"""Core geometry for Delaunay-based cone centerline planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from sim_car.cone_fusion import normalize_color

try:
    from scipy.spatial import Delaunay as _ScipyDelaunay
except Exception:  # pragma: no cover - optional dependency
    _ScipyDelaunay = None


@dataclass
class CoreConfig:
    max_cone_range_m: float = 25.0
    behind_drop_m: float = 5.0
    min_confidence: float = 0.3
    use_unknown_cones: bool = True
    infer_unknown_by_side: bool = True
    include_orange: bool = False
    min_colored_cones: int = 6
    min_required_cones: int = 6

    min_cross_edge_m: float = 0.8
    max_cross_edge_m: float = 6.0
    cross_edge_lateral_ratio: float = 0.6
    min_cross_edges: int = 3

    min_spacing_m: float = 0.5
    path_resolution_m: float = 0.5
    max_path_length_m: float = 30.0


@dataclass
class CoreResult:
    filtered_points: np.ndarray
    filtered_colors: list[str]
    triangulation_edges: np.ndarray
    candidate_edges: np.ndarray
    selected_edges: np.ndarray
    midpoints_raw: np.ndarray
    centerline: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    used_fallback: bool
    status: str


def compute_centerline(
    points_xy: np.ndarray,
    colors: list[str],
    confidences: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: CoreConfig,
) -> CoreResult:
    """Build a local centerline from tracked cones in a common planning frame."""
    if points_xy.size == 0:
        return _empty_result('no cones available')

    normalized = [normalize_color(c) for c in colors]
    if config.infer_unknown_by_side:
        normalized = _infer_unknown_by_side(
            points_xy=points_xy,
            colors=normalized,
            vehicle_xy=vehicle_xy,
            vehicle_yaw=vehicle_yaw,
        )
    mask_geom = _geometry_filter(points_xy, vehicle_xy, vehicle_yaw, config)
    mask_conf = confidences >= float(config.min_confidence)

    colored_mask = np.array([c in {'blue', 'yellow'} for c in normalized], dtype=bool)
    unknown_mask = np.array([c == 'unknown' for c in normalized], dtype=bool)
    orange_mask = np.array([c == 'orange' for c in normalized], dtype=bool)

    if config.include_orange:
        base_color_mask = colored_mask | orange_mask
    else:
        base_color_mask = colored_mask

    selected_mask = mask_geom & mask_conf & base_color_mask
    selected_colored_count = int(np.count_nonzero(selected_mask & colored_mask))

    if config.use_unknown_cones and selected_colored_count < int(config.min_colored_cones):
        selected_mask = mask_geom & mask_conf & (base_color_mask | unknown_mask)

    filtered_points = points_xy[selected_mask]
    filtered_colors = [normalized[idx] for idx in np.where(selected_mask)[0]]

    if filtered_points.shape[0] < int(config.min_required_cones):
        return _empty_result(
            f'usable cones below minimum ({filtered_points.shape[0]} < {int(config.min_required_cones)})',
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
        )

    tri_edges, tri_ok = _build_edges(filtered_points)
    fallback_reason: Optional[str] = None
    if tri_edges.shape[0] == 0:
        fallback_reason = 'triangulation failed'

    candidate_edges, selected_edges = _pick_cross_edges(
        points=filtered_points,
        colors=filtered_colors,
        edges=tri_edges,
        yaw=vehicle_yaw,
        config=config,
    )

    used_fallback = False
    if selected_edges.shape[0] < int(config.min_cross_edges):
        used_fallback = True
        fallback_reason = fallback_reason or 'insufficient blue-yellow cross edges'
        fallback_pairs = _nearest_blue_yellow_pairs(filtered_points, filtered_colors, vehicle_yaw, config)
        if fallback_pairs.shape[0] > selected_edges.shape[0]:
            selected_edges = fallback_pairs
            candidate_edges = fallback_pairs

    if selected_edges.shape[0] == 0:
        status = fallback_reason or 'no valid cross edges'
        return _empty_result(
            status,
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            triangulation_edges=tri_edges,
            candidate_edges=candidate_edges,
            used_fallback=used_fallback,
        )

    midpoints = 0.5 * (filtered_points[selected_edges[:, 0]] + filtered_points[selected_edges[:, 1]])
    ordered_midpoints = _order_forward_points(midpoints, vehicle_xy, vehicle_yaw, config.behind_drop_m)
    ordered_midpoints = _dedup_spacing(ordered_midpoints, config.min_spacing_m)
    if ordered_midpoints.shape[0] == 0:
        return _empty_result(
            'midpoint ordering removed all candidates',
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            triangulation_edges=tri_edges,
            candidate_edges=candidate_edges,
            selected_edges=selected_edges,
            used_fallback=used_fallback,
        )

    smooth = _moving_average(ordered_midpoints, window=3)
    centerline = _resample_path(
        smooth,
        resolution_m=config.path_resolution_m,
        max_length_m=config.max_path_length_m,
    )

    left_boundary = _order_boundary(filtered_points, filtered_colors, 'blue', vehicle_xy, vehicle_yaw)
    right_boundary = _order_boundary(filtered_points, filtered_colors, 'yellow', vehicle_xy, vehicle_yaw)

    if not tri_ok and fallback_reason is None:
        fallback_reason = 'triangulation fallback used'

    status = 'ok' if centerline.shape[0] > 0 else (fallback_reason or 'centerline generation failed')
    return CoreResult(
        filtered_points=filtered_points,
        filtered_colors=filtered_colors,
        triangulation_edges=tri_edges,
        candidate_edges=candidate_edges,
        selected_edges=selected_edges,
        midpoints_raw=ordered_midpoints,
        centerline=centerline,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        used_fallback=used_fallback or (not tri_ok),
        status=status,
    )


def _empty_result(
    status: str,
    *,
    filtered_points: Optional[np.ndarray] = None,
    filtered_colors: Optional[list[str]] = None,
    triangulation_edges: Optional[np.ndarray] = None,
    candidate_edges: Optional[np.ndarray] = None,
    selected_edges: Optional[np.ndarray] = None,
    used_fallback: bool = False,
) -> CoreResult:
    return CoreResult(
        filtered_points=filtered_points if filtered_points is not None else np.empty((0, 2), dtype=np.float64),
        filtered_colors=filtered_colors if filtered_colors is not None else [],
        triangulation_edges=(
            triangulation_edges if triangulation_edges is not None else np.empty((0, 2), dtype=np.int64)
        ),
        candidate_edges=candidate_edges if candidate_edges is not None else np.empty((0, 2), dtype=np.int64),
        selected_edges=selected_edges if selected_edges is not None else np.empty((0, 2), dtype=np.int64),
        midpoints_raw=np.empty((0, 2), dtype=np.float64),
        centerline=np.empty((0, 2), dtype=np.float64),
        left_boundary=np.empty((0, 2), dtype=np.float64),
        right_boundary=np.empty((0, 2), dtype=np.float64),
        used_fallback=used_fallback,
        status=status,
    )


def _geometry_filter(
    points_xy: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: CoreConfig,
) -> np.ndarray:
    rel = points_xy - np.asarray(vehicle_xy, dtype=np.float64).reshape(1, 2)
    ranges = np.hypot(rel[:, 0], rel[:, 1])
    vx, _ = _rotate_into_vehicle(rel, vehicle_yaw)
    return (ranges <= float(config.max_cone_range_m)) & (vx >= -float(config.behind_drop_m))


def _infer_unknown_by_side(
    *,
    points_xy: np.ndarray,
    colors: list[str],
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
) -> list[str]:
    if points_xy.size == 0 or not colors:
        return colors

    rel = points_xy - np.asarray(vehicle_xy, dtype=np.float64).reshape(1, 2)
    vx, vy = _rotate_into_vehicle(rel, vehicle_yaw)
    inferred = list(colors)
    for idx, color in enumerate(colors):
        if color != 'unknown':
            continue
        angle = math.atan2(float(vy[idx]), float(vx[idx]))
        inferred[idx] = 'blue' if angle >= 0.0 else 'yellow'
    return inferred


def _build_edges(points: np.ndarray) -> tuple[np.ndarray, bool]:
    if points.shape[0] < 3:
        return np.empty((0, 2), dtype=np.int64), False

    tri_edges: set[tuple[int, int]] = set()
    if _ScipyDelaunay is not None:
        try:
            tri = _ScipyDelaunay(points)
            simplices = np.asarray(tri.simplices, dtype=np.int64)
            for simplex in simplices:
                a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
                tri_edges.add(tuple(sorted((a, b))))
                tri_edges.add(tuple(sorted((b, c))))
                tri_edges.add(tuple(sorted((a, c))))
            if tri_edges:
                return np.array(sorted(tri_edges), dtype=np.int64), True
        except Exception:
            pass

    # Fallback: build a sparse undirected graph from local nearest neighbors.
    dxy = points[:, None, :] - points[None, :, :]
    dist = np.hypot(dxy[:, :, 0], dxy[:, :, 1])
    np.fill_diagonal(dist, np.inf)
    neighbor_count = max(2, min(4, points.shape[0] - 1))
    nearest = np.argsort(dist, axis=1)[:, :neighbor_count]
    for i in range(points.shape[0]):
        for j in nearest[i]:
            tri_edges.add(tuple(sorted((int(i), int(j)))))
    if not tri_edges:
        return np.empty((0, 2), dtype=np.int64), False
    return np.array(sorted(tri_edges), dtype=np.int64), False


def _pick_cross_edges(
    *,
    points: np.ndarray,
    colors: list[str],
    edges: np.ndarray,
    yaw: float,
    config: CoreConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if edges.shape[0] == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return empty, empty

    lengths = np.hypot(
        points[edges[:, 0], 0] - points[edges[:, 1], 0],
        points[edges[:, 0], 1] - points[edges[:, 1], 1],
    )
    len_ok = (lengths >= float(config.min_cross_edge_m)) & (lengths <= float(config.max_cross_edge_m))

    vec = points[edges[:, 1]] - points[edges[:, 0]]
    vx, vy = _rotate_into_vehicle(vec, yaw)
    norm = np.maximum(np.hypot(vx, vy), 1e-6)
    lateral_ratio = np.abs(vy) / norm
    lateral_ok = lateral_ratio >= float(config.cross_edge_lateral_ratio)

    geom_mask = len_ok & lateral_ok
    geom_edges = edges[geom_mask]
    if geom_edges.shape[0] == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return empty, empty

    color_mask = np.array(
        [
            (colors[int(a)] == 'blue' and colors[int(b)] == 'yellow')
            or (colors[int(a)] == 'yellow' and colors[int(b)] == 'blue')
            for a, b in geom_edges
        ],
        dtype=bool,
    )
    colored_edges = geom_edges[color_mask]
    if colored_edges.shape[0] >= int(config.min_cross_edges):
        return colored_edges, colored_edges
    return geom_edges, geom_edges


def _nearest_blue_yellow_pairs(
    points: np.ndarray,
    colors: list[str],
    yaw: float,
    config: CoreConfig,
) -> np.ndarray:
    blue_idx = [idx for idx, color in enumerate(colors) if color == 'blue']
    yellow_idx = [idx for idx, color in enumerate(colors) if color == 'yellow']
    if not blue_idx or not yellow_idx:
        return np.empty((0, 2), dtype=np.int64)

    candidates: list[tuple[float, int, int]] = []
    for b in blue_idx:
        for y in yellow_idx:
            vec = points[y] - points[b]
            dist = float(np.hypot(vec[0], vec[1]))
            if dist < float(config.min_cross_edge_m) or dist > float(config.max_cross_edge_m):
                continue
            vx, vy = _rotate_into_vehicle(vec.reshape(1, 2), yaw)
            ratio = abs(float(vy[0])) / max(float(np.hypot(vx[0], vy[0])), 1e-6)
            if ratio < float(config.cross_edge_lateral_ratio):
                continue
            candidates.append((dist, b, y))

    candidates.sort(key=lambda item: item[0])
    used_blue: set[int] = set()
    used_yellow: set[int] = set()
    out: list[tuple[int, int]] = []
    for _dist, b, y in candidates:
        if b in used_blue or y in used_yellow:
            continue
        used_blue.add(b)
        used_yellow.add(y)
        out.append((min(b, y), max(b, y)))

    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def _order_forward_points(
    points: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    behind_drop_m: float,
) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    rel = points - np.asarray(vehicle_xy, dtype=np.float64).reshape(1, 2)
    vx, _ = _rotate_into_vehicle(rel, vehicle_yaw)
    keep = vx >= -float(behind_drop_m)
    kept = points[keep]
    kept_vx = vx[keep]
    if kept.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    order = np.argsort(kept_vx)
    return kept[order]


def _dedup_spacing(points: np.ndarray, min_spacing_m: float) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    kept = [points[0]]
    min_spacing = float(max(1e-6, min_spacing_m))
    for idx in range(1, points.shape[0]):
        if np.hypot(*(points[idx] - kept[-1])) >= min_spacing:
            kept.append(points[idx])
    return np.asarray(kept, dtype=np.float64)


def _moving_average(points: np.ndarray, window: int) -> np.ndarray:
    if points.shape[0] < 3 or window <= 1:
        return points
    radius = max(1, int(window) // 2)
    out = np.array(points, copy=True)
    for idx in range(points.shape[0]):
        lo = max(0, idx - radius)
        hi = min(points.shape[0], idx + radius + 1)
        out[idx] = np.mean(points[lo:hi], axis=0)
    return out


def _resample_path(points: np.ndarray, resolution_m: float, max_length_m: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    if points.shape[0] == 1:
        return points.copy()

    seg = points[1:] - points[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])
    if total <= 1e-6:
        return points[:1].copy()

    capped = min(total, float(max_length_m))
    step = max(float(resolution_m), 0.05)
    samples = np.arange(0.0, capped + 1e-9, step)
    if samples.size == 0 or samples[-1] < capped:
        samples = np.concatenate((samples, [capped]))

    x = np.interp(samples, cum, points[:, 0])
    y = np.interp(samples, cum, points[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


def _order_boundary(
    points: np.ndarray,
    colors: list[str],
    boundary_color: str,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
) -> np.ndarray:
    idx = [i for i, c in enumerate(colors) if c == boundary_color]
    if not idx:
        return np.empty((0, 2), dtype=np.float64)
    subset = points[np.asarray(idx, dtype=np.int64)]
    return _order_forward_points(subset, vehicle_xy, vehicle_yaw, behind_drop_m=999.0)


def _rotate_into_vehicle(vec_xy: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    vx = cos_y * vec_xy[:, 0] + sin_y * vec_xy[:, 1]
    vy = -sin_y * vec_xy[:, 0] + cos_y * vec_xy[:, 1]
    return vx, vy
