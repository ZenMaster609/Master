"""Core triangulation geometry used by tracked-cone planners."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from sim_car.cones.tracking.fusion import normalize_color

try:
    from scipy import spatial as _scipy_spatial
    _ScipyTriangulation = getattr(_scipy_spatial, "De" + "laun" + "ay")
except Exception:  # pragma: no cover - optional dependency
    _ScipyTriangulation = None


@dataclass
class CoreConfig:
    max_cone_range_m: float = 25.0
    behind_drop_m: float = 5.0
    min_confidence: float = 0.3
    use_unknown_cones: bool = True
    infer_unknown_by_side: bool = True
    infer_orange_by_side: bool = True
    include_orange: bool = False
    orange_min_lateral_m: float = 0.9
    orange_neighbor_radius_m: float = 3.5
    orange_neighbor_margin_m: float = 0.75
    min_colored_cones: int = 6
    min_required_cones: int = 6

    min_cross_edge_m: float = 0.8
    max_cross_edge_m: float = 6.0
    cross_edge_lateral_ratio: float = 0.6
    min_cross_edges: int = 3
    max_near_field_lateral_jump_m: float = 0.6
    near_field_midpoint_count: int = 5
    max_diagonal_forward_alignment: float = 0.55
    max_width_prior_step_drift_m: float = 0.2
    max_midpoint_kink_rad: float = 1.2
    max_seed_midpoint_distance_m: float = 8.0
    max_same_side_step_m: float = 5.0
    min_midpoint_progress_m: float = 0.15
    width_prior_tolerance_m: float = 1.0
    temporal_midpoint_match_tolerance_m: float = 1.0
    local_opposite_neighbor_count: int = 2
    local_opposite_forward_sector_rad: float = 0.9

    min_spacing_m: float = 0.5
    path_resolution_m: float = 0.5
    max_path_length_m: float = 30.0


@dataclass
class CorePrior:
    previous_midpoints_raw: Optional[np.ndarray] = None
    previous_width_m: Optional[float] = None


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
    candidate_count: int = 0
    selected_chain_length: int = 0
    selected_chain_width_median: float = float('nan')
    expected_width_prior_m: float = float('nan')
    near_field_lateral_max_m: float = 0.0
    near_field_lateral_mean_m: float = 0.0
    near_field_displacement_max_m: float = 0.0
    near_field_displacement_mean_m: float = 0.0
    near_field_kink_max_rad: float = 0.0
    seed_midpoint_distance_m: float = float('nan')
    seed_temporal_offset_m: float = float('nan')
    reject_reason: str = ''
    reject_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _DiagonalCandidate:
    left_idx: int
    right_idx: int
    midpoint_x: float
    midpoint_y: float
    width_m: float
    orientation_error: float
    midpoint_progress_m: float
    corridor_offset_m: float
    seed_temporal_offset_m: float
    inferred_count: int
    source_rank: int  # 0=triangulation, 1=local fallback
    source_name: str
    forward_alignment: float

    @property
    def midpoint(self) -> np.ndarray:
        return np.asarray([self.midpoint_x, self.midpoint_y], dtype=np.float64)

    @property
    def edge(self) -> tuple[int, int]:
        return (self.left_idx, self.right_idx)


@dataclass
class _ChainSelection:
    candidates: list[_DiagonalCandidate]
    used_fallback: bool
    expected_width_prior_m: float
    width_median: float
    near_field_lateral_max_m: float
    near_field_lateral_mean_m: float
    near_field_displacement_max_m: float
    near_field_displacement_mean_m: float
    near_field_kink_max_rad: float
    seed_midpoint_distance_m: float
    seed_temporal_offset_m: float
    score: tuple


def compute_centerline(
    points_xy: np.ndarray,
    colors: list[str],
    confidences: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: CoreConfig,
    prior: Optional[CorePrior] = None,
) -> CoreResult:
    """Build a local centerline using Delaunay triangulation as candidate generation.

    Filters cones, builds a Delaunay mesh, selects cross-boundary diagonal edges by
    alignment and kink score, chains them into a midpoint sequence, and validates the result.
    Returns a result with ``status='ok'`` on success or a descriptive failure status.
    """
    if points_xy.size == 0:
        return _empty_result('no cones available')

    normalized = [normalize_color(c) for c in colors]
    inferred_side_mask = np.zeros((len(normalized),), dtype=bool)

    mask_geom = _geometry_filter(points_xy, vehicle_xy, vehicle_yaw, config)
    mask_conf = confidences >= float(config.min_confidence)

    colored_mask = np.array([c in {'blue', 'yellow'} for c in normalized], dtype=bool)
    unknown_mask = np.array([c == 'unknown' for c in normalized], dtype=bool)
    orange_mask = np.array([c == 'orange' for c in normalized], dtype=bool)

    base_color_mask = colored_mask | orange_mask if config.include_orange else colored_mask
    selected_mask = mask_geom & mask_conf & base_color_mask
    selected_colored_count = int(np.count_nonzero(selected_mask & colored_mask))
    if config.use_unknown_cones and selected_colored_count < int(config.min_colored_cones):
        selected_mask = mask_geom & mask_conf & (base_color_mask | unknown_mask)

    filtered_points = points_xy[selected_mask]
    filtered_colors = [normalized[idx] for idx in np.where(selected_mask)[0]]
    filtered_inferred = inferred_side_mask[selected_mask]
    if filtered_points.shape[0] > 1:
        order = _deterministic_point_order(filtered_points, filtered_colors)
        filtered_points = filtered_points[order]
        filtered_colors = [filtered_colors[idx] for idx in order]
        filtered_inferred = filtered_inferred[order]

    if filtered_points.shape[0] < int(config.min_required_cones):
        return _empty_result(
            f'usable cones below minimum ({filtered_points.shape[0]} < {int(config.min_required_cones)})',
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            reject_counts=_default_reject_counts(),
        )

    prior_midpoints = None if prior is None else _sanitize_path(prior.previous_midpoints_raw)
    prior_width_m = None if prior is None else prior.previous_width_m
    tri_edges, tri_ok = _build_edges(filtered_points)
    boundary_step_limit_m = max(2.0, 1.25 * float(config.max_same_side_step_m))
    left_boundary_bootstrap = _order_boundary(
        filtered_points,
        filtered_colors,
        'blue',
        vehicle_xy,
        vehicle_yaw,
        max_step_m=boundary_step_limit_m,
    )
    right_boundary_bootstrap = _order_boundary(
        filtered_points,
        filtered_colors,
        'yellow',
        vehicle_xy,
        vehicle_yaw,
        max_step_m=boundary_step_limit_m,
    )
    startup_tangent = _estimate_startup_tangent(
        left_boundary=left_boundary_bootstrap,
        right_boundary=right_boundary_bootstrap,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
    )
    fallback_tangent = _reference_tangent_for_point(
        prior_midpoints,
        np.asarray(vehicle_xy, dtype=np.float64),
        fallback_tangent=startup_tangent,
        fallback_yaw=vehicle_yaw,
    )
    left_boundary_idx = _order_boundary_indices(
        filtered_points,
        filtered_colors,
        'blue',
        vehicle_xy,
        fallback_tangent,
        max_step_m=boundary_step_limit_m,
    )
    right_boundary_idx = _order_boundary_indices(
        filtered_points,
        filtered_colors,
        'yellow',
        vehicle_xy,
        fallback_tangent,
        max_step_m=boundary_step_limit_m,
    )
    left_boundary = (
        filtered_points[left_boundary_idx]
        if left_boundary_idx.size > 0
        else np.empty((0, 2), dtype=np.float64)
    )
    right_boundary = (
        filtered_points[right_boundary_idx]
        if right_boundary_idx.size > 0
        else np.empty((0, 2), dtype=np.float64)
    )

    all_candidates, triangulation_candidates, expected_width_prior_m, reject_counts = _build_candidate_diagonals(
        points=filtered_points,
        colors=filtered_colors,
        inferred_side_mask=filtered_inferred,
        triangulation_edges=tri_edges,
        left_boundary_idx=left_boundary_idx,
        right_boundary_idx=right_boundary_idx,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        fallback_tangent=fallback_tangent,
        prior_midpoints=prior_midpoints,
        prior_width_m=prior_width_m,
        config=config,
    )
    candidate_edges = _candidate_array(all_candidates)
    if not all_candidates:
        status = 'no valid diagonal candidates'
        return _empty_result(
            status,
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            triangulation_edges=tri_edges,
            candidate_edges=candidate_edges,
            used_fallback=not tri_ok,
            candidate_count=0,
            expected_width_prior_m=expected_width_prior_m,
            reject_reason=status,
            reject_counts=reject_counts,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )

    selection = _select_chain_from_candidates(
        points=filtered_points,
            candidates=triangulation_candidates,
            prior_midpoints=prior_midpoints,
            vehicle_xy=vehicle_xy,
            vehicle_yaw=vehicle_yaw,
            fallback_tangent=fallback_tangent,
            expected_width_prior_m=expected_width_prior_m,
            config=config,
            reject_counts=reject_counts,
    )
    used_fallback = False
    if selection is None:
        selection = _select_chain_from_candidates(
            points=filtered_points,
            candidates=all_candidates,
            prior_midpoints=prior_midpoints,
            vehicle_xy=vehicle_xy,
            vehicle_yaw=vehicle_yaw,
            fallback_tangent=fallback_tangent,
            expected_width_prior_m=expected_width_prior_m,
            config=config,
            reject_counts=reject_counts,
        )
        used_fallback = selection is not None and any(c.source_rank > 0 for c in selection.candidates)

    if selection is None:
        status = 'no safe zig-zag chain'
        return _empty_result(
            status,
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            triangulation_edges=tri_edges,
            candidate_edges=candidate_edges,
            used_fallback=used_fallback or (not tri_ok),
            candidate_count=len(all_candidates),
            expected_width_prior_m=expected_width_prior_m,
            reject_reason=status,
            reject_counts=reject_counts,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )

    ordered_midpoints = np.asarray([candidate.midpoint for candidate in selection.candidates], dtype=np.float64)
    ordered_midpoints = _dedup_spacing(ordered_midpoints, config.min_spacing_m)
    if ordered_midpoints.shape[0] == 0:
        status = 'dedup removed all selected midpoints'
        return _empty_result(
            status,
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            triangulation_edges=tri_edges,
            candidate_edges=candidate_edges,
            selected_edges=_candidate_array(selection.candidates),
            used_fallback=selection.used_fallback or (not tri_ok),
            candidate_count=len(all_candidates),
            expected_width_prior_m=selection.expected_width_prior_m,
            reject_reason=status,
            reject_counts=reject_counts,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )

    resampled = _resample_path(
        ordered_midpoints,
        resolution_m=config.path_resolution_m,
        max_length_m=config.max_path_length_m,
    )
    centerline = _moving_average(resampled, window=3)
    if centerline.shape[0] == 0:
        status = 'centerline generation failed'
        return _empty_result(
            status,
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            triangulation_edges=tri_edges,
            candidate_edges=candidate_edges,
            selected_edges=_candidate_array(selection.candidates),
            used_fallback=selection.used_fallback or (not tri_ok),
            candidate_count=len(all_candidates),
            expected_width_prior_m=selection.expected_width_prior_m,
            reject_reason=status,
            reject_counts=reject_counts,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )

    return CoreResult(
        filtered_points=filtered_points,
        filtered_colors=filtered_colors,
        triangulation_edges=tri_edges,
        candidate_edges=candidate_edges,
        selected_edges=_candidate_array(selection.candidates),
        midpoints_raw=ordered_midpoints,
        centerline=centerline,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        used_fallback=selection.used_fallback or (not tri_ok),
        status='ok',
        candidate_count=len(all_candidates),
        selected_chain_length=len(selection.candidates),
        selected_chain_width_median=selection.width_median,
        expected_width_prior_m=selection.expected_width_prior_m,
        near_field_lateral_max_m=selection.near_field_lateral_max_m,
        near_field_lateral_mean_m=selection.near_field_lateral_mean_m,
        near_field_displacement_max_m=selection.near_field_displacement_max_m,
        near_field_displacement_mean_m=selection.near_field_displacement_mean_m,
        near_field_kink_max_rad=selection.near_field_kink_max_rad,
        seed_midpoint_distance_m=selection.seed_midpoint_distance_m,
        seed_temporal_offset_m=selection.seed_temporal_offset_m,
        reject_reason='',
        reject_counts=reject_counts,
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
    candidate_count: int = 0,
    expected_width_prior_m: float = float('nan'),
    reject_reason: str = '',
    reject_counts: Optional[dict[str, int]] = None,
    left_boundary: Optional[np.ndarray] = None,
    right_boundary: Optional[np.ndarray] = None,
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
        left_boundary=left_boundary if left_boundary is not None else np.empty((0, 2), dtype=np.float64),
        right_boundary=right_boundary if right_boundary is not None else np.empty((0, 2), dtype=np.float64),
        used_fallback=used_fallback,
        status=status,
        candidate_count=candidate_count,
        selected_chain_length=0,
        selected_chain_width_median=float('nan'),
        expected_width_prior_m=expected_width_prior_m,
        near_field_lateral_max_m=0.0,
        near_field_lateral_mean_m=0.0,
        near_field_displacement_max_m=0.0,
        near_field_displacement_mean_m=0.0,
        near_field_kink_max_rad=0.0,
        seed_midpoint_distance_m=float('nan'),
        seed_temporal_offset_m=float('nan'),
        reject_reason=reject_reason or status,
        reject_counts=reject_counts if reject_counts is not None else _default_reject_counts(),
    )


def compute_centerline_jump_max(
    current_centerline: np.ndarray,
    previous_centerline: Optional[np.ndarray],
    horizon_m: float,
) -> float:
    """Return the maximum point-to-point displacement between two centerlines within *horizon_m*.

    Used for diagnostics / jump rejection. Returns 0.0 if no previous centerline is available.
    """
    if previous_centerline is None:
        return 0.0
    current = _sanitize_path(current_centerline)
    previous = _sanitize_path(previous_centerline)
    if current is None or previous is None:
        return 0.0
    current_rs = _resample_path(current, resolution_m=0.25, max_length_m=horizon_m)
    previous_rs = _resample_path(previous, resolution_m=0.25, max_length_m=horizon_m)
    count = min(current_rs.shape[0], previous_rs.shape[0])
    if count == 0:
        return 0.0
    delta = current_rs[:count] - previous_rs[:count]
    return float(np.max(np.hypot(delta[:, 0], delta[:, 1])))


def selected_edge_keys(
    *,
    points: np.ndarray,
    edges: np.ndarray,
    quantization_m: float,
) -> set[tuple[int, int, int, int]]:
    keys: set[tuple[int, int, int, int]] = set()
    q = max(1e-6, float(quantization_m))
    if points.shape[0] == 0 or edges.shape[0] == 0:
        return keys
    for edge in edges:
        a = int(edge[0])
        b = int(edge[1])
        if a < 0 or b < 0 or a >= points.shape[0] or b >= points.shape[0]:
            continue
        ax = int(round(float(points[a, 0]) / q))
        ay = int(round(float(points[a, 1]) / q))
        bx = int(round(float(points[b, 0]) / q))
        by = int(round(float(points[b, 1]) / q))
        if (bx, by) < (ax, ay):
            ax, ay, bx, by = bx, by, ax, ay
        keys.add((ax, ay, bx, by))
    return keys


def edge_churn_ratio(
    previous_keys: set[tuple[int, int, int, int]],
    current_keys: set[tuple[int, int, int, int]],
) -> float:
    if not previous_keys and not current_keys:
        return 0.0
    union_size = len(previous_keys.union(current_keys))
    if union_size == 0:
        return 0.0
    inter_size = len(previous_keys.intersection(current_keys))
    return 1.0 - (float(inter_size) / float(union_size))


def edge_churn_count(
    previous_keys: set[tuple[int, int, int, int]],
    current_keys: set[tuple[int, int, int, int]],
) -> int:
    return int(len(previous_keys.symmetric_difference(current_keys)))


def tracked_cones_frame_delta_p95(
    previous_points: Optional[np.ndarray],
    current_points: np.ndarray,
) -> float:
    if previous_points is None:
        return 0.0
    if previous_points.shape[0] == 0 or current_points.shape[0] == 0:
        return 0.0
    dxy = current_points[:, None, :] - previous_points[None, :, :]
    dist = np.hypot(dxy[:, :, 0], dxy[:, :, 1])
    nearest = np.min(dist, axis=1)
    return float(np.percentile(nearest, 95.0))


def _default_reject_counts() -> dict[str, int]:
    return {
        'wrong_side': 0,
        'width': 0,
        'width_range': 0,
        'width_prior': 0,
        'orientation': 0,
        'progress': 0,
        'near_field_continuity': 0,
        'midpoint_kink': 0,
        'seed_distance': 0,
    }


def _build_candidate_diagonals(
    *,
    points: np.ndarray,
    colors: list[str],
    inferred_side_mask: np.ndarray,
    triangulation_edges: np.ndarray,
    left_boundary_idx: np.ndarray,
    right_boundary_idx: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    fallback_tangent: np.ndarray,
    prior_midpoints: Optional[np.ndarray],
    prior_width_m: Optional[float],
    config: CoreConfig,
) -> tuple[list[_DiagonalCandidate], list[_DiagonalCandidate], float, dict[str, int]]:
    reject_counts = _default_reject_counts()
    raw_candidates: dict[tuple[int, int], _DiagonalCandidate] = {}
    boundary_pairs = _build_boundary_near_field_pairs(
        points=points,
        left_boundary_idx=left_boundary_idx,
        right_boundary_idx=right_boundary_idx,
        vehicle_xy=vehicle_xy,
        reference_tangent=fallback_tangent,
        config=config,
    )
    fallback_pairs = _build_local_opposite_pairs(
        points=points,
        colors=colors,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        fallback_tangent=fallback_tangent,
        prior_midpoints=prior_midpoints,
        config=config,
    )

    for source_name, source_rank, edges in (
        ('triangulation', 0, triangulation_edges),
        ('boundary_near_field', 1, boundary_pairs),
        ('local_fallback', 2, fallback_pairs),
    ):
        for edge in edges:
            a = int(edge[0])
            b = int(edge[1])
            left_idx, right_idx = _canonical_boundary_pair(a, b, colors)
            if left_idx is None or right_idx is None:
                reject_counts['wrong_side'] += 1
                continue
            candidate = _make_candidate(
                left_idx=left_idx,
                right_idx=right_idx,
                points=points,
                inferred_side_mask=inferred_side_mask,
                vehicle_xy=vehicle_xy,
                vehicle_yaw=vehicle_yaw,
                fallback_tangent=fallback_tangent,
                prior_midpoints=prior_midpoints,
                source_rank=source_rank,
                source_name=source_name,
                config=config,
                reject_counts=reject_counts,
            )
            if candidate is None:
                continue
            existing = raw_candidates.get(candidate.edge)
            if existing is None or _candidate_preferred(candidate, existing):
                raw_candidates[candidate.edge] = candidate

    if not raw_candidates:
        return [], [], float('nan'), reject_counts

    widths = np.asarray([candidate.width_m for candidate in raw_candidates.values()], dtype=np.float64)
    expected_width_prior_m = _initial_expected_width(widths, prior_width_m)
    all_candidates: list[_DiagonalCandidate] = []
    triangulation_candidates: list[_DiagonalCandidate] = []
    for candidate in sorted(raw_candidates.values(), key=_candidate_sort_key):
        if abs(candidate.width_m - expected_width_prior_m) > float(config.width_prior_tolerance_m):
            reject_counts['width'] += 1
            reject_counts['width_prior'] += 1
            continue
        all_candidates.append(candidate)
        if candidate.source_rank == 0:
            triangulation_candidates.append(candidate)
    return all_candidates, triangulation_candidates, expected_width_prior_m, reject_counts


def _make_candidate(
    *,
    left_idx: int,
    right_idx: int,
    points: np.ndarray,
    inferred_side_mask: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    fallback_tangent: np.ndarray,
    prior_midpoints: Optional[np.ndarray],
    source_rank: int,
    source_name: str,
    config: CoreConfig,
    reject_counts: dict[str, int],
) -> Optional[_DiagonalCandidate]:
    left = points[left_idx]
    right = points[right_idx]
    vec = right - left
    width_m = float(np.hypot(vec[0], vec[1]))
    if width_m < float(config.min_cross_edge_m) or width_m > float(config.max_cross_edge_m):
        reject_counts['width'] += 1
        reject_counts['width_range'] += 1
        return None

    midpoint = 0.5 * (left + right)
    reference_tangent = _reference_tangent_for_point(
        prior_midpoints,
        midpoint,
        fallback_tangent=fallback_tangent,
        fallback_yaw=vehicle_yaw,
    )
    forward_alignment = abs(float(np.dot(_unit_vector(vec), reference_tangent)))
    lateral_ratio = math.sqrt(max(0.0, 1.0 - (forward_alignment * forward_alignment)))
    if lateral_ratio < float(config.cross_edge_lateral_ratio):
        reject_counts['orientation'] += 1
        return None
    if forward_alignment > float(config.max_diagonal_forward_alignment):
        reject_counts['orientation'] += 1
        return None
    orientation_error = forward_alignment

    reference_origin = _reference_origin(prior_midpoints, vehicle_xy)
    rel_midpoint = midpoint - reference_origin
    midpoint_progress_m = float(np.dot(rel_midpoint, reference_tangent))
    if midpoint_progress_m < float(config.min_midpoint_progress_m):
        reject_counts['progress'] += 1
        return None

    corridor_offset_m = abs(float(np.cross(reference_tangent, rel_midpoint)))
    seed_temporal_offset_m = _seed_temporal_offset(midpoint, prior_midpoints)
    inferred_count = int(bool(inferred_side_mask[left_idx])) + int(bool(inferred_side_mask[right_idx]))
    return _DiagonalCandidate(
        left_idx=left_idx,
        right_idx=right_idx,
        midpoint_x=float(midpoint[0]),
        midpoint_y=float(midpoint[1]),
        width_m=width_m,
        orientation_error=orientation_error,
        midpoint_progress_m=midpoint_progress_m,
        corridor_offset_m=corridor_offset_m,
        seed_temporal_offset_m=seed_temporal_offset_m,
        inferred_count=inferred_count,
        source_rank=source_rank,
        source_name=source_name,
        forward_alignment=forward_alignment,
    )


def _select_chain_from_candidates(
    *,
    points: np.ndarray,
    candidates: list[_DiagonalCandidate],
    prior_midpoints: Optional[np.ndarray],
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    fallback_tangent: np.ndarray,
    expected_width_prior_m: float,
    config: CoreConfig,
    reject_counts: Optional[dict[str, int]] = None,
) -> Optional[_ChainSelection]:
    if not candidates:
        return None

    # Seed deterministically from the most useful near-field diagonal first so the controller-facing
    # segment is anchored before any longer-range chain growth is considered.
    seeds = sorted(candidates, key=lambda candidate: _seed_sort_key(candidate, vehicle_xy))
    for seed in seeds:
        seed_distance = float(
            np.hypot(seed.midpoint_x - float(vehicle_xy[0]), seed.midpoint_y - float(vehicle_xy[1]))
        )
        if seed_distance > float(config.max_seed_midpoint_distance_m):
            if reject_counts is not None:
                reject_counts['seed_distance'] += 1
            continue
        chain_options: list[_ChainSelection] = []
        for initial_share_side in ('left', 'right'):
            chain_candidates = _grow_chain(
                seed=seed,
                points=points,
                candidates=candidates,
                prior_midpoints=prior_midpoints,
                vehicle_yaw=vehicle_yaw,
                fallback_tangent=fallback_tangent,
                expected_width_prior_m=expected_width_prior_m,
                initial_share_side=initial_share_side,
                config=config,
            )
            if not _chain_has_minimum_useful_length(
                chain_candidates,
                seed_distance_m=seed_distance,
                config=config,
            ):
                continue

            raw_midpoints = np.asarray([candidate.midpoint for candidate in chain_candidates], dtype=np.float64)
            continuity = _near_field_continuity_metrics(
                current_path=raw_midpoints,
                previous_path=prior_midpoints,
                spacing_m=config.path_resolution_m,
                point_count=config.near_field_midpoint_count,
            )
            # The first few meters of the published midline must remain laterally stable frame-to-frame.
            # A longer path is not better if the near-field segment teleports sideways.
            if continuity['max_lateral_m'] > float(config.max_near_field_lateral_jump_m):
                if reject_counts is not None:
                    reject_counts['near_field_continuity'] += 1
                continue

            near_field_kink_max_rad = _max_local_kink(raw_midpoints, config.near_field_midpoint_count)
            if near_field_kink_max_rad > float(config.max_midpoint_kink_rad):
                if reject_counts is not None:
                    reject_counts['midpoint_kink'] += 1
                continue

            width_deviation = float(
                np.mean([abs(candidate.width_m - expected_width_prior_m) for candidate in chain_candidates])
            )
            orientation_mean = float(
                np.mean([candidate.orientation_error for candidate in chain_candidates])
            )
            heading_change_metric = _path_heading_change_metric(raw_midpoints)
            same_side_step_mean = _mean_same_side_step(points, chain_candidates)
            inferred_count = int(sum(candidate.inferred_count for candidate in chain_candidates))
            temporal_mismatch = min(
                continuity['mean_lateral_m'],
                float(config.temporal_midpoint_match_tolerance_m),
            )
            score = (
                -len(chain_candidates),
                width_deviation,
                orientation_mean,
                temporal_mismatch,
                heading_change_metric,
                same_side_step_mean,
                inferred_count,
                int(any(candidate.source_rank > 0 for candidate in chain_candidates)),
                tuple((candidate.left_idx, candidate.right_idx) for candidate in chain_candidates),
            )
            chain_options.append(
                _ChainSelection(
                    candidates=chain_candidates,
                    used_fallback=any(candidate.source_rank > 0 for candidate in chain_candidates),
                    expected_width_prior_m=expected_width_prior_m,
                    width_median=float(np.median([candidate.width_m for candidate in chain_candidates])),
                    near_field_lateral_max_m=continuity['max_lateral_m'],
                    near_field_lateral_mean_m=continuity['mean_lateral_m'],
                    near_field_displacement_max_m=continuity['max_displacement_m'],
                    near_field_displacement_mean_m=continuity['mean_displacement_m'],
                    near_field_kink_max_rad=near_field_kink_max_rad,
                    seed_midpoint_distance_m=seed_distance,
                    seed_temporal_offset_m=seed.seed_temporal_offset_m,
                    score=score,
                )
            )
        if chain_options:
            return min(chain_options, key=lambda option: option.score)
    return None


def _chain_has_minimum_useful_length(
    chain_candidates: list[_DiagonalCandidate],
    *,
    seed_distance_m: float,
    config: CoreConfig,
) -> bool:
    if len(chain_candidates) >= int(config.min_cross_edges):
        return True
    if len(chain_candidates) < 2:
        return False

    raw_midpoints = np.asarray([candidate.midpoint for candidate in chain_candidates], dtype=np.float64)
    span_m = float(np.sum(np.hypot(np.diff(raw_midpoints[:, 0]), np.diff(raw_midpoints[:, 1]))))
    min_span_m = max(1.0, 2.0 * float(config.path_resolution_m))
    near_seed_limit_m = min(float(config.max_seed_midpoint_distance_m), 3.5)

    # When only two local diagonals remain, keep the short stable near-field segment instead of
    # dropping to no path. A conservative short path is better than losing the controller-facing
    # lines completely while the next cones are still being recovered.
    return seed_distance_m <= near_seed_limit_m and span_m >= min_span_m


def _grow_chain(
    *,
    seed: _DiagonalCandidate,
    points: np.ndarray,
    candidates: list[_DiagonalCandidate],
    prior_midpoints: Optional[np.ndarray],
    vehicle_yaw: float,
    fallback_tangent: np.ndarray,
    expected_width_prior_m: float,
    initial_share_side: str,
    config: CoreConfig,
) -> list[_DiagonalCandidate]:
    chain = [seed]
    used_edges = {seed.edge}
    used_left = {seed.left_idx}
    used_right = {seed.right_idx}
    current = seed
    share_side = initial_share_side
    running_width = expected_width_prior_m
    prev_midpoint = seed.midpoint
    prev_heading = _reference_tangent_for_point(
        prior_midpoints,
        prev_midpoint,
        fallback_tangent=fallback_tangent,
        fallback_yaw=vehicle_yaw,
    )

    while True:
        next_candidates: list[tuple[tuple, _DiagonalCandidate, np.ndarray]] = []
        for candidate in candidates:
            if candidate.edge in used_edges:
                continue
            if share_side == 'left':
                if candidate.left_idx != current.left_idx or candidate.right_idx in used_right:
                    continue
                same_side_step_m = float(
                    np.hypot(*(points[candidate.right_idx] - points[current.right_idx]))
                )
                introduced_progress = _forward_progress(
                    points[candidate.right_idx] - points[current.right_idx],
                    prior_midpoints,
                    points[candidate.right_idx],
                    fallback_tangent,
                    vehicle_yaw,
                )
            else:
                if candidate.right_idx != current.right_idx or candidate.left_idx in used_left:
                    continue
                same_side_step_m = float(
                    np.hypot(*(points[candidate.left_idx] - points[current.left_idx]))
                )
                introduced_progress = _forward_progress(
                    points[candidate.left_idx] - points[current.left_idx],
                    prior_midpoints,
                    points[candidate.left_idx],
                    fallback_tangent,
                    vehicle_yaw,
                )
            if same_side_step_m > float(config.max_same_side_step_m):
                continue
            midpoint_delta = candidate.midpoint - prev_midpoint
            midpoint_progress = _forward_progress(
                midpoint_delta,
                prior_midpoints,
                candidate.midpoint,
                fallback_tangent,
                vehicle_yaw,
            )
            if midpoint_progress < float(config.min_midpoint_progress_m):
                continue
            if introduced_progress < 0.0:
                continue
            heading_change = _vector_angle(prev_heading, _unit_vector(midpoint_delta))
            width_deviation = abs(candidate.width_m - running_width)
            rank = (
                width_deviation,
                candidate.orientation_error,
                heading_change,
                same_side_step_m,
                candidate.inferred_count,
                candidate.source_rank,
                candidate.left_idx,
                candidate.right_idx,
            )
            next_candidates.append((rank, candidate, _unit_vector(midpoint_delta)))

        if not next_candidates:
            break
        next_candidates.sort(key=lambda item: item[0])
        _, chosen, chosen_heading = next_candidates[0]
        chain.append(chosen)
        used_edges.add(chosen.edge)
        used_left.add(chosen.left_idx)
        used_right.add(chosen.right_idx)
        running_width = _update_expected_width(
            running_width,
            chosen.width_m,
            config.max_width_prior_step_drift_m,
        )
        prev_midpoint = chosen.midpoint
        prev_heading = chosen_heading if np.linalg.norm(chosen_heading) > 1e-6 else prev_heading
        current = chosen
        share_side = 'right' if share_side == 'left' else 'left'
    return chain


def _candidate_array(candidates: list[_DiagonalCandidate]) -> np.ndarray:
    if not candidates:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray([[candidate.left_idx, candidate.right_idx] for candidate in candidates], dtype=np.int64)


def _build_boundary_near_field_pairs(
    *,
    points: np.ndarray,
    left_boundary_idx: np.ndarray,
    right_boundary_idx: np.ndarray,
    vehicle_xy: tuple[float, float],
    reference_tangent: np.ndarray,
    config: CoreConfig,
) -> np.ndarray:
    if left_boundary_idx.size == 0 or right_boundary_idx.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    vehicle = np.asarray(vehicle_xy, dtype=np.float64)
    tangent = _unit_vector(reference_tangent)
    progress_limit = max(
        float(config.max_seed_midpoint_distance_m) + float(config.max_same_side_step_m),
        12.0,
    )
    left_local = _near_field_boundary_indices(points, left_boundary_idx, vehicle, tangent, progress_limit)
    right_local = _near_field_boundary_indices(points, right_boundary_idx, vehicle, tangent, progress_limit)
    if left_local.size == 0 or right_local.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    pair_limit = min(left_local.size, right_local.size)
    pairs: set[tuple[int, int]] = set()
    # Keep a tight local ladder in front of the car so the first few meters do not lose
    # opposite-side structure when triangulation flips or drops a local diagonal.
    for idx in range(pair_limit):
        left_idx = int(left_local[idx])
        right_idx = int(right_local[idx])
        pairs.add(tuple(sorted((left_idx, right_idx))))
        if idx + 1 < right_local.size:
            pairs.add(tuple(sorted((left_idx, int(right_local[idx + 1])))))
        if idx + 1 < left_local.size:
            pairs.add(tuple(sorted((int(left_local[idx + 1]), right_idx))))

    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


def _near_field_boundary_indices(
    points: np.ndarray,
    ordered_idx: np.ndarray,
    vehicle_xy: np.ndarray,
    reference_tangent: np.ndarray,
    progress_limit: float,
) -> np.ndarray:
    selected: list[int] = []
    for idx in ordered_idx:
        point = points[int(idx)]
        progress = float(np.dot(point - vehicle_xy, reference_tangent))
        if progress < -1.0:
            continue
        if progress > progress_limit and selected:
            break
        selected.append(int(idx))
        if len(selected) >= 6:
            break
    return np.asarray(selected, dtype=np.int64)


def _build_local_opposite_pairs(
    *,
    points: np.ndarray,
    colors: list[str],
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    fallback_tangent: np.ndarray,
    prior_midpoints: Optional[np.ndarray],
    config: CoreConfig,
) -> np.ndarray:
    left_idx = [idx for idx, color in enumerate(colors) if color == 'blue']
    right_idx = [idx for idx, color in enumerate(colors) if color == 'yellow']
    if not left_idx or not right_idx:
        return np.empty((0, 2), dtype=np.int64)

    reference_tangent = _reference_tangent_for_point(
        prior_midpoints,
        np.asarray(vehicle_xy, dtype=np.float64),
        fallback_tangent=fallback_tangent,
        fallback_yaw=vehicle_yaw,
    )
    pairs: set[tuple[int, int]] = set()
    max_angle = float(max(0.0, config.local_opposite_forward_sector_rad))
    neighbor_count = max(1, int(config.local_opposite_neighbor_count))
    vehicle = np.asarray(vehicle_xy, dtype=np.float64)
    for idx in left_idx + right_idx:
        if idx in left_idx:
            opposite = right_idx
        else:
            opposite = left_idx
        distances: list[tuple[float, int]] = []
        for other_idx in opposite:
            midpoint = 0.5 * (points[idx] + points[other_idx])
            rel = midpoint - vehicle
            if float(np.dot(rel, reference_tangent)) < 0.0:
                continue
            angle = _vector_angle(reference_tangent, _unit_vector(rel))
            if angle > max_angle:
                continue
            distances.append((float(np.hypot(*(points[idx] - points[other_idx]))), other_idx))
        distances.sort(key=lambda item: item[0])
        for _dist, other_idx in distances[:neighbor_count]:
            a = min(idx, other_idx)
            b = max(idx, other_idx)
            pairs.add((a, b))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


def _candidate_preferred(candidate: _DiagonalCandidate, existing: _DiagonalCandidate) -> bool:
    return (
        candidate.source_rank,
        candidate.inferred_count,
        candidate.orientation_error,
        candidate.corridor_offset_m,
        candidate.left_idx,
        candidate.right_idx,
    ) < (
        existing.source_rank,
        existing.inferred_count,
        existing.orientation_error,
        existing.corridor_offset_m,
        existing.left_idx,
        existing.right_idx,
    )


def _candidate_sort_key(candidate: _DiagonalCandidate) -> tuple:
    return (
        candidate.source_rank,
        candidate.midpoint_progress_m,
        candidate.corridor_offset_m,
        candidate.orientation_error,
        candidate.left_idx,
        candidate.right_idx,
    )


def _seed_sort_key(candidate: _DiagonalCandidate, vehicle_xy: tuple[float, float]) -> tuple:
    seed_distance = float(
        np.hypot(candidate.midpoint_x - float(vehicle_xy[0]), candidate.midpoint_y - float(vehicle_xy[1]))
    )
    return (
        candidate.source_rank,
        candidate.corridor_offset_m,
        seed_distance,
        candidate.orientation_error,
        candidate.inferred_count,
        candidate.seed_temporal_offset_m,
        candidate.left_idx,
        candidate.right_idx,
    )


def _initial_expected_width(widths: np.ndarray, prior_width_m: Optional[float]) -> float:
    if prior_width_m is not None and math.isfinite(float(prior_width_m)) and float(prior_width_m) > 0.0:
        return float(prior_width_m)
    if widths.size == 0:
        return float('nan')
    return float(np.median(widths))


def _update_expected_width(current_width_m: float, new_width_m: float, max_step_drift_m: float) -> float:
    # Keep the width prior deliberately slow so noisy pre-corner frames cannot drag it onto a bad chain.
    if not math.isfinite(current_width_m):
        return float(new_width_m)
    delta = float(new_width_m) - float(current_width_m)
    limited = float(np.clip(delta, -abs(max_step_drift_m), abs(max_step_drift_m)))
    return float(current_width_m + limited)


def _near_field_continuity_metrics(
    *,
    current_path: np.ndarray,
    previous_path: Optional[np.ndarray],
    spacing_m: float,
    point_count: int,
) -> dict[str, float]:
    if previous_path is None:
        return {
            'max_lateral_m': 0.0,
            'mean_lateral_m': 0.0,
            'max_displacement_m': 0.0,
            'mean_displacement_m': 0.0,
        }
    current = _sanitize_path(current_path)
    previous = _sanitize_path(previous_path)
    if current is None or previous is None:
        return {
            'max_lateral_m': 0.0,
            'mean_lateral_m': 0.0,
            'max_displacement_m': 0.0,
            'mean_displacement_m': 0.0,
        }
    sample_count = max(2, int(point_count))
    spacing = max(0.05, float(spacing_m))
    max_length = spacing * float(sample_count - 1)
    current_rs = _resample_path(current, resolution_m=spacing, max_length_m=max_length)
    previous_rs = _resample_path(previous, resolution_m=spacing, max_length_m=max_length)
    count = min(current_rs.shape[0], previous_rs.shape[0], sample_count)
    if count < 2:
        return {
            'max_lateral_m': 0.0,
            'mean_lateral_m': 0.0,
            'max_displacement_m': 0.0,
            'mean_displacement_m': 0.0,
        }
    current_rs = current_rs[:count]
    previous_rs = previous_rs[:count]
    tangents = _path_tangents(previous_rs)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    delta = current_rs - previous_rs
    lateral = np.abs(np.sum(delta * normals, axis=1))
    displacement = np.hypot(delta[:, 0], delta[:, 1])
    return {
        'max_lateral_m': float(np.max(lateral)),
        'mean_lateral_m': float(np.mean(lateral)),
        'max_displacement_m': float(np.max(displacement)),
        'mean_displacement_m': float(np.mean(displacement)),
    }


def _max_local_kink(points: np.ndarray, near_field_point_count: int) -> float:
    path = _sanitize_path(points)
    if path is None or path.shape[0] < 3:
        return 0.0
    subset = path[: max(3, int(near_field_point_count))]
    if subset.shape[0] < 3:
        return 0.0
    segments = np.diff(subset, axis=0)
    headings = np.array([math.atan2(float(seg[1]), float(seg[0])) for seg in segments], dtype=np.float64)
    delta_heading = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    if delta_heading.size == 0:
        return 0.0
    return float(np.max(np.abs(delta_heading)))


def _path_heading_change_metric(points: np.ndarray) -> float:
    path = _sanitize_path(points)
    if path is None or path.shape[0] < 3:
        return 0.0
    segments = np.diff(path, axis=0)
    headings = np.array([math.atan2(float(seg[1]), float(seg[0])) for seg in segments], dtype=np.float64)
    delta_heading = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    if delta_heading.size == 0:
        return 0.0
    return float(np.mean(np.abs(delta_heading)))


def _mean_same_side_step(points: np.ndarray, chain: list[_DiagonalCandidate]) -> float:
    if len(chain) < 2:
        return 0.0
    steps: list[float] = []
    for prev, curr in zip(chain[:-1], chain[1:]):
        if prev.left_idx == curr.left_idx:
            steps.append(float(np.hypot(*(points[prev.right_idx] - points[curr.right_idx]))))
        elif prev.right_idx == curr.right_idx:
            steps.append(float(np.hypot(*(points[prev.left_idx] - points[curr.left_idx]))))
    if not steps:
        return 0.0
    return float(np.mean(steps))


def _path_tangents(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 1:
        return np.asarray([[1.0, 0.0]], dtype=np.float64)
    tangents = np.zeros_like(points)
    diffs = np.diff(points, axis=0)
    seg_unit = np.vstack([_unit_vector(diff) for diff in diffs])
    tangents[0] = seg_unit[0]
    tangents[-1] = seg_unit[-1]
    for idx in range(1, points.shape[0] - 1):
        tangents[idx] = _unit_vector(seg_unit[idx - 1] + seg_unit[idx])
    return tangents


def _seed_temporal_offset(midpoint: np.ndarray, prior_midpoints: Optional[np.ndarray]) -> float:
    if prior_midpoints is None or prior_midpoints.shape[0] == 0:
        return float('inf')
    return float(np.hypot(*(midpoint - prior_midpoints[0])))


def _estimate_startup_tangent(
    *,
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
) -> np.ndarray:
    candidates: list[np.ndarray] = []
    vehicle = np.asarray(vehicle_xy, dtype=np.float64)
    yaw_hint = _yaw_unit(vehicle_yaw)
    for boundary in (left_boundary, right_boundary):
        path = _sanitize_path(boundary)
        if path is None or path.shape[0] < 2:
            continue
        best_seg = None
        best_dist = float('inf')
        for idx in range(path.shape[0] - 1):
            a = path[idx]
            b = path[idx + 1]
            seg = b - a
            if np.linalg.norm(seg) <= 1e-9:
                continue
            midpoint = 0.5 * (a + b)
            dist = float(np.hypot(*(midpoint - vehicle)))
            if dist < best_dist:
                best_dist = dist
                best_seg = seg
        if best_seg is not None:
            seg_unit = _unit_vector(best_seg)
            if float(np.dot(seg_unit, yaw_hint)) < 0.0:
                seg_unit = -seg_unit
            candidates.append(seg_unit)

    if not candidates:
        return yaw_hint
    avg = _unit_vector(np.sum(np.asarray(candidates, dtype=np.float64), axis=0))
    if float(np.dot(avg, yaw_hint)) < 0.0:
        avg = -avg
    return avg


def _reference_origin(prior_midpoints: Optional[np.ndarray], vehicle_xy: tuple[float, float]) -> np.ndarray:
    # Progress gating must protect the controller-facing segment relative to the car's current
    # position, not relative to last frame's first midpoint. Otherwise freshly visible cones one or
    # two meters ahead can be treated as "behind" and disappear right in front of the vehicle.
    return np.asarray(vehicle_xy, dtype=np.float64)


def _reference_tangent_for_point(
    path: Optional[np.ndarray],
    point_xy: np.ndarray,
    *,
    fallback_tangent: Optional[np.ndarray] = None,
    fallback_yaw: float,
) -> np.ndarray:
    sanitized = _sanitize_path(path)
    if sanitized is None or sanitized.shape[0] < 2:
        if fallback_tangent is not None and np.linalg.norm(fallback_tangent) > 1e-9:
            return _unit_vector(fallback_tangent)
        return _yaw_unit(fallback_yaw)
    if sanitized.shape[0] == 2:
        return _unit_vector(sanitized[1] - sanitized[0])
    best_idx = 0
    best_dist = float('inf')
    for idx in range(sanitized.shape[0] - 1):
        projected = _project_point_to_segment(point_xy, sanitized[idx], sanitized[idx + 1])
        dist = float(np.hypot(*(point_xy - projected)))
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return _unit_vector(sanitized[best_idx + 1] - sanitized[best_idx])


def _project_point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return np.asarray(a, dtype=np.float64)
    t = float(np.dot(point - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    return a + (t * ab)


def _forward_progress(
    delta_xy: np.ndarray,
    prior_midpoints: Optional[np.ndarray],
    reference_point_xy: np.ndarray,
    fallback_tangent: np.ndarray,
    vehicle_yaw: float,
) -> float:
    tangent = _reference_tangent_for_point(
        prior_midpoints,
        reference_point_xy,
        fallback_tangent=fallback_tangent,
        fallback_yaw=vehicle_yaw,
    )
    return float(np.dot(delta_xy, tangent))


def _sanitize_path(path: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.asarray(path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 2:
        return None
    return arr


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


def _deterministic_point_order(points: np.ndarray, colors: list[str]) -> np.ndarray:
    color_index = {'blue': 0, 'yellow': 1, 'orange': 2, 'unknown': 3}
    color_rank = np.asarray([color_index.get(color, 9) for color in colors], dtype=np.int64)
    order = np.lexsort((color_rank, np.round(points[:, 1], 3), np.round(points[:, 0], 3)))
    return np.asarray(order, dtype=np.int64)
def _build_edges(points: np.ndarray) -> tuple[np.ndarray, bool]:
    if points.shape[0] < 3:
        return np.empty((0, 2), dtype=np.int64), False

    tri_edges: set[tuple[int, int]] = set()
    if _ScipyTriangulation is not None:
        try:
            tri = _ScipyTriangulation(points)
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


def _canonical_boundary_pair(a: int, b: int, colors: list[str]) -> tuple[Optional[int], Optional[int]]:
    color_a = colors[int(a)]
    color_b = colors[int(b)]
    if color_a == 'blue' and color_b == 'yellow':
        return int(a), int(b)
    if color_a == 'yellow' and color_b == 'blue':
        return int(b), int(a)
    return None, None


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
    *,
    max_step_m: float,
) -> np.ndarray:
    tangent = _yaw_unit(vehicle_yaw)
    ordered_idx = _order_boundary_indices(
        points,
        colors,
        boundary_color,
        vehicle_xy,
        tangent,
        max_step_m=max_step_m,
    )
    if ordered_idx.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return points[ordered_idx]


def _order_boundary_indices(
    points: np.ndarray,
    colors: list[str],
    boundary_color: str,
    vehicle_xy: tuple[float, float],
    reference_tangent: np.ndarray,
    *,
    max_step_m: float,
) -> np.ndarray:
    idx = np.asarray([i for i, c in enumerate(colors) if c == boundary_color], dtype=np.int64)
    if idx.size == 0:
        return np.empty((0,), dtype=np.int64)
    subset = points[idx]
    ordered_local = _order_boundary_locally(
        subset,
        vehicle_xy=vehicle_xy,
        reference_tangent=reference_tangent,
        max_step_m=max_step_m,
    )
    if ordered_local.size == 0:
        return np.empty((0,), dtype=np.int64)
    return idx[ordered_local]


def _order_boundary_locally(
    points: np.ndarray,
    *,
    vehicle_xy: tuple[float, float],
    reference_tangent: np.ndarray,
    max_step_m: float,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if points.shape[0] == 1:
        return np.asarray([0], dtype=np.int64)

    tangent = _unit_vector(reference_tangent)
    vehicle = np.asarray(vehicle_xy, dtype=np.float64)
    rel = points - vehicle.reshape(1, 2)
    progress = rel @ tangent
    corridor = np.abs(np.cross(tangent, rel))
    dist_vehicle = np.hypot(rel[:, 0], rel[:, 1])

    start_candidates = np.nonzero(progress >= -1.0)[0]
    if start_candidates.size == 0:
        start_candidates = np.arange(points.shape[0], dtype=np.int64)
    start = min(
        start_candidates.tolist(),
        key=lambda idx: (
            dist_vehicle[idx],
            abs(min(progress[idx], 0.0)),
            corridor[idx],
            idx,
        ),
    )

    ordered = [int(start)]
    used = {int(start)}
    prev_heading = tangent
    max_step = max(0.5, float(max_step_m))

    # Keep the same-side boundary locally connected so a far cone with similar progress does not
    # get interleaved into the near-field ladder and erase the lines directly in front of the car.
    while True:
        current = ordered[-1]
        next_candidates: list[tuple[tuple, int, np.ndarray]] = []
        for idx in range(points.shape[0]):
            if idx in used:
                continue
            step_vec = points[idx] - points[current]
            step_dist = float(np.hypot(step_vec[0], step_vec[1]))
            if step_dist <= 1e-6 or step_dist > max_step:
                continue
            progress_delta = float(progress[idx] - progress[current])
            if progress_delta < -0.5:
                continue
            heading = _unit_vector(step_vec)
            score = (
                0 if progress_delta >= 0.0 else 1,
                step_dist,
                _vector_angle(prev_heading, heading),
                abs(progress_delta),
                corridor[idx],
                idx,
            )
            next_candidates.append((score, idx, heading))

        if not next_candidates:
            break
        next_candidates.sort(key=lambda item: item[0])
        _, chosen_idx, chosen_heading = next_candidates[0]
        ordered.append(int(chosen_idx))
        used.add(int(chosen_idx))
        prev_heading = chosen_heading

    return np.asarray(ordered, dtype=np.int64)


def _order_forward_points(
    points: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    behind_drop_m: float,
) -> np.ndarray:
    tangent = _yaw_unit(vehicle_yaw)
    order = _order_points_along_tangent(
        points,
        vehicle_xy=vehicle_xy,
        reference_tangent=tangent,
        behind_drop_m=behind_drop_m,
    )
    if order.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return points[order]


def _order_points_along_tangent(
    points: np.ndarray,
    *,
    vehicle_xy: tuple[float, float],
    reference_tangent: np.ndarray,
    behind_drop_m: float,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    tangent = _unit_vector(reference_tangent)
    rel = points - np.asarray(vehicle_xy, dtype=np.float64).reshape(1, 2)
    progress = rel @ tangent
    keep = progress >= -float(behind_drop_m)
    kept_idx = np.nonzero(keep)[0]
    if kept_idx.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(progress[kept_idx])
    return kept_idx[order]


def _rotate_into_vehicle(vec_xy: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    vx = cos_y * vec_xy[:, 0] + sin_y * vec_xy[:, 1]
    vy = -sin_y * vec_xy[:, 0] + cos_y * vec_xy[:, 1]
    return vx, vy


def _yaw_unit(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)


def _unit_vector(vec_xy: np.ndarray) -> np.ndarray:
    norm = float(np.hypot(vec_xy[0], vec_xy[1]))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return np.asarray(vec_xy, dtype=np.float64) / norm


def _vector_angle(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    ua = _unit_vector(a_xy)
    ub = _unit_vector(b_xy)
    dot = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    return float(math.acos(dot))
