"""Shared algorithm helpers for planner core modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math
from typing import Any, Optional

import numpy as np

from sim_car.cones.tracking.fusion import normalize_color

_INITIAL_REJECT_COUNT = 0  # Reject counters start at zero before candidate checks run.
_COLOR_RANK_BLUE = 0  # Blue cones sort before yellow for deterministic left/right pairing.
_COLOR_RANK_YELLOW = 1  # Yellow cones sort after blue but before unknown cones.
_COLOR_RANK_OTHER = 2  # Unknown/other colors sort last so known boundaries win ties.
_MIN_RESAMPLE_RESOLUTION_M = 0.05  # Resampling below 5 cm amplifies jitter without useful detail.
_RESAMPLE_ENDPOINT_EPSILON_M = 1e-9  # Includes the exact endpoint despite floating-point roundoff.
_MIN_PATH_LENGTH_M = 1e-6  # Sub-micrometer paths are degenerate and collapse to their first point.
_MIN_HEADING_DELTA_DISTANCE_M = 1e-9  # Treats sub-nanometer movement as no heading information.


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))


def _default_reject_counts(reason_keys: Iterable[str]) -> dict[str, int]:
    return {str(key): _INITIAL_REJECT_COUNT for key in reason_keys}


def _deterministic_order(
    local_points: np.ndarray,
    global_points: np.ndarray,
    colors: list[str],
) -> np.ndarray:
    color_rank = np.asarray(
        [
            _COLOR_RANK_BLUE
            if color == "blue"
            else _COLOR_RANK_YELLOW
            if color == "yellow"
            else _COLOR_RANK_OTHER
            for color in colors
        ],
        dtype=np.int64,
    )
    return np.lexsort(
        (
            global_points[:, 1],
            global_points[:, 0],
            local_points[:, 1],
            np.abs(local_points[:, 1]),
            local_points[:, 0],
            color_rank,
        )
    )


def _filter_and_order_cones(
    *,
    points_xy: np.ndarray,
    colors: list[str],
    confidences: np.ndarray,
    track_ids: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: Any,
    filtered_cones_type: type,
    geometry_filter: Callable[[np.ndarray, Any], np.ndarray],
    include_unknown: bool,
    raw_colors: Optional[list[str]] = None,
    include_raw_colors: bool = False,
) -> Any:
    from sim_car.planning.tracked_cone_planner_geometry import to_vehicle_frame as _to_vehicle_frame

    normalized = [normalize_color(c) for c in colors]
    normalized_raw = _normalized_raw_colors(raw_colors, colors, normalized)
    local_points = _to_vehicle_frame(points_xy, vehicle_xy, vehicle_yaw)
    selected_mask = _cone_selection_mask(
        local_points=local_points,
        confidences=confidences,
        normalized_colors=normalized,
        config=config,
        geometry_filter=geometry_filter,
        include_unknown=include_unknown,
    )
    fields = _ordered_filtered_cone_fields(
        points_xy=points_xy,
        local_points=local_points,
        track_ids=track_ids,
        normalized_colors=normalized,
        normalized_raw_colors=normalized_raw,
        selected_mask=selected_mask,
        include_raw_colors=include_raw_colors,
    )
    return filtered_cones_type(**fields)


def _ordered_filtered_cone_fields(
    *,
    points_xy: np.ndarray,
    local_points: np.ndarray,
    track_ids: np.ndarray,
    normalized_colors: list[str],
    normalized_raw_colors: list[str],
    selected_mask: np.ndarray,
    include_raw_colors: bool,
) -> dict[str, Any]:
    indices = np.where(selected_mask)[0]
    filtered_points = np.asarray(points_xy[selected_mask], dtype=np.float64)
    filtered_local = np.asarray(local_points[selected_mask], dtype=np.float64)
    filtered_track_ids = np.asarray(track_ids[selected_mask], dtype=np.int64)
    filtered_colors = [normalized_colors[i] for i in indices]
    colored_count = int(np.count_nonzero(
        np.array([c in {"blue", "yellow"} for c in filtered_colors], dtype=bool)
    ))

    order = _deterministic_order(filtered_local, filtered_points, filtered_colors)
    fields = {
        "points": filtered_points[order],
        "local": filtered_local[order],
        "track_ids": filtered_track_ids[order],
        "colors": [filtered_colors[i] for i in order],
        "colored_count": colored_count,
    }
    if include_raw_colors:
        filtered_raw_colors = [normalized_raw_colors[i] for i in indices]
        fields["raw_colors"] = [filtered_raw_colors[i] for i in order]
    return fields


def _normalized_raw_colors(
    raw_colors: Optional[list[str]],
    colors: list[str],
    normalized_colors: list[str],
) -> list[str]:
    if raw_colors is not None and len(raw_colors) == len(colors):
        return [normalize_color(c) for c in raw_colors]
    return list(normalized_colors)


def _cone_selection_mask(
    *,
    local_points: np.ndarray,
    confidences: np.ndarray,
    normalized_colors: list[str],
    config: Any,
    geometry_filter: Callable[[np.ndarray, Any], np.ndarray],
    include_unknown: bool,
) -> np.ndarray:
    mask_geom = geometry_filter(local_points, config)
    mask_conf = confidences >= float(config.min_confidence)
    colored_mask = np.array([c in {"blue", "yellow"} for c in normalized_colors], dtype=bool)
    if not include_unknown:
        return mask_geom & mask_conf & colored_mask
    unknown_mask = np.array([c == "unknown" for c in normalized_colors], dtype=bool)
    return mask_geom & mask_conf & (
        colored_mask | (unknown_mask if config.allow_unknown_pair_completion else False)
    )


def _build_boundary_chain(
    *,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    side_indices: np.ndarray,
    config: Any,
    boundary_chain_type: type,
    collect_rejection_reasons: bool = False,
    min_step_m: Optional[float] = None,
) -> Any:
    from sim_car.planning.tracked_cone_planner_geometry import build_boundary_chain_data

    chain = build_boundary_chain_data(
        filtered_points=filtered_points,
        filtered_local=filtered_local,
        side_indices=side_indices,
        config=config,
        collect_rejection_reasons=collect_rejection_reasons,
        min_step_m=min_step_m,
    )
    fields = {
        "filtered_indices": chain.filtered_indices,
        "global_points": chain.global_points,
        "local_points": chain.local_points,
        "tangents_local": chain.tangents_local,
        "mean_heading_change_rad": chain.mean_heading_change_rad,
        "forward_extent_m": chain.forward_extent_m,
    }
    if collect_rejection_reasons:
        fields["rejected_reasons_by_filtered_index"] = dict(
            chain.rejected_reasons_by_filtered_index
        )
    return boundary_chain_type(**fields)


def _resample_path(points: np.ndarray, resolution_m: float, max_length_m: float) -> np.ndarray:
    from sim_car.planning.tracked_cone_planner_geometry import (
        path_cumulative_lengths as _path_cumulative_lengths,
    )

    if points.shape[0] <= 1:
        return np.asarray(points, dtype=np.float64)

    cumulative = _path_cumulative_lengths(points)
    total = min(float(cumulative[-1]), float(max_length_m))
    if total <= _MIN_PATH_LENGTH_M:
        return np.asarray(points[:1], dtype=np.float64)

    step = max(_MIN_RESAMPLE_RESOLUTION_M, float(resolution_m))
    samples = np.arange(
        0.0,
        total + _RESAMPLE_ENDPOINT_EPSILON_M,
        step,
        dtype=np.float64,
    )
    if samples.size == 0 or samples[-1] < total:
        samples = np.concatenate((samples, [total]))
    x = np.interp(samples, cumulative, points[:, 0])
    y = np.interp(samples, cumulative, points[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


def _finalize_path(
    points: np.ndarray,
    config: Any,
    *,
    smoothing_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    path = np.asarray(points, dtype=np.float64)
    active_smoothing_fn = smoothing_fn
    if active_smoothing_fn is None and hasattr(config, "smoothing_window"):
        from sim_car.planning.tracked_cone_planner_geometry import moving_average as _moving_average

        active_smoothing_fn = _moving_average
    if active_smoothing_fn is not None and int(config.smoothing_window) > 1:
        path = active_smoothing_fn(path, int(config.smoothing_window))
    return _resample_path(
        path,
        resolution_m=float(config.path_resolution_m),
        max_length_m=float(config.max_path_length_m),
    )


def _path_length(points: np.ndarray) -> float:
    from sim_car.planning.tracked_cone_planner_geometry import (
        path_cumulative_lengths as _path_cumulative_lengths,
    )

    cumulative = _path_cumulative_lengths(np.asarray(points, dtype=np.float64))
    return float(cumulative[-1]) if cumulative.size > 0 else 0.0


def _path_start_heading_error(path_local: np.ndarray) -> float:
    if path_local.shape[0] < 2:
        return 0.0
    delta = path_local[1] - path_local[0]
    if float(np.hypot(delta[0], delta[1])) <= _MIN_HEADING_DELTA_DISTANCE_M:
        return 0.0
    return float(math.atan2(float(delta[1]), float(delta[0])))


def _forward_extent_m(path_local: np.ndarray) -> float:
    if path_local.shape[0] == 0:
        return 0.0
    x_span = float(np.max(path_local[:, 0]) - np.min(path_local[:, 0]))
    if path_local.shape[0] < 2:
        return x_span
    diffs = np.diff(path_local, axis=0)
    path_length = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    return max(x_span, path_length)


def _first_point_distance(path_local: np.ndarray) -> float:
    if path_local.shape[0] == 0:
        return float("nan")
    return float(np.hypot(path_local[0, 0], path_local[0, 1]))


def _merge_reject_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)


def _empty_result_fields(
    *,
    filtered_points: Optional[np.ndarray] = None,
    filtered_colors: Optional[list[str]] = None,
    left_boundary: Optional[np.ndarray] = None,
    right_boundary: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    return {
        "filtered_points": (
            np.asarray(filtered_points, dtype=np.float64)
            if filtered_points is not None
            else np.empty((0, 2), dtype=np.float64)
        ),
        "filtered_colors": list(filtered_colors or []),
        "candidate_edges": np.empty((0, 2), dtype=np.int64),
        "selected_edges": np.empty((0, 2), dtype=np.int64),
        "selected_pair_track_ids": np.empty((0, 2), dtype=np.int64),
        "midpoints_raw": np.empty((0, 2), dtype=np.float64),
        "centerline": np.empty((0, 2), dtype=np.float64),
        "left_boundary": (
            np.asarray(left_boundary, dtype=np.float64)
            if left_boundary is not None
            else np.empty((0, 2), dtype=np.float64)
        ),
        "right_boundary": (
            np.asarray(right_boundary, dtype=np.float64)
            if right_boundary is not None
            else np.empty((0, 2), dtype=np.float64)
        ),
        "used_fallback": False,
    }
