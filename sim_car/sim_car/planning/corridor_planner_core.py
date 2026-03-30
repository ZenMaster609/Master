"""Core geometry for the corridor planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from sim_car.cones.tracking.fusion import normalize_color


@dataclass
class CorridorPlannerConfig:
    max_cone_range_m: float = 25.0
    planning_horizon_m: float = 25.0
    max_lateral_range_m: float = 8.0
    behind_drop_m: float = 5.0
    min_confidence: float = 0.3
    min_required_cones: int = 4

    min_step_m: float = 0.8
    max_step_m: float = 6.0
    max_heading_change_rad: float = 1.0
    min_forward_progress_m: float = 0.2
    min_chain_length: int = 3

    initial_width_m: float = 3.6
    min_width_m: float = 2.4
    max_width_m: float = 4.8
    width_filter_alpha: float = 0.15
    max_width_delta_per_update_m: float = 0.2
    min_trustworthy_pairs: int = 3

    boundary_resample_dx: float = 0.5
    min_corridor_width_m: float = 2.2
    max_corridor_width_m: float = 5.5
    min_required_corridor_samples: int = 5
    path_fit_smoothing_window: int = 5
    membership_margin_m: float = 0.15

    path_resolution_m: float = 0.5
    max_path_length_m: float = 30.0

    min_path_points: int = 4
    min_forward_extent_m: float = 2.0
    jump_check_horizon_m: float = 8.0
    max_near_field_lateral_jump_m: float = 0.8
    max_heading_delta_rad: float = 0.75
    max_initial_heading_error_rad: float = 1.0
    max_curvature: float = 0.45


@dataclass
class CorridorPlannerPrior:
    previous_centerline: Optional[np.ndarray] = None
    previous_width_m: Optional[float] = None
    previous_mode: str = "none"


@dataclass
class CorridorPlannerResult:
    filtered_points: np.ndarray
    filtered_colors: list[str]
    triangulation_edges: np.ndarray
    candidate_edges: np.ndarray
    selected_edges: np.ndarray
    selected_pair_track_ids: np.ndarray
    midpoints_raw: np.ndarray
    centerline: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    used_fallback: bool
    status: str
    candidate_count: int = 0
    selected_chain_length: int = 0
    selected_chain_width_median: float = float("nan")
    expected_width_prior_m: float = float("nan")
    near_field_lateral_max_m: float = 0.0
    near_field_lateral_mean_m: float = 0.0
    near_field_displacement_max_m: float = 0.0
    near_field_displacement_mean_m: float = 0.0
    near_field_kink_max_rad: float = 0.0
    seed_midpoint_distance_m: float = float("nan")
    seed_temporal_offset_m: float = float("nan")
    reject_reason: str = ""
    reject_counts: dict[str, int] = field(default_factory=dict)
    planner_mode: str = "none"
    active_boundary_side: str = ""
    raw_offset_path: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    pair_segments: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2, 2), dtype=np.float64)
    )
    accepted_pair_count: int = 0
    left_chain_length: int = 0
    right_chain_length: int = 0
    filtered_track_width_m: float = float("nan")
    unknown_pair_count: int = 0
    corridor_width_min_m: float = float("nan")
    corridor_width_max_m: float = float("nan")


@dataclass
class _BoundaryChain:
    filtered_indices: np.ndarray
    global_points: np.ndarray
    local_points: np.ndarray
    tangents_local: np.ndarray
    mean_heading_change_rad: float
    forward_extent_m: float


def compute_corridor_centerline(
    points_xy: np.ndarray,
    colors: list[str],
    confidences: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: CorridorPlannerConfig,
    prior: Optional[CorridorPlannerPrior] = None,
    track_ids: Optional[np.ndarray] = None,
) -> CorridorPlannerResult:
    if points_xy.size == 0:
        return _empty_result("no cones available")

    if track_ids is None or len(track_ids) != points_xy.shape[0]:
        track_ids = np.arange(points_xy.shape[0], dtype=np.int64)
    else:
        track_ids = np.asarray(track_ids, dtype=np.int64)

    normalized = [normalize_color(color) for color in colors]
    local_points = _to_vehicle_frame(points_xy, vehicle_xy, vehicle_yaw)

    mask_geom = _geometry_filter(local_points, config)
    mask_conf = confidences >= float(config.min_confidence)
    colored_mask = np.array([color in {"blue", "yellow"} for color in normalized], dtype=bool)
    selected_mask = mask_geom & mask_conf & colored_mask

    filtered_points = np.asarray(points_xy[selected_mask], dtype=np.float64)
    filtered_local = np.asarray(local_points[selected_mask], dtype=np.float64)
    filtered_track_ids = np.asarray(track_ids[selected_mask], dtype=np.int64)
    filtered_colors = [normalized[idx] for idx in np.where(selected_mask)[0]]
    colored_count = int(np.count_nonzero(colored_mask[selected_mask]))
    if colored_count == 0:
        return _empty_result(
            "no colored cones in planning region",
            reject_counts=_default_reject_counts(),
        )

    order = _deterministic_order(filtered_local, filtered_points, filtered_colors)
    filtered_points = filtered_points[order]
    filtered_local = filtered_local[order]
    filtered_track_ids = filtered_track_ids[order]
    filtered_colors = [filtered_colors[idx] for idx in order]

    if colored_count < int(config.min_required_cones):
        return _empty_result(
            f"usable colored cones below minimum ({colored_count} < {int(config.min_required_cones)})",
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            reject_counts=_default_reject_counts(),
        )

    left_indices = np.flatnonzero(np.array([color == "blue" for color in filtered_colors], dtype=bool))
    right_indices = np.flatnonzero(np.array([color == "yellow" for color in filtered_colors], dtype=bool))
    left_chain = _build_boundary_chain(filtered_points, filtered_local, left_indices, config)
    right_chain = _build_boundary_chain(filtered_points, filtered_local, right_indices, config)

    reject_counts = _default_reject_counts()
    expected_width = _clamp(
        prior.previous_width_m if prior and prior.previous_width_m is not None else config.initial_width_m,
        config.min_width_m,
        config.max_width_m,
    )

    if (
        left_chain.filtered_indices.size < int(config.min_chain_length)
        or right_chain.filtered_indices.size < int(config.min_chain_length)
    ):
        return _result_with_metadata(
            result=_empty_result(
                "no reliable corridor boundaries",
                filtered_points=filtered_points,
                filtered_colors=filtered_colors,
                left_boundary=left_chain.global_points,
                right_boundary=right_chain.global_points,
                reject_counts=reject_counts,
                reject_reason="no reliable corridor boundaries",
            ),
            left_chain=left_chain,
            right_chain=right_chain,
            planner_mode="none",
            filtered_track_width_m=expected_width,
        )

    corridor = _build_corridor(
        left_chain=left_chain,
        right_chain=right_chain,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        config=config,
    )
    if corridor is None:
        reject_counts["corridor_geometry"] += 1
        return _result_with_metadata(
            result=_empty_result(
                "no valid corridor overlap",
                filtered_points=filtered_points,
                filtered_colors=filtered_colors,
                left_boundary=left_chain.global_points,
                right_boundary=right_chain.global_points,
                reject_counts=reject_counts,
                reject_reason="no valid corridor overlap",
            ),
            left_chain=left_chain,
            right_chain=right_chain,
            planner_mode="none",
            filtered_track_width_m=expected_width,
        )

    corridor_sample_count = int(np.asarray(corridor["anchors_local"], dtype=np.float64).shape[0])
    left_boundary = np.asarray(corridor["left_global"], dtype=np.float64)
    right_boundary = np.asarray(corridor["right_global"], dtype=np.float64)
    center_anchors = np.asarray(corridor["anchors_global"], dtype=np.float64)
    raw_centerline = np.asarray(corridor["centerline_global"], dtype=np.float64)
    corridor_rungs = np.asarray(corridor["rungs_global"], dtype=np.float64)
    widths = np.asarray(corridor["widths_m"], dtype=np.float64)

    width_min_m = float(np.min(widths)) if widths.size else float("nan")
    width_max_m = float(np.max(widths)) if widths.size else float("nan")
    width_median_m = float(np.median(widths)) if widths.size else float("nan")

    if corridor_sample_count >= int(config.min_trustworthy_pairs) and math.isfinite(width_median_m):
        measured_width_m = width_median_m
    else:
        measured_width_m = float("nan")

    centerline = _resample_path(
        raw_centerline,
        resolution_m=float(config.path_resolution_m),
        max_length_m=float(config.max_path_length_m),
    )
    centerline_local = _to_vehicle_frame(centerline, vehicle_xy, vehicle_yaw)
    seed_distance_m = _first_point_distance(centerline_local)
    near_field = _near_field_delta_metrics(
        current=centerline,
        previous=None if prior is None else prior.previous_centerline,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        horizon_m=config.jump_check_horizon_m,
    )
    heading_delta_max = _path_heading_delta_max(centerline_local)
    curvature_max = _path_curvature_abs_max(centerline_local)

    status = "ok"
    reject_reason = ""
    if corridor_sample_count < int(config.min_required_corridor_samples):
        reject_counts["corridor_samples"] += 1
        status = "too few valid corridor samples"
        reject_reason = status
    elif centerline.shape[0] < int(config.min_path_points):
        status = "path has too few points"
        reject_reason = status
    elif not np.all(np.isfinite(centerline)):
        status = "path contains non-finite geometry"
        reject_reason = status
    elif _forward_extent_m(centerline_local) < float(config.min_forward_extent_m):
        status = "path forward extent too short"
        reject_reason = status
    elif _path_violates_corridor(
        centerline_local=centerline_local,
        corridor_center_local=np.asarray(corridor["anchors_local"], dtype=np.float64),
        corridor_widths_m=np.asarray(corridor["widths_m"], dtype=np.float64),
        membership_margin_m=float(config.membership_margin_m),
    ):
        reject_counts["path_outside_corridor"] += 1
        status = "path exits corridor"
        reject_reason = status
    else:
        initial_heading_error = abs(_path_start_heading_error(centerline_local))
        if initial_heading_error > float(config.max_initial_heading_error_rad):
            reject_counts["heading"] += 1
            status = "path heading flip near vehicle"
            reject_reason = status
        elif near_field["lateral_max_m"] > _effective_near_field_jump_limit(
            config=config,
            corridor_sample_count=corridor_sample_count,
        ):
            reject_counts["near_field_continuity"] += 1
            status = "near-field continuity rejected fresh path"
            reject_reason = status
        elif heading_delta_max > float(config.max_heading_delta_rad):
            reject_counts["heading"] += 1
            status = "path heading delta exceeded limit"
            reject_reason = status
        elif curvature_max > _effective_curvature_limit(
            config=config,
            corridor_sample_count=corridor_sample_count,
        ):
            reject_counts["curvature"] += 1
            status = "path curvature exceeded limit"
            reject_reason = status
        elif _path_self_intersects(centerline):
            reject_counts["corridor_geometry"] += 1
            status = "path self-crossing detected"
            reject_reason = status

    if status != "ok" and reject_reason not in {
        "path has too few points",
        "path forward extent too short",
        "near-field continuity rejected fresh path",
    }:
        centerline = np.empty((0, 2), dtype=np.float64)

    result = CorridorPlannerResult(
        filtered_points=filtered_points,
        filtered_colors=filtered_colors,
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.empty((0, 2), dtype=np.int64),
        selected_pair_track_ids=np.empty((0, 2), dtype=np.int64),
        midpoints_raw=center_anchors,
        centerline=centerline,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        used_fallback=False,
        status=status,
        candidate_count=corridor_sample_count,
        selected_chain_length=corridor_sample_count,
        selected_chain_width_median=width_median_m,
        expected_width_prior_m=float(expected_width),
        near_field_lateral_max_m=float(near_field["lateral_max_m"]),
        near_field_lateral_mean_m=float(near_field["lateral_mean_m"]),
        near_field_displacement_max_m=float(near_field["displacement_max_m"]),
        near_field_displacement_mean_m=float(near_field["displacement_mean_m"]),
        near_field_kink_max_rad=float(heading_delta_max),
        seed_midpoint_distance_m=float(seed_distance_m),
        seed_temporal_offset_m=float("nan"),
        reject_reason=reject_reason,
        reject_counts=reject_counts,
        planner_mode="corridor" if status == "ok" else "none",
        active_boundary_side="",
        raw_offset_path=np.empty((0, 2), dtype=np.float64),
        pair_segments=corridor_rungs,
        accepted_pair_count=corridor_sample_count,
        left_chain_length=int(left_chain.filtered_indices.size),
        right_chain_length=int(right_chain.filtered_indices.size),
        filtered_track_width_m=float(expected_width),
        unknown_pair_count=0,
        corridor_width_min_m=width_min_m,
        corridor_width_max_m=width_max_m,
    )
    if status == "ok" and math.isfinite(measured_width_m):
        result.filtered_track_width_m = float(measured_width_m)
    return result


def update_track_width_estimate(
    previous_width_m: Optional[float],
    measured_width_m: Optional[float],
    config: CorridorPlannerConfig,
) -> float:
    width = (
        config.initial_width_m
        if previous_width_m is None or not math.isfinite(float(previous_width_m))
        else float(previous_width_m)
    )
    width = _clamp(width, config.min_width_m, config.max_width_m)
    if measured_width_m is None or not math.isfinite(float(measured_width_m)):
        return width

    measured = _clamp(float(measured_width_m), config.min_width_m, config.max_width_m)
    delta = _clamp(
        measured - width,
        -float(config.max_width_delta_per_update_m),
        float(config.max_width_delta_per_update_m),
    )
    alpha = _clamp(float(config.width_filter_alpha), 0.0, 1.0)
    updated = width + (alpha * delta)
    return _clamp(updated, config.min_width_m, config.max_width_m)


def _geometry_filter(local_points: np.ndarray, config: CorridorPlannerConfig) -> np.ndarray:
    distance = np.hypot(local_points[:, 0], local_points[:, 1])
    return (
        np.isfinite(local_points[:, 0])
        & np.isfinite(local_points[:, 1])
        & (distance <= float(config.max_cone_range_m))
        & (local_points[:, 0] >= -float(config.behind_drop_m))
        & (local_points[:, 0] <= float(config.planning_horizon_m))
        & (np.abs(local_points[:, 1]) <= float(config.max_lateral_range_m))
    )


def _deterministic_order(
    local_points: np.ndarray,
    global_points: np.ndarray,
    colors: list[str],
) -> np.ndarray:
    color_rank = np.asarray(
        [0 if color == "blue" else 1 if color == "yellow" else 2 for color in colors],
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


def _build_boundary_chain(
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    side_indices: np.ndarray,
    config: CorridorPlannerConfig,
) -> _BoundaryChain:
    if side_indices.size == 0:
        return _BoundaryChain(
            filtered_indices=np.empty((0,), dtype=np.int64),
            global_points=np.empty((0, 2), dtype=np.float64),
            local_points=np.empty((0, 2), dtype=np.float64),
            tangents_local=np.empty((0, 2), dtype=np.float64),
            mean_heading_change_rad=float("inf"),
            forward_extent_m=0.0,
        )

    side_local = filtered_local[side_indices]
    seed_pos = _select_seed_index(side_local)
    if seed_pos < 0:
        return _BoundaryChain(
            filtered_indices=np.empty((0,), dtype=np.int64),
            global_points=np.empty((0, 2), dtype=np.float64),
            local_points=np.empty((0, 2), dtype=np.float64),
            tangents_local=np.empty((0, 2), dtype=np.float64),
            mean_heading_change_rad=float("inf"),
            forward_extent_m=0.0,
        )

    chain_positions = [seed_pos]
    remaining = [idx for idx in range(side_indices.size) if idx != seed_pos]
    heading = np.asarray([1.0, 0.0], dtype=np.float64)
    heading_changes: list[float] = []

    while remaining:
        current_local = side_local[chain_positions[-1]]
        best_pos = None
        best_score: Optional[tuple[float, float, float, float, float, int]] = None
        best_heading = heading
        best_heading_change = 0.0
        for candidate_pos in remaining:
            candidate_local = side_local[candidate_pos]
            delta = candidate_local - current_local
            delta_x = float(delta[0])
            lateral_shift = float(abs(delta[1]))
            turn_progress = (
                delta_x >= -max(0.5, float(config.min_forward_progress_m))
                and lateral_shift >= max(0.10, 0.5 * float(config.min_forward_progress_m))
            )
            if not _candidate_progresses_from_vehicle(
                current_local=current_local,
                candidate_local=candidate_local,
                min_progress_m=float(config.min_forward_progress_m),
            ) and not turn_progress:
                continue
            distance = float(np.hypot(delta[0], delta[1]))
            min_step = float(config.min_step_m)
            if turn_progress:
                min_step = min(min_step, 0.35)
            if distance < min_step or distance > float(config.max_step_m):
                continue
            step_heading = delta / distance
            forward = float(np.dot(delta, heading))
            if forward < float(config.min_forward_progress_m):
                if not turn_progress or delta_x < -max(0.5, float(config.min_forward_progress_m)):
                    continue
            heading_change = abs(_angle_between(heading, step_heading))
            max_heading_change = float(config.max_heading_change_rad)
            if turn_progress:
                max_heading_change = max(max_heading_change, 1.55)
            if heading_change > max_heading_change:
                continue
            if _candidate_is_shadowed(
                current_local=current_local,
                candidate_pos=candidate_pos,
                side_local=side_local,
                remaining=remaining,
            ):
                continue
            score = (
                distance,
                heading_change,
                -lateral_shift,
                -max(delta_x, 0.0),
                -max(forward, 0.0),
                candidate_pos,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_pos = candidate_pos
                best_heading = step_heading
                best_heading_change = heading_change

        if best_pos is None:
            break
        chain_positions.append(best_pos)
        remaining.remove(best_pos)
        heading = best_heading
        heading_changes.append(best_heading_change)

    primary_chain = _make_boundary_chain_from_positions(
        filtered_points=filtered_points,
        filtered_local=filtered_local,
        side_indices=side_indices,
        chain_positions=np.asarray(chain_positions, dtype=np.int64),
        heading_changes=heading_changes,
    )
    fallback_chain = _bearing_sorted_boundary_chain(
        filtered_points=filtered_points,
        filtered_local=filtered_local,
        side_indices=side_indices,
        config=config,
    )
    if _should_prefer_fallback_chain(primary_chain, fallback_chain, config):
        return fallback_chain
    return primary_chain


def _make_boundary_chain_from_positions(
    *,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    side_indices: np.ndarray,
    chain_positions: np.ndarray,
    heading_changes: list[float],
) -> _BoundaryChain:
    if chain_positions.size == 0:
        return _BoundaryChain(
            filtered_indices=np.empty((0,), dtype=np.int64),
            global_points=np.empty((0, 2), dtype=np.float64),
            local_points=np.empty((0, 2), dtype=np.float64),
            tangents_local=np.empty((0, 2), dtype=np.float64),
            mean_heading_change_rad=float("inf"),
            forward_extent_m=0.0,
        )
    filtered_indices = side_indices[chain_positions]
    global_points = filtered_points[filtered_indices]
    local_points = filtered_local[filtered_indices]
    tangents_local = _estimate_tangents(local_points)
    mean_heading_change = float(np.mean(heading_changes)) if heading_changes else 0.0
    forward_extent = (
        float(np.max(local_points[:, 0]) - np.min(local_points[:, 0]))
        if local_points.shape[0] > 0
        else 0.0
    )
    return _BoundaryChain(
        filtered_indices=filtered_indices,
        global_points=global_points,
        local_points=local_points,
        tangents_local=tangents_local,
        mean_heading_change_rad=mean_heading_change,
        forward_extent_m=forward_extent,
    )


def _bearing_sorted_boundary_chain(
    *,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    side_indices: np.ndarray,
    config: CorridorPlannerConfig,
) -> _BoundaryChain:
    if side_indices.size == 0:
        return _BoundaryChain(
            filtered_indices=np.empty((0,), dtype=np.int64),
            global_points=np.empty((0, 2), dtype=np.float64),
            local_points=np.empty((0, 2), dtype=np.float64),
            tangents_local=np.empty((0, 2), dtype=np.float64),
            mean_heading_change_rad=float("inf"),
            forward_extent_m=0.0,
        )
    side_local = filtered_local[side_indices]
    seed_pos = _select_seed_index(side_local)
    if seed_pos < 0:
        ranges = np.hypot(side_local[:, 0], side_local[:, 1])
        seed_pos = int(np.argmin(ranges))

    bearings = np.arctan2(side_local[:, 1], side_local[:, 0])
    seed_bearing = float(bearings[seed_pos])
    deltas = np.arctan2(np.sin(bearings - seed_bearing), np.cos(bearings - seed_bearing))

    negative_branch = sorted(
        [idx for idx in range(side_local.shape[0]) if idx != seed_pos and deltas[idx] < -1e-3],
        key=lambda idx: float(deltas[idx]),
    )
    positive_branch = sorted(
        [idx for idx in range(side_local.shape[0]) if idx != seed_pos and deltas[idx] >= -1e-3],
        key=lambda idx: float(deltas[idx]),
    )
    order = negative_branch + [seed_pos] + positive_branch

    kept: list[int] = []
    last_point: Optional[np.ndarray] = None
    min_spacing = min(float(config.min_step_m), 0.35)
    for pos in order:
        point = side_local[int(pos)]
        if last_point is not None:
            step = float(np.hypot(point[0] - last_point[0], point[1] - last_point[1]))
            if step < min_spacing:
                continue
        kept.append(int(pos))
        last_point = point

    heading_changes: list[float] = []
    if len(kept) >= 3:
        ordered_points = side_local[np.asarray(kept, dtype=np.int64)]
        diffs = np.diff(ordered_points, axis=0)
        for idx in range(diffs.shape[0] - 1):
            a = diffs[idx]
            b = diffs[idx + 1]
            if float(np.hypot(a[0], a[1])) <= 1e-9 or float(np.hypot(b[0], b[1])) <= 1e-9:
                continue
            heading_changes.append(abs(_angle_between(a, b)))
    return _make_boundary_chain_from_positions(
        filtered_points=filtered_points,
        filtered_local=filtered_local,
        side_indices=side_indices,
        chain_positions=np.asarray(kept, dtype=np.int64),
        heading_changes=heading_changes,
    )


def _should_prefer_fallback_chain(
    primary_chain: _BoundaryChain,
    fallback_chain: _BoundaryChain,
    config: CorridorPlannerConfig,
) -> bool:
    fallback_count = int(fallback_chain.filtered_indices.size)
    primary_count = int(primary_chain.filtered_indices.size)
    if fallback_count < int(config.min_chain_length):
        return False
    if fallback_count <= primary_count:
        return False
    if primary_count < int(config.min_chain_length):
        return True
    primary_length = _path_length(primary_chain.local_points)
    fallback_length = _path_length(fallback_chain.local_points)
    if fallback_count >= (primary_count + 2) and fallback_length >= (0.9 * primary_length):
        return True
    return False


def _build_corridor(
    *,
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: CorridorPlannerConfig,
) -> Optional[dict[str, np.ndarray]]:
    dx = max(0.05, float(config.boundary_resample_dx))
    left_local = _resample_boundary_by_station(left_chain.local_points, dx)
    right_local = _resample_boundary_by_station(right_chain.local_points, dx)
    if left_local is None or right_local is None:
        return None

    corridor_candidates: list[dict[str, np.ndarray]] = []

    station_count = min(left_local.shape[0], right_local.shape[0])
    if station_count >= 2:
        station_candidate = _build_corridor_candidate(
            left_local=np.asarray(left_local[:station_count], dtype=np.float64),
            right_local=np.asarray(right_local[:station_count], dtype=np.float64),
            config=config,
        )
        if station_candidate is not None:
            corridor_candidates.append(station_candidate)

    progress_count = max(left_local.shape[0], right_local.shape[0])
    if progress_count >= 2:
        normalized_candidate = _build_corridor_candidate(
            left_local=_resample_to_count(left_chain.local_points, progress_count),
            right_local=_resample_to_count(right_chain.local_points, progress_count),
            config=config,
        )
        if normalized_candidate is not None:
            corridor_candidates.append(normalized_candidate)

    if not corridor_candidates:
        return None

    chosen = max(
        corridor_candidates,
        key=lambda candidate: (
            int(candidate["anchors_local"].shape[0]),
            -float(np.std(candidate["widths_m"])) if candidate["widths_m"].size else float("-inf"),
            -float(np.max(candidate["widths_m"])) if candidate["widths_m"].size else float("-inf"),
        ),
    )

    left_local = np.asarray(chosen["left_local"], dtype=np.float64)
    right_local = np.asarray(chosen["right_local"], dtype=np.float64)
    widths = np.asarray(chosen["widths_m"], dtype=np.float64)
    anchors_local = np.asarray(chosen["anchors_local"], dtype=np.float64)

    centerline_local = _fit_centerline_from_anchors(anchors_local, config)

    left_global = _from_vehicle_frame(left_local, vehicle_xy, vehicle_yaw)
    right_global = _from_vehicle_frame(right_local, vehicle_xy, vehicle_yaw)
    anchors_global = _from_vehicle_frame(anchors_local, vehicle_xy, vehicle_yaw)
    centerline_global = _from_vehicle_frame(centerline_local, vehicle_xy, vehicle_yaw)
    rungs_global = np.empty((anchors_local.shape[0], 2, 2), dtype=np.float64)
    rungs_global[:, 0, :] = left_global
    rungs_global[:, 1, :] = right_global

    return {
        "x_local": anchors_local[:, 0].copy(),
        "anchors_local": anchors_local,
        "widths_m": widths,
        "left_global": left_global,
        "right_global": right_global,
        "anchors_global": anchors_global,
        "centerline_global": centerline_global,
        "rungs_global": rungs_global,
    }


def _build_corridor_candidate(
    *,
    left_local: np.ndarray,
    right_local: np.ndarray,
    config: CorridorPlannerConfig,
) -> Optional[dict[str, np.ndarray]]:
    if left_local.shape[0] < 2 or right_local.shape[0] < 2:
        return None

    widths = np.hypot(left_local[:, 0] - right_local[:, 0], left_local[:, 1] - right_local[:, 1])
    valid_mask = _corridor_valid_mask(
        left_local=left_local,
        right_local=right_local,
        widths=widths,
        config=config,
    )
    valid_mask = _fill_small_invalid_gaps(valid_mask, max_gap=1)
    valid_slice = _longest_valid_slice(valid_mask)
    if valid_slice is None:
        return None

    left_valid = np.asarray(left_local[valid_slice], dtype=np.float64)
    right_valid = np.asarray(right_local[valid_slice], dtype=np.float64)
    widths_valid = np.asarray(widths[valid_slice], dtype=np.float64)
    if left_valid.shape[0] < int(config.min_required_corridor_samples):
        return None

    anchors_local = 0.5 * (left_valid + right_valid)
    return {
        "left_local": left_valid,
        "right_local": right_valid,
        "widths_m": widths_valid,
        "anchors_local": anchors_local,
    }


def _corridor_valid_mask(
    *,
    left_local: np.ndarray,
    right_local: np.ndarray,
    widths: np.ndarray,
    config: CorridorPlannerConfig,
) -> np.ndarray:
    valid_mask = np.isfinite(widths)
    valid_mask &= np.isfinite(left_local[:, 0]) & np.isfinite(left_local[:, 1])
    valid_mask &= np.isfinite(right_local[:, 0]) & np.isfinite(right_local[:, 1])
    valid_mask &= widths >= float(config.min_corridor_width_m)
    valid_mask &= widths <= float(config.max_corridor_width_m)
    valid_mask &= left_local[:, 0] >= -float(config.behind_drop_m)
    valid_mask &= right_local[:, 0] >= -float(config.behind_drop_m)
    valid_mask &= left_local[:, 0] <= float(config.planning_horizon_m)
    valid_mask &= right_local[:, 0] <= float(config.planning_horizon_m)
    return valid_mask


def _fill_small_invalid_gaps(valid_mask: np.ndarray, max_gap: int) -> np.ndarray:
    closed = np.asarray(valid_mask, dtype=bool).copy()
    if max_gap <= 0 or closed.size < 3:
        return closed
    idx = 0
    while idx < closed.size:
        if closed[idx]:
            idx += 1
            continue
        start = idx
        while idx < closed.size and not closed[idx]:
            idx += 1
        stop = idx
        gap = stop - start
        if start == 0 or stop >= closed.size:
            continue
        if gap <= max_gap and closed[start - 1] and closed[stop]:
            closed[start:stop] = True
    return closed


def _boundary_interp_series(local_points: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if local_points.shape[0] < 2:
        return None
    valid_mask = np.all(np.isfinite(local_points), axis=1)
    pts = np.asarray(local_points[valid_mask], dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    # Preserve boundary chain order through turns. Sorting by x alone scrambles
    # curved boundaries and makes the corridor disappear as soon as a side
    # bends back slightly in vehicle-local coordinates.
    x_progress = np.maximum.accumulate(pts[:, 0])
    series: list[list[float]] = []
    counts: list[int] = []
    eps = 1e-3
    for idx, (_raw_x, y_val) in enumerate(pts):
        x_val = float(x_progress[idx])
        if not series:
            series.append([x_val, float(y_val)])
            counts.append(1)
            continue
        if abs(x_val - float(series[-1][0])) <= eps:
            prev_count = counts[-1]
            prev_x, prev_y = series[-1]
            series[-1][0] = max(prev_x, x_val)
            series[-1][1] = ((prev_y * prev_count) + float(y_val)) / float(prev_count + 1)
            counts[-1] = prev_count + 1
            continue
        if x_val < float(series[-1][0]) + eps:
            x_val = float(series[-1][0]) + eps
        series.append([x_val, float(y_val)])
        counts.append(1)

    if len(series) < 2:
        return None
    arr = np.asarray(series, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def _longest_valid_slice(valid_mask: np.ndarray) -> Optional[slice]:
    best_start = -1
    best_len = 0
    run_start = -1
    for idx, is_valid in enumerate(valid_mask):
        if bool(is_valid):
            if run_start < 0:
                run_start = idx
        elif run_start >= 0:
            run_len = idx - run_start
            if run_len > best_len:
                best_start = run_start
                best_len = run_len
            run_start = -1
    if run_start >= 0:
        run_len = len(valid_mask) - run_start
        if run_len > best_len:
            best_start = run_start
            best_len = run_len
    if best_start < 0 or best_len <= 0:
        return None
    return slice(best_start, best_start + best_len)


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 2 or window <= 1:
        return arr
    radius = max(1, int(window)) // 2
    out = np.empty_like(arr)
    for idx in range(arr.size):
        start = max(0, idx - radius)
        stop = min(arr.size, idx + radius + 1)
        out[idx] = float(np.mean(arr[start:stop]))
    return out


def _effective_near_field_jump_limit(
    *,
    config: CorridorPlannerConfig,
    corridor_sample_count: int,
) -> float:
    limit = float(config.max_near_field_lateral_jump_m)
    if corridor_sample_count >= int(config.min_required_corridor_samples) + 2:
        limit *= 1.75
    return limit


def _effective_curvature_limit(
    *,
    config: CorridorPlannerConfig,
    corridor_sample_count: int,
) -> float:
    limit = float(config.max_curvature)
    if limit < 0.2:
        return limit
    if corridor_sample_count >= int(config.min_required_corridor_samples) + 2:
        limit *= 1.75
    return limit


def _moving_average_points(points: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape[0] <= 2 or window <= 1:
        return arr
    out = np.empty_like(arr)
    out[:, 0] = _moving_average_1d(arr[:, 0], window)
    out[:, 1] = _moving_average_1d(arr[:, 1], window)
    return out


def _fit_centerline_from_anchors(
    anchors_local: np.ndarray,
    config: CorridorPlannerConfig,
) -> np.ndarray:
    fitted = np.asarray(anchors_local, dtype=np.float64)
    window = max(1, int(config.path_fit_smoothing_window))
    if fitted.shape[0] <= 2 or window <= 1:
        return fitted

    best = np.array(fitted, copy=True)
    best_curvature = float("inf")
    for _ in range(4):
        candidate = _moving_average_points(fitted, window)
        candidate_rs = _resample_path(
            candidate,
            resolution_m=float(config.path_resolution_m),
            max_length_m=float(config.max_path_length_m),
        )
        curvature = _path_curvature_abs_max(candidate_rs)
        if curvature < best_curvature:
            best = candidate
            best_curvature = curvature
        if curvature <= float(config.max_curvature):
            return candidate
        fitted = candidate
    return best


def _resample_boundary_by_station(points: np.ndarray, spacing_m: float) -> Optional[np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    valid_mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[valid_mask]
    if pts.shape[0] < 2:
        return None

    cumulative = _path_cumulative_lengths(pts)
    total = float(cumulative[-1])
    if total <= 1e-6:
        return None

    spacing = max(0.05, float(spacing_m))
    samples = np.arange(0.0, total + 1e-9, spacing, dtype=np.float64)
    if samples.size == 0 or samples[-1] < total:
        samples = np.concatenate((samples, [total]))
    x = np.interp(samples, cumulative, pts[:, 0])
    y = np.interp(samples, cumulative, pts[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


def _resample_to_count(points: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if count <= 0 or pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if count == 1:
        return np.asarray(pts[:1], dtype=np.float64)
    if pts.shape[0] == 1:
        return np.repeat(np.asarray(pts[:1], dtype=np.float64), count, axis=0)

    cumulative = _path_cumulative_lengths(pts)
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.repeat(np.asarray(pts[:1], dtype=np.float64), count, axis=0)

    samples = np.linspace(0.0, total, count, dtype=np.float64)
    x = np.interp(samples, cumulative, pts[:, 0])
    y = np.interp(samples, cumulative, pts[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


def _path_violates_corridor(
    *,
    centerline_local: np.ndarray,
    corridor_center_local: np.ndarray,
    corridor_widths_m: np.ndarray,
    membership_margin_m: float,
) -> bool:
    if centerline_local.shape[0] == 0 or corridor_center_local.shape[0] == 0:
        return True
    count = corridor_center_local.shape[0]
    centerline_rs = _resample_to_count(centerline_local, count)
    if centerline_rs.shape[0] != count:
        return True
    deviation = np.hypot(
        centerline_rs[:, 0] - corridor_center_local[:, 0],
        centerline_rs[:, 1] - corridor_center_local[:, 1],
    )
    allowed = (0.5 * corridor_widths_m) + max(0.0, float(membership_margin_m))
    return bool(np.any(deviation > allowed))


def _select_seed_index(side_local: np.ndarray) -> int:
    candidates = np.flatnonzero(side_local[:, 0] >= 0.0)
    if candidates.size == 0:
        return -1

    best_pos = -1
    best_score = None
    for pos in candidates:
        x = float(side_local[pos, 0])
        y = float(side_local[pos, 1])
        score = (x, abs(y), math.hypot(x, y), int(pos))
        if best_score is None or score < best_score:
            best_score = score
            best_pos = int(pos)
    return best_pos


def _candidate_progresses_from_vehicle(
    *,
    current_local: np.ndarray,
    candidate_local: np.ndarray,
    min_progress_m: float,
) -> bool:
    x_margin = max(0.05, 0.5 * float(min_progress_m))
    if float(candidate_local[0]) >= float(current_local[0]) - x_margin:
        return True

    current_y = float(current_local[1])
    candidate_y = float(candidate_local[1])
    if abs(current_y) <= 0.05:
        return False
    same_side = current_y * candidate_y >= 0.0
    outboard_progress = abs(candidate_y) >= abs(current_y) + (0.5 * float(min_progress_m))
    return bool(same_side and outboard_progress)


def _candidate_is_shadowed(
    *,
    current_local: np.ndarray,
    candidate_pos: int,
    side_local: np.ndarray,
    remaining: list[int],
) -> bool:
    candidate_delta = side_local[candidate_pos] - current_local
    candidate_distance = float(np.hypot(candidate_delta[0], candidate_delta[1]))
    if candidate_distance <= 1e-9:
        return True
    candidate_dir = candidate_delta / candidate_distance

    for other_pos in remaining:
        if other_pos == candidate_pos:
            continue
        other_delta = side_local[other_pos] - current_local
        other_distance = float(np.hypot(other_delta[0], other_delta[1]))
        if other_distance <= 1e-9 or other_distance >= candidate_distance:
            continue
        other_dir = other_delta / other_distance
        if abs(_angle_between(candidate_dir, other_dir)) > 0.30:
            continue
        if float(np.dot(other_delta, candidate_dir)) <= 0.0:
            continue
        return True
    return False


def _estimate_tangents(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    tangents = np.empty_like(points)
    for idx in range(points.shape[0]):
        if idx == 0:
            delta = points[1] - points[0] if points.shape[0] > 1 else np.asarray([1.0, 0.0], dtype=np.float64)
        elif idx == points.shape[0] - 1:
            delta = points[-1] - points[-2]
        else:
            delta = points[idx + 1] - points[idx - 1]
        norm = float(np.hypot(delta[0], delta[1]))
        tangents[idx] = (
            np.asarray([1.0, 0.0], dtype=np.float64)
            if norm <= 1e-9
            else (delta / norm)
        )
    return tangents


def _resample_path(points: np.ndarray, resolution_m: float, max_length_m: float) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.asarray(points, dtype=np.float64)

    cumulative = _path_cumulative_lengths(points)
    total = min(float(cumulative[-1]), float(max_length_m))
    if total <= 1e-6:
        return np.asarray(points[:1], dtype=np.float64)

    step = max(0.05, float(resolution_m))
    samples = np.arange(0.0, total + 1e-9, step, dtype=np.float64)
    if samples.size == 0 or samples[-1] < total:
        samples = np.concatenate((samples, [total]))
    x = np.interp(samples, cumulative, points[:, 0])
    y = np.interp(samples, cumulative, points[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


def _finalize_path(points: np.ndarray, config: CorridorPlannerConfig) -> np.ndarray:
    return _resample_path(
        np.asarray(points, dtype=np.float64),
        resolution_m=float(config.path_resolution_m),
        max_length_m=float(config.max_path_length_m),
    )


def _path_cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.asarray([0.0], dtype=np.float64)
    diffs = np.diff(points, axis=0)
    lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate(([0.0], np.cumsum(lengths))).astype(np.float64)


def _path_length(points: np.ndarray) -> float:
    cumulative = _path_cumulative_lengths(np.asarray(points, dtype=np.float64))
    return float(cumulative[-1]) if cumulative.size > 0 else 0.0


def _near_field_delta_metrics(
    *,
    current: np.ndarray,
    previous: Optional[np.ndarray],
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    horizon_m: float,
) -> dict[str, float]:
    if previous is None or previous.shape[0] < 2 or current.shape[0] < 2:
        return {
            "lateral_max_m": 0.0,
            "lateral_mean_m": 0.0,
            "displacement_max_m": 0.0,
            "displacement_mean_m": 0.0,
        }

    current_local = _to_vehicle_frame(current, vehicle_xy, vehicle_yaw)
    previous_local = _to_vehicle_frame(previous, vehicle_xy, vehicle_yaw)
    current_rs = _resample_path(current_local, 0.25, horizon_m)
    previous_rs = _resample_path(previous_local, 0.25, horizon_m)
    count = min(current_rs.shape[0], previous_rs.shape[0])
    if count <= 0:
        return {
            "lateral_max_m": 0.0,
            "lateral_mean_m": 0.0,
            "displacement_max_m": 0.0,
            "displacement_mean_m": 0.0,
        }

    delta = current_rs[:count] - previous_rs[:count]
    lateral = np.abs(delta[:, 1])
    displacement = np.hypot(delta[:, 0], delta[:, 1])
    return {
        "lateral_max_m": float(np.max(lateral)) if lateral.size else 0.0,
        "lateral_mean_m": float(np.mean(lateral)) if lateral.size else 0.0,
        "displacement_max_m": float(np.max(displacement)) if displacement.size else 0.0,
        "displacement_mean_m": float(np.mean(displacement)) if displacement.size else 0.0,
    }


def _path_heading_delta_max(path_local: np.ndarray) -> float:
    if path_local.shape[0] < 3:
        return 0.0
    diffs = np.diff(path_local, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    delta = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    return float(np.max(np.abs(delta))) if delta.size else 0.0


def _path_start_heading_error(path_local: np.ndarray) -> float:
    if path_local.shape[0] < 2:
        return 0.0
    delta = path_local[1] - path_local[0]
    if float(np.hypot(delta[0], delta[1])) <= 1e-9:
        return 0.0
    return float(math.atan2(delta[1], delta[0]))


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
    first = np.asarray(path_local[0], dtype=np.float64)
    return float(np.hypot(first[0], first[1]))


def _path_curvature_abs_max(path_local: np.ndarray) -> float:
    if path_local.shape[0] < 3:
        return 0.0
    diffs = np.diff(path_local, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    valid = seg_len > 1e-6
    if np.count_nonzero(valid) < 2:
        return 0.0
    headings = np.arctan2(diffs[valid, 1], diffs[valid, 0])
    if headings.size < 2:
        return 0.0
    delta_heading = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    ds = 0.5 * (seg_len[valid][1:] + seg_len[valid][:-1])
    valid_ds = ds > 1e-6
    if not np.any(valid_ds):
        return 0.0
    curvature = np.abs(delta_heading[valid_ds] / ds[valid_ds])
    return float(np.max(curvature)) if curvature.size else 0.0


def _path_self_intersects(path_xy: np.ndarray) -> bool:
    if path_xy.shape[0] < 4:
        return False
    for idx in range(path_xy.shape[0] - 3):
        a0 = path_xy[idx]
        a1 = path_xy[idx + 1]
        for jdx in range(idx + 2, path_xy.shape[0] - 1):
            if jdx == idx + 1:
                continue
            b0 = path_xy[jdx]
            b1 = path_xy[jdx + 1]
            if _segments_intersect(a0, a1, b0, b1):
                return True
    return False


def _segments_intersect(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a0, a1, b0)
    o2 = orient(a0, a1, b1)
    o3 = orient(b0, b1, a0)
    o4 = orient(b0, b1, a1)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _to_vehicle_frame(
    points_xy: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
) -> np.ndarray:
    dx = points_xy[:, 0] - float(vehicle_xy[0])
    dy = points_xy[:, 1] - float(vehicle_xy[1])
    cos_yaw = math.cos(vehicle_yaw)
    sin_yaw = math.sin(vehicle_yaw)
    x_local = (cos_yaw * dx) + (sin_yaw * dy)
    y_local = (-sin_yaw * dx) + (cos_yaw * dy)
    return np.column_stack((x_local, y_local)).astype(np.float64)


def _from_vehicle_frame(
    local_points: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
) -> np.ndarray:
    cos_yaw = math.cos(vehicle_yaw)
    sin_yaw = math.sin(vehicle_yaw)
    x_global = (
        float(vehicle_xy[0])
        + (cos_yaw * local_points[:, 0])
        - (sin_yaw * local_points[:, 1])
    )
    y_global = (
        float(vehicle_xy[1])
        + (sin_yaw * local_points[:, 0])
        + (cos_yaw * local_points[:, 1])
    )
    return np.column_stack((x_global, y_global)).astype(np.float64)


def _angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    angle_a = math.atan2(float(vec_a[1]), float(vec_a[0]))
    angle_b = math.atan2(float(vec_b[1]), float(vec_b[0]))
    return math.atan2(math.sin(angle_b - angle_a), math.cos(angle_b - angle_a))


def _default_reject_counts() -> dict[str, int]:
    return {
        "corridor_geometry": 0,
        "corridor_samples": 0,
        "path_outside_corridor": 0,
        "heading": 0,
        "curvature": 0,
        "near_field_continuity": 0,
    }


def _empty_result(
    status: str,
    *,
    filtered_points: Optional[np.ndarray] = None,
    filtered_colors: Optional[list[str]] = None,
    left_boundary: Optional[np.ndarray] = None,
    right_boundary: Optional[np.ndarray] = None,
    reject_counts: Optional[dict[str, int]] = None,
    reject_reason: str = "",
) -> CorridorPlannerResult:
    return CorridorPlannerResult(
        filtered_points=(
            np.asarray(filtered_points, dtype=np.float64)
            if filtered_points is not None
            else np.empty((0, 2), dtype=np.float64)
        ),
        filtered_colors=list(filtered_colors or []),
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.empty((0, 2), dtype=np.int64),
        selected_pair_track_ids=np.empty((0, 2), dtype=np.int64),
        midpoints_raw=np.empty((0, 2), dtype=np.float64),
        centerline=np.empty((0, 2), dtype=np.float64),
        left_boundary=(
            np.asarray(left_boundary, dtype=np.float64)
            if left_boundary is not None
            else np.empty((0, 2), dtype=np.float64)
        ),
        right_boundary=(
            np.asarray(right_boundary, dtype=np.float64)
            if right_boundary is not None
            else np.empty((0, 2), dtype=np.float64)
        ),
        used_fallback=False,
        status=status,
        reject_counts=reject_counts or _default_reject_counts(),
        reject_reason=reject_reason,
    )


def _result_with_metadata(
    *,
    result: CorridorPlannerResult,
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
    planner_mode: str,
    filtered_track_width_m: float,
) -> CorridorPlannerResult:
    result.left_chain_length = int(left_chain.filtered_indices.size)
    result.right_chain_length = int(right_chain.filtered_indices.size)
    result.left_boundary = left_chain.global_points
    result.right_boundary = right_chain.global_points
    result.planner_mode = planner_mode
    result.filtered_track_width_m = float(filtered_track_width_m)
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))
