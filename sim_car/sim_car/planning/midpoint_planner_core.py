"""Core geometry for the midpoint boundary planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from sim_car.cones.tracking.fusion import normalize_color

_MIDPOINT_CHAIN_BACKTRACK_TOLERANCE_M = 0.25


@dataclass
class MidpointPlannerConfig:
    max_cone_range_m: float = 25.0
    behind_drop_m: float = 5.0
    min_confidence: float = 0.3
    min_required_cones: int = 4
    allow_unknown_pair_completion: bool = True
    unknown_pair_search_radius_m: float = 1.25
    unknown_pair_max_longitudinal_error_m: float = 1.5
    unknown_pair_max_width_error_m: float = 0.9
    max_consecutive_unknown_pairs: int = 2

    min_step_m: float = 0.8
    max_step_m: float = 10.0
    max_heading_change_rad: float = 1.0
    min_forward_progress_m: float = 0.2
    min_chain_length: int = 3

    min_pair_width_m: float = 2.2
    max_pair_width_m: float = 5.5
    max_width_jump_m: float = 0.8
    min_pair_count: int = 3
    pair_reassignment_margin: float = 0.25
    pair_inward_projection_tolerance_m: float = 0.15
    pairing_tangent_neighbor_count: int = 4
    enforce_opposite_color_pairing: bool = True
    enforce_geometry_pairing_gate: bool = True

    initial_width_m: float = 3.6
    min_width_m: float = 2.4
    max_width_m: float = 4.8
    width_filter_alpha: float = 0.15
    max_width_delta_per_update_m: float = 0.2
    min_trustworthy_pairs: int = 3

    path_resolution_m: float = 0.5
    max_path_length_m: float = 30.0
    smoothing_window: int = 3
    max_heading_delta_rad: float = 0.75
    max_midpoint_segment_length_m: float = 7.5
    midpoint_order_reference_handoff_m: float = 6.0
    midpoint_order_history_size: int = 3
    midpoint_order_backtrack_tolerance_m: float = 0.35

    min_path_points: int = 4
    min_forward_extent_m: float = 2.0
    jump_check_horizon_m: float = 8.0
    max_near_field_lateral_jump_m: float = 0.6
    max_near_field_lateral_jump_m_sparse_pairs: float = 0.9
    max_start_heading_error_rad: float = 1.0


@dataclass
class MidpointPlannerPrior:
    previous_centerline: Optional[np.ndarray] = None
    previous_width_m: Optional[float] = None
    previous_mode: str = "none"
    previous_pairs: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class MidpointPlannerResult:
    filtered_points: np.ndarray
    filtered_colors: list[str]
    triangulation_edges: np.ndarray
    candidate_edges: np.ndarray
    selected_edges: np.ndarray
    selected_pair_track_ids: np.ndarray
    midpoints_raw: np.ndarray
    centerline: np.ndarray
    prevalidation_centerline: np.ndarray
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


@dataclass
class _BoundaryChain:
    filtered_indices: np.ndarray
    global_points: np.ndarray
    local_points: np.ndarray
    tangents_local: np.ndarray
    mean_heading_change_rad: float
    forward_extent_m: float


@dataclass
class _BoundaryPair:
    left_filtered_idx: int
    right_filtered_idx: int
    left_track_id: int
    right_track_id: int
    left_global: np.ndarray
    right_global: np.ndarray
    left_local: np.ndarray
    right_local: np.ndarray
    width_m: float

    @property
    def midpoint_global(self) -> np.ndarray:
        return 0.5 * (self.left_global + self.right_global)

    @property
    def midpoint_local(self) -> np.ndarray:
        return 0.5 * (self.left_local + self.right_local)


def compute_midpoint_centerline(
    points_xy: np.ndarray,
    colors: list[str],
    confidences: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: MidpointPlannerConfig,
    prior: Optional[MidpointPlannerPrior] = None,
    track_ids: Optional[np.ndarray] = None,
    raw_colors: Optional[list[str]] = None,
) -> MidpointPlannerResult:
    """Compute a local centerline by pairing left/right boundary cones and taking midpoints.

    Filters cones by geometry and confidence, builds boundary chains, pairs them by width
    constraints, orders the midpoints forward, resamples, and validates the result.
    Returns a result with ``status='ok'`` on success or a descriptive failure status.
    """
    if points_xy.size == 0:
        return _empty_result("no cones available")

    if track_ids is None or len(track_ids) != points_xy.shape[0]:
        track_ids = np.arange(points_xy.shape[0], dtype=np.int64)
    else:
        track_ids = np.asarray(track_ids, dtype=np.int64)

    normalized = [normalize_color(color) for color in colors]
    if raw_colors is not None and len(raw_colors) == len(colors):
        normalized_raw = [normalize_color(color) for color in raw_colors]
    else:
        normalized_raw = list(normalized)
    local_points = _to_vehicle_frame(points_xy, vehicle_xy, vehicle_yaw)

    mask_geom = _geometry_filter(local_points, config)
    mask_conf = confidences >= float(config.min_confidence)
    colored_mask = np.array([color in {"blue", "yellow"} for color in normalized], dtype=bool)
    unknown_mask = np.array([color == "unknown" for color in normalized], dtype=bool)
    selected_mask = mask_geom & mask_conf & (
        colored_mask | (unknown_mask if config.allow_unknown_pair_completion else False)
    )

    filtered_points = points_xy[selected_mask]
    filtered_local = local_points[selected_mask]
    filtered_track_ids = track_ids[selected_mask]
    filtered_colors = [normalized[idx] for idx in np.where(selected_mask)[0]]
    filtered_raw_colors = [normalized_raw[idx] for idx in np.where(selected_mask)[0]]
    colored_count = int(np.count_nonzero(np.array([color in {"blue", "yellow"} for color in filtered_colors], dtype=bool)))
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
    filtered_raw_colors = [filtered_raw_colors[idx] for idx in order]

    if colored_count < int(config.min_required_cones):
        return _empty_result(
            f"usable colored cones below minimum ({colored_count} < {int(config.min_required_cones)})",
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            reject_counts=_default_reject_counts(),
        )

    left_indices = np.flatnonzero(np.array([color == "blue" for color in filtered_colors], dtype=bool))
    right_indices = np.flatnonzero(np.array([color == "yellow" for color in filtered_colors], dtype=bool))
    unknown_indices = np.flatnonzero(np.array([color == "unknown" for color in filtered_colors], dtype=bool))
    left_chain = _build_boundary_chain(filtered_points, filtered_local, left_indices, config)
    right_chain = _build_boundary_chain(filtered_points, filtered_local, right_indices, config)

    reject_counts = _default_reject_counts()
    expected_width = _clamp(
        prior.previous_width_m if prior and prior.previous_width_m is not None else config.initial_width_m,
        config.min_width_m,
        config.max_width_m,
    )
    candidate_edges = np.empty((0, 2), dtype=np.int64)

    pairs: list[_BoundaryPair] = []
    midpoint_chain = np.empty((0, 2), dtype=np.float64)
    raw_offset_path = np.empty((0, 2), dtype=np.float64)
    selected_edges = np.empty((0, 2), dtype=np.int64)
    used_fallback = False
    planner_mode = "none"
    active_boundary_side = ""
    candidate_count = 0
    measured_width_m = float("nan")

    left_boundary_count = int(left_indices.size)
    right_boundary_count = int(right_indices.size)
    pairing_possible = left_boundary_count > 0 and (
        right_boundary_count > 0 or unknown_indices.size > 0
    )
    if pairing_possible:
        pairs, candidate_count, unknown_pair_count, pair_reject_counts = _pair_boundary_chains(
            filtered_points=filtered_points,
            filtered_local=filtered_local,
            filtered_track_ids=filtered_track_ids,
            filtered_raw_colors=filtered_raw_colors,
            left_indices=left_indices,
            right_indices=right_indices,
            unknown_indices=unknown_indices,
            expected_width_m=expected_width,
            config=config,
            prior=prior,
            left_chain=left_chain,
            right_chain=right_chain,
        )
        _merge_reject_counts(reject_counts, pair_reject_counts)
        pairs = _order_pairs_into_midpoint_chain(
            pairs,
            config=config,
        )
        pairs = _trim_pairs_by_midpoint_step_length(
            pairs,
            max_segment_length_m=float(config.max_midpoint_segment_length_m),
        )
        if pairs:
            measured_width_m = float(np.median([pair.width_m for pair in pairs]))
        if len(pairs) >= int(config.min_pair_count):
            midpoint_chain = np.vstack([pair.midpoint_global for pair in pairs]).astype(np.float64)
            selected_edges = np.asarray(
                [[pair.left_filtered_idx, pair.right_filtered_idx] for pair in pairs],
                dtype=np.int64,
            )
            selected_pair_track_ids = np.asarray(
                [[pair.left_track_id, pair.right_track_id] for pair in pairs],
                dtype=np.int64,
            )
            planner_mode = "midpoint"
        else:
            unknown_pair_count = 0
            selected_pair_track_ids = np.empty((0, 2), dtype=np.int64)
    else:
        unknown_pair_count = 0
        selected_pair_track_ids = np.empty((0, 2), dtype=np.int64)

    if planner_mode == "none":
        return _result_with_metadata(
            result=_empty_result(
                "no reliable midpoint chain",
                filtered_points=filtered_points,
                filtered_colors=filtered_colors,
                left_boundary=left_chain.global_points,
                right_boundary=right_chain.global_points,
                reject_counts=reject_counts,
                reject_reason="no reliable midpoint chain",
            ),
            left_chain=left_chain,
            right_chain=right_chain,
            planner_mode="none",
            filtered_track_width_m=expected_width,
            left_boundary_count=left_boundary_count,
            right_boundary_count=right_boundary_count,
        )

    raw_curve = midpoint_chain
    centerline = _finalize_path(raw_curve, config)
    centerline_local = _to_vehicle_frame(centerline, vehicle_xy, vehicle_yaw)
    seed_distance_m = _first_point_distance(centerline_local)
    near_field = _near_field_delta_metrics(
        current=centerline,
        previous=None if prior is None else prior.previous_centerline,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        horizon_m=config.jump_check_horizon_m,
    )
    near_field_kink_max_rad = _path_heading_delta_max(centerline_local)
    continuity_threshold_m = float(config.max_near_field_lateral_jump_m)
    if len(pairs) <= max(3, int(config.min_pair_count)):
        continuity_threshold_m = max(
            continuity_threshold_m,
            float(config.max_near_field_lateral_jump_m_sparse_pairs),
        )

    prevalidation_centerline = np.array(centerline, copy=True)
    status = "ok"
    reject_reason = ""
    min_path_points = int(config.min_path_points)
    min_forward_extent_m = float(config.min_forward_extent_m)
    if centerline.shape[0] < min_path_points:
        status = "path has too few points"
        reject_reason = status
    elif not np.all(np.isfinite(centerline)):
        status = "path contains non-finite geometry"
        reject_reason = status
    elif _forward_extent_m(centerline_local) < min_forward_extent_m:
        status = "path forward extent too short"
        reject_reason = status
    else:
        start_heading_error = abs(_path_start_heading_error(centerline_local))
        if start_heading_error > float(config.max_start_heading_error_rad):
            reject_counts["midpoint_kink"] += 1
            status = "path heading flip near vehicle"
            reject_reason = status
        elif near_field["lateral_max_m"] > continuity_threshold_m:
            reject_counts["near_field_continuity"] += 1
            status = "near-field continuity rejected fresh path"
            reject_reason = status
        elif near_field_kink_max_rad > float(config.max_heading_delta_rad):
            reject_counts["midpoint_kink"] += 1
            status = "path heading delta exceeded limit"
            reject_reason = status
        elif _path_self_intersects(centerline):
            status = "path self-crossing detected"
            reject_reason = status

    if status != "ok":
        centerline = np.empty((0, 2), dtype=np.float64)

    result = MidpointPlannerResult(
        filtered_points=filtered_points,
        filtered_colors=filtered_colors,
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=candidate_edges,
        selected_edges=selected_edges,
        selected_pair_track_ids=selected_pair_track_ids,
        midpoints_raw=midpoint_chain,
        centerline=centerline,
        prevalidation_centerline=prevalidation_centerline,
        left_boundary=left_chain.global_points,
        right_boundary=right_chain.global_points,
        used_fallback=False,
        status=status,
        candidate_count=int(candidate_count),
        selected_chain_length=int(
            len(pairs)
        ),
        selected_chain_width_median=float(measured_width_m),
        expected_width_prior_m=float(expected_width),
        near_field_lateral_max_m=float(near_field["lateral_max_m"]),
        near_field_lateral_mean_m=float(near_field["lateral_mean_m"]),
        near_field_displacement_max_m=float(near_field["displacement_max_m"]),
        near_field_displacement_mean_m=float(near_field["displacement_mean_m"]),
        near_field_kink_max_rad=float(near_field_kink_max_rad),
        seed_midpoint_distance_m=float(seed_distance_m),
        seed_temporal_offset_m=float("nan"),
        reject_reason=reject_reason,
        reject_counts=reject_counts,
        planner_mode=planner_mode,
        active_boundary_side="",
        raw_offset_path=np.empty((0, 2), dtype=np.float64),
        pair_segments=np.asarray(
            [[pair.left_global, pair.right_global] for pair in pairs],
            dtype=np.float64,
        ) if pairs else np.empty((0, 2, 2), dtype=np.float64),
        accepted_pair_count=len(pairs),
        left_chain_length=left_boundary_count,
        right_chain_length=right_boundary_count,
        filtered_track_width_m=float(expected_width),
        unknown_pair_count=int(unknown_pair_count),
    )
    return result


def update_track_width_estimate(
    previous_width_m: Optional[float],
    measured_width_m: Optional[float],
    config: MidpointPlannerConfig,
) -> float:
    """Exponential-moving-average update of the running track width estimate.

    The update is rate-limited by ``config.max_width_delta_per_update_m`` and
    the result is clamped to ``[config.min_width_m, config.max_width_m]``.
    """
    width = (
        config.initial_width_m if previous_width_m is None or not math.isfinite(float(previous_width_m))
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


def _geometry_filter(local_points: np.ndarray, config: MidpointPlannerConfig) -> np.ndarray:
    distance = np.hypot(local_points[:, 0], local_points[:, 1])
    return (
        np.isfinite(local_points[:, 0])
        & np.isfinite(local_points[:, 1])
        & (distance <= float(config.max_cone_range_m))
        & (local_points[:, 0] >= -float(config.behind_drop_m))
    )


def _deterministic_order(
    local_points: np.ndarray,
    global_points: np.ndarray,
    colors: list[str],
) -> np.ndarray:
    color_rank = np.asarray(
        [
            0 if color == "blue" else 1 if color == "yellow" else 2
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


def _build_boundary_chain(
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    side_indices: np.ndarray,
    config: MidpointPlannerConfig,
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
        current_range = float(np.hypot(current_local[0], current_local[1]))
        best_pos = None
        best_score: Optional[tuple[float, float, float, float, int]] = None
        best_heading = heading
        best_heading_change = 0.0
        for candidate_pos in remaining:
            candidate_local = side_local[candidate_pos]
            candidate_range = float(np.hypot(candidate_local[0], candidate_local[1]))
            radial_progress = candidate_range - current_range
            if radial_progress < max(0.05, 0.5 * float(config.min_forward_progress_m)):
                continue
            if not _candidate_progresses_from_vehicle(
                current_local=current_local,
                candidate_local=candidate_local,
                min_progress_m=float(config.min_forward_progress_m),
            ):
                continue
            delta = side_local[candidate_pos] - current_local
            distance = float(np.hypot(delta[0], delta[1]))
            if distance < float(config.min_step_m) or distance > float(config.max_step_m):
                continue
            step_heading = delta / distance
            forward = float(np.dot(delta, heading))
            if forward < float(config.min_forward_progress_m):
                continue
            heading_change = abs(_angle_between(heading, step_heading))
            if heading_change > float(config.max_heading_change_rad):
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
                radial_progress,
                -forward,
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

    filtered_indices = side_indices[chain_positions]
    global_points = filtered_points[filtered_indices]
    local_points = filtered_local[filtered_indices]
    tangents_local = _estimate_tangents(local_points)
    mean_heading_change = (
        float(np.mean(heading_changes)) if heading_changes else 0.0
    )
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


def _pair_boundary_chains(
    *,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    filtered_raw_colors: Optional[list[str]] = None,
    left_indices: Optional[np.ndarray] = None,
    right_indices: Optional[np.ndarray] = None,
    unknown_indices: np.ndarray,
    expected_width_m: float,
    config: MidpointPlannerConfig,
    prior: Optional[MidpointPlannerPrior],
    left_chain: Optional[_BoundaryChain] = None,
    right_chain: Optional[_BoundaryChain] = None,
) -> tuple[list[_BoundaryPair], int, int, dict[str, int]]:
    reject_counts = _default_reject_counts()
    candidate_count = 0
    unknown_pair_count = 0
    use_legacy_side_gate = left_indices is None and left_chain is not None
    inward_projection_tolerance_m = max(
        0.0,
        float(config.pair_inward_projection_tolerance_m),
    )
    if left_indices is None:
        left_indices = (
            np.asarray(left_chain.filtered_indices, dtype=np.int64)
            if left_chain is not None
            else np.empty((0,), dtype=np.int64)
        )
    else:
        left_indices = np.asarray(left_indices, dtype=np.int64)
    if right_indices is None:
        right_indices = (
            np.asarray(right_chain.filtered_indices, dtype=np.int64)
            if right_chain is not None
            else np.empty((0,), dtype=np.int64)
        )
    else:
        right_indices = np.asarray(right_indices, dtype=np.int64)
    if left_indices.size == 0 or (right_indices.size == 0 and unknown_indices.size == 0):
        return [], 0, 0, reject_counts

    left_local = filtered_local[left_indices]
    if (
        use_legacy_side_gate
        and left_chain is not None
        and left_chain.tangents_local.shape[0] == left_indices.size
    ):
        left_tangents = np.asarray(left_chain.tangents_local, dtype=np.float64)
    else:
        left_tangents = _estimate_side_tangents(
            side_local=left_local,
            side="blue",
            neighbor_count=max(2, int(config.pairing_tangent_neighbor_count)),
        )

    candidate_options: list[dict[str, object]] = []

    for left_pos, anchor_filtered_idx in enumerate(left_indices):
        anchor_filtered_idx = int(anchor_filtered_idx)
        anchor_local = np.asarray(filtered_local[anchor_filtered_idx], dtype=np.float64)
        anchor_global = np.asarray(filtered_points[anchor_filtered_idx], dtype=np.float64)
        anchor_tangent = np.asarray(left_tangents[left_pos], dtype=np.float64)
        anchor_track_id = int(filtered_track_ids[anchor_filtered_idx])
        anchor_raw_color = (
            normalize_color(filtered_raw_colors[anchor_filtered_idx])
            if filtered_raw_colors is not None and anchor_filtered_idx < len(filtered_raw_colors)
            else "unknown"
        )
        inward_normal = _inward_normal(anchor_tangent, "blue")
        anchor_options: list[dict[str, object]] = []

        for partner_filtered_idx in right_indices:
            partner_filtered_idx = int(partner_filtered_idx)
            other_local = np.asarray(filtered_local[partner_filtered_idx], dtype=np.float64)
            partner_raw_color = (
                normalize_color(filtered_raw_colors[partner_filtered_idx])
                if filtered_raw_colors is not None and partner_filtered_idx < len(filtered_raw_colors)
                else "unknown"
            )
            if config.enforce_opposite_color_pairing:
                raw_pair = {anchor_raw_color, partner_raw_color}
                if raw_pair not in (
                    {"blue", "yellow"},
                    {"orange", "blue"},
                    {"orange", "yellow"},
                    {"orange"},
                ):
                    reject_counts["color"] += 1
                    continue
            delta = other_local - anchor_local
            width_m = float(np.hypot(delta[0], delta[1]))
            if width_m < float(config.min_pair_width_m) or width_m > float(config.max_pair_width_m):
                reject_counts["width_range"] += 1
                continue

            inward_distance = float(np.dot(delta, inward_normal))
            midpoint_local = 0.5 * (anchor_local + other_local)
            lateral_progress = float(anchor_local[1] - other_local[1])
            if config.enforce_geometry_pairing_gate:
                if use_legacy_side_gate:
                    wrong_side = inward_distance <= -inward_projection_tolerance_m
                else:
                    wrong_side = (
                        inward_distance <= -inward_projection_tolerance_m
                        and lateral_progress <= inward_projection_tolerance_m
                    )
                midpoint_outside_pair_span = (
                    abs(float(midpoint_local[1]))
                    > (0.5 * width_m)
                )
                if wrong_side or midpoint_outside_pair_span:
                    reject_counts["wrong_side"] += 1
                    continue

            candidate_count += 1
            longitudinal_offset = abs(float(other_local[0] - anchor_local[0]))
            progress_offset = abs(
                float(np.hypot(other_local[0], other_local[1]))
                - float(np.hypot(anchor_local[0], anchor_local[1]))
            )
            width_error = abs(width_m - float(expected_width_m))
            inward_penalty = max(0.0, -inward_distance)
            cost = float(width_m)
            anchor_options.append(
                {
                    "use_unknown": False,
                    "anchor_filtered_idx": anchor_filtered_idx,
                    "anchor_track_id": anchor_track_id,
                    "anchor_global": anchor_global,
                    "anchor_local": anchor_local,
                    "partner_filtered_idx": partner_filtered_idx,
                    "partner_track_id": int(filtered_track_ids[partner_filtered_idx]),
                    "partner_global": np.asarray(filtered_points[partner_filtered_idx], dtype=np.float64),
                    "partner_local": other_local,
                    "width_m": float(width_m),
                    "cost": float(cost),
                    "selection_cost": float(cost),
                    "sort_key": (
                        width_m,
                        width_error,
                        longitudinal_offset,
                        progress_offset,
                        inward_penalty,
                        abs(float(midpoint_local[1])),
                        partner_filtered_idx,
                    ),
                }
            )

        if config.allow_unknown_pair_completion and unknown_indices.size > 0:
            expected_partner_local = anchor_local + (inward_normal * float(expected_width_m))
            for filtered_idx in unknown_indices:
                unknown_idx = int(filtered_idx)
                if config.enforce_opposite_color_pairing:
                    reject_counts["color"] += 1
                    continue
                unknown_local = filtered_local[unknown_idx]
                delta = unknown_local - anchor_local
                width_m = float(np.hypot(delta[0], delta[1]))
                if width_m < float(config.min_pair_width_m) or width_m > float(config.max_pair_width_m):
                    continue
                inward_distance = float(np.dot(delta, inward_normal))
                midpoint_local = 0.5 * (anchor_local + unknown_local)
                if config.enforce_geometry_pairing_gate:
                    if use_legacy_side_gate:
                        wrong_side = inward_distance <= -inward_projection_tolerance_m
                    else:
                        wrong_side = (
                            inward_distance <= -inward_projection_tolerance_m
                            and float(anchor_local[1] - unknown_local[1]) <= inward_projection_tolerance_m
                        )
                    midpoint_outside_pair_span = (
                        abs(float(midpoint_local[1]))
                        > (0.5 * width_m)
                    )
                    if wrong_side or midpoint_outside_pair_span:
                        reject_counts["wrong_side"] += 1
                        continue
                longitudinal_error = abs(float(np.dot(unknown_local - expected_partner_local, anchor_tangent)))
                width_error = abs(width_m - float(expected_width_m))
                radial_error = float(np.hypot(*(unknown_local - expected_partner_local)))
                if longitudinal_error > float(config.unknown_pair_max_longitudinal_error_m):
                    continue
                if width_error > float(config.unknown_pair_max_width_error_m):
                    continue
                if radial_error > float(config.unknown_pair_search_radius_m):
                    continue
                candidate_count += 1
                inward_penalty = max(0.0, -inward_distance)
                anchor_options.append(
                    {
                        "use_unknown": True,
                        "anchor_filtered_idx": anchor_filtered_idx,
                        "anchor_track_id": anchor_track_id,
                        "anchor_global": anchor_global,
                        "anchor_local": anchor_local,
                        "partner_filtered_idx": unknown_idx,
                        "partner_track_id": int(filtered_track_ids[unknown_idx]),
                        "partner_global": np.asarray(filtered_points[unknown_idx], dtype=np.float64),
                        "partner_local": np.asarray(filtered_local[unknown_idx], dtype=np.float64),
                        "width_m": float(width_m),
                        "cost": float(
                            longitudinal_error + width_error + radial_error + inward_penalty + 0.05
                        ),
                        "selection_cost": float(
                            longitudinal_error + width_error + radial_error + inward_penalty + 0.05
                        ),
                        "sort_key": (
                            inward_penalty,
                            longitudinal_error,
                            width_error,
                            radial_error,
                            unknown_idx,
                        ),
                    }
                )
        if not anchor_options:
            continue
        anchor_options.sort(key=lambda option: option["sort_key"])
        candidate_options.extend(anchor_options)

    candidate_options.sort(
        key=lambda option: (
            float(option["selection_cost"]),
            1 if bool(option["use_unknown"]) else 0,
            float(option["width_m"]),
            float(np.hypot(option["anchor_local"][0], option["anchor_local"][1])),
            float(option["anchor_local"][0]),
            abs(float(option["anchor_local"][1])),
            int(option["anchor_track_id"]),
            int(option["partner_track_id"]),
        )
    )

    pairs: list[_BoundaryPair] = []
    used_left_indices: set[int] = set()
    used_partner_indices: set[int] = set()
    used_unknown_indices: set[int] = set()
    for chosen in candidate_options:
        left_filtered_idx = int(chosen["anchor_filtered_idx"])
        partner_filtered_idx = int(chosen["partner_filtered_idx"])
        if left_filtered_idx in used_left_indices or partner_filtered_idx in used_partner_indices:
            continue
        if bool(chosen["use_unknown"]) and partner_filtered_idx in used_unknown_indices:
            continue
        pair = _BoundaryPair(
            left_filtered_idx=left_filtered_idx,
            right_filtered_idx=partner_filtered_idx,
            left_track_id=int(chosen["anchor_track_id"]),
            right_track_id=int(chosen["partner_track_id"]),
            left_global=np.asarray(chosen["anchor_global"], dtype=np.float64),
            right_global=np.asarray(chosen["partner_global"], dtype=np.float64),
            left_local=np.asarray(chosen["anchor_local"], dtype=np.float64),
            right_local=np.asarray(chosen["partner_local"], dtype=np.float64),
            width_m=float(chosen["width_m"]),
        )
        pairs.append(pair)
        used_left_indices.add(left_filtered_idx)
        used_partner_indices.add(partner_filtered_idx)
        if bool(chosen["use_unknown"]):
            used_unknown_indices.add(partner_filtered_idx)
            unknown_pair_count += 1

    return pairs, candidate_count, unknown_pair_count, reject_counts


def _trim_pairs_by_midpoint_step_length(
    pairs: list[_BoundaryPair],
    *,
    max_segment_length_m: float,
) -> list[_BoundaryPair]:
    if len(pairs) <= 1:
        return list(pairs)
    limit_m = float(max_segment_length_m)
    if not math.isfinite(limit_m) or limit_m <= 0.0:
        return list(pairs)

    trimmed: list[_BoundaryPair] = [pairs[0]]
    previous_midpoint = np.asarray(pairs[0].midpoint_global, dtype=np.float64)
    for pair in pairs[1:]:
        midpoint = np.asarray(pair.midpoint_global, dtype=np.float64)
        if float(np.hypot(*(midpoint - previous_midpoint))) > limit_m:
            break
        trimmed.append(pair)
        previous_midpoint = midpoint
    return trimmed


def _order_pairs_into_midpoint_chain(
    pairs: list[_BoundaryPair],
    *,
    config: Optional[MidpointPlannerConfig] = None,
    max_segment_length_m: Optional[float] = None,
) -> list[_BoundaryPair]:
    if len(pairs) <= 1:
        return list(pairs)

    if config is None:
        config = MidpointPlannerConfig()
        if max_segment_length_m is not None:
            config.max_midpoint_segment_length_m = float(max_segment_length_m)

    limit_m = float(config.max_midpoint_segment_length_m)
    if not math.isfinite(limit_m) or limit_m <= 0.0:
        return list(pairs)

    local_midpoints = [
        np.asarray(pair.midpoint_local, dtype=np.float64)
        for pair in pairs
    ]
    ordered_path_length_m = 0.0

    def _start_key(idx: int) -> tuple[float, float, float, float, int]:
        midpoint = local_midpoints[idx]
        x_val = float(midpoint[0])
        y_val = float(midpoint[1])
        distance = float(np.hypot(x_val, y_val))
        return (
            0.0 if x_val >= 0.0 else 1.0,
            distance,
            max(0.0, -x_val),
            abs(y_val),
            idx,
        )

    start_idx = min(range(len(pairs)), key=_start_key)
    ordered: list[_BoundaryPair] = [pairs[start_idx]]
    used_indices: set[int] = {start_idx}
    current_midpoint = np.asarray(local_midpoints[start_idx], dtype=np.float64)

    while len(ordered) < len(pairs):
        reference_direction, _ = _midpoint_progress_reference(
            ordered_pairs=ordered,
            handoff_distance_m=float(config.midpoint_order_reference_handoff_m),
            history_size=max(2, int(config.midpoint_order_history_size)),
            ordered_path_length_m=float(ordered_path_length_m),
        )
        best_idx: Optional[int] = None
        best_cost = float("inf")

        for idx, midpoint in enumerate(local_midpoints):
            if idx in used_indices:
                continue
            delta = midpoint - current_midpoint
            distance = float(np.hypot(delta[0], delta[1]))
            if distance <= 1e-9 or distance > limit_m:
                continue

            forward_progress_m = float(np.dot(delta, reference_direction))
            max_backward_step_m = max(
                float(config.midpoint_order_backtrack_tolerance_m),
                0.5 * float(config.min_forward_progress_m),
            )
            if forward_progress_m < -max_backward_step_m:
                continue

            backward_progress_penalty = max(0.0, -forward_progress_m)
            width_jump_penalty = max(
                0.0,
                abs(float(pairs[idx].width_m) - float(ordered[-1].width_m))
                - float(config.max_width_jump_m),
            )
            cost = (
                distance
                + (2.0 * backward_progress_penalty)
                + (0.25 * width_jump_penalty)
            )
            if cost < best_cost:
                best_cost = cost
                best_idx = idx

        if best_idx is None:
            break

        step_delta = np.asarray(local_midpoints[best_idx], dtype=np.float64) - current_midpoint
        ordered_path_length_m += float(np.hypot(step_delta[0], step_delta[1]))
        ordered.append(pairs[best_idx])
        used_indices.add(best_idx)
        current_midpoint = np.asarray(local_midpoints[best_idx], dtype=np.float64)

    return ordered


def _midpoint_progress_reference(
    *,
    ordered_pairs: list[_BoundaryPair],
    handoff_distance_m: float,
    history_size: int,
    ordered_path_length_m: float,
) -> tuple[np.ndarray, bool]:
    vehicle_forward = np.asarray([1.0, 0.0], dtype=np.float64)
    if len(ordered_pairs) <= 1:
        # Use direction from vehicle toward the first pair as the initial
        # reference so candidates behind the vehicle in x but further along
        # a turning track are not immediately filtered out.
        mp = np.asarray(ordered_pairs[-1].midpoint_local, dtype=np.float64)
        mp_norm = float(np.hypot(mp[0], mp[1]))
        if mp_norm > 1e-9:
            return mp / mp_norm, True
        return vehicle_forward, True

    midpoint_history = np.asarray(
        [pair.midpoint_local for pair in ordered_pairs],
        dtype=np.float64,
    )
    trend_start = max(0, midpoint_history.shape[0] - int(history_size))
    trend_delta = midpoint_history[-1] - midpoint_history[trend_start]
    trend_norm = float(np.hypot(trend_delta[0], trend_delta[1]))
    if trend_norm <= 1e-9:
        trend_delta = midpoint_history[-1] - midpoint_history[-2]
        trend_norm = float(np.hypot(trend_delta[0], trend_delta[1]))
    if trend_norm <= 1e-9:
        return vehicle_forward, True
    trend_direction = trend_delta / trend_norm

    handoff_distance_m = max(0.5, float(handoff_distance_m))
    if ordered_path_length_m >= handoff_distance_m:
        return trend_direction, False

    alpha = float(np.clip(ordered_path_length_m / handoff_distance_m, 0.0, 1.0))
    blended = ((1.0 - alpha) * vehicle_forward) + (alpha * trend_direction)
    blended_norm = float(np.hypot(blended[0], blended[1]))
    if blended_norm <= 1e-9:
        return vehicle_forward, True
    return (blended / blended_norm), True


def _estimate_side_tangents(
    *,
    side_local: np.ndarray,
    side: str,
    neighbor_count: int,
) -> np.ndarray:
    if side_local.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if side_local.shape[0] == 1:
        return np.asarray([[1.0, 0.0]], dtype=np.float64)

    tangents = np.empty((side_local.shape[0], 2), dtype=np.float64)
    for idx, anchor in enumerate(side_local):
        delta = side_local - anchor
        distance = np.hypot(delta[:, 0], delta[:, 1])
        neighbor_order = np.argsort(distance)
        support_indices = [
            int(pos)
            for pos in neighbor_order
            if int(pos) != idx
        ][: max(1, min(int(neighbor_count), side_local.shape[0] - 1))]
        support = np.vstack((anchor[None, :], side_local[support_indices]))
        centered = support - np.mean(support, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        tangent = np.asarray(vh[0], dtype=np.float64)
        tangent_norm = float(np.hypot(tangent[0], tangent[1]))
        if tangent_norm <= 1e-9:
            tangent = np.asarray([1.0, 0.0], dtype=np.float64)
        else:
            tangent = tangent / tangent_norm
        tangents[idx] = _orient_tangent_for_side(
            tangent=tangent,
            side=side,
            anchor_local=np.asarray(anchor, dtype=np.float64),
        )
    return tangents


def _orient_tangent_for_side(
    *,
    tangent: np.ndarray,
    side: str,
    anchor_local: np.ndarray,
) -> np.ndarray:
    best_tangent = np.asarray(tangent, dtype=np.float64)
    best_score: Optional[tuple[float, float, float]] = None
    center_hint = -np.asarray(anchor_local, dtype=np.float64)
    for candidate in (tangent, -tangent):
        inward = _inward_normal(np.asarray(candidate, dtype=np.float64), side)
        score = (
            float(np.dot(inward, center_hint)),
            float(candidate[0]),
            -abs(float(candidate[1])),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_tangent = np.asarray(candidate, dtype=np.float64)
    return best_tangent


def _select_fallback_chain(
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
    config: MidpointPlannerConfig,
) -> tuple[Optional[_BoundaryChain], str]:
    candidates: list[tuple[tuple[float, float, float, int], _BoundaryChain, str]] = []
    min_chain_length = int(config.min_chain_length)

    if left_chain.filtered_indices.size >= min_chain_length:
        candidates.append(
            (
                (
                    -float(left_chain.forward_extent_m),
                    -float(left_chain.filtered_indices.size),
                    float(left_chain.mean_heading_change_rad),
                    0,
                ),
                left_chain,
                "blue",
            )
        )
    if right_chain.filtered_indices.size >= min_chain_length:
        candidates.append(
            (
                (
                    -float(right_chain.forward_extent_m),
                    -float(right_chain.filtered_indices.size),
                    float(right_chain.mean_heading_change_rad),
                    1,
                ),
                right_chain,
                "yellow",
            )
        )

    if not candidates:
        return None, ""
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def _offset_boundary_chain(
    *,
    chain: _BoundaryChain,
    side: str,
    width_m: float,
) -> np.ndarray:
    if chain.global_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    offset = []
    offset_distance = 0.5 * float(width_m)
    tangents_global = _estimate_tangents(chain.global_points)
    for idx, point in enumerate(chain.global_points):
        normal = _inward_normal(tangents_global[idx], side)
        offset.append(point + (offset_distance * normal))
    return np.asarray(offset, dtype=np.float64)


def _inward_normal(tangent: np.ndarray, side: str) -> np.ndarray:
    if side == "blue":
        normal = np.asarray([tangent[1], -tangent[0]], dtype=np.float64)
    else:
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    norm = float(np.hypot(normal[0], normal[1]))
    if norm <= 1e-9:
        return np.asarray([0.0, 0.0], dtype=np.float64)
    return normal / norm


def _estimate_tangents(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.shape[0] == 1:
        return np.asarray([[1.0, 0.0]], dtype=np.float64)

    tangents = np.empty_like(points)
    for idx in range(points.shape[0]):
        if idx == 0:
            delta = points[1] - points[0]
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


def _finalize_path(points: np.ndarray, config: MidpointPlannerConfig) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    path = np.asarray(points, dtype=np.float64)
    if int(config.smoothing_window) > 1:
        path = _moving_average(path, int(config.smoothing_window))
    path = _resample_path(
        path,
        resolution_m=float(config.path_resolution_m),
        max_length_m=float(config.max_path_length_m),
    )
    return path


def _moving_average(points: np.ndarray, window: int) -> np.ndarray:
    if points.shape[0] <= 2 or window <= 1:
        return np.asarray(points, dtype=np.float64)

    radius = max(1, int(window)) // 2
    out = np.empty_like(points)
    for idx in range(points.shape[0]):
        start = max(0, idx - radius)
        stop = min(points.shape[0], idx + radius + 1)
        out[idx] = np.mean(points[start:stop], axis=0)
    return out


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
    if delta.size == 0:
        return 0.0
    return float(np.max(np.abs(delta)))


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
    return float(np.hypot(path_local[0, 0], path_local[0, 1]))


def _path_self_intersects(points: np.ndarray) -> bool:
    if points.shape[0] < 4:
        return False
    for idx in range(points.shape[0] - 1):
        a0 = points[idx]
        a1 = points[idx + 1]
        for jdx in range(idx + 2, points.shape[0] - 1):
            if idx == 0 and jdx == points.shape[0] - 2:
                continue
            b0 = points[jdx]
            b1 = points[jdx + 1]
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


def _angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    angle_a = math.atan2(float(vec_a[1]), float(vec_a[0]))
    angle_b = math.atan2(float(vec_b[1]), float(vec_b[0]))
    return math.atan2(math.sin(angle_b - angle_a), math.cos(angle_b - angle_a))


def _merge_reject_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)


def _default_reject_counts() -> dict[str, int]:
    return {
        "color": 0,
        "wrong_side": 0,
        "width": 0,
        "width_range": 0,
        "width_prior": 0,
        "orientation": 0,
        "progress": 0,
        "near_field_continuity": 0,
        "midpoint_kink": 0,
        "seed_distance": 0,
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
) -> MidpointPlannerResult:
    return MidpointPlannerResult(
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
        prevalidation_centerline=np.empty((0, 2), dtype=np.float64),
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
    result: MidpointPlannerResult,
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
    planner_mode: str,
    filtered_track_width_m: float,
    left_boundary_count: Optional[int] = None,
    right_boundary_count: Optional[int] = None,
) -> MidpointPlannerResult:
    result.left_chain_length = (
        int(left_chain.filtered_indices.size)
        if left_boundary_count is None
        else int(left_boundary_count)
    )
    result.right_chain_length = (
        int(right_chain.filtered_indices.size)
        if right_boundary_count is None
        else int(right_boundary_count)
    )
    result.left_boundary = left_chain.global_points
    result.right_boundary = right_chain.global_points
    result.planner_mode = planner_mode
    result.filtered_track_width_m = float(filtered_track_width_m)
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))
