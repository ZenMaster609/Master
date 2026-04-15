"""Core geometry for the single-boundary planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from sim_car.cones.tracking.fusion import normalize_color


@dataclass
class SingleBoundaryPlannerConfig:
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
    max_step_m: float = 6.0
    max_heading_change_rad: float = 1.0
    min_forward_progress_m: float = 0.2
    min_chain_length: int = 2

    min_pair_width_m: float = 2.2
    max_pair_width_m: float = 5.5
    max_width_jump_m: float = 0.8
    min_pair_count: int = 3
    pair_reassignment_margin: float = 0.25

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

    min_path_points: int = 2
    min_forward_extent_m: float = 1.0
    jump_check_horizon_m: float = 8.0
    max_near_field_lateral_jump_m: float = 0.6
    max_near_field_lateral_jump_m_sparse_pairs: float = 0.9
    max_near_field_lateral_jump_m_single_boundary: float = 5.0
    max_start_heading_error_rad: float = 1.0


@dataclass
class SingleBoundaryPlannerPrior:
    previous_centerline: Optional[np.ndarray] = None
    previous_width_m: Optional[float] = None
    previous_mode: str = "none"
    previous_pairs: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class SingleBoundaryPlannerResult:
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


def compute_single_boundary_centerline(
    points_xy: np.ndarray,
    colors: list[str],
    confidences: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: SingleBoundaryPlannerConfig,
    prior: Optional[SingleBoundaryPlannerPrior] = None,
    track_ids: Optional[np.ndarray] = None,
) -> SingleBoundaryPlannerResult:
    """Compute a local centerline using only one visible boundary.

    Attempts normal paired midpoint planning first. If that fails and a single boundary
    chain is available, offsets it laterally by the estimated track half-width to produce
    a centerline. Returns a result with ``status='ok'`` on success.
    """
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
    unknown_mask = np.array([color == "unknown" for color in normalized], dtype=bool)
    selected_mask = mask_geom & mask_conf & (
        colored_mask | (unknown_mask if config.allow_unknown_pair_completion else False)
    )

    filtered_points = points_xy[selected_mask]
    filtered_local = local_points[selected_mask]
    filtered_track_ids = track_ids[selected_mask]
    filtered_colors = [normalized[idx] for idx in np.where(selected_mask)[0]]
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

    pairing_possible = (
        (
            left_chain.filtered_indices.size >= int(config.min_chain_length)
            or right_chain.filtered_indices.size >= int(config.min_chain_length)
        ) and (
            right_chain.filtered_indices.size > 0
            or left_chain.filtered_indices.size > 0
            or unknown_indices.size > 0
        )
    )
    if pairing_possible:
        pairs, candidate_count, unknown_pair_count, pair_reject_counts = _pair_boundary_chains(
            filtered_points=filtered_points,
            filtered_local=filtered_local,
            filtered_track_ids=filtered_track_ids,
            left_chain=left_chain,
            right_chain=right_chain,
            unknown_indices=unknown_indices,
            expected_width_m=expected_width,
            config=config,
            prior=prior,
        )
        _merge_reject_counts(reject_counts, pair_reject_counts)
        if pairs:
            measured_width_m = float(np.median([pair.width_m for pair in pairs]))
        unknown_pair_count = 0
        selected_pair_track_ids = np.empty((0, 2), dtype=np.int64)
    else:
        unknown_pair_count = 0
        selected_pair_track_ids = np.empty((0, 2), dtype=np.int64)

    fallback_chain, fallback_side = _select_fallback_chain(left_chain, right_chain, config)
    if fallback_chain is not None:
        planner_mode = "single_boundary"
        active_boundary_side = fallback_side
        used_fallback = True
        raw_offset_path = _offset_boundary_chain(
            chain=fallback_chain,
            side=fallback_side,
            width_m=expected_width,
        )
    else:
        return _result_with_metadata(
            result=_empty_result(
                "no reliable boundary chain",
                filtered_points=filtered_points,
                filtered_colors=filtered_colors,
                left_boundary=left_chain.global_points,
                right_boundary=right_chain.global_points,
                reject_counts=reject_counts,
                reject_reason="no reliable boundary chain",
            ),
            left_chain=left_chain,
            right_chain=right_chain,
            planner_mode="none",
            filtered_track_width_m=expected_width,
        )

    raw_curve = midpoint_chain if planner_mode == "midpoint" else raw_offset_path
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
    if planner_mode == "single_boundary":
        continuity_threshold_m = max(
            continuity_threshold_m,
            float(config.max_near_field_lateral_jump_m_single_boundary),
        )
    elif len(pairs) <= max(3, int(config.min_pair_count)):
        continuity_threshold_m = max(
            continuity_threshold_m,
            float(config.max_near_field_lateral_jump_m_sparse_pairs),
        )

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

    result = SingleBoundaryPlannerResult(
        filtered_points=filtered_points,
        filtered_colors=filtered_colors,
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=candidate_edges,
        selected_edges=selected_edges,
        selected_pair_track_ids=selected_pair_track_ids,
        midpoints_raw=midpoint_chain,
        centerline=centerline,
        left_boundary=left_chain.global_points,
        right_boundary=right_chain.global_points,
        used_fallback=used_fallback,
        status=status,
        candidate_count=int(candidate_count),
        selected_chain_length=int(
            len(pairs) if planner_mode == "midpoint" else fallback_chain.filtered_indices.size  # type: ignore[union-attr]
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
        active_boundary_side=active_boundary_side,
        raw_offset_path=raw_offset_path,
        pair_segments=np.asarray(
            [[pair.left_global, pair.right_global] for pair in pairs],
            dtype=np.float64,
        ) if pairs else np.empty((0, 2, 2), dtype=np.float64),
        accepted_pair_count=len(pairs),
        left_chain_length=int(left_chain.filtered_indices.size),
        right_chain_length=int(right_chain.filtered_indices.size),
        filtered_track_width_m=float(expected_width),
        unknown_pair_count=int(unknown_pair_count),
    )
    return result


def update_track_width_estimate(
    previous_width_m: Optional[float],
    measured_width_m: Optional[float],
    config: SingleBoundaryPlannerConfig,
) -> float:
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


def _geometry_filter(local_points: np.ndarray, config: SingleBoundaryPlannerConfig) -> np.ndarray:
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
    config: SingleBoundaryPlannerConfig,
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

    filtered_indices = side_indices[np.asarray(chain_positions, dtype=np.int64)]
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
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
    unknown_indices: np.ndarray,
    expected_width_m: float,
    config: SingleBoundaryPlannerConfig,
    prior: Optional[SingleBoundaryPlannerPrior],
) -> tuple[list[_BoundaryPair], int, int, dict[str, int]]:
    reject_counts = _default_reject_counts()
    candidate_count = 0
    unknown_pair_count = 0

    if left_chain.filtered_indices.size >= right_chain.filtered_indices.size:
        anchor_chain = left_chain
        other_chain = right_chain
        anchor_side = "blue"
    elif right_chain.filtered_indices.size > 0:
        anchor_chain = right_chain
        other_chain = left_chain
        anchor_side = "yellow"
    elif left_chain.filtered_indices.size > 0:
        anchor_chain = left_chain
        other_chain = right_chain
        anchor_side = "blue"
    else:
        anchor_chain = right_chain
        other_chain = left_chain
        anchor_side = "yellow"

    pairs: list[_BoundaryPair] = []
    next_other_start = 0
    last_width: Optional[float] = None
    last_partner_progress = float("-inf")
    used_unknown_indices: set[int] = set()
    consecutive_unknown_pairs = 0
    previous_pairs = list(prior.previous_pairs) if prior is not None else []
    previous_partner_by_anchor: dict[int, int] = {}
    for left_track_id, right_track_id in previous_pairs:
        if anchor_side == 'blue':
            previous_partner_by_anchor[int(left_track_id)] = int(right_track_id)
        else:
            previous_partner_by_anchor[int(right_track_id)] = int(left_track_id)

    for anchor_pos in range(anchor_chain.filtered_indices.size):
        anchor_local = anchor_chain.local_points[anchor_pos]
        anchor_global = anchor_chain.global_points[anchor_pos]
        anchor_tangent = anchor_chain.tangents_local[anchor_pos]
        anchor_filtered_idx = int(anchor_chain.filtered_indices[anchor_pos])
        anchor_track_id = int(filtered_track_ids[anchor_filtered_idx])
        inward_normal = _inward_normal(anchor_tangent, anchor_side)
        candidate_options: list[dict[str, object]] = []
        for other_pos in range(next_other_start, other_chain.filtered_indices.size):
            other_local = other_chain.local_points[other_pos]
            if float(other_local[0]) < (last_partner_progress - float(config.min_forward_progress_m)):
                reject_counts["progress"] += 1
                continue
            delta = other_local - anchor_local
            width_m = float(np.hypot(delta[0], delta[1]))
            if width_m < float(config.min_pair_width_m) or width_m > float(config.max_pair_width_m):
                reject_counts["width_range"] += 1
                continue

            inward_distance = float(np.dot(delta, inward_normal))
            if inward_distance <= 0.0:
                reject_counts["wrong_side"] += 1
                continue

            candidate_count += 1
            longitudinal_offset = abs(float(np.dot(delta, anchor_tangent)))
            cost = longitudinal_offset + abs(width_m - float(expected_width_m))
            partner_filtered_idx = int(other_chain.filtered_indices[other_pos])
            candidate_options.append(
                {
                    'use_unknown': False,
                    'other_pos': other_pos,
                    'partner_filtered_idx': partner_filtered_idx,
                    'partner_track_id': int(filtered_track_ids[partner_filtered_idx]),
                    'partner_global': np.asarray(other_chain.global_points[other_pos], dtype=np.float64),
                    'partner_local': np.asarray(other_chain.local_points[other_pos], dtype=np.float64),
                    'width_m': float(width_m),
                    'cost': float(cost),
                    'sort_key': (
                        longitudinal_offset,
                        abs(width_m - float(expected_width_m)),
                        width_m,
                        other_pos,
                    ),
                }
            )

        if config.allow_unknown_pair_completion and unknown_indices.size > 0:
            expected_partner_local = anchor_local + (inward_normal * float(expected_width_m))
            for filtered_idx in unknown_indices:
                unknown_idx = int(filtered_idx)
                if unknown_idx in used_unknown_indices:
                    continue
                unknown_local = filtered_local[unknown_idx]
                if float(unknown_local[0]) < (last_partner_progress - float(config.min_forward_progress_m)):
                    continue
                delta = unknown_local - anchor_local
                width_m = float(np.hypot(delta[0], delta[1]))
                if width_m < float(config.min_pair_width_m) or width_m > float(config.max_pair_width_m):
                    continue
                inward_distance = float(np.dot(delta, inward_normal))
                if inward_distance <= 0.0:
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
                candidate_options.append(
                    {
                        'use_unknown': True,
                        'other_pos': -1,
                        'partner_filtered_idx': unknown_idx,
                        'partner_track_id': int(filtered_track_ids[unknown_idx]),
                        'partner_global': np.asarray(filtered_points[unknown_idx], dtype=np.float64),
                        'partner_local': np.asarray(filtered_local[unknown_idx], dtype=np.float64),
                        'width_m': float(width_m),
                        'cost': float(longitudinal_error + width_error + radial_error + 0.05),
                        'sort_key': (
                            longitudinal_error,
                            width_error,
                            radial_error,
                            unknown_idx,
                        ),
                    }
                )

        if not candidate_options:
            continue
        candidate_options.sort(key=lambda option: option['sort_key'])
        chosen = candidate_options[0]
        preferred_partner_track_id = previous_partner_by_anchor.get(anchor_track_id)
        if preferred_partner_track_id is not None:
            preferred = next(
                (option for option in candidate_options if int(option['partner_track_id']) == preferred_partner_track_id),
                None,
            )
            if preferred is not None:
                best_cost = float(chosen['cost'])
                preferred_cost = float(preferred['cost'])
                if preferred_cost <= (best_cost + float(config.pair_reassignment_margin)):
                    chosen = preferred

        use_unknown = bool(chosen['use_unknown'])
        if use_unknown and consecutive_unknown_pairs >= int(config.max_consecutive_unknown_pairs):
            non_unknown = next((option for option in candidate_options if not bool(option['use_unknown'])), None)
            if non_unknown is None:
                continue
            chosen = non_unknown
            use_unknown = False

        candidate_width = float(chosen['width_m'])
        if last_width is not None and abs(candidate_width - last_width) > float(config.max_width_jump_m):
            reject_counts["width"] += 1
            continue

        if anchor_side == "blue":
            right_filtered_idx = int(chosen['partner_filtered_idx'])
            right_global = np.asarray(chosen['partner_global'], dtype=np.float64)
            right_local = np.asarray(chosen['partner_local'], dtype=np.float64)
            pair = _BoundaryPair(
                left_filtered_idx=anchor_filtered_idx,
                right_filtered_idx=right_filtered_idx,
                left_track_id=anchor_track_id,
                right_track_id=int(chosen['partner_track_id']),
                left_global=np.asarray(anchor_global, dtype=np.float64),
                right_global=right_global,
                left_local=np.asarray(anchor_local, dtype=np.float64),
                right_local=right_local,
                width_m=float(candidate_width),
            )
        else:
            left_filtered_idx = int(chosen['partner_filtered_idx'])
            left_global = np.asarray(chosen['partner_global'], dtype=np.float64)
            left_local = np.asarray(chosen['partner_local'], dtype=np.float64)
            pair = _BoundaryPair(
                left_filtered_idx=left_filtered_idx,
                right_filtered_idx=anchor_filtered_idx,
                left_track_id=int(chosen['partner_track_id']),
                right_track_id=anchor_track_id,
                left_global=left_global,
                right_global=np.asarray(anchor_global, dtype=np.float64),
                left_local=left_local,
                right_local=np.asarray(anchor_local, dtype=np.float64),
                width_m=float(candidate_width),
            )
        pairs.append(pair)
        if use_unknown:
            used_unknown_indices.add(int(chosen['partner_filtered_idx']))
            unknown_pair_count += 1
            consecutive_unknown_pairs += 1
            last_partner_progress = float(filtered_local[int(chosen['partner_filtered_idx']), 0])
        else:
            next_other_start = int(chosen['other_pos']) + 1
            consecutive_unknown_pairs = 0
            last_partner_progress = float(np.asarray(chosen['partner_local'])[0])
        last_width = candidate_width

    return pairs, candidate_count, unknown_pair_count, reject_counts


def _select_fallback_chain(
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
    config: SingleBoundaryPlannerConfig,
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


def _finalize_path(points: np.ndarray, config: SingleBoundaryPlannerConfig) -> np.ndarray:
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
    alignment = _path_alignment_metrics(
        current_local=current_local,
        previous_local=previous_local,
        horizon_m=min(float(horizon_m), 3.0),
    )
    return {
        "lateral_max_m": float(alignment["lateral_max_m"]),
        "lateral_mean_m": float(alignment["lateral_mean_m"]),
        "displacement_max_m": float(alignment["displacement_max_m"]),
        "displacement_mean_m": float(alignment["displacement_mean_m"]),
    }


def _path_alignment_metrics(
    *,
    current_local: Optional[np.ndarray],
    previous_local: Optional[np.ndarray],
    horizon_m: float,
) -> dict[str, float]:
    empty = {
        "lateral_max_m": 0.0,
        "lateral_mean_m": 0.0,
        "displacement_max_m": 0.0,
        "displacement_mean_m": 0.0,
        "heading_delta_rad": 0.0,
    }
    if current_local is None or previous_local is None:
        return empty
    current_prefix = _local_forward_prefix(
        np.asarray(current_local, dtype=np.float64),
        horizon_m=float(horizon_m),
    )
    previous_prefix = _local_forward_prefix(
        np.asarray(previous_local, dtype=np.float64),
        horizon_m=float(horizon_m),
    )
    if current_prefix.shape[0] < 2 or previous_prefix.shape[0] < 2:
        return empty

    count = min(current_prefix.shape[0], previous_prefix.shape[0])
    if count < 2:
        return empty
    delta = current_prefix[:count] - previous_prefix[:count]
    lateral = np.abs(delta[:, 1])
    displacement = np.hypot(delta[:, 0], delta[:, 1])
    current_heading = _path_start_heading_error(current_prefix[:count])
    previous_heading = _path_start_heading_error(previous_prefix[:count])
    heading_delta = abs(
        float(
            math.atan2(
                math.sin(current_heading - previous_heading),
                math.cos(current_heading - previous_heading),
            )
        )
    )
    return {
        "lateral_max_m": float(np.max(lateral)) if lateral.size else 0.0,
        "lateral_mean_m": float(np.mean(lateral)) if lateral.size else 0.0,
        "displacement_max_m": float(np.max(displacement)) if displacement.size else 0.0,
        "displacement_mean_m": float(np.mean(displacement)) if displacement.size else 0.0,
        "heading_delta_rad": heading_delta,
    }


def _local_forward_prefix(path_local: np.ndarray, *, horizon_m: float) -> np.ndarray:
    pts = np.asarray(path_local, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)
    valid_mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & (pts[:, 0] >= -0.1)
    pts = pts[valid_mask]
    if pts.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)
    cumulative = _path_cumulative_lengths(pts)
    total = min(float(cumulative[-1]), max(0.25, float(horizon_m)))
    if total <= 1e-6:
        return np.asarray(pts[:1], dtype=np.float64)
    return _resample_path(pts, 0.25, total)


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
) -> SingleBoundaryPlannerResult:
    return SingleBoundaryPlannerResult(
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
    result: SingleBoundaryPlannerResult,
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
    planner_mode: str,
    filtered_track_width_m: float,
) -> SingleBoundaryPlannerResult:
    result.left_chain_length = int(left_chain.filtered_indices.size)
    result.right_chain_length = int(right_chain.filtered_indices.size)
    result.left_boundary = left_chain.global_points
    result.right_boundary = right_chain.global_points
    result.planner_mode = planner_mode
    result.filtered_track_width_m = float(filtered_track_width_m)
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))
