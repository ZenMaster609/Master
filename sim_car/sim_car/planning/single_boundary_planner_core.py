"""Core geometry for the single-boundary planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.tracked_cone_planner_geometry import (
    build_boundary_chain_data,
    estimate_tangents as _estimate_tangents,
    inward_distance,
    inward_normal as _inward_normal,
    pair_width_in_range,
    prefer_previous_partner_option,
    unknown_partner_check,
    unknown_partner_within_limits,
    update_track_width_estimate,
    width_jump_exceeds,
)


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


@dataclass
class _FilteredCones:
    points: np.ndarray
    local: np.ndarray
    track_ids: np.ndarray
    colors: list[str]
    colored_count: int


def _filter_and_order_cones(
    points_xy: np.ndarray,
    colors: list[str],
    confidences: np.ndarray,
    track_ids: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: SingleBoundaryPlannerConfig,
) -> "_FilteredCones":
    normalized = [normalize_color(c) for c in colors]
    local_points = _to_vehicle_frame(points_xy, vehicle_xy, vehicle_yaw)

    mask_geom = _geometry_filter(local_points, config)
    mask_conf = confidences >= float(config.min_confidence)
    colored_mask = np.array([c in {"blue", "yellow"} for c in normalized], dtype=bool)
    unknown_mask = np.array([c == "unknown" for c in normalized], dtype=bool)
    selected_mask = mask_geom & mask_conf & (
        colored_mask | (unknown_mask if config.allow_unknown_pair_completion else False)
    )

    indices = np.where(selected_mask)[0]
    filtered_points = points_xy[selected_mask]
    filtered_local = local_points[selected_mask]
    filtered_track_ids = track_ids[selected_mask]
    filtered_colors = [normalized[i] for i in indices]
    colored_count = int(np.count_nonzero(
        np.array([c in {"blue", "yellow"} for c in filtered_colors], dtype=bool)
    ))

    order = _deterministic_order(filtered_local, filtered_points, filtered_colors)
    return _FilteredCones(
        points=filtered_points[order],
        local=filtered_local[order],
        track_ids=filtered_track_ids[order],
        colors=[filtered_colors[i] for i in order],
        colored_count=colored_count,
    )


def _validate_path(
    centerline: np.ndarray,
    centerline_local: np.ndarray,
    near_field: dict,
    heading_delta_max: float,
    continuity_threshold_m: float,
    reject_counts: dict[str, int],
    config: SingleBoundaryPlannerConfig,
) -> str:
    """Run path quality checks. Returns reject reason (empty string means ok). Mutates reject_counts."""
    if centerline.shape[0] < int(config.min_path_points):
        return "path has too few points"
    if not np.all(np.isfinite(centerline)):
        return "path contains non-finite geometry"
    if _forward_extent_m(centerline_local) < float(config.min_forward_extent_m):
        return "path forward extent too short"
    start_heading_error = abs(_path_start_heading_error(centerline_local))
    if start_heading_error > float(config.max_start_heading_error_rad):
        reject_counts["midpoint_kink"] += 1
        return "path heading flip near vehicle"
    if near_field["lateral_max_m"] > continuity_threshold_m:
        reject_counts["near_field_continuity"] += 1
        return "near-field continuity rejected fresh path"
    if heading_delta_max > float(config.max_heading_delta_rad):
        reject_counts["midpoint_kink"] += 1
        return "path heading delta exceeded limit"
    if _path_self_intersects(centerline):
        return "path self-crossing detected"
    return ""


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

    If a single boundary
    chain is available, offsets it laterally by the estimated track half-width to produce
    a centerline. Returns a result with ``status='ok'`` on success.
    """
    if points_xy.size == 0:
        return _empty_result("no cones available")

    if track_ids is None or len(track_ids) != points_xy.shape[0]:
        track_ids = np.arange(points_xy.shape[0], dtype=np.int64)
    else:
        track_ids = np.asarray(track_ids, dtype=np.int64)

    # --- Filter & order cone inputs ---
    cones = _filter_and_order_cones(
        points_xy, colors, confidences, track_ids,
        vehicle_xy, vehicle_yaw, config,
    )
    if cones.colored_count == 0:
        return _empty_result(
            "no colored cones in planning region",
            reject_counts=_default_reject_counts(),
        )
    if cones.colored_count < int(config.min_required_cones):
        return _empty_result(
            f"usable colored cones below minimum ({cones.colored_count} < {int(config.min_required_cones)})",
            filtered_points=cones.points,
            filtered_colors=cones.colors,
            reject_counts=_default_reject_counts(),
        )

    # --- Build boundary chains ---
    left_indices = np.flatnonzero(np.array([c == "blue" for c in cones.colors], dtype=bool))
    right_indices = np.flatnonzero(np.array([c == "yellow" for c in cones.colors], dtype=bool))
    unknown_indices = np.flatnonzero(np.array([c == "unknown" for c in cones.colors], dtype=bool))
    left_chain = _build_boundary_chain(cones.points, cones.local, left_indices, config)
    right_chain = _build_boundary_chain(cones.points, cones.local, right_indices, config)

    reject_counts = _default_reject_counts()
    expected_width = _clamp(
        prior.previous_width_m if prior and prior.previous_width_m is not None else config.initial_width_m,
        config.min_width_m,
        config.max_width_m,
    )

    # --- Attempt pairing (for width estimation) ---
    candidate_edges = np.empty((0, 2), dtype=np.int64)
    pairs: list[_BoundaryPair] = []
    midpoint_chain = np.empty((0, 2), dtype=np.float64)
    selected_edges = np.empty((0, 2), dtype=np.int64)
    planner_mode = "none"
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
            filtered_points=cones.points,
            filtered_local=cones.local,
            filtered_track_ids=cones.track_ids,
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

    # --- Select & offset fallback boundary chain ---
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
                filtered_points=cones.points,
                filtered_colors=cones.colors,
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

    # --- Compute path metrics ---
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
    heading_delta_max = _path_heading_delta_max(centerline_local)
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

    # --- Validate path ---
    reject_reason = _validate_path(
        centerline, centerline_local, near_field, heading_delta_max,
        continuity_threshold_m, reject_counts, config,
    )
    status = reject_reason if reject_reason else "ok"
    if status != "ok":
        centerline = np.empty((0, 2), dtype=np.float64)

    # --- Assemble result ---
    return SingleBoundaryPlannerResult(
        filtered_points=cones.points,
        filtered_colors=cones.colors,
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
        near_field_kink_max_rad=float(heading_delta_max),
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
    chain = build_boundary_chain_data(
        filtered_points=filtered_points,
        filtered_local=filtered_local,
        side_indices=side_indices,
        config=config,
    )
    return _BoundaryChain(
        filtered_indices=chain.filtered_indices,
        global_points=chain.global_points,
        local_points=chain.local_points,
        tangents_local=chain.tangents_local,
        mean_heading_change_rad=chain.mean_heading_change_rad,
        forward_extent_m=chain.forward_extent_m,
    )


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
            option, reject_key = _real_partner_option(
                anchor_local=anchor_local,
                anchor_tangent=anchor_tangent,
                inward_normal=inward_normal,
                other_chain=other_chain,
                other_pos=other_pos,
                filtered_track_ids=filtered_track_ids,
                expected_width_m=expected_width_m,
                last_partner_progress=last_partner_progress,
                config=config,
            )
            if option is None:
                if reject_key:
                    reject_counts[reject_key] += 1
                continue
            candidate_count += 1
            candidate_options.append(option)

        if config.allow_unknown_pair_completion and unknown_indices.size > 0:
            expected_partner_local = anchor_local + (inward_normal * float(expected_width_m))
            for filtered_idx in unknown_indices:
                unknown_idx = int(filtered_idx)
                if unknown_idx in used_unknown_indices:
                    continue
                option = _unknown_partner_option(
                    anchor_local=anchor_local,
                    anchor_tangent=anchor_tangent,
                    inward_normal=inward_normal,
                    filtered_points=filtered_points,
                    filtered_local=filtered_local,
                    filtered_track_ids=filtered_track_ids,
                    unknown_idx=unknown_idx,
                    expected_partner_local=expected_partner_local,
                    expected_width_m=expected_width_m,
                    last_partner_progress=last_partner_progress,
                    config=config,
                )
                if option is None:
                    continue
                candidate_count += 1
                candidate_options.append(option)

        if not candidate_options:
            continue
        candidate_options.sort(key=lambda option: option['sort_key'])
        chosen = candidate_options[0]
        chosen = prefer_previous_partner_option(
            options=candidate_options,
            current_option=chosen,
            preferred_partner_track_id=previous_partner_by_anchor.get(anchor_track_id),
            reassignment_margin=config.pair_reassignment_margin,
        )

        use_unknown = bool(chosen['use_unknown'])
        if use_unknown and consecutive_unknown_pairs >= int(config.max_consecutive_unknown_pairs):
            non_unknown = next((option for option in candidate_options if not bool(option['use_unknown'])), None)
            if non_unknown is None:
                continue
            chosen = non_unknown
            use_unknown = False

        candidate_width = float(chosen['width_m'])
        if width_jump_exceeds(last_width, candidate_width, config.max_width_jump_m):
            reject_counts["width"] += 1
            continue

        pairs.append(
            _boundary_pair_from_option(
                chosen=chosen,
                anchor_side=anchor_side,
                anchor_filtered_idx=anchor_filtered_idx,
                anchor_track_id=anchor_track_id,
                anchor_global=anchor_global,
                anchor_local=anchor_local,
                width_m=candidate_width,
            )
        )
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


def _real_partner_option(
    *,
    anchor_local: np.ndarray,
    anchor_tangent: np.ndarray,
    inward_normal: np.ndarray,
    other_chain: _BoundaryChain,
    other_pos: int,
    filtered_track_ids: np.ndarray,
    expected_width_m: float,
    last_partner_progress: float,
    config: SingleBoundaryPlannerConfig,
) -> tuple[Optional[dict[str, object]], str]:
    other_local = np.asarray(other_chain.local_points[other_pos], dtype=np.float64)
    if float(other_local[0]) < (last_partner_progress - float(config.min_forward_progress_m)):
        return None, "progress"

    delta = other_local - anchor_local
    width_m = float(np.hypot(delta[0], delta[1]))
    if not pair_width_in_range(width_m, config.min_pair_width_m, config.max_pair_width_m):
        return None, "width_range"

    if inward_distance(delta, inward_normal) <= 0.0:
        return None, "wrong_side"

    longitudinal_offset = abs(float(np.dot(delta, anchor_tangent)))
    width_error = abs(width_m - float(expected_width_m))
    partner_filtered_idx = int(other_chain.filtered_indices[other_pos])
    return (
        {
            "use_unknown": False,
            "other_pos": int(other_pos),
            "partner_filtered_idx": partner_filtered_idx,
            "partner_track_id": int(filtered_track_ids[partner_filtered_idx]),
            "partner_global": np.asarray(other_chain.global_points[other_pos], dtype=np.float64),
            "partner_local": other_local,
            "width_m": float(width_m),
            "cost": float(longitudinal_offset + width_error),
            "sort_key": (
                longitudinal_offset,
                width_error,
                width_m,
                int(other_pos),
            ),
        },
        "",
    )


def _unknown_partner_option(
    *,
    anchor_local: np.ndarray,
    anchor_tangent: np.ndarray,
    inward_normal: np.ndarray,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    unknown_idx: int,
    expected_partner_local: np.ndarray,
    expected_width_m: float,
    last_partner_progress: float,
    config: SingleBoundaryPlannerConfig,
) -> Optional[dict[str, object]]:
    unknown_local = np.asarray(filtered_local[unknown_idx], dtype=np.float64)
    if float(unknown_local[0]) < (last_partner_progress - float(config.min_forward_progress_m)):
        return None

    delta = unknown_local - anchor_local
    width_m = float(np.hypot(delta[0], delta[1]))
    if not pair_width_in_range(width_m, config.min_pair_width_m, config.max_pair_width_m):
        return None
    if inward_distance(delta, inward_normal) <= 0.0:
        return None

    check = unknown_partner_check(
        partner_local=unknown_local,
        expected_partner_local=expected_partner_local,
        anchor_tangent=anchor_tangent,
        width_m=width_m,
        expected_width_m=expected_width_m,
    )
    if not unknown_partner_within_limits(
        check,
        max_longitudinal_error_m=config.unknown_pair_max_longitudinal_error_m,
        max_width_error_m=config.unknown_pair_max_width_error_m,
        search_radius_m=config.unknown_pair_search_radius_m,
    ):
        return None

    return {
        "use_unknown": True,
        "other_pos": -1,
        "partner_filtered_idx": int(unknown_idx),
        "partner_track_id": int(filtered_track_ids[unknown_idx]),
        "partner_global": np.asarray(filtered_points[unknown_idx], dtype=np.float64),
        "partner_local": unknown_local,
        "width_m": float(width_m),
        "cost": float(check.cost + 0.05),
        "sort_key": (
            check.longitudinal_error_m,
            check.width_error_m,
            check.radial_error_m,
            int(unknown_idx),
        ),
    }


def _boundary_pair_from_option(
    *,
    chosen: dict[str, object],
    anchor_side: str,
    anchor_filtered_idx: int,
    anchor_track_id: int,
    anchor_global: np.ndarray,
    anchor_local: np.ndarray,
    width_m: float,
) -> _BoundaryPair:
    partner_filtered_idx = int(chosen["partner_filtered_idx"])
    partner_track_id = int(chosen["partner_track_id"])
    partner_global = np.asarray(chosen["partner_global"], dtype=np.float64)
    partner_local = np.asarray(chosen["partner_local"], dtype=np.float64)

    if anchor_side == "blue":
        return _BoundaryPair(
            left_filtered_idx=anchor_filtered_idx,
            right_filtered_idx=partner_filtered_idx,
            left_track_id=anchor_track_id,
            right_track_id=partner_track_id,
            left_global=np.asarray(anchor_global, dtype=np.float64),
            right_global=partner_global,
            left_local=np.asarray(anchor_local, dtype=np.float64),
            right_local=partner_local,
            width_m=float(width_m),
        )

    return _BoundaryPair(
        left_filtered_idx=partner_filtered_idx,
        right_filtered_idx=anchor_filtered_idx,
        left_track_id=partner_track_id,
        right_track_id=anchor_track_id,
        left_global=partner_global,
        right_global=np.asarray(anchor_global, dtype=np.float64),
        left_local=partner_local,
        right_local=np.asarray(anchor_local, dtype=np.float64),
        width_m=float(width_m),
    )


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
