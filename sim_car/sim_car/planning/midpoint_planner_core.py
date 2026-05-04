"""Core geometry for the midpoint boundary planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.planner_config_base import BasePlannerConfig
from sim_car.planning.tracked_cone_planner_geometry import (
    build_boundary_chain_data,
    estimate_tangents as _estimate_tangents,
    inward_distance,
    inward_normal as _inward_normal,
    midpoint_outside_pair_span,
    moving_average as _moving_average,
    pair_width_in_range,
    path_cumulative_lengths as _path_cumulative_lengths,
    path_heading_delta_max as _path_heading_delta_max,
    path_self_intersects as _path_self_intersects,
    to_vehicle_frame as _to_vehicle_frame,
    unknown_partner_check,
    unknown_partner_within_limits,
    update_track_width_estimate,
)

_MIDPOINT_CHAIN_BACKTRACK_TOLERANCE_M = 0.25


@dataclass
class MidpointPlannerConfig(BasePlannerConfig):
    allow_unknown_pair_completion: bool = True
    unknown_pair_search_radius_m: float = 1.25
    unknown_pair_max_longitudinal_error_m: float = 1.5
    unknown_pair_max_width_error_m: float = 0.9
    max_consecutive_unknown_pairs: int = 2

    max_step_m: float = 10.0
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

    smoothing_window: int = 3
    max_heading_delta_rad: float = 0.75
    max_midpoint_segment_length_m: float = 7.5
    midpoint_order_reference_handoff_m: float = 6.0
    midpoint_order_history_size: int = 3
    midpoint_order_backtrack_tolerance_m: float = 0.35

    min_path_points: int = 4
    min_forward_extent_m: float = 2.0
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


@dataclass
class _FilteredCones:
    points: np.ndarray
    local: np.ndarray
    track_ids: np.ndarray
    colors: list[str]
    raw_colors: list[str]
    colored_count: int


@dataclass
class _BoundaryIndices:
    left: np.ndarray
    right: np.ndarray
    unknown: np.ndarray


@dataclass
class _PairingIndices:
    left: np.ndarray
    right: np.ndarray
    use_legacy_side_gate: bool


@dataclass
class _PairAnchor:
    filtered_idx: int
    track_id: int
    global_point: np.ndarray
    local_point: np.ndarray
    tangent: np.ndarray
    raw_color: str
    inward_normal: np.ndarray


@dataclass
class _PairCandidate:
    use_unknown: bool
    anchor_filtered_idx: int
    anchor_track_id: int
    anchor_global: np.ndarray
    anchor_local: np.ndarray
    partner_filtered_idx: int
    partner_track_id: int
    partner_global: np.ndarray
    partner_local: np.ndarray
    width_m: float
    cost: float
    selection_cost: float
    sort_key: tuple[float | int, ...]


@dataclass
class _PairingOutcome:
    pairs: list[_BoundaryPair]
    candidate_count: int
    unknown_pair_count: int
    reject_counts: dict[str, int]


@dataclass
class _MidpointPairingResult:
    candidate_edges: np.ndarray
    pairs: list[_BoundaryPair]
    midpoint_chain: np.ndarray
    selected_edges: np.ndarray
    selected_pair_track_ids: np.ndarray
    planner_mode: str
    candidate_count: int
    measured_width_m: float
    unknown_pair_count: int


@dataclass
class _NearFieldMetrics:
    lateral_max_m: float = 0.0
    lateral_mean_m: float = 0.0
    displacement_max_m: float = 0.0
    displacement_mean_m: float = 0.0


@dataclass
class _ValidatedMidpointPath:
    centerline: np.ndarray
    prevalidation_centerline: np.ndarray
    near_field: _NearFieldMetrics
    heading_delta_max: float
    seed_distance_m: float
    reject_reason: str
    status: str


@dataclass
class _MidpointComputationContext:
    cones: _FilteredCones
    left_chain: _BoundaryChain
    right_chain: _BoundaryChain
    pairing: _MidpointPairingResult
    expected_width_m: float
    reject_counts: dict[str, int]
    left_boundary_count: int
    right_boundary_count: int


def _filter_and_order_cones(
    points_xy: np.ndarray,
    colors: list[str],
    confidences: np.ndarray,
    track_ids: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: MidpointPlannerConfig,
    raw_colors: Optional[list[str]] = None,
) -> "_FilteredCones":
    normalized = [normalize_color(c) for c in colors]
    normalized_raw = _normalized_raw_colors(raw_colors, colors, normalized)
    local_points = _to_vehicle_frame(points_xy, vehicle_xy, vehicle_yaw)
    selected_mask = _cone_selection_mask(local_points, confidences, normalized, config)

    indices = np.where(selected_mask)[0]
    filtered_points = points_xy[selected_mask]
    filtered_local = local_points[selected_mask]
    filtered_track_ids = track_ids[selected_mask]
    filtered_colors = [normalized[i] for i in indices]
    filtered_raw_colors = [normalized_raw[i] for i in indices]
    colored_count = int(np.count_nonzero(
        np.array([c in {"blue", "yellow"} for c in filtered_colors], dtype=bool)
    ))

    order = _deterministic_order(filtered_local, filtered_points, filtered_colors)
    return _FilteredCones(
        points=filtered_points[order],
        local=filtered_local[order],
        track_ids=filtered_track_ids[order],
        colors=[filtered_colors[i] for i in order],
        raw_colors=[filtered_raw_colors[i] for i in order],
        colored_count=colored_count,
    )


def _normalized_raw_colors(
    raw_colors: Optional[list[str]],
    colors: list[str],
    normalized_colors: list[str],
) -> list[str]:
    if raw_colors is not None and len(raw_colors) == len(colors):
        return [normalize_color(c) for c in raw_colors]
    return list(normalized_colors)


def _cone_selection_mask(
    local_points: np.ndarray,
    confidences: np.ndarray,
    normalized_colors: list[str],
    config: MidpointPlannerConfig,
) -> np.ndarray:
    mask_geom = _geometry_filter(local_points, config)
    mask_conf = confidences >= float(config.min_confidence)
    colored_mask = np.array([c in {"blue", "yellow"} for c in normalized_colors], dtype=bool)
    unknown_mask = np.array([c == "unknown" for c in normalized_colors], dtype=bool)
    return mask_geom & mask_conf & (
        colored_mask | (unknown_mask if config.allow_unknown_pair_completion else False)
    )


def _validate_path(
    centerline: np.ndarray,
    centerline_local: np.ndarray,
    near_field: _NearFieldMetrics,
    heading_delta_max: float,
    continuity_threshold_m: float,
    reject_counts: dict[str, int],
    config: MidpointPlannerConfig,
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
    if near_field.lateral_max_m > continuity_threshold_m:
        reject_counts["near_field_continuity"] += 1
        return "near-field continuity rejected fresh path"
    if heading_delta_max > float(config.max_heading_delta_rad):
        reject_counts["midpoint_kink"] += 1
        return "path heading delta exceeded limit"
    if _path_self_intersects(centerline):
        return "path self-crossing detected"
    return ""


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
    """Compute a local centerline from midpoint pairs."""
    if points_xy.size == 0:
        return _empty_result("no cones available")

    track_ids = _normalized_track_ids(track_ids, points_xy.shape[0])
    cones = _filter_and_order_cones(
        points_xy, colors, confidences, track_ids,
        vehicle_xy, vehicle_yaw, config, raw_colors,
    )
    early_result = _filtered_cones_failure_result(cones, config)
    if early_result is not None:
        return early_result
    return _compute_midpoint_from_cones(cones, vehicle_xy, vehicle_yaw, config, prior)


def _compute_midpoint_from_cones(
    cones: _FilteredCones,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: MidpointPlannerConfig,
    prior: Optional[MidpointPlannerPrior],
) -> MidpointPlannerResult:
    context = _midpoint_computation_context(cones, config, prior)
    if context.pairing.planner_mode == "none":
        return _no_reliable_midpoint_result(context)
    validated = _validate_midpoint_centerline(
        context.pairing, vehicle_xy, vehicle_yaw, config,
        prior, context.reject_counts,
    )
    return _midpoint_result(context, validated)


def _midpoint_computation_context(
    cones: _FilteredCones,
    config: MidpointPlannerConfig,
    prior: Optional[MidpointPlannerPrior],
) -> _MidpointComputationContext:
    boundary_indices = _boundary_indices_from_colors(cones.colors)
    left_chain, right_chain = _build_midpoint_boundary_chains(cones, boundary_indices, config)
    reject_counts = _default_reject_counts()
    expected_width = _expected_width_m(prior, config)

    pairing = _compute_midpoint_pairing(
        cones=cones,
        boundary_indices=boundary_indices,
        expected_width_m=expected_width,
        config=config,
        prior=prior,
        left_chain=left_chain,
        right_chain=right_chain,
        reject_counts=reject_counts,
    )
    return _MidpointComputationContext(
        cones=cones,
        left_chain=left_chain,
        right_chain=right_chain,
        pairing=pairing,
        expected_width_m=expected_width,
        reject_counts=reject_counts,
        left_boundary_count=int(boundary_indices.left.size),
        right_boundary_count=int(boundary_indices.right.size),
    )


def _normalized_track_ids(
    track_ids: Optional[np.ndarray],
    point_count: int,
) -> np.ndarray:
    if track_ids is None or len(track_ids) != point_count:
        return np.arange(point_count, dtype=np.int64)
    return np.asarray(track_ids, dtype=np.int64)


def _filtered_cones_failure_result(
    cones: _FilteredCones,
    config: MidpointPlannerConfig,
) -> Optional[MidpointPlannerResult]:
    if cones.colored_count == 0:
        return _empty_result(
            "no colored cones in planning region",
            reject_counts=_default_reject_counts(),
        )
    minimum = int(config.min_required_cones)
    if cones.colored_count >= minimum:
        return None
    return _empty_result(
        f"usable colored cones below minimum ({cones.colored_count} < {minimum})",
        filtered_points=cones.points,
        filtered_colors=cones.colors,
        reject_counts=_default_reject_counts(),
    )


def _boundary_indices_from_colors(colors: list[str]) -> _BoundaryIndices:
    color_array = np.asarray(colors, dtype=object)
    return _BoundaryIndices(
        left=np.flatnonzero(color_array == "blue"),
        right=np.flatnonzero(color_array == "yellow"),
        unknown=np.flatnonzero(color_array == "unknown"),
    )


def _build_midpoint_boundary_chains(
    cones: _FilteredCones,
    boundary_indices: _BoundaryIndices,
    config: MidpointPlannerConfig,
) -> tuple[_BoundaryChain, _BoundaryChain]:
    return (
        _build_boundary_chain(cones.points, cones.local, boundary_indices.left, config),
        _build_boundary_chain(cones.points, cones.local, boundary_indices.right, config),
    )


def _expected_width_m(
    prior: Optional[MidpointPlannerPrior],
    config: MidpointPlannerConfig,
) -> float:
    prior_width = (
        prior.previous_width_m
        if prior is not None and prior.previous_width_m is not None
        else config.initial_width_m
    )
    return _clamp(prior_width, config.min_width_m, config.max_width_m)


def _compute_midpoint_pairing(
    *,
    cones: _FilteredCones,
    boundary_indices: _BoundaryIndices,
    expected_width_m: float,
    config: MidpointPlannerConfig,
    prior: Optional[MidpointPlannerPrior],
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
    reject_counts: dict[str, int],
) -> _MidpointPairingResult:
    pairing = _build_boundary_pairing(
        cones, boundary_indices, expected_width_m,
        config, prior, left_chain, right_chain,
    )
    _merge_reject_counts(reject_counts, pairing.reject_counts)
    ordered_pairs = _ordered_midpoint_pairs(pairing.pairs, config)
    return _midpoint_pairing_result(
        ordered_pairs, pairing.candidate_count,
        pairing.unknown_pair_count, config,
    )


def _build_boundary_pairing(
    cones: _FilteredCones,
    boundary_indices: _BoundaryIndices,
    expected_width_m: float,
    config: MidpointPlannerConfig,
    prior: Optional[MidpointPlannerPrior],
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
) -> _PairingOutcome:
    if not _has_pairing_inputs(boundary_indices):
        return _PairingOutcome([], 0, 0, _default_reject_counts())
    pairs, candidate_count, unknown_pair_count, reject_counts = _pair_boundary_chains(
        filtered_points=cones.points,
        filtered_local=cones.local,
        filtered_track_ids=cones.track_ids,
        filtered_raw_colors=cones.raw_colors,
        left_indices=boundary_indices.left,
        right_indices=boundary_indices.right,
        unknown_indices=boundary_indices.unknown,
        expected_width_m=expected_width_m,
        config=config,
        prior=prior,
        left_chain=left_chain,
        right_chain=right_chain,
    )
    return _PairingOutcome(pairs, candidate_count, unknown_pair_count, reject_counts)


def _has_pairing_inputs(boundary_indices: _BoundaryIndices) -> bool:
    return boundary_indices.left.size > 0 and (
        boundary_indices.right.size > 0 or boundary_indices.unknown.size > 0
    )


def _ordered_midpoint_pairs(
    pairs: list[_BoundaryPair],
    config: MidpointPlannerConfig,
) -> list[_BoundaryPair]:
    ordered = _order_pairs_into_midpoint_chain(pairs, config=config)
    return _trim_pairs_by_midpoint_step_length(
        ordered,
        max_segment_length_m=float(config.max_midpoint_segment_length_m),
    )


def _midpoint_pairing_result(
    pairs: list[_BoundaryPair],
    candidate_count: int,
    unknown_pair_count: int,
    config: MidpointPlannerConfig,
) -> _MidpointPairingResult:
    measured_width_m = _median_pair_width_m(pairs)
    if len(pairs) < int(config.min_pair_count):
        return _empty_midpoint_pairing(candidate_count, measured_width_m)
    return _MidpointPairingResult(
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        pairs=pairs,
        midpoint_chain=np.vstack([pair.midpoint_global for pair in pairs]).astype(np.float64),
        selected_edges=_selected_pair_edges(pairs),
        selected_pair_track_ids=_selected_pair_track_ids(pairs),
        planner_mode="midpoint",
        candidate_count=int(candidate_count),
        measured_width_m=float(measured_width_m),
        unknown_pair_count=int(unknown_pair_count),
    )


def _median_pair_width_m(pairs: list[_BoundaryPair]) -> float:
    if not pairs:
        return float("nan")
    return float(np.median([pair.width_m for pair in pairs]))


def _empty_midpoint_pairing(
    candidate_count: int,
    measured_width_m: float,
) -> _MidpointPairingResult:
    return _MidpointPairingResult(
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        pairs=[],
        midpoint_chain=np.empty((0, 2), dtype=np.float64),
        selected_edges=np.empty((0, 2), dtype=np.int64),
        selected_pair_track_ids=np.empty((0, 2), dtype=np.int64),
        planner_mode="none",
        candidate_count=int(candidate_count),
        measured_width_m=float(measured_width_m),
        unknown_pair_count=0,
    )


def _selected_pair_edges(pairs: list[_BoundaryPair]) -> np.ndarray:
    return np.asarray(
        [[pair.left_filtered_idx, pair.right_filtered_idx] for pair in pairs],
        dtype=np.int64,
    )


def _selected_pair_track_ids(pairs: list[_BoundaryPair]) -> np.ndarray:
    return np.asarray(
        [[pair.left_track_id, pair.right_track_id] for pair in pairs],
        dtype=np.int64,
    )


def _no_reliable_midpoint_result(context: _MidpointComputationContext) -> MidpointPlannerResult:
    return _result_with_metadata(
        result=_empty_result(
            "no reliable midpoint chain",
            filtered_points=context.cones.points,
            filtered_colors=context.cones.colors,
            left_boundary=context.left_chain.global_points,
            right_boundary=context.right_chain.global_points,
            reject_counts=context.reject_counts,
            reject_reason="no reliable midpoint chain",
        ),
        left_chain=context.left_chain,
        right_chain=context.right_chain,
        planner_mode="none",
        filtered_track_width_m=context.expected_width_m,
        left_boundary_count=context.left_boundary_count,
        right_boundary_count=context.right_boundary_count,
    )


def _validate_midpoint_centerline(
    pairing: _MidpointPairingResult,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: MidpointPlannerConfig,
    prior: Optional[MidpointPlannerPrior],
    reject_counts: dict[str, int],
) -> _ValidatedMidpointPath:
    centerline = _finalize_path(pairing.midpoint_chain, config)
    centerline_local = _to_vehicle_frame(centerline, vehicle_xy, vehicle_yaw)
    near_field = _near_field_delta_metrics(
        current=centerline,
        previous=None if prior is None else prior.previous_centerline,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        horizon_m=config.jump_check_horizon_m,
    )
    heading_delta_max = _path_heading_delta_max(centerline_local)
    continuity_threshold_m = _continuity_threshold_m(pairing.pairs, config)
    prevalidation_centerline = np.array(centerline, copy=True)
    reject_reason = _validate_path(
        centerline, centerline_local, near_field, heading_delta_max,
        continuity_threshold_m, reject_counts, config,
    )
    status = reject_reason if reject_reason else "ok"
    if status != "ok":
        centerline = np.empty((0, 2), dtype=np.float64)
    return _ValidatedMidpointPath(
        centerline=centerline,
        prevalidation_centerline=prevalidation_centerline,
        near_field=near_field,
        heading_delta_max=float(heading_delta_max),
        seed_distance_m=_first_point_distance(centerline_local),
        reject_reason=reject_reason,
        status=status,
    )


def _continuity_threshold_m(
    pairs: list[_BoundaryPair],
    config: MidpointPlannerConfig,
) -> float:
    threshold_m = float(config.max_near_field_lateral_jump_m)
    if len(pairs) <= max(3, int(config.min_pair_count)):
        return max(threshold_m, float(config.max_near_field_lateral_jump_m_sparse_pairs))
    return threshold_m


def _midpoint_result(
    context: _MidpointComputationContext,
    validated: _ValidatedMidpointPath,
) -> MidpointPlannerResult:
    result = _base_midpoint_result(context, validated)
    _apply_midpoint_result_metrics(result, context, validated)
    return result


def _base_midpoint_result(
    context: _MidpointComputationContext,
    validated: _ValidatedMidpointPath,
) -> MidpointPlannerResult:
    return MidpointPlannerResult(
        filtered_points=context.cones.points,
        filtered_colors=context.cones.colors,
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=context.pairing.candidate_edges,
        selected_edges=context.pairing.selected_edges,
        selected_pair_track_ids=context.pairing.selected_pair_track_ids,
        midpoints_raw=context.pairing.midpoint_chain,
        centerline=validated.centerline,
        prevalidation_centerline=validated.prevalidation_centerline,
        left_boundary=context.left_chain.global_points,
        right_boundary=context.right_chain.global_points,
        used_fallback=False,
        status=validated.status,
        reject_reason=validated.reject_reason,
        reject_counts=context.reject_counts,
    )


def _apply_midpoint_result_metrics(
    result: MidpointPlannerResult,
    context: _MidpointComputationContext,
    validated: _ValidatedMidpointPath,
) -> None:
    result.candidate_count = int(context.pairing.candidate_count)
    result.selected_chain_length = int(len(context.pairing.pairs))
    result.selected_chain_width_median = float(context.pairing.measured_width_m)
    result.expected_width_prior_m = float(context.expected_width_m)
    result.near_field_lateral_max_m = float(validated.near_field.lateral_max_m)
    result.near_field_lateral_mean_m = float(validated.near_field.lateral_mean_m)
    result.near_field_displacement_max_m = float(validated.near_field.displacement_max_m)
    result.near_field_displacement_mean_m = float(validated.near_field.displacement_mean_m)
    result.near_field_kink_max_rad = float(validated.heading_delta_max)
    result.seed_midpoint_distance_m = float(validated.seed_distance_m)
    result.seed_temporal_offset_m = float("nan")
    result.planner_mode = context.pairing.planner_mode
    result.active_boundary_side = ""
    result.raw_offset_path = np.empty((0, 2), dtype=np.float64)
    result.pair_segments = _pair_segments(context.pairing.pairs)
    result.accepted_pair_count = len(context.pairing.pairs)
    result.left_chain_length = context.left_boundary_count
    result.right_chain_length = context.right_boundary_count
    result.filtered_track_width_m = float(context.expected_width_m)
    result.unknown_pair_count = int(context.pairing.unknown_pair_count)


def _pair_segments(pairs: list[_BoundaryPair]) -> np.ndarray:
    if not pairs:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.asarray(
        [[pair.left_global, pair.right_global] for pair in pairs],
        dtype=np.float64,
    )


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
    pairing_indices = _resolve_pairing_indices(left_indices, right_indices, left_chain, right_chain)
    if _pairing_indices_empty(pairing_indices, unknown_indices):
        return [], 0, 0, reject_counts

    left_tangents = _left_pairing_tangents(
        filtered_local=filtered_local,
        pairing_indices=pairing_indices,
        left_chain=left_chain,
        config=config,
    )
    candidate_options = _pair_candidates_for_all_anchors(
        filtered_points=filtered_points,
        filtered_local=filtered_local,
        filtered_track_ids=filtered_track_ids,
        filtered_raw_colors=filtered_raw_colors,
        pairing_indices=pairing_indices,
        unknown_indices=unknown_indices,
        left_tangents=left_tangents,
        expected_width_m=expected_width_m,
        config=config,
        reject_counts=reject_counts,
    )
    pairs, unknown_pair_count = _select_boundary_pairs(candidate_options)
    return pairs, len(candidate_options), unknown_pair_count, reject_counts


def _resolve_pairing_indices(
    left_indices: Optional[np.ndarray],
    right_indices: Optional[np.ndarray],
    left_chain: Optional[_BoundaryChain],
    right_chain: Optional[_BoundaryChain],
) -> _PairingIndices:
    return _PairingIndices(
        left=_chain_or_explicit_indices(left_indices, left_chain),
        right=_chain_or_explicit_indices(right_indices, right_chain),
        use_legacy_side_gate=left_indices is None and left_chain is not None,
    )


def _chain_or_explicit_indices(
    indices: Optional[np.ndarray],
    chain: Optional[_BoundaryChain],
) -> np.ndarray:
    if indices is not None:
        return np.asarray(indices, dtype=np.int64)
    if chain is None:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(chain.filtered_indices, dtype=np.int64)


def _pairing_indices_empty(
    pairing_indices: _PairingIndices,
    unknown_indices: np.ndarray,
) -> bool:
    return pairing_indices.left.size == 0 or (
        pairing_indices.right.size == 0 and unknown_indices.size == 0
    )


def _left_pairing_tangents(
    *,
    filtered_local: np.ndarray,
    pairing_indices: _PairingIndices,
    left_chain: Optional[_BoundaryChain],
    config: MidpointPlannerConfig,
) -> np.ndarray:
    if _can_use_chain_tangents(pairing_indices, left_chain):
        return np.asarray(left_chain.tangents_local, dtype=np.float64)
    return _estimate_side_tangents(
        side_local=filtered_local[pairing_indices.left],
        side="blue",
        neighbor_count=max(2, int(config.pairing_tangent_neighbor_count)),
    )


def _can_use_chain_tangents(
    pairing_indices: _PairingIndices,
    left_chain: Optional[_BoundaryChain],
) -> bool:
    return (
        pairing_indices.use_legacy_side_gate
        and left_chain is not None
        and left_chain.tangents_local.shape[0] == pairing_indices.left.size
    )


def _pair_candidates_for_all_anchors(
    *,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    filtered_raw_colors: Optional[list[str]],
    pairing_indices: _PairingIndices,
    unknown_indices: np.ndarray,
    left_tangents: np.ndarray,
    expected_width_m: float,
    config: MidpointPlannerConfig,
    reject_counts: dict[str, int],
) -> list[_PairCandidate]:
    candidate_options: list[_PairCandidate] = []
    for left_pos, anchor_filtered_idx in enumerate(pairing_indices.left):
        anchor = _pair_anchor(
            filtered_idx=int(anchor_filtered_idx),
            tangent=left_tangents[left_pos],
            filtered_points=filtered_points,
            filtered_local=filtered_local,
            filtered_track_ids=filtered_track_ids,
            filtered_raw_colors=filtered_raw_colors,
        )
        anchor_options = _pair_candidates_for_anchor(
            anchor=anchor,
            filtered_points=filtered_points,
            filtered_local=filtered_local,
            filtered_track_ids=filtered_track_ids,
            filtered_raw_colors=filtered_raw_colors,
            pairing_indices=pairing_indices,
            unknown_indices=unknown_indices,
            expected_width_m=expected_width_m,
            config=config,
            reject_counts=reject_counts,
        )
        anchor_options.sort(key=lambda option: option.sort_key)
        candidate_options.extend(anchor_options)
    candidate_options.sort(key=_pair_candidate_selection_key)
    return candidate_options


def _pair_anchor(
    *,
    filtered_idx: int,
    tangent: np.ndarray,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    filtered_raw_colors: Optional[list[str]],
) -> _PairAnchor:
    tangent = np.asarray(tangent, dtype=np.float64)
    return _PairAnchor(
        filtered_idx=filtered_idx,
        track_id=int(filtered_track_ids[filtered_idx]),
        global_point=np.asarray(filtered_points[filtered_idx], dtype=np.float64),
        local_point=np.asarray(filtered_local[filtered_idx], dtype=np.float64),
        tangent=tangent,
        raw_color=_raw_color_at(filtered_raw_colors, filtered_idx),
        inward_normal=_inward_normal(tangent, "blue"),
    )


def _raw_color_at(raw_colors: Optional[list[str]], filtered_idx: int) -> str:
    if raw_colors is not None and filtered_idx < len(raw_colors):
        return normalize_color(raw_colors[filtered_idx])
    return "unknown"


def _pair_candidates_for_anchor(
    *,
    anchor: _PairAnchor,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    filtered_raw_colors: Optional[list[str]],
    pairing_indices: _PairingIndices,
    unknown_indices: np.ndarray,
    expected_width_m: float,
    config: MidpointPlannerConfig,
    reject_counts: dict[str, int],
) -> list[_PairCandidate]:
    options = _real_pair_candidates_for_anchor(
        anchor, pairing_indices, filtered_points, filtered_local,
        filtered_track_ids, filtered_raw_colors, expected_width_m,
        config, reject_counts,
    )
    if config.allow_unknown_pair_completion and unknown_indices.size > 0:
        options.extend(
            _unknown_pair_candidates_for_anchor(
                anchor, unknown_indices, filtered_points, filtered_local,
                filtered_track_ids, expected_width_m, config,
                pairing_indices.use_legacy_side_gate, reject_counts,
            )
        )
    return options


def _real_pair_candidates_for_anchor(
    anchor: _PairAnchor,
    pairing_indices: _PairingIndices,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    filtered_raw_colors: Optional[list[str]],
    expected_width_m: float,
    config: MidpointPlannerConfig,
    reject_counts: dict[str, int],
) -> list[_PairCandidate]:
    options: list[_PairCandidate] = []
    for partner_filtered_idx in pairing_indices.right:
        candidate, reject_key = _real_pair_candidate(
            anchor=anchor,
            partner_filtered_idx=int(partner_filtered_idx),
            filtered_points=filtered_points,
            filtered_local=filtered_local,
            filtered_track_ids=filtered_track_ids,
            filtered_raw_colors=filtered_raw_colors,
            expected_width_m=expected_width_m,
            config=config,
            use_legacy_side_gate=pairing_indices.use_legacy_side_gate,
        )
        if candidate is None:
            if reject_key:
                reject_counts[reject_key] += 1
            continue
        options.append(candidate)
    return options


def _real_pair_candidate(
    *,
    anchor: _PairAnchor,
    partner_filtered_idx: int,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    filtered_raw_colors: Optional[list[str]],
    expected_width_m: float,
    config: MidpointPlannerConfig,
    use_legacy_side_gate: bool,
) -> tuple[Optional[_PairCandidate], str]:
    other_local = np.asarray(filtered_local[partner_filtered_idx], dtype=np.float64)
    partner_raw_color = _raw_color_at(filtered_raw_colors, partner_filtered_idx)
    if not _raw_pair_allowed(anchor.raw_color, partner_raw_color, config):
        return None, "color"

    delta = other_local - anchor.local_point
    width_m = float(np.hypot(delta[0], delta[1]))
    if not pair_width_in_range(width_m, config.min_pair_width_m, config.max_pair_width_m):
        return None, "width_range"

    inward_distance_m = inward_distance(delta, anchor.inward_normal)
    if _pair_geometry_rejected(
        anchor.local_point, other_local, inward_distance_m,
        width_m, config, use_legacy_side_gate,
    ):
        return None, "wrong_side"
    return _real_pair_candidate_from_geometry(
        anchor, partner_filtered_idx, filtered_points, filtered_track_ids,
        other_local, width_m, inward_distance_m, expected_width_m,
    ), ""


def _raw_pair_allowed(
    anchor_raw_color: str,
    partner_raw_color: str,
    config: MidpointPlannerConfig,
) -> bool:
    if not config.enforce_opposite_color_pairing:
        return True
    return {anchor_raw_color, partner_raw_color} in (
        {"blue", "yellow"},
        {"orange", "blue"},
        {"orange", "yellow"},
        {"orange"},
    )


def _pair_geometry_rejected(
    anchor_local: np.ndarray,
    partner_local: np.ndarray,
    inward_distance_m: float,
    width_m: float,
    config: MidpointPlannerConfig,
    use_legacy_side_gate: bool,
) -> bool:
    if not config.enforce_geometry_pairing_gate:
        return False
    tolerance_m = max(0.0, float(config.pair_inward_projection_tolerance_m))
    midpoint_local = 0.5 * (anchor_local + partner_local)
    if use_legacy_side_gate:
        wrong_side = inward_distance_m <= -tolerance_m
    else:
        lateral_progress = float(anchor_local[1] - partner_local[1])
        wrong_side = inward_distance_m <= -tolerance_m and lateral_progress <= tolerance_m
    return wrong_side or midpoint_outside_pair_span(midpoint_local, width_m)


def _real_pair_candidate_from_geometry(
    anchor: _PairAnchor,
    partner_filtered_idx: int,
    filtered_points: np.ndarray,
    filtered_track_ids: np.ndarray,
    other_local: np.ndarray,
    width_m: float,
    inward_distance_m: float,
    expected_width_m: float,
) -> _PairCandidate:
    return _PairCandidate(
        use_unknown=False,
        anchor_filtered_idx=anchor.filtered_idx,
        anchor_track_id=anchor.track_id,
        anchor_global=anchor.global_point,
        anchor_local=anchor.local_point,
        partner_filtered_idx=partner_filtered_idx,
        partner_track_id=int(filtered_track_ids[partner_filtered_idx]),
        partner_global=np.asarray(filtered_points[partner_filtered_idx], dtype=np.float64),
        partner_local=other_local,
        width_m=float(width_m),
        cost=float(width_m),
        selection_cost=float(width_m),
        sort_key=_real_pair_sort_key(
            anchor, other_local, width_m, inward_distance_m,
            expected_width_m, partner_filtered_idx,
        ),
    )


def _real_pair_sort_key(
    anchor: _PairAnchor,
    other_local: np.ndarray,
    width_m: float,
    inward_distance_m: float,
    expected_width_m: float,
    partner_filtered_idx: int,
) -> tuple[float | int, ...]:
    midpoint_local = 0.5 * (anchor.local_point + other_local)
    longitudinal_offset = abs(float(other_local[0] - anchor.local_point[0]))
    progress_offset = abs(
        float(np.hypot(other_local[0], other_local[1]))
        - float(np.hypot(anchor.local_point[0], anchor.local_point[1]))
    )
    width_error = abs(width_m - float(expected_width_m))
    inward_penalty = max(0.0, -inward_distance_m)
    return (
        width_m,
        width_error,
        longitudinal_offset,
        progress_offset,
        inward_penalty,
        abs(float(midpoint_local[1])),
        partner_filtered_idx,
    )


def _unknown_pair_candidates_for_anchor(
    anchor: _PairAnchor,
    unknown_indices: np.ndarray,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    expected_width_m: float,
    config: MidpointPlannerConfig,
    use_legacy_side_gate: bool,
    reject_counts: dict[str, int],
) -> list[_PairCandidate]:
    options: list[_PairCandidate] = []
    expected_partner_local = anchor.local_point + (anchor.inward_normal * float(expected_width_m))
    for filtered_idx in unknown_indices:
        candidate, reject_key = _unknown_pair_candidate(
            anchor, int(filtered_idx), filtered_points, filtered_local,
            filtered_track_ids, expected_partner_local, expected_width_m,
            config, use_legacy_side_gate,
        )
        if candidate is None:
            if reject_key:
                reject_counts[reject_key] += 1
            continue
        options.append(candidate)
    return options


def _unknown_pair_candidate(
    anchor: _PairAnchor,
    unknown_idx: int,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    expected_partner_local: np.ndarray,
    expected_width_m: float,
    config: MidpointPlannerConfig,
    use_legacy_side_gate: bool,
) -> tuple[Optional[_PairCandidate], str]:
    if config.enforce_opposite_color_pairing:
        return None, "color"
    unknown_local = np.asarray(filtered_local[unknown_idx], dtype=np.float64)
    delta = unknown_local - anchor.local_point
    width_m = float(np.hypot(delta[0], delta[1]))
    if not pair_width_in_range(width_m, config.min_pair_width_m, config.max_pair_width_m):
        return None, ""
    inward_distance_m = inward_distance(delta, anchor.inward_normal)
    if _pair_geometry_rejected(
        anchor.local_point, unknown_local, inward_distance_m,
        width_m, config, use_legacy_side_gate,
    ):
        return None, "wrong_side"
    return _unknown_pair_candidate_from_geometry(
        anchor, unknown_idx, filtered_points, filtered_local, filtered_track_ids,
        unknown_local, width_m, inward_distance_m, expected_partner_local,
        expected_width_m, config,
    )


def _unknown_pair_candidate_from_geometry(
    anchor: _PairAnchor,
    unknown_idx: int,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    unknown_local: np.ndarray,
    width_m: float,
    inward_distance_m: float,
    expected_partner_local: np.ndarray,
    expected_width_m: float,
    config: MidpointPlannerConfig,
) -> tuple[Optional[_PairCandidate], str]:
    unknown_check = _accepted_unknown_partner_check(
        anchor, unknown_local, width_m, expected_partner_local,
        expected_width_m, config,
    )
    if unknown_check is None:
        return None, ""
    inward_penalty = max(0.0, -inward_distance_m)
    cost = float(unknown_check.cost + inward_penalty + 0.05)
    return _PairCandidate(
        use_unknown=True,
        anchor_filtered_idx=anchor.filtered_idx,
        anchor_track_id=anchor.track_id,
        anchor_global=anchor.global_point,
        anchor_local=anchor.local_point,
        partner_filtered_idx=unknown_idx,
        partner_track_id=int(filtered_track_ids[unknown_idx]),
        partner_global=np.asarray(filtered_points[unknown_idx], dtype=np.float64),
        partner_local=np.asarray(filtered_local[unknown_idx], dtype=np.float64),
        width_m=float(width_m),
        cost=cost,
        selection_cost=cost,
        sort_key=_unknown_pair_sort_key(unknown_check, inward_penalty, unknown_idx),
    ), ""


def _accepted_unknown_partner_check(
    anchor: _PairAnchor,
    unknown_local: np.ndarray,
    width_m: float,
    expected_partner_local: np.ndarray,
    expected_width_m: float,
    config: MidpointPlannerConfig,
):
    unknown_check = unknown_partner_check(
        partner_local=unknown_local,
        expected_partner_local=expected_partner_local,
        anchor_tangent=anchor.tangent,
        width_m=width_m,
        expected_width_m=expected_width_m,
    )
    if unknown_partner_within_limits(
        unknown_check,
        max_longitudinal_error_m=config.unknown_pair_max_longitudinal_error_m,
        max_width_error_m=config.unknown_pair_max_width_error_m,
        search_radius_m=config.unknown_pair_search_radius_m,
    ):
        return unknown_check
    return None


def _unknown_pair_sort_key(
    unknown_check,
    inward_penalty: float,
    unknown_idx: int,
) -> tuple[float | int, ...]:
    return (
        inward_penalty,
        unknown_check.longitudinal_error_m,
        unknown_check.width_error_m,
        unknown_check.radial_error_m,
        unknown_idx,
    )


def _pair_candidate_selection_key(
    option: _PairCandidate,
) -> tuple[float, int, float, float, float, float, int, int]:
    return (
        float(option.selection_cost),
        1 if option.use_unknown else 0,
        float(option.width_m),
        float(np.hypot(option.anchor_local[0], option.anchor_local[1])),
        float(option.anchor_local[0]),
        abs(float(option.anchor_local[1])),
        int(option.anchor_track_id),
        int(option.partner_track_id),
    )


def _select_boundary_pairs(
    candidate_options: list[_PairCandidate],
) -> tuple[list[_BoundaryPair], int]:
    pairs: list[_BoundaryPair] = []
    used_left_indices: set[int] = set()
    used_partner_indices: set[int] = set()
    used_unknown_indices: set[int] = set()
    unknown_pair_count = 0
    for chosen in candidate_options:
        if _candidate_already_used(chosen, used_left_indices, used_partner_indices, used_unknown_indices):
            continue
        pairs.append(_boundary_pair_from_candidate(chosen))
        used_left_indices.add(chosen.anchor_filtered_idx)
        used_partner_indices.add(chosen.partner_filtered_idx)
        if chosen.use_unknown:
            used_unknown_indices.add(chosen.partner_filtered_idx)
            unknown_pair_count += 1
    return pairs, unknown_pair_count


def _candidate_already_used(
    chosen: _PairCandidate,
    used_left_indices: set[int],
    used_partner_indices: set[int],
    used_unknown_indices: set[int],
) -> bool:
    if chosen.anchor_filtered_idx in used_left_indices:
        return True
    if chosen.partner_filtered_idx in used_partner_indices:
        return True
    return chosen.use_unknown and chosen.partner_filtered_idx in used_unknown_indices


def _boundary_pair_from_candidate(chosen: _PairCandidate) -> _BoundaryPair:
    return _BoundaryPair(
        left_filtered_idx=int(chosen.anchor_filtered_idx),
        right_filtered_idx=int(chosen.partner_filtered_idx),
        left_track_id=int(chosen.anchor_track_id),
        right_track_id=int(chosen.partner_track_id),
        left_global=np.asarray(chosen.anchor_global, dtype=np.float64),
        right_global=np.asarray(chosen.partner_global, dtype=np.float64),
        left_local=np.asarray(chosen.anchor_local, dtype=np.float64),
        right_local=np.asarray(chosen.partner_local, dtype=np.float64),
        width_m=float(chosen.width_m),
    )


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

    config = _midpoint_order_config(config, max_segment_length_m)
    limit_m = float(config.max_midpoint_segment_length_m)
    if not math.isfinite(limit_m) or limit_m <= 0.0:
        return list(pairs)

    local_midpoints = _local_pair_midpoints(pairs)
    start_idx = _midpoint_chain_start_idx(local_midpoints)
    ordered: list[_BoundaryPair] = [pairs[start_idx]]
    used_indices: set[int] = {start_idx}
    current_midpoint = np.asarray(local_midpoints[start_idx], dtype=np.float64)
    return _extend_midpoint_chain(
        pairs=pairs,
        local_midpoints=local_midpoints,
        ordered=ordered,
        used_indices=used_indices,
        current_midpoint=current_midpoint,
        limit_m=limit_m,
        config=config,
    )


def _midpoint_order_config(
    config: Optional[MidpointPlannerConfig],
    max_segment_length_m: Optional[float],
) -> MidpointPlannerConfig:
    if config is not None:
        return config
    resolved = MidpointPlannerConfig()
    if max_segment_length_m is not None:
        resolved.max_midpoint_segment_length_m = float(max_segment_length_m)
    return resolved


def _local_pair_midpoints(pairs: list[_BoundaryPair]) -> list[np.ndarray]:
    return [np.asarray(pair.midpoint_local, dtype=np.float64) for pair in pairs]


def _midpoint_chain_start_idx(local_midpoints: list[np.ndarray]) -> int:
    return min(range(len(local_midpoints)), key=lambda idx: _midpoint_start_key(local_midpoints, idx))


def _midpoint_start_key(
    local_midpoints: list[np.ndarray],
    idx: int,
) -> tuple[float, float, float, float, int]:
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


def _extend_midpoint_chain(
    *,
    pairs: list[_BoundaryPair],
    local_midpoints: list[np.ndarray],
    ordered: list[_BoundaryPair],
    used_indices: set[int],
    current_midpoint: np.ndarray,
    limit_m: float,
    config: MidpointPlannerConfig,
) -> list[_BoundaryPair]:
    ordered_path_length_m = 0.0
    while len(ordered) < len(pairs):
        best_idx = _next_midpoint_pair_index(
            pairs=pairs,
            local_midpoints=local_midpoints,
            ordered=ordered,
            used_indices=used_indices,
            current_midpoint=current_midpoint,
            ordered_path_length_m=ordered_path_length_m,
            limit_m=limit_m,
            config=config,
        )
        if best_idx is None:
            break

        step_delta = np.asarray(local_midpoints[best_idx], dtype=np.float64) - current_midpoint
        ordered_path_length_m += float(np.hypot(step_delta[0], step_delta[1]))
        ordered.append(pairs[best_idx])
        used_indices.add(best_idx)
        current_midpoint = np.asarray(local_midpoints[best_idx], dtype=np.float64)
    return ordered


def _next_midpoint_pair_index(
    *,
    pairs: list[_BoundaryPair],
    local_midpoints: list[np.ndarray],
    ordered: list[_BoundaryPair],
    used_indices: set[int],
    current_midpoint: np.ndarray,
    ordered_path_length_m: float,
    limit_m: float,
    config: MidpointPlannerConfig,
) -> Optional[int]:
    reference_direction, _ = _midpoint_progress_reference(
        ordered_pairs=ordered,
        handoff_distance_m=float(config.midpoint_order_reference_handoff_m),
        history_size=max(2, int(config.midpoint_order_history_size)),
        ordered_path_length_m=float(ordered_path_length_m),
    )
    best_idx: Optional[int] = None
    best_cost = float("inf")
    for idx, midpoint in enumerate(local_midpoints):
        cost = _midpoint_candidate_cost(
            idx, midpoint, pairs, ordered, used_indices,
            current_midpoint, reference_direction, limit_m, config,
        )
        if cost is not None and cost < best_cost:
            best_cost = cost
            best_idx = idx
    return best_idx


def _midpoint_candidate_cost(
    idx: int,
    midpoint: np.ndarray,
    pairs: list[_BoundaryPair],
    ordered: list[_BoundaryPair],
    used_indices: set[int],
    current_midpoint: np.ndarray,
    reference_direction: np.ndarray,
    limit_m: float,
    config: MidpointPlannerConfig,
) -> Optional[float]:
    if idx in used_indices:
        return None
    delta = midpoint - current_midpoint
    distance = float(np.hypot(delta[0], delta[1]))
    if distance <= 1e-9 or distance > limit_m:
        return None
    forward_progress_m = float(np.dot(delta, reference_direction))
    max_backward_step_m = max(
        float(config.midpoint_order_backtrack_tolerance_m),
        0.5 * float(config.min_forward_progress_m),
    )
    if forward_progress_m < -max_backward_step_m:
        return None
    backward_progress_penalty = max(0.0, -forward_progress_m)
    width_jump_penalty = _midpoint_width_jump_penalty(pairs[idx], ordered[-1], config)
    return distance + (2.0 * backward_progress_penalty) + (0.25 * width_jump_penalty)


def _midpoint_width_jump_penalty(
    candidate: _BoundaryPair,
    previous: _BoundaryPair,
    config: MidpointPlannerConfig,
) -> float:
    return max(
        0.0,
        abs(float(candidate.width_m) - float(previous.width_m))
        - float(config.max_width_jump_m),
    )


def _midpoint_progress_reference(
    *,
    ordered_pairs: list[_BoundaryPair],
    handoff_distance_m: float,
    history_size: int,
    ordered_path_length_m: float,
) -> tuple[np.ndarray, bool]:
    vehicle_forward = np.asarray([1.0, 0.0], dtype=np.float64)
    if len(ordered_pairs) <= 1:
        return _initial_midpoint_reference(ordered_pairs, vehicle_forward)

    trend_direction = _midpoint_trend_direction(ordered_pairs, history_size)
    if trend_direction is None:
        return vehicle_forward, True

    handoff_distance_m = max(0.5, float(handoff_distance_m))
    if ordered_path_length_m >= handoff_distance_m:
        return trend_direction, False
    return _blended_midpoint_reference(
        vehicle_forward, trend_direction, ordered_path_length_m, handoff_distance_m,
    )


def _initial_midpoint_reference(
    ordered_pairs: list[_BoundaryPair],
    vehicle_forward: np.ndarray,
) -> tuple[np.ndarray, bool]:
    # Use direction from vehicle toward the first pair as the initial
    # reference so candidates behind the vehicle in x but further along
    # a turning track are not immediately filtered out.
    mp = np.asarray(ordered_pairs[-1].midpoint_local, dtype=np.float64)
    mp_norm = float(np.hypot(mp[0], mp[1]))
    if mp_norm > 1e-9:
        return mp / mp_norm, True
    return vehicle_forward, True


def _midpoint_trend_direction(
    ordered_pairs: list[_BoundaryPair],
    history_size: int,
) -> Optional[np.ndarray]:
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
        return None
    return trend_delta / trend_norm


def _blended_midpoint_reference(
    vehicle_forward: np.ndarray,
    trend_direction: np.ndarray,
    ordered_path_length_m: float,
    handoff_distance_m: float,
) -> tuple[np.ndarray, bool]:
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
) -> _NearFieldMetrics:
    if previous is None or previous.shape[0] < 2 or current.shape[0] < 2:
        return _NearFieldMetrics()

    current_local = _to_vehicle_frame(current, vehicle_xy, vehicle_yaw)
    previous_local = _to_vehicle_frame(previous, vehicle_xy, vehicle_yaw)
    current_rs = _resample_path(current_local, 0.25, horizon_m)
    previous_rs = _resample_path(previous_local, 0.25, horizon_m)
    count = min(current_rs.shape[0], previous_rs.shape[0])
    if count <= 0:
        return _NearFieldMetrics()

    delta = current_rs[:count] - previous_rs[:count]
    lateral = np.abs(delta[:, 1])
    displacement = np.hypot(delta[:, 0], delta[:, 1])
    return _NearFieldMetrics(
        lateral_max_m=float(np.max(lateral)) if lateral.size else 0.0,
        lateral_mean_m=float(np.mean(lateral)) if lateral.size else 0.0,
        displacement_max_m=float(np.max(displacement)) if displacement.size else 0.0,
        displacement_mean_m=float(np.mean(displacement)) if displacement.size else 0.0,
    )


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
