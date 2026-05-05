"""Core geometry for the corridor planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from sim_car.planning.planner_config_base import BasePlannerConfig
from sim_car.planning.planner_utils import (
    _build_boundary_chain,
    _clamp,
    _default_reject_counts,
    _empty_result_fields,
    _filter_and_order_cones,
    _finalize_path,
    _first_point_distance,
    _forward_extent_m,
    _path_start_heading_error,
    _resample_path,
)
from sim_car.planning.tracked_cone_planner_geometry import (
    estimate_tangents as _estimate_tangents,
    moving_average as _moving_average_points,
    pair_width_in_range,
    path_cumulative_lengths as _path_cumulative_lengths,
    path_curvature_abs_max as _path_curvature_abs_max,
    path_heading_delta_max as _path_heading_delta_max,
    path_self_intersects,
    to_vehicle_frame as _to_vehicle_frame,
    update_track_width_estimate,
)

# Meters; keeps boundary chain growth from accepting near-duplicate corridor points.
CORRIDOR_CHAIN_MIN_STEP_M = 0.35
# Meters; prevents zero or near-zero station spacing during boundary resampling.
MIN_BOUNDARY_RESAMPLE_SPACING_M = 0.05
# Count; a corridor needs one left/right station pair before and after interpolation.
MIN_CORRIDOR_STATION_COUNT = 2
# Samples; closes only one-sample gaps to avoid bridging sustained invalid corridor regions.
MAX_CORRIDOR_GAP_FILL_SAMPLES = 1
# Meters; caps prior alignment scoring to the near vehicle path segment.
PRIOR_ALIGNMENT_HORIZON_M = 4.0
# Meters; caps continuity checks to the near vehicle path segment.
NEAR_FIELD_ALIGNMENT_HORIZON_M = 3.0
_REJECT_COUNT_KEYS = (
    "corridor_geometry",
    "corridor_samples",
    "path_outside_corridor",
    "heading",
    "curvature",
    "near_field_continuity",
)


@dataclass
class CorridorPlannerConfig(BasePlannerConfig):
    planning_horizon_m: float = 25.0
    max_lateral_range_m: float = 8.0

    max_step_m: float = 6.0
    min_chain_length: int = 3

    boundary_resample_dx: float = 0.5
    min_corridor_width_m: float = 2.2
    max_corridor_width_m: float = 5.5
    min_required_corridor_samples: int = 5
    path_fit_smoothing_window: int = 5
    membership_margin_m: float = 0.15

    min_path_points: int = 4
    min_forward_extent_m: float = 2.0
    max_near_field_lateral_jump_m: float = 0.8
    max_heading_delta_rad: float = 0.75
    max_initial_heading_error_rad: float = 3.0 * math.pi / 4.0
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
    raw_left_chain_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    raw_right_chain_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    corridor_pair_audit_segments: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2, 2), dtype=np.float64)
    )
    corridor_pair_audit_anchors_local: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    corridor_pair_audit_widths_m: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    corridor_pair_audit_reasons: list[str] = field(default_factory=list)
    used_left_track_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    used_right_track_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    chain_rejection_reasons_by_track_id: dict[int, str] = field(default_factory=dict)
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
    rejected_reasons_by_filtered_index: dict[int, str] = field(default_factory=dict)


@dataclass
class _FilteredCones:
    points: np.ndarray
    local: np.ndarray
    track_ids: np.ndarray
    colors: list[str]
    colored_count: int


@dataclass
class _BoundaryChains:
    left: _BoundaryChain
    right: _BoundaryChain


@dataclass
class _CorridorCandidate:
    left_local: np.ndarray
    right_local: np.ndarray
    widths_m: np.ndarray
    anchors_local: np.ndarray
    centerline_local: np.ndarray
    width_std_m: float
    width_range_m: float
    centerline_curvature_abs_max_1pm: float
    centerline_heading_delta_max_rad: float
    prior_lateral_mean_m: float
    prior_lateral_max_m: float
    prior_heading_delta_rad: float
    audit_left_local: np.ndarray
    audit_right_local: np.ndarray
    audit_anchors_local: np.ndarray
    audit_widths_m: np.ndarray
    audit_reasons: list[str]


@dataclass
class _CorridorCandidateParts:
    left_valid: np.ndarray
    right_valid: np.ndarray
    widths_valid: np.ndarray
    anchors_local: np.ndarray
    centerline_local: np.ndarray
    audit_left_local: np.ndarray
    audit_right_local: np.ndarray
    audit_widths_m: np.ndarray
    audit_reasons: list[str]


@dataclass
class _BuiltCorridor:
    anchors_local: np.ndarray
    widths_m: np.ndarray
    left_global: np.ndarray
    right_global: np.ndarray
    anchors_global: np.ndarray
    centerline_global: np.ndarray
    rungs_global: np.ndarray
    audit_anchors_local: np.ndarray
    audit_widths_m: np.ndarray
    audit_reasons: list[str]
    audit_rungs_global: np.ndarray


@dataclass
class _PathDeltaMetrics:
    lateral_max_m: float
    lateral_mean_m: float
    displacement_max_m: float
    displacement_mean_m: float
    heading_delta_rad: float = 0.0


@dataclass
class _CorridorPathMetrics:
    centerline: np.ndarray
    centerline_local: np.ndarray
    prevalidation_centerline: np.ndarray
    near_field: _PathDeltaMetrics
    heading_delta_max_rad: float
    curvature_max_1pm: float
    seed_distance_m: float


@dataclass
class _CorridorInputs:
    cones: _FilteredCones
    chains: _BoundaryChains
    reject_counts: dict[str, int]
    expected_width_m: float


@dataclass
class _CorridorRequest:
    points_xy: np.ndarray
    colors: list[str]
    confidences: np.ndarray
    vehicle_xy: tuple[float, float]
    vehicle_yaw: float
    config: CorridorPlannerConfig
    prior: Optional[CorridorPlannerPrior]
    track_ids: Optional[np.ndarray]


def _validate_corridor_path(
    centerline: np.ndarray,
    centerline_local: np.ndarray,
    corridor: _BuiltCorridor,
    near_field: _PathDeltaMetrics,
    heading_delta_max: float,
    curvature_max: float,
    corridor_sample_count: int,
    reject_counts: dict[str, int],
    config: CorridorPlannerConfig,
) -> str:
    """Run corridor path quality checks."""
    basic_reject = _basic_corridor_path_rejection(
        centerline, centerline_local, corridor,
        corridor_sample_count, reject_counts, config,
    )
    if basic_reject:
        return basic_reject
    return _corridor_continuity_rejection(
        centerline=centerline,
        near_field=near_field,
        heading_delta_max=heading_delta_max,
        curvature_max=curvature_max,
        corridor_sample_count=corridor_sample_count,
        reject_counts=reject_counts,
        config=config,
    )


def _basic_corridor_path_rejection(
    centerline: np.ndarray,
    centerline_local: np.ndarray,
    corridor: _BuiltCorridor,
    corridor_sample_count: int,
    reject_counts: dict[str, int],
    config: CorridorPlannerConfig,
) -> str:
    if corridor_sample_count < int(config.min_required_corridor_samples):
        reject_counts["corridor_samples"] += 1
        return "too few valid corridor samples"
    if centerline.shape[0] < int(config.min_path_points):
        return "path has too few points"
    if not np.all(np.isfinite(centerline)):
        return "path contains non-finite geometry"
    if _forward_extent_m(centerline_local) < float(config.min_forward_extent_m):
        return "path forward extent too short"
    if _path_violates_corridor(
        centerline_local=centerline_local,
        corridor_center_local=corridor.anchors_local,
        corridor_widths_m=corridor.widths_m,
        membership_margin_m=float(config.membership_margin_m),
    ):
        reject_counts["path_outside_corridor"] += 1
        return "path exits corridor"
    initial_heading_error = abs(_path_start_heading_error(centerline_local))
    if initial_heading_error > float(config.max_initial_heading_error_rad):
        reject_counts["heading"] += 1
        return "path heading flip near vehicle"
    return ""


def _corridor_continuity_rejection(
    *,
    centerline: np.ndarray,
    near_field: _PathDeltaMetrics,
    heading_delta_max: float,
    curvature_max: float,
    corridor_sample_count: int,
    reject_counts: dict[str, int],
    config: CorridorPlannerConfig,
) -> str:
    if near_field.lateral_max_m > _effective_near_field_jump_limit(
        config=config,
        corridor_sample_count=corridor_sample_count,
    ):
        reject_counts["near_field_continuity"] += 1
        return "near-field continuity rejected fresh path"
    if heading_delta_max > float(config.max_heading_delta_rad):
        reject_counts["heading"] += 1
        return "path heading delta exceeded limit"
    if curvature_max > _effective_curvature_limit(
        config=config,
        corridor_sample_count=corridor_sample_count,
        heading_delta_max_rad=heading_delta_max,
    ):
        reject_counts["curvature"] += 1
        return "path curvature exceeded limit"
    if path_self_intersects(centerline, skip_closing_segment=False):
        reject_counts["corridor_geometry"] += 1
        return "path self-crossing detected"
    return ""


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
    """Compute a centerline from the valid corridor between left/right boundaries."""
    request = _CorridorRequest(
        points_xy, colors, confidences, vehicle_xy,
        vehicle_yaw, config, prior, track_ids,
    )
    return _compute_corridor_centerline_from_request(request)


def _compute_corridor_centerline_from_request(
    request: _CorridorRequest,
) -> CorridorPlannerResult:
    if request.points_xy.size == 0:
        return _empty_result("no cones available")

    inputs, input_failure = _prepare_corridor_inputs_from_request(request)
    if input_failure is not None:
        return input_failure
    assert inputs is not None

    corridor, corridor_failure = _build_corridor_or_failure(
        inputs=inputs,
        prior=request.prior,
        vehicle_xy=request.vehicle_xy,
        vehicle_yaw=request.vehicle_yaw,
        config=request.config,
    )
    if corridor_failure is not None:
        return corridor_failure
    assert corridor is not None

    return _corridor_result_from_inputs(request, inputs, corridor)


def _prepare_corridor_inputs_from_request(
    request: _CorridorRequest,
) -> tuple[Optional[_CorridorInputs], Optional[CorridorPlannerResult]]:
    return _prepare_corridor_inputs(
        points_xy=request.points_xy,
        colors=request.colors,
        confidences=request.confidences,
        track_ids=request.track_ids,
        vehicle_xy=request.vehicle_xy,
        vehicle_yaw=request.vehicle_yaw,
        prior=request.prior,
        config=request.config,
    )


def _corridor_result_from_inputs(
    request: _CorridorRequest,
    inputs: _CorridorInputs,
    corridor: _BuiltCorridor,
) -> CorridorPlannerResult:
    metrics = _compute_corridor_path_metrics(
        corridor=corridor,
        prior=request.prior,
        vehicle_xy=request.vehicle_xy,
        vehicle_yaw=request.vehicle_yaw,
        config=request.config,
    )
    centerline, status, reject_reason = _validate_corridor_metrics(
        metrics=metrics,
        corridor=corridor,
        reject_counts=inputs.reject_counts,
        config=request.config,
    )

    return _assemble_corridor_result(
        cones=inputs.cones,
        chains=inputs.chains,
        corridor=corridor,
        metrics=metrics,
        centerline=centerline,
        status=status,
        reject_reason=reject_reason,
        reject_counts=inputs.reject_counts,
        expected_width_m=inputs.expected_width_m,
        config=request.config,
    )


def _prepare_corridor_inputs(
    *,
    points_xy: np.ndarray,
    colors: list[str],
    confidences: np.ndarray,
    track_ids: Optional[np.ndarray],
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    prior: Optional[CorridorPlannerPrior],
    config: CorridorPlannerConfig,
) -> tuple[Optional[_CorridorInputs], Optional[CorridorPlannerResult]]:
    normalized_track_ids = _normalize_track_ids(points_xy, track_ids)
    cones = _filter_and_order_cones(
        points_xy=points_xy,
        colors=colors,
        confidences=confidences,
        track_ids=normalized_track_ids,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        config=config,
        filtered_cones_type=_FilteredCones,
        geometry_filter=_geometry_filter,
        include_unknown=False,
    )
    cone_failure = _cone_filter_failure_result(cones, config)
    if cone_failure is not None:
        return None, cone_failure

    chains = _make_boundary_chains(cones, config)
    reject_counts = _default_reject_counts(_REJECT_COUNT_KEYS)
    expected_width_m = _expected_width_m(prior, config)
    chain_failure = _boundary_chain_failure_result(
        cones=cones,
        chains=chains,
        reject_counts=reject_counts,
        expected_width_m=expected_width_m,
        config=config,
    )
    if chain_failure is not None:
        return None, chain_failure
    return _CorridorInputs(cones, chains, reject_counts, expected_width_m), None


def _build_corridor_or_failure(
    *,
    inputs: _CorridorInputs,
    prior: Optional[CorridorPlannerPrior],
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: CorridorPlannerConfig,
) -> tuple[Optional[_BuiltCorridor], Optional[CorridorPlannerResult]]:
    prior_centerline_local = _prior_centerline_local(prior, vehicle_xy, vehicle_yaw)
    corridor, _build_reason = _build_corridor(
        left_chain=inputs.chains.left,
        right_chain=inputs.chains.right,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        config=config,
        prior_centerline_local=prior_centerline_local,
    )
    if corridor is not None:
        return corridor, None
    inputs.reject_counts["corridor_geometry"] += 1
    return None, _failure_result_with_chains(
        status="no valid corridor overlap",
        cones=inputs.cones,
        chains=inputs.chains,
        reject_counts=inputs.reject_counts,
        expected_width_m=inputs.expected_width_m,
    )


def _normalize_track_ids(
    points_xy: np.ndarray,
    track_ids: Optional[np.ndarray],
) -> np.ndarray:
    if track_ids is None or len(track_ids) != points_xy.shape[0]:
        return np.arange(points_xy.shape[0], dtype=np.int64)
    return np.asarray(track_ids, dtype=np.int64)


def _cone_filter_failure_result(
    cones: _FilteredCones,
    config: CorridorPlannerConfig,
) -> Optional[CorridorPlannerResult]:
    if cones.colored_count == 0:
        return _empty_result(
            "no colored cones in planning region",
            reject_counts=_default_reject_counts(_REJECT_COUNT_KEYS),
        )
    if cones.colored_count >= int(config.min_required_cones):
        return None
    return _empty_result(
        f"usable colored cones below minimum ({cones.colored_count} < {int(config.min_required_cones)})",
        filtered_points=cones.points,
        filtered_colors=cones.colors,
        reject_counts=_default_reject_counts(_REJECT_COUNT_KEYS),
    )


def _make_boundary_chains(
    cones: _FilteredCones,
    config: CorridorPlannerConfig,
) -> _BoundaryChains:
    left_indices = np.flatnonzero(np.array([c == "blue" for c in cones.colors], dtype=bool))
    right_indices = np.flatnonzero(np.array([c == "yellow" for c in cones.colors], dtype=bool))
    return _BoundaryChains(
        left=_build_boundary_chain(
            filtered_points=cones.points,
            filtered_local=cones.local,
            side_indices=left_indices,
            config=config,
            boundary_chain_type=_BoundaryChain,
            collect_rejection_reasons=True,
            min_step_m=min(float(config.min_step_m), CORRIDOR_CHAIN_MIN_STEP_M),
        ),
        right=_build_boundary_chain(
            filtered_points=cones.points,
            filtered_local=cones.local,
            side_indices=right_indices,
            config=config,
            boundary_chain_type=_BoundaryChain,
            collect_rejection_reasons=True,
            min_step_m=min(float(config.min_step_m), CORRIDOR_CHAIN_MIN_STEP_M),
        ),
    )


def _expected_width_m(
    prior: Optional[CorridorPlannerPrior],
    config: CorridorPlannerConfig,
) -> float:
    prior_width_m = (
        prior.previous_width_m
        if prior is not None and prior.previous_width_m is not None
        else config.initial_width_m
    )
    return _clamp(prior_width_m, config.min_width_m, config.max_width_m)


def _boundary_chain_failure_result(
    *,
    cones: _FilteredCones,
    chains: _BoundaryChains,
    reject_counts: dict[str, int],
    expected_width_m: float,
    config: CorridorPlannerConfig,
) -> Optional[CorridorPlannerResult]:
    if (
        chains.left.filtered_indices.size >= int(config.min_chain_length)
        and chains.right.filtered_indices.size >= int(config.min_chain_length)
    ):
        return None
    return _failure_result_with_chains(
        status="no reliable corridor boundaries",
        cones=cones,
        chains=chains,
        reject_counts=reject_counts,
        expected_width_m=expected_width_m,
    )


def _failure_result_with_chains(
    *,
    status: str,
    cones: _FilteredCones,
    chains: _BoundaryChains,
    reject_counts: dict[str, int],
    expected_width_m: float,
) -> CorridorPlannerResult:
    return _result_with_metadata(
        result=_empty_result(
            status,
            filtered_points=cones.points,
            filtered_colors=cones.colors,
            left_boundary=chains.left.global_points,
            right_boundary=chains.right.global_points,
            reject_counts=reject_counts,
            reject_reason=status,
        ),
        left_chain=chains.left,
        right_chain=chains.right,
        filtered_track_ids=cones.track_ids,
        planner_mode="none",
        filtered_track_width_m=expected_width_m,
    )


def _prior_centerline_local(
    prior: Optional[CorridorPlannerPrior],
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
) -> Optional[np.ndarray]:
    if prior is None or prior.previous_centerline is None:
        return None
    previous_centerline = np.asarray(prior.previous_centerline, dtype=np.float64)
    if previous_centerline.shape[0] < MIN_CORRIDOR_STATION_COUNT:
        return None
    if not np.all(np.isfinite(previous_centerline)):
        return None
    return _to_vehicle_frame(previous_centerline, vehicle_xy, vehicle_yaw)


def _compute_corridor_path_metrics(
    *,
    corridor: _BuiltCorridor,
    prior: Optional[CorridorPlannerPrior],
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: CorridorPlannerConfig,
) -> _CorridorPathMetrics:
    centerline = _finalize_path(corridor.centerline_global, config)
    centerline_local = _to_vehicle_frame(centerline, vehicle_xy, vehicle_yaw)
    near_field = _near_field_delta_metrics(
        current=centerline,
        previous=None if prior is None else prior.previous_centerline,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        horizon_m=config.jump_check_horizon_m,
    )
    return _CorridorPathMetrics(
        centerline=centerline,
        centerline_local=centerline_local,
        prevalidation_centerline=np.array(centerline, copy=True),
        near_field=near_field,
        heading_delta_max_rad=_path_heading_delta_max(centerline_local),
        curvature_max_1pm=_path_curvature_abs_max(centerline_local),
        seed_distance_m=_first_point_distance(centerline_local),
    )


def _validate_corridor_metrics(
    *,
    metrics: _CorridorPathMetrics,
    corridor: _BuiltCorridor,
    reject_counts: dict[str, int],
    config: CorridorPlannerConfig,
) -> tuple[np.ndarray, str, str]:
    reject_reason = _validate_corridor_path(
        metrics.centerline,
        metrics.centerline_local,
        corridor,
        metrics.near_field,
        metrics.heading_delta_max_rad,
        metrics.curvature_max_1pm,
        int(corridor.anchors_local.shape[0]),
        reject_counts,
        config,
    )
    status = reject_reason if reject_reason else "ok"
    return _centerline_for_status(metrics.centerline, status), status, reject_reason


def _centerline_for_status(centerline: np.ndarray, status: str) -> np.ndarray:
    if status == "ok" or status in {
        "path has too few points",
        "path forward extent too short",
        "near-field continuity rejected fresh path",
    }:
        return centerline
    return np.empty((0, 2), dtype=np.float64)


def _assemble_corridor_result(
    *,
    cones: _FilteredCones,
    chains: _BoundaryChains,
    corridor: _BuiltCorridor,
    metrics: _CorridorPathMetrics,
    centerline: np.ndarray,
    status: str,
    reject_reason: str,
    reject_counts: dict[str, int],
    expected_width_m: float,
    config: CorridorPlannerConfig,
) -> CorridorPlannerResult:
    result = _new_corridor_result(cones, corridor, metrics, centerline, status)
    _apply_corridor_scalar_metadata(
        result, corridor, metrics, status, reject_reason,
        reject_counts, expected_width_m, config,
    )
    _apply_corridor_array_metadata(result, cones, chains, corridor)
    return result


def _new_corridor_result(
    cones: _FilteredCones,
    corridor: _BuiltCorridor,
    metrics: _CorridorPathMetrics,
    centerline: np.ndarray,
    status: str,
) -> CorridorPlannerResult:
    return CorridorPlannerResult(
        filtered_points=cones.points,
        filtered_colors=cones.colors,
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.empty((0, 2), dtype=np.int64),
        selected_pair_track_ids=np.empty((0, 2), dtype=np.int64),
        midpoints_raw=corridor.anchors_global,
        centerline=centerline,
        prevalidation_centerline=metrics.prevalidation_centerline,
        left_boundary=corridor.left_global,
        right_boundary=corridor.right_global,
        used_fallback=False,
        status=status,
    )


def _apply_corridor_scalar_metadata(
    result: CorridorPlannerResult,
    corridor: _BuiltCorridor,
    metrics: _CorridorPathMetrics,
    status: str,
    reject_reason: str,
    reject_counts: dict[str, int],
    expected_width_m: float,
    config: CorridorPlannerConfig,
) -> None:
    corridor_sample_count = int(corridor.anchors_local.shape[0])
    width_median_m = _corridor_width_median(corridor.widths_m)
    result.candidate_count = corridor_sample_count
    result.selected_chain_length = corridor_sample_count
    result.selected_chain_width_median = width_median_m
    result.expected_width_prior_m = float(expected_width_m)
    result.near_field_lateral_max_m = float(metrics.near_field.lateral_max_m)
    result.near_field_lateral_mean_m = float(metrics.near_field.lateral_mean_m)
    result.near_field_displacement_max_m = float(metrics.near_field.displacement_max_m)
    result.near_field_displacement_mean_m = float(metrics.near_field.displacement_mean_m)
    result.near_field_kink_max_rad = float(metrics.heading_delta_max_rad)
    result.seed_midpoint_distance_m = float(metrics.seed_distance_m)
    result.reject_reason = reject_reason
    result.reject_counts = reject_counts
    result.planner_mode = "corridor" if status == "ok" else "none"
    result.accepted_pair_count = corridor_sample_count
    result.filtered_track_width_m = _filtered_track_width_m(
        status, corridor_sample_count, width_median_m, expected_width_m, config,
    )
    result.corridor_width_min_m = _corridor_width_min(corridor.widths_m)
    result.corridor_width_max_m = _corridor_width_max(corridor.widths_m)


def _apply_corridor_array_metadata(
    result: CorridorPlannerResult,
    cones: _FilteredCones,
    chains: _BoundaryChains,
    corridor: _BuiltCorridor,
) -> None:
    result.pair_segments = corridor.rungs_global
    result.raw_left_chain_points = chains.left.global_points
    result.raw_right_chain_points = chains.right.global_points
    result.corridor_pair_audit_segments = corridor.audit_rungs_global
    result.corridor_pair_audit_anchors_local = corridor.audit_anchors_local
    result.corridor_pair_audit_widths_m = corridor.audit_widths_m
    result.corridor_pair_audit_reasons = list(corridor.audit_reasons)
    result.used_left_track_ids = np.asarray(
        cones.track_ids[chains.left.filtered_indices], dtype=np.int64,
    )
    result.used_right_track_ids = np.asarray(
        cones.track_ids[chains.right.filtered_indices], dtype=np.int64,
    )
    result.chain_rejection_reasons_by_track_id = _chain_rejection_reasons_by_track_id(
        filtered_track_ids=cones.track_ids,
        left_chain=chains.left,
        right_chain=chains.right,
    )
    result.left_chain_length = int(chains.left.filtered_indices.size)
    result.right_chain_length = int(chains.right.filtered_indices.size)


def _filtered_track_width_m(
    status: str,
    corridor_sample_count: int,
    width_median_m: float,
    expected_width_m: float,
    config: CorridorPlannerConfig,
) -> float:
    if (
        status == "ok"
        and corridor_sample_count >= int(config.min_trustworthy_pairs)
        and math.isfinite(width_median_m)
    ):
        return float(width_median_m)
    return float(expected_width_m)


def _corridor_width_min(widths_m: np.ndarray) -> float:
    return float(np.min(widths_m)) if widths_m.size else float("nan")


def _corridor_width_max(widths_m: np.ndarray) -> float:
    return float(np.max(widths_m)) if widths_m.size else float("nan")


def _corridor_width_median(widths_m: np.ndarray) -> float:
    return float(np.median(widths_m)) if widths_m.size else float("nan")


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


def _build_corridor(
    *,
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: CorridorPlannerConfig,
    prior_centerline_local: Optional[np.ndarray] = None,
) -> tuple[Optional[_BuiltCorridor], str]:
    dx = max(MIN_BOUNDARY_RESAMPLE_SPACING_M, float(config.boundary_resample_dx))
    left_local = _resample_boundary_by_station(left_chain.local_points, dx)
    right_local = _resample_boundary_by_station(right_chain.local_points, dx)
    if left_local is None or right_local is None:
        return None, "boundary resample failed"

    station_count = min(left_local.shape[0], right_local.shape[0])
    if station_count < MIN_CORRIDOR_STATION_COUNT:
        return None, "too few corridor stations"

    chosen, reason = _build_corridor_candidate(
        left_local=np.asarray(left_local[:station_count], dtype=np.float64),
        right_local=np.asarray(right_local[:station_count], dtype=np.float64),
        config=config,
        prior_centerline_local=prior_centerline_local,
    )
    if chosen is None:
        return None, reason
    return _materialize_built_corridor(
        chosen, vehicle_xy, vehicle_yaw, config,
    ), ""


def _materialize_built_corridor(
    candidate: _CorridorCandidate,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: CorridorPlannerConfig,
) -> _BuiltCorridor:
    centerline_local = _fit_centerline_from_anchors(candidate.anchors_local, config)
    left_global = _from_vehicle_frame(candidate.left_local, vehicle_xy, vehicle_yaw)
    right_global = _from_vehicle_frame(candidate.right_local, vehicle_xy, vehicle_yaw)
    audit_left_global = _from_vehicle_frame(candidate.audit_left_local, vehicle_xy, vehicle_yaw)
    audit_right_global = _from_vehicle_frame(candidate.audit_right_local, vehicle_xy, vehicle_yaw)
    return _BuiltCorridor(
        anchors_local=candidate.anchors_local,
        widths_m=candidate.widths_m,
        left_global=left_global,
        right_global=right_global,
        anchors_global=_from_vehicle_frame(candidate.anchors_local, vehicle_xy, vehicle_yaw),
        centerline_global=_from_vehicle_frame(centerline_local, vehicle_xy, vehicle_yaw),
        rungs_global=_rungs_from_boundaries(left_global, right_global),
        audit_anchors_local=candidate.audit_anchors_local,
        audit_widths_m=candidate.audit_widths_m,
        audit_reasons=list(candidate.audit_reasons),
        audit_rungs_global=_rungs_from_boundaries(audit_left_global, audit_right_global),
    )


def _rungs_from_boundaries(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    rungs = np.empty((left.shape[0], 2, 2), dtype=np.float64)
    rungs[:, 0, :] = left
    rungs[:, 1, :] = right
    return rungs


def _build_corridor_candidate(
    *,
    left_local: np.ndarray,
    right_local: np.ndarray,
    config: CorridorPlannerConfig,
    prior_centerline_local: Optional[np.ndarray] = None,
) -> tuple[Optional[_CorridorCandidate], str]:
    if (
        left_local.shape[0] < MIN_CORRIDOR_STATION_COUNT
        or right_local.shape[0] < MIN_CORRIDOR_STATION_COUNT
    ):
        return None, "too few boundary samples"

    valid_slice, widths, raw_valid_mask, reason = _corridor_candidate_slice(
        left_local, right_local, config,
    )
    if valid_slice is None:
        return None, reason
    return _corridor_candidate_from_slice(
        left_local=left_local,
        right_local=right_local,
        widths=widths,
        raw_valid_mask=raw_valid_mask,
        valid_slice=valid_slice,
        config=config,
        prior_centerline_local=prior_centerline_local,
    )


def _corridor_candidate_slice(
    left_local: np.ndarray,
    right_local: np.ndarray,
    config: CorridorPlannerConfig,
) -> tuple[Optional[slice], np.ndarray, np.ndarray, str]:
    widths = np.hypot(left_local[:, 0] - right_local[:, 0], left_local[:, 1] - right_local[:, 1])
    raw_valid_mask = _corridor_valid_mask(
        left_local=left_local,
        right_local=right_local,
        widths=widths,
        config=config,
    )
    valid_mask = _fill_small_invalid_gaps(
        raw_valid_mask, max_gap=MAX_CORRIDOR_GAP_FILL_SAMPLES,
    )
    valid_slice = _longest_valid_slice(valid_mask)
    reason = "" if valid_slice is not None else "no valid corridor slice"
    return valid_slice, widths, raw_valid_mask, reason


def _corridor_candidate_from_slice(
    *,
    left_local: np.ndarray,
    right_local: np.ndarray,
    widths: np.ndarray,
    raw_valid_mask: np.ndarray,
    valid_slice: slice,
    config: CorridorPlannerConfig,
    prior_centerline_local: Optional[np.ndarray],
) -> tuple[Optional[_CorridorCandidate], str]:
    audit_reasons = _corridor_pair_audit_reasons(
        left_local=left_local,
        right_local=right_local,
        widths=widths,
        raw_valid_mask=raw_valid_mask,
        accepted_slice=valid_slice,
        config=config,
    )
    parts = _corridor_candidate_parts(
        left_local, right_local, widths, valid_slice, audit_reasons, config,
    )
    if parts is None:
        return None, "too few valid corridor samples"
    return _new_corridor_candidate(parts, prior_centerline_local, config), ""


def _corridor_candidate_parts(
    left_local: np.ndarray,
    right_local: np.ndarray,
    widths: np.ndarray,
    valid_slice: slice,
    audit_reasons: list[str],
    config: CorridorPlannerConfig,
) -> Optional[_CorridorCandidateParts]:
    left_valid = np.asarray(left_local[valid_slice], dtype=np.float64)
    right_valid = np.asarray(right_local[valid_slice], dtype=np.float64)
    widths_valid = np.asarray(widths[valid_slice], dtype=np.float64)
    if left_valid.shape[0] < int(config.min_required_corridor_samples):
        return None
    anchors_local = 0.5 * (left_valid + right_valid)
    centerline_local = _fit_centerline_from_anchors(anchors_local, config)
    return _CorridorCandidateParts(
        left_valid=left_valid,
        right_valid=right_valid,
        widths_valid=widths_valid,
        anchors_local=anchors_local,
        centerline_local=centerline_local,
        audit_left_local=np.asarray(left_local, dtype=np.float64),
        audit_right_local=np.asarray(right_local, dtype=np.float64),
        audit_widths_m=np.asarray(widths, dtype=np.float64),
        audit_reasons=audit_reasons,
    )


def _new_corridor_candidate(
    parts: _CorridorCandidateParts,
    prior_centerline_local: Optional[np.ndarray],
    config: CorridorPlannerConfig,
) -> _CorridorCandidate:
    width_std_m = float(np.std(parts.widths_valid)) if parts.widths_valid.size else float("inf")
    centerline_heading_delta_max = _path_heading_delta_max(parts.centerline_local)
    prior_alignment = _candidate_prior_alignment(
        parts.centerline_local, prior_centerline_local, config,
    )
    return _CorridorCandidate(
        left_local=parts.left_valid,
        right_local=parts.right_valid,
        widths_m=parts.widths_valid,
        anchors_local=parts.anchors_local,
        centerline_local=parts.centerline_local,
        width_std_m=width_std_m,
        width_range_m=_corridor_candidate_width_range_m(parts.widths_valid),
        centerline_curvature_abs_max_1pm=_corridor_candidate_curvature(parts, config),
        centerline_heading_delta_max_rad=centerline_heading_delta_max,
        prior_lateral_mean_m=float(prior_alignment.lateral_mean_m),
        prior_lateral_max_m=float(prior_alignment.lateral_max_m),
        prior_heading_delta_rad=float(prior_alignment.heading_delta_rad),
        audit_left_local=parts.audit_left_local,
        audit_right_local=parts.audit_right_local,
        audit_anchors_local=0.5 * (parts.audit_left_local + parts.audit_right_local),
        audit_widths_m=parts.audit_widths_m,
        audit_reasons=parts.audit_reasons,
    )


def _corridor_candidate_width_range_m(widths_m: np.ndarray) -> float:
    if widths_m.size:
        return float(np.max(widths_m) - np.min(widths_m))
    return float("inf")


def _corridor_candidate_curvature(
    parts: _CorridorCandidateParts,
    config: CorridorPlannerConfig,
) -> float:
    return _path_curvature_abs_max(
        _resample_path(
            parts.centerline_local,
            resolution_m=float(config.path_resolution_m),
            max_length_m=float(config.max_path_length_m),
        )
    )


def _candidate_prior_alignment(
    centerline_local: np.ndarray,
    prior_centerline_local: Optional[np.ndarray],
    config: CorridorPlannerConfig,
) -> _PathDeltaMetrics:
    if (
        prior_centerline_local is None
        or prior_centerline_local.shape[0] < MIN_CORRIDOR_STATION_COUNT
    ):
        return _nan_path_delta_metrics()
    return _path_alignment_metrics(
        current_local=centerline_local,
        previous_local=prior_centerline_local,
        horizon_m=min(float(config.jump_check_horizon_m), PRIOR_ALIGNMENT_HORIZON_M),
    )


def _nan_path_delta_metrics() -> _PathDeltaMetrics:
    return _PathDeltaMetrics(
        lateral_max_m=float("nan"),
        lateral_mean_m=float("nan"),
        displacement_max_m=float("nan"),
        displacement_mean_m=float("nan"),
        heading_delta_rad=float("nan"),
    )


def _corridor_candidate_score(
    candidate: _CorridorCandidate,
    config: CorridorPlannerConfig,
) -> tuple[float, ...]:
    del config
    anchor_count = float(candidate.anchors_local.shape[0])
    width_std_m = float(candidate.width_std_m)
    width_range_m = float(candidate.width_range_m)
    curvature_abs_max = float(candidate.centerline_curvature_abs_max_1pm)
    heading_delta_max = float(candidate.centerline_heading_delta_max_rad)
    prior_lateral_mean_m = float(candidate.prior_lateral_mean_m)
    prior_lateral_max_m = float(candidate.prior_lateral_max_m)
    prior_heading_delta_rad = float(candidate.prior_heading_delta_rad)
    has_prior_alignment = 1.0 if math.isfinite(prior_lateral_mean_m) else 0.0
    return (
        has_prior_alignment,
        -prior_lateral_mean_m if has_prior_alignment else 0.0,
        -prior_lateral_max_m if has_prior_alignment else 0.0,
        -prior_heading_delta_rad if has_prior_alignment else 0.0,
        anchor_count,
        -width_std_m,
        -width_range_m,
        -curvature_abs_max,
        -heading_delta_max,
    )


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
    valid_mask &= np.asarray(
        [
            pair_width_in_range(width, config.min_corridor_width_m, config.max_corridor_width_m)
            for width in widths
        ],
        dtype=bool,
    )
    valid_mask &= left_local[:, 0] >= -float(config.behind_drop_m)
    valid_mask &= right_local[:, 0] >= -float(config.behind_drop_m)
    valid_mask &= left_local[:, 0] <= float(config.planning_horizon_m)
    valid_mask &= right_local[:, 0] <= float(config.planning_horizon_m)
    return valid_mask


def _corridor_pair_audit_reasons(
    *,
    left_local: np.ndarray,
    right_local: np.ndarray,
    widths: np.ndarray,
    raw_valid_mask: np.ndarray,
    accepted_slice: slice,
    config: CorridorPlannerConfig,
) -> list[str]:
    accepted = np.zeros((len(widths),), dtype=bool)
    accepted[accepted_slice] = True
    reasons: list[str] = []
    for idx in range(len(widths)):
        if bool(accepted[idx]):
            reasons.append("pair_valid")
            continue
        reasons.append(
            _corridor_pair_audit_reason(
                left=np.asarray(left_local[idx], dtype=np.float64),
                right=np.asarray(right_local[idx], dtype=np.float64),
                width_m=float(widths[idx]),
                is_raw_valid=bool(raw_valid_mask[idx]),
                config=config,
            )
        )
    return reasons


def _corridor_pair_audit_reason(
    *,
    left: np.ndarray,
    right: np.ndarray,
    width_m: float,
    is_raw_valid: bool,
    config: CorridorPlannerConfig,
) -> str:
    if (
        not math.isfinite(width_m)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        return "pair_nonfinite"
    if width_m < float(config.min_corridor_width_m):
        return "pair_width_too_narrow"
    if width_m > float(config.max_corridor_width_m):
        return "pair_width_too_wide"
    if float(left[0]) < -float(config.behind_drop_m):
        return "pair_left_behind"
    if float(right[0]) < -float(config.behind_drop_m):
        return "pair_right_behind"
    if float(left[0]) > float(config.planning_horizon_m):
        return "pair_left_beyond_horizon"
    if float(right[0]) > float(config.planning_horizon_m):
        return "pair_right_beyond_horizon"
    if is_raw_valid:
        return "pair_not_in_longest_valid_slice"
    return "pair_rejected_unknown"


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
    heading_delta_max_rad: float = 0.0,
) -> float:
    limit = float(config.max_curvature)
    if limit < 0.2:
        return limit
    if corridor_sample_count >= int(config.min_required_corridor_samples) + 2:
        limit *= 1.75
    if float(heading_delta_max_rad) >= 0.28:
        limit *= 1.35
    elif float(heading_delta_max_rad) >= 0.18:
        limit *= 1.15
    return limit


def _fit_centerline_from_anchors(
    anchors_local: np.ndarray,
    config: CorridorPlannerConfig,
) -> np.ndarray:
    fitted = np.asarray(anchors_local, dtype=np.float64)
    window = max(1, int(config.path_fit_smoothing_window))
    anchor_heading_delta_max = _path_heading_delta_max(fitted)
    curvature_limit = float(config.max_curvature)
    if anchor_heading_delta_max >= 0.28:
        window = max(1, window - 2)
        curvature_limit *= 1.35
    elif anchor_heading_delta_max >= 0.14:
        window = max(1, window - 1)
        curvature_limit *= 1.15
    if fitted.shape[0] <= 2 or window <= 1:
        return fitted

    best = np.array(fitted, copy=True)
    best_curvature = float("inf")
    for _ in range(3):
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
        if curvature <= curvature_limit:
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


def _near_field_delta_metrics(
    *,
    current: np.ndarray,
    previous: Optional[np.ndarray],
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    horizon_m: float,
) -> _PathDeltaMetrics:
    if previous is None or previous.shape[0] < 2 or current.shape[0] < 2:
        return _zero_path_delta_metrics()

    current_local = _to_vehicle_frame(current, vehicle_xy, vehicle_yaw)
    previous_local = _to_vehicle_frame(previous, vehicle_xy, vehicle_yaw)
    return _path_alignment_metrics(
        current_local=current_local,
        previous_local=previous_local,
        horizon_m=min(float(horizon_m), NEAR_FIELD_ALIGNMENT_HORIZON_M),
    )


def _path_alignment_metrics(
    *,
    current_local: Optional[np.ndarray],
    previous_local: Optional[np.ndarray],
    horizon_m: float,
) -> _PathDeltaMetrics:
    empty = _zero_path_delta_metrics()
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
    return _path_delta_metrics_from_prefixes(current_prefix, previous_prefix, count)


def _path_delta_metrics_from_prefixes(
    current_prefix: np.ndarray,
    previous_prefix: np.ndarray,
    count: int,
) -> _PathDeltaMetrics:
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
    return _PathDeltaMetrics(
        lateral_max_m=float(np.max(lateral)) if lateral.size else 0.0,
        lateral_mean_m=float(np.mean(lateral)) if lateral.size else 0.0,
        displacement_max_m=float(np.max(displacement)) if displacement.size else 0.0,
        displacement_mean_m=float(np.mean(displacement)) if displacement.size else 0.0,
        heading_delta_rad=heading_delta,
    )


def _zero_path_delta_metrics() -> _PathDeltaMetrics:
    return _PathDeltaMetrics(
        lateral_max_m=0.0,
        lateral_mean_m=0.0,
        displacement_max_m=0.0,
        displacement_mean_m=0.0,
        heading_delta_rad=0.0,
    )


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
    samples = np.arange(0.0, total + 1e-9, 0.25, dtype=np.float64)
    if samples.size == 0 or samples[-1] < total:
        samples = np.concatenate((samples, [total]))
    x = np.interp(samples, cumulative, pts[:, 0])
    y = np.interp(samples, cumulative, pts[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


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
        **_empty_result_fields(
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        ),
        prevalidation_centerline=np.empty((0, 2), dtype=np.float64),
        status=status,
        reject_counts=reject_counts or _default_reject_counts(_REJECT_COUNT_KEYS),
        reject_reason=reject_reason,
    )


def _result_with_metadata(
    *,
    result: CorridorPlannerResult,
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
    filtered_track_ids: Optional[np.ndarray] = None,
    planner_mode: str,
    filtered_track_width_m: float,
) -> CorridorPlannerResult:
    track_ids = (
        np.asarray(filtered_track_ids, dtype=np.int64)
        if filtered_track_ids is not None
        else None
    )
    result.left_chain_length = int(left_chain.filtered_indices.size)
    result.right_chain_length = int(right_chain.filtered_indices.size)
    result.left_boundary = left_chain.global_points
    result.right_boundary = right_chain.global_points
    result.raw_left_chain_points = left_chain.global_points
    result.raw_right_chain_points = right_chain.global_points
    if track_ids is not None and track_ids.size > 0:
        result.used_left_track_ids = np.asarray(
            track_ids[left_chain.filtered_indices],
            dtype=np.int64,
        )
        result.used_right_track_ids = np.asarray(
            track_ids[right_chain.filtered_indices],
            dtype=np.int64,
        )
        result.chain_rejection_reasons_by_track_id = _chain_rejection_reasons_by_track_id(
            filtered_track_ids=track_ids,
            left_chain=left_chain,
            right_chain=right_chain,
        )
    result.planner_mode = planner_mode
    result.filtered_track_width_m = float(filtered_track_width_m)
    return result


def _chain_rejection_reasons_by_track_id(
    *,
    filtered_track_ids: np.ndarray,
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
) -> dict[int, str]:
    track_ids = np.asarray(filtered_track_ids, dtype=np.int64)
    out: dict[int, str] = {}
    for filtered_idx, reason in left_chain.rejected_reasons_by_filtered_index.items():
        idx = int(filtered_idx)
        if 0 <= idx < track_ids.size:
            out[int(track_ids[idx])] = str(reason)
    for filtered_idx, reason in right_chain.rejected_reasons_by_filtered_index.items():
        idx = int(filtered_idx)
        if 0 <= idx < track_ids.size:
            out[int(track_ids[idx])] = str(reason)
    return out
