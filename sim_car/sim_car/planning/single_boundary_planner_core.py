"""Core geometry for the single-boundary planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, cast

import numpy as np

from sim_car.planning.planner_config_base import BasePlannerConfig
from sim_car.planning.tracked_cone_planner_geometry import (
    _build_boundary_chain,
    _clamp,
    _default_reject_counts,
    _empty_result_fields,
    _filter_and_order_cones,
    _finalize_path,
    _first_point_distance,
    _forward_extent_m,
    _merge_reject_counts,
    _path_start_heading_error,
    _resample_path,
    estimate_tangents as _estimate_tangents,
    inward_distance,
    inward_normal as _inward_normal,
    pair_width_in_range,
    path_cumulative_lengths as _path_cumulative_lengths,
    path_heading_delta_max as _path_heading_delta_max,
    path_self_intersects as _path_self_intersects,
    to_vehicle_frame as _to_vehicle_frame,
    UnknownPartnerCheck,
    unknown_partner_check,
    unknown_partner_within_limits,
    width_jump_exceeds,
)


# Unknown partner candidates get a small cost penalty so real colored partners win ties.
_UNKNOWN_PARTNER_COST_BIAS_M = 0.05
# Sparse paths get the relaxed continuity threshold when only the minimum pair support exists.
_MIN_SPARSE_PAIR_CONTINUITY_COUNT = 3
# Near-field alignment is capped to the immediate path in front of the vehicle.
_MAX_NEAR_FIELD_ALIGNMENT_HORIZON_M = 3.0
# A single detected boundary is offset by half of the estimated track width.
_TRACK_HALF_WIDTH_SCALE = 0.5
# A pair midpoint is the average of its two boundary endpoints.
_MIDPOINT_ENDPOINT_WEIGHT = 0.5
# Near-field comparison allows a small behind-vehicle margin for the seed point.
_LOCAL_FORWARD_BACKTRACK_MARGIN_M = 0.1
# The near-field prefix must cover at least a short physical segment.
_MIN_LOCAL_PREFIX_LENGTH_M = 0.25
# Degenerate paths below this length are represented by their first point only.
_MIN_PATH_LENGTH_M = 1e-6
_REJECT_COUNT_KEYS = (
    "wrong_side",
    "width",
    "width_range",
    "width_prior",
    "orientation",
    "progress",
    "near_field_continuity",
    "midpoint_kink",
    "seed_distance",
)


@dataclass
class SingleBoundaryPlannerConfig(BasePlannerConfig):
    allow_unknown_pair_completion: bool = True
    unknown_pair_search_radius_m: float = 1.25
    unknown_pair_max_longitudinal_error_m: float = 1.5
    unknown_pair_max_width_error_m: float = 0.9
    max_consecutive_unknown_pairs: int = 2

    max_step_m: float = 6.0
    min_chain_length: int = 2

    min_pair_width_m: float = 2.2
    max_pair_width_m: float = 5.5
    max_width_jump_m: float = 0.8
    min_pair_count: int = 3
    pair_reassignment_margin: float = 0.25

    smoothing_window: int = 3
    max_heading_delta_rad: float = 0.75

    min_path_points: int = 2
    min_forward_extent_m: float = 1.0
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
        return _MIDPOINT_ENDPOINT_WEIGHT * (self.left_global + self.right_global)


@dataclass
class _FilteredCones:
    points: np.ndarray
    local: np.ndarray
    track_ids: np.ndarray
    colors: list[str]
    colored_count: int


@dataclass(frozen=True)
class _BoundaryChains:
    left: _BoundaryChain
    right: _BoundaryChain
    unknown_indices: np.ndarray


@dataclass(frozen=True)
class _PairingAnchor:
    anchor_chain: _BoundaryChain
    other_chain: _BoundaryChain
    anchor_side: str


@dataclass(frozen=True)
class _AnchorContext:
    local: np.ndarray
    global_point: np.ndarray
    tangent: np.ndarray
    filtered_idx: int
    track_id: int
    inward_normal: np.ndarray


@dataclass(frozen=True)
class _PartnerOption:
    use_unknown: bool
    other_pos: int
    partner_filtered_idx: int
    partner_track_id: int
    partner_global: np.ndarray
    partner_local: np.ndarray
    width_m: float
    cost: float
    sort_key: tuple[float, float, float, int]


@dataclass
class _PairingState:
    pairs: list[_BoundaryPair] = field(default_factory=list)
    next_other_start: int = 0
    last_width: Optional[float] = None
    last_partner_progress: float = float("-inf")
    used_unknown_indices: set[int] = field(default_factory=set)
    consecutive_unknown_pairs: int = 0
    candidate_count: int = 0
    unknown_pair_count: int = 0


@dataclass(frozen=True)
class _PairingResult:
    pairs: list[_BoundaryPair]
    candidate_count: int
    unknown_pair_count: int
    measured_width_m: float
    reject_counts: dict[str, int]


@dataclass(frozen=True)
class _PairingSearch:
    anchor: _PairingAnchor
    state: _PairingState
    previous_partner_by_anchor: dict[int, int]
    cones: _FilteredCones
    unknown_indices: np.ndarray
    expected_width_m: float
    reject_counts: dict[str, int]
    config: SingleBoundaryPlannerConfig


@dataclass(frozen=True)
class _FallbackSelection:
    chain: _BoundaryChain
    side: str
    raw_offset_path: np.ndarray


@dataclass(frozen=True)
class _NearFieldMetrics:
    lateral_max_m: float = 0.0
    lateral_mean_m: float = 0.0
    displacement_max_m: float = 0.0
    displacement_mean_m: float = 0.0

    def __getitem__(self, key: str) -> float:
        try:
            return float(getattr(self, key))
        except AttributeError as exc:
            raise KeyError(key) from exc


@dataclass(frozen=True)
class _PathAlignmentMetrics(_NearFieldMetrics):
    heading_delta_rad: float = 0.0


@dataclass(frozen=True)
class _CandidatePath:
    centerline: np.ndarray
    centerline_local: np.ndarray
    seed_distance_m: float
    near_field: _NearFieldMetrics
    heading_delta_max: float
    continuity_threshold_m: float


@dataclass(frozen=True)
class _PlannerRequest:
    points_xy: np.ndarray
    colors: list[str]
    confidences: np.ndarray
    vehicle_xy: tuple[float, float]
    vehicle_yaw: float
    config: SingleBoundaryPlannerConfig
    prior: Optional[SingleBoundaryPlannerPrior]
    track_ids: Optional[np.ndarray]


@dataclass(frozen=True)
class _PreparedPlanning:
    cones: _FilteredCones
    chains: _BoundaryChains
    reject_counts: dict[str, int]
    expected_width_m: float
    pairing: _PairingResult


def _validate_path(
    centerline: np.ndarray,
    centerline_local: np.ndarray,
    near_field: _NearFieldMetrics,
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
    if near_field.lateral_max_m > continuity_threshold_m:
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
    """Compute a local centerline using only one visible boundary."""
    return _compute_single_boundary_centerline(
        _PlannerRequest(
            points_xy=points_xy,
            colors=colors,
            confidences=confidences,
            vehicle_xy=vehicle_xy,
            vehicle_yaw=vehicle_yaw,
            config=config,
            prior=prior,
            track_ids=track_ids,
        )
    )


def _compute_single_boundary_centerline(request: _PlannerRequest) -> SingleBoundaryPlannerResult:
    if request.points_xy.size == 0:
        return _empty_result("no cones available")

    prepared = _prepare_planning(request)
    if isinstance(prepared, SingleBoundaryPlannerResult):
        return prepared

    fallback = _select_fallback_path(prepared.chains, prepared.expected_width_m, request.config)
    if fallback is None:
        return _unreliable_boundary_result(prepared)

    candidate_path = _candidate_path_for_request(prepared, fallback, request)
    centerline, status, reject_reason = _validated_centerline(
        candidate_path, prepared.reject_counts, request.config,
    )

    return _assemble_single_boundary_result(
        prepared=prepared,
        fallback=fallback,
        candidate_path=candidate_path,
        centerline=centerline,
        status=status,
        reject_reason=reject_reason,
    )


def _candidate_path_for_request(
    prepared: _PreparedPlanning,
    fallback: _FallbackSelection,
    request: _PlannerRequest,
) -> _CandidatePath:
    return _candidate_path_metrics(
        raw_curve=fallback.raw_offset_path,
        planner_mode="single_boundary",
        pair_count=len(prepared.pairing.pairs),
        vehicle_xy=request.vehicle_xy,
        vehicle_yaw=request.vehicle_yaw,
        config=request.config,
        prior=request.prior,
    )


def _prepare_planning(
    request: _PlannerRequest,
) -> _PreparedPlanning | SingleBoundaryPlannerResult:
    track_ids = _coerce_track_ids(request.track_ids, request.points_xy.shape[0])
    cones = _filter_and_order_cones(
        points_xy=request.points_xy,
        colors=request.colors,
        confidences=request.confidences,
        track_ids=track_ids,
        vehicle_xy=request.vehicle_xy,
        vehicle_yaw=request.vehicle_yaw,
        config=request.config,
        filtered_cones_type=_FilteredCones,
        geometry_filter=_geometry_filter,
        include_unknown=True,
    )
    input_rejection = _reject_insufficient_colored_cones(cones, request.config)
    if input_rejection is not None:
        return input_rejection

    chains = _make_boundary_chains(cones, request.config)
    reject_counts = _default_reject_counts(_REJECT_COUNT_KEYS)
    expected_width_m = _expected_width_m(request.prior, request.config)
    pairing = _attempt_pairing(
        cones=cones,
        chains=chains,
        expected_width_m=expected_width_m,
        config=request.config,
        prior=request.prior,
    )
    _merge_reject_counts(reject_counts, pairing.reject_counts)
    return _PreparedPlanning(cones, chains, reject_counts, expected_width_m, pairing)


def _coerce_track_ids(track_ids: Optional[np.ndarray], point_count: int) -> np.ndarray:
    if track_ids is None or len(track_ids) != point_count:
        return np.arange(point_count, dtype=np.int64)
    return np.asarray(track_ids, dtype=np.int64)


def _reject_insufficient_colored_cones(
    cones: _FilteredCones,
    config: SingleBoundaryPlannerConfig,
) -> Optional[SingleBoundaryPlannerResult]:
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
    config: SingleBoundaryPlannerConfig,
) -> _BoundaryChains:
    left_indices = np.flatnonzero(np.array([c == "blue" for c in cones.colors], dtype=bool))
    right_indices = np.flatnonzero(np.array([c == "yellow" for c in cones.colors], dtype=bool))
    unknown_indices = np.flatnonzero(np.array([c == "unknown" for c in cones.colors], dtype=bool))
    return _BoundaryChains(
        left=_build_boundary_chain(
            filtered_points=cones.points,
            filtered_local=cones.local,
            side_indices=left_indices,
            config=config,
            boundary_chain_type=_BoundaryChain,
        ),
        right=_build_boundary_chain(
            filtered_points=cones.points,
            filtered_local=cones.local,
            side_indices=right_indices,
            config=config,
            boundary_chain_type=_BoundaryChain,
        ),
        unknown_indices=unknown_indices,
    )


def _expected_width_m(
    prior: Optional[SingleBoundaryPlannerPrior],
    config: SingleBoundaryPlannerConfig,
) -> float:
    prior_width = (
        prior.previous_width_m
        if prior is not None and prior.previous_width_m is not None
        else config.initial_width_m
    )
    return _clamp(prior_width, config.min_width_m, config.max_width_m)


def _attempt_pairing(
    *,
    cones: _FilteredCones,
    chains: _BoundaryChains,
    expected_width_m: float,
    config: SingleBoundaryPlannerConfig,
    prior: Optional[SingleBoundaryPlannerPrior],
) -> _PairingResult:
    if not _pairing_possible(chains, config):
        return _empty_pairing_result()
    pairs, candidate_count, _unknown_pair_count, reject_counts = _pair_boundary_chains(
        cones=cones,
        chains=chains,
        expected_width_m=expected_width_m,
        config=config,
        prior=prior,
    )
    measured_width_m = float(np.median([pair.width_m for pair in pairs])) if pairs else float("nan")
    return _PairingResult(
        pairs=pairs,
        candidate_count=candidate_count,
        unknown_pair_count=0,
        measured_width_m=measured_width_m,
        reject_counts=reject_counts,
    )


def _pairing_possible(
    chains: _BoundaryChains,
    config: SingleBoundaryPlannerConfig,
) -> bool:
    has_chain = (
        chains.left.filtered_indices.size >= int(config.min_chain_length)
        or chains.right.filtered_indices.size >= int(config.min_chain_length)
    )
    has_partner_pool = (
        chains.right.filtered_indices.size > 0
        or chains.left.filtered_indices.size > 0
        or chains.unknown_indices.size > 0
    )
    return bool(has_chain and has_partner_pool)


def _empty_pairing_result() -> _PairingResult:
    return _PairingResult(
        pairs=[],
        candidate_count=0,
        unknown_pair_count=0,
        measured_width_m=float("nan"),
        reject_counts=_default_reject_counts(_REJECT_COUNT_KEYS),
    )


def _select_fallback_path(
    chains: _BoundaryChains,
    expected_width_m: float,
    config: SingleBoundaryPlannerConfig,
) -> Optional[_FallbackSelection]:
    fallback_chain, fallback_side = _select_fallback_chain(chains.left, chains.right, config)
    if fallback_chain is None:
        return None
    return _FallbackSelection(
        chain=fallback_chain,
        side=fallback_side,
        raw_offset_path=_offset_boundary_chain(
            chain=fallback_chain,
            side=fallback_side,
            width_m=expected_width_m,
        ),
    )


def _unreliable_boundary_result(prepared: _PreparedPlanning) -> SingleBoundaryPlannerResult:
    return _result_with_metadata(
        result=_empty_result(
            "no reliable boundary chain",
            filtered_points=prepared.cones.points,
            filtered_colors=prepared.cones.colors,
            left_boundary=prepared.chains.left.global_points,
            right_boundary=prepared.chains.right.global_points,
            reject_counts=prepared.reject_counts,
            reject_reason="no reliable boundary chain",
        ),
        left_chain=prepared.chains.left,
        right_chain=prepared.chains.right,
        planner_mode="none",
        filtered_track_width_m=prepared.expected_width_m,
    )


def _candidate_path_metrics(
    *,
    raw_curve: np.ndarray,
    planner_mode: str,
    pair_count: int,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    config: SingleBoundaryPlannerConfig,
    prior: Optional[SingleBoundaryPlannerPrior],
) -> _CandidatePath:
    centerline = _finalize_path(raw_curve, config)
    centerline_local = _to_vehicle_frame(centerline, vehicle_xy, vehicle_yaw)
    near_field = _near_field_delta_metrics(
        current=centerline,
        previous=None if prior is None else prior.previous_centerline,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        horizon_m=config.jump_check_horizon_m,
    )
    return _CandidatePath(
        centerline=centerline,
        centerline_local=centerline_local,
        seed_distance_m=_first_point_distance(centerline_local),
        near_field=near_field,
        heading_delta_max=_path_heading_delta_max(centerline_local),
        continuity_threshold_m=_continuity_threshold_m(planner_mode, pair_count, config),
    )


def _continuity_threshold_m(
    planner_mode: str,
    pair_count: int,
    config: SingleBoundaryPlannerConfig,
) -> float:
    threshold_m = float(config.max_near_field_lateral_jump_m)
    if planner_mode == "single_boundary":
        return max(threshold_m, float(config.max_near_field_lateral_jump_m_single_boundary))
    sparse_pair_count = max(_MIN_SPARSE_PAIR_CONTINUITY_COUNT, int(config.min_pair_count))
    if pair_count <= sparse_pair_count:
        return max(threshold_m, float(config.max_near_field_lateral_jump_m_sparse_pairs))
    return threshold_m


def _validated_centerline(
    candidate_path: _CandidatePath,
    reject_counts: dict[str, int],
    config: SingleBoundaryPlannerConfig,
) -> tuple[np.ndarray, str, str]:
    reject_reason = _validate_path(
        candidate_path.centerline,
        candidate_path.centerline_local,
        candidate_path.near_field,
        candidate_path.heading_delta_max,
        candidate_path.continuity_threshold_m,
        reject_counts,
        config,
    )
    status = reject_reason if reject_reason else "ok"
    centerline = (
        np.empty((0, 2), dtype=np.float64)
        if status != "ok"
        else candidate_path.centerline
    )
    return centerline, status, reject_reason


def _assemble_single_boundary_result(
    *,
    prepared: _PreparedPlanning,
    fallback: _FallbackSelection,
    candidate_path: _CandidatePath,
    centerline: np.ndarray,
    status: str,
    reject_reason: str,
) -> SingleBoundaryPlannerResult:
    result = _base_single_boundary_result(
        prepared=prepared,
        centerline=centerline,
        status=status,
        reject_reason=reject_reason,
    )
    _apply_single_boundary_path_metadata(result, fallback, candidate_path)
    _apply_single_boundary_pair_metadata(result, prepared)
    return result


def _base_single_boundary_result(
    *,
    prepared: _PreparedPlanning,
    centerline: np.ndarray,
    status: str,
    reject_reason: str,
) -> SingleBoundaryPlannerResult:
    return SingleBoundaryPlannerResult(
        filtered_points=prepared.cones.points,
        filtered_colors=prepared.cones.colors,
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.empty((0, 2), dtype=np.int64),
        selected_pair_track_ids=np.empty((0, 2), dtype=np.int64),
        midpoints_raw=np.empty((0, 2), dtype=np.float64),
        centerline=centerline,
        left_boundary=prepared.chains.left.global_points,
        right_boundary=prepared.chains.right.global_points,
        used_fallback=True,
        status=status,
        reject_reason=reject_reason,
        reject_counts=prepared.reject_counts,
    )


def _apply_single_boundary_path_metadata(
    result: SingleBoundaryPlannerResult,
    fallback: _FallbackSelection,
    candidate_path: _CandidatePath,
) -> None:
    result.selected_chain_length = int(fallback.chain.filtered_indices.size)
    result.near_field_lateral_max_m = float(candidate_path.near_field.lateral_max_m)
    result.near_field_lateral_mean_m = float(candidate_path.near_field.lateral_mean_m)
    result.near_field_displacement_max_m = float(
        candidate_path.near_field.displacement_max_m
    )
    result.near_field_displacement_mean_m = float(
        candidate_path.near_field.displacement_mean_m
    )
    result.near_field_kink_max_rad = float(candidate_path.heading_delta_max)
    result.seed_midpoint_distance_m = float(candidate_path.seed_distance_m)
    result.seed_temporal_offset_m = float("nan")
    result.planner_mode = "single_boundary"
    result.active_boundary_side = fallback.side
    result.raw_offset_path = fallback.raw_offset_path


def _apply_single_boundary_pair_metadata(
    result: SingleBoundaryPlannerResult,
    prepared: _PreparedPlanning,
) -> None:
    result.candidate_count = int(prepared.pairing.candidate_count)
    result.selected_chain_width_median = float(prepared.pairing.measured_width_m)
    result.expected_width_prior_m = float(prepared.expected_width_m)
    result.pair_segments = _pair_segments(prepared.pairing.pairs)
    result.accepted_pair_count = len(prepared.pairing.pairs)
    result.left_chain_length = int(prepared.chains.left.filtered_indices.size)
    result.right_chain_length = int(prepared.chains.right.filtered_indices.size)
    result.filtered_track_width_m = float(prepared.expected_width_m)
    result.unknown_pair_count = int(prepared.pairing.unknown_pair_count)


def _pair_segments(pairs: list[_BoundaryPair]) -> np.ndarray:
    if not pairs:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.asarray(
        [[pair.left_global, pair.right_global] for pair in pairs],
        dtype=np.float64,
    )


def _geometry_filter(local_points: np.ndarray, config: SingleBoundaryPlannerConfig) -> np.ndarray:
    distance = np.hypot(local_points[:, 0], local_points[:, 1])
    return (
        np.isfinite(local_points[:, 0])
        & np.isfinite(local_points[:, 1])
        & (distance <= float(config.max_cone_range_m))
        & (local_points[:, 0] >= -float(config.behind_drop_m))
    )


def _pair_boundary_chains(
    *,
    cones: _FilteredCones,
    chains: _BoundaryChains,
    expected_width_m: float,
    config: SingleBoundaryPlannerConfig,
    prior: Optional[SingleBoundaryPlannerPrior],
) -> tuple[list[_BoundaryPair], int, int, dict[str, int]]:
    reject_counts = _default_reject_counts(_REJECT_COUNT_KEYS)
    anchor = _select_pairing_anchor(chains.left, chains.right)
    state = _PairingState()
    previous_partner_by_anchor = _previous_partner_by_anchor(prior, anchor.anchor_side)
    search = _PairingSearch(
        anchor=anchor,
        state=state,
        previous_partner_by_anchor=previous_partner_by_anchor,
        cones=cones,
        unknown_indices=chains.unknown_indices,
        expected_width_m=expected_width_m,
        reject_counts=reject_counts,
        config=config,
    )

    for anchor_pos in range(anchor.anchor_chain.filtered_indices.size):
        _process_pairing_anchor(anchor_pos, search)

    return state.pairs, state.candidate_count, state.unknown_pair_count, reject_counts


def _process_pairing_anchor(anchor_pos: int, search: _PairingSearch) -> None:
    context = _anchor_context(search.anchor, anchor_pos, search.cones.track_ids)
    options = _candidate_partner_options(
        context=context,
        anchor=search.anchor,
        state=search.state,
        filtered_points=search.cones.points,
        filtered_local=search.cones.local,
        filtered_track_ids=search.cones.track_ids,
        unknown_indices=search.unknown_indices,
        expected_width_m=search.expected_width_m,
        reject_counts=search.reject_counts,
        config=search.config,
    )
    chosen = _choose_partner_option(
        options=options,
        anchor_track_id=context.track_id,
        previous_partner_by_anchor=search.previous_partner_by_anchor,
        state=search.state,
        config=search.config,
    )
    if chosen is None:
        return
    _accept_partner_option(
        chosen=chosen,
        context=context,
        anchor=search.anchor,
        state=search.state,
        filtered_local=search.cones.local,
        reject_counts=search.reject_counts,
        config=search.config,
    )


def _select_pairing_anchor(
    left_chain: _BoundaryChain,
    right_chain: _BoundaryChain,
) -> _PairingAnchor:
    if left_chain.filtered_indices.size >= right_chain.filtered_indices.size:
        return _PairingAnchor(left_chain, right_chain, "blue")
    if right_chain.filtered_indices.size > 0:
        return _PairingAnchor(right_chain, left_chain, "yellow")
    if left_chain.filtered_indices.size > 0:
        return _PairingAnchor(left_chain, right_chain, "blue")
    return _PairingAnchor(right_chain, left_chain, "yellow")


def _previous_partner_by_anchor(
    prior: Optional[SingleBoundaryPlannerPrior],
    anchor_side: str,
) -> dict[int, int]:
    previous_pairs = list(prior.previous_pairs) if prior is not None else []
    previous_partner_by_anchor: dict[int, int] = {}
    for left_track_id, right_track_id in previous_pairs:
        if anchor_side == "blue":
            previous_partner_by_anchor[int(left_track_id)] = int(right_track_id)
        else:
            previous_partner_by_anchor[int(right_track_id)] = int(left_track_id)
    return previous_partner_by_anchor


def _anchor_context(
    anchor: _PairingAnchor,
    anchor_pos: int,
    filtered_track_ids: np.ndarray,
) -> _AnchorContext:
    filtered_idx = int(anchor.anchor_chain.filtered_indices[anchor_pos])
    tangent = anchor.anchor_chain.tangents_local[anchor_pos]
    return _AnchorContext(
        local=anchor.anchor_chain.local_points[anchor_pos],
        global_point=anchor.anchor_chain.global_points[anchor_pos],
        tangent=tangent,
        filtered_idx=filtered_idx,
        track_id=int(filtered_track_ids[filtered_idx]),
        inward_normal=_inward_normal(tangent, anchor.anchor_side),
    )


def _candidate_partner_options(
    *,
    context: _AnchorContext,
    anchor: _PairingAnchor,
    state: _PairingState,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    unknown_indices: np.ndarray,
    expected_width_m: float,
    reject_counts: dict[str, int],
    config: SingleBoundaryPlannerConfig,
) -> list[_PartnerOption]:
    options = _real_partner_options(
        context, anchor, state, filtered_track_ids,
        expected_width_m, reject_counts, config,
    )
    options.extend(
        _unknown_partner_options(
            context, state, filtered_points, filtered_local,
            filtered_track_ids, unknown_indices, expected_width_m, config,
        )
    )
    return options


def _real_partner_options(
    context: _AnchorContext,
    anchor: _PairingAnchor,
    state: _PairingState,
    filtered_track_ids: np.ndarray,
    expected_width_m: float,
    reject_counts: dict[str, int],
    config: SingleBoundaryPlannerConfig,
) -> list[_PartnerOption]:
    options: list[_PartnerOption] = []
    for other_pos in range(state.next_other_start, anchor.other_chain.filtered_indices.size):
        option, reject_key = _real_partner_option(
            context=context,
            other_chain=anchor.other_chain,
            other_pos=other_pos,
            filtered_track_ids=filtered_track_ids,
            expected_width_m=expected_width_m,
            last_partner_progress=state.last_partner_progress,
            config=config,
        )
        if option is None:
            if reject_key:
                reject_counts[reject_key] += 1
            continue
        state.candidate_count += 1
        options.append(option)
    return options


def _unknown_partner_options(
    context: _AnchorContext,
    state: _PairingState,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    unknown_indices: np.ndarray,
    expected_width_m: float,
    config: SingleBoundaryPlannerConfig,
) -> list[_PartnerOption]:
    if not config.allow_unknown_pair_completion or unknown_indices.size == 0:
        return []
    options: list[_PartnerOption] = []
    expected_partner_local = context.local + (context.inward_normal * float(expected_width_m))
    for filtered_idx in unknown_indices:
        option = _unknown_option_for_index(
            context, state, filtered_points, filtered_local,
            filtered_track_ids, int(filtered_idx), expected_partner_local,
            expected_width_m, config,
        )
        if option is None:
            continue
        state.candidate_count += 1
        options.append(option)
    return options


def _unknown_option_for_index(
    context: _AnchorContext,
    state: _PairingState,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    unknown_idx: int,
    expected_partner_local: np.ndarray,
    expected_width_m: float,
    config: SingleBoundaryPlannerConfig,
) -> Optional[_PartnerOption]:
    if unknown_idx in state.used_unknown_indices:
        return None
    option, _reject_reason = _unknown_partner_option(
        context=context,
        filtered_points=filtered_points,
        filtered_local=filtered_local,
        filtered_track_ids=filtered_track_ids,
        unknown_idx=unknown_idx,
        expected_partner_local=expected_partner_local,
        expected_width_m=expected_width_m,
        last_partner_progress=state.last_partner_progress,
        config=config,
    )
    return option


def _choose_partner_option(
    *,
    options: list[_PartnerOption],
    anchor_track_id: int,
    previous_partner_by_anchor: dict[int, int],
    state: _PairingState,
    config: SingleBoundaryPlannerConfig,
) -> Optional[_PartnerOption]:
    if not options:
        return None
    options.sort(key=lambda option: option.sort_key)
    chosen = _prefer_previous_partner_option(
        options=options,
        current_option=options[0],
        preferred_partner_track_id=previous_partner_by_anchor.get(anchor_track_id),
        reassignment_margin=config.pair_reassignment_margin,
    )
    if not _can_accept_unknown_option(chosen, options, state, config):
        return None
    return _known_option_when_unknown_limit_reached(chosen, options, state, config)


def _can_accept_unknown_option(
    chosen: _PartnerOption,
    options: list[_PartnerOption],
    state: _PairingState,
    config: SingleBoundaryPlannerConfig,
) -> bool:
    if not chosen.use_unknown:
        return True
    if state.consecutive_unknown_pairs < int(config.max_consecutive_unknown_pairs):
        return True
    return any(not option.use_unknown for option in options)


def _known_option_when_unknown_limit_reached(
    chosen: _PartnerOption,
    options: list[_PartnerOption],
    state: _PairingState,
    config: SingleBoundaryPlannerConfig,
) -> _PartnerOption:
    if not chosen.use_unknown:
        return chosen
    if state.consecutive_unknown_pairs < int(config.max_consecutive_unknown_pairs):
        return chosen
    return next(option for option in options if not option.use_unknown)


def _accept_partner_option(
    *,
    chosen: _PartnerOption,
    context: _AnchorContext,
    anchor: _PairingAnchor,
    state: _PairingState,
    filtered_local: np.ndarray,
    reject_counts: dict[str, int],
    config: SingleBoundaryPlannerConfig,
) -> None:
    candidate_width = float(chosen.width_m)
    if width_jump_exceeds(state.last_width, candidate_width, config.max_width_jump_m):
        reject_counts["width"] += 1
        return
    state.pairs.append(
        _boundary_pair_from_option(
            chosen=chosen,
            anchor_side=anchor.anchor_side,
            anchor_filtered_idx=context.filtered_idx,
            anchor_track_id=context.track_id,
            anchor_global=context.global_point,
            anchor_local=context.local,
            width_m=candidate_width,
        )
    )
    _update_pairing_state(chosen, state, filtered_local, candidate_width)


def _update_pairing_state(
    chosen: _PartnerOption,
    state: _PairingState,
    filtered_local: np.ndarray,
    candidate_width_m: float,
) -> None:
    if chosen.use_unknown:
        state.used_unknown_indices.add(int(chosen.partner_filtered_idx))
        state.unknown_pair_count += 1
        state.consecutive_unknown_pairs += 1
        state.last_partner_progress = float(filtered_local[int(chosen.partner_filtered_idx), 0])
    else:
        state.next_other_start = int(chosen.other_pos) + 1
        state.consecutive_unknown_pairs = 0
        state.last_partner_progress = float(np.asarray(chosen.partner_local)[0])
    state.last_width = candidate_width_m


def _prefer_previous_partner_option(
    *,
    options: list[_PartnerOption],
    current_option: _PartnerOption,
    preferred_partner_track_id: Optional[int],
    reassignment_margin: float,
) -> _PartnerOption:
    if preferred_partner_track_id is None:
        return current_option
    preferred = next(
        (
            option
            for option in options
            if int(option.partner_track_id) == int(preferred_partner_track_id)
        ),
        None,
    )
    if preferred is None:
        return current_option
    if float(preferred.cost) <= (
        float(current_option.cost) + float(reassignment_margin)
    ):
        return preferred
    return current_option


def _real_partner_option(
    *,
    context: _AnchorContext,
    other_chain: _BoundaryChain,
    other_pos: int,
    filtered_track_ids: np.ndarray,
    expected_width_m: float,
    last_partner_progress: float,
    config: SingleBoundaryPlannerConfig,
) -> tuple[Optional[_PartnerOption], str]:
    other_local = np.asarray(other_chain.local_points[other_pos], dtype=np.float64)
    if float(other_local[0]) < (last_partner_progress - float(config.min_forward_progress_m)):
        return None, "progress"

    delta = other_local - context.local
    width_m = float(np.hypot(delta[0], delta[1]))
    if not pair_width_in_range(width_m, config.min_pair_width_m, config.max_pair_width_m):
        return None, "width_range"

    if inward_distance(delta, context.inward_normal) <= 0.0:
        return None, "wrong_side"

    longitudinal_offset = abs(float(np.dot(delta, context.tangent)))
    width_error = abs(width_m - float(expected_width_m))
    partner_filtered_idx = int(other_chain.filtered_indices[other_pos])
    sort_key = (longitudinal_offset, width_error, width_m, int(other_pos))
    partner_global = np.asarray(other_chain.global_points[other_pos], dtype=np.float64)
    option = _PartnerOption(
        False, int(other_pos), partner_filtered_idx,
        int(filtered_track_ids[partner_filtered_idx]), partner_global,
        other_local, float(width_m), float(longitudinal_offset + width_error), sort_key,
    )
    return option, ""


def _unknown_partner_option(
    *,
    context: _AnchorContext,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    filtered_track_ids: np.ndarray,
    unknown_idx: int,
    expected_partner_local: np.ndarray,
    expected_width_m: float,
    last_partner_progress: float,
    config: SingleBoundaryPlannerConfig,
) -> tuple[Optional[_PartnerOption], str]:
    unknown_local = np.asarray(filtered_local[unknown_idx], dtype=np.float64)
    check, width_m, reject_reason = _unknown_partner_validation(
        context=context,
        unknown_local=unknown_local,
        expected_partner_local=expected_partner_local,
        expected_width_m=expected_width_m,
        last_partner_progress=last_partner_progress,
        config=config,
    )
    if reject_reason:
        return None, reject_reason
    option = _unknown_partner_option_from_check(
        check=check,
        width_m=width_m,
        unknown_idx=unknown_idx,
        unknown_local=unknown_local,
        filtered_points=filtered_points,
        filtered_track_ids=filtered_track_ids,
    )
    return option, ""


def _unknown_partner_validation(
    *,
    context: _AnchorContext,
    unknown_local: np.ndarray,
    expected_partner_local: np.ndarray,
    expected_width_m: float,
    last_partner_progress: float,
    config: SingleBoundaryPlannerConfig,
) -> tuple[Optional[UnknownPartnerCheck], float, str]:
    if float(unknown_local[0]) < (last_partner_progress - float(config.min_forward_progress_m)):
        return None, 0.0, "progress"

    delta = unknown_local - context.local
    width_m = float(np.hypot(delta[0], delta[1]))
    if not pair_width_in_range(width_m, config.min_pair_width_m, config.max_pair_width_m):
        return None, width_m, "width_range"
    if inward_distance(delta, context.inward_normal) <= 0.0:
        return None, width_m, "wrong_side"

    check = unknown_partner_check(
        partner_local=unknown_local,
        expected_partner_local=expected_partner_local,
        anchor_tangent=context.tangent,
        width_m=width_m,
        expected_width_m=expected_width_m,
    )
    if _unknown_partner_exceeds_limits(check, config):
        return None, width_m, "unknown_partner_limits"
    return check, width_m, ""


def _unknown_partner_exceeds_limits(
    check: UnknownPartnerCheck,
    config: SingleBoundaryPlannerConfig,
) -> bool:
    return not unknown_partner_within_limits(
        check,
        max_longitudinal_error_m=config.unknown_pair_max_longitudinal_error_m,
        max_width_error_m=config.unknown_pair_max_width_error_m,
        search_radius_m=config.unknown_pair_search_radius_m,
    )


def _unknown_partner_option_from_check(
    *,
    check: Optional[UnknownPartnerCheck],
    width_m: float,
    unknown_idx: int,
    unknown_local: np.ndarray,
    filtered_points: np.ndarray,
    filtered_track_ids: np.ndarray,
) -> _PartnerOption:
    check = cast(UnknownPartnerCheck, check)
    sort_key = (
        check.longitudinal_error_m,
        check.width_error_m,
        check.radial_error_m,
        int(unknown_idx),
    )
    option = _PartnerOption(
        True, -1, int(unknown_idx), int(filtered_track_ids[unknown_idx]),
        np.asarray(filtered_points[unknown_idx], dtype=np.float64), unknown_local,
        float(width_m), float(check.cost + _UNKNOWN_PARTNER_COST_BIAS_M), sort_key,
    )
    return option


def _boundary_pair_from_option(
    *,
    chosen: _PartnerOption,
    anchor_side: str,
    anchor_filtered_idx: int,
    anchor_track_id: int,
    anchor_global: np.ndarray,
    anchor_local: np.ndarray,
    width_m: float,
) -> _BoundaryPair:
    partner_filtered_idx = int(chosen.partner_filtered_idx)
    partner_track_id = int(chosen.partner_track_id)
    partner_global = np.asarray(chosen.partner_global, dtype=np.float64)
    partner_local = np.asarray(chosen.partner_local, dtype=np.float64)

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
    offset_distance = _TRACK_HALF_WIDTH_SCALE * float(width_m)
    tangents_global = _estimate_tangents(chain.global_points)
    for idx, point in enumerate(chain.global_points):
        normal = _inward_normal(tangents_global[idx], side)
        offset.append(point + (offset_distance * normal))
    return np.asarray(offset, dtype=np.float64)


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
    alignment = _path_alignment_metrics(
        current_local=current_local,
        previous_local=previous_local,
        horizon_m=min(float(horizon_m), _MAX_NEAR_FIELD_ALIGNMENT_HORIZON_M),
    )
    return _NearFieldMetrics(
        lateral_max_m=float(alignment.lateral_max_m),
        lateral_mean_m=float(alignment.lateral_mean_m),
        displacement_max_m=float(alignment.displacement_max_m),
        displacement_mean_m=float(alignment.displacement_mean_m),
    )


def _path_alignment_metrics(
    *,
    current_local: Optional[np.ndarray],
    previous_local: Optional[np.ndarray],
    horizon_m: float,
) -> _PathAlignmentMetrics:
    empty = _PathAlignmentMetrics()
    prefixes = _alignment_prefixes(current_local, previous_local, horizon_m)
    if prefixes is None:
        return empty

    current_prefix, previous_prefix = prefixes
    count = min(current_prefix.shape[0], previous_prefix.shape[0])
    if count < 2:
        return empty
    delta = current_prefix[:count] - previous_prefix[:count]
    lateral = np.abs(delta[:, 1])
    displacement = np.hypot(delta[:, 0], delta[:, 1])
    return _PathAlignmentMetrics(
        lateral_max_m=float(np.max(lateral)) if lateral.size else 0.0,
        lateral_mean_m=float(np.mean(lateral)) if lateral.size else 0.0,
        displacement_max_m=float(np.max(displacement)) if displacement.size else 0.0,
        displacement_mean_m=float(np.mean(displacement)) if displacement.size else 0.0,
        heading_delta_rad=_alignment_heading_delta_rad(current_prefix, previous_prefix, count),
    )


def _alignment_prefixes(
    current_local: Optional[np.ndarray],
    previous_local: Optional[np.ndarray],
    horizon_m: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if current_local is None or previous_local is None:
        return None
    current_prefix = _local_forward_prefix(
        np.asarray(current_local, dtype=np.float64),
        horizon_m=float(horizon_m),
    )
    previous_prefix = _local_forward_prefix(
        np.asarray(previous_local, dtype=np.float64),
        horizon_m=float(horizon_m),
    )
    if current_prefix.shape[0] < 2 or previous_prefix.shape[0] < 2:
        return None
    return current_prefix, previous_prefix


def _alignment_heading_delta_rad(
    current_prefix: np.ndarray,
    previous_prefix: np.ndarray,
    count: int,
) -> float:
    current_heading = _path_start_heading_error(current_prefix[:count])
    previous_heading = _path_start_heading_error(previous_prefix[:count])
    return abs(
        float(
            math.atan2(
                math.sin(current_heading - previous_heading),
                math.cos(current_heading - previous_heading),
            )
        )
    )


def _local_forward_prefix(path_local: np.ndarray, *, horizon_m: float) -> np.ndarray:
    pts = np.asarray(path_local, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)
    valid_mask = (
        np.isfinite(pts[:, 0])
        & np.isfinite(pts[:, 1])
        & (pts[:, 0] >= -_LOCAL_FORWARD_BACKTRACK_MARGIN_M)
    )
    pts = pts[valid_mask]
    if pts.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)
    cumulative = _path_cumulative_lengths(pts)
    total = min(float(cumulative[-1]), max(_MIN_LOCAL_PREFIX_LENGTH_M, float(horizon_m)))
    if total <= _MIN_PATH_LENGTH_M:
        return np.asarray(pts[:1], dtype=np.float64)
    return _resample_path(pts, _MIN_LOCAL_PREFIX_LENGTH_M, total)


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
        **_empty_result_fields(
            filtered_points=filtered_points,
            filtered_colors=filtered_colors,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        ),
        status=status,
        reject_counts=reject_counts or _default_reject_counts(_REJECT_COUNT_KEYS),
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
