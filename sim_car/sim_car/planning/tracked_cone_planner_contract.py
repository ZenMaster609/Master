"""Small public parameter contract and profile builders for tracked-cone planners."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from sim_car.planning.controller_config import build_steering_controller
from sim_car.planning.midline_memory import MidlineMemoryConfig
from sim_car.planning.pipeline_defaults import (
    BASE_FRAME_DEFAULT,
    ODOM_FRAME_DEFAULT,
    ODOM_TOPIC_DEFAULT,
    PLANNER_MAX_RANGE_M_DEFAULT,
    PLANNER_MIN_CONFIDENCE_DEFAULT,
    TF_TIMEOUT_S_DEFAULT,
    TRACKED_CONES_TOPIC_DEFAULT,
    WHEELBASE_M_DEFAULT,
)

PARAM_STATUS_ACTIVE = "active"
PARAM_STATUS_COMPATIBILITY_ONLY = "compatibility_only"

SUPPORTED_TRACKED_CONE_CONTROLLER_TYPES = frozenset({"stanley", "pure_pursuit", "none"})
SUPPORTED_PLANNER_PROFILES = frozenset({"default"})


@dataclass(frozen=True)
class PlannerPublicConfig:
    profile: str = "default"
    max_range_m: float = PLANNER_MAX_RANGE_M_DEFAULT
    min_confidence: float = PLANNER_MIN_CONFIDENCE_DEFAULT

    planning_frame: str = ODOM_FRAME_DEFAULT
    odom_frame: str = ODOM_FRAME_DEFAULT
    base_frame: str = BASE_FRAME_DEFAULT
    tf_timeout_s: float = TF_TIMEOUT_S_DEFAULT
    vehicle_wheelbase_m: float = WHEELBASE_M_DEFAULT

    tracked_cones_topic: str = TRACKED_CONES_TOPIC_DEFAULT
    cmd_topic: str = "/cmd"
    centerline_topic: str = "/planned_centerline"
    viz_topic: str = "/planner_viz"
    points_topic: str = "/planned_centerline_points"
    odom_topic: str = ODOM_TOPIC_DEFAULT
    diagnostics_topic: str = ""

    publish_rate_hz: float = 180.0
    log_throttle_s: float = 1.0

    controller_type: str = "stanley"
    odom_lag_compensation_ms: float = 0.0
    stop_if_no_path: bool = True

    speed_min_mps: float = 1.0
    speed_max_mps: float = 4.0
    curvature_speed_gain: float = 4.0
    lowpass_speed_alpha: float = 0.15

    stanley_k_gain: float = 1.2
    stanley_heading_gain: float = 1.6
    stanley_lookahead_idx_offset: int = 0
    pure_pursuit_lookahead_m: float = 3.0
    pure_pursuit_min_lookahead_m: float = 1.5
    pure_pursuit_max_lookahead_m: float = 8.0

    enable_debug_markers: bool = True
    publish_points_topic: bool = False
    show_lookahead_point: bool = True
    lap_tracking_target_laps: int = 0


PUBLIC_TRACKED_CONE_PLANNER_DEFAULTS: dict[str, object] = {
    "planner.profile": "default",
    "planner.max_range_m": PLANNER_MAX_RANGE_M_DEFAULT,
    "planner.min_confidence": PLANNER_MIN_CONFIDENCE_DEFAULT,
    "frames.planning_frame": ODOM_FRAME_DEFAULT,
    "frames.odom_frame": ODOM_FRAME_DEFAULT,
    "frames.base_frame": BASE_FRAME_DEFAULT,
    "frames.tf_timeout_s": TF_TIMEOUT_S_DEFAULT,
    "vehicle.wheelbase_m": WHEELBASE_M_DEFAULT,
    "topics.tracked_cones_topic": TRACKED_CONES_TOPIC_DEFAULT,
    "topics.cmd_topic": "/cmd",
    "topics.centerline_topic": "/planned_centerline",
    "topics.viz_topic": "/planner_viz",
    "topics.points_topic": "/planned_centerline_points",
    "topics.odom_topic": ODOM_TOPIC_DEFAULT,
    "topics.diagnostics_topic": "",
    "runtime.publish_rate_hz": 180.0,
    "runtime.log_throttle_s": 1.0,
    "control.controller_type": "stanley",
    "control.odom_lag_compensation_ms": 0.0,
    "control.stop_if_no_path": True,
    "speed_control.speed_min_mps": 1.0,
    "speed_control.speed_max_mps": 4.0,
    "speed_control.curvature_speed_gain": 4.0,
    "speed_control.lowpass_speed_alpha": 0.15,
    "stanley.k_gain": 1.2,
    "stanley.heading_gain": 1.6,
    "stanley.lookahead_idx_offset": 0,
    "pure_pursuit.lookahead_m": 3.0,
    "pure_pursuit.min_lookahead_m": 1.5,
    "pure_pursuit.max_lookahead_m": 8.0,
    "debug.enable_markers": True,
    "debug.publish_points_topic": False,
    "debug.show_lookahead_point": True,
    "lap_tracking.target_laps": 0,
}

# Kept as an import alias during the refactor; it is no longer the public ROS
# surface. New code should use PUBLIC_TRACKED_CONE_PLANNER_DEFAULTS.
COMMON_MIGRATED_TRACKED_CONE_PLANNER_DEFAULTS = PUBLIC_TRACKED_CONE_PLANNER_DEFAULTS


@dataclass(frozen=True)
class PlannerAlgorithmProfile:
    max_cone_range_m: float = PLANNER_MAX_RANGE_M_DEFAULT
    min_confidence: float = PLANNER_MIN_CONFIDENCE_DEFAULT
    behind_drop_m: float = 5.0
    min_required_cones: int = 4
    centerline_path_resolution_m: float = 0.5
    max_path_length_m: float = 30.0
    jump_check_horizon_m: float = 8.0

    infer_unknown_by_side: bool = True
    infer_orange_by_side: bool = True
    orange_min_lateral_m: float = 0.9
    orange_neighbor_radius_m: float = 3.5
    orange_neighbor_margin_m: float = 0.75

    boundary_min_step_m: float = 0.8
    boundary_max_step_m: float = 10.5
    boundary_max_heading_change_rad: float = 0.95
    boundary_min_forward_progress_m: float = 0.0002

    initial_width_m: float = 3.6
    min_width_m: float = 2.4
    max_width_m: float = 4.8
    width_filter_alpha: float = 0.18
    max_width_delta_per_update_m: float = 0.2

    validation_hold_last_valid_s: float = 2.5
    validation_hold_exit_clean_frames: int = 2
    candidate_jump_reject_threshold_m: float = 0.45
    candidate_jump_recover_frames: int = 3
    candidate_min_points: int = 4
    candidate_min_extent_m: float = 2.0
    centerline_jump_horizon_m: float = 8.0
    publish_control_debug: bool = True

    show_raw_cones: bool = False
    show_boundary_chains: bool = False
    show_pair_lines: bool = False
    show_raw_midpoint_chain: bool = False
    show_raw_prevalidation_centerline: bool = False
    show_candidate_edges: bool = False
    show_selected_edges: bool = False

    midline_memory: MidlineMemoryConfig = MidlineMemoryConfig()


DEFAULT_ALGORITHM_PROFILE = PlannerAlgorithmProfile()
PROFILE_BY_NAME = {"default": DEFAULT_ALGORITHM_PROFILE}


@dataclass(frozen=True)
class MigratedTrackedConePlannerCommonConfig:
    planning_frame: str
    odom_frame: str
    base_frame: str
    tf_timeout_s: float
    vehicle_wheelbase_m: float
    tracked_cones_topic: str
    cmd_topic: str
    centerline_topic: str
    viz_topic: str
    points_topic: str
    odom_topic: str
    infer_unknown_by_side: bool
    infer_orange_by_side: bool
    orange_min_lateral_m: float
    orange_neighbor_radius_m: float
    orange_neighbor_margin_m: float
    centerline_path_resolution_m: float
    temporal_alpha: float
    enable_temporal_smoothing: bool
    smoothing_alpha: float
    enable_near_field_freeze: bool
    freeze_near_field_m: float
    freeze_blend_length_m: float
    enable_committed_near_field: bool
    commit_plan_horizon_m: float
    commit_stable_frames: int
    commit_update_max_churn_ratio: float
    midline_horizon_m: float
    midline_station_spacing_m: float
    midline_near_distance_m: float
    midline_mid_distance_m: float
    midline_control_handoff_distance_m: float
    midline_near_alpha: float
    midline_mid_alpha: float
    midline_far_alpha: float
    midline_near_max_shift_m: float
    midline_mid_max_shift_m: float
    midline_far_max_shift_m: float
    midline_min_buffer_confidence: float
    midline_hold_last_valid_duration_s: float
    midline_min_estimated_extent_m: float
    midline_max_estimation_extension_m: float
    midline_max_estimation_join_lateral_m: float
    midline_max_estimation_join_heading_rad: float
    midline_allow_tangent_estimate_without_memory: bool
    midline_max_tangent_estimation_extension_m: float
    publish_rate_hz: float
    log_throttle_s: float
    controller_type: str
    odom_lag_compensation_s: float
    _controller: Any
    stop_if_no_path: bool
    speed_min_mps: float
    speed_max_mps: float
    curvature_speed_gain: float
    lowpass_speed_alpha: float
    hold_last_valid_s: float
    hold_exit_clean_frames: int
    candidate_jump_reject_threshold_m: float
    candidate_jump_recover_frames: int
    candidate_min_points: int
    candidate_min_extent_m: float
    diagnostics_topic: str
    centerline_jump_horizon_m: float
    publish_control_debug: bool
    enable_debug_markers: bool
    show_raw_cones: bool
    show_boundary_chains: bool
    show_pair_lines: bool
    show_raw_midpoint_chain: bool
    show_raw_prevalidation_centerline: bool
    publish_points_topic: bool
    show_lookahead_point: bool
    show_candidate_edges: bool
    show_selected_edges: bool
    lap_tracking_target_laps: int


def declare_tracked_cone_planner_parameters(node: Any, *, diagnostics_topic_default: str) -> None:
    for name, value in PUBLIC_TRACKED_CONE_PLANNER_DEFAULTS.items():
        default = diagnostics_topic_default if name == "topics.diagnostics_topic" else value
        node.declare_parameter(name, default)


def normalize_tracked_cone_controller_type(controller_type: str) -> str:
    normalized = str(controller_type).strip().lower() or "stanley"
    if normalized not in SUPPORTED_TRACKED_CONE_CONTROLLER_TYPES:
        raise ValueError(
            "Unsupported control.controller_type '%s'. Supported values: stanley, pure_pursuit, none"
            % normalized
        )
    return normalized


def _normalize_profile(profile: str) -> str:
    normalized = str(profile).strip().lower() or "default"
    if normalized not in SUPPORTED_PLANNER_PROFILES:
        raise ValueError("Unsupported planner.profile '%s'. Supported value: default" % normalized)
    return normalized


def _read_public_config(node: Any, *, diagnostics_topic_fallback: str) -> PlannerPublicConfig:
    profile = _normalize_profile(node.get_parameter("planner.profile").value)
    diagnostics_topic = (
        str(node.get_parameter("topics.diagnostics_topic").value).strip()
        or diagnostics_topic_fallback
    )
    return PlannerPublicConfig(
        profile=profile,
        max_range_m=max(1.0, float(node.get_parameter("planner.max_range_m").value)),
        min_confidence=max(0.0, min(1.0, float(node.get_parameter("planner.min_confidence").value))),
        planning_frame=str(node.get_parameter("frames.planning_frame").value).strip() or ODOM_FRAME_DEFAULT,
        odom_frame=str(node.get_parameter("frames.odom_frame").value).strip() or ODOM_FRAME_DEFAULT,
        base_frame=str(node.get_parameter("frames.base_frame").value).strip() or BASE_FRAME_DEFAULT,
        tf_timeout_s=max(0.0, float(node.get_parameter("frames.tf_timeout_s").value)),
        vehicle_wheelbase_m=max(0.1, float(node.get_parameter("vehicle.wheelbase_m").value)),
        tracked_cones_topic=str(node.get_parameter("topics.tracked_cones_topic").value).strip(),
        cmd_topic=str(node.get_parameter("topics.cmd_topic").value).strip(),
        centerline_topic=str(node.get_parameter("topics.centerline_topic").value).strip(),
        viz_topic=str(node.get_parameter("topics.viz_topic").value).strip(),
        points_topic=str(node.get_parameter("topics.points_topic").value).strip(),
        odom_topic=str(node.get_parameter("topics.odom_topic").value).strip(),
        diagnostics_topic=diagnostics_topic,
        publish_rate_hz=max(1.0, float(node.get_parameter("runtime.publish_rate_hz").value)),
        log_throttle_s=max(0.1, float(node.get_parameter("runtime.log_throttle_s").value)),
        controller_type=normalize_tracked_cone_controller_type(
            node.get_parameter("control.controller_type").value
        ),
        odom_lag_compensation_ms=min(
            max(0.0, float(node.get_parameter("control.odom_lag_compensation_ms").value)),
            150.0,
        ),
        stop_if_no_path=bool(node.get_parameter("control.stop_if_no_path").value),
        speed_min_mps=max(0.0, float(node.get_parameter("speed_control.speed_min_mps").value)),
        speed_max_mps=float(node.get_parameter("speed_control.speed_max_mps").value),
        curvature_speed_gain=max(
            0.0,
            float(node.get_parameter("speed_control.curvature_speed_gain").value),
        ),
        lowpass_speed_alpha=max(
            0.0,
            min(1.0, float(node.get_parameter("speed_control.lowpass_speed_alpha").value)),
        ),
        stanley_k_gain=max(0.0, float(node.get_parameter("stanley.k_gain").value)),
        stanley_heading_gain=float(node.get_parameter("stanley.heading_gain").value),
        stanley_lookahead_idx_offset=max(0, int(node.get_parameter("stanley.lookahead_idx_offset").value)),
        pure_pursuit_lookahead_m=max(0.0, float(node.get_parameter("pure_pursuit.lookahead_m").value)),
        pure_pursuit_min_lookahead_m=max(
            0.01,
            float(node.get_parameter("pure_pursuit.min_lookahead_m").value),
        ),
        pure_pursuit_max_lookahead_m=max(
            0.01,
            float(node.get_parameter("pure_pursuit.max_lookahead_m").value),
        ),
        enable_debug_markers=bool(node.get_parameter("debug.enable_markers").value),
        publish_points_topic=bool(node.get_parameter("debug.publish_points_topic").value),
        show_lookahead_point=bool(node.get_parameter("debug.show_lookahead_point").value),
        lap_tracking_target_laps=max(0, int(node.get_parameter("lap_tracking.target_laps").value)),
    )


def planner_algorithm_profile(public: PlannerPublicConfig) -> PlannerAlgorithmProfile:
    base = PROFILE_BY_NAME[public.profile]
    return PlannerAlgorithmProfile(
        max_cone_range_m=public.max_range_m,
        min_confidence=public.min_confidence,
        behind_drop_m=base.behind_drop_m,
        min_required_cones=base.min_required_cones,
        centerline_path_resolution_m=base.centerline_path_resolution_m,
        max_path_length_m=public.max_range_m,
        jump_check_horizon_m=base.jump_check_horizon_m,
        infer_unknown_by_side=base.infer_unknown_by_side,
        infer_orange_by_side=base.infer_orange_by_side,
        orange_min_lateral_m=base.orange_min_lateral_m,
        orange_neighbor_radius_m=base.orange_neighbor_radius_m,
        orange_neighbor_margin_m=base.orange_neighbor_margin_m,
        boundary_min_step_m=base.boundary_min_step_m,
        boundary_max_step_m=base.boundary_max_step_m,
        boundary_max_heading_change_rad=base.boundary_max_heading_change_rad,
        boundary_min_forward_progress_m=base.boundary_min_forward_progress_m,
        initial_width_m=base.initial_width_m,
        min_width_m=base.min_width_m,
        max_width_m=base.max_width_m,
        width_filter_alpha=base.width_filter_alpha,
        max_width_delta_per_update_m=base.max_width_delta_per_update_m,
        validation_hold_last_valid_s=base.validation_hold_last_valid_s,
        validation_hold_exit_clean_frames=base.validation_hold_exit_clean_frames,
        candidate_jump_reject_threshold_m=base.candidate_jump_reject_threshold_m,
        candidate_jump_recover_frames=base.candidate_jump_recover_frames,
        candidate_min_points=base.candidate_min_points,
        candidate_min_extent_m=base.candidate_min_extent_m,
        centerline_jump_horizon_m=base.centerline_jump_horizon_m,
        publish_control_debug=base.publish_control_debug,
        show_raw_cones=base.show_raw_cones,
        show_boundary_chains=base.show_boundary_chains,
        show_pair_lines=base.show_pair_lines,
        show_raw_midpoint_chain=base.show_raw_midpoint_chain,
        show_raw_prevalidation_centerline=base.show_raw_prevalidation_centerline,
        show_candidate_edges=base.show_candidate_edges,
        show_selected_edges=base.show_selected_edges,
        midline_memory=base.midline_memory,
    )


def build_tracked_cone_controller(*, node: Any, controller_type: str, publish_rate_hz: float):
    normalized = normalize_tracked_cone_controller_type(controller_type)
    if normalized == "none":
        return None
    return build_steering_controller(
        node=node,
        controller_type=normalized,
        publish_rate_hz=publish_rate_hz,
    )


def log_tracked_cone_controller_mode(node: Any, *, controller_type: str) -> None:
    if controller_type == "none":
        node.get_logger().info("control.controller_type 'none'; controller output is disabled")


def emit_migrated_planner_compatibility_warnings(
    node: Any, *, planner_label: str, temporal_alpha: float
) -> None:
    del node
    del planner_label
    del temporal_alpha


def _expose_controller_overrides(node: Any, public: PlannerPublicConfig) -> None:
    node.vehicle_wheelbase_m = public.vehicle_wheelbase_m
    node.stanley_k_gain = public.stanley_k_gain
    node.stanley_heading_gain = public.stanley_heading_gain
    node.stanley_lookahead_idx_offset = public.stanley_lookahead_idx_offset
    node.pure_pursuit_lookahead_m = public.pure_pursuit_lookahead_m
    node.pure_pursuit_min_lookahead_m = public.pure_pursuit_min_lookahead_m
    node.pure_pursuit_max_lookahead_m = public.pure_pursuit_max_lookahead_m


def read_migrated_tracked_cone_planner_common_config(
    node: Any,
    *,
    planner_label: str,
    diagnostics_topic_fallback: str,
) -> MigratedTrackedConePlannerCommonConfig:
    del planner_label
    public = _read_public_config(node, diagnostics_topic_fallback=diagnostics_topic_fallback)
    _expose_controller_overrides(node, public)
    profile = planner_algorithm_profile(public)
    node._planner_public_config = public
    node._planner_algorithm_profile = profile
    speed_max = max(public.speed_min_mps, public.speed_max_mps)
    midline = profile.midline_memory
    hold_last_valid_s = max(
        midline.hold_last_valid_duration_s,
        profile.validation_hold_last_valid_s,
    )

    common = MigratedTrackedConePlannerCommonConfig(
        planning_frame=public.planning_frame,
        odom_frame=public.odom_frame,
        base_frame=public.base_frame,
        tf_timeout_s=public.tf_timeout_s,
        vehicle_wheelbase_m=public.vehicle_wheelbase_m,
        tracked_cones_topic=public.tracked_cones_topic,
        cmd_topic=public.cmd_topic,
        centerline_topic=public.centerline_topic,
        viz_topic=public.viz_topic,
        points_topic=public.points_topic,
        odom_topic=public.odom_topic,
        infer_unknown_by_side=profile.infer_unknown_by_side,
        infer_orange_by_side=profile.infer_orange_by_side,
        orange_min_lateral_m=profile.orange_min_lateral_m,
        orange_neighbor_radius_m=profile.orange_neighbor_radius_m,
        orange_neighbor_margin_m=profile.orange_neighbor_margin_m,
        centerline_path_resolution_m=max(0.05, profile.centerline_path_resolution_m),
        temporal_alpha=0.25,
        enable_temporal_smoothing=False,
        smoothing_alpha=0.25,
        enable_near_field_freeze=False,
        freeze_near_field_m=0.0,
        freeze_blend_length_m=0.0,
        enable_committed_near_field=False,
        commit_plan_horizon_m=0.0,
        commit_stable_frames=1,
        commit_update_max_churn_ratio=1.0,
        midline_horizon_m=max(1.0, midline.horizon_m),
        midline_station_spacing_m=max(0.05, midline.station_spacing_m),
        midline_near_distance_m=max(0.0, midline.near_distance_m),
        midline_mid_distance_m=max(midline.near_distance_m, midline.mid_distance_m),
        midline_control_handoff_distance_m=max(
            midline.station_spacing_m,
            getattr(midline, "control_handoff_distance_m", 1.5),
        ),
        midline_near_alpha=max(0.0, min(1.0, midline.near_alpha)),
        midline_mid_alpha=max(0.0, min(1.0, midline.mid_alpha)),
        midline_far_alpha=max(0.0, min(1.0, midline.far_alpha)),
        midline_near_max_shift_m=max(0.0, midline.near_max_lateral_shift_m),
        midline_mid_max_shift_m=max(midline.near_max_lateral_shift_m, midline.mid_max_lateral_shift_m),
        midline_far_max_shift_m=max(midline.mid_max_lateral_shift_m, midline.far_max_lateral_shift_m),
        midline_min_buffer_confidence=max(0.0, min(1.0, midline.min_buffer_confidence)),
        midline_hold_last_valid_duration_s=max(0.0, midline.hold_last_valid_duration_s),
        midline_min_estimated_extent_m=max(0.5, midline.min_estimated_extent_m),
        midline_max_estimation_extension_m=max(0.0, midline.max_estimation_extension_m),
        midline_max_estimation_join_lateral_m=max(0.0, midline.max_estimation_join_lateral_m),
        midline_max_estimation_join_heading_rad=max(0.0, midline.max_estimation_join_heading_rad),
        midline_allow_tangent_estimate_without_memory=bool(
            midline.allow_tangent_estimate_without_memory
        ),
        midline_max_tangent_estimation_extension_m=max(
            0.0,
            midline.max_tangent_estimation_extension_m,
        ),
        publish_rate_hz=public.publish_rate_hz,
        log_throttle_s=public.log_throttle_s,
        controller_type=public.controller_type,
        odom_lag_compensation_s=public.odom_lag_compensation_ms / 1000.0,
        _controller=build_tracked_cone_controller(
            node=node,
            controller_type=public.controller_type,
            publish_rate_hz=public.publish_rate_hz,
        ),
        stop_if_no_path=public.stop_if_no_path,
        speed_min_mps=public.speed_min_mps,
        speed_max_mps=speed_max,
        curvature_speed_gain=public.curvature_speed_gain,
        lowpass_speed_alpha=public.lowpass_speed_alpha,
        hold_last_valid_s=hold_last_valid_s,
        hold_exit_clean_frames=max(1, profile.validation_hold_exit_clean_frames),
        candidate_jump_reject_threshold_m=max(0.0, profile.candidate_jump_reject_threshold_m),
        candidate_jump_recover_frames=max(1, profile.candidate_jump_recover_frames),
        candidate_min_points=max(2, profile.candidate_min_points),
        candidate_min_extent_m=max(0.5, profile.candidate_min_extent_m),
        diagnostics_topic=public.diagnostics_topic,
        centerline_jump_horizon_m=max(0.5, profile.centerline_jump_horizon_m),
        publish_control_debug=profile.publish_control_debug,
        enable_debug_markers=public.enable_debug_markers,
        show_raw_cones=profile.show_raw_cones,
        show_boundary_chains=profile.show_boundary_chains,
        show_pair_lines=profile.show_pair_lines,
        show_raw_midpoint_chain=profile.show_raw_midpoint_chain,
        show_raw_prevalidation_centerline=profile.show_raw_prevalidation_centerline,
        publish_points_topic=public.publish_points_topic,
        show_lookahead_point=public.show_lookahead_point,
        show_candidate_edges=profile.show_candidate_edges,
        show_selected_edges=profile.show_selected_edges,
        lap_tracking_target_laps=public.lap_tracking_target_laps,
    )
    log_tracked_cone_controller_mode(node, controller_type=common.controller_type)
    return common


def apply_common_config_to_node(node: Any, common: MigratedTrackedConePlannerCommonConfig) -> None:
    for field in fields(common):
        setattr(node, field.name, getattr(common, field.name))
