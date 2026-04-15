"""Common configuration schema and factory helpers for tracked-cone planner nodes."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any

from sim_car.planning.controller_config import build_steering_controller

PARAM_STATUS_ACTIVE = 'active'
PARAM_STATUS_COMPATIBILITY_ONLY = 'compatibility_only'

SUPPORTED_TRACKED_CONE_CONTROLLER_TYPES = frozenset({'stanley', 'pure_pursuit', 'none'})

COMMON_MIGRATED_TRACKED_CONE_PLANNER_DEFAULTS: dict[str, object] = {
    'boundary_chain.max_heading_change_rad': 0.95,
    'boundary_chain.max_step_m': 10.5,
    'boundary_chain.min_forward_progress_m': 0.0002,
    'boundary_chain.min_step_m': 0.8,
    'centerline.max_path_length_m': 30.0,
    'centerline.path_resolution_m': 0.5,
    'centerline.temporal_alpha': 0.25,
    'control.controller_type': 'stanley',
    'control.odom_lag_compensation_ms': 0.0,
    'control.stop_if_no_path': True,
    'debug.enable_markers': True,
    'debug.publish_points_topic': False,
    'debug.show_boundary_chains': True,
    'debug.show_lookahead_point': True,
    'debug.show_pair_lines': True,
    'debug.show_raw_cones': True,
    'debug.show_raw_midpoint_chain': True,
    'debug.show_raw_prevalidation_centerline': True,
    'diagnostics.centerline_jump_horizon_m': 8.0,
    'diagnostics.edge_churn_warn_threshold': 0.4,
    'diagnostics.edge_quantization_m': 0.05,
    'diagnostics.jump_warn_threshold_m': 0.8,
    'diagnostics.publish_control_debug': True,
    'diagnostics.publish_thesis_context': False,
    'filtering.behind_drop_m': 5.0,
    'filtering.infer_orange_by_side': True,
    'filtering.infer_unknown_by_side': True,
    'filtering.max_cone_range_m': 25.0,
    'filtering.min_confidence': 0.3,
    'filtering.min_required_cones': 4,
    'filtering.orange_min_lateral_m': 0.9,
    'filtering.orange_neighbor_margin_m': 0.75,
    'filtering.orange_neighbor_radius_m': 3.5,
    'frames.base_frame': 'front_axle',
    'frames.odom_frame': 'odom',
    'frames.planning_frame': 'odom',
    'frames.tf_timeout_s': 0.03,
    'midline_memory.control_handoff_distance_m': 1.5,
    'midline_memory.far_alpha': 0.30,
    'midline_memory.far_max_lateral_shift_m': 0.35,
    'midline_memory.hold_last_valid_duration_s': 3.0,
    'midline_memory.horizon_m': 30.0,
    'midline_memory.mid_alpha': 0.12,
    'midline_memory.mid_distance_m': 12.0,
    'midline_memory.mid_max_lateral_shift_m': 0.18,
    'midline_memory.min_buffer_confidence': 0.2,
    'midline_memory.min_estimated_extent_m': 6.0,
    'midline_memory.near_alpha': 0.04,
    'midline_memory.near_distance_m': 4.0,
    'midline_memory.near_max_lateral_shift_m': 0.07,
    'midline_memory.station_spacing_m': 0.5,
    'midline_memory.max_estimation_extension_m': 4.0,
    'midline_memory.max_estimation_join_lateral_m': 0.5,
    'midline_memory.max_estimation_join_heading_rad': 0.45,
    'midline_memory.allow_tangent_estimate_without_memory': True,
    'midline_memory.max_tangent_estimation_extension_m': 2.0,
    'pure_pursuit.lookahead_gain': 0.0,
    'pure_pursuit.lookahead_m': 3.0,
    'pure_pursuit.max_lookahead_m': 8.0,
    'pure_pursuit.min_lookahead_m': 1.5,
    'pure_pursuit.steering_limit_rad': 0.52,
    'pure_pursuit.steering_lowpass_alpha': 1.0,
    'pure_pursuit.steering_rate_limit_rad_s': 10.0,
    'pure_pursuit.wheelbase_m': 1.65,
    'runtime.log_throttle_s': 1.0,
    'runtime.publish_rate_hz': 180.0,
    'speed_control.curvature_speed_gain': 4.0,
    'speed_control.lowpass_speed_alpha': 0.15,
    'speed_control.speed_max_mps': 4.0,
    'speed_control.speed_min_mps': 1.0,
    'stanley.cross_track_deadband_m': 0.0,
    'stanley.heading_gain': 1.6,
    'stanley.k_gain': 1.2,
    'stanley.lookahead_idx_offset': 0,
    'stanley.softening_speed_mps': 0.0,
    'stanley.steering_limit_rad': 0.52,
    'stanley.steering_lowpass_alpha': 1.0,
    'stanley.steering_rate_limit_rad_s': 10.0,
    'stanley.use_yaw_rate_damping': True,
    'stanley.wheelbase_m': 1.65,
    'stanley.yaw_rate_damping_gain': 0.0,
    'topics.centerline_topic': '/planned_centerline',
    'topics.cmd_topic': '/cmd',
    'topics.odom_topic': '/sim/odom',
    'topics.points_topic': '/planned_centerline_points',
    'topics.tracked_cones_topic': '/tracked_cones',
    'topics.viz_topic': '/planner_viz',
    'validation.candidate_jump_recover_frames': 3,
    'validation.candidate_jump_reject_threshold_m': 0.45,
    'validation.candidate_min_extent_m': 2.0,
    'validation.candidate_min_points': 4,
    'validation.hold_exit_clean_frames': 2,
    'validation.hold_last_valid_s': 2.5,
    'validation.jump_check_horizon_m': 8.0,
    'validation.max_near_field_lateral_jump_m': 0.6,
    'width_estimation.alpha': 0.18,
    'width_estimation.initial_width_m': 3.6,
    'width_estimation.max_delta_per_update_m': 0.2,
    'width_estimation.max_width_m': 4.8,
    'width_estimation.min_width_m': 2.4,
}

MIGRATED_TRACKED_CONE_PARAMETER_STATUS = {
    'centerline.temporal_alpha': PARAM_STATUS_COMPATIBILITY_ONLY,
}

MIGRATED_TRACKED_CONE_COMPATIBILITY_DEFAULTS = {
    'centerline.temporal_alpha': 0.25,
}


@dataclass(frozen=True)
class MigratedTrackedConePlannerCommonConfig:
    planning_frame: str
    odom_frame: str
    base_frame: str
    tf_timeout_s: float
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
    edge_quantization_m: float
    jump_warn_threshold_m: float
    edge_churn_warn_threshold: float
    publish_control_debug: bool
    publish_thesis_context: bool
    enable_debug_markers: bool
    show_raw_cones: bool
    show_boundary_chains: bool
    show_pair_lines: bool
    show_raw_midpoint_chain: bool
    show_raw_prevalidation_centerline: bool
    publish_points_topic: bool
    show_lookahead_point: bool
    show_triangulation_edges: bool
    show_candidate_edges: bool
    show_selected_edges: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_tracked_cone_controller_type(controller_type: str) -> str:
    normalized = str(controller_type).strip().lower() or 'stanley'
    if normalized not in SUPPORTED_TRACKED_CONE_CONTROLLER_TYPES:
        raise ValueError(
            "Unsupported control.controller_type '%s'. Supported values: stanley, pure_pursuit, none"
            % normalized
        )
    return normalized


def build_tracked_cone_controller(*, node: Any, controller_type: str, publish_rate_hz: float):
    normalized = normalize_tracked_cone_controller_type(controller_type)
    if normalized == 'none':
        return None
    return build_steering_controller(node=node, controller_type=normalized, publish_rate_hz=publish_rate_hz)


def log_tracked_cone_controller_mode(node: Any, *, controller_type: str) -> None:
    if controller_type == 'none':
        node.get_logger().info("control.controller_type 'none'; controller output is disabled")


def emit_migrated_planner_compatibility_warnings(
    node: Any, *, planner_label: str, temporal_alpha: float
) -> None:
    default = float(MIGRATED_TRACKED_CONE_COMPATIBILITY_DEFAULTS['centerline.temporal_alpha'])
    if math.isclose(float(temporal_alpha), default, rel_tol=0.0, abs_tol=1e-9):
        return
    node.get_logger().warn(
        "Parameter 'centerline.temporal_alpha' is compatibility-only for %s; "
        'whole-path temporal smoothing is disabled, and midline memory now owns path stability.'
        % planner_label
    )


def read_migrated_tracked_cone_planner_common_config(
    node: Any,
    *,
    planner_label: str,
    diagnostics_topic_fallback: str,
) -> MigratedTrackedConePlannerCommonConfig:
    """Read all common planner parameters from *node* and return a frozen config object."""

    # Runtime
    publish_rate_hz = max(1.0, float(node.get_parameter('runtime.publish_rate_hz').value))
    log_throttle_s = max(0.1, float(node.get_parameter('runtime.log_throttle_s').value))

    # Control
    controller_type = normalize_tracked_cone_controller_type(
        node.get_parameter('control.controller_type').value
    )
    odom_lag_compensation_s = min(
        max(0.0, float(node.get_parameter('control.odom_lag_compensation_ms').value)), 150.0
    ) / 1000.0
    stop_if_no_path = bool(node.get_parameter('control.stop_if_no_path').value)

    # Frames
    planning_frame = str(node.get_parameter('frames.planning_frame').value).strip() or 'odom'
    odom_frame = str(node.get_parameter('frames.odom_frame').value).strip() or 'odom'
    base_frame = str(node.get_parameter('frames.base_frame').value).strip() or 'front_axle'
    tf_timeout_s = max(0.0, float(node.get_parameter('frames.tf_timeout_s').value))

    # Topics
    tracked_cones_topic = str(node.get_parameter('topics.tracked_cones_topic').value).strip()
    cmd_topic = str(node.get_parameter('topics.cmd_topic').value).strip()
    centerline_topic = str(node.get_parameter('topics.centerline_topic').value).strip()
    viz_topic = str(node.get_parameter('topics.viz_topic').value).strip()
    points_topic = str(node.get_parameter('topics.points_topic').value).strip()
    odom_topic = str(node.get_parameter('topics.odom_topic').value).strip()

    # Filtering
    infer_unknown_by_side = bool(node.get_parameter('filtering.infer_unknown_by_side').value)
    infer_orange_by_side = bool(node.get_parameter('filtering.infer_orange_by_side').value)
    orange_min_lateral_m = float(node.get_parameter('filtering.orange_min_lateral_m').value)
    orange_neighbor_radius_m = float(node.get_parameter('filtering.orange_neighbor_radius_m').value)
    orange_neighbor_margin_m = float(node.get_parameter('filtering.orange_neighbor_margin_m').value)

    # Centerline
    centerline_path_resolution_m = max(
        0.05, float(node.get_parameter('centerline.path_resolution_m').value)
    )
    temporal_alpha = max(0.0, min(1.0, float(node.get_parameter('centerline.temporal_alpha').value)))

    # Midline memory — distances are lower-bounded by earlier levels
    midline_horizon_m = max(1.0, float(node.get_parameter('midline_memory.horizon_m').value))
    station_spacing = max(0.05, float(node.get_parameter('midline_memory.station_spacing_m').value))
    near_dist = max(0.0, float(node.get_parameter('midline_memory.near_distance_m').value))
    mid_dist = max(near_dist, float(node.get_parameter('midline_memory.mid_distance_m').value))
    handoff_dist = max(
        station_spacing,
        float(node.get_parameter('midline_memory.control_handoff_distance_m').value),
    )

    # Midline lateral shift limits — each level >= the previous
    near_shift = max(0.0, float(node.get_parameter('midline_memory.near_max_lateral_shift_m').value))
    mid_shift = max(near_shift, float(node.get_parameter('midline_memory.mid_max_lateral_shift_m').value))
    far_shift = max(mid_shift, float(node.get_parameter('midline_memory.far_max_lateral_shift_m').value))

    near_alpha = max(0.0, min(1.0, float(node.get_parameter('midline_memory.near_alpha').value)))
    mid_alpha = max(0.0, min(1.0, float(node.get_parameter('midline_memory.mid_alpha').value)))
    far_alpha = max(0.0, min(1.0, float(node.get_parameter('midline_memory.far_alpha').value)))
    min_buffer_confidence = max(
        0.0, min(1.0, float(node.get_parameter('midline_memory.min_buffer_confidence').value))
    )
    midline_hold = max(0.0, float(node.get_parameter('midline_memory.hold_last_valid_duration_s').value))
    midline_min_estimated_extent_m = max(
        0.5, float(node.get_parameter('midline_memory.min_estimated_extent_m').value)
    )
    midline_max_estimation_extension_m = max(
        0.0, float(node.get_parameter('midline_memory.max_estimation_extension_m').value)
    )
    midline_max_estimation_join_lateral_m = max(
        0.0, float(node.get_parameter('midline_memory.max_estimation_join_lateral_m').value)
    )
    midline_max_estimation_join_heading_rad = max(
        0.0, float(node.get_parameter('midline_memory.max_estimation_join_heading_rad').value)
    )
    midline_allow_tangent = bool(
        node.get_parameter('midline_memory.allow_tangent_estimate_without_memory').value
    )
    midline_max_tangent_extension_m = max(
        0.0, float(node.get_parameter('midline_memory.max_tangent_estimation_extension_m').value)
    )

    # Speed control
    speed_min = max(0.0, float(node.get_parameter('speed_control.speed_min_mps').value))
    speed_max = max(speed_min, float(node.get_parameter('speed_control.speed_max_mps').value))
    curvature_speed_gain = max(
        0.0, float(node.get_parameter('speed_control.curvature_speed_gain').value)
    )
    lowpass_speed_alpha = max(
        0.0, min(1.0, float(node.get_parameter('speed_control.lowpass_speed_alpha').value))
    )

    # Validation / path hold
    hold_last_valid_s = max(midline_hold, float(node.get_parameter('validation.hold_last_valid_s').value))
    hold_exit_clean_frames = max(1, int(node.get_parameter('validation.hold_exit_clean_frames').value))
    candidate_jump_reject_threshold_m = max(
        0.0, float(node.get_parameter('validation.candidate_jump_reject_threshold_m').value)
    )
    candidate_jump_recover_frames = max(
        1, int(node.get_parameter('validation.candidate_jump_recover_frames').value)
    )
    candidate_min_points = max(2, int(node.get_parameter('validation.candidate_min_points').value))
    candidate_min_extent_m = max(
        0.5, float(node.get_parameter('validation.candidate_min_extent_m').value)
    )

    # Diagnostics
    diagnostics_topic = (
        str(node.get_parameter('diagnostics.topic').value).strip() or diagnostics_topic_fallback
    )
    centerline_jump_horizon_m = max(
        0.5, float(node.get_parameter('diagnostics.centerline_jump_horizon_m').value)
    )
    edge_quantization_m = max(1e-6, float(node.get_parameter('diagnostics.edge_quantization_m').value))
    jump_warn_threshold_m = max(0.0, float(node.get_parameter('diagnostics.jump_warn_threshold_m').value))
    edge_churn_warn_threshold = max(
        0.0, float(node.get_parameter('diagnostics.edge_churn_warn_threshold').value)
    )
    publish_control_debug = bool(node.get_parameter('diagnostics.publish_control_debug').value)
    publish_thesis_context = bool(node.get_parameter('diagnostics.publish_thesis_context').value)

    # Debug markers
    enable_debug_markers = bool(node.get_parameter('debug.enable_markers').value)
    show_raw_cones = bool(node.get_parameter('debug.show_raw_cones').value)
    show_boundary_chains = bool(node.get_parameter('debug.show_boundary_chains').value)
    show_pair_lines = bool(node.get_parameter('debug.show_pair_lines').value)
    show_raw_midpoint_chain = bool(node.get_parameter('debug.show_raw_midpoint_chain').value)
    show_raw_prevalidation_centerline = bool(
        node.get_parameter('debug.show_raw_prevalidation_centerline').value
    )
    publish_points_topic = bool(node.get_parameter('debug.publish_points_topic').value)
    show_lookahead_point = bool(node.get_parameter('debug.show_lookahead_point').value)

    common = MigratedTrackedConePlannerCommonConfig(
        # Frames
        planning_frame=planning_frame,
        odom_frame=odom_frame,
        base_frame=base_frame,
        tf_timeout_s=tf_timeout_s,
        # Topics
        tracked_cones_topic=tracked_cones_topic,
        cmd_topic=cmd_topic,
        centerline_topic=centerline_topic,
        viz_topic=viz_topic,
        points_topic=points_topic,
        odom_topic=odom_topic,
        # Filtering
        infer_unknown_by_side=infer_unknown_by_side,
        infer_orange_by_side=infer_orange_by_side,
        orange_min_lateral_m=orange_min_lateral_m,
        orange_neighbor_radius_m=orange_neighbor_radius_m,
        orange_neighbor_margin_m=orange_neighbor_margin_m,
        # Centerline
        centerline_path_resolution_m=centerline_path_resolution_m,
        temporal_alpha=temporal_alpha,
        # Temporal smoothing — disabled; midline memory owns stability
        enable_temporal_smoothing=False,
        smoothing_alpha=temporal_alpha,
        enable_near_field_freeze=False,
        freeze_near_field_m=0.0,
        freeze_blend_length_m=0.0,
        enable_committed_near_field=False,
        commit_plan_horizon_m=0.0,
        commit_stable_frames=1,
        commit_update_max_churn_ratio=1.0,
        # Midline memory
        midline_horizon_m=midline_horizon_m,
        midline_station_spacing_m=station_spacing,
        midline_near_distance_m=near_dist,
        midline_mid_distance_m=mid_dist,
        midline_control_handoff_distance_m=handoff_dist,
        midline_near_alpha=near_alpha,
        midline_mid_alpha=mid_alpha,
        midline_far_alpha=far_alpha,
        midline_near_max_shift_m=near_shift,
        midline_mid_max_shift_m=mid_shift,
        midline_far_max_shift_m=far_shift,
        midline_min_buffer_confidence=min_buffer_confidence,
        midline_hold_last_valid_duration_s=midline_hold,
        midline_min_estimated_extent_m=midline_min_estimated_extent_m,
        midline_max_estimation_extension_m=midline_max_estimation_extension_m,
        midline_max_estimation_join_lateral_m=midline_max_estimation_join_lateral_m,
        midline_max_estimation_join_heading_rad=midline_max_estimation_join_heading_rad,
        midline_allow_tangent_estimate_without_memory=midline_allow_tangent,
        midline_max_tangent_estimation_extension_m=midline_max_tangent_extension_m,
        # Runtime
        publish_rate_hz=publish_rate_hz,
        log_throttle_s=log_throttle_s,
        # Control
        controller_type=controller_type,
        odom_lag_compensation_s=odom_lag_compensation_s,
        _controller=build_tracked_cone_controller(
            node=node, controller_type=controller_type, publish_rate_hz=publish_rate_hz
        ),
        stop_if_no_path=stop_if_no_path,
        # Speed control
        speed_min_mps=speed_min,
        speed_max_mps=speed_max,
        curvature_speed_gain=curvature_speed_gain,
        lowpass_speed_alpha=lowpass_speed_alpha,
        # Validation / hold
        hold_last_valid_s=hold_last_valid_s,
        hold_exit_clean_frames=hold_exit_clean_frames,
        candidate_jump_reject_threshold_m=candidate_jump_reject_threshold_m,
        candidate_jump_recover_frames=candidate_jump_recover_frames,
        candidate_min_points=candidate_min_points,
        candidate_min_extent_m=candidate_min_extent_m,
        # Diagnostics
        diagnostics_topic=diagnostics_topic,
        centerline_jump_horizon_m=centerline_jump_horizon_m,
        edge_quantization_m=edge_quantization_m,
        jump_warn_threshold_m=jump_warn_threshold_m,
        edge_churn_warn_threshold=edge_churn_warn_threshold,
        publish_control_debug=publish_control_debug,
        publish_thesis_context=publish_thesis_context,
        # Debug markers
        enable_debug_markers=enable_debug_markers,
        show_raw_cones=show_raw_cones,
        show_boundary_chains=show_boundary_chains,
        show_pair_lines=show_pair_lines,
        show_raw_midpoint_chain=show_raw_midpoint_chain,
        show_raw_prevalidation_centerline=show_raw_prevalidation_centerline,
        publish_points_topic=publish_points_topic,
        show_lookahead_point=show_lookahead_point,
        show_triangulation_edges=False,
        show_candidate_edges=False,
        show_selected_edges=False,
    )
    emit_migrated_planner_compatibility_warnings(
        node, planner_label=planner_label, temporal_alpha=common.temporal_alpha
    )
    log_tracked_cone_controller_mode(node, controller_type=common.controller_type)
    return common


def apply_common_config_to_node(node: Any, common: MigratedTrackedConePlannerCommonConfig) -> None:
    for field in fields(common):
        setattr(node, field.name, getattr(common, field.name))
