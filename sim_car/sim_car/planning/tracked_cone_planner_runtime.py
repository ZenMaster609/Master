#!/usr/bin/env python3
"""Shared tracked-cone planner runtime used by migrated planners."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import Buffer, TransformListener
from vehicle_plotter_msgs.msg import ConeDetectionArray
from visualization_msgs.msg import MarkerArray

from sim_car.cones.tracking.pose import (
    convert_odom_child_pose_to_base_frame,
    project_planar_pose_constant_twist,
)
from sim_car.cones.tracking.fusion import normalize_color, resolve_boundary_colors_for_planning
from sim_car.planning.triangulation_planner_core import (
    CoreConfig,
    CorePrior,
    CoreResult,
    compute_centerline,
    compute_centerline_jump_max,
    edge_churn_count,
    edge_churn_ratio,
    selected_edge_keys,
    tracked_cones_frame_delta_p95,
)
from sim_car.planning.tracked_cone_planner_contract import (
    build_tracked_cone_controller,
    log_tracked_cone_controller_mode,
    normalize_tracked_cone_controller_type,
)
from sim_car.planning.path_stability import (
    extract_forward_path_from_pose,
    path_cumulative_lengths,
    resample_to_count,
    sample_path_at_lengths,
    splice_frozen_near_field,
)
from sim_car.planning.tracked_cone_planner_geometry import (
    _base_point_to_odom,
    _odom_point_to_base,
    _transform_point,
    _yaw_from_quat,
)
from sim_car.planning.planner_runtime_types import PlannerIdentity
from sim_car.planning.planning_state_machine import StateMachineMixin
from sim_car.planning.planning_diagnostics import DiagnosticsMixin
from sim_car.planning.planning_visualization import VisualizationMixin

@dataclass
class _PlanResult:
    result: CoreResult
    centerline: np.ndarray
    raw_centerline: np.ndarray
    raw_midpoint_chain: np.ndarray
    status: str
    publish_mode: str
    hold_reason: str
    near_field_freeze_active: bool
    committed_near_field_active: bool
    centerline_jump_max_m: float
    selected_edge_churn: float
    selected_edge_churn_count: int
    tracked_delta_p95_m: float


@dataclass
class _ControlResult:
    cmd_speed: float = 0.0
    cmd_steering: float = 0.0
    lookahead: float = 0.0
    control_target_frame: Optional[np.ndarray] = None
    control_debug_metrics: Optional[dict] = None
    zero_cmd_sent_flag: int = 0
    controller_failed: bool = False
    control_path_point_count: int = 0


class TrackedConePlannerRuntime(DiagnosticsMixin, VisualizationMixin, StateMachineMixin, Node):
    """Consumes tracked cones and publishes a centerline path + debug markers."""

    def __init__(self) -> None:
        self._planner_identity = self._planner_identity_config()
        super().__init__(self._planner_identity.node_name)
        self._declare_parameters()
        self._read_parameters()

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_cones_msg: Optional[ConeDetectionArray] = None
        self._latest_odom_msg: Optional[Odometry] = None
        self._latest_speed_mps: float = 0.0
        self._latest_yaw_rate_rps: float = 0.0
        self._last_throttled_log_sec: dict[str, float] = {}
        self._previous_centerline: Optional[np.ndarray] = None
        self._previous_raw_centerline: Optional[np.ndarray] = None
        self._previous_tracked_points: Optional[np.ndarray] = None
        self._previous_edge_keys: set[tuple[int, int, int, int]] = set()
        self._last_valid_centerline: Optional[np.ndarray] = None
        self._last_valid_raw_midpoint_chain: Optional[np.ndarray] = None
        self._last_valid_width_m: Optional[float] = None
        self._last_valid_time_sec: float = -1.0
        self._committed_centerline: Optional[np.ndarray] = None
        self._commit_stable_frame_count: int = 0
        self._hold_mode_active: bool = False
        self._hold_clean_frame_count: int = 0
        self._last_speed_cmd: Optional[float] = None
        self._last_steering_cmd: Optional[float] = None
        self._last_operator_state: Optional[str] = None
        self._last_operator_reason: Optional[str] = None
        self._active_planner_mode = self._planner_identity.planner_mode
        self._remembered_cone_viz_points = np.empty((0, 2), dtype=np.float64)
        self._remembered_cone_viz_colors: list[str] = []

        self._cmd_pub = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
        self._path_pub = self.create_publisher(Path, self.centerline_topic, 10)
        self._viz_pub = self.create_publisher(MarkerArray, self.viz_topic, 10)
        self._points_pub = self.create_publisher(PoseArray, self.points_topic, 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, self.diagnostics_topic, 10)

        self.create_subscription(
            ConeDetectionArray,
            self.tracked_cones_topic,
            self._cones_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_cb,
            qos_profile_sensor_data,
        )

        loop_hz = max(1.0, float(self.publish_rate_hz))
        self.create_timer(1.0 / loop_hz, self._on_timer)

        self.get_logger().info(
            f'{self._planner_identity.node_name} ready '
            f'cones={self.tracked_cones_topic} odom={self.odom_topic} '
            f'cmd={self.cmd_topic} path={self.centerline_topic} viz={self.viz_topic} '
            f'planning_frame={self.planning_frame} controller={self.controller_type}'
        )

    def _planner_identity_config(self) -> PlannerIdentity:
        return PlannerIdentity(
            node_name='tracked_cone_planner_runtime',
            planner_mode='tracked_cone',
            diagnostics_prefix='tracked_cone_planner',
            diagnostics_topic='/tracked_cone_planner/diagnostics',
        )

    def _declare_parameters(self) -> None:
        defaults = {
            'frames.planning_frame': 'odom',
            'frames.odom_frame': 'odom',
            'frames.base_frame': 'front_axle',
            'frames.tf_timeout_s': 0.03,
            'topics.tracked_cones_topic': '/tracked_cones',
            'topics.cmd_topic': '/cmd',
            'topics.centerline_topic': '/planned_centerline',
            'topics.viz_topic': '/planner_viz',
            'topics.points_topic': '/planned_centerline_points',
            'topics.odom_topic': '/sim/odom',
            'filtering.max_cone_range_m': 25.0,
            'filtering.behind_drop_m': 5.0,
            'filtering.min_confidence': 0.3,
            'filtering.use_unknown_cones': True,
            'filtering.infer_unknown_by_side': True,
            'filtering.infer_orange_by_side': True,
            'filtering.include_orange': False,
            'filtering.orange_min_lateral_m': 0.9,
            'filtering.orange_neighbor_radius_m': 3.5,
            'filtering.orange_neighbor_margin_m': 0.75,
            'filtering.min_colored_cones': 6,
            'filtering.min_required_cones': 6,
            'edge_selection.min_cross_edge_m': 0.8,
            'edge_selection.max_cross_edge_m': 6.0,
            'edge_selection.cross_edge_lateral_ratio': 0.6,
            'edge_selection.min_cross_edges': 3,
            'edge_selection.max_near_field_lateral_jump_m': 0.6,
            'edge_selection.near_field_midpoint_count': 5,
            'edge_selection.max_diagonal_forward_alignment': 0.55,
            'edge_selection.max_width_prior_step_drift_m': 0.2,
            'edge_selection.max_midpoint_kink_rad': 1.2,
            'edge_selection.max_seed_midpoint_distance_m': 8.0,
            'edge_selection.max_same_side_step_m': 5.0,
            'edge_selection.min_midpoint_progress_m': 0.15,
            'edge_selection.width_prior_tolerance_m': 1.0,
            'edge_selection.temporal_midpoint_match_tolerance_m': 1.0,
            'edge_selection.local_opposite_neighbor_count': 2,
            'edge_selection.local_opposite_forward_sector_rad': 0.9,
            'centerline.min_spacing_m': 0.5,
            'centerline.path_resolution_m': 0.5,
            'centerline.max_path_length_m': 30.0,
            'centerline.enable_temporal_smoothing': True,
            'centerline.smoothing_alpha': 0.3,
            'centerline.enable_near_field_freeze': True,
            'centerline.freeze_near_field_m': 3.0,
            'centerline.freeze_blend_length_m': 1.0,
            'centerline.enable_committed_near_field': True,
            'centerline.commit_plan_horizon_m': 8.0,
            'centerline.commit_stable_frames': 3,
            'centerline.commit_update_max_churn_ratio': 0.45,
            'runtime.publish_rate_hz': 180.0,
            'runtime.log_throttle_s': 1.0,
            'control.controller_type': 'stanley',
            'control.odom_lag_compensation_ms': 0.0,
            'control.stop_if_no_path': True,
            'stanley.k_gain': 1.2,
            'stanley.softening_speed_mps': 0.0,
            'stanley.heading_gain': 1.0,
            'stanley.lookahead_idx_offset': 0,
            'stanley.steering_limit_rad': 0.52,
            'stanley.steering_lowpass_alpha': 1.0,
            'stanley.steering_rate_limit_rad_s': 10.0,
            'stanley.use_yaw_rate_damping': True,
            'stanley.yaw_rate_damping_gain': 0.0,
            'stanley.wheelbase_m': 1.65,
            'stanley.cross_track_deadband_m': 0.0,
            'speed_control.speed_min_mps': 1.0,
            'speed_control.speed_max_mps': 1.8,
            'speed_control.curvature_speed_gain': 4.0,
            'speed_control.lowpass_speed_alpha': 0.15,
            'validation.hold_last_valid_s': 1.25,
            'validation.hold_exit_clean_frames': 2,
            'diagnostics.topic': self._planner_identity.diagnostics_topic,
            'diagnostics.centerline_jump_horizon_m': 8.0,
            'diagnostics.edge_quantization_m': 0.05,
            'diagnostics.jump_warn_threshold_m': 0.8,
            'diagnostics.edge_churn_warn_threshold': 0.55,
            'diagnostics.publish_control_debug': True,
                    'debug.enable_markers': True,
            'debug.show_raw_cones': False,
            'debug.show_triangulation_edges': True,
            'debug.show_candidate_edges': True,
            'debug.show_selected_edges': True,
            'debug.show_raw_midpoint_chain': True,
            'debug.show_raw_prevalidation_centerline': True,
            'debug.publish_points_topic': False,
            'debug.show_lookahead_point': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        self.planning_frame = str(self.get_parameter('frames.planning_frame').value).strip() or 'odom'
        self.odom_frame = str(self.get_parameter('frames.odom_frame').value).strip() or 'odom'
        self.base_frame = str(self.get_parameter('frames.base_frame').value).strip() or 'front_axle'
        self.tf_timeout_s = max(0.0, float(self.get_parameter('frames.tf_timeout_s').value))

        self.tracked_cones_topic = str(self.get_parameter('topics.tracked_cones_topic').value)
        self.cmd_topic = str(self.get_parameter('topics.cmd_topic').value)
        self.centerline_topic = str(self.get_parameter('topics.centerline_topic').value)
        self.viz_topic = str(self.get_parameter('topics.viz_topic').value)
        self.points_topic = str(self.get_parameter('topics.points_topic').value)
        self.odom_topic = str(self.get_parameter('topics.odom_topic').value)
        self.infer_unknown_by_side = bool(self.get_parameter('filtering.infer_unknown_by_side').value)
        self.infer_orange_by_side = bool(self.get_parameter('filtering.infer_orange_by_side').value)
        self.orange_min_lateral_m = float(self.get_parameter('filtering.orange_min_lateral_m').value)
        self.orange_neighbor_radius_m = float(
            self.get_parameter('filtering.orange_neighbor_radius_m').value
        )
        self.orange_neighbor_margin_m = float(
            self.get_parameter('filtering.orange_neighbor_margin_m').value
        )

        self.enable_temporal_smoothing = bool(self.get_parameter('centerline.enable_temporal_smoothing').value)
        self.centerline_path_resolution_m = max(
            0.05,
            float(self.get_parameter('centerline.path_resolution_m').value),
        )
        self.smoothing_alpha = float(
            np.clip(float(self.get_parameter('centerline.smoothing_alpha').value), 0.0, 1.0)
        )
        self.enable_near_field_freeze = bool(self.get_parameter('centerline.enable_near_field_freeze').value)
        self.freeze_near_field_m = max(
            0.0,
            float(self.get_parameter('centerline.freeze_near_field_m').value),
        )
        self.freeze_blend_length_m = max(
            0.0,
            float(self.get_parameter('centerline.freeze_blend_length_m').value),
        )
        self.enable_committed_near_field = bool(
            self.get_parameter('centerline.enable_committed_near_field').value
        )
        self.commit_plan_horizon_m = max(
            0.0,
            float(self.get_parameter('centerline.commit_plan_horizon_m').value),
        )
        self.commit_stable_frames = max(
            1,
            int(self.get_parameter('centerline.commit_stable_frames').value),
        )
        self.commit_update_max_churn_ratio = max(
            0.0,
            float(self.get_parameter('centerline.commit_update_max_churn_ratio').value),
        )

        self.publish_rate_hz = max(1.0, float(self.get_parameter('runtime.publish_rate_hz').value))
        self.log_throttle_s = max(0.1, float(self.get_parameter('runtime.log_throttle_s').value))

        self.controller_type = (
            normalize_tracked_cone_controller_type(
                self.get_parameter('control.controller_type').value
            )
        )
        self.odom_lag_compensation_s = min(
            max(0.0, float(self.get_parameter('control.odom_lag_compensation_ms').value)),
            150.0,
        ) / 1000.0
        self._controller = build_tracked_cone_controller(
            node=self,
            controller_type=self.controller_type,
            publish_rate_hz=self.publish_rate_hz,
        )
        self.stop_if_no_path = bool(self.get_parameter('control.stop_if_no_path').value)
        log_tracked_cone_controller_mode(self, controller_type=self.controller_type)

        self.speed_min_mps = max(0.0, float(self.get_parameter('speed_control.speed_min_mps').value))
        self.speed_max_mps = max(self.speed_min_mps, float(self.get_parameter('speed_control.speed_max_mps').value))
        self.curvature_speed_gain = max(0.0, float(self.get_parameter('speed_control.curvature_speed_gain').value))
        self.lowpass_speed_alpha = float(
            np.clip(float(self.get_parameter('speed_control.lowpass_speed_alpha').value), 0.0, 1.0)
        )
        self.hold_last_valid_s = max(0.0, float(self.get_parameter('validation.hold_last_valid_s').value))
        self.hold_exit_clean_frames = max(
            1,
            int(self.get_parameter('validation.hold_exit_clean_frames').value),
        )
        self.diagnostics_topic = (
            str(self.get_parameter('diagnostics.topic').value).strip()
            or self._planner_identity.diagnostics_topic
        )
        self.centerline_jump_horizon_m = max(
            0.5,
            float(self.get_parameter('diagnostics.centerline_jump_horizon_m').value),
        )
        self.edge_quantization_m = max(
            1e-6,
            float(self.get_parameter('diagnostics.edge_quantization_m').value),
        )
        self.jump_warn_threshold_m = max(
            0.0,
            float(self.get_parameter('diagnostics.jump_warn_threshold_m').value),
        )
        self.edge_churn_warn_threshold = max(
            0.0,
            float(self.get_parameter('diagnostics.edge_churn_warn_threshold').value),
        )
        self.publish_control_debug = bool(
            self.get_parameter('diagnostics.publish_control_debug').value
        )

        self.enable_debug_markers = bool(self.get_parameter('debug.enable_markers').value)
        self.show_raw_cones = bool(self.get_parameter('debug.show_raw_cones').value)
        self.show_triangulation_edges = bool(self.get_parameter('debug.show_triangulation_edges').value)
        self.show_candidate_edges = bool(self.get_parameter('debug.show_candidate_edges').value)
        self.show_selected_edges = bool(self.get_parameter('debug.show_selected_edges').value)
        self.show_raw_midpoint_chain = bool(
            self.get_parameter('debug.show_raw_midpoint_chain').value
        )
        self.show_raw_prevalidation_centerline = bool(
            self.get_parameter('debug.show_raw_prevalidation_centerline').value
        )
        self.publish_points_topic = bool(self.get_parameter('debug.publish_points_topic').value)
        self.show_lookahead_point = bool(self.get_parameter('debug.show_lookahead_point').value)

        self._core_config = CoreConfig(
            max_cone_range_m=float(self.get_parameter('filtering.max_cone_range_m').value),
            behind_drop_m=float(self.get_parameter('filtering.behind_drop_m').value),
            min_confidence=float(self.get_parameter('filtering.min_confidence').value),
            use_unknown_cones=bool(self.get_parameter('filtering.use_unknown_cones').value),
            include_orange=bool(self.get_parameter('filtering.include_orange').value),
            min_colored_cones=max(1, int(self.get_parameter('filtering.min_colored_cones').value)),
            min_required_cones=max(2, int(self.get_parameter('filtering.min_required_cones').value)),
            min_cross_edge_m=float(self.get_parameter('edge_selection.min_cross_edge_m').value),
            max_cross_edge_m=float(self.get_parameter('edge_selection.max_cross_edge_m').value),
            cross_edge_lateral_ratio=float(self.get_parameter('edge_selection.cross_edge_lateral_ratio').value),
            min_cross_edges=max(1, int(self.get_parameter('edge_selection.min_cross_edges').value)),
            max_near_field_lateral_jump_m=float(
                self.get_parameter('edge_selection.max_near_field_lateral_jump_m').value
            ),
            near_field_midpoint_count=max(
                2,
                int(self.get_parameter('edge_selection.near_field_midpoint_count').value),
            ),
            max_diagonal_forward_alignment=float(
                self.get_parameter('edge_selection.max_diagonal_forward_alignment').value
            ),
            max_width_prior_step_drift_m=float(
                self.get_parameter('edge_selection.max_width_prior_step_drift_m').value
            ),
            max_midpoint_kink_rad=float(self.get_parameter('edge_selection.max_midpoint_kink_rad').value),
            max_seed_midpoint_distance_m=float(
                self.get_parameter('edge_selection.max_seed_midpoint_distance_m').value
            ),
            max_same_side_step_m=float(self.get_parameter('edge_selection.max_same_side_step_m').value),
            min_midpoint_progress_m=float(
                self.get_parameter('edge_selection.min_midpoint_progress_m').value
            ),
            width_prior_tolerance_m=float(
                self.get_parameter('edge_selection.width_prior_tolerance_m').value
            ),
            temporal_midpoint_match_tolerance_m=float(
                self.get_parameter('edge_selection.temporal_midpoint_match_tolerance_m').value
            ),
            local_opposite_neighbor_count=max(
                1,
                int(self.get_parameter('edge_selection.local_opposite_neighbor_count').value),
            ),
            local_opposite_forward_sector_rad=float(
                self.get_parameter('edge_selection.local_opposite_forward_sector_rad').value
            ),
            min_spacing_m=float(self.get_parameter('centerline.min_spacing_m').value),
            path_resolution_m=float(self.get_parameter('centerline.path_resolution_m').value),
            max_path_length_m=float(self.get_parameter('centerline.max_path_length_m').value),
        )

    def _cones_cb(self, msg: ConeDetectionArray) -> None:
        self._latest_cones_msg = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom_msg = msg
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        frame_tokens = f'{msg.header.frame_id}/{msg.child_frame_id}'.lower()
        if 'base_link' in frame_tokens or 'base_footprint' in frame_tokens:
            self._latest_speed_mps = abs(vx)
            self._latest_yaw_rate_rps = float(msg.twist.twist.angular.z)
        else:
            self._latest_speed_mps = float(math.hypot(vx, vy))
            self._latest_yaw_rate_rps = float(msg.twist.twist.angular.z)

    def _on_timer(self) -> None:
        cones_msg = self._latest_cones_msg
        if cones_msg is None:
            self._publish_waiting_cycle(self.odom_frame, 'waiting for /tracked_cones', 'waiting_for_cones')
            return

        inputs = self._resolve_planning_inputs(cones_msg)
        if inputs is None:
            return  # empty cycle already published inside _resolve_planning_inputs

        target_frame, vehicle_x, vehicle_y, vehicle_yaw, points_xy, colors, confidences = inputs
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9

        plan = self._run_and_stabilize_plan(
            target_frame=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            points_xy=points_xy, colors=colors, confidences=confidences,
            now_sec=now_sec,
        )
        ctrl = self._run_controller_step(
            target_frame=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            centerline=plan.centerline,
        )

        hold_remaining_s = self._hold_remaining_s(now_sec)
        operator_state, operator_reason = self._determine_operator_state(
            centerline=plan.centerline, publish_mode=plan.publish_mode,
            status=plan.status, result=plan.result,
            ctrl=ctrl, hold_remaining_s=hold_remaining_s,
        )
        self._log_operator_state_transition(
            operator_state=operator_state, operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            selected_chain_length=int(plan.result.selected_chain_length),
        )
        self._active_planner_mode = self._planner_identity.planner_mode
        self._publish_plan_cycle(
            target_frame=target_frame, plan=plan, ctrl=ctrl,
            operator_state=operator_state, operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
        )

    def _publish_waiting_cycle(self, frame_id: str, status: str, reason: str) -> None:
        """Publish a zero-command empty cycle with operator_state='waiting'."""
        zero_cmd_sent = int(self._apply_no_path_behavior())
        self._publish_empty_cycle(
            frame_id=frame_id, status=status,
            operator_state='waiting', operator_reason=reason,
            cmd_speed=0.0, cmd_steering=0.0, lookahead=0.0,
            zero_cmd_sent_flag=zero_cmd_sent,
        )

    def _publish_plan_cycle(
        self,
        *,
        target_frame: str,
        plan: _PlanResult,
        ctrl: _ControlResult,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
    ) -> None:
        """Publish diagnostics and path/marker outputs for one planning cycle."""
        self._publish_diagnostics(
            frame_id=target_frame,
            centerline_jump_max_m=plan.centerline_jump_max_m,
            selected_edge_churn_ratio=plan.selected_edge_churn,
            tracked_cones_frame_delta_p95_m=plan.tracked_delta_p95_m,
            centerline_point_count=int(plan.centerline.shape[0]),
            selected_edge_count=int(plan.result.selected_edges.shape[0]),
            status=plan.status,
            control_debug_metrics=ctrl.control_debug_metrics,
            planner_metrics=self._build_planner_metrics(
                plan=plan, ctrl=ctrl,
                operator_state=operator_state, operator_reason=operator_reason,
                hold_remaining_s=hold_remaining_s,
            ),
        )
        self._publish_outputs(
            frame_id=target_frame,
            centerline=plan.centerline, raw_centerline=plan.raw_centerline,
            raw_midpoint_chain=plan.raw_midpoint_chain, result=plan.result,
            status=plan.status, control_target_frame=ctrl.control_target_frame,
            cmd_speed=ctrl.cmd_speed, cmd_steering=ctrl.cmd_steering, lookahead=ctrl.lookahead,
            operator_state=operator_state, operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            control_path_point_count=ctrl.control_path_point_count,
            candidate_diagonal_count=int(plan.result.candidate_count),
            selected_chain_length=int(plan.result.selected_chain_length),
            seed_midpoint_distance_m=float(plan.result.seed_midpoint_distance_m),
            near_field_lateral_max_m=float(plan.result.near_field_lateral_max_m),
            near_field_midpoint_kink_max_rad=float(plan.result.near_field_kink_max_rad),
        )

    def _resolve_planning_inputs(
        self,
        cones_msg: ConeDetectionArray,
    ) -> Optional[tuple]:
        """Resolve target frame, vehicle pose, and transformed cones; publish empty cycle and return None on any failure."""
        source_frame = str(cones_msg.header.frame_id).strip() or self.odom_frame
        target_frame = self._resolve_planning_frame(source_frame, cones_msg.header.stamp)

        pose = self._resolve_vehicle_pose(target_frame, cones_msg.header.stamp)
        if pose is None and target_frame != self.odom_frame:
            self._warn_throttled('fallback_odom', f'cannot resolve base pose in {target_frame}; falling back to odom')
            target_frame = self.odom_frame
            pose = self._resolve_vehicle_pose(target_frame, cones_msg.header.stamp)

        if pose is None:
            self._publish_waiting_cycle(target_frame, 'missing vehicle pose (tf and /sim/odom unavailable)', 'missing_vehicle_pose')
            return None

        vehicle_x, vehicle_y, vehicle_yaw = pose
        target_frame, vehicle_x, vehicle_y, vehicle_yaw, points_xy, colors, confidences = (
            self._resolve_cone_data(cones_msg, source_frame, target_frame, vehicle_x, vehicle_y, vehicle_yaw)
            or (None,) * 7
        )
        if points_xy is None:
            return None  # empty cycle already published inside _resolve_cone_data

        self._update_remembered_cone_viz(points_xy=points_xy, colors=colors)
        return target_frame, vehicle_x, vehicle_y, vehicle_yaw, points_xy, colors, confidences

    def _resolve_cone_data(
        self,
        cones_msg: ConeDetectionArray,
        source_frame: str,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> Optional[tuple]:
        """Transform cones into target_frame, falling back to odom frame if needed. Returns None (with empty cycle published) on failure."""
        points_xy, colors, confidences = self._convert_cones_to_frame(
            cones_msg, source_frame, target_frame,
            vehicle_xy=(vehicle_x, vehicle_y), vehicle_yaw=vehicle_yaw,
        )
        if points_xy is None and target_frame != self.odom_frame:
            self._warn_throttled('fallback_odom_cones', f'cannot transform cones to {target_frame}; planning in odom frame')
            target_frame = self.odom_frame
            pose = self._resolve_vehicle_pose(target_frame, cones_msg.header.stamp)
            if pose is None:
                self._publish_waiting_cycle(self.odom_frame, 'cone transform failed and odom pose unavailable', 'cone_transform_unavailable')
                return None
            vehicle_x, vehicle_y, vehicle_yaw = pose
            points_xy, colors, confidences = self._convert_cones_to_frame(
                cones_msg, source_frame, target_frame,
                vehicle_xy=(vehicle_x, vehicle_y), vehicle_yaw=vehicle_yaw,
            )
        if points_xy is None:
            self._publish_waiting_cycle(target_frame, 'cone transform unavailable', 'cone_transform_unavailable')
            return None
        return target_frame, vehicle_x, vehicle_y, vehicle_yaw, points_xy, colors, confidences

    def _run_and_stabilize_plan(self, *, target_frame: str, vehicle_x: float, vehicle_y: float,
                                vehicle_yaw: float, points_xy: np.ndarray, colors: list,
                                confidences: np.ndarray, now_sec: float) -> _PlanResult:
        """Run centerline algorithm, compute metrics, and apply hold/smoothing/freeze stability."""
        result = compute_centerline(
            points_xy=points_xy, colors=colors, confidences=confidences,
            vehicle_xy=(vehicle_x, vehicle_y), vehicle_yaw=vehicle_yaw,
            config=self._core_config,
            prior=CorePrior(
                previous_midpoints_raw=(
                    np.array(self._last_valid_raw_midpoint_chain, copy=True)
                    if self._last_valid_raw_midpoint_chain is not None else None
                ),
                previous_width_m=self._last_valid_width_m,
            ),
        )
        tracked_delta_p95_m, previous_edge_keys, churn_count, churn_ratio, jump_max_m = (
            self._compute_plan_metrics(result, points_xy)
        )
        raw_centerline, raw_midpoint_chain = result.centerline, result.midpoints_raw
        centerline, status, publish_mode, hold_reason, nf_freeze, committed_nf = (
            self._apply_hold_and_smoothing(
                result=result, raw_centerline=raw_centerline, raw_midpoint_chain=raw_midpoint_chain,
                target_frame=target_frame, vehicle_x=vehicle_x, vehicle_y=vehicle_y,
                selected_edge_churn=churn_ratio, previous_edge_keys=previous_edge_keys,
                now_sec=now_sec,
            )
        )
        return _PlanResult(
            result=result, centerline=centerline,
            raw_centerline=raw_centerline, raw_midpoint_chain=raw_midpoint_chain,
            status=status, publish_mode=publish_mode, hold_reason=hold_reason,
            near_field_freeze_active=nf_freeze, committed_near_field_active=committed_nf,
            centerline_jump_max_m=jump_max_m, selected_edge_churn=churn_ratio,
            selected_edge_churn_count=churn_count, tracked_delta_p95_m=tracked_delta_p95_m,
        )

    def _compute_plan_metrics(
        self, result: CoreResult, points_xy: np.ndarray
    ) -> tuple:
        """Update tracked-cone and edge-churn state; return (tracked_delta_p95_m, previous_edge_keys, churn_count, churn_ratio, jump_max_m)."""
        tracked_delta_p95_m = tracked_cones_frame_delta_p95(self._previous_tracked_points, points_xy)
        self._previous_tracked_points = np.array(points_xy, copy=True)

        previous_edge_keys = set(self._previous_edge_keys)
        selected_keys = selected_edge_keys(
            points=result.filtered_points, edges=result.selected_edges,
            quantization_m=self.edge_quantization_m,
        )
        churn_count = edge_churn_count(previous_edge_keys, selected_keys)
        churn_ratio = edge_churn_ratio(previous_edge_keys, selected_keys)
        self._previous_edge_keys = set(selected_keys)

        raw_centerline = result.centerline
        jump_max_m = compute_centerline_jump_max(raw_centerline, self._previous_raw_centerline, self.centerline_jump_horizon_m)
        self._previous_raw_centerline = np.array(raw_centerline, copy=True) if raw_centerline.shape[0] > 0 else None

        if jump_max_m > self.jump_warn_threshold_m:
            self._warn_throttled('centerline_jump_warn', f'centerline jump {jump_max_m:.3f} m exceeded threshold {self.jump_warn_threshold_m:.3f} m')
        if churn_ratio > self.edge_churn_warn_threshold:
            self._warn_throttled('edge_churn_warn', f'selected edge churn {churn_ratio:.3f} exceeded threshold {self.edge_churn_warn_threshold:.3f}')
        return tracked_delta_p95_m, previous_edge_keys, churn_count, churn_ratio, jump_max_m

    def _apply_hold_and_smoothing(
        self,
        *,
        result: CoreResult,
        raw_centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        selected_edge_churn: float,
        previous_edge_keys: set,
        now_sec: float,
    ) -> tuple:
        """Apply hold hysteresis then near-field stability. Returns (centerline, status, publish_mode, hold_reason, nf_freeze_active, committed_nf_active)."""
        centerline, status, publish_mode, hold_reason = self._apply_hold_hysteresis(
            result=result, raw_centerline=raw_centerline, now_sec=now_sec,
        )
        centerline, status, near_field_freeze_active, committed_near_field_active = (
            self._apply_near_field_stability(
                result=result, centerline=centerline, raw_midpoint_chain=raw_midpoint_chain,
                status=status, publish_mode=publish_mode,
                target_frame=target_frame, vehicle_x=vehicle_x, vehicle_y=vehicle_y,
                selected_edge_churn=selected_edge_churn, previous_edge_keys=previous_edge_keys,
                now_sec=now_sec,
            )
        )
        return centerline, status, publish_mode, hold_reason, near_field_freeze_active, committed_near_field_active

    def _apply_hold_hysteresis(
        self,
        *,
        result: CoreResult,
        raw_centerline: np.ndarray,
        now_sec: float,
    ) -> tuple:
        """Run hold hysteresis state machine. Returns (centerline, status, publish_mode, hold_reason)."""
        centerline = raw_centerline
        status = result.status
        plan_hold_active = False
        publish_mode = 'fresh'
        hold_reason = result.reject_reason
        # The first few meters of the published midline must remain laterally stable frame-to-frame.
        # A longer path is not better if the near-field segment teleports sideways.
        if centerline.shape[0] > 0:
            plan_ok = True
            if plan_ok:
                if self._advance_hold_hysteresis(plan_ok=True):
                    held = self._held_centerline(now_sec)
                    if held is not None:
                        centerline, plan_hold_active, publish_mode = held, True, 'held'
                        status = (f'{status}; hysteresis holding previous valid centerline '
                                  f'({self._hold_clean_frame_count}/{self.hold_exit_clean_frames})')
                if not plan_hold_active:
                    publish_mode = 'fresh'
            else:
                centerline, plan_hold_active, publish_mode, status = self._hold_on_invalid_plan(status, now_sec)
        else:
            centerline, plan_hold_active, publish_mode, status = self._hold_on_no_centerline(centerline, status, now_sec)
        if plan_hold_active:
            hold_reason = result.reject_reason or status
        return centerline, status, publish_mode, hold_reason

    def _hold_on_invalid_plan(self, status: str, now_sec: float) -> tuple:
        """Advance hysteresis for a bad plan and try to hold the previous valid centerline."""
        self._advance_hold_hysteresis(plan_ok=False)
        held = self._held_centerline(now_sec)
        if held is not None:
            return held, True, 'held', f'{status}; holding previous valid centerline'
        return np.empty((0, 2), dtype=np.float64), False, 'held', f'{status}; rejected unstable centerline'

    def _hold_on_no_centerline(self, centerline: np.ndarray, status: str, now_sec: float) -> tuple:
        """Advance hysteresis for an empty result and try to hold the previous valid centerline."""
        self._advance_hold_hysteresis(plan_ok=False)
        held = self._held_centerline(now_sec)
        if held is not None:
            return held, True, 'held', f'{status}; holding previous valid centerline'
        return centerline, False, 'held', status

    def _apply_near_field_stability(self, *, result: CoreResult, centerline: np.ndarray,
                                    raw_midpoint_chain: np.ndarray, status: str, publish_mode: str,
                                    target_frame: str, vehicle_x: float, vehicle_y: float,
                                    selected_edge_churn: float, previous_edge_keys: set,
                                    now_sec: float) -> tuple:
        """Apply temporal smoothing, near-field freeze, and record valid plan. Returns (centerline, status, nf_freeze_active, committed_nf_active)."""
        near_field_freeze_active = False
        committed_near_field_active = False
        if self.enable_temporal_smoothing and publish_mode == 'fresh':
            centerline = self._apply_temporal_smoothing(centerline)
            self._previous_centerline = np.array(centerline, copy=True)
        if publish_mode == 'fresh':
            self._update_committed_near_field(
                centerline=centerline, selected_edge_churn=selected_edge_churn,
                selected_chain_length=int(result.selected_chain_length),
                previous_edge_keys=previous_edge_keys,
            )
            centerline, near_field_freeze_active, committed_near_field_active = self._apply_near_field_freeze(
                centerline, frame_id=target_frame, vehicle_x=vehicle_x, vehicle_y=vehicle_y,
            )
            if centerline.shape[0] > 0:
                self._previous_centerline = np.array(centerline, copy=True)
                self._record_valid_plan(
                    now_sec=now_sec, centerline=centerline, raw_midpoint_chain=raw_midpoint_chain,
                    selected_chain_width_median=result.selected_chain_width_median,
                )
                if near_field_freeze_active:
                    status = f'{status}; frozen near-field {self.freeze_near_field_m:.1f} m before blending into fresh far field'
                if committed_near_field_active:
                    status = f'{status}; committed near-field {self.commit_plan_horizon_m:.1f} m trusted over high-churn refresh'
        elif publish_mode == 'held' and centerline.shape[0] > 0:
            self._previous_centerline = np.array(centerline, copy=True)
        return centerline, status, near_field_freeze_active, committed_near_field_active

    def _run_controller_step(
        self,
        *,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        centerline: np.ndarray,
    ) -> _ControlResult:
        """Compute control commands from centerline; publish cmd if controller succeeds."""
        ctrl = _ControlResult()
        control_path = self._centerline_to_vehicle_frame(
            centerline=centerline, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        ctrl.control_path_point_count = int(control_path.shape[0])

        if control_path.shape[0] >= 1 and self._controller is not None:
            ctrl = self._compute_controller_output(ctrl, control_path, target_frame, vehicle_x, vehicle_y, vehicle_yaw)
        elif control_path.shape[0] >= 1:
            ctrl.zero_cmd_sent_flag = int(self._apply_controller_disabled_behavior())
        else:
            ctrl.zero_cmd_sent_flag = int(self._apply_no_path_behavior())
            if self._last_speed_cmd is not None:
                ctrl.cmd_speed = float(self._last_speed_cmd)
            if self._last_steering_cmd is not None:
                ctrl.cmd_steering = float(self._last_steering_cmd)
        return ctrl

    def _compute_controller_output(self, ctrl: _ControlResult, control_path: np.ndarray,
                                   target_frame: str, vehicle_x: float, vehicle_y: float,
                                   vehicle_yaw: float) -> _ControlResult:
        """Call controller.compute and populate ctrl; on ValueError mark controller_failed."""
        try:
            controller_output = self._controller.compute(
                control_path=control_path,
                speed_mps=self._latest_speed_mps,
                yaw_rate_rps=self._latest_yaw_rate_rps,
            )
        except ValueError as exc:
            self._warn_throttled('controller_compute_error', f'controller compute failed: {exc}')
            ctrl.controller_failed = True
            ctrl.zero_cmd_sent_flag = int(self._apply_no_path_behavior())
            if self._last_speed_cmd is not None:
                ctrl.cmd_speed = float(self._last_speed_cmd)
            if self._last_steering_cmd is not None:
                ctrl.cmd_steering = float(self._last_steering_cmd)
            return ctrl
        ctrl.cmd_steering = float(controller_output.steering_rad)
        ctrl.cmd_speed = self._compute_speed_command(float(controller_output.kappa))
        ctrl.lookahead = float(controller_output.lookahead_m)
        self._last_speed_cmd = ctrl.cmd_speed
        self._last_steering_cmd = ctrl.cmd_steering
        self._publish_cmd(ctrl.cmd_speed, ctrl.cmd_steering)
        control_target_base = np.asarray(controller_output.target_point_base, dtype=np.float64)
        if self._is_alias(target_frame, self.base_frame):
            ctrl.control_target_frame = np.array(control_target_base, copy=True)
        elif self._is_alias(target_frame, self.odom_frame):
            tx, ty = _base_point_to_odom(
                float(control_target_base[0]), float(control_target_base[1]),
                vehicle_x, vehicle_y, vehicle_yaw,
            )
            ctrl.control_target_frame = np.array([tx, ty], dtype=np.float64)
        if controller_output.stanley_debug is not None:
            ctrl.control_debug_metrics = self._extract_stanley_debug_metrics(
                controller_output.stanley_debug, ctrl.control_target_frame)
        return ctrl

    def _extract_stanley_debug_metrics(self, debug: object, control_target_frame: Optional[np.ndarray]) -> dict:
        """Build the stanley debug metrics dict from a controller debug object."""
        return {
            'heading_error_rad': float(debug.heading_error_rad),
            'cross_track_error_m': float(debug.cross_track_error_m),
            'vehicle_speed_mps': float(debug.vehicle_speed_mps),
            'speed_term_mps': float(debug.speed_term_mps),
            'heading_contribution_rad': float(debug.heading_contribution_rad),
            'cross_track_contribution_rad': float(debug.cross_track_contribution_rad),
            'yaw_rate_damping_contribution_rad': float(debug.yaw_rate_damping_contribution_rad),
            'yaw_rate_rps': float(self._latest_yaw_rate_rps),
            'raw_steering_cmd_rad': float(debug.raw_steering_cmd_rad),
            'steering_after_clamp_rad': float(debug.steering_after_clamp_rad),
            'steering_after_filter_rad': float(debug.steering_after_filter_rad),
            'steering_after_rate_limit_rad': float(debug.steering_after_rate_limit_rad),
            'final_steering_cmd_rad': float(debug.final_steering_cmd_rad),
            'steering_saturated_flag': 1.0 if bool(debug.steering_saturated_flag) else 0.0,
            'nearest_path_index': float(debug.nearest_path_index),
            'heading_path_index': float(debug.heading_path_index),
            'target_point_x_base_m': float(debug.target_point_x_base_m),
            'target_point_y_base_m': float(debug.target_point_y_base_m),
            'target_point_x_frame_m': float(control_target_frame[0]) if control_target_frame is not None else float('nan'),
            'target_point_y_frame_m': float(control_target_frame[1]) if control_target_frame is not None else float('nan'),
        }

    def _determine_operator_state(self, *, centerline: np.ndarray, publish_mode: str, status: str,
                                   result: CoreResult, ctrl: _ControlResult,
                                   hold_remaining_s: float) -> tuple[str, str]:
        """Derive operator_state and operator_reason from plan and control results."""
        operator_state = 'fresh'
        operator_reason = 'none'
        core_reject_reason = self._normalize_core_reject_reason(result)
        if centerline.shape[0] == 0:
            if self._last_valid_centerline is not None and hold_remaining_s <= 0.0:
                operator_state, operator_reason = 'stopped', 'hold_expired_no_path'
            elif core_reject_reason != 'none':
                operator_state, operator_reason = 'stopped', core_reject_reason
            elif ctrl.zero_cmd_sent_flag:
                operator_state, operator_reason = 'stopped', 'stop_if_no_path'
            else:
                operator_state, operator_reason = 'held', 'holding_previous_valid'
        elif publish_mode == 'held':
            operator_state = 'held'
            if centerline.shape[0] > 0 and 'hysteresis holding previous valid centerline' in status:
                operator_reason = 'hysteresis_holding'
            elif core_reject_reason != 'none':
                operator_reason = core_reject_reason
            else:
                operator_reason = 'holding_previous_valid'
        if ctrl.controller_failed:
            operator_state, operator_reason = 'stopped', 'controller_compute_failed'
        elif self._controller is None and centerline.shape[0] > 0:
            operator_state, operator_reason = 'stopped', 'controller_disabled'
        elif centerline.shape[0] > 0 and ctrl.control_path_point_count <= 0 and operator_state != 'waiting':
            operator_state, operator_reason = 'stopped', 'no_control_path'
        if ctrl.zero_cmd_sent_flag and operator_reason == 'none':
            operator_state, operator_reason = 'stopped', 'stop_if_no_path'
        if ctrl.control_path_point_count > 0 and centerline.shape[0] > 0 and ctrl.zero_cmd_sent_flag == 0 and publish_mode == 'fresh':
            operator_state = 'fresh'
        return operator_state, operator_reason

    def _build_planner_metrics(
        self,
        *,
        plan: _PlanResult,
        ctrl: _ControlResult,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
    ) -> dict:
        """Build the planner_metrics dict for _publish_diagnostics."""
        metrics = self._build_result_quality_metrics(plan)
        metrics.update(self._build_operator_state_metrics(
            plan=plan, ctrl=ctrl,
            operator_state=operator_state, operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
        ))
        return metrics

    def _build_result_quality_metrics(self, plan: _PlanResult) -> dict:
        """Build the subset of planner_metrics describing algorithm result quality."""
        r = plan.result
        return {
            'candidate_diagonal_count': r.candidate_count,
            'selected_chain_length': r.selected_chain_length,
            'selected_chain_median_width_m': r.selected_chain_width_median,
            'expected_width_prior_m': r.expected_width_prior_m,
            'reject_wrong_side_count': r.reject_counts.get('wrong_side', 0),
            'reject_width_count': r.reject_counts.get('width', 0),
            'reject_width_range_count': r.reject_counts.get('width_range', 0),
            'reject_width_prior_count': r.reject_counts.get('width_prior', 0),
            'reject_orientation_count': r.reject_counts.get('orientation', 0),
            'reject_progress_count': r.reject_counts.get('progress', 0),
            'reject_near_field_continuity_count': r.reject_counts.get('near_field_continuity', 0),
            'reject_midpoint_kink_count': r.reject_counts.get('midpoint_kink', 0),
            'reject_seed_distance_count': r.reject_counts.get('seed_distance', 0),
            'near_field_lateral_max_m': r.near_field_lateral_max_m,
            'near_field_lateral_mean_m': r.near_field_lateral_mean_m,
            'near_field_displacement_max_m': r.near_field_displacement_max_m,
            'near_field_displacement_mean_m': r.near_field_displacement_mean_m,
            'near_field_midpoint_kink_max_rad': r.near_field_kink_max_rad,
            'seed_midpoint_distance_m': r.seed_midpoint_distance_m,
            'seed_temporal_offset_m': r.seed_temporal_offset_m,
            'selected_chain_churn_count': plan.selected_edge_churn_count,
            'selected_chain_churn_ratio': plan.selected_edge_churn,
        }

    def _build_operator_state_metrics(
        self,
        *,
        plan: _PlanResult,
        ctrl: _ControlResult,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
    ) -> dict:
        """Build the subset of planner_metrics describing operator state and hold/freeze flags."""
        return {
            'publish_mode': plan.publish_mode,
            'hold_mode_active_flag': 1 if self._hold_mode_active else 0,
            'hold_clean_frame_count': self._hold_clean_frame_count,
            'hold_reason': plan.hold_reason or '',
            'committed_near_field_active_flag': 1 if plan.committed_near_field_active else 0,
            'commit_stable_frame_count': self._commit_stable_frame_count,
            'near_field_freeze_active_flag': 1 if plan.near_field_freeze_active else 0,
            'planner_state_code': self._operator_state_code(operator_state),
            'fresh_publish_flag': 1 if operator_state == 'fresh' else 0,
            'held_publish_flag': 1 if operator_state == 'held' else 0,
            'stopped_flag': 1 if operator_state == 'stopped' else 0,
            'waiting_flag': 1 if operator_state == 'waiting' else 0,
            'operator_reason_code': self._operator_reason_code(operator_reason),
            'operator_reason': operator_reason,
            'hold_remaining_s': hold_remaining_s,
            'control_path_point_count': ctrl.control_path_point_count,
            'zero_cmd_sent_flag': ctrl.zero_cmd_sent_flag,
            'planner_mode': self._active_planner_mode,
        }

    def _centerline_to_vehicle_frame(
        self,
        *,
        centerline: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> np.ndarray:
        if centerline.shape[0] == 0:
            return centerline
        if self._is_alias(frame_id, self.base_frame):
            return centerline
        if not self._is_alias(frame_id, self.odom_frame):
            return np.empty((0, 2), dtype=np.float64)
        out = np.empty_like(centerline)
        for idx in range(centerline.shape[0]):
            xb, yb = _odom_point_to_base(
                centerline[idx, 0],
                centerline[idx, 1],
                vehicle_x,
                vehicle_y,
                vehicle_yaw,
            )
            out[idx, 0] = xb
            out[idx, 1] = yb
        return out

    def _compute_speed_command(self, kappa: float) -> float:
        v_des = self.speed_max_mps / (1.0 + self.curvature_speed_gain * abs(kappa))
        v_des = float(np.clip(v_des, self.speed_min_mps, self.speed_max_mps))
        if self._last_speed_cmd is None:
            return v_des
        alpha = self.lowpass_speed_alpha
        return float((alpha * v_des) + ((1.0 - alpha) * float(self._last_speed_cmd)))

    def _publish_cmd(self, speed_mps: float, steering_rad: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed_mps)
        msg.drive.steering_angle = float(steering_rad)
        self._cmd_pub.publish(msg)

    def _resolve_planning_frame(self, source_frame: str, stamp) -> str:
        requested = self.planning_frame
        if requested == self.odom_frame:
            return requested

        has_tf = (
            self._lookup_transform_with_alias(requested, source_frame, stamp)[0] is not None
            and self._lookup_transform_with_alias(requested, self.base_frame, stamp)[0] is not None
        )
        if has_tf:
            return requested
        return self.odom_frame

    def _resolve_vehicle_pose(self, frame_id: str, stamp) -> Optional[tuple[float, float, float]]:
        tf_msg, _resolved_target, _resolved_source = self._lookup_transform_with_alias(
            frame_id,
            self.base_frame,
            stamp,
        )
        if tf_msg is not None:
            tx = float(tf_msg.transform.translation.x)
            ty = float(tf_msg.transform.translation.y)
            q = tf_msg.transform.rotation
            yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
            return self._compensate_vehicle_pose(frame_id, (tx, ty, yaw))

        if not self._is_alias(frame_id, self.odom_frame):
            return None

        odom = self._latest_odom_msg
        if odom is None:
            return None
        if not self._is_alias(odom.header.frame_id, self.odom_frame):
            return None

        pose = odom.pose.pose
        tx = float(pose.position.x)
        ty = float(pose.position.y)
        q = pose.orientation
        yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        base_pose = self._convert_odom_child_pose_to_base_frame(
            child_frame=str(odom.child_frame_id).strip(),
            tx=tx,
            ty=ty,
            yaw=yaw,
        )
        return self._compensate_vehicle_pose(frame_id, base_pose)

    def _compensate_vehicle_pose(
        self,
        frame_id: str,
        pose: Optional[tuple[float, float, float]],
    ) -> Optional[tuple[float, float, float]]:
        if pose is None:
            return None
        if self._is_alias(frame_id, self.base_frame):
            return pose
        return project_planar_pose_constant_twist(
            pose,
            speed_mps=self._latest_speed_mps,
            yaw_rate_rps=self._latest_yaw_rate_rps,
            delay_s=float(getattr(self, 'odom_lag_compensation_s', 0.0)),
        )

    def _convert_odom_child_pose_to_base_frame(
        self,
        *,
        child_frame: str,
        tx: float,
        ty: float,
        yaw: float,
    ) -> Optional[tuple[float, float, float]]:
        return convert_odom_child_pose_to_base_frame(
            child_frame=child_frame,
            base_frame=self.base_frame,
            tx=tx,
            ty=ty,
            yaw=yaw,
            wheelbase_m=self._configured_wheelbase_m(),
            is_alias=self._is_alias,
        )

    def _configured_wheelbase_m(self) -> float:
        stanley_wheelbase = float(self.get_parameter('stanley.wheelbase_m').value)
        return max(0.1, stanley_wheelbase)

    def _convert_cones_to_frame(
        self,
        msg: ConeDetectionArray,
        source_frame: str,
        target_frame: str,
        *,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> tuple[Optional[np.ndarray], list[str], np.ndarray]:
        tf_msg = None
        if not self._is_alias(source_frame, target_frame):
            tf_msg, _resolved_target, _resolved_source = self._lookup_transform_with_alias(
                target_frame,
                source_frame,
                msg.header.stamp,
            )
            if tf_msg is None:
                fallback = self._convert_with_odom_pose_fallback(
                    msg,
                    source_frame,
                    target_frame,
                    vehicle_xy=vehicle_xy,
                    vehicle_yaw=vehicle_yaw,
                )
                if fallback is None:
                    return None, [], np.empty((0,), dtype=np.float64)
                return fallback

        points: list[tuple[float, float]] = []
        raw_colors: list[str] = []
        boundary_hints: list[str] = []
        confs: list[float] = []
        for cone in msg.cones:
            x = float(cone.position.x)
            y = float(cone.position.y)
            z = float(cone.position.z)
            if tf_msg is not None:
                x, y, z = _transform_point(tf_msg, x, y, z)
            points.append((x, y))
            raw_colors.append(normalize_color(cone.color))
            boundary_hints.append(str(getattr(cone, 'boundary_color', '')).strip())
            confs.append(float(np.clip(float(cone.confidence), 0.0, 1.0)))

        if not points:
            return np.empty((0, 2), dtype=np.float64), [], np.empty((0,), dtype=np.float64)
        points_xy = np.asarray(points, dtype=np.float64)
        colors = resolve_boundary_colors_for_planning(
            points_xy=points_xy,
            raw_colors=raw_colors,
            boundary_hints=boundary_hints,
            vehicle_xy=vehicle_xy,
            vehicle_yaw=vehicle_yaw,
            infer_unknown=self.infer_unknown_by_side,
            infer_orange=self.infer_orange_by_side,
            orange_min_lateral_m=self.orange_min_lateral_m,
            orange_neighbor_radius_m=self.orange_neighbor_radius_m,
            orange_neighbor_margin_m=self.orange_neighbor_margin_m,
        )
        return points_xy, colors, np.asarray(confs, dtype=np.float64)

    def _convert_with_odom_pose_fallback(
        self,
        msg: ConeDetectionArray,
        source_frame: str,
        target_frame: str,
        *,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> Optional[tuple[np.ndarray, list[str], np.ndarray]]:
        odom_pose = self._resolve_vehicle_pose(self.odom_frame, msg.header.stamp)
        if odom_pose is None:
            return None

        odom_x, odom_y, odom_yaw = odom_pose
        if self._is_alias(source_frame, self.odom_frame) and self._is_alias(target_frame, self.base_frame):
            return self._convert_odom_cones_to_base(msg, odom_x, odom_y, odom_yaw, vehicle_xy, vehicle_yaw)
        if self._is_alias(source_frame, self.base_frame) and self._is_alias(target_frame, self.odom_frame):
            return self._convert_base_cones_to_odom(msg, odom_x, odom_y, odom_yaw, vehicle_xy, vehicle_yaw)
        return None

    def _convert_odom_cones_to_base(
        self,
        msg: ConeDetectionArray,
        odom_x: float, odom_y: float, odom_yaw: float,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Transform cones from odom frame into base_link frame using known odom pose."""
        points: list[tuple[float, float]] = []
        raw_colors: list[str] = []
        boundary_hints: list[str] = []
        confs: list[float] = []
        for cone in msg.cones:
            x_base, y_base = _odom_point_to_base(
                float(cone.position.x), float(cone.position.y), odom_x, odom_y, odom_yaw,
            )
            points.append((x_base, y_base))
            raw_colors.append(normalize_color(cone.color))
            boundary_hints.append(str(getattr(cone, 'boundary_color', '')).strip())
            confs.append(float(np.clip(float(cone.confidence), 0.0, 1.0)))
        points_xy = np.asarray(points, dtype=np.float64)
        colors = resolve_boundary_colors_for_planning(
            points_xy=points_xy, raw_colors=raw_colors, boundary_hints=boundary_hints,
            vehicle_xy=vehicle_xy, vehicle_yaw=vehicle_yaw,
            infer_unknown=self.infer_unknown_by_side, infer_orange=self.infer_orange_by_side,
            orange_min_lateral_m=self.orange_min_lateral_m,
            orange_neighbor_radius_m=self.orange_neighbor_radius_m,
            orange_neighbor_margin_m=self.orange_neighbor_margin_m,
        )
        return points_xy, colors, np.asarray(confs, dtype=np.float64)

    def _convert_base_cones_to_odom(
        self,
        msg: ConeDetectionArray,
        odom_x: float, odom_y: float, odom_yaw: float,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Transform cones from base_link frame into odom frame using known odom pose."""
        cos_y = math.cos(odom_yaw)
        sin_y = math.sin(odom_yaw)
        points: list[tuple[float, float]] = []
        raw_colors: list[str] = []
        boundary_hints: list[str] = []
        confs: list[float] = []
        for cone in msg.cones:
            xb = float(cone.position.x)
            yb = float(cone.position.y)
            points.append((odom_x + (cos_y * xb) - (sin_y * yb), odom_y + (sin_y * xb) + (cos_y * yb)))
            raw_colors.append(normalize_color(cone.color))
            boundary_hints.append(str(getattr(cone, 'boundary_color', '')).strip())
            confs.append(float(np.clip(float(cone.confidence), 0.0, 1.0)))
        points_xy = np.asarray(points, dtype=np.float64)
        colors = resolve_boundary_colors_for_planning(
            points_xy=points_xy, raw_colors=raw_colors, boundary_hints=boundary_hints,
            vehicle_xy=vehicle_xy, vehicle_yaw=vehicle_yaw,
            infer_unknown=self.infer_unknown_by_side, infer_orange=self.infer_orange_by_side,
            orange_min_lateral_m=self.orange_min_lateral_m,
            orange_neighbor_radius_m=self.orange_neighbor_radius_m,
            orange_neighbor_margin_m=self.orange_neighbor_margin_m,
        )
        return points_xy, colors, np.asarray(confs, dtype=np.float64)

    def _apply_temporal_smoothing(self, centerline: np.ndarray) -> np.ndarray:
        if centerline.shape[0] == 0:
            return centerline
        prev = self._previous_centerline
        if prev is None or prev.shape[0] < 2 or centerline.shape[0] < 2:
            return centerline

        count_diff = abs(prev.shape[0] - centerline.shape[0])
        if count_diff > max(2, int(0.25 * centerline.shape[0])):
            return centerline

        prev_rs = self._resample_to_count(prev, centerline.shape[0])
        return (self.smoothing_alpha * centerline) + ((1.0 - self.smoothing_alpha) * prev_rs)

    def _apply_near_field_freeze(
        self,
        centerline: np.ndarray,
        *,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
    ) -> tuple[np.ndarray, bool, bool]:
        if not self.enable_near_field_freeze or self.freeze_near_field_m <= 0.0:
            return centerline, False, False
        previous, committed_active = self._freeze_reference_path(
            frame_id=frame_id,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
        )
        if centerline.shape[0] < 2 or previous is None or previous.shape[0] < 2:
            return centerline, False, committed_active
        freeze_horizon_m = self.freeze_near_field_m
        if committed_active:
            freeze_horizon_m = max(freeze_horizon_m, self.commit_plan_horizon_m)
        spliced, applied = self._splice_frozen_near_field(
            previous=previous,
            current=centerline,
            freeze_horizon_m=freeze_horizon_m,
            blend_length_m=self.freeze_blend_length_m,
            resolution_m=self.centerline_path_resolution_m,
        )
        return spliced, applied, committed_active

    def _freeze_reference_path(
        self,
        *,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
    ) -> tuple[Optional[np.ndarray], bool]:
        if (
            self.enable_committed_near_field
            and self.commit_plan_horizon_m > 0.0
            and self._is_alias(frame_id, self.odom_frame)
            and self._committed_centerline is not None
            and self._committed_centerline.shape[0] >= 2
        ):
            committed = self._extract_forward_path_from_pose(
                path=self._committed_centerline,
                vehicle_xy=(vehicle_x, vehicle_y),
                resolution_m=self.centerline_path_resolution_m,
            )
            if committed is not None and committed.shape[0] >= 2:
                return committed, True
            self._committed_centerline = None
            self._commit_stable_frame_count = 0
        return self._last_valid_centerline, False

    @staticmethod
    def _splice_frozen_near_field(
        *,
        previous: np.ndarray,
        current: np.ndarray,
        freeze_horizon_m: float,
        blend_length_m: float,
        resolution_m: float,
    ) -> tuple[np.ndarray, bool]:
        return splice_frozen_near_field(
            previous=previous,
            current=current,
            freeze_horizon_m=freeze_horizon_m,
            blend_length_m=blend_length_m,
            resolution_m=resolution_m,
        )

    @staticmethod
    def _path_cumulative_lengths(points: np.ndarray) -> np.ndarray:
        return path_cumulative_lengths(points)

    @staticmethod
    def _sample_path_at_lengths(points: np.ndarray, cum_lengths: np.ndarray, samples: np.ndarray) -> np.ndarray:
        return sample_path_at_lengths(points, cum_lengths, samples)

    @staticmethod
    def _extract_forward_path_from_pose(
        *,
        path: np.ndarray,
        vehicle_xy: tuple[float, float],
        resolution_m: float,
    ) -> Optional[np.ndarray]:
        return extract_forward_path_from_pose(
            path=path,
            vehicle_xy=vehicle_xy,
            resolution_m=resolution_m,
        )

    @staticmethod
    def _project_point_to_path_s(path: np.ndarray, cum_lengths: np.ndarray, point_xy: np.ndarray) -> float:
        best_s = 0.0
        best_dist = float('inf')
        for idx in range(path.shape[0] - 1):
            a = path[idx]
            b = path[idx + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1e-12:
                t = 0.0
                projected = a
            else:
                t = float(np.clip(np.dot(point_xy - a, ab) / denom, 0.0, 1.0))
                projected = a + (t * ab)
            dist = float(np.hypot(*(point_xy - projected)))
            if dist < best_dist:
                best_dist = dist
                best_s = float(cum_lengths[idx] + (t * np.hypot(ab[0], ab[1])))
        return best_s

    @staticmethod
    def _resample_to_count(points: np.ndarray, count: int) -> np.ndarray:
        return resample_to_count(points, count)

    def _publish_outputs(
        self,
        *,
        frame_id: str,
        centerline: np.ndarray,
        raw_centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        result: Optional[CoreResult],
        status: str,
        control_target_frame: Optional[np.ndarray],
        cmd_speed: float,
        cmd_steering: float,
        lookahead: float,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
        control_path_point_count: int,
        candidate_diagonal_count: int,
        selected_chain_length: int,
        seed_midpoint_distance_m: float,
        near_field_lateral_max_m: float,
        near_field_midpoint_kink_max_rad: float,
    ) -> None:
        now = self.get_clock().now().to_msg()
        path_msg = self._publish_path_and_points(frame_id=frame_id, centerline=centerline, now=now)
        if self.enable_debug_markers:
            self._publish_viz_markers(
                now=now, frame_id=frame_id,
                centerline=centerline, raw_centerline=raw_centerline,
                raw_midpoint_chain=raw_midpoint_chain, result=result,
                status=status, control_target_frame=control_target_frame,
                operator_state=operator_state, operator_reason=operator_reason,
                cmd_speed=cmd_speed, cmd_steering=cmd_steering, lookahead=lookahead,
                candidate_diagonal_count=candidate_diagonal_count,
                selected_chain_length=selected_chain_length,
                seed_midpoint_distance_m=seed_midpoint_distance_m,
                near_field_lateral_max_m=near_field_lateral_max_m,
                near_field_midpoint_kink_max_rad=near_field_midpoint_kink_max_rad,
                hold_remaining_s=hold_remaining_s,
            )

    def _publish_path_and_points(self, *, frame_id: str, centerline: np.ndarray, now: object) -> Path:
        """Build and publish the Path message; also publish PoseArray if configured."""
        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = frame_id
        for idx in range(centerline.shape[0]):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(centerline[idx, 0])
            pose.pose.position.y = float(centerline[idx, 1])
            pose.pose.position.z = 0.0
            yaw = self._path_point_yaw(centerline, idx)
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            path_msg.poses.append(pose)
        self._path_pub.publish(path_msg)
        if self.publish_points_topic:
            points_msg = PoseArray()
            points_msg.header = path_msg.header
            for pose_stamped in path_msg.poses:
                points_msg.poses.append(pose_stamped.pose)
            self._points_pub.publish(points_msg)
        return path_msg

    def _publish_viz_markers(self, *, now: object, frame_id: str, centerline: np.ndarray,
                             raw_centerline: np.ndarray, raw_midpoint_chain: np.ndarray,
                             result: Optional[CoreResult], status: str,
                             control_target_frame: Optional[np.ndarray],
                             operator_state: str, operator_reason: str,
                             cmd_speed: float, cmd_steering: float, lookahead: float,
                             candidate_diagonal_count: int, selected_chain_length: int,
                             seed_midpoint_distance_m: float, near_field_lateral_max_m: float,
                             near_field_midpoint_kink_max_rad: float, hold_remaining_s: float) -> None:
        """Build and publish debug visualization markers."""
        status_text = self._build_operator_status_text(
            operator_state=operator_state, operator_reason=operator_reason,
            centerline_point_count=int(centerline.shape[0]),
            cmd_speed=cmd_speed, cmd_steering=cmd_steering, lookahead=lookahead,
            candidate_diagonal_count=candidate_diagonal_count,
            selected_chain_length=selected_chain_length,
            seed_midpoint_distance_m=seed_midpoint_distance_m,
            near_field_lateral_max_m=near_field_lateral_max_m,
            near_field_midpoint_kink_max_rad=near_field_midpoint_kink_max_rad,
            hold_remaining_s=hold_remaining_s,
        )
        markers = self._build_markers(
            now=now, frame_id=frame_id, result=result,
            centerline=centerline, raw_centerline=raw_centerline,
            raw_midpoint_chain=raw_midpoint_chain,
            status=status_text, operator_state=operator_state,
            control_target_frame=control_target_frame,
        )
        self._viz_pub.publish(markers)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackedConePlannerRuntime()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
