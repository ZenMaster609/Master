#!/usr/bin/env python3
"""Delaunay-based centerline planner over tracked cone detections."""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetectionArray
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cones.tracking.fusion import normalize_color, resolve_boundary_colors_for_planning
from sim_car.controllers.factory import create_steering_controller
from sim_car.controllers.stanley_controller import StanleyConfig
from sim_car.planning.delaunay_planner_core import (
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
from sim_car.planning.path_stability import (
    extract_forward_path_from_pose,
    path_cumulative_lengths,
    resample_to_count,
    sample_path_at_lengths,
    splice_frozen_near_field,
)
from sim_car.planning.planner_runtime_types import PlannerIdentity

_OPERATOR_STATE_CODES = {
    'waiting': 0,
    'fresh': 1,
    'held': 2,
    'stopped': 3,
}

_OPERATOR_REASON_CODES = {
    'none': 0,
    'waiting_for_cones': 1,
    'missing_vehicle_pose': 2,
    'cone_transform_unavailable': 3,
    'no_safe_chain': 4,
    'near_field_continuity': 5,
    'seed_distance': 6,
    'midpoint_kink': 7,
    'hysteresis_holding': 8,
    'holding_previous_valid': 9,
    'hold_expired_no_path': 10,
    'no_control_path': 11,
    'controller_compute_failed': 12,
    'stop_if_no_path': 13,
    'controller_disabled': 14,
}

_OPERATOR_REASON_LABELS = {
    'none': 'fresh path accepted',
    'waiting_for_cones': 'waiting for /tracked_cones',
    'missing_vehicle_pose': 'missing vehicle pose',
    'cone_transform_unavailable': 'cone transform unavailable',
    'no_safe_chain': 'no safe fresh chain',
    'near_field_continuity': 'near-field continuity rejected fresh chain',
    'seed_distance': 'seed midpoint too far ahead',
    'midpoint_kink': 'near-field midpoint kink too large',
    'hysteresis_holding': 'holding previous valid path during hysteresis',
    'holding_previous_valid': 'holding previous valid path',
    'hold_expired_no_path': 'held path expired and no fresh path is available',
    'no_control_path': 'no control path available',
    'controller_compute_failed': 'controller compute failed',
    'stop_if_no_path': 'stop_if_no_path sent zero command',
    'controller_disabled': 'controller disabled',
}

_OPERATOR_STATE_COLORS = {
    'waiting': (0.9, 0.9, 0.9),
    'fresh': (0.2, 1.0, 0.3),
    'held': (1.0, 0.9, 0.2),
    'stopped': (1.0, 0.2, 0.2),
}


class DelaunayPlannerNode(Node):
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
            node_name='delaunay_planner_node',
            planner_mode='delaunay',
            diagnostics_prefix='delaunay_planner',
            diagnostics_topic='/delaunay_planner/diagnostics',
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
            'diagnostics.publish_thesis_context': False,
            'debug.enable_markers': True,
            'debug.show_raw_cones': False,
            'debug.show_delaunay_edges': True,
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
            str(self.get_parameter('control.controller_type').value).strip().lower() or 'stanley'
        )
        if self.controller_type not in {'stanley', 'pure_pursuit', 'none'}:
            raise ValueError(
                "Unsupported control.controller_type '%s'. Supported values: stanley, pure_pursuit, none"
                % self.controller_type
            )
        stanley_config = StanleyConfig(
            k_gain=max(0.0, float(self.get_parameter('stanley.k_gain').value)),
            softening_speed_mps=max(0.0, float(self.get_parameter('stanley.softening_speed_mps').value)),
            heading_gain=float(self.get_parameter('stanley.heading_gain').value),
            lookahead_idx_offset=max(0, int(self.get_parameter('stanley.lookahead_idx_offset').value)),
            steering_limit_rad=max(0.01, float(self.get_parameter('stanley.steering_limit_rad').value)),
            steering_lowpass_alpha=float(
                np.clip(float(self.get_parameter('stanley.steering_lowpass_alpha').value), 0.0, 1.0)
            ),
            steering_rate_limit_rad_s=max(
                0.0,
                float(self.get_parameter('stanley.steering_rate_limit_rad_s').value),
            ),
            use_yaw_rate_damping=bool(self.get_parameter('stanley.use_yaw_rate_damping').value),
            yaw_rate_damping_gain=max(
                0.0,
                float(self.get_parameter('stanley.yaw_rate_damping_gain').value),
            ),
            wheelbase_m=max(0.1, float(self.get_parameter('stanley.wheelbase_m').value)),
            cross_track_deadband_m=max(
                0.0,
                float(self.get_parameter('stanley.cross_track_deadband_m').value),
            ),
        )
        self._controller = (
            create_steering_controller(
                controller_type=self.controller_type,
                stanley_config=stanley_config,
                publish_rate_hz=self.publish_rate_hz,
            )
            if self.controller_type == 'stanley'
            else None
        )
        self.stop_if_no_path = bool(self.get_parameter('control.stop_if_no_path').value)
        if self.controller_type == 'pure_pursuit':
            self.get_logger().warn(
                "control.controller_type 'pure_pursuit' is a placeholder; controller output is disabled"
            )
        elif self.controller_type == 'none':
            self.get_logger().info("control.controller_type 'none'; controller output is disabled")

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
        self.publish_thesis_context = bool(
            self.get_parameter('diagnostics.publish_thesis_context').value
        )

        self.enable_debug_markers = bool(self.get_parameter('debug.enable_markers').value)
        self.show_raw_cones = bool(self.get_parameter('debug.show_raw_cones').value)
        self.show_delaunay_edges = bool(self.get_parameter('debug.show_delaunay_edges').value)
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
        control_target_base: Optional[np.ndarray] = None
        control_target_frame: Optional[np.ndarray] = None
        control_debug_metrics: Optional[dict[str, float]] = None
        cmd_speed = 0.0
        cmd_steering = 0.0
        lookahead = 0.0

        cones_msg = self._latest_cones_msg
        if cones_msg is None:
            zero_cmd_sent = int(self._apply_no_path_behavior())
            self._publish_empty_cycle(
                frame_id=self.odom_frame,
                status='waiting for /tracked_cones',
                operator_state='waiting',
                operator_reason='waiting_for_cones',
                cmd_speed=cmd_speed,
                cmd_steering=cmd_steering,
                lookahead=lookahead,
                zero_cmd_sent_flag=zero_cmd_sent,
            )
            return

        source_frame = str(cones_msg.header.frame_id).strip() or self.odom_frame
        target_frame = self._resolve_planning_frame(source_frame, cones_msg.header.stamp)

        pose = self._resolve_vehicle_pose(target_frame, cones_msg.header.stamp)
        if pose is None and target_frame != self.odom_frame:
            self._warn_throttled('fallback_odom', f'cannot resolve base pose in {target_frame}; falling back to odom')
            target_frame = self.odom_frame
            pose = self._resolve_vehicle_pose(target_frame, cones_msg.header.stamp)

        if pose is None:
            zero_cmd_sent = int(self._apply_no_path_behavior())
            self._publish_empty_cycle(
                frame_id=target_frame,
                status='missing vehicle pose (tf and /sim/odom unavailable)',
                operator_state='waiting',
                operator_reason='missing_vehicle_pose',
                cmd_speed=cmd_speed,
                cmd_steering=cmd_steering,
                lookahead=lookahead,
                zero_cmd_sent_flag=zero_cmd_sent,
            )
            return

        vehicle_x, vehicle_y, vehicle_yaw = pose
        points_xy, colors, confidences = self._convert_cones_to_frame(
            cones_msg,
            source_frame,
            target_frame,
            vehicle_xy=(vehicle_x, vehicle_y),
            vehicle_yaw=vehicle_yaw,
        )
        if points_xy is None:
            if target_frame != self.odom_frame:
                self._warn_throttled(
                    'fallback_odom_cones',
                    f'cannot transform cones to {target_frame}; planning in odom frame',
                )
                target_frame = self.odom_frame
                pose = self._resolve_vehicle_pose(target_frame, cones_msg.header.stamp)
                if pose is None:
                    zero_cmd_sent = int(self._apply_no_path_behavior())
                    self._publish_empty_cycle(
                        frame_id=self.odom_frame,
                        status='cone transform failed and odom pose unavailable',
                        operator_state='waiting',
                        operator_reason='cone_transform_unavailable',
                        cmd_speed=cmd_speed,
                        cmd_steering=cmd_steering,
                        lookahead=lookahead,
                        zero_cmd_sent_flag=zero_cmd_sent,
                    )
                    return
                vehicle_x, vehicle_y, vehicle_yaw = pose
                points_xy, colors, confidences = self._convert_cones_to_frame(
                    cones_msg,
                    source_frame,
                    target_frame,
                    vehicle_xy=(vehicle_x, vehicle_y),
                    vehicle_yaw=vehicle_yaw,
                )

        if points_xy is None:
            zero_cmd_sent = int(self._apply_no_path_behavior())
            self._publish_empty_cycle(
                frame_id=target_frame,
                status='cone transform unavailable',
                operator_state='waiting',
                operator_reason='cone_transform_unavailable',
                cmd_speed=cmd_speed,
                cmd_steering=cmd_steering,
                lookahead=lookahead,
                zero_cmd_sent_flag=zero_cmd_sent,
            )
            return

        self._update_remembered_cone_viz(points_xy=points_xy, colors=colors)
        result = compute_centerline(
            points_xy=points_xy,
            colors=colors,
            confidences=confidences,
            vehicle_xy=(vehicle_x, vehicle_y),
            vehicle_yaw=vehicle_yaw,
            config=self._core_config,
            prior=CorePrior(
                previous_midpoints_raw=(
                    np.array(self._last_valid_raw_midpoint_chain, copy=True)
                    if self._last_valid_raw_midpoint_chain is not None
                    else None
                ),
                previous_width_m=self._last_valid_width_m,
            ),
        )
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        tracked_delta_p95_m = tracked_cones_frame_delta_p95(self._previous_tracked_points, points_xy)
        self._previous_tracked_points = np.array(points_xy, copy=True)

        previous_edge_keys = set(self._previous_edge_keys)
        selected_keys = selected_edge_keys(
            points=result.filtered_points,
            edges=result.selected_edges,
            quantization_m=self.edge_quantization_m,
        )
        selected_edge_churn_count = edge_churn_count(previous_edge_keys, selected_keys)
        selected_edge_churn = edge_churn_ratio(previous_edge_keys, selected_keys)
        self._previous_edge_keys = set(selected_keys)

        raw_centerline = result.centerline
        raw_midpoint_chain = result.midpoints_raw
        centerline_jump_max_m = compute_centerline_jump_max(
            raw_centerline,
            self._previous_raw_centerline,
            self.centerline_jump_horizon_m,
        )
        self._previous_raw_centerline = np.array(raw_centerline, copy=True) if raw_centerline.shape[0] > 0 else None

        if centerline_jump_max_m > self.jump_warn_threshold_m:
            self._warn_throttled(
                'centerline_jump_warn',
                f'centerline jump {centerline_jump_max_m:.3f} m exceeded threshold {self.jump_warn_threshold_m:.3f} m',
            )
        if selected_edge_churn > self.edge_churn_warn_threshold:
            self._warn_throttled(
                'edge_churn_warn',
                f'selected edge churn {selected_edge_churn:.3f} exceeded threshold {self.edge_churn_warn_threshold:.3f}',
            )

        centerline = raw_centerline
        status = result.status
        plan_valid = False
        plan_hold_active = False
        publish_mode = 'fresh'
        hold_reason = result.reject_reason
        zero_cmd_sent_flag = 0
        controller_failed = False
        near_field_freeze_active = False
        committed_near_field_active = False

        # The first few meters of the published midline must remain laterally stable frame-to-frame.
        # A longer path is not better if the near-field segment teleports sideways.
        if centerline.shape[0] > 0:
            plan_ok = True
            if plan_ok:
                plan_valid = True
                continue_holding = self._advance_hold_hysteresis(plan_ok=True)
                if continue_holding:
                    held_centerline = self._held_centerline(now_sec)
                    if held_centerline is not None:
                        centerline = held_centerline
                        plan_hold_active = True
                        publish_mode = 'held'
                        status = (
                            f'{status}; hysteresis holding previous valid centerline '
                            f'({self._hold_clean_frame_count}/{self.hold_exit_clean_frames})'
                        )
                if not plan_hold_active:
                    publish_mode = 'fresh'
            else:
                self._advance_hold_hysteresis(plan_ok=False)
                held_centerline = self._held_centerline(now_sec)
                if held_centerline is not None:
                    centerline = held_centerline
                    plan_hold_active = True
                    publish_mode = 'held'
                    status = f'{status}; holding previous valid centerline'
                else:
                    centerline = np.empty((0, 2), dtype=np.float64)
                    publish_mode = 'held'
                    status = f'{status}; rejected unstable centerline'
        else:
            self._advance_hold_hysteresis(plan_ok=False)
            held_centerline = self._held_centerline(now_sec)
            if held_centerline is not None:
                centerline = held_centerline
                plan_hold_active = True
                publish_mode = 'held'
                status = f'{status}; holding previous valid centerline'
            else:
                publish_mode = 'held'

        if plan_hold_active:
            hold_reason = result.reject_reason or status

        if self.enable_temporal_smoothing and publish_mode == 'fresh':
            centerline = self._apply_temporal_smoothing(centerline)
            self._previous_centerline = np.array(centerline, copy=True)
        if publish_mode == 'fresh':
            self._update_committed_near_field(
                centerline=centerline,
                selected_edge_churn=selected_edge_churn,
                selected_chain_length=int(result.selected_chain_length),
                previous_edge_keys=previous_edge_keys,
            )
            centerline, near_field_freeze_active, committed_near_field_active = self._apply_near_field_freeze(
                centerline,
                frame_id=target_frame,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
            )
            if centerline.shape[0] > 0:
                self._previous_centerline = np.array(centerline, copy=True)
                self._record_valid_plan(
                    now_sec=now_sec,
                    centerline=centerline,
                    raw_midpoint_chain=raw_midpoint_chain,
                    selected_chain_width_median=result.selected_chain_width_median,
                )
                if near_field_freeze_active:
                    status = (
                        f'{status}; frozen near-field '
                        f'{self.freeze_near_field_m:.1f} m before blending into fresh far field'
                    )
                if committed_near_field_active:
                    status = (
                        f'{status}; committed near-field '
                        f'{self.commit_plan_horizon_m:.1f} m trusted over high-churn refresh'
                    )
        elif publish_mode == 'held' and centerline.shape[0] > 0:
            self._previous_centerline = np.array(centerline, copy=True)

        control_path = self._centerline_to_vehicle_frame(
            centerline=centerline,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        if control_path.shape[0] >= 1 and self._controller is not None:
            try:
                controller_output = self._controller.compute(
                    control_path=control_path,
                    speed_mps=self._latest_speed_mps,
                    yaw_rate_rps=self._latest_yaw_rate_rps,
                )
            except ValueError as exc:
                self._warn_throttled('controller_compute_error', f'controller compute failed: {exc}')
                controller_failed = True
                zero_cmd_sent_flag = int(self._apply_no_path_behavior())
                if self._last_speed_cmd is not None:
                    cmd_speed = float(self._last_speed_cmd)
                if self._last_steering_cmd is not None:
                    cmd_steering = float(self._last_steering_cmd)
            else:
                cmd_steering = float(controller_output.steering_rad)
                cmd_speed = self._compute_speed_command(float(controller_output.kappa))
                lookahead = float(controller_output.lookahead_m)
                control_target_base = np.asarray(controller_output.target_point_base, dtype=np.float64)
                self._last_speed_cmd = cmd_speed
                self._last_steering_cmd = cmd_steering
                self._publish_cmd(cmd_speed, cmd_steering)

                if self._is_alias(target_frame, self.base_frame):
                    control_target_frame = np.array(control_target_base, copy=True)
                elif self._is_alias(target_frame, self.odom_frame):
                    tx, ty = self._base_point_to_odom(
                        float(control_target_base[0]),
                        float(control_target_base[1]),
                        vehicle_x,
                        vehicle_y,
                        vehicle_yaw,
                    )
                    control_target_frame = np.array([tx, ty], dtype=np.float64)
                if controller_output.stanley_debug is not None:
                    debug = controller_output.stanley_debug
                    control_debug_metrics = {
                        'heading_error_rad': float(debug.heading_error_rad),
                        'cross_track_error_m': float(debug.cross_track_error_m),
                        'vehicle_speed_mps': float(debug.vehicle_speed_mps),
                        'speed_term_mps': float(debug.speed_term_mps),
                        'heading_contribution_rad': float(debug.heading_contribution_rad),
                        'cross_track_contribution_rad': float(debug.cross_track_contribution_rad),
                        'yaw_rate_damping_contribution_rad': float(
                            debug.yaw_rate_damping_contribution_rad
                        ),
                        'yaw_rate_rps': float(self._latest_yaw_rate_rps),
                        'raw_steering_cmd_rad': float(debug.raw_steering_cmd_rad),
                        'steering_after_clamp_rad': float(debug.steering_after_clamp_rad),
                        'steering_after_filter_rad': float(debug.steering_after_filter_rad),
                        'steering_after_rate_limit_rad': float(debug.steering_after_rate_limit_rad),
                        'final_steering_cmd_rad': float(debug.final_steering_cmd_rad),
                        'steering_saturated_flag': (
                            1.0 if bool(debug.steering_saturated_flag) else 0.0
                        ),
                        'nearest_path_index': float(debug.nearest_path_index),
                        'heading_path_index': float(debug.heading_path_index),
                        'target_point_x_base_m': float(debug.target_point_x_base_m),
                        'target_point_y_base_m': float(debug.target_point_y_base_m),
                        'target_point_x_frame_m': (
                            float(control_target_frame[0])
                            if control_target_frame is not None
                            else float('nan')
                        ),
                        'target_point_y_frame_m': (
                            float(control_target_frame[1])
                            if control_target_frame is not None
                            else float('nan')
                        ),
                    }
        elif control_path.shape[0] >= 1:
            zero_cmd_sent_flag = int(self._apply_controller_disabled_behavior())
        else:
            zero_cmd_sent_flag = int(self._apply_no_path_behavior())
            if self._last_speed_cmd is not None:
                cmd_speed = float(self._last_speed_cmd)
            if self._last_steering_cmd is not None:
                cmd_steering = float(self._last_steering_cmd)

        control_path_point_count = int(control_path.shape[0])
        hold_remaining_s = self._hold_remaining_s(now_sec)
        operator_state = 'fresh'
        operator_reason = 'none'
        core_reject_reason = self._normalize_core_reject_reason(result)
        if centerline.shape[0] == 0:
            if self._last_valid_centerline is not None and hold_remaining_s <= 0.0:
                operator_state = 'stopped'
                operator_reason = 'hold_expired_no_path'
            elif core_reject_reason != 'none':
                operator_state = 'stopped'
                operator_reason = core_reject_reason
            elif zero_cmd_sent_flag:
                operator_state = 'stopped'
                operator_reason = 'stop_if_no_path'
            else:
                operator_state = 'held'
                operator_reason = 'holding_previous_valid'
        elif publish_mode == 'held':
            operator_state = 'held'
            if centerline.shape[0] > 0 and 'hysteresis holding previous valid centerline' in status:
                operator_reason = 'hysteresis_holding'
            elif core_reject_reason != 'none':
                operator_reason = core_reject_reason
            else:
                operator_reason = 'holding_previous_valid'
        else:
            operator_state = 'fresh'
            operator_reason = 'none'

        if controller_failed:
            operator_state = 'stopped'
            operator_reason = 'controller_compute_failed'
        elif self._controller is None and centerline.shape[0] > 0:
            operator_state = 'stopped'
            operator_reason = 'controller_disabled'
        elif centerline.shape[0] > 0 and control_path_point_count <= 0 and operator_state != 'waiting':
            operator_state = 'stopped'
            operator_reason = 'no_control_path'
        if zero_cmd_sent_flag and operator_reason == 'none':
            operator_state = 'stopped'
            operator_reason = 'stop_if_no_path'
        if control_path_point_count > 0 and centerline.shape[0] > 0 and zero_cmd_sent_flag == 0 and publish_mode == 'fresh':
            operator_state = 'fresh'

        self._log_operator_state_transition(
            operator_state=operator_state,
            operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            selected_chain_length=int(result.selected_chain_length),
        )
        self._active_planner_mode = self._planner_identity.planner_mode

        self._publish_diagnostics(
            frame_id=target_frame,
            centerline_jump_max_m=centerline_jump_max_m,
            selected_edge_churn_ratio=selected_edge_churn,
            tracked_cones_frame_delta_p95_m=tracked_delta_p95_m,
            centerline_point_count=int(centerline.shape[0]),
            selected_edge_count=int(result.selected_edges.shape[0]),
            status=status,
            control_debug_metrics=control_debug_metrics,
            thesis_context_metrics={
                'plan_valid_flag': 1.0 if plan_valid else 0.0,
                'plan_hold_active_flag': 1.0 if plan_hold_active else 0.0,
                'plan_fallback_flag': 1.0 if bool(result.used_fallback) else 0.0,
                'path_length_m': self._path_length_m(centerline),
                'path_curvature_abs_p95_1pm': self._path_curvature_abs_p95(centerline),
            },
            planner_metrics={
                'candidate_diagonal_count': result.candidate_count,
                'selected_chain_length': result.selected_chain_length,
                'selected_chain_median_width_m': result.selected_chain_width_median,
                'expected_width_prior_m': result.expected_width_prior_m,
                'reject_wrong_side_count': result.reject_counts.get('wrong_side', 0),
                'reject_width_count': result.reject_counts.get('width', 0),
                'reject_width_range_count': result.reject_counts.get('width_range', 0),
                'reject_width_prior_count': result.reject_counts.get('width_prior', 0),
                'reject_orientation_count': result.reject_counts.get('orientation', 0),
                'reject_progress_count': result.reject_counts.get('progress', 0),
                'reject_near_field_continuity_count': result.reject_counts.get('near_field_continuity', 0),
                'reject_midpoint_kink_count': result.reject_counts.get('midpoint_kink', 0),
                'reject_seed_distance_count': result.reject_counts.get('seed_distance', 0),
                'near_field_lateral_max_m': result.near_field_lateral_max_m,
                'near_field_lateral_mean_m': result.near_field_lateral_mean_m,
                'near_field_displacement_max_m': result.near_field_displacement_max_m,
                'near_field_displacement_mean_m': result.near_field_displacement_mean_m,
                'near_field_midpoint_kink_max_rad': result.near_field_kink_max_rad,
                'seed_midpoint_distance_m': result.seed_midpoint_distance_m,
                'seed_temporal_offset_m': result.seed_temporal_offset_m,
                'selected_chain_churn_count': selected_edge_churn_count,
                'selected_chain_churn_ratio': selected_edge_churn,
                'publish_mode': publish_mode,
                'hold_mode_active_flag': 1 if self._hold_mode_active else 0,
                'hold_clean_frame_count': self._hold_clean_frame_count,
                'hold_reason': hold_reason or '',
                'committed_near_field_active_flag': 1 if committed_near_field_active else 0,
                'commit_stable_frame_count': self._commit_stable_frame_count,
                'near_field_freeze_active_flag': 1 if near_field_freeze_active else 0,
                'planner_state_code': self._operator_state_code(operator_state),
                'fresh_publish_flag': 1 if operator_state == 'fresh' else 0,
                'held_publish_flag': 1 if operator_state == 'held' else 0,
                'stopped_flag': 1 if operator_state == 'stopped' else 0,
                'waiting_flag': 1 if operator_state == 'waiting' else 0,
                'operator_reason_code': self._operator_reason_code(operator_reason),
                'operator_reason': operator_reason,
                'hold_remaining_s': hold_remaining_s,
                'control_path_point_count': control_path_point_count,
                'zero_cmd_sent_flag': zero_cmd_sent_flag,
                'planner_mode': self._active_planner_mode,
            },
        )

        self._publish_outputs(
            frame_id=target_frame,
            centerline=centerline,
            raw_centerline=raw_centerline,
            raw_midpoint_chain=raw_midpoint_chain,
            result=result,
            status=status,
            control_target_frame=control_target_frame,
            cmd_speed=cmd_speed,
            cmd_steering=cmd_steering,
            lookahead=lookahead,
            operator_state=operator_state,
            operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            control_path_point_count=control_path_point_count,
            candidate_diagonal_count=int(result.candidate_count),
            selected_chain_length=int(result.selected_chain_length),
            seed_midpoint_distance_m=float(result.seed_midpoint_distance_m),
            near_field_lateral_max_m=float(result.near_field_lateral_max_m),
            near_field_midpoint_kink_max_rad=float(result.near_field_kink_max_rad),
        )

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
            xb, yb = self._odom_point_to_base(centerline[idx, 0], centerline[idx, 1], vehicle_x, vehicle_y, vehicle_yaw)
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

    def _apply_no_path_behavior(self) -> bool:
        if self.stop_if_no_path:
            self._publish_cmd(0.0, 0.0)
            self._last_speed_cmd = 0.0
            self._last_steering_cmd = 0.0
            return True
        if self._last_speed_cmd is not None and self._last_steering_cmd is not None:
            self._publish_cmd(float(self._last_speed_cmd), float(self._last_steering_cmd))
        return False

    def _apply_controller_disabled_behavior(self) -> bool:
        if self.stop_if_no_path:
            self._publish_cmd(0.0, 0.0)
            self._last_speed_cmd = 0.0
            self._last_steering_cmd = 0.0
            return True
        return False

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
            yaw = self._yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
            return tx, ty, yaw

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
        yaw = self._yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        return self._convert_odom_child_pose_to_base_frame(
            child_frame=str(odom.child_frame_id).strip(),
            tx=tx,
            ty=ty,
            yaw=yaw,
        )

    def _convert_odom_child_pose_to_base_frame(
        self,
        *,
        child_frame: str,
        tx: float,
        ty: float,
        yaw: float,
    ) -> Optional[tuple[float, float, float]]:
        if self._is_alias(child_frame, self.base_frame):
            return tx, ty, yaw

        child_is_body_center = self._is_alias(child_frame, 'base_footprint') or self._is_alias(child_frame, 'base_link')
        base_is_front_axle = self._is_alias(self.base_frame, 'front_axle')
        if child_is_body_center and base_is_front_axle:
            front_axle_offset_m = 0.5 * self._configured_wheelbase_m()
            x_front, y_front = self._base_point_to_odom(front_axle_offset_m, 0.0, tx, ty, yaw)
            return x_front, y_front, yaw

        if self._is_alias(child_frame, 'front_axle') and child_is_body_center:
            return tx, ty, yaw

        return None

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
                x, y, z = self._transform_point(tf_msg, x, y, z)
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
        target_is_base = self._is_alias(target_frame, self.base_frame)
        source_is_base = self._is_alias(source_frame, self.base_frame)
        source_is_odom = self._is_alias(source_frame, self.odom_frame)

        if source_is_odom and target_is_base:
            points: list[tuple[float, float]] = []
            raw_colors: list[str] = []
            boundary_hints: list[str] = []
            confs: list[float] = []
            for cone in msg.cones:
                x_base, y_base = self._odom_point_to_base(
                    float(cone.position.x),
                    float(cone.position.y),
                    odom_x,
                    odom_y,
                    odom_yaw,
                )
                points.append((x_base, y_base))
                raw_colors.append(normalize_color(cone.color))
                boundary_hints.append(str(getattr(cone, 'boundary_color', '')).strip())
                confs.append(float(np.clip(float(cone.confidence), 0.0, 1.0)))
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

        if source_is_base and self._is_alias(target_frame, self.odom_frame):
            cos_y = math.cos(odom_yaw)
            sin_y = math.sin(odom_yaw)
            points = []
            raw_colors = []
            boundary_hints = []
            confs = []
            for cone in msg.cones:
                xb = float(cone.position.x)
                yb = float(cone.position.y)
                xo = odom_x + (cos_y * xb) - (sin_y * yb)
                yo = odom_y + (sin_y * xb) + (cos_y * yb)
                points.append((xo, yo))
                raw_colors.append(normalize_color(cone.color))
                boundary_hints.append(str(getattr(cone, 'boundary_color', '')).strip())
                confs.append(float(np.clip(float(cone.confidence), 0.0, 1.0)))
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

        return None

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

    def _update_committed_near_field(
        self,
        *,
        centerline: np.ndarray,
        selected_edge_churn: float,
        selected_chain_length: int,
        previous_edge_keys: set[tuple[int, int, int, int]],
    ) -> None:
        if not self.enable_committed_near_field or self.commit_plan_horizon_m <= 0.0:
            return
        if centerline.shape[0] < 2 or selected_chain_length <= 0:
            self._commit_stable_frame_count = 0
            return

        stable_enough = (not previous_edge_keys) or (
            selected_edge_churn <= self.commit_update_max_churn_ratio
        )
        if not stable_enough:
            self._commit_stable_frame_count = 0
            return

        self._commit_stable_frame_count += 1
        if self._commit_stable_frame_count < self.commit_stable_frames:
            return

        self._committed_centerline = np.array(centerline, copy=True)

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

    def _record_valid_plan(
        self,
        *,
        now_sec: float,
        centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        selected_chain_width_median: float,
    ) -> None:
        self._last_valid_centerline = np.array(centerline, copy=True)
        self._last_valid_raw_midpoint_chain = (
            np.array(raw_midpoint_chain, copy=True)
            if raw_midpoint_chain.shape[0] > 0
            else None
        )
        self._last_valid_width_m = (
            float(selected_chain_width_median)
            if math.isfinite(float(selected_chain_width_median))
            else self._last_valid_width_m
        )
        self._last_valid_time_sec = now_sec

    def _advance_hold_hysteresis(self, *, plan_ok: bool) -> bool:
        if not plan_ok:
            self._hold_mode_active = True
            self._hold_clean_frame_count = 0
            return True
        # Re-enter fresh publishing only after consecutive clean frames so borderline cases do not
        # flap between held and fresh controller inputs.
        if not self._hold_mode_active:
            return False
        self._hold_clean_frame_count += 1
        if self._hold_clean_frame_count < self.hold_exit_clean_frames:
            return True
        self._hold_mode_active = False
        self._hold_clean_frame_count = 0
        return False

    def _held_centerline(self, now_sec: float) -> Optional[np.ndarray]:
        if self._last_valid_centerline is None:
            return None
        if self._last_valid_time_sec < 0.0:
            return None
        if (now_sec - self._last_valid_time_sec) > self.hold_last_valid_s:
            return None
        return np.array(self._last_valid_centerline, copy=True)

    def _hold_remaining_s(self, now_sec: float) -> float:
        if self._last_valid_centerline is None or self._last_valid_time_sec < 0.0:
            return 0.0
        return max(0.0, float(self.hold_last_valid_s) - max(0.0, now_sec - self._last_valid_time_sec))

    @staticmethod
    def _operator_state_code(state: str) -> int:
        return int(_OPERATOR_STATE_CODES.get(state, 0))

    @staticmethod
    def _operator_reason_code(reason: str) -> int:
        return int(_OPERATOR_REASON_CODES.get(reason, 0))

    @staticmethod
    def _operator_reason_label(reason: str) -> str:
        return _OPERATOR_REASON_LABELS.get(reason, reason.replace('_', ' '))

    @staticmethod
    def _operator_state_color(state: str) -> tuple[float, float, float]:
        return _OPERATOR_STATE_COLORS.get(state, _OPERATOR_STATE_COLORS['waiting'])

    def _normalize_core_reject_reason(self, result: Optional[CoreResult]) -> str:
        if result is None:
            return 'none'
        reject_counts = result.reject_counts or {}
        if int(reject_counts.get('near_field_continuity', 0)) > 0:
            return 'near_field_continuity'
        if int(reject_counts.get('seed_distance', 0)) > 0:
            return 'seed_distance'
        if int(reject_counts.get('midpoint_kink', 0)) > 0:
            return 'midpoint_kink'
        text = (result.reject_reason or result.status or '').strip().lower()
        if text in {
            'no valid diagonal candidates',
            'no safe zig-zag chain',
            'centerline generation failed',
            'dedup removed all selected midpoints',
            'no cones available',
        }:
            return 'no_safe_chain'
        if text.startswith('usable cones below minimum'):
            return 'no_safe_chain'
        return 'none'

    def _log_operator_state_transition(
        self,
        *,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
        selected_chain_length: int,
    ) -> None:
        if operator_state == self._last_operator_state and operator_reason == self._last_operator_reason:
            return
        prev_state = self._last_operator_state or 'none'
        prev_reason = self._last_operator_reason or 'none'
        self.get_logger().info(
            'planner state %s -> %s | reason=%s | prev_reason=%s | hold_remaining=%.2fs | chain=%d | clean_frames=%d/%d'
            % (
                prev_state,
                operator_state,
                operator_reason,
                prev_reason,
                float(hold_remaining_s),
                int(selected_chain_length),
                int(self._hold_clean_frame_count),
                int(self.hold_exit_clean_frames),
            )
        )
        self._last_operator_state = operator_state
        self._last_operator_reason = operator_reason

    @staticmethod
    def _fmt_metric(value: float, suffix: str = '', *, digits: int = 2) -> str:
        value_f = float(value)
        if not math.isfinite(value_f):
            return 'n/a'
        return f'{value_f:.{digits}f}{suffix}'

    def _build_operator_status_text(
        self,
        *,
        operator_state: str,
        operator_reason: str,
        centerline_point_count: int,
        cmd_speed: float,
        cmd_steering: float,
        lookahead: float,
        candidate_diagonal_count: int,
        selected_chain_length: int,
        seed_midpoint_distance_m: float,
        near_field_lateral_max_m: float,
        near_field_midpoint_kink_max_rad: float,
        hold_remaining_s: float,
    ) -> str:
        return '\n'.join(
            [
                f'STATE: {operator_state.upper()}',
                f'REASON: {self._operator_reason_label(operator_reason)}',
                (
                    f'PATH: {int(centerline_point_count)} pts | '
                    f'chain={int(selected_chain_length)} | cand={int(candidate_diagonal_count)}'
                ),
                (
                    f'CMD: v={self._fmt_metric(cmd_speed, " m/s")} | '
                    f'delta={self._fmt_metric(cmd_steering, " rad")} | '
                    f'Ld={self._fmt_metric(lookahead, " m")}'
                ),
                (
                    f'NF: lat={self._fmt_metric(near_field_lateral_max_m, " m")} | '
                    f'kink={self._fmt_metric(near_field_midpoint_kink_max_rad, " rad")} | '
                    f'seed={self._fmt_metric(seed_midpoint_distance_m, " m")}'
                ),
                (
                    f'HOLD: {self._fmt_metric(hold_remaining_s, " s")} left | '
                    f'clean {int(self._hold_clean_frame_count)}/{int(self.hold_exit_clean_frames)}'
                ),
            ]
        )

    def _publish_empty_cycle(
        self,
        *,
        frame_id: str,
        status: str,
        operator_state: str,
        operator_reason: str,
        cmd_speed: float,
        cmd_steering: float,
        lookahead: float,
        zero_cmd_sent_flag: int,
    ) -> None:
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        hold_remaining_s = self._hold_remaining_s(now_sec)
        self._log_operator_state_transition(
            operator_state=operator_state,
            operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            selected_chain_length=0,
        )
        self._publish_diagnostics(
            frame_id=frame_id,
            centerline_jump_max_m=0.0,
            selected_edge_churn_ratio=0.0,
            tracked_cones_frame_delta_p95_m=0.0,
            centerline_point_count=0,
            selected_edge_count=0,
            status=status,
            planner_metrics={
                'candidate_diagonal_count': 0,
                'selected_chain_length': 0,
                'selected_chain_median_width_m': float('nan'),
                'expected_width_prior_m': float('nan'),
                'reject_wrong_side_count': 0,
                'reject_width_count': 0,
                'reject_width_range_count': 0,
                'reject_width_prior_count': 0,
                'reject_orientation_count': 0,
                'reject_progress_count': 0,
                'reject_near_field_continuity_count': 0,
                'reject_midpoint_kink_count': 0,
                'reject_seed_distance_count': 0,
                'near_field_lateral_max_m': float('nan'),
                'near_field_lateral_mean_m': float('nan'),
                'near_field_displacement_max_m': float('nan'),
                'near_field_displacement_mean_m': float('nan'),
                'near_field_midpoint_kink_max_rad': float('nan'),
                'seed_midpoint_distance_m': float('nan'),
                'seed_temporal_offset_m': float('nan'),
                'selected_chain_churn_count': 0,
                'selected_chain_churn_ratio': 0.0,
                'publish_mode': operator_state if operator_state in {'fresh', 'held'} else 'held',
                'hold_mode_active_flag': 1 if self._hold_mode_active else 0,
                'hold_clean_frame_count': self._hold_clean_frame_count,
                'hold_reason': '',
                'planner_state_code': self._operator_state_code(operator_state),
                'fresh_publish_flag': 1 if operator_state == 'fresh' else 0,
                'held_publish_flag': 1 if operator_state == 'held' else 0,
                'stopped_flag': 1 if operator_state == 'stopped' else 0,
                'waiting_flag': 1 if operator_state == 'waiting' else 0,
                'operator_reason_code': self._operator_reason_code(operator_reason),
                'operator_reason': operator_reason,
                'hold_remaining_s': hold_remaining_s,
                'control_path_point_count': 0,
                'zero_cmd_sent_flag': int(zero_cmd_sent_flag),
                'planner_mode': self._active_planner_mode,
            },
        )
        self._publish_outputs(
            frame_id=frame_id,
            centerline=np.empty((0, 2), dtype=np.float64),
            raw_centerline=np.empty((0, 2), dtype=np.float64),
            raw_midpoint_chain=np.empty((0, 2), dtype=np.float64),
            result=None,
            status=status,
            control_target_frame=None,
            cmd_speed=cmd_speed,
            cmd_steering=cmd_steering,
            lookahead=lookahead,
            operator_state=operator_state,
            operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            control_path_point_count=0,
            candidate_diagonal_count=0,
            selected_chain_length=0,
            seed_midpoint_distance_m=float('nan'),
            near_field_lateral_max_m=float('nan'),
            near_field_midpoint_kink_max_rad=float('nan'),
        )

    def _publish_diagnostics(
        self,
        *,
        frame_id: str,
        centerline_jump_max_m: float,
        selected_edge_churn_ratio: float,
        tracked_cones_frame_delta_p95_m: float,
        centerline_point_count: int,
        selected_edge_count: int,
        status: str,
        control_debug_metrics: Optional[dict[str, float]] = None,
        thesis_context_metrics: Optional[dict[str, float]] = None,
        planner_metrics: Optional[dict[str, object]] = None,
    ) -> None:
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        diag = DiagnosticStatus()
        diag.name = f'{self._planner_identity.diagnostics_prefix}/stability'
        diag.hardware_id = self._planner_identity.hardware_id
        diag.level = DiagnosticStatus.OK
        diag.message = status
        diag.values = [
            KeyValue(key='centerline_jump_max_m', value=f'{centerline_jump_max_m:.6f}'),
            KeyValue(key='selected_edge_churn_ratio', value=f'{selected_edge_churn_ratio:.6f}'),
            KeyValue(key='tracked_cones_frame_delta_p95_m', value=f'{tracked_cones_frame_delta_p95_m:.6f}'),
            KeyValue(key='centerline_point_count', value=str(int(centerline_point_count))),
            KeyValue(key='selected_edge_count', value=str(int(selected_edge_count))),
        ]
        if planner_metrics is not None:
            for key in sorted(planner_metrics):
                diag.values.append(KeyValue(key=str(key), value=self._diag_value_string(planner_metrics[key])))
        msg.status.append(diag)

        if self.publish_control_debug:
            merged = self._default_control_debug_metrics()
            if control_debug_metrics is not None:
                merged.update(control_debug_metrics)
            control_diag = DiagnosticStatus()
            control_diag.name = f'{self._planner_identity.diagnostics_prefix}/control_debug'
            control_diag.hardware_id = self._planner_identity.hardware_id
            control_diag.level = DiagnosticStatus.OK
            control_diag.message = 'stanley debug signals'
            control_diag.values = [
                KeyValue(key='heading_error_rad', value=f"{merged['heading_error_rad']:.6f}"),
                KeyValue(key='cross_track_error_m', value=f"{merged['cross_track_error_m']:.6f}"),
                KeyValue(key='vehicle_speed_mps', value=f"{merged['vehicle_speed_mps']:.6f}"),
                KeyValue(key='speed_term_mps', value=f"{merged['speed_term_mps']:.6f}"),
                KeyValue(key='heading_contribution_rad', value=f"{merged['heading_contribution_rad']:.6f}"),
                KeyValue(
                    key='cross_track_contribution_rad',
                    value=f"{merged['cross_track_contribution_rad']:.6f}",
                ),
                KeyValue(
                    key='yaw_rate_damping_contribution_rad',
                    value=f"{merged['yaw_rate_damping_contribution_rad']:.6f}",
                ),
                KeyValue(key='yaw_rate_rps', value=f"{merged['yaw_rate_rps']:.6f}"),
                KeyValue(key='raw_steering_cmd_rad', value=f"{merged['raw_steering_cmd_rad']:.6f}"),
                KeyValue(
                    key='steering_after_clamp_rad',
                    value=f"{merged['steering_after_clamp_rad']:.6f}",
                ),
                KeyValue(
                    key='steering_after_filter_rad',
                    value=f"{merged['steering_after_filter_rad']:.6f}",
                ),
                KeyValue(
                    key='steering_after_rate_limit_rad',
                    value=f"{merged['steering_after_rate_limit_rad']:.6f}",
                ),
                KeyValue(key='final_steering_cmd_rad', value=f"{merged['final_steering_cmd_rad']:.6f}"),
                KeyValue(
                    key='steering_saturated_flag',
                    value=str(int(round(merged['steering_saturated_flag']))),
                ),
                KeyValue(key='nearest_path_index', value=str(int(round(merged['nearest_path_index'])))),
                KeyValue(key='heading_path_index', value=str(int(round(merged['heading_path_index'])))),
                KeyValue(
                    key='target_point_x_base_m',
                    value=f"{merged['target_point_x_base_m']:.6f}",
                ),
                KeyValue(
                    key='target_point_y_base_m',
                    value=f"{merged['target_point_y_base_m']:.6f}",
                ),
                KeyValue(
                    key='target_point_x_frame_m',
                    value=f"{merged['target_point_x_frame_m']:.6f}",
                ),
                KeyValue(
                    key='target_point_y_frame_m',
                    value=f"{merged['target_point_y_frame_m']:.6f}",
                ),
            ]
            msg.status.append(control_diag)
        if self.publish_thesis_context:
            merged_thesis = self._default_thesis_context_metrics()
            if thesis_context_metrics is not None:
                merged_thesis.update(thesis_context_metrics)
            thesis_diag = DiagnosticStatus()
            thesis_diag.name = f'{self._planner_identity.diagnostics_prefix}/thesis_context'
            thesis_diag.hardware_id = self._planner_identity.hardware_id
            thesis_diag.level = DiagnosticStatus.OK
            thesis_diag.message = 'planner context for controller thesis diagnostics'
            thesis_diag.values = [
                KeyValue(
                    key='plan_valid_flag',
                    value=str(int(round(merged_thesis['plan_valid_flag']))),
                ),
                KeyValue(
                    key='plan_hold_active_flag',
                    value=str(int(round(merged_thesis['plan_hold_active_flag']))),
                ),
                KeyValue(
                    key='plan_fallback_flag',
                    value=str(int(round(merged_thesis['plan_fallback_flag']))),
                ),
                KeyValue(key='centerline_point_count', value=str(int(centerline_point_count))),
                KeyValue(key='selected_edge_count', value=str(int(selected_edge_count))),
                KeyValue(key='centerline_jump_max_m', value=f'{centerline_jump_max_m:.6f}'),
                KeyValue(key='selected_edge_churn_ratio', value=f'{selected_edge_churn_ratio:.6f}'),
                KeyValue(
                    key='tracked_cones_frame_delta_p95_m',
                    value=f'{tracked_cones_frame_delta_p95_m:.6f}',
                ),
                KeyValue(key='path_length_m', value=f"{merged_thesis['path_length_m']:.6f}"),
                KeyValue(
                    key='path_curvature_abs_p95_1pm',
                    value=f"{merged_thesis['path_curvature_abs_p95_1pm']:.6f}",
                ),
            ]
            msg.status.append(thesis_diag)
        self._diag_pub.publish(msg)

    @staticmethod
    def _default_control_debug_metrics() -> dict[str, float]:
        nan = float('nan')
        return {
            'heading_error_rad': nan,
            'cross_track_error_m': nan,
            'vehicle_speed_mps': nan,
            'speed_term_mps': nan,
            'heading_contribution_rad': nan,
            'cross_track_contribution_rad': nan,
            'yaw_rate_damping_contribution_rad': nan,
            'yaw_rate_rps': nan,
            'raw_steering_cmd_rad': nan,
            'steering_after_clamp_rad': nan,
            'steering_after_filter_rad': nan,
            'steering_after_rate_limit_rad': nan,
            'final_steering_cmd_rad': nan,
            'steering_saturated_flag': 0.0,
            'nearest_path_index': -1.0,
            'heading_path_index': -1.0,
            'target_point_x_base_m': nan,
            'target_point_y_base_m': nan,
            'target_point_x_frame_m': nan,
            'target_point_y_frame_m': nan,
        }

    @staticmethod
    def _diag_value_string(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            value_f = float(value)
            return f'{value_f:.6f}' if math.isfinite(value_f) else 'nan'
        return str(value)

    @staticmethod
    def _default_thesis_context_metrics() -> dict[str, float]:
        nan = float('nan')
        return {
            'plan_valid_flag': 0.0,
            'plan_hold_active_flag': 0.0,
            'plan_fallback_flag': 0.0,
            'path_length_m': nan,
            'path_curvature_abs_p95_1pm': nan,
        }

    @staticmethod
    def _path_length_m(centerline: np.ndarray) -> float:
        if centerline.shape[0] < 2:
            return float('nan')
        diffs = np.diff(centerline, axis=0)
        lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        if lengths.size == 0:
            return float('nan')
        return float(np.sum(lengths))

    @staticmethod
    def _path_curvature_abs_p95(centerline: np.ndarray) -> float:
        if centerline.shape[0] < 3:
            return float('nan')
        diffs = np.diff(centerline, axis=0)
        seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
        valid = seg_len > 1e-6
        if np.count_nonzero(valid) < 2:
            return float('nan')
        headings = np.arctan2(diffs[valid, 1], diffs[valid, 0])
        if headings.size < 2:
            return float('nan')
        delta_heading = np.arctan2(
            np.sin(np.diff(headings)),
            np.cos(np.diff(headings)),
        )
        ds = 0.5 * (seg_len[valid][1:] + seg_len[valid][:-1])
        valid_ds = ds > 1e-6
        if not np.any(valid_ds):
            return float('nan')
        curvature = np.abs(delta_heading[valid_ds] / ds[valid_ds])
        if curvature.size == 0:
            return float('nan')
        return float(np.percentile(curvature, 95.0))

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

        if self.enable_debug_markers:
            status_text = self._build_operator_status_text(
                operator_state=operator_state,
                operator_reason=operator_reason,
                centerline_point_count=int(centerline.shape[0]),
                cmd_speed=cmd_speed,
                cmd_steering=cmd_steering,
                lookahead=lookahead,
                candidate_diagonal_count=candidate_diagonal_count,
                selected_chain_length=selected_chain_length,
                seed_midpoint_distance_m=seed_midpoint_distance_m,
                near_field_lateral_max_m=near_field_lateral_max_m,
                near_field_midpoint_kink_max_rad=near_field_midpoint_kink_max_rad,
                hold_remaining_s=hold_remaining_s,
            )
            markers = self._build_markers(
                now=now,
                frame_id=frame_id,
                result=result,
                centerline=centerline,
                raw_centerline=raw_centerline,
                raw_midpoint_chain=raw_midpoint_chain,
                status=status_text,
                operator_state=operator_state,
                control_target_frame=control_target_frame,
            )
            self._viz_pub.publish(markers)

    def _build_markers(
        self,
        *,
        now,
        frame_id: str,
        result: Optional[CoreResult],
        centerline: np.ndarray,
        raw_centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        status: str,
        operator_state: str,
        control_target_frame: Optional[np.ndarray],
    ) -> MarkerArray:
        arr = MarkerArray()

        clear = Marker()
        clear.header.frame_id = frame_id
        clear.header.stamp = now
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        marker_id = 1
        if result is None:
            marker_id = self._append_status_marker(
                arr,
                marker_id,
                frame_id,
                now,
                status,
                operator_state=operator_state,
            )
            return arr

        if self.show_raw_cones:
            marker_id = self._append_remembered_cone_marker(
                arr,
                marker_id,
                frame_id,
                now,
            )

        if self.show_delaunay_edges:
            arr.markers.append(
                self._make_edge_list_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns='delaunay_edges',
                    points=result.filtered_points,
                    edges=result.triangulation_edges,
                    color=(0.3, 0.7, 1.0, 0.35),
                    width=0.03,
                )
            )
            marker_id += 1

        if self.show_candidate_edges:
            arr.markers.append(
                self._make_edge_list_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns='candidate_cross_edges',
                    points=result.filtered_points,
                    edges=result.candidate_edges,
                    color=(1.0, 0.6, 0.1, 0.8),
                    width=0.06,
                )
            )
            marker_id += 1

        if self.show_selected_edges:
            arr.markers.append(
                self._make_edge_list_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns='selected_cross_edges',
                    points=result.filtered_points,
                    edges=result.selected_edges,
                    color=(0.2, 1.0, 0.3, 0.95),
                    width=0.08,
                )
            )
            marker_id += 1

        if self.show_raw_midpoint_chain:
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns='raw_midpoint_chain',
                    points=raw_midpoint_chain,
                    color=(1.0, 1.0, 1.0, 0.95),
                    width=0.06,
                    z_offset=0.03,
                )
            )
            marker_id += 1

        if self.show_raw_prevalidation_centerline:
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns='raw_prevalidation_centerline',
                    points=raw_centerline,
                    color=(1.0, 0.15, 0.85, 0.9),
                    width=0.07,
                    z_offset=0.05,
                )
            )
            marker_id += 1

        arr.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns='boundary_left',
                points=result.left_boundary,
                color=(0.2, 0.45, 1.0, 0.95),
                width=0.07,
                z_offset=0.02,
            )
        )
        marker_id += 1

        arr.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns='boundary_right',
                points=result.right_boundary,
                color=(1.0, 0.9, 0.2, 0.95),
                width=0.07,
                z_offset=0.02,
            )
        )
        marker_id += 1

        arr.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns='centerline',
                points=centerline,
                color=(0.95, 0.15, 0.15, 1.0),
                width=0.09,
                z_offset=0.07,
            )
        )
        marker_id += 1

        if self.show_lookahead_point and control_target_frame is not None:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = now
            marker.ns = 'lookahead'
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.position.x = float(control_target_frame[0])
            marker.pose.position.y = float(control_target_frame[1])
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            arr.markers.append(marker)
            marker_id += 1

        self._append_status_marker(
            arr,
            marker_id,
            frame_id,
            now,
            status,
            operator_state=operator_state,
        )
        return arr

    @staticmethod
    def _path_point_yaw(path_xy: np.ndarray, idx: int) -> float:
        if path_xy.shape[0] < 2:
            return 0.0
        if idx == path_xy.shape[0] - 1:
            dx, dy = path_xy[idx] - path_xy[idx - 1]
        else:
            dx, dy = path_xy[idx + 1] - path_xy[idx]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return float(math.atan2(dy, dx))

    @staticmethod
    def _make_points_marker(
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        color: tuple[float, float, float, float],
        scale: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.pose.orientation.w = 1.0
        for x, y in points:
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = 0.03
            marker.points.append(pt)
        return marker

    def _update_remembered_cone_viz(
        self,
        *,
        points_xy: np.ndarray,
        colors: list[str],
    ) -> None:
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            self._remembered_cone_viz_points = np.empty((0, 2), dtype=np.float64)
            self._remembered_cone_viz_colors = []
            return

        normalized_colors = [normalize_color(color) for color in colors]
        if len(normalized_colors) < points.shape[0]:
            normalized_colors.extend(['unknown'] * (points.shape[0] - len(normalized_colors)))
        elif len(normalized_colors) > points.shape[0]:
            normalized_colors = normalized_colors[: points.shape[0]]

        finite_mask = np.all(np.isfinite(points), axis=1)
        self._remembered_cone_viz_points = np.array(points[finite_mask], copy=True)
        self._remembered_cone_viz_colors = [
            color
            for color, keep in zip(normalized_colors, finite_mask)
            if bool(keep)
        ]

    def _append_remembered_cone_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
    ) -> int:
        points = np.asarray(
            getattr(self, '_remembered_cone_viz_points', np.empty((0, 2), dtype=np.float64)),
            dtype=np.float64,
        )
        colors = list(getattr(self, '_remembered_cone_viz_colors', []))
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            return marker_id
        if len(colors) < points.shape[0]:
            colors.extend(['unknown'] * (points.shape[0] - len(colors)))
        elif len(colors) > points.shape[0]:
            colors = colors[: points.shape[0]]

        markers.markers.append(
            self._make_colored_points_marker(
                frame_id=frame_id,
                stamp=stamp,
                marker_id=marker_id,
                ns='remembered_cones',
                points=points,
                colors=colors,
                scale=0.18,
            )
        )
        return marker_id + 1

    @classmethod
    def _make_colored_points_marker(
        cls,
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        colors: list[str],
        scale: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        for (x, y), color_name in zip(np.asarray(points, dtype=np.float64), colors):
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = 0.03
            marker.points.append(pt)

            rgba = ColorRGBA()
            rgba.r, rgba.g, rgba.b, rgba.a = cls._cone_marker_rgba(color_name)
            marker.colors.append(rgba)
        return marker

    @staticmethod
    def _cone_marker_rgba(color_name: str) -> tuple[float, float, float, float]:
        color = normalize_color(color_name)
        if color == 'blue':
            return 0.2, 0.55, 1.0, 0.95
        if color == 'yellow':
            return 1.0, 0.92, 0.25, 0.95
        if color == 'orange':
            return 1.0, 0.55, 0.15, 0.95
        return 0.8, 0.8, 0.8, 0.80

    @staticmethod
    def _make_edge_list_marker(
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        edges: np.ndarray,
        color: tuple[float, float, float, float],
        width: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.pose.orientation.w = 1.0

        for edge in edges:
            a = int(edge[0])
            b = int(edge[1])
            if a < 0 or b < 0 or a >= points.shape[0] or b >= points.shape[0]:
                continue
            p0 = Point()
            p0.x = float(points[a, 0])
            p0.y = float(points[a, 1])
            p0.z = 0.02
            p1 = Point()
            p1.x = float(points[b, 0])
            p1.y = float(points[b, 1])
            p1.z = 0.02
            marker.points.append(p0)
            marker.points.append(p1)
        return marker

    @staticmethod
    def _make_line_strip_marker(
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        color: tuple[float, float, float, float],
        width: float,
        z_offset: float = 0.02,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.pose.orientation.w = 1.0
        for x, y in points:
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = float(z_offset)
            marker.points.append(pt)
        return marker

    @staticmethod
    def _append_status_marker(
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        status: str,
        *,
        operator_state: str,
    ) -> int:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'status'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 0.32
        marker.color.a = 1.0
        color = _OPERATOR_STATE_COLORS.get(operator_state, _OPERATOR_STATE_COLORS['waiting'])
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.pose.position.x = 1.5
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.35
        marker.pose.orientation.w = 1.0
        marker.text = status
        markers.markers.append(marker)
        return marker_id + 1

    def _warn_throttled(self, key: str, message: str) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get(key, -1.0)
        if (now_sec - last_sec) >= self.log_throttle_s:
            self.get_logger().warn(message)
            self._last_throttled_log_sec[key] = now_sec

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp):
        timeout = Duration(seconds=self.tf_timeout_s)
        try:
            stamp_time = Time.from_msg(stamp)
            return self._tf_buffer.lookup_transform(target_frame, source_frame, stamp_time, timeout=timeout)
        except (TransformException, ValueError):
            pass
        try:
            return self._tf_buffer.lookup_transform(target_frame, source_frame, Time(), timeout=timeout)
        except TransformException:
            return None

    def _lookup_transform_with_alias(self, target_frame: str, source_frame: str, stamp):
        for target_candidate in self._frame_aliases(target_frame):
            for source_candidate in self._frame_aliases(source_frame):
                tf_msg = self._lookup_transform(target_candidate, source_candidate, stamp)
                if tf_msg is not None:
                    return tf_msg, target_candidate, source_candidate
        return None, target_frame, source_frame

    def _frame_aliases(self, frame: str) -> list[str]:
        token = str(frame).strip()
        out: list[str] = []

        def add(candidate: str) -> None:
            c = candidate.strip()
            if c and c not in out:
                out.append(c)

        add(token)
        leaf = token.split('/')[-1] if token else ''
        add(leaf)

        if token in {'base_link', 'base_footprint'} or leaf in {'base_link', 'base_footprint'}:
            add(token.replace('base_link', 'base_footprint'))
            add(token.replace('base_footprint', 'base_link'))
            if '/' in token:
                prefix = token.rsplit('/', 1)[0]
                add(f'{prefix}/base_link')
                add(f'{prefix}/base_footprint')
            add('base_link')
            add('base_footprint')
        return out

    def _is_alias(self, frame_a: str, frame_b: str) -> bool:
        a = set(self._frame_aliases(frame_a))
        b = set(self._frame_aliases(frame_b))
        return bool(a.intersection(b))

    @staticmethod
    def _transform_point(transform, x: float, y: float, z: float) -> tuple[float, float, float]:
        t = transform.transform.translation
        q = transform.transform.rotation
        qx = float(q.x)
        qy = float(q.y)
        qz = float(q.z)
        qw = float(q.w)

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)
        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)

        tx = float(t.x)
        ty = float(t.y)
        tz = float(t.z)

        px = (r00 * x) + (r01 * y) + (r02 * z) + tx
        py = (r10 * x) + (r11 * y) + (r12 * z) + ty
        pz = (r20 * x) + (r21 * y) + (r22 * z) + tz
        return px, py, pz

    @staticmethod
    def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _odom_point_to_base(x_odom: float, y_odom: float, tx: float, ty: float, yaw: float) -> tuple[float, float]:
        dx = x_odom - tx
        dy = y_odom - ty
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x_base = cos_y * dx + sin_y * dy
        y_base = -sin_y * dx + cos_y * dy
        return x_base, y_base

    @staticmethod
    def _base_point_to_odom(x_base: float, y_base: float, tx: float, ty: float, yaw: float) -> tuple[float, float]:
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x_odom = tx + (cos_y * x_base) - (sin_y * y_base)
        y_odom = ty + (sin_y * x_base) + (cos_y * y_base)
        return x_odom, y_odom


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DelaunayPlannerNode()
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
