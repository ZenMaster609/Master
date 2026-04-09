#!/usr/bin/env python3
"""Straight-line debug planner for acceleration controller testing."""

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
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cones.tracking.pose import (
    convert_odom_child_pose_to_base_frame,
    project_planar_pose_constant_twist,
)
from sim_car.planning.controller_config import build_steering_controller
from sim_car.planning.planner_runtime_types import PlannerIdentity
from sim_car.planning.tracked_cone_planner_contract import (
    log_tracked_cone_controller_mode,
    normalize_tracked_cone_controller_type,
)

_OPERATOR_STATE_CODES = {
    'waiting': 0,
    'fresh': 1,
    'held': 2,
    'stopped': 3,
}

_OPERATOR_REASON_CODES = {
    'none': 0,
    'missing_vehicle_pose': 2,
    'no_control_path': 11,
    'controller_compute_failed': 12,
    'stop_if_no_path': 13,
    'controller_disabled': 14,
}

_OPERATOR_REASON_LABELS = {
    'none': 'straight-line path active',
    'missing_vehicle_pose': 'missing vehicle pose',
    'no_control_path': 'vehicle passed line end',
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


class LineTestPlannerNode(Node):
    """Publishes a fixed straight centerline and optional steering commands."""

    def __init__(self) -> None:
        self._planner_identity = PlannerIdentity(
            node_name='linetest_planner_node',
            planner_mode='linetest',
            diagnostics_prefix='linetest_planner',
            diagnostics_topic='/linetest_planner/diagnostics',
        )
        super().__init__(self._planner_identity.node_name)

        self._declare_parameters()
        self._read_parameters()

        self._latest_odom_msg: Optional[Odometry] = None
        self._latest_speed_mps: float = 0.0
        self._latest_yaw_rate_rps: float = 0.0
        self._last_speed_cmd: Optional[float] = None
        self._last_steering_cmd: Optional[float] = None
        self._last_throttled_log_sec: dict[str, float] = {}

        self._cmd_pub = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
        self._path_pub = self.create_publisher(Path, self.centerline_topic, 10)
        self._viz_pub = self.create_publisher(MarkerArray, self.viz_topic, 10)
        self._points_pub = self.create_publisher(PoseArray, self.points_topic, 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, self.diagnostics_topic, 10)

        self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_cb,
            qos_profile_sensor_data,
        )

        loop_hz = max(1.0, float(self.publish_rate_hz))
        self.create_timer(1.0 / loop_hz, self._on_timer)

        if not self._is_alias(self.planning_frame, self.odom_frame):
            self.get_logger().warn(
                "frames.planning_frame '%s' is unsupported for linetest; publishing in '%s'"
                % (self.planning_frame, self.odom_frame)
            )

        self.get_logger().info(
            f'{self._planner_identity.node_name} ready '
            f'odom={self.odom_topic} cmd={self.cmd_topic} path={self.centerline_topic} '
            f'controller={self.controller_type} line=({self.line_start_x_m:.2f}, {self.line_start_y_m:.2f})'
            f' -> ({self.line_end_x_m:.2f}, {self.line_end_y_m:.2f})'
        )

    def _declare_parameters(self) -> None:
        defaults = {
            'frames.planning_frame': 'odom',
            'frames.odom_frame': 'odom',
            'frames.base_frame': 'front_axle',
            'topics.cmd_topic': '/cmd',
            'topics.centerline_topic': '/planned_centerline',
            'topics.viz_topic': '/planner_viz',
            'topics.points_topic': '/planned_centerline_points',
            'topics.odom_topic': '/sim/odom',
            'runtime.publish_rate_hz': 60.0,
            'runtime.log_throttle_s': 1.0,
            'control.controller_type': 'stanley',
            'control.odom_lag_compensation_ms': 0.0,
            'control.stop_if_no_path': True,
            'stanley.k_gain': 1.2,
            'stanley.softening_speed_mps': 0.0,
            'stanley.heading_gain': 1.6,
            'stanley.lookahead_idx_offset': 0,
            'stanley.steering_limit_rad': 0.52,
            'stanley.steering_lowpass_alpha': 1.0,
            'stanley.steering_rate_limit_rad_s': 10.0,
            'stanley.use_yaw_rate_damping': True,
            'stanley.yaw_rate_damping_gain': 0.0,
            'stanley.wheelbase_m': 1.65,
            'stanley.cross_track_deadband_m': 0.0,
            'pure_pursuit.lookahead_m': 3.0,
            'pure_pursuit.min_lookahead_m': 1.5,
            'pure_pursuit.max_lookahead_m': 8.0,
            'pure_pursuit.lookahead_gain': 0.0,
            'pure_pursuit.steering_limit_rad': 0.52,
            'pure_pursuit.steering_lowpass_alpha': 1.0,
            'pure_pursuit.steering_rate_limit_rad_s': 10.0,
            'pure_pursuit.wheelbase_m': 1.65,
            'speed_control.speed_min_mps': 1.0,
            'speed_control.speed_max_mps': 4.0,
            'speed_control.curvature_speed_gain': 4.0,
            'speed_control.lowpass_speed_alpha': 0.15,
            'diagnostics.topic': '/linetest_planner/diagnostics',
            'diagnostics.publish_control_debug': True,
            'diagnostics.publish_thesis_context': False,
            'debug.enable_markers': True,
            'debug.publish_points_topic': False,
            'debug.show_lookahead_point': True,
            'line.start_x_m': -38.5,
            'line.start_y_m': 0.0,
            'line.end_x_m': 50.0,
            'line.end_y_m': 0.0,
            'line.point_spacing_m': 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        self.planning_frame = str(self.get_parameter('frames.planning_frame').value).strip() or 'odom'
        self.odom_frame = str(self.get_parameter('frames.odom_frame').value).strip() or 'odom'
        self.base_frame = str(self.get_parameter('frames.base_frame').value).strip() or 'front_axle'

        self.cmd_topic = str(self.get_parameter('topics.cmd_topic').value)
        self.centerline_topic = str(self.get_parameter('topics.centerline_topic').value)
        self.viz_topic = str(self.get_parameter('topics.viz_topic').value)
        self.points_topic = str(self.get_parameter('topics.points_topic').value)
        self.odom_topic = str(self.get_parameter('topics.odom_topic').value)

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
        self._controller = self._build_steering_controller() if self.controller_type != 'none' else None
        self.stop_if_no_path = bool(self.get_parameter('control.stop_if_no_path').value)
        log_tracked_cone_controller_mode(self, controller_type=self.controller_type)

        self.speed_min_mps = max(0.0, float(self.get_parameter('speed_control.speed_min_mps').value))
        self.speed_max_mps = max(self.speed_min_mps, float(self.get_parameter('speed_control.speed_max_mps').value))
        self.curvature_speed_gain = max(0.0, float(self.get_parameter('speed_control.curvature_speed_gain').value))
        self.lowpass_speed_alpha = float(
            np.clip(float(self.get_parameter('speed_control.lowpass_speed_alpha').value), 0.0, 1.0)
        )

        self.diagnostics_topic = (
            str(self.get_parameter('diagnostics.topic').value).strip()
            or self._planner_identity.diagnostics_topic
        )
        self.publish_control_debug = bool(self.get_parameter('diagnostics.publish_control_debug').value)
        self.publish_thesis_context = bool(self.get_parameter('diagnostics.publish_thesis_context').value)

        self.enable_debug_markers = bool(self.get_parameter('debug.enable_markers').value)
        self.publish_points_topic = bool(self.get_parameter('debug.publish_points_topic').value)
        self.show_lookahead_point = bool(self.get_parameter('debug.show_lookahead_point').value)

        self.line_start_x_m = float(self.get_parameter('line.start_x_m').value)
        self.line_start_y_m = float(self.get_parameter('line.start_y_m').value)
        self.line_end_x_m = float(self.get_parameter('line.end_x_m').value)
        self.line_end_y_m = float(self.get_parameter('line.end_y_m').value)
        self.line_point_spacing_m = max(0.05, float(self.get_parameter('line.point_spacing_m').value))

        self._line_start_xy = np.array([self.line_start_x_m, self.line_start_y_m], dtype=np.float64)
        self._line_end_xy = np.array([self.line_end_x_m, self.line_end_y_m], dtype=np.float64)
        delta = self._line_end_xy - self._line_start_xy
        self._line_length_m = float(np.hypot(delta[0], delta[1]))
        if self._line_length_m <= 1e-9:
            self._line_direction_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            self._line_direction_xy = delta / self._line_length_m
        self._full_centerline = self._generate_line_path(
            self._line_start_xy,
            self._line_end_xy,
            self.line_point_spacing_m,
        )

    def _build_steering_controller(self):
        return build_steering_controller(
            node=self,
            controller_type=self.controller_type,
            publish_rate_hz=self.publish_rate_hz,
        )

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
        frame_id = self.odom_frame
        centerline = np.array(self._full_centerline, copy=True)
        control_path = np.empty((0, 2), dtype=np.float64)
        control_target_base: Optional[np.ndarray] = None
        control_target_world: Optional[np.ndarray] = None
        control_debug_metrics: Optional[dict[str, float]] = None
        operator_state = 'fresh'
        operator_reason = 'none'
        cmd_speed = 0.0
        cmd_steering = 0.0
        lookahead = 0.0
        zero_cmd_sent_flag = 0
        vehicle_pose = self._resolve_vehicle_pose()

        if vehicle_pose is None:
            operator_state = 'waiting'
            operator_reason = 'missing_vehicle_pose'
            if self._controller is not None:
                zero_cmd_sent_flag = int(self._apply_no_path_behavior())
        else:
            vehicle_x, vehicle_y, vehicle_yaw = vehicle_pose
            control_world = self._build_forward_control_path(vehicle_x, vehicle_y)
            control_path = self._centerline_to_vehicle_frame(
                centerline=control_world,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
            )

            if self._controller is None:
                operator_reason = 'controller_disabled'
            elif control_path.shape[0] == 0:
                operator_state = 'stopped'
                operator_reason = 'no_control_path'
                zero_cmd_sent_flag = int(self._apply_no_path_behavior())
            else:
                try:
                    controller_output = self._controller.compute(
                        control_path=control_path,
                        speed_mps=self._latest_speed_mps,
                        yaw_rate_rps=self._latest_yaw_rate_rps,
                    )
                except Exception as exc:
                    operator_state = 'stopped'
                    operator_reason = 'controller_compute_failed'
                    zero_cmd_sent_flag = int(self._apply_no_path_behavior())
                    self._throttled_warn('controller_compute_failed', f'linetest controller failure: {exc}')
                else:
                    cmd_steering = float(controller_output.steering_rad)
                    cmd_speed = self._compute_speed_command(float(controller_output.kappa))
                    lookahead = float(controller_output.lookahead_m)
                    control_target_base = np.asarray(controller_output.target_point_base, dtype=np.float64)
                    control_target_world = np.asarray(
                        self._base_point_to_odom(
                            float(control_target_base[0]),
                            float(control_target_base[1]),
                            vehicle_x,
                            vehicle_y,
                            vehicle_yaw,
                        ),
                        dtype=np.float64,
                    )
                    control_debug_metrics = self._control_debug_metrics(
                        controller_output=controller_output,
                        target_point_world=control_target_world,
                    )
                    self._last_speed_cmd = cmd_speed
                    self._last_steering_cmd = cmd_steering
                    self._publish_cmd(cmd_speed, cmd_steering)

        status_text = self._build_status_text(
            operator_state=operator_state,
            operator_reason=operator_reason,
            centerline_point_count=int(centerline.shape[0]),
            control_path_point_count=int(control_path.shape[0]),
            cmd_speed=cmd_speed,
            cmd_steering=cmd_steering,
            lookahead=lookahead,
        )

        self._publish_outputs(
            frame_id=frame_id,
            centerline=centerline,
            status_text=status_text,
            operator_state=operator_state,
            control_target_world=control_target_world,
            vehicle_pose=vehicle_pose,
        )
        self._publish_diagnostics(
            frame_id=frame_id,
            centerline=centerline,
            control_path_point_count=int(control_path.shape[0]),
            operator_state=operator_state,
            operator_reason=operator_reason,
            zero_cmd_sent_flag=zero_cmd_sent_flag,
            control_debug_metrics=control_debug_metrics,
            thesis_context_metrics={
                'plan_valid_flag': 1.0 if centerline.shape[0] > 0 else 0.0,
                'plan_hold_active_flag': 0.0,
                'plan_fallback_flag': 0.0,
                'path_length_m': self._path_length_m(centerline),
                'path_curvature_abs_p95_1pm': self._path_curvature_abs_p95(centerline),
            },
        )

    def _resolve_vehicle_pose(self) -> Optional[tuple[float, float, float]]:
        odom = self._latest_odom_msg
        if odom is None:
            return None
        if not self._is_alias(str(odom.header.frame_id).strip(), self.odom_frame):
            return None

        pose = odom.pose.pose
        tx = float(pose.position.x)
        ty = float(pose.position.y)
        q = pose.orientation
        yaw = self._yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        child_frame = str(odom.child_frame_id).strip()
        base_pose = self._convert_odom_child_pose_to_base_frame(
            child_frame=child_frame,
            tx=tx,
            ty=ty,
            yaw=yaw,
        )
        if base_pose is None:
            return None
        return project_planar_pose_constant_twist(
            base_pose,
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
        wheelbase = float(self.get_parameter('stanley.wheelbase_m').value)
        return max(0.1, wheelbase)

    def _build_forward_control_path(self, vehicle_x: float, vehicle_y: float) -> np.ndarray:
        if self._line_length_m <= 1e-9:
            return np.empty((0, 2), dtype=np.float64)

        vehicle_xy = np.array([vehicle_x, vehicle_y], dtype=np.float64)
        station_m = float(np.dot(vehicle_xy - self._line_start_xy, self._line_direction_xy))
        start_station_m = max(0.0, station_m)
        if start_station_m >= self._line_length_m:
            return np.empty((0, 2), dtype=np.float64)

        stations = [start_station_m]
        next_station_m = math.floor(start_station_m / self.line_point_spacing_m + 1.0) * self.line_point_spacing_m
        while next_station_m < self._line_length_m:
            stations.append(float(next_station_m))
            next_station_m += self.line_point_spacing_m
        if self._line_length_m - stations[-1] > 1e-6:
            stations.append(self._line_length_m)

        return np.asarray(
            [self._line_start_xy + (self._line_direction_xy * station) for station in stations],
            dtype=np.float64,
        )

    def _centerline_to_vehicle_frame(
        self,
        *,
        centerline: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> np.ndarray:
        if centerline.shape[0] == 0:
            return centerline
        out = np.empty_like(centerline)
        for idx in range(centerline.shape[0]):
            out[idx, 0], out[idx, 1] = self._odom_point_to_base(
                float(centerline[idx, 0]),
                float(centerline[idx, 1]),
                vehicle_x,
                vehicle_y,
                vehicle_yaw,
            )
        return out

    def _publish_outputs(
        self,
        *,
        frame_id: str,
        centerline: np.ndarray,
        status_text: str,
        operator_state: str,
        control_target_world: Optional[np.ndarray],
        vehicle_pose: Optional[tuple[float, float, float]],
    ) -> None:
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
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
            markers = self._build_markers(
                frame_id=frame_id,
                stamp=path_msg.header.stamp,
                centerline=centerline,
                status_text=status_text,
                operator_state=operator_state,
                control_target_world=control_target_world,
                vehicle_pose=vehicle_pose,
            )
            self._viz_pub.publish(markers)

    def _build_markers(
        self,
        *,
        frame_id: str,
        stamp,
        centerline: np.ndarray,
        status_text: str,
        operator_state: str,
        control_target_world: Optional[np.ndarray],
        vehicle_pose: Optional[tuple[float, float, float]],
    ) -> MarkerArray:
        markers = MarkerArray()

        clear = Marker()
        clear.header.frame_id = frame_id
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        markers.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id,
                stamp=stamp,
                marker_id=1,
                ns='centerline',
                points=centerline,
                color=(0.95, 0.15, 0.15, 1.0),
                width=0.09,
                z_offset=0.07,
            )
        )

        if self.show_lookahead_point and control_target_world is not None:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = 'lookahead_point'
            marker.id = 2
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(control_target_world[0])
            marker.pose.position.y = float(control_target_world[1])
            marker.pose.position.z = 0.15
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.22
            marker.scale.y = 0.22
            marker.scale.z = 0.22
            marker.color.r = 0.2
            marker.color.g = 1.0
            marker.color.b = 0.3
            marker.color.a = 0.95
            markers.markers.append(marker)

        status_marker = Marker()
        status_marker.header.frame_id = frame_id
        status_marker.header.stamp = stamp
        status_marker.ns = 'status'
        status_marker.id = 3
        status_marker.type = Marker.TEXT_VIEW_FACING
        status_marker.action = Marker.ADD
        if vehicle_pose is None:
            status_marker.pose.position.x = float(self.line_start_x_m)
            status_marker.pose.position.y = float(self.line_start_y_m)
        else:
            status_marker.pose.position.x = float(vehicle_pose[0])
            status_marker.pose.position.y = float(vehicle_pose[1])
        status_marker.pose.position.z = 1.2
        status_marker.pose.orientation.w = 1.0
        status_marker.scale.z = 0.32
        status_marker.text = status_text
        color = _OPERATOR_STATE_COLORS.get(operator_state, (1.0, 1.0, 1.0))
        status_marker.color.r = color[0]
        status_marker.color.g = color[1]
        status_marker.color.b = color[2]
        status_marker.color.a = 0.95
        markers.markers.append(status_marker)

        return markers

    def _publish_diagnostics(
        self,
        *,
        frame_id: str,
        centerline: np.ndarray,
        control_path_point_count: int,
        operator_state: str,
        operator_reason: str,
        zero_cmd_sent_flag: int,
        control_debug_metrics: Optional[dict[str, float]] = None,
        thesis_context_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        stability = DiagnosticStatus()
        stability.name = f'{self._planner_identity.diagnostics_prefix}/stability'
        stability.hardware_id = self._planner_identity.hardware_id
        stability.level = DiagnosticStatus.OK
        stability.message = _OPERATOR_REASON_LABELS.get(operator_reason, operator_reason)

        planner_metrics = {
            'path_length_m': self._path_length_m(centerline),
            'path_curvature_abs_p95_1pm': self._path_curvature_abs_p95(centerline),
            'planner_state_code': self._operator_state_code(operator_state),
            'fresh_publish_flag': 1 if operator_state == 'fresh' else 0,
            'held_publish_flag': 0,
            'stopped_flag': 1 if operator_state == 'stopped' else 0,
            'waiting_flag': 1 if operator_state == 'waiting' else 0,
            'operator_reason_code': self._operator_reason_code(operator_reason),
            'hold_remaining_s': 0.0,
            'control_path_point_count': int(control_path_point_count),
            'zero_cmd_sent_flag': int(zero_cmd_sent_flag),
            'planner_mode': self._planner_identity.planner_mode,
        }
        stability.values = [
            KeyValue(key='centerline_jump_max_m', value='0.000000'),
            KeyValue(key='selected_edge_churn_ratio', value='0.000000'),
            KeyValue(key='tracked_cones_frame_delta_p95_m', value='nan'),
            KeyValue(key='centerline_point_count', value=str(int(centerline.shape[0]))),
            KeyValue(key='selected_edge_count', value='0'),
        ]
        for key in sorted(planner_metrics):
            stability.values.append(KeyValue(key=str(key), value=self._diag_value_string(planner_metrics[key])))
        msg.status.append(stability)

        if self.publish_control_debug:
            merged = self._default_control_debug_metrics()
            if control_debug_metrics is not None:
                merged.update(control_debug_metrics)
            control_diag = DiagnosticStatus()
            control_diag.name = f'{self._planner_identity.diagnostics_prefix}/control_debug'
            control_diag.hardware_id = self._planner_identity.hardware_id
            control_diag.level = DiagnosticStatus.OK
            control_diag.message = 'controller debug signals'
            control_diag.values = [
                KeyValue(key='heading_error_rad', value=f"{merged['heading_error_rad']:.6f}"),
                KeyValue(key='cross_track_error_m', value=f"{merged['cross_track_error_m']:.6f}"),
                KeyValue(key='vehicle_speed_mps', value=f"{merged['vehicle_speed_mps']:.6f}"),
                KeyValue(key='speed_term_mps', value=f"{merged['speed_term_mps']:.6f}"),
                KeyValue(key='heading_contribution_rad', value=f"{merged['heading_contribution_rad']:.6f}"),
                KeyValue(key='cross_track_contribution_rad', value=f"{merged['cross_track_contribution_rad']:.6f}"),
                KeyValue(
                    key='yaw_rate_damping_contribution_rad',
                    value=f"{merged['yaw_rate_damping_contribution_rad']:.6f}",
                ),
                KeyValue(key='yaw_rate_rps', value=f"{merged['yaw_rate_rps']:.6f}"),
                KeyValue(key='raw_steering_cmd_rad', value=f"{merged['raw_steering_cmd_rad']:.6f}"),
                KeyValue(key='steering_after_clamp_rad', value=f"{merged['steering_after_clamp_rad']:.6f}"),
                KeyValue(key='steering_after_filter_rad', value=f"{merged['steering_after_filter_rad']:.6f}"),
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
                KeyValue(key='target_point_x_base_m', value=f"{merged['target_point_x_base_m']:.6f}"),
                KeyValue(key='target_point_y_base_m', value=f"{merged['target_point_y_base_m']:.6f}"),
                KeyValue(key='target_point_x_frame_m', value=f"{merged['target_point_x_frame_m']:.6f}"),
                KeyValue(key='target_point_y_frame_m', value=f"{merged['target_point_y_frame_m']:.6f}"),
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
                KeyValue(key='plan_valid_flag', value=str(int(round(merged_thesis['plan_valid_flag'])))),
                KeyValue(
                    key='plan_hold_active_flag',
                    value=str(int(round(merged_thesis['plan_hold_active_flag']))),
                ),
                KeyValue(key='plan_fallback_flag', value=str(int(round(merged_thesis['plan_fallback_flag'])))),
                KeyValue(key='centerline_point_count', value=str(int(centerline.shape[0]))),
                KeyValue(key='selected_edge_count', value='0'),
                KeyValue(key='centerline_jump_max_m', value='0.000000'),
                KeyValue(key='selected_edge_churn_ratio', value='0.000000'),
                KeyValue(key='path_length_m', value=f"{merged_thesis['path_length_m']:.6f}"),
                KeyValue(
                    key='path_curvature_abs_p95_1pm',
                    value=f"{merged_thesis['path_curvature_abs_p95_1pm']:.6f}",
                ),
            ]
            msg.status.append(thesis_diag)

        self._diag_pub.publish(msg)

    def _control_debug_metrics(
        self,
        *,
        controller_output,
        target_point_world: np.ndarray,
    ) -> dict[str, float]:
        metrics = self._default_control_debug_metrics()
        metrics['target_point_x_frame_m'] = float(target_point_world[0])
        metrics['target_point_y_frame_m'] = float(target_point_world[1])
        metrics['vehicle_speed_mps'] = float(self._latest_speed_mps)
        metrics['yaw_rate_rps'] = float(self._latest_yaw_rate_rps)
        if controller_output.stanley_debug is not None:
            debug = controller_output.stanley_debug
            metrics.update(
                {
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
                    'steering_saturated_flag': float(int(debug.steering_saturated_flag)),
                    'nearest_path_index': float(debug.nearest_path_index),
                    'heading_path_index': float(debug.heading_path_index),
                    'target_point_x_base_m': float(debug.target_point_x_base_m),
                    'target_point_y_base_m': float(debug.target_point_y_base_m),
                }
            )
        else:
            metrics.update(
                {
                    'raw_steering_cmd_rad': float(controller_output.steering_rad),
                    'steering_after_clamp_rad': float(controller_output.steering_rad),
                    'steering_after_filter_rad': float(controller_output.steering_rad),
                    'steering_after_rate_limit_rad': float(controller_output.steering_rad),
                    'final_steering_cmd_rad': float(controller_output.steering_rad),
                    'target_point_x_base_m': float(controller_output.target_point_base[0]),
                    'target_point_y_base_m': float(controller_output.target_point_base[1]),
                }
            )
        return metrics

    def _build_status_text(
        self,
        *,
        operator_state: str,
        operator_reason: str,
        centerline_point_count: int,
        control_path_point_count: int,
        cmd_speed: float,
        cmd_steering: float,
        lookahead: float,
    ) -> str:
        return (
            f'{operator_state.upper()} | '
            f'{_OPERATOR_REASON_LABELS.get(operator_reason, operator_reason)}\n'
            f'line pts={centerline_point_count} ctrl pts={control_path_point_count}\n'
            f'CMD: v={cmd_speed:.2f} m/s delta={cmd_steering:.3f} rad lookahead={lookahead:.2f} m'
        )

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

    def _publish_cmd(self, speed_mps: float, steering_rad: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed_mps)
        msg.drive.steering_angle = float(steering_rad)
        self._cmd_pub.publish(msg)

    def _throttled_warn(self, key: str, message: str) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get(key, -1.0)
        if last_sec >= 0.0 and (now_sec - last_sec) < self.log_throttle_s:
            return
        self._last_throttled_log_sec[key] = now_sec
        self.get_logger().warn(message)

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
    def _operator_state_code(operator_state: str) -> int:
        return int(_OPERATOR_STATE_CODES.get(operator_state, 0))

    @staticmethod
    def _operator_reason_code(operator_reason: str) -> int:
        return int(_OPERATOR_REASON_CODES.get(operator_reason, 0))

    @staticmethod
    def _generate_line_path(start_xy: np.ndarray, end_xy: np.ndarray, spacing_m: float) -> np.ndarray:
        start = np.asarray(start_xy, dtype=np.float64)
        end = np.asarray(end_xy, dtype=np.float64)
        spacing = max(0.05, float(spacing_m))
        delta = end - start
        length = float(np.hypot(delta[0], delta[1]))
        if length <= 1e-9:
            return np.asarray([start], dtype=np.float64)

        direction = delta / length
        stations = list(np.arange(0.0, length, spacing, dtype=np.float64))
        if not stations or abs(float(stations[-1]) - length) > 1e-6:
            stations.append(length)
        return np.asarray(
            [start + (direction * float(station)) for station in stations],
            dtype=np.float64,
        )

    @staticmethod
    def _path_point_yaw(centerline: np.ndarray, idx: int) -> float:
        if centerline.shape[0] < 2:
            return 0.0
        if idx <= 0:
            vec = centerline[1] - centerline[0]
        elif idx >= centerline.shape[0] - 1:
            vec = centerline[-1] - centerline[-2]
        else:
            vec = centerline[idx + 1] - centerline[idx - 1]
        if float(np.hypot(vec[0], vec[1])) <= 1e-9:
            return 0.0
        return float(math.atan2(float(vec[1]), float(vec[0])))

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

    @staticmethod
    def _frame_aliases(frame: str) -> set[str]:
        out: set[str] = set()

        def add(token: str) -> None:
            normalized = str(token).strip().strip('/')
            if normalized:
                out.add(normalized)

        add(frame)
        normalized = str(frame).strip().strip('/').lower()
        if normalized == 'odom':
            add('odom')
        if normalized in {'front_axle', 'base_link', 'base_footprint'}:
            add('front_axle')
            add('base_link')
            add('base_footprint')
        return out

    def _is_alias(self, frame_a: str, frame_b: str) -> bool:
        return bool(self._frame_aliases(frame_a).intersection(self._frame_aliases(frame_b)))

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
        z_offset: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(width)
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        for xy in points:
            point = Point()
            point.x = float(xy[0])
            point.y = float(xy[1])
            point.z = float(z_offset)
            marker.points.append(point)
        return marker


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LineTestPlannerNode()
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
