#!/usr/bin/env python3
"""Delaunay-based centerline planner over tracked cone detections."""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetectionArray
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cone_fusion import normalize_color
from sim_car.delaunay_planner_core import CoreConfig, CoreResult, compute_centerline


class DelaunayPlannerNode(Node):
    """Consumes tracked cones and publishes a centerline path + debug markers."""

    def __init__(self) -> None:
        super().__init__('delaunay_planner_node')
        self._declare_parameters()
        self._read_parameters()

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_cones_msg: Optional[ConeDetectionArray] = None
        self._latest_odom_msg: Optional[Odometry] = None
        self._latest_speed_mps: float = 0.0
        self._last_throttled_log_sec: dict[str, float] = {}
        self._previous_centerline: Optional[np.ndarray] = None
        self._last_speed_cmd: Optional[float] = None
        self._last_steering_cmd: Optional[float] = None

        self._cmd_pub = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
        self._path_pub = self.create_publisher(Path, self.centerline_topic, 10)
        self._viz_pub = self.create_publisher(MarkerArray, self.viz_topic, 10)
        self._points_pub = self.create_publisher(PoseArray, self.points_topic, 10)

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
            10,
        )

        loop_hz = max(1.0, float(self.publish_rate_hz))
        self.create_timer(1.0 / loop_hz, self._on_timer)

        self.get_logger().info(
            'delaunay_planner_node ready '
            f'cones={self.tracked_cones_topic} odom={self.odom_topic} '
            f'cmd={self.cmd_topic} path={self.centerline_topic} viz={self.viz_topic} '
            f'planning_frame={self.planning_frame}'
        )

    def _declare_parameters(self) -> None:
        defaults = {
            'frames.planning_frame': 'odom',
            'frames.odom_frame': 'odom',
            'frames.base_frame': 'base_link',
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
            'filtering.include_orange': False,
            'filtering.min_colored_cones': 6,
            'filtering.min_required_cones': 6,
            'edge_selection.min_cross_edge_m': 0.8,
            'edge_selection.max_cross_edge_m': 6.0,
            'edge_selection.cross_edge_lateral_ratio': 0.6,
            'edge_selection.min_cross_edges': 3,
            'centerline.min_spacing_m': 0.5,
            'centerline.path_resolution_m': 0.5,
            'centerline.max_path_length_m': 30.0,
            'centerline.enable_temporal_smoothing': True,
            'centerline.smoothing_alpha': 0.3,
            'runtime.publish_rate_hz': 20.0,
            'runtime.log_throttle_s': 1.0,
            'pure_pursuit.lookahead_min_m': 3.0,
            'pure_pursuit.lookahead_gain': 0.35,
            'pure_pursuit.steering_limit_rad': 0.32,
            'pure_pursuit.wheelbase_m': 1.6,
            'pure_pursuit.stop_if_no_path': False,
            'speed_control.speed_min_mps': 1.0,
            'speed_control.speed_max_mps': 1.8,
            'speed_control.curvature_speed_gain': 4.0,
            'speed_control.lowpass_speed_alpha': 0.15,
            'debug.enable_markers': True,
            'debug.show_raw_cones': False,
            'debug.show_delaunay_edges': True,
            'debug.show_candidate_edges': True,
            'debug.show_selected_edges': True,
            'debug.publish_points_topic': False,
            'debug.show_lookahead_point': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        self.planning_frame = str(self.get_parameter('frames.planning_frame').value).strip() or 'odom'
        self.odom_frame = str(self.get_parameter('frames.odom_frame').value).strip() or 'odom'
        self.base_frame = str(self.get_parameter('frames.base_frame').value).strip() or 'base_link'
        self.tf_timeout_s = max(0.0, float(self.get_parameter('frames.tf_timeout_s').value))

        self.tracked_cones_topic = str(self.get_parameter('topics.tracked_cones_topic').value)
        self.cmd_topic = str(self.get_parameter('topics.cmd_topic').value)
        self.centerline_topic = str(self.get_parameter('topics.centerline_topic').value)
        self.viz_topic = str(self.get_parameter('topics.viz_topic').value)
        self.points_topic = str(self.get_parameter('topics.points_topic').value)
        self.odom_topic = str(self.get_parameter('topics.odom_topic').value)

        self.enable_temporal_smoothing = bool(self.get_parameter('centerline.enable_temporal_smoothing').value)
        self.smoothing_alpha = float(
            np.clip(float(self.get_parameter('centerline.smoothing_alpha').value), 0.0, 1.0)
        )

        self.publish_rate_hz = max(1.0, float(self.get_parameter('runtime.publish_rate_hz').value))
        self.log_throttle_s = max(0.1, float(self.get_parameter('runtime.log_throttle_s').value))

        self.lookahead_min_m = max(0.5, float(self.get_parameter('pure_pursuit.lookahead_min_m').value))
        self.lookahead_gain = max(0.0, float(self.get_parameter('pure_pursuit.lookahead_gain').value))
        self.steering_limit_rad = max(0.01, float(self.get_parameter('pure_pursuit.steering_limit_rad').value))
        self.wheelbase_m = max(0.1, float(self.get_parameter('pure_pursuit.wheelbase_m').value))
        self.stop_if_no_path = bool(self.get_parameter('pure_pursuit.stop_if_no_path').value)

        self.speed_min_mps = max(0.0, float(self.get_parameter('speed_control.speed_min_mps').value))
        self.speed_max_mps = max(self.speed_min_mps, float(self.get_parameter('speed_control.speed_max_mps').value))
        self.curvature_speed_gain = max(0.0, float(self.get_parameter('speed_control.curvature_speed_gain').value))
        self.lowpass_speed_alpha = float(
            np.clip(float(self.get_parameter('speed_control.lowpass_speed_alpha').value), 0.0, 1.0)
        )

        self.enable_debug_markers = bool(self.get_parameter('debug.enable_markers').value)
        self.show_raw_cones = bool(self.get_parameter('debug.show_raw_cones').value)
        self.show_delaunay_edges = bool(self.get_parameter('debug.show_delaunay_edges').value)
        self.show_candidate_edges = bool(self.get_parameter('debug.show_candidate_edges').value)
        self.show_selected_edges = bool(self.get_parameter('debug.show_selected_edges').value)
        self.publish_points_topic = bool(self.get_parameter('debug.publish_points_topic').value)
        self.show_lookahead_point = bool(self.get_parameter('debug.show_lookahead_point').value)

        self._core_config = CoreConfig(
            max_cone_range_m=float(self.get_parameter('filtering.max_cone_range_m').value),
            behind_drop_m=float(self.get_parameter('filtering.behind_drop_m').value),
            min_confidence=float(self.get_parameter('filtering.min_confidence').value),
            use_unknown_cones=bool(self.get_parameter('filtering.use_unknown_cones').value),
            infer_unknown_by_side=bool(self.get_parameter('filtering.infer_unknown_by_side').value),
            include_orange=bool(self.get_parameter('filtering.include_orange').value),
            min_colored_cones=max(1, int(self.get_parameter('filtering.min_colored_cones').value)),
            min_required_cones=max(2, int(self.get_parameter('filtering.min_required_cones').value)),
            min_cross_edge_m=float(self.get_parameter('edge_selection.min_cross_edge_m').value),
            max_cross_edge_m=float(self.get_parameter('edge_selection.max_cross_edge_m').value),
            cross_edge_lateral_ratio=float(self.get_parameter('edge_selection.cross_edge_lateral_ratio').value),
            min_cross_edges=max(1, int(self.get_parameter('edge_selection.min_cross_edges').value)),
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
        else:
            self._latest_speed_mps = float(math.hypot(vx, vy))

    def _on_timer(self) -> None:
        control_target_base: Optional[np.ndarray] = None
        control_target_frame: Optional[np.ndarray] = None
        cmd_speed = 0.0
        cmd_steering = 0.0
        lookahead = 0.0

        cones_msg = self._latest_cones_msg
        if cones_msg is None:
            self._apply_no_path_behavior()
            self._publish_outputs(
                frame_id=self.odom_frame,
                centerline=np.empty((0, 2), dtype=np.float64),
                result=None,
                status='waiting for /tracked_cones',
                control_target_frame=control_target_frame,
                cmd_speed=cmd_speed,
                cmd_steering=cmd_steering,
                lookahead=lookahead,
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
            self._apply_no_path_behavior()
            self._publish_outputs(
                frame_id=target_frame,
                centerline=np.empty((0, 2), dtype=np.float64),
                result=None,
                status='missing vehicle pose (tf and /sim/odom unavailable)',
                control_target_frame=control_target_frame,
                cmd_speed=cmd_speed,
                cmd_steering=cmd_steering,
                lookahead=lookahead,
            )
            return

        vehicle_x, vehicle_y, vehicle_yaw = pose
        points_xy, colors, confidences = self._convert_cones_to_frame(cones_msg, source_frame, target_frame)
        if points_xy is None:
            if target_frame != self.odom_frame:
                self._warn_throttled(
                    'fallback_odom_cones',
                    f'cannot transform cones to {target_frame}; planning in odom frame',
                )
                target_frame = self.odom_frame
                pose = self._resolve_vehicle_pose(target_frame, cones_msg.header.stamp)
                if pose is None:
                    self._apply_no_path_behavior()
                    self._publish_outputs(
                        frame_id=self.odom_frame,
                        centerline=np.empty((0, 2), dtype=np.float64),
                        result=None,
                        status='cone transform failed and odom pose unavailable',
                        control_target_frame=control_target_frame,
                        cmd_speed=cmd_speed,
                        cmd_steering=cmd_steering,
                        lookahead=lookahead,
                    )
                    return
                vehicle_x, vehicle_y, vehicle_yaw = pose
                points_xy, colors, confidences = self._convert_cones_to_frame(cones_msg, source_frame, target_frame)

        if points_xy is None:
            self._apply_no_path_behavior()
            self._publish_outputs(
                frame_id=target_frame,
                centerline=np.empty((0, 2), dtype=np.float64),
                result=None,
                status='cone transform unavailable',
                control_target_frame=control_target_frame,
                cmd_speed=cmd_speed,
                cmd_steering=cmd_steering,
                lookahead=lookahead,
            )
            return

        result = compute_centerline(
            points_xy=points_xy,
            colors=colors,
            confidences=confidences,
            vehicle_xy=(vehicle_x, vehicle_y),
            vehicle_yaw=vehicle_yaw,
            config=self._core_config,
        )

        centerline = result.centerline
        if self.enable_temporal_smoothing:
            centerline = self._apply_temporal_smoothing(centerline)

        control_path = self._centerline_to_vehicle_frame(
            centerline=centerline,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        if control_path.shape[0] >= 1:
            cmd_steering, cmd_speed, _kappa, lookahead, control_target_base = self._compute_control(control_path)
            self._publish_cmd(cmd_speed, cmd_steering)
            if control_target_base is not None:
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
        else:
            self._apply_no_path_behavior()
            if self._last_speed_cmd is not None:
                cmd_speed = float(self._last_speed_cmd)
            if self._last_steering_cmd is not None:
                cmd_steering = float(self._last_steering_cmd)

        self._publish_outputs(
            frame_id=target_frame,
            centerline=centerline,
            result=result,
            status=result.status,
            control_target_frame=control_target_frame,
            cmd_speed=cmd_speed,
            cmd_steering=cmd_steering,
            lookahead=lookahead,
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

    def _compute_control(self, control_path: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
        speed = max(0.0, float(self._latest_speed_mps))
        lookahead = max(self.lookahead_min_m, self.lookahead_min_m + self.lookahead_gain * speed)

        if control_path.shape[0] <= 1:
            target = control_path[0]
        else:
            segment_lengths = np.hypot(np.diff(control_path[:, 0]), np.diff(control_path[:, 1]))
            arc_lengths = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths)))
            idx = int(np.searchsorted(arc_lengths, lookahead, side='left'))
            idx = min(max(idx, 0), control_path.shape[0] - 1)
            target = control_path[idx]

        target_distance = max(float(np.hypot(target[0], target[1])), 1e-3)
        kappa = 2.0 * float(target[1]) / max(target_distance * target_distance, 1e-6)
        steering = math.atan(self.wheelbase_m * kappa)
        steering = float(np.clip(steering, -self.steering_limit_rad, self.steering_limit_rad))

        v_des = self.speed_max_mps / (1.0 + self.curvature_speed_gain * abs(kappa))
        v_des = float(np.clip(v_des, self.speed_min_mps, self.speed_max_mps))
        if self._last_speed_cmd is None:
            v_cmd = v_des
        else:
            alpha = self.lowpass_speed_alpha
            v_cmd = alpha * v_des + (1.0 - alpha) * float(self._last_speed_cmd)

        self._last_speed_cmd = v_cmd
        self._last_steering_cmd = steering
        return steering, v_cmd, kappa, lookahead, target

    def _apply_no_path_behavior(self) -> None:
        if self.stop_if_no_path:
            self._publish_cmd(0.0, 0.0)
            self._last_speed_cmd = 0.0
            self._last_steering_cmd = 0.0
            return
        if self._last_speed_cmd is not None and self._last_steering_cmd is not None:
            self._publish_cmd(float(self._last_speed_cmd), float(self._last_steering_cmd))

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
        if not self._is_alias(odom.child_frame_id, self.base_frame):
            return None

        pose = odom.pose.pose
        tx = float(pose.position.x)
        ty = float(pose.position.y)
        q = pose.orientation
        yaw = self._yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        return tx, ty, yaw

    def _convert_cones_to_frame(
        self,
        msg: ConeDetectionArray,
        source_frame: str,
        target_frame: str,
    ) -> tuple[Optional[np.ndarray], list[str], np.ndarray]:
        tf_msg = None
        if not self._is_alias(source_frame, target_frame):
            tf_msg, _resolved_target, _resolved_source = self._lookup_transform_with_alias(
                target_frame,
                source_frame,
                msg.header.stamp,
            )
            if tf_msg is None:
                fallback = self._convert_with_odom_pose_fallback(msg, source_frame, target_frame)
                if fallback is None:
                    return None, [], np.empty((0,), dtype=np.float64)
                return fallback

        points: list[tuple[float, float]] = []
        colors: list[str] = []
        confs: list[float] = []
        for cone in msg.cones:
            x = float(cone.position.x)
            y = float(cone.position.y)
            z = float(cone.position.z)
            if tf_msg is not None:
                x, y, z = self._transform_point(tf_msg, x, y, z)
            points.append((x, y))
            colors.append(normalize_color(cone.color))
            confs.append(float(np.clip(float(cone.confidence), 0.0, 1.0)))

        if not points:
            return np.empty((0, 2), dtype=np.float64), [], np.empty((0,), dtype=np.float64)
        return np.asarray(points, dtype=np.float64), colors, np.asarray(confs, dtype=np.float64)

    def _convert_with_odom_pose_fallback(
        self,
        msg: ConeDetectionArray,
        source_frame: str,
        target_frame: str,
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
            colors: list[str] = []
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
                colors.append(normalize_color(cone.color))
                confs.append(float(np.clip(float(cone.confidence), 0.0, 1.0)))
            return np.asarray(points, dtype=np.float64), colors, np.asarray(confs, dtype=np.float64)

        if source_is_base and self._is_alias(target_frame, self.odom_frame):
            cos_y = math.cos(odom_yaw)
            sin_y = math.sin(odom_yaw)
            points = []
            colors = []
            confs = []
            for cone in msg.cones:
                xb = float(cone.position.x)
                yb = float(cone.position.y)
                xo = odom_x + (cos_y * xb) - (sin_y * yb)
                yo = odom_y + (sin_y * xb) + (cos_y * yb)
                points.append((xo, yo))
                colors.append(normalize_color(cone.color))
                confs.append(float(np.clip(float(cone.confidence), 0.0, 1.0)))
            return np.asarray(points, dtype=np.float64), colors, np.asarray(confs, dtype=np.float64)

        return None

    def _apply_temporal_smoothing(self, centerline: np.ndarray) -> np.ndarray:
        if centerline.shape[0] == 0:
            self._previous_centerline = None
            return centerline
        prev = self._previous_centerline
        if prev is None or prev.shape[0] < 2 or centerline.shape[0] < 2:
            self._previous_centerline = centerline
            return centerline

        count_diff = abs(prev.shape[0] - centerline.shape[0])
        if count_diff > max(2, int(0.25 * centerline.shape[0])):
            self._previous_centerline = centerline
            return centerline

        prev_rs = self._resample_to_count(prev, centerline.shape[0])
        blended = (self.smoothing_alpha * centerline) + ((1.0 - self.smoothing_alpha) * prev_rs)
        self._previous_centerline = blended
        return blended

    @staticmethod
    def _resample_to_count(points: np.ndarray, count: int) -> np.ndarray:
        if points.shape[0] == count:
            return points
        if points.shape[0] <= 1 or count <= 1:
            return np.repeat(points[:1], count, axis=0)

        seg = points[1:] - points[:-1]
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        cum = np.concatenate(([0.0], np.cumsum(seg_len)))
        total = max(float(cum[-1]), 1e-6)
        samples = np.linspace(0.0, total, count)
        x = np.interp(samples, cum, points[:, 0])
        y = np.interp(samples, cum, points[:, 1])
        return np.column_stack((x, y)).astype(np.float64)

    def _publish_outputs(
        self,
        *,
        frame_id: str,
        centerline: np.ndarray,
        result: Optional[CoreResult],
        status: str,
        control_target_frame: Optional[np.ndarray],
        cmd_speed: float,
        cmd_steering: float,
        lookahead: float,
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
            status_text = (
                f'{status}  v_cmd={cmd_speed:.2f} m/s  '
                f'delta={cmd_steering:.2f} rad  Ld={lookahead:.2f} m'
            )
            markers = self._build_markers(
                now=now,
                frame_id=frame_id,
                result=result,
                centerline=centerline,
                status=status_text,
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
        status: str,
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
            marker_id = self._append_status_marker(arr, marker_id, frame_id, now, status)
            return arr

        if self.show_raw_cones and result.filtered_points.shape[0] > 0:
            marker = self._make_points_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns='filtered_cones',
                points=result.filtered_points,
                color=(0.8, 0.8, 0.8, 0.65),
                scale=0.18,
            )
            arr.markers.append(marker)
            marker_id += 1

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

        arr.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns='boundary_left',
                points=result.left_boundary,
                color=(0.2, 0.45, 1.0, 0.95),
                width=0.07,
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

        self._append_status_marker(arr, marker_id, frame_id, now, status)
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
            pt.z = 0.02
            marker.points.append(pt)
        return marker

    @staticmethod
    def _append_status_marker(
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        status: str,
    ) -> int:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'status'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.pose.position.x = 1.5
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.2
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
