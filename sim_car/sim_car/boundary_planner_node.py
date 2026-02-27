#!/usr/bin/env python3
"""Boundary-based cone planner with Pure Pursuit tracking."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import warnings
from typing import Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetectionArray
from visualization_msgs.msg import Marker, MarkerArray

NP_RANK_WARNING = getattr(np, 'RankWarning', RuntimeWarning)


@dataclass
class PlanResult:
    centerline: np.ndarray
    left_raw: np.ndarray
    right_raw: np.ndarray
    left_fit: np.ndarray
    right_fit: np.ndarray
    left_coeffs: Optional[np.ndarray]
    right_coeffs: Optional[np.ndarray]


class BoundaryPlannerNode(Node):
    """Fits cone boundaries, generates centerline, and tracks with Pure Pursuit."""

    def __init__(self) -> None:
        super().__init__('boundary_planner_node')

        self._declare_parameters()
        self._read_parameters()

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_cones_msg: Optional[ConeDetectionArray] = None
        self._latest_odom_msg: Optional[Odometry] = None
        self._latest_speed_mps: float = 0.0

        self._last_valid_plan: Optional[PlanResult] = None
        self._last_valid_time_sec: float = -1.0
        self._last_speed_cmd: Optional[float] = None
        self._last_steering_cmd: Optional[float] = None
        self._last_throttled_log_sec: dict[str, float] = {}
        self._active_base_frame: str = self.base_frame

        self._cmd_pub = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
        self._path_pub = self.create_publisher(Path, self.path_topic, 10)
        self._markers_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)

        self.create_subscription(
            ConeDetectionArray,
            self.cones_topic,
            self._cones_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_cb,
            10,
        )

        loop_hz = max(1.0, float(self.loop_rate_hz))
        self.create_timer(1.0 / loop_hz, self._on_timer)

        self.get_logger().info(
            'boundary_planner_node ready '
            f'cones={self.cones_topic} odom={self.odom_topic} cmd={self.cmd_topic} '
            f'base_frame={self.base_frame} fit={self.fit_model} ransac={self.use_ransac}'
        )

    def _declare_parameters(self) -> None:
        defaults = {
            'topics.cones_topic': 'sim/stereo/perception/cones_3d',
            'topics.odom_topic': 'sim/odom',
            'topics.cmd_topic': '/cmd',
            'topics.path_topic': 'sim/planner/centerline_path',
            'topics.markers_topic': 'sim/planner/markers',
            'gating.min_confidence': 0.35,
            'gating.x_min_m': 0.5,
            'gating.x_max_m': 20.0,
            'gating.y_abs_max_m': 8.0,
            'gating.max_cone_age_s': 0.25,
            'gating.hold_last_valid_s': 0.50,
            'fitting.fit_model': 'poly2',
            'fitting.min_points_per_side': 4,
            'fitting.max_points_per_side': 15,
            'fitting.use_ransac': True,
            'fitting.ransac_iters': 60,
            'fitting.ransac_inlier_thresh_m': 0.25,
            'path_generation.horizon_m': 15.0,
            'path_generation.sample_count': 25,
            'path_generation.track_width_m': 3.0,
            'path_generation.side_offset_m': 0.0,
            'pure_pursuit.lookahead_min_m': 2.5,
            'pure_pursuit.lookahead_gain': 0.6,
            'pure_pursuit.steering_limit_rad': 0.5,
            'pure_pursuit.wheelbase_m': 1.6,
            'pure_pursuit.stop_if_no_path': True,
            'speed_control.speed_min_mps': 1.5,
            'speed_control.speed_max_mps': 8.0,
            'speed_control.curvature_speed_gain': 3.0,
            'speed_control.lowpass_speed_alpha': 0.25,
            'debug.publish_path': True,
            'debug.publish_markers': True,
            'debug.log_throttle_s': 1.0,
            'frames.base_frame': 'base_footprint',
            'frames.tf_timeout_s': 0.02,
            'colors.left_boundary': ['blue'],
            'colors.right_boundary': ['yellow'],
            'planner_rate_hz': 30.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        self.cones_topic = str(self.get_parameter('topics.cones_topic').value)
        self.odom_topic = str(self.get_parameter('topics.odom_topic').value)
        self.cmd_topic = str(self.get_parameter('topics.cmd_topic').value)
        self.path_topic = str(self.get_parameter('topics.path_topic').value)
        self.markers_topic = str(self.get_parameter('topics.markers_topic').value)

        self.min_confidence = float(self.get_parameter('gating.min_confidence').value)
        self.x_min_m = float(self.get_parameter('gating.x_min_m').value)
        self.x_max_m = float(self.get_parameter('gating.x_max_m').value)
        self.y_abs_max_m = float(self.get_parameter('gating.y_abs_max_m').value)
        self.max_cone_age_s = max(0.01, float(self.get_parameter('gating.max_cone_age_s').value))
        self.hold_last_valid_s = max(0.0, float(self.get_parameter('gating.hold_last_valid_s').value))

        self.fit_model = str(self.get_parameter('fitting.fit_model').value).strip().lower()
        if self.fit_model not in {'poly1', 'poly2'}:
            self.fit_model = 'poly2'
        self.fit_degree = 1 if self.fit_model == 'poly1' else 2
        self.min_points_per_side = max(2, int(self.get_parameter('fitting.min_points_per_side').value))
        self.max_points_per_side = max(
            self.min_points_per_side,
            int(self.get_parameter('fitting.max_points_per_side').value),
        )
        self.use_ransac = bool(self.get_parameter('fitting.use_ransac').value)
        self.ransac_iters = max(1, int(self.get_parameter('fitting.ransac_iters').value))
        self.ransac_inlier_thresh_m = max(0.01, float(self.get_parameter('fitting.ransac_inlier_thresh_m').value))

        self.horizon_m = max(2.0, float(self.get_parameter('path_generation.horizon_m').value))
        self.sample_count = max(5, int(self.get_parameter('path_generation.sample_count').value))
        self.track_width_m = max(0.2, float(self.get_parameter('path_generation.track_width_m').value))
        self.side_offset_m = float(self.get_parameter('path_generation.side_offset_m').value)

        self.lookahead_min_m = max(0.5, float(self.get_parameter('pure_pursuit.lookahead_min_m').value))
        self.lookahead_gain = max(0.0, float(self.get_parameter('pure_pursuit.lookahead_gain').value))
        self.steering_limit_rad = max(0.01, float(self.get_parameter('pure_pursuit.steering_limit_rad').value))
        self.wheelbase_m = max(0.1, float(self.get_parameter('pure_pursuit.wheelbase_m').value))
        self.stop_if_no_path = bool(self.get_parameter('pure_pursuit.stop_if_no_path').value)

        self.speed_min_mps = max(0.0, float(self.get_parameter('speed_control.speed_min_mps').value))
        self.speed_max_mps = max(self.speed_min_mps, float(self.get_parameter('speed_control.speed_max_mps').value))
        self.curvature_speed_gain = max(0.0, float(self.get_parameter('speed_control.curvature_speed_gain').value))
        self.lowpass_speed_alpha = float(self.get_parameter('speed_control.lowpass_speed_alpha').value)
        self.lowpass_speed_alpha = max(0.0, min(1.0, self.lowpass_speed_alpha))

        self.publish_path = bool(self.get_parameter('debug.publish_path').value)
        self.publish_markers = bool(self.get_parameter('debug.publish_markers').value)
        self.log_throttle_s = max(0.1, float(self.get_parameter('debug.log_throttle_s').value))

        self.base_frame = str(self.get_parameter('frames.base_frame').value).strip() or 'base_footprint'
        self.tf_timeout_s = max(0.0, float(self.get_parameter('frames.tf_timeout_s').value))
        self.loop_rate_hz = max(1.0, float(self.get_parameter('planner_rate_hz').value))

        self.left_boundary_colors = {
            self._normalize_color(token) for token in self.get_parameter('colors.left_boundary').value
        }
        self.right_boundary_colors = {
            self._normalize_color(token) for token in self.get_parameter('colors.right_boundary').value
        }
        self.left_boundary_colors.discard('unknown')
        self.right_boundary_colors.discard('unknown')
        if not self.left_boundary_colors:
            self.left_boundary_colors = {'blue'}
        if not self.right_boundary_colors:
            self.right_boundary_colors = {'yellow'}

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
        now = self.get_clock().now()
        now_sec = self._clock_to_sec(now.nanoseconds)
        current_plan: Optional[PlanResult] = None

        cones_msg = self._latest_cones_msg
        if cones_msg is not None:
            cone_stamp_sec = self._stamp_to_sec(cones_msg.header.stamp.sec, cones_msg.header.stamp.nanosec)
            age = now_sec - cone_stamp_sec
            if age <= self.max_cone_age_s:
                current_plan = self._build_plan(cones_msg)
            else:
                self._warn_throttled(
                    'stale_cones',
                    f'cones stale by {age:.3f}s (limit {self.max_cone_age_s:.3f}s)',
                )

        if current_plan is not None:
            self._last_valid_plan = current_plan
            self._last_valid_time_sec = now_sec
        else:
            if (
                self._last_valid_plan is not None
                and self._last_valid_time_sec > 0.0
                and (now_sec - self._last_valid_time_sec) <= self.hold_last_valid_s
            ):
                current_plan = self._last_valid_plan

        if current_plan is None:
            self._publish_no_path(now)
            return

        steering, speed_cmd, kappa, lookahead, target_point = self._compute_control(current_plan.centerline)
        self._publish_cmd(now, speed_cmd, steering)

        if self.publish_path:
            self._publish_centerline_path(now, current_plan.centerline)
        if self.publish_markers:
            self._publish_debug_markers(
                now=now,
                plan=current_plan,
                target_point=target_point,
                speed_cmd=speed_cmd,
                lookahead=lookahead,
                steering=steering,
                kappa=kappa,
            )

    def _build_plan(self, cones_msg: ConeDetectionArray) -> Optional[PlanResult]:
        left_raw, right_raw, plan_frame = self._extract_boundary_points(cones_msg)
        self._active_base_frame = plan_frame
        left_coeffs = self._fit_boundary(left_raw)
        right_coeffs = self._fit_boundary(right_raw)

        centerline, sample_x = self._build_centerline(left_coeffs, right_coeffs, left_raw, right_raw)
        if centerline is None:
            return None

        left_fit = self._evaluate_fit(left_coeffs, sample_x)
        right_fit = self._evaluate_fit(right_coeffs, sample_x)
        return PlanResult(
            centerline=centerline,
            left_raw=left_raw,
            right_raw=right_raw,
            left_fit=left_fit,
            right_fit=right_fit,
            left_coeffs=left_coeffs,
            right_coeffs=right_coeffs,
        )

    def _extract_boundary_points(self, cones_msg: ConeDetectionArray) -> tuple[np.ndarray, np.ndarray, str]:
        frame_id = str(cones_msg.header.frame_id).strip()
        if not frame_id:
            frame_id = self.base_frame
            self._warn_throttled('empty_frame', 'cones message has empty frame_id; assuming base frame')

        effective_base_frame = self.base_frame
        transform = None
        if frame_id != self.base_frame:
            transform = self._lookup_transform(
                target_frame=self.base_frame,
                source_frame=frame_id,
                sec=cones_msg.header.stamp.sec,
                nanosec=cones_msg.header.stamp.nanosec,
            )
            if transform is None:
                candidate_base = self._resolve_namespaced_base_frame(
                    source_frame=frame_id,
                    requested_base=self.base_frame,
                )
                if candidate_base:
                    if frame_id == candidate_base:
                        effective_base_frame = candidate_base
                    else:
                        candidate_tf = self._lookup_transform(
                            target_frame=candidate_base,
                            source_frame=frame_id,
                            sec=cones_msg.header.stamp.sec,
                            nanosec=cones_msg.header.stamp.nanosec,
                        )
                        if candidate_tf is not None:
                            transform = candidate_tf
                            effective_base_frame = candidate_base
                    if effective_base_frame != self.base_frame:
                        self._warn_throttled(
                            'namespaced_base_frame',
                            f'using resolved base frame "{effective_base_frame}" instead of "{self.base_frame}"',
                        )
                if transform is None and frame_id != effective_base_frame:
                    self._warn_throttled(
                        'missing_tf',
                        f'cannot transform cones from "{frame_id}" to "{self.base_frame}"',
                    )
                    return np.empty((0, 2)), np.empty((0, 2)), self.base_frame

        left_points: list[tuple[float, float]] = []
        right_points: list[tuple[float, float]] = []

        for cone in cones_msg.cones:
            confidence = float(cone.confidence)
            if confidence < self.min_confidence:
                continue

            x = float(cone.position.x)
            y = float(cone.position.y)
            z = float(cone.position.z)
            if transform is not None:
                x, y, z = self._transform_point(transform, x, y, z)

            if x < self.x_min_m or x > self.x_max_m:
                continue
            if abs(y) > self.y_abs_max_m:
                continue

            color = self._normalize_color(cone.color)
            if color in self.left_boundary_colors:
                left_points.append((x, y))
            elif color in self.right_boundary_colors:
                right_points.append((x, y))

        left_array = np.array(left_points, dtype=np.float64) if left_points else np.empty((0, 2))
        right_array = np.array(right_points, dtype=np.float64) if right_points else np.empty((0, 2))
        return left_array, right_array, effective_base_frame

    def _fit_boundary(self, points: np.ndarray) -> Optional[np.ndarray]:
        if points.shape[0] < self.min_points_per_side:
            return None

        ordered = points[np.argsort(points[:, 0])]
        selected = ordered[: self.max_points_per_side]
        if selected.shape[0] < self.min_points_per_side:
            return None

        if self.use_ransac:
            coeffs = self._fit_poly_ransac(selected, degree=self.fit_degree)
            if coeffs is not None:
                return coeffs

        return self._fit_poly_ls(selected, degree=self.fit_degree)

    def _fit_poly_ransac(self, points: np.ndarray, degree: int) -> Optional[np.ndarray]:
        min_subset = degree + 1
        if points.shape[0] < min_subset:
            return None

        best_inliers: Optional[np.ndarray] = None
        best_count = -1
        best_error = float('inf')
        rng = np.random.default_rng()

        for _ in range(self.ransac_iters):
            subset_idx = rng.choice(points.shape[0], size=min_subset, replace=False)
            subset = points[subset_idx]
            coeffs = self._fit_poly_ls(subset, degree=degree)
            if coeffs is None:
                continue

            y_hat = np.polyval(coeffs, points[:, 0])
            residual = np.abs(y_hat - points[:, 1])
            inliers = residual <= self.ransac_inlier_thresh_m
            count = int(np.count_nonzero(inliers))
            if count <= 0:
                continue

            inlier_error = float(np.sum(residual[inliers]))
            if count > best_count or (count == best_count and inlier_error < best_error):
                best_count = count
                best_error = inlier_error
                best_inliers = inliers

        if best_inliers is None:
            return None

        inlier_count = int(np.count_nonzero(best_inliers))
        if inlier_count < max(self.min_points_per_side, min_subset):
            return None

        return self._fit_poly_ls(points[best_inliers], degree=degree)

    @staticmethod
    def _fit_poly_ls(points: np.ndarray, degree: int) -> Optional[np.ndarray]:
        if points.shape[0] < (degree + 1):
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', NP_RANK_WARNING)
                coeffs = np.polyfit(points[:, 0], points[:, 1], deg=degree)
            return coeffs.astype(np.float64, copy=False)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None

    def _build_centerline(
        self,
        left_coeffs: Optional[np.ndarray],
        right_coeffs: Optional[np.ndarray],
        left_points: np.ndarray,
        right_points: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if left_coeffs is None and right_coeffs is None:
            return None, None

        max_seen_x = 0.0
        if left_points.shape[0] > 0:
            max_seen_x = max(max_seen_x, float(np.max(left_points[:, 0])))
        if right_points.shape[0] > 0:
            max_seen_x = max(max_seen_x, float(np.max(right_points[:, 0])))

        horizon_x = self.horizon_m if max_seen_x <= 0.0 else min(self.horizon_m, max_seen_x)
        if horizon_x <= 0.2:
            return None, None

        sample_x = np.linspace(0.0, horizon_x, num=self.sample_count, dtype=np.float64)
        if left_coeffs is not None and right_coeffs is not None:
            y_center = 0.5 * (np.polyval(left_coeffs, sample_x) + np.polyval(right_coeffs, sample_x))
        elif left_coeffs is not None:
            y_center = np.polyval(left_coeffs, sample_x) - 0.5 * self.track_width_m
        else:
            y_center = np.polyval(right_coeffs, sample_x) + 0.5 * self.track_width_m

        y_center = y_center + self.side_offset_m
        centerline = np.column_stack((sample_x, y_center))
        return centerline, sample_x

    @staticmethod
    def _evaluate_fit(coeffs: Optional[np.ndarray], sample_x: Optional[np.ndarray]) -> np.ndarray:
        if coeffs is None or sample_x is None:
            return np.empty((0, 2), dtype=np.float64)
        y_values = np.polyval(coeffs, sample_x)
        return np.column_stack((sample_x, y_values))

    def _compute_control(self, centerline: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
        speed = max(0.0, float(self._latest_speed_mps))
        lookahead = max(self.lookahead_min_m, self.lookahead_min_m + self.lookahead_gain * speed)

        distances = np.hypot(centerline[:, 0], centerline[:, 1])
        idx = int(np.argmin(np.abs(distances - lookahead)))
        target = centerline[idx]

        denom = max(lookahead * lookahead, 1e-6)
        kappa = 2.0 * float(target[1]) / denom
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

    def _publish_no_path(self, now) -> None:
        if self.stop_if_no_path:
            self._publish_cmd(now, speed_mps=0.0, steering_rad=0.0)
            self._last_speed_cmd = 0.0
            self._last_steering_cmd = 0.0
        elif self._last_speed_cmd is not None and self._last_steering_cmd is not None:
            self._publish_cmd(now, speed_mps=float(self._last_speed_cmd), steering_rad=float(self._last_steering_cmd))

        if self.publish_path:
            self._publish_centerline_path(now, np.empty((0, 2), dtype=np.float64))
        if self.publish_markers:
            self._publish_clear_markers(now)

    def _publish_cmd(self, now, speed_mps: float, steering_rad: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = now.to_msg()
        msg.drive.speed = float(speed_mps)
        msg.drive.steering_angle = float(steering_rad)
        self._cmd_pub.publish(msg)

    def _publish_centerline_path(self, now, centerline: np.ndarray) -> None:
        path_msg = Path()
        path_msg.header.stamp = now.to_msg()
        path_msg.header.frame_id = self._active_base_frame
        for x, y in centerline:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self._path_pub.publish(path_msg)

    def _publish_clear_markers(self, now) -> None:
        arr = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.stamp = now.to_msg()
        clear_marker.header.frame_id = self._active_base_frame
        clear_marker.action = Marker.DELETEALL
        arr.markers.append(clear_marker)
        self._markers_pub.publish(arr)

    def _publish_debug_markers(
        self,
        now,
        plan: PlanResult,
        target_point: np.ndarray,
        speed_cmd: float,
        lookahead: float,
        steering: float,
        kappa: float,
    ) -> None:
        markers = MarkerArray()

        clear_marker = Marker()
        clear_marker.header.stamp = now.to_msg()
        clear_marker.header.frame_id = self._active_base_frame
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        marker_id = 0
        marker_id = self._append_sphere_list(
            markers,
            marker_id=marker_id,
            now=now,
            ns='raw_left',
            points_xy=plan.left_raw,
            rgb=(0.1, 0.4, 1.0),
            scale=0.22,
        )
        marker_id = self._append_sphere_list(
            markers,
            marker_id=marker_id,
            now=now,
            ns='raw_right',
            points_xy=plan.right_raw,
            rgb=(1.0, 0.9, 0.1),
            scale=0.22,
        )
        marker_id = self._append_line_strip(
            markers,
            marker_id=marker_id,
            now=now,
            ns='fit_left',
            points_xy=plan.left_fit,
            rgb=(0.0, 0.2, 0.9),
            width=0.08,
        )
        marker_id = self._append_line_strip(
            markers,
            marker_id=marker_id,
            now=now,
            ns='fit_right',
            points_xy=plan.right_fit,
            rgb=(0.9, 0.7, 0.0),
            width=0.08,
        )
        marker_id = self._append_line_strip(
            markers,
            marker_id=marker_id,
            now=now,
            ns='centerline',
            points_xy=plan.centerline,
            rgb=(0.9, 0.1, 0.1),
            width=0.10,
        )
        marker_id = self._append_target_marker(
            markers,
            marker_id=marker_id,
            now=now,
            target_xy=target_point,
        )
        self._append_text_marker(
            markers,
            marker_id=marker_id,
            now=now,
            text=f'v={speed_cmd:.2f} m/s  Ld={lookahead:.2f} m  delta={steering:.3f} rad  kappa={kappa:.3f}',
        )

        self._markers_pub.publish(markers)

    def _append_sphere_list(
        self,
        markers: MarkerArray,
        marker_id: int,
        now,
        ns: str,
        points_xy: np.ndarray,
        rgb: tuple[float, float, float],
        scale: float,
    ) -> int:
        marker = Marker()
        marker.header.stamp = now.to_msg()
        marker.header.frame_id = self._active_base_frame
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = 0.95
        marker.color.r = float(rgb[0])
        marker.color.g = float(rgb[1])
        marker.color.b = float(rgb[2])

        for x, y in points_xy:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            marker.points.append(point)

        markers.markers.append(marker)
        return marker_id + 1

    def _append_line_strip(
        self,
        markers: MarkerArray,
        marker_id: int,
        now,
        ns: str,
        points_xy: np.ndarray,
        rgb: tuple[float, float, float],
        width: float,
    ) -> int:
        marker = Marker()
        marker.header.stamp = now.to_msg()
        marker.header.frame_id = self._active_base_frame
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.a = 0.95
        marker.color.r = float(rgb[0])
        marker.color.g = float(rgb[1])
        marker.color.b = float(rgb[2])

        for x, y in points_xy:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            marker.points.append(point)

        markers.markers.append(marker)
        return marker_id + 1

    def _append_target_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        now,
        target_xy: np.ndarray,
    ) -> int:
        marker = Marker()
        marker.header.stamp = now.to_msg()
        marker.header.frame_id = self._active_base_frame
        marker.ns = 'target'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.28
        marker.scale.y = 0.28
        marker.scale.z = 0.28
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.position.x = float(target_xy[0])
        marker.pose.position.y = float(target_xy[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        markers.markers.append(marker)
        return marker_id + 1

    def _append_text_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        now,
        text: str,
    ) -> int:
        marker = Marker()
        marker.header.stamp = now.to_msg()
        marker.header.frame_id = self._active_base_frame
        marker.ns = 'status'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 0.35
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.pose.position.x = 1.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0
        marker.text = text
        markers.markers.append(marker)
        return marker_id + 1

    def _lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        sec: int,
        nanosec: int,
    ):
        try:
            stamp = Time(seconds=int(sec), nanoseconds=int(nanosec))
            timeout = Duration(seconds=self.tf_timeout_s)
            return self._tf_buffer.lookup_transform(target_frame, source_frame, stamp, timeout=timeout)
        except TransformException:
            return None

    @staticmethod
    def _transform_point(transform, x: float, y: float, z: float) -> tuple[float, float, float]:
        q = transform.transform.rotation
        t = transform.transform.translation

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

        px = r00 * x + r01 * y + r02 * z + float(t.x)
        py = r10 * x + r11 * y + r12 * z + float(t.y)
        pz = r20 * x + r21 * y + r22 * z + float(t.z)
        return float(px), float(py), float(pz)

    def _warn_throttled(self, key: str, message: str) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get(key, -1.0)
        if (now_sec - last_sec) >= self.log_throttle_s:
            self.get_logger().warn(message)
            self._last_throttled_log_sec[key] = now_sec

    @staticmethod
    def _normalize_color(token: str) -> str:
        value = str(token).strip().lower().replace('-', '_').replace(' ', '_')
        if 'big_orange' in value or ('big' in value and 'orange' in value):
            return 'big_orange'
        if 'orange' in value:
            return 'orange'
        if 'yellow' in value:
            return 'yellow'
        if 'blue' in value:
            return 'blue'
        return 'unknown'

    @staticmethod
    def _resolve_namespaced_base_frame(source_frame: str, requested_base: str) -> str:
        requested = str(requested_base).strip().strip('/')
        source = str(source_frame).strip().strip('/')
        if not requested or not source:
            return ''
        if '/' in requested:
            return ''
        marker = f'/{requested}/'
        source_with_slashes = f'/{source}/'
        idx = source_with_slashes.find(marker)
        if idx < 0:
            return ''
        prefix = source_with_slashes[1:idx].strip('/')
        if not prefix:
            return requested
        return f'{prefix}/{requested}'

    @staticmethod
    def _stamp_to_sec(sec: int, nanosec: int) -> float:
        return float(sec) + float(nanosec) * 1e-9

    @staticmethod
    def _clock_to_sec(nanoseconds: int) -> float:
        return float(nanoseconds) * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoundaryPlannerNode()
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
