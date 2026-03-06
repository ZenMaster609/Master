#!/usr/bin/env python3
"""Boundary-based cone planner with Pure Pursuit tracking."""

from __future__ import annotations

from collections import deque
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
    control_centerline: np.ndarray
    left_raw: np.ndarray
    right_raw: np.ndarray
    left_fit_input: np.ndarray
    right_fit_input: np.ndarray
    left_fit: np.ndarray
    right_fit: np.ndarray
    left_coeffs: Optional[np.ndarray]
    right_coeffs: Optional[np.ndarray]
    left_raw_count: int
    right_raw_count: int
    left_used_count: int
    right_used_count: int
    left_fit_ok: bool
    right_fit_ok: bool


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
        self._startup_delay_released: bool = self.startup_delay_s <= 0.0
        self._recent_valid_centerlines: deque[tuple[float, np.ndarray]] = deque()

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
            f'base_frame={self.base_frame} fit={self.fit_model} ransac={self.use_ransac} '
            f'startup_delay_s={self.startup_delay_s:.2f}'
        )
        self.get_logger().info(
            'cone axis mapping '
            f'forward={self.cone_forward_sign:+.0f}*{self.cone_forward_axis} '
            f'lateral={self.cone_lateral_sign:+.0f}*{self.cone_lateral_axis}'
        )
        if self.startup_delay_s > 0.0:
            self.get_logger().info(
                f'holding planning/control until simulation time reaches {self.startup_delay_s:.2f}s'
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
            'gating.startup_delay_s': 10.0,
            'fitting.fit_model': 'poly2',
            'fitting.min_points_per_side': 4,
            'fitting.max_points_per_side': 15,
            'fitting.allow_single_point_fallback': True,
            'fitting.use_ransac': True,
            'fitting.ransac_iters': 60,
            'fitting.ransac_inlier_thresh_m': 0.25,
            'path_generation.preview_x_max_m': -1.0,
            'path_generation.preview_horizon_m': -1.0,
            'path_generation.control_horizon_m': 5.5,
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
            'debug.show_cone_counts_in_status': True,
            'debug.log_throttle_s': 1.0,
            'frames.base_frame': 'base_footprint',
            'frames.tf_timeout_s': 0.02,
            'colors.left_boundary': ['blue'],
            'colors.right_boundary': ['yellow'],
            'colors.split_unmapped_by_y_sign': True,
            'colors.enforce_expected_side_sign': True,
            'colors.side_sign_min_abs_y_m': 0.1,
            'cone_axes.forward_axis': 'x',
            'cone_axes.lateral_axis': 'y',
            'cone_axes.forward_sign': 1.0,
            'cone_axes.lateral_sign': 1.0,
            'validation.reject_crossing_paths': True,
            'validation.min_boundary_separation_m': 0.2,
            'validation.max_history_frames': 8,
            'validation.consistency_horizon_m': 6.0,
            'validation.max_centerline_jump_m': 1.25,
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
        self.startup_delay_s = max(0.0, float(self.get_parameter('gating.startup_delay_s').value))

        self.fit_model = str(self.get_parameter('fitting.fit_model').value).strip().lower()
        if self.fit_model not in {'poly1', 'poly2'}:
            self.fit_model = 'poly2'
        self.fit_degree = 1 if self.fit_model == 'poly1' else 2
        self.min_points_per_side = max(2, int(self.get_parameter('fitting.min_points_per_side').value))
        self.max_points_per_side = max(
            self.min_points_per_side,
            int(self.get_parameter('fitting.max_points_per_side').value),
        )
        self.allow_single_point_fallback = bool(self.get_parameter('fitting.allow_single_point_fallback').value)
        self.use_ransac = bool(self.get_parameter('fitting.use_ransac').value)
        self.ransac_iters = max(1, int(self.get_parameter('fitting.ransac_iters').value))
        self.ransac_inlier_thresh_m = max(0.01, float(self.get_parameter('fitting.ransac_inlier_thresh_m').value))

        self.legacy_horizon_m = max(2.0, float(self.get_parameter('path_generation.horizon_m').value))
        preview_x_max_raw = float(self.get_parameter('path_generation.preview_x_max_m').value)
        if preview_x_max_raw > 0.0:
            self.preview_x_max_m = preview_x_max_raw
        else:
            self.preview_x_max_m = self.x_max_m

        preview_horizon_raw = float(self.get_parameter('path_generation.preview_horizon_m').value)
        self._using_legacy_preview_horizon = preview_horizon_raw <= 0.0
        if self._using_legacy_preview_horizon:
            self.preview_horizon_m = self.legacy_horizon_m
        else:
            self.preview_horizon_m = max(2.0, preview_horizon_raw)
        self.control_horizon_m = max(
            0.5,
            float(self.get_parameter('path_generation.control_horizon_m').value),
        )
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
        self.show_cone_counts_in_status = bool(self.get_parameter('debug.show_cone_counts_in_status').value)
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
        self.split_unmapped_by_y_sign = bool(self.get_parameter('colors.split_unmapped_by_y_sign').value)
        self.enforce_expected_side_sign = bool(self.get_parameter('colors.enforce_expected_side_sign').value)
        self.side_sign_min_abs_y_m = max(0.0, float(self.get_parameter('colors.side_sign_min_abs_y_m').value))
        self.left_boundary_colors.discard('unknown')
        self.right_boundary_colors.discard('unknown')
        if not self.left_boundary_colors:
            self.left_boundary_colors = {'blue'}
        if not self.right_boundary_colors:
            self.right_boundary_colors = {'yellow'}

        self.cone_forward_axis = self._read_axis_parameter('cone_axes.forward_axis', default='x')
        self.cone_lateral_axis = self._read_axis_parameter('cone_axes.lateral_axis', default='y')
        if self.cone_lateral_axis == self.cone_forward_axis:
            self.get_logger().warn(
                f'cone_axes.lateral_axis matches forward_axis "{self.cone_forward_axis}"; using "y" instead'
            )
            self.cone_lateral_axis = 'y' if self.cone_forward_axis != 'y' else 'x'
        self.cone_forward_sign = self._read_axis_sign_parameter('cone_axes.forward_sign')
        self.cone_lateral_sign = self._read_axis_sign_parameter('cone_axes.lateral_sign')

        self.reject_crossing_paths = bool(self.get_parameter('validation.reject_crossing_paths').value)
        self.min_boundary_separation_m = max(
            0.0,
            float(self.get_parameter('validation.min_boundary_separation_m').value),
        )
        self.max_history_frames = max(1, int(self.get_parameter('validation.max_history_frames').value))
        self.consistency_horizon_m = max(0.5, float(self.get_parameter('validation.consistency_horizon_m').value))
        self.max_centerline_jump_m = max(
            0.0,
            float(self.get_parameter('validation.max_centerline_jump_m').value),
        )

        if self.preview_x_max_m > self.x_max_m:
            self.get_logger().warn(
                f'path_generation.preview_x_max_m={self.preview_x_max_m:.2f} exceeds '
                f'gating.x_max_m={self.x_max_m:.2f}; clamping to gating limit'
            )
            self.preview_x_max_m = self.x_max_m
        if self.preview_horizon_m < self.control_horizon_m:
            self.get_logger().warn(
                f'path_generation.preview_horizon_m={self.preview_horizon_m:.2f} is shorter than '
                f'control_horizon_m={self.control_horizon_m:.2f}; clamping control horizon'
            )
            self.control_horizon_m = self.preview_horizon_m
        if self._using_legacy_preview_horizon:
            self.get_logger().info(
                'path_generation.preview_horizon_m not set; falling back to path_generation.horizon_m'
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
        now = self.get_clock().now()
        now_sec = self._clock_to_sec(now.nanoseconds)
        if now_sec < self.startup_delay_s:
            self._publish_no_path(now)
            return
        if not self._startup_delay_released:
            self._startup_delay_released = True
            self.get_logger().info('startup delay elapsed; enabling planning/control')

        current_plan: Optional[PlanResult] = None

        cones_msg = self._latest_cones_msg
        if cones_msg is not None:
            cone_stamp_sec = self._stamp_to_sec(cones_msg.header.stamp.sec, cones_msg.header.stamp.nanosec)
            age = now_sec - cone_stamp_sec
            if age <= self.max_cone_age_s:
                current_plan = self._build_plan(cones_msg, now_sec)
            else:
                self._warn_throttled(
                    'stale_cones',
                    f'cones stale by {age:.3f}s (limit {self.max_cone_age_s:.3f}s)',
                )

        if current_plan is not None and not self._validate_plan(current_plan, now_sec):
            current_plan = None

        if current_plan is not None:
            self._last_valid_plan = current_plan
            self._last_valid_time_sec = now_sec
            self._record_valid_centerline(now_sec, current_plan.centerline)
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

        steering, speed_cmd, kappa, lookahead, target_point = self._compute_control(current_plan.control_centerline)
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

    def _build_plan(self, cones_msg: ConeDetectionArray, now_sec: float) -> Optional[PlanResult]:
        left_raw, right_raw, plan_frame = self._extract_boundary_points(cones_msg)
        self._active_base_frame = plan_frame
        left_raw = self._limit_points_by_x(left_raw, self.preview_x_max_m)
        right_raw = self._limit_points_by_x(right_raw, self.preview_x_max_m)
        left_coeffs, left_fit_input = self._fit_boundary(left_raw)
        right_coeffs, right_fit_input = self._fit_boundary(right_raw)

        previous_plan: Optional[PlanResult] = None
        if (
            self._last_valid_plan is not None
            and self._last_valid_time_sec > 0.0
            and (now_sec - self._last_valid_time_sec) <= self.hold_last_valid_s
        ):
            previous_plan = self._last_valid_plan

        reused_previous_side = False
        if previous_plan is not None:
            if left_raw.shape[0] < self.min_points_per_side and previous_plan.left_coeffs is not None:
                left_coeffs = previous_plan.left_coeffs
                if left_fit_input.shape[0] == 0:
                    left_fit_input = previous_plan.left_fit_input
                reused_previous_side = True
            if right_raw.shape[0] < self.min_points_per_side and previous_plan.right_coeffs is not None:
                right_coeffs = previous_plan.right_coeffs
                if right_fit_input.shape[0] == 0:
                    right_fit_input = previous_plan.right_fit_input
                reused_previous_side = True
        if reused_previous_side:
            self._warn_throttled(
                'reuse_previous_side_fit',
                'reusing previous boundary fit for sparse/missing side to avoid abrupt path jumps',
            )

        centerline, sample_x = self._build_centerline(
            left_coeffs,
            right_coeffs,
            left_raw,
            right_raw,
            horizon_m=self.preview_horizon_m,
        )
        if centerline is None:
            return None
        control_centerline = self._truncate_centerline(centerline, self.control_horizon_m)
        if control_centerline.shape[0] == 0:
            control_centerline = centerline

        left_fit = self._evaluate_fit(left_coeffs, sample_x)
        right_fit = self._evaluate_fit(right_coeffs, sample_x)
        return PlanResult(
            centerline=centerline,
            control_centerline=control_centerline,
            left_raw=left_raw,
            right_raw=right_raw,
            left_fit_input=left_fit_input,
            right_fit_input=right_fit_input,
            left_fit=left_fit,
            right_fit=right_fit,
            left_coeffs=left_coeffs,
            right_coeffs=right_coeffs,
            left_raw_count=int(left_raw.shape[0]),
            right_raw_count=int(right_raw.shape[0]),
            left_used_count=int(left_fit_input.shape[0]),
            right_used_count=int(right_fit_input.shape[0]),
            left_fit_ok=left_coeffs is not None,
            right_fit_ok=right_coeffs is not None,
        )

    def _extract_boundary_points(self, cones_msg: ConeDetectionArray) -> tuple[np.ndarray, np.ndarray, str]:
        frame_id = str(cones_msg.header.frame_id).strip()
        if not frame_id:
            frame_id = self.base_frame
            self._warn_throttled('empty_frame', 'cones message has empty frame_id; assuming base frame')

        effective_base_frame = self.base_frame
        transform = None
        if frame_id != self.base_frame:
            target_candidates = self._target_frame_candidates(
                source_frame=frame_id,
                requested_base=self.base_frame,
            )
            source_candidates = self._source_frame_candidates(
                source_frame=frame_id,
                requested_base=self.base_frame,
            )
            resolved_source = frame_id
            found = False
            for target_frame in target_candidates:
                if frame_id == target_frame:
                    effective_base_frame = target_frame
                    found = True
                    break
                for source_frame in source_candidates:
                    if source_frame == target_frame:
                        continue
                    candidate_tf = self._lookup_transform(
                        target_frame=target_frame,
                        source_frame=source_frame,
                        sec=cones_msg.header.stamp.sec,
                        nanosec=cones_msg.header.stamp.nanosec,
                    )
                    if candidate_tf is None:
                        continue
                    transform = candidate_tf
                    effective_base_frame = target_frame
                    resolved_source = source_frame
                    found = True
                    break
                if found:
                    break

            if not found:
                self._warn_throttled(
                    'missing_tf',
                    f'cannot transform cones from "{frame_id}" to "{self.base_frame}"',
                )
                return np.empty((0, 2)), np.empty((0, 2)), self.base_frame

            if effective_base_frame != self.base_frame or resolved_source != frame_id:
                self._warn_throttled(
                    'namespaced_base_frame',
                    f'using transform {resolved_source}->{effective_base_frame} '
                    f'instead of {frame_id}->{self.base_frame}',
                )

        left_points: list[tuple[float, float]] = []
        right_points: list[tuple[float, float]] = []
        left_wrong_side_points: list[tuple[float, float]] = []
        right_wrong_side_points: list[tuple[float, float]] = []
        gated_count = 0
        assigned_count = 0
        fallback_split_count = 0
        unmapped_count = 0
        wrong_side_count = 0

        for cone in cones_msg.cones:
            confidence = float(cone.confidence)
            if confidence < self.min_confidence:
                continue

            px = float(cone.position.x)
            py = float(cone.position.y)
            pz = float(cone.position.z)
            if transform is not None:
                px, py, pz = self._transform_point(transform, px, py, pz)

            x = self.cone_forward_sign * self._axis_value(px, py, pz, self.cone_forward_axis)
            y = self.cone_lateral_sign * self._axis_value(px, py, pz, self.cone_lateral_axis)

            if x < self.x_min_m or x > self.x_max_m:
                continue
            if abs(y) > self.y_abs_max_m:
                continue
            gated_count += 1

            color = self._normalize_color(cone.color)
            if color in self.left_boundary_colors:
                if self.enforce_expected_side_sign and y < -self.side_sign_min_abs_y_m:
                    left_wrong_side_points.append((x, y))
                    wrong_side_count += 1
                    continue
                left_points.append((x, y))
                assigned_count += 1
            elif color in self.right_boundary_colors:
                if self.enforce_expected_side_sign and y > self.side_sign_min_abs_y_m:
                    right_wrong_side_points.append((x, y))
                    wrong_side_count += 1
                    continue
                right_points.append((x, y))
                assigned_count += 1
            elif self.split_unmapped_by_y_sign:
                if y >= 0.0:
                    left_points.append((x, y))
                else:
                    right_points.append((x, y))
                assigned_count += 1
                fallback_split_count += 1
            else:
                unmapped_count += 1

        if fallback_split_count > 0:
            self._warn_throttled(
                'fallback_color_split',
                f'color fallback split active for {fallback_split_count} cones '
                '(unmapped colors; using y sign)',
            )
        if wrong_side_count > 0:
            self._warn_throttled(
                'wrong_side_reject',
                f'rejected {wrong_side_count} cones that disagreed with expected boundary side sign',
            )
        used_wrong_side_count = 0
        if self.enforce_expected_side_sign:
            if not left_points and left_wrong_side_points:
                left_points.extend(left_wrong_side_points)
                used_wrong_side_count += len(left_wrong_side_points)
            if not right_points and right_wrong_side_points:
                right_points.extend(right_wrong_side_points)
                used_wrong_side_count += len(right_wrong_side_points)
        if used_wrong_side_count > 0:
            self._warn_throttled(
                'wrong_side_fallback',
                f'using {used_wrong_side_count} cones despite side-sign mismatch because '
                'that boundary had no surviving cones',
            )

        final_assigned_count = len(left_points) + len(right_points)
        if gated_count > 0 and final_assigned_count == 0:
            self._warn_throttled(
                'color_mapping_empty',
                'no cones remained after gating/color split; '
                f'left={sorted(self.left_boundary_colors)} right={sorted(self.right_boundary_colors)} '
                f'unmapped={unmapped_count}',
            )

        left_array = np.array(left_points, dtype=np.float64) if left_points else np.empty((0, 2))
        right_array = np.array(right_points, dtype=np.float64) if right_points else np.empty((0, 2))
        return left_array, right_array, effective_base_frame

    def _fit_boundary(self, points: np.ndarray) -> tuple[Optional[np.ndarray], np.ndarray]:
        if points.shape[0] <= 0:
            return None, np.empty((0, 2), dtype=np.float64)

        ordered = points[np.argsort(points[:, 0])]
        selected = ordered[: self.max_points_per_side]
        if selected.shape[0] == 1 and self.allow_single_point_fallback:
            return self._single_point_coeffs(float(selected[0, 1]), degree=self.fit_degree), selected
        if selected.shape[0] < self.min_points_per_side:
            return None, selected

        if self.use_ransac:
            coeffs = self._fit_poly_ransac(selected, degree=self.fit_degree)
            if coeffs is not None:
                return coeffs, selected

        return self._fit_poly_ls(selected, degree=self.fit_degree), selected

    @staticmethod
    def _single_point_coeffs(y_value: float, degree: int) -> np.ndarray:
        if degree <= 1:
            return np.array([0.0, y_value], dtype=np.float64)
        return np.array([0.0, 0.0, y_value], dtype=np.float64)

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
        horizon_m: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if left_coeffs is None and right_coeffs is None:
            return None, None

        max_seen_x = 0.0
        if left_points.shape[0] > 0:
            max_seen_x = max(max_seen_x, float(np.max(left_points[:, 0])))
        if right_points.shape[0] > 0:
            max_seen_x = max(max_seen_x, float(np.max(right_points[:, 0])))

        horizon_x = horizon_m if max_seen_x <= 0.0 else min(horizon_m, max_seen_x)
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
    def _limit_points_by_x(points: np.ndarray, x_max_m: float) -> np.ndarray:
        if points.shape[0] == 0:
            return points
        return points[points[:, 0] <= x_max_m]

    @staticmethod
    def _truncate_centerline(centerline: np.ndarray, horizon_m: float) -> np.ndarray:
        if centerline.shape[0] == 0:
            return centerline
        if horizon_m <= 0.0:
            return centerline
        mask = centerline[:, 0] <= horizon_m
        truncated = centerline[mask]
        if truncated.shape[0] == 0:
            return centerline[:1].copy()
        if truncated.shape[0] == centerline.shape[0]:
            return truncated
        next_idx = truncated.shape[0]
        if next_idx < centerline.shape[0]:
            prev_point = truncated[-1]
            next_point = centerline[next_idx]
            if next_point[0] > prev_point[0]:
                ratio = (horizon_m - prev_point[0]) / max(next_point[0] - prev_point[0], 1e-6)
                ratio = float(np.clip(ratio, 0.0, 1.0))
                interp_y = prev_point[1] + ratio * (next_point[1] - prev_point[1])
                trunc_point = np.array([[horizon_m, interp_y]], dtype=np.float64)
                truncated = np.vstack((truncated, trunc_point))
        return truncated

    def _validate_plan(self, plan: PlanResult, now_sec: float) -> bool:
        if self.reject_crossing_paths:
            crossing_reason = self._plan_crossing_reason(plan)
            if crossing_reason is not None:
                self._warn_throttled('reject_crossing_plan', crossing_reason)
                return False

        if self.max_centerline_jump_m > 0.0 and self._plan_has_large_jump(plan.centerline, now_sec):
            self._warn_throttled(
                'reject_centerline_jump',
                'rejecting plan because centerline contradicts recent valid history',
            )
            return False

        return True

    def _plan_crossing_reason(self, plan: PlanResult) -> Optional[str]:
        eps = self.min_boundary_separation_m

        if plan.left_fit.shape[0] > 0 and plan.right_fit.shape[0] > 0:
            left_y = plan.left_fit[:, 1]
            right_y = plan.right_fit[:, 1]
            if np.any((left_y - right_y) <= eps):
                return (
                    'rejecting plan because fitted boundaries cross or collapse '
                    f'(min separation <= {eps:.2f} m)'
                )

        if plan.left_fit.shape[0] > 0 and plan.centerline.shape[0] > 0:
            center_y = np.interp(plan.left_fit[:, 0], plan.centerline[:, 0], plan.centerline[:, 1])
            if np.any((plan.left_fit[:, 1] - center_y) <= eps):
                return (
                    'rejecting plan because centerline crosses left boundary '
                    f'(margin <= {eps:.2f} m)'
                )

        if plan.right_fit.shape[0] > 0 and plan.centerline.shape[0] > 0:
            center_y = np.interp(plan.right_fit[:, 0], plan.centerline[:, 0], plan.centerline[:, 1])
            if np.any((center_y - plan.right_fit[:, 1]) <= eps):
                return (
                    'rejecting plan because centerline crosses right boundary '
                    f'(margin <= {eps:.2f} m)'
                )

        return None

    def _plan_has_large_jump(self, centerline: np.ndarray, now_sec: float) -> bool:
        if centerline.shape[0] < 2:
            return False

        self._prune_centerline_history(now_sec)
        if not self._recent_valid_centerlines:
            return False

        horizon_limit = min(self.consistency_horizon_m, float(centerline[-1, 0]))
        if horizon_limit <= 0.25:
            return False

        sample_mask = centerline[:, 0] <= horizon_limit
        if not np.any(sample_mask):
            return False
        sample_x = centerline[sample_mask, 0]
        current_y = centerline[sample_mask, 1]

        reference_rows: list[np.ndarray] = []
        for _, previous_centerline in self._recent_valid_centerlines:
            if previous_centerline.shape[0] < 2:
                continue
            previous_horizon = float(previous_centerline[-1, 0])
            overlap_limit = min(horizon_limit, previous_horizon)
            if overlap_limit <= 0.25:
                continue
            overlap_mask = sample_x <= overlap_limit
            if not np.any(overlap_mask):
                continue
            interp_y = np.interp(
                sample_x[overlap_mask],
                previous_centerline[:, 0],
                previous_centerline[:, 1],
            )
            row = np.full(sample_x.shape, np.nan, dtype=np.float64)
            row[overlap_mask] = interp_y
            reference_rows.append(row)

        if not reference_rows:
            return False

        stacked = np.vstack(reference_rows)
        reference_y = np.nanmedian(stacked, axis=0)
        valid_mask = np.isfinite(reference_y)
        if not np.any(valid_mask):
            return False

        deviation = np.abs(current_y[valid_mask] - reference_y[valid_mask])
        return bool(np.max(deviation) > self.max_centerline_jump_m)

    def _record_valid_centerline(self, now_sec: float, centerline: np.ndarray) -> None:
        if centerline.shape[0] == 0:
            return
        self._recent_valid_centerlines.append((now_sec, np.array(centerline, copy=True)))
        self._prune_centerline_history(now_sec)
        while len(self._recent_valid_centerlines) > self.max_history_frames:
            self._recent_valid_centerlines.popleft()

    def _prune_centerline_history(self, now_sec: float) -> None:
        while len(self._recent_valid_centerlines) > self.max_history_frames:
            self._recent_valid_centerlines.popleft()

    @staticmethod
    def _evaluate_fit(coeffs: Optional[np.ndarray], sample_x: Optional[np.ndarray]) -> np.ndarray:
        if coeffs is None or sample_x is None:
            return np.empty((0, 2), dtype=np.float64)
        y_values = np.polyval(coeffs, sample_x)
        return np.column_stack((sample_x, y_values))

    def _compute_control(self, centerline: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
        speed = max(0.0, float(self._latest_speed_mps))
        lookahead = max(self.lookahead_min_m, self.lookahead_min_m + self.lookahead_gain * speed)

        if centerline.shape[0] <= 1:
            idx = 0
        else:
            segment_lengths = np.hypot(
                np.diff(centerline[:, 0]),
                np.diff(centerline[:, 1]),
            )
            arc_lengths = np.concatenate((
                np.array([0.0], dtype=np.float64),
                np.cumsum(segment_lengths, dtype=np.float64),
            ))
            idx = int(np.searchsorted(arc_lengths, lookahead, side='left'))
            idx = min(max(idx, 0), centerline.shape[0] - 1)
        target = centerline[idx]

        target_distance = max(float(np.hypot(target[0], target[1])), 1e-3)
        denom = max(target_distance * target_distance, 1e-6)
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
            scale=0.18,
            alpha=0.45,
        )
        marker_id = self._append_sphere_list(
            markers,
            marker_id=marker_id,
            now=now,
            ns='raw_right',
            points_xy=plan.right_raw,
            rgb=(1.0, 0.9, 0.1),
            scale=0.18,
            alpha=0.45,
        )
        marker_id = self._append_sphere_list(
            markers,
            marker_id=marker_id,
            now=now,
            ns='fit_input_left',
            points_xy=plan.left_fit_input,
            rgb=(0.1, 0.4, 1.0),
            scale=0.28,
            alpha=0.95,
        )
        marker_id = self._append_sphere_list(
            markers,
            marker_id=marker_id,
            now=now,
            ns='fit_input_right',
            points_xy=plan.right_fit_input,
            rgb=(1.0, 0.9, 0.1),
            scale=0.28,
            alpha=0.95,
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
        status_text = f'v={speed_cmd:.2f} m/s  Ld={lookahead:.2f} m  delta={steering:.3f} rad  kappa={kappa:.3f}'
        if self.show_cone_counts_in_status:
            left_fit_state = 'ok' if plan.left_fit_ok else 'no'
            right_fit_state = 'ok' if plan.right_fit_ok else 'no'
            status_text += (
                f'\nL raw/used/fit={plan.left_raw_count}/{plan.left_used_count}/{left_fit_state}'
                f'  R raw/used/fit={plan.right_raw_count}/{plan.right_used_count}/{right_fit_state}'
            )

        self._append_text_marker(
            markers,
            marker_id=marker_id,
            now=now,
            text=status_text,
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
        alpha: float = 0.95,
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
        marker.color.a = float(alpha)
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
        timeout = Duration(seconds=self.tf_timeout_s)
        try_stamps = [
            Time(seconds=int(sec), nanoseconds=int(nanosec)),
            Time(),  # fallback: latest available transform
        ]
        for stamp in try_stamps:
            try:
                return self._tf_buffer.lookup_transform(target_frame, source_frame, stamp, timeout=timeout)
            except TransformException:
                continue
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

    def _read_axis_parameter(self, name: str, default: str) -> str:
        value = str(self.get_parameter(name).value).strip().lower()
        if value not in {'x', 'y', 'z'}:
            self.get_logger().warn(f'{name}="{value}" invalid; using "{default}"')
            return default
        return value

    def _read_axis_sign_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        return -1.0 if value < 0.0 else 1.0

    @staticmethod
    def _axis_value(px: float, py: float, pz: float, axis: str) -> float:
        if axis == 'x':
            return px
        if axis == 'y':
            return py
        return pz

    @staticmethod
    def _normalize_color(token: str) -> str:
        value = str(token).strip().lower().replace('-', '_').replace(' ', '_')
        if (
            'big_orange' in value
            or 'large_orange' in value
            or (('big' in value or 'large' in value) and 'orange' in value)
        ):
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

    @classmethod
    def _target_frame_candidates(cls, source_frame: str, requested_base: str) -> list[str]:
        requested = str(requested_base).strip().strip('/')
        if not requested:
            return []
        candidates = [requested]
        namespaced = cls._resolve_namespaced_base_frame(
            source_frame=source_frame,
            requested_base=requested,
        )
        if namespaced and namespaced not in candidates:
            candidates.append(namespaced)
        return candidates

    @classmethod
    def _source_frame_candidates(cls, source_frame: str, requested_base: str) -> list[str]:
        source = str(source_frame).strip().strip('/')
        if not source:
            return []

        candidates: list[str] = [source]
        if '/' in source:
            parts = [p for p in source.split('/') if p]
            if parts:
                leaf = parts[-1]
                if leaf not in candidates:
                    candidates.append(leaf)
                namespace_leaf = f'{parts[0]}/{leaf}'
                if namespace_leaf not in candidates:
                    candidates.append(namespace_leaf)

            requested = str(requested_base).strip().strip('/')
            marker = f'/{requested}/' if requested else ''
            source_with_slashes = f'/{source}/'
            if marker and marker in source_with_slashes:
                idx = source_with_slashes.find(marker)
                prefix = source_with_slashes[1:idx].strip('/')
                suffix_start = idx + len(marker)
                suffix = source_with_slashes[suffix_start:-1].strip('/')
                if prefix and suffix:
                    prefixed_suffix = f'{prefix}/{suffix}'
                    if prefixed_suffix not in candidates:
                        candidates.append(prefixed_suffix)

        expanded = list(candidates)
        for token in candidates:
            if token.endswith('_camera'):
                link_token = token[:-7] + '_link'
                if link_token not in expanded:
                    expanded.append(link_token)
        return expanded

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
