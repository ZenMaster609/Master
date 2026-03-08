#!/usr/bin/env python3
"""Cone-pair midpoint planner with Pure Pursuit tracking."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
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


@dataclass
class PlanResult:
    preview_path: np.ndarray
    control_path: np.ndarray
    control_valid: bool
    mode: str
    left_min_clearance_m: float
    right_min_clearance_m: float
    left_raw: np.ndarray
    right_raw: np.ndarray
    left_filtered: np.ndarray
    right_filtered: np.ndarray
    paired_left: np.ndarray
    paired_right: np.ndarray
    midpoints: np.ndarray
    left_raw_count: int
    right_raw_count: int
    left_filtered_count: int
    right_filtered_count: int
    pair_count: int

    def wall_avoidance_active(self, min_clearance_m: float, enabled: bool) -> bool:
        if not enabled:
            return False
        return (
            math.isfinite(self.left_min_clearance_m) and self.left_min_clearance_m < min_clearance_m
        ) or (
            math.isfinite(self.right_min_clearance_m) and self.right_min_clearance_m < min_clearance_m
        )


class PairMidpointPlannerNode(Node):
    """Builds a centerline from local cone pairs and tracks it with Pure Pursuit."""

    def __init__(self) -> None:
        super().__init__('pair_midpoint_planner_node')

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
        self._default_corridor_half_width_m: float = 1.5
        self._last_corridor_half_width_m: float = self._default_corridor_half_width_m
        self._one_sided_steering_bias_rad: float = 0.0

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

        self.create_timer(1.0 / self.loop_rate_hz, self._on_timer)

        self.get_logger().info(
            'pair_midpoint_planner_node ready '
            f'cones={self.cones_topic} odom={self.odom_topic} cmd={self.cmd_topic} '
            f'base_frame={self.base_frame} pair_by={self.pair_by} '
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
            'topics.path_topic': 'sim/planner/pair_midpoint_path',
            'topics.markers_topic': 'sim/planner/pair_midpoint_markers',
            'gating.min_confidence': 0.35,
            'gating.x_min_m': 0.5,
            'gating.x_max_m': 12.0,
            'gating.y_abs_max_m': 8.0,
            'gating.max_cone_age_s': 20.0,
            'gating.hold_last_valid_s': 2.0,
            'gating.startup_delay_s': 10.0,
            'colors.left_boundary': ['blue'],
            'colors.right_boundary': ['yellow'],
            'colors.split_unmapped_by_y_sign': True,
            'pairing.max_pair_dx_m': 2.5,
            'pairing.max_pair_dist_m': 4.5,
            'pairing.min_pairs_required': 2,
            'pairing.allow_unbalanced_sides': True,
            'pairing.pair_by': 'greedy_x',
            'pairing.reuse_unpaired_for_tail': False,
            'path_generation.preview_horizon_m': 9.0,
            'path_generation.control_horizon_m': 5.5,
            'path_generation.sample_count': 21,
            'path_generation.smoothing_window': 3,
            'path_generation.enforce_monotonic_x': True,
            'pure_pursuit.lookahead_min_m': 3.0,
            'pure_pursuit.lookahead_gain': 0.35,
            'pure_pursuit.steering_limit_rad': 0.52,
            'pure_pursuit.wheelbase_m': 1.65,
            'pure_pursuit.stop_if_no_path': False,
            'wall_avoidance.enabled': True,
            'wall_avoidance.min_clearance_m': 0.20,
            'wall_avoidance.steering_gain_rad_per_m': 0.90,
            'wall_avoidance.lookahead_margin_m': 1.0,
            'one_sided_bias.enabled': True,
            'one_sided_bias.increment_rad_per_frame': 0.003,
            'one_sided_bias.max_bias_rad': 0.12,
            'speed_control.speed_min_mps': 1.0,
            'speed_control.speed_max_mps': 1.8,
            'speed_control.curvature_speed_gain': 4.0,
            'speed_control.lowpass_speed_alpha': 0.15,
            'debug.publish_path': True,
            'debug.publish_markers': True,
            'debug.show_counts_in_status': True,
            'debug.log_throttle_s': 1.0,
            'frames.base_frame': 'base_footprint',
            'frames.tf_timeout_s': 0.02,
            'cone_axes.forward_axis': 'x',
            'cone_axes.lateral_axis': 'y',
            'cone_axes.forward_sign': 1.0,
            'cone_axes.lateral_sign': 1.0,
            'planner_rate_hz': 180.0,
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
        self.split_unmapped_by_y_sign = bool(self.get_parameter('colors.split_unmapped_by_y_sign').value)

        self.max_pair_dx_m = max(0.0, float(self.get_parameter('pairing.max_pair_dx_m').value))
        self.max_pair_dist_m = max(0.0, float(self.get_parameter('pairing.max_pair_dist_m').value))
        self.min_pairs_required = max(1, int(self.get_parameter('pairing.min_pairs_required').value))
        self.allow_unbalanced_sides = bool(self.get_parameter('pairing.allow_unbalanced_sides').value)
        self.pair_by = str(self.get_parameter('pairing.pair_by').value).strip().lower()
        if self.pair_by not in {'greedy_x', 'nearest'}:
            self.get_logger().warn(f'pairing.pair_by="{self.pair_by}" invalid; using "greedy_x"')
            self.pair_by = 'greedy_x'
        self.reuse_unpaired_for_tail = bool(self.get_parameter('pairing.reuse_unpaired_for_tail').value)
        if self.reuse_unpaired_for_tail:
            self.get_logger().warn('pairing.reuse_unpaired_for_tail is declared but inactive in v1')

        self.preview_horizon_m = max(0.5, float(self.get_parameter('path_generation.preview_horizon_m').value))
        self.control_horizon_m = max(0.5, float(self.get_parameter('path_generation.control_horizon_m').value))
        self.sample_count = max(2, int(self.get_parameter('path_generation.sample_count').value))
        self.smoothing_window = max(1, int(self.get_parameter('path_generation.smoothing_window').value))
        self.enforce_monotonic_x = bool(self.get_parameter('path_generation.enforce_monotonic_x').value)
        if self.control_horizon_m > self.preview_horizon_m:
            self.get_logger().warn(
                f'path_generation.control_horizon_m={self.control_horizon_m:.2f} exceeds '
                f'preview_horizon_m={self.preview_horizon_m:.2f}; clamping control horizon'
            )
            self.control_horizon_m = self.preview_horizon_m

        self.lookahead_min_m = max(0.5, float(self.get_parameter('pure_pursuit.lookahead_min_m').value))
        self.lookahead_gain = max(0.0, float(self.get_parameter('pure_pursuit.lookahead_gain').value))
        self.steering_limit_rad = max(0.01, float(self.get_parameter('pure_pursuit.steering_limit_rad').value))
        self.wheelbase_m = max(0.1, float(self.get_parameter('pure_pursuit.wheelbase_m').value))
        self.stop_if_no_path = bool(self.get_parameter('pure_pursuit.stop_if_no_path').value)
        self.wall_avoidance_enabled = bool(self.get_parameter('wall_avoidance.enabled').value)
        self.wall_min_clearance_m = max(0.0, float(self.get_parameter('wall_avoidance.min_clearance_m').value))
        self.wall_steering_gain_rad_per_m = max(
            0.0,
            float(self.get_parameter('wall_avoidance.steering_gain_rad_per_m').value),
        )
        self.wall_lookahead_margin_m = max(
            0.0,
            float(self.get_parameter('wall_avoidance.lookahead_margin_m').value),
        )
        self.one_sided_bias_enabled = bool(self.get_parameter('one_sided_bias.enabled').value)
        self.one_sided_bias_increment_rad = max(
            0.0,
            float(self.get_parameter('one_sided_bias.increment_rad_per_frame').value),
        )
        self.one_sided_bias_max_rad = max(
            0.0,
            float(self.get_parameter('one_sided_bias.max_bias_rad').value),
        )

        self.speed_min_mps = max(0.0, float(self.get_parameter('speed_control.speed_min_mps').value))
        self.speed_max_mps = max(self.speed_min_mps, float(self.get_parameter('speed_control.speed_max_mps').value))
        self.curvature_speed_gain = max(0.0, float(self.get_parameter('speed_control.curvature_speed_gain').value))
        self.lowpass_speed_alpha = float(self.get_parameter('speed_control.lowpass_speed_alpha').value)
        self.lowpass_speed_alpha = float(np.clip(self.lowpass_speed_alpha, 0.0, 1.0))

        self.publish_path = bool(self.get_parameter('debug.publish_path').value)
        self.publish_markers = bool(self.get_parameter('debug.publish_markers').value)
        self.show_counts_in_status = bool(self.get_parameter('debug.show_counts_in_status').value)
        self.log_throttle_s = max(0.1, float(self.get_parameter('debug.log_throttle_s').value))

        self.base_frame = str(self.get_parameter('frames.base_frame').value).strip() or 'base_footprint'
        self.tf_timeout_s = max(0.0, float(self.get_parameter('frames.tf_timeout_s').value))

        self.cone_forward_axis = self._read_axis_parameter('cone_axes.forward_axis', default='x')
        self.cone_lateral_axis = self._read_axis_parameter('cone_axes.lateral_axis', default='y')
        if self.cone_lateral_axis == self.cone_forward_axis:
            self.get_logger().warn(
                f'cone_axes.lateral_axis matches forward_axis "{self.cone_forward_axis}"; using "y" instead'
            )
            self.cone_lateral_axis = 'y' if self.cone_forward_axis != 'y' else 'x'
        self.cone_forward_sign = self._read_axis_sign_parameter('cone_axes.forward_sign')
        self.cone_lateral_sign = self._read_axis_sign_parameter('cone_axes.lateral_sign')

        self.loop_rate_hz = max(1.0, float(self.get_parameter('planner_rate_hz').value))

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

        debug_plan: Optional[PlanResult] = None
        control_plan: Optional[PlanResult] = None
        cones_msg = self._latest_cones_msg
        if cones_msg is not None:
            cone_stamp_sec = self._stamp_to_sec(cones_msg.header.stamp.sec, cones_msg.header.stamp.nanosec)
            age = now_sec - cone_stamp_sec
            if age <= self.max_cone_age_s:
                debug_plan = self._build_plan(cones_msg)
            else:
                self._warn_throttled(
                    'stale_cones',
                    f'cones stale by {age:.3f}s (limit {self.max_cone_age_s:.3f}s)',
                )

        if debug_plan is not None and debug_plan.control_valid:
            self._last_valid_plan = debug_plan
            self._last_valid_time_sec = now_sec
            control_plan = debug_plan
        elif (
            self._last_valid_plan is not None
            and self._last_valid_time_sec > 0.0
            and (now_sec - self._last_valid_time_sec) <= self.hold_last_valid_s
        ):
            control_plan = self._last_valid_plan

        if debug_plan is None:
            debug_plan = control_plan

        if debug_plan is None:
            self._publish_no_path(now)
            return

        steering = 0.0
        speed_cmd = float(self._last_speed_cmd) if self._last_speed_cmd is not None else 0.0
        kappa = 0.0
        lookahead = 0.0
        target_point = debug_plan.preview_path[0] if debug_plan.preview_path.shape[0] > 0 else np.zeros(2, dtype=np.float64)
        if control_plan is not None:
            control_plan = self._with_wall_clearance(control_plan)
            if debug_plan is control_plan:
                debug_plan = control_plan
            self._update_one_sided_bias(debug_plan)
            steering, speed_cmd, kappa, lookahead, target_point = self._compute_control(control_plan.control_path)
            steering = self._apply_wall_clearance_bias(steering, control_plan)
            steering = self._apply_one_sided_bias(steering, control_plan)
            kappa = math.tan(steering) / max(self.wheelbase_m, 1e-6)
            v_des = self.speed_max_mps / (1.0 + self.curvature_speed_gain * abs(kappa))
            v_des = float(np.clip(v_des, self.speed_min_mps, self.speed_max_mps))
            if self._last_speed_cmd is None:
                speed_cmd = v_des
            else:
                alpha = self.lowpass_speed_alpha
                speed_cmd = alpha * v_des + (1.0 - alpha) * float(self._last_speed_cmd)
            self._last_speed_cmd = speed_cmd
            self._last_steering_cmd = steering
            self._publish_cmd(now, speed_cmd, steering)
        elif self.stop_if_no_path:
            self._update_one_sided_bias(debug_plan)
            self._publish_cmd(now, speed_mps=0.0, steering_rad=0.0)
            self._last_speed_cmd = 0.0
            self._last_steering_cmd = 0.0
        elif self._last_speed_cmd is not None and self._last_steering_cmd is not None:
            self._update_one_sided_bias(debug_plan)
            self._publish_cmd(now, speed_mps=float(self._last_speed_cmd), steering_rad=float(self._last_steering_cmd))

        if self.publish_path:
            self._publish_preview_path(now, debug_plan.preview_path)
        if self.publish_markers:
            self._publish_debug_markers(
                now=now,
                plan=debug_plan,
                target_point=target_point,
                speed_cmd=speed_cmd,
                lookahead=lookahead,
                steering=steering,
                kappa=kappa,
            )

    def _build_plan(self, cones_msg: ConeDetectionArray) -> Optional[PlanResult]:
        extracted = self._extract_boundary_points(cones_msg)
        if extracted is None:
            return None
        left_raw, right_raw, left_filtered, right_filtered, plan_frame = extracted
        self._active_base_frame = plan_frame

        paired_left, paired_right, midpoints = self._pair_boundaries(left_filtered, right_filtered)
        if midpoints.shape[0] == 0:
            self._warn_throttled(
                'too_few_pairs',
                f'only {midpoints.shape[0]} cone pairs available; need at least {self.min_pairs_required}',
            )
            return PlanResult(
                preview_path=np.empty((0, 2), dtype=np.float64),
                control_path=np.empty((0, 2), dtype=np.float64),
                control_valid=False,
                mode='no_path',
                left_min_clearance_m=float('inf'),
                right_min_clearance_m=float('inf'),
                left_raw=left_raw,
                right_raw=right_raw,
                left_filtered=left_filtered,
                right_filtered=right_filtered,
                paired_left=paired_left,
                paired_right=paired_right,
                midpoints=np.empty((0, 2), dtype=np.float64),
                left_raw_count=int(left_raw.shape[0]),
                right_raw_count=int(right_raw.shape[0]),
                left_filtered_count=int(left_filtered.shape[0]),
                right_filtered_count=int(right_filtered.shape[0]),
                pair_count=0,
            )

        if midpoints.shape[0] < self.min_pairs_required:
            fallback_plan = self._build_single_boundary_fallback_plan(
                left_raw=left_raw,
                right_raw=right_raw,
                left_filtered=left_filtered,
                right_filtered=right_filtered,
            )
            if fallback_plan is not None:
                self._warn_throttled(
                    'single_boundary_fallback',
                    f'paired path has only {midpoints.shape[0]} pair(s); using {fallback_plan.mode} fallback',
                )
                return fallback_plan

        ordered_midpoints = self._prepare_midpoints(midpoints)
        preview_path = self._truncate_path_by_arc_length(ordered_midpoints, self.preview_horizon_m)
        control_path = self._truncate_path_by_arc_length(preview_path, self.control_horizon_m)
        if control_path.shape[0] == 0:
            control_path = preview_path[:1].copy()
        control_valid = preview_path.shape[0] >= self.min_pairs_required
        if not control_valid:
            self._warn_throttled(
                'too_few_pairs',
                f'only {preview_path.shape[0]} cone pairs available; need at least {self.min_pairs_required}',
            )
            fallback_plan = self._build_single_boundary_fallback_plan(
                left_raw=left_raw,
                right_raw=right_raw,
                left_filtered=left_filtered,
                right_filtered=right_filtered,
            )
            if fallback_plan is not None:
                self._warn_throttled(
                    'single_boundary_fallback',
                    f'preview path has only {preview_path.shape[0]} point(s); using {fallback_plan.mode} fallback',
                )
                return fallback_plan

        self._update_corridor_width_from_pairs(paired_left, paired_right)

        return PlanResult(
            preview_path=preview_path,
            control_path=control_path,
            control_valid=control_valid,
            mode='paired',
            left_min_clearance_m=float('inf'),
            right_min_clearance_m=float('inf'),
            left_raw=left_raw,
            right_raw=right_raw,
            left_filtered=left_filtered,
            right_filtered=right_filtered,
            paired_left=paired_left,
            paired_right=paired_right,
            midpoints=ordered_midpoints,
            left_raw_count=int(left_raw.shape[0]),
            right_raw_count=int(right_raw.shape[0]),
            left_filtered_count=int(left_filtered.shape[0]),
            right_filtered_count=int(right_filtered.shape[0]),
            pair_count=int(min(paired_left.shape[0], paired_right.shape[0])),
        )

    def _build_single_boundary_fallback_plan(
        self,
        left_raw: np.ndarray,
        right_raw: np.ndarray,
        left_filtered: np.ndarray,
        right_filtered: np.ndarray,
    ) -> Optional[PlanResult]:
        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        if left_filtered.shape[0] >= 2:
            candidates.append(('left_only', left_filtered, right_filtered))
        if right_filtered.shape[0] >= 2:
            candidates.append(('right_only', right_filtered, left_filtered))
        if not candidates:
            return None

        best_plan: Optional[PlanResult] = None
        best_score: tuple[float, float] = (-1.0, -1.0)
        for mode, boundary_points, opposite_points in candidates:
            preview_path = self._build_single_boundary_centerline(boundary_points, mode=mode)
            if preview_path is None or preview_path.shape[0] < 2:
                continue
            control_path = self._truncate_path_by_arc_length(preview_path, self.control_horizon_m)
            if control_path.shape[0] == 0:
                control_path = preview_path[:1].copy()
            score = (
                float(boundary_points.shape[0]),
                self._path_arc_length(preview_path),
            )
            if score <= best_score:
                continue
            if mode == 'left_only':
                paired_left = boundary_points.copy()
                paired_right = np.empty((0, 2), dtype=np.float64)
            else:
                paired_left = np.empty((0, 2), dtype=np.float64)
                paired_right = boundary_points.copy()
            best_plan = PlanResult(
                preview_path=preview_path,
                control_path=control_path,
                control_valid=True,
                mode=mode,
                left_min_clearance_m=float('inf'),
                right_min_clearance_m=float('inf'),
                left_raw=left_raw,
                right_raw=right_raw,
                left_filtered=left_filtered,
                right_filtered=right_filtered,
                paired_left=paired_left,
                paired_right=paired_right,
                midpoints=preview_path.copy(),
                left_raw_count=int(left_raw.shape[0]),
                right_raw_count=int(right_raw.shape[0]),
                left_filtered_count=int(left_filtered.shape[0]),
                right_filtered_count=int(right_filtered.shape[0]),
                pair_count=0,
            )
            best_score = score
        return best_plan

    def _build_single_boundary_centerline(self, boundary_points: np.ndarray, mode: str) -> Optional[np.ndarray]:
        ordered_boundary = self._prepare_midpoints(boundary_points)
        if ordered_boundary.shape[0] < 2:
            return None

        tangent = np.zeros_like(ordered_boundary)
        tangent[0] = ordered_boundary[min(1, ordered_boundary.shape[0] - 1)] - ordered_boundary[0]
        tangent[-1] = ordered_boundary[-1] - ordered_boundary[max(0, ordered_boundary.shape[0] - 2)]
        if ordered_boundary.shape[0] > 2:
            tangent[1:-1] = ordered_boundary[2:] - ordered_boundary[:-2]

        tangent_norm = np.hypot(tangent[:, 0], tangent[:, 1])
        tangent_norm = np.maximum(tangent_norm, 1e-6)
        unit_tangent = tangent / tangent_norm[:, None]
        left_normal = np.column_stack((-unit_tangent[:, 1], unit_tangent[:, 0]))

        inward_sign = -1.0 if mode == 'left_only' else 1.0
        half_width_m = self._estimate_corridor_half_width()
        centerline = ordered_boundary + (inward_sign * half_width_m) * left_normal
        centerline = self._prepare_midpoints(centerline)
        centerline = self._blend_with_previous_centerline(centerline)
        centerline = self._extend_path_along_tangent(centerline, self.preview_horizon_m)
        preview_path = self._truncate_path_by_arc_length(centerline, self.preview_horizon_m)
        return preview_path if preview_path.shape[0] >= 2 else None

    def _estimate_corridor_half_width(self) -> float:
        return float(np.clip(self._last_corridor_half_width_m, 0.8, 2.5))

    def _update_corridor_width_from_pairs(self, paired_left: np.ndarray, paired_right: np.ndarray) -> None:
        count = min(paired_left.shape[0], paired_right.shape[0])
        if count <= 0:
            return
        width = np.median(np.hypot(
            paired_left[:count, 0] - paired_right[:count, 0],
            paired_left[:count, 1] - paired_right[:count, 1],
        ))
        if not math.isfinite(float(width)) or width <= 0.2:
            return
        half_width = 0.5 * float(width)
        self._last_corridor_half_width_m = float(np.clip(half_width, 0.8, 2.5))

    def _blend_with_previous_centerline(self, path_xy: np.ndarray) -> np.ndarray:
        if path_xy.shape[0] < 2 or self._last_valid_plan is None:
            return path_xy
        previous = self._last_valid_plan.preview_path
        if previous.shape[0] < 2:
            return path_xy
        x_min = float(previous[0, 0])
        x_max = float(previous[-1, 0])
        mask = (path_xy[:, 0] >= x_min) & (path_xy[:, 0] <= x_max)
        if not np.any(mask):
            return path_xy
        blended = np.array(path_xy, copy=True)
        prev_y = np.interp(blended[mask, 0], previous[:, 0], previous[:, 1])
        overlap_x = blended[mask, 0]
        if overlap_x.shape[0] == 1:
            previous_weight = np.array([0.75], dtype=np.float64)
        else:
            transition_span = max(
                min(self.preview_horizon_m, self.control_horizon_m + 2.0),
                1.5,
            )
            ramp = np.clip(overlap_x / transition_span, 0.0, 1.0)
            previous_weight = 0.80 - 0.45 * ramp
        blended[mask, 1] = previous_weight * prev_y + (1.0 - previous_weight) * blended[mask, 1]
        return blended

    def _extend_path_along_tangent(self, path_xy: np.ndarray, horizon_m: float) -> np.ndarray:
        if path_xy.shape[0] < 2:
            return path_xy
        current_length = self._path_arc_length(path_xy)
        if current_length >= horizon_m:
            return path_xy
        last_segment = path_xy[-1] - path_xy[-2]
        seg_norm = float(np.hypot(last_segment[0], last_segment[1]))
        if seg_norm <= 1e-6:
            return path_xy
        unit_tangent = last_segment / seg_norm
        extension = horizon_m - current_length
        extra_point = path_xy[-1] + extension * unit_tangent
        return np.vstack((path_xy, extra_point.reshape(1, 2)))

    @staticmethod
    def _path_arc_length(path_xy: np.ndarray) -> float:
        if path_xy.shape[0] < 2:
            return 0.0
        return float(np.sum(np.hypot(np.diff(path_xy[:, 0]), np.diff(path_xy[:, 1]))))

    def _with_wall_clearance(self, plan: PlanResult) -> PlanResult:
        left_clearance, right_clearance = self._estimate_wall_clearances(
            path_xy=plan.control_path,
            left_points=plan.left_filtered,
            right_points=plan.right_filtered,
        )
        plan.left_min_clearance_m = left_clearance
        plan.right_min_clearance_m = right_clearance
        return plan

    def _estimate_wall_clearances(
        self,
        path_xy: np.ndarray,
        left_points: np.ndarray,
        right_points: np.ndarray,
    ) -> tuple[float, float]:
        if path_xy.shape[0] == 0:
            return float('inf'), float('inf')

        horizon_x = float(path_xy[-1, 0]) + self.wall_lookahead_margin_m
        left_clearance = self._estimate_side_clearance(path_xy, left_points, True, horizon_x)
        right_clearance = self._estimate_side_clearance(path_xy, right_points, False, horizon_x)
        return left_clearance, right_clearance

    @staticmethod
    def _estimate_side_clearance(
        path_xy: np.ndarray,
        wall_points: np.ndarray,
        is_left: bool,
        horizon_x: float,
    ) -> float:
        if path_xy.shape[0] == 0 or wall_points.shape[0] == 0:
            return float('inf')
        mask = (wall_points[:, 0] >= 0.0) & (wall_points[:, 0] <= horizon_x)
        relevant = wall_points[mask]
        if relevant.shape[0] == 0:
            return float('inf')

        eval_x = np.clip(relevant[:, 0], float(path_xy[0, 0]), float(path_xy[-1, 0]))
        path_y = np.interp(eval_x, path_xy[:, 0], path_xy[:, 1])
        if is_left:
            clearance = relevant[:, 1] - path_y
        else:
            clearance = path_y - relevant[:, 1]
        return float(np.min(clearance)) if clearance.size > 0 else float('inf')

    def _apply_wall_clearance_bias(self, steering: float, plan: PlanResult) -> float:
        if not self.wall_avoidance_enabled:
            return steering

        correction = 0.0
        if math.isfinite(plan.left_min_clearance_m) and plan.left_min_clearance_m < self.wall_min_clearance_m:
            correction -= self.wall_steering_gain_rad_per_m * (self.wall_min_clearance_m - plan.left_min_clearance_m)
        if math.isfinite(plan.right_min_clearance_m) and plan.right_min_clearance_m < self.wall_min_clearance_m:
            correction += self.wall_steering_gain_rad_per_m * (self.wall_min_clearance_m - plan.right_min_clearance_m)
        return float(np.clip(steering + correction, -self.steering_limit_rad, self.steering_limit_rad))

    def _update_one_sided_bias(self, plan: Optional[PlanResult]) -> None:
        if plan is None or not self.one_sided_bias_enabled:
            self._one_sided_steering_bias_rad = 0.0
            return
        if plan.pair_count >= 1:
            self._one_sided_steering_bias_rad = 0.0
            return
        if plan.mode == 'left_only':
            self._one_sided_steering_bias_rad = float(np.clip(
                self._one_sided_steering_bias_rad - self.one_sided_bias_increment_rad,
                -self.one_sided_bias_max_rad,
                self.one_sided_bias_max_rad,
            ))
        elif plan.mode == 'right_only':
            self._one_sided_steering_bias_rad = float(np.clip(
                self._one_sided_steering_bias_rad + self.one_sided_bias_increment_rad,
                -self.one_sided_bias_max_rad,
                self.one_sided_bias_max_rad,
            ))
        else:
            self._one_sided_steering_bias_rad = 0.0

    def _apply_one_sided_bias(self, steering: float, plan: PlanResult) -> float:
        if not self.one_sided_bias_enabled:
            return steering
        if plan.mode not in {'left_only', 'right_only'}:
            return steering
        return float(np.clip(
            steering + self._one_sided_steering_bias_rad,
            -self.steering_limit_rad,
            self.steering_limit_rad,
        ))

    def _extract_boundary_points(
        self,
        cones_msg: ConeDetectionArray,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
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
                return None

            if effective_base_frame != self.base_frame or resolved_source != frame_id:
                self._warn_throttled(
                    'namespaced_base_frame',
                    f'using transform {resolved_source}->{effective_base_frame} '
                    f'instead of {frame_id}->{self.base_frame}',
                )

        left_raw: list[tuple[float, float]] = []
        right_raw: list[tuple[float, float]] = []
        left_filtered: list[tuple[float, float]] = []
        right_filtered: list[tuple[float, float]] = []

        for cone in cones_msg.cones:
            px = float(cone.position.x)
            py = float(cone.position.y)
            pz = float(cone.position.z)
            if transform is not None:
                px, py, pz = self._transform_point(transform, px, py, pz)

            x = self.cone_forward_sign * self._axis_value(px, py, pz, self.cone_forward_axis)
            y = self.cone_lateral_sign * self._axis_value(px, py, pz, self.cone_lateral_axis)

            side = self._classify_boundary_side(self._normalize_color(cone.color), y)
            if side is None:
                continue

            if side == 'left':
                left_raw.append((x, y))
            else:
                right_raw.append((x, y))

            if float(cone.confidence) < self.min_confidence:
                continue
            if x < self.x_min_m or x > self.x_max_m:
                continue
            if abs(y) > self.y_abs_max_m:
                continue

            if side == 'left':
                left_filtered.append((x, y))
            else:
                right_filtered.append((x, y))

        left_raw_arr = np.array(left_raw, dtype=np.float64) if left_raw else np.empty((0, 2), dtype=np.float64)
        right_raw_arr = np.array(right_raw, dtype=np.float64) if right_raw else np.empty((0, 2), dtype=np.float64)
        left_filtered_arr = (
            np.array(left_filtered, dtype=np.float64) if left_filtered else np.empty((0, 2), dtype=np.float64)
        )
        right_filtered_arr = (
            np.array(right_filtered, dtype=np.float64) if right_filtered else np.empty((0, 2), dtype=np.float64)
        )
        return left_raw_arr, right_raw_arr, left_filtered_arr, right_filtered_arr, effective_base_frame

    def _classify_boundary_side(self, color: str, y_value: float) -> Optional[str]:
        if color in self.left_boundary_colors:
            return 'left'
        if color in self.right_boundary_colors:
            return 'right'
        if self.split_unmapped_by_y_sign:
            return 'left' if y_value >= 0.0 else 'right'
        return None

    def _pair_boundaries(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if left_points.shape[0] == 0 or right_points.shape[0] == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
            )

        left_sorted = left_points[np.argsort(left_points[:, 0])]
        right_sorted = right_points[np.argsort(right_points[:, 0])]

        if self.pair_by == 'nearest':
            primary = self._pair_boundaries_nearest(left_sorted, right_sorted, enforce_dx=True)
            if primary[2].shape[0] >= self.min_pairs_required or not self.allow_unbalanced_sides:
                return primary
            relaxed = self._pair_boundaries_nearest(left_sorted, right_sorted, enforce_dx=False)
        else:
            primary = self._pair_boundaries_greedy_x(left_sorted, right_sorted, enforce_dx=True)
            if primary[2].shape[0] >= self.min_pairs_required or not self.allow_unbalanced_sides:
                return primary
            relaxed = self._pair_boundaries_greedy_x(left_sorted, right_sorted, enforce_dx=False)

        if relaxed[2].shape[0] > primary[2].shape[0]:
            self._warn_throttled(
                'relaxed_pairing',
                'strict forward-offset pairing found too few matches; using relaxed local pairing fallback',
            )
            return relaxed
        return primary

    def _pair_boundaries_greedy_x(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray,
        enforce_dx: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_count = left_points.shape[0]
        right_count = right_points.shape[0]
        pair_cost = np.full((left_count, right_count), np.inf, dtype=np.float64)

        for left_idx, left_point in enumerate(left_points):
            deltas = right_points - left_point
            dist = np.hypot(deltas[:, 0], deltas[:, 1])
            dx = np.abs(deltas[:, 0])
            if enforce_dx:
                valid = (dx <= self.max_pair_dx_m) & (dist <= self.max_pair_dist_m)
            else:
                valid = dist <= self.max_pair_dist_m
            if np.any(valid):
                midpoint_x = 0.5 * (left_point[0] + right_points[valid, 0])
                pair_cost[left_idx, valid] = dx[valid] + 0.25 * dist[valid] + 1e-3 * midpoint_x

        # Monotone matching DP:
        # maximize number of pairs first, then minimize total pairing cost.
        best_pairs = np.zeros((left_count + 1, right_count + 1), dtype=np.int32)
        best_cost = np.full((left_count + 1, right_count + 1), np.inf, dtype=np.float64)
        best_cost[0, :] = 0.0
        best_cost[:, 0] = 0.0
        decision = np.zeros((left_count + 1, right_count + 1), dtype=np.int8)

        for left_idx in range(1, left_count + 1):
            for right_idx in range(1, right_count + 1):
                skip_left_pairs = best_pairs[left_idx - 1, right_idx]
                skip_left_cost = best_cost[left_idx - 1, right_idx]
                skip_right_pairs = best_pairs[left_idx, right_idx - 1]
                skip_right_cost = best_cost[left_idx, right_idx - 1]

                if (
                    skip_left_pairs > skip_right_pairs
                    or (
                        skip_left_pairs == skip_right_pairs
                        and skip_left_cost <= skip_right_cost
                    )
                ):
                    best_pairs[left_idx, right_idx] = skip_left_pairs
                    best_cost[left_idx, right_idx] = skip_left_cost
                    decision[left_idx, right_idx] = 1
                else:
                    best_pairs[left_idx, right_idx] = skip_right_pairs
                    best_cost[left_idx, right_idx] = skip_right_cost
                    decision[left_idx, right_idx] = 2

                cost = pair_cost[left_idx - 1, right_idx - 1]
                if math.isfinite(float(cost)):
                    pair_pairs = best_pairs[left_idx - 1, right_idx - 1] + 1
                    pair_total_cost = best_cost[left_idx - 1, right_idx - 1] + float(cost)
                    if (
                        pair_pairs > best_pairs[left_idx, right_idx]
                        or (
                            pair_pairs == best_pairs[left_idx, right_idx]
                            and pair_total_cost < best_cost[left_idx, right_idx]
                        )
                    ):
                        best_pairs[left_idx, right_idx] = pair_pairs
                        best_cost[left_idx, right_idx] = pair_total_cost
                        decision[left_idx, right_idx] = 3

        paired_left: list[np.ndarray] = []
        paired_right: list[np.ndarray] = []
        midpoints: list[np.ndarray] = []
        left_idx = left_count
        right_idx = right_count
        while left_idx > 0 and right_idx > 0:
            step = int(decision[left_idx, right_idx])
            if step == 3:
                left_point = left_points[left_idx - 1]
                right_point = right_points[right_idx - 1]
                midpoint = 0.5 * (left_point + right_point)
                paired_left.append(left_point)
                paired_right.append(right_point)
                midpoints.append(midpoint)
                left_idx -= 1
                right_idx -= 1
            elif step == 1:
                left_idx -= 1
            else:
                right_idx -= 1

        paired_left.reverse()
        paired_right.reverse()
        midpoints.reverse()
        return self._paired_arrays(paired_left, paired_right, midpoints)

    def _pair_boundaries_nearest(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray,
        enforce_dx: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        used_right: set[int] = set()
        pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for left_point in left_points:
            deltas = right_points - left_point
            dist = np.hypot(deltas[:, 0], deltas[:, 1])
            dx = np.abs(deltas[:, 0])
            if enforce_dx:
                candidates = np.where((dx <= self.max_pair_dx_m) & (dist <= self.max_pair_dist_m))[0]
            else:
                candidates = np.where(dist <= self.max_pair_dist_m)[0]
            best_idx: Optional[int] = None
            best_dist = float('inf')
            best_dx = float('inf')
            for right_idx in candidates:
                if int(right_idx) in used_right:
                    continue
                cand_dist = float(dist[right_idx])
                cand_dx = float(dx[right_idx])
                if cand_dist < best_dist or (math.isclose(cand_dist, best_dist) and cand_dx < best_dx):
                    best_idx = int(right_idx)
                    best_dist = cand_dist
                    best_dx = cand_dx
            if best_idx is None:
                continue
            used_right.add(best_idx)
            right_point = right_points[best_idx]
            midpoint = 0.5 * (left_point + right_point)
            pairs.append((left_point, right_point, midpoint))

        pairs.sort(key=lambda pair: float(pair[2][0]))
        paired_left = [pair[0] for pair in pairs]
        paired_right = [pair[1] for pair in pairs]
        midpoints = [pair[2] for pair in pairs]
        return self._paired_arrays(paired_left, paired_right, midpoints)

    @staticmethod
    def _paired_arrays(
        paired_left: list[np.ndarray],
        paired_right: list[np.ndarray],
        midpoints: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_arr = np.array(paired_left, dtype=np.float64) if paired_left else np.empty((0, 2), dtype=np.float64)
        right_arr = np.array(paired_right, dtype=np.float64) if paired_right else np.empty((0, 2), dtype=np.float64)
        midpoint_arr = np.array(midpoints, dtype=np.float64) if midpoints else np.empty((0, 2), dtype=np.float64)
        if midpoint_arr.shape[0] > 0:
            order = np.argsort(midpoint_arr[:, 0])
            left_arr = left_arr[order]
            right_arr = right_arr[order]
            midpoint_arr = midpoint_arr[order]
        return left_arr, right_arr, midpoint_arr

    def _prepare_midpoints(self, midpoints: np.ndarray) -> np.ndarray:
        if midpoints.shape[0] == 0:
            return midpoints
        ordered = midpoints[np.argsort(midpoints[:, 0])]
        if self.enforce_monotonic_x:
            kept: list[np.ndarray] = []
            last_x: Optional[float] = None
            for point in ordered:
                x_value = float(point[0])
                if last_x is None or x_value > (last_x + 1e-3):
                    kept.append(point)
                    last_x = x_value
            ordered = np.array(kept, dtype=np.float64) if kept else np.empty((0, 2), dtype=np.float64)
        if ordered.shape[0] == 0 or self.smoothing_window < 2:
            return ordered

        y_values = ordered[:, 1]
        half_window = self.smoothing_window // 2
        smoothed_y = np.empty_like(y_values)
        for idx in range(y_values.shape[0]):
            start = max(0, idx - half_window)
            end = min(y_values.shape[0], idx + half_window + 1)
            smoothed_y[idx] = float(np.mean(y_values[start:end]))
        return np.column_stack((ordered[:, 0], smoothed_y))

    @staticmethod
    def _truncate_path_by_arc_length(path_xy: np.ndarray, horizon_m: float) -> np.ndarray:
        if path_xy.shape[0] == 0:
            return path_xy
        if path_xy.shape[0] == 1 or horizon_m <= 0.0:
            return path_xy.copy()

        segment_lengths = np.hypot(np.diff(path_xy[:, 0]), np.diff(path_xy[:, 1]))
        arc_lengths = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths)))
        keep_mask = arc_lengths <= horizon_m
        truncated = path_xy[keep_mask]

        if truncated.shape[0] == 0:
            return path_xy[:1].copy()
        if truncated.shape[0] == path_xy.shape[0]:
            return truncated

        next_idx = truncated.shape[0]
        prev_idx = max(0, next_idx - 1)
        prev_point = path_xy[prev_idx]
        next_point = path_xy[next_idx]
        prev_arc = arc_lengths[prev_idx]
        next_arc = arc_lengths[next_idx]
        if next_arc <= prev_arc:
            return truncated

        ratio = float(np.clip((horizon_m - prev_arc) / (next_arc - prev_arc), 0.0, 1.0))
        interp_point = prev_point + ratio * (next_point - prev_point)
        return np.vstack((truncated, interp_point.reshape(1, 2)))

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

    def _publish_no_path(self, now) -> None:
        if self.stop_if_no_path:
            self._publish_cmd(now, speed_mps=0.0, steering_rad=0.0)
            self._last_speed_cmd = 0.0
            self._last_steering_cmd = 0.0
        elif self._last_speed_cmd is not None and self._last_steering_cmd is not None:
            self._publish_cmd(now, speed_mps=float(self._last_speed_cmd), steering_rad=float(self._last_steering_cmd))

        if self.publish_path:
            self._publish_preview_path(now, np.empty((0, 2), dtype=np.float64))
        if self.publish_markers:
            self._publish_clear_markers(now)

    def _publish_cmd(self, now, speed_mps: float, steering_rad: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = now.to_msg()
        msg.drive.speed = float(speed_mps)
        msg.drive.steering_angle = float(steering_rad)
        self._cmd_pub.publish(msg)

    def _publish_preview_path(self, now, preview_path: np.ndarray) -> None:
        path_msg = Path()
        path_msg.header.stamp = now.to_msg()
        path_msg.header.frame_id = self._active_base_frame
        for x_value, y_value in preview_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x_value)
            pose.pose.position.y = float(y_value)
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
            markers, marker_id, now, 'raw_left', plan.left_raw, (0.1, 0.4, 1.0), 0.16, 0.35
        )
        marker_id = self._append_sphere_list(
            markers, marker_id, now, 'raw_right', plan.right_raw, (1.0, 0.9, 0.1), 0.16, 0.35
        )
        marker_id = self._append_sphere_list(
            markers, marker_id, now, 'filtered_left', plan.left_filtered, (0.1, 0.4, 1.0), 0.24, 0.95
        )
        marker_id = self._append_sphere_list(
            markers, marker_id, now, 'filtered_right', plan.right_filtered, (1.0, 0.9, 0.1), 0.24, 0.95
        )
        marker_id = self._append_sphere_list(
            markers, marker_id, now, 'paired_left', plan.paired_left, (0.0, 0.2, 0.9), 0.30, 1.0
        )
        marker_id = self._append_sphere_list(
            markers, marker_id, now, 'paired_right', plan.paired_right, (0.9, 0.7, 0.0), 0.30, 1.0
        )
        marker_id = self._append_pair_links(
            markers, marker_id, now, 'pair_links', plan.paired_left, plan.paired_right, (0.8, 0.8, 0.8), 0.06
        )
        marker_id = self._append_sphere_list(
            markers, marker_id, now, 'midpoints', plan.midpoints, (0.95, 0.2, 0.2), 0.20, 0.95
        )
        marker_id = self._append_line_strip(
            markers, marker_id, now, 'midpoint_path', plan.preview_path, (0.9, 0.1, 0.1), 0.10
        )
        marker_id = self._append_target_marker(markers, marker_id, now, target_point)

        wall_state = 'off'
        if self.wall_avoidance_enabled:
            wall_state = 'active' if plan.wall_avoidance_active(self.wall_min_clearance_m, True) else 'idle'

        status_text = (
            f'v={speed_cmd:.2f} m/s  Ld={lookahead:.2f} m  '
            f'delta={steering:.2f} rad  kappa={kappa:.2f}  mode={plan.mode}  wall={wall_state}'
        )
        if self.show_counts_in_status:
            status_text += (
                f'\nL={plan.left_filtered_count}  R={plan.right_filtered_count}  pairs={plan.pair_count}'
                f'  clrL={plan.left_min_clearance_m:.2f}  clrR={plan.right_min_clearance_m:.2f}'
                f'  bias={self._one_sided_steering_bias_rad:.3f}'
            )
        marker_id = self._append_text_marker(markers, marker_id, now, status_text)
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
        alpha: float,
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
        for x_value, y_value in points_xy:
            point = Point()
            point.x = float(x_value)
            point.y = float(y_value)
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
        for x_value, y_value in points_xy:
            point = Point()
            point.x = float(x_value)
            point.y = float(y_value)
            point.z = 0.0
            marker.points.append(point)
        markers.markers.append(marker)
        return marker_id + 1

    def _append_pair_links(
        self,
        markers: MarkerArray,
        marker_id: int,
        now,
        ns: str,
        left_points_xy: np.ndarray,
        right_points_xy: np.ndarray,
        rgb: tuple[float, float, float],
        width: float,
    ) -> int:
        count = min(left_points_xy.shape[0], right_points_xy.shape[0])
        marker = Marker()
        marker.header.stamp = now.to_msg()
        marker.header.frame_id = self._active_base_frame
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.a = 0.90
        marker.color.r = float(rgb[0])
        marker.color.g = float(rgb[1])
        marker.color.b = float(rgb[2])
        for idx in range(count):
            left_point = Point()
            left_point.x = float(left_points_xy[idx, 0])
            left_point.y = float(left_points_xy[idx, 1])
            left_point.z = 0.0
            marker.points.append(left_point)

            right_point = Point()
            right_point.x = float(right_points_xy[idx, 0])
            right_point.y = float(right_points_xy[idx, 1])
            right_point.z = 0.0
            marker.points.append(right_point)
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
            Time(),
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
        namespaced = cls._resolve_namespaced_base_frame(source_frame=source_frame, requested_base=requested)
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
            parts = [part for part in source.split('/') if part]
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
        return float(sec) + 1e-9 * float(nanosec)

    @staticmethod
    def _clock_to_sec(nanoseconds: int) -> float:
        return 1e-9 * float(nanoseconds)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = PairMidpointPlannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
