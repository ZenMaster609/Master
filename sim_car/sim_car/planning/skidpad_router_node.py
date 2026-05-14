#!/usr/bin/env python3
"""Skidpad-specific cone router for deterministic branch selection."""

from __future__ import annotations

from copy import deepcopy
import math
import os
import signal
import threading
from typing import Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Float32
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.skidpad_router_core import (
    AccelerationStopRowDetection,
    SkidpadRouterSnapshot,
    SkidpadStateMachine,
    boundary_color_from_lateral_y,
    config_from_parameters,
    detect_acceleration_stop_row,
    detect_stop_line_pair,
)


class SkidpadRouterNode(Node):
    """Republish cones for the currently active skidpad branch."""

    def __init__(self) -> None:
        super().__init__("skidpad_router_node")
        self._declare_parameters()
        self._read_parameters()

        self._latest_pose_xy: Optional[tuple[float, float]] = None
        self._latest_yaw_rad: float = 0.0
        self._latest_speed_mps: float = 0.0
        self._latest_odom_stamp = None
        self._latest_snapshot = self._make_initial_snapshot()
        self._last_warn_sec = -1.0
        self._parking_mode_active = False
        self._orange_only_cone_count = 0
        self._parking_boundary_override_count = 0
        self._acceleration_parking_latched = False
        self._stop_override_active = False
        self._route_complete_shutdown_requested = False
        self._parking_complete_shutdown_requested = False
        self._acceleration_parked_since_sec: Optional[float] = None
        self._stop_line_forward_distance_m: float = float("nan")
        self._stop_line_marker_points_odom: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._spawn_pose_xy: Optional[tuple[float, float]] = None
        self._approach_distance_from_spawn_m: float = 0.0

        self._cones_pub = self.create_publisher(ConeDetectionArray, self.output_topic, 10)
        self._steering_lock_pub = self.create_publisher(Bool, '/steering_lock', 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, self.diagnostics_topic, 10)
        self._viz_pub = self.create_publisher(MarkerArray, self.viz_topic, 10)
        self._cmd_pub = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
        self._brake_pub = self.create_publisher(Float32, self.brake_cmd_topic, 10)
        self._stop_override_timer = None
        _timer_period = (
            self.stop_override_publish_period_s
            if self.stop_override_publish_period_s > 0.0
            else (1.0 / 120.0)
        )
        if self.stop_override_publish_period_s > 0.0 or self._approach_lock_active:
            self._stop_override_timer = self.create_timer(_timer_period, self._stop_override_timer_cb)

        self.create_subscription(
            ConeDetectionArray,
            self.input_topic,
            self._cones_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_cb,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "skidpad_router_node ready "
            f"in={self.input_topic} out={self.output_topic} odom={self.odom_topic}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "frames.odom_frame": "odom",
            "frames.base_frame_aliases": ["front_axle", "base_link", "base_footprint"],
            "topics.input_topic": "/tracked_cones",
            "topics.output_topic": "/tracked_cones/skidpad_routed",
            "topics.odom_topic": "/sim/odom",
            "topics.cmd_topic": "/cmd",
            "topics.brake_cmd_topic": "/sim/brake_cmd",
            "topics.diagnostics_topic": "/skidpad_router/diagnostics",
            "topics.viz_topic": "/skidpad_router/markers",
            "routing.event_mode": "skidpad",
            "geometry.crossroads_center_xy": [0.0, 0.0],
            "geometry.crossroads_half_extents_xy": [3.0, 2.5],
            "geometry.left_circle_center_xy": [-9.25, 0.0],
            "geometry.right_circle_center_xy": [9.25, 0.0],
            "geometry.circle_radius_window_m": [6.5, 11.5],
            "routing.route_sequence": ["right", "right", "left", "left", "straight"],
            "routing.route_laps": 1,
            "routing.lap_complete_angle_rad": 5.5,
            "routing.lobe_radius_m": 12.5,
            "routing.right_lobe_min_x_m": -1.0,
            "routing.left_lobe_max_x_m": 1.0,
            "routing.straight_corridor_half_width_m": 2.0,
            "routing.straight_min_y_m": -1.0,
            "routing.shutdown_on_route_complete": False,
            "routing.shutdown_on_parking_complete": False,
            "parking.corridor_half_width_m": 2.0,
            "parking.start_y_m": 8.5,
            "parking.parked_speed_threshold_mps": 0.5,
            "parking.parked_hold_time_s": 1.0,
            "parking.test_park_only": False,
            "parking.acceleration_activation_distance_m": 10.0,
            "parking.acceleration_stop_row_cluster_depth_m": 0.75,
            "parking.acceleration_stop_row_min_cluster_count": 4,
            "parking.acceleration_stop_row_min_lateral_span_m": 2.0,
            "parking.acceleration_stop_row_min_points_per_side": 1,
            "parking.stop_margin_m": 1.0,
            "parking.target_margin_m": 3.5,
            "parking.stop_line_pair_max_distance_m": 1.0,
            "parking.stop_override_publish_rate_hz": 120.0,
            "parking.stop_approach_speed_gain": 1.5,
            "routing.approach_straight_lock_distance_m": 12.0,
            "parking.brake_activation_margin_m": 1.0,
            "parking.brake_command": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        self.odom_frame = str(self.get_parameter("frames.odom_frame").value).strip() or "odom"
        self.base_frame_aliases = {
            str(value).strip()
            for value in self.get_parameter("frames.base_frame_aliases").value
            if str(value).strip()
        }
        self.input_topic = str(self.get_parameter("topics.input_topic").value).strip() or "/tracked_cones"
        self.output_topic = str(self.get_parameter("topics.output_topic").value).strip() or "/tracked_cones/skidpad_routed"
        self.odom_topic = str(self.get_parameter("topics.odom_topic").value).strip() or "/sim/odom"
        self.cmd_topic = str(self.get_parameter("topics.cmd_topic").value).strip() or "/cmd"
        self.brake_cmd_topic = str(self.get_parameter("topics.brake_cmd_topic").value).strip() or "/sim/brake_cmd"
        self.diagnostics_topic = (
            str(self.get_parameter("topics.diagnostics_topic").value).strip() or "/skidpad_router/diagnostics"
        )
        self.viz_topic = str(self.get_parameter("topics.viz_topic").value).strip() or "/skidpad_router/markers"
        self.event_mode = str(self.get_parameter("routing.event_mode").value).strip().lower() or "skidpad"
        self.shutdown_on_route_complete = bool(self.get_parameter("routing.shutdown_on_route_complete").value)
        self.shutdown_on_parking_complete = bool(self.get_parameter("routing.shutdown_on_parking_complete").value)
        self.acceleration_activation_distance_m = float(
            self.get_parameter("parking.acceleration_activation_distance_m").value
        )
        self.acceleration_stop_row_cluster_depth_m = float(
            self.get_parameter("parking.acceleration_stop_row_cluster_depth_m").value
        )
        self.acceleration_stop_row_min_cluster_count = int(
            self.get_parameter("parking.acceleration_stop_row_min_cluster_count").value
        )
        self.acceleration_stop_row_min_lateral_span_m = float(
            self.get_parameter("parking.acceleration_stop_row_min_lateral_span_m").value
        )
        self.acceleration_stop_row_min_points_per_side = int(
            self.get_parameter("parking.acceleration_stop_row_min_points_per_side").value
        )
        self.stop_margin_m = float(self.get_parameter("parking.stop_margin_m").value)
        self.target_margin_m = float(self.get_parameter("parking.target_margin_m").value)
        self.stop_line_pair_max_distance_m = float(self.get_parameter("parking.stop_line_pair_max_distance_m").value)
        stop_override_rate_hz = float(self.get_parameter("parking.stop_override_publish_rate_hz").value)
        self.stop_override_publish_period_s = 0.0 if stop_override_rate_hz <= 0.0 else (1.0 / stop_override_rate_hz)
        self.stop_approach_speed_gain = float(self.get_parameter("parking.stop_approach_speed_gain").value)
        self.brake_activation_margin_m = float(self.get_parameter("parking.brake_activation_margin_m").value)
        self.brake_command = float(np.clip(float(self.get_parameter("parking.brake_command").value), 0.0, 1.0))
        self.approach_straight_lock_distance_m = float(
            self.get_parameter("routing.approach_straight_lock_distance_m").value
        )
        self._approach_lock_active: bool = self.approach_straight_lock_distance_m > 0.0

        self._state_machine = SkidpadStateMachine(
            config_from_parameters(
                crossroads_center_xy=self.get_parameter("geometry.crossroads_center_xy").value,
                crossroads_half_extents_xy=self.get_parameter("geometry.crossroads_half_extents_xy").value,
                left_circle_center_xy=self.get_parameter("geometry.left_circle_center_xy").value,
                right_circle_center_xy=self.get_parameter("geometry.right_circle_center_xy").value,
                circle_radius_window_m=self.get_parameter("geometry.circle_radius_window_m").value,
                route_sequence=self.get_parameter("routing.route_sequence").value,
                route_laps=int(self.get_parameter("routing.route_laps").value),
                lap_complete_angle_rad=float(self.get_parameter("routing.lap_complete_angle_rad").value),
                parking_corridor_half_width_m=float(self.get_parameter("parking.corridor_half_width_m").value),
                parking_start_y_m=float(self.get_parameter("parking.start_y_m").value),
                straight_corridor_half_width_m=float(
                    self.get_parameter("routing.straight_corridor_half_width_m").value
                ),
                straight_min_y_m=float(self.get_parameter("routing.straight_min_y_m").value),
                parked_speed_threshold_mps=float(self.get_parameter("parking.parked_speed_threshold_mps").value),
                parked_hold_time_s=float(self.get_parameter("parking.parked_hold_time_s").value),
                test_park_only=bool(self.get_parameter("parking.test_park_only").value),
                lobe_radius_m=float(self.get_parameter("routing.lobe_radius_m").value),
                right_lobe_min_x_m=float(self.get_parameter("routing.right_lobe_min_x_m").value),
                left_lobe_max_x_m=float(self.get_parameter("routing.left_lobe_max_x_m").value),
            )
        )

    def _make_initial_snapshot(self) -> SkidpadRouterSnapshot:
        return SkidpadRouterSnapshot(
            stage_name=self._state_machine.stage_name(),
            active_branch=self._state_machine.active_branch(),
            completed_laps=self._state_machine.completed_laps,
            route_index=self._state_machine.route_index,
            current_route_pass=self._state_machine.current_route_pass(),
            total_route_passes=self._state_machine.total_route_passes(),
            completed_route_passes=self._state_machine.completed_route_passes(),
            in_crossroads=False,
            lap_angle_accum_rad=0.0,
            lap_armed=False,
            parked=self._state_machine.parked,
            just_completed_lap=False,
            just_entered_crossroads=False,
        )

    def _odom_cb(self, msg: Odometry) -> None:
        x_m = float(msg.pose.pose.position.x)
        y_m = float(msg.pose.pose.position.y)
        yaw_rad = _yaw_from_quaternion(
            float(msg.pose.pose.orientation.x),
            float(msg.pose.pose.orientation.y),
            float(msg.pose.pose.orientation.z),
            float(msg.pose.pose.orientation.w),
        )
        speed_mps = math.hypot(float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y))
        stamp_sec = float(msg.header.stamp.sec) + (float(msg.header.stamp.nanosec) * 1e-9)

        self._latest_pose_xy = (x_m, y_m)
        self._latest_yaw_rad = yaw_rad
        self._latest_speed_mps = speed_mps
        self._latest_odom_stamp = msg.header.stamp

        if self._spawn_pose_xy is None:
            self._spawn_pose_xy = (x_m, y_m)
        if self._approach_lock_active:
            self._approach_distance_from_spawn_m = math.hypot(
                x_m - self._spawn_pose_xy[0], y_m - self._spawn_pose_xy[1]
            )
            if self._approach_distance_from_spawn_m >= self.approach_straight_lock_distance_m:
                self._approach_lock_active = False
                self._publish_steering_lock(False)
                self.get_logger().info(
                    f"Approach straight lock released at {self._approach_distance_from_spawn_m:.1f} m from spawn."
                )

        self._latest_snapshot = self._state_machine.update(
            x_m=x_m,
            y_m=y_m,
            speed_mps=speed_mps,
            now_sec=stamp_sec,
        )
        self._update_latched_stop_line_distance_from_pose()
        if self._stop_override_active:
            self._publish_parking_override_cmd(msg.header.stamp)
        self._publish_diagnostics(msg.header.stamp)
        self._publish_markers(msg.header.stamp)
        self._maybe_shutdown_after_route_complete()
        self._maybe_shutdown_after_parking_complete(stamp=msg.header.stamp)

    def _cones_cb(self, msg: ConeDetectionArray) -> None:
        if self._latest_pose_xy is None:
            self._throttled_warn("waiting for odom before routing skidpad cones")
            return

        cone_points_odom, converted_cones = self._convert_cones_to_odom(msg)
        if cone_points_odom is None:
            return

        routed_msg = ConeDetectionArray()
        routed_msg.header = deepcopy(msg.header)
        routed_msg.header.frame_id = self.odom_frame
        self._update_acceleration_parking_latch(converted_cones=converted_cones, cone_points_odom=cone_points_odom)
        self._parking_mode_active = self._parking_mode_is_active()
        self._orange_only_cone_count = 0
        self._parking_boundary_override_count = 0
        if not self._parking_mode_active:
            self._clear_stop_line_state()
        else:
            self._update_latched_stop_line_distance_from_pose()

        if self._parking_mode_active:
            routed_msg.cones = self._route_parking_cones(
                converted_cones=converted_cones,
                cone_points_odom=cone_points_odom,
            )
        else:
            routed_msg.cones = self._route_non_parking_cones(
                converted_cones=converted_cones,
                cone_points_odom=cone_points_odom,
            )

        self._update_stop_override_from_routed_cones(routed_cones=routed_msg.cones)
        if self._stop_override_active:
            self._publish_parking_override_cmd(msg.header.stamp)
        self._cones_pub.publish(routed_msg)
        stamp = self._latest_odom_stamp if self._latest_odom_stamp is not None else msg.header.stamp
        self._publish_diagnostics(stamp)
        self._publish_markers(stamp)

    def _parking_mode_is_active(self) -> bool:
        if self.event_mode == "acceleration":
            return bool(self._acceleration_parking_latched)
        return self._state_machine.active_branch() == "straight"

    def _update_acceleration_parking_latch(
        self,
        *,
        converted_cones: list[ConeDetection],
        cone_points_odom: np.ndarray,
    ) -> None:
        if self.event_mode != "acceleration" or self._acceleration_parking_latched:
            return

        if self._latest_pose_xy is None or cone_points_odom.size == 0:
            return

        trigger_distance_m = max(0.0, float(self.acceleration_activation_distance_m))
        for cone, point_xy in zip(converted_cones, cone_points_odom):
            if normalize_color(getattr(cone, "color", "")) != "orange":
                continue
            dx = float(point_xy[0]) - float(self._latest_pose_xy[0])
            dy = float(point_xy[1]) - float(self._latest_pose_xy[1])
            if math.hypot(dx, dy) <= trigger_distance_m:
                self._acceleration_parking_latched = True
                return

    def _route_non_parking_cones(
        self,
        *,
        converted_cones: list[ConeDetection],
        cone_points_odom: np.ndarray,
    ) -> list[ConeDetection]:
        if not converted_cones or cone_points_odom.size == 0:
            return []

        if self.event_mode == "acceleration":
            return list(converted_cones)

        mask = self._state_machine.route_mask(cone_points_odom)
        return [cone for cone, keep in zip(converted_cones, mask) if keep]

    def _route_parking_cones(
        self,
        *,
        converted_cones: list[ConeDetection],
        cone_points_odom: np.ndarray,
    ) -> list[ConeDetection]:
        if not converted_cones or cone_points_odom.size == 0:
            return []

        if self.event_mode == "acceleration":
            mask = np.ones((cone_points_odom.shape[0],), dtype=bool)
        else:
            mask = self._state_machine.route_mask(cone_points_odom)
        filtered_cones: list[ConeDetection] = []
        filtered_points: list[np.ndarray] = []
        for cone, point_xy, keep in zip(converted_cones, cone_points_odom, mask):
            if not keep:
                continue
            if normalize_color(getattr(cone, "color", "")) != "orange":
                continue
            filtered_cones.append(cone)
            filtered_points.append(np.asarray(point_xy, dtype=np.float64))

        orange_points_odom = (
            np.vstack(filtered_points) if filtered_points else np.empty((0, 2), dtype=np.float64)
        )
        stop_line_mask = (
            self._detect_acceleration_stop_row_mask(orange_points_odom)
            if self.event_mode == "acceleration"
            else self._detect_stop_line_cluster_mask(orange_points_odom)
        )
        if self._stop_line_marker_points_odom is not None and orange_points_odom.size > 0:
            stop_line_mask = stop_line_mask | self._mask_points_on_latched_stop_line(orange_points_odom)
        routed: list[ConeDetection] = []
        orange_count = len(filtered_cones)
        override_count = 0
        for idx, (cone, point_xy) in enumerate(zip(filtered_cones, orange_points_odom)):
            if idx < stop_line_mask.shape[0] and bool(stop_line_mask[idx]):
                continue
            routed_cone = deepcopy(cone)
            boundary_color = self._boundary_color_for_odom_point(float(point_xy[0]), float(point_xy[1]))
            existing_boundary = str(getattr(routed_cone, "boundary_color", "")).strip().lower()
            if existing_boundary and existing_boundary != boundary_color:
                override_count += 1
            routed_cone.boundary_color = boundary_color
            routed.append(routed_cone)

        self._orange_only_cone_count = orange_count
        self._parking_boundary_override_count = override_count
        return routed

    def _convert_cones_to_odom(
        self,
        msg: ConeDetectionArray,
    ) -> tuple[Optional[np.ndarray], list[ConeDetection]]:
        if self._latest_pose_xy is None:
            return None, []

        source_frame = str(msg.header.frame_id).strip() or self.odom_frame
        odom_points: list[tuple[float, float]] = []
        routed_cones: list[ConeDetection] = []
        for cone in msg.cones:
            x_odom, y_odom = self._point_to_odom(
                x_m=float(cone.position.x),
                y_m=float(cone.position.y),
                source_frame=source_frame,
            )
            routed_cone = deepcopy(cone)
            routed_cone.position.x = float(x_odom)
            routed_cone.position.y = float(y_odom)
            routed_cone.position.z = float(cone.position.z)
            odom_points.append((x_odom, y_odom))
            routed_cones.append(routed_cone)

        return np.asarray(odom_points, dtype=np.float64), routed_cones

    def _point_to_odom(self, *, x_m: float, y_m: float, source_frame: str) -> tuple[float, float]:
        normalized = str(source_frame).strip()
        if not normalized or normalized == self.odom_frame:
            return float(x_m), float(y_m)
        if normalized in self.base_frame_aliases:
            odom_x, odom_y = self._latest_pose_xy if self._latest_pose_xy is not None else (0.0, 0.0)
            cos_yaw = math.cos(self._latest_yaw_rad)
            sin_yaw = math.sin(self._latest_yaw_rad)
            x_odom = odom_x + (cos_yaw * float(x_m)) - (sin_yaw * float(y_m))
            y_odom = odom_y + (sin_yaw * float(x_m)) + (cos_yaw * float(y_m))
            return x_odom, y_odom

        self._throttled_warn(f"unknown cone frame '{normalized}', assuming odom geometry")
        return float(x_m), float(y_m)

    def _vehicle_point_to_odom(self, *, x_forward_m: float, y_lateral_m: float) -> tuple[float, float]:
        odom_x, odom_y = self._latest_pose_xy if self._latest_pose_xy is not None else (0.0, 0.0)
        cos_yaw = math.cos(self._latest_yaw_rad)
        sin_yaw = math.sin(self._latest_yaw_rad)
        x_odom = odom_x + (cos_yaw * float(x_forward_m)) - (sin_yaw * float(y_lateral_m))
        y_odom = odom_y + (sin_yaw * float(x_forward_m)) + (cos_yaw * float(y_lateral_m))
        return float(x_odom), float(y_odom)

    def _vehicle_frame_lateral_y_for_odom_point(self, x_m: float, y_m: float) -> float:
        odom_x, odom_y = self._latest_pose_xy if self._latest_pose_xy is not None else (0.0, 0.0)
        dx = float(x_m) - float(odom_x)
        dy = float(y_m) - float(odom_y)
        return (-math.sin(self._latest_yaw_rad) * dx) + (math.cos(self._latest_yaw_rad) * dy)

    def _boundary_color_for_odom_point(self, x_m: float, y_m: float) -> str:
        lateral_y = self._vehicle_frame_lateral_y_for_odom_point(x_m, y_m)
        return boundary_color_from_lateral_y(lateral_y)

    def _odom_points_to_vehicle_frame(self, points_xy: np.ndarray) -> np.ndarray:
        if points_xy.size == 0 or self._latest_pose_xy is None:
            return np.empty((0, 2), dtype=np.float64)
        rel = np.asarray(points_xy, dtype=np.float64) - np.asarray(self._latest_pose_xy, dtype=np.float64).reshape(1, 2)
        cos_yaw = math.cos(self._latest_yaw_rad)
        sin_yaw = math.sin(self._latest_yaw_rad)
        x_forward = (cos_yaw * rel[:, 0]) + (sin_yaw * rel[:, 1])
        y_lateral = (-sin_yaw * rel[:, 0]) + (cos_yaw * rel[:, 1])
        return np.column_stack((x_forward, y_lateral))

    def _update_stop_override_from_routed_cones(self, *, routed_cones: list[ConeDetection]) -> None:
        if self._stop_line_marker_points_odom is not None:
            self._update_latched_stop_line_distance_from_pose()
            return

        if not self._parking_mode_active or not routed_cones:
            self._stop_line_forward_distance_m = float("nan")
            return

    def _clear_stop_line_state(self) -> None:
        self._stop_line_forward_distance_m = float("nan")
        self._stop_line_marker_points_odom = None
        if self._stop_override_active:
            self._publish_brake_cmd(0.0)
        self._stop_override_active = False

    def _update_latched_stop_line_distance_from_pose(self) -> None:
        if self._latest_pose_xy is None or self._stop_line_marker_points_odom is None:
            if not self._stop_override_active:
                self._stop_line_forward_distance_m = float("nan")
            return

        left_xy, right_xy = self._stop_line_marker_points_odom
        midpoint = np.asarray(
            [[0.5 * (float(left_xy[0]) + float(right_xy[0])), 0.5 * (float(left_xy[1]) + float(right_xy[1]))]],
            dtype=np.float64,
        )
        midpoint_vehicle = self._odom_points_to_vehicle_frame(midpoint)
        if midpoint_vehicle.size == 0:
            return

        self._stop_line_forward_distance_m = float(midpoint_vehicle[0, 0])
        activation_distance_m = max(0.0, float(self.stop_margin_m), float(self.target_margin_m))
        if float(self._stop_line_forward_distance_m) <= activation_distance_m:
            self._stop_override_active = True

    def _mask_points_on_latched_stop_line(self, orange_points_odom: np.ndarray) -> np.ndarray:
        if orange_points_odom.size == 0 or self._stop_line_marker_points_odom is None:
            return np.zeros((orange_points_odom.shape[0],), dtype=bool)

        left_xy, right_xy = self._stop_line_marker_points_odom
        p0 = np.asarray(left_xy, dtype=np.float64)
        p1 = np.asarray(right_xy, dtype=np.float64)
        line_vec = p1 - p0
        line_len = float(np.linalg.norm(line_vec))
        if line_len <= 1e-6:
            return np.zeros((orange_points_odom.shape[0],), dtype=bool)

        tangent = line_vec / line_len
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        rel = np.asarray(orange_points_odom, dtype=np.float64) - p0.reshape(1, 2)
        along = rel @ tangent
        normal_dist = np.abs(rel @ normal)
        along_margin_m = 0.5
        normal_margin_m = max(0.75, float(self.stop_line_pair_max_distance_m))
        return (
            (along >= -along_margin_m)
            & (along <= (line_len + along_margin_m))
            & (normal_dist <= normal_margin_m)
        )

    def _detect_stop_line_cluster_mask(self, orange_points_odom: np.ndarray) -> np.ndarray:
        if orange_points_odom.size == 0:
            return np.empty((0,), dtype=bool)

        orange_vehicle_points = self._odom_points_to_vehicle_frame(orange_points_odom)
        detected_pair = detect_stop_line_pair(
            orange_vehicle_points,
            max_pair_distance_m=self.stop_line_pair_max_distance_m,
        )
        if detected_pair is None:
            return np.zeros((orange_points_odom.shape[0],), dtype=bool)

        idx_a, idx_b, forward_distance_m = detected_pair
        point_a = np.asarray(orange_points_odom[idx_a], dtype=np.float64)
        point_b = np.asarray(orange_points_odom[idx_b], dtype=np.float64)
        midpoint = 0.5 * (point_a + point_b)
        half_width_m = max(
            float(self._state_machine.config.parking_corridor_half_width_m),
            0.5 * float(np.linalg.norm(point_b - point_a)),
        )

        self._stop_line_forward_distance_m = forward_distance_m
        self._stop_line_marker_points_odom = self._perpendicular_line_through_midpoint(
            midpoint_xy=midpoint,
            half_width_m=half_width_m,
        )
        return self._mask_points_on_latched_stop_line(orange_points_odom)

    def _detect_acceleration_stop_row_mask(self, orange_points_odom: np.ndarray) -> np.ndarray:
        if orange_points_odom.size == 0:
            return np.empty((0,), dtype=bool)

        orange_vehicle_points = self._odom_points_to_vehicle_frame(orange_points_odom)
        detection = detect_acceleration_stop_row(
            orange_vehicle_points,
            cluster_depth_m=self.acceleration_stop_row_cluster_depth_m,
            min_cluster_count=self.acceleration_stop_row_min_cluster_count,
            min_lateral_span_m=self.acceleration_stop_row_min_lateral_span_m,
            min_points_per_side=self.acceleration_stop_row_min_points_per_side,
        )
        if detection is None:
            detection = self._infer_acceleration_frontier_stop_row(orange_vehicle_points)
        if detection is None:
            return np.zeros((orange_points_odom.shape[0],), dtype=bool)

        self._apply_acceleration_stop_row_detection(detection=detection, orange_points_odom=orange_points_odom)
        return self._mask_points_on_latched_stop_line(orange_points_odom)

    def _apply_acceleration_stop_row_detection(
        self,
        *,
        detection: AccelerationStopRowDetection,
        orange_points_odom: np.ndarray,
    ) -> None:
        del orange_points_odom
        left_point_xy = self._vehicle_point_to_odom(
            x_forward_m=float(detection.forward_distance_m),
            y_lateral_m=float(detection.max_lateral_y_m),
        )
        right_point_xy = self._vehicle_point_to_odom(
            x_forward_m=float(detection.forward_distance_m),
            y_lateral_m=float(detection.min_lateral_y_m),
        )

        self._stop_line_forward_distance_m = float(detection.forward_distance_m)
        self._stop_line_marker_points_odom = (left_point_xy, right_point_xy)

    def _infer_acceleration_frontier_stop_row(
        self,
        orange_vehicle_points: np.ndarray,
    ) -> Optional[AccelerationStopRowDetection]:
        points = np.asarray(orange_vehicle_points, dtype=np.float64)
        if points.size == 0 or points.ndim != 2 or points.shape[1] != 2:
            return None

        valid_indices = np.flatnonzero(np.isfinite(points).all(axis=1) & (points[:, 0] > 0.0))
        if valid_indices.size < 4:
            return None

        valid_points = points[valid_indices]
        left_points = valid_points[valid_points[:, 1] >= 0.0]
        right_points = valid_points[valid_points[:, 1] < 0.0]
        if left_points.shape[0] == 0 or right_points.shape[0] == 0:
            return None

        frontier_idx = int(valid_indices[int(np.argmax(valid_points[:, 0]))])
        left_median_y_m = float(np.median(left_points[:, 1]))
        right_median_y_m = float(np.median(right_points[:, 1]))
        center_y_m = 0.5 * (left_median_y_m + right_median_y_m)
        half_width_m = max(0.75, 0.5 * (left_median_y_m - right_median_y_m))

        return AccelerationStopRowDetection(
            point_indices=(frontier_idx,),
            forward_distance_m=float(points[frontier_idx, 0]),
            min_lateral_y_m=float(center_y_m - half_width_m),
            max_lateral_y_m=float(center_y_m + half_width_m),
        )

    def _perpendicular_line_through_midpoint(
        self,
        *,
        midpoint_xy: np.ndarray,
        half_width_m: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        lateral_axis = np.asarray(
            [
                -math.sin(self._latest_yaw_rad),
                math.cos(self._latest_yaw_rad),
            ],
            dtype=np.float64,
        )
        midpoint = np.asarray(midpoint_xy, dtype=np.float64)
        return (
            tuple((midpoint - (float(half_width_m) * lateral_axis)).tolist()),
            tuple((midpoint + (float(half_width_m) * lateral_axis)).tolist()),
        )

    def _stop_override_timer_cb(self) -> None:
        stamp = self._latest_odom_stamp
        if stamp is None:
            stamp = self.get_clock().now().to_msg()
        if self._approach_lock_active:
            self._publish_steering_lock(True)
            return
        if not self._stop_override_active:
            return
        self._publish_parking_override_cmd(stamp)

    def _publish_steering_lock(self, locked: bool) -> None:
        msg = Bool()
        msg.data = locked
        self._steering_lock_pub.publish(msg)

    def _publish_parking_override_cmd(self, stamp) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = stamp
        msg.drive.speed = self._parking_override_speed_mps()
        msg.drive.steering_angle = 0.0
        self._cmd_pub.publish(msg)
        self._publish_brake_cmd(self._parking_override_brake_cmd())

    def _parking_override_speed_mps(self) -> float:
        target_forward_m = self._target_forward_distance_m()
        if not np.isfinite(target_forward_m) or target_forward_m <= 0.0:
            return 0.0

        speed_gain = max(0.0, float(self.stop_approach_speed_gain))
        if speed_gain <= 0.0:
            return 0.0

        return float(max(0.0, min(float(self._latest_speed_mps), speed_gain * target_forward_m)))

    def _parking_override_brake_cmd(self) -> float:
        target_forward_m = self._target_forward_distance_m()
        if not np.isfinite(target_forward_m):
            return 0.0
        if target_forward_m <= max(0.0, float(self.brake_activation_margin_m)):
            return float(self.brake_command)
        return 0.0

    def _publish_brake_cmd(self, command: float) -> None:
        msg = Float32()
        msg.data = float(np.clip(float(command), 0.0, 1.0))
        self._brake_pub.publish(msg)

    def _publish_diagnostics(self, stamp) -> None:
        snapshot = self._latest_snapshot
        status = DiagnosticStatus()
        status.name = "skidpad_router/state"
        status.level = DiagnosticStatus.OK
        status.message = snapshot.stage_name
        status.hardware_id = self.get_name()
        status.values = self._build_diagnostic_values(snapshot)
        msg = DiagnosticArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.status = [status]
        self._diag_pub.publish(msg)

    def _publish_markers(self, stamp) -> None:
        snapshot = self._latest_snapshot
        cfg = self._state_machine.config
        markers = MarkerArray()

        if self.event_mode == "acceleration":
            markers.markers.append(self._make_delete_marker(ns="stage_text", marker_id=0, stamp=stamp))
            markers.markers.append(self._make_delete_marker(ns="left_circle", marker_id=1, stamp=stamp))
            markers.markers.append(self._make_delete_marker(ns="right_circle", marker_id=2, stamp=stamp))
            markers.markers.append(self._make_delete_marker(ns="crossroads", marker_id=3, stamp=stamp))
            markers.markers.append(self._make_stop_line_marker(marker_id=4, stamp=stamp))
            markers.markers.append(self._make_stop_target_marker(marker_id=5, stamp=stamp))
            markers.markers.append(self._make_stop_target_center_marker(marker_id=6, stamp=stamp))
            markers.markers.append(self._make_acceleration_parking_status_marker(marker_id=7, stamp=stamp))
            self._viz_pub.publish(markers)
            return

        text_marker = Marker()
        text_marker.header.frame_id = self.odom_frame
        text_marker.header.stamp = stamp
        text_marker.ns = "stage_text"
        text_marker.id = 0
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.orientation.w = 1.0
        text_marker.pose.position.x = float(cfg.crossroads_center_xy[0])
        text_marker.pose.position.y = float(cfg.crossroads_center_xy[1])
        text_marker.pose.position.z = 1.5
        text_marker.scale.z = 0.9
        text_marker.color.a = 1.0
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.text = self._build_status_text(snapshot)
        markers.markers.append(text_marker)

        markers.markers.append(
            self._make_circle_marker(
                marker_id=1,
                stamp=stamp,
                center_xy=cfg.left_circle_center_xy,
                radius_m=float(np.mean(cfg.circle_radius_window_m)),
                color_rgb=(0.0, 0.0, 0.0),
                ns="left_circle",
            )
        )
        markers.markers.append(
            self._make_circle_marker(
                marker_id=2,
                stamp=stamp,
                center_xy=cfg.right_circle_center_xy,
                radius_m=float(np.mean(cfg.circle_radius_window_m)),
                color_rgb=(0.0, 0.0, 0.0),
                ns="right_circle",
            )
        )
        markers.markers.append(self._make_crossroads_marker(marker_id=3, stamp=stamp))
        markers.markers.append(self._make_stop_line_marker(marker_id=4, stamp=stamp))
        markers.markers.append(self._make_stop_target_marker(marker_id=5, stamp=stamp))
        markers.markers.append(self._make_delete_marker(ns="parking_target_center", marker_id=6, stamp=stamp))
        markers.markers.append(self._make_delete_marker(ns="parking_status", marker_id=7, stamp=stamp))
        self._viz_pub.publish(markers)

    def _make_delete_marker(self, *, ns: str, marker_id: int, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.action = Marker.DELETE
        return marker

    def _make_crossroads_marker(self, *, marker_id: int, stamp) -> Marker:
        cfg = self._state_machine.config
        cx, cy = cfg.crossroads_center_xy
        hx, hy = cfg.crossroads_half_extents_xy
        corners = [
            (cx - hx, cy - hy),
            (cx + hx, cy - hy),
            (cx + hx, cy + hy),
            (cx - hx, cy + hy),
            (cx - hx, cy - hy),
        ]
        marker = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp = stamp
        marker.ns = "crossroads"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in corners]
        return marker

    def _make_circle_marker(
        self,
        *,
        marker_id: int,
        stamp,
        center_xy: tuple[float, float],
        radius_m: float,
        color_rgb: tuple[float, float, float],
        ns: str,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.06
        marker.color.a = 0.85
        marker.color.r = float(color_rgb[0])
        marker.color.g = float(color_rgb[1])
        marker.color.b = float(color_rgb[2])
        cx, cy = center_xy
        marker.points = [
            Point(
                x=float(cx + (radius_m * math.cos(theta))),
                y=float(cy + (radius_m * math.sin(theta))),
                z=0.05,
            )
            for theta in np.linspace(0.0, 2.0 * math.pi, 33)
        ]
        return marker

    def _make_stop_line_marker(self, *, marker_id: int, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp = stamp
        marker.ns = "stop_line"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.35
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        if (
            self._latest_pose_xy is None
            or not self._parking_mode_active
            or self._stop_line_marker_points_odom is None
        ):
            marker.action = Marker.DELETE
            return marker

        marker.action = Marker.ADD
        left_point_xy, right_point_xy = self._stop_line_marker_points_odom
        marker.points = [
            Point(x=float(left_point_xy[0]), y=float(left_point_xy[1]), z=0.25),
            Point(x=float(right_point_xy[0]), y=float(right_point_xy[1]), z=0.25),
        ]
        return marker

    def _make_stop_target_marker(self, *, marker_id: int, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp = stamp
        marker.ns = "stop_target"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.45
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        if (
            self._latest_pose_xy is None
            or not self._parking_mode_active
            or self._stop_line_marker_points_odom is None
        ):
            marker.action = Marker.DELETE
            return marker

        marker.action = Marker.ADD
        target_line = self._stop_target_line_points_odom()
        if target_line is None:
            marker.action = Marker.DELETE
            return marker

        left_point_xy, right_point_xy = target_line
        marker.points = [
            Point(
                x=float(left_point_xy[0]),
                y=float(left_point_xy[1]),
                z=0.22,
            ),
            Point(
                x=float(right_point_xy[0]),
                y=float(right_point_xy[1]),
                z=0.22,
            ),
        ]
        return marker

    def _make_stop_target_center_marker(self, *, marker_id: int, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp = stamp
        marker.ns = "parking_target_center"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        marker.color.a = 0.95
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 0.2

        target_center_xy = self._stop_target_center_odom()
        if target_center_xy is None:
            marker.action = Marker.DELETE
            return marker

        marker.action = Marker.ADD
        marker.pose.position.x = float(target_center_xy[0])
        marker.pose.position.y = float(target_center_xy[1])
        marker.pose.position.z = 0.3
        return marker

    def _make_acceleration_parking_status_marker(self, *, marker_id: int, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp = stamp
        marker.ns = "parking_status"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.8
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        if self.event_mode != "acceleration" or self._latest_pose_xy is None:
            marker.action = Marker.DELETE
            return marker

        marker.action = Marker.ADD
        status_anchor_xy = self._stop_target_center_odom()
        if status_anchor_xy is None:
            status_anchor_xy = self._latest_pose_xy
        marker.pose.position.x = float(status_anchor_xy[0])
        marker.pose.position.y = float(status_anchor_xy[1])
        marker.pose.position.z = 1.6
        marker.text = (
            f"parking_found={int(self._stop_line_marker_points_odom is not None)} "
            f"stop_latched={int(self._stop_line_marker_points_odom is not None)}\n"
            f"row_forward_m={self._format_distance(self._stop_line_forward_distance_m)} "
            f"target_forward_m={self._format_distance(self._target_forward_distance_m())}"
        )
        return marker

    def _stop_target_line_points_odom(
        self,
    ) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        if self._latest_pose_xy is None or self._stop_line_marker_points_odom is None:
            return None

        left_point_xy, right_point_xy = self._stop_line_marker_points_odom
        stop_offset_m = max(0.0, float(self.target_margin_m))
        backward_dx = stop_offset_m * math.cos(self._latest_yaw_rad)
        backward_dy = stop_offset_m * math.sin(self._latest_yaw_rad)
        return (
            (float(left_point_xy[0] - backward_dx), float(left_point_xy[1] - backward_dy)),
            (float(right_point_xy[0] - backward_dx), float(right_point_xy[1] - backward_dy)),
        )

    def _stop_target_center_odom(self) -> Optional[tuple[float, float]]:
        target_line = self._stop_target_line_points_odom()
        if target_line is None:
            return None

        left_point_xy, right_point_xy = target_line
        return (
            0.5 * (float(left_point_xy[0]) + float(right_point_xy[0])),
            0.5 * (float(left_point_xy[1]) + float(right_point_xy[1])),
        )

    def _target_forward_distance_m(self) -> float:
        if not np.isfinite(self._stop_line_forward_distance_m):
            return float("nan")
        return float(self._stop_line_forward_distance_m) - max(0.0, float(self.target_margin_m))

    def _format_distance(self, distance_m: float) -> str:
        return f"{distance_m:.2f}" if np.isfinite(distance_m) else "nan"

    def _build_diagnostic_values(self, snapshot: SkidpadRouterSnapshot) -> list[KeyValue]:
        return [
            KeyValue(key="stage_name", value=snapshot.stage_name),
            KeyValue(key="active_branch", value=snapshot.active_branch),
            KeyValue(key="completed_laps", value=str(snapshot.completed_laps)),
            KeyValue(key="route_index", value=str(snapshot.route_index)),
            KeyValue(key="current_route_pass", value=str(snapshot.current_route_pass)),
            KeyValue(key="total_route_passes", value=str(snapshot.total_route_passes)),
            KeyValue(key="completed_route_passes", value=str(snapshot.completed_route_passes)),
            KeyValue(key="in_crossroads", value=str(int(snapshot.in_crossroads))),
            KeyValue(key="lap_angle_accum_rad", value=f"{snapshot.lap_angle_accum_rad:.6f}"),
            KeyValue(key="lap_armed", value=str(int(snapshot.lap_armed))),
            KeyValue(key="parked", value=str(int(snapshot.parked))),
            KeyValue(key="parking_mode_active", value=str(int(self._parking_mode_active))),
            KeyValue(key="acceleration_parking_latched", value=str(int(self._acceleration_parking_latched))),
            KeyValue(key="orange_only_cone_count", value=str(int(self._orange_only_cone_count))),
            KeyValue(
                key="parking_boundary_override_count",
                value=str(int(self._parking_boundary_override_count)),
            ),
            KeyValue(key="stop_override_active", value=str(int(self._stop_override_active))),
            KeyValue(key="stop_line_latched", value=str(int(self._stop_line_marker_points_odom is not None))),
            KeyValue(key="stop_line_forward_distance_m", value=f"{self._stop_line_forward_distance_m:.6f}"),
            KeyValue(key="target_forward_distance_m", value=f"{self._target_forward_distance_m():.6f}"),
            KeyValue(key="vehicle_speed_mps", value=f"{self._latest_speed_mps:.6f}"),
            KeyValue(key="approach_lock_active", value=str(int(self._approach_lock_active))),
            KeyValue(key="approach_distance_from_spawn_m", value=f"{self._approach_distance_from_spawn_m:.2f}"),
        ]

    def _maybe_shutdown_after_route_complete(self) -> None:
        if self._route_complete_shutdown_requested:
            return
        if not self.shutdown_on_route_complete or self.event_mode != "skidpad":
            return
        snapshot = self._latest_snapshot
        if snapshot.total_route_passes <= 0:
            return
        if snapshot.completed_route_passes < snapshot.total_route_passes:
            return

        self._route_complete_shutdown_requested = True
        self.get_logger().info(
            "skidpad route target reached: "
            f"{snapshot.completed_route_passes}/{snapshot.total_route_passes} route_laps. "
            "Exiting router so the top-level launch can shut down the run."
        )
        self._request_process_exit(parent_delay_s=0.0, force_delay_s=2.0)
        if rclpy.ok():
            rclpy.shutdown()

    def _maybe_shutdown_after_parking_complete(self, *, stamp) -> None:
        if self._parking_complete_shutdown_requested:
            return
        if not self.shutdown_on_parking_complete or self.event_mode not in {"skidpad", "acceleration"}:
            self._acceleration_parked_since_sec = None
            return
        if self.event_mode == "skidpad":
            snapshot = self._latest_snapshot
            if not snapshot.parked:
                return
            self._parking_complete_shutdown_requested = True
            self.get_logger().info(
                "skidpad parking target reached: "
                f"stage={snapshot.stage_name} route={snapshot.completed_route_passes}/{snapshot.total_route_passes}. "
                "Exiting router so the top-level launch can shut down the run."
            )
            self._request_process_exit(parent_delay_s=0.0, force_delay_s=2.0)
            if rclpy.ok():
                rclpy.shutdown()
            return

        if not self._parking_mode_active or self._stop_line_marker_points_odom is None or not self._stop_override_active:
            self._acceleration_parked_since_sec = None
            return
        target_forward_m = self._target_forward_distance_m()
        if not np.isfinite(target_forward_m) or target_forward_m > 0.0:
            self._acceleration_parked_since_sec = None
            return

        speed_threshold_mps = float(self._state_machine.config.parked_speed_threshold_mps)
        if float(self._latest_speed_mps) > speed_threshold_mps:
            self._acceleration_parked_since_sec = None
            return

        now_sec = float(getattr(stamp, "sec", 0)) + (float(getattr(stamp, "nanosec", 0)) * 1e-9)
        if now_sec <= 0.0:
            now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        if self._acceleration_parked_since_sec is None:
            self._acceleration_parked_since_sec = now_sec
            return

        hold_time_s = float(self._state_machine.config.parked_hold_time_s)
        if (now_sec - self._acceleration_parked_since_sec) < hold_time_s:
            return

        self._parking_complete_shutdown_requested = True
        self.get_logger().info(
            "acceleration parking target reached: "
            f"speed={self._latest_speed_mps:.3f}m/s target_forward={target_forward_m:.3f}m. "
            "Exiting router so the top-level launch can shut down the run."
        )
        self._request_process_exit(parent_delay_s=0.0, force_delay_s=2.0)
        if rclpy.ok():
            rclpy.shutdown()

    def _request_process_exit(self, *, parent_delay_s: float = 0.0, force_delay_s: float = 2.0) -> None:
        # Route completion should stop the full launched sim, not only this router node.
        parent_pid = os.getppid()
        process_group_id = os.getpgrp()

        def _signal_parent_launch() -> None:
            try:
                if parent_pid > 1:
                    os.kill(parent_pid, signal.SIGINT)
            except Exception as exc:
                self.get_logger().warn(f"Failed to interrupt parent launch process for route-complete exit: {exc}")

        def _force_exit_process_group() -> None:
            try:
                if parent_pid > 1:
                    os.kill(parent_pid, signal.SIGINT)
            except Exception:
                pass
            try:
                os.killpg(process_group_id, signal.SIGINT)
            except Exception:
                pass
            os._exit(0)

        try:
            parent_timer = threading.Timer(max(0.0, float(parent_delay_s)), _signal_parent_launch)
            parent_timer.daemon = True
            parent_timer.start()
            force_timer = threading.Timer(max(0.0, float(force_delay_s)), _force_exit_process_group)
            force_timer.daemon = True
            force_timer.start()
        except Exception as exc:
            self.get_logger().warn(f"Failed to schedule launch shutdown for route-complete exit: {exc}")
            if parent_pid > 1:
                os.kill(parent_pid, signal.SIGINT)
            os.killpg(process_group_id, signal.SIGINT)

    def _build_status_text(self, snapshot: SkidpadRouterSnapshot) -> str:
        stop_line_latched = int(self._stop_line_marker_points_odom is not None)
        stop_line_distance_text = (
            f"{self._stop_line_forward_distance_m:.2f}" if np.isfinite(self._stop_line_forward_distance_m) else "nan"
        )
        route_progress = "off"
        if snapshot.total_route_passes > 0:
            route_progress = f"{snapshot.current_route_pass}/{snapshot.total_route_passes}"
        return (
            f"stage={snapshot.stage_name} branch={snapshot.active_branch} "
            f"route={route_progress} circle_laps={snapshot.completed_laps} armed={int(snapshot.lap_armed)}\n"
            f"stop_latched={stop_line_latched} stop_forward_m={stop_line_distance_text}"
        )

    def _throttled_warn(self, message: str, throttle_s: float = 1.0) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self._last_warn_sec < 0.0 or (now_sec - self._last_warn_sec) >= float(throttle_s):
            self.get_logger().warn(message)
            self._last_warn_sec = now_sec


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = SkidpadRouterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
