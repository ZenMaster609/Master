#!/usr/bin/env python3
"""Live controller diagnostics plotting window."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from ..logging.stanley_debug_plots import StanleyDebugLivePlot
from ..logging.steering_diagnostics import (
    PLANNER_DIAG_DEFAULTS,
    heading_error,
    nearest_point_on_polyline,
    parse_planner_diag,
    signed_cross_track_error,
)


class ControllerDiagnosticsPlotNode(Node):
    """Compute and display the live controller diagnostics window."""

    def __init__(self) -> None:
        super().__init__("controller_diagnostics_plot_node")

        self.declare_parameter("controller_diagnostics_rate_hz", 50.0)
        self.declare_parameter("controller_diagnostics_cmd_topic", "/cmd")
        self.declare_parameter("controller_diagnostics_steering_topic", "/sim/steering_angle")
        self.declare_parameter(
            "controller_diagnostics_joint_states_topic", "/sim/raw/joint_states"
        )
        self.declare_parameter("controller_diagnostics_odom_topic", "/sim/odom")
        self.declare_parameter("controller_diagnostics_path_topic", "/planned_centerline")
        self.declare_parameter(
            "controller_diagnostics_planner_diag_topic",
            "/delaunay_planner/diagnostics",
        )
        self.declare_parameter("controller_diagnostics_live_plot_rate_hz", 10.0)
        self.declare_parameter("controller_diagnostics_live_buffer_sec", 30.0)

        self._diagnostics_rate_hz = max(
            1.0,
            float(self.get_parameter("controller_diagnostics_rate_hz").value),
        )
        self._cmd_topic = (
            str(self.get_parameter("controller_diagnostics_cmd_topic").value).strip()
            or "/cmd"
        )
        self._steering_topic = (
            str(self.get_parameter("controller_diagnostics_steering_topic").value).strip()
            or "/sim/steering_angle"
        )
        self._joint_states_topic = (
            str(
                self.get_parameter("controller_diagnostics_joint_states_topic").value
            ).strip()
            or "/sim/raw/joint_states"
        )
        self._odom_topic = (
            str(self.get_parameter("controller_diagnostics_odom_topic").value).strip()
            or "/sim/odom"
        )
        self._path_topic = (
            str(self.get_parameter("controller_diagnostics_path_topic").value).strip()
            or "/planned_centerline"
        )
        self._planner_diag_topic = (
            str(
                self.get_parameter("controller_diagnostics_planner_diag_topic").value
            ).strip()
            or "/delaunay_planner/diagnostics"
        )
        self._plot_rate_hz = max(
            0.1,
            float(self.get_parameter("controller_diagnostics_live_plot_rate_hz").value),
        )
        self._plot_buffer_sec = max(
            2.0,
            float(self.get_parameter("controller_diagnostics_live_buffer_sec").value),
        )

        self._cmd_stamp_sec = float("nan")
        self._cmd_recv_sec = float("nan")
        self._desired_steering_rad = float("nan")
        self._desired_speed_mps = float("nan")
        self._actual_steering_deg = float("nan")
        self._vehicle_x_m = float("nan")
        self._vehicle_y_m = float("nan")
        self._vehicle_yaw_rad = float("nan")
        self._vehicle_yaw_rate_rps = float("nan")
        self._vehicle_speed_mps = float("nan")
        self._centerline_xy = np.empty((0, 2), dtype=np.float64)
        self._planner_metrics: Dict[str, float] = dict(PLANNER_DIAG_DEFAULTS)

        self._plot_stride = max(
            1,
            int(round(self._diagnostics_rate_hz / self._plot_rate_hz)),
        )
        self._plot_counter = 0
        self._plot: Optional[StanleyDebugLivePlot] = None

        self.create_subscription(
            AckermannDriveStamped,
            self._cmd_topic,
            self._cmd_callback,
            10,
        )
        self.create_subscription(
            Float32,
            self._steering_topic,
            self._actual_callback,
            10,
        )
        self.create_subscription(
            JointState,
            self._joint_states_topic,
            self._joint_states_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            self._odom_topic,
            self._odom_callback,
            10,
        )
        self.create_subscription(
            NavPath,
            self._path_topic,
            self._path_callback,
            10,
        )
        self.create_subscription(
            DiagnosticArray,
            self._planner_diag_topic,
            self._planner_callback,
            10,
        )

        try:
            self._plot = StanleyDebugLivePlot(
                buffer_sec=self._plot_buffer_sec,
                sample_rate_hz=self._plot_rate_hz,
                title="Controller Diagnostics (Live)",
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(
                f"Failed to initialize controller diagnostics live plot: {exc}"
            )

        self._timer = self.create_timer(
            1.0 / self._diagnostics_rate_hz,
            self._sample,
        )
        rclpy.get_default_context().on_shutdown(self.shutdown)

        self.get_logger().info(
            "Controller diagnostics plot enabled: "
            f"cmd={self._cmd_topic} steering={self._steering_topic} "
            f"joint_states={self._joint_states_topic} odom={self._odom_topic} "
            f"path={self._path_topic}"
        )

    def _cmd_callback(self, msg: AckermannDriveStamped) -> None:
        self._desired_steering_rad = float(msg.drive.steering_angle)
        self._desired_speed_mps = float(msg.drive.speed)
        self._cmd_recv_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        self._cmd_stamp_sec = (
            float(msg.header.stamp.sec) + (float(msg.header.stamp.nanosec) * 1e-9)
        )

    def _actual_callback(self, msg: Float32) -> None:
        self._actual_steering_deg = float(msg.data)

    def _joint_states_callback(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        candidates = (
            ("steering_fl_joint", "steering_fr_joint"),
            ("left_steering_hinge_joint", "right_steering_hinge_joint"),
            ("left_steering_joint", "right_steering_joint"),
        )

        idx_left = -1
        idx_right = -1
        for left_name, right_name in candidates:
            if left_name in msg.name and right_name in msg.name:
                idx_left = msg.name.index(left_name)
                idx_right = msg.name.index(right_name)
                break

        if idx_left < 0 or idx_right < 0:
            for idx, joint_name in enumerate(msg.name):
                token = str(joint_name).lower()
                if "steering" not in token:
                    continue
                if idx_left < 0 and ("left" in token or token.endswith("_fl_joint")):
                    idx_left = idx
                elif idx_right < 0 and (
                    "right" in token or token.endswith("_fr_joint")
                ):
                    idx_right = idx
            if idx_left < 0 or idx_right < 0:
                return

        if idx_left >= len(msg.position) or idx_right >= len(msg.position):
            return
        left_rad = float(msg.position[idx_left])
        right_rad = float(msg.position[idx_right])
        self._actual_steering_deg = math.degrees(0.5 * (left_rad + right_rad))

    def _odom_callback(self, msg: Odometry) -> None:
        self._vehicle_x_m = float(msg.pose.pose.position.x)
        self._vehicle_y_m = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self._vehicle_yaw_rad = self._yaw_from_quat(
            float(q.x),
            float(q.y),
            float(q.z),
            float(q.w),
        )
        self._vehicle_yaw_rate_rps = float(msg.twist.twist.angular.z)
        self._vehicle_speed_mps = float(
            math.hypot(
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.linear.y),
            )
        )

    def _path_callback(self, msg: NavPath) -> None:
        if not msg.poses:
            self._centerline_xy = np.empty((0, 2), dtype=np.float64)
            return
        points = np.empty((len(msg.poses), 2), dtype=np.float64)
        for idx, pose_stamped in enumerate(msg.poses):
            points[idx, 0] = float(pose_stamped.pose.position.x)
            points[idx, 1] = float(pose_stamped.pose.position.y)
        self._centerline_xy = points

    def _planner_callback(self, msg: DiagnosticArray) -> None:
        self._planner_metrics = parse_planner_diag(msg)

    def _sample(self) -> None:
        if self._plot is None:
            return

        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        desired = float(self._desired_steering_rad)
        actual_deg = float(self._actual_steering_deg)
        actual_rad = (
            math.radians(actual_deg) if math.isfinite(actual_deg) else float("nan")
        )
        cmd_age_sec = (
            now_sec - self._cmd_recv_sec if math.isfinite(self._cmd_recv_sec) else float("nan")
        )

        centerline_count = int(self._centerline_xy.shape[0])
        centerline_available = 1 if centerline_count >= 2 else 0
        cte_geom = float("nan")
        heading_geom = float("nan")
        nearest_idx_geom = float("nan")
        nearest_path_point = np.array([float("nan"), float("nan")], dtype=np.float64)
        if centerline_available and math.isfinite(self._vehicle_x_m) and math.isfinite(self._vehicle_y_m):
            nearest_idx_int, nearest_path_point = nearest_point_on_polyline(
                self._vehicle_x_m,
                self._vehicle_y_m,
                self._centerline_xy,
            )
            nearest_idx_geom = float(nearest_idx_int)
            cte_geom, tangent_yaw = signed_cross_track_error(
                self._vehicle_x_m,
                self._vehicle_y_m,
                self._centerline_xy,
            )
            if math.isfinite(tangent_yaw) and math.isfinite(self._vehicle_yaw_rad):
                heading_geom = heading_error(self._vehicle_yaw_rad, tangent_yaw)

        planner_jump = float(self._planner_metrics.get("centerline_jump_max_m", float("nan")))
        planner_churn = float(
            self._planner_metrics.get("selected_edge_churn_ratio", float("nan"))
        )
        planner_tracked_delta = float(
            self._planner_metrics.get("tracked_cones_frame_delta_p95_m", float("nan"))
        )
        cte_ctrl = float(self._planner_metrics.get("cross_track_error_m", float("nan")))
        heading_ctrl = float(self._planner_metrics.get("heading_error_rad", float("nan")))
        cte_m = cte_ctrl if math.isfinite(cte_ctrl) else cte_geom
        heading_err = heading_ctrl if math.isfinite(heading_ctrl) else heading_geom

        raw_cmd = float(self._planner_metrics.get("raw_steering_cmd_rad", float("nan")))
        final_cmd = float(self._planner_metrics.get("final_steering_cmd_rad", float("nan")))
        if not math.isfinite(final_cmd):
            final_cmd = desired
        steering_ref = final_cmd if math.isfinite(final_cmd) else desired
        err = (
            (actual_rad - steering_ref)
            if math.isfinite(actual_rad) and math.isfinite(steering_ref)
            else float("nan")
        )

        vehicle_speed_dbg = float(
            self._planner_metrics.get("vehicle_speed_mps", float("nan"))
        )
        vehicle_speed = (
            vehicle_speed_dbg if math.isfinite(vehicle_speed_dbg) else self._vehicle_speed_mps
        )
        nearest_path_index = float(
            self._planner_metrics.get("nearest_path_index", float("nan"))
        )
        if not math.isfinite(nearest_path_index):
            nearest_path_index = nearest_idx_geom

        row = {
            "timestamp_sec": now_sec,
            "cmd_stamp_sec": self._cmd_stamp_sec,
            "cmd_age_sec": cmd_age_sec,
            "desired_steering_rad": desired,
            "desired_speed_mps": self._desired_speed_mps,
            "actual_steering_deg": actual_deg,
            "actual_steering_rad": actual_rad,
            "steering_error_rad": err,
            "steering_error_abs_rad": abs(err) if math.isfinite(err) else float("nan"),
            "raw_steering_cmd_rad": raw_cmd,
            "final_steering_cmd_rad": final_cmd,
            "steering_after_clamp_rad": float(
                self._planner_metrics.get("steering_after_clamp_rad", float("nan"))
            ),
            "steering_after_filter_rad": float(
                self._planner_metrics.get("steering_after_filter_rad", float("nan"))
            ),
            "steering_after_rate_limit_rad": float(
                self._planner_metrics.get("steering_after_rate_limit_rad", float("nan"))
            ),
            "steering_saturated_flag": float(
                self._planner_metrics.get("steering_saturated_flag", float("nan"))
            ),
            "vehicle_x_m": self._vehicle_x_m,
            "vehicle_y_m": self._vehicle_y_m,
            "vehicle_yaw_rad": self._vehicle_yaw_rad,
            "vehicle_yaw_rate_rps": self._vehicle_yaw_rate_rps,
            "vehicle_speed_mps": vehicle_speed,
            "speed_term_mps": float(
                self._planner_metrics.get("speed_term_mps", float("nan"))
            ),
            "centerline_available": centerline_available,
            "centerline_point_count": centerline_count,
            "cte_m": cte_m,
            "cte_abs_m": abs(cte_m) if math.isfinite(cte_m) else float("nan"),
            "heading_error_rad": heading_err,
            "heading_error_abs_rad": abs(heading_err)
            if math.isfinite(heading_err)
            else float("nan"),
            "heading_contribution_rad": float(
                self._planner_metrics.get("heading_contribution_rad", float("nan"))
            ),
            "cross_track_contribution_rad": float(
                self._planner_metrics.get("cross_track_contribution_rad", float("nan"))
            ),
            "yaw_rate_damping_contribution_rad": float(
                self._planner_metrics.get(
                    "yaw_rate_damping_contribution_rad", float("nan")
                )
            ),
            "nearest_path_index": nearest_path_index,
            "heading_path_index": float(
                self._planner_metrics.get("heading_path_index", float("nan"))
            ),
            "target_point_x_base_m": float(
                self._planner_metrics.get("target_point_x_base_m", float("nan"))
            ),
            "target_point_y_base_m": float(
                self._planner_metrics.get("target_point_y_base_m", float("nan"))
            ),
            "target_point_x_frame_m": float(
                self._planner_metrics.get("target_point_x_frame_m", float("nan"))
            ),
            "target_point_y_frame_m": float(
                self._planner_metrics.get("target_point_y_frame_m", float("nan"))
            ),
            "nearest_path_point_x_m": float(nearest_path_point[0]),
            "nearest_path_point_y_m": float(nearest_path_point[1]),
            "planner_centerline_jump_max_m": planner_jump,
            "planner_selected_edge_churn_ratio": planner_churn,
            "planner_tracked_cones_frame_delta_p95_m": planner_tracked_delta,
        }

        self._plot_counter += 1
        if self._plot_counter >= self._plot_stride:
            if not self._plot.update(row):
                self.get_logger().warn("Controller diagnostics live plot closed")
                self._plot.close()
                self._plot = None
            self._plot_counter = 0

    def shutdown(self) -> None:
        if self._plot is not None:
            self._plot.close()
            self._plot = None

    @staticmethod
    def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerDiagnosticsPlotNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
