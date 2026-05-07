from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetectionArray
from visualization_msgs.msg import MarkerArray

from sim_car.cones.tracking.fusion import normalize_color, resolve_boundary_colors_for_planning
from sim_car.cones.tracking.pose import (
    convert_odom_child_pose_to_base_frame,
    project_planar_pose_constant_twist,
)
from sim_car.planning.tracked_cone_planner_contract import build_steering_controller
from sim_car.planning.midline_memory import (
    CommittedMidlineMemory,
    MidlineCandidate,
    MidlineMemoryConfig,
)
from sim_car.planning.tracked_cone_planner_geometry import (
    extract_forward_path_from_pose,
    resample_to_count,
    sample_path_at_lengths,
)
from sim_car.planning.planning_diagnostics import DiagnosticsMixin
from sim_car.planning.planning_state_machine import StateMachineMixin
from sim_car.planning.planning_visualization import VisualizationMixin
from sim_car.planning.planner_constants import (
    MSG_TRACK_STATE_CONFIRMED,
    MSG_TRACK_STATE_TENTATIVE,
)
from sim_car.planning.planner_constants import TrackedConePlanningFrame, TrackedConePlanningMetadata
from sim_car.planning.tracked_cone_planner_geometry import (
    _base_point_to_odom,
    _odom_point_to_base,
    _transform_point,
    _yaw_from_quat,
    path_cumulative_lengths,
)


class PlannerNodeUtilitiesMixin:
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
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp_time,
                timeout=timeout,
            )
        except (TransformException, ValueError):
            pass
        try:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=timeout,
            )
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
            normalized = candidate.strip().strip("/")
            if normalized and normalized not in out:
                out.append(normalized)

        add(token)
        leaf = token.strip("/").split("/")[-1] if token else ""
        add(leaf)

        if leaf in {"front_axle", "base_link", "base_footprint"}:
            prefix = token.strip("/").rsplit("/", 1)[0] if "/" in token.strip("/") else ""
            for alias in ("front_axle", "base_link", "base_footprint"):
                add(f"{prefix}/{alias}" if prefix else alias)
                add(alias)
        return out

    def _is_alias(self, frame_a: str, frame_b: str) -> bool:
        a = set(self._frame_aliases(frame_a))
        b = set(self._frame_aliases(frame_b))
        return bool(a.intersection(b))


class TrackedConePlannerBase(
    PlannerNodeUtilitiesMixin,
    DiagnosticsMixin,
    VisualizationMixin,
    StateMachineMixin,
    Node,
):
    """Shared tracked-cone planner runtime used by midpoint/single-boundary planners."""

    # Width of the planned centerline strip in viz markers.  Subclasses that
    # render at a different width (e.g. SingleBoundaryPlannerNode uses 0.09)
    # should shadow this attribute in the class body.
    _centerline_marker_width_m: float = 0.20

    # -----------------------------------------------------------------------
    # Shared __init__ helpers — identical across all three planner nodes
    # -----------------------------------------------------------------------

    def _init_common_planner_state(self) -> None:
        """Initialise the ~50 state attributes shared by all three planner nodes."""
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_cones_msg = None
        self._latest_odom_msg = None
        self._latest_speed_mps = 0.0
        self._latest_yaw_rate_rps = 0.0
        self._last_throttled_log_sec: dict[str, float] = {}
        self._previous_centerline = None
        self._previous_raw_centerline = None
        self._previous_tracked_points = None
        self._previous_edge_keys: set = set()
        self._last_valid_centerline = None
        self._last_valid_raw_midpoint_chain = None
        self._last_valid_pair_segments = None
        self._last_valid_pair_track_ids = None
        self._current_pair_segments_for_viz = None
        self._last_valid_width_m = self._filtered_track_width_m
        self._last_valid_time_sec = -1.0
        self._committed_centerline = None
        self._commit_stable_frame_count = 0
        self._hold_mode_active = False
        self._hold_clean_frame_count = 0
        self._last_speed_cmd = None
        self._last_steering_cmd = None
        self._last_operator_state = None
        self._last_operator_reason = None
        self._midline_buffer_path = None
        self._midline_buffer_confidence = 0.0
        self._midline_buffer_last_update_sec = -1.0
        self._midline_memory = None
        self._last_midline_update_mode = "hold"
        self._last_midline_candidate_update_ok = False
        self._last_midline_candidate_update_reason = "ok"
        self._last_midline_candidate_jump_m = float("nan")
        self._last_midline_near_lateral_delta_max_m = float("nan")
        self._last_midline_buffer_confidence = 0.0
        self._midline_recovery_count = 0
        self._last_viz_left_boundary = None
        self._last_viz_right_boundary = None
        self._last_viz_raw_offset_path = None
        self._last_viz_pair_segments = None
        self._last_viz_raw_midpoint_chain = None
        self._candidate_jump_reject_streak = 0
        self._lap_tracking_completed_laps = 0
        self._lap_tracking_armed = True

        self._active_planner_mode = "waiting"
        self._active_remembered_cone_count = 0
        self._active_stale_cone_count = 0
        self._active_left_chain_length = 0
        self._active_right_chain_length = 0
        self._active_pair_count = 0
        self._active_unknown_pair_count = 0
        self._active_filtered_track_width_m = self._filtered_track_width_m
        self._active_held_path_flag = 0

    def _init_common_ros_interfaces(self) -> None:
        """Create the five standard publishers, two standard subscriptions, and the loop timer."""
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
            f"{self.get_name()} ready "
            f"cones={self.tracked_cones_topic} odom={self.odom_topic} "
            f"cmd={self.cmd_topic} path={self.centerline_topic} viz={self.viz_topic} "
            f"planning_frame={self.planning_frame} controller={self.controller_type}"
        )

    def _build_steering_controller(self):
        return build_steering_controller(
            node=self,
            controller_type=self.controller_type,
            publish_rate_hz=self.publish_rate_hz,
        )

    def _cones_cb(self, msg: ConeDetectionArray) -> None:
        self._latest_cones_msg = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom_msg = msg
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        frame_tokens = f"{msg.header.frame_id}/{msg.child_frame_id}".lower()
        if "base_link" in frame_tokens or "base_footprint" in frame_tokens:
            self._latest_speed_mps = abs(vx)
            self._latest_yaw_rate_rps = float(msg.twist.twist.angular.z)
        else:
            self._latest_speed_mps = float(math.hypot(vx, vy))
            self._latest_yaw_rate_rps = float(msg.twist.twist.angular.z)

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

    def _execute_controller(
        self,
        control_path: np.ndarray,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple:
        try:
            controller_output = self._controller.compute(
                control_path=control_path,
                speed_mps=self._latest_speed_mps,
                yaw_rate_rps=self._latest_yaw_rate_rps,
            )
        except ValueError as exc:
            self._warn_throttled("controller_compute_error", f"controller compute failed: {exc}")
            zero_flag = int(self._apply_no_path_behavior())
            cmd_speed = float(self._last_speed_cmd) if self._last_speed_cmd is not None else 0.0
            cmd_steering = float(self._last_steering_cmd) if self._last_steering_cmd is not None else 0.0
            return None, None, cmd_speed, cmd_steering, 0.0, zero_flag, True
        cmd_steering = float(controller_output.steering_rad)
        cmd_speed = self._compute_speed_command(float(controller_output.kappa))
        lookahead = float(controller_output.lookahead_m)
        control_target_base = np.asarray(controller_output.target_point_base, dtype=np.float64)
        self._last_speed_cmd = cmd_speed
        self._last_steering_cmd = cmd_steering
        self._publish_cmd(cmd_speed, cmd_steering)
        control_target_frame = self._resolve_control_target_frame(
            control_target_base, target_frame, vehicle_x, vehicle_y, vehicle_yaw,
        )
        ctrl_debug = None
        if controller_output.stanley_debug is not None:
            ctrl_debug = self._build_stanley_debug_metrics(
                controller_output.stanley_debug, control_target_frame,
            )
        return control_target_frame, ctrl_debug, cmd_speed, cmd_steering, lookahead, 0, False

    def _build_stanley_debug_metrics(
        self,
        debug,
        control_target_frame: Optional[np.ndarray],
    ) -> dict[str, float]:
        return {
            "heading_error_rad": float(debug.heading_error_rad),
            "cross_track_error_m": float(debug.cross_track_error_m),
            "vehicle_speed_mps": float(debug.vehicle_speed_mps),
            "speed_term_mps": float(debug.speed_term_mps),
            "heading_contribution_rad": float(debug.heading_contribution_rad),
            "cross_track_contribution_rad": float(debug.cross_track_contribution_rad),
            "yaw_rate_damping_contribution_rad": float(debug.yaw_rate_damping_contribution_rad),
            "yaw_rate_rps": float(self._latest_yaw_rate_rps),
            "raw_steering_cmd_rad": float(debug.raw_steering_cmd_rad),
            "steering_after_clamp_rad": float(debug.steering_after_clamp_rad),
            "steering_after_filter_rad": float(debug.steering_after_filter_rad),
            "steering_after_rate_limit_rad": float(debug.steering_after_rate_limit_rad),
            "final_steering_cmd_rad": float(debug.final_steering_cmd_rad),
            "steering_saturated_flag": 1.0 if bool(debug.steering_saturated_flag) else 0.0,
            "nearest_path_index": float(debug.nearest_path_index),
            "heading_path_index": float(debug.heading_path_index),
            "target_point_x_base_m": float(debug.target_point_x_base_m),
            "target_point_y_base_m": float(debug.target_point_y_base_m),
            "target_point_x_frame_m": (
                float(control_target_frame[0]) if control_target_frame is not None else float("nan")
            ),
            "target_point_y_frame_m": (
                float(control_target_frame[1]) if control_target_frame is not None else float("nan")
            ),
        }

    def _buffer_and_anchor_centerline(
        self,
        result,
        raw_centerline: np.ndarray,
        candidate_source: str,
        raw_midpoint_chain: Optional[np.ndarray],
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
    ) -> tuple:
        candidate_update_ok, candidate_update_reason = self._candidate_path_is_updateable(
            candidate_centerline=raw_centerline, vehicle_x=vehicle_x, vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw, result=result, candidate_source=candidate_source,
        )
        support_centerline = (
            raw_midpoint_chain
            if raw_midpoint_chain is not None
            else getattr(result, "raw_offset_path", raw_centerline)
        )
        centerline = self._update_midline_buffer(
            candidate_centerline=raw_centerline, candidate_source=candidate_source,
            candidate_update_ok=candidate_update_ok, candidate_update_reason=candidate_update_reason,
            frame_id=target_frame, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            result=result, now_sec=now_sec, support_centerline=support_centerline,
        )
        candidate_update_ok = bool(getattr(self, "_last_midline_candidate_update_ok", candidate_update_ok))
        candidate_update_reason = str(getattr(self, "_last_midline_candidate_update_reason", candidate_update_reason))
        buffered_centerline = np.array(centerline, copy=True)
        centerline = self._anchor_centerline_near_vehicle(
            centerline=centerline, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        return centerline, buffered_centerline, candidate_update_ok, candidate_update_reason

    def _build_midline_status(
        self,
        result,
        raw_centerline: np.ndarray,
        centerline: np.ndarray,
        candidate_source: str,
        candidate_update_ok: bool,
        candidate_update_reason: str,
        pair_segments_for_viz: np.ndarray,
        raw_midpoint_chain: Optional[np.ndarray],
        target_frame: Optional[str],
        vehicle_x: Optional[float],
        vehicle_y: Optional[float],
        vehicle_yaw: Optional[float],
        now_sec: float,
    ) -> tuple:
        status = result.status
        if not candidate_update_ok and centerline.shape[0] > 0:
            status = f"{status}; holding stored midline"
        elif self._is_publishing_stored_midline(raw_centerline, centerline):
            status = f"{status}; publishing stored midline"
        if candidate_source != "validated" and raw_centerline.shape[0] > 0:
            status = f"{status}; using {candidate_source}"
        self._update_midline_pair_memory(
            result=result,
            raw_centerline=raw_centerline,
            centerline=centerline,
            pair_segments_for_viz=pair_segments_for_viz,
            target_frame=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
            now_sec=now_sec,
        )
        if raw_centerline.shape[0] == 0 and self._last_valid_pair_segments is not None and centerline.shape[0] > 0:
            pair_segments_for_viz = np.array(self._last_valid_pair_segments, copy=True)
        return status, pair_segments_for_viz, raw_midpoint_chain

    def _update_midline_pair_memory(
        self,
        *,
        result,
        raw_centerline: np.ndarray,
        centerline: np.ndarray,
        pair_segments_for_viz: np.ndarray,
        target_frame: Optional[str],
        vehicle_x: Optional[float],
        vehicle_y: Optional[float],
        vehicle_yaw: Optional[float],
        now_sec: float,
    ) -> None:
        if target_frame is not None and result.status == "ok" and result.pair_segments.size > 0:
            self._remember_pairs(
                result=result, frame_id=target_frame,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw, now_sec=now_sec,
            )
        elif centerline.shape[0] > 0 and raw_centerline.shape[0] > 0:
            self._remember_pairs(result=result, now_sec=now_sec)
            remember_valid = getattr(self, "_remember_valid_pair_geometry", None)
            if remember_valid is not None:
                remember_valid(result, pair_segments_for_viz)
        if centerline.shape[0] > 0 and raw_centerline.shape[0] > 0:
            self._last_valid_pair_segments = (
                pair_segments_for_viz if pair_segments_for_viz.size > 0 else self._last_valid_pair_segments
            )

    @staticmethod
    def _is_publishing_stored_midline(raw_centerline: np.ndarray, centerline: np.ndarray) -> bool:
        return (
            raw_centerline.shape[0] > 0
            and centerline.shape[0] > 0
            and (raw_centerline.shape != centerline.shape or not np.allclose(raw_centerline, centerline))
        )

    def _apply_hold_logic(
        self,
        result,
        centerline: np.ndarray,
        candidate_update_ok: bool,
        candidate_update_reason: str,
        status: str,
        now_sec: float,
        pair_segments_for_viz: Optional[np.ndarray] = None,
        raw_midpoint_chain: Optional[np.ndarray] = None,
    ) -> tuple:
        plan_hold_active = False
        publish_mode = "fresh"
        hold_reason = result.reject_reason
        if centerline.shape[0] > 0 and candidate_update_ok:
            continue_holding = self._advance_hold_hysteresis(plan_ok=True)
            if continue_holding:
                held_centerline = self._held_centerline(now_sec)
                if held_centerline is not None:
                    centerline = held_centerline
                    plan_hold_active = True
                    publish_mode = "held"
                    status = (
                        f"{status}; hysteresis holding previous valid centerline "
                        f"({self._hold_clean_frame_count}/{self.hold_exit_clean_frames})"
                    )
        else:
            self._advance_hold_hysteresis(plan_ok=False)
            held_centerline = self._held_centerline(now_sec)
            if held_centerline is not None:
                centerline = held_centerline
                plan_hold_active = True
                publish_mode = "held"
                status = f"{status}; holding previous valid centerline"
            else:
                publish_mode = "held"
        if plan_hold_active:
            hold_reason = result.reject_reason or candidate_update_reason or status
        return centerline, publish_mode, plan_hold_active, hold_reason, status

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
            delay_s=float(getattr(self, "odom_lag_compensation_s", 0.0)),
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
        return max(0.1, float(getattr(self, "vehicle_wheelbase_m", 1.65)))

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
            boundary_hints.append(str(getattr(cone, "boundary_color", "")).strip())
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
        odom_x: float,
        odom_y: float,
        odom_yaw: float,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        points: list[tuple[float, float]] = []
        raw_colors: list[str] = []
        boundary_hints: list[str] = []
        confs: list[float] = []
        for cone in msg.cones:
            x_base, y_base = _odom_point_to_base(
                float(cone.position.x),
                float(cone.position.y),
                odom_x,
                odom_y,
                odom_yaw,
            )
            points.append((x_base, y_base))
            raw_colors.append(normalize_color(cone.color))
            boundary_hints.append(str(getattr(cone, "boundary_color", "")).strip())
            confs.append(float(np.clip(float(cone.confidence), 0.0, 1.0)))
        return self._resolve_planner_cone_colors(points, raw_colors, boundary_hints, confs, vehicle_xy, vehicle_yaw)

    def _convert_base_cones_to_odom(
        self,
        msg: ConeDetectionArray,
        odom_x: float,
        odom_y: float,
        odom_yaw: float,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
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
            boundary_hints.append(str(getattr(cone, "boundary_color", "")).strip())
            confs.append(float(np.clip(float(cone.confidence), 0.0, 1.0)))
        return self._resolve_planner_cone_colors(points, raw_colors, boundary_hints, confs, vehicle_xy, vehicle_yaw)

    def _resolve_planner_cone_colors(
        self,
        points: list[tuple[float, float]],
        raw_colors: list[str],
        boundary_hints: list[str],
        confs: list[float],
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
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

    @staticmethod
    def _path_cumulative_lengths(points: np.ndarray) -> np.ndarray:
        return path_cumulative_lengths(points)

    @staticmethod
    def _sample_path_at_lengths(
        points: np.ndarray,
        cum_lengths: np.ndarray,
        samples: np.ndarray,
    ) -> np.ndarray:
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
    def _resample_to_count(points: np.ndarray, count: int) -> np.ndarray:
        return resample_to_count(points, count)

    def _publish_outputs(
        self,
        *,
        frame_id: str,
        centerline: np.ndarray,
        raw_centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        result: Optional[object],
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
        del control_path_point_count
        now = self.get_clock().now().to_msg()
        self._publish_path_and_points(frame_id=frame_id, centerline=centerline, now=now)
        if self.enable_debug_markers:
            self._publish_viz_markers(
                now=now,
                frame_id=frame_id,
                centerline=centerline,
                raw_centerline=raw_centerline,
                raw_midpoint_chain=raw_midpoint_chain,
                result=result,
                status=status,
                control_target_frame=control_target_frame,
                operator_state=operator_state,
                operator_reason=operator_reason,
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

    def _publish_path_and_points(self, *, frame_id: str, centerline: np.ndarray, now: object) -> Path:
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

    def _publish_viz_markers(
        self,
        *,
        now: object,
        frame_id: str,
        centerline: np.ndarray,
        raw_centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        result: Optional[object],
        status: str,
        control_target_frame: Optional[np.ndarray],
        operator_state: str,
        operator_reason: str,
        cmd_speed: float,
        cmd_steering: float,
        lookahead: float,
        candidate_diagonal_count: int,
        selected_chain_length: int,
        seed_midpoint_distance_m: float,
        near_field_lateral_max_m: float,
        near_field_midpoint_kink_max_rad: float,
        hold_remaining_s: float,
    ) -> None:
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

    @staticmethod
    def _extract_cone_metadata(msg: ConeDetectionArray) -> TrackedConePlanningMetadata:
        track_ids: list[int] = []
        track_states: list[int] = []
        track_confidences: list[float] = []
        for idx, cone in enumerate(msg.cones):
            track_id = int(getattr(cone, "track_id", 0))
            track_state = int(getattr(cone, "track_state", MSG_TRACK_STATE_CONFIRMED))
            track_conf = float(getattr(cone, "track_confidence", 0.0))
            track_ids.append(track_id if track_id > 0 else idx)
            track_states.append(track_state)
            track_confidences.append(max(0.0, min(1.0, track_conf)))
        return TrackedConePlanningMetadata(
            track_ids=np.asarray(track_ids, dtype=np.int64),
            track_states=np.asarray(track_states, dtype=np.int64),
            track_confidences=np.asarray(track_confidences, dtype=np.float64),
        )

    def _tentative_cone_is_usable_for_planning(
        self,
        *,
        raw_color: str,
        boundary_color: str,
    ) -> bool:
        del raw_color
        del boundary_color
        return False

    def _planner_confidences_for_msg(
        self,
        *,
        msg: ConeDetectionArray,
        confidences: np.ndarray,
        metadata: TrackedConePlanningMetadata,
    ) -> np.ndarray:
        planner_confidences = np.asarray(confidences, dtype=np.float64).copy()
        valid_track_conf_mask = metadata.track_confidences > 1e-6
        planner_confidences[valid_track_conf_mask] = metadata.track_confidences[valid_track_conf_mask]
        tentative_keep_mask = np.asarray(
            [
                self._tentative_cone_is_usable_for_planning(
                    raw_color=getattr(cone, "color", ""),
                    boundary_color=getattr(cone, "boundary_color", ""),
                )
                for cone in msg.cones
            ],
            dtype=bool,
        )
        planner_confidences[
            (metadata.track_ids > 0)
            & (metadata.track_states == MSG_TRACK_STATE_TENTATIVE)
            & (~tentative_keep_mask)
        ] = 0.0
        return planner_confidences

    def _tracked_cone_planning_frame(
        self,
        *,
        msg: ConeDetectionArray,
        points_xy: np.ndarray,
        colors: list[str],
        confidences: np.ndarray,
    ) -> TrackedConePlanningFrame:
        metadata = self._extract_cone_metadata(msg)
        raw_colors = [normalize_color(getattr(cone, "color", "")) for cone in msg.cones]
        boundary_hints = [str(getattr(cone, "boundary_color", "")).strip().lower() for cone in msg.cones]
        return TrackedConePlanningFrame(
            points_xy=np.asarray(points_xy, dtype=np.float64),
            colors=list(colors),
            raw_confidences=np.asarray(confidences, dtype=np.float64),
            planner_confidences=self._planner_confidences_for_msg(
                msg=msg,
                confidences=confidences,
                metadata=metadata,
            ),
            raw_colors=raw_colors,
            boundary_hints=boundary_hints,
            metadata=metadata,
        )

    def _clear_midline_buffer(self) -> None:
        self._midline_buffer_path = None
        self._midline_buffer_confidence = 0.0
        self._midline_buffer_last_update_sec = -1.0
        memory = getattr(self, "_midline_memory", None)
        if memory is not None:
            memory.clear()

    def _update_candidate_jump_reject_streak(
        self,
        *,
        candidate_update_ok: bool,
        candidate_update_reason: str,
    ) -> tuple[bool, str]:
        if not candidate_update_ok and candidate_update_reason == "candidate_jump_rejected":
            self._candidate_jump_reject_streak += 1
            if self._candidate_jump_reject_streak >= self.candidate_jump_recover_frames:
                self._clear_midline_buffer()
                return True, "candidate_jump_recovery"
        else:
            self._candidate_jump_reject_streak = 0
        return candidate_update_ok, candidate_update_reason

    def _midline_memory_config(self) -> MidlineMemoryConfig:
        near_shift = float(getattr(self, "midline_near_max_shift_m", 0.07))
        mid_shift = float(getattr(self, "midline_mid_max_shift_m", 0.18))
        far_shift = float(getattr(self, "midline_far_max_shift_m", 0.35))
        return MidlineMemoryConfig(
            horizon_m=max(1.0, float(getattr(self, "midline_horizon_m", 30.0))),
            station_spacing_m=max(0.05, float(getattr(self, "midline_station_spacing_m", 0.5))),
            near_distance_m=max(0.0, float(getattr(self, "midline_near_distance_m", 4.0))),
            mid_distance_m=max(
                float(getattr(self, "midline_near_distance_m", 4.0)),
                float(getattr(self, "midline_mid_distance_m", 12.0)),
            ),
            near_alpha=float(np.clip(float(getattr(self, "midline_near_alpha", 0.04)), 0.0, 1.0)),
            mid_alpha=float(np.clip(float(getattr(self, "midline_mid_alpha", 0.12)), 0.0, 1.0)),
            far_alpha=float(np.clip(float(getattr(self, "midline_far_alpha", 0.30)), 0.0, 1.0)),
            near_max_lateral_shift_m=max(0.0, near_shift),
            mid_max_lateral_shift_m=max(near_shift, mid_shift),
            far_max_lateral_shift_m=max(mid_shift, far_shift),
            min_buffer_confidence=float(
                np.clip(float(getattr(self, "midline_min_buffer_confidence", 0.2)), 0.0, 1.0)
            ),
            hold_last_valid_duration_s=max(
                0.0,
                float(getattr(self, "midline_hold_last_valid_duration_s", getattr(self, "hold_last_valid_s", 3.0))),
            ),
            candidate_min_points=max(2, int(getattr(self, "candidate_min_points", 2))),
            candidate_min_extent_m=max(0.5, float(getattr(self, "candidate_min_extent_m", 1.0))),
            candidate_jump_reject_threshold_m=max(
                0.0,
                float(getattr(self, "candidate_jump_reject_threshold_m", 0.45)),
            ),
            candidate_jump_recover_frames=max(1, int(getattr(self, "candidate_jump_recover_frames", 3))),
            jump_check_horizon_m=max(0.5, float(getattr(self, "centerline_jump_horizon_m", 8.0))),
            min_estimated_extent_m=max(
                0.5,
                float(getattr(self, "midline_min_estimated_extent_m", 6.0)),
            ),
            max_estimation_extension_m=max(
                0.0,
                float(getattr(self, "midline_max_estimation_extension_m", 4.0)),
            ),
            max_estimation_join_lateral_m=max(
                0.0,
                float(getattr(self, "midline_max_estimation_join_lateral_m", 0.5)),
            ),
            max_estimation_join_heading_rad=max(
                0.0,
                float(getattr(self, "midline_max_estimation_join_heading_rad", 0.45)),
            ),
            allow_tangent_estimate_without_memory=bool(
                getattr(self, "midline_allow_tangent_estimate_without_memory", True)
            ),
            max_tangent_estimation_extension_m=max(
                0.0,
                float(getattr(self, "midline_max_tangent_estimation_extension_m", 2.0)),
            ),
        )

    def _ensure_midline_memory(self) -> CommittedMidlineMemory:
        config = self._midline_memory_config()
        memory = getattr(self, "_midline_memory", None)
        if memory is None:
            memory = CommittedMidlineMemory(config)
            existing_path = getattr(self, "_midline_buffer_path", None)
            if existing_path is not None and np.asarray(existing_path).shape[0] >= 2:
                memory.path = np.array(existing_path, copy=True)
                memory.confidence = float(getattr(self, "_midline_buffer_confidence", 1.0))
                memory.last_update_sec = float(getattr(self, "_midline_buffer_last_update_sec", -1.0))
            self._midline_memory = memory
        else:
            memory.config = config
        return memory

    def _update_midline_memory_common(
        self,
        *,
        candidate_centerline: np.ndarray,
        candidate_source: str,
        candidate_update_ok: bool | None,
        candidate_update_reason: str,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
        support_centerline: np.ndarray | None = None,
        direct_commit: bool = False,
        allow_estimation: bool = False,
    ) -> np.ndarray:
        self._last_midline_update_mode = "hold"
        self._last_midline_candidate_update_ok = False
        self._last_midline_candidate_update_reason = str(candidate_update_reason or "ok")
        self._last_midline_candidate_jump_m = float("nan")
        self._last_midline_near_lateral_delta_max_m = float("nan")
        self._last_midline_buffer_confidence = float(getattr(self, "_midline_buffer_confidence", 0.0))
        self._midline_recovery_count = int(getattr(self, "_midline_recovery_count", 0))
        self._last_midline_estimation_mode = "none"
        self._last_midline_estimated_point_count = 0
        self._last_midline_estimated_extent_m = 0.0
        self._last_midline_live_prefix_extent_m = 0.0
        self._last_midline_estimation_join_lateral_m = float("nan")
        self._last_midline_estimation_join_heading_rad = float("nan")

        candidate_path = np.asarray(candidate_centerline, dtype=np.float64)
        support_path = (
            np.asarray(support_centerline, dtype=np.float64)
            if support_centerline is not None
            else None
        )
        if not self._is_alias(frame_id, self.odom_frame):
            if candidate_path.shape[0] > 0:
                self._midline_buffer_path = np.array(candidate_path, copy=True)
                self._midline_buffer_last_update_sec = now_sec
                self._midline_buffer_confidence = 1.0
                self._last_midline_buffer_confidence = 1.0
                self._last_midline_candidate_update_ok = True
                self._last_midline_update_mode = "direct"
            return candidate_path

        memory = self._ensure_midline_memory()
        updateable = bool(candidate_update_ok) if candidate_update_ok is not None else True
        result = memory.update(
            candidate=MidlineCandidate(
                centerline=candidate_path,
                source=str(candidate_source),
                updateable=updateable,
                update_reason=str(candidate_update_reason or "ok"),
                support_path=support_path,
                direct_commit=bool(direct_commit),
                allow_estimation=bool(allow_estimation),
            ),
            vehicle_xy=(float(vehicle_x), float(vehicle_y)),
            vehicle_yaw=float(vehicle_yaw),
            now_sec=float(now_sec),
        )
        self._midline_buffer_path = (
            np.array(memory.path, copy=True)
            if memory.path is not None and memory.path.shape[0] >= 2
            else None
        )
        self._midline_buffer_last_update_sec = float(memory.last_update_sec)
        self._midline_buffer_confidence = float(memory.confidence)
        self._last_midline_update_mode = str(result.update_mode)
        self._last_midline_candidate_update_ok = bool(result.candidate_accepted)
        self._last_midline_candidate_update_reason = str(result.reason)
        self._last_midline_candidate_jump_m = float(result.candidate_jump_m)
        self._last_midline_near_lateral_delta_max_m = float(
            result.near_field_lateral_delta_max_m
        )
        self._last_midline_buffer_confidence = float(result.buffer_confidence)
        self._midline_recovery_count = int(result.recovery_count)
        self._last_midline_estimation_mode = str(result.estimation_mode)
        self._last_midline_estimated_point_count = int(result.estimated_point_count)
        self._last_midline_estimated_extent_m = float(result.estimated_extent_m)
        self._last_midline_live_prefix_extent_m = float(result.live_prefix_extent_m)
        self._last_midline_estimation_join_lateral_m = float(result.estimation_join_lateral_m)
        self._last_midline_estimation_join_heading_rad = float(result.estimation_join_heading_rad)
        return np.array(result.centerline, copy=True)

    def _midline_estimation_metrics_for_diagnostics(self) -> dict[str, object]:
        return {
            "midline_estimation_mode": getattr(self, "_last_midline_estimation_mode", "none"),
            "midline_estimated_point_count": int(
                getattr(self, "_last_midline_estimated_point_count", 0)
            ),
            "midline_estimated_extent_m": float(
                getattr(self, "_last_midline_estimated_extent_m", 0.0)
            ),
            "midline_live_prefix_extent_m": float(
                getattr(self, "_last_midline_live_prefix_extent_m", 0.0)
            ),
            "midline_estimation_join_lateral_m": float(
                getattr(self, "_last_midline_estimation_join_lateral_m", float("nan"))
            ),
            "midline_estimation_join_heading_rad": float(
                getattr(self, "_last_midline_estimation_join_heading_rad", float("nan"))
            ),
        }

    # -----------------------------------------------------------------------
    # Shared geometry helpers — identical across midpoint/corridor/single_boundary
    # -----------------------------------------------------------------------

    def _midline_blend_params(self, distance_ahead: float) -> tuple[float, float]:
        if distance_ahead <= self.midline_near_distance_m:
            return self.midline_near_alpha, self.midline_near_max_shift_m
        if distance_ahead <= self.midline_mid_distance_m:
            return self.midline_mid_alpha, self.midline_mid_max_shift_m
        return self.midline_far_alpha, self.midline_far_max_shift_m

    @staticmethod
    def _path_forward_extent_local(path_local: np.ndarray) -> float:
        if path_local.shape[0] == 0:
            return 0.0
        if path_local.shape[0] == 1:
            return max(0.0, float(path_local[0, 0]))
        max_x_idx = int(np.argmax(path_local[:, 0]))
        forward_segment = path_local[: max_x_idx + 1]
        if forward_segment.shape[0] >= 2:
            diffs = np.diff(forward_segment, axis=0)
            path_length = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        else:
            path_length = 0.0
        x_reach = max(0.0, float(path_local[max_x_idx, 0]))
        return max(path_length, x_reach)

    def _resample_midline_stations(self, path: np.ndarray) -> np.ndarray:
        if path.shape[0] < 2:
            return np.array(path, copy=True)
        cumulative = self._path_cumulative_lengths(path)
        total = min(float(cumulative[-1]), float(self.midline_horizon_m))
        if total <= 1e-6:
            return np.asarray(path[:1], dtype=np.float64)
        step = max(0.05, float(self.midline_station_spacing_m))
        samples = np.arange(0.0, total + 1e-9, step, dtype=np.float64)
        if samples.size == 0 or samples[-1] < total:
            samples = np.concatenate((samples, [total]))
        return self._sample_path_at_lengths(path, cumulative, samples)

    def _candidate_forward_extent_m(
        self,
        *,
        centerline: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> float:
        if centerline.shape[0] == 0:
            return 0.0
        local_path = self._centerline_to_vehicle_frame(
            centerline=centerline,
            frame_id=frame_id,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        return self._path_forward_extent_local(local_path)

    def _prepare_centerline_for_current_pose(
        self,
        *,
        centerline: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        preserve_live_lateral_near_vehicle: bool = False,
    ) -> np.ndarray:
        if centerline.shape[0] == 0:
            return centerline
        if not self._is_alias(frame_id, self.odom_frame):
            return centerline
        forward = self._extract_forward_path_from_pose(
            path=centerline,
            vehicle_xy=(vehicle_x, vehicle_y),
            resolution_m=self.midline_station_spacing_m,
        )
        prepared = (
            np.array(forward, copy=True)
            if forward is not None and forward.shape[0] > 0
            else np.array(centerline, copy=True)
        )
        return self._anchor_centerline_near_vehicle(
            centerline=prepared,
            frame_id=frame_id,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
            preserve_live_lateral_near_vehicle=preserve_live_lateral_near_vehicle,
        )

    def _should_preserve_near_vehicle_lateral(
        self, local: np.ndarray, anchor_length_m: float
    ) -> bool:
        """Hook for subclasses to override the near-vehicle lateral anchoring decision."""
        return False

    def _anchor_centerline_near_vehicle(
        self,
        *,
        centerline: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        preserve_live_lateral_near_vehicle: bool = False,
    ) -> np.ndarray:
        if centerline.shape[0] == 0:
            return centerline
        if not self._is_alias(frame_id, self.odom_frame):
            return centerline

        local = self._centerline_to_vehicle_frame(
            centerline=centerline,
            frame_id=frame_id,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        if local.shape[0] == 0:
            return centerline

        keep_mask = local[:, 0] >= -0.1
        local = local[keep_mask]
        if local.shape[0] == 0:
            local = np.array([[0.0, 0.0]], dtype=np.float64)

        anchor_length_m = max(0.5, min(1.5, float(self.midline_near_distance_m)))
        preserve = preserve_live_lateral_near_vehicle or self._should_preserve_near_vehicle_lateral(
            local, anchor_length_m
        )
        for idx in range(local.shape[0]):
            x_val = float(local[idx, 0])
            if x_val <= 0.0:
                local[idx, 1] = 0.0
                continue
            if not preserve and x_val < anchor_length_m:
                local[idx, 1] *= x_val / anchor_length_m

        if float(local[0, 0]) > anchor_length_m:
            # Path starts well ahead of the vehicle (e.g. committed from pair
            # memory that has no near-vehicle points). Insert the vehicle origin
            # rather than replacing the first pair point — otherwise zeroing it
            # creates a single long beam segment to local[1].
            local = np.vstack((np.array([[0.0, 0.0]], dtype=np.float64), local))
        else:
            local[0, 0] = 0.0
            local[0, 1] = 0.0
        if local.shape[0] == 1:
            local = np.vstack((local, np.array([[max(0.5, anchor_length_m), 0.0]], dtype=np.float64)))

        anchored = np.empty_like(local)
        for idx in range(local.shape[0]):
            ox, oy = _base_point_to_odom(
                float(local[idx, 0]),
                float(local[idx, 1]),
                vehicle_x,
                vehicle_y,
                vehicle_yaw,
            )
            anchored[idx, 0] = ox
            anchored[idx, 1] = oy
        return anchored

    # -----------------------------------------------------------------------
    # Shared logging helper
    # -----------------------------------------------------------------------

    def _log_mode_summary(
        self,
        *,
        mode: str,
        result: Any,
        operator_state: str,
        operator_reason: str,
        hold_active: bool,
    ) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get("mode_summary", -1.0)
        if (now_sec - last_sec) < self.log_throttle_s:
            return
        self.get_logger().info(
            (
                "mode=%s state=%s reason=%s status=%s tracks=%d stale=%d left=%d right=%d pairs=%d unknown=%d width=%.2f held=%d"
            )
            % (
                mode,
                operator_state,
                operator_reason,
                result.status,
                int(self._active_remembered_cone_count),
                int(self._active_stale_cone_count),
                int(result.left_chain_length),
                int(result.right_chain_length),
                int(result.accepted_pair_count),
                int(result.unknown_pair_count),
                float(self._filtered_track_width_m),
                1 if hold_active else 0,
            )
        )
        self._last_throttled_log_sec["mode_summary"] = now_sec

    # -----------------------------------------------------------------------
    # Timer preamble — shared by all three planner nodes
    # -----------------------------------------------------------------------

    def _resolve_cone_planning_context(
        self,
    ) -> Optional[tuple[Any, str, float, float, float, np.ndarray, list, np.ndarray]]:
        """Snapshot cones, resolve vehicle pose and frame.

        Returns (cones_msg, target_frame, vehicle_x, vehicle_y, vehicle_yaw,
        points_xy, colors, confidences) or None if a failure was detected and
        an empty cycle was already published.
        """
        cones_msg = self._latest_cones_msg
        if cones_msg is None:
            zero_cmd_sent = int(self._apply_no_path_behavior())
            self._publish_empty_cycle(
                frame_id=self.odom_frame,
                status="waiting for /tracked_cones",
                operator_state="waiting",
                operator_reason="waiting_for_cones",
                cmd_speed=0.0,
                cmd_steering=0.0,
                lookahead=0.0,
                zero_cmd_sent_flag=zero_cmd_sent,
            )
            return None

        source_frame = str(cones_msg.header.frame_id).strip() or self.odom_frame
        target_frame = self._resolve_planning_frame(source_frame, cones_msg.header.stamp)

        pose = self._resolve_vehicle_pose(target_frame, cones_msg.header.stamp)
        if pose is None and target_frame != self.odom_frame:
            self._warn_throttled(
                "fallback_odom",
                f"cannot resolve base pose in {target_frame}; falling back to odom",
            )
            target_frame = self.odom_frame
            pose = self._resolve_vehicle_pose(target_frame, cones_msg.header.stamp)

        if pose is None:
            zero_cmd_sent = int(self._apply_no_path_behavior())
            self._publish_empty_cycle(
                frame_id=target_frame,
                status="missing vehicle pose (tf and /sim/odom unavailable)",
                operator_state="waiting",
                operator_reason="missing_vehicle_pose",
                cmd_speed=0.0,
                cmd_steering=0.0,
                lookahead=0.0,
                zero_cmd_sent_flag=zero_cmd_sent,
            )
            return None

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
                    "fallback_odom_cones",
                    f"cannot transform cones to {target_frame}; planning in odom frame",
                )
                target_frame = self.odom_frame
                pose = self._resolve_vehicle_pose(target_frame, cones_msg.header.stamp)
                if pose is None:
                    zero_cmd_sent = int(self._apply_no_path_behavior())
                    self._publish_empty_cycle(
                        frame_id=self.odom_frame,
                        status="cone transform failed and odom pose unavailable",
                        operator_state="waiting",
                        operator_reason="cone_transform_unavailable",
                        cmd_speed=0.0,
                        cmd_steering=0.0,
                        lookahead=0.0,
                        zero_cmd_sent_flag=zero_cmd_sent,
                    )
                    return None
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
                status="cone transform unavailable",
                operator_state="waiting",
                operator_reason="cone_transform_unavailable",
                cmd_speed=0.0,
                cmd_steering=0.0,
                lookahead=0.0,
                zero_cmd_sent_flag=zero_cmd_sent,
            )
            return None

        return cones_msg, target_frame, vehicle_x, vehicle_y, vehicle_yaw, points_xy, colors, confidences

    def _update_smalltrack_lap_from_orange_cones(
        self,
        *,
        cones_msg: ConeDetectionArray,
        points_xy: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> None:
        orange_idx = [
            idx
            for idx, cone in enumerate(cones_msg.cones)
            if normalize_color(getattr(cone, "color", "")) == "orange"
        ]
        points = np.asarray(points_xy, dtype=np.float64)
        if not orange_idx or points.ndim != 2 or points.shape[0] <= max(orange_idx):
            return

        rel = points[np.asarray(orange_idx, dtype=np.int64)] - np.asarray(
            [vehicle_x, vehicle_y],
            dtype=np.float64,
        )
        cos_yaw = math.cos(float(vehicle_yaw))
        sin_yaw = math.sin(float(vehicle_yaw))
        forward = (cos_yaw * rel[:, 0]) + (sin_yaw * rel[:, 1])
        lateral = (-sin_yaw * rel[:, 0]) + (cos_yaw * rel[:, 1])
        near_mask = np.abs(lateral) <= 5.0
        if not np.any(near_mask):
            return

        forward = forward[near_mask]
        lateral = lateral[near_mask]
        order = np.argsort(np.abs(forward) + (0.25 * np.abs(lateral)))
        gate_forward_m = float(np.mean(forward[order[: min(2, len(order))]]))

        if gate_forward_m > 1.0:
            self._lap_tracking_armed = True
        elif self._lap_tracking_armed and gate_forward_m <= 0.0:
            self._lap_tracking_completed_laps += 1
            self._lap_tracking_armed = False
            self.get_logger().info(
                f"smalltrack laps={self._lap_tracking_completed_laps}/{self.lap_tracking_target_laps or 0}"
            )
