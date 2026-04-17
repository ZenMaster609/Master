from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Point, PoseArray
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import Buffer, TransformListener
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.controller_config import build_steering_controller
from sim_car.planning.midline_memory import (
    CommittedMidlineMemory,
    MidlineCandidate,
    MidlineMemoryConfig,
)
from sim_car.planning.planner_runtime_types import TrackedConePlanningFrame, TrackedConePlanningMetadata
from sim_car.planning.tracked_cone_planner_runtime import TrackedConePlannerRuntime

MSG_TRACK_STATE_TENTATIVE = int(getattr(ConeDetection, "TRACK_STATE_TENTATIVE", 0))
MSG_TRACK_STATE_CONFIRMED = int(getattr(ConeDetection, "TRACK_STATE_CONFIRMED", 1))
MSG_TRACK_STATE_STALE = int(getattr(ConeDetection, "TRACK_STATE_STALE", 2))


class TrackedConePlannerBase(TrackedConePlannerRuntime):
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
            ox, oy = self._base_point_to_odom(
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
    # Shared visualization helpers
    # -----------------------------------------------------------------------

    def _make_pair_segment_marker(
        self,
        *,
        frame_id: str,
        stamp: Any,
        marker_id: int,
        ns: str,
        pair_segments: Optional[np.ndarray],
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
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(width)
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        if pair_segments is None:
            return marker
        for segment in np.asarray(pair_segments, dtype=np.float64):
            if segment.shape != (2, 2):
                continue
            for x, y in segment:
                point = self._point_msg(float(x), float(y), 0.03)
                marker.points.append(point)
        return marker

    @staticmethod
    def _point_msg(x: float, y: float, z: float) -> Point:
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        return point

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
    # Shared marker builder — midpoint and single_boundary planners
    # -----------------------------------------------------------------------

    def _build_markers(
        self,
        *,
        now,
        frame_id: str,
        result: Optional[Any],
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
            self._append_status_marker(
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

        left_boundary = np.array(result.left_boundary, copy=True)
        right_boundary = np.array(result.right_boundary, copy=True)
        if left_boundary.shape[0] > 0:
            self._last_viz_left_boundary = np.array(left_boundary, copy=True)
        elif self._last_viz_left_boundary is not None:
            left_boundary = np.array(self._last_viz_left_boundary, copy=True)
        if right_boundary.shape[0] > 0:
            self._last_viz_right_boundary = np.array(right_boundary, copy=True)
        elif self._last_viz_right_boundary is not None:
            right_boundary = np.array(self._last_viz_right_boundary, copy=True)

        if self.show_boundary_chains:
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="boundary_left",
                    points=left_boundary,
                    color=(0.2, 0.45, 1.0, 0.95),
                    width=0.10,
                    z_offset=0.12,
                )
            )
            marker_id += 1
            left_points_marker = self._make_points_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns="boundary_left_points",
                points=left_boundary,
                color=(0.2, 0.55, 1.0, 1.0),
                scale=0.22,
            )
            for point in left_points_marker.points:
                point.z = 0.13
            arr.markers.append(left_points_marker)
            marker_id += 1
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="boundary_right",
                    points=right_boundary,
                    color=(1.0, 0.9, 0.2, 0.95),
                    width=0.10,
                    z_offset=0.14,
                )
            )
            marker_id += 1
            right_points_marker = self._make_points_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns="boundary_right_points",
                points=right_boundary,
                color=(1.0, 0.92, 0.25, 1.0),
                scale=0.22,
            )
            for point in right_points_marker.points:
                point.z = 0.15
            arr.markers.append(right_points_marker)
            marker_id += 1

        active_boundary = np.empty((0, 2), dtype=np.float64)
        if result.planner_mode == "single_boundary":
            if result.active_boundary_side == "blue":
                active_boundary = left_boundary
            elif result.active_boundary_side == "yellow":
                active_boundary = right_boundary
        if active_boundary.shape[0] > 0:
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="single_boundary_active",
                    points=active_boundary,
                    color=(0.15, 1.0, 0.2, 1.0),
                    width=0.16,
                    z_offset=0.20,
                )
            )
            marker_id += 1

        if self.show_pair_lines:
            pair_segs = self._current_pair_segments_for_viz
            if pair_segs is not None and pair_segs.size > 0:
                self._last_viz_pair_segments = np.array(pair_segs, copy=True)
            elif self._last_viz_pair_segments is not None:
                pair_segs = self._last_viz_pair_segments
            arr.markers.append(
                self._make_pair_segment_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="accepted_pairs",
                    pair_segments=pair_segs,
                    color=(0.2, 1.0, 0.3, 0.95),
                    width=0.07,
                )
            )
            marker_id += 1

        if self.show_raw_midpoint_chain:
            midpoint_chain = raw_midpoint_chain
            if midpoint_chain.size > 0:
                self._last_viz_raw_midpoint_chain = np.array(midpoint_chain, copy=True)
            elif self._last_viz_raw_midpoint_chain is not None:
                midpoint_chain = np.array(self._last_viz_raw_midpoint_chain, copy=True)
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="raw_midpoint_chain",
                    points=midpoint_chain,
                    color=(1.0, 1.0, 1.0, 0.95),
                    width=0.06,
                    z_offset=0.03,
                )
            )
            marker_id += 1

        if self.show_raw_offset_path:
            raw_offset_path = np.array(result.raw_offset_path, copy=True)
            if raw_offset_path.shape[0] > 0:
                self._last_viz_raw_offset_path = np.array(raw_offset_path, copy=True)
            elif self._last_viz_raw_offset_path is not None:
                raw_offset_path = np.array(self._last_viz_raw_offset_path, copy=True)
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="raw_offset_path",
                    points=raw_offset_path,
                    color=(0.2, 1.0, 1.0, 0.9),
                    width=0.09,
                    z_offset=0.18,
                )
            )
            marker_id += 1
            raw_offset_points_marker = self._make_points_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns="raw_offset_path_points",
                points=raw_offset_path,
                color=(0.2, 1.0, 1.0, 1.0),
                scale=0.20,
            )
            for point in raw_offset_points_marker.points:
                point.z = 0.19
            arr.markers.append(raw_offset_points_marker)
            marker_id += 1

        if self.show_raw_prevalidation_centerline:
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="raw_prevalidation_centerline",
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
                ns="centerline",
                points=centerline,
                color=(0.95, 0.15, 0.15, 1.0),
                width=self._centerline_marker_width_m,
                z_offset=0.07,
            )
        )
        marker_id += 1

        if self.show_lookahead_point and control_target_frame is not None:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = now
            marker.ns = "lookahead"
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
