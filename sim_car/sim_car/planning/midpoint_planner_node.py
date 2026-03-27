#!/usr/bin/env python3
"""Midpoint boundary planner over tracked cone detections."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Point, PoseArray
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import Buffer, TransformListener
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.controllers.factory import create_steering_controller
from sim_car.controllers.stanley_controller import StanleyConfig
from sim_car.planning.delaunay_planner_core import (
    compute_centerline_jump_max,
    edge_churn_count,
    edge_churn_ratio,
    selected_edge_keys,
    tracked_cones_frame_delta_p95,
)
from sim_car.planning.midpoint_planner_core import (
    MidpointPlannerConfig,
    MidpointPlannerPrior,
    MidpointPlannerResult,
    _finalize_path,
    compute_midpoint_centerline,
    update_track_width_estimate,
)
from sim_car.planning.planner_runtime_types import PlannerIdentity
from sim_car.planning.tracked_cone_planner_base import TrackedConePlannerBase

MSG_TRACK_STATE_TENTATIVE = int(getattr(ConeDetection, "TRACK_STATE_TENTATIVE", 0))
MSG_TRACK_STATE_CONFIRMED = int(getattr(ConeDetection, "TRACK_STATE_CONFIRMED", 1))
MSG_TRACK_STATE_STALE = int(getattr(ConeDetection, "TRACK_STATE_STALE", 2))
_PAIR_PASSED_MARGIN_M = 0.5
_MIN_PROJECTABLE_PAIR_COUNT = 2


@dataclass
class _PairMemoryEntry:
    left_track_id: int
    right_track_id: int
    midpoint_x_odom: float
    midpoint_y_odom: float
    left_x_odom: float
    left_y_odom: float
    right_x_odom: float
    right_y_odom: float


class MidpointPlannerNode(TrackedConePlannerBase):
    """Tracked-cone midpoint planner with shared path-memory stabilization."""

    def __init__(self) -> None:
        self._planner_identity = PlannerIdentity(
            node_name="midpoint_planner_node",
            planner_mode="midpoint",
            diagnostics_prefix="midpoint_planner",
            diagnostics_topic="/midpoint_planner/diagnostics",
        )
        Node.__init__(self, self._planner_identity.node_name)
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
        self._last_valid_pair_segments: Optional[np.ndarray] = None
        self._last_valid_pair_track_ids: Optional[np.ndarray] = None
        self._current_pair_segments_for_viz: Optional[np.ndarray] = None
        self._last_valid_width_m: Optional[float] = self._filtered_track_width_m
        self._last_valid_time_sec: float = -1.0
        self._committed_centerline: Optional[np.ndarray] = None
        self._commit_stable_frame_count: int = 0
        self._hold_mode_active: bool = False
        self._hold_clean_frame_count: int = 0
        self._last_speed_cmd: Optional[float] = None
        self._last_steering_cmd: Optional[float] = None
        self._last_operator_state: Optional[str] = None
        self._last_operator_reason: Optional[str] = None
        self._pair_memory: list[_PairMemoryEntry] = []
        self._midline_buffer_path: Optional[np.ndarray] = None
        self._midline_buffer_confidence: float = 0.0
        self._midline_buffer_last_update_sec: float = -1.0
        self._last_viz_left_boundary: Optional[np.ndarray] = None
        self._last_viz_right_boundary: Optional[np.ndarray] = None
        self._last_viz_raw_offset_path: Optional[np.ndarray] = None
        self._candidate_jump_reject_streak: int = 0

        self._active_planner_mode = "waiting"
        self._active_remembered_cone_count = 0
        self._active_stale_cone_count = 0
        self._active_left_chain_length = 0
        self._active_right_chain_length = 0
        self._active_pair_count = 0
        self._active_unknown_pair_count = 0
        self._active_filtered_track_width_m = self._filtered_track_width_m
        self._active_held_path_flag = 0

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
            "midpoint_planner_node ready "
            f"cones={self.tracked_cones_topic} odom={self.odom_topic} "
            f"cmd={self.cmd_topic} path={self.centerline_topic} viz={self.viz_topic} "
            f"planning_frame={self.planning_frame} controller={self.controller_type}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "frames.planning_frame": "odom",
            "frames.odom_frame": "odom",
            "frames.base_frame": "front_axle",
            "frames.tf_timeout_s": 0.03,
            "topics.tracked_cones_topic": "/tracked_cones",
            "topics.cmd_topic": "/cmd",
            "topics.centerline_topic": "/planned_centerline",
            "topics.viz_topic": "/planner_viz",
            "topics.points_topic": "/planned_centerline_points",
            "topics.odom_topic": "/sim/odom",
            "filtering.max_cone_range_m": 25.0,
            "filtering.behind_drop_m": 5.0,
            "filtering.min_confidence": 0.3,
            "filtering.min_required_cones": 4,
            "filtering.infer_unknown_by_side": True,
            "filtering.infer_orange_by_side": True,
            "filtering.orange_min_lateral_m": 0.9,
            "filtering.orange_neighbor_radius_m": 3.5,
            "filtering.orange_neighbor_margin_m": 0.75,
            "filtering.allow_unknown_pair_completion": True,
            "filtering.unknown_pair_search_radius_m": 1.25,
            "filtering.unknown_pair_max_longitudinal_error_m": 1.5,
            "filtering.unknown_pair_max_width_error_m": 0.9,
            "filtering.max_consecutive_unknown_pairs": 2,
            "boundary_chain.min_step_m": 0.8,
            "boundary_chain.max_step_m": 6.0,
            "boundary_chain.max_heading_change_rad": 1.0,
            "boundary_chain.min_forward_progress_m": 0.2,
            "boundary_chain.min_chain_length": 3,
            "pairing.min_pair_width_m": 2.2,
            "pairing.max_pair_width_m": 5.5,
            "pairing.max_width_jump_m": 0.8,
            "pairing.min_pair_count": 3,
            "pairing.pair_hold_time_s": 1.25,
            "pairing.pair_reassignment_margin": 0.25,
            "width_estimation.initial_width_m": 3.6,
            "width_estimation.min_width_m": 2.4,
            "width_estimation.max_width_m": 4.8,
            "width_estimation.alpha": 0.15,
            "width_estimation.max_delta_per_update_m": 0.2,
            "width_estimation.min_trustworthy_pairs": 3,
            "centerline.path_resolution_m": 0.5,
            "centerline.max_path_length_m": 30.0,
            "centerline.smoothing_window": 3,
            "centerline.temporal_alpha": 0.25,
            "centerline.max_heading_delta_rad": 0.75,
            "midline_memory.horizon_m": 30.0,
            "midline_memory.station_spacing_m": 0.5,
            "midline_memory.near_distance_m": 4.0,
            "midline_memory.mid_distance_m": 12.0,
            "midline_memory.control_handoff_distance_m": 1.5,
            "midline_memory.near_alpha": 0.06,
            "midline_memory.mid_alpha": 0.18,
            "midline_memory.far_alpha": 0.35,
            "midline_memory.near_max_lateral_shift_m": 0.10,
            "midline_memory.mid_max_lateral_shift_m": 0.20,
            "midline_memory.far_max_lateral_shift_m": 0.40,
            "midline_memory.hold_last_valid_duration_s": 2.5,
            "midline_memory.min_buffer_confidence": 0.20,
            "runtime.publish_rate_hz": 180.0,
            "runtime.log_throttle_s": 1.0,
            "control.controller_type": "stanley",
            "control.stop_if_no_path": True,
            "stanley.k_gain": 1.2,
            "stanley.softening_speed_mps": 0.0,
            "stanley.heading_gain": 1.0,
            "stanley.lookahead_idx_offset": 0,
            "stanley.steering_limit_rad": 0.52,
            "stanley.steering_lowpass_alpha": 1.0,
            "stanley.steering_rate_limit_rad_s": 10.0,
            "stanley.use_yaw_rate_damping": True,
            "stanley.yaw_rate_damping_gain": 0.0,
            "stanley.wheelbase_m": 1.65,
            "stanley.cross_track_deadband_m": 0.0,
            "speed_control.speed_min_mps": 1.0,
            "speed_control.speed_max_mps": 1.8,
            "speed_control.curvature_speed_gain": 4.0,
            "speed_control.lowpass_speed_alpha": 0.15,
            "validation.min_path_points": 4,
            "validation.min_forward_extent_m": 2.0,
            "validation.jump_check_horizon_m": 8.0,
            "validation.max_near_field_lateral_jump_m": 0.6,
            "validation.max_near_field_lateral_jump_m_sparse_pairs": 0.9,
            "validation.max_start_heading_error_rad": 1.0,
            "validation.hold_last_valid_s": 2.5,
            "validation.hold_exit_clean_frames": 2,
            "validation.candidate_jump_reject_threshold_m": 1.0,
            "validation.candidate_jump_recover_frames": 3,
            "validation.candidate_min_points": 4,
            "validation.candidate_min_extent_m": 2.0,
            "diagnostics.topic": "/midpoint_planner/diagnostics",
            "diagnostics.centerline_jump_horizon_m": 8.0,
            "diagnostics.edge_quantization_m": 0.05,
            "diagnostics.jump_warn_threshold_m": 0.8,
            "diagnostics.edge_churn_warn_threshold": 0.55,
            "diagnostics.publish_control_debug": True,
            "diagnostics.publish_thesis_context": False,
            "debug.enable_markers": True,
            "debug.show_raw_cones": False,
            "debug.show_boundary_chains": True,
            "debug.show_pair_lines": True,
            "debug.show_raw_midpoint_chain": True,
            "debug.show_raw_offset_path": True,
            "debug.show_raw_prevalidation_centerline": True,
            "debug.publish_points_topic": False,
            "debug.show_lookahead_point": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        self.planning_frame = str(self.get_parameter("frames.planning_frame").value).strip() or "odom"
        self.odom_frame = str(self.get_parameter("frames.odom_frame").value).strip() or "odom"
        self.base_frame = str(self.get_parameter("frames.base_frame").value).strip() or "front_axle"
        self.tf_timeout_s = max(0.0, float(self.get_parameter("frames.tf_timeout_s").value))

        self.tracked_cones_topic = str(self.get_parameter("topics.tracked_cones_topic").value)
        self.cmd_topic = str(self.get_parameter("topics.cmd_topic").value)
        self.centerline_topic = str(self.get_parameter("topics.centerline_topic").value)
        self.viz_topic = str(self.get_parameter("topics.viz_topic").value)
        self.points_topic = str(self.get_parameter("topics.points_topic").value)
        self.odom_topic = str(self.get_parameter("topics.odom_topic").value)
        self.infer_unknown_by_side = bool(self.get_parameter("filtering.infer_unknown_by_side").value)
        self.infer_orange_by_side = bool(self.get_parameter("filtering.infer_orange_by_side").value)
        self.orange_min_lateral_m = float(self.get_parameter("filtering.orange_min_lateral_m").value)
        self.orange_neighbor_radius_m = float(
            self.get_parameter("filtering.orange_neighbor_radius_m").value
        )
        self.orange_neighbor_margin_m = float(
            self.get_parameter("filtering.orange_neighbor_margin_m").value
        )

        self.centerline_path_resolution_m = max(
            0.05,
            float(self.get_parameter("centerline.path_resolution_m").value),
        )
        self.temporal_alpha = float(
            np.clip(float(self.get_parameter("centerline.temporal_alpha").value), 0.0, 1.0)
        )
        # The hybrid planner now owns temporal stability through the persistent
        # midline buffer. Keep the legacy parameter readable for compatibility,
        # but do not stack a second whole-path smoother on top of it.
        self.enable_temporal_smoothing = False
        self.smoothing_alpha = self.temporal_alpha
        self.enable_near_field_freeze = False
        self.freeze_near_field_m = 0.0
        self.freeze_blend_length_m = 0.0
        self.enable_committed_near_field = False
        self.commit_plan_horizon_m = 0.0
        self.commit_stable_frames = 1
        self.commit_update_max_churn_ratio = 1.0
        self.pair_hold_time_s = max(0.0, float(self.get_parameter("pairing.pair_hold_time_s").value))
        self.midline_horizon_m = max(1.0, float(self.get_parameter("midline_memory.horizon_m").value))
        self.midline_station_spacing_m = max(
            0.05,
            float(self.get_parameter("midline_memory.station_spacing_m").value),
        )
        self.midline_near_distance_m = max(
            0.0,
            float(self.get_parameter("midline_memory.near_distance_m").value),
        )
        self.midline_mid_distance_m = max(
            self.midline_near_distance_m,
            float(self.get_parameter("midline_memory.mid_distance_m").value),
        )
        self.midline_control_handoff_distance_m = max(
            self.midline_station_spacing_m,
            float(self.get_parameter("midline_memory.control_handoff_distance_m").value),
        )
        self.midline_near_alpha = float(
            np.clip(float(self.get_parameter("midline_memory.near_alpha").value), 0.0, 1.0)
        )
        self.midline_mid_alpha = float(
            np.clip(float(self.get_parameter("midline_memory.mid_alpha").value), 0.0, 1.0)
        )
        self.midline_far_alpha = float(
            np.clip(float(self.get_parameter("midline_memory.far_alpha").value), 0.0, 1.0)
        )
        self.midline_near_max_shift_m = max(
            0.0,
            float(self.get_parameter("midline_memory.near_max_lateral_shift_m").value),
        )
        self.midline_mid_max_shift_m = max(
            self.midline_near_max_shift_m,
            float(self.get_parameter("midline_memory.mid_max_lateral_shift_m").value),
        )
        self.midline_far_max_shift_m = max(
            self.midline_mid_max_shift_m,
            float(self.get_parameter("midline_memory.far_max_lateral_shift_m").value),
        )
        self.midline_min_buffer_confidence = float(
            np.clip(float(self.get_parameter("midline_memory.min_buffer_confidence").value), 0.0, 1.0)
        )
        self.midline_hold_last_valid_duration_s = max(
            0.0,
            float(self.get_parameter("midline_memory.hold_last_valid_duration_s").value),
        )

        self.publish_rate_hz = max(1.0, float(self.get_parameter("runtime.publish_rate_hz").value))
        self.log_throttle_s = max(0.1, float(self.get_parameter("runtime.log_throttle_s").value))

        self.controller_type = (
            str(self.get_parameter("control.controller_type").value).strip().lower() or "stanley"
        )
        if self.controller_type not in {"stanley", "pure_pursuit", "none"}:
            raise ValueError(
                "Unsupported control.controller_type '%s'. Supported values: stanley, pure_pursuit, none"
                % self.controller_type
            )
        stanley_config = StanleyConfig(
            k_gain=max(0.0, float(self.get_parameter("stanley.k_gain").value)),
            softening_speed_mps=max(0.0, float(self.get_parameter("stanley.softening_speed_mps").value)),
            heading_gain=float(self.get_parameter("stanley.heading_gain").value),
            lookahead_idx_offset=max(0, int(self.get_parameter("stanley.lookahead_idx_offset").value)),
            steering_limit_rad=max(0.01, float(self.get_parameter("stanley.steering_limit_rad").value)),
            steering_lowpass_alpha=float(
                np.clip(float(self.get_parameter("stanley.steering_lowpass_alpha").value), 0.0, 1.0)
            ),
            steering_rate_limit_rad_s=max(
                0.0,
                float(self.get_parameter("stanley.steering_rate_limit_rad_s").value),
            ),
            use_yaw_rate_damping=bool(self.get_parameter("stanley.use_yaw_rate_damping").value),
            yaw_rate_damping_gain=max(
                0.0,
                float(self.get_parameter("stanley.yaw_rate_damping_gain").value),
            ),
            wheelbase_m=max(0.1, float(self.get_parameter("stanley.wheelbase_m").value)),
            cross_track_deadband_m=max(
                0.0,
                float(self.get_parameter("stanley.cross_track_deadband_m").value),
            ),
        )
        self._controller = (
            create_steering_controller(
                controller_type=self.controller_type,
                stanley_config=stanley_config,
                publish_rate_hz=self.publish_rate_hz,
            )
            if self.controller_type == "stanley"
            else None
        )
        self.stop_if_no_path = bool(self.get_parameter("control.stop_if_no_path").value)
        if self.controller_type == "pure_pursuit":
            self.get_logger().warn(
                "control.controller_type 'pure_pursuit' is a placeholder; controller output is disabled"
            )
        elif self.controller_type == "none":
            self.get_logger().info("control.controller_type 'none'; controller output is disabled")

        self.speed_min_mps = max(0.0, float(self.get_parameter("speed_control.speed_min_mps").value))
        self.speed_max_mps = max(self.speed_min_mps, float(self.get_parameter("speed_control.speed_max_mps").value))
        self.curvature_speed_gain = max(0.0, float(self.get_parameter("speed_control.curvature_speed_gain").value))
        self.lowpass_speed_alpha = float(
            np.clip(float(self.get_parameter("speed_control.lowpass_speed_alpha").value), 0.0, 1.0)
        )
        self.hold_last_valid_s = max(
            self.midline_hold_last_valid_duration_s,
            float(self.get_parameter("validation.hold_last_valid_s").value),
        )
        self.hold_exit_clean_frames = max(
            1,
            int(self.get_parameter("validation.hold_exit_clean_frames").value),
        )
        self.candidate_jump_reject_threshold_m = max(
            0.0,
            float(self.get_parameter("validation.candidate_jump_reject_threshold_m").value),
        )
        self.candidate_jump_recover_frames = max(
            1,
            int(self.get_parameter("validation.candidate_jump_recover_frames").value),
        )
        self.candidate_min_points = max(
            2,
            int(self.get_parameter("validation.candidate_min_points").value),
        )
        self.candidate_min_extent_m = max(
            0.5,
            float(self.get_parameter("validation.candidate_min_extent_m").value),
        )
        self.diagnostics_topic = (
            str(self.get_parameter("diagnostics.topic").value).strip()
            or self._planner_identity.diagnostics_topic
        )
        self.centerline_jump_horizon_m = max(
            0.5,
            float(self.get_parameter("diagnostics.centerline_jump_horizon_m").value),
        )
        self.edge_quantization_m = max(
            1e-6,
            float(self.get_parameter("diagnostics.edge_quantization_m").value),
        )
        self.jump_warn_threshold_m = max(
            0.0,
            float(self.get_parameter("diagnostics.jump_warn_threshold_m").value),
        )
        self.edge_churn_warn_threshold = max(
            0.0,
            float(self.get_parameter("diagnostics.edge_churn_warn_threshold").value),
        )
        self.publish_control_debug = bool(
            self.get_parameter("diagnostics.publish_control_debug").value
        )
        self.publish_thesis_context = bool(
            self.get_parameter("diagnostics.publish_thesis_context").value
        )

        self.enable_debug_markers = bool(self.get_parameter("debug.enable_markers").value)
        self.show_raw_cones = bool(self.get_parameter("debug.show_raw_cones").value)
        self.show_boundary_chains = bool(self.get_parameter("debug.show_boundary_chains").value)
        self.show_pair_lines = bool(self.get_parameter("debug.show_pair_lines").value)
        self.show_raw_midpoint_chain = bool(
            self.get_parameter("debug.show_raw_midpoint_chain").value
        )
        self.show_raw_offset_path = bool(
            self.get_parameter("debug.show_raw_offset_path").value
        )
        self.show_raw_prevalidation_centerline = bool(
            self.get_parameter("debug.show_raw_prevalidation_centerline").value
        )
        self.publish_points_topic = bool(self.get_parameter("debug.publish_points_topic").value)
        self.show_lookahead_point = bool(self.get_parameter("debug.show_lookahead_point").value)
        self.show_delaunay_edges = False
        self.show_candidate_edges = False
        self.show_selected_edges = False

        self._core_config = MidpointPlannerConfig(
            max_cone_range_m=float(self.get_parameter("filtering.max_cone_range_m").value),
            behind_drop_m=float(self.get_parameter("filtering.behind_drop_m").value),
            min_confidence=float(self.get_parameter("filtering.min_confidence").value),
            min_required_cones=max(2, int(self.get_parameter("filtering.min_required_cones").value)),
            allow_unknown_pair_completion=bool(
                self.get_parameter("filtering.allow_unknown_pair_completion").value
            ),
            unknown_pair_search_radius_m=float(
                self.get_parameter("filtering.unknown_pair_search_radius_m").value
            ),
            unknown_pair_max_longitudinal_error_m=float(
                self.get_parameter("filtering.unknown_pair_max_longitudinal_error_m").value
            ),
            unknown_pair_max_width_error_m=float(
                self.get_parameter("filtering.unknown_pair_max_width_error_m").value
            ),
            max_consecutive_unknown_pairs=max(
                0,
                int(self.get_parameter("filtering.max_consecutive_unknown_pairs").value),
            ),
            min_step_m=float(self.get_parameter("boundary_chain.min_step_m").value),
            max_step_m=float(self.get_parameter("boundary_chain.max_step_m").value),
            max_heading_change_rad=float(self.get_parameter("boundary_chain.max_heading_change_rad").value),
            min_forward_progress_m=float(
                self.get_parameter("boundary_chain.min_forward_progress_m").value
            ),
            min_chain_length=max(2, int(self.get_parameter("boundary_chain.min_chain_length").value)),
            min_pair_width_m=float(self.get_parameter("pairing.min_pair_width_m").value),
            max_pair_width_m=float(self.get_parameter("pairing.max_pair_width_m").value),
            max_width_jump_m=float(self.get_parameter("pairing.max_width_jump_m").value),
            min_pair_count=max(1, int(self.get_parameter("pairing.min_pair_count").value)),
            pair_reassignment_margin=float(
                self.get_parameter("pairing.pair_reassignment_margin").value
            ),
            initial_width_m=float(self.get_parameter("width_estimation.initial_width_m").value),
            min_width_m=float(self.get_parameter("width_estimation.min_width_m").value),
            max_width_m=float(self.get_parameter("width_estimation.max_width_m").value),
            width_filter_alpha=float(self.get_parameter("width_estimation.alpha").value),
            max_width_delta_per_update_m=float(
                self.get_parameter("width_estimation.max_delta_per_update_m").value
            ),
            min_trustworthy_pairs=max(
                1,
                int(self.get_parameter("width_estimation.min_trustworthy_pairs").value),
            ),
            path_resolution_m=float(self.get_parameter("centerline.path_resolution_m").value),
            max_path_length_m=float(self.get_parameter("centerline.max_path_length_m").value),
            smoothing_window=max(1, int(self.get_parameter("centerline.smoothing_window").value)),
            max_heading_delta_rad=float(self.get_parameter("centerline.max_heading_delta_rad").value),
            min_path_points=max(2, int(self.get_parameter("validation.min_path_points").value)),
            min_forward_extent_m=float(self.get_parameter("validation.min_forward_extent_m").value),
            jump_check_horizon_m=float(self.get_parameter("validation.jump_check_horizon_m").value),
            max_near_field_lateral_jump_m=float(
                self.get_parameter("validation.max_near_field_lateral_jump_m").value
            ),
            max_near_field_lateral_jump_m_sparse_pairs=float(
                self.get_parameter("validation.max_near_field_lateral_jump_m_sparse_pairs").value
            ),
            max_start_heading_error_rad=float(
                self.get_parameter("validation.max_start_heading_error_rad").value
            ),
        )
        self._filtered_track_width_m = float(self._core_config.initial_width_m)

    def _on_timer(self) -> None:
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
                status="waiting for /tracked_cones",
                operator_state="waiting",
                operator_reason="waiting_for_cones",
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
                status="cone transform unavailable",
                operator_state="waiting",
                operator_reason="cone_transform_unavailable",
                cmd_speed=cmd_speed,
                cmd_steering=cmd_steering,
                lookahead=lookahead,
                zero_cmd_sent_flag=zero_cmd_sent,
            )
            return

        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        track_ids, track_states, track_confidences = self._extract_cone_metadata(cones_msg)
        self._active_remembered_cone_count = int(len(cones_msg.cones))
        self._active_stale_cone_count = int(np.count_nonzero(track_states == MSG_TRACK_STATE_STALE))
        planner_confidences = np.array(confidences, copy=True)
        valid_track_conf_mask = track_confidences > 1e-6
        planner_confidences[valid_track_conf_mask] = track_confidences[valid_track_conf_mask]
        planner_confidences[
            (track_ids > 0)
            & (track_states == MSG_TRACK_STATE_TENTATIVE)
        ] = 0.0
        remembered_pair_entries = self._active_pair_memory_entries(
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )

        result = compute_midpoint_centerline(
            points_xy=points_xy,
            colors=colors,
            confidences=planner_confidences,
            track_ids=track_ids,
            vehicle_xy=(vehicle_x, vehicle_y),
            vehicle_yaw=vehicle_yaw,
            config=self._core_config,
            prior=MidpointPlannerPrior(
                previous_centerline=(
                    np.array(self._midline_buffer_path, copy=True)
                    if self._midline_buffer_path is not None
                    else (
                        np.array(self._last_valid_centerline, copy=True)
                        if self._last_valid_centerline is not None
                        else None
                )
                ),
                previous_width_m=self._filtered_track_width_m,
                previous_mode=self._active_planner_mode,
                previous_pairs=[
                    (entry.left_track_id, entry.right_track_id)
                    for entry in remembered_pair_entries
                ],
            ),
        )
        if (
            result.accepted_pair_count >= int(self._core_config.min_trustworthy_pairs)
            and math.isfinite(float(result.selected_chain_width_median))
        ):
            self._filtered_track_width_m = update_track_width_estimate(
                self._filtered_track_width_m,
                result.selected_chain_width_median,
                self._core_config,
            )
        result.filtered_track_width_m = float(self._filtered_track_width_m)

        raw_centerline, candidate_source = self._select_candidate_centerline(result)
        raw_midpoint_chain = np.array(result.midpoints_raw, copy=True)
        pair_segments_for_viz = np.array(result.pair_segments, copy=True)
        live_pair_entries = self._pair_entries_from_segments(
            pair_track_ids=result.selected_pair_track_ids,
            pair_segments=result.pair_segments,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        combined_pair_entries = self._merge_pair_entries(
            remembered_entries=remembered_pair_entries,
            live_entries=live_pair_entries,
        )
        combined_pair_segments, combined_midpoint_chain = self._pair_geometry_from_memory(
            combined_pair_entries
        )
        if combined_pair_segments.size > 0:
            pair_segments_for_viz = combined_pair_segments
        if combined_midpoint_chain.size > 0:
            raw_midpoint_chain = combined_midpoint_chain
        projected_pair_candidate = self._project_midpoint_chain_candidate(
            midpoint_chain=raw_midpoint_chain,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        if projected_pair_candidate.shape[0] > 0 and (
            raw_centerline.shape[0] == 0
            or self._candidate_forward_extent_m(
                centerline=raw_centerline,
                frame_id=target_frame,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
            ) < self._minimum_projected_forward_extent_m()
        ):
            raw_centerline = projected_pair_candidate
            candidate_source = "projected_pairs"
        result.planner_mode = self._planner_identity.planner_mode

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

        centerline_jump_max_m = compute_centerline_jump_max(
            raw_centerline,
            self._previous_raw_centerline,
            self.centerline_jump_horizon_m,
        )
        self._previous_raw_centerline = (
            np.array(raw_centerline, copy=True) if raw_centerline.shape[0] > 0 else None
        )

        if centerline_jump_max_m > self.jump_warn_threshold_m:
            self._warn_throttled(
                "centerline_jump_warn",
                (
                    f"centerline jump {centerline_jump_max_m:.3f} m exceeded threshold "
                    f"{self.jump_warn_threshold_m:.3f} m"
                ),
            )
        if selected_edge_churn > self.edge_churn_warn_threshold:
            self._warn_throttled(
                "edge_churn_warn",
                (
                    f"selected pair churn {selected_edge_churn:.3f} exceeded threshold "
                    f"{self.edge_churn_warn_threshold:.3f}"
                ),
            )

        candidate_update_ok, candidate_update_reason = self._candidate_path_is_updateable(
            candidate_centerline=raw_centerline,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
            result=result,
            candidate_source=candidate_source,
        )
        candidate_update_ok, candidate_update_reason = self._update_candidate_jump_reject_streak(
            candidate_update_ok=candidate_update_ok,
            candidate_update_reason=candidate_update_reason,
        )
        centerline = self._update_midline_buffer(
            candidate_centerline=raw_centerline,
            candidate_source=candidate_source,
            candidate_update_ok=candidate_update_ok,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
            result=result,
            now_sec=now_sec,
        )
        centerline = self._anchor_centerline_near_vehicle(
            centerline=centerline,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        status = result.status
        if not candidate_update_ok and centerline.shape[0] > 0:
            status = f"{status}; holding stored midline"
        elif (
            raw_centerline.shape[0] > 0
            and centerline.shape[0] > 0
            and (
                raw_centerline.shape != centerline.shape
                or not np.allclose(raw_centerline, centerline)
            )
        ):
            status = f"{status}; publishing stored midline"
        if candidate_source != "validated" and raw_centerline.shape[0] > 0:
            status = f"{status}; using {candidate_source}"
        if centerline.shape[0] > 0 and raw_centerline.shape[0] > 0:
            self._remember_pairs(
                result=result,
                frame_id=target_frame,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
                track_ids=track_ids,
                track_states=track_states,
                planner_confidences=planner_confidences,
            )
            self._last_valid_pair_segments = (
                np.array(pair_segments_for_viz, copy=True)
                if pair_segments_for_viz.size > 0
                else self._last_valid_pair_segments
            )
            self._last_valid_pair_track_ids = (
                np.array(result.selected_pair_track_ids, copy=True)
                if result.selected_pair_track_ids.size > 0
                else self._last_valid_pair_track_ids
            )

        plan_valid = False
        plan_hold_active = False
        publish_mode = "fresh"
        hold_reason = result.reject_reason
        zero_cmd_sent_flag = 0
        controller_failed = False

        if centerline.shape[0] > 0 and candidate_update_ok:
            plan_valid = True
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

        if publish_mode == "held" and centerline.shape[0] > 0:
            held_pair_segments, held_raw_midpoint_chain = self._held_pair_geometry(now_sec=now_sec)
            if pair_segments_for_viz.size == 0 and held_pair_segments is not None:
                pair_segments_for_viz = held_pair_segments
            if raw_midpoint_chain.size == 0 and held_raw_midpoint_chain is not None:
                raw_midpoint_chain = held_raw_midpoint_chain

        centerline = self._prepare_centerline_for_current_pose(
            centerline=centerline,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )

        if self.enable_temporal_smoothing and publish_mode == "fresh":
            centerline = self._apply_temporal_smoothing(centerline)
            self._previous_centerline = np.array(centerline, copy=True)
        elif publish_mode == "held" and centerline.shape[0] > 0:
            self._previous_centerline = np.array(centerline, copy=True)

        if publish_mode == "fresh" and centerline.shape[0] > 0:
            self._record_valid_plan(
                now_sec=now_sec,
                centerline=centerline,
                raw_midpoint_chain=raw_midpoint_chain,
                selected_chain_width_median=result.selected_chain_width_median,
            )
            if pair_segments_for_viz.size > 0:
                self._last_valid_pair_segments = np.array(pair_segments_for_viz, copy=True)
            if result.selected_pair_track_ids.size > 0:
                self._last_valid_pair_track_ids = np.array(result.selected_pair_track_ids, copy=True)

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
                self._warn_throttled("controller_compute_error", f"controller compute failed: {exc}")
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
                        "heading_error_rad": float(debug.heading_error_rad),
                        "cross_track_error_m": float(debug.cross_track_error_m),
                        "vehicle_speed_mps": float(debug.vehicle_speed_mps),
                        "speed_term_mps": float(debug.speed_term_mps),
                        "heading_contribution_rad": float(debug.heading_contribution_rad),
                        "cross_track_contribution_rad": float(debug.cross_track_contribution_rad),
                        "yaw_rate_damping_contribution_rad": float(
                            debug.yaw_rate_damping_contribution_rad
                        ),
                        "yaw_rate_rps": float(self._latest_yaw_rate_rps),
                        "raw_steering_cmd_rad": float(debug.raw_steering_cmd_rad),
                        "steering_after_clamp_rad": float(debug.steering_after_clamp_rad),
                        "steering_after_filter_rad": float(debug.steering_after_filter_rad),
                        "steering_after_rate_limit_rad": float(debug.steering_after_rate_limit_rad),
                        "final_steering_cmd_rad": float(debug.final_steering_cmd_rad),
                        "steering_saturated_flag": (
                            1.0 if bool(debug.steering_saturated_flag) else 0.0
                        ),
                        "nearest_path_index": float(debug.nearest_path_index),
                        "heading_path_index": float(debug.heading_path_index),
                        "target_point_x_base_m": float(debug.target_point_x_base_m),
                        "target_point_y_base_m": float(debug.target_point_y_base_m),
                        "target_point_x_frame_m": (
                            float(control_target_frame[0])
                            if control_target_frame is not None
                            else float("nan")
                        ),
                        "target_point_y_frame_m": (
                            float(control_target_frame[1])
                            if control_target_frame is not None
                            else float("nan")
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
        operator_state = "fresh"
        operator_reason = "none"
        core_reject_reason = self._normalize_core_reject_reason(result)
        if centerline.shape[0] == 0:
            if self._last_valid_centerline is not None and hold_remaining_s <= 0.0:
                operator_state = "stopped"
                operator_reason = "hold_expired_no_path"
            elif core_reject_reason != "none":
                operator_state = "stopped"
                operator_reason = core_reject_reason
            elif zero_cmd_sent_flag:
                operator_state = "stopped"
                operator_reason = "stop_if_no_path"
            else:
                operator_state = "held"
                operator_reason = "holding_previous_valid"
        elif publish_mode == "held":
            operator_state = "held"
            if centerline.shape[0] > 0 and "hysteresis holding previous valid centerline" in status:
                operator_reason = "hysteresis_holding"
            elif core_reject_reason != "none":
                operator_reason = core_reject_reason
            else:
                operator_reason = "holding_previous_valid"
        else:
            operator_state = "fresh"
            operator_reason = "none"

        if controller_failed:
            operator_state = "stopped"
            operator_reason = "controller_compute_failed"
        elif self._controller is None and centerline.shape[0] > 0:
            operator_state = "stopped"
            operator_reason = "controller_disabled"
        elif centerline.shape[0] > 0 and control_path_point_count <= 0 and operator_state != "waiting":
            operator_state = "stopped"
            operator_reason = "no_control_path"
        if zero_cmd_sent_flag and operator_reason == "none":
            operator_state = "stopped"
            operator_reason = "stop_if_no_path"
        if (
            control_path_point_count > 0
            and centerline.shape[0] > 0
            and zero_cmd_sent_flag == 0
            and publish_mode == "fresh"
        ):
            operator_state = "fresh"

        self._active_planner_mode = (
            "holding_last_valid"
            if publish_mode == "held" and centerline.shape[0] > 0
            else result.planner_mode
        )
        self._active_left_chain_length = int(result.left_chain_length)
        self._active_right_chain_length = int(result.right_chain_length)
        self._active_pair_count = self._published_pair_count(pair_segments_for_viz, result)
        self._active_unknown_pair_count = int(result.unknown_pair_count)
        self._active_filtered_track_width_m = float(self._filtered_track_width_m)
        self._active_held_path_flag = 1 if publish_mode == "held" and centerline.shape[0] > 0 else 0

        self._log_operator_state_transition(
            operator_state=operator_state,
            operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            selected_chain_length=int(result.selected_chain_length),
        )
        self._log_mode_summary(
            mode=self._active_planner_mode,
            result=result,
            operator_state=operator_state,
            operator_reason=operator_reason,
            hold_active=bool(self._active_held_path_flag),
        )

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
                "plan_valid_flag": 1.0 if plan_valid else 0.0,
                "plan_hold_active_flag": 1.0 if plan_hold_active else 0.0,
                "plan_fallback_flag": 1.0 if bool(result.used_fallback) else 0.0,
                "path_length_m": self._path_length_m(centerline),
                "path_curvature_abs_p95_1pm": self._path_curvature_abs_p95(centerline),
            },
            planner_metrics={
                "candidate_diagonal_count": result.candidate_count,
                "selected_chain_length": result.selected_chain_length,
                "selected_chain_median_width_m": result.selected_chain_width_median,
                "expected_width_prior_m": result.expected_width_prior_m,
                "reject_wrong_side_count": result.reject_counts.get("wrong_side", 0),
                "reject_width_count": result.reject_counts.get("width", 0),
                "reject_width_range_count": result.reject_counts.get("width_range", 0),
                "reject_width_prior_count": result.reject_counts.get("width_prior", 0),
                "reject_orientation_count": result.reject_counts.get("orientation", 0),
                "reject_progress_count": result.reject_counts.get("progress", 0),
                "reject_near_field_continuity_count": result.reject_counts.get("near_field_continuity", 0),
                "reject_midpoint_kink_count": result.reject_counts.get("midpoint_kink", 0),
                "reject_seed_distance_count": result.reject_counts.get("seed_distance", 0),
                "near_field_lateral_max_m": result.near_field_lateral_max_m,
                "near_field_lateral_mean_m": result.near_field_lateral_mean_m,
                "near_field_displacement_max_m": result.near_field_displacement_max_m,
                "near_field_displacement_mean_m": result.near_field_displacement_mean_m,
                "near_field_midpoint_kink_max_rad": result.near_field_kink_max_rad,
                "seed_midpoint_distance_m": result.seed_midpoint_distance_m,
                "seed_temporal_offset_m": result.seed_temporal_offset_m,
                "selected_chain_churn_count": selected_edge_churn_count,
                "selected_chain_churn_ratio": selected_edge_churn,
                "publish_mode": publish_mode,
                "hold_mode_active_flag": 1 if self._hold_mode_active else 0,
                "hold_clean_frame_count": self._hold_clean_frame_count,
                "hold_reason": hold_reason or "",
                "planner_state_code": self._operator_state_code(operator_state),
                "fresh_publish_flag": 1 if operator_state == "fresh" else 0,
                "held_publish_flag": 1 if operator_state == "held" else 0,
                "stopped_flag": 1 if operator_state == "stopped" else 0,
                "waiting_flag": 1 if operator_state == "waiting" else 0,
                "operator_reason_code": self._operator_reason_code(operator_reason),
                "operator_reason": operator_reason,
                "hold_remaining_s": hold_remaining_s,
                "control_path_point_count": control_path_point_count,
                "zero_cmd_sent_flag": zero_cmd_sent_flag,
                "planner_mode": self._active_planner_mode,
                "remembered_cone_count": self._active_remembered_cone_count,
                "remembered_stale_cone_count": self._active_stale_cone_count,
                "left_chain_length": result.left_chain_length,
                "right_chain_length": result.right_chain_length,
                "accepted_pair_count": result.accepted_pair_count,
                "unknown_pair_count": result.unknown_pair_count,
                "filtered_track_width_m": self._filtered_track_width_m,
                "held_path_flag": self._active_held_path_flag,
                "raw_candidate_point_count": int(raw_centerline.shape[0]),
            },
        )

        self._current_pair_segments_for_viz = np.array(pair_segments_for_viz, copy=True)
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

    @staticmethod
    def _extract_cone_metadata(msg: ConeDetectionArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return (
            np.asarray(track_ids, dtype=np.int64),
            np.asarray(track_states, dtype=np.int64),
            np.asarray(track_confidences, dtype=np.float64),
        )

    def _active_pair_memory_entries(
        self,
        *,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[_PairMemoryEntry]:
        keep: list[_PairMemoryEntry] = []
        for entry in self._pair_memory:
            midpoint_x_base, midpoint_y_base = self._odom_point_to_base(
                entry.midpoint_x_odom,
                entry.midpoint_y_odom,
                vehicle_x,
                vehicle_y,
                vehicle_yaw,
            )
            del midpoint_y_base
            if midpoint_x_base >= -_PAIR_PASSED_MARGIN_M:
                keep.append(entry)
        self._pair_memory = keep
        return keep

    def _active_pair_memory(
        self,
        *,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[tuple[int, int]]:
        return [
            (entry.left_track_id, entry.right_track_id)
            for entry in self._active_pair_memory_entries(
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
            )
        ]

    @staticmethod
    def _pair_geometry_from_memory(
        entries: list[_PairMemoryEntry],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not entries:
            return (
                np.empty((0, 2, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
            )
        pair_segments = np.empty((len(entries), 2, 2), dtype=np.float64)
        midpoint_chain = np.empty((len(entries), 2), dtype=np.float64)
        for idx, entry in enumerate(entries):
            pair_segments[idx, 0, 0] = float(entry.left_x_odom)
            pair_segments[idx, 0, 1] = float(entry.left_y_odom)
            pair_segments[idx, 1, 0] = float(entry.right_x_odom)
            pair_segments[idx, 1, 1] = float(entry.right_y_odom)
            midpoint_chain[idx, 0] = float(entry.midpoint_x_odom)
            midpoint_chain[idx, 1] = float(entry.midpoint_y_odom)
        return pair_segments, midpoint_chain

    def _pair_entries_from_segments(
        self,
        *,
        pair_track_ids: np.ndarray,
        pair_segments: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[_PairMemoryEntry]:
        if pair_track_ids.size == 0 or pair_segments.size == 0:
            return []
        entries: list[_PairMemoryEntry] = []
        for pair_ids, pair_segment in zip(
            np.asarray(pair_track_ids, dtype=np.int64),
            np.asarray(pair_segments, dtype=np.float64),
        ):
            if pair_segment.shape != (2, 2):
                continue
            left_point = np.asarray(pair_segment[0], dtype=np.float64)
            right_point = np.asarray(pair_segment[1], dtype=np.float64)
            midpoint = 0.5 * (left_point + right_point)
            left_x_odom = float(left_point[0])
            left_y_odom = float(left_point[1])
            right_x_odom = float(right_point[0])
            right_y_odom = float(right_point[1])
            midpoint_x_odom = float(midpoint[0])
            midpoint_y_odom = float(midpoint[1])
            if self._is_alias(frame_id, self.base_frame):
                left_x_odom, left_y_odom = self._base_point_to_odom(
                    left_x_odom,
                    left_y_odom,
                    vehicle_x,
                    vehicle_y,
                    vehicle_yaw,
                )
                right_x_odom, right_y_odom = self._base_point_to_odom(
                    right_x_odom,
                    right_y_odom,
                    vehicle_x,
                    vehicle_y,
                    vehicle_yaw,
                )
                midpoint_x_odom, midpoint_y_odom = self._base_point_to_odom(
                    midpoint_x_odom,
                    midpoint_y_odom,
                    vehicle_x,
                    vehicle_y,
                    vehicle_yaw,
                )
            elif not self._is_alias(frame_id, self.odom_frame):
                continue
            entries.append(
                _PairMemoryEntry(
                    left_track_id=int(pair_ids[0]),
                    right_track_id=int(pair_ids[1]),
                    midpoint_x_odom=float(midpoint_x_odom),
                    midpoint_y_odom=float(midpoint_y_odom),
                    left_x_odom=float(left_x_odom),
                    left_y_odom=float(left_y_odom),
                    right_x_odom=float(right_x_odom),
                    right_y_odom=float(right_y_odom),
                )
            )
        return entries

    @staticmethod
    def _merge_pair_entries(
        *,
        remembered_entries: list[_PairMemoryEntry],
        live_entries: list[_PairMemoryEntry],
    ) -> list[_PairMemoryEntry]:
        merged: dict[tuple[int, int], _PairMemoryEntry] = {}
        for entry in remembered_entries:
            merged[(entry.left_track_id, entry.right_track_id)] = entry
        for entry in live_entries:
            merged[(entry.left_track_id, entry.right_track_id)] = entry
        return list(merged.values())

    def _remember_pairs(
        self,
        *,
        result: MidpointPlannerResult,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        track_ids: np.ndarray,
        track_states: np.ndarray,
        planner_confidences: np.ndarray,
    ) -> None:
        if result.selected_pair_track_ids.size == 0 or result.pair_segments.size == 0:
            self._active_pair_count = 0
            return

        track_state_by_id = {
            int(track_id): int(track_state)
            for track_id, track_state in zip(track_ids, track_states)
        }
        del planner_confidences
        remembered_pairs: list[_PairMemoryEntry] = []
        for pair_ids, pair_segment in zip(
            np.asarray(result.selected_pair_track_ids, dtype=np.int64),
            np.asarray(result.pair_segments, dtype=np.float64),
        ):
            if pair_segment.shape != (2, 2):
                continue
            left_track_id = int(pair_ids[0])
            right_track_id = int(pair_ids[1])
            left_state = track_state_by_id.get(left_track_id, MSG_TRACK_STATE_TENTATIVE)
            right_state = track_state_by_id.get(right_track_id, MSG_TRACK_STATE_TENTATIVE)
            if left_state == MSG_TRACK_STATE_TENTATIVE or right_state == MSG_TRACK_STATE_TENTATIVE:
                continue
            left_point = np.asarray(pair_segment[0], dtype=np.float64)
            right_point = np.asarray(pair_segment[1], dtype=np.float64)
            midpoint = 0.5 * (left_point + right_point)
            left_x_odom = float(left_point[0])
            left_y_odom = float(left_point[1])
            right_x_odom = float(right_point[0])
            right_y_odom = float(right_point[1])
            midpoint_x_odom = float(midpoint[0])
            midpoint_y_odom = float(midpoint[1])
            if self._is_alias(frame_id, self.base_frame):
                left_x_odom, left_y_odom = self._base_point_to_odom(
                    left_x_odom,
                    left_y_odom,
                    vehicle_x,
                    vehicle_y,
                    vehicle_yaw,
                )
                right_x_odom, right_y_odom = self._base_point_to_odom(
                    right_x_odom,
                    right_y_odom,
                    vehicle_x,
                    vehicle_y,
                    vehicle_yaw,
                )
                midpoint_x_odom, midpoint_y_odom = self._base_point_to_odom(
                    midpoint_x_odom,
                    midpoint_y_odom,
                    vehicle_x,
                    vehicle_y,
                    vehicle_yaw,
                )
            elif not self._is_alias(frame_id, self.odom_frame):
                continue
            remembered_pairs.append(
                _PairMemoryEntry(
                    left_track_id=left_track_id,
                    right_track_id=right_track_id,
                    midpoint_x_odom=float(midpoint_x_odom),
                    midpoint_y_odom=float(midpoint_y_odom),
                    left_x_odom=float(left_x_odom),
                    left_y_odom=float(left_y_odom),
                    right_x_odom=float(right_x_odom),
                    right_y_odom=float(right_y_odom),
                )
            )
        self._pair_memory = remembered_pairs

    def _held_pair_geometry(
        self,
        *,
        now_sec: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._last_valid_time_sec < 0.0:
            return None, None
        if self._hold_remaining_s(now_sec) <= 0.0:
            return None, None
        pair_segments = (
            np.array(self._last_valid_pair_segments, copy=True)
            if self._last_valid_pair_segments is not None
            else None
        )
        raw_midpoint_chain = (
            np.array(self._last_valid_raw_midpoint_chain, copy=True)
            if self._last_valid_raw_midpoint_chain is not None
            else None
        )
        return pair_segments, raw_midpoint_chain

    @staticmethod
    def _published_pair_count(
        pair_segments_for_viz: np.ndarray,
        result: MidpointPlannerResult,
    ) -> int:
        if pair_segments_for_viz.size > 0:
            return int(pair_segments_for_viz.shape[0])
        return int(result.accepted_pair_count)

    def _update_candidate_jump_reject_streak(
        self,
        *,
        candidate_update_ok: bool,
        candidate_update_reason: str,
    ) -> tuple[bool, str]:
        if not candidate_update_ok and candidate_update_reason == "candidate_jump_rejected":
            self._candidate_jump_reject_streak += 1
        else:
            self._candidate_jump_reject_streak = 0
        return candidate_update_ok, candidate_update_reason

    def _apply_mode_hysteresis(self, candidate_mode: str) -> str:
        return "midpoint" if candidate_mode != "none" else "none"

    def _update_midline_buffer(
        self,
        *,
        candidate_centerline: np.ndarray,
        candidate_source: str,
        candidate_update_ok: Optional[bool],
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        result: MidpointPlannerResult,
        now_sec: float,
    ) -> np.ndarray:
        if not self._is_alias(frame_id, self.odom_frame):
            if candidate_centerline.shape[0] > 0:
                self._midline_buffer_path = np.array(candidate_centerline, copy=True)
                self._midline_buffer_last_update_sec = now_sec
                self._midline_buffer_confidence = 1.0
            return candidate_centerline

        candidate_valid = (
            bool(candidate_update_ok)
            if candidate_update_ok is not None
            else self._candidate_path_is_updateable(
                candidate_centerline=candidate_centerline,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
                result=result,
                candidate_source=candidate_source,
            )[0]
        )
        stored_forward = None
        if self._midline_buffer_path is not None and self._midline_buffer_path.shape[0] >= 2:
            stored_forward = self._extract_forward_path_from_pose(
                path=self._midline_buffer_path,
                vehicle_xy=(vehicle_x, vehicle_y),
                resolution_m=self.midline_station_spacing_m,
            )

        if candidate_valid:
            candidate_forward = self._extract_forward_path_from_pose(
                path=candidate_centerline,
                vehicle_xy=(vehicle_x, vehicle_y),
                resolution_m=self.midline_station_spacing_m,
            )
            if candidate_forward is None or candidate_forward.shape[0] < 2:
                candidate_forward = np.array(candidate_centerline, copy=True)
            candidate_samples = self._resample_midline_stations(candidate_forward)
            if stored_forward is None or stored_forward.shape[0] < 2:
                updated = candidate_samples
            else:
                stored_samples = self._resample_midline_stations(stored_forward)
                updated = self._blend_midline_samples(
                    stored_samples=stored_samples,
                    candidate_samples=candidate_samples,
                    vehicle_x=vehicle_x,
                    vehicle_y=vehicle_y,
                    vehicle_yaw=vehicle_yaw,
                )
            self._midline_buffer_path = np.array(updated, copy=True)
            self._midline_buffer_last_update_sec = now_sec
            self._midline_buffer_confidence = min(1.0, self._midline_buffer_confidence + 0.25)
            return updated

        if self._midline_buffer_path is None or self._midline_buffer_path.shape[0] < 2:
            return np.empty((0, 2), dtype=np.float64)
        if self._midline_buffer_last_update_sec < 0.0:
            return np.empty((0, 2), dtype=np.float64)
        if (now_sec - self._midline_buffer_last_update_sec) > self.midline_hold_last_valid_duration_s:
            self._midline_buffer_path = None
            return np.empty((0, 2), dtype=np.float64)
        # Midline dropout retention is governed by elapsed hold time. Per-cycle
        # confidence decay made the stored path expire in a few frames at high
        # planner rates, which defeats the configured hold window.
        if stored_forward is not None and stored_forward.shape[0] >= 2:
            return self._resample_midline_stations(stored_forward)
        return np.array(self._midline_buffer_path, copy=True)

    def _candidate_path_is_updateable(
        self,
        *,
        candidate_centerline: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        result: MidpointPlannerResult,
        candidate_source: str = "validated",
    ) -> tuple[bool, str]:
        if candidate_centerline.shape[0] < self.candidate_min_points:
            return False, "candidate_too_short"
        if not np.all(np.isfinite(candidate_centerline)):
            return False, "candidate_non_finite"
        candidate_local = self._centerline_to_vehicle_frame(
            centerline=candidate_centerline,
            frame_id=self.odom_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        if candidate_local.shape[0] < self.candidate_min_points:
            return False, "candidate_no_local_path"
        if self._path_forward_extent_local(candidate_local) < self.candidate_min_extent_m:
            return False, "candidate_extent_too_short"
        if (
            not self._hold_mode_active
            and self._midline_buffer_path is not None
            and self._midline_buffer_path.shape[0] >= 2
        ):
            jump = compute_centerline_jump_max(
                candidate_centerline,
                self._midline_buffer_path,
                min(self.midline_horizon_m, self.centerline_jump_horizon_m),
            )
            if jump > self.candidate_jump_reject_threshold_m:
                return False, "candidate_jump_rejected"
        if candidate_source in {"single_boundary_raw_offset", "remembered_pairs", "projected_pairs"}:
            return True, f"{candidate_source}_soft_accept"
        if result.status != "ok":
            return False, result.reject_reason or result.status
        return True, "ok"

    def _minimum_projected_forward_extent_m(self) -> float:
        return max(
            float(self.candidate_min_extent_m),
            float(getattr(self._core_config, "min_forward_extent_m", self.candidate_min_extent_m)),
        )

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

    def _project_midpoint_chain_candidate(
        self,
        *,
        midpoint_chain: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> np.ndarray:
        if midpoint_chain.shape[0] < _MIN_PROJECTABLE_PAIR_COUNT:
            return np.empty((0, 2), dtype=np.float64)
        local_chain = self._centerline_to_vehicle_frame(
            centerline=midpoint_chain,
            frame_id=frame_id,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        if local_chain.shape[0] < _MIN_PROJECTABLE_PAIR_COUNT:
            return np.empty((0, 2), dtype=np.float64)

        projected_local = np.array(local_chain, copy=True)
        target_extent_m = self._minimum_projected_forward_extent_m()
        if self._path_forward_extent_local(projected_local) < target_extent_m:
            tail_count = min(3, projected_local.shape[0])
            direction = projected_local[-1] - projected_local[-tail_count]
            direction_norm = float(np.hypot(direction[0], direction[1]))
            if direction_norm <= 1e-9 and projected_local.shape[0] >= 2:
                direction = projected_local[-1] - projected_local[-2]
                direction_norm = float(np.hypot(direction[0], direction[1]))
            if direction_norm <= 1e-9 or float(direction[0]) <= 1e-6:
                direction = np.asarray([1.0, 0.0], dtype=np.float64)
            else:
                direction = direction / direction_norm
            step_m = max(0.25, float(self.centerline_path_resolution_m))
            while self._path_forward_extent_local(projected_local) < target_extent_m:
                next_point = projected_local[-1] + (direction * step_m)
                if next_point[0] <= projected_local[-1, 0]:
                    next_point[0] = projected_local[-1, 0] + step_m
                projected_local = np.vstack((projected_local, next_point))

        projected_global = np.empty_like(projected_local)
        for idx in range(projected_local.shape[0]):
            ox, oy = self._base_point_to_odom(
                float(projected_local[idx, 0]),
                float(projected_local[idx, 1]),
                vehicle_x,
                vehicle_y,
                vehicle_yaw,
            )
            projected_global[idx, 0] = ox
            projected_global[idx, 1] = oy
        return _finalize_path(projected_global, self._core_config)

    def _select_candidate_centerline(
        self,
        result: MidpointPlannerResult,
    ) -> tuple[np.ndarray, str]:
        if result.centerline.shape[0] > 0:
            return np.array(result.centerline, copy=True), "validated"
        return np.empty((0, 2), dtype=np.float64), "none"

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

    def _blend_midline_samples(
        self,
        *,
        stored_samples: np.ndarray,
        candidate_samples: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> np.ndarray:
        if stored_samples.shape[0] == 0:
            return np.array(candidate_samples, copy=True)
        if candidate_samples.shape[0] == 0:
            return np.array(stored_samples, copy=True)
        count = min(stored_samples.shape[0], candidate_samples.shape[0])
        stored_local = self._centerline_to_vehicle_frame(
            centerline=stored_samples[:count],
            frame_id=self.odom_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        candidate_local = self._centerline_to_vehicle_frame(
            centerline=candidate_samples[:count],
            frame_id=self.odom_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        updated_local = np.array(stored_local, copy=True)
        step = max(0.05, float(self.midline_station_spacing_m))
        for idx in range(count):
            distance_ahead = float(idx) * step
            alpha, max_shift = self._midline_blend_params(distance_ahead)
            delta_local = candidate_local[idx] - stored_local[idx]
            if distance_ahead <= self.midline_control_handoff_distance_m:
                updated_local[idx] = candidate_local[idx]
                continue
            # Keep the candidate longitudinal progression and stabilize only the
            # lateral shape. Blending x here can freeze curvature and make the
            # controller chase an outdated heading segment.
            updated_local[idx, 0] = candidate_local[idx, 0]
            updated_local[idx, 1] = stored_local[idx, 1] + float(
                np.clip(alpha * delta_local[1], -max_shift, max_shift)
            )
            if idx > 0:
                updated_local[idx, 0] = max(updated_local[idx, 0], updated_local[idx - 1, 0] + 0.05)
        updated = np.empty((count, 2), dtype=np.float64)
        for idx in range(count):
            ox, oy = self._base_point_to_odom(
                float(updated_local[idx, 0]),
                float(updated_local[idx, 1]),
                vehicle_x,
                vehicle_y,
                vehicle_yaw,
            )
            updated[idx, 0] = ox
            updated[idx, 1] = oy
        if candidate_samples.shape[0] > count:
            updated = np.vstack((updated, candidate_samples[count:]))
        elif stored_samples.shape[0] > count:
            updated = np.vstack((updated, stored_samples[count:]))
        return updated

    def _midline_blend_params(self, distance_ahead: float) -> tuple[float, float]:
        if distance_ahead <= self.midline_near_distance_m:
            return self.midline_near_alpha, self.midline_near_max_shift_m
        if distance_ahead <= self.midline_mid_distance_m:
            return self.midline_mid_alpha, self.midline_mid_max_shift_m
        return self.midline_far_alpha, self.midline_far_max_shift_m

    def _prepare_centerline_for_current_pose(
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
        )

    def _anchor_centerline_near_vehicle(
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

        # Do not let the controller chase a laterally offset path origin.
        keep_mask = local[:, 0] >= -0.1
        local = local[keep_mask]
        if local.shape[0] == 0:
            local = np.array([[0.0, 0.0]], dtype=np.float64)

        anchor_length_m = max(0.5, min(1.5, float(self.midline_near_distance_m)))
        for idx in range(local.shape[0]):
            x_val = float(local[idx, 0])
            if x_val <= 0.0:
                local[idx, 1] = 0.0
                continue
            if x_val < anchor_length_m:
                local[idx, 1] *= x_val / anchor_length_m

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

    @staticmethod
    def _path_forward_extent_local(path_local: np.ndarray) -> float:
        if path_local.shape[0] == 0:
            return 0.0
        if path_local.shape[0] == 1:
            return max(0.0, float(path_local[0, 0]))
        diffs = np.diff(path_local, axis=0)
        path_length = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        x_span = float(np.max(path_local[:, 0]) - np.min(path_local[:, 0]))
        return max(path_length, x_span)

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
        self._active_planner_mode = "waiting"
        self._active_remembered_cone_count = 0
        self._active_stale_cone_count = 0
        self._active_left_chain_length = 0
        self._active_right_chain_length = 0
        self._active_pair_count = 0
        self._active_unknown_pair_count = 0
        self._active_filtered_track_width_m = float(self._filtered_track_width_m)
        self._active_held_path_flag = 0
        super()._publish_empty_cycle(
            frame_id=frame_id,
            status=status,
            operator_state=operator_state,
            operator_reason=operator_reason,
            cmd_speed=cmd_speed,
            cmd_steering=cmd_steering,
            lookahead=lookahead,
            zero_cmd_sent_flag=zero_cmd_sent_flag,
        )

    def _normalize_core_reject_reason(self, result: Optional[MidpointPlannerResult]) -> str:
        if result is None:
            return "none"
        reject_counts = result.reject_counts or {}
        if int(reject_counts.get("near_field_continuity", 0)) > 0:
            return "near_field_continuity"
        if int(reject_counts.get("midpoint_kink", 0)) > 0:
            return "midpoint_kink"
        text = (result.reject_reason or result.status or "").strip().lower()
        if text.startswith("usable cones below minimum"):
            return "no_safe_chain"
        if text in {
            "no cones available",
            "no colored cones in planning region",
            "no reliable midpoint chain",
            "no reliable boundary chain",
            "path has too few points",
            "path forward extent too short",
        }:
            return "no_safe_chain"
        return "none"

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
        del candidate_diagonal_count
        del selected_chain_length
        return "\n".join(
            [
                f"STATE: {operator_state.upper()}",
                f"MODE: {self._active_planner_mode.upper()}",
                f"REASON: {self._operator_reason_label(operator_reason)}",
                (
                    f"TRACKS: remembered={int(self._active_remembered_cone_count)} | "
                    f"stale={int(self._active_stale_cone_count)}"
                ),
                (
                    f"BOUNDARIES: L={int(self._active_left_chain_length)} | "
                    f"R={int(self._active_right_chain_length)} | "
                    f"pairs={int(self._active_pair_count)} | "
                    f"unknown={int(self._active_unknown_pair_count)} | "
                    f"path={int(centerline_point_count)} pts"
                ),
                (
                    f"WIDTH: {self._fmt_metric(self._active_filtered_track_width_m, ' m')} | "
                    f"HALF: {self._fmt_metric(0.5 * float(self._active_filtered_track_width_m), ' m')}"
                ),
                (
                    f"CMD: v={self._fmt_metric(cmd_speed, ' m/s')} | "
                    f"delta={self._fmt_metric(cmd_steering, ' rad')} | "
                    f"Ld={self._fmt_metric(lookahead, ' m')}"
                ),
                (
                    f"NF: lat={self._fmt_metric(near_field_lateral_max_m, ' m')} | "
                    f"kink={self._fmt_metric(near_field_midpoint_kink_max_rad, ' rad')} | "
                    f"seed={self._fmt_metric(seed_midpoint_distance_m, ' m')}"
                ),
                (
                    f"HOLD: {self._fmt_metric(hold_remaining_s, ' s')} left | "
                    f"held={int(self._active_held_path_flag)} | "
                    f"clean {int(self._hold_clean_frame_count)}/{int(self.hold_exit_clean_frames)}"
                ),
            ]
        )

    def _build_markers(
        self,
        *,
        now,
        frame_id: str,
        result: Optional[MidpointPlannerResult],
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

        if self.show_raw_cones and result.filtered_points.shape[0] > 0:
            arr.markers.append(
                self._make_points_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="filtered_cones",
                    points=result.filtered_points,
                    color=(0.8, 0.8, 0.8, 0.65),
                    scale=0.18,
                )
            )
            marker_id += 1

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
            arr.markers.append(
                self._make_pair_segment_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="accepted_pairs",
                    pair_segments=self._current_pair_segments_for_viz,
                    color=(0.2, 1.0, 0.3, 0.95),
                    width=0.07,
                )
            )
            marker_id += 1

        if self.show_raw_midpoint_chain:
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="raw_midpoint_chain",
                    points=raw_midpoint_chain,
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
                width=0.09,
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

    def _make_pair_segment_marker(
        self,
        *,
        frame_id: str,
        stamp,
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
    def _point_msg(x: float, y: float, z: float):
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        return point

    def _log_mode_summary(
        self,
        *,
        mode: str,
        result: MidpointPlannerResult,
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MidpointPlannerNode()
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
