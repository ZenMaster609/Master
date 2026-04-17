#!/usr/bin/env python3
"""Corridor planner over tracked cone detections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.triangulation_planner_core import (
    compute_centerline_jump_max,
    edge_churn_count,
    edge_churn_ratio,
    selected_edge_keys,
    tracked_cones_frame_delta_p95,
)
from sim_car.planning.planner_runtime_types import PlannerIdentity
from sim_car.planning.tracked_cone_planner_contract import (
    COMMON_MIGRATED_TRACKED_CONE_PLANNER_DEFAULTS,
    apply_common_config_to_node,
    read_migrated_tracked_cone_planner_common_config,
)
from sim_car.planning.corridor_planner_core import (
    CorridorPlannerConfig,
    CorridorPlannerPrior,
    CorridorPlannerResult,
    compute_corridor_centerline,
    update_track_width_estimate,
)
from sim_car.planning.tracked_cone_planner_base import TrackedConePlannerBase

MSG_TRACK_STATE_TENTATIVE = int(getattr(ConeDetection, "TRACK_STATE_TENTATIVE", 0))
MSG_TRACK_STATE_CONFIRMED = int(getattr(ConeDetection, "TRACK_STATE_CONFIRMED", 1))
MSG_TRACK_STATE_STALE = int(getattr(ConeDetection, "TRACK_STATE_STALE", 2))
_PAIR_PASSED_MARGIN_M = 0.5
_CORRIDOR_MIDPOINT_SOURCE = "corridor_midpoints"
_CORRIDOR_MIDPOINT_MIN_POINTS = 2
_CENTERLINE_MARKER_WIDTH_M = 0.20
_CORRIDOR_ANALYSIS_SAMPLE_COUNT = 8
_CORRIDOR_ANALYSIS_SAMPLE_SPACING_M = 1.0
_ANCHOR_TAPER_GATE_LATERAL_M = 0.20
_ANCHOR_TAPER_GATE_HEADING_RAD = 0.18
_VALIDATED_JUMP_ACCEPT_HORIZON_M = 3.0
_VALIDATED_JUMP_ACCEPT_LATERAL_MAX_M = 0.45
_VALIDATED_JUMP_ACCEPT_LATERAL_MEAN_M = 0.25
_VALIDATED_JUMP_ACCEPT_HEADING_DELTA_RAD = 0.30
_CONE_AUDIT_REASONS = (
    "used_left_chain",
    "used_right_chain",
    "chain_step_too_close",
    "chain_step_too_far",
    "chain_no_forward_progress",
    "chain_radial_regression",
    "chain_forward_projection",
    "chain_heading_change",
    "chain_shadowed",
    "chain_not_best_next_step",
    "chain_no_forward_seed",
    "chain_unreached",
    "rejected_geometry_range",
    "rejected_geometry_behind",
    "rejected_geometry_horizon",
    "rejected_geometry_lateral",
    "rejected_confidence",
    "rejected_tentative",
    "rejected_color",
    "rejected_nonfinite",
)
_CORRIDOR_PAIR_AUDIT_REASONS = (
    "pair_valid",
    "pair_width_too_narrow",
    "pair_width_too_wide",
    "pair_left_behind",
    "pair_right_behind",
    "pair_left_beyond_horizon",
    "pair_right_beyond_horizon",
    "pair_not_in_longest_valid_slice",
    "pair_nonfinite",
    "pair_rejected_unknown",
)


@dataclass
class _ConeAuditEntry:
    track_id: int
    reason: str
    point_xy: np.ndarray
    local_x_m: float
    local_y_m: float
    raw_color: str
    resolved_color: str
    track_state: int
    confidence: float
    track_confidence: float
    color_confidence: float
    missed_count: int
    last_seen_age_sec: float
    memory_only: bool


@dataclass
class _PairMemoryEntry:
    left_x_odom: float
    left_y_odom: float
    right_x_odom: float
    right_y_odom: float
    midpoint_x_odom: float
    midpoint_y_odom: float
    last_valid_sec: float


class CorridorPlannerNode(TrackedConePlannerBase):
    """Tracked-cone corridor planner with shared path-memory stabilization."""

    def __init__(self) -> None:
        self._planner_identity = PlannerIdentity(
            node_name="corridor_planner_node",
            planner_mode="corridor",
            diagnostics_prefix="corridor_planner",
            diagnostics_topic="/corridor_planner/diagnostics",
        )
        Node.__init__(self, self._planner_identity.node_name)
        self._declare_parameters()
        self._read_parameters()

        self._init_common_planner_state()
        self._pair_memory: list[_PairMemoryEntry] = []
        self._active_cone_audit_counts = self._empty_cone_audit_counts()

        self._init_common_ros_interfaces()
        self._cone_audit_viz_pub = self.create_publisher(MarkerArray, self.cone_audit_viz_topic, 10)

    def _declare_parameters(self) -> None:
        defaults = dict(COMMON_MIGRATED_TRACKED_CONE_PLANNER_DEFAULTS)
        defaults.update({
            "filtering.planning_horizon_m": 20.0,
            "filtering.max_lateral_range_m": 14.0,
            "boundary_chain.min_chain_length": 3,
            'boundary_chain.max_heading_change_rad': 2.35,

            "width_estimation.min_trustworthy_pairs": 3,
            "corridor.min_corridor_width_m": 2.2,
            "corridor.max_corridor_width_m": 6.4,
            "corridor.boundary_resample_dx": 0.5,
            "corridor.min_required_corridor_samples": 5,
            "corridor.path_fit_smoothing_window": 5,
            "corridor.membership_margin_m": 0.15,
            "midline_memory.pair_memory_retention_s": 12.0,
            "lap_tracking.target_laps": 0,
            "validation.candidate_min_points": 2,
            "validation.candidate_min_extent_m": 0.25,
            "validation.min_path_points": 4,
            "validation.min_forward_extent_m": 2.0,
            "validation.max_heading_delta_rad": 0.75,
            "validation.max_initial_heading_error_rad": 1.0,
            "validation.max_curvature": 0.45,
            "debug.enable_cone_audit_markers": False,
            "debug.cone_audit_viz_topic": "/corridor_planner/cone_audit_viz",
            "debug.cone_audit_show_labels": True,
            "debug.cone_audit_max_labels": 80,
            "debug.show_corridor_pair_audit": False,
            "debug.corridor_pair_audit_show_labels": True,
            "debug.corridor_pair_audit_max_labels": 80,
            "diagnostics.topic": "/corridor_planner/diagnostics",
        })
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        common = read_migrated_tracked_cone_planner_common_config(
            self,
            planner_label='corridor planner',
            diagnostics_topic_fallback=self._planner_identity.diagnostics_topic,
        )
        apply_common_config_to_node(self, common)
        self.pair_memory_retention_s = max(
            self.midline_hold_last_valid_duration_s,
            float(self.get_parameter("midline_memory.pair_memory_retention_s").value),
        )
        self.lap_tracking_target_laps = max(
            0,
            int(self.get_parameter("lap_tracking.target_laps").value),
        )
        self.enable_cone_audit_markers = bool(
            self.get_parameter("debug.enable_cone_audit_markers").value
        )
        self.cone_audit_viz_topic = str(
            self.get_parameter("debug.cone_audit_viz_topic").value
        ).strip() or "/corridor_planner/cone_audit_viz"
        self.cone_audit_show_labels = bool(
            self.get_parameter("debug.cone_audit_show_labels").value
        )
        self.cone_audit_max_labels = max(
            0,
            int(self.get_parameter("debug.cone_audit_max_labels").value),
        )
        self.show_corridor_pair_audit = bool(
            self.get_parameter("debug.show_corridor_pair_audit").value
        )
        self.corridor_pair_audit_show_labels = bool(
            self.get_parameter("debug.corridor_pair_audit_show_labels").value
        )
        self.corridor_pair_audit_max_labels = max(
            0,
            int(self.get_parameter("debug.corridor_pair_audit_max_labels").value),
        )

        self._core_config = CorridorPlannerConfig(
            max_cone_range_m=float(self.get_parameter("filtering.max_cone_range_m").value),
            planning_horizon_m=float(self.get_parameter("filtering.planning_horizon_m").value),
            max_lateral_range_m=float(self.get_parameter("filtering.max_lateral_range_m").value),
            behind_drop_m=float(self.get_parameter("filtering.behind_drop_m").value),
            min_confidence=float(self.get_parameter("filtering.min_confidence").value),
            min_required_cones=max(2, int(self.get_parameter("filtering.min_required_cones").value)),
            min_step_m=float(self.get_parameter("boundary_chain.min_step_m").value),
            max_step_m=float(self.get_parameter("boundary_chain.max_step_m").value),
            max_heading_change_rad=float(self.get_parameter("boundary_chain.max_heading_change_rad").value),
            min_forward_progress_m=float(
                self.get_parameter("boundary_chain.min_forward_progress_m").value
            ),
            min_chain_length=max(2, int(self.get_parameter("boundary_chain.min_chain_length").value)),
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
            boundary_resample_dx=float(self.get_parameter("corridor.boundary_resample_dx").value),
            min_corridor_width_m=float(self.get_parameter("corridor.min_corridor_width_m").value),
            max_corridor_width_m=float(self.get_parameter("corridor.max_corridor_width_m").value),
            min_required_corridor_samples=max(
                2,
                int(self.get_parameter("corridor.min_required_corridor_samples").value),
            ),
            path_fit_smoothing_window=max(
                1,
                int(self.get_parameter("corridor.path_fit_smoothing_window").value),
            ),
            membership_margin_m=float(self.get_parameter("corridor.membership_margin_m").value),
            path_resolution_m=float(self.get_parameter("centerline.path_resolution_m").value),
            max_path_length_m=float(self.get_parameter("centerline.max_path_length_m").value),
            min_path_points=max(2, int(self.get_parameter("validation.min_path_points").value)),
            min_forward_extent_m=float(self.get_parameter("validation.min_forward_extent_m").value),
            jump_check_horizon_m=float(self.get_parameter("validation.jump_check_horizon_m").value),
            max_near_field_lateral_jump_m=float(
                self.get_parameter("validation.max_near_field_lateral_jump_m").value
            ),
            max_heading_delta_rad=float(
                self.get_parameter("validation.max_heading_delta_rad").value
            ),
            max_initial_heading_error_rad=float(
                self.get_parameter("validation.max_initial_heading_error_rad").value
            ),
            max_curvature=float(
                self.get_parameter("validation.max_curvature").value
            ),
        )
        self._filtered_track_width_m = float(self._core_config.initial_width_m)

    def _on_timer(self) -> None:
        ctx = self._resolve_cone_planning_context()
        if ctx is None:
            return
        cones_msg, target_frame, vehicle_x, vehicle_y, vehicle_yaw, points_xy, colors, confidences = ctx
        self._update_smalltrack_lap_from_orange_cones(
            cones_msg=cones_msg,
            points_xy=points_xy,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        control_target_frame: Optional[np.ndarray] = None
        control_debug_metrics: Optional[dict[str, float]] = None
        cmd_speed = 0.0
        cmd_steering = 0.0
        lookahead = 0.0

        self._update_remembered_cone_viz(points_xy=points_xy, colors=colors)
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        raw_orange_count = sum(
            1 for cone in cones_msg.cones if normalize_color(getattr(cone, "color", "")) == "orange"
        )
        boundary_hint_count = sum(
            1 for cone in cones_msg.cones if str(getattr(cone, "boundary_color", "")).strip()
        )
        resolved_blue_count = sum(1 for color in colors if color == "blue")
        resolved_yellow_count = sum(1 for color in colors if color == "yellow")
        planning_frame = self._tracked_cone_planning_frame(
            msg=cones_msg,
            points_xy=points_xy,
            colors=colors,
            confidences=confidences,
        )
        self._active_remembered_cone_count = int(len(cones_msg.cones))
        self._active_stale_cone_count = int(
            np.count_nonzero(planning_frame.track_states == MSG_TRACK_STATE_STALE)
        )

        result = compute_corridor_centerline(
            points_xy=planning_frame.points_xy,
            colors=planning_frame.colors,
            confidences=planning_frame.planner_confidences,
            track_ids=planning_frame.track_ids,
            vehicle_xy=(vehicle_x, vehicle_y),
            vehicle_yaw=vehicle_yaw,
            config=self._core_config,
            prior=CorridorPlannerPrior(
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
        cone_audit_entries = self._build_cone_audit_entries(
            msg=cones_msg,
            planning_frame=planning_frame,
            result=result,
            vehicle_xy=(vehicle_x, vehicle_y),
            vehicle_yaw=vehicle_yaw,
            now_sec=now_sec,
        )
        self._active_cone_audit_counts = self._cone_audit_counts(cone_audit_entries)
        self._publish_cone_audit_markers(
            frame_id=target_frame,
            stamp=self.get_clock().now().to_msg(),
            entries=cone_audit_entries,
        )

        remembered_pair_entries = self._active_pair_memory_entries(
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        live_center_anchors = np.array(result.midpoints_raw, copy=True)
        raw_midpoint_chain = np.array(live_center_anchors, copy=True)
        pair_segments_for_viz = np.array(result.pair_segments, copy=True)
        live_pair_entries = self._pair_entries_from_segments(
            pair_segments=result.pair_segments,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
            now_sec=now_sec,
        )
        combined_pair_entries = self._merge_pair_entries(
            remembered_entries=remembered_pair_entries,
            live_entries=live_pair_entries,
        )
        combined_pair_entries = self._sort_pair_entries_by_forward_progress(
            entries=combined_pair_entries,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        combined_pair_segments, combined_midpoint_chain = self._pair_geometry_from_memory(
            combined_pair_entries
        )
        if combined_pair_segments.size > 0:
            pair_segments_for_viz = combined_pair_segments
        if combined_midpoint_chain.size > 0:
            raw_midpoint_chain = combined_midpoint_chain
        raw_centerline, candidate_source = self._select_candidate_centerline(
            result=result,
            support_chain=raw_midpoint_chain,
            memory_midpoint_chain=combined_midpoint_chain,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
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
        centerline = self._update_midline_buffer(
            candidate_centerline=raw_centerline,
            candidate_source=candidate_source,
            candidate_update_ok=candidate_update_ok,
            candidate_update_reason=candidate_update_reason,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
            result=result,
            now_sec=now_sec,
            support_centerline=raw_midpoint_chain,
        )
        candidate_update_ok = bool(
            getattr(self, "_last_midline_candidate_update_ok", candidate_update_ok)
        )
        candidate_update_reason = str(
            getattr(self, "_last_midline_candidate_update_reason", candidate_update_reason)
        )
        buffered_centerline = np.array(centerline, copy=True)
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
        if result.status == "ok" and result.pair_segments.size > 0:
            self._remember_pairs(
                result=result,
                frame_id=target_frame,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
                now_sec=now_sec,
            )
        if centerline.shape[0] > 0 and raw_centerline.shape[0] > 0:
            self._last_valid_pair_segments = (
                pair_segments_for_viz
                if pair_segments_for_viz.size > 0
                else self._last_valid_pair_segments
            )
        elif self._last_valid_pair_segments is not None and centerline.shape[0] > 0:
            pair_segments_for_viz = np.array(self._last_valid_pair_segments, copy=True)

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
            if self._last_valid_pair_segments is not None:
                pair_segments_for_viz = np.array(self._last_valid_pair_segments, copy=True)
            if self._last_valid_raw_midpoint_chain is not None and raw_midpoint_chain.size == 0:
                raw_midpoint_chain = np.array(self._last_valid_raw_midpoint_chain, copy=True)

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

        control_path = self._centerline_to_vehicle_frame(
            centerline=centerline,
            frame_id=target_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        corridor_analysis_metrics = self._corridor_analysis_metrics(
            raw_anchor_path=live_center_anchors,
            prevalidation_centerline=result.prevalidation_centerline,
            buffered_centerline=buffered_centerline,
            control_path_local=control_path,
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
        self._active_pair_count = int(pair_segments_for_viz.shape[0]) if pair_segments_for_viz.size > 0 else int(result.accepted_pair_count)
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
            selected_edge_count=int(pair_segments_for_viz.shape[0]),
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
                "corridor_sample_count": result.accepted_pair_count,
                "corridor_width_min_m": result.corridor_width_min_m,
                "corridor_width_median_m": result.selected_chain_width_median,
                "corridor_width_max_m": result.corridor_width_max_m,
                "left_chain_length": result.left_chain_length,
                "right_chain_length": result.right_chain_length,
                "raw_orange_count": raw_orange_count,
                "resolved_blue_count": resolved_blue_count,
                "resolved_yellow_count": resolved_yellow_count,
                "boundary_hint_count": boundary_hint_count,
                "candidate_source": candidate_source,
                "midline_update_mode": (
                    "hold" if publish_mode == "held" else self._last_midline_update_mode
                ),
                "midline_update_reason": getattr(self, "_last_midline_candidate_update_reason", ""),
                "midline_candidate_jump_m": getattr(self, "_last_midline_candidate_jump_m", float("nan")),
                "midline_near_lateral_delta_max_m": getattr(
                    self,
                    "_last_midline_near_lateral_delta_max_m",
                    float("nan"),
                ),
                "midline_buffer_confidence": getattr(
                    self,
                    "_last_midline_buffer_confidence",
                    float("nan"),
                ),
                "midline_recovery_count": getattr(self, "_midline_recovery_count", 0),
                **self._midline_estimation_metrics_for_diagnostics(),
                "reject_corridor_geometry_count": result.reject_counts.get("corridor_geometry", 0),
                "reject_corridor_samples_count": result.reject_counts.get("corridor_samples", 0),
                "reject_path_outside_corridor_count": result.reject_counts.get("path_outside_corridor", 0),
                "reject_heading_count": result.reject_counts.get("heading", 0),
                "reject_curvature_count": result.reject_counts.get("curvature", 0),
                "reject_near_field_continuity_count": result.reject_counts.get("near_field_continuity", 0),
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
                **self._active_cone_audit_counts,
                **self._corridor_pair_audit_counts(result),
                **corridor_analysis_metrics,
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

    def _build_cone_audit_entries(
        self,
        *,
        msg: ConeDetectionArray,
        planning_frame,
        result: CorridorPlannerResult,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
        now_sec: float,
    ) -> list[_ConeAuditEntry]:
        points = np.asarray(planning_frame.points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            points = np.empty((0, 2), dtype=np.float64)
        local_points = self._audit_points_to_vehicle_frame(points, vehicle_xy, vehicle_yaw)
        used_left = {int(track_id) for track_id in np.asarray(result.used_left_track_ids, dtype=np.int64)}
        used_right = {int(track_id) for track_id in np.asarray(result.used_right_track_ids, dtype=np.int64)}
        entries: list[_ConeAuditEntry] = []
        for idx, cone in enumerate(msg.cones):
            point_xy = points[idx] if idx < points.shape[0] else np.asarray([float("nan"), float("nan")])
            local_xy = (
                local_points[idx]
                if idx < local_points.shape[0]
                else np.asarray([float("nan"), float("nan")], dtype=np.float64)
            )
            track_id = self._audit_array_int(planning_frame.track_ids, idx, idx)
            track_state = self._audit_array_int(planning_frame.track_states, idx, MSG_TRACK_STATE_CONFIRMED)
            confidence = self._audit_array_float(
                planning_frame.raw_confidences,
                idx,
                float(getattr(cone, "confidence", 0.0)),
            )
            planner_confidence = self._audit_array_float(planning_frame.planner_confidences, idx, confidence)
            track_confidence = self._audit_array_float(
                planning_frame.track_confidences,
                idx,
                float(getattr(cone, "track_confidence", confidence)),
            )
            raw_color = (
                planning_frame.raw_colors[idx]
                if idx < len(planning_frame.raw_colors)
                else normalize_color(getattr(cone, "color", ""))
            )
            resolved_color = (
                planning_frame.colors[idx]
                if idx < len(planning_frame.colors)
                else normalize_color(getattr(cone, "boundary_color", ""))
            )
            missed_count = int(getattr(cone, "missed_count", 0))
            age_sec = self._stamp_age_sec(getattr(cone, "last_seen", None), now_sec)
            memory_only = bool(track_state == MSG_TRACK_STATE_STALE or missed_count > 0)
            reason = self._classify_cone_audit_reason(
                track_id=track_id,
                local_xy=local_xy,
                resolved_color=resolved_color,
                track_state=track_state,
                planner_confidence=planner_confidence,
                used_left=used_left,
                used_right=used_right,
                chain_rejection_reasons_by_track_id=result.chain_rejection_reasons_by_track_id,
            )
            entries.append(
                _ConeAuditEntry(
                    track_id=int(track_id),
                    reason=reason,
                    point_xy=np.asarray(point_xy, dtype=np.float64),
                    local_x_m=float(local_xy[0]),
                    local_y_m=float(local_xy[1]),
                    raw_color=str(raw_color),
                    resolved_color=str(resolved_color),
                    track_state=int(track_state),
                    confidence=float(confidence),
                    track_confidence=float(track_confidence),
                    color_confidence=float(getattr(cone, "color_confidence", float("nan"))),
                    missed_count=missed_count,
                    last_seen_age_sec=float(age_sec),
                    memory_only=memory_only,
                )
            )
        return entries

    def _classify_cone_audit_reason(
        self,
        *,
        track_id: int,
        local_xy: np.ndarray,
        resolved_color: str,
        track_state: int,
        planner_confidence: float,
        used_left: set[int],
        used_right: set[int],
        chain_rejection_reasons_by_track_id: dict[int, str],
    ) -> str:
        if not np.all(np.isfinite(local_xy)):
            return "rejected_nonfinite"
        if int(track_state) == MSG_TRACK_STATE_TENTATIVE:
            return "rejected_tentative"
        x_m = float(local_xy[0])
        y_m = float(local_xy[1])
        distance_m = float(math.hypot(x_m, y_m))
        if x_m < -float(self._core_config.behind_drop_m):
            return "rejected_geometry_behind"
        if x_m > float(self._core_config.planning_horizon_m):
            return "rejected_geometry_horizon"
        if abs(y_m) > float(self._core_config.max_lateral_range_m):
            return "rejected_geometry_lateral"
        if distance_m > float(self._core_config.max_cone_range_m):
            return "rejected_geometry_range"
        if (
            not math.isfinite(float(planner_confidence))
            or planner_confidence < float(self._core_config.min_confidence)
        ):
            return "rejected_confidence"
        if normalize_color(resolved_color) not in {"blue", "yellow"}:
            return "rejected_color"
        if int(track_id) in used_left:
            return "used_left_chain"
        if int(track_id) in used_right:
            return "used_right_chain"
        chain_reason = chain_rejection_reasons_by_track_id.get(int(track_id), "")
        return str(chain_reason) if chain_reason else "chain_unreached"

    @staticmethod
    def _audit_points_to_vehicle_frame(
        points_xy: np.ndarray,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        shifted = points - np.asarray(vehicle_xy, dtype=np.float64).reshape(1, 2)
        cos_yaw = math.cos(float(vehicle_yaw))
        sin_yaw = math.sin(float(vehicle_yaw))
        return np.column_stack(
            (
                (cos_yaw * shifted[:, 0]) + (sin_yaw * shifted[:, 1]),
                (-sin_yaw * shifted[:, 0]) + (cos_yaw * shifted[:, 1]),
            )
        ).astype(np.float64)

    @staticmethod
    def _audit_array_int(values: np.ndarray, idx: int, default: int) -> int:
        arr = np.asarray(values, dtype=np.int64)
        if idx < 0 or idx >= arr.size:
            return int(default)
        return int(arr[idx])

    @staticmethod
    def _audit_array_float(values: np.ndarray, idx: int, default: float) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if idx < 0 or idx >= arr.size:
            return float(default)
        return float(arr[idx])

    @staticmethod
    def _stamp_age_sec(stamp, now_sec: float) -> float:
        if stamp is None:
            return float("nan")
        stamp_sec = float(getattr(stamp, "sec", 0)) + (float(getattr(stamp, "nanosec", 0)) * 1e-9)
        if stamp_sec <= 0.0:
            return float("nan")
        return max(0.0, float(now_sec) - stamp_sec)

    @staticmethod
    def _empty_cone_audit_counts() -> dict[str, int]:
        counts = {
            "cone_audit_received_count": 0,
            "cone_audit_used_left_count": 0,
            "cone_audit_used_right_count": 0,
            "cone_audit_stale_count": 0,
            "cone_audit_live_count": 0,
        }
        for reason in _CONE_AUDIT_REASONS:
            counts[f"cone_audit_{reason}_count"] = 0
        return counts

    def _cone_audit_counts(self, entries: list[_ConeAuditEntry]) -> dict[str, int]:
        counts = self._empty_cone_audit_counts()
        counts["cone_audit_received_count"] = int(len(entries))
        for entry in entries:
            counts[f"cone_audit_{entry.reason}_count"] = counts.get(
                f"cone_audit_{entry.reason}_count",
                0,
            ) + 1
            if entry.reason == "used_left_chain":
                counts["cone_audit_used_left_count"] += 1
            elif entry.reason == "used_right_chain":
                counts["cone_audit_used_right_count"] += 1
            if entry.memory_only:
                counts["cone_audit_stale_count"] += 1
            else:
                counts["cone_audit_live_count"] += 1
        return counts

    @staticmethod
    def _corridor_pair_audit_counts(result: CorridorPlannerResult) -> dict[str, int]:
        counts = {
            "corridor_pair_audit_total_count": 0,
            "corridor_pair_audit_rejected_count": 0,
        }
        for reason in _CORRIDOR_PAIR_AUDIT_REASONS:
            counts[f"corridor_pair_audit_{reason}_count"] = 0
        for reason in result.corridor_pair_audit_reasons:
            key = f"corridor_pair_audit_{reason}_count"
            counts[key] = counts.get(key, 0) + 1
            counts["corridor_pair_audit_total_count"] += 1
            if reason != "pair_valid":
                counts["corridor_pair_audit_rejected_count"] += 1
        return counts

    def _publish_cone_audit_markers(
        self,
        *,
        frame_id: str,
        stamp,
        entries: list[_ConeAuditEntry],
    ) -> None:
        if not self.enable_cone_audit_markers:
            return
        markers = self._build_cone_audit_markers(frame_id=frame_id, stamp=stamp, entries=entries)
        self._cone_audit_viz_pub.publish(markers)

    def _build_cone_audit_markers(
        self,
        *,
        frame_id: str,
        stamp,
        entries: list[_ConeAuditEntry],
    ) -> MarkerArray:
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = frame_id
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        marker_id = 1
        label_count = 0
        for entry in entries:
            if not np.all(np.isfinite(entry.point_xy)):
                continue
            main = Marker()
            main.header.frame_id = frame_id
            main.header.stamp = stamp
            main.ns = self._cone_audit_marker_namespace(entry.reason)
            main.id = marker_id
            marker_id += 1
            main.type = Marker.CUBE if entry.reason.startswith("rejected_geometry") else Marker.SPHERE
            main.action = Marker.ADD
            main.pose.position.x = float(entry.point_xy[0])
            main.pose.position.y = float(entry.point_xy[1])
            main.pose.position.z = 0.18
            main.pose.orientation.w = 1.0
            scale = self._cone_audit_marker_scale(entry.reason)
            main.scale.x = scale
            main.scale.y = scale
            main.scale.z = scale
            r, g, b, a = self._cone_audit_marker_rgba(entry)
            main.color.r = r
            main.color.g = g
            main.color.b = b
            main.color.a = a
            arr.markers.append(main)

            if entry.memory_only:
                halo = Marker()
                halo.header.frame_id = frame_id
                halo.header.stamp = stamp
                halo.ns = "cone_audit_stale_halo"
                halo.id = marker_id
                marker_id += 1
                halo.type = Marker.CYLINDER
                halo.action = Marker.ADD
                halo.pose.position.x = float(entry.point_xy[0])
                halo.pose.position.y = float(entry.point_xy[1])
                halo.pose.position.z = 0.03
                halo.pose.orientation.w = 1.0
                halo.scale.x = 0.55
                halo.scale.y = 0.55
                halo.scale.z = 0.03
                halo.color.r = 0.1
                halo.color.g = 0.95
                halo.color.b = 1.0
                halo.color.a = 0.55
                arr.markers.append(halo)

            if self.cone_audit_show_labels and label_count < self.cone_audit_max_labels:
                label = Marker()
                label.header.frame_id = frame_id
                label.header.stamp = stamp
                label.ns = "cone_audit_labels"
                label.id = marker_id
                marker_id += 1
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = float(entry.point_xy[0])
                label.pose.position.y = float(entry.point_xy[1])
                label.pose.position.z = 0.75
                label.pose.orientation.w = 1.0
                label.scale.z = 0.16
                label.color.r = 1.0
                label.color.g = 1.0
                label.color.b = 1.0
                label.color.a = 0.95
                label.text = self._cone_audit_label(entry)
                arr.markers.append(label)
                label_count += 1
        return arr

    @staticmethod
    def _cone_audit_marker_namespace(reason: str) -> str:
        if reason in {"used_left_chain", "used_right_chain"}:
            return f"cone_audit_{reason}"
        if reason.startswith("chain_"):
            return "cone_audit_chain_rejected"
        if reason.startswith("rejected_geometry"):
            return "cone_audit_rejected_geometry"
        if reason in {"rejected_confidence", "rejected_tentative"}:
            return "cone_audit_rejected_confidence"
        if reason == "rejected_color":
            return "cone_audit_rejected_color"
        return "cone_audit_rejected_other"

    @staticmethod
    def _cone_audit_marker_scale(reason: str) -> float:
        if reason in {"used_left_chain", "used_right_chain"}:
            return 0.32
        if reason.startswith("chain_"):
            return 0.22
        return 0.25

    @staticmethod
    def _cone_audit_marker_rgba(entry: _ConeAuditEntry) -> tuple[float, float, float, float]:
        if entry.reason == "used_left_chain":
            return 0.2, 0.55, 1.0, 0.95
        if entry.reason == "used_right_chain":
            return 1.0, 0.9, 0.2, 0.95
        if entry.reason.startswith("chain_"):
            return 0.92, 0.92, 0.92, 0.82
        if entry.reason.startswith("rejected_geometry"):
            return 1.0, 0.05, 0.05, 0.9
        if entry.reason in {"rejected_confidence", "rejected_tentative"}:
            return 1.0, 0.0, 0.85, 0.9
        if entry.reason == "rejected_color":
            return 1.0, 0.45, 0.05, 0.9
        return 0.95, 0.95, 0.95, 0.65

    @staticmethod
    def _cone_audit_state_label(track_state: int) -> str:
        if int(track_state) == MSG_TRACK_STATE_TENTATIVE:
            return "tentative"
        if int(track_state) == MSG_TRACK_STATE_STALE:
            return "stale"
        return "confirmed"

    def _cone_audit_label(self, entry: _ConeAuditEntry) -> str:
        age_text = "nan" if not math.isfinite(entry.last_seen_age_sec) else f"{entry.last_seen_age_sec:.2f}s"
        suffix = " memory" if entry.memory_only else ""
        return (
            f"id={entry.track_id} {self._cone_audit_state_label(entry.track_state)}{suffix}\n"
            f"raw={entry.raw_color} side={entry.resolved_color} "
            f"conf={entry.track_confidence:.2f}/{entry.confidence:.2f}\n"
            f"local=({entry.local_x_m:.1f},{entry.local_y_m:.1f}) age={age_text}\n"
            f"{entry.reason}"
        )

    def _active_pair_memory_entries(
        self,
        *,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[_PairMemoryEntry]:
        keep: list[_PairMemoryEntry] = []
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        for entry in self._pair_memory:
            midpoint_x_base, midpoint_y_base = self._odom_point_to_base(
                entry.midpoint_x_odom,
                entry.midpoint_y_odom,
                vehicle_x,
                vehicle_y,
                vehicle_yaw,
            )
            del midpoint_y_base
            if midpoint_x_base < -_PAIR_PASSED_MARGIN_M:
                continue
            if (now_sec - float(entry.last_valid_sec)) > self.pair_memory_retention_s:
                continue
            if midpoint_x_base <= (self.midline_horizon_m + self._core_config.max_path_length_m):
                keep.append(entry)
        self._pair_memory = keep
        return keep

    def _pair_entries_from_segments(
        self,
        *,
        pair_segments: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
    ) -> list[_PairMemoryEntry]:
        if pair_segments.size == 0:
            return []
        entries: list[_PairMemoryEntry] = []
        for pair_segment in np.asarray(pair_segments, dtype=np.float64):
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
                    left_x_odom, left_y_odom, vehicle_x, vehicle_y, vehicle_yaw
                )
                right_x_odom, right_y_odom = self._base_point_to_odom(
                    right_x_odom, right_y_odom, vehicle_x, vehicle_y, vehicle_yaw
                )
                midpoint_x_odom, midpoint_y_odom = self._base_point_to_odom(
                    midpoint_x_odom, midpoint_y_odom, vehicle_x, vehicle_y, vehicle_yaw
                )
            elif not self._is_alias(frame_id, self.odom_frame):
                continue
            entries.append(
                _PairMemoryEntry(
                    left_x_odom=left_x_odom,
                    left_y_odom=left_y_odom,
                    right_x_odom=right_x_odom,
                    right_y_odom=right_y_odom,
                    midpoint_x_odom=midpoint_x_odom,
                    midpoint_y_odom=midpoint_y_odom,
                    last_valid_sec=float(now_sec),
                )
            )
        return entries

    @staticmethod
    def _merge_pair_entries(
        *,
        remembered_entries: list[_PairMemoryEntry],
        live_entries: list[_PairMemoryEntry],
    ) -> list[_PairMemoryEntry]:
        merged: list[_PairMemoryEntry] = []
        for entry in remembered_entries + live_entries:
            duplicate_idx = None
            for idx, existing in enumerate(merged):
                dx = float(existing.midpoint_x_odom - entry.midpoint_x_odom)
                dy = float(existing.midpoint_y_odom - entry.midpoint_y_odom)
                if math.hypot(dx, dy) <= 0.35:
                    duplicate_idx = idx
                    break
            if duplicate_idx is None:
                merged.append(entry)
            else:
                merged[duplicate_idx] = entry
        return merged

    def _sort_pair_entries_by_forward_progress(
        self,
        *,
        entries: list[_PairMemoryEntry],
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[_PairMemoryEntry]:
        if len(entries) <= 1:
            return list(entries)
        local_midpoints = np.empty((len(entries), 2), dtype=np.float64)
        for idx, entry in enumerate(entries):
            midpoint_x_base, midpoint_y_base = self._odom_point_to_base(
                entry.midpoint_x_odom,
                entry.midpoint_y_odom,
                vehicle_x,
                vehicle_y,
                vehicle_yaw,
            )
            local_midpoints[idx, 0] = float(midpoint_x_base)
            local_midpoints[idx, 1] = float(midpoint_y_base)

        seed_candidates = np.flatnonzero(local_midpoints[:, 0] >= -_PAIR_PASSED_MARGIN_M)
        if seed_candidates.size > 0:
            seed_idx = int(
                min(
                    seed_candidates.tolist(),
                    key=lambda idx: (
                        max(float(local_midpoints[idx, 0]), 0.0),
                        abs(float(local_midpoints[idx, 1])),
                        float(np.hypot(local_midpoints[idx, 0], local_midpoints[idx, 1])),
                        idx,
                    ),
                )
            )
        else:
            seed_idx = int(np.argmin(np.hypot(local_midpoints[:, 0], local_midpoints[:, 1])))

        ordered_indices = [seed_idx]
        remaining = {idx for idx in range(len(entries)) if idx != seed_idx}
        heading = np.asarray([1.0, 0.0], dtype=np.float64)
        current_range = float(np.hypot(local_midpoints[seed_idx, 0], local_midpoints[seed_idx, 1]))

        while remaining:
            current_idx = ordered_indices[-1]
            current_point = local_midpoints[current_idx]
            best_idx = None
            best_score = None
            for candidate_idx in remaining:
                delta = local_midpoints[candidate_idx] - current_point
                distance = float(np.hypot(delta[0], delta[1]))
                if distance <= 1e-6:
                    continue
                candidate_range = float(np.hypot(local_midpoints[candidate_idx, 0], local_midpoints[candidate_idx, 1]))
                if candidate_range < current_range - 0.20:
                    continue
                step_dir = delta / distance
                heading_error = abs(math.atan2(step_dir[1], step_dir[0]) - math.atan2(heading[1], heading[0]))
                heading_error = abs(math.atan2(math.sin(heading_error), math.cos(heading_error)))
                backward_m = max(0.0, -float(delta[0]))
                score = (
                    backward_m > 0.75,
                    backward_m,
                    heading_error,
                    distance,
                    abs(float(delta[1])),
                    candidate_idx,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = candidate_idx
            if best_idx is None:
                break
            delta = local_midpoints[best_idx] - current_point
            delta_norm = float(np.hypot(delta[0], delta[1]))
            if delta_norm > 1e-6:
                heading = delta / delta_norm
            ordered_indices.append(best_idx)
            remaining.remove(best_idx)
            current_range = float(np.hypot(local_midpoints[best_idx, 0], local_midpoints[best_idx, 1]))

        return [entries[idx] for idx in ordered_indices]

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

    def _remember_pairs(
        self,
        *,
        result: CorridorPlannerResult,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
    ) -> None:
        live_entries = self._pair_entries_from_segments(
            pair_segments=result.pair_segments,
            frame_id=frame_id,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
            now_sec=now_sec,
        )
        if not live_entries:
            return
        remembered_entries = self._active_pair_memory_entries(
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        merged = self._merge_pair_entries(
            remembered_entries=remembered_entries,
            live_entries=live_entries,
        )
        self._pair_memory = self._sort_pair_entries_by_forward_progress(
            entries=merged,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )

    def _update_midline_buffer(
        self,
        *,
        candidate_centerline: np.ndarray,
        candidate_source: str,
        candidate_update_ok: Optional[bool],
        candidate_update_reason: str = "ok",
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        result: CorridorPlannerResult,
        now_sec: float,
        support_centerline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        direct_commit = (
            candidate_source in {"validated", _CORRIDOR_MIDPOINT_SOURCE}
            and candidate_update_ok is not False
            and np.asarray(candidate_centerline, dtype=np.float64).shape[0] >= 2
        )
        return self._update_midline_memory_common(
            candidate_centerline=candidate_centerline,
            candidate_source=candidate_source,
            candidate_update_ok=candidate_update_ok,
            candidate_update_reason=candidate_update_reason,
            frame_id=frame_id,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
            now_sec=now_sec,
            support_centerline=support_centerline,
            direct_commit=direct_commit,
        )

    def _candidate_path_is_updateable(
        self,
        *,
        candidate_centerline: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        result: CorridorPlannerResult,
        candidate_source: str = "validated",
    ) -> tuple[bool, str]:
        min_points = (
            _CORRIDOR_MIDPOINT_MIN_POINTS
            if candidate_source == _CORRIDOR_MIDPOINT_SOURCE
            else self.candidate_min_points
        )
        if candidate_centerline.shape[0] < min_points:
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
        if candidate_local.shape[0] < min_points:
            return False, "candidate_no_local_path"
        if self._path_forward_extent_local(candidate_local) < self.candidate_min_extent_m:
            return False, "candidate_extent_too_short"
        if candidate_source == _CORRIDOR_MIDPOINT_SOURCE:
            return True, "ok"
        if candidate_source != "validated":
            return False, result.reject_reason or result.status or "unsupported_candidate_source"
        if result.status != "ok":
            return False, result.reject_reason or result.status
        return True, "ok"

    def _candidate_transition_metrics(
        self,
        *,
        candidate_centerline: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        horizon_m: float,
    ) -> dict[str, float]:
        empty = {
            "sample_count": 0.0,
            "lateral_max_m": float("inf"),
            "lateral_mean_m": float("inf"),
            "displacement_max_m": float("inf"),
            "displacement_mean_m": float("inf"),
            "heading_delta_rad": float("inf"),
        }
        if self._midline_buffer_path is None or self._midline_buffer_path.shape[0] < 2:
            return empty
        stored_forward = self._extract_forward_path_from_pose(
            path=self._midline_buffer_path,
            vehicle_xy=(vehicle_x, vehicle_y),
            resolution_m=self.midline_station_spacing_m,
        )
        candidate_forward = self._extract_forward_path_from_pose(
            path=candidate_centerline,
            vehicle_xy=(vehicle_x, vehicle_y),
            resolution_m=self.midline_station_spacing_m,
        )
        stored_path = (
            np.asarray(stored_forward, dtype=np.float64)
            if stored_forward is not None and stored_forward.shape[0] >= 2
            else np.asarray(self._midline_buffer_path, dtype=np.float64)
        )
        candidate_path = (
            np.asarray(candidate_forward, dtype=np.float64)
            if candidate_forward is not None and candidate_forward.shape[0] >= 2
            else np.asarray(candidate_centerline, dtype=np.float64)
        )
        if stored_path.shape[0] < 2 or candidate_path.shape[0] < 2:
            return empty

        stored_samples = self._resample_midline_stations(stored_path)
        candidate_samples = self._resample_midline_stations(candidate_path)
        count = min(stored_samples.shape[0], candidate_samples.shape[0])
        if count < 2:
            return empty

        # Use the actual sampled arc length rather than the configured station
        # spacing so the near-field acceptance gate remains correct even if the
        # sampled paths are irregular or test doubles bypass the resampler.
        horizon_limit_m = max(0.25, float(horizon_m))
        stored_limit = max(
            2,
            int(np.searchsorted(self._path_cumulative_lengths(stored_samples), horizon_limit_m, side="right")),
        )
        candidate_limit = max(
            2,
            int(np.searchsorted(self._path_cumulative_lengths(candidate_samples), horizon_limit_m, side="right")),
        )
        count = min(count, stored_limit, candidate_limit)
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
        if stored_local.shape[0] < 2 or candidate_local.shape[0] < 2:
            return empty

        delta = candidate_local - stored_local
        lateral = np.abs(delta[:, 1])
        displacement = np.hypot(delta[:, 0], delta[:, 1])
        stored_heading = self._path_start_heading_local(stored_local)
        candidate_heading = self._path_start_heading_local(candidate_local)
        heading_delta = abs(
            float(
                math.atan2(
                    math.sin(candidate_heading - stored_heading),
                    math.cos(candidate_heading - stored_heading),
                )
            )
        )
        return {
            "sample_count": float(count),
            "lateral_max_m": float(np.max(lateral)),
            "lateral_mean_m": float(np.mean(lateral)),
            "displacement_max_m": float(np.max(displacement)),
            "displacement_mean_m": float(np.mean(displacement)),
            "heading_delta_rad": heading_delta,
        }

    def _select_candidate_centerline(
        self,
        *,
        result: CorridorPlannerResult,
        support_chain: np.ndarray,
        memory_midpoint_chain: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple[np.ndarray, str]:
        candidates: list[tuple[float, int, str, np.ndarray]] = []

        def add_candidate(path: np.ndarray, source: str, priority: int) -> None:
            candidate = self._finite_corridor_path(path)
            if candidate.shape[0] < _CORRIDOR_MIDPOINT_MIN_POINTS:
                return
            extent = self._candidate_forward_extent_m(
                centerline=candidate,
                frame_id=frame_id,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
            )
            if extent <= 1e-6:
                return
            candidates.append((extent, priority, source, candidate))

        if result.status == "ok" and result.centerline.shape[0] > 0:
            add_candidate(result.centerline, "validated", 30)

        # The visible corridor anchor chain is the most literal "pair midpoint to
        # pair midpoint" path. Prefer it when it reaches farther than the fitted
        # core centerline, including remembered pairs beyond the current live fit.
        add_candidate(support_chain, _CORRIDOR_MIDPOINT_SOURCE, 20)
        add_candidate(result.prevalidation_centerline, _CORRIDOR_MIDPOINT_SOURCE, 10)
        add_candidate(memory_midpoint_chain, _CORRIDOR_MIDPOINT_SOURCE, 5)

        if not candidates:
            return np.empty((0, 2), dtype=np.float64), "none"

        best_extent = max(extent for extent, _, _, _ in candidates)
        extent_tolerance_m = max(0.05, 0.5 * float(self.midline_station_spacing_m))
        near_best = [
            candidate
            for candidate in candidates
            if candidate[0] >= (best_extent - extent_tolerance_m)
        ]
        _, _, source, path = max(near_best, key=lambda candidate: (candidate[1], candidate[0]))
        return np.array(path, copy=True), source

    @staticmethod
    def _finite_corridor_path(path: np.ndarray) -> np.ndarray:
        arr = np.asarray(path, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return np.empty((0, 2), dtype=np.float64)
        finite_mask = np.all(np.isfinite(arr), axis=1)
        if not np.any(finite_mask):
            return np.empty((0, 2), dtype=np.float64)
        return np.array(arr[finite_mask], copy=True)

    @staticmethod
    def _local_heading_delta_at_index(path_local: np.ndarray, idx: int) -> float:
        if path_local.shape[0] < 3:
            return 0.0
        lo = max(0, int(idx) - 1)
        hi = min(path_local.shape[0] - 1, int(idx) + 1)
        if hi - lo < 2:
            return 0.0
        first = path_local[lo + 1] - path_local[lo]
        second = path_local[hi] - path_local[hi - 1]
        first_norm = float(np.hypot(first[0], first[1]))
        second_norm = float(np.hypot(second[0], second[1]))
        if first_norm <= 1e-6 or second_norm <= 1e-6:
            return 0.0
        first /= first_norm
        second /= second_norm
        return abs(float(math.atan2((first[0] * second[1]) - (first[1] * second[0]), np.dot(first, second))))

    @staticmethod
    def _path_start_heading_local(path_local: np.ndarray) -> float:
        if path_local.shape[0] < 2:
            return 0.0
        delta = path_local[1] - path_local[0]
        if float(np.hypot(delta[0], delta[1])) <= 1e-6:
            return 0.0
        return float(math.atan2(delta[1], delta[0]))

    @staticmethod
    def _should_preserve_near_vehicle_lateral(local_path: np.ndarray, anchor_length_m: float) -> bool:
        if local_path.shape[0] < 2:
            return False
        near_mask = (local_path[:, 0] >= 0.0) & (local_path[:, 0] <= float(anchor_length_m))
        if not np.any(near_mask):
            return False
        near_points = np.asarray(local_path[near_mask], dtype=np.float64)
        lateral_max = float(np.max(np.abs(near_points[:, 1])))
        if lateral_max > _ANCHOR_TAPER_GATE_LATERAL_M:
            return False
        if near_points.shape[0] < 2:
            return True
        delta = near_points[min(1, near_points.shape[0] - 1)] - near_points[0]
        if float(np.hypot(delta[0], delta[1])) <= 1e-6:
            return True
        heading = abs(float(math.atan2(delta[1], delta[0])))
        return heading <= _ANCHOR_TAPER_GATE_HEADING_RAD

    def _corridor_analysis_metrics(
        self,
        *,
        raw_anchor_path: np.ndarray,
        prevalidation_centerline: np.ndarray,
        buffered_centerline: np.ndarray,
        control_path_local: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        samples_by_prefix = {
            "corridor_raw_anchor": self._corridor_analysis_sample_path_in_vehicle_frame(
                path=raw_anchor_path,
                frame_id=frame_id,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
            ),
            "corridor_prevalidation_centerline": self._corridor_analysis_sample_path_in_vehicle_frame(
                path=prevalidation_centerline,
                frame_id=frame_id,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
            ),
            "corridor_buffer_centerline": self._corridor_analysis_sample_path_in_vehicle_frame(
                path=buffered_centerline,
                frame_id=frame_id,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
            ),
            "corridor_control_path": self._corridor_analysis_sample_local_path(
                np.asarray(control_path_local, dtype=np.float64)
            ),
        }
        for prefix, samples in samples_by_prefix.items():
            metrics[f"{prefix}_point_count"] = float(samples.shape[0])
            for idx in range(_CORRIDOR_ANALYSIS_SAMPLE_COUNT):
                if idx < samples.shape[0]:
                    metrics[f"{prefix}_p{idx}_x_m"] = float(samples[idx, 0])
                    metrics[f"{prefix}_p{idx}_y_m"] = float(samples[idx, 1])
                else:
                    metrics[f"{prefix}_p{idx}_x_m"] = float("nan")
                    metrics[f"{prefix}_p{idx}_y_m"] = float("nan")
        return metrics

    def _corridor_analysis_sample_path_in_vehicle_frame(
        self,
        *,
        path: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> np.ndarray:
        if path.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        if self._is_alias(frame_id, self.odom_frame):
            forward = self._extract_forward_path_from_pose(
                path=np.asarray(path, dtype=np.float64),
                vehicle_xy=(vehicle_x, vehicle_y),
                resolution_m=min(
                    float(self.midline_station_spacing_m),
                    float(_CORRIDOR_ANALYSIS_SAMPLE_SPACING_M),
                ),
            )
            working = (
                np.asarray(forward, dtype=np.float64)
                if forward is not None and forward.shape[0] > 0
                else np.asarray(path, dtype=np.float64)
            )
            local = self._centerline_to_vehicle_frame(
                centerline=working,
                frame_id=frame_id,
                vehicle_x=vehicle_x,
                vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw,
            )
            return self._corridor_analysis_sample_local_path(local)
        if self._is_alias(frame_id, self.base_frame):
            return self._corridor_analysis_sample_local_path(np.asarray(path, dtype=np.float64))
        return np.empty((0, 2), dtype=np.float64)

    def _corridor_analysis_sample_local_path(self, path_local: np.ndarray) -> np.ndarray:
        path_local = np.asarray(path_local, dtype=np.float64)
        if path_local.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        keep_mask = np.isfinite(path_local[:, 0]) & np.isfinite(path_local[:, 1]) & (path_local[:, 0] >= -0.1)
        local = path_local[keep_mask]
        if local.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        if local.shape[0] == 1:
            return np.asarray(local[:1], dtype=np.float64)

        cumulative = self._path_cumulative_lengths(local)
        max_distance_m = float(_CORRIDOR_ANALYSIS_SAMPLE_SPACING_M) * float(_CORRIDOR_ANALYSIS_SAMPLE_COUNT - 1)
        total = min(float(cumulative[-1]), max_distance_m)
        if total <= 1e-6:
            return np.asarray(local[:1], dtype=np.float64)
        samples = np.arange(
            0.0,
            total + 1e-9,
            float(_CORRIDOR_ANALYSIS_SAMPLE_SPACING_M),
            dtype=np.float64,
        )
        if samples.size == 0 or samples[-1] < total:
            samples = np.concatenate((samples, [total]))
        sampled = self._sample_path_at_lengths(local, cumulative, samples)
        return np.asarray(sampled[:_CORRIDOR_ANALYSIS_SAMPLE_COUNT], dtype=np.float64)

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
        self._active_cone_audit_counts = self._empty_cone_audit_counts()
        self._publish_cone_audit_markers(
            frame_id=frame_id,
            stamp=self.get_clock().now().to_msg(),
            entries=[],
        )
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

    def _normalize_core_reject_reason(self, result: Optional[CorridorPlannerResult]) -> str:
        if result is None:
            return "none"
        reject_counts = result.reject_counts or {}
        if int(reject_counts.get("near_field_continuity", 0)) > 0:
            return "near_field_continuity"
        if int(reject_counts.get("heading", 0)) > 0:
            return "midpoint_kink"
        text = (result.reject_reason or result.status or "").strip().lower()
        if text.startswith("usable cones below minimum"):
            return "no_safe_chain"
        if text in {
            "no cones available",
            "no colored cones in planning region",
            "no reliable corridor boundaries",
            "no valid corridor overlap",
            "too few valid corridor samples",
            "path exits corridor",
            "path curvature exceeded limit",
            "path self-crossing detected",
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
        del centerline_point_count
        del cmd_speed
        del cmd_steering
        del lookahead
        del seed_midpoint_distance_m
        del near_field_lateral_max_m
        del near_field_midpoint_kink_max_rad
        del hold_remaining_s
        return "\n".join(
            [
                f"STATE: {operator_state.upper()}",
                f"MODE: {self._active_planner_mode.upper()}",
                f"REASON: {self._operator_reason_label(operator_reason)}",
                self._lap_status_text(),
            ]
        )

    def _lap_status_text(self) -> str:
        if self.lap_tracking_target_laps > 0:
            return f"LAPS: {int(self._lap_tracking_completed_laps)}/{int(self.lap_tracking_target_laps)}"
        return f"LAPS: {int(self._lap_tracking_completed_laps)}/off"

    def _build_markers(
        self,
        *,
        now,
        frame_id: str,
        result: Optional[CorridorPlannerResult],
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
        raw_left_chain = np.asarray(result.raw_left_chain_points, dtype=np.float64)
        raw_right_chain = np.asarray(result.raw_right_chain_points, dtype=np.float64)
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
                    ns="raw_chain_left",
                    points=raw_left_chain,
                    color=(0.05, 0.25, 1.0, 0.95),
                    width=0.06,
                    z_offset=0.22,
                )
            )
            marker_id += 1
            raw_left_points_marker = self._make_points_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns="raw_chain_left_points",
                points=raw_left_chain,
                color=(0.05, 0.25, 1.0, 1.0),
                scale=0.34,
            )
            for point in raw_left_points_marker.points:
                point.z = 0.23
            arr.markers.append(raw_left_points_marker)
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
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="raw_chain_right",
                    points=raw_right_chain,
                    color=(1.0, 0.82, 0.0, 0.95),
                    width=0.06,
                    z_offset=0.24,
                )
            )
            marker_id += 1
            raw_right_points_marker = self._make_points_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns="raw_chain_right_points",
                points=raw_right_chain,
                color=(1.0, 0.82, 0.0, 1.0),
                scale=0.34,
            )
            for point in raw_right_points_marker.points:
                point.z = 0.25
            arr.markers.append(raw_right_points_marker)
            marker_id += 1

        if self.show_pair_lines:
            arr.markers.append(
                self._make_pair_segment_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="corridor_rungs",
                    pair_segments=self._current_pair_segments_for_viz,
                    color=(0.2, 1.0, 0.3, 0.95),
                    width=0.07,
                )
            )
            marker_id += 1

        if self.show_corridor_pair_audit:
            marker_id = self._append_corridor_pair_audit_markers(
                arr=arr,
                marker_id=marker_id,
                frame_id=frame_id,
                stamp=now,
                result=result,
            )

        if self.show_raw_midpoint_chain:
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns="corridor_center_anchors",
                    points=raw_midpoint_chain,
                    color=(1.0, 1.0, 1.0, 0.95),
                    width=0.06,
                    z_offset=0.03,
                )
            )
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
                width=_CENTERLINE_MARKER_WIDTH_M,
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

    def _append_corridor_pair_audit_markers(
        self,
        *,
        arr: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: CorridorPlannerResult,
    ) -> int:
        segments = np.asarray(result.corridor_pair_audit_segments, dtype=np.float64)
        reasons = list(result.corridor_pair_audit_reasons)
        widths = np.asarray(result.corridor_pair_audit_widths_m, dtype=np.float64)
        anchors_local = np.asarray(result.corridor_pair_audit_anchors_local, dtype=np.float64)
        count = min(len(reasons), segments.shape[0], widths.size, anchors_local.shape[0])
        if count <= 0:
            return marker_id

        for reason in _CORRIDOR_PAIR_AUDIT_REASONS:
            if reason == "pair_valid":
                continue
            reason_indices = [
                idx
                for idx in range(count)
                if reasons[idx] == reason and np.all(np.isfinite(segments[idx]))
            ]
            if not reason_indices:
                continue
            arr.markers.append(
                self._make_pair_segment_marker(
                    frame_id=frame_id,
                    stamp=stamp,
                    marker_id=marker_id,
                    ns=f"corridor_pair_audit_{reason}",
                    pair_segments=segments[reason_indices],
                    color=self._corridor_pair_audit_rgba(reason),
                    width=0.05,
                )
            )
            marker_id += 1

        if not self.corridor_pair_audit_show_labels:
            return marker_id

        label_count = 0
        for idx in range(count):
            reason = reasons[idx]
            if reason == "pair_valid" or not np.all(np.isfinite(segments[idx])):
                continue
            midpoint = 0.5 * (segments[idx, 0, :] + segments[idx, 1, :])
            if not np.all(np.isfinite(midpoint)):
                continue
            label = Marker()
            label.header.frame_id = frame_id
            label.header.stamp = stamp
            label.ns = "corridor_pair_audit_labels"
            label.id = marker_id
            marker_id += 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(midpoint[0])
            label.pose.position.y = float(midpoint[1])
            label.pose.position.z = 0.85
            label.pose.orientation.w = 1.0
            label.scale.z = 0.14
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 0.95
            label.text = (
                f"pair[{idx}] {reason}\n"
                f"w={float(widths[idx]):.2f}m "
                f"local=({float(anchors_local[idx, 0]):.1f},"
                f"{float(anchors_local[idx, 1]):.1f})"
            )
            arr.markers.append(label)
            label_count += 1
            if label_count >= self.corridor_pair_audit_max_labels:
                break
        return marker_id

    @staticmethod
    def _corridor_pair_audit_rgba(reason: str) -> tuple[float, float, float, float]:
        if reason == "pair_width_too_narrow":
            return 1.0, 0.05, 0.05, 0.9
        if reason == "pair_width_too_wide":
            return 1.0, 0.45, 0.05, 0.9
        if reason in {"pair_left_beyond_horizon", "pair_right_beyond_horizon"}:
            return 0.0, 0.85, 1.0, 0.9
        if reason in {"pair_left_behind", "pair_right_behind"}:
            return 1.0, 0.0, 0.85, 0.9
        if reason == "pair_not_in_longest_valid_slice":
            return 0.85, 0.85, 0.85, 0.75
        return 0.65, 0.2, 1.0, 0.9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CorridorPlannerNode()
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
