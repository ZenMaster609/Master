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
from vehicle_plotter_msgs.msg import ConeDetectionArray
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.planner_runtime_types import PlannerIdentity
from sim_car.planning.planner_constants import (
    MSG_TRACK_STATE_CONFIRMED,
    MSG_TRACK_STATE_STALE,
    MSG_TRACK_STATE_TENTATIVE,
    PAIR_PASSED_MARGIN_M as _PAIR_PASSED_MARGIN_M,
    VALIDATED_JUMP_ACCEPT_HEADING_DELTA_RAD as _VALIDATED_JUMP_ACCEPT_HEADING_DELTA_RAD,
    VALIDATED_JUMP_ACCEPT_HORIZON_M as _VALIDATED_JUMP_ACCEPT_HORIZON_M,
    VALIDATED_JUMP_ACCEPT_LATERAL_MAX_M as _VALIDATED_JUMP_ACCEPT_LATERAL_MAX_M,
    VALIDATED_JUMP_ACCEPT_LATERAL_MEAN_M as _VALIDATED_JUMP_ACCEPT_LATERAL_MEAN_M,
)
from sim_car.planning.tracked_cone_planner_contract import (
    apply_common_config_to_node,
    declare_tracked_cone_planner_parameters,
    read_migrated_tracked_cone_planner_common_config,
)
from sim_car.planning.corridor_planner_core import (
    CorridorPlannerConfig,
    CorridorPlannerPrior,
    CorridorPlannerResult,
    compute_corridor_centerline,
    update_track_width_estimate,
)
from sim_car.planning.planning_visualization import CORRIDOR_PAIR_AUDIT_REASONS
from sim_car.planning.tracked_cone_planner_base import TrackedConePlannerBase
from sim_car.planning.tracked_cone_planner_geometry import (
    _base_point_to_odom,
    _odom_point_to_base,
)

_CORRIDOR_MIDPOINT_SOURCE = "corridor_midpoints"
_CORRIDOR_MIDPOINT_MIN_POINTS = 2  # Two points are required to form a drawable midpoint path.
_CORRIDOR_ANALYSIS_SAMPLE_COUNT = 8  # Eight samples keep diagnostics compact while showing near-field shape.
_CORRIDOR_ANALYSIS_SAMPLE_SPACING_M = 1.0  # One-meter spacing matches the operator-scale path preview.
_ANCHOR_TAPER_GATE_LATERAL_M = 0.20  # Allows small lateral drift while detecting off-axis starts.
_ANCHOR_TAPER_GATE_HEADING_RAD = 0.18  # Roughly ten degrees keeps the near-vehicle anchor stable.
_NANOSECONDS_TO_SECONDS = 1e-9  # ROS clock timestamps are nanoseconds; planner metrics use seconds.
_CANDIDATE_EXTENT_EPSILON_M = 1e-6  # Treats sub-micrometer forward extent as no usable path.
_CANDIDATE_EXTENT_TOLERANCE_MIN_M = 0.05  # Five centimeters avoids winner flips from resampling noise.
_CANDIDATE_EXTENT_TOLERANCE_STATION_FACTOR = 0.5  # Half-station tolerance preserves prior selection behavior.
_VALIDATED_CANDIDATE_PRIORITY = 30  # Validated core paths win ties when their extent is comparable.
_SUPPORT_CHAIN_CANDIDATE_PRIORITY = 20  # Live pair midpoints are preferred over stale midpoint fallbacks.
_PREVALIDATION_CANDIDATE_PRIORITY = 10  # Prevalidation paths are useful only behind live support chains.
_MEMORY_CHAIN_CANDIDATE_PRIORITY = 5  # Remembered pairs are the weakest fallback source.
_TRANSITION_MIN_PATH_POINTS = 2  # Two samples are required to measure displacement and heading change.
_TRANSITION_MIN_HORIZON_M = 0.25  # Keeps the near-field comparison meaningful for very short horizons.
_PAIR_SEGMENT_ENDPOINT_COUNT = 2  # Pair memory stores left and right cone endpoints only.
_PAIR_SEGMENT_COORD_COUNT = 2  # Cone endpoints are planar x/y coordinates.
_PAIR_MIDPOINT_WEIGHT = 0.5  # Midpoint is the average of the left and right endpoints.
_PAIR_MEMORY_MERGE_DISTANCE_M = 0.35  # Merges remembered/live pairs that describe the same corridor gap.
_PAIR_SORT_MIN_STEP_M = 1e-6  # Avoids unstable ordering from duplicate midpoint coordinates.
_PAIR_SORT_RANGE_REGRESSION_TOLERANCE_M = 0.20  # Allows small range regressions on curved paths.
_PAIR_SORT_BACKWARD_GATE_M = 0.75  # Strongly penalizes steps that move behind the vehicle.
_PAIR_SORT_INITIAL_HEADING = np.asarray([1.0, 0.0], dtype=np.float64)
_CONE_AUDIT_MARKER_START_ID = 1  # ID zero is reserved for the DELETEALL marker.
_CONE_AUDIT_MAIN_Z_M = 0.18  # Raises cone markers above ground enough to remain visible.
_CONE_AUDIT_HALO_Z_M = 0.03  # Keeps stale halos flat against the ground plane.
_CONE_AUDIT_HALO_DIAMETER_M = 0.55  # Halo is wider than a cone marker so stale tracks stand out.
_CONE_AUDIT_HALO_HEIGHT_M = 0.03  # Thin cylinder reads as a ground ring in RViz.
_CONE_AUDIT_LABEL_Z_M = 0.75  # Labels sit above cones to reduce overlap with markers.
_CONE_AUDIT_LABEL_HEIGHT_M = 0.16  # Label text remains readable without dominating the view.
_CONE_AUDIT_HALO_RGBA = (0.1, 0.95, 1.0, 0.55)
_CONE_AUDIT_LABEL_RGBA = (1.0, 1.0, 1.0, 0.95)
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
class _ConeAuditContext:
    points: np.ndarray
    local_points: np.ndarray
    used_left: set[int]
    used_right: set[int]
    chain_rejection_reasons_by_track_id: dict[int, str]


@dataclass
class _PairMemoryEntry:
    left_x_odom: float
    left_y_odom: float
    right_x_odom: float
    right_y_odom: float
    midpoint_x_odom: float
    midpoint_y_odom: float
    last_valid_sec: float


@dataclass
class _InputMetrics:
    raw_orange_count: int
    boundary_hint_count: int
    resolved_blue_count: int
    resolved_yellow_count: int
    planning_frame: object  # TrackedConePlanningFrame


@dataclass
class _MidlineUpdateResult:
    centerline: np.ndarray
    buffered_centerline: np.ndarray
    raw_midpoint_chain: np.ndarray
    pair_segments_for_viz: np.ndarray
    publish_mode: str
    candidate_update_ok: bool
    candidate_update_reason: str
    status: str
    hold_reason: str
    plan_hold_active: bool


@dataclass
class _ControllerOutput:
    control_path: np.ndarray
    control_target_frame: Optional[np.ndarray]
    control_debug_metrics: Optional[dict]
    cmd_speed: float
    cmd_steering: float
    lookahead: float
    zero_cmd_sent_flag: int
    controller_failed: bool
    control_path_point_count: int


@dataclass
class _TransitionPaths:
    stored_path: np.ndarray
    candidate_path: np.ndarray


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
        declare_tracked_cone_planner_parameters(
            self,
            diagnostics_topic_default=self._planner_identity.diagnostics_topic,
            defaults_override={"boundary_chain.max_heading_change_rad": 2.35},
        )
        defaults = {}
        defaults.update({
            "filtering.max_lateral_range_m": 14.0,
            "boundary_chain.min_chain_length": 3,
            "width_estimation.min_trustworthy_pairs": 3,
            "corridor.min_corridor_width_m": 2.2,
            "corridor.max_corridor_width_m": 6.4,
            "corridor.boundary_resample_dx": 0.5,
            "corridor.min_required_corridor_samples": 5,
            "corridor.path_fit_smoothing_window": 5,
            "corridor.membership_margin_m": 0.15,
            "midline_memory.pair_memory_retention_s": 12.0,
            "validation.min_path_points": 4,
            "validation.min_forward_extent_m": 2.0,
            "validation.max_heading_delta_rad": 0.75,
            "validation.max_initial_heading_error_rad": 3.0 * math.pi / 4.0,
            "validation.max_curvature": 0.45,
            "debug.enable_cone_audit_markers": False,
            "debug.cone_audit_viz_topic": "/corridor_planner/cone_audit_viz",
            "debug.cone_audit_show_labels": True,
            "debug.cone_audit_max_labels": 80,
            "debug.show_corridor_pair_audit": False,
            "debug.corridor_pair_audit_show_labels": True,
            "debug.corridor_pair_audit_max_labels": 80,
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
        self._read_runtime_parameters()
        self._core_config = self._build_core_config()
        self._filtered_track_width_m = float(self._core_config.initial_width_m)

    def _read_runtime_parameters(self) -> None:
        self.pair_memory_retention_s = max(
            self.midline_hold_last_valid_duration_s,
            float(self.get_parameter("midline_memory.pair_memory_retention_s").value),
        )
        self.enable_cone_audit_markers = bool(self.get_parameter("debug.enable_cone_audit_markers").value)
        self.cone_audit_viz_topic = (
            str(self.get_parameter("debug.cone_audit_viz_topic").value).strip()
            or "/corridor_planner/cone_audit_viz"
        )
        self.cone_audit_show_labels = bool(self.get_parameter("debug.cone_audit_show_labels").value)
        self.cone_audit_max_labels = max(0, int(self.get_parameter("debug.cone_audit_max_labels").value))
        self.show_corridor_pair_audit = bool(self.get_parameter("debug.show_corridor_pair_audit").value)
        self.corridor_pair_audit_show_labels = bool(
            self.get_parameter("debug.corridor_pair_audit_show_labels").value
        )
        self.corridor_pair_audit_max_labels = max(
            0,
            int(self.get_parameter("debug.corridor_pair_audit_max_labels").value),
        )

    def _build_core_config(self) -> CorridorPlannerConfig:
        values = {}
        values.update(self._filtering_config_values())
        values.update(self._boundary_chain_and_width_config_values())
        values.update(self._corridor_and_centerline_config_values())
        values.update(self._validation_config_values())
        return CorridorPlannerConfig(**values)

    def _filtering_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            "max_cone_range_m": profile.max_cone_range_m,
            "planning_horizon_m": profile.planning_horizon_m,
            "max_lateral_range_m": float(self.get_parameter("filtering.max_lateral_range_m").value),
            "behind_drop_m": profile.behind_drop_m,
            "min_confidence": profile.min_confidence,
            "min_required_cones": max(2, profile.min_required_cones),
        }

    def _boundary_chain_and_width_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            "min_step_m": profile.boundary_min_step_m,
            "max_step_m": profile.boundary_max_step_m,
            "max_heading_change_rad": float(self.get_parameter("boundary_chain.max_heading_change_rad").value),
            "min_forward_progress_m": profile.boundary_min_forward_progress_m,
            "min_chain_length": max(2, int(self.get_parameter("boundary_chain.min_chain_length").value)),
            "initial_width_m": profile.initial_width_m,
            "min_width_m": profile.min_width_m,
            "max_width_m": profile.max_width_m,
            "width_filter_alpha": profile.width_filter_alpha,
            "max_width_delta_per_update_m": profile.max_width_delta_per_update_m,
            "min_trustworthy_pairs": max(1, int(self.get_parameter("width_estimation.min_trustworthy_pairs").value)),
        }

    def _corridor_and_centerline_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            "boundary_resample_dx": float(self.get_parameter("corridor.boundary_resample_dx").value),
            "min_corridor_width_m": float(self.get_parameter("corridor.min_corridor_width_m").value),
            "max_corridor_width_m": float(self.get_parameter("corridor.max_corridor_width_m").value),
            "min_required_corridor_samples": max(
                2,
                int(self.get_parameter("corridor.min_required_corridor_samples").value),
            ),
            "path_fit_smoothing_window": max(1, int(self.get_parameter("corridor.path_fit_smoothing_window").value)),
            "membership_margin_m": float(self.get_parameter("corridor.membership_margin_m").value),
            "path_resolution_m": profile.centerline_path_resolution_m,
            "max_path_length_m": profile.max_path_length_m,
        }

    def _validation_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            "min_path_points": max(2, int(self.get_parameter("validation.min_path_points").value)),
            "min_forward_extent_m": float(self.get_parameter("validation.min_forward_extent_m").value),
            "jump_check_horizon_m": profile.jump_check_horizon_m,
            "max_near_field_lateral_jump_m": profile.max_near_field_lateral_jump_m,
            "max_heading_delta_rad": float(self.get_parameter("validation.max_heading_delta_rad").value),
            "max_initial_heading_error_rad": float(
                self.get_parameter("validation.max_initial_heading_error_rad").value
            ),
            "max_curvature": float(self.get_parameter("validation.max_curvature").value),
        }

    def _on_timer(self) -> None:
        if (ctx := self._resolve_cone_planning_context()) is None:
            return
        cones_msg, target_frame, vehicle_x, vehicle_y, vehicle_yaw, points_xy, colors, confidences = ctx
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        self._update_smalltrack_lap_from_orange_cones(
            cones_msg=cones_msg, points_xy=points_xy,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        self._update_remembered_cone_viz(points_xy=points_xy, colors=colors)
        input_metrics = self._collect_input_metrics(cones_msg, points_xy, colors, confidences)
        result = self._run_corridor_planner(input_metrics.planning_frame, vehicle_x, vehicle_y, vehicle_yaw)
        cone_audit_entries = self._process_cone_audit(
            cones_msg, input_metrics.planning_frame, result,
            target_frame, vehicle_x, vehicle_y, vehicle_yaw, now_sec,
        )
        pair_segs_viz, raw_midpoint_chain, combined_midpoint_chain = self._resolve_pair_geometry(
            result, target_frame, vehicle_x, vehicle_y, vehicle_yaw, now_sec,
        )
        raw_centerline, candidate_source = self._select_candidate_centerline(
            result=result, support_chain=raw_midpoint_chain,
            memory_midpoint_chain=combined_midpoint_chain,
            frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        result.planner_mode = self._planner_identity.planner_mode
        del points_xy
        midline = self._apply_midline_updates(
            result, raw_centerline, candidate_source, raw_midpoint_chain,
            pair_segs_viz, target_frame, vehicle_x, vehicle_y, vehicle_yaw, now_sec,
        )
        live_center_anchors = np.array(result.midpoints_raw, copy=True)
        ctrl = self._run_controller(
            midline.centerline, midline.buffered_centerline, live_center_anchors,
            result.prevalidation_centerline, target_frame, vehicle_x, vehicle_y, vehicle_yaw,
        )
        hold_remaining_s = self._hold_remaining_s(now_sec)
        operator_state, operator_reason = self._determine_operator_state(
            result, midline, ctrl, hold_remaining_s,
        )
        self._update_active_state_counters(result, midline, operator_state)
        self._log_cycle_state(operator_state, operator_reason, hold_remaining_s, result)
        diag_metrics = self._build_diagnostics_metrics(
            result, midline, ctrl, input_metrics, operator_state, operator_reason,
            hold_remaining_s, cone_audit_entries,
        )
        self._publish_cycle_results(
            target_frame, raw_centerline, result, midline, ctrl,
            diag_metrics, operator_state, operator_reason, hold_remaining_s,
        )

    # ------------------------------------------------------------------
    # _on_timer stage helpers
    # ------------------------------------------------------------------

    def _collect_input_metrics(
        self,
        cones_msg,
        points_xy: np.ndarray,
        colors: list,
        confidences: np.ndarray,
    ) -> _InputMetrics:
        raw_orange_count = sum(
            1 for cone in cones_msg.cones if normalize_color(getattr(cone, "color", "")) == "orange"
        )
        boundary_hint_count = sum(
            1 for cone in cones_msg.cones if str(getattr(cone, "boundary_color", "")).strip()
        )
        resolved_blue_count = sum(1 for color in colors if color == "blue")
        resolved_yellow_count = sum(1 for color in colors if color == "yellow")
        planning_frame = self._tracked_cone_planning_frame(
            msg=cones_msg, points_xy=points_xy, colors=colors, confidences=confidences,
        )
        self._active_remembered_cone_count = int(len(cones_msg.cones))
        self._active_stale_cone_count = int(
            np.count_nonzero(planning_frame.track_states == MSG_TRACK_STATE_STALE)
        )
        return _InputMetrics(
            raw_orange_count=raw_orange_count,
            boundary_hint_count=boundary_hint_count,
            resolved_blue_count=resolved_blue_count,
            resolved_yellow_count=resolved_yellow_count,
            planning_frame=planning_frame,
        )

    def _run_corridor_planner(
        self,
        planning_frame,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> "CorridorPlannerResult":
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
                self._filtered_track_width_m, result.selected_chain_width_median, self._core_config,
            )
        result.filtered_track_width_m = float(self._filtered_track_width_m)
        return result

    def _process_cone_audit(
        self,
        cones_msg,
        planning_frame,
        result,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
    ) -> list:
        cone_audit_entries = self._build_cone_audit_entries(
            msg=cones_msg, planning_frame=planning_frame, result=result,
            vehicle_xy=(vehicle_x, vehicle_y), vehicle_yaw=vehicle_yaw, now_sec=now_sec,
        )
        self._active_cone_audit_counts = self._cone_audit_counts(cone_audit_entries)
        self._publish_cone_audit_markers(
            frame_id=target_frame, stamp=self.get_clock().now().to_msg(), entries=cone_audit_entries,
        )
        return cone_audit_entries

    def _resolve_pair_geometry(
        self,
        result,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
    ) -> tuple:
        remembered_pair_entries = self._active_pair_memory_entries(
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        raw_midpoint_chain = np.array(result.midpoints_raw, copy=True)
        pair_segments_for_viz = np.array(result.pair_segments, copy=True)
        live_pair_entries = self._pair_entries_from_segments(
            pair_segments=result.pair_segments, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw, now_sec=now_sec,
        )
        combined_pair_entries = self._merge_pair_entries(
            remembered_entries=remembered_pair_entries, live_entries=live_pair_entries,
        )
        combined_pair_entries = self._sort_pair_entries_by_forward_progress(
            entries=combined_pair_entries,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        combined_pair_segments, combined_midpoint_chain = self._pair_geometry_from_memory(combined_pair_entries)
        if combined_pair_segments.size > 0:
            pair_segments_for_viz = combined_pair_segments
        if combined_midpoint_chain.size > 0:
            raw_midpoint_chain = combined_midpoint_chain
        return pair_segments_for_viz, raw_midpoint_chain, combined_midpoint_chain

    def _apply_midline_updates(  # noqa: PLR0913
        self, result, raw_centerline, candidate_source, raw_midpoint_chain,
        pair_segments_for_viz, target_frame, vehicle_x, vehicle_y, vehicle_yaw, now_sec,
    ) -> _MidlineUpdateResult:
        centerline, buffered_centerline, candidate_update_ok, candidate_update_reason = (
            self._buffer_and_anchor_centerline(
                result, raw_centerline, candidate_source, raw_midpoint_chain,
                target_frame, vehicle_x, vehicle_y, vehicle_yaw, now_sec,
            )
        )
        status, pair_segments_for_viz, raw_midpoint_chain = self._build_midline_status(
            result, raw_centerline, centerline, candidate_source, candidate_update_ok,
            candidate_update_reason, pair_segments_for_viz, raw_midpoint_chain,
            target_frame, vehicle_x, vehicle_y, vehicle_yaw, now_sec,
        )
        centerline, publish_mode, plan_hold_active, hold_reason, status = self._apply_hold_logic(
            result, centerline, candidate_update_ok, candidate_update_reason, status, now_sec,
            pair_segments_for_viz, raw_midpoint_chain,
        )
        pair_segments_for_viz, raw_midpoint_chain = self._refresh_hold_viz(
            plan_hold_active, pair_segments_for_viz, raw_midpoint_chain,
        )
        centerline = self._prepare_centerline_for_current_pose(
            centerline=centerline, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        centerline = self._finalize_fresh_centerline(
            centerline, raw_midpoint_chain, pair_segments_for_viz, result, publish_mode, now_sec,
        )
        if publish_mode == "fresh" and centerline.shape[0] > 0 and pair_segments_for_viz.size > 0:
            self._last_valid_pair_segments = np.array(pair_segments_for_viz, copy=True)
        return _MidlineUpdateResult(
            centerline=centerline, buffered_centerline=buffered_centerline,
            raw_midpoint_chain=raw_midpoint_chain, pair_segments_for_viz=pair_segments_for_viz,
            publish_mode=publish_mode, candidate_update_ok=candidate_update_ok,
            candidate_update_reason=candidate_update_reason, status=status,
            hold_reason=hold_reason, plan_hold_active=plan_hold_active,
        )

    def _refresh_hold_viz(
        self,
        plan_hold_active: bool,
        pair_segments_for_viz: np.ndarray,
        raw_midpoint_chain: np.ndarray,
    ) -> tuple:
        if plan_hold_active:
            if self._last_valid_pair_segments is not None:
                pair_segments_for_viz = np.array(self._last_valid_pair_segments, copy=True)
            if self._last_valid_raw_midpoint_chain is not None and raw_midpoint_chain.size == 0:
                raw_midpoint_chain = np.array(self._last_valid_raw_midpoint_chain, copy=True)
        return pair_segments_for_viz, raw_midpoint_chain

    def _finalize_fresh_centerline(
        self,
        centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        pair_segments_for_viz: np.ndarray,
        result,
        publish_mode: str,
        now_sec: float,
    ) -> np.ndarray:
        if self.enable_temporal_smoothing and publish_mode == "fresh":
            centerline = self._apply_temporal_smoothing(centerline)
            self._previous_centerline = np.array(centerline, copy=True)
        elif publish_mode == "held" and centerline.shape[0] > 0:
            self._previous_centerline = np.array(centerline, copy=True)
        if publish_mode == "fresh" and centerline.shape[0] > 0:
            self._record_valid_plan(
                now_sec=now_sec, centerline=centerline, raw_midpoint_chain=raw_midpoint_chain,
                selected_chain_width_median=result.selected_chain_width_median,
            )
        return centerline

    def _run_controller(
        self,
        centerline: np.ndarray,
        buffered_centerline: np.ndarray,
        live_center_anchors: np.ndarray,
        prevalidation_centerline,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> _ControllerOutput:
        control_path = self._centerline_to_vehicle_frame(
            centerline=centerline, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        self._last_corridor_analysis_metrics = self._corridor_analysis_metrics(
            raw_anchor_path=live_center_anchors, prevalidation_centerline=prevalidation_centerline,
            buffered_centerline=buffered_centerline, control_path_local=control_path,
            frame_id=target_frame, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        ctrl_target, ctrl_debug, cmd_speed, cmd_steering, lookahead, zero_flag, failed = (
            self._dispatch_controller(control_path, target_frame, vehicle_x, vehicle_y, vehicle_yaw)
        )
        return _ControllerOutput(
            control_path=control_path, control_target_frame=ctrl_target,
            control_debug_metrics=ctrl_debug, cmd_speed=cmd_speed,
            cmd_steering=cmd_steering, lookahead=lookahead,
            zero_cmd_sent_flag=zero_flag, controller_failed=failed,
            control_path_point_count=int(control_path.shape[0]),
        )

    def _dispatch_controller(
        self,
        control_path: np.ndarray,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple:
        if control_path.shape[0] >= 1 and self._controller is not None:
            return self._execute_controller(control_path, target_frame, vehicle_x, vehicle_y, vehicle_yaw)
        if control_path.shape[0] >= 1:
            return None, None, 0.0, 0.0, 0.0, int(self._apply_controller_disabled_behavior()), False
        zero_flag = int(self._apply_no_path_behavior())
        cmd_speed = float(self._last_speed_cmd) if self._last_speed_cmd is not None else 0.0
        cmd_steering = float(self._last_steering_cmd) if self._last_steering_cmd is not None else 0.0
        return None, None, cmd_speed, cmd_steering, 0.0, zero_flag, False

    def _resolve_control_target_frame(
        self,
        control_target_base: np.ndarray,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> Optional[np.ndarray]:
        if self._is_alias(target_frame, self.base_frame):
            return np.array(control_target_base, copy=True)
        if self._is_alias(target_frame, self.odom_frame):
            tx, ty = _base_point_to_odom(
                float(control_target_base[0]), float(control_target_base[1]),
                vehicle_x, vehicle_y, vehicle_yaw,
            )
            return np.array([tx, ty], dtype=np.float64)
        return None

    def _determine_operator_state(
        self, result, midline: _MidlineUpdateResult, ctrl: _ControllerOutput,
        hold_remaining_s: float,
    ) -> tuple:
        centerline = midline.centerline
        publish_mode = midline.publish_mode
        core_reject_reason = self._normalize_core_reject_reason(result)
        if centerline.shape[0] == 0:
            if self._last_valid_centerline is not None and hold_remaining_s <= 0.0:
                return "stopped", "hold_expired_no_path"
            if core_reject_reason != "none":
                return "stopped", core_reject_reason
            if ctrl.zero_cmd_sent_flag:
                return "stopped", "stop_if_no_path"
            return "held", "holding_previous_valid"
        if publish_mode == "held":
            if "hysteresis holding previous valid centerline" in midline.status:
                return "held", "hysteresis_holding"
            if core_reject_reason != "none":
                return "held", core_reject_reason
            return "held", "holding_previous_valid"
        if ctrl.controller_failed:
            return "stopped", "controller_compute_failed"
        if self._controller is None and centerline.shape[0] > 0:
            return "stopped", "controller_disabled"
        if centerline.shape[0] > 0 and ctrl.control_path_point_count <= 0:
            return "stopped", "no_control_path"
        if ctrl.zero_cmd_sent_flag:
            return "stopped", "stop_if_no_path"
        if ctrl.control_path_point_count > 0 and centerline.shape[0] > 0 and ctrl.zero_cmd_sent_flag == 0:
            return "fresh", "none"
        return "fresh", "none"

    def _update_active_state_counters(self, result, midline: _MidlineUpdateResult, operator_state: str) -> None:
        self._active_planner_mode = (
            "holding_last_valid"
            if midline.publish_mode == "held" and midline.centerline.shape[0] > 0
            else result.planner_mode
        )
        self._active_left_chain_length = int(result.left_chain_length)
        self._active_right_chain_length = int(result.right_chain_length)
        self._active_pair_count = (
            int(midline.pair_segments_for_viz.shape[0])
            if midline.pair_segments_for_viz.size > 0
            else int(result.accepted_pair_count)
        )
        self._active_unknown_pair_count = int(result.unknown_pair_count)
        self._active_filtered_track_width_m = float(self._filtered_track_width_m)
        self._active_held_path_flag = (
            1 if midline.publish_mode == "held" and midline.centerline.shape[0] > 0 else 0
        )

    def _log_cycle_state(
        self,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
        result,
    ) -> None:
        self._log_operator_state_transition(
            operator_state=operator_state, operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            selected_chain_length=int(result.selected_chain_length),
        )
        self._log_mode_summary(
            mode=self._active_planner_mode, result=result,
            operator_state=operator_state, operator_reason=operator_reason,
            hold_active=bool(self._active_held_path_flag),
        )

    def _build_diagnostics_metrics(
        self,
        result,
        midline: _MidlineUpdateResult,
        ctrl: _ControllerOutput,
        input_metrics: _InputMetrics,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
        cone_audit_entries: list,
    ) -> dict:
        corridor_analysis_metrics = getattr(self, "_last_corridor_analysis_metrics", {})
        return {
            **self._build_corridor_result_metrics(result, input_metrics, midline),
            **self._build_operator_control_metrics(
                result, midline, ctrl, operator_state, operator_reason, hold_remaining_s,
            ),
            **self._active_cone_audit_counts,
            **self._corridor_pair_audit_counts(result),
            **corridor_analysis_metrics,
        }

    def _build_corridor_result_metrics(
        self, result, input_metrics: _InputMetrics, midline: _MidlineUpdateResult,
    ) -> dict:
        return {
            **self._build_corridor_chain_metrics(result, input_metrics),
            **self._build_midline_and_reject_metrics(result, midline),
        }

    def _build_corridor_chain_metrics(self, result, input_metrics: _InputMetrics) -> dict:
        return {
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
            "raw_orange_count": input_metrics.raw_orange_count,
            "resolved_blue_count": input_metrics.resolved_blue_count,
            "resolved_yellow_count": input_metrics.resolved_yellow_count,
            "boundary_hint_count": input_metrics.boundary_hint_count,
        }

    def _build_midline_and_reject_metrics(
        self, result, midline: _MidlineUpdateResult,
    ) -> dict:
        return {
            "candidate_source": midline.candidate_update_reason,
            "midline_update_mode": (
                "hold" if midline.publish_mode == "held" else self._last_midline_update_mode
            ),
            "midline_update_reason": getattr(self, "_last_midline_candidate_update_reason", ""),
            "midline_candidate_jump_m": getattr(self, "_last_midline_candidate_jump_m", float("nan")),
            "midline_near_lateral_delta_max_m": getattr(self, "_last_midline_near_lateral_delta_max_m", float("nan")),
            "midline_buffer_confidence": getattr(self, "_last_midline_buffer_confidence", float("nan")),
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
        }

    def _build_operator_control_metrics(
        self,
        result,
        midline: _MidlineUpdateResult,
        ctrl: _ControllerOutput,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
    ) -> dict:
        return {
            "publish_mode": midline.publish_mode,
            "hold_mode_active_flag": 1 if self._hold_mode_active else 0,
            "hold_clean_frame_count": self._hold_clean_frame_count,
            "hold_reason": midline.hold_reason or "",
            "planner_state_code": self._operator_state_code(operator_state),
            "fresh_publish_flag": 1 if operator_state == "fresh" else 0,
            "held_publish_flag": 1 if operator_state == "held" else 0,
            "stopped_flag": 1 if operator_state == "stopped" else 0,
            "waiting_flag": 1 if operator_state == "waiting" else 0,
            "operator_reason_code": self._operator_reason_code(operator_reason),
            "operator_reason": operator_reason,
            "hold_remaining_s": hold_remaining_s,
            "control_path_point_count": ctrl.control_path_point_count,
            "zero_cmd_sent_flag": ctrl.zero_cmd_sent_flag,
            "planner_mode": self._active_planner_mode,
            "remembered_cone_count": self._active_remembered_cone_count,
            "remembered_stale_cone_count": self._active_stale_cone_count,
            "accepted_pair_count": result.accepted_pair_count,
            "unknown_pair_count": result.unknown_pair_count,
            "filtered_track_width_m": self._filtered_track_width_m,
            "held_path_flag": self._active_held_path_flag,
            "raw_candidate_point_count": int(midline.centerline.shape[0]),
        }

    def _publish_cycle_results(  # noqa: PLR0913
        self, target_frame, raw_centerline, result, midline: _MidlineUpdateResult,
        ctrl: _ControllerOutput,
        diag_metrics, operator_state, operator_reason, hold_remaining_s,
    ) -> None:
        self._publish_diagnostics(
            frame_id=target_frame,
            centerline_point_count=int(midline.centerline.shape[0]),
            selected_edge_count=int(midline.pair_segments_for_viz.shape[0]),
            status=midline.status,
            control_debug_metrics=ctrl.control_debug_metrics,
            planner_metrics=diag_metrics,
        )
        self._current_pair_segments_for_viz = np.array(midline.pair_segments_for_viz, copy=True)
        self._publish_outputs(
            frame_id=target_frame,
            centerline=midline.centerline,
            raw_centerline=raw_centerline,
            raw_midpoint_chain=midline.raw_midpoint_chain,
            result=result,
            status=midline.status,
            control_target_frame=ctrl.control_target_frame,
            cmd_speed=ctrl.cmd_speed,
            cmd_steering=ctrl.cmd_steering,
            lookahead=ctrl.lookahead,
            operator_state=operator_state,
            operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            control_path_point_count=ctrl.control_path_point_count,
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
        context = self._build_cone_audit_context(
            planning_frame=planning_frame, result=result,
            vehicle_xy=vehicle_xy, vehicle_yaw=vehicle_yaw,
        )
        return [
            self._build_single_cone_audit_entry(
                idx=idx, cone=cone, planning_frame=planning_frame,
                context=context, now_sec=now_sec,
            )
            for idx, cone in enumerate(msg.cones)
        ]

    def _build_cone_audit_context(
        self,
        *,
        planning_frame,
        result: CorridorPlannerResult,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> _ConeAuditContext:
        points = np.asarray(planning_frame.points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            points = np.empty((0, 2), dtype=np.float64)
        return _ConeAuditContext(
            points=points,
            local_points=self._audit_points_to_vehicle_frame(points, vehicle_xy, vehicle_yaw),
            used_left={int(track_id) for track_id in np.asarray(result.used_left_track_ids, dtype=np.int64)},
            used_right={int(track_id) for track_id in np.asarray(result.used_right_track_ids, dtype=np.int64)},
            chain_rejection_reasons_by_track_id=result.chain_rejection_reasons_by_track_id,
        )

    def _build_single_cone_audit_entry(
        self, *,
        idx: int,
        cone,
        planning_frame,
        context: _ConeAuditContext,
        now_sec: float,
    ) -> _ConeAuditEntry:
        point_xy = self._audit_point_at(context.points, idx)
        local_xy = self._audit_point_at(context.local_points, idx)
        track_id = self._audit_array_int(planning_frame.track_ids, idx, idx)
        track_state = self._audit_array_int(planning_frame.track_states, idx, MSG_TRACK_STATE_CONFIRMED)
        confidence = self._audit_array_float(
            planning_frame.raw_confidences, idx, float(getattr(cone, "confidence", 0.0)),
        )
        planner_confidence = self._audit_array_float(planning_frame.planner_confidences, idx, confidence)
        track_confidence = self._audit_array_float(
            planning_frame.track_confidences, idx, float(getattr(cone, "track_confidence", confidence)),
        )
        raw_color = self._audit_color(planning_frame.raw_colors, idx, getattr(cone, "color", ""))
        resolved_color = self._audit_color(
            planning_frame.colors, idx, getattr(cone, "boundary_color", ""),
        )
        missed_count = int(getattr(cone, "missed_count", 0))
        reason = self._classify_cone_audit_reason(
            track_id=track_id, local_xy=local_xy, resolved_color=resolved_color,
            track_state=track_state, planner_confidence=planner_confidence,
            used_left=context.used_left, used_right=context.used_right,
            chain_rejection_reasons_by_track_id=context.chain_rejection_reasons_by_track_id,
        )
        return _ConeAuditEntry(
            track_id=int(track_id), reason=reason, point_xy=np.asarray(point_xy, dtype=np.float64),
            local_x_m=float(local_xy[0]), local_y_m=float(local_xy[1]), raw_color=str(raw_color),
            resolved_color=str(resolved_color), track_state=int(track_state),
            confidence=float(confidence), track_confidence=float(track_confidence),
            color_confidence=float(getattr(cone, "color_confidence", float("nan"))),
            missed_count=missed_count,
            last_seen_age_sec=float(self._stamp_age_sec(getattr(cone, "last_seen", None), now_sec)),
            memory_only=bool(track_state == MSG_TRACK_STATE_STALE or missed_count > 0),
        )

    @staticmethod
    def _audit_point_at(points: np.ndarray, idx: int) -> np.ndarray:
        if idx < points.shape[0]:
            return points[idx]
        return np.asarray([float("nan"), float("nan")], dtype=np.float64)

    @staticmethod
    def _audit_color(values: list, idx: int, fallback: str) -> str:
        if idx < len(values):
            return str(values[idx])
        return normalize_color(fallback)

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
        for reason in CORRIDOR_PAIR_AUDIT_REASONS:
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
        arr.markers.append(self._cone_audit_clear_marker(frame_id=frame_id, stamp=stamp))
        label_count = 0
        for entry in entries:
            if not np.all(np.isfinite(entry.point_xy)):
                continue
            arr.markers.append(self._build_cone_audit_main_marker(
                frame_id=frame_id, stamp=stamp, entry=entry, marker_id=len(arr.markers),
            ))
            if entry.memory_only:
                arr.markers.append(self._build_cone_audit_halo_marker(
                    frame_id=frame_id, stamp=stamp, entry=entry, marker_id=len(arr.markers),
                ))
            if self.cone_audit_show_labels and label_count < self.cone_audit_max_labels:
                arr.markers.append(self._build_cone_audit_label_marker(
                    frame_id=frame_id, stamp=stamp, entry=entry, marker_id=len(arr.markers),
                ))
                label_count += 1
        return arr

    @staticmethod
    def _cone_audit_clear_marker(*, frame_id: str, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.action = Marker.DELETEALL
        return marker

    def _build_cone_audit_main_marker(
        self, *, frame_id: str, stamp, entry: _ConeAuditEntry, marker_id: int,
    ) -> Marker:
        marker = self._cone_audit_base_marker(frame_id, stamp, entry, marker_id)
        marker.ns = self._cone_audit_marker_namespace(entry.reason)
        marker.type = Marker.CUBE if entry.reason.startswith("rejected_geometry") else Marker.SPHERE
        marker.pose.position.z = _CONE_AUDIT_MAIN_Z_M
        scale = self._cone_audit_marker_scale(entry.reason)
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        self._apply_marker_rgba(marker, self._cone_audit_marker_rgba(entry))
        return marker

    def _build_cone_audit_halo_marker(
        self, *, frame_id: str, stamp, entry: _ConeAuditEntry, marker_id: int,
    ) -> Marker:
        marker = self._cone_audit_base_marker(frame_id, stamp, entry, marker_id)
        marker.ns = "cone_audit_stale_halo"
        marker.type = Marker.CYLINDER
        marker.pose.position.z = _CONE_AUDIT_HALO_Z_M
        marker.scale.x = _CONE_AUDIT_HALO_DIAMETER_M
        marker.scale.y = _CONE_AUDIT_HALO_DIAMETER_M
        marker.scale.z = _CONE_AUDIT_HALO_HEIGHT_M
        self._apply_marker_rgba(marker, _CONE_AUDIT_HALO_RGBA)
        return marker

    def _build_cone_audit_label_marker(
        self, *, frame_id: str, stamp, entry: _ConeAuditEntry, marker_id: int,
    ) -> Marker:
        marker = self._cone_audit_base_marker(frame_id, stamp, entry, marker_id)
        marker.ns = "cone_audit_labels"
        marker.type = Marker.TEXT_VIEW_FACING
        marker.pose.position.z = _CONE_AUDIT_LABEL_Z_M
        marker.scale.z = _CONE_AUDIT_LABEL_HEIGHT_M
        marker.text = self._cone_audit_label(entry)
        self._apply_marker_rgba(marker, _CONE_AUDIT_LABEL_RGBA)
        return marker

    @staticmethod
    def _cone_audit_base_marker(frame_id: str, stamp, entry: _ConeAuditEntry, marker_id: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.id = _CONE_AUDIT_MARKER_START_ID if marker_id < 1 else marker_id
        marker.action = Marker.ADD
        marker.pose.position.x = float(entry.point_xy[0])
        marker.pose.position.y = float(entry.point_xy[1])
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _apply_marker_rgba(marker: Marker, rgba: tuple[float, float, float, float]) -> None:
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba

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
            midpoint_x_base, midpoint_y_base = _odom_point_to_base(
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
            entry = self._transform_pair_segment_to_odom(
                pair_segment=pair_segment, frame_id=frame_id,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
                now_sec=now_sec,
            )
            if entry is None:
                continue
            entries.append(entry)
        return entries

    def _transform_pair_segment_to_odom(
        self,
        *,
        pair_segment: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
    ) -> Optional[_PairMemoryEntry]:
        if pair_segment.shape != (_PAIR_SEGMENT_ENDPOINT_COUNT, _PAIR_SEGMENT_COORD_COUNT):
            return None
        left_point = np.asarray(pair_segment[0], dtype=np.float64)
        right_point = np.asarray(pair_segment[1], dtype=np.float64)
        midpoint = _PAIR_MIDPOINT_WEIGHT * (left_point + right_point)
        left = (float(left_point[0]), float(left_point[1]))
        right = (float(right_point[0]), float(right_point[1]))
        middle = (float(midpoint[0]), float(midpoint[1]))
        if self._is_alias(frame_id, self.base_frame):
            left = _base_point_to_odom(*left, vehicle_x, vehicle_y, vehicle_yaw)
            right = _base_point_to_odom(*right, vehicle_x, vehicle_y, vehicle_yaw)
            middle = _base_point_to_odom(*middle, vehicle_x, vehicle_y, vehicle_yaw)
        elif not self._is_alias(frame_id, self.odom_frame):
            return None
        return _PairMemoryEntry(
            left_x_odom=float(left[0]), left_y_odom=float(left[1]),
            right_x_odom=float(right[0]), right_y_odom=float(right[1]),
            midpoint_x_odom=float(middle[0]), midpoint_y_odom=float(middle[1]),
            last_valid_sec=float(now_sec),
        )

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
                if math.hypot(dx, dy) <= _PAIR_MEMORY_MERGE_DISTANCE_M:
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
        local_midpoints = self._project_pair_midpoints_to_local(
            entries=entries, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        seed_idx = self._select_sort_seed_index(local_midpoints)
        ordered_indices = self._greedy_chain_sort(local_midpoints, seed_idx)
        return [entries[idx] for idx in ordered_indices]

    @staticmethod
    def _project_pair_midpoints_to_local(
        *,
        entries: list[_PairMemoryEntry],
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> np.ndarray:
        local_midpoints = np.empty((len(entries), _PAIR_SEGMENT_COORD_COUNT), dtype=np.float64)
        for idx, entry in enumerate(entries):
            midpoint_x_base, midpoint_y_base = _odom_point_to_base(
                entry.midpoint_x_odom, entry.midpoint_y_odom, vehicle_x, vehicle_y, vehicle_yaw,
            )
            local_midpoints[idx, 0] = float(midpoint_x_base)
            local_midpoints[idx, 1] = float(midpoint_y_base)
        return local_midpoints

    @staticmethod
    def _select_sort_seed_index(local_midpoints: np.ndarray) -> int:
        seed_candidates = np.flatnonzero(local_midpoints[:, 0] >= -_PAIR_PASSED_MARGIN_M)
        if seed_candidates.size > 0:
            return int(
                min(seed_candidates.tolist(), key=lambda idx: (
                    max(float(local_midpoints[idx, 0]), 0.0),
                    abs(float(local_midpoints[idx, 1])),
                    float(np.hypot(local_midpoints[idx, 0], local_midpoints[idx, 1])),
                    idx,
                ))
            )
        return int(np.argmin(np.hypot(local_midpoints[:, 0], local_midpoints[:, 1])))

    @staticmethod
    def _greedy_chain_sort(local_midpoints: np.ndarray, seed_idx: int) -> list[int]:
        ordered_indices = [seed_idx]
        remaining = {idx for idx in range(local_midpoints.shape[0]) if idx != seed_idx}
        heading = np.array(_PAIR_SORT_INITIAL_HEADING, copy=True)
        current_range = float(np.hypot(local_midpoints[seed_idx, 0], local_midpoints[seed_idx, 1]))
        while remaining:
            current_idx = ordered_indices[-1]
            current_point = local_midpoints[current_idx]
            best_idx = CorridorPlannerNode._best_chain_sort_candidate(
                remaining, local_midpoints, current_point, heading, current_range,
            )
            if best_idx is None:
                break
            delta = local_midpoints[best_idx] - current_point
            delta_norm = float(np.hypot(delta[0], delta[1]))
            if delta_norm > _PAIR_SORT_MIN_STEP_M:
                heading = delta / delta_norm
            ordered_indices.append(best_idx)
            remaining.remove(best_idx)
            current_range = float(np.hypot(local_midpoints[best_idx, 0], local_midpoints[best_idx, 1]))
        return ordered_indices

    @staticmethod
    def _best_chain_sort_candidate(
        remaining: set[int],
        local_midpoints: np.ndarray,
        current_point: np.ndarray,
        heading: np.ndarray,
        current_range: float,
    ) -> Optional[int]:
        best_idx = None
        best_score = None
        for candidate_idx in remaining:
            score = CorridorPlannerNode._chain_sort_score(
                local_midpoints, current_point, heading, current_range, candidate_idx,
            )
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_idx = candidate_idx
        return best_idx

    @staticmethod
    def _chain_sort_score(
        local_midpoints: np.ndarray,
        current_point: np.ndarray,
        heading: np.ndarray,
        current_range: float,
        candidate_idx: int,
    ) -> Optional[tuple[bool, float, float, float, float, int]]:
        delta = local_midpoints[candidate_idx] - current_point
        distance = float(np.hypot(delta[0], delta[1]))
        if distance <= _PAIR_SORT_MIN_STEP_M:
            return None
        candidate_range = float(np.hypot(local_midpoints[candidate_idx, 0], local_midpoints[candidate_idx, 1]))
        if candidate_range < current_range - _PAIR_SORT_RANGE_REGRESSION_TOLERANCE_M:
            return None
        step_dir = delta / distance
        heading_error = abs(math.atan2(step_dir[1], step_dir[0]) - math.atan2(heading[1], heading[0]))
        heading_error = abs(math.atan2(math.sin(heading_error), math.cos(heading_error)))
        backward_m = max(0.0, -float(delta[0]))
        return (
            backward_m > _PAIR_SORT_BACKWARD_GATE_M,
            backward_m, heading_error, distance, abs(float(delta[1])), candidate_idx,
        )

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
        frame_id: Optional[str] = None,
        vehicle_x: Optional[float] = None,
        vehicle_y: Optional[float] = None,
        vehicle_yaw: Optional[float] = None,
        now_sec: float,
    ) -> None:
        if frame_id is None or vehicle_x is None or vehicle_y is None or vehicle_yaw is None:
            return
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
        empty = self._empty_candidate_transition_metrics()
        paths = self._resolve_transition_paths(candidate_centerline, vehicle_x, vehicle_y)
        if paths is None:
            return empty
        stored_samples, candidate_samples = self._sample_transition_paths(paths, horizon_m)
        if stored_samples is None or candidate_samples is None:
            return empty
        stored_local, candidate_local = self._transition_paths_to_vehicle_frame(
            stored_samples, candidate_samples, vehicle_x, vehicle_y, vehicle_yaw,
        )
        if stored_local is None or candidate_local is None:
            return empty
        return self._summarize_transition_delta(stored_local, candidate_local)

    @staticmethod
    def _empty_candidate_transition_metrics() -> dict[str, float]:
        return {
            "sample_count": 0.0,
            "lateral_max_m": float("inf"),
            "lateral_mean_m": float("inf"),
            "displacement_max_m": float("inf"),
            "displacement_mean_m": float("inf"),
            "heading_delta_rad": float("inf"),
        }

    def _resolve_transition_paths(
        self,
        candidate_centerline: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
    ) -> Optional[_TransitionPaths]:
        if self._midline_buffer_path is None:
            return None
        if self._midline_buffer_path.shape[0] < _TRANSITION_MIN_PATH_POINTS:
            return None
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
        return self._select_transition_paths(stored_forward, candidate_forward, candidate_centerline)

    def _select_transition_paths(
        self,
        stored_forward: Optional[np.ndarray],
        candidate_forward: Optional[np.ndarray],
        candidate_centerline: np.ndarray,
    ) -> Optional[_TransitionPaths]:
        stored_path = self._transition_path_or_fallback(stored_forward, self._midline_buffer_path)
        candidate_path = self._transition_path_or_fallback(candidate_forward, candidate_centerline)
        if stored_path.shape[0] < _TRANSITION_MIN_PATH_POINTS:
            return None
        if candidate_path.shape[0] < _TRANSITION_MIN_PATH_POINTS:
            return None
        return _TransitionPaths(stored_path=stored_path, candidate_path=candidate_path)

    @staticmethod
    def _transition_path_or_fallback(path: Optional[np.ndarray], fallback: np.ndarray) -> np.ndarray:
        if path is not None and path.shape[0] >= _TRANSITION_MIN_PATH_POINTS:
            return np.asarray(path, dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64)

    def _sample_transition_paths(
        self,
        paths: _TransitionPaths,
        horizon_m: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        stored_samples = self._resample_midline_stations(paths.stored_path)
        candidate_samples = self._resample_midline_stations(paths.candidate_path)
        count = min(stored_samples.shape[0], candidate_samples.shape[0])
        if count < _TRANSITION_MIN_PATH_POINTS:
            return None, None
        count = self._transition_sample_count(stored_samples, candidate_samples, count, horizon_m)
        return stored_samples[:count], candidate_samples[:count]

    def _transition_sample_count(
        self,
        stored_samples: np.ndarray,
        candidate_samples: np.ndarray,
        count: int,
        horizon_m: float,
    ) -> int:
        horizon_limit_m = max(_TRANSITION_MIN_HORIZON_M, float(horizon_m))
        stored_limit = max(
            _TRANSITION_MIN_PATH_POINTS,
            int(np.searchsorted(self._path_cumulative_lengths(stored_samples), horizon_limit_m, side="right")),
        )
        candidate_limit = max(
            _TRANSITION_MIN_PATH_POINTS,
            int(np.searchsorted(self._path_cumulative_lengths(candidate_samples), horizon_limit_m, side="right")),
        )
        return min(count, stored_limit, candidate_limit)

    def _transition_paths_to_vehicle_frame(
        self,
        stored_samples: np.ndarray,
        candidate_samples: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        stored_local = self._transition_path_to_vehicle_frame(stored_samples, vehicle_x, vehicle_y, vehicle_yaw)
        candidate_local = self._transition_path_to_vehicle_frame(candidate_samples, vehicle_x, vehicle_y, vehicle_yaw)
        if stored_local.shape[0] < _TRANSITION_MIN_PATH_POINTS:
            return None, None
        if candidate_local.shape[0] < _TRANSITION_MIN_PATH_POINTS:
            return None, None
        return stored_local, candidate_local

    def _transition_path_to_vehicle_frame(
        self,
        path: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> np.ndarray:
        return self._centerline_to_vehicle_frame(
            centerline=path,
            frame_id=self.odom_frame,
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )

    def _summarize_transition_delta(
        self,
        stored_local: np.ndarray,
        candidate_local: np.ndarray,
    ) -> dict[str, float]:
        delta = candidate_local - stored_local
        lateral = np.abs(delta[:, 1])
        displacement = np.hypot(delta[:, 0], delta[:, 1])
        return {
            "sample_count": float(stored_local.shape[0]),
            "lateral_max_m": float(np.max(lateral)),
            "lateral_mean_m": float(np.mean(lateral)),
            "displacement_max_m": float(np.max(displacement)),
            "displacement_mean_m": float(np.mean(displacement)),
            "heading_delta_rad": self._transition_heading_delta(stored_local, candidate_local),
        }

    def _transition_heading_delta(self, stored_local: np.ndarray, candidate_local: np.ndarray) -> float:
        stored_heading = self._path_start_heading_local(stored_local)
        candidate_heading = self._path_start_heading_local(candidate_local)
        return abs(
            float(
                math.atan2(
                    math.sin(candidate_heading - stored_heading),
                    math.cos(candidate_heading - stored_heading),
                )
            )
        )

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
        if result.status == "ok" and result.centerline.shape[0] > 0:
            candidates.extend(self._candidate_entry(
                result.centerline, "validated", _VALIDATED_CANDIDATE_PRIORITY,
                frame_id, vehicle_x, vehicle_y, vehicle_yaw,
            ))
        candidates.extend(self._candidate_entry(
            support_chain, _CORRIDOR_MIDPOINT_SOURCE, _SUPPORT_CHAIN_CANDIDATE_PRIORITY,
            frame_id, vehicle_x, vehicle_y, vehicle_yaw,
        ))
        candidates.extend(self._candidate_entry(
            result.prevalidation_centerline, _CORRIDOR_MIDPOINT_SOURCE,
            _PREVALIDATION_CANDIDATE_PRIORITY, frame_id, vehicle_x, vehicle_y, vehicle_yaw,
        ))
        candidates.extend(self._candidate_entry(
            memory_midpoint_chain, _CORRIDOR_MIDPOINT_SOURCE, _MEMORY_CHAIN_CANDIDATE_PRIORITY,
            frame_id, vehicle_x, vehicle_y, vehicle_yaw,
        ))
        path, source = self._pick_best_candidate(candidates, self.midline_station_spacing_m)
        return np.array(path, copy=True), source

    def _candidate_entry(
        self,
        path: np.ndarray,
        source: str,
        priority: int,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[tuple[float, int, str, np.ndarray]]:
        candidate = self._finite_corridor_path(path)
        if candidate.shape[0] < _CORRIDOR_MIDPOINT_MIN_POINTS:
            return []
        extent = self._candidate_forward_extent_m(
            centerline=candidate, frame_id=frame_id,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        if extent <= _CANDIDATE_EXTENT_EPSILON_M:
            return []
        return [(extent, priority, source, candidate)]

    @staticmethod
    def _pick_best_candidate(
        candidates: list[tuple[float, int, str, np.ndarray]],
        station_spacing_m: float,
    ) -> tuple[np.ndarray, str]:
        if not candidates:
            return np.empty((0, 2), dtype=np.float64), "none"
        best_extent = max(extent for extent, _, _, _ in candidates)
        extent_tolerance_m = max(
            _CANDIDATE_EXTENT_TOLERANCE_MIN_M,
            _CANDIDATE_EXTENT_TOLERANCE_STATION_FACTOR * float(station_spacing_m),
        )
        near_best = [
            candidate for candidate in candidates
            if candidate[0] >= (best_extent - extent_tolerance_m)
        ]
        _, _, source, path = max(near_best, key=lambda candidate: (candidate[1], candidate[0]))
        return path, source

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
        for prefix, samples in self._corridor_analysis_samples_by_prefix(
            raw_anchor_path=raw_anchor_path,
            prevalidation_centerline=prevalidation_centerline,
            buffered_centerline=buffered_centerline,
            control_path_local=control_path_local,
            frame_id=frame_id, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        ).items():
            self._corridor_analysis_fill_prefix_metrics(metrics, prefix, samples)
        return metrics

    def _corridor_analysis_samples_by_prefix(
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
    ) -> dict[str, np.ndarray]:
        return {
            "corridor_raw_anchor": self._corridor_analysis_frame_path(raw_anchor_path, frame_id, vehicle_x, vehicle_y, vehicle_yaw),
            "corridor_prevalidation_centerline": self._corridor_analysis_frame_path(
                prevalidation_centerline, frame_id, vehicle_x, vehicle_y, vehicle_yaw,
            ),
            "corridor_buffer_centerline": self._corridor_analysis_frame_path(
                buffered_centerline, frame_id, vehicle_x, vehicle_y, vehicle_yaw,
            ),
            "corridor_control_path": self._corridor_analysis_sample_local_path(
                np.asarray(control_path_local, dtype=np.float64)
            ),
        }

    def _corridor_analysis_frame_path(
        self,
        path: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> np.ndarray:
        return self._corridor_analysis_sample_path_in_vehicle_frame(
            path=path, frame_id=frame_id,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )

    @staticmethod
    def _corridor_analysis_fill_prefix_metrics(
        metrics: dict[str, float],
        prefix: str,
        samples: np.ndarray,
    ) -> None:
        metrics[f"{prefix}_point_count"] = float(samples.shape[0])
        for idx in range(_CORRIDOR_ANALYSIS_SAMPLE_COUNT):
            if idx < samples.shape[0]:
                metrics[f"{prefix}_p{idx}_x_m"] = float(samples[idx, 0])
                metrics[f"{prefix}_p{idx}_y_m"] = float(samples[idx, 1])
            else:
                metrics[f"{prefix}_p{idx}_x_m"] = float("nan")
                metrics[f"{prefix}_p{idx}_y_m"] = float("nan")

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
