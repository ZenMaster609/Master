#!/usr/bin/env python3
"""Single-boundary planner over tracked cone detections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vehicle_plotter_msgs.msg import ConeDetectionArray

from sim_car.planning.planner_runtime_types import PlannerIdentity
from sim_car.planning.planner_constants import (
    MSG_TRACK_STATE_STALE,
    MSG_TRACK_STATE_TENTATIVE,
    VALIDATED_JUMP_ACCEPT_HEADING_DELTA_RAD as _VALIDATED_JUMP_ACCEPT_HEADING_DELTA_RAD,
    VALIDATED_JUMP_ACCEPT_HORIZON_M as _VALIDATED_JUMP_ACCEPT_HORIZON_M,
    VALIDATED_JUMP_ACCEPT_LATERAL_MAX_M as _VALIDATED_JUMP_ACCEPT_LATERAL_MAX_M,
    VALIDATED_JUMP_ACCEPT_LATERAL_MEAN_M as _VALIDATED_JUMP_ACCEPT_LATERAL_MEAN_M,
)
from sim_car.planning.tracked_cone_planner_contract import (
    COMMON_MIGRATED_TRACKED_CONE_PLANNER_DEFAULTS,
    apply_common_config_to_node,
    read_migrated_tracked_cone_planner_common_config,
)
from sim_car.planning.single_boundary_planner_core import (
    SingleBoundaryPlannerConfig,
    SingleBoundaryPlannerPrior,
    SingleBoundaryPlannerResult,
    _finalize_path,
    compute_single_boundary_centerline,
)
from sim_car.planning.tracked_cone_planner_base import TrackedConePlannerBase
from sim_car.planning.tracked_cone_planner_geometry import (
    _base_point_to_odom,
    update_track_width_estimate,
)


_NANOSECONDS_TO_SECONDS = 1e-9  # ROS clock timestamps are nanoseconds; planner metrics use seconds.
_TRANSITION_MIN_PATH_POINTS = 2  # Two samples are required to measure path displacement and heading.
_TRANSITION_MIN_HORIZON_M = 0.25  # Keeps the near-field comparison meaningful for very short horizons.
_TRANSITION_FORWARD_X_MIN_M = -0.1  # Allows tiny pose-noise drift behind the vehicle while filtering old path.
_TRANSITION_LENGTH_EPSILON_M = 1e-6  # Treats sub-micrometer local paths as degenerate.
_TRANSITION_MIN_STATION_STEP_M = 0.05  # Prevents overly dense samples if station spacing is misconfigured.
_TRANSITION_SAMPLE_END_EPSILON_M = 1e-9  # Includes the final sample despite floating-point roundoff.


@dataclass
class _PairMemoryEntry:
    left_track_id: int
    right_track_id: int
    last_valid_sec: float


@dataclass
class _InputMetrics:
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
    control_debug_metrics: Optional[dict[str, float]]
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


class SingleBoundaryPlannerNode(TrackedConePlannerBase):
    """Tracked-cone single-boundary planner with shared path-memory stabilization."""

    _centerline_marker_width_m: float = 0.09

    def __init__(self) -> None:
        self._planner_identity = PlannerIdentity(
            node_name="single_boundary_planner_node",
            planner_mode="single_boundary",
            diagnostics_prefix="single_boundary_planner",
            diagnostics_topic="/single_boundary_planner/diagnostics",
        )
        Node.__init__(self, self._planner_identity.node_name)
        self._declare_parameters()
        self._read_parameters()

        self._init_common_planner_state()

        self._init_common_ros_interfaces()

    def _declare_parameters(self) -> None:
        defaults = dict(COMMON_MIGRATED_TRACKED_CONE_PLANNER_DEFAULTS)
        defaults.update({
            "filtering.allow_unknown_pair_completion": True,
            "filtering.unknown_pair_search_radius_m": 1.25,
            "filtering.unknown_pair_max_longitudinal_error_m": 1.5,
            "filtering.unknown_pair_max_width_error_m": 0.9,
            "filtering.max_consecutive_unknown_pairs": 2,
            "boundary_chain.min_chain_length": 2,
            "pairing.min_pair_width_m": 2.2,
            "pairing.max_pair_width_m": 5.4,
            "pairing.max_width_jump_m": 0.75,
            "pairing.min_pair_count": 3,
            "pairing.pair_reassignment_margin": 0.25,
            "width_estimation.min_trustworthy_pairs": 3,
            "centerline.smoothing_window": 3,
            "centerline.max_heading_delta_rad": 0.75,
            "lap_tracking.target_laps": 0,
            "validation.min_path_points": 2,
            "validation.min_forward_extent_m": 1.0,
            "validation.max_near_field_lateral_jump_m_sparse_pairs": 0.9,
            "validation.max_near_field_lateral_jump_m_single_boundary": 5.0,
            "validation.max_start_heading_error_rad": 1.0,
            "diagnostics.topic": "/single_boundary_planner/diagnostics",
            "debug.show_raw_offset_path": True,
        })
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        common = read_migrated_tracked_cone_planner_common_config(
            self,
            planner_label='single-boundary planner',
            diagnostics_topic_fallback=self._planner_identity.diagnostics_topic,
        )
        apply_common_config_to_node(self, common)
        self._read_runtime_parameters()
        self._core_config = self._build_core_config()
        self._filtered_track_width_m = float(self._core_config.initial_width_m)

    def _read_runtime_parameters(self) -> None:
        self.lap_tracking_target_laps = max(
            0,
            int(self.get_parameter("lap_tracking.target_laps").value),
        )
        self.show_raw_offset_path = bool(
            self.get_parameter("debug.show_raw_offset_path").value
        )

    def _build_core_config(self) -> SingleBoundaryPlannerConfig:
        values = {}
        values.update(self._filtering_config_values())
        values.update(self._boundary_chain_config_values())
        values.update(self._pairing_config_values())
        values.update(self._width_estimation_config_values())
        values.update(self._centerline_config_values())
        values.update(self._validation_config_values())
        return SingleBoundaryPlannerConfig(**values)

    def _filtering_config_values(self) -> dict:
        return {
            "max_cone_range_m": float(self.get_parameter("filtering.max_cone_range_m").value),
            "behind_drop_m": float(self.get_parameter("filtering.behind_drop_m").value),
            "min_confidence": float(self.get_parameter("filtering.min_confidence").value),
            "min_required_cones": max(2, int(self.get_parameter("filtering.min_required_cones").value)),
            "allow_unknown_pair_completion": bool(
                self.get_parameter("filtering.allow_unknown_pair_completion").value
            ),
            "unknown_pair_search_radius_m": float(
                self.get_parameter("filtering.unknown_pair_search_radius_m").value
            ),
            "unknown_pair_max_longitudinal_error_m": float(
                self.get_parameter("filtering.unknown_pair_max_longitudinal_error_m").value
            ),
            "unknown_pair_max_width_error_m": float(
                self.get_parameter("filtering.unknown_pair_max_width_error_m").value
            ),
            "max_consecutive_unknown_pairs": max(
                0,
                int(self.get_parameter("filtering.max_consecutive_unknown_pairs").value),
            ),
        }

    def _boundary_chain_config_values(self) -> dict:
        return {
            "min_step_m": float(self.get_parameter("boundary_chain.min_step_m").value),
            "max_step_m": float(self.get_parameter("boundary_chain.max_step_m").value),
            "max_heading_change_rad": float(self.get_parameter("boundary_chain.max_heading_change_rad").value),
            "min_forward_progress_m": float(
                self.get_parameter("boundary_chain.min_forward_progress_m").value
            ),
            "min_chain_length": max(2, int(self.get_parameter("boundary_chain.min_chain_length").value)),
        }

    def _pairing_config_values(self) -> dict:
        return {
            "min_pair_width_m": float(self.get_parameter("pairing.min_pair_width_m").value),
            "max_pair_width_m": float(self.get_parameter("pairing.max_pair_width_m").value),
            "max_width_jump_m": float(self.get_parameter("pairing.max_width_jump_m").value),
            "min_pair_count": max(1, int(self.get_parameter("pairing.min_pair_count").value)),
            "pair_reassignment_margin": float(
                self.get_parameter("pairing.pair_reassignment_margin").value
            ),
        }

    def _width_estimation_config_values(self) -> dict:
        return {
            "initial_width_m": float(self.get_parameter("width_estimation.initial_width_m").value),
            "min_width_m": float(self.get_parameter("width_estimation.min_width_m").value),
            "max_width_m": float(self.get_parameter("width_estimation.max_width_m").value),
            "width_filter_alpha": float(self.get_parameter("width_estimation.alpha").value),
            "max_width_delta_per_update_m": float(
                self.get_parameter("width_estimation.max_delta_per_update_m").value
            ),
            "min_trustworthy_pairs": max(
                1,
                int(self.get_parameter("width_estimation.min_trustworthy_pairs").value),
            ),
        }

    def _centerline_config_values(self) -> dict:
        return {
            "path_resolution_m": float(self.get_parameter("centerline.path_resolution_m").value),
            "max_path_length_m": float(self.get_parameter("centerline.max_path_length_m").value),
            "smoothing_window": max(1, int(self.get_parameter("centerline.smoothing_window").value)),
            "max_heading_delta_rad": float(self.get_parameter("centerline.max_heading_delta_rad").value),
        }

    def _validation_config_values(self) -> dict:
        return {
            "min_path_points": max(2, int(self.get_parameter("validation.min_path_points").value)),
            "min_forward_extent_m": float(self.get_parameter("validation.min_forward_extent_m").value),
            "jump_check_horizon_m": float(self.get_parameter("validation.jump_check_horizon_m").value),
            "max_near_field_lateral_jump_m": float(
                self.get_parameter("validation.max_near_field_lateral_jump_m").value
            ),
            "max_near_field_lateral_jump_m_sparse_pairs": float(
                self.get_parameter("validation.max_near_field_lateral_jump_m_sparse_pairs").value
            ),
            "max_near_field_lateral_jump_m_single_boundary": float(
                self.get_parameter("validation.max_near_field_lateral_jump_m_single_boundary").value
            ),
            "max_start_heading_error_rad": float(
                self.get_parameter("validation.max_start_heading_error_rad").value
            ),
        }

    def _lap_status_text(self) -> str:
        if self.lap_tracking_target_laps > 0:
            return f"LAPS: {int(self._lap_tracking_completed_laps)}/{int(self.lap_tracking_target_laps)}"
        return f"LAPS: {int(self._lap_tracking_completed_laps)}/off"

    def _on_timer(self) -> None:
        ctx = self._resolve_cone_planning_context()
        if ctx is None:
            return
        cones_msg, target_frame, vehicle_x, vehicle_y, vehicle_yaw, points_xy, colors, confidences = ctx
        now_sec = float(self.get_clock().now().nanoseconds) * _NANOSECONDS_TO_SECONDS
        self._update_smalltrack_lap_from_orange_cones(
            cones_msg=cones_msg, points_xy=points_xy,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        self._update_remembered_cone_viz(points_xy=points_xy, colors=colors)
        input_metrics = self._collect_input_metrics(cones_msg, points_xy, colors, confidences)
        result = self._run_single_boundary_planner(
            input_metrics.planning_frame, vehicle_x, vehicle_y, vehicle_yaw,
        )
        raw_centerline, candidate_source = self._select_candidate_centerline(result)
        raw_midpoint_chain = np.array(result.midpoints_raw, copy=True)
        pair_segments_for_viz = np.array(result.pair_segments, copy=True)
        result.planner_mode = self._planner_identity.planner_mode
        del points_xy
        midline = self._apply_midline_updates(
            result, raw_centerline, candidate_source, raw_midpoint_chain,
            pair_segments_for_viz, target_frame, vehicle_x, vehicle_y, vehicle_yaw, now_sec,
        )
        ctrl = self._run_controller(midline.centerline, target_frame, vehicle_x, vehicle_y, vehicle_yaw)
        hold_remaining_s = self._hold_remaining_s(now_sec)
        operator_state, operator_reason = self._determine_operator_state(
            result, midline, ctrl, hold_remaining_s,
        )
        self._update_active_state_counters(result, midline)
        self._log_cycle_state(operator_state, operator_reason, hold_remaining_s, result)
        diag_metrics = self._build_diagnostics_metrics(
            result, midline, ctrl, raw_centerline, candidate_source,
            operator_state, operator_reason, hold_remaining_s,
        )
        self._publish_cycle_results(
            target_frame, raw_centerline, result, midline, ctrl,
            diag_metrics, operator_state, operator_reason, hold_remaining_s,
        )

    def _collect_input_metrics(
        self,
        cones_msg,
        points_xy: np.ndarray,
        colors: list,
        confidences: np.ndarray,
    ) -> _InputMetrics:
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
        return _InputMetrics(planning_frame=planning_frame)

    def _run_single_boundary_planner(
        self,
        planning_frame,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> SingleBoundaryPlannerResult:
        result = compute_single_boundary_centerline(
            points_xy=planning_frame.points_xy, colors=planning_frame.colors,
            confidences=planning_frame.planner_confidences, track_ids=planning_frame.track_ids,
            vehicle_xy=(vehicle_x, vehicle_y), vehicle_yaw=vehicle_yaw,
            config=self._core_config, prior=self._single_boundary_prior(),
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

    def _single_boundary_prior(self) -> SingleBoundaryPlannerPrior:
        return SingleBoundaryPlannerPrior(
            previous_centerline=self._previous_centerline_prior(),
            previous_width_m=self._filtered_track_width_m,
            previous_mode=self._active_planner_mode,
            previous_pairs=[],
        )

    def _previous_centerline_prior(self) -> Optional[np.ndarray]:
        if self._midline_buffer_path is not None:
            return np.array(self._midline_buffer_path, copy=True)
        if self._last_valid_centerline is not None:
            return np.array(self._last_valid_centerline, copy=True)
        return None

    def _apply_midline_updates(  # noqa: PLR0913
        self, result, raw_centerline, candidate_source, raw_midpoint_chain,
        pair_segments_for_viz, target_frame, vehicle_x, vehicle_y, vehicle_yaw, now_sec,
    ) -> _MidlineUpdateResult:
        centerline, buffered_centerline, candidate_update_ok, candidate_update_reason = (
            self._buffer_and_anchor_centerline(
                result, raw_centerline, candidate_source, target_frame,
                vehicle_x, vehicle_y, vehicle_yaw, now_sec,
            )
        )
        status, pair_segments_for_viz = self._build_midline_status(
            result, raw_centerline, centerline, candidate_source, candidate_update_ok,
            candidate_update_reason, pair_segments_for_viz, now_sec,
        )
        centerline, publish_mode, plan_hold_active, hold_reason, status = self._apply_hold_logic(
            result, centerline, candidate_update_ok, candidate_update_reason, status, now_sec,
        )
        pair_segments_for_viz = self._refresh_hold_viz(plan_hold_active, pair_segments_for_viz)
        centerline = self._prepare_centerline_for_current_pose(
            centerline=centerline, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        centerline = self._finalize_fresh_centerline(
            centerline, raw_midpoint_chain, pair_segments_for_viz, result, publish_mode, now_sec,
        )
        return _MidlineUpdateResult(
            centerline, buffered_centerline, raw_midpoint_chain, pair_segments_for_viz,
            publish_mode, candidate_update_ok, candidate_update_reason, status,
            hold_reason, plan_hold_active,
        )

    def _buffer_and_anchor_centerline(
        self, result, raw_centerline, candidate_source, target_frame,
        vehicle_x, vehicle_y, vehicle_yaw, now_sec,
    ) -> tuple:
        candidate_update_ok, candidate_update_reason = self._candidate_path_is_updateable(
            candidate_centerline=raw_centerline, vehicle_x=vehicle_x, vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw, result=result, candidate_source=candidate_source,
        )
        centerline = self._update_midline_buffer(
            candidate_centerline=raw_centerline, candidate_source=candidate_source,
            candidate_update_ok=candidate_update_ok, candidate_update_reason=candidate_update_reason,
            frame_id=target_frame, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            result=result, now_sec=now_sec, support_centerline=result.raw_offset_path,
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
        self, result, raw_centerline, centerline, candidate_source,
        candidate_update_ok, candidate_update_reason, pair_segments_for_viz, now_sec,
    ) -> tuple:
        status = result.status
        if not candidate_update_ok and centerline.shape[0] > 0:
            status = f"{status}; holding stored midline"
        elif self._is_publishing_stored_midline(raw_centerline, centerline):
            status = f"{status}; publishing stored midline"
        if candidate_source != "validated" and raw_centerline.shape[0] > 0:
            status = f"{status}; using {candidate_source}"
        if centerline.shape[0] > 0 and raw_centerline.shape[0] > 0:
            self._remember_pairs(result=result, now_sec=now_sec)
            self._remember_valid_pair_geometry(result, pair_segments_for_viz)
        elif self._last_valid_pair_segments is not None and centerline.shape[0] > 0:
            pair_segments_for_viz = np.array(self._last_valid_pair_segments, copy=True)
        return status, pair_segments_for_viz

    def _is_publishing_stored_midline(self, raw_centerline: np.ndarray, centerline: np.ndarray) -> bool:
        return (
            raw_centerline.shape[0] > 0
            and centerline.shape[0] > 0
            and (raw_centerline.shape != centerline.shape or not np.allclose(raw_centerline, centerline))
        )

    def _remember_valid_pair_geometry(self, result, pair_segments_for_viz: np.ndarray) -> None:
        self._last_valid_pair_segments = (
            pair_segments_for_viz if pair_segments_for_viz.size > 0 else self._last_valid_pair_segments
        )
        self._last_valid_pair_track_ids = (
            np.array(result.selected_pair_track_ids, copy=True)
            if result.selected_pair_track_ids.size > 0
            else self._last_valid_pair_track_ids
        )

    def _apply_hold_logic(
        self, result, centerline, candidate_update_ok, candidate_update_reason, status, now_sec,
    ) -> tuple:
        plan_hold_active = False
        publish_mode = "fresh"
        hold_reason = result.reject_reason
        if centerline.shape[0] > 0 and candidate_update_ok:
            centerline, plan_hold_active, publish_mode, status = self._hold_after_valid_plan(
                centerline, publish_mode, status, now_sec,
            )
        else:
            centerline, plan_hold_active, publish_mode, status = self._hold_after_invalid_plan(
                centerline, status, now_sec,
            )
        if plan_hold_active:
            hold_reason = result.reject_reason or candidate_update_reason or status
        return centerline, publish_mode, plan_hold_active, hold_reason, status

    def _hold_after_valid_plan(self, centerline, publish_mode: str, status: str, now_sec: float) -> tuple:
        continue_holding = self._advance_hold_hysteresis(plan_ok=True)
        if continue_holding:
            held_centerline = self._held_centerline(now_sec)
            if held_centerline is not None:
                centerline = held_centerline
                publish_mode = "held"
                status = (
                    f"{status}; hysteresis holding previous valid centerline "
                    f"({self._hold_clean_frame_count}/{self.hold_exit_clean_frames})"
                )
                return centerline, True, publish_mode, status
        return centerline, False, publish_mode, status

    def _hold_after_invalid_plan(self, centerline, status: str, now_sec: float) -> tuple:
        self._advance_hold_hysteresis(plan_ok=False)
        held_centerline = self._held_centerline(now_sec)
        if held_centerline is not None:
            return held_centerline, True, "held", f"{status}; holding previous valid centerline"
        return centerline, False, "held", status

    def _refresh_hold_viz(self, plan_hold_active: bool, pair_segments_for_viz: np.ndarray) -> np.ndarray:
        if plan_hold_active and self._last_valid_pair_segments is not None:
            return np.array(self._last_valid_pair_segments, copy=True)
        return pair_segments_for_viz

    def _finalize_fresh_centerline(
        self, centerline, raw_midpoint_chain, pair_segments_for_viz, result, publish_mode, now_sec,
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
            self._remember_fresh_pair_geometry(result, pair_segments_for_viz)
        return centerline

    def _remember_fresh_pair_geometry(self, result, pair_segments_for_viz: np.ndarray) -> None:
        if pair_segments_for_viz.size > 0:
            self._last_valid_pair_segments = np.array(pair_segments_for_viz, copy=True)
        if result.selected_pair_track_ids.size > 0:
            self._last_valid_pair_track_ids = np.array(result.selected_pair_track_ids, copy=True)

    def _run_controller(self, centerline, target_frame, vehicle_x, vehicle_y, vehicle_yaw) -> _ControllerOutput:
        control_path = self._centerline_to_vehicle_frame(
            centerline=centerline, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        ctrl_target, ctrl_debug, cmd_speed, cmd_steering, lookahead, zero_flag, failed = (
            self._dispatch_controller(control_path, target_frame, vehicle_x, vehicle_y, vehicle_yaw)
        )
        return _ControllerOutput(
            control_path, ctrl_target, ctrl_debug, cmd_speed, cmd_steering, lookahead,
            zero_flag, failed, int(control_path.shape[0]),
        )

    def _dispatch_controller(
        self, control_path: np.ndarray, target_frame: str, vehicle_x: float, vehicle_y: float, vehicle_yaw: float,
    ) -> tuple:
        if control_path.shape[0] >= 1 and self._controller is not None:
            return self._execute_controller(control_path, target_frame, vehicle_x, vehicle_y, vehicle_yaw)
        if control_path.shape[0] >= 1:
            return None, None, 0.0, 0.0, 0.0, int(self._apply_controller_disabled_behavior()), False
        zero_flag = int(self._apply_no_path_behavior())
        cmd_speed = float(self._last_speed_cmd) if self._last_speed_cmd is not None else 0.0
        cmd_steering = float(self._last_steering_cmd) if self._last_steering_cmd is not None else 0.0
        return None, None, cmd_speed, cmd_steering, 0.0, zero_flag, False

    def _execute_controller(
        self, control_path: np.ndarray, target_frame: str, vehicle_x: float, vehicle_y: float, vehicle_yaw: float,
    ) -> tuple:
        try:
            controller_output = self._controller.compute(
                control_path=control_path, speed_mps=self._latest_speed_mps,
                yaw_rate_rps=self._latest_yaw_rate_rps,
            )
        except ValueError as exc:
            self._warn_throttled("controller_compute_error", f"controller compute failed: {exc}")
            zero_flag = int(self._apply_no_path_behavior())
            cmd_speed = float(self._last_speed_cmd) if self._last_speed_cmd is not None else 0.0
            cmd_steering = float(self._last_steering_cmd) if self._last_steering_cmd is not None else 0.0
            return None, None, cmd_speed, cmd_steering, 0.0, zero_flag, True
        return self._controller_success_output(controller_output, target_frame, vehicle_x, vehicle_y, vehicle_yaw)

    def _controller_success_output(self, controller_output, target_frame, vehicle_x, vehicle_y, vehicle_yaw) -> tuple:
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

    def _resolve_control_target_frame(
        self, control_target_base: np.ndarray, target_frame: str,
        vehicle_x: float, vehicle_y: float, vehicle_yaw: float,
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

    def _build_stanley_debug_metrics(self, debug, control_target_frame: Optional[np.ndarray]) -> dict[str, float]:
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
            **self._stanley_target_frame_metrics(control_target_frame),
        }

    def _stanley_target_frame_metrics(self, control_target_frame: Optional[np.ndarray]) -> dict[str, float]:
        return {
            "target_point_x_frame_m": (
                float(control_target_frame[0]) if control_target_frame is not None else float("nan")
            ),
            "target_point_y_frame_m": (
                float(control_target_frame[1]) if control_target_frame is not None else float("nan")
            ),
        }

    def _determine_operator_state(
        self, result, midline: _MidlineUpdateResult, ctrl: _ControllerOutput, hold_remaining_s: float,
    ) -> tuple[str, str]:
        operator_state, operator_reason = self._base_operator_state(result, midline, ctrl, hold_remaining_s)
        if ctrl.controller_failed:
            operator_state, operator_reason = "stopped", "controller_compute_failed"
        elif self._controller is None and midline.centerline.shape[0] > 0:
            operator_state, operator_reason = "stopped", "controller_disabled"
        elif midline.centerline.shape[0] > 0 and ctrl.control_path_point_count <= 0:
            operator_state, operator_reason = "stopped", "no_control_path"
        if ctrl.zero_cmd_sent_flag and operator_reason == "none":
            operator_state, operator_reason = "stopped", "stop_if_no_path"
        if self._can_report_fresh_operator_state(midline, ctrl):
            operator_state = "fresh"
        return operator_state, operator_reason

    def _base_operator_state(
        self, result, midline: _MidlineUpdateResult, ctrl: _ControllerOutput, hold_remaining_s: float,
    ) -> tuple:
        core_reject_reason = self._normalize_core_reject_reason(result)
        if midline.centerline.shape[0] == 0:
            if self._last_valid_centerline is not None and hold_remaining_s <= 0.0:
                return "stopped", "hold_expired_no_path"
            if core_reject_reason != "none":
                return "stopped", core_reject_reason
            if ctrl.zero_cmd_sent_flag:
                return "stopped", "stop_if_no_path"
            return "held", "holding_previous_valid"
        if midline.publish_mode == "held":
            return self._held_operator_state(core_reject_reason, midline.status)
        return "fresh", "none"

    def _held_operator_state(self, core_reject_reason: str, status: str) -> tuple[str, str]:
        if "hysteresis holding previous valid centerline" in status:
            return "held", "hysteresis_holding"
        if core_reject_reason != "none":
            return "held", core_reject_reason
        return "held", "holding_previous_valid"

    def _can_report_fresh_operator_state(self, midline: _MidlineUpdateResult, ctrl: _ControllerOutput) -> bool:
        return (
            ctrl.control_path_point_count > 0
            and midline.centerline.shape[0] > 0
            and ctrl.zero_cmd_sent_flag == 0
            and midline.publish_mode == "fresh"
        )

    def _update_active_state_counters(self, result, midline: _MidlineUpdateResult) -> None:
        self._active_planner_mode = (
            "holding_last_valid"
            if midline.publish_mode == "held" and midline.centerline.shape[0] > 0
            else result.planner_mode
        )
        self._active_left_chain_length = int(result.left_chain_length)
        self._active_right_chain_length = int(result.right_chain_length)
        self._active_pair_count = self._active_pair_count_from_result(result, midline)
        self._active_unknown_pair_count = int(result.unknown_pair_count)
        self._active_filtered_track_width_m = float(self._filtered_track_width_m)
        self._active_held_path_flag = (
            1 if midline.publish_mode == "held" and midline.centerline.shape[0] > 0 else 0
        )

    def _active_pair_count_from_result(self, result, midline: _MidlineUpdateResult) -> int:
        if midline.pair_segments_for_viz.size > 0:
            return int(midline.pair_segments_for_viz.shape[0])
        return int(result.accepted_pair_count)

    def _log_cycle_state(self, operator_state: str, operator_reason: str, hold_remaining_s: float, result) -> None:
        self._log_operator_state_transition(
            operator_state=operator_state, operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s, selected_chain_length=int(result.selected_chain_length),
        )
        self._log_mode_summary(
            mode=self._active_planner_mode, result=result,
            operator_state=operator_state, operator_reason=operator_reason,
            hold_active=bool(self._active_held_path_flag),
        )

    def _build_diagnostics_metrics(
        self, result, midline, ctrl, raw_centerline, candidate_source,
        operator_state, operator_reason, hold_remaining_s,
    ) -> dict:
        return {
            **self._build_single_boundary_result_metrics(result),
            **self._build_operator_control_metrics(
                result, midline, ctrl, raw_centerline, candidate_source,
                operator_state, operator_reason, hold_remaining_s,
            ),
        }

    def _build_single_boundary_result_metrics(self, result) -> dict:
        return {
            "candidate_diagonal_count": result.candidate_count,
            "selected_chain_length": result.selected_chain_length,
            "selected_chain_median_width_m": result.selected_chain_width_median,
            "expected_width_prior_m": result.expected_width_prior_m,
            **self._build_single_boundary_reject_metrics(result),
            **self._build_single_boundary_path_metrics(result),
        }

    def _build_single_boundary_reject_metrics(self, result) -> dict:
        return {
            "reject_wrong_side_count": result.reject_counts.get("wrong_side", 0),
            "reject_width_count": result.reject_counts.get("width", 0),
            "reject_width_range_count": result.reject_counts.get("width_range", 0),
            "reject_width_prior_count": result.reject_counts.get("width_prior", 0),
            "reject_orientation_count": result.reject_counts.get("orientation", 0),
            "reject_progress_count": result.reject_counts.get("progress", 0),
            "reject_near_field_continuity_count": result.reject_counts.get("near_field_continuity", 0),
            "reject_midpoint_kink_count": result.reject_counts.get("midpoint_kink", 0),
            "reject_seed_distance_count": result.reject_counts.get("seed_distance", 0),
        }

    def _build_single_boundary_path_metrics(self, result) -> dict:
        return {
            "near_field_lateral_max_m": result.near_field_lateral_max_m,
            "near_field_lateral_mean_m": result.near_field_lateral_mean_m,
            "near_field_displacement_max_m": result.near_field_displacement_max_m,
            "near_field_displacement_mean_m": result.near_field_displacement_mean_m,
            "near_field_midpoint_kink_max_rad": result.near_field_kink_max_rad,
            "seed_midpoint_distance_m": result.seed_midpoint_distance_m,
            "seed_temporal_offset_m": result.seed_temporal_offset_m,
        }

    def _build_operator_control_metrics(
        self, result, midline, ctrl, raw_centerline, candidate_source,
        operator_state, operator_reason, hold_remaining_s,
    ) -> dict:
        return {
            **self._build_operator_state_metrics(midline, operator_state, operator_reason, hold_remaining_s),
            "control_path_point_count": ctrl.control_path_point_count,
            "zero_cmd_sent_flag": ctrl.zero_cmd_sent_flag,
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
            "candidate_source": candidate_source,
            **self._build_midline_diagnostic_metrics(midline),
        }

    def _build_operator_state_metrics(self, midline, operator_state, operator_reason, hold_remaining_s) -> dict:
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
        }

    def _build_midline_diagnostic_metrics(self, midline) -> dict:
        return {
            "midline_update_mode": "hold" if midline.publish_mode == "held" else self._last_midline_update_mode,
            "midline_update_reason": getattr(self, "_last_midline_candidate_update_reason", ""),
            "midline_candidate_jump_m": getattr(self, "_last_midline_candidate_jump_m", float("nan")),
            "midline_near_lateral_delta_max_m": getattr(
                self, "_last_midline_near_lateral_delta_max_m", float("nan"),
            ),
            "midline_buffer_confidence": getattr(self, "_last_midline_buffer_confidence", float("nan")),
            "midline_recovery_count": getattr(self, "_midline_recovery_count", 0),
            **self._midline_estimation_metrics_for_diagnostics(),
        }

    def _publish_cycle_results(
        self, target_frame, raw_centerline, result, midline, ctrl,
        diag_metrics, operator_state, operator_reason, hold_remaining_s,
    ) -> None:
        self._publish_diagnostics(
            frame_id=target_frame,
            centerline_point_count=int(midline.centerline.shape[0]),
            selected_edge_count=int(result.selected_edges.shape[0]), status=midline.status,
            control_debug_metrics=ctrl.control_debug_metrics, planner_metrics=diag_metrics,
        )
        self._current_pair_segments_for_viz = np.array(midline.pair_segments_for_viz, copy=True)
        self._publish_outputs(
            frame_id=target_frame, centerline=midline.centerline, raw_centerline=raw_centerline,
            raw_midpoint_chain=midline.raw_midpoint_chain, result=result, status=midline.status,
            control_target_frame=ctrl.control_target_frame, cmd_speed=ctrl.cmd_speed,
            cmd_steering=ctrl.cmd_steering, lookahead=ctrl.lookahead,
            operator_state=operator_state, operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s, control_path_point_count=ctrl.control_path_point_count,
            candidate_diagonal_count=int(result.candidate_count),
            selected_chain_length=int(result.selected_chain_length),
            seed_midpoint_distance_m=float(result.seed_midpoint_distance_m),
            near_field_lateral_max_m=float(result.near_field_lateral_max_m),
            near_field_midpoint_kink_max_rad=float(result.near_field_kink_max_rad),
        )

    def _remember_pairs(self, *, result: SingleBoundaryPlannerResult, now_sec: float) -> None:
        del result
        del now_sec

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
        result: SingleBoundaryPlannerResult,
        now_sec: float,
        support_centerline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        direct_commit = (
            candidate_source in {"validated", "single_boundary_raw_offset"}
            and candidate_update_ok is not False
            and candidate_centerline.shape[0] >= 2
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
        result: SingleBoundaryPlannerResult,
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
        if candidate_source == "single_boundary_raw_offset":
            return True, "single_boundary_raw_offset_soft_accept"
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
            stored_samples, candidate_samples, vehicle_x, vehicle_y, vehicle_yaw, horizon_m,
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
        horizon_m: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        stored_local = self._transition_path_to_vehicle_frame(stored_samples, vehicle_x, vehicle_y, vehicle_yaw)
        candidate_local = self._transition_path_to_vehicle_frame(candidate_samples, vehicle_x, vehicle_y, vehicle_yaw)
        stored_local = self._local_forward_prefix_samples(path_local=stored_local, horizon_m=horizon_m)
        candidate_local = self._local_forward_prefix_samples(path_local=candidate_local, horizon_m=horizon_m)
        count = min(stored_local.shape[0], candidate_local.shape[0])
        if count < _TRANSITION_MIN_PATH_POINTS:
            return None, None
        return stored_local[:count], candidate_local[:count]

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
            "lateral_max_m": float(np.max(lateral)) if lateral.size else 0.0,
            "lateral_mean_m": float(np.mean(lateral)) if lateral.size else 0.0,
            "displacement_max_m": float(np.max(displacement)) if displacement.size else 0.0,
            "displacement_mean_m": float(np.mean(displacement)) if displacement.size else 0.0,
            "heading_delta_rad": self._transition_heading_delta(stored_local, candidate_local),
        }

    @staticmethod
    def _transition_heading_delta(stored_local: np.ndarray, candidate_local: np.ndarray) -> float:
        candidate_heading = math.atan2(
            float(candidate_local[1, 1] - candidate_local[0, 1]),
            float(candidate_local[1, 0] - candidate_local[0, 0]),
        )
        stored_heading = math.atan2(
            float(stored_local[1, 1] - stored_local[0, 1]),
            float(stored_local[1, 0] - stored_local[0, 0]),
        )
        return abs(
            float(
                math.atan2(
                    math.sin(candidate_heading - stored_heading),
                    math.cos(candidate_heading - stored_heading),
                )
            )
        )

    def _local_forward_prefix_samples(
        self,
        *,
        path_local: np.ndarray,
        horizon_m: float,
    ) -> np.ndarray:
        pts = np.asarray(path_local, dtype=np.float64)
        if pts.shape[0] < _TRANSITION_MIN_PATH_POINTS:
            return np.empty((0, 2), dtype=np.float64)
        valid_mask = (
            np.isfinite(pts[:, 0])
            & np.isfinite(pts[:, 1])
            & (pts[:, 0] >= _TRANSITION_FORWARD_X_MIN_M)
        )
        pts = pts[valid_mask]
        if pts.shape[0] < _TRANSITION_MIN_PATH_POINTS:
            return np.empty((0, 2), dtype=np.float64)
        cumulative = self._path_cumulative_lengths(pts)
        total = min(float(cumulative[-1]), max(_TRANSITION_MIN_HORIZON_M, float(horizon_m)))
        if total <= _TRANSITION_LENGTH_EPSILON_M:
            return np.asarray(pts[:1], dtype=np.float64)
        step = max(_TRANSITION_MIN_STATION_STEP_M, float(self.midline_station_spacing_m))
        samples = np.arange(0.0, total + _TRANSITION_SAMPLE_END_EPSILON_M, step, dtype=np.float64)
        if samples.size == 0 or samples[-1] < total:
            samples = np.concatenate((samples, [total]))
        return self._sample_path_at_lengths(pts, cumulative, samples)

    def _select_candidate_centerline(
        self,
        result: SingleBoundaryPlannerResult,
    ) -> tuple[np.ndarray, str]:
        if result.centerline.shape[0] > 0:
            return np.array(result.centerline, copy=True), "validated"
        if result.planner_mode == "single_boundary" and result.raw_offset_path.shape[0] > 0:
            fallback = _finalize_path(result.raw_offset_path, self._core_config)
            if fallback.shape[0] > 0:
                return np.array(fallback, copy=True), "single_boundary_raw_offset"
        return np.empty((0, 2), dtype=np.float64), "none"

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

    def _normalize_core_reject_reason(self, result: Optional[SingleBoundaryPlannerResult]) -> str:
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



def main(args=None) -> None:
    rclpy.init(args=args)
    node = SingleBoundaryPlannerNode()
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
