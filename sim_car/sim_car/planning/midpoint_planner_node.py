#!/usr/bin/env python3
"""Midpoint boundary planner over tracked cone detections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vehicle_plotter_msgs.msg import ConeDetectionArray

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.triangulation_planner_core import (
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
    compute_midpoint_centerline,
    update_track_width_estimate,
)
from sim_car.planning.planner_runtime_types import PlannerIdentity
from sim_car.planning.planner_constants import (
    CENTERLINE_MARKER_WIDTH_M as _CENTERLINE_MARKER_WIDTH_M,
    MSG_TRACK_STATE_STALE,
    MSG_TRACK_STATE_TENTATIVE,
    PAIR_PASSED_MARGIN_M as _PAIR_PASSED_MARGIN_M,
)
from sim_car.planning.tracked_cone_planner_contract import (
    COMMON_MIGRATED_TRACKED_CONE_PLANNER_DEFAULTS,
    apply_common_config_to_node,
    read_migrated_tracked_cone_planner_common_config,
)
from sim_car.planning.tracked_cone_planner_base import TrackedConePlannerBase
from sim_car.planning.tracked_cone_planner_geometry import (
    _base_point_to_odom,
    _odom_point_to_base,
)

_MIDPOINT_CHAIN_SOURCE = "midpoint_chain"
_MIDPOINT_CHAIN_MIN_POINTS = 2


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

        self._init_common_planner_state()
        self._pair_memory: list[_PairMemoryEntry] = []
        self._active_chain_stage = "waiting"
        self._active_reject_wrong_side_count = 0
        self._active_reject_width_count = 0
        self._active_reject_width_range_count = 0
        self._active_reject_progress_count = 0
        self._active_reject_orientation_count = 0

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
            "pairing.max_pair_width_m": 8.0,
            "pairing.max_width_jump_m": 2.0,
            "pairing.min_pair_count": 2,
            "pairing.pair_hold_time_s": 1.25,
            "pairing.pair_reassignment_margin": 0.25,
            "pairing.pair_inward_projection_tolerance_m": 0.15,
            "pairing.tangent_neighbor_count": 4,
            "pairing.enforce_opposite_color_pairing": True,
            "pairing.enforce_geometry_pairing_gate": False,
            "width_estimation.min_trustworthy_pairs": 2,
            "centerline.smoothing_window": 3,
            "centerline.max_heading_delta_rad": 0.75,
            "centerline.max_midpoint_segment_length_m": 10.0,
            "centerline.midpoint_order_reference_handoff_m": 6.0,
            "centerline.midpoint_order_history_size": 3,
            "centerline.midpoint_order_backtrack_tolerance_m": 0.35,
            "lap_tracking.target_laps": 0,
            "validation.candidate_min_points": 2,
            "validation.candidate_min_extent_m": 0.25,
            "validation.min_path_points": 4,
            "validation.min_forward_extent_m": 2.0,
            "validation.max_near_field_lateral_jump_m_sparse_pairs": 0.9,
            "validation.max_start_heading_error_rad": 1.0,
            "diagnostics.topic": "/midpoint_planner/diagnostics",
            "debug.show_raw_offset_path": True,
        })
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        common = read_migrated_tracked_cone_planner_common_config(
            self,
            planner_label='midpoint planner',
            diagnostics_topic_fallback=self._planner_identity.diagnostics_topic,
        )
        apply_common_config_to_node(self, common)
        self.show_raw_offset_path = bool(
            self.get_parameter("debug.show_raw_offset_path").value
        )
        self.lap_tracking_target_laps = max(
            0,
            int(self.get_parameter("lap_tracking.target_laps").value),
        )
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
            pair_inward_projection_tolerance_m=max(
                0.0,
                float(self.get_parameter("pairing.pair_inward_projection_tolerance_m").value),
            ),
            pairing_tangent_neighbor_count=max(
                2,
                int(self.get_parameter("pairing.tangent_neighbor_count").value),
            ),
            enforce_opposite_color_pairing=bool(
                self.get_parameter("pairing.enforce_opposite_color_pairing").value
            ),
            enforce_geometry_pairing_gate=bool(
                self.get_parameter("pairing.enforce_geometry_pairing_gate").value
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
            max_midpoint_segment_length_m=max(
                self.centerline_path_resolution_m,
                float(self.get_parameter("centerline.max_midpoint_segment_length_m").value),
            ),
            midpoint_order_reference_handoff_m=max(
                self.centerline_path_resolution_m,
                float(self.get_parameter("centerline.midpoint_order_reference_handoff_m").value),
            ),
            midpoint_order_history_size=max(
                2,
                int(self.get_parameter("centerline.midpoint_order_history_size").value),
            ),
            midpoint_order_backtrack_tolerance_m=max(
                0.0,
                float(self.get_parameter("centerline.midpoint_order_backtrack_tolerance_m").value),
            ),
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
        ctx = self._resolve_cone_planning_context()
        if ctx is None:
            return
        cones_msg, target_frame, vehicle_x, vehicle_y, vehicle_yaw, points_xy, colors, confidences = ctx
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        self._update_smalltrack_lap_from_orange_cones(
            cones_msg=cones_msg, points_xy=points_xy,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        self._update_remembered_cone_viz(points_xy=points_xy, colors=colors)
        cone_counts = self._count_cone_types(cones_msg=cones_msg, colors=colors)
        planning_frame, result, pair_segments_for_viz, raw_midpoint_chain = (
            self._prepare_planning_inputs(
                cones_msg=cones_msg, points_xy=points_xy, colors=colors,
                confidences=confidences, target_frame=target_frame,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            )
        )
        result.planner_mode = self._planner_identity.planner_mode
        pipeline = self._run_planning_pipeline(
            result=result, planning_frame=planning_frame, points_xy=points_xy,
            raw_midpoint_chain=raw_midpoint_chain, pair_segments_for_viz=pair_segments_for_viz,
            target_frame=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw, now_sec=now_sec,
        )
        self._update_active_diagnostics_fields(
            result=result, publish_mode=pipeline["publish_mode"],
            centerline=pipeline["centerline"],
            pair_segments_for_viz=pipeline["pair_segments_for_viz"],
        )
        self._publish_cycle(
            target_frame=target_frame, result=result, cone_counts=cone_counts,
            now_sec=now_sec, pipeline=pipeline,
        )

    def _run_planning_pipeline(
        self,
        *,
        result,
        planning_frame,
        points_xy: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        pair_segments_for_viz: np.ndarray,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
    ) -> dict:
        tracked_delta_p95_m, selected_edge_churn, selected_edge_churn_count = (
            self._compute_stability_metrics(result=result, points_xy=points_xy)
        )
        raw_centerline, candidate_source = self._select_candidate_centerline(
            result=result, support_chain=raw_midpoint_chain, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        centerline_jump_max_m = self._update_centerline_jump_metric(raw_centerline)
        self._emit_stability_warnings(
            centerline_jump_max_m=centerline_jump_max_m, selected_edge_churn=selected_edge_churn,
        )
        centerline, candidate_update_ok, candidate_update_reason, status = self._resolve_centerline(
            result=result, raw_centerline=raw_centerline, candidate_source=candidate_source,
            planning_frame=planning_frame, pair_segments_for_viz=pair_segments_for_viz,
            raw_midpoint_chain=raw_midpoint_chain, target_frame=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw, now_sec=now_sec,
        )
        centerline, publish_mode, hold_reason, pair_segments_for_viz, raw_midpoint_chain = (
            self._resolve_publish_mode(
                centerline=centerline, candidate_update_ok=candidate_update_ok,
                candidate_update_reason=candidate_update_reason, result=result,
                pair_segments_for_viz=pair_segments_for_viz, raw_midpoint_chain=raw_midpoint_chain,
                target_frame=target_frame, vehicle_x=vehicle_x, vehicle_y=vehicle_y,
                vehicle_yaw=vehicle_yaw, now_sec=now_sec, status=status,
            )
        )
        centerline = self._apply_smoothing_and_record(
            centerline=centerline, publish_mode=publish_mode, now_sec=now_sec,
            raw_midpoint_chain=raw_midpoint_chain,
            selected_chain_width_median=result.selected_chain_width_median,
            pair_segments_for_viz=pair_segments_for_viz,
            selected_pair_track_ids=result.selected_pair_track_ids,
        )
        control_path = self._centerline_to_vehicle_frame(
            centerline=centerline, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        cmd_speed, cmd_steering, lookahead, control_target_frame, control_debug_metrics, \
            zero_cmd_sent_flag, controller_failed = self._run_controller_step(
                control_path=control_path, target_frame=target_frame,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            )
        hold_remaining_s = self._hold_remaining_s(now_sec)
        operator_state, operator_reason = self._resolve_operator_state(
            centerline=centerline, publish_mode=publish_mode, status=status,
            zero_cmd_sent_flag=zero_cmd_sent_flag, controller_failed=controller_failed,
            control_path_point_count=int(control_path.shape[0]),
            hold_remaining_s=hold_remaining_s, result=result,
        )
        return self._assemble_pipeline_dict(
            centerline=centerline, raw_centerline=raw_centerline,
            raw_midpoint_chain=raw_midpoint_chain, pair_segments_for_viz=pair_segments_for_viz,
            candidate_source=candidate_source, publish_mode=publish_mode,
            hold_reason=hold_reason, status=status, operator_state=operator_state,
            operator_reason=operator_reason, hold_remaining_s=hold_remaining_s,
            control_path=control_path, control_target_frame=control_target_frame,
            control_debug_metrics=control_debug_metrics, cmd_speed=cmd_speed,
            cmd_steering=cmd_steering, lookahead=lookahead, zero_cmd_sent_flag=zero_cmd_sent_flag,
            controller_failed=controller_failed, selected_edge_churn=selected_edge_churn,
            selected_edge_churn_count=selected_edge_churn_count,
            tracked_delta_p95_m=tracked_delta_p95_m, centerline_jump_max_m=centerline_jump_max_m,
        )

    @staticmethod
    def _assemble_pipeline_dict(
        *,
        centerline, raw_centerline, raw_midpoint_chain, pair_segments_for_viz,
        candidate_source, publish_mode, hold_reason, status,
        operator_state, operator_reason, hold_remaining_s, control_path,
        control_target_frame, control_debug_metrics, cmd_speed, cmd_steering,
        lookahead, zero_cmd_sent_flag, controller_failed, selected_edge_churn,
        selected_edge_churn_count, tracked_delta_p95_m, centerline_jump_max_m,
    ) -> dict:
        return {
            "centerline": centerline, "raw_centerline": raw_centerline,
            "raw_midpoint_chain": raw_midpoint_chain, "pair_segments_for_viz": pair_segments_for_viz,
            "candidate_source": candidate_source, "publish_mode": publish_mode,
            "hold_reason": hold_reason, "status": status,
            "operator_state": operator_state, "operator_reason": operator_reason,
            "hold_remaining_s": hold_remaining_s, "control_path": control_path,
            "control_target_frame": control_target_frame,
            "control_debug_metrics": control_debug_metrics,
            "cmd_speed": cmd_speed, "cmd_steering": cmd_steering, "lookahead": lookahead,
            "zero_cmd_sent_flag": zero_cmd_sent_flag, "controller_failed": controller_failed,
            "selected_edge_churn": selected_edge_churn,
            "selected_edge_churn_count": selected_edge_churn_count,
            "tracked_delta_p95_m": tracked_delta_p95_m, "centerline_jump_max_m": centerline_jump_max_m,
        }

    def _prepare_planning_inputs(
        self,
        *,
        cones_msg,
        points_xy: np.ndarray,
        colors,
        confidences,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple:
        planning_frame = self._tracked_cone_planning_frame(
            msg=cones_msg, points_xy=points_xy, colors=colors, confidences=confidences,
        )
        self._active_remembered_cone_count = int(len(cones_msg.cones))
        self._active_stale_cone_count = int(
            np.count_nonzero(planning_frame.track_states == MSG_TRACK_STATE_STALE)
        )
        result, pair_segments_for_viz, raw_midpoint_chain = self._run_core_planning(
            planning_frame=planning_frame, target_frame=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        return planning_frame, result, pair_segments_for_viz, raw_midpoint_chain

    def _publish_cycle(
        self,
        *,
        target_frame: str,
        result,
        cone_counts: dict,
        now_sec: float,
        pipeline: dict,
    ) -> None:
        p = pipeline
        control_path_point_count = int(p["control_path"].shape[0])
        self._log_operator_state_transition(
            operator_state=p["operator_state"], operator_reason=p["operator_reason"],
            hold_remaining_s=p["hold_remaining_s"],
            selected_chain_length=int(result.selected_chain_length),
        )
        self._log_mode_summary(
            mode=self._active_planner_mode, result=result,
            operator_state=p["operator_state"], operator_reason=p["operator_reason"],
            hold_active=bool(self._active_held_path_flag),
        )
        self._publish_diagnostics(
            frame_id=target_frame, centerline_jump_max_m=p["centerline_jump_max_m"],
            selected_edge_churn_ratio=p["selected_edge_churn"],
            tracked_cones_frame_delta_p95_m=p["tracked_delta_p95_m"],
            centerline_point_count=int(p["centerline"].shape[0]),
            selected_edge_count=int(result.selected_edges.shape[0]),
            status=p["status"], control_debug_metrics=p["control_debug_metrics"],
            planner_metrics=self._build_planner_metrics(
                result=result, raw_centerline=p["raw_centerline"], cone_counts=cone_counts,
                candidate_source=p["candidate_source"], publish_mode=p["publish_mode"],
                hold_reason=p["hold_reason"], operator_state=p["operator_state"],
                operator_reason=p["operator_reason"], hold_remaining_s=p["hold_remaining_s"],
                control_path_point_count=control_path_point_count,
                zero_cmd_sent_flag=p["zero_cmd_sent_flag"],
                selected_edge_churn=p["selected_edge_churn"],
                selected_edge_churn_count=p["selected_edge_churn_count"],
            ),
        )
        self._current_pair_segments_for_viz = np.array(p["pair_segments_for_viz"], copy=True)
        self._publish_outputs(
            frame_id=target_frame, centerline=p["centerline"],
            raw_centerline=p["raw_centerline"], raw_midpoint_chain=p["raw_midpoint_chain"],
            result=result, status=p["status"],
            control_target_frame=p["control_target_frame"], cmd_speed=p["cmd_speed"],
            cmd_steering=p["cmd_steering"], lookahead=p["lookahead"],
            operator_state=p["operator_state"], operator_reason=p["operator_reason"],
            hold_remaining_s=p["hold_remaining_s"],
            control_path_point_count=control_path_point_count,
            candidate_diagonal_count=int(result.candidate_count),
            selected_chain_length=int(result.selected_chain_length),
            seed_midpoint_distance_m=float(result.seed_midpoint_distance_m),
            near_field_lateral_max_m=float(result.near_field_lateral_max_m),
            near_field_midpoint_kink_max_rad=float(result.near_field_kink_max_rad),
        )

    def _count_cone_types(self, *, cones_msg, colors) -> dict:
        return {
            "raw_orange_count": sum(
                1 for cone in cones_msg.cones
                if normalize_color(getattr(cone, "color", "")) == "orange"
            ),
            "boundary_hint_count": sum(
                1 for cone in cones_msg.cones
                if str(getattr(cone, "boundary_color", "")).strip()
            ),
            "resolved_blue_count": sum(1 for color in colors if color == "blue"),
            "resolved_yellow_count": sum(1 for color in colors if color == "yellow"),
        }

    def _run_core_planning(
        self,
        *,
        planning_frame,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple:
        remembered_pair_entries = self._active_pair_memory_entries(
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        result = self._call_core_planner(
            planning_frame=planning_frame, remembered_pair_entries=remembered_pair_entries,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        pair_segments_for_viz, raw_midpoint_chain = self._merge_remembered_pair_geometry(
            result=result, remembered_pair_entries=remembered_pair_entries,
            target_frame=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        return result, pair_segments_for_viz, raw_midpoint_chain

    def _call_core_planner(
        self,
        *,
        planning_frame,
        remembered_pair_entries: list,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ):
        prior = self._build_midpoint_prior(
            remembered_pair_entries=remembered_pair_entries,
        )
        result = compute_midpoint_centerline(
            points_xy=planning_frame.points_xy,
            colors=planning_frame.colors,
            confidences=planning_frame.planner_confidences,
            track_ids=planning_frame.track_ids,
            raw_colors=planning_frame.raw_colors,
            vehicle_xy=(vehicle_x, vehicle_y),
            vehicle_yaw=vehicle_yaw,
            config=self._core_config,
            prior=prior,
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

    def _build_midpoint_prior(self, *, remembered_pair_entries: list) -> MidpointPlannerPrior:
        previous_centerline = None
        if self._midline_buffer_path is not None:
            previous_centerline = np.array(self._midline_buffer_path, copy=True)
        elif self._last_valid_centerline is not None:
            previous_centerline = np.array(self._last_valid_centerline, copy=True)
        return MidpointPlannerPrior(
            previous_centerline=previous_centerline,
            previous_width_m=self._filtered_track_width_m,
            previous_mode=self._active_planner_mode,
            previous_pairs=[
                (e.left_track_id, e.right_track_id) for e in remembered_pair_entries
            ],
        )

    def _merge_remembered_pair_geometry(
        self,
        *,
        result,
        remembered_pair_entries: list,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple:
        live_pair_entries = self._pair_entries_from_segments(
            pair_track_ids=result.selected_pair_track_ids,
            pair_segments=result.pair_segments,
            frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        combined_pair_entries = self._merge_pair_entries(
            remembered_entries=remembered_pair_entries, live_entries=live_pair_entries,
        )
        combined_pair_entries = self._sort_pair_entries_by_forward_progress(
            entries=combined_pair_entries,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        combined_pair_segments, combined_midpoint_chain = self._pair_geometry_from_memory(
            combined_pair_entries
        )
        pair_segments_for_viz = np.array(result.pair_segments, copy=True)
        raw_midpoint_chain = np.array(result.midpoints_raw, copy=True)
        if combined_pair_segments.size > 0:
            pair_segments_for_viz = combined_pair_segments
        if combined_midpoint_chain.size > 0:
            raw_midpoint_chain = combined_midpoint_chain
        return pair_segments_for_viz, raw_midpoint_chain

    def _compute_stability_metrics(
        self,
        *,
        result,
        points_xy: np.ndarray,
    ) -> tuple:
        tracked_delta_p95_m = tracked_cones_frame_delta_p95(self._previous_tracked_points, points_xy)
        self._previous_tracked_points = np.array(points_xy, copy=True)
        previous_edge_keys = set(self._previous_edge_keys)
        selected_keys = selected_edge_keys(
            points=result.filtered_points,
            edges=result.selected_edges,
            quantization_m=self.edge_quantization_m,
        )
        churn_count = edge_churn_count(previous_edge_keys, selected_keys)
        churn_ratio = edge_churn_ratio(previous_edge_keys, selected_keys)
        self._previous_edge_keys = set(selected_keys)
        return tracked_delta_p95_m, churn_ratio, churn_count

    def _update_centerline_jump_metric(self, raw_centerline: np.ndarray) -> float:
        jump_max_m = compute_centerline_jump_max(
            raw_centerline, self._previous_raw_centerline, self.centerline_jump_horizon_m,
        )
        self._previous_raw_centerline = (
            np.array(raw_centerline, copy=True) if raw_centerline.shape[0] > 0 else None
        )
        return jump_max_m

    def _emit_stability_warnings(
        self, *, centerline_jump_max_m: float, selected_edge_churn: float
    ) -> None:
        if centerline_jump_max_m > self.jump_warn_threshold_m:
            self._warn_throttled(
                "centerline_jump_warn",
                f"centerline jump {centerline_jump_max_m:.3f} m exceeded threshold "
                f"{self.jump_warn_threshold_m:.3f} m",
            )
        if selected_edge_churn > self.edge_churn_warn_threshold:
            self._warn_throttled(
                "edge_churn_warn",
                f"selected pair churn {selected_edge_churn:.3f} exceeded threshold "
                f"{self.edge_churn_warn_threshold:.3f}",
            )

    def _resolve_centerline(
        self,
        *,
        result,
        raw_centerline: np.ndarray,
        candidate_source: str,
        planning_frame,
        pair_segments_for_viz: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
    ) -> tuple:
        candidate_update_ok, candidate_update_reason = self._candidate_path_is_updateable(
            candidate_centerline=raw_centerline,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            result=result, candidate_source=candidate_source,
        )
        centerline = self._update_midline_buffer(
            candidate_centerline=raw_centerline, candidate_source=candidate_source,
            candidate_update_ok=candidate_update_ok,
            candidate_update_reason=candidate_update_reason,
            frame_id=target_frame, vehicle_x=vehicle_x, vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw, result=result, now_sec=now_sec,
            support_centerline=raw_midpoint_chain,
        )
        candidate_update_ok = bool(
            getattr(self, "_last_midline_candidate_update_ok", candidate_update_ok)
        )
        candidate_update_reason = str(
            getattr(self, "_last_midline_candidate_update_reason", candidate_update_reason)
        )
        centerline = self._anchor_centerline_near_vehicle(
            centerline=centerline, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        status = self._annotate_centerline_status(
            result=result, raw_centerline=raw_centerline, centerline=centerline,
            candidate_source=candidate_source, candidate_update_ok=candidate_update_ok,
        )
        if centerline.shape[0] > 0 and raw_centerline.shape[0] > 0:
            self._update_valid_pair_cache(
                result=result, frame_id=target_frame,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
                planning_frame=planning_frame, pair_segments_for_viz=pair_segments_for_viz,
            )
        return centerline, candidate_update_ok, candidate_update_reason, status

    def _annotate_centerline_status(
        self,
        *,
        result,
        raw_centerline: np.ndarray,
        centerline: np.ndarray,
        candidate_source: str,
        candidate_update_ok: bool,
    ) -> str:
        status = result.status
        if not candidate_update_ok and centerline.shape[0] > 0:
            status = f"{status}; holding stored midline"
        elif (
            raw_centerline.shape[0] > 0 and centerline.shape[0] > 0
            and (
                raw_centerline.shape != centerline.shape
                or not np.allclose(raw_centerline, centerline)
            )
        ):
            status = f"{status}; publishing stored midline"
        if candidate_source != "validated" and raw_centerline.shape[0] > 0:
            status = f"{status}; using {candidate_source}"
        return status

    def _update_valid_pair_cache(
        self,
        *,
        result,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        planning_frame,
        pair_segments_for_viz: np.ndarray,
    ) -> None:
        self._remember_pairs(
            result=result, frame_id=frame_id,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            track_ids=planning_frame.track_ids, track_states=planning_frame.track_states,
            planner_confidences=planning_frame.planner_confidences,
        )
        self._last_valid_pair_segments = (
            np.array(pair_segments_for_viz, copy=True)
            if pair_segments_for_viz.size > 0 else self._last_valid_pair_segments
        )
        self._last_valid_pair_track_ids = (
            np.array(result.selected_pair_track_ids, copy=True)
            if result.selected_pair_track_ids.size > 0 else self._last_valid_pair_track_ids
        )

    def _resolve_publish_mode(
        self,
        *,
        centerline: np.ndarray,
        candidate_update_ok: bool,
        candidate_update_reason: str,
        result,
        pair_segments_for_viz: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
        status: str,
    ) -> tuple:
        publish_mode = "fresh"
        hold_reason = result.reject_reason
        if centerline.shape[0] > 0 and candidate_update_ok:
            continue_holding = self._advance_hold_hysteresis(plan_ok=True)
            if continue_holding:
                held_centerline = self._held_centerline(now_sec)
                if held_centerline is not None:
                    centerline = held_centerline
                    publish_mode = "held"
                    hold_reason = result.reject_reason or candidate_update_reason or status
                    status = (
                        f"{status}; hysteresis holding previous valid centerline "
                        f"({self._hold_clean_frame_count}/{self.hold_exit_clean_frames})"
                    )
        else:
            self._advance_hold_hysteresis(plan_ok=False)
            held_centerline = self._held_centerline(now_sec)
            if held_centerline is not None:
                centerline = held_centerline
                publish_mode = "held"
                hold_reason = result.reject_reason or candidate_update_reason or status
                status = f"{status}; holding previous valid centerline"
            else:
                publish_mode = "held"
        if publish_mode == "held" and centerline.shape[0] > 0:
            held_pair_segments, held_raw_midpoint_chain = self._held_pair_geometry(now_sec=now_sec)
            if pair_segments_for_viz.size == 0 and held_pair_segments is not None:
                pair_segments_for_viz = held_pair_segments
            if raw_midpoint_chain.size == 0 and held_raw_midpoint_chain is not None:
                raw_midpoint_chain = held_raw_midpoint_chain
        centerline = self._prepare_centerline_for_current_pose(
            centerline=centerline, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        return centerline, publish_mode, hold_reason, pair_segments_for_viz, raw_midpoint_chain

    def _apply_smoothing_and_record(
        self,
        *,
        centerline: np.ndarray,
        publish_mode: str,
        now_sec: float,
        raw_midpoint_chain: np.ndarray,
        selected_chain_width_median: float,
        pair_segments_for_viz: np.ndarray,
        selected_pair_track_ids: np.ndarray,
    ) -> np.ndarray:
        if self.enable_temporal_smoothing and publish_mode == "fresh":
            centerline = self._apply_temporal_smoothing(centerline)
            self._previous_centerline = np.array(centerline, copy=True)
        elif publish_mode == "held" and centerline.shape[0] > 0:
            self._previous_centerline = np.array(centerline, copy=True)
        if publish_mode == "fresh" and centerline.shape[0] > 0:
            self._record_valid_plan(
                now_sec=now_sec, centerline=centerline,
                raw_midpoint_chain=raw_midpoint_chain,
                selected_chain_width_median=selected_chain_width_median,
            )
            if pair_segments_for_viz.size > 0:
                self._last_valid_pair_segments = np.array(pair_segments_for_viz, copy=True)
            if selected_pair_track_ids.size > 0:
                self._last_valid_pair_track_ids = np.array(selected_pair_track_ids, copy=True)
        return centerline

    def _run_controller_step(
        self,
        *,
        control_path: np.ndarray,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple:
        cmd_speed, cmd_steering, lookahead = 0.0, 0.0, 0.0
        control_target_frame: Optional[np.ndarray] = None
        control_debug_metrics: Optional[dict[str, float]] = None
        zero_cmd_sent_flag = 0
        controller_failed = False
        if control_path.shape[0] >= 1 and self._controller is not None:
            try:
                controller_output = self._controller.compute(
                    control_path=control_path,
                    speed_mps=self._latest_speed_mps,
                    yaw_rate_rps=self._latest_yaw_rate_rps,
                )
            except ValueError as exc:
                controller_failed, zero_cmd_sent_flag, cmd_speed, cmd_steering = (
                    self._handle_controller_error(exc)
                )
            else:
                cmd_speed, cmd_steering, lookahead, control_target_frame, control_debug_metrics = (
                    self._apply_successful_controller_output(
                        controller_output=controller_output, target_frame=target_frame,
                        vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
                    )
                )
        elif control_path.shape[0] >= 1:
            zero_cmd_sent_flag = int(self._apply_controller_disabled_behavior())
        else:
            zero_cmd_sent_flag = int(self._apply_no_path_behavior())
            if self._last_speed_cmd is not None:
                cmd_speed = float(self._last_speed_cmd)
            if self._last_steering_cmd is not None:
                cmd_steering = float(self._last_steering_cmd)
        return (
            cmd_speed, cmd_steering, lookahead,
            control_target_frame, control_debug_metrics,
            zero_cmd_sent_flag, controller_failed,
        )

    def _handle_controller_error(self, exc: ValueError) -> tuple:
        self._warn_throttled("controller_compute_error", f"controller compute failed: {exc}")
        zero_cmd_sent_flag = int(self._apply_no_path_behavior())
        cmd_speed = float(self._last_speed_cmd) if self._last_speed_cmd is not None else 0.0
        cmd_steering = float(self._last_steering_cmd) if self._last_steering_cmd is not None else 0.0
        return True, zero_cmd_sent_flag, cmd_speed, cmd_steering

    def _apply_successful_controller_output(
        self,
        *,
        controller_output,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple:
        cmd_steering = float(controller_output.steering_rad)
        cmd_speed = self._compute_speed_command(float(controller_output.kappa))
        lookahead = float(controller_output.lookahead_m)
        self._last_speed_cmd = cmd_speed
        self._last_steering_cmd = cmd_steering
        self._publish_cmd(cmd_speed, cmd_steering)
        control_target_frame, control_debug_metrics = self._extract_controller_debug(
            controller_output=controller_output, target_frame=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        return cmd_speed, cmd_steering, lookahead, control_target_frame, control_debug_metrics

    def _extract_controller_debug(
        self,
        *,
        controller_output,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple:
        control_target_base = np.asarray(controller_output.target_point_base, dtype=np.float64)
        control_target_frame: Optional[np.ndarray] = None
        if self._is_alias(target_frame, self.base_frame):
            control_target_frame = np.array(control_target_base, copy=True)
        elif self._is_alias(target_frame, self.odom_frame):
            tx, ty = _base_point_to_odom(
                float(control_target_base[0]), float(control_target_base[1]),
                vehicle_x, vehicle_y, vehicle_yaw,
            )
            control_target_frame = np.array([tx, ty], dtype=np.float64)
        control_debug_metrics: Optional[dict[str, float]] = None
        if controller_output.stanley_debug is not None:
            control_debug_metrics = self._build_stanley_debug_metrics(
                debug=controller_output.stanley_debug,
                control_target_frame=control_target_frame,
            )
        return control_target_frame, control_debug_metrics

    def _build_stanley_debug_metrics(self, *, debug, control_target_frame) -> dict:
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

    def _resolve_operator_state(
        self,
        *,
        centerline: np.ndarray,
        publish_mode: str,
        status: str,
        zero_cmd_sent_flag: int,
        controller_failed: bool,
        control_path_point_count: int,
        hold_remaining_s: float,
        result,
    ) -> tuple:
        core_reject_reason = self._normalize_core_reject_reason(result)
        if centerline.shape[0] == 0:
            if self._last_valid_centerline is not None and hold_remaining_s <= 0.0:
                operator_state, operator_reason = "stopped", "hold_expired_no_path"
            elif core_reject_reason != "none":
                operator_state, operator_reason = "stopped", core_reject_reason
            elif zero_cmd_sent_flag:
                operator_state, operator_reason = "stopped", "stop_if_no_path"
            else:
                operator_state, operator_reason = "held", "holding_previous_valid"
        elif publish_mode == "held":
            operator_state = "held"
            if centerline.shape[0] > 0 and "hysteresis holding previous valid centerline" in status:
                operator_reason = "hysteresis_holding"
            elif core_reject_reason != "none":
                operator_reason = core_reject_reason
            else:
                operator_reason = "holding_previous_valid"
        else:
            operator_state, operator_reason = "fresh", "none"
        if controller_failed:
            operator_state, operator_reason = "stopped", "controller_compute_failed"
        elif self._controller is None and centerline.shape[0] > 0:
            operator_state, operator_reason = "stopped", "controller_disabled"
        elif centerline.shape[0] > 0 and control_path_point_count <= 0 and operator_state != "waiting":
            operator_state, operator_reason = "stopped", "no_control_path"
        if zero_cmd_sent_flag and operator_reason == "none":
            operator_state, operator_reason = "stopped", "stop_if_no_path"
        if (
            control_path_point_count > 0 and centerline.shape[0] > 0
            and zero_cmd_sent_flag == 0 and publish_mode == "fresh"
        ):
            operator_state = "fresh"
        return operator_state, operator_reason

    def _update_active_diagnostics_fields(
        self, *, result, publish_mode: str, centerline: np.ndarray, pair_segments_for_viz: np.ndarray
    ) -> None:
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
        self._active_chain_stage = self._midpoint_debug_stage(result)
        self._active_reject_wrong_side_count = int(result.reject_counts.get("wrong_side", 0))
        self._active_reject_width_count = int(result.reject_counts.get("width", 0))
        self._active_reject_width_range_count = int(result.reject_counts.get("width_range", 0))
        self._active_reject_progress_count = int(result.reject_counts.get("progress", 0))
        self._active_reject_orientation_count = int(result.reject_counts.get("orientation", 0))

    def _build_planner_metrics(
        self,
        *,
        result,
        raw_centerline: np.ndarray,
        cone_counts: dict,
        candidate_source: str,
        publish_mode: str,
        hold_reason,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
        control_path_point_count: int,
        zero_cmd_sent_flag: int,
        selected_edge_churn: float,
        selected_edge_churn_count: int,
    ) -> dict:
        return {
            **self._build_result_quality_metrics(
                result=result, selected_edge_churn=selected_edge_churn,
                selected_edge_churn_count=selected_edge_churn_count,
            ),
            **self._build_operator_state_metrics(
                publish_mode=publish_mode, hold_reason=hold_reason,
                operator_state=operator_state, operator_reason=operator_reason,
                hold_remaining_s=hold_remaining_s, control_path_point_count=control_path_point_count,
                zero_cmd_sent_flag=zero_cmd_sent_flag,
            ),
            **self._build_chain_count_metrics(result=result),
            "raw_candidate_point_count": int(raw_centerline.shape[0]),
            **cone_counts,
            "candidate_source": candidate_source,
            **self._build_midline_buffer_metrics(publish_mode=publish_mode),
        }

    def _build_result_quality_metrics(
        self,
        *,
        result,
        selected_edge_churn: float,
        selected_edge_churn_count: int,
    ) -> dict:
        rc = result.reject_counts
        return {
            "candidate_diagonal_count": result.candidate_count,
            "selected_chain_length": result.selected_chain_length,
            "selected_chain_median_width_m": result.selected_chain_width_median,
            "expected_width_prior_m": result.expected_width_prior_m,
            "reject_wrong_side_count": rc.get("wrong_side", 0),
            "reject_width_count": rc.get("width", 0),
            "reject_width_range_count": rc.get("width_range", 0),
            "reject_width_prior_count": rc.get("width_prior", 0),
            "reject_orientation_count": rc.get("orientation", 0),
            "reject_progress_count": rc.get("progress", 0),
            "reject_near_field_continuity_count": rc.get("near_field_continuity", 0),
            "reject_midpoint_kink_count": rc.get("midpoint_kink", 0),
            "reject_seed_distance_count": rc.get("seed_distance", 0),
            "near_field_lateral_max_m": result.near_field_lateral_max_m,
            "near_field_lateral_mean_m": result.near_field_lateral_mean_m,
            "near_field_displacement_max_m": result.near_field_displacement_max_m,
            "near_field_displacement_mean_m": result.near_field_displacement_mean_m,
            "near_field_midpoint_kink_max_rad": result.near_field_kink_max_rad,
            "seed_midpoint_distance_m": result.seed_midpoint_distance_m,
            "seed_temporal_offset_m": result.seed_temporal_offset_m,
            "selected_chain_churn_count": selected_edge_churn_count,
            "selected_chain_churn_ratio": selected_edge_churn,
        }

    def _build_operator_state_metrics(
        self,
        *,
        publish_mode: str,
        hold_reason,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
        control_path_point_count: int,
        zero_cmd_sent_flag: int,
    ) -> dict:
        return {
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
        }

    def _build_chain_count_metrics(self, *, result) -> dict:
        return {
            "left_chain_length": result.left_chain_length,
            "right_chain_length": result.right_chain_length,
            "accepted_pair_count": result.accepted_pair_count,
            "unknown_pair_count": result.unknown_pair_count,
            "filtered_track_width_m": self._filtered_track_width_m,
            "held_path_flag": self._active_held_path_flag,
        }

    def _build_midline_buffer_metrics(self, *, publish_mode: str) -> dict:
        return {
            "midline_update_mode": (
                "hold" if publish_mode == "held" else self._last_midline_update_mode
            ),
            "midline_update_reason": getattr(self, "_last_midline_candidate_update_reason", ""),
            "midline_candidate_jump_m": getattr(self, "_last_midline_candidate_jump_m", float("nan")),
            "midline_near_lateral_delta_max_m": getattr(
                self, "_last_midline_near_lateral_delta_max_m", float("nan"),
            ),
            "midline_buffer_confidence": getattr(
                self, "_last_midline_buffer_confidence", float("nan"),
            ),
            "midline_recovery_count": getattr(self, "_midline_recovery_count", 0),
            **self._midline_estimation_metrics_for_diagnostics(),
        }

    @staticmethod
    def _tentative_cone_is_usable_for_planning(
        *,
        raw_color: str,
        boundary_color: str,
    ) -> bool:
        normalized_boundary = str(boundary_color).strip().lower()
        if normalized_boundary in {"blue", "yellow"}:
            return True
        return normalize_color(raw_color) in {"blue", "yellow", "orange"}

    def _active_pair_memory_entries(
        self,
        *,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[_PairMemoryEntry]:
        keep: list[_PairMemoryEntry] = []
        for entry in self._pair_memory:
            midpoint_x_base, midpoint_y_base = _odom_point_to_base(
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

        local = [
            _odom_point_to_base(
                e.midpoint_x_odom, e.midpoint_y_odom, vehicle_x, vehicle_y, vehicle_yaw
            )
            for e in entries
        ]

        # Pick the starting entry: prefer ahead of vehicle, then nearest.
        def start_key(i: int) -> tuple[float, float, float, int]:
            x, y = local[i]
            return (0.0 if x >= 0.0 else 1.0, math.hypot(x, y), abs(y), i)

        remaining = list(range(len(entries)))
        start = min(remaining, key=start_key)
        ordered = [start]
        remaining.remove(start)

        # Use direction from vehicle to first pair as the initial reference so
        # pairs that are to the side (during a turn) are not misordered.
        sp = local[start]
        sp_norm = math.hypot(sp[0], sp[1])
        ref = np.array(sp, dtype=np.float64) / sp_norm if sp_norm > 1e-9 else np.array([1.0, 0.0])

        _history_size = 3
        _backtrack_tol_m = 0.35

        while remaining:
            curr = np.array(local[ordered[-1]], dtype=np.float64)
            # Update reference direction from recent chain history.
            if len(ordered) >= 2:
                hist = max(0, len(ordered) - _history_size)
                delta = curr - np.array(local[ordered[hist]], dtype=np.float64)
                norm = float(np.hypot(delta[0], delta[1]))
                if norm > 1e-9:
                    ref = delta / norm

            best_i: Optional[int] = None
            best_cost = float("inf")
            for i in remaining:
                pt = np.array(local[i], dtype=np.float64)
                delta = pt - curr
                dist = float(np.hypot(delta[0], delta[1]))
                if dist < 1e-9:
                    continue
                forward = float(np.dot(delta, ref))
                if forward < -_backtrack_tol_m:
                    continue
                cost = dist + 2.0 * max(0.0, -forward)
                if cost < best_cost:
                    best_cost = cost
                    best_i = i

            if best_i is None:
                break
            ordered.append(best_i)
            remaining.remove(best_i)

        # Append any unreachable entries at the end as a fallback.
        for i in remaining:
            ordered.append(i)

        return [entries[i] for i in ordered]

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
                left_x_odom, left_y_odom = _base_point_to_odom(
                    left_x_odom,
                    left_y_odom,
                    vehicle_x,
                    vehicle_y,
                    vehicle_yaw,
                )
                right_x_odom, right_y_odom = _base_point_to_odom(
                    right_x_odom,
                    right_y_odom,
                    vehicle_x,
                    vehicle_y,
                    vehicle_yaw,
                )
                midpoint_x_odom, midpoint_y_odom = _base_point_to_odom(
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
                left_x_odom, left_y_odom = _base_point_to_odom(
                    left_x_odom,
                    left_y_odom,
                    vehicle_x,
                    vehicle_y,
                    vehicle_yaw,
                )
                right_x_odom, right_y_odom = _base_point_to_odom(
                    right_x_odom,
                    right_y_odom,
                    vehicle_x,
                    vehicle_y,
                    vehicle_yaw,
                )
                midpoint_x_odom, midpoint_y_odom = _base_point_to_odom(
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
        merged: dict[tuple[int, int], _PairMemoryEntry] = {}
        for entry in self._pair_memory:
            merged[(entry.left_track_id, entry.right_track_id)] = entry
        for entry in remembered_pairs:
            key = (entry.left_track_id, entry.right_track_id)
            # A live pair that shares a track ID with a memory pair means the
            # cone is now matched to a better partner — evict the stale pairing.
            stale = [k for k in merged if k != key and (k[0] == key[0] or k[1] == key[1])]
            for k in stale:
                del merged[k]
            merged[key] = entry
        self._pair_memory = list(merged.values())

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
        result: MidpointPlannerResult,
        now_sec: float,
        support_centerline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        direct_commit = self._midpoint_candidate_should_commit_directly(
            result=result,
            candidate_centerline=candidate_centerline,
            candidate_source=candidate_source,
            candidate_update_ok=candidate_update_ok,
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

    def _midpoint_candidate_should_commit_directly(
        self,
        *,
        result: MidpointPlannerResult,
        candidate_centerline: np.ndarray,
        candidate_source: str,
        candidate_update_ok: Optional[bool],
    ) -> bool:
        if candidate_source not in {"validated", _MIDPOINT_CHAIN_SOURCE}:
            return False
        if candidate_update_ok is False:
            return False
        if np.asarray(candidate_centerline, dtype=np.float64).shape[0] < _MIDPOINT_CHAIN_MIN_POINTS:
            return False
        return True

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
        min_points = (
            _MIDPOINT_CHAIN_MIN_POINTS
            if candidate_source == _MIDPOINT_CHAIN_SOURCE
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
        if candidate_source == _MIDPOINT_CHAIN_SOURCE:
            return True, "ok"
        if candidate_source != "validated":
            return False, result.reject_reason or result.status or "unsupported_candidate_source"
        if result.status != "ok":
            return False, result.reject_reason or result.status
        return True, "ok"

    def _select_candidate_centerline(
        self,
        *,
        result: MidpointPlannerResult,
        support_chain: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple[np.ndarray, str]:
        candidates: list[tuple[float, int, str, np.ndarray]] = []

        def add_candidate(path: np.ndarray, source: str, priority: int) -> None:
            candidate = self._finite_midpoint_path(path)
            if candidate.shape[0] < _MIDPOINT_CHAIN_MIN_POINTS:
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
        add_candidate(support_chain, _MIDPOINT_CHAIN_SOURCE, 20)

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
    def _finite_midpoint_path(path: np.ndarray) -> np.ndarray:
        arr = np.asarray(path, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return np.empty((0, 2), dtype=np.float64)
        finite_mask = np.all(np.isfinite(arr), axis=1)
        if not np.any(finite_mask):
            return np.empty((0, 2), dtype=np.float64)
        return np.array(arr[finite_mask], copy=True)

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
        self._active_chain_stage = "waiting"
        self._active_reject_wrong_side_count = 0
        self._active_reject_width_count = 0
        self._active_reject_width_range_count = 0
        self._active_reject_progress_count = 0
        self._active_reject_orientation_count = 0
        self._last_midline_update_mode = "hold"
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
        del centerline_point_count
        del cmd_speed
        del cmd_steering
        del lookahead
        del seed_midpoint_distance_m
        del near_field_lateral_max_m
        del near_field_midpoint_kink_max_rad
        del hold_remaining_s
        reject_parts: list[str] = []
        if int(self._active_reject_wrong_side_count) > 0:
            reject_parts.append(f"wrong={int(self._active_reject_wrong_side_count)}")
        if int(self._active_reject_width_count) > 0:
            reject_parts.append(f"width={int(self._active_reject_width_count)}")
        if int(self._active_reject_width_range_count) > 0:
            reject_parts.append(f"range={int(self._active_reject_width_range_count)}")
        if int(self._active_reject_progress_count) > 0:
            reject_parts.append(f"progress={int(self._active_reject_progress_count)}")
        if int(self._active_reject_orientation_count) > 0:
            reject_parts.append(f"orient={int(self._active_reject_orientation_count)}")

        lines = [
            f"STATE: {operator_state.upper()}",
            f"MODE: {self._active_planner_mode.upper()}",
            f"REASON: {self._operator_reason_label(operator_reason)}",
            (
                f"FLOW: stage={str(self._active_chain_stage).upper()} | "
                f"L={int(self._active_left_chain_length)} | "
                f"R={int(self._active_right_chain_length)} | "
                f"pairs={int(self._active_pair_count)} | "
                f"unknown={int(self._active_unknown_pair_count)}"
            ),
        ]
        if reject_parts:
            lines.append("REJECTS: " + " | ".join(reject_parts))
        lines.append(self._lap_status_text())
        return "\n".join(
            lines
        )

    def _lap_status_text(self) -> str:
        if self.lap_tracking_target_laps > 0:
            return f"LAPS: {int(self._lap_tracking_completed_laps)}/{int(self.lap_tracking_target_laps)}"
        return f"LAPS: {int(self._lap_tracking_completed_laps)}/off"

    def _midpoint_debug_stage(self, result: MidpointPlannerResult) -> str:
        pair_ready = int(result.accepted_pair_count) >= int(
            getattr(self._core_config, "min_pair_count", 0)
        )
        if pair_ready:
            return "planning"
        if int(result.accepted_pair_count) > 0:
            return "ordering"
        if int(result.left_chain_length) > 0 and int(result.right_chain_length) > 0:
            return "pairing"
        return "searching"


    def _blend_midline_samples(
        self,
        *,
        stored_samples: np.ndarray,
        candidate_samples: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        direct_prefix_distance_m: Optional[float] = None,
    ) -> np.ndarray:
        if stored_samples.shape[0] == 0:
            return np.array(candidate_samples, copy=True)
        if candidate_samples.shape[0] == 0:
            return np.array(stored_samples, copy=True)
        count = min(stored_samples.shape[0], candidate_samples.shape[0])
        stored_local, candidate_local = self._project_samples_to_vehicle_frame(
            stored_samples=stored_samples, candidate_samples=candidate_samples,
            count=count, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        updated_local = self._apply_lateral_clip(
            stored_local=stored_local, candidate_local=candidate_local,
            count=count,
            direct_prefix_distance_m=direct_prefix_distance_m,
        )
        updated = self._convert_blended_local_to_odom(
            updated_local=updated_local, count=count,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        if candidate_samples.shape[0] > count:
            updated = np.vstack((updated, candidate_samples[count:]))
        elif stored_samples.shape[0] > count:
            updated = np.vstack((updated, stored_samples[count:]))
        return updated

    def _project_samples_to_vehicle_frame(
        self,
        *,
        stored_samples: np.ndarray,
        candidate_samples: np.ndarray,
        count: int,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple:
        stored_local = self._centerline_to_vehicle_frame(
            centerline=stored_samples[:count], frame_id=self.odom_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        candidate_local = self._centerline_to_vehicle_frame(
            centerline=candidate_samples[:count], frame_id=self.odom_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        return stored_local, candidate_local

    def _apply_lateral_clip(
        self,
        *,
        stored_local: np.ndarray,
        candidate_local: np.ndarray,
        count: int,
        direct_prefix_distance_m: Optional[float],
    ) -> np.ndarray:
        step = max(0.05, float(self.midline_station_spacing_m))
        direct_prefix_distance = (
            0.0 if direct_prefix_distance_m is None else float(direct_prefix_distance_m)
        )
        updated_local = np.array(stored_local, copy=True)
        for idx in range(count):
            distance_ahead = float(idx) * step
            alpha, max_shift = self._midline_blend_params(distance_ahead)
            delta_local = candidate_local[idx] - stored_local[idx]
            if distance_ahead <= direct_prefix_distance:
                updated_local[idx] = candidate_local[idx]
                continue
            updated_local[idx, 0] = candidate_local[idx, 0]
            lateral_delta = float(delta_local[1])
            if abs(lateral_delta) <= max_shift:
                updated_local[idx, 1] = candidate_local[idx, 1]
            else:
                updated_local[idx, 1] = stored_local[idx, 1] + float(
                    np.clip(alpha * lateral_delta, -max_shift, max_shift)
                )
            if idx > 0:
                updated_local[idx, 0] = max(updated_local[idx, 0], updated_local[idx - 1, 0] + 0.05)
        return updated_local

    def _convert_blended_local_to_odom(
        self,
        *,
        updated_local: np.ndarray,
        count: int,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> np.ndarray:
        updated = np.empty((count, 2), dtype=np.float64)
        for idx in range(count):
            ox, oy = _base_point_to_odom(
                float(updated_local[idx, 0]), float(updated_local[idx, 1]),
                vehicle_x, vehicle_y, vehicle_yaw,
            )
            updated[idx, 0] = ox
            updated[idx, 1] = oy
        return updated


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
