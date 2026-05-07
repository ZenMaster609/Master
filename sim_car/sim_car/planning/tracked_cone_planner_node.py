#!/usr/bin/env python3
"""Tracked-cone planner nodes: midpoint, corridor, and single-boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.corridor_planner_core import (
    CorridorPlannerConfig,
    CorridorPlannerPrior,
    CorridorPlannerResult,
    compute_corridor_centerline,
)
from sim_car.planning.midpoint_planner_core import (
    MidpointPlannerConfig,
    MidpointPlannerPrior,
    MidpointPlannerResult,
    compute_midpoint_centerline,
)
from sim_car.planning.planner_constants import (
    CENTERLINE_MARKER_WIDTH_M as _CENTERLINE_MARKER_WIDTH_M,
    MSG_TRACK_STATE_CONFIRMED,
    MSG_TRACK_STATE_STALE,
    MSG_TRACK_STATE_TENTATIVE,
    PAIR_PASSED_MARGIN_M as _PAIR_PASSED_MARGIN_M,
    VALIDATED_JUMP_ACCEPT_HEADING_DELTA_RAD as _VALIDATED_JUMP_ACCEPT_HEADING_DELTA_RAD,
    VALIDATED_JUMP_ACCEPT_HORIZON_M as _VALIDATED_JUMP_ACCEPT_HORIZON_M,
    VALIDATED_JUMP_ACCEPT_LATERAL_MAX_M as _VALIDATED_JUMP_ACCEPT_LATERAL_MAX_M,
    VALIDATED_JUMP_ACCEPT_LATERAL_MEAN_M as _VALIDATED_JUMP_ACCEPT_LATERAL_MEAN_M,
)
from sim_car.planning.planner_runtime_types import PlannerIdentity
from sim_car.planning.single_boundary_planner_core import (
    SingleBoundaryPlannerConfig,
    SingleBoundaryPlannerPrior,
    SingleBoundaryPlannerResult,
    compute_single_boundary_centerline,
)
from sim_car.planning.tracked_cone_planner_base import TrackedConePlannerBase
from sim_car.planning.tracked_cone_planner_contract import (
    apply_common_config_to_node,
    declare_tracked_cone_planner_parameters,
    read_migrated_tracked_cone_planner_common_config,
)
from sim_car.planning.tracked_cone_planner_geometry import (
    _base_point_to_odom,
    _finalize_path,
    _odom_point_to_base,
    update_track_width_estimate,
)

# -- Shared numeric constants -------------------------------------------------

_NANOSECONDS_TO_SECONDS = 1e-9  # ROS clock timestamps are nanoseconds; planner metrics use seconds.
_CANDIDATE_EXTENT_EPSILON_M = 1e-6  # Treats sub-micrometer forward extent as no usable path.
_CANDIDATE_EXTENT_TOLERANCE_MIN_M = 0.05  # Five centimeters avoids winner flips from resampling noise.
_CANDIDATE_EXTENT_TOLERANCE_STATION_FACTOR = 0.5  # Half-station tolerance preserves prior selection behavior.
_VALIDATED_CANDIDATE_PRIORITY = 30  # Validated core paths win ties when their extent is comparable.
_SUPPORT_CHAIN_CANDIDATE_PRIORITY = 20  # Live chains are preferred over weaker fallback sources.
_PAIR_SEGMENT_ENDPOINT_COUNT = 2  # Pair memory stores left and right cone endpoints only.
_PAIR_SEGMENT_COORD_COUNT = 2  # Cone endpoints are planar x/y coordinates.
_PAIR_MIDPOINT_WEIGHT = 0.5  # Midpoint is the average of the left and right endpoints.
_TRANSITION_MIN_PATH_POINTS = 2  # Two samples are required to measure path displacement and heading.
_TRANSITION_MIN_HORIZON_M = 0.25  # Keeps the near-field comparison meaningful for very short horizons.

# -- Corridor-specific constants ----------------------------------------------

_CORRIDOR_MIDPOINT_SOURCE = "corridor_midpoints"
_CORRIDOR_MIDPOINT_MIN_POINTS = 2  # Two points are required to form a drawable midpoint path.
_CORRIDOR_ANALYSIS_SAMPLE_COUNT = 8  # Eight samples keep diagnostics compact while showing near-field shape.
_CORRIDOR_ANALYSIS_SAMPLE_SPACING_M = 1.0  # One-meter spacing matches the operator-scale path preview.
_ANCHOR_TAPER_GATE_LATERAL_M = 0.20  # Allows small lateral drift while detecting off-axis starts.
_ANCHOR_TAPER_GATE_HEADING_RAD = 0.18  # Roughly ten degrees keeps the near-vehicle anchor stable.
_PREVALIDATION_CANDIDATE_PRIORITY = 10  # Prevalidation paths are useful only behind live support chains.
_MEMORY_CHAIN_CANDIDATE_PRIORITY = 5  # Remembered pairs are the weakest fallback source.
_PAIR_MEMORY_MERGE_DISTANCE_M = 0.35  # Merges remembered/live pairs that describe the same corridor gap.
_PAIR_SORT_MIN_STEP_M = 1e-6  # Avoids unstable ordering from duplicate midpoint coordinates.
_PAIR_SORT_RANGE_REGRESSION_TOLERANCE_M = 0.20  # Allows small range regressions on curved paths.
_PAIR_SORT_BACKWARD_GATE_M = 0.75  # Strongly penalizes steps that move behind the vehicle.
_PAIR_SORT_INITIAL_HEADING = np.asarray([1.0, 0.0], dtype=np.float64)

# -- Midpoint-specific constants ----------------------------------------------

_MIDPOINT_CHAIN_SOURCE = "midpoint_chain"
_MIDPOINT_CHAIN_MIN_POINTS = 2  # Two points are required to form a drawable midpoint path.
_PAIR_SORT_MIN_NORM = 1e-9  # Avoids unstable directions from duplicate midpoint coordinates.
_PAIR_SORT_INITIAL_REFERENCE = np.asarray([1.0, 0.0], dtype=np.float64)
_PAIR_SORT_HISTORY_SIZE = 3  # Recent three pairs smooth the direction estimate without lagging turns.
_PAIR_SORT_BACKTRACK_TOLERANCE_M = 0.35  # Allows small local backtracking on curved paths.
_PAIR_SORT_BACKTRACK_COST_WEIGHT = 2.0  # Penalizes reverse progress while still allowing recovery.

# -- Single-boundary-specific constants ---------------------------------------

_TRANSITION_FORWARD_X_MIN_M = -0.1  # Allows tiny pose-noise drift behind the vehicle while filtering old path.
_TRANSITION_LENGTH_EPSILON_M = 1e-6  # Treats sub-micrometer local paths as degenerate.
_TRANSITION_MIN_STATION_STEP_M = 0.05  # Prevents overly dense samples if station spacing is misconfigured.
_TRANSITION_SAMPLE_END_EPSILON_M = 1e-9  # Includes the final sample despite floating-point roundoff.

# =============================================================================
# Shared dataclasses (identical across corridor and single-boundary)
# =============================================================================


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


# =============================================================================
# Per-planner dataclasses
# =============================================================================


@dataclass
class _MidpointPairMemoryEntry:
    left_track_id: int
    right_track_id: int
    midpoint_x_odom: float
    midpoint_y_odom: float
    left_x_odom: float
    left_y_odom: float
    right_x_odom: float
    right_y_odom: float


@dataclass
class _CorridorPairMemoryEntry:
    left_x_odom: float
    left_y_odom: float
    right_x_odom: float
    right_y_odom: float
    midpoint_x_odom: float
    midpoint_y_odom: float
    last_valid_sec: float


@dataclass
class _SingleBoundaryPairMemoryEntry:
    left_track_id: int
    right_track_id: int
    last_valid_sec: float


@dataclass
class _CorridorInputMetrics:
    raw_orange_count: int
    boundary_hint_count: int
    resolved_blue_count: int
    resolved_yellow_count: int
    planning_frame: object  # TrackedConePlanningFrame


@dataclass
class _SingleBoundaryInputMetrics:
    planning_frame: object  # TrackedConePlanningFrame


# =============================================================================
# Shared base node
# =============================================================================


class GenericTrackedConePlannerNode(TrackedConePlannerBase):
    """Shared skeleton for all three tracked-cone planner nodes."""

    def __init__(self, identity: PlannerIdentity) -> None:
        self._planner_identity = identity
        Node.__init__(self, identity.node_name)
        self._declare_parameters()
        self._read_parameters()
        self._init_common_planner_state()
        self._init_algorithm_state()
        self._init_common_ros_interfaces()

    # -- Parameter lifecycle --------------------------------------------------

    def _declare_parameters(self) -> None:
        declare_tracked_cone_planner_parameters(
            self,
            diagnostics_topic_default=self._planner_identity.diagnostics_topic,
            **self._declare_parameter_overrides(),
        )
        self._declare_algorithm_parameters()

    def _declare_parameter_overrides(self) -> dict:
        return {}

    def _declare_algorithm_parameters(self) -> None:
        pass

    def _read_parameters(self) -> None:
        common = read_migrated_tracked_cone_planner_common_config(
            self,
            planner_label=self._planner_label(),
            diagnostics_topic_fallback=self._planner_identity.diagnostics_topic,
        )
        apply_common_config_to_node(self, common)
        self._read_algorithm_parameters()
        self._core_config = self._build_core_config()
        self._filtered_track_width_m = float(self._core_config.initial_width_m)

    def _planner_label(self) -> str:
        return self._planner_identity.planner_mode

    def _read_algorithm_parameters(self) -> None:
        pass

    def _build_core_config(self):
        raise NotImplementedError

    def _init_algorithm_state(self) -> None:
        pass

    # -- Shared helpers -------------------------------------------------------

    def _apply_width_estimate_update(self, result) -> None:
        if (
            result.accepted_pair_count >= int(self._core_config.min_trustworthy_pairs)
            and math.isfinite(float(result.selected_chain_width_median))
        ):
            self._filtered_track_width_m = update_track_width_estimate(
                self._filtered_track_width_m, result.selected_chain_width_median, self._core_config,
            )
        result.filtered_track_width_m = float(self._filtered_track_width_m)

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

    def _lap_status_text(self) -> str:
        if self.lap_tracking_target_laps > 0:
            return f"LAPS: {int(self._lap_tracking_completed_laps)}/{int(self.lap_tracking_target_laps)}"
        return f"LAPS: {int(self._lap_tracking_completed_laps)}/off"

    # -- Transition path helpers shared by corridor and single-boundary -------

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


# =============================================================================
# MidpointPlannerNode
# =============================================================================


class MidpointPlannerNode(GenericTrackedConePlannerNode):
    """Tracked-cone midpoint planner with shared path-memory stabilization."""

    def __init__(self) -> None:
        super().__init__(PlannerIdentity(
            node_name="midpoint_planner_node",
            planner_mode="midpoint",
            diagnostics_prefix="midpoint_planner",
            diagnostics_topic="/midpoint_planner/diagnostics",
        ))

    def _init_algorithm_state(self) -> None:
        self._pair_memory: list[_MidpointPairMemoryEntry] = []
        self._active_chain_stage = "waiting"
        self._active_reject_wrong_side_count = 0
        self._active_reject_width_count = 0
        self._active_reject_width_range_count = 0
        self._active_reject_progress_count = 0
        self._active_reject_orientation_count = 0

    def _planner_label(self) -> str:
        return 'midpoint planner'

    def _declare_algorithm_parameters(self) -> None:
        defaults = {
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
            "validation.min_path_points": 4,
            "validation.min_forward_extent_m": 2.0,
            "validation.max_near_field_lateral_jump_m_sparse_pairs": 0.9,
            "validation.max_start_heading_error_rad": 1.0,
            "debug.show_raw_offset_path": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_algorithm_parameters(self) -> None:
        self.show_raw_offset_path = bool(self.get_parameter("debug.show_raw_offset_path").value)

    def _build_core_config(self) -> MidpointPlannerConfig:
        values = {}
        values.update(self._filtering_config_values())
        values.update(self._boundary_chain_config_values())
        values.update(self._pairing_and_width_config_values())
        values.update(self._centerline_and_validation_config_values())
        return MidpointPlannerConfig(**values)

    def _filtering_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            "max_cone_range_m": profile.max_cone_range_m,
            "behind_drop_m": profile.behind_drop_m,
            "min_confidence": profile.min_confidence,
            "min_required_cones": max(2, profile.min_required_cones),
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
        profile = self._planner_algorithm_profile
        return {
            "min_step_m": profile.boundary_min_step_m,
            "max_step_m": profile.boundary_max_step_m,
            "max_heading_change_rad": profile.boundary_max_heading_change_rad,
            "min_forward_progress_m": profile.boundary_min_forward_progress_m,
            "min_chain_length": max(2, int(self.get_parameter("boundary_chain.min_chain_length").value)),
        }

    def _pairing_and_width_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            **self._pairing_config_values(),
            "initial_width_m": profile.initial_width_m,
            "min_width_m": profile.min_width_m,
            "max_width_m": profile.max_width_m,
            "width_filter_alpha": profile.width_filter_alpha,
            "max_width_delta_per_update_m": profile.max_width_delta_per_update_m,
            "min_trustworthy_pairs": max(1, int(self.get_parameter("width_estimation.min_trustworthy_pairs").value)),
        }

    def _pairing_config_values(self) -> dict:
        return {
            "min_pair_width_m": float(self.get_parameter("pairing.min_pair_width_m").value),
            "max_pair_width_m": float(self.get_parameter("pairing.max_pair_width_m").value),
            "max_width_jump_m": float(self.get_parameter("pairing.max_width_jump_m").value),
            "min_pair_count": max(1, int(self.get_parameter("pairing.min_pair_count").value)),
            "pair_reassignment_margin": float(self.get_parameter("pairing.pair_reassignment_margin").value),
            "pair_inward_projection_tolerance_m": max(
                0.0,
                float(self.get_parameter("pairing.pair_inward_projection_tolerance_m").value),
            ),
            "pairing_tangent_neighbor_count": max(2, int(self.get_parameter("pairing.tangent_neighbor_count").value)),
            "enforce_opposite_color_pairing": bool(
                self.get_parameter("pairing.enforce_opposite_color_pairing").value
            ),
            "enforce_geometry_pairing_gate": bool(self.get_parameter("pairing.enforce_geometry_pairing_gate").value),
        }

    def _centerline_and_validation_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            **self._centerline_config_values(),
            "min_path_points": max(2, int(self.get_parameter("validation.min_path_points").value)),
            "min_forward_extent_m": float(self.get_parameter("validation.min_forward_extent_m").value),
            "jump_check_horizon_m": profile.jump_check_horizon_m,
            "max_near_field_lateral_jump_m": profile.max_near_field_lateral_jump_m,
            "max_near_field_lateral_jump_m_sparse_pairs": float(
                self.get_parameter("validation.max_near_field_lateral_jump_m_sparse_pairs").value
            ),
            "max_start_heading_error_rad": float(self.get_parameter("validation.max_start_heading_error_rad").value),
        }

    def _centerline_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            "path_resolution_m": profile.centerline_path_resolution_m,
            "max_path_length_m": profile.max_path_length_m,
            "smoothing_window": max(1, int(self.get_parameter("centerline.smoothing_window").value)),
            "max_heading_delta_rad": float(self.get_parameter("centerline.max_heading_delta_rad").value),
            "max_midpoint_segment_length_m": max(
                self.centerline_path_resolution_m,
                float(self.get_parameter("centerline.max_midpoint_segment_length_m").value),
            ),
            "midpoint_order_reference_handoff_m": max(
                self.centerline_path_resolution_m,
                float(self.get_parameter("centerline.midpoint_order_reference_handoff_m").value),
            ),
            "midpoint_order_history_size": max(2, int(self.get_parameter("centerline.midpoint_order_history_size").value)),
            "midpoint_order_backtrack_tolerance_m": max(
                0.0,
                float(self.get_parameter("centerline.midpoint_order_backtrack_tolerance_m").value),
            ),
        }

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
        del points_xy
        raw_centerline, candidate_source = self._select_candidate_centerline(
            result=result, support_chain=raw_midpoint_chain, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        centerline, publish_mode, hold_reason, status, pair_segments_for_viz, raw_midpoint_chain = (
            self._resolve_publishable_centerline(
                result=result, planning_frame=planning_frame, raw_centerline=raw_centerline,
                candidate_source=candidate_source, pair_segments_for_viz=pair_segments_for_viz,
                raw_midpoint_chain=raw_midpoint_chain, target_frame=target_frame,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw, now_sec=now_sec,
            )
        )
        controller_state = self._run_controller_and_state(
            centerline=centerline, publish_mode=publish_mode, status=status,
            target_frame=target_frame, vehicle_x=vehicle_x, vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw, now_sec=now_sec, result=result,
        )
        return self._assemble_pipeline_dict(
            centerline=centerline, raw_centerline=raw_centerline,
            raw_midpoint_chain=raw_midpoint_chain, pair_segments_for_viz=pair_segments_for_viz,
            candidate_source=candidate_source, publish_mode=publish_mode,
            hold_reason=hold_reason, status=status, **controller_state,
        )

    def _resolve_publishable_centerline(
        self,
        *,
        result,
        planning_frame,
        raw_centerline: np.ndarray,
        candidate_source: str,
        pair_segments_for_viz: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
    ) -> tuple:
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
        return centerline, publish_mode, hold_reason, status, pair_segments_for_viz, raw_midpoint_chain

    def _run_controller_and_state(
        self,
        *,
        centerline: np.ndarray,
        publish_mode: str,
        status: str,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
        result,
    ) -> dict:
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
        return {
            "operator_state": operator_state, "operator_reason": operator_reason,
            "hold_remaining_s": hold_remaining_s, "control_path": control_path,
            "control_target_frame": control_target_frame,
            "control_debug_metrics": control_debug_metrics, "cmd_speed": cmd_speed,
            "cmd_steering": cmd_steering, "lookahead": lookahead,
            "zero_cmd_sent_flag": zero_cmd_sent_flag, "controller_failed": controller_failed,
        }

    @staticmethod
    def _assemble_pipeline_dict(
        *,
        centerline, raw_centerline, raw_midpoint_chain, pair_segments_for_viz,
        candidate_source, publish_mode, hold_reason, status,
        operator_state, operator_reason, hold_remaining_s, control_path,
        control_target_frame, control_debug_metrics, cmd_speed, cmd_steering,
        lookahead, zero_cmd_sent_flag, controller_failed,
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
        self._log_publish_cycle_state(result=result, pipeline=p)
        planner_metrics = self._build_publish_diagnostics_call(
            result=result, pipeline=p, cone_counts=cone_counts,
            control_path_point_count=control_path_point_count,
        )
        self._publish_diagnostics(
            frame_id=target_frame,
            centerline_point_count=int(p["centerline"].shape[0]),
            selected_edge_count=int(result.selected_edges.shape[0]),
            status=p["status"], control_debug_metrics=p["control_debug_metrics"],
            planner_metrics=planner_metrics,
        )
        self._publish_cycle_outputs(target_frame, result, p, control_path_point_count)

    def _log_publish_cycle_state(self, *, result, pipeline: dict) -> None:
        self._log_operator_state_transition(
            operator_state=pipeline["operator_state"], operator_reason=pipeline["operator_reason"],
            hold_remaining_s=pipeline["hold_remaining_s"],
            selected_chain_length=int(result.selected_chain_length),
        )
        self._log_mode_summary(
            mode=self._active_planner_mode, result=result,
            operator_state=pipeline["operator_state"], operator_reason=pipeline["operator_reason"],
            hold_active=bool(self._active_held_path_flag),
        )

    def _publish_cycle_outputs(
        self,
        target_frame: str,
        result,
        pipeline: dict,
        control_path_point_count: int,
    ) -> None:
        p = pipeline
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

    def _build_publish_diagnostics_call(
        self,
        *,
        result,
        pipeline: dict,
        cone_counts: dict,
        control_path_point_count: int,
    ) -> dict:
        return self._build_planner_metrics(
            result=result, raw_centerline=pipeline["raw_centerline"], cone_counts=cone_counts,
            candidate_source=pipeline["candidate_source"], publish_mode=pipeline["publish_mode"],
            hold_reason=pipeline["hold_reason"], operator_state=pipeline["operator_state"],
            operator_reason=pipeline["operator_reason"], hold_remaining_s=pipeline["hold_remaining_s"],
            control_path_point_count=control_path_point_count,
            zero_cmd_sent_flag=pipeline["zero_cmd_sent_flag"],
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
        prior = self._build_midpoint_prior(remembered_pair_entries=remembered_pair_entries)
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
        self._apply_width_estimate_update(result)
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
        return self._finalize_centerline_and_cache(
            result=result, raw_centerline=raw_centerline, candidate_source=candidate_source,
            candidate_update_ok=candidate_update_ok, candidate_update_reason=candidate_update_reason,
            centerline=centerline, planning_frame=planning_frame,
            pair_segments_for_viz=pair_segments_for_viz, target_frame=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )

    def _finalize_centerline_and_cache(
        self,
        *,
        result,
        raw_centerline: np.ndarray,
        candidate_source: str,
        candidate_update_ok: bool,
        candidate_update_reason: str,
        centerline: np.ndarray,
        planning_frame,
        pair_segments_for_viz: np.ndarray,
        target_frame: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> tuple:
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
        centerline, publish_mode, hold_reason, _status = self._resolve_hold_publish_state(
            centerline=centerline, candidate_update_ok=candidate_update_ok,
            candidate_update_reason=candidate_update_reason, result=result,
            now_sec=now_sec, status=status,
        )
        pair_segments_for_viz, raw_midpoint_chain = self._apply_held_pair_geometry(
            publish_mode=publish_mode, centerline=centerline,
            pair_segments_for_viz=pair_segments_for_viz,
            raw_midpoint_chain=raw_midpoint_chain, now_sec=now_sec,
        )
        centerline = self._prepare_centerline_for_current_pose(
            centerline=centerline, frame_id=target_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        return centerline, publish_mode, hold_reason, pair_segments_for_viz, raw_midpoint_chain

    def _resolve_hold_publish_state(
        self,
        *,
        centerline: np.ndarray,
        candidate_update_ok: bool,
        candidate_update_reason: str,
        result,
        now_sec: float,
        status: str,
    ) -> tuple:
        if centerline.shape[0] > 0 and candidate_update_ok:
            return self._fresh_or_hysteresis_hold_state(
                centerline, result, candidate_update_reason, now_sec, status,
            )
        self._advance_hold_hysteresis(plan_ok=False)
        held_centerline = self._held_centerline(now_sec)
        if held_centerline is None:
            return centerline, "held", result.reject_reason, status
        hold_reason = result.reject_reason or candidate_update_reason or status
        return held_centerline, "held", hold_reason, f"{status}; holding previous valid centerline"

    def _fresh_or_hysteresis_hold_state(
        self,
        centerline: np.ndarray,
        result,
        candidate_update_reason: str,
        now_sec: float,
        status: str,
    ) -> tuple:
        if not self._advance_hold_hysteresis(plan_ok=True):
            return centerline, "fresh", result.reject_reason, status
        held_centerline = self._held_centerline(now_sec)
        if held_centerline is None:
            return centerline, "fresh", result.reject_reason, status
        hold_reason = result.reject_reason or candidate_update_reason or status
        status = (
            f"{status}; hysteresis holding previous valid centerline "
            f"({self._hold_clean_frame_count}/{self.hold_exit_clean_frames})"
        )
        return held_centerline, "held", hold_reason, status

    def _apply_held_pair_geometry(
        self,
        *,
        publish_mode: str,
        centerline: np.ndarray,
        pair_segments_for_viz: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        now_sec: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if publish_mode != "held" or centerline.shape[0] == 0:
            return pair_segments_for_viz, raw_midpoint_chain
        held_pair_segments, held_raw_midpoint_chain = self._held_pair_geometry(now_sec=now_sec)
        if pair_segments_for_viz.size == 0 and held_pair_segments is not None:
            pair_segments_for_viz = held_pair_segments
        if raw_midpoint_chain.size == 0 and held_raw_midpoint_chain is not None:
            raw_midpoint_chain = held_raw_midpoint_chain
        return pair_segments_for_viz, raw_midpoint_chain

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
        control_debug_metrics: Optional[dict] = None
        zero_cmd_sent_flag = 0
        controller_failed = False
        if control_path.shape[0] >= 1 and self._controller is not None:
            return self._execute_controller_compute(
                control_path=control_path, target_frame=target_frame,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
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

    def _execute_controller_compute(
        self,
        *,
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
            failed, zero_flag, cmd_speed, cmd_steering = self._handle_controller_error(exc)
            return cmd_speed, cmd_steering, 0.0, None, None, zero_flag, failed
        cmd_speed, cmd_steering, lookahead, control_target_frame, control_debug_metrics = (
            self._apply_successful_controller_output(
                controller_output=controller_output, target_frame=target_frame,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            )
        )
        return cmd_speed, cmd_steering, lookahead, control_target_frame, control_debug_metrics, 0, False

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
        control_debug_metrics: Optional[dict] = None
        if controller_output.stanley_debug is not None:
            control_debug_metrics = self._build_stanley_debug_metrics(
                debug=controller_output.stanley_debug,
                control_target_frame=control_target_frame,
            )
        return control_target_frame, control_debug_metrics

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
        return self._apply_operator_state_overrides(
            operator_state=operator_state, operator_reason=operator_reason,
            centerline=centerline, publish_mode=publish_mode,
            zero_cmd_sent_flag=zero_cmd_sent_flag, controller_failed=controller_failed,
            control_path_point_count=control_path_point_count,
        )

    def _apply_operator_state_overrides(
        self,
        *,
        operator_state: str,
        operator_reason: str,
        centerline: np.ndarray,
        publish_mode: str,
        zero_cmd_sent_flag: int,
        controller_failed: bool,
        control_path_point_count: int,
    ) -> tuple[str, str]:
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
    ) -> dict:
        return {
            **self._build_result_quality_metrics(result=result),
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

    def _build_result_quality_metrics(self, *, result) -> dict:
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
    def _tentative_cone_is_usable_for_planning(*, raw_color: str, boundary_color: str) -> bool:
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
    ) -> list[_MidpointPairMemoryEntry]:
        keep: list[_MidpointPairMemoryEntry] = []
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
        entries: list[_MidpointPairMemoryEntry],
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
        entries: list[_MidpointPairMemoryEntry],
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[_MidpointPairMemoryEntry]:
        if len(entries) <= 1:
            return list(entries)
        local = self._compute_local_midpoints(
            entries=entries, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        ordered = self._greedy_direction_chain(local, self._pick_chain_start(local))
        return [entries[i] for i in ordered]

    @staticmethod
    def _compute_local_midpoints(
        *,
        entries: list[_MidpointPairMemoryEntry],
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[tuple[float, float]]:
        return [
            _odom_point_to_base(e.midpoint_x_odom, e.midpoint_y_odom, vehicle_x, vehicle_y, vehicle_yaw)
            for e in entries
        ]

    @staticmethod
    def _pick_chain_start(local: list[tuple[float, float]]) -> int:
        def start_key(i: int) -> tuple[float, float, float, int]:
            x, y = local[i]
            return (0.0 if x >= 0.0 else 1.0, math.hypot(x, y), abs(y), i)
        return min(range(len(local)), key=start_key)

    @staticmethod
    def _greedy_direction_chain(local: list[tuple[float, float]], start: int) -> list[int]:
        remaining = list(range(len(local)))
        ordered = [start]
        remaining.remove(start)
        ref = MidpointPlannerNode._initial_sort_reference(local[start])
        while remaining:
            curr = np.array(local[ordered[-1]], dtype=np.float64)
            ref = MidpointPlannerNode._update_direction_reference(local, ordered, curr, ref)
            best_i = MidpointPlannerNode._best_direction_candidate(local, remaining, curr, ref)
            if best_i is None:
                break
            ordered.append(best_i)
            remaining.remove(best_i)
        ordered.extend(remaining)
        return ordered

    @staticmethod
    def _initial_sort_reference(start_point: tuple[float, float]) -> np.ndarray:
        norm = math.hypot(start_point[0], start_point[1])
        if norm > _PAIR_SORT_MIN_NORM:
            return np.array(start_point, dtype=np.float64) / norm
        return np.array(_PAIR_SORT_INITIAL_REFERENCE, copy=True)

    @staticmethod
    def _update_direction_reference(
        local: list[tuple[float, float]],
        ordered: list[int],
        current: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        if len(ordered) < _PAIR_SEGMENT_ENDPOINT_COUNT:
            return reference
        history_idx = max(0, len(ordered) - _PAIR_SORT_HISTORY_SIZE)
        delta = current - np.array(local[ordered[history_idx]], dtype=np.float64)
        norm = float(np.hypot(delta[0], delta[1]))
        return delta / norm if norm > _PAIR_SORT_MIN_NORM else reference

    @staticmethod
    def _best_direction_candidate(
        local: list[tuple[float, float]],
        remaining: list[int],
        current: np.ndarray,
        reference: np.ndarray,
    ) -> Optional[int]:
        best_i: Optional[int] = None
        best_cost = float("inf")
        for i in remaining:
            cost = MidpointPlannerNode._direction_candidate_cost(local[i], current, reference)
            if cost is not None and cost < best_cost:
                best_cost = cost
                best_i = i
        return best_i

    @staticmethod
    def _direction_candidate_cost(
        point: tuple[float, float],
        current: np.ndarray,
        reference: np.ndarray,
    ) -> Optional[float]:
        delta = np.array(point, dtype=np.float64) - current
        dist = float(np.hypot(delta[0], delta[1]))
        if dist < _PAIR_SORT_MIN_NORM:
            return None
        forward = float(np.dot(delta, reference))
        if forward < -_PAIR_SORT_BACKTRACK_TOLERANCE_M:
            return None
        return dist + _PAIR_SORT_BACKTRACK_COST_WEIGHT * max(0.0, -forward)

    def _pair_entries_from_segments(
        self,
        *,
        pair_track_ids: np.ndarray,
        pair_segments: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[_MidpointPairMemoryEntry]:
        if pair_track_ids.size == 0 or pair_segments.size == 0:
            return []
        entries: list[_MidpointPairMemoryEntry] = []
        for pair_ids, pair_segment in zip(
            np.asarray(pair_track_ids, dtype=np.int64),
            np.asarray(pair_segments, dtype=np.float64),
        ):
            entry = self._transform_pair_segment_to_odom_with_ids(
                pair_ids=pair_ids, pair_segment=pair_segment, frame_id=frame_id,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            )
            if entry is None:
                continue
            entries.append(entry)
        return entries

    def _transform_pair_segment_to_odom_with_ids(
        self,
        *,
        pair_ids: np.ndarray,
        pair_segment: np.ndarray,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> Optional[_MidpointPairMemoryEntry]:
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
        return _MidpointPairMemoryEntry(
            left_track_id=int(pair_ids[0]), right_track_id=int(pair_ids[1]),
            midpoint_x_odom=float(middle[0]), midpoint_y_odom=float(middle[1]),
            left_x_odom=float(left[0]), left_y_odom=float(left[1]),
            right_x_odom=float(right[0]), right_y_odom=float(right[1]),
        )

    @staticmethod
    def _merge_pair_entries(
        *,
        remembered_entries: list[_MidpointPairMemoryEntry],
        live_entries: list[_MidpointPairMemoryEntry],
    ) -> list[_MidpointPairMemoryEntry]:
        merged: dict[tuple[int, int], _MidpointPairMemoryEntry] = {}
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
        del planner_confidences
        remembered_pairs = self._filter_eligible_remembered_pairs(
            result=result, frame_id=frame_id, vehicle_x=vehicle_x,
            vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            track_state_by_id={
                int(track_id): int(track_state)
                for track_id, track_state in zip(track_ids, track_states)
            },
        )
        self._pair_memory = self._merge_new_pairs_into_memory(remembered_pairs)

    def _filter_eligible_remembered_pairs(
        self,
        *,
        result: MidpointPlannerResult,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        track_state_by_id: dict[int, int],
    ) -> list[_MidpointPairMemoryEntry]:
        remembered_pairs: list[_MidpointPairMemoryEntry] = []
        for pair_ids, pair_segment in zip(
            np.asarray(result.selected_pair_track_ids, dtype=np.int64),
            np.asarray(result.pair_segments, dtype=np.float64),
        ):
            left_track_id = int(pair_ids[0])
            right_track_id = int(pair_ids[1])
            left_state = track_state_by_id.get(left_track_id, MSG_TRACK_STATE_TENTATIVE)
            right_state = track_state_by_id.get(right_track_id, MSG_TRACK_STATE_TENTATIVE)
            if left_state == MSG_TRACK_STATE_TENTATIVE or right_state == MSG_TRACK_STATE_TENTATIVE:
                continue
            entry = self._transform_pair_segment_to_odom_with_ids(
                pair_ids=pair_ids, pair_segment=pair_segment, frame_id=frame_id,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            )
            if entry is None:
                continue
            remembered_pairs.append(entry)
        return remembered_pairs

    def _merge_new_pairs_into_memory(
        self,
        remembered_pairs: list[_MidpointPairMemoryEntry],
    ) -> list[_MidpointPairMemoryEntry]:
        merged: dict[tuple[int, int], _MidpointPairMemoryEntry] = {}
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
        return list(merged.values())

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
        if result.status == "ok" and result.centerline.shape[0] > 0:
            candidates.extend(self._candidate_entry(
                result.centerline, "validated", _VALIDATED_CANDIDATE_PRIORITY,
                frame_id, vehicle_x, vehicle_y, vehicle_yaw,
            ))
        candidates.extend(self._candidate_entry(
            support_chain, _MIDPOINT_CHAIN_SOURCE, _SUPPORT_CHAIN_CANDIDATE_PRIORITY,
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
        candidate = self._finite_midpoint_path(path)
        if candidate.shape[0] < _MIDPOINT_CHAIN_MIN_POINTS:
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
        del (
            candidate_diagonal_count, selected_chain_length, centerline_point_count,
            cmd_speed, cmd_steering, lookahead, seed_midpoint_distance_m,
            near_field_lateral_max_m, near_field_midpoint_kink_max_rad, hold_remaining_s,
        )
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
        reject_parts_text = self._build_reject_parts_text()
        if reject_parts_text:
            lines.append(reject_parts_text)
        lines.append(self._lap_status_text())
        return "\n".join(lines)

    def _build_reject_parts_text(self) -> str:
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
        return "REJECTS: " + " | ".join(reject_parts) if reject_parts else ""

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


# =============================================================================
# CorridorPlannerNode
# =============================================================================


class CorridorPlannerNode(GenericTrackedConePlannerNode):
    """Tracked-cone corridor planner with shared path-memory stabilization."""

    def __init__(self) -> None:
        super().__init__(PlannerIdentity(
            node_name="corridor_planner_node",
            planner_mode="corridor",
            diagnostics_prefix="corridor_planner",
            diagnostics_topic="/corridor_planner/diagnostics",
        ))

    def _planner_label(self) -> str:
        return 'corridor planner'

    def _declare_parameter_overrides(self) -> dict:
        return {"defaults_override": {"boundary_chain.max_heading_change_rad": 2.35}}

    def _declare_algorithm_parameters(self) -> None:
        defaults = {
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
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_algorithm_parameters(self) -> None:
        self.pair_memory_retention_s = max(
            self.midline_hold_last_valid_duration_s,
            float(self.get_parameter("midline_memory.pair_memory_retention_s").value),
        )

    def _init_algorithm_state(self) -> None:
        self._pair_memory: list[_CorridorPairMemoryEntry] = []

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
                2, int(self.get_parameter("corridor.min_required_corridor_samples").value),
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
            result, midline, ctrl, input_metrics, operator_state, operator_reason, hold_remaining_s,
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
    ) -> _CorridorInputMetrics:
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
        return _CorridorInputMetrics(
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
        self._apply_width_estimate_update(result)
        return result

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
        input_metrics: _CorridorInputMetrics,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
    ) -> dict:
        corridor_analysis_metrics = getattr(self, "_last_corridor_analysis_metrics", {})
        return {
            **self._build_corridor_result_metrics(result, input_metrics, midline),
            **self._build_operator_control_metrics(
                result, midline, ctrl, operator_state, operator_reason, hold_remaining_s,
            ),
            **corridor_analysis_metrics,
        }

    def _build_corridor_result_metrics(
        self, result, input_metrics: _CorridorInputMetrics, midline: _MidlineUpdateResult,
    ) -> dict:
        return {
            **self._build_corridor_chain_metrics(result, input_metrics),
            **self._build_midline_and_reject_metrics(result, midline),
        }

    def _build_corridor_chain_metrics(self, result, input_metrics: _CorridorInputMetrics) -> dict:
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

    # ------------------------------------------------------------------
    # Pair memory methods
    # ------------------------------------------------------------------

    def _active_pair_memory_entries(
        self,
        *,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[_CorridorPairMemoryEntry]:
        keep: list[_CorridorPairMemoryEntry] = []
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        for entry in self._pair_memory:
            midpoint_x_base, midpoint_y_base = _odom_point_to_base(
                entry.midpoint_x_odom, entry.midpoint_y_odom, vehicle_x, vehicle_y, vehicle_yaw,
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
    ) -> list[_CorridorPairMemoryEntry]:
        if pair_segments.size == 0:
            return []
        entries: list[_CorridorPairMemoryEntry] = []
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
    ) -> Optional[_CorridorPairMemoryEntry]:
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
        return _CorridorPairMemoryEntry(
            left_x_odom=float(left[0]), left_y_odom=float(left[1]),
            right_x_odom=float(right[0]), right_y_odom=float(right[1]),
            midpoint_x_odom=float(middle[0]), midpoint_y_odom=float(middle[1]),
            last_valid_sec=float(now_sec),
        )

    @staticmethod
    def _merge_pair_entries(
        *,
        remembered_entries: list[_CorridorPairMemoryEntry],
        live_entries: list[_CorridorPairMemoryEntry],
    ) -> list[_CorridorPairMemoryEntry]:
        merged: list[_CorridorPairMemoryEntry] = []
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
        entries: list[_CorridorPairMemoryEntry],
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> list[_CorridorPairMemoryEntry]:
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
        entries: list[_CorridorPairMemoryEntry],
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
        entries: list[_CorridorPairMemoryEntry],
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
            frame_id=frame_id, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            now_sec=now_sec,
        )
        if not live_entries:
            return
        remembered_entries = self._active_pair_memory_entries(
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
        )
        merged = self._merge_pair_entries(
            remembered_entries=remembered_entries, live_entries=live_entries,
        )
        self._pair_memory = self._sort_pair_entries_by_forward_progress(
            entries=merged, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
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
            frame_id=frame_id, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            now_sec=now_sec, support_centerline=support_centerline, direct_commit=direct_commit,
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
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
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
        return abs(float(math.atan2(
            math.sin(candidate_heading - stored_heading),
            math.cos(candidate_heading - stored_heading),
        )))

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

    # ------------------------------------------------------------------
    # Corridor analysis diagnostics
    # ------------------------------------------------------------------

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
            "corridor_raw_anchor": self._corridor_analysis_frame_path(
                raw_anchor_path, frame_id, vehicle_x, vehicle_y, vehicle_yaw,
            ),
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
                centerline=working, frame_id=frame_id,
                vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
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
            0.0, total + 1e-9, float(_CORRIDOR_ANALYSIS_SAMPLE_SPACING_M), dtype=np.float64,
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
        super()._publish_empty_cycle(
            frame_id=frame_id, status=status, operator_state=operator_state,
            operator_reason=operator_reason, cmd_speed=cmd_speed,
            cmd_steering=cmd_steering, lookahead=lookahead,
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
        del candidate_diagonal_count, selected_chain_length, centerline_point_count
        del cmd_speed, cmd_steering, lookahead, seed_midpoint_distance_m
        del near_field_lateral_max_m, near_field_midpoint_kink_max_rad, hold_remaining_s
        return "\n".join([
            f"STATE: {operator_state.upper()}",
            f"MODE: {self._active_planner_mode.upper()}",
            f"REASON: {self._operator_reason_label(operator_reason)}",
            self._lap_status_text(),
        ])


# =============================================================================
# SingleBoundaryPlannerNode
# =============================================================================


class SingleBoundaryPlannerNode(GenericTrackedConePlannerNode):
    """Tracked-cone single-boundary planner with shared path-memory stabilization."""

    _centerline_marker_width_m: float = 0.09

    def __init__(self) -> None:
        super().__init__(PlannerIdentity(
            node_name="single_boundary_planner_node",
            planner_mode="single_boundary",
            diagnostics_prefix="single_boundary_planner",
            diagnostics_topic="/single_boundary_planner/diagnostics",
        ))

    def _planner_label(self) -> str:
        return 'single-boundary planner'

    def _declare_algorithm_parameters(self) -> None:
        defaults = {
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
            "validation.min_path_points": 2,
            "validation.min_forward_extent_m": 1.0,
            "validation.max_near_field_lateral_jump_m_sparse_pairs": 0.9,
            "validation.max_near_field_lateral_jump_m_single_boundary": 5.0,
            "validation.max_start_heading_error_rad": 1.0,
            "debug.show_raw_offset_path": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_algorithm_parameters(self) -> None:
        self.show_raw_offset_path = bool(self.get_parameter("debug.show_raw_offset_path").value)

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
        profile = self._planner_algorithm_profile
        return {
            "max_cone_range_m": profile.max_cone_range_m,
            "behind_drop_m": profile.behind_drop_m,
            "min_confidence": profile.min_confidence,
            "min_required_cones": max(2, profile.min_required_cones),
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
                0, int(self.get_parameter("filtering.max_consecutive_unknown_pairs").value),
            ),
        }

    def _boundary_chain_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            "min_step_m": profile.boundary_min_step_m,
            "max_step_m": profile.boundary_max_step_m,
            "max_heading_change_rad": profile.boundary_max_heading_change_rad,
            "min_forward_progress_m": profile.boundary_min_forward_progress_m,
            "min_chain_length": max(2, int(self.get_parameter("boundary_chain.min_chain_length").value)),
        }

    def _pairing_config_values(self) -> dict:
        return {
            "min_pair_width_m": float(self.get_parameter("pairing.min_pair_width_m").value),
            "max_pair_width_m": float(self.get_parameter("pairing.max_pair_width_m").value),
            "max_width_jump_m": float(self.get_parameter("pairing.max_width_jump_m").value),
            "min_pair_count": max(1, int(self.get_parameter("pairing.min_pair_count").value)),
            "pair_reassignment_margin": float(self.get_parameter("pairing.pair_reassignment_margin").value),
        }

    def _width_estimation_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            "initial_width_m": profile.initial_width_m,
            "min_width_m": profile.min_width_m,
            "max_width_m": profile.max_width_m,
            "width_filter_alpha": profile.width_filter_alpha,
            "max_width_delta_per_update_m": profile.max_width_delta_per_update_m,
            "min_trustworthy_pairs": max(
                1, int(self.get_parameter("width_estimation.min_trustworthy_pairs").value)
            ),
        }

    def _centerline_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            "path_resolution_m": profile.centerline_path_resolution_m,
            "max_path_length_m": profile.max_path_length_m,
            "smoothing_window": max(1, int(self.get_parameter("centerline.smoothing_window").value)),
            "max_heading_delta_rad": float(self.get_parameter("centerline.max_heading_delta_rad").value),
        }

    def _validation_config_values(self) -> dict:
        profile = self._planner_algorithm_profile
        return {
            "min_path_points": max(2, int(self.get_parameter("validation.min_path_points").value)),
            "min_forward_extent_m": float(self.get_parameter("validation.min_forward_extent_m").value),
            "jump_check_horizon_m": profile.jump_check_horizon_m,
            "max_near_field_lateral_jump_m": profile.max_near_field_lateral_jump_m,
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

    # ------------------------------------------------------------------
    # _on_timer stage helpers
    # ------------------------------------------------------------------

    def _collect_input_metrics(
        self,
        cones_msg,
        points_xy: np.ndarray,
        colors: list,
        confidences: np.ndarray,
    ) -> _SingleBoundaryInputMetrics:
        planning_frame = self._tracked_cone_planning_frame(
            msg=cones_msg, points_xy=points_xy, colors=colors, confidences=confidences,
        )
        self._active_remembered_cone_count = int(len(cones_msg.cones))
        self._active_stale_cone_count = int(
            np.count_nonzero(planning_frame.track_states == MSG_TRACK_STATE_STALE)
        )
        return _SingleBoundaryInputMetrics(planning_frame=planning_frame)

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
        self._apply_width_estimate_update(result)
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
                result, raw_centerline, candidate_source, result.raw_offset_path, target_frame,
                vehicle_x, vehicle_y, vehicle_yaw, now_sec,
            )
        )
        status, pair_segments_for_viz, _raw_midpoint_chain = self._build_midline_status(
            result, raw_centerline, centerline, candidate_source, candidate_update_ok,
            candidate_update_reason, pair_segments_for_viz, None, None, None, None, None, now_sec,
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

    def _remember_valid_pair_geometry(self, result, pair_segments_for_viz: np.ndarray) -> None:
        self._last_valid_pair_segments = (
            pair_segments_for_viz if pair_segments_for_viz.size > 0 else self._last_valid_pair_segments
        )
        self._last_valid_pair_track_ids = (
            np.array(result.selected_pair_track_ids, copy=True)
            if result.selected_pair_track_ids.size > 0
            else self._last_valid_pair_track_ids
        )

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
            frame_id=frame_id, vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
            now_sec=now_sec, support_centerline=support_centerline, direct_commit=direct_commit,
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
            centerline=candidate_centerline, frame_id=self.odom_frame,
            vehicle_x=vehicle_x, vehicle_y=vehicle_y, vehicle_yaw=vehicle_yaw,
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
        return abs(float(math.atan2(
            math.sin(candidate_heading - stored_heading),
            math.cos(candidate_heading - stored_heading),
        )))

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
            frame_id=frame_id, status=status, operator_state=operator_state,
            operator_reason=operator_reason, cmd_speed=cmd_speed,
            cmd_steering=cmd_steering, lookahead=lookahead,
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
        del candidate_diagonal_count, selected_chain_length, centerline_point_count
        del cmd_speed, cmd_steering, lookahead, seed_midpoint_distance_m
        del near_field_lateral_max_m, near_field_midpoint_kink_max_rad, hold_remaining_s
        return "\n".join([
            f"STATE: {operator_state.upper()}",
            f"MODE: {self._active_planner_mode.upper()}",
            f"REASON: {self._operator_reason_label(operator_reason)}",
            self._lap_status_text(),
        ])


# =============================================================================
# Entry points
# =============================================================================


def main_midpoint(args=None) -> None:
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


def main_corridor(args=None) -> None:
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


def main_single_boundary(args=None) -> None:
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
