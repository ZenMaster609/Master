#!/usr/bin/env python3
"""Single-boundary planner over tracked cone detections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import rclpy
from eufs_msgs.msg import ConeArrayWithCovariance
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray

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
from sim_car.planning.single_boundary_planner_core import (
    SingleBoundaryPlannerConfig,
    SingleBoundaryPlannerPrior,
    SingleBoundaryPlannerResult,
    _finalize_path,
    compute_single_boundary_centerline,
    update_track_width_estimate,
)
from sim_car.planning.tracked_cone_planner_base import TrackedConePlannerBase

MSG_TRACK_STATE_TENTATIVE = int(getattr(ConeDetection, "TRACK_STATE_TENTATIVE", 0))
MSG_TRACK_STATE_CONFIRMED = int(getattr(ConeDetection, "TRACK_STATE_CONFIRMED", 1))
MSG_TRACK_STATE_STALE = int(getattr(ConeDetection, "TRACK_STATE_STALE", 2))
IDENTITY_WORLD_FRAMES = frozenset({"map", "odom"})
_VALIDATED_JUMP_ACCEPT_HORIZON_M = 3.0
_VALIDATED_JUMP_ACCEPT_LATERAL_MAX_M = 0.45
_VALIDATED_JUMP_ACCEPT_LATERAL_MEAN_M = 0.25
_VALIDATED_JUMP_ACCEPT_HEADING_DELTA_RAD = 0.30


@dataclass
class _PairMemoryEntry:
    left_track_id: int
    right_track_id: int
    last_valid_sec: float


@dataclass(frozen=True)
class _LapGateInfo:
    frame_id: str
    segment_xy: np.ndarray


@dataclass(frozen=True)
class _LapCounterSnapshot:
    completed_laps: int
    gate_crossings: int
    just_completed_lap: bool


class _LapGateCounter:
    def __init__(
        self,
        gate_segment_xy: np.ndarray,
        *,
        min_lap_travel_m: float,
        min_lap_time_sec: float = 5.0,
        near_gate_distance_m: float = 6.0,
    ) -> None:
        gate = _as_xy(gate_segment_xy)
        if gate.shape != (2, 2):
            raise ValueError("gate_segment_xy must contain exactly two XY points")
        if float(np.hypot(*(gate[1] - gate[0]))) <= 1e-6:
            raise ValueError("gate_segment_xy must span a non-zero line segment")
        self._gate_segment_xy = np.asarray(gate, dtype=np.float64)
        self._min_lap_travel_m = max(1.0, float(min_lap_travel_m))
        self._min_lap_time_sec = max(0.0, float(min_lap_time_sec))
        self._near_gate_distance_m = max(0.5, float(near_gate_distance_m))
        self._prev_point_xy: Optional[np.ndarray] = None
        self._last_crossing_time_sec = float("nan")
        self._distance_since_last_crossing_m = 0.0
        self._completed_laps = 0
        self._gate_crossings = 0
        self._ignore_first_crossing = False

    def update(self, point_xy: np.ndarray, timestamp_sec: float) -> _LapCounterSnapshot:
        point = np.asarray(point_xy, dtype=np.float64).reshape(2,)
        now_sec = float(timestamp_sec)
        if not np.all(np.isfinite(point)):
            return _LapCounterSnapshot(
                completed_laps=int(self._completed_laps),
                gate_crossings=int(self._gate_crossings),
                just_completed_lap=False,
            )

        if self._prev_point_xy is None:
            self._prev_point_xy = point
            self._ignore_first_crossing = (
                _point_to_segment_distance_m(point, self._gate_segment_xy) <= self._near_gate_distance_m
            )
            return _LapCounterSnapshot(
                completed_laps=int(self._completed_laps),
                gate_crossings=int(self._gate_crossings),
                just_completed_lap=False,
            )

        step_m = float(np.hypot(*(point - self._prev_point_xy)))
        if math.isfinite(step_m):
            self._distance_since_last_crossing_m += step_m

        just_completed_lap = False
        if _segments_intersect_2d(
            self._prev_point_xy,
            point,
            self._gate_segment_xy[0],
            self._gate_segment_xy[1],
        ):
            should_ignore = self._gate_crossings == 0 and self._ignore_first_crossing
            enough_distance = self._distance_since_last_crossing_m >= self._min_lap_travel_m
            enough_time = (
                not math.isfinite(self._last_crossing_time_sec)
                or not math.isfinite(now_sec)
                or (now_sec - self._last_crossing_time_sec) >= self._min_lap_time_sec
            )
            if not should_ignore and enough_distance and enough_time:
                self._completed_laps += 1
                just_completed_lap = True
            self._gate_crossings += 1
            self._distance_since_last_crossing_m = 0.0
            self._last_crossing_time_sec = now_sec

        self._prev_point_xy = point
        return _LapCounterSnapshot(
            completed_laps=int(self._completed_laps),
            gate_crossings=int(self._gate_crossings),
            just_completed_lap=bool(just_completed_lap),
        )


def _as_xy(points) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.reshape(arr, (-1, 2))
    return arr[np.all(np.isfinite(arr), axis=1)]


def _point_to_segment_distance_m(point_xy: np.ndarray, segment_xy: np.ndarray) -> float:
    segment_xy = _as_xy(segment_xy)
    point_xy = np.asarray(point_xy, dtype=np.float64).reshape(2,)
    if segment_xy.shape[0] == 0:
        return float("nan")
    if segment_xy.shape[0] == 1:
        return float(np.hypot(*(point_xy - segment_xy[0])))
    start = segment_xy[0]
    end = segment_xy[1]
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom <= 1e-12:
        nearest = start
    else:
        t = float(np.clip(np.dot(point_xy - start, delta) / denom, 0.0, 1.0))
        nearest = start + (t * delta)
    return float(np.hypot(*(point_xy - nearest)))


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] * b[1]) - (a[1] * b[0]))


def _orientation_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return _cross_2d(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64), np.asarray(c, dtype=np.float64) - np.asarray(a, dtype=np.float64))


def _point_on_segment_2d(point_xy: np.ndarray, seg_start_xy: np.ndarray, seg_end_xy: np.ndarray) -> bool:
    point = np.asarray(point_xy, dtype=np.float64)
    start = np.asarray(seg_start_xy, dtype=np.float64)
    end = np.asarray(seg_end_xy, dtype=np.float64)
    eps = 1e-9
    return (
        min(start[0], end[0]) - eps <= point[0] <= max(start[0], end[0]) + eps
        and min(start[1], end[1]) - eps <= point[1] <= max(start[1], end[1]) + eps
    )


def _should_assume_identity_transform(source_frame: str, target_frame: str) -> bool:
    source = str(source_frame).strip().lower()
    target = str(target_frame).strip().lower()
    if not source or not target or source == target:
        return False
    return source in IDENTITY_WORLD_FRAMES and target in IDENTITY_WORLD_FRAMES


def _segments_intersect_2d(
    a_start_xy: np.ndarray,
    a_end_xy: np.ndarray,
    b_start_xy: np.ndarray,
    b_end_xy: np.ndarray,
) -> bool:
    a0 = np.asarray(a_start_xy, dtype=np.float64)
    a1 = np.asarray(a_end_xy, dtype=np.float64)
    b0 = np.asarray(b_start_xy, dtype=np.float64)
    b1 = np.asarray(b_end_xy, dtype=np.float64)
    eps = 1e-9
    o1 = _orientation_2d(a0, a1, b0)
    o2 = _orientation_2d(a0, a1, b1)
    o3 = _orientation_2d(b0, b1, a0)
    o4 = _orientation_2d(b0, b1, a1)
    if (o1 * o2) < -eps and (o3 * o4) < -eps:
        return True
    if abs(o1) <= eps and _point_on_segment_2d(b0, a0, a1):
        return True
    if abs(o2) <= eps and _point_on_segment_2d(b1, a0, a1):
        return True
    if abs(o3) <= eps and _point_on_segment_2d(a0, b0, b1):
        return True
    if abs(o4) <= eps and _point_on_segment_2d(a1, b0, b1):
        return True
    return False


def _build_smalltrack_lap_gate(big_orange_xy: np.ndarray, frame_id: str) -> Optional[_LapGateInfo]:
    points_xy = _as_xy(big_orange_xy)
    if points_xy.shape[0] < 4:
        return None
    candidates: list[tuple[float, int, int]] = []
    for idx_a in range(points_xy.shape[0]):
        for idx_b in range(idx_a + 1, points_xy.shape[0]):
            candidates.append((float(np.hypot(*(points_xy[idx_b] - points_xy[idx_a]))), idx_a, idx_b))
    candidates.sort(key=lambda item: item[0])
    used: set[int] = set()
    pair_midpoints: list[np.ndarray] = []
    for _, idx_a, idx_b in candidates:
        if idx_a in used or idx_b in used:
            continue
        used.add(idx_a)
        used.add(idx_b)
        pair_midpoints.append(0.5 * (points_xy[idx_a] + points_xy[idx_b]))
        if len(pair_midpoints) == 2:
            break
    gate_segment_xy = _as_xy(pair_midpoints)
    if gate_segment_xy.shape != (2, 2):
        return None
    if float(np.hypot(*(gate_segment_xy[1] - gate_segment_xy[0]))) <= 0.25:
        return None
    order = np.lexsort((gate_segment_xy[:, 0], gate_segment_xy[:, 1]))
    return _LapGateInfo(
        frame_id=str(frame_id).strip() or "map",
        segment_xy=np.asarray(gate_segment_xy[order], dtype=np.float64),
    )


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
        self._lap_tracking_gate = None
        self._lap_tracking_counter = None
        self._lap_tracking_completed_laps = 0

        self._init_common_ros_interfaces()
        if self.lap_tracking_gt_track_topic:
            self.create_subscription(
                ConeArrayWithCovariance,
                self.lap_tracking_gt_track_topic,
                self._lap_tracking_gt_cb,
                10,
            )

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
            "lap_tracking.gt_track_topic": "/ground_truth/track",
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
        self.lap_tracking_gt_track_topic = str(
            self.get_parameter("lap_tracking.gt_track_topic").value
        ).strip() or "/ground_truth/track"
        self.lap_tracking_target_laps = max(
            0,
            int(self.get_parameter("lap_tracking.target_laps").value),
        )
        self.show_raw_offset_path = bool(
            self.get_parameter("debug.show_raw_offset_path").value
        )

        self._core_config = SingleBoundaryPlannerConfig(
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
            max_near_field_lateral_jump_m_single_boundary=float(
                self.get_parameter("validation.max_near_field_lateral_jump_m_single_boundary").value
            ),
            max_start_heading_error_rad=float(
                self.get_parameter("validation.max_start_heading_error_rad").value
            ),
        )
        self._filtered_track_width_m = float(self._core_config.initial_width_m)

    def _lap_tracking_gt_cb(self, msg: ConeArrayWithCovariance) -> None:
        big_orange_xy = [
            (float(cone.point.x), float(cone.point.y))
            for cone in msg.big_orange_cones
        ]
        gate = _build_smalltrack_lap_gate(
            np.asarray(big_orange_xy, dtype=np.float64),
            str(msg.header.frame_id).strip() or "map",
        )
        if gate is None:
            return
        self._lap_tracking_gate = gate
        if self._lap_tracking_counter is None:
            gate_length_m = float(np.hypot(*(gate.segment_xy[1] - gate.segment_xy[0])))
            self._lap_tracking_counter = _LapGateCounter(
                gate.segment_xy,
                min_lap_travel_m=20.0,
                min_lap_time_sec=5.0,
                near_gate_distance_m=max(4.0, 2.0 * gate_length_m),
            )

    def _odom_cb(self, msg: Odometry) -> None:
        super()._odom_cb(msg)
        self._update_lap_tracking(msg)

    def _update_lap_tracking(self, msg: Odometry) -> None:
        if self._lap_tracking_gate is None or self._lap_tracking_counter is None:
            return
        point_gate = self._transform_xy_to_frame(
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            source_frame=str(msg.header.frame_id).strip() or self.odom_frame,
            target_frame=self._lap_tracking_gate.frame_id,
            stamp=msg.header.stamp,
        )
        if point_gate is None:
            return
        timestamp_sec = float(msg.header.stamp.sec) + (float(msg.header.stamp.nanosec) * 1e-9)
        if timestamp_sec <= 0.0:
            timestamp_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        snapshot = self._lap_tracking_counter.update(point_gate, timestamp_sec)
        self._lap_tracking_completed_laps = int(snapshot.completed_laps)
        if snapshot.just_completed_lap:
            self.get_logger().info(
                f"smalltrack laps={snapshot.completed_laps}/{self.lap_tracking_target_laps or 0}"
            )

    def _transform_xy_to_frame(
        self,
        *,
        x: float,
        y: float,
        source_frame: str,
        target_frame: str,
        stamp,
    ) -> Optional[np.ndarray]:
        if not source_frame or not target_frame:
            return None
        if (
            source_frame == target_frame
            or self._is_alias(source_frame, target_frame)
            or _should_assume_identity_transform(source_frame, target_frame)
        ):
            return np.asarray([float(x), float(y)], dtype=np.float64)
        tf_msg, _, _ = self._lookup_transform_with_alias(target_frame, source_frame, stamp)
        if tf_msg is None:
            return None
        px, py, _ = self._transform_point(tf_msg, float(x), float(y), 0.0)
        return np.asarray([px, py], dtype=np.float64)

    def _lap_status_text(self) -> str:
        if self._lap_tracking_gate is None:
            return f"LAPS: {int(self._lap_tracking_completed_laps)}/off"
        if self.lap_tracking_target_laps > 0:
            return f"LAPS: {int(self._lap_tracking_completed_laps)}/{int(self.lap_tracking_target_laps)}"
        return f"LAPS: {int(self._lap_tracking_completed_laps)}/off"

    def _on_timer(self) -> None:
        ctx = self._resolve_cone_planning_context()
        if ctx is None:
            return
        cones_msg, target_frame, vehicle_x, vehicle_y, vehicle_yaw, points_xy, colors, confidences = ctx
        control_target_frame: Optional[np.ndarray] = None
        control_debug_metrics: Optional[dict[str, float]] = None
        cmd_speed = 0.0
        cmd_steering = 0.0
        lookahead = 0.0

        self._update_remembered_cone_viz(points_xy=points_xy, colors=colors)
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
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

        result = compute_single_boundary_centerline(
            points_xy=planning_frame.points_xy,
            colors=planning_frame.colors,
            confidences=planning_frame.planner_confidences,
            track_ids=planning_frame.track_ids,
            vehicle_xy=(vehicle_x, vehicle_y),
            vehicle_yaw=vehicle_yaw,
            config=self._core_config,
            prior=SingleBoundaryPlannerPrior(
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
                previous_pairs=[],
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
            support_centerline=result.raw_offset_path,
        )
        candidate_update_ok = bool(
            getattr(self, "_last_midline_candidate_update_ok", candidate_update_ok)
        )
        candidate_update_reason = str(
            getattr(self, "_last_midline_candidate_update_reason", candidate_update_reason)
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
            self._remember_pairs(result=result, now_sec=now_sec)
            self._last_valid_pair_segments = pair_segments_for_viz if pair_segments_for_viz.size > 0 else self._last_valid_pair_segments
            self._last_valid_pair_track_ids = (
                np.array(result.selected_pair_track_ids, copy=True)
                if result.selected_pair_track_ids.size > 0
                else self._last_valid_pair_track_ids
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
        stored_local = self._local_forward_prefix_samples(
            path_local=stored_local,
            horizon_m=horizon_m,
        )
        candidate_local = self._local_forward_prefix_samples(
            path_local=candidate_local,
            horizon_m=horizon_m,
        )
        count = min(stored_local.shape[0], candidate_local.shape[0])
        if count < 2:
            return empty

        stored_local = stored_local[:count]
        candidate_local = candidate_local[:count]
        delta = candidate_local - stored_local
        lateral = np.abs(delta[:, 1])
        displacement = np.hypot(delta[:, 0], delta[:, 1])
        candidate_heading = math.atan2(
            float(candidate_local[1, 1] - candidate_local[0, 1]),
            float(candidate_local[1, 0] - candidate_local[0, 0]),
        )
        stored_heading = math.atan2(
            float(stored_local[1, 1] - stored_local[0, 1]),
            float(stored_local[1, 0] - stored_local[0, 0]),
        )
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
            "lateral_max_m": float(np.max(lateral)) if lateral.size else 0.0,
            "lateral_mean_m": float(np.mean(lateral)) if lateral.size else 0.0,
            "displacement_max_m": float(np.max(displacement)) if displacement.size else 0.0,
            "displacement_mean_m": float(np.mean(displacement)) if displacement.size else 0.0,
            "heading_delta_rad": heading_delta,
        }

    def _local_forward_prefix_samples(
        self,
        *,
        path_local: np.ndarray,
        horizon_m: float,
    ) -> np.ndarray:
        pts = np.asarray(path_local, dtype=np.float64)
        if pts.shape[0] < 2:
            return np.empty((0, 2), dtype=np.float64)
        valid_mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & (pts[:, 0] >= -0.1)
        pts = pts[valid_mask]
        if pts.shape[0] < 2:
            return np.empty((0, 2), dtype=np.float64)
        cumulative = self._path_cumulative_lengths(pts)
        total = min(float(cumulative[-1]), max(0.25, float(horizon_m)))
        if total <= 1e-6:
            return np.asarray(pts[:1], dtype=np.float64)
        step = max(0.05, float(self.midline_station_spacing_m))
        samples = np.arange(0.0, total + 1e-9, step, dtype=np.float64)
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
