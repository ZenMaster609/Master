from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from builtin_interfaces.msg import Time as TimeMsg

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray  # noqa: E402
    from sim_car.planning.planner_runtime_types import PlannerIdentity  # noqa: E402
    from sim_car.planning.single_boundary_planner_core import SingleBoundaryPlannerResult  # noqa: E402
    from sim_car.planning.single_boundary_planner_node import MSG_TRACK_STATE_TENTATIVE  # noqa: E402
    from sim_car.planning.single_boundary_planner_node import SingleBoundaryPlannerNode  # noqa: E402
except ImportError as exc:  # pragma: no cover - depends on generated ROS interfaces
    pytest.skip(f"ROS planner node imports unavailable: {exc}", allow_module_level=True)


class _FakePublisher:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, msg: object) -> None:
        self.messages.append(msg)


class _FakeNow:
    def __init__(self, stamp: TimeMsg) -> None:
        self._stamp = stamp
        self.nanoseconds = (int(stamp.sec) * 1_000_000_000) + int(stamp.nanosec)

    def to_msg(self) -> TimeMsg:
        return self._stamp


class _FakeClock:
    def __init__(self, stamp: TimeMsg) -> None:
        self._stamp = stamp

    def now(self) -> _FakeNow:
        return _FakeNow(self._stamp)


def _make_node() -> SingleBoundaryPlannerNode:
    node = object.__new__(SingleBoundaryPlannerNode)
    node._planner_identity = PlannerIdentity(
        node_name="single_boundary_planner_node",
        planner_mode="single_boundary",
        diagnostics_prefix="single_boundary_planner",
        diagnostics_topic="/single_boundary_planner/diagnostics",
    )
    node._diag_pub = _FakePublisher()
    node.publish_control_debug = False
    node.publish_thesis_context = False
    node._hold_mode_active = False
    node._hold_clean_frame_count = 0
    node._active_planner_mode = "single_boundary"
    node._last_midline_update_mode = "hold"
    node.show_raw_cones = False
    node.show_boundary_chains = False
    node.show_pair_lines = False
    node.show_raw_midpoint_chain = False
    node.show_raw_offset_path = False
    node.show_raw_prevalidation_centerline = False
    node.show_lookahead_point = False
    node._current_pair_segments_for_viz = None
    node._last_viz_left_boundary = None
    node._last_viz_right_boundary = None
    node._last_viz_raw_offset_path = None
    node._midline_buffer_path = None
    node._midline_buffer_confidence = 0.0
    node._midline_buffer_last_update_sec = -1.0
    node.midline_hold_last_valid_duration_s = 1.25
    node.midline_min_buffer_confidence = 0.1
    node.midline_station_spacing_m = 0.5
    node.midline_horizon_m = 20.0
    node.midline_near_distance_m = 4.0
    node.midline_mid_distance_m = 12.0
    node.midline_control_handoff_distance_m = 1.5
    node.midline_near_alpha = 0.06
    node.midline_mid_alpha = 0.18
    node.midline_far_alpha = 0.35
    node.midline_near_max_shift_m = 0.10
    node.midline_mid_max_shift_m = 0.20
    node.midline_far_max_shift_m = 0.40
    node.centerline_jump_horizon_m = 8.0
    node.candidate_jump_reject_threshold_m = 1.0
    node.candidate_min_points = 2
    node.candidate_min_extent_m = 1.0
    node.odom_frame = "odom"
    node.base_frame = "front_axle"
    node._is_alias = lambda frame_a, frame_b: frame_a == frame_b
    node.get_clock = lambda: _FakeClock(TimeMsg(sec=1, nanosec=0))
    return node


def _sample_result() -> SingleBoundaryPlannerResult:
    return SingleBoundaryPlannerResult(
        filtered_points=np.empty((0, 2), dtype=np.float64),
        filtered_colors=[],
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.empty((0, 2), dtype=np.int64),
        selected_pair_track_ids=np.empty((0, 2), dtype=np.int64),
        midpoints_raw=np.empty((0, 2), dtype=np.float64),
        centerline=np.array([[2.0, 0.1], [4.0, 0.2]], dtype=np.float64),
        left_boundary=np.array([[2.0, 1.8], [4.0, 2.0]], dtype=np.float64),
        right_boundary=np.empty((0, 2), dtype=np.float64),
        used_fallback=True,
        status="ok",
        planner_mode="single_boundary",
        active_boundary_side="blue",
        raw_offset_path=np.array([[2.0, 0.1], [4.0, 0.2]], dtype=np.float64),
    )


def _marker_map(marker_array) -> dict[str, object]:
    return {
        marker.ns: marker
        for marker in marker_array.markers
        if getattr(marker, "ns", "")
    }


def _marker_xy(marker) -> np.ndarray:
    return np.asarray([[point.x, point.y] for point in marker.points], dtype=np.float64)


def test_publish_diagnostics_uses_single_boundary_identity():
    node = _make_node()
    node._publish_diagnostics(
        frame_id="odom",
        centerline_jump_max_m=0.1,
        selected_edge_churn_ratio=0.2,
        tracked_cones_frame_delta_p95_m=0.3,
        centerline_point_count=2,
        selected_edge_count=0,
        status="ok",
        planner_metrics={"planner_mode": "single_boundary"},
    )
    msg = node._diag_pub.messages[-1]
    assert msg.status[0].name == "single_boundary_planner/stability"


def test_tracked_cone_planning_frame_rejects_tentative_cones_for_single_boundary():
    node = _make_node()
    msg = ConeDetectionArray()
    msg.header.frame_id = "odom"
    msg.header.stamp = TimeMsg(sec=0, nanosec=0)

    cone = ConeDetection()
    cone.color = "orange"
    cone.boundary_color = "blue"
    cone.confidence = 0.2
    cone.track_id = 41
    cone.track_state = MSG_TRACK_STATE_TENTATIVE
    cone.track_confidence = 0.85
    msg.cones.append(cone)

    planning_frame = node._tracked_cone_planning_frame(
        msg=msg,
        points_xy=np.array([[2.0, 1.0]], dtype=np.float64),
        colors=["blue"],
        confidences=np.array([0.2], dtype=np.float64),
    )

    assert planning_frame.track_ids.tolist() == [41]
    assert planning_frame.track_states.tolist() == [MSG_TRACK_STATE_TENTATIVE]
    assert np.allclose(planning_frame.planner_confidences, np.array([0.0], dtype=np.float64))


def test_single_boundary_reject_reason_maps_to_no_safe_chain():
    node = _make_node()
    result = _sample_result()
    result.status = "no reliable boundary chain"
    result.reject_reason = result.status
    assert node._normalize_core_reject_reason(result) == "no_safe_chain"


def test_build_markers_show_remembered_cones_instead_of_filtered_subset():
    node = _make_node()
    node.show_raw_cones = True
    result = _sample_result()
    result.filtered_points = np.array([[9.0, 9.0]], dtype=np.float64)
    node._update_remembered_cone_viz(
        points_xy=np.array([[1.0, 1.0], [2.0, -2.0]], dtype=np.float64),
        colors=["blue", "yellow"],
    )

    markers = node._build_markers(
        now=TimeMsg(sec=1, nanosec=0),
        frame_id="odom",
        result=result,
        centerline=result.centerline,
        raw_centerline=result.centerline,
        raw_midpoint_chain=result.midpoints_raw,
        status="ok",
        operator_state="fresh",
        control_target_frame=None,
    )
    by_ns = _marker_map(markers)
    assert "remembered_cones" in by_ns
    assert np.allclose(
        _marker_xy(by_ns["remembered_cones"]),
        np.array([[1.0, 1.0], [2.0, -2.0]], dtype=np.float64),
    )


def test_candidate_path_accepts_validated_jump_when_near_field_stays_aligned():
    node = _make_node()
    node._midline_buffer_path = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        dtype=np.float64,
    )
    node._extract_forward_path_from_pose = lambda path, vehicle_xy, resolution_m: np.array(path, copy=True)
    node._resample_midline_stations = lambda path: np.array(path, copy=True)
    candidate = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.05], [3.0, 0.10], [4.0, 1.30], [5.0, 1.30]],
        dtype=np.float64,
    )
    result = _sample_result()

    ok, reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="validated",
    )

    assert ok is True
    assert reason == "ok"


def test_candidate_path_rejects_validated_jump_when_near_field_is_shifted():
    node = _make_node()
    node._midline_buffer_path = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    node._extract_forward_path_from_pose = lambda path, vehicle_xy, resolution_m: np.array(path, copy=True)
    node._resample_midline_stations = lambda path: np.array(path, copy=True)
    candidate = np.array([[0.0, 1.2], [2.0, 1.2], [4.0, 1.2]], dtype=np.float64)
    result = _sample_result()

    ok, reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="validated",
    )

    assert ok is True
    assert reason == "ok"


def test_validated_near_field_jump_ok_replaces_buffer_directly():
    node = _make_node()
    node.midline_min_estimated_extent_m = 5.0
    stored_path = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        dtype=np.float64,
    )
    candidate = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.05], [3.0, 0.10], [4.0, 1.30], [5.0, 1.30]],
        dtype=np.float64,
    )
    node._midline_buffer_path = np.array(stored_path, copy=True)
    node._midline_buffer_confidence = 1.0
    node._midline_buffer_last_update_sec = 10.0
    node._extract_forward_path_from_pose = lambda path, vehicle_xy, resolution_m: np.array(path, copy=True)
    node._resample_midline_stations = lambda path: np.array(path, copy=True)
    result = _sample_result()

    updated = node._update_midline_buffer(
        candidate_centerline=candidate,
        candidate_source="validated",
        candidate_update_ok=True,
        candidate_update_reason="candidate_jump_near_field_ok",
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        now_sec=11.0,
    )

    assert node._last_midline_update_mode == "direct"
    candidate_y = np.interp(updated[:, 0], candidate[:, 0], candidate[:, 1])
    assert np.allclose(updated[:, 1], candidate_y, atol=1e-3)


def test_single_boundary_raw_offset_soft_accept_still_works():
    node = _make_node()
    result = _sample_result()
    result.status = "near-field continuity rejected fresh path"
    result.reject_reason = result.status
    candidate = np.array([[0.0, 0.0], [1.5, 0.1], [3.0, 0.2]], dtype=np.float64)

    ok, reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="single_boundary_raw_offset",
    )

    assert ok is True
    assert reason == "single_boundary_raw_offset_soft_accept"
