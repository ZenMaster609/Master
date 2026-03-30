from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time as TimeMsg

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from sim_car.planning.corridor_planner_core import CorridorPlannerResult  # noqa: E402
    from sim_car.planning.corridor_planner_node import CorridorPlannerNode  # noqa: E402
    from sim_car.planning.planner_runtime_types import PlannerIdentity  # noqa: E402
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


def _make_node() -> CorridorPlannerNode:
    node = object.__new__(CorridorPlannerNode)
    node._planner_identity = PlannerIdentity(
        node_name="corridor_planner_node",
        planner_mode="corridor",
        diagnostics_prefix="corridor_planner",
        diagnostics_topic="/corridor_planner/diagnostics",
    )
    node._diag_pub = _FakePublisher()
    node.publish_control_debug = False
    node.publish_thesis_context = False
    node._hold_mode_active = False
    node._hold_clean_frame_count = 0
    node._active_planner_mode = "corridor"
    node._active_remembered_cone_count = 8
    node._active_stale_cone_count = 0
    node._active_left_chain_length = 4
    node._active_right_chain_length = 4
    node._active_pair_count = 5
    node._active_unknown_pair_count = 0
    node._active_filtered_track_width_m = 3.6
    node._active_held_path_flag = 0
    node.show_raw_cones = False
    node.show_boundary_chains = True
    node.show_pair_lines = True
    node.show_raw_midpoint_chain = True
    node.show_raw_prevalidation_centerline = True
    node.show_lookahead_point = False
    node._current_pair_segments_for_viz = None
    node._pair_memory = []
    node._candidate_jump_reject_streak = 0
    node._last_midline_update_mode = "hold"
    node._midline_buffer_path = None
    node._midline_buffer_confidence = 0.0
    node._midline_buffer_last_update_sec = -1.0
    node._last_viz_left_boundary = None
    node._last_viz_right_boundary = None
    node._last_valid_centerline = None
    node._last_valid_raw_midpoint_chain = None
    node._last_valid_pair_segments = None
    node._last_valid_time_sec = -1.0
    node.hold_last_valid_s = 1.25
    node.midline_hold_last_valid_duration_s = 1.25
    node.pair_memory_retention_s = 8.0
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
    node.centerline_path_resolution_m = 0.5
    node.candidate_jump_reject_threshold_m = 1.0
    node.candidate_jump_recover_frames = 3
    node.candidate_min_points = 2
    node.candidate_min_extent_m = 1.0
    node.odom_frame = "odom"
    node.base_frame = "front_axle"
    node._core_config = SimpleNamespace(
        path_resolution_m=0.5,
        max_path_length_m=20.0,
        min_forward_extent_m=2.0,
        min_required_corridor_samples=3,
    )
    node._filtered_track_width_m = 3.6
    node._is_alias = lambda frame_a, frame_b: frame_a == frame_b
    node.get_clock = lambda: _FakeClock(TimeMsg(sec=1, nanosec=0))
    return node


def _sample_result() -> CorridorPlannerResult:
    left = np.array([[2.0, 1.8], [4.0, 1.8], [6.0, 1.8]], dtype=np.float64)
    right = np.array([[2.0, -1.8], [4.0, -1.8], [6.0, -1.8]], dtype=np.float64)
    anchors = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float64)
    rungs = np.empty((3, 2, 2), dtype=np.float64)
    rungs[:, 0, :] = left
    rungs[:, 1, :] = right
    return CorridorPlannerResult(
        filtered_points=np.vstack((left, right)),
        filtered_colors=["blue", "blue", "blue", "yellow", "yellow", "yellow"],
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.empty((0, 2), dtype=np.int64),
        selected_pair_track_ids=np.empty((0, 2), dtype=np.int64),
        midpoints_raw=anchors,
        centerline=np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float64),
        prevalidation_centerline=np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float64),
        left_boundary=left,
        right_boundary=right,
        used_fallback=False,
        status="ok",
        planner_mode="corridor",
        pair_segments=rungs,
        accepted_pair_count=3,
        left_chain_length=3,
        right_chain_length=3,
        selected_chain_width_median=3.6,
        corridor_width_min_m=3.6,
        corridor_width_max_m=3.6,
    )


def _marker_map(marker_array) -> dict[str, object]:
    return {
        marker.ns: marker
        for marker in marker_array.markers
        if getattr(marker, "ns", "")
    }


def _marker_xy(marker) -> np.ndarray:
    return np.asarray([[point.x, point.y] for point in marker.points], dtype=np.float64)


def test_publish_diagnostics_uses_corridor_identity():
    node = _make_node()
    node._publish_diagnostics(
        frame_id="odom",
        centerline_jump_max_m=0.1,
        selected_edge_churn_ratio=0.2,
        tracked_cones_frame_delta_p95_m=0.3,
        centerline_point_count=3,
        selected_edge_count=3,
        status="ok",
        planner_metrics={
            "planner_mode": "corridor",
            "corridor_sample_count": 5,
            "corridor_width_min_m": 3.2,
            "corridor_width_median_m": 3.5,
            "corridor_width_max_m": 3.8,
        },
    )
    msg = node._diag_pub.messages[-1]
    assert msg.status[0].name == "corridor_planner/stability"
    values = {item.key: item.value for item in msg.status[0].values}
    assert values["planner_mode"] == "corridor"
    assert values["corridor_sample_count"] == "5"


def test_normalize_core_reject_reason_maps_to_no_safe_chain():
    node = _make_node()
    result = _sample_result()
    result.status = "no valid corridor overlap"
    result.reject_reason = result.status
    result.centerline = np.empty((0, 2), dtype=np.float64)
    assert node._normalize_core_reject_reason(result) == "no_safe_chain"


def test_build_markers_include_corridor_anchor_and_rung_namespaces():
    node = _make_node()
    result = _sample_result()
    node._current_pair_segments_for_viz = result.pair_segments

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
    assert "corridor_center_anchors" in by_ns
    assert "corridor_rungs" in by_ns
    assert "centerline" in by_ns


def test_build_markers_show_remembered_cones_instead_of_filtered_subset():
    node = _make_node()
    node.show_raw_cones = True
    result = _sample_result()
    result.filtered_points = np.array([[8.0, 8.0]], dtype=np.float64)
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


def test_held_centerline_returns_last_valid_path_within_timeout():
    node = _make_node()
    node._last_valid_centerline = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    node._last_valid_time_sec = 10.0

    held = node._held_centerline(10.5)
    assert held is not None
    assert np.allclose(held, node._last_valid_centerline)


def test_candidate_path_rejects_projected_corridor_when_core_status_is_not_ok():
    node = _make_node()
    result = _sample_result()
    result.status = "no reliable corridor boundaries"
    result.reject_reason = result.status
    candidate = np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]], dtype=np.float64)

    ok, reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="projected_corridor",
    )

    assert ok is False
    assert reason == result.status


def test_candidate_path_rejects_soft_corridor_when_core_status_is_not_ok():
    node = _make_node()
    result = _sample_result()
    result.status = "path forward extent too short"
    result.reject_reason = result.status
    candidate = np.array([[0.0, 0.0], [0.9, 0.1]], dtype=np.float64)

    ok, reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="soft_corridor",
    )

    assert ok is False
    assert reason == "candidate_extent_too_short"


def test_remembered_corridor_geometry_can_project_candidate():
    node = _make_node()
    result = _sample_result()

    node._remember_pairs(
        result=result,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        now_sec=5.0,
    )
    entries = node._active_pair_memory_entries(
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )
    pair_segments, midpoint_chain = node._pair_geometry_from_memory(entries)
    candidate = node._project_corridor_memory_candidate(
        midpoint_chain=midpoint_chain,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert pair_segments.shape[0] == 3
    assert midpoint_chain.shape[0] == 3
    assert candidate.shape[0] >= 3


def test_valid_live_corridor_candidate_blends_into_existing_buffer():
    node = _make_node()
    stored_path = np.array([[0.0, 0.8], [1.0, 0.8], [2.0, 0.8], [3.0, 0.8]], dtype=np.float64)
    candidate = np.array([[0.0, 0.2], [1.0, 0.2], [2.0, 0.2], [3.0, 0.2]], dtype=np.float64)
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
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        now_sec=11.0,
    )

    assert node._last_midline_update_mode == "blend"
    assert not np.allclose(updated, candidate)
    assert abs(updated[1, 1] - stored_path[1, 1]) <= node.midline_near_max_shift_m + 1e-9
    assert updated[1, 1] > candidate[1, 1]


def test_candidate_path_rejects_projected_corridor_jump_against_stored_midline():
    node = _make_node()
    node._midline_buffer_path = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    candidate = np.array([[0.0, 1.2], [2.0, 1.2], [4.0, 1.2]], dtype=np.float64)
    result = _sample_result()
    result.status = "no reliable corridor boundaries"
    result.reject_reason = result.status

    ok, reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="projected_corridor",
    )

    assert ok is False
    assert reason == "candidate_jump_rejected"


def test_select_candidate_centerline_recovers_live_corridor_after_near_field_reject():
    node = _make_node()
    result = _sample_result()
    result.status = "near-field continuity rejected fresh path"
    result.reject_reason = result.status

    centerline, source = node._select_candidate_centerline(
        result=result,
        support_chain=result.midpoints_raw,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert source == "recoverable_live_path"
    assert np.allclose(centerline, result.prevalidation_centerline)


def test_select_candidate_centerline_completes_recoverable_corridor_prefix():
    node = _make_node()
    result = _sample_result()
    result.status = "too few valid corridor samples"
    result.reject_reason = result.status
    result.centerline = np.array([[0.5, 0.0], [2.8, 0.1]], dtype=np.float64)
    result.prevalidation_centerline = np.array([[0.5, 0.0], [2.8, 0.1]], dtype=np.float64)
    result.midpoints_raw = np.array([[0.5, 0.0], [2.8, 0.1]], dtype=np.float64)

    centerline, source = node._select_candidate_centerline(
        result=result,
        support_chain=result.midpoints_raw,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert source == "completed_live_prefix"
    assert centerline.shape[0] >= result.prevalidation_centerline.shape[0]
    assert np.allclose(centerline[: result.prevalidation_centerline.shape[0]], result.prevalidation_centerline)
