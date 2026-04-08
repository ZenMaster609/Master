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
    from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray  # noqa: E402
    from sim_car.planning.midpoint_planner_core import MidpointPlannerResult  # noqa: E402
    from sim_car.planning.midpoint_planner_node import (  # noqa: E402
        MSG_TRACK_STATE_CONFIRMED,
        MSG_TRACK_STATE_STALE,
        MSG_TRACK_STATE_TENTATIVE,
        _PairMemoryEntry,
    )
    from sim_car.planning.midpoint_planner_node import MidpointPlannerNode  # noqa: E402
    from sim_car.planning.planner_runtime_types import PlannerIdentity  # noqa: E402
    from sim_car.controllers.pure_pursuit_controller import PurePursuitController  # noqa: E402
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


def _make_node() -> MidpointPlannerNode:
    node = object.__new__(MidpointPlannerNode)
    node._planner_identity = PlannerIdentity(
        node_name="midpoint_planner_node",
        planner_mode="midpoint",
        diagnostics_prefix="midpoint_planner",
        diagnostics_topic="/midpoint_planner/diagnostics",
    )
    node._diag_pub = _FakePublisher()
    node.publish_control_debug = False
    node.publish_thesis_context = False
    node._hold_mode_active = False
    node._hold_clean_frame_count = 0
    node._active_planner_mode = "midpoint"
    node.odom_frame = "odom"
    node.base_frame = "front_axle"
    node.infer_unknown_by_side = True
    node.infer_orange_by_side = True
    node.orange_min_lateral_m = 0.9
    node.orange_neighbor_radius_m = 3.5
    node.orange_neighbor_margin_m = 0.75
    node._pair_memory = []
    node._core_config = type(
        "Cfg",
        (),
        {
            "min_confidence": 0.3,
            "min_forward_extent_m": 2.0,
            "min_pair_count": 3,
            "min_trustworthy_pairs": 3,
            "path_resolution_m": 0.5,
            "max_path_length_m": 30.0,
            "smoothing_window": 1,
        },
    )()
    node.show_raw_cones = False
    node.show_boundary_chains = False
    node.show_pair_lines = True
    node.show_raw_midpoint_chain = True
    node.show_raw_offset_path = False
    node.show_raw_prevalidation_centerline = False
    node.show_lookahead_point = False
    node._current_pair_segments_for_viz = None
    node._last_viz_left_boundary = None
    node._last_viz_right_boundary = None
    node._last_viz_raw_offset_path = None
    node._last_valid_centerline = None
    node._last_valid_raw_midpoint_chain = None
    node._last_valid_pair_segments = None
    node._last_valid_time_sec = -1.0
    node.hold_last_valid_s = 1.25
    node.centerline_path_resolution_m = 0.5
    node.candidate_min_points = 2
    node.candidate_min_extent_m = 0.5
    node.midline_station_spacing_m = 0.5
    node.midline_control_handoff_distance_m = 1.5
    node.midline_near_distance_m = 4.0
    node.midline_mid_distance_m = 12.0
    node.midline_near_alpha = 0.06
    node.midline_mid_alpha = 0.18
    node.midline_far_alpha = 0.35
    node.midline_near_max_shift_m = 0.10
    node.midline_mid_max_shift_m = 0.20
    node.midline_far_max_shift_m = 0.40
    node.midline_horizon_m = 30.0
    node.centerline_jump_horizon_m = 8.0
    node.candidate_jump_reject_threshold_m = 1.0
    node.candidate_jump_recover_frames = 3
    node._candidate_jump_reject_streak = 0
    node._midline_buffer_path = None
    node._midline_buffer_confidence = 0.0
    node._midline_buffer_last_update_sec = -1.0
    node._last_midline_update_mode = "hold"
    node._active_left_chain_length = 0
    node._active_right_chain_length = 0
    node._active_pair_count = 0
    node._active_unknown_pair_count = 0
    node._active_filtered_track_width_m = 3.6
    node._active_held_path_flag = 0
    node._active_chain_stage = "waiting"
    node._active_reject_wrong_side_count = 0
    node._active_reject_width_count = 0
    node._active_reject_width_range_count = 0
    node._active_reject_progress_count = 0
    node._active_reject_orientation_count = 0
    node._is_alias = lambda frame_a, frame_b: frame_a == frame_b
    node.get_clock = lambda: _FakeClock(TimeMsg(sec=1, nanosec=0))
    return node


def _sample_result() -> MidpointPlannerResult:
    return MidpointPlannerResult(
        filtered_points=np.empty((0, 2), dtype=np.float64),
        filtered_colors=[],
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.empty((0, 2), dtype=np.int64),
        selected_pair_track_ids=np.empty((0, 2), dtype=np.int64),
        midpoints_raw=np.array([[2.0, 0.0], [4.0, 0.0]], dtype=np.float64),
        centerline=np.array([[2.0, 0.0], [4.0, 0.0]], dtype=np.float64),
        prevalidation_centerline=np.array([[2.0, 0.0], [4.0, 0.0]], dtype=np.float64),
        left_boundary=np.empty((0, 2), dtype=np.float64),
        right_boundary=np.empty((0, 2), dtype=np.float64),
        used_fallback=False,
        status="ok",
        planner_mode="midpoint",
        pair_segments=np.empty((0, 2, 2), dtype=np.float64),
        accepted_pair_count=0,
    )


def _marker_map(marker_array) -> dict[str, object]:
    return {
        marker.ns: marker
        for marker in marker_array.markers
        if getattr(marker, "ns", "")
    }


def _marker_xy(marker) -> np.ndarray:
    return np.asarray([[point.x, point.y] for point in marker.points], dtype=np.float64)


def _cone_msg(points: list[tuple[float, float]], *, color: str, boundary_color: str = "") -> ConeDetectionArray:
    msg = ConeDetectionArray()
    msg.header.frame_id = "odom"
    msg.header.stamp = TimeMsg(sec=0, nanosec=0)
    for x, y in points:
        cone = ConeDetection()
        cone.color = color
        cone.boundary_color = boundary_color
        cone.confidence = 0.9
        cone.position.x = float(x)
        cone.position.y = float(y)
        cone.position.z = 0.0
        msg.cones.append(cone)
    return msg


def test_publish_diagnostics_uses_midpoint_identity():
    node = _make_node()
    node._publish_diagnostics(
        frame_id="odom",
        centerline_jump_max_m=0.1,
        selected_edge_churn_ratio=0.2,
        tracked_cones_frame_delta_p95_m=0.3,
        centerline_point_count=2,
        selected_edge_count=1,
        status="ok",
        planner_metrics={
            "planner_mode": "midpoint",
            "raw_orange_count": 2,
            "resolved_blue_count": 1,
            "resolved_yellow_count": 1,
            "boundary_hint_count": 2,
            "candidate_source": "validated",
            "midline_update_mode": "direct",
        },
    )
    msg = node._diag_pub.messages[-1]
    assert msg.status[0].name == "midpoint_planner/stability"
    values = {item.key: item.value for item in msg.status[0].values}
    assert values["raw_orange_count"] == "2"
    assert values["resolved_blue_count"] == "1"
    assert values["resolved_yellow_count"] == "1"
    assert values["boundary_hint_count"] == "2"
    assert values["candidate_source"] == "validated"
    assert values["midline_update_mode"] == "direct"


def test_tracked_cone_planning_frame_keeps_hinted_tentative_cones_for_midpoint():
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
    assert planning_frame.boundary_hints == ["blue"]
    assert planning_frame.raw_colors == ["orange"]
    assert np.allclose(planning_frame.planner_confidences, np.array([0.85], dtype=np.float64))


def test_build_operator_status_text_shows_chain_stage_and_pair_rejects():
    node = _make_node()
    node._active_planner_mode = "midpoint"
    node._active_left_chain_length = 4
    node._active_right_chain_length = 4
    node._active_pair_count = 1
    node._active_unknown_pair_count = 0
    node._active_chain_stage = "pairing"
    node._active_reject_wrong_side_count = 5
    node._active_reject_width_count = 1
    node._active_reject_width_range_count = 3
    node._active_reject_progress_count = 0
    node._active_reject_orientation_count = 0

    text = node._build_operator_status_text(
        operator_state="fresh",
        operator_reason="none",
        centerline_point_count=3,
        cmd_speed=1.2,
        cmd_steering=0.1,
        lookahead=2.0,
        candidate_diagonal_count=99,
        selected_chain_length=77,
        seed_midpoint_distance_m=1.0,
        near_field_lateral_max_m=0.2,
        near_field_midpoint_kink_max_rad=0.1,
        hold_remaining_s=0.5,
    )

    assert "FLOW: stage=PAIRING" in text
    assert "wrong=5" in text
    assert "range=3" in text
    assert "NF:" not in text
    assert "seed=" not in text


def test_build_steering_controller_creates_active_pure_pursuit_controller():
    params = {
        "stanley.k_gain": 1.2,
        "stanley.softening_speed_mps": 0.0,
        "stanley.heading_gain": 1.6,
        "stanley.lookahead_idx_offset": 0,
        "stanley.steering_limit_rad": 0.52,
        "stanley.steering_lowpass_alpha": 1.0,
        "stanley.steering_rate_limit_rad_s": 10.0,
        "stanley.use_yaw_rate_damping": True,
        "stanley.yaw_rate_damping_gain": 0.0,
        "stanley.wheelbase_m": 1.65,
        "stanley.cross_track_deadband_m": 0.0,
        "pure_pursuit.lookahead_m": 2.0,
        "pure_pursuit.min_lookahead_m": 1.0,
        "pure_pursuit.max_lookahead_m": 5.0,
        "pure_pursuit.lookahead_gain": 0.5,
        "pure_pursuit.steering_limit_rad": 0.52,
        "pure_pursuit.steering_lowpass_alpha": 1.0,
        "pure_pursuit.steering_rate_limit_rad_s": 0.0,
        "pure_pursuit.wheelbase_m": 1.65,
    }

    node = object.__new__(MidpointPlannerNode)
    node.controller_type = "pure_pursuit"
    node.publish_rate_hz = 20.0
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])

    controller = node._build_steering_controller()
    output = controller.compute(
        control_path=np.array([[2.0, 1.0], [12.0, 1.0]], dtype=np.float64),
        speed_mps=3.0,
        yaw_rate_rps=0.0,
    )

    assert isinstance(controller, PurePursuitController)
    assert np.isfinite(output.steering_rad)
    assert output.steering_rad > 0.0


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


def test_convert_cones_to_frame_resolves_orange_with_and_without_boundary_hints():
    node = _make_node()
    points = [(2.0, 1.8), (2.0, -1.8), (4.0, 1.8), (4.0, -1.8)]

    direct_msg = _cone_msg(points, color="orange")
    hinted_msg = ConeDetectionArray()
    hinted_msg.header = direct_msg.header
    expected_colors = ["blue", "yellow", "blue", "yellow"]
    for (x, y), boundary_color in zip(points, expected_colors):
        cone = ConeDetection()
        cone.color = "orange"
        cone.boundary_color = boundary_color
        cone.confidence = 0.9
        cone.position.x = float(x)
        cone.position.y = float(y)
        cone.position.z = 0.0
        hinted_msg.cones.append(cone)

    direct_points, direct_colors, direct_conf = node._convert_cones_to_frame(
        direct_msg,
        "odom",
        "odom",
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
    )
    hinted_points, hinted_colors, hinted_conf = node._convert_cones_to_frame(
        hinted_msg,
        "odom",
        "odom",
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
    )

    assert np.allclose(direct_points, hinted_points)
    assert direct_colors == expected_colors
    assert hinted_colors == expected_colors
    assert np.allclose(direct_conf, 0.9)
    assert np.allclose(hinted_conf, 0.9)


def test_normalize_core_reject_reason_maps_to_no_safe_chain():
    node = _make_node()
    result = _sample_result()
    result.status = "no reliable midpoint chain"
    result.reject_reason = result.status
    assert node._normalize_core_reject_reason(result) == "no_safe_chain"


def test_held_pair_geometry_returns_last_valid_geometry_within_hold_timeout():
    node = _make_node()
    node._last_valid_time_sec = 10.0
    node._last_valid_centerline = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    node._last_valid_pair_segments = np.array(
        [
            [[1.0, 1.0], [1.0, -1.0]],
            [[3.0, 1.0], [3.0, -1.0]],
        ],
        dtype=np.float64,
    )
    node._last_valid_raw_midpoint_chain = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float64)

    held_pair_segments, held_raw_midpoint_chain = node._held_pair_geometry(now_sec=10.5)

    assert np.allclose(held_pair_segments, node._last_valid_pair_segments)
    assert np.allclose(held_raw_midpoint_chain, node._last_valid_raw_midpoint_chain)


def test_held_pair_geometry_expires_with_hold_timeout():
    node = _make_node()
    node._last_valid_time_sec = 10.0
    node._last_valid_centerline = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    node._last_valid_pair_segments = np.array(
        [[[1.0, 1.0], [1.0, -1.0]]],
        dtype=np.float64,
    )
    node._last_valid_raw_midpoint_chain = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float64)

    held_pair_segments, held_raw_midpoint_chain = node._held_pair_geometry(
        now_sec=10.0 + node.hold_last_valid_s + 0.01
    )

    assert held_pair_segments is None
    assert held_raw_midpoint_chain is None


def test_build_markers_publish_held_pair_geometry_when_current_frame_loses_pairs():
    node = _make_node()
    held_pair_segments = np.array(
        [
            [[1.0, 1.0], [1.0, -1.0]],
            [[3.0, 1.0], [3.0, -1.0]],
        ],
        dtype=np.float64,
    )
    held_raw_midpoint_chain = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    node._current_pair_segments_for_viz = np.array(held_pair_segments, copy=True)
    result = _sample_result()

    markers = node._build_markers(
        now=TimeMsg(sec=9, nanosec=1),
        frame_id="odom",
        result=result,
        centerline=np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64),
        raw_centerline=np.empty((0, 2), dtype=np.float64),
        raw_midpoint_chain=held_raw_midpoint_chain,
        status="ok",
        operator_state="held",
        control_target_frame=None,
    )

    by_ns = _marker_map(markers)
    assert np.allclose(_marker_xy(by_ns["accepted_pairs"]), held_pair_segments.reshape(-1, 2))
    assert np.allclose(_marker_xy(by_ns["raw_midpoint_chain"]), held_raw_midpoint_chain)


def test_build_markers_clear_pair_geometry_after_hold_timeout():
    node = _make_node()
    result = _sample_result()

    markers = node._build_markers(
        now=TimeMsg(sec=9, nanosec=1),
        frame_id="odom",
        result=result,
        centerline=np.empty((0, 2), dtype=np.float64),
        raw_centerline=np.empty((0, 2), dtype=np.float64),
        raw_midpoint_chain=np.empty((0, 2), dtype=np.float64),
        status="expired",
        operator_state="stopped",
        control_target_frame=None,
    )

    by_ns = _marker_map(markers)
    assert _marker_xy(by_ns["accepted_pairs"]).size == 0
    assert _marker_xy(by_ns["raw_midpoint_chain"]).size == 0


def test_published_pair_count_prefers_held_pair_geometry():
    node = _make_node()
    result = _sample_result()
    held_pair_segments = np.array(
        [
            [[1.0, 1.0], [1.0, -1.0]],
            [[3.0, 1.0], [3.0, -1.0]],
        ],
        dtype=np.float64,
    )

    assert node._published_pair_count(held_pair_segments, result) == 2


def test_active_pair_memory_keeps_pairs_until_vehicle_has_passed_them():
    node = _make_node()
    node._pair_memory = [
        _PairMemoryEntry(1, 2, 2.0, 0.0, 2.0, 1.0, 2.0, -1.0),
        _PairMemoryEntry(3, 4, -0.4, 0.0, -0.4, 1.0, -0.4, -1.0),
        _PairMemoryEntry(5, 6, -0.6, 0.0, -0.6, 1.0, -0.6, -1.0),
    ]

    active = node._active_pair_memory(vehicle_x=0.0, vehicle_y=0.0, vehicle_yaw=0.0)

    assert active == [(1, 2), (3, 4)]
    assert [(entry.left_track_id, entry.right_track_id) for entry in node._pair_memory] == [(1, 2), (3, 4)]


def test_pair_geometry_from_memory_restores_pair_segments_and_midpoints():
    node = _make_node()
    entries = [
        _PairMemoryEntry(1, 2, 2.0, 0.0, 2.0, 1.0, 2.0, -1.0),
        _PairMemoryEntry(3, 4, 4.0, 0.0, 4.0, 1.2, 4.0, -1.2),
    ]

    pair_segments, midpoint_chain = node._pair_geometry_from_memory(entries)

    assert np.allclose(
        pair_segments,
        np.array(
            [
                [[2.0, 1.0], [2.0, -1.0]],
                [[4.0, 1.2], [4.0, -1.2]],
            ],
            dtype=np.float64,
        ),
    )
    assert np.allclose(midpoint_chain, np.array([[2.0, 0.0], [4.0, 0.0]], dtype=np.float64))


def test_merge_pair_entries_keeps_live_pairs_and_retains_missing_remembered_pairs():
    node = _make_node()
    remembered_entries = [
        _PairMemoryEntry(1, 2, 2.0, 0.0, 2.0, 1.0, 2.0, -1.0),
        _PairMemoryEntry(3, 4, 4.0, 0.0, 4.0, 1.0, 4.0, -1.0),
        _PairMemoryEntry(5, 6, 6.0, 0.0, 6.0, 1.0, 6.0, -1.0),
    ]
    live_entries = [
        _PairMemoryEntry(3, 4, 4.1, 0.1, 4.1, 1.1, 4.1, -0.9),
    ]

    merged = node._merge_pair_entries(
        remembered_entries=remembered_entries,
        live_entries=live_entries,
    )

    assert [(entry.left_track_id, entry.right_track_id) for entry in merged] == [(1, 2), (3, 4), (5, 6)]
    assert abs(merged[1].midpoint_x_odom - 4.1) < 1e-9
    assert abs(merged[1].midpoint_y_odom - 0.1) < 1e-9


def test_sort_pair_entries_by_forward_progress_orders_projected_pairs_deterministically():
    node = _make_node()
    entries = [
        _PairMemoryEntry(5, 6, 4.0, 0.0, 4.0, 1.0, 4.0, -1.0),
        _PairMemoryEntry(1, 2, 2.0, 0.0, 2.0, 1.0, 2.0, -1.0),
        _PairMemoryEntry(3, 4, 3.0, 0.1, 3.0, 1.1, 3.0, -0.9),
    ]

    ordered = node._sort_pair_entries_by_forward_progress(
        entries=entries,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )
    _, midpoint_chain = node._pair_geometry_from_memory(ordered)
    candidate = node._project_midpoint_chain_candidate(
        midpoint_chain=midpoint_chain,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert [(entry.left_track_id, entry.right_track_id) for entry in ordered] == [(1, 2), (3, 4), (5, 6)]
    assert np.all(np.diff(midpoint_chain[:, 0]) > 0.0)
    assert candidate.shape[0] >= 2
    assert np.all(np.diff(candidate[:, 0]) > -1e-9)


def test_remember_pairs_keeps_confirmed_and_stale_pairs_with_sufficient_confidence():
    node = _make_node()
    result = _sample_result()
    result.selected_pair_track_ids = np.array([[11, 12]], dtype=np.int64)
    result.pair_segments = np.array([[[2.0, 1.0], [2.0, -1.0]]], dtype=np.float64)

    node._remember_pairs(
        result=result,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        track_ids=np.array([11, 12], dtype=np.int64),
        track_states=np.array([MSG_TRACK_STATE_CONFIRMED, MSG_TRACK_STATE_STALE], dtype=np.int64),
        planner_confidences=np.array([0.9, 0.8], dtype=np.float64),
    )

    assert len(node._pair_memory) == 1
    assert node._pair_memory[0].left_track_id == 11
    assert node._pair_memory[0].right_track_id == 12
    assert abs(node._pair_memory[0].midpoint_x_odom - 2.0) < 1e-9
    assert abs(node._pair_memory[0].midpoint_y_odom - 0.0) < 1e-9
    assert abs(node._pair_memory[0].left_x_odom - 2.0) < 1e-9
    assert abs(node._pair_memory[0].right_y_odom + 1.0) < 1e-9


def test_remember_pairs_rejects_tentative_tracks():
    node = _make_node()
    result = _sample_result()
    result.selected_pair_track_ids = np.array([[21, 22]], dtype=np.int64)
    result.pair_segments = np.array([[[2.0, 1.0], [2.0, -1.0]]], dtype=np.float64)

    node._remember_pairs(
        result=result,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        track_ids=np.array([21, 22], dtype=np.int64),
        track_states=np.array([MSG_TRACK_STATE_TENTATIVE, MSG_TRACK_STATE_CONFIRMED], dtype=np.int64),
        planner_confidences=np.array([0.9, 0.9], dtype=np.float64),
    )

    assert node._pair_memory == []


def test_remember_pairs_keeps_non_tentative_pairs_even_with_low_confidence_metadata():
    node = _make_node()
    result = _sample_result()
    result.selected_pair_track_ids = np.array([[31, 32]], dtype=np.int64)
    result.pair_segments = np.array([[[2.0, 1.0], [2.0, -1.0]]], dtype=np.float64)

    node._remember_pairs(
        result=result,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        track_ids=np.array([31, 32], dtype=np.int64),
        track_states=np.array([MSG_TRACK_STATE_CONFIRMED, MSG_TRACK_STATE_CONFIRMED], dtype=np.int64),
        planner_confidences=np.array([0.29, 0.9], dtype=np.float64),
    )

    assert len(node._pair_memory) == 1
    assert node._pair_memory[0].left_track_id == 31
    assert node._pair_memory[0].right_track_id == 32


def test_project_midpoint_chain_candidate_extends_sparse_pairs_to_minimum_extent():
    node = _make_node()

    candidate = node._project_midpoint_chain_candidate(
        midpoint_chain=np.array([[0.8, 0.0], [1.3, 0.2]], dtype=np.float64),
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert candidate.shape[0] >= 2
    assert node._candidate_forward_extent_m(
        centerline=candidate,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    ) >= node._minimum_projected_forward_extent_m()


def test_blend_midline_samples_snaps_to_candidate_when_within_allowed_shift():
    node = _make_node()
    stored = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    candidate = np.array(
        [[0.0, 0.0], [1.0, 0.08], [2.0, 0.09], [3.0, 0.08], [4.0, 0.07]],
        dtype=np.float64,
    )

    updated = node._blend_midline_samples(
        stored_samples=stored,
        candidate_samples=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert np.allclose(updated[4], candidate[4])


def test_blend_midline_samples_moves_toward_candidate_more_aggressively_when_far_apart():
    node = _make_node()
    stored = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        dtype=np.float64,
    )
    candidate = np.array(
        [[0.0, 0.0], [1.0, 0.6], [2.0, 0.8], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]],
        dtype=np.float64,
    )

    updated = node._blend_midline_samples(
        stored_samples=stored,
        candidate_samples=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert abs(updated[4, 1] - stored[4, 1]) <= node.midline_near_max_shift_m + 1e-9
    assert updated[4, 1] > 0.0
    assert abs(updated[5, 1] - stored[5, 1]) <= node.midline_near_max_shift_m + 1e-9
    assert updated[5, 1] > 0.0


def test_update_midline_buffer_expires_stored_path_when_confidence_drops_below_minimum():
    node = _make_node()
    stored_path = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    node._midline_buffer_path = np.array(stored_path, copy=True)
    node._midline_buffer_confidence = 1.0
    node._midline_buffer_last_update_sec = 10.0
    node.midline_hold_last_valid_duration_s = 2.5
    node.midline_min_buffer_confidence = 0.2
    node.midline_station_spacing_m = 0.5
    node._extract_forward_path_from_pose = lambda path, vehicle_xy, resolution_m: np.array(path, copy=True)
    node._resample_midline_stations = lambda path: np.array(path, copy=True)
    node._is_alias = lambda frame_a, frame_b: frame_a == frame_b

    result = _sample_result()

    for failed_now_sec in np.linspace(10.01, 10.80, 8):
        held = node._update_midline_buffer(
            candidate_centerline=np.empty((0, 2), dtype=np.float64),
            candidate_source="none",
            candidate_update_ok=False,
            frame_id="odom",
            vehicle_x=0.0,
            vehicle_y=0.0,
            vehicle_yaw=0.0,
            result=result,
            now_sec=float(failed_now_sec),
        )
        assert np.allclose(held, stored_path)

    expired = node._update_midline_buffer(
        candidate_centerline=np.empty((0, 2), dtype=np.float64),
        candidate_source="none",
        candidate_update_ok=False,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        now_sec=10.81,
    )

    assert expired.size == 0


def test_candidate_jump_reject_streak_triggers_buffer_reset_and_recovery():
    node = _make_node()
    stored_path = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    node._candidate_jump_reject_streak = 0
    node._midline_buffer_path = np.array(stored_path, copy=True)
    node._midline_buffer_confidence = 1.0
    node._midline_buffer_last_update_sec = 10.0
    node.candidate_jump_recover_frames = 3

    for _ in range(2):
        candidate_update_ok, candidate_update_reason = node._update_candidate_jump_reject_streak(
            candidate_update_ok=False,
            candidate_update_reason="candidate_jump_rejected",
        )
        assert not candidate_update_ok
        assert candidate_update_reason == "candidate_jump_rejected"

    candidate_update_ok, candidate_update_reason = node._update_candidate_jump_reject_streak(
        candidate_update_ok=False,
        candidate_update_reason="candidate_jump_rejected",
    )

    assert candidate_update_ok
    assert candidate_update_reason == "candidate_jump_recovery"
    assert node._candidate_jump_reject_streak == 3
    assert node._midline_buffer_path is None
    assert node._midline_buffer_confidence == 0.0
    assert node._midline_buffer_last_update_sec == -1.0


def test_recovery_directly_replaces_stored_midline_and_preserves_live_pair_shape_near_vehicle():
    node = _make_node()
    candidate = np.array([[0.0, 0.2], [1.0, 0.2], [2.0, 0.2], [3.0, 0.2]], dtype=np.float64)
    node._midline_buffer_path = None
    node._midline_buffer_confidence = 0.0
    node._midline_buffer_last_update_sec = -1.0
    node._extract_forward_path_from_pose = lambda path, vehicle_xy, resolution_m: np.array(path, copy=True)
    node._resample_midline_stations = lambda path: np.array(path, copy=True)
    result = _sample_result()
    result.accepted_pair_count = 3
    result.status = "ok"

    updated = node._update_midline_buffer(
        candidate_centerline=candidate,
        candidate_source="validated",
        candidate_update_ok=True,
        candidate_update_reason="candidate_jump_recovery",
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        now_sec=11.0,
    )
    anchored = node._anchor_centerline_near_vehicle(
        centerline=updated,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        preserve_live_lateral_near_vehicle=True,
    )

    assert np.allclose(updated, candidate)
    assert node._last_midline_update_mode == "recovery"
    assert np.allclose(anchored[0], np.array([0.0, 0.0], dtype=np.float64))
    assert np.allclose(anchored[1:], candidate[1:])


def test_valid_live_midpoint_candidate_blends_into_existing_buffer():
    node = _make_node()
    stored_path = np.array([[0.0, 0.9], [1.0, 0.9], [2.0, 0.9], [3.0, 0.9]], dtype=np.float64)
    candidate = np.array([[0.0, 0.2], [1.0, 0.2], [2.0, 0.2], [3.0, 0.2]], dtype=np.float64)
    node._midline_buffer_path = np.array(stored_path, copy=True)
    node._midline_buffer_confidence = 1.0
    node._midline_buffer_last_update_sec = 10.0
    node._extract_forward_path_from_pose = lambda path, vehicle_xy, resolution_m: np.array(path, copy=True)
    node._resample_midline_stations = lambda path: np.array(path, copy=True)
    result = _sample_result()
    result.accepted_pair_count = 3
    result.status = "ok"

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
    assert np.allclose(updated[0], candidate[0])
    assert abs(updated[1, 1] - stored_path[1, 1]) <= node.midline_near_max_shift_m + 1e-9
    assert updated[1, 1] > candidate[1, 1]


def test_candidate_path_rejects_large_jump_even_while_hold_mode_active():
    node = _make_node()
    node._midline_buffer_path = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    candidate = np.array([[0.0, 2.0], [2.0, 2.0], [4.0, 2.0]], dtype=np.float64)
    result = _sample_result()

    node._hold_mode_active = False
    reject_ok, reject_reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="validated",
    )

    node._hold_mode_active = True
    recover_ok, recover_reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="validated",
    )

    assert not reject_ok
    assert reject_reason == "candidate_jump_rejected"
    assert not recover_ok
    assert recover_reason == "candidate_jump_rejected"


def test_candidate_path_rejects_remembered_pairs_when_core_status_is_not_ok():
    node = _make_node()
    candidate = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    result = _sample_result()
    result.status = "no reliable midpoint chain"
    result.reject_reason = result.status

    candidate_ok, candidate_reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="remembered_pairs",
    )

    assert not candidate_ok
    assert candidate_reason == result.status


def test_candidate_path_rejects_projected_pairs_on_jump_against_stored_midline():
    node = _make_node()
    node._midline_buffer_path = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    candidate = np.array([[0.0, 1.2], [2.0, 1.2], [4.0, 1.2]], dtype=np.float64)
    result = _sample_result()
    result.status = "no reliable midpoint chain"
    result.reject_reason = result.status

    candidate_ok, candidate_reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="projected_pairs",
    )

    assert not candidate_ok
    assert candidate_reason == "candidate_jump_rejected"


def test_select_candidate_centerline_recovers_live_path_after_near_field_reject():
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


def test_select_candidate_centerline_completes_recoverable_live_prefix():
    node = _make_node()
    result = _sample_result()
    result.status = "path forward extent too short"
    result.reject_reason = result.status
    result.centerline = np.empty((0, 2), dtype=np.float64)
    result.prevalidation_centerline = np.array([[0.5, 0.0], [2.7, 0.1]], dtype=np.float64)
    result.midpoints_raw = np.array([[0.5, 0.0], [2.7, 0.1]], dtype=np.float64)

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


def test_select_candidate_centerline_bridges_to_live_pair_midline_when_path_disappears():
    node = _make_node()
    result = _sample_result()
    result.status = "path heading delta exceeded limit"
    result.reject_reason = result.status
    result.centerline = np.empty((0, 2), dtype=np.float64)
    result.prevalidation_centerline = np.empty((0, 2), dtype=np.float64)
    result.accepted_pair_count = 3
    result.midpoints_raw = np.array(
        [[2.0, 1.0], [4.0, 1.0], [6.0, 1.0]],
        dtype=np.float64,
    )

    centerline, source = node._select_candidate_centerline(
        result=result,
        support_chain=result.midpoints_raw,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert source == "pair_midline_bridge"
    assert centerline.shape[0] >= result.midpoints_raw.shape[0]
    assert np.allclose(centerline[0], np.array([0.0, 0.0], dtype=np.float64))
    assert np.allclose(centerline[-1], result.midpoints_raw[-1])


def test_candidate_path_accepts_pair_midline_bridge_from_live_pairs():
    node = _make_node()
    result = _sample_result()
    result.status = "path heading delta exceeded limit"
    result.reject_reason = result.status
    result.centerline = np.empty((0, 2), dtype=np.float64)
    result.prevalidation_centerline = np.empty((0, 2), dtype=np.float64)
    result.accepted_pair_count = 3
    result.midpoints_raw = np.array(
        [[2.0, 1.0], [4.0, 1.0], [6.0, 1.0]],
        dtype=np.float64,
    )
    candidate = node._build_pair_midline_bridge_candidate(
        pair_midline=result.midpoints_raw,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    candidate_ok, candidate_reason = node._candidate_path_is_updateable(
        candidate_centerline=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        result=result,
        candidate_source="pair_midline_bridge",
    )

    assert candidate_ok
    assert candidate_reason == "ok"
