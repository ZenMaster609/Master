from __future__ import annotations

import pathlib
import sys

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.planning.hybrid_boundary_planner_core import HybridBoundaryResult
from sim_car.planning.hybrid_boundary_planner_node import HybridBoundaryPlannerNode


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


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warn_messages: list[str] = []

    def info(self, msg: str) -> None:
        self.info_messages.append(msg)

    def warn(self, msg: str) -> None:
        self.warn_messages.append(msg)


def _make_node() -> HybridBoundaryPlannerNode:
    node = object.__new__(HybridBoundaryPlannerNode)
    node.show_raw_cones = False
    node.show_boundary_chains = True
    node.show_pair_lines = True
    node.show_raw_midpoint_chain = True
    node.show_raw_offset_path = True
    node.show_raw_prevalidation_centerline = True
    node.show_lookahead_point = False
    node.enable_debug_markers = True
    node.publish_points_topic = True
    node.odom_frame = "odom"
    node.base_frame = "front_axle"
    node._path_pub = _FakePublisher()
    node._points_pub = _FakePublisher()
    node._viz_pub = _FakePublisher()
    node._diag_pub = _FakePublisher()
    node.publish_control_debug = False
    node.publish_thesis_context = False
    node._hold_mode_active = False
    node._hold_clean_frame_count = 0
    node._last_operator_state = None
    node._last_operator_reason = None
    node.hold_exit_clean_frames = 2
    node.hold_last_valid_s = 1.25
    node.centerline_path_resolution_m = 0.5
    node.enable_temporal_smoothing = True
    node.smoothing_alpha = 0.25
    node.midline_station_spacing_m = 0.5
    node.midline_control_handoff_distance_m = 1.5
    node.enable_near_field_freeze = False
    node.freeze_near_field_m = 0.0
    node.freeze_blend_length_m = 0.0
    node.enable_committed_near_field = False
    node.commit_plan_horizon_m = 0.0
    node.commit_stable_frames = 1
    node.commit_update_max_churn_ratio = 1.0
    node._committed_centerline = None
    node._commit_stable_frame_count = 0
    node._last_valid_centerline = None
    node._last_valid_raw_midpoint_chain = None
    node._last_valid_pair_segments = None
    node._last_valid_pair_track_ids = None
    node._current_pair_segments_for_viz = None
    node._last_valid_width_m = 3.6
    node._filtered_track_width_m = 3.6
    node._last_valid_time_sec = -1.0
    node._last_throttled_log_sec = {}
    node.log_throttle_s = 0.0
    node._active_planner_mode = "midpoint"
    node._active_remembered_cone_count = 6
    node._active_stale_cone_count = 1
    node._active_left_chain_length = 4
    node._active_right_chain_length = 4
    node._active_pair_count = 4
    node._active_unknown_pair_count = 0
    node._active_filtered_track_width_m = 3.6
    node._active_held_path_flag = 0
    node._fake_logger = _FakeLogger()
    node._pair_memory = []
    node._mode_enter_good_count = 0
    node._mode_exit_bad_count = 0
    node._midline_buffer_path = None
    node._midline_buffer_confidence = 0.0
    node._midline_buffer_last_update_sec = -1.0
    node._last_viz_left_boundary = None
    node._last_viz_right_boundary = None
    node._last_viz_raw_offset_path = None
    node._candidate_jump_reject_streak = 0
    node.candidate_jump_recover_frames = 3
    node.get_clock = lambda: _FakeClock(TimeMsg(sec=123, nanosec=456))
    node.get_logger = lambda: node._fake_logger
    return node


def _sample_result() -> HybridBoundaryResult:
    return HybridBoundaryResult(
        filtered_points=np.array(
            [[2.0, 1.8], [4.0, 1.8], [6.0, 1.8], [2.0, -1.8], [4.0, -1.8], [6.0, -1.8]],
            dtype=np.float64,
        ),
        filtered_colors=["blue", "blue", "blue", "yellow", "yellow", "yellow"],
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.array([[0, 3], [1, 4], [2, 5]], dtype=np.int64),
        selected_pair_track_ids=np.array([[10, 20], [11, 21], [12, 22]], dtype=np.int64),
        midpoints_raw=np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float64),
        centerline=np.array([[2.0, 0.0], [4.0, 0.1], [6.0, 0.3]], dtype=np.float64),
        left_boundary=np.array([[2.0, 1.8], [4.0, 1.8], [6.0, 1.8]], dtype=np.float64),
        right_boundary=np.array([[2.0, -1.8], [4.0, -1.8], [6.0, -1.8]], dtype=np.float64),
        used_fallback=False,
        status="ok",
        candidate_count=5,
        selected_chain_length=3,
        selected_chain_width_median=3.6,
        expected_width_prior_m=3.6,
        near_field_lateral_max_m=0.1,
        near_field_lateral_mean_m=0.05,
        near_field_displacement_max_m=0.12,
        near_field_displacement_mean_m=0.06,
        near_field_kink_max_rad=0.1,
        seed_midpoint_distance_m=2.0,
        reject_counts={},
        planner_mode="midpoint",
        raw_offset_path=np.array([[2.0, 0.2], [4.0, 0.3], [6.0, 0.5]], dtype=np.float64),
        pair_segments=np.array(
            [
                [[2.0, 1.8], [2.0, -1.8]],
                [[4.0, 1.8], [4.0, -1.8]],
                [[6.0, 1.8], [6.0, -1.8]],
            ],
            dtype=np.float64,
        ),
        accepted_pair_count=3,
        left_chain_length=3,
        right_chain_length=3,
        filtered_track_width_m=3.6,
        unknown_pair_count=1,
    )


def _marker_map(marker_array) -> dict[str, object]:
    return {
        marker.ns: marker
        for marker in marker_array.markers
        if getattr(marker, "ns", "")
    }


def _marker_xy(marker) -> np.ndarray:
    return np.asarray([[point.x, point.y] for point in marker.points], dtype=np.float64)


def test_build_markers_include_pairs_offset_and_final_centerline():
    node = _make_node()
    result = _sample_result()
    node._current_pair_segments_for_viz = np.array(result.pair_segments, copy=True)

    markers = node._build_markers(
        now=TimeMsg(sec=9, nanosec=1),
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
    assert "accepted_pairs" in by_ns
    assert "raw_offset_path" in by_ns
    assert "boundary_left" in by_ns
    assert "boundary_right" in by_ns
    assert np.allclose(_marker_xy(by_ns["raw_midpoint_chain"]), result.midpoints_raw)
    assert np.allclose(_marker_xy(by_ns["raw_offset_path"]), result.raw_offset_path)
    assert np.allclose(_marker_xy(by_ns["centerline"]), result.centerline)


def test_build_markers_highlight_active_single_boundary_side():
    node = _make_node()
    result = _sample_result()
    result.planner_mode = "single_boundary"
    result.active_boundary_side = "blue"

    markers = node._build_markers(
        now=TimeMsg(sec=9, nanosec=1),
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
    assert "single_boundary_active" in by_ns
    assert np.allclose(_marker_xy(by_ns["single_boundary_active"]), result.left_boundary)


def test_publish_outputs_status_text_includes_mode_and_width():
    node = _make_node()
    result = _sample_result()
    node._current_pair_segments_for_viz = np.array(result.pair_segments, copy=True)
    node._active_planner_mode = "single_boundary"
    node._active_left_chain_length = 4
    node._active_right_chain_length = 0
    node._active_pair_count = 0
    node._active_unknown_pair_count = 1
    node._active_filtered_track_width_m = 3.7
    node._active_held_path_flag = 1

    node._publish_outputs(
        frame_id="odom",
        centerline=result.centerline,
        raw_centerline=result.centerline,
        raw_midpoint_chain=result.midpoints_raw,
        result=result,
        status="ok",
        control_target_frame=None,
        cmd_speed=1.5,
        cmd_steering=0.1,
        lookahead=3.0,
        operator_state="held",
        operator_reason="holding_previous_valid",
        hold_remaining_s=1.0,
        control_path_point_count=3,
        candidate_diagonal_count=5,
        selected_chain_length=3,
        seed_midpoint_distance_m=2.0,
        near_field_lateral_max_m=0.1,
        near_field_midpoint_kink_max_rad=0.1,
    )

    markers = node._viz_pub.messages[-1]
    status_marker = _marker_map(markers)["status"]
    assert "MODE: SINGLE_BOUNDARY" in status_marker.text
    assert "WIDTH: 3.70 m" in status_marker.text
    assert "HALF: 1.85 m" in status_marker.text
    assert "unknown=1" in status_marker.text
    assert "held=1" in status_marker.text


def test_publish_diagnostics_keeps_delaunay_status_names_and_new_metrics():
    node = _make_node()

    node._publish_diagnostics(
        frame_id="odom",
        centerline_jump_max_m=0.1,
        selected_edge_churn_ratio=0.25,
        tracked_cones_frame_delta_p95_m=0.05,
        centerline_point_count=12,
        selected_edge_count=4,
        status="ok",
        planner_metrics={
            "planner_mode": "single_boundary",
            "left_chain_length": 4,
            "right_chain_length": 0,
            "accepted_pair_count": 0,
            "unknown_pair_count": 1,
            "filtered_track_width_m": 3.7,
            "held_path_flag": 1,
        },
    )

    diag_msg = node._diag_pub.messages[-1]
    assert diag_msg.status[0].name == "delaunay_planner/stability"
    values = {item.key: item.value for item in diag_msg.status[0].values}
    assert values["planner_mode"] == "single_boundary"
    assert values["left_chain_length"] == "4"
    assert values["unknown_pair_count"] == "1"
    assert values["filtered_track_width_m"] == "3.700000"


def test_blend_midline_samples_preserves_fresh_near_field_geometry():
    node = _make_node()
    stored = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
            [2.0, 0.0],
            [2.5, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )
    candidate = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.2],
            [1.0, 0.4],
            [1.5, 0.6],
            [2.0, 0.8],
            [2.5, 1.0],
            [3.0, 1.2],
        ],
        dtype=np.float64,
    )

    blended = node._blend_midline_samples(
        stored_samples=stored,
        candidate_samples=candidate,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert np.allclose(blended[:4], candidate[:4])
    assert float(blended[4, 1]) < float(candidate[4, 1])
