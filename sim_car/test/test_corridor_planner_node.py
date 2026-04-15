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
    from sim_car.planning.corridor_planner_core import CorridorPlannerResult  # noqa: E402
    from sim_car.planning.corridor_planner_node import MSG_TRACK_STATE_CONFIRMED  # noqa: E402
    from sim_car.planning.corridor_planner_node import MSG_TRACK_STATE_STALE  # noqa: E402
    from sim_car.planning.corridor_planner_node import MSG_TRACK_STATE_TENTATIVE  # noqa: E402
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
    node.show_corridor_pair_audit = True
    node.corridor_pair_audit_show_labels = True
    node.corridor_pair_audit_max_labels = 80
    node.enable_cone_audit_markers = True
    node.cone_audit_show_labels = True
    node.cone_audit_max_labels = 80
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
        max_cone_range_m=25.0,
        planning_horizon_m=25.0,
        max_lateral_range_m=8.0,
        behind_drop_m=5.0,
        min_confidence=0.3,
        path_resolution_m=0.5,
        max_path_length_m=20.0,
        min_forward_extent_m=2.0,
        min_required_corridor_samples=3,
    )
    node.lap_tracking_target_laps = 0
    node._filtered_track_width_m = 3.6
    node._is_alias = lambda frame_a, frame_b: frame_a == frame_b
    node.get_clock = lambda: _FakeClock(TimeMsg(sec=1, nanosec=0))
    return node


def _cone(
    *,
    track_id: int,
    color: str = "blue",
    boundary_color: str = "blue",
    confidence: float = 0.9,
    track_confidence: float = 0.9,
    track_state: int = MSG_TRACK_STATE_CONFIRMED,
    missed_count: int = 0,
    last_seen_sec: int = 1,
) -> ConeDetection:
    cone = ConeDetection()
    cone.color = color
    cone.boundary_color = boundary_color
    cone.confidence = confidence
    cone.track_id = track_id
    cone.track_state = track_state
    cone.track_confidence = track_confidence
    cone.color_confidence = 0.8
    cone.missed_count = missed_count
    cone.last_seen = TimeMsg(sec=last_seen_sec, nanosec=0)
    return cone


def _sample_result() -> CorridorPlannerResult:
    left = np.array([[2.0, 1.8], [4.0, 1.8], [6.0, 1.8]], dtype=np.float64)
    right = np.array([[2.0, -1.8], [4.0, -1.8], [6.0, -1.8]], dtype=np.float64)
    raw_left = np.array([[1.8, 1.7], [4.1, 1.9], [6.3, 1.8]], dtype=np.float64)
    raw_right = np.array([[1.9, -1.7], [4.2, -1.9], [6.4, -1.8]], dtype=np.float64)
    anchors = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float64)
    rungs = np.empty((3, 2, 2), dtype=np.float64)
    rungs[:, 0, :] = left
    rungs[:, 1, :] = right
    rejected_wide = np.array([[[8.0, 2.0], [8.0, -4.8]]], dtype=np.float64)
    audit_segments = np.concatenate((rungs, rejected_wide), axis=0)
    audit_anchors = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, -1.4]], dtype=np.float64)
    audit_widths = np.array([3.6, 3.6, 3.6, 6.8], dtype=np.float64)
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
        raw_left_chain_points=raw_left,
        raw_right_chain_points=raw_right,
        corridor_pair_audit_segments=audit_segments,
        corridor_pair_audit_anchors_local=audit_anchors,
        corridor_pair_audit_widths_m=audit_widths,
        corridor_pair_audit_reasons=[
            "pair_valid",
            "pair_valid",
            "pair_valid",
            "pair_width_too_wide",
        ],
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


def test_cone_audit_classifies_all_rejection_and_usage_buckets():
    node = _make_node()
    result = _sample_result()
    result.used_left_track_ids = np.asarray([1], dtype=np.int64)
    result.used_right_track_ids = np.asarray([2], dtype=np.int64)
    result.chain_rejection_reasons_by_track_id = {3: "chain_step_too_far"}

    msg = ConeDetectionArray()
    msg.header.frame_id = "odom"
    msg.header.stamp = TimeMsg(sec=2, nanosec=0)
    cones = [
        _cone(track_id=1, color="blue", boundary_color="blue"),
        _cone(track_id=2, color="yellow", boundary_color="yellow"),
        _cone(track_id=3, color="blue", boundary_color="blue"),
        _cone(track_id=4, color="blue", boundary_color="blue"),
        _cone(track_id=5, color="blue", boundary_color="blue"),
        _cone(track_id=6, color="blue", boundary_color="blue"),
        _cone(track_id=7, color="blue", boundary_color="blue", track_confidence=0.1),
        _cone(
            track_id=8,
            color="blue",
            boundary_color="blue",
            track_state=MSG_TRACK_STATE_TENTATIVE,
        ),
        _cone(track_id=9, color="unknown", boundary_color="", track_confidence=0.9),
        _cone(track_id=10, color="blue", boundary_color="blue"),
        _cone(track_id=11, color="blue", boundary_color="blue"),
    ]
    msg.cones.extend(cones)
    points = np.asarray(
        [
            [2.0, 1.8],
            [2.0, -1.8],
            [3.0, 1.8],
            [-6.0, 0.0],
            [26.0, 0.0],
            [5.0, 9.0],
            [4.0, 1.8],
            [5.0, 1.8],
            [6.0, 0.0],
            [float("nan"), 0.0],
            [24.0, 8.0],
        ],
        dtype=np.float64,
    )
    colors = [
        "blue",
        "yellow",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "unknown",
        "blue",
        "blue",
    ]
    planning_frame = node._tracked_cone_planning_frame(
        msg=msg,
        points_xy=points,
        colors=colors,
        confidences=np.full((len(cones),), 0.9, dtype=np.float64),
    )

    entries = node._build_cone_audit_entries(
        msg=msg,
        planning_frame=planning_frame,
        result=result,
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=2.0,
    )
    reasons = {entry.track_id: entry.reason for entry in entries}

    assert reasons[1] == "used_left_chain"
    assert reasons[2] == "used_right_chain"
    assert reasons[3] == "chain_step_too_far"
    assert reasons[4] == "rejected_geometry_behind"
    assert reasons[5] == "rejected_geometry_horizon"
    assert reasons[6] == "rejected_geometry_lateral"
    assert reasons[7] == "rejected_confidence"
    assert reasons[8] == "rejected_tentative"
    assert reasons[9] == "rejected_color"
    assert reasons[10] == "rejected_nonfinite"
    assert reasons[11] == "rejected_geometry_range"

    counts = node._cone_audit_counts(entries)
    assert counts["cone_audit_received_count"] == len(cones)
    assert counts["cone_audit_used_left_count"] == 1
    assert counts["cone_audit_used_right_count"] == 1
    assert counts["cone_audit_chain_step_too_far_count"] == 1


def test_cone_audit_markers_include_namespaces_labels_and_stale_halo():
    node = _make_node()
    used = SimpleNamespace(
        track_id=1,
        reason="used_left_chain",
        point_xy=np.asarray([1.0, 1.0], dtype=np.float64),
        local_x_m=1.0,
        local_y_m=1.0,
        raw_color="blue",
        resolved_color="blue",
        track_state=MSG_TRACK_STATE_CONFIRMED,
        confidence=0.9,
        track_confidence=0.9,
        color_confidence=0.8,
        missed_count=0,
        last_seen_age_sec=0.1,
        memory_only=False,
    )
    stale_rejected = SimpleNamespace(
        track_id=2,
        reason="rejected_geometry_lateral",
        point_xy=np.asarray([1.0, 9.0], dtype=np.float64),
        local_x_m=1.0,
        local_y_m=9.0,
        raw_color="yellow",
        resolved_color="yellow",
        track_state=MSG_TRACK_STATE_STALE,
        confidence=0.9,
        track_confidence=0.9,
        color_confidence=0.8,
        missed_count=2,
        last_seen_age_sec=0.7,
        memory_only=True,
    )
    chain_rejected = SimpleNamespace(
        track_id=3,
        reason="chain_heading_change",
        point_xy=np.asarray([3.0, 1.5], dtype=np.float64),
        local_x_m=3.0,
        local_y_m=1.5,
        raw_color="blue",
        resolved_color="blue",
        track_state=MSG_TRACK_STATE_CONFIRMED,
        confidence=0.9,
        track_confidence=0.9,
        color_confidence=0.8,
        missed_count=0,
        last_seen_age_sec=0.1,
        memory_only=False,
    )

    markers = node._build_cone_audit_markers(
        frame_id="odom",
        stamp=TimeMsg(sec=2, nanosec=0),
        entries=[used, stale_rejected, chain_rejected],
    )
    by_ns = _marker_map(markers)

    assert "cone_audit_used_left_chain" in by_ns
    assert "cone_audit_chain_rejected" in by_ns
    assert "cone_audit_rejected_geometry" in by_ns
    assert "cone_audit_stale_halo" in by_ns
    assert "cone_audit_labels" in by_ns
    label_text = "\n".join(marker.text for marker in markers.markers if marker.ns == "cone_audit_labels")
    assert "id=1" in label_text
    assert "used_left_chain" in label_text
    assert "chain_heading_change" in label_text
    assert "rejected_geometry_lateral" in label_text


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


def test_tracked_cone_planning_frame_rejects_tentative_cones_for_corridor():
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
    assert "raw_chain_left" in by_ns
    assert "raw_chain_left_points" in by_ns
    assert "raw_chain_right" in by_ns
    assert "raw_chain_right_points" in by_ns
    assert "corridor_pair_audit_pair_width_too_wide" in by_ns
    assert "corridor_pair_audit_labels" in by_ns
    assert np.allclose(_marker_xy(by_ns["raw_chain_left_points"]), result.raw_left_chain_points)
    assert np.allclose(_marker_xy(by_ns["raw_chain_right_points"]), result.raw_right_chain_points)
    label_text = "\n".join(
        marker.text
        for marker in markers.markers
        if marker.ns == "corridor_pair_audit_labels"
    )
    assert "pair_width_too_wide" in label_text
    assert "w=6.80m" in label_text


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


def test_lap_status_text_uses_configured_target_laps():
    node = _make_node()

    assert node._lap_status_text() == "LAPS: 0/off"

    node.lap_tracking_target_laps = 1

    assert node._lap_status_text() == "LAPS: 0/1"


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


def test_remembered_corridor_geometry_does_not_supply_memory_candidate():
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
    result.status = "no reliable corridor boundaries"
    result.reject_reason = result.status
    result.centerline = np.empty((0, 2), dtype=np.float64)
    result.prevalidation_centerline = np.empty((0, 2), dtype=np.float64)
    candidate, source = node._select_candidate_centerline(
        result=result,
        support_chain=midpoint_chain,
        memory_midpoint_chain=midpoint_chain,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert pair_segments.shape[0] == 3
    assert midpoint_chain.shape[0] == 3
    assert source == "none"
    assert candidate.shape == (0, 2)


def test_valid_live_corridor_candidate_directly_replaces_existing_buffer():
    node = _make_node()
    node.midline_min_estimated_extent_m = 3.0
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

    assert node._last_midline_update_mode == "direct"
    assert np.allclose(updated[:, 1], 0.2)
    assert np.allclose(updated[[0, -1]], candidate[[0, -1]])


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
    assert reason == "no reliable corridor boundaries"


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


def test_select_candidate_centerline_does_not_recover_near_field_corridor_jump():
    node = _make_node()
    result = _sample_result()
    result.status = "near-field continuity rejected fresh path"
    result.reject_reason = result.status

    centerline, source = node._select_candidate_centerline(
        result=result,
        support_chain=result.midpoints_raw,
        memory_midpoint_chain=result.midpoints_raw,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert source == "none"
    assert centerline.shape == (0, 2)


def test_select_candidate_centerline_rejects_short_corridor_prefix():
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
        memory_midpoint_chain=result.midpoints_raw,
        frame_id="odom",
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
    )

    assert source == "none"
    assert centerline.shape == (0, 2)
