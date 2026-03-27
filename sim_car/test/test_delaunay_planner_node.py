from __future__ import annotations

import pathlib
import sys

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.planning.delaunay_planner_core import CoreResult
from sim_car.planning.delaunay_planner_node import DelaunayPlannerNode
from sim_car.planning.planner_runtime_types import PlannerIdentity


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


def _make_node() -> DelaunayPlannerNode:
    node = object.__new__(DelaunayPlannerNode)
    node._planner_identity = PlannerIdentity(
        node_name='delaunay_planner_node',
        planner_mode='delaunay',
        diagnostics_prefix='delaunay_planner',
        diagnostics_topic='/delaunay_planner/diagnostics',
    )
    node.show_raw_cones = False
    node.show_delaunay_edges = False
    node.show_candidate_edges = False
    node.show_selected_edges = False
    node.show_raw_midpoint_chain = True
    node.show_raw_prevalidation_centerline = True
    node.show_lookahead_point = False
    node.enable_debug_markers = True
    node.publish_points_topic = True
    node.odom_frame = 'odom'
    node.base_frame = 'front_axle'
    node.infer_unknown_by_side = True
    node.infer_orange_by_side = True
    node.orange_min_lateral_m = 0.9
    node.orange_neighbor_radius_m = 3.5
    node.orange_neighbor_margin_m = 0.75
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
    node.smoothing_alpha = 0.3
    node.enable_near_field_freeze = True
    node.freeze_near_field_m = 3.0
    node.freeze_blend_length_m = 1.0
    node.enable_committed_near_field = True
    node.commit_plan_horizon_m = 8.0
    node.commit_stable_frames = 3
    node.commit_update_max_churn_ratio = 0.45
    node._committed_centerline = None
    node._commit_stable_frame_count = 0
    node._last_valid_centerline = None
    node._last_valid_raw_midpoint_chain = None
    node._last_valid_width_m = None
    node._last_valid_time_sec = -1.0
    node._last_throttled_log_sec = {}
    node.log_throttle_s = 0.0
    node._fake_logger = _FakeLogger()
    node.get_clock = lambda: _FakeClock(TimeMsg(sec=123, nanosec=456))
    node.get_logger = lambda: node._fake_logger
    return node


def _sample_result() -> CoreResult:
    return CoreResult(
        filtered_points=np.array([[1.0, 1.0], [1.0, -1.0], [3.0, 1.0], [3.0, -1.0]], dtype=np.float64),
        filtered_colors=['blue', 'yellow', 'blue', 'yellow'],
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.array([[0, 1], [2, 3]], dtype=np.int64),
        midpoints_raw=np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float64),
        centerline=np.array([[1.0, 0.2], [3.0, 0.2], [5.0, 0.2]], dtype=np.float64),
        left_boundary=np.array([[1.0, 1.0], [3.0, 1.0]], dtype=np.float64),
        right_boundary=np.array([[1.0, -1.0], [3.0, -1.0]], dtype=np.float64),
        used_fallback=False,
        status='ok',
    )


def _cone_msg(points: list[tuple[float, float]], *, color: str, boundary_color: str = '') -> ConeDetectionArray:
    msg = ConeDetectionArray()
    msg.header.frame_id = 'odom'
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


def _marker_map(marker_array) -> dict[str, object]:
    return {
        marker.ns: marker
        for marker in marker_array.markers
        if getattr(marker, 'ns', '')
    }


def _marker_xy(marker) -> np.ndarray:
    return np.asarray([[point.x, point.y] for point in marker.points], dtype=np.float64)


def _marker_z(marker) -> np.ndarray:
    return np.asarray([point.z for point in marker.points], dtype=np.float64)


def test_build_markers_include_raw_midpoint_chain_and_raw_prevalidation_centerline():
    node = _make_node()
    result = _sample_result()
    final_centerline = np.array([[0.5, -0.1], [2.5, -0.1], [4.5, -0.1]], dtype=np.float64)

    markers = node._build_markers(
        now=TimeMsg(sec=9, nanosec=1),
        frame_id='odom',
        result=result,
        centerline=final_centerline,
        raw_centerline=result.centerline,
        raw_midpoint_chain=result.midpoints_raw,
        status='ok',
        operator_state='fresh',
        control_target_frame=None,
    )

    by_ns = _marker_map(markers)
    assert np.allclose(_marker_xy(by_ns['raw_midpoint_chain']), result.midpoints_raw)
    assert np.allclose(_marker_xy(by_ns['raw_prevalidation_centerline']), result.centerline)
    assert np.allclose(_marker_xy(by_ns['centerline']), final_centerline)
    assert np.allclose(_marker_z(by_ns['raw_midpoint_chain']), 0.03)
    assert np.allclose(_marker_z(by_ns['raw_prevalidation_centerline']), 0.05)
    assert np.allclose(_marker_z(by_ns['centerline']), 0.07)
    assert by_ns['status'].text.startswith('ok')
    assert abs(by_ns['status'].color.g - 1.0) < 1e-9


def test_build_markers_omit_new_raw_paths_when_disabled():
    node = _make_node()
    node.show_raw_midpoint_chain = False
    node.show_raw_prevalidation_centerline = False
    result = _sample_result()

    markers = node._build_markers(
        now=TimeMsg(sec=9, nanosec=1),
        frame_id='odom',
        result=result,
        centerline=result.centerline,
        raw_centerline=result.centerline,
        raw_midpoint_chain=result.midpoints_raw,
        status='ok',
        operator_state='held',
        control_target_frame=None,
    )

    by_ns = _marker_map(markers)
    assert 'raw_midpoint_chain' not in by_ns
    assert 'raw_prevalidation_centerline' not in by_ns
    assert 'centerline' in by_ns


def test_publish_outputs_keeps_path_topic_on_final_centerline_only():
    node = _make_node()
    result = _sample_result()
    final_centerline = np.array([[0.2, -0.3], [2.2, -0.3], [4.2, -0.3]], dtype=np.float64)

    node._publish_outputs(
        frame_id='odom',
        centerline=final_centerline,
        raw_centerline=result.centerline,
        raw_midpoint_chain=result.midpoints_raw,
        result=result,
        status='held previous valid centerline',
        control_target_frame=None,
        cmd_speed=1.5,
        cmd_steering=0.1,
        lookahead=3.0,
        operator_state='held',
        operator_reason='near_field_continuity',
        hold_remaining_s=1.1,
        control_path_point_count=3,
        candidate_diagonal_count=9,
        selected_chain_length=4,
        seed_midpoint_distance_m=2.3,
        near_field_lateral_max_m=0.12,
        near_field_midpoint_kink_max_rad=0.35,
    )

    path_msg = node._path_pub.messages[-1]
    points_msg = node._points_pub.messages[-1]
    markers = node._viz_pub.messages[-1]
    published_xy = np.asarray(
        [[pose.pose.position.x, pose.pose.position.y] for pose in path_msg.poses],
        dtype=np.float64,
    )
    points_xy = np.asarray(
        [[pose.position.x, pose.position.y] for pose in points_msg.poses],
        dtype=np.float64,
    )
    by_ns = _marker_map(markers)

    assert np.allclose(published_xy, final_centerline)
    assert np.allclose(points_xy, final_centerline)
    assert np.allclose(_marker_xy(by_ns['centerline']), final_centerline)
    assert np.allclose(_marker_xy(by_ns['raw_prevalidation_centerline']), result.centerline)
    assert np.allclose(_marker_xy(by_ns['raw_midpoint_chain']), result.midpoints_raw)
    assert 'STATE: HELD' in by_ns['status'].text
    assert 'REASON: near-field continuity rejected fresh chain' in by_ns['status'].text
    assert abs(by_ns['status'].color.r - 1.0) < 1e-9
    assert abs(by_ns['status'].color.g - 0.9) < 1e-9


def test_publish_diagnostics_includes_planner_metrics():
    node = _make_node()

    node._publish_diagnostics(
        frame_id='odom',
        centerline_jump_max_m=0.1,
        selected_edge_churn_ratio=0.25,
        tracked_cones_frame_delta_p95_m=0.05,
        centerline_point_count=12,
        selected_edge_count=4,
        status='ok',
        planner_metrics={
            'candidate_diagonal_count': 9,
            'publish_mode': 'held',
            'near_field_lateral_max_m': 0.12,
            'planner_state_code': 2,
            'operator_reason_code': 5,
            'hold_remaining_s': 1.2,
            'control_path_point_count': 4,
            'zero_cmd_sent_flag': 1,
        },
    )

    diag_array = node._diag_pub.messages[-1]
    diag = diag_array.status[0]
    values = {item.key: item.value for item in diag.values}

    assert values['candidate_diagonal_count'] == '9'
    assert values['publish_mode'] == 'held'
    assert values['near_field_lateral_max_m'] == '0.120000'
    assert values['planner_state_code'] == '2'
    assert values['operator_reason_code'] == '5'
    assert values['hold_remaining_s'] == '1.200000'
    assert values['control_path_point_count'] == '4'
    assert values['zero_cmd_sent_flag'] == '1'


def test_convert_cones_to_frame_matches_boundary_hint_and_direct_fallback():
    node = _make_node()
    points = [(2.0, 1.8), (2.0, -1.8), (4.0, 1.8), (4.0, -1.8)]

    direct_msg = _cone_msg(points, color='orange')
    hinted_msg = ConeDetectionArray()
    hinted_msg.header = direct_msg.header
    expected_colors = ['blue', 'yellow', 'blue', 'yellow']
    for (x, y), boundary_color in zip(points, expected_colors):
        cone = ConeDetection()
        cone.color = 'orange'
        cone.boundary_color = boundary_color
        cone.confidence = 0.9
        cone.position.x = float(x)
        cone.position.y = float(y)
        cone.position.z = 0.0
        hinted_msg.cones.append(cone)

    direct_points, direct_colors, direct_conf = node._convert_cones_to_frame(
        direct_msg,
        'odom',
        'odom',
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
    )
    hinted_points, hinted_colors, hinted_conf = node._convert_cones_to_frame(
        hinted_msg,
        'odom',
        'odom',
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
    )

    assert np.allclose(direct_points, hinted_points)
    assert direct_colors == expected_colors
    assert hinted_colors == expected_colors
    assert np.allclose(direct_conf, hinted_conf)


def test_near_field_freeze_splices_previous_prefix_into_fresh_path():
    node = _make_node()
    node._last_valid_centerline = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        dtype=np.float64,
    )
    current = np.array(
        [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]],
        dtype=np.float64,
    )

    spliced, applied, committed = node._apply_near_field_freeze(
        current,
        frame_id='odom',
        vehicle_x=0.0,
        vehicle_y=0.0,
    )

    assert applied
    assert not committed
    assert np.allclose(spliced[:3, 1], 0.0)
    assert float(spliced[-1, 1]) > 0.9


def test_committed_near_field_is_preferred_over_last_valid_when_active():
    node = _make_node()
    node._last_valid_centerline = np.array(
        [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0]],
        dtype=np.float64,
    )
    node._committed_centerline = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        dtype=np.float64,
    )
    current = np.array(
        [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]],
        dtype=np.float64,
    )

    spliced, applied, committed = node._apply_near_field_freeze(
        current,
        frame_id='odom',
        vehicle_x=0.5,
        vehicle_y=0.0,
    )

    assert applied
    assert committed
    assert np.allclose(spliced[:2, 1], 0.0)


def test_committed_near_field_updates_only_after_stable_frames():
    node = _make_node()
    current = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float64,
    )

    node._update_committed_near_field(
        centerline=current,
        selected_edge_churn=1.0,
        selected_chain_length=3,
        previous_edge_keys=set(),
    )
    assert node._committed_centerline is None

    for _ in range(3):
        node._update_committed_near_field(
            centerline=current,
            selected_edge_churn=0.2,
            selected_chain_length=3,
            previous_edge_keys={(0, 0, 1, 1)},
        )

    assert node._committed_centerline is not None
    assert np.allclose(node._committed_centerline, current)


def test_record_valid_plan_tracks_raw_midpoints_and_width():
    node = _make_node()
    centerline = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.1]], dtype=np.float64)
    raw_midpoints = np.array([[1.0, 0.0], [2.0, 0.05]], dtype=np.float64)

    node._record_valid_plan(
        now_sec=10.0,
        centerline=centerline,
        raw_midpoint_chain=raw_midpoints,
        selected_chain_width_median=4.3,
    )

    assert np.allclose(node._last_valid_centerline, centerline)
    assert np.allclose(node._last_valid_raw_midpoint_chain, raw_midpoints)
    assert abs(node._last_valid_width_m - 4.3) < 1e-6
    assert node._last_valid_time_sec == 10.0


def test_hold_hysteresis_requires_clean_reentry_frames():
    node = _make_node()

    assert node._advance_hold_hysteresis(plan_ok=False)
    assert node._hold_mode_active
    assert node._hold_clean_frame_count == 0

    assert node._advance_hold_hysteresis(plan_ok=True)
    assert node._hold_mode_active
    assert node._hold_clean_frame_count == 1

    assert not node._advance_hold_hysteresis(plan_ok=True)
    assert not node._hold_mode_active
    assert node._hold_clean_frame_count == 0


def test_held_centerline_only_uses_timeout_not_legacy_history():
    node = _make_node()
    centerline = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)

    node._record_valid_plan(
        now_sec=10.0,
        centerline=centerline,
        raw_midpoint_chain=centerline,
        selected_chain_width_median=4.0,
    )

    held = node._held_centerline(now_sec=10.5)
    expired = node._held_centerline(now_sec=11.5 + node.hold_last_valid_s)

    assert np.allclose(held, centerline)
    assert expired is None


def test_operator_status_text_is_readable():
    node = _make_node()

    text = node._build_operator_status_text(
        operator_state='stopped',
        operator_reason='hold_expired_no_path',
        centerline_point_count=0,
        cmd_speed=0.0,
        cmd_steering=0.0,
        lookahead=0.0,
        candidate_diagonal_count=6,
        selected_chain_length=0,
        seed_midpoint_distance_m=float('nan'),
        near_field_lateral_max_m=0.42,
        near_field_midpoint_kink_max_rad=0.91,
        hold_remaining_s=0.0,
    )

    assert 'STATE: STOPPED' in text
    assert 'REASON: held path expired and no fresh path is available' in text
    assert 'PATH: 0 pts | chain=0 | cand=6' in text


def test_log_operator_state_transition_only_logs_on_change():
    node = _make_node()

    node._log_operator_state_transition(
        operator_state='held',
        operator_reason='near_field_continuity',
        hold_remaining_s=1.1,
        selected_chain_length=4,
    )
    node._log_operator_state_transition(
        operator_state='held',
        operator_reason='near_field_continuity',
        hold_remaining_s=1.0,
        selected_chain_length=4,
    )
    node._log_operator_state_transition(
        operator_state='fresh',
        operator_reason='none',
        hold_remaining_s=1.0,
        selected_chain_length=5,
    )

    assert len(node._fake_logger.info_messages) == 2
    assert 'planner state none -> held | reason=near_field_continuity' in node._fake_logger.info_messages[0]
    assert 'planner state held -> fresh | reason=none' in node._fake_logger.info_messages[1]
