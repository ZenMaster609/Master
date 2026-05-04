from __future__ import annotations

import math
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time as TimeMsg
from nav_msgs.msg import Odometry

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance  # noqa: E402
    from sim_car.controllers.base import ControllerOutput  # noqa: E402
    from sim_car.controllers.pure_pursuit_controller import PurePursuitController  # noqa: E402
    from sim_car.planning.ground_truth_midline import (  # noqa: E402
        build_forward_path_from_loop,
        build_gt_midline_from_cones,
    )
    from sim_car.planning.linetest_planner_node import LineTestPlannerNode  # noqa: E402
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


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warn_messages: list[str] = []

    def info(self, msg: str) -> None:
        self.info_messages.append(msg)

    def warn(self, msg: str) -> None:
        self.warn_messages.append(msg)


class _CapturingController:
    def __init__(self) -> None:
        self.control_paths: list[np.ndarray] = []

    def compute(
        self,
        *,
        control_path: np.ndarray,
        speed_mps: float,
        yaw_rate_rps: float,
    ) -> ControllerOutput:
        del speed_mps
        del yaw_rate_rps
        captured = np.asarray(control_path, dtype=np.float64)
        self.control_paths.append(np.array(captured, copy=True))
        target = captured[min(1, captured.shape[0] - 1)]
        return ControllerOutput(
            steering_rad=0.0,
            kappa=0.0,
            lookahead_m=float(np.hypot(target[0], target[1])),
            target_point_base=np.asarray(target, dtype=np.float64),
        )


def _make_node() -> LineTestPlannerNode:
    node = object.__new__(LineTestPlannerNode)
    node._planner_identity = PlannerIdentity(
        node_name='linetest_planner_node',
        planner_mode='linetest',
        diagnostics_prefix='linetest_planner',
        diagnostics_topic='/linetest_planner/diagnostics',
    )
    node._cmd_pub = _FakePublisher()
    node._brake_pub = _FakePublisher()
    node._path_pub = _FakePublisher()
    node._points_pub = _FakePublisher()
    node._viz_pub = _FakePublisher()
    node._diag_pub = _FakePublisher()
    node.enable_debug_markers = False
    node.publish_points_topic = False
    node.show_lookahead_point = False
    node.publish_control_debug = False
    node.controller_type = 'none'
    node._controller = None
    node.stop_if_no_path = True
    node.odom_frame = 'odom'
    node.base_frame = 'front_axle'
    node.planning_frame = 'odom'
    node.cmd_topic = '/cmd'
    node.brake_cmd_topic = '/sim/brake_cmd'
    node.centerline_topic = '/planned_centerline'
    node.viz_topic = '/planner_viz'
    node.points_topic = '/planned_centerline_points'
    node.odom_topic = '/sim/odom'
    node.tracked_cones_topic = '/tracked_cones'
    node.gt_track_topic = '/ground_truth/track'
    node.publish_rate_hz = 60.0
    node.log_throttle_s = 0.0
    node.path_source = 'fixed_line'
    node.gt_midline_resolution_m = 0.5
    node.gt_control_horizon_m = 30.0
    node.odom_lag_compensation_s = 0.0
    node.speed_min_mps = 1.0
    node.speed_max_mps = 4.17
    node.curvature_speed_gain = 4.0
    node.lowpass_speed_alpha = 0.15
    node.brake_activation_distance_m = 1.0
    node.brake_command = 1.0
    node.line_start_x_m = -38.5
    node.line_start_y_m = 0.0
    node.line_end_x_m = 50.0
    node.line_end_y_m = 0.0
    node.line_point_spacing_m = 0.5
    node._line_start_xy = np.array([node.line_start_x_m, node.line_start_y_m], dtype=np.float64)
    node._line_end_xy = np.array([node.line_end_x_m, node.line_end_y_m], dtype=np.float64)
    node._line_direction_xy = np.array([1.0, 0.0], dtype=np.float64)
    node._line_length_m = float(node.line_end_x_m - node.line_start_x_m)
    node._full_centerline = LineTestPlannerNode._generate_line_path(
        node._line_start_xy,
        node._line_end_xy,
        node.line_point_spacing_m,
    )
    node._latest_odom_msg = None
    node._latest_gt_track_msg = None
    node._gt_midline_source = None
    node._gt_anchor_xy = None
    node._gt_anchor_heading_xy = None
    node._gt_track_sub = None
    node.lap_tracking_target_laps = 0
    node._latest_cones_msg = None
    node._lap_tracking_completed_laps = 0
    node._lap_tracking_armed = True
    node._latest_speed_mps = 0.0
    node._latest_yaw_rate_rps = 0.0
    node._last_speed_cmd = None
    node._last_steering_cmd = None
    node._last_throttled_log_sec = {}
    node._configured_wheelbase_m = lambda: 1.65
    node._fake_logger = _FakeLogger()
    node.get_clock = lambda: _FakeClock(TimeMsg(sec=1, nanosec=0))
    node.get_logger = lambda: node._fake_logger
    return node


def _odom_msg(
    *,
    x: float,
    y: float,
    yaw: float,
    child_frame_id: str = 'front_axle',
    vx: float = 0.0,
    vy: float = 0.0,
    yaw_rate: float = 0.0,
) -> Odometry:
    msg = Odometry()
    msg.header.frame_id = 'odom'
    msg.child_frame_id = child_frame_id
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.orientation.z = math.sin(0.5 * yaw)
    msg.pose.pose.orientation.w = math.cos(0.5 * yaw)
    msg.twist.twist.linear.x = float(vx)
    msg.twist.twist.linear.y = float(vy)
    msg.twist.twist.angular.z = float(yaw_rate)
    return msg


def _cone(x: float, y: float) -> ConeWithCovariance:
    cone = ConeWithCovariance()
    cone.point.x = float(x)
    cone.point.y = float(y)
    cone.point.z = 0.0
    return cone


def _gt_track_msg(blue_xy: np.ndarray, yellow_xy: np.ndarray) -> ConeArrayWithCovariance:
    msg = ConeArrayWithCovariance()
    msg.header.frame_id = 'map'
    msg.blue_cones = [_cone(float(x), float(y)) for x, y in blue_xy]
    msg.yellow_cones = [_cone(float(x), float(y)) for x, y in yellow_xy]
    return msg


def test_gt_midline_from_unordered_cones_stays_centered():
    blue = np.asarray([[15.0, 1.5], [0.0, 1.5], [10.0, 1.5], [5.0, 1.5]], dtype=np.float64)
    yellow = np.asarray([[10.0, -1.5], [0.0, -1.5], [15.0, -1.5], [5.0, -1.5]], dtype=np.float64)

    midline = build_gt_midline_from_cones(
        blue_xy=blue,
        yellow_xy=yellow,
        start_xy=np.asarray([0.0, 0.0], dtype=np.float64),
        heading_xy=np.asarray([1.0, 0.0], dtype=np.float64),
        frame_id='odom',
        resolution_m=0.5,
    )

    assert midline.midline_xy.shape[0] >= 10
    assert np.allclose(midline.midline_xy[:, 1], 0.0, atol=1e-6)
    assert np.max(midline.midline_xy[:, 0]) >= 14.5


def test_gt_midline_loop_like_cones_produce_closed_path():
    angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    blue = np.column_stack((6.5 * np.cos(angles), 6.5 * np.sin(angles)))
    yellow = np.column_stack((3.5 * np.cos(angles), 3.5 * np.sin(angles)))

    midline = build_gt_midline_from_cones(
        blue_xy=blue,
        yellow_xy=yellow,
        start_xy=np.asarray([5.0, 0.0], dtype=np.float64),
        heading_xy=np.asarray([0.0, 1.0], dtype=np.float64),
        frame_id='odom',
        resolution_m=0.5,
    )

    assert midline.midline_xy.shape[0] >= 40
    assert np.hypot(*(midline.midline_xy[0] - midline.midline_xy[-1])) <= 1e-6


def test_forward_loop_sampling_wraps_after_projected_end_segment():
    loop_path = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
        dtype=np.float64,
    )

    forward = build_forward_path_from_loop(
        path_xy=loop_path,
        vehicle_xy=np.asarray([0.0, 0.1], dtype=np.float64),
        resolution_m=0.5,
        horizon_m=1.5,
    )

    assert forward.shape[0] >= 3
    assert np.any(forward[:, 0] > 0.4)
    assert np.min(forward[:, 1]) <= 1e-6


def test_generate_line_path_spans_full_configured_segment():
    path = LineTestPlannerNode._generate_line_path(
        np.array([-38.5, 0.0], dtype=np.float64),
        np.array([50.0, 0.0], dtype=np.float64),
        0.5,
    )

    assert np.allclose(path[0], np.array([-38.5, 0.0], dtype=np.float64))
    assert np.allclose(path[-1], np.array([50.0, 0.0], dtype=np.float64))
    assert np.allclose(path[:, 1], 0.0)
    assert np.all(np.diff(path[:, 0]) > 0.0)


def test_build_steering_controller_creates_active_pure_pursuit_controller():
    params = {
        'stanley.k_gain': 1.2,
        'stanley.softening_speed_mps': 0.0,
        'stanley.heading_gain': 1.6,
        'stanley.lookahead_idx_offset': 0,
        'stanley.steering_limit_rad': 0.52,
        'stanley.steering_lowpass_alpha': 1.0,
        'stanley.steering_rate_limit_rad_s': 10.0,
        'stanley.use_yaw_rate_damping': True,
        'stanley.yaw_rate_damping_gain': 0.0,
        'stanley.wheelbase_m': 1.65,
        'stanley.cross_track_deadband_m': 0.0,
        'pure_pursuit.lookahead_m': 2.0,
        'pure_pursuit.min_lookahead_m': 1.0,
        'pure_pursuit.max_lookahead_m': 5.0,
        'pure_pursuit.lookahead_gain': 0.5,
        'pure_pursuit.steering_limit_rad': 0.52,
        'pure_pursuit.steering_lowpass_alpha': 1.0,
        'pure_pursuit.steering_rate_limit_rad_s': 0.0,
        'pure_pursuit.wheelbase_m': 1.65,
    }

    node = object.__new__(LineTestPlannerNode)
    node.controller_type = 'pure_pursuit'
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


def test_publish_outputs_emits_straight_path_with_aligned_yaw():
    node = _make_node()
    centerline = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)

    node._publish_outputs(
        frame_id='odom',
        centerline=centerline,
        status_text='ok',
        operator_state='fresh',
        control_target_world=None,
        vehicle_pose=(0.0, 0.0, 0.0),
    )

    path_msg = node._path_pub.messages[-1]
    xy = np.asarray([[pose.pose.position.x, pose.pose.position.y] for pose in path_msg.poses], dtype=np.float64)

    assert np.allclose(xy, centerline)
    assert all(abs(pose.pose.orientation.z) <= 1e-9 for pose in path_msg.poses)
    assert all(abs(pose.pose.orientation.w - 1.0) <= 1e-9 for pose in path_msg.poses)


def test_resolve_vehicle_pose_converts_body_center_odom_to_front_axle():
    node = _make_node()
    node._latest_odom_msg = _odom_msg(x=-38.5, y=0.0, yaw=0.0, child_frame_id='base_footprint')

    pose = node._resolve_vehicle_pose()

    assert pose is not None
    assert abs(pose[0] - (-37.675)) < 1e-9
    assert abs(pose[1]) < 1e-9
    assert abs(pose[2]) < 1e-9


@pytest.mark.parametrize('controller_type', ['stanley', 'pure_pursuit'])
def test_odom_lag_compensation_shifts_linetest_control_path_for_both_controllers(controller_type: str):
    node = _make_node()
    node.controller_type = controller_type
    node._controller = _CapturingController()
    node._latest_odom_msg = _odom_msg(x=-38.5, y=0.0, yaw=0.0)
    node._latest_speed_mps = 2.0
    node._latest_yaw_rate_rps = 0.0
    node.odom_lag_compensation_s = 0.04

    node._on_timer()

    captured_path = node._controller.control_paths[-1]
    assert captured_path.shape[0] >= 2
    assert captured_path[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert captured_path[1, 0] == pytest.approx(0.42, abs=1e-9)


def test_linetest_brakes_near_configured_line_end() -> None:
    node = _make_node()
    node.controller_type = 'stanley'
    node._controller = _CapturingController()
    node._latest_odom_msg = _odom_msg(x=49.2, y=0.0, yaw=0.0, vx=4.0)
    node._latest_speed_mps = 4.0
    node._latest_yaw_rate_rps = 0.0

    node._on_timer()

    assert node._cmd_pub.messages[-1].drive.speed == pytest.approx(0.0)
    assert node._brake_pub.messages[-1].data == pytest.approx(1.0)


def test_gt_linetest_waits_without_gt_midline_and_sends_zero_command() -> None:
    node = _make_node()
    node.path_source = 'ground_truth_midline'
    node.controller_type = 'stanley'
    node._controller = _CapturingController()
    node._latest_odom_msg = _odom_msg(x=5.0, y=0.0, yaw=0.0, vx=1.0)
    node._latest_speed_mps = 1.0

    node._on_timer()

    assert node._cmd_pub.messages[-1].drive.speed == pytest.approx(0.0)
    assert len(node._path_pub.messages[-1].poses) == 0
    values = {item.key: item.value for item in node._diag_pub.messages[-1].status[0].values}
    assert values['operator_reason_code'] == str(LineTestPlannerNode._operator_reason_code('missing_gt_midline'))


def test_gt_linetest_publishes_midline_and_tracks_forward_loop_without_end_brake() -> None:
    node = _make_node()
    node.path_source = 'ground_truth_midline'
    node.controller_type = 'stanley'
    node._controller = _CapturingController()
    node.brake_activation_distance_m = 999.0
    node._latest_odom_msg = _odom_msg(x=5.0, y=0.0, yaw=math.pi / 2.0, vx=2.0)
    node._latest_speed_mps = 2.0
    angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    node._gt_track_cb(
        _gt_track_msg(
            blue_xy=np.column_stack((6.5 * np.cos(angles), 6.5 * np.sin(angles))),
            yellow_xy=np.column_stack((3.5 * np.cos(angles), 3.5 * np.sin(angles))),
        )
    )

    node._on_timer()

    assert len(node._path_pub.messages[-1].poses) >= 40
    captured_path = node._controller.control_paths[-1]
    assert captured_path.shape[0] >= 3
    assert captured_path[0, 0] == pytest.approx(0.0, abs=0.15)
    assert node._brake_pub.messages[-1].data == pytest.approx(0.0)


def test_lap_status_text_uses_completed_laps() -> None:
    node = _make_node()

    assert node._lap_status_text() == 'LAPS: 0/off'

    node.lap_tracking_target_laps = 10
    node._lap_tracking_completed_laps = 3

    assert node._lap_status_text() == 'LAPS: 3/10'


def test_publish_diagnostics_uses_linetest_identity_and_expected_keys():
    node = _make_node()
    centerline = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)

    node._publish_diagnostics(
        frame_id='odom',
        centerline=centerline,
        control_path_point_count=3,
        operator_state='fresh',
        operator_reason='none',
        zero_cmd_sent_flag=0,
    )

    msg = node._diag_pub.messages[-1]
    assert msg.status[0].name == 'linetest_planner/stability'
    values = {item.key: item.value for item in msg.status[0].values}
    assert values['centerline_point_count'] == '3'
    assert values['selected_edge_count'] == '0'
    assert values['planner_mode'] == 'linetest'
    assert values['control_path_point_count'] == '3'
    assert float(values['path_length_m']) > 0.0


def test_on_timer_with_controller_disabled_publishes_path_and_diagnostics_without_cmd():
    node = _make_node()
    node._latest_odom_msg = _odom_msg(x=-38.5, y=0.0, yaw=0.0)

    node._on_timer()

    assert len(node._cmd_pub.messages) == 0
    assert len(node._path_pub.messages) == 1
    assert len(node._diag_pub.messages) == 1

    path_msg = node._path_pub.messages[-1]
    first_pose = path_msg.poses[0].pose.position
    last_pose = path_msg.poses[-1].pose.position
    assert abs(first_pose.x - node.line_start_x_m) <= 1e-9
    assert abs(first_pose.y - node.line_start_y_m) <= 1e-9
    assert abs(last_pose.x - node.line_end_x_m) <= 1e-9
    assert abs(last_pose.y - node.line_end_y_m) <= 1e-9

    diag_msg = node._diag_pub.messages[-1]
    stability_values = {item.key: item.value for item in diag_msg.status[0].values}
    assert stability_values['fresh_publish_flag'] == '1'
    assert stability_values['operator_reason_code'] == str(LineTestPlannerNode._operator_reason_code('controller_disabled'))
