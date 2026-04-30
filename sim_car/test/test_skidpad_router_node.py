from __future__ import annotations

import pathlib
import signal
import sys

import numpy as np
import pytest
from builtin_interfaces.msg import Time as TimeMsg


TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from vehicle_plotter_msgs.msg import ConeDetection  # noqa: E402
    from sim_car.planning.skidpad_router_core import (  # noqa: E402
        SkidpadRouterConfig,
        SkidpadRouterSnapshot,
        SkidpadStateMachine,
    )
    from sim_car.planning.skidpad_router_node import SkidpadRouterNode  # noqa: E402
except ImportError as exc:  # pragma: no cover - depends on generated ROS interfaces
    pytest.skip(f"ROS router node imports unavailable: {exc}", allow_module_level=True)


class _PublisherRecorder:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, msg) -> None:
        self.messages.append(msg)


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warn_messages: list[str] = []

    def info(self, msg: str) -> None:
        self.info_messages.append(msg)

    def warn(self, msg: str) -> None:
        self.warn_messages.append(msg)


def _make_node(*, test_park_only: bool = False, route_laps: int = 1) -> SkidpadRouterNode:
    node = object.__new__(SkidpadRouterNode)
    node._state_machine = SkidpadStateMachine(
        SkidpadRouterConfig(
            test_park_only=test_park_only,
            route_laps=route_laps,
        )
    )
    node._latest_pose_xy = (0.0, 0.0)
    node._latest_yaw_rad = 0.0
    node._latest_speed_mps = 0.0
    node._orange_only_cone_count = 0
    node._parking_boundary_override_count = 0
    node._parking_mode_active = False
    node._acceleration_parking_latched = False
    node.event_mode = "skidpad"
    node.shutdown_on_route_complete = False
    node.shutdown_on_parking_complete = False
    node._route_complete_shutdown_requested = False
    node._parking_complete_shutdown_requested = False
    node._acceleration_parked_since_sec = None
    node.odom_frame = "odom"
    node.acceleration_activation_distance_m = 10.0
    node.acceleration_stop_row_cluster_depth_m = 0.75
    node.acceleration_stop_row_min_cluster_count = 4
    node.acceleration_stop_row_min_lateral_span_m = 2.0
    node.acceleration_stop_row_min_points_per_side = 1
    node.stop_line_pair_max_distance_m = 1.0
    node.stop_margin_m = 1.0
    node.target_margin_m = 3.5
    node.stop_approach_speed_gain = 1.5
    node.brake_activation_margin_m = 1.0
    node.brake_command = 1.0
    node._stop_override_active = False
    node._stop_line_forward_distance_m = float("nan")
    node._stop_line_marker_points_odom = None
    node._diag_pub = _PublisherRecorder()
    node._viz_pub = _PublisherRecorder()
    node._cmd_pub = _PublisherRecorder()
    node._brake_pub = _PublisherRecorder()
    node.get_name = lambda: "skidpad_router_node"
    node._fake_logger = _FakeLogger()
    node.get_logger = lambda: node._fake_logger
    node._latest_snapshot = SkidpadRouterSnapshot(
        stage_name=node._state_machine.stage_name(),
        active_branch=node._state_machine.active_branch(),
        completed_laps=node._state_machine.completed_laps,
        route_index=node._state_machine.route_index,
        current_route_pass=node._state_machine.current_route_pass(),
        total_route_passes=node._state_machine.total_route_passes(),
        completed_route_passes=node._state_machine.completed_route_passes(),
        in_crossroads=False,
        lap_angle_accum_rad=0.0,
        lap_armed=False,
        parked=False,
        just_completed_lap=False,
        just_entered_crossroads=False,
    )
    return node


def _cone(*, x_m: float, y_m: float, color: str, boundary_color: str = "") -> ConeDetection:
    cone = ConeDetection()
    cone.position.x = float(x_m)
    cone.position.y = float(y_m)
    cone.position.z = 0.0
    cone.color = color
    cone.boundary_color = boundary_color
    cone.confidence = 1.0
    cone.track_confidence = 1.0
    cone.color_confidence = 1.0
    return cone


def test_route_parking_cones_drops_non_orange_and_overwrites_bad_boundary_hints() -> None:
    node = _make_node(test_park_only=True)
    node._state_machine.approach_complete = True

    cones = [
        _cone(x_m=0.0, y_m=10.0, color="orange", boundary_color="yellow"),
        _cone(x_m=1.0, y_m=11.0, color="big_orange", boundary_color=""),
        _cone(x_m=0.0, y_m=9.0, color="blue", boundary_color="blue"),
        _cone(x_m=10.0, y_m=0.0, color="orange", boundary_color="blue"),
    ]
    points = np.asarray(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [0.0, 9.0],
            [10.0, 0.0],
        ],
        dtype=np.float64,
    )

    routed = node._route_parking_cones(converted_cones=cones, cone_points_odom=points)

    assert len(routed) == 2
    assert [cone.color for cone in routed] == ["orange", "big_orange"]
    assert [cone.boundary_color for cone in routed] == ["blue", "blue"]
    assert node._orange_only_cone_count == 2
    assert node._parking_boundary_override_count == 1


def test_route_non_parking_cones_passes_all_acceleration_cones_through() -> None:
    node = _make_node()
    node.event_mode = "acceleration"

    cones = [
        _cone(x_m=-30.0, y_m=1.5, color="blue", boundary_color="blue"),
        _cone(x_m=-30.0, y_m=-1.5, color="yellow", boundary_color="yellow"),
        _cone(x_m=50.0, y_m=0.75, color="orange", boundary_color=""),
    ]
    points = np.asarray(
        [
            [-30.0, 1.5],
            [-30.0, -1.5],
            [50.0, 0.75],
        ],
        dtype=np.float64,
    )

    routed = node._route_non_parking_cones(converted_cones=cones, cone_points_odom=points)

    assert routed == cones


def test_route_non_parking_cones_keeps_skidpad_branch_masking() -> None:
    node = _make_node()
    _ = _enter_skidpad_approach(node)

    cones = [
        _cone(x_m=0.0, y_m=0.0, color="blue", boundary_color="blue"),
        _cone(x_m=10.0, y_m=0.0, color="blue", boundary_color="blue"),
        _cone(x_m=-10.0, y_m=0.0, color="yellow", boundary_color="yellow"),
        _cone(x_m=0.0, y_m=12.0, color="orange", boundary_color=""),
    ]
    points = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [-10.0, 0.0],
            [0.0, 12.0],
        ],
        dtype=np.float64,
    )

    routed = node._route_non_parking_cones(converted_cones=cones, cone_points_odom=points)

    assert [(cone.position.x, cone.position.y) for cone in routed] == [(0.0, 0.0), (10.0, 0.0)]


def test_route_parking_cones_uses_vehicle_frame_lateral_sign() -> None:
    node = _make_node(test_park_only=True)
    node._latest_pose_xy = (1.0, 2.0)
    node._latest_yaw_rad = np.pi / 2.0
    node._state_machine.approach_complete = True

    cones = [
        _cone(x_m=2.5, y_m=2.0, color="orange", boundary_color="yellow"),
        _cone(x_m=-0.5, y_m=2.0, color="orange", boundary_color="blue"),
    ]
    points = np.asarray(
        [
            [2.5, 2.0],
            [-0.5, 2.0],
        ],
        dtype=np.float64,
    )

    routed = node._route_parking_cones(converted_cones=cones, cone_points_odom=points)

    assert [cone.boundary_color for cone in routed] == ["yellow", "blue"]


def test_route_parking_cones_removes_detected_front_stop_line_row() -> None:
    node = _make_node(test_park_only=True)
    node._latest_yaw_rad = np.pi / 2.0
    node._state_machine.approach_complete = True

    cones = [
        _cone(x_m=-1.6, y_m=12.0, color="orange"),
        _cone(x_m=1.6, y_m=12.0, color="orange"),
        _cone(x_m=-0.5, y_m=12.0, color="orange"),
        _cone(x_m=0.5, y_m=12.0, color="orange"),
        _cone(x_m=-1.6, y_m=10.0, color="orange"),
        _cone(x_m=1.6, y_m=10.0, color="orange"),
    ]
    points = np.asarray(
        [
            [-1.6, 12.0],
            [1.6, 12.0],
            [-0.5, 12.0],
            [0.5, 12.0],
            [-1.6, 10.0],
            [1.6, 10.0],
        ],
        dtype=np.float64,
    )

    routed = node._route_parking_cones(converted_cones=cones, cone_points_odom=points)

    assert len(routed) == 2
    assert {(round(cone.position.x, 1), round(cone.position.y, 1)) for cone in routed} == {(-1.6, 10.0), (1.6, 10.0)}
    assert np.isfinite(node._stop_line_forward_distance_m)
    assert node._stop_line_marker_points_odom is not None


def test_acceleration_parking_cones_still_drop_non_orange_and_override_boundary_hints() -> None:
    node = _make_node()
    node.event_mode = "acceleration"
    node._acceleration_parking_latched = True

    cones = [
        _cone(x_m=0.0, y_m=10.0, color="orange", boundary_color="yellow"),
        _cone(x_m=1.0, y_m=11.0, color="orange", boundary_color=""),
        _cone(x_m=0.0, y_m=9.0, color="blue", boundary_color="blue"),
    ]
    points = np.asarray(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [0.0, 9.0],
        ],
        dtype=np.float64,
    )

    routed = node._route_parking_cones(converted_cones=cones, cone_points_odom=points)

    assert len(routed) == 2
    assert [cone.color for cone in routed] == ["orange", "orange"]
    assert [cone.boundary_color for cone in routed] == ["blue", "blue"]
    assert node._orange_only_cone_count == 2
    assert node._parking_boundary_override_count == 1


def test_acceleration_parking_cones_remove_detected_finish_stop_row() -> None:
    node = _make_node()
    node.event_mode = "acceleration"
    node._acceleration_parking_latched = True
    node._latest_yaw_rad = 0.0

    cones = [
        _cone(x_m=30.0, y_m=1.5, color="orange"),
        _cone(x_m=30.0, y_m=-1.5, color="orange"),
        _cone(x_m=35.0, y_m=1.5, color="orange"),
        _cone(x_m=35.0, y_m=-1.5, color="orange"),
        _cone(x_m=40.0, y_m=1.5, color="orange"),
        _cone(x_m=40.0, y_m=-1.5, color="orange"),
        _cone(x_m=45.0, y_m=1.5, color="orange"),
        _cone(x_m=45.0, y_m=-1.5, color="orange"),
        _cone(x_m=50.0, y_m=1.5, color="orange"),
        _cone(x_m=50.0, y_m=-1.5, color="orange"),
        _cone(x_m=50.0, y_m=0.75, color="orange"),
        _cone(x_m=50.0, y_m=-0.75, color="orange"),
    ]
    points = np.asarray(
        [
            [30.0, 1.5],
            [30.0, -1.5],
            [35.0, 1.5],
            [35.0, -1.5],
            [40.0, 1.5],
            [40.0, -1.5],
            [45.0, 1.5],
            [45.0, -1.5],
            [50.0, 1.5],
            [50.0, -1.5],
            [50.0, 0.75],
            [50.0, -0.75],
        ],
        dtype=np.float64,
    )

    routed = node._route_parking_cones(converted_cones=cones, cone_points_odom=points)

    assert len(routed) == 8
    assert {(round(cone.position.x, 2), round(cone.position.y, 2)) for cone in routed} == {
        (30.0, 1.5),
        (30.0, -1.5),
        (35.0, 1.5),
        (35.0, -1.5),
        (40.0, 1.5),
        (40.0, -1.5),
        (45.0, 1.5),
        (45.0, -1.5),
    }
    assert np.isfinite(node._stop_line_forward_distance_m)
    assert node._stop_line_marker_points_odom is not None
    assert {
        (round(point[0], 2), round(point[1], 2))
        for point in node._stop_line_marker_points_odom
    } == {(50.0, 1.5), (50.0, -1.5)}


def test_acceleration_parking_cones_fallback_infer_stop_row_from_frontier_cone() -> None:
    node = _make_node()
    node.event_mode = "acceleration"
    node._acceleration_parking_latched = True
    node._latest_yaw_rad = 0.0

    cones = [
        _cone(x_m=30.0, y_m=1.5, color="orange"),
        _cone(x_m=30.0, y_m=-1.5, color="orange"),
        _cone(x_m=35.0, y_m=1.5, color="orange"),
        _cone(x_m=35.0, y_m=-1.5, color="orange"),
        _cone(x_m=40.0, y_m=1.5, color="orange"),
        _cone(x_m=40.0, y_m=-1.5, color="orange"),
        _cone(x_m=45.0, y_m=1.5, color="orange"),
        _cone(x_m=45.0, y_m=-1.5, color="orange"),
        _cone(x_m=49.8, y_m=-1.15, color="orange"),
    ]
    points = np.asarray(
        [
            [30.0, 1.5],
            [30.0, -1.5],
            [35.0, 1.5],
            [35.0, -1.5],
            [40.0, 1.5],
            [40.0, -1.5],
            [45.0, 1.5],
            [45.0, -1.5],
            [49.8, -1.15],
        ],
        dtype=np.float64,
    )

    routed = node._route_parking_cones(converted_cones=cones, cone_points_odom=points)

    assert len(routed) == 8
    assert {(round(cone.position.x, 2), round(cone.position.y, 2)) for cone in routed} == {
        (30.0, 1.5),
        (30.0, -1.5),
        (35.0, 1.5),
        (35.0, -1.5),
        (40.0, 1.5),
        (40.0, -1.5),
        (45.0, 1.5),
        (45.0, -1.5),
    }
    assert np.isfinite(node._stop_line_forward_distance_m)
    assert round(node._stop_line_forward_distance_m, 2) == 49.8
    assert node._stop_line_marker_points_odom is not None
    assert {
        (round(point[0], 2), round(point[1], 2))
        for point in node._stop_line_marker_points_odom
    } == {(49.8, 1.5), (49.8, -1.5)}


def test_route_parking_cones_keeps_filtering_latched_stop_row_when_next_frame_is_noisy() -> None:
    node = _make_node(test_park_only=True)
    node._latest_yaw_rad = np.pi / 2.0
    node._state_machine.approach_complete = True

    first_points = np.asarray(
        [
            [-1.6, 12.0],
            [1.6, 12.0],
            [-0.5, 12.0],
            [0.5, 12.0],
            [-1.6, 10.0],
            [1.6, 10.0],
        ],
        dtype=np.float64,
    )
    first_cones = [_cone(x_m=float(point[0]), y_m=float(point[1]), color="orange") for point in first_points]
    node._route_parking_cones(converted_cones=first_cones, cone_points_odom=first_points)

    second_points = np.asarray(
        [
            [-1.55, 12.3],
            [1.55, 11.7],
            [-0.45, 12.2],
            [0.45, 11.8],
            [-1.6, 10.0],
            [1.6, 10.0],
        ],
        dtype=np.float64,
    )
    second_cones = [_cone(x_m=float(point[0]), y_m=float(point[1]), color="orange") for point in second_points]

    routed = node._route_parking_cones(converted_cones=second_cones, cone_points_odom=second_points)

    assert len(routed) == 2
    assert {(round(cone.position.x, 1), round(cone.position.y, 1)) for cone in routed} == {(-1.6, 10.0), (1.6, 10.0)}


def test_acceleration_stop_target_marker_appears_before_detected_stop_line() -> None:
    node = _make_node()
    node.event_mode = "acceleration"
    node._parking_mode_active = True
    node.stop_margin_m = 4.0
    node.target_margin_m = 3.5
    node._stop_line_forward_distance_m = 50.0
    node._stop_line_marker_points_odom = ((50.0, 1.5), (50.0, -1.5))

    target_marker = node._make_stop_target_marker(marker_id=5, stamp=TimeMsg(sec=1, nanosec=0))
    center_marker = node._make_stop_target_center_marker(marker_id=6, stamp=TimeMsg(sec=1, nanosec=0))

    assert target_marker.action == target_marker.ADD
    assert [(round(point.x, 2), round(point.y, 2)) for point in target_marker.points] == [
        (46.5, 1.5),
        (46.5, -1.5),
    ]
    assert center_marker.action == center_marker.ADD
    assert round(center_marker.pose.position.x, 2) == 46.5
    assert round(center_marker.pose.position.y, 2) == 0.0


def test_parking_override_slows_to_target_margin_before_full_stop() -> None:
    node = _make_node()
    node.event_mode = "acceleration"
    node._parking_mode_active = True
    node._stop_override_active = True
    node._latest_speed_mps = 4.0
    node.target_margin_m = 3.5
    node.stop_approach_speed_gain = 1.5
    node.brake_activation_margin_m = 1.0
    node._stop_line_forward_distance_m = 5.0
    node._stop_line_marker_points_odom = ((5.0, 1.5), (5.0, -1.5))

    node._publish_parking_override_cmd(TimeMsg(sec=1, nanosec=0))

    assert len(node._cmd_pub.messages) == 1
    assert node._cmd_pub.messages[0].drive.speed == pytest.approx(2.25)
    assert len(node._brake_pub.messages) == 1
    assert node._brake_pub.messages[0].data == pytest.approx(0.0)

    node._stop_line_forward_distance_m = 3.5
    node._publish_parking_override_cmd(TimeMsg(sec=2, nanosec=0))

    assert len(node._cmd_pub.messages) == 2
    assert node._cmd_pub.messages[1].drive.speed == pytest.approx(0.0)
    assert len(node._brake_pub.messages) == 2
    assert node._brake_pub.messages[1].data == pytest.approx(1.0)


def test_acceleration_parking_status_marker_switches_from_searching_to_found() -> None:
    node = _make_node()
    node.event_mode = "acceleration"
    node._parking_mode_active = True

    searching_marker = node._make_acceleration_parking_status_marker(marker_id=7, stamp=TimeMsg(sec=1, nanosec=0))

    assert searching_marker.action == searching_marker.ADD
    assert "parking_found=0" in searching_marker.text
    assert "stop_latched=0" in searching_marker.text

    node._stop_line_forward_distance_m = 50.0
    node._stop_line_marker_points_odom = ((50.0, 1.5), (50.0, -1.5))
    found_marker = node._make_acceleration_parking_status_marker(marker_id=7, stamp=TimeMsg(sec=2, nanosec=0))

    assert found_marker.action == found_marker.ADD
    assert "parking_found=1" in found_marker.text
    assert "stop_latched=1" in found_marker.text
    assert "row_forward_m=50.00" in found_marker.text


def test_build_status_text_includes_route_counter_and_circle_laps() -> None:
    node = _make_node(route_laps=3)
    node._latest_snapshot = SkidpadRouterSnapshot(
        stage_name="left_1",
        active_branch="left",
        completed_laps=4,
        route_index=4,
        current_route_pass=2,
        total_route_passes=3,
        completed_route_passes=1,
        in_crossroads=False,
        lap_angle_accum_rad=0.0,
        lap_armed=True,
        parked=False,
        just_completed_lap=False,
        just_entered_crossroads=False,
    )

    text = node._build_status_text(node._latest_snapshot)

    assert "route=2/3" in text
    assert "circle_laps=4" in text
    assert "stage=left_1" in text


def test_publish_diagnostics_includes_route_progress_values() -> None:
    node = _make_node(route_laps=2)
    node._latest_snapshot = SkidpadRouterSnapshot(
        stage_name="right_2",
        active_branch="right",
        completed_laps=1,
        route_index=1,
        current_route_pass=1,
        total_route_passes=2,
        completed_route_passes=0,
        in_crossroads=True,
        lap_angle_accum_rad=5.7,
        lap_armed=True,
        parked=False,
        just_completed_lap=False,
        just_entered_crossroads=False,
    )

    node._publish_diagnostics(TimeMsg(sec=5, nanosec=0))

    assert len(node._diag_pub.messages) == 1
    status = node._diag_pub.messages[0].status[0]
    values = {item.key: item.value for item in status.values}
    assert values["current_route_pass"] == "1"
    assert values["total_route_passes"] == "2"
    assert values["completed_route_passes"] == "0"
    assert values["completed_laps"] == "1"


def test_route_complete_shutdown_requests_rclpy_shutdown_once(monkeypatch) -> None:
    shutdown_calls: list[bool] = []
    signal_calls: list[tuple[int, int]] = []
    timer_started: list[float] = []

    class _FakeTimer:
        def __init__(self, delay_s, callback) -> None:
            self.delay_s = float(delay_s)
            self.callback = callback
            self.daemon = False

        def start(self) -> None:
            timer_started.append(self.delay_s)
            if self.delay_s == 0.0:
                self.callback()

    monkeypatch.setattr("sim_car.planning.skidpad_router_node.rclpy.ok", lambda: True)
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.rclpy.shutdown",
        lambda: shutdown_calls.append(True),
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.getpgrp",
        lambda: 12345,
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.getppid",
        lambda: 23456,
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.kill",
        lambda pid, sig: signal_calls.append((pid, sig)),
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.killpg",
        lambda pgid, sig: signal_calls.append((pgid, sig)),
    )
    monkeypatch.setattr("sim_car.planning.skidpad_router_node.threading.Timer", _FakeTimer)
    node = _make_node(route_laps=1)
    node.shutdown_on_route_complete = True
    node._latest_snapshot = SkidpadRouterSnapshot(
        stage_name="straight",
        active_branch="straight",
        completed_laps=4,
        route_index=4,
        current_route_pass=1,
        total_route_passes=1,
        completed_route_passes=1,
        in_crossroads=True,
        lap_angle_accum_rad=0.0,
        lap_armed=False,
        parked=False,
        just_completed_lap=True,
        just_entered_crossroads=True,
    )

    node._maybe_shutdown_after_route_complete()
    node._maybe_shutdown_after_route_complete()

    assert shutdown_calls == [True]
    assert signal_calls == [(23456, signal.SIGINT)]
    assert timer_started == [0.0, 2.0]
    assert node._route_complete_shutdown_requested
    assert "skidpad route target reached" in node._fake_logger.info_messages[0]


def test_acceleration_parking_complete_shutdown_requests_rclpy_shutdown_once(monkeypatch) -> None:
    shutdown_calls: list[bool] = []
    signal_calls: list[tuple[int, int]] = []
    timer_started: list[float] = []

    class _FakeTimer:
        def __init__(self, delay_s, callback) -> None:
            self.delay_s = float(delay_s)
            self.callback = callback
            self.daemon = False

        def start(self) -> None:
            timer_started.append(self.delay_s)
            if self.delay_s == 0.0:
                self.callback()

    monkeypatch.setattr("sim_car.planning.skidpad_router_node.rclpy.ok", lambda: True)
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.rclpy.shutdown",
        lambda: shutdown_calls.append(True),
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.getpgrp",
        lambda: 12345,
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.getppid",
        lambda: 23456,
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.kill",
        lambda pid, sig: signal_calls.append((pid, sig)),
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.killpg",
        lambda pgid, sig: signal_calls.append((pgid, sig)),
    )
    monkeypatch.setattr("sim_car.planning.skidpad_router_node.threading.Timer", _FakeTimer)
    node = _make_node()
    node.event_mode = "acceleration"
    node.shutdown_on_parking_complete = True
    node._parking_mode_active = True
    node._stop_override_active = True
    node._stop_line_marker_points_odom = ((1.0, 1.5), (1.0, -1.5))
    node._stop_line_forward_distance_m = node.target_margin_m
    node._latest_speed_mps = 0.0

    node._maybe_shutdown_after_parking_complete(stamp=TimeMsg(sec=10, nanosec=0))
    node._maybe_shutdown_after_parking_complete(stamp=TimeMsg(sec=10, nanosec=500_000_000))
    node._maybe_shutdown_after_parking_complete(stamp=TimeMsg(sec=11, nanosec=0))
    node._maybe_shutdown_after_parking_complete(stamp=TimeMsg(sec=12, nanosec=0))

    assert shutdown_calls == [True]
    assert signal_calls == [(23456, signal.SIGINT)]
    assert timer_started == [0.0, 2.0]
    assert node._parking_complete_shutdown_requested
    assert "acceleration parking target reached" in node._fake_logger.info_messages[0]


def test_skidpad_parking_complete_shutdown_waits_for_parked_state(monkeypatch) -> None:
    shutdown_calls: list[bool] = []
    signal_calls: list[tuple[int, int]] = []
    timer_started: list[float] = []

    class _FakeTimer:
        def __init__(self, delay_s, callback) -> None:
            self.delay_s = float(delay_s)
            self.callback = callback
            self.daemon = False

        def start(self) -> None:
            timer_started.append(self.delay_s)
            if self.delay_s == 0.0:
                self.callback()

    monkeypatch.setattr("sim_car.planning.skidpad_router_node.rclpy.ok", lambda: True)
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.rclpy.shutdown",
        lambda: shutdown_calls.append(True),
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.getpgrp",
        lambda: 12345,
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.getppid",
        lambda: 23456,
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.kill",
        lambda pid, sig: signal_calls.append((pid, sig)),
    )
    monkeypatch.setattr(
        "sim_car.planning.skidpad_router_node.os.killpg",
        lambda pgid, sig: signal_calls.append((pgid, sig)),
    )
    monkeypatch.setattr("sim_car.planning.skidpad_router_node.threading.Timer", _FakeTimer)
    node = _make_node(route_laps=1)
    node.shutdown_on_parking_complete = True
    node._latest_snapshot = SkidpadRouterSnapshot(
        stage_name="straight",
        active_branch="straight",
        completed_laps=4,
        route_index=4,
        current_route_pass=1,
        total_route_passes=1,
        completed_route_passes=1,
        in_crossroads=True,
        lap_angle_accum_rad=0.0,
        lap_armed=False,
        parked=False,
        just_completed_lap=True,
        just_entered_crossroads=True,
    )

    node._maybe_shutdown_after_parking_complete(stamp=TimeMsg(sec=10, nanosec=0))

    assert shutdown_calls == []
    assert signal_calls == []
    assert timer_started == []

    node._latest_snapshot = SkidpadRouterSnapshot(
        stage_name="parked",
        active_branch="straight",
        completed_laps=4,
        route_index=4,
        current_route_pass=1,
        total_route_passes=1,
        completed_route_passes=1,
        in_crossroads=False,
        lap_angle_accum_rad=0.0,
        lap_armed=False,
        parked=True,
        just_completed_lap=False,
        just_entered_crossroads=False,
    )

    node._maybe_shutdown_after_parking_complete(stamp=TimeMsg(sec=11, nanosec=0))
    node._maybe_shutdown_after_parking_complete(stamp=TimeMsg(sec=12, nanosec=0))

    assert shutdown_calls == [True]
    assert signal_calls == [(23456, signal.SIGINT)]
    assert timer_started == [0.0, 2.0]
    assert node._parking_complete_shutdown_requested
    assert "skidpad parking target reached" in node._fake_logger.info_messages[0]


def _enter_skidpad_approach(node: SkidpadRouterNode) -> float:
    now_sec = 0.0
    for x_m, y_m in [(0.0, -4.0), (0.0, -1.5), (0.0, 0.0)]:
        node._latest_snapshot = node._state_machine.update(
            x_m=x_m,
            y_m=y_m,
            speed_mps=3.0,
            now_sec=now_sec,
        )
        now_sec += 0.1
    return now_sec
