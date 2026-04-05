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
    node.event_mode = "skidpad"
    node.odom_frame = "odom"
    node.stop_line_pair_max_distance_m = 1.0
    node.stop_margin_m = 1.0
    node._stop_override_active = False
    node._stop_line_forward_distance_m = float("nan")
    node._stop_line_marker_points_odom = None
    node._diag_pub = _PublisherRecorder()
    node._viz_pub = _PublisherRecorder()
    node.get_name = lambda: "skidpad_router_node"
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


def test_build_synthetic_cones_assigns_side_from_vehicle_frame_lateral_sign() -> None:
    node = _make_node(test_park_only=True)
    node._latest_pose_xy = (0.0, 0.0)
    node._latest_yaw_rad = np.pi / 2.0
    node._state_machine.approach_complete = True

    cones = node._build_synthetic_cones(
        existing_xy=np.empty((0, 2), dtype=np.float64),
        stamp=TimeMsg(sec=0, nanosec=0),
    )

    assert len(cones) == 6
    assert [cone.color for cone in cones] == ["orange"] * 6
    assert [cone.boundary_color for cone in cones] == ["blue", "yellow", "blue", "yellow", "blue", "yellow"]


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
