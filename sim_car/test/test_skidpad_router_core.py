from __future__ import annotations

import pathlib
import sys

import numpy as np


TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.planning.skidpad_router_core import (
    SkidpadRouterConfig,
    SkidpadStateMachine,
    boundary_color_from_lateral_y,
    detect_stop_line_pair,
    detect_stop_line_forward_distance_m,
)


def _advance(machine: SkidpadStateMachine, points_xy: list[tuple[float, float]], *, start_t: float = 0.0) -> float:
    now_sec = float(start_t)
    for x_m, y_m in points_xy:
        machine.update(x_m=x_m, y_m=y_m, speed_mps=3.0, now_sec=now_sec)
        now_sec += 0.1
    return now_sec


def _enter_crossroads(machine: SkidpadStateMachine, *, start_t: float = 0.0) -> float:
    return _advance(machine, [(0.0, -4.0), (0.0, -1.5), (0.0, 0.0)], start_t=start_t)


def _complete_lap(
    machine: SkidpadStateMachine,
    *,
    center_xy: tuple[float, float],
    radius_m: float = 9.0,
    start_t: float = 0.0,
) -> float:
    cx, cy = center_xy
    angles = np.linspace(-1.1, 5.4, 48)
    points = [(cx + (radius_m * np.cos(theta)), cy + (radius_m * np.sin(theta))) for theta in angles]
    points.extend([(3.5 if cx > 0.0 else -3.5, 0.0), (0.0, 0.0)])
    return _advance(machine, points, start_t=start_t)


def test_initial_center_entry_does_not_increment_laps() -> None:
    machine = SkidpadStateMachine(SkidpadRouterConfig())

    _enter_crossroads(machine)

    snapshot = machine.update(x_m=0.0, y_m=0.5, speed_mps=1.0, now_sec=0.4)
    assert snapshot.completed_laps == 0
    assert snapshot.stage_name == "right_1"
    assert snapshot.active_branch == "right"


def test_full_right_circle_then_center_reentry_counts_one_lap() -> None:
    machine = SkidpadStateMachine(SkidpadRouterConfig())

    now_sec = _enter_crossroads(machine)
    now_sec = _complete_lap(machine, center_xy=machine.config.right_circle_center_xy, start_t=now_sec)

    snapshot = machine.update(x_m=0.0, y_m=0.0, speed_mps=1.0, now_sec=now_sec)
    assert snapshot.completed_laps == 1
    assert snapshot.stage_name == "right_2"
    assert snapshot.active_branch == "right"


def test_partial_loop_and_center_dither_do_not_increment_lap() -> None:
    machine = SkidpadStateMachine(SkidpadRouterConfig())

    now_sec = _enter_crossroads(machine)
    cx, cy = machine.config.right_circle_center_xy
    partial_angles = np.linspace(-1.1, 2.2, 16)
    partial_points = [(cx + (9.0 * np.cos(theta)), cy + (9.0 * np.sin(theta))) for theta in partial_angles]
    partial_points.extend([(3.5, 0.0), (0.0, 0.0), (3.5, 0.2), (0.0, 0.1)])
    _advance(machine, partial_points, start_t=now_sec)

    snapshot = machine.update(x_m=0.0, y_m=-0.2, speed_mps=0.5, now_sec=now_sec + 2.0)
    assert snapshot.completed_laps == 0
    assert snapshot.stage_name == "right_1"


def test_branch_sequence_advances_right_right_left_left_straight() -> None:
    machine = SkidpadStateMachine(SkidpadRouterConfig())

    now_sec = _enter_crossroads(machine)
    assert machine.stage_name() == "right_1"

    now_sec = _complete_lap(machine, center_xy=machine.config.right_circle_center_xy, start_t=now_sec)
    assert machine.stage_name() == "right_2"

    now_sec = _complete_lap(machine, center_xy=machine.config.right_circle_center_xy, start_t=now_sec)
    assert machine.stage_name() == "left_1"

    now_sec = _complete_lap(machine, center_xy=machine.config.left_circle_center_xy, start_t=now_sec)
    assert machine.stage_name() == "left_2"

    _complete_lap(machine, center_xy=machine.config.left_circle_center_xy, start_t=now_sec)
    assert machine.stage_name() == "straight"
    assert machine.active_branch() == "straight"
    assert machine.completed_laps == 4


def test_right_routing_suppresses_left_lobe_and_parking() -> None:
    machine = SkidpadStateMachine(SkidpadRouterConfig())
    _enter_crossroads(machine)

    points = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [-10.0, 0.0],
            [0.0, 12.0],
        ],
        dtype=np.float64,
    )
    mask = machine.route_mask(points)
    assert mask.tolist() == [True, True, False, False]


def test_left_routing_suppresses_right_lobe_and_parking() -> None:
    machine = SkidpadStateMachine(SkidpadRouterConfig())
    now_sec = _enter_crossroads(machine)
    _complete_lap(machine, center_xy=machine.config.right_circle_center_xy, start_t=now_sec)
    _complete_lap(machine, center_xy=machine.config.right_circle_center_xy, start_t=now_sec + 10.0)

    points = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [-10.0, 0.0],
            [0.0, 12.0],
        ],
        dtype=np.float64,
    )
    mask = machine.route_mask(points)
    assert mask.tolist() == [True, False, True, False]


def test_straight_routing_adds_synthetic_bridge_pairs() -> None:
    machine = SkidpadStateMachine(SkidpadRouterConfig())
    now_sec = _enter_crossroads(machine)
    now_sec = _complete_lap(machine, center_xy=machine.config.right_circle_center_xy, start_t=now_sec)
    now_sec = _complete_lap(machine, center_xy=machine.config.right_circle_center_xy, start_t=now_sec)
    now_sec = _complete_lap(machine, center_xy=machine.config.left_circle_center_xy, start_t=now_sec)
    _complete_lap(machine, center_xy=machine.config.left_circle_center_xy, start_t=now_sec)

    assert machine.stage_name() == "straight"
    synthetic = machine.synthetic_cone_pairs()
    assert len(synthetic) == 6
    assert np.allclose([point[1] for point in synthetic], [3.5, 3.5, 5.5, 5.5, 7.5, 7.5])


def test_test_park_only_skips_laps_and_routes_straight() -> None:
    machine = SkidpadStateMachine(SkidpadRouterConfig(test_park_only=True))

    assert machine.active_branch() == "straight"
    assert machine.stage_name() == "approach"

    _enter_crossroads(machine)

    assert machine.active_branch() == "straight"
    assert machine.stage_name() == "straight"
    assert machine.completed_laps == 0

    points = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [-10.0, 0.0],
        ],
        dtype=np.float64,
    )
    mask = machine.route_mask(points)
    assert mask.tolist() == [True, True, False, False]


def test_detect_stop_line_forward_distance_uses_front_four_cone_row() -> None:
    points = np.asarray(
        [
            [4.0, -1.5],
            [5.9, -1.5],
            [6.0, 1.5],
            [6.1, -0.6],
            [6.0, 0.6],
        ],
        dtype=np.float64,
    )
    distance_m = detect_stop_line_forward_distance_m(
        points,
        cluster_depth_m=0.4,
        min_lateral_span_m=1.0,
        min_cluster_count=4,
        min_points_per_side=1,
    )
    assert distance_m is not None
    assert abs(distance_m - 6.0) < 1e-9


def test_detect_stop_line_forward_distance_allows_scattered_front_row_when_depth_is_loose() -> None:
    points = np.asarray(
        [
            [8.9, -1.6],
            [9.7, -0.6],
            [10.4, 0.5],
            [11.1, 1.7],
            [7.0, -1.6],
            [7.0, 1.6],
        ],
        dtype=np.float64,
    )
    distance_m = detect_stop_line_forward_distance_m(
        points,
        cluster_depth_m=2.5,
        min_lateral_span_m=1.0,
        min_cluster_count=4,
        min_points_per_side=1,
    )
    assert distance_m is not None
    assert abs(distance_m - 10.05) < 1e-9


def test_detect_stop_line_forward_distance_rejects_cluster_with_too_few_cones() -> None:
    points = np.asarray(
        [
            [5.9, -1.5],
            [6.0, 1.5],
            [6.1, -0.6],
        ],
        dtype=np.float64,
    )
    assert (
        detect_stop_line_forward_distance_m(
            points,
            cluster_depth_m=0.4,
            min_lateral_span_m=1.0,
            min_cluster_count=4,
            min_points_per_side=1,
        )
        is None
    )


def test_detect_stop_line_forward_distance_rejects_front_cluster_without_both_sides() -> None:
    points = np.asarray(
        [
            [5.9, 0.2],
            [6.0, 0.6],
            [6.1, 1.1],
            [6.0, 1.6],
        ],
        dtype=np.float64,
    )
    assert (
        detect_stop_line_forward_distance_m(
            points,
            cluster_depth_m=0.4,
            min_lateral_span_m=1.0,
            min_cluster_count=4,
            min_points_per_side=1,
        )
        is None
    )


def test_detect_stop_line_pair_picks_furthest_ahead_close_orange_pair() -> None:
    points = np.asarray(
        [
            [6.0, -0.2],
            [6.7, 0.1],
            [8.8, -0.4],
            [9.4, 0.0],
            [7.0, 1.5],
        ],
        dtype=np.float64,
    )
    detected = detect_stop_line_pair(points, max_pair_distance_m=1.0)
    assert detected is not None
    idx_a, idx_b, forward_distance_m = detected
    assert {idx_a, idx_b} == {2, 3}
    assert abs(forward_distance_m - 9.1) < 1e-9


def test_detect_stop_line_pair_rejects_pairs_outside_distance_threshold() -> None:
    points = np.asarray(
        [
            [8.8, -0.6],
            [10.0, 0.6],
            [7.0, -1.5],
            [7.0, 1.5],
        ],
        dtype=np.float64,
    )
    assert detect_stop_line_pair(points, max_pair_distance_m=1.0) is None


def test_boundary_color_from_lateral_y_matches_planner_convention() -> None:
    assert boundary_color_from_lateral_y(0.0) == "blue"
    assert boundary_color_from_lateral_y(0.5) == "blue"
    assert boundary_color_from_lateral_y(-0.5) == "yellow"
