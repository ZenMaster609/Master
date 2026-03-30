from __future__ import annotations

import pathlib
import sys

import numpy as np

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.cones.tracking.fusion import resolve_boundary_colors_for_planning  # noqa: E402
from sim_car.planning.corridor_planner_core import (  # noqa: E402
    CorridorPlannerConfig,
    CorridorPlannerPrior,
    _build_boundary_chain,
    compute_corridor_centerline,
    update_track_width_estimate,
)


def _cfg(**overrides) -> CorridorPlannerConfig:
    cfg = CorridorPlannerConfig(
        min_required_cones=4,
        min_chain_length=3,
        min_required_corridor_samples=4,
        min_path_points=4,
        min_forward_extent_m=2.0,
        path_resolution_m=0.5,
        max_path_length_m=20.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_straight_corridor_uses_centered_path():
    points = np.array(
        [[2.0, 1.8], [2.0, -1.8], [4.0, 1.8], [4.0, -1.8], [6.0, 1.8], [6.0, -1.8], [8.0, 1.8], [8.0, -1.8]],
        dtype=np.float64,
    )
    colors = ["blue", "yellow", "blue", "yellow", "blue", "yellow", "blue", "yellow"]
    conf = np.ones((8,), dtype=np.float64)

    result = compute_corridor_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        CorridorPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.planner_mode == "corridor"
    assert result.accepted_pair_count >= 4
    assert np.allclose(result.centerline[:, 1], 0.0, atol=1e-6)


def test_mild_asymmetry_stays_inside_corridor():
    points = np.array(
        [
            [2.0, 1.8], [2.0, -1.6],
            [4.0, 1.9], [4.0, -1.5],
            [6.0, 2.0], [6.0, -1.4],
            [8.0, 2.2], [8.0, -1.3],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "yellow"] * 4
    conf = np.ones((8,), dtype=np.float64)

    result = compute_corridor_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        CorridorPlannerPrior(previous_width_m=3.4),
    )

    assert result.status == "ok"
    assert result.centerline.shape[0] >= 4
    left_y = np.interp(result.centerline[:, 0], result.left_boundary[:, 0], result.left_boundary[:, 1])
    right_y = np.interp(result.centerline[:, 0], result.right_boundary[:, 0], result.right_boundary[:, 1])
    assert np.all(result.centerline[:, 1] <= left_y + 1e-6)
    assert np.all(result.centerline[:, 1] >= right_y - 1e-6)


def test_missing_boundary_returns_no_path():
    points = np.array([[2.0, 1.8], [4.0, 1.9], [6.0, 2.0], [8.0, 2.1]], dtype=np.float64)
    colors = ["blue", "blue", "blue", "blue"]
    conf = np.ones((4,), dtype=np.float64)

    result = compute_corridor_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        CorridorPlannerPrior(previous_width_m=3.6),
    )

    assert result.planner_mode == "none"
    assert result.centerline.shape[0] == 0
    assert "corridor" in result.status


def test_invalid_corridor_geometry_is_rejected():
    points = np.array(
        [[2.0, 1.5], [2.0, -1.5], [4.0, 0.4], [4.0, -0.4], [6.0, -0.2], [6.0, 0.2], [8.0, -0.6], [8.0, 0.6]],
        dtype=np.float64,
    )
    colors = ["blue", "yellow"] * 4
    conf = np.ones((8,), dtype=np.float64)

    result = compute_corridor_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        CorridorPlannerPrior(previous_width_m=3.6),
    )

    assert result.centerline.shape[0] == 0
    assert result.reject_counts["corridor_geometry"] > 0


def test_excessive_curvature_is_rejected():
    points = np.array(
        [
            [2.0, 2.0], [2.0, -2.0],
            [3.0, 2.2], [3.0, -1.8],
            [4.0, 2.8], [4.0, -1.2],
            [5.0, 3.8], [5.0, -0.2],
            [6.0, 5.2], [6.0, 1.2],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "yellow"] * 5
    conf = np.ones((10,), dtype=np.float64)

    result = compute_corridor_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(max_curvature=0.1),
        CorridorPlannerPrior(previous_width_m=4.0),
    )

    assert result.centerline.shape[0] == 0
    assert result.reject_counts["curvature"] > 0


def test_width_estimate_updates_slowly_and_clamps_delta():
    cfg = _cfg(initial_width_m=3.6, width_filter_alpha=0.2, max_width_delta_per_update_m=0.15)
    updated = update_track_width_estimate(3.6, 4.5, cfg)
    assert abs(updated - 3.63) < 1e-9


def test_orange_cones_can_plan_when_resolved_upstream():
    points = np.array(
        [[2.0, 1.8], [2.0, -1.8], [4.0, 1.8], [4.0, -1.8], [6.0, 1.8], [6.0, -1.8], [8.0, 1.8], [8.0, -1.8]],
        dtype=np.float64,
    )
    colors = resolve_boundary_colors_for_planning(
        points_xy=points,
        raw_colors=["orange"] * len(points),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
    )
    conf = np.ones((8,), dtype=np.float64)

    result = compute_corridor_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        CorridorPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.centerline.shape[0] >= 4


def test_turning_boundary_with_small_x_regression_keeps_corridor():
    points = np.array(
        [
            [2.0, 1.8], [2.0, -1.8],
            [4.0, 1.9], [4.0, -1.7],
            [6.0, 2.1], [6.0, -1.5],
            [6.4, 2.6], [6.5, -1.0],
            [6.1, 3.2], [6.3, -0.4],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "yellow"] * 5
    conf = np.ones((10,), dtype=np.float64)

    result = compute_corridor_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(min_required_corridor_samples=3),
        CorridorPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.accepted_pair_count >= 3


def test_deeper_turn_keeps_extending_both_boundary_chains():
    points = np.array(
        [
            [2.0, 1.8], [2.0, -1.8],
            [4.0, 1.9], [4.0, -1.7],
            [6.0, 2.2], [6.1, -1.4],
            [6.7, 3.0], [7.0, -0.7],
            [6.2, 4.1], [6.8, -0.2],
            [5.3, 5.0], [6.0, -0.1],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "yellow"] * 6
    conf = np.ones((12,), dtype=np.float64)

    result = compute_corridor_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(min_required_corridor_samples=4),
        CorridorPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.left_chain_length >= 5
    assert result.right_chain_length >= 5
    assert result.centerline.shape[0] >= 5


def test_prevalidation_centerline_is_preserved_when_validation_rejects_corridor_path():
    points = np.array(
        [[2.0, 1.8], [2.0, -1.8], [4.0, 1.8], [4.0, -1.8], [6.0, 1.8], [6.0, -1.8], [8.0, 1.8], [8.0, -1.8]],
        dtype=np.float64,
    )
    colors = ["blue", "yellow", "blue", "yellow", "blue", "yellow", "blue", "yellow"]
    conf = np.ones((8,), dtype=np.float64)

    result = compute_corridor_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(max_initial_heading_error_rad=-1.0),
        CorridorPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "path heading flip near vehicle"
    assert result.centerline.shape[0] == 0
    assert result.prevalidation_centerline.shape[0] >= 2


def test_concentric_arc_corridor_pairs_boundaries_through_turn():
    theta = np.linspace(0.0, 1.0, 7, dtype=np.float64)
    left = np.column_stack((6.5 * np.sin(theta), 6.5 * (1.0 - np.cos(theta)) + 1.8))
    right = np.column_stack((3.2 * np.sin(theta), 3.2 * (1.0 - np.cos(theta)) - 1.8))
    points = np.empty((left.shape[0] + right.shape[0], 2), dtype=np.float64)
    colors: list[str] = []
    for idx in range(left.shape[0]):
        points[2 * idx] = left[idx]
        points[(2 * idx) + 1] = right[idx]
        colors.extend(["blue", "yellow"])
    conf = np.ones((points.shape[0],), dtype=np.float64)

    result = compute_corridor_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(
            min_required_corridor_samples=4,
            max_corridor_width_m=5.5,
            max_curvature=0.8,
        ),
        CorridorPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.accepted_pair_count >= 4
    assert result.centerline.shape[0] >= 4
    assert np.max(result.centerline[:, 1]) > 0.6


def test_boundary_chain_fallback_orders_wraparound_arc():
    arc = np.array(
        [
            [1.35075576, -2.10367746],
            [0.54751672, -2.43930839],
            [-0.32211124, -2.47916203],
            [-1.15268173, -2.21840592],
            [-1.84348429, -1.68865795],
        ],
        dtype=np.float64,
    )
    chain = _build_boundary_chain(
        filtered_points=arc,
        filtered_local=arc,
        side_indices=np.arange(arc.shape[0], dtype=np.int64),
        config=_cfg(),
    )

    assert chain.filtered_indices.size >= 4
