from __future__ import annotations

import pathlib
import sys

import numpy as np

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import sim_car.planning.single_boundary_planner_core as core  # noqa: E402
from sim_car.cones.tracking.fusion import resolve_boundary_colors_for_planning  # noqa: E402
from sim_car.planning.single_boundary_planner_core import (  # noqa: E402
    SingleBoundaryPlannerConfig,
    SingleBoundaryPlannerPrior,
    compute_single_boundary_centerline,
)


def _cfg(**overrides) -> SingleBoundaryPlannerConfig:
    cfg = SingleBoundaryPlannerConfig(
        min_required_cones=4,
        path_resolution_m=0.5,
        max_path_length_m=20.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_single_boundary_dropout_uses_offset_path():
    points = np.array([[2.0, 1.8], [4.0, 2.1], [6.0, 2.6], [8.0, 3.3]], dtype=np.float64)
    colors = ["blue", "blue", "blue", "blue"]
    conf = np.ones((4,), dtype=np.float64)

    result = compute_single_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        SingleBoundaryPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.planner_mode == "single_boundary"
    assert result.raw_offset_path.shape[0] >= 3
    assert result.centerline.shape[0] >= 3


def test_single_boundary_still_estimates_width_from_visible_pairs():
    points = np.array(
        [
            [2.0, 1.6],
            [2.0, -1.6],
            [4.0, 1.6],
            [4.0, -1.6],
            [6.0, 1.6],
            [6.0, -1.6],
            [8.0, 1.6],
            [8.0, -1.6],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "yellow", "blue", "yellow", "blue", "yellow", "blue", "yellow"]
    conf = np.ones((8,), dtype=np.float64)

    result = compute_single_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(initial_width_m=3.6),
        SingleBoundaryPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.accepted_pair_count >= 3
    assert abs(result.selected_chain_width_median - 3.2) < 1e-6
    assert result.planner_mode == "single_boundary"


def test_single_boundary_accepts_short_shorter_path_than_midpoint_defaults():
    points = np.array([[2.0, 1.8], [4.0, 2.1]], dtype=np.float64)
    colors = ["blue", "blue"]
    conf = np.ones((2,), dtype=np.float64)

    result = compute_single_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(min_required_cones=2),
        SingleBoundaryPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.centerline.shape[0] >= 2


def test_all_orange_track_is_resolved_to_single_boundary_path():
    points = np.array(
        [[2.0, 1.8], [2.0, -1.8], [4.0, 1.8], [4.0, -1.8], [6.0, 1.8], [6.0, -1.8]],
        dtype=np.float64,
    )
    colors = resolve_boundary_colors_for_planning(
        points_xy=points,
        raw_colors=["orange"] * len(points),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
    )
    conf = np.ones((6,), dtype=np.float64)

    result = compute_single_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        SingleBoundaryPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.centerline.shape[0] >= 3


def test_near_field_delta_metrics_ignore_far_field_divergence_when_prefix_stays_aligned():
    previous = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        dtype=np.float64,
    )
    current = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.05], [3.0, 0.10], [4.0, 1.30], [5.0, 1.30]],
        dtype=np.float64,
    )

    near_field = core._near_field_delta_metrics(
        current=current,
        previous=previous,
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        horizon_m=8.0,
    )

    assert near_field["lateral_max_m"] <= 0.10 + 1e-9
    assert near_field["lateral_mean_m"] < 0.05


def test_near_field_delta_metrics_reject_actual_prefix_shift():
    previous = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        dtype=np.float64,
    )
    current = np.array(
        [[0.0, 0.75], [1.0, 0.75], [2.0, 0.80], [3.0, 0.90], [4.0, 1.20]],
        dtype=np.float64,
    )

    near_field = core._near_field_delta_metrics(
        current=current,
        previous=previous,
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        horizon_m=8.0,
    )

    assert near_field["lateral_max_m"] >= 0.75 - 1e-9
