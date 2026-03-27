from __future__ import annotations

import pathlib
import sys

import numpy as np

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.cones.tracking.fusion import resolve_boundary_colors_for_planning  # noqa: E402
from sim_car.planning.midpoint_planner_core import (  # noqa: E402
    MidpointPlannerConfig,
    MidpointPlannerPrior,
    compute_midpoint_centerline,
    update_track_width_estimate,
)


def _cfg(**overrides) -> MidpointPlannerConfig:
    cfg = MidpointPlannerConfig(
        min_required_cones=4,
        min_chain_length=3,
        min_pair_count=3,
        min_path_points=3,
        min_forward_extent_m=2.0,
        path_resolution_m=0.5,
        max_path_length_m=20.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_straight_track_uses_midpoint_mode():
    points = np.array(
        [[2.0, 1.8], [2.0, -1.8], [4.0, 1.8], [4.0, -1.8], [6.0, 1.8], [6.0, -1.8]],
        dtype=np.float64,
    )
    colors = ["blue", "yellow", "blue", "yellow", "blue", "yellow"]
    conf = np.ones((6,), dtype=np.float64)

    result = compute_midpoint_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        MidpointPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.planner_mode == "midpoint"
    assert result.accepted_pair_count >= 3
    assert np.allclose(result.centerline[:, 1], 0.0, atol=1e-6)


def test_missing_opposite_boundary_returns_no_midpoint_path():
    points = np.array([[2.0, 1.8], [4.0, 2.1], [6.0, 2.6], [8.0, 3.3]], dtype=np.float64)
    colors = ["blue", "blue", "blue", "blue"]
    conf = np.ones((4,), dtype=np.float64)

    result = compute_midpoint_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        MidpointPlannerPrior(previous_width_m=3.6),
    )

    assert result.planner_mode == "none"
    assert result.centerline.shape[0] == 0
    assert "midpoint" in result.status


def test_width_estimate_updates_slowly_and_clamps_delta():
    cfg = _cfg(initial_width_m=3.6, width_filter_alpha=0.2, max_width_delta_per_update_m=0.15)
    updated = update_track_width_estimate(3.6, 4.5, cfg)
    assert abs(updated - 3.63) < 1e-9


def test_all_orange_track_is_resolved_to_midpoint_path():
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

    result = compute_midpoint_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        MidpointPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.planner_mode == "midpoint"
