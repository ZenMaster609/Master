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
    _BoundaryPair,
    MidpointPlannerPrior,
    _BoundaryChain,
    _order_pairs_into_midpoint_chain,
    _pair_boundary_chains,
    _trim_pairs_by_midpoint_step_length,
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


def test_prevalidation_centerline_is_preserved_when_short_path_is_rejected():
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
        _cfg(min_path_points=8),
        MidpointPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "path has too few points"
    assert result.centerline.shape[0] == 0
    assert result.prevalidation_centerline.shape[0] >= 2


def test_turning_boundary_with_small_x_regression_keeps_midpoint_pairs():
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
    assert result.accepted_pair_count >= 3


def test_deeper_turn_keeps_extending_midpoint_boundary_chains():
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
    assert result.left_chain_length >= 4
    assert result.right_chain_length >= 4
    assert result.accepted_pair_count >= 3


def test_midpoint_pairing_is_not_blocked_by_boundary_chain_length_requirement():
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
        _cfg(min_chain_length=10),
        MidpointPlannerPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.accepted_pair_count >= 3
    assert np.allclose(result.centerline[:, 1], 0.0, atol=1e-6)


def test_pairing_allows_small_negative_inward_projection_with_tolerance():
    filtered_points = np.array([[2.0, 1.8], [2.0, -1.6]], dtype=np.float64)
    filtered_local = np.array(filtered_points, copy=True)
    filtered_track_ids = np.array([101, 202], dtype=np.int64)

    left_chain = _BoundaryChain(
        filtered_indices=np.array([0], dtype=np.int64),
        global_points=np.array([[2.0, 1.8]], dtype=np.float64),
        local_points=np.array([[2.0, 1.8]], dtype=np.float64),
        tangents_local=np.array([[-0.02, 0.9998]], dtype=np.float64),
        mean_heading_change_rad=0.0,
        forward_extent_m=0.0,
    )
    right_chain = _BoundaryChain(
        filtered_indices=np.array([1], dtype=np.int64),
        global_points=np.array([[2.0, -1.6]], dtype=np.float64),
        local_points=np.array([[2.0, -1.6]], dtype=np.float64),
        tangents_local=np.array([[1.0, 0.0]], dtype=np.float64),
        mean_heading_change_rad=0.0,
        forward_extent_m=0.0,
    )

    strict_cfg = _cfg(pair_inward_projection_tolerance_m=0.0)
    strict_pairs, _strict_candidates, _strict_unknown, strict_rejects = _pair_boundary_chains(
        filtered_points=filtered_points,
        filtered_local=filtered_local,
        filtered_track_ids=filtered_track_ids,
        left_chain=left_chain,
        right_chain=right_chain,
        unknown_indices=np.empty((0,), dtype=np.int64),
        expected_width_m=3.6,
        config=strict_cfg,
        prior=None,
    )

    tolerant_cfg = _cfg(pair_inward_projection_tolerance_m=0.15)
    tolerant_pairs, _tol_candidates, _tol_unknown, tolerant_rejects = _pair_boundary_chains(
        filtered_points=filtered_points,
        filtered_local=filtered_local,
        filtered_track_ids=filtered_track_ids,
        left_chain=left_chain,
        right_chain=right_chain,
        unknown_indices=np.empty((0,), dtype=np.int64),
        expected_width_m=3.6,
        config=tolerant_cfg,
        prior=None,
    )

    assert len(strict_pairs) == 0
    assert strict_rejects["wrong_side"] >= 1
    assert len(tolerant_pairs) == 1
    assert tolerant_rejects["wrong_side"] == 0


def test_midpoint_chain_is_trimmed_at_first_oversized_midpoint_jump():
    pairs = [
        _BoundaryPair(
            left_filtered_idx=0,
            right_filtered_idx=1,
            left_track_id=10,
            right_track_id=20,
            left_global=np.array([2.0, 1.8], dtype=np.float64),
            right_global=np.array([2.0, -1.8], dtype=np.float64),
            left_local=np.array([2.0, 1.8], dtype=np.float64),
            right_local=np.array([2.0, -1.8], dtype=np.float64),
            width_m=3.6,
        ),
        _BoundaryPair(
            left_filtered_idx=2,
            right_filtered_idx=3,
            left_track_id=11,
            right_track_id=21,
            left_global=np.array([5.0, 1.8], dtype=np.float64),
            right_global=np.array([5.0, -1.8], dtype=np.float64),
            left_local=np.array([5.0, 1.8], dtype=np.float64),
            right_local=np.array([5.0, -1.8], dtype=np.float64),
            width_m=3.6,
        ),
        _BoundaryPair(
            left_filtered_idx=4,
            right_filtered_idx=5,
            left_track_id=12,
            right_track_id=22,
            left_global=np.array([13.0, 1.8], dtype=np.float64),
            right_global=np.array([13.0, -1.8], dtype=np.float64),
            left_local=np.array([13.0, 1.8], dtype=np.float64),
            right_local=np.array([13.0, -1.8], dtype=np.float64),
            width_m=3.6,
        ),
    ]

    trimmed = _trim_pairs_by_midpoint_step_length(
        pairs,
        max_segment_length_m=6.0,
    )

    assert len(trimmed) == 2
    assert trimmed[0].left_track_id == 10
    assert trimmed[1].left_track_id == 11


def test_midpoint_chain_order_prefers_forward_geometric_continuation():
    pairs = [
        _BoundaryPair(
            left_filtered_idx=0,
            right_filtered_idx=1,
            left_track_id=10,
            right_track_id=20,
            left_global=np.array([2.0, 1.8], dtype=np.float64),
            right_global=np.array([2.0, -1.8], dtype=np.float64),
            left_local=np.array([2.0, 1.8], dtype=np.float64),
            right_local=np.array([2.0, -1.8], dtype=np.float64),
            width_m=3.6,
        ),
        _BoundaryPair(
            left_filtered_idx=2,
            right_filtered_idx=3,
            left_track_id=30,
            right_track_id=40,
            left_global=np.array([3.5, -0.2], dtype=np.float64),
            right_global=np.array([3.7, -4.2], dtype=np.float64),
            left_local=np.array([3.5, -0.2], dtype=np.float64),
            right_local=np.array([3.7, -4.2], dtype=np.float64),
            width_m=4.0,
        ),
        _BoundaryPair(
            left_filtered_idx=4,
            right_filtered_idx=5,
            left_track_id=11,
            right_track_id=21,
            left_global=np.array([4.0, 1.9], dtype=np.float64),
            right_global=np.array([4.0, -1.7], dtype=np.float64),
            left_local=np.array([4.0, 1.9], dtype=np.float64),
            right_local=np.array([4.0, -1.7], dtype=np.float64),
            width_m=3.6,
        ),
    ]

    ordered = _order_pairs_into_midpoint_chain(
        pairs,
        max_segment_length_m=6.0,
    )

    assert [pair.left_track_id for pair in ordered] == [10, 11]


def test_midpoint_chain_order_hands_off_from_vehicle_forward_to_recent_midline_trend():
    pairs = [
        _BoundaryPair(
            left_filtered_idx=0,
            right_filtered_idx=1,
            left_track_id=10,
            right_track_id=20,
            left_global=np.array([1.0, 1.8], dtype=np.float64),
            right_global=np.array([1.0, -1.8], dtype=np.float64),
            left_local=np.array([1.0, 1.8], dtype=np.float64),
            right_local=np.array([1.0, -1.8], dtype=np.float64),
            width_m=3.6,
        ),
        _BoundaryPair(
            left_filtered_idx=2,
            right_filtered_idx=3,
            left_track_id=11,
            right_track_id=21,
            left_global=np.array([2.7, 1.9], dtype=np.float64),
            right_global=np.array([2.7, -1.7], dtype=np.float64),
            left_local=np.array([2.7, 1.9], dtype=np.float64),
            right_local=np.array([2.7, -1.7], dtype=np.float64),
            width_m=3.6,
        ),
        _BoundaryPair(
            left_filtered_idx=4,
            right_filtered_idx=5,
            left_track_id=12,
            right_track_id=22,
            left_global=np.array([3.8, 2.4], dtype=np.float64),
            right_global=np.array([3.8, -1.2], dtype=np.float64),
            left_local=np.array([3.8, 2.4], dtype=np.float64),
            right_local=np.array([3.8, -1.2], dtype=np.float64),
            width_m=3.6,
        ),
        _BoundaryPair(
            left_filtered_idx=6,
            right_filtered_idx=7,
            left_track_id=13,
            right_track_id=23,
            left_global=np.array([3.4, 3.8], dtype=np.float64),
            right_global=np.array([3.4, 0.2], dtype=np.float64),
            left_local=np.array([3.4, 3.8], dtype=np.float64),
            right_local=np.array([3.4, 0.2], dtype=np.float64),
            width_m=3.6,
        ),
        _BoundaryPair(
            left_filtered_idx=8,
            right_filtered_idx=9,
            left_track_id=30,
            right_track_id=40,
            left_global=np.array([5.5, 1.0], dtype=np.float64),
            right_global=np.array([5.5, -2.6], dtype=np.float64),
            left_local=np.array([5.5, 1.0], dtype=np.float64),
            right_local=np.array([5.5, -2.6], dtype=np.float64),
            width_m=3.6,
        ),
    ]

    ordered = _order_pairs_into_midpoint_chain(
        pairs,
        config=_cfg(
            max_midpoint_segment_length_m=6.0,
            midpoint_order_reference_handoff_m=1.5,
            midpoint_order_history_size=2,
            midpoint_order_backtrack_tolerance_m=0.1,
        ),
    )

    assert [pair.left_track_id for pair in ordered] == [10, 11, 12, 13]
