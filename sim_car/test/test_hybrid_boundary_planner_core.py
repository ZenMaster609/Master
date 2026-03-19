from __future__ import annotations

import pathlib
import sys

import numpy as np

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.planning.hybrid_boundary_planner_core import (
    HybridBoundaryConfig,
    HybridBoundaryPrior,
    _build_boundary_chain,
    compute_hybrid_boundary_centerline,
    update_track_width_estimate,
)


def _cfg(**overrides) -> HybridBoundaryConfig:
    cfg = HybridBoundaryConfig(
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
        [
            [2.0, 1.8],
            [2.0, -1.8],
            [4.0, 1.8],
            [4.0, -1.8],
            [6.0, 1.8],
            [6.0, -1.8],
            [8.0, 1.8],
            [8.0, -1.8],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "yellow", "blue", "yellow", "blue", "yellow", "blue", "yellow"]
    conf = np.ones((8,), dtype=np.float64)

    result = compute_hybrid_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(),
        HybridBoundaryPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.planner_mode == "midpoint"
    assert result.accepted_pair_count >= 3
    assert result.left_chain_length == 4
    assert result.right_chain_length == 4
    assert np.allclose(result.centerline[:, 1], 0.0, atol=1e-6)


def test_single_boundary_dropout_uses_offset_fallback():
    points = np.array(
        [
            [2.0, 1.8],
            [4.0, 2.1],
            [6.0, 2.6],
            [8.0, 3.3],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "blue", "blue", "blue"]
    conf = np.ones((4,), dtype=np.float64)

    result = compute_hybrid_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(min_required_cones=4, min_chain_length=3),
        HybridBoundaryPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.planner_mode == "single_boundary"
    assert result.used_fallback
    assert result.raw_offset_path.shape[0] >= 3
    assert result.centerline.shape[0] >= 3


def test_force_single_boundary_skips_midpoint_even_with_both_sides_present():
    points = np.array(
        [
            [2.0, 1.8],
            [2.0, -1.8],
            [4.0, 1.8],
            [4.0, -1.8],
            [6.0, 1.8],
            [6.0, -1.8],
            [8.0, 1.8],
            [8.0, -1.8],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "yellow", "blue", "yellow", "blue", "yellow", "blue", "yellow"]
    conf = np.ones((8,), dtype=np.float64)

    result = compute_hybrid_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(force_single_boundary=True),
        HybridBoundaryPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.planner_mode == "single_boundary"
    assert result.accepted_pair_count == 0
    assert result.used_fallback
    assert result.raw_offset_path.shape[0] >= 3
    assert result.centerline.shape[0] >= 3


def test_force_single_boundary_accepts_shorter_single_side_path():
    points = np.array(
        [
            [2.0, 1.8],
            [3.2, 2.0],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "blue"]
    conf = np.ones((2,), dtype=np.float64)

    result = compute_hybrid_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(
            force_single_boundary=True,
            min_required_cones=2,
            min_chain_length=3,
            min_path_points=4,
            min_forward_extent_m=2.0,
        ),
        HybridBoundaryPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.planner_mode == "single_boundary"
    assert result.used_fallback
    assert result.centerline.shape[0] >= 2


def test_width_estimate_updates_slowly_and_clamps_delta():
    cfg = _cfg(
        initial_width_m=3.6,
        min_width_m=2.4,
        max_width_m=4.8,
        width_filter_alpha=0.15,
        max_width_delta_per_update_m=0.2,
    )

    updated = update_track_width_estimate(3.6, 4.5, cfg)

    assert abs(updated - 3.63) < 1e-9


def test_unknown_cones_can_complete_missing_opposite_side_pairs():
    points = np.array(
        [
            [2.0, 1.8],
            [4.0, 1.8],
            [6.0, 1.8],
            [8.0, 1.8],
            [2.0, -1.8],
            [4.0, -1.8],
            [6.0, -1.8],
            [8.0, -1.8],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "blue", "blue", "blue", "yellow", "yellow", "unknown", "unknown"]
    conf = np.ones((8,), dtype=np.float64)

    result = compute_hybrid_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(
            allow_unknown_pair_completion=True,
            min_pair_count=4,
            unknown_pair_search_radius_m=0.6,
            unknown_pair_max_longitudinal_error_m=0.5,
            unknown_pair_max_width_error_m=0.5,
        ),
        HybridBoundaryPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.planner_mode == "midpoint"
    assert result.accepted_pair_count == 4
    assert result.unknown_pair_count == 2


def test_bad_unknown_cones_do_not_force_midpoint_pairs():
    points = np.array(
        [
            [2.0, 1.8],
            [4.0, 2.1],
            [6.0, 2.6],
            [8.0, 3.3],
            [2.0, -1.8],
            [5.5, 2.4],
            [7.5, 3.8],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "blue", "blue", "blue", "yellow", "unknown", "unknown"]
    conf = np.ones((7,), dtype=np.float64)

    result = compute_hybrid_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(
            allow_unknown_pair_completion=True,
            unknown_pair_search_radius_m=0.7,
            unknown_pair_max_longitudinal_error_m=0.5,
            unknown_pair_max_width_error_m=0.4,
        ),
        HybridBoundaryPrior(previous_width_m=3.6),
    )

    assert result.status == "ok"
    assert result.planner_mode == "single_boundary"
    assert result.unknown_pair_count == 0


def test_near_field_jump_is_rejected_against_previous_path():
    points = np.array(
        [
            [2.0, 2.8],
            [2.0, -0.8],
            [4.0, 2.9],
            [4.0, -0.7],
            [6.0, 3.0],
            [6.0, -0.6],
            [8.0, 3.1],
            [8.0, -0.5],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "yellow", "blue", "yellow", "blue", "yellow", "blue", "yellow"]
    conf = np.ones((8,), dtype=np.float64)
    prior = HybridBoundaryPrior(
        previous_centerline=np.array(
            [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0], [7.0, 0.0]],
            dtype=np.float64,
        ),
        previous_width_m=3.6,
    )

    result = compute_hybrid_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(max_near_field_lateral_jump_m=0.4),
        prior,
    )

    assert result.status == "near-field continuity rejected fresh path"
    assert result.reject_counts["near_field_continuity"] > 0


def test_heading_delta_limit_rejects_kinky_path():
    points = np.array(
        [
            [2.0, 1.8],
            [2.0, -1.8],
            [4.0, 2.0],
            [4.0, -1.6],
            [6.0, 2.8],
            [6.0, -0.8],
            [8.0, 4.0],
            [8.0, 0.4],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "yellow", "blue", "yellow", "blue", "yellow", "blue", "yellow"]
    conf = np.ones((8,), dtype=np.float64)

    result = compute_hybrid_boundary_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(max_heading_delta_rad=0.08),
        HybridBoundaryPrior(previous_width_m=3.6),
    )

    assert result.status == "path heading delta exceeded limit"
    assert result.reject_counts["midpoint_kink"] > 0


def test_identical_input_produces_identical_output():
    points = np.array(
        [
            [2.0, 1.8],
            [2.0, -1.8],
            [4.0, 1.9],
            [4.0, -1.7],
            [6.0, 2.1],
            [6.0, -1.5],
            [8.0, 2.5],
            [8.0, -1.1],
        ],
        dtype=np.float64,
    )
    colors = ["blue", "yellow", "blue", "yellow", "blue", "yellow", "blue", "yellow"]
    conf = np.ones((8,), dtype=np.float64)
    cfg = _cfg()
    prior = HybridBoundaryPrior(previous_width_m=3.6)

    first = compute_hybrid_boundary_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg, prior)
    second = compute_hybrid_boundary_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg, prior)

    assert first.status == second.status == "ok"
    assert first.planner_mode == second.planner_mode
    assert np.allclose(first.selected_edges, second.selected_edges)
    assert np.allclose(first.centerline, second.centerline)


def test_boundary_chain_prefers_nearest_next_cone_without_skipping():
    filtered_points = np.array(
        [
            [2.0, 1.8],
            [4.0, 1.9],
            [6.0, 2.0],
            [8.0, 2.1],
        ],
        dtype=np.float64,
    )
    filtered_local = np.array(filtered_points, copy=True)
    side_indices = np.array([0, 1, 2, 3], dtype=np.int64)

    chain = _build_boundary_chain(filtered_points, filtered_local, side_indices, _cfg())

    expected = np.array(
        [[2.0, 1.8], [4.0, 1.9], [6.0, 2.0], [8.0, 2.1]],
        dtype=np.float64,
    )
    assert np.allclose(chain.local_points, expected)


def test_boundary_chain_keeps_progressing_through_sharp_side_turn():
    filtered_points = np.array(
        [
            [2.0, 1.8],
            [3.8, 2.1],
            [3.9, 4.2],
            [3.8, 6.4],
        ],
        dtype=np.float64,
    )
    filtered_local = np.array(filtered_points, copy=True)
    side_indices = np.array([0, 1, 2, 3], dtype=np.int64)

    chain = _build_boundary_chain(
        filtered_points,
        filtered_local,
        side_indices,
        _cfg(max_heading_change_rad=1.4, min_forward_progress_m=0.1),
    )

    assert chain.local_points.shape[0] == 4
    assert np.allclose(chain.local_points[0], [2.0, 1.8])
    assert np.allclose(chain.local_points[-1], [3.8, 6.4])
