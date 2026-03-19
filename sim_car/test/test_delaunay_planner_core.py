from __future__ import annotations

import pathlib
import sys

import numpy as np

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import sim_car.planning.delaunay_planner_core as core


def _cfg(**overrides) -> core.CoreConfig:
    cfg = core.CoreConfig(
        min_required_cones=4,
        min_cross_edges=2,
        max_near_field_lateral_jump_m=0.6,
        near_field_midpoint_count=5,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_pre_turn_ambiguity_prefers_local_zig_zag():
    points = np.array([
        [2.0, 2.0],
        [2.0, -2.0],
        [4.0, 2.2],
        [4.0, -2.0],
        [7.0, 4.5],
        [6.0, -1.6],
        [8.0, 5.5],
        [8.0, -0.4],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((8,), dtype=np.float64)
    prior = core.CorePrior(
        previous_midpoints_raw=np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float64),
        previous_width_m=4.0,
    )

    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, _cfg(), prior=prior)

    assert result.status == 'ok'
    assert result.selected_chain_length >= 3
    assert np.all(np.abs(result.midpoints_raw[:, 1]) < 1.0)


def test_delaunay_miss_recovered_by_local_fallback(monkeypatch):
    points = np.array([
        [2.0, 1.8],
        [2.0, -1.8],
        [4.0, 1.9],
        [4.0, -1.9],
        [6.0, 2.0],
        [6.0, -2.0],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((6,), dtype=np.float64)

    monkeypatch.setattr(
        core,
        '_build_edges',
        lambda _points: (np.empty((0, 2), dtype=np.int64), False),
    )
    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, _cfg())

    assert result.status == 'ok'
    assert result.used_fallback
    assert result.selected_chain_length >= 2


def test_along_track_diagonal_rejected_by_orientation_gate():
    points = np.array([
        [2.0, 1.0],
        [4.0, 1.1],
        [6.0, 1.2],
        [2.2, -1.0],
        [4.2, -1.1],
        [6.2, -1.2],
    ], dtype=np.float64)
    colors = ['blue', 'blue', 'blue', 'yellow', 'yellow', 'yellow']
    conf = np.ones((6,), dtype=np.float64)

    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, _cfg())

    assert result.status != 'ok'
    assert result.reject_counts['orientation'] > 0


def test_near_field_teleport_rejected_against_previous_raw_chain():
    points = np.array([
        [2.0, 3.0],
        [2.0, -0.6],
        [4.0, 3.1],
        [4.0, -0.5],
        [6.0, 3.2],
        [6.0, -0.4],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((6,), dtype=np.float64)
    prior = core.CorePrior(
        previous_midpoints_raw=np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float64),
        previous_width_m=3.6,
    )

    result = core.compute_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(max_near_field_lateral_jump_m=0.4),
        prior=prior,
    )

    assert result.status != 'ok'
    assert result.reject_counts['near_field_continuity'] > 0


def test_real_corner_entry_allowed_when_geometry_is_consistent():
    points = np.array([
        [2.0, 1.8],
        [2.0, -1.8],
        [4.0, 2.0],
        [4.0, -1.6],
        [6.0, 2.5],
        [6.0, -1.1],
        [8.0, 3.2],
        [8.0, -0.4],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((8,), dtype=np.float64)
    prior = core.CorePrior(
        previous_midpoints_raw=np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float64),
        previous_width_m=3.6,
    )

    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, _cfg(), prior=prior)

    assert result.status == 'ok'
    assert result.near_field_lateral_max_m < 0.6


def test_seed_midpoint_too_far_is_rejected():
    points = np.array([
        [12.0, 1.8],
        [12.0, -1.8],
        [14.0, 1.9],
        [14.0, -1.9],
        [16.0, 2.0],
        [16.0, -2.0],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((6,), dtype=np.float64)

    result = core.compute_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(max_seed_midpoint_distance_m=6.0),
    )

    assert result.status != 'ok'
    assert result.reject_counts['seed_distance'] > 0


def test_two_diagonal_near_field_chain_is_kept_instead_of_dropping_to_zero():
    points = np.array([
        [2.0, 1.8],
        [2.0, -1.8],
        [4.0, 1.8],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue']
    conf = np.ones((3,), dtype=np.float64)

    result = core.compute_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(min_required_cones=3, min_cross_edges=3),
    )

    assert result.status == 'ok'
    assert result.selected_chain_length == 2
    assert result.seed_midpoint_distance_m <= 3.5


def test_candidate_progress_is_measured_from_vehicle_not_previous_seed():
    points = np.array([
        [2.0, 1.8],
        [2.0, -1.8],
        [4.0, 1.8],
        [4.0, -1.8],
        [6.0, 1.8],
        [6.0, -1.8],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((6,), dtype=np.float64)
    prior = core.CorePrior(
        previous_midpoints_raw=np.array([[4.0, 0.0], [6.0, 0.0], [8.0, 0.0]], dtype=np.float64),
        previous_width_m=3.6,
    )

    result = core.compute_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(min_cross_edges=3),
        prior=prior,
    )

    assert result.status == 'ok'
    assert result.reject_counts['progress'] == 0
    assert result.selected_chain_length >= 3
    assert np.allclose(result.midpoints_raw[0], np.array([2.0, 0.0]))


def test_startup_tangent_comes_from_cone_geometry_not_vehicle_yaw():
    left_boundary = np.array([
        [5.07326, -10.121974],
        [1.441203, -11.932402],
        [-2.466965, -13.999331],
    ], dtype=np.float64)
    right_boundary = np.array([
        [4.67237, -5.183963],
        [1.824002, -6.670729],
        [-1.94798, -8.549423],
    ], dtype=np.float64)
    vehicle_xy = (9.58, -5.2)
    vehicle_yaw = 3.75
    rejected_edge = np.array([1.441203, -11.932402]) - np.array([1.824002, -6.670729])

    startup_tangent = core._estimate_startup_tangent(
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
    )
    yaw_tangent = core._yaw_unit(vehicle_yaw)
    startup_alignment = abs(float(np.dot(core._unit_vector(rejected_edge), startup_tangent)))
    yaw_alignment = abs(float(np.dot(core._unit_vector(rejected_edge), yaw_tangent)))

    assert startup_alignment < 0.55
    assert yaw_alignment > 0.55


def test_startup_geometry_accepts_local_chain_even_if_vehicle_yaw_is_off():
    points = np.array([
        [4.67237, -5.183963],
        [5.07326, -10.121974],
        [1.824002, -6.670729],
        [1.441203, -11.932402],
        [-1.94798, -8.549423],
        [-2.466965, -13.999331],
    ], dtype=np.float64)
    colors = ['yellow', 'blue', 'yellow', 'blue', 'yellow', 'blue']
    conf = np.ones((6,), dtype=np.float64)

    result = core.compute_centerline(
        points,
        colors,
        conf,
        (9.58, -5.2),
        3.75,
        _cfg(max_seed_midpoint_distance_m=12.0),
    )

    assert result.status == 'ok'
    assert result.selected_chain_length >= 2
    assert result.reject_counts['orientation'] == 0
    assert result.reject_counts['seed_distance'] == 0


def test_small_track_spawn_width_gate_above_five_meters_is_needed():
    points = np.array([
        [4.67237, -5.183963],
        [5.07326, -10.121974],
        [1.824002, -6.670729],
        [1.441203, -11.932402],
        [-1.94798, -8.549423],
        [-2.466965, -13.999331],
    ], dtype=np.float64)
    colors = ['yellow', 'blue', 'yellow', 'blue', 'yellow', 'blue']
    conf = np.ones((6,), dtype=np.float64)

    too_tight = core.compute_centerline(
        points,
        colors,
        conf,
        (9.58, -5.2),
        3.75,
        _cfg(max_cross_edge_m=5.0, max_seed_midpoint_distance_m=12.0, min_cross_edges=3),
    )
    corrected = core.compute_centerline(
        points,
        colors,
        conf,
        (9.58, -5.2),
        3.75,
        _cfg(max_cross_edge_m=5.4, max_seed_midpoint_distance_m=12.0, min_cross_edges=3),
    )

    assert too_tight.status != 'ok'
    assert too_tight.reject_counts['width'] > 0
    assert corrected.status == 'ok'
    assert corrected.selected_chain_length >= 4


def test_boundary_near_field_pairs_keep_local_ladder_in_front_of_vehicle():
    points = np.array([
        [5.07326, -10.121974],   # left 0
        [1.441203, -11.932402],  # left 1
        [-2.466965, -13.999331], # left 2
        [4.67237, -5.183963],    # right 0
        [1.824002, -6.670729],   # right 1
        [-1.94798, -8.549423],   # right 2
    ], dtype=np.float64)
    left_idx = np.array([0, 1, 2], dtype=np.int64)
    right_idx = np.array([3, 4, 5], dtype=np.int64)
    tangent = np.array([-0.89, -0.45], dtype=np.float64)

    pairs = core._build_boundary_near_field_pairs(
        points=points,
        left_boundary_idx=left_idx,
        right_boundary_idx=right_idx,
        vehicle_xy=(9.58, -5.2),
        reference_tangent=tangent,
        config=_cfg(max_seed_midpoint_distance_m=8.0, max_same_side_step_m=5.0),
    )

    as_pairs = {tuple(map(int, pair)) for pair in pairs.tolist()}
    assert (0, 3) in as_pairs
    assert (0, 4) in as_pairs
    assert (1, 3) in as_pairs
    assert (1, 4) in as_pairs


def test_boundary_order_prefers_local_same_side_continuation():
    points = np.array([
        [4.80910279, -5.04329004],
        [1.95319712, -6.71117767],
        [-1.75103629, -8.62416587],
        [-3.93795828, 8.46931923],
    ], dtype=np.float64)
    colors = ['yellow', 'yellow', 'yellow', 'yellow']
    vehicle_xy = (3.233129609988708, -8.156905936403199)
    tangent = np.array([-0.91, -0.41], dtype=np.float64)

    ordered_idx = core._order_boundary_indices(
        points,
        colors,
        'yellow',
        vehicle_xy,
        tangent,
        max_step_m=6.25,
    )

    assert ordered_idx.tolist()[:2] == [1, 2]
    assert 3 not in ordered_idx.tolist()[:2]


def test_shared_vertex_alternation_still_holds():
    points = np.array([
        [2.0, 1.8],
        [2.0, -1.8],
        [4.0, 1.9],
        [4.0, -1.9],
        [6.0, 2.0],
        [6.0, -2.0],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((6,), dtype=np.float64)

    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, _cfg())

    assert result.status == 'ok'
    assert result.selected_edges.shape[0] >= 3
    for prev, curr in zip(result.selected_edges[:-1], result.selected_edges[1:]):
        shared = set(map(int, prev)).intersection(set(map(int, curr)))
        assert len(shared) == 1


def test_direct_color_beats_inferred_when_both_valid():
    points = np.array([
        [2.0, 1.8],
        [2.0, -1.8],
        [4.0, 2.0],
        [4.0, -2.0],
        [4.2, 2.05],
        [6.0, -2.2],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'unknown', 'yellow']
    conf = np.ones((6,), dtype=np.float64)

    result = core.compute_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(infer_unknown_by_side=True, use_unknown_cones=True),
    )

    assert result.status == 'ok'
    assert np.any(np.all(np.isclose(result.midpoints_raw, np.array([4.0, 0.0])), axis=1))


def test_width_prior_drift_is_capped():
    updated = core._update_expected_width(4.0, 5.0, 0.2)
    assert abs(updated - 4.2) < 1e-6


def test_selected_edge_churn_uses_canonical_diagonal_identity():
    points = np.array([
        [2.0, 1.0],
        [2.0, -1.0],
        [4.0, 1.0],
        [4.0, -1.0],
    ], dtype=np.float64)
    edges_a = np.array([[0, 1], [2, 3]], dtype=np.int64)
    edges_b = np.array([[0, 1], [1, 2]], dtype=np.int64)

    keys_a = core.selected_edge_keys(points=points, edges=edges_a, quantization_m=0.05)
    keys_b = core.selected_edge_keys(points=points, edges=edges_b, quantization_m=0.05)

    assert core.edge_churn_count(keys_a, keys_b) == 2
    assert abs(core.edge_churn_ratio(keys_a, keys_b) - (2.0 / 3.0)) < 1e-6


def test_resample_before_smooth_preserves_even_spacing():
    points = np.array([
        [2.0, 1.8],
        [2.0, -1.8],
        [4.0, 1.9],
        [4.0, -1.9],
        [6.0, 2.0],
        [6.0, -2.0],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((6,), dtype=np.float64)

    result = core.compute_centerline(
        points,
        colors,
        conf,
        (0.0, 0.0),
        0.0,
        _cfg(path_resolution_m=0.5, min_cross_edges=2),
    )

    assert result.status == 'ok'
    seg = np.diff(result.centerline, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    assert np.all(seg_len > 0.15)
    assert np.all(seg_len < 0.8)
