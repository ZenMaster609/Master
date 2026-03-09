from __future__ import annotations

import pathlib
import sys

import numpy as np

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import sim_car.delaunay_planner_core as core


def test_color_filtering_unknown_gate_behavior():
    points = np.array([
        [4.0, 2.0],
        [4.0, -2.0],
        [6.0, 2.0],
        [6.0, -2.0],
        [8.0, 2.1],
        [8.0, -2.1],
        [10.0, 0.0],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow', 'unknown']
    conf = np.ones((7,), dtype=np.float64)

    cfg = core.CoreConfig(
        min_colored_cones=10,
        use_unknown_cones=True,
        infer_unknown_by_side=False,
        min_required_cones=6,
    )
    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg)
    assert result.filtered_points.shape[0] == 7

    cfg2 = core.CoreConfig(
        min_colored_cones=1,
        use_unknown_cones=True,
        infer_unknown_by_side=False,
        min_required_cones=6,
    )
    result2 = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg2)
    assert result2.filtered_points.shape[0] == 6


def test_edge_selection_has_unique_pairs_and_cross_edges():
    points = np.array([
        [3.0, 1.8],
        [3.0, -1.8],
        [6.0, 2.0],
        [6.1, -2.0],
        [9.0, 2.1],
        [9.0, -2.1],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((6,), dtype=np.float64)

    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, core.CoreConfig())
    tri_edges = result.triangulation_edges
    if tri_edges.shape[0] > 0:
        tuples = {(int(a), int(b)) for a, b in tri_edges}
        assert len(tuples) == tri_edges.shape[0]
    assert result.selected_edges.shape[0] >= 1


def test_midpoint_ordering_and_spacing_with_resampling():
    points = np.array([
        [2.0, 1.5],
        [2.0, -1.5],
        [4.0, 1.5],
        [4.0, -1.5],
        [6.0, 1.5],
        [6.0, -1.5],
        [8.0, 1.5],
        [8.0, -1.5],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((8,), dtype=np.float64)

    cfg = core.CoreConfig(min_spacing_m=0.5, path_resolution_m=0.5, max_path_length_m=20.0)
    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg)

    assert result.centerline.shape[0] > 2
    dx = np.diff(result.centerline[:, 0])
    assert np.all(dx >= -1e-6)


def test_temporal_blend_formula_example():
    alpha = 0.3
    old = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    new = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    smoothed = (alpha * new) + ((1.0 - alpha) * old)
    assert np.allclose(smoothed[:, 1], np.array([0.3, 0.3]))


def test_delaunay_failure_fallback_to_nearest_pairs(monkeypatch):
    points = np.array([
        [3.0, 1.7],
        [3.0, -1.7],
        [5.0, 1.8],
        [5.0, -1.8],
        [7.0, 1.9],
        [7.0, -1.9],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((6,), dtype=np.float64)

    def _fake_build_edges(_points):
        return np.empty((0, 2), dtype=np.int64), False

    monkeypatch.setattr(core, '_build_edges', _fake_build_edges)
    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, core.CoreConfig())
    assert result.used_fallback
    assert result.selected_edges.shape[0] >= 1
    assert result.centerline.shape[0] >= 1


def test_unknown_side_inference_can_be_toggled():
    points = np.array([
        [3.0, 1.8],
        [3.0, -1.8],
        [6.0, 2.0],
        [6.0, -2.0],
        [5.0, 1.4],
        [5.0, -1.4],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'unknown', 'unknown']
    conf = np.ones((6,), dtype=np.float64)

    cfg_off = core.CoreConfig(
        use_unknown_cones=False,
        infer_unknown_by_side=False,
        min_required_cones=4,
    )
    result_off = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg_off)
    assert result_off.filtered_points.shape[0] == 4
    assert result_off.filtered_colors.count('unknown') == 0

    cfg_on = core.CoreConfig(
        use_unknown_cones=False,
        infer_unknown_by_side=True,
        min_required_cones=4,
    )
    result_on = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg_on)
    assert result_on.filtered_points.shape[0] == 6
    assert result_on.filtered_colors.count('unknown') == 0
    assert result_on.filtered_colors.count('blue') == 3
    assert result_on.filtered_colors.count('yellow') == 3


def test_orange_side_inference_uses_clear_lateral_separation():
    points = np.array([
        [3.0, 1.8],
        [3.0, -1.8],
        [6.0, 2.0],
        [6.0, -2.0],
        [5.0, 1.4],
        [5.0, -1.4],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'orange', 'orange']
    conf = np.ones((6,), dtype=np.float64)

    cfg = core.CoreConfig(
        use_unknown_cones=False,
        include_orange=False,
        infer_unknown_by_side=True,
        infer_orange_by_side=True,
        min_required_cones=4,
    )
    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg)

    assert result.filtered_points.shape[0] == 6
    assert result.filtered_colors.count('orange') == 0
    assert result.filtered_colors.count('blue') == 3
    assert result.filtered_colors.count('yellow') == 3


def test_orange_near_center_is_left_unmapped_when_side_is_ambiguous():
    points = np.array([
        [3.0, 1.8],
        [3.0, -1.8],
        [6.0, 2.0],
        [6.0, -2.0],
        [5.0, 0.15],
        [5.0, -0.15],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'orange', 'orange']
    conf = np.ones((6,), dtype=np.float64)

    cfg = core.CoreConfig(
        use_unknown_cones=False,
        include_orange=False,
        infer_unknown_by_side=True,
        infer_orange_by_side=True,
        min_required_cones=4,
    )
    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg)

    assert result.filtered_points.shape[0] == 4
    assert result.filtered_colors.count('orange') == 0
    assert result.filtered_colors.count('blue') == 2
    assert result.filtered_colors.count('yellow') == 2


def test_inferred_orange_pair_cannot_become_the_only_cross_edge_reference():
    points = np.array([
        [2.0, 1.8],
        [2.0, -1.8],
        [5.0, 1.4],
        [5.0, -1.4],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'orange', 'orange']
    conf = np.ones((4,), dtype=np.float64)

    cfg = core.CoreConfig(
        use_unknown_cones=False,
        infer_unknown_by_side=True,
        infer_orange_by_side=True,
        include_orange=False,
        min_required_cones=4,
        min_cross_edges=1,
    )
    result = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg)

    assert result.filtered_points.shape[0] == 4
    selected_edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in result.selected_edges
    }
    assert (0, 1) in selected_edges
    assert (2, 3) not in selected_edges


def test_selected_edge_churn_ratio_and_key_generation():
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
    churn = core.edge_churn_ratio(keys_a, keys_b)

    assert len(keys_a) == 2
    assert len(keys_b) == 2
    assert abs(churn - (2.0 / 3.0)) < 1e-6


def test_centerline_jump_metric_tracks_large_vs_small_shift():
    prev = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float64)
    smooth = np.array([[0.0, 0.05], [2.0, 0.05], [4.0, 0.05], [6.0, 0.05]], dtype=np.float64)
    jumpy = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 1.0], [6.0, 1.0]], dtype=np.float64)

    smooth_jump = core.compute_centerline_jump_max(smooth, prev, horizon_m=8.0)
    jumpy_jump = core.compute_centerline_jump_max(jumpy, prev, horizon_m=8.0)

    assert smooth_jump < 0.1
    assert jumpy_jump > 0.9


def test_deterministic_centerline_under_input_permutation():
    points = np.array([
        [3.0, 1.8],
        [3.0, -1.8],
        [6.0, 2.0],
        [6.0, -2.0],
        [9.0, 2.2],
        [9.0, -2.2],
        [12.0, 2.4],
        [12.0, -2.4],
    ], dtype=np.float64)
    colors = ['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow']
    conf = np.ones((8,), dtype=np.float64)
    cfg = core.CoreConfig(min_required_cones=6)

    base = core.compute_centerline(points, colors, conf, (0.0, 0.0), 0.0, cfg)

    perm = np.array([6, 1, 4, 3, 0, 5, 2, 7], dtype=np.int64)
    perm_points = points[perm]
    perm_colors = [colors[idx] for idx in perm]
    perm_conf = conf[perm]
    permuted = core.compute_centerline(perm_points, perm_colors, perm_conf, (0.0, 0.0), 0.0, cfg)

    assert base.centerline.shape == permuted.centerline.shape
    assert np.allclose(base.centerline, permuted.centerline)


def test_steering_lowpass_step_response_is_bounded():
    alpha = 0.35
    previous = 0.0
    commanded = []
    for _ in range(5):
        previous = (alpha * 1.0) + ((1.0 - alpha) * previous)
        commanded.append(previous)
    assert 0.0 < commanded[0] < 1.0
    assert commanded[-1] < 1.0
    assert commanded[-1] > commanded[0]


def test_tracked_cones_frame_delta_p95_reflects_motion():
    prev = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    curr = np.array([[0.1, 0.0], [1.1, 1.0], [2.1, 2.0]], dtype=np.float64)
    delta = core.tracked_cones_frame_delta_p95(prev, curr)
    assert 0.09 <= delta <= 0.11
