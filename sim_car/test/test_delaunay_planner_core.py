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
