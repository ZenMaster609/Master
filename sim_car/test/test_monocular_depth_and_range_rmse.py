from __future__ import annotations

import math
import importlib.util
import pathlib
import sys

import numpy as np


TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.perception.monocular_depth import estimate_axis_depth_from_bbox_height
from sim_car.perception.range_rmse_analyzer import RangeRMSEAnalyzer
from sim_car.perception.range_rmse_live_plot import RangeRMSELivePlot

TOOLS_ROOT = PACKAGE_ROOT.parent / 'tools'
FIT_MONO_PATH = TOOLS_ROOT / 'fit_monocular_depth.py'


def test_estimate_axis_depth_from_bbox_height_valid():
    depth_m = estimate_axis_depth_from_bbox_height(
        fy_px=600.0,
        cone_height_m=0.3034,
        bbox_height_px=100.0,
    )
    assert depth_m is not None
    assert math.isclose(depth_m, 1.8204, rel_tol=1e-6)


def test_estimate_axis_depth_from_bbox_height_invalid_inputs():
    assert estimate_axis_depth_from_bbox_height(0.0, 0.3034, 100.0) is None
    assert estimate_axis_depth_from_bbox_height(600.0, 0.0, 100.0) is None
    assert estimate_axis_depth_from_bbox_height(600.0, 0.3034, 0.0) is None
    assert estimate_axis_depth_from_bbox_height(float('nan'), 0.3034, 100.0) is None


def test_estimate_axis_depth_from_bbox_height_positive_offset_increases_depth():
    base = estimate_axis_depth_from_bbox_height(600.0, 0.3034, 100.0, bbox_height_offset_px=0.0)
    corrected = estimate_axis_depth_from_bbox_height(600.0, 0.3034, 100.0, bbox_height_offset_px=5.0)
    assert base is not None
    assert corrected is not None
    assert corrected > base


def test_estimate_axis_depth_from_bbox_height_invalid_effective_height():
    assert estimate_axis_depth_from_bbox_height(600.0, 0.3034, 5.0, bbox_height_offset_px=4.5) is None
    assert estimate_axis_depth_from_bbox_height(600.0, 0.3034, 5.0, bbox_height_offset_px=5.0) is None


def test_range_rmse_analyzer_single_source_combined_rmse():
    analyzer = RangeRMSEAnalyzer(range_min_m=0.0, range_max_m=4.0, bin_width_m=1.0)
    analyzer.add_sample(source='monocular', gt_range_m=0.2, error_m=0.5)
    analyzer.add_sample(source='monocular', gt_range_m=0.9, error_m=1.5)

    stats = analyzer.compute_binned_rmse()

    assert stats.total_counts.tolist() == [2, 0, 0, 0]
    assert 'monocular' in stats.source_rmse
    assert math.isclose(float(stats.source_rmse['monocular'][0]), math.sqrt(1.25), rel_tol=1e-6)
    assert np.isnan(stats.source_rmse['monocular'][1])


def test_range_rmse_analyzer_multi_source_and_classification_counts():
    analyzer = RangeRMSEAnalyzer(range_min_m=0.0, range_max_m=4.0, bin_width_m=1.0)
    analyzer.add_sample(
        source='monocular',
        gt_range_m=1.2,
        error_m=0.3,
        predicted_class_id=1,
        ground_truth_class_id=1,
    )
    analyzer.add_sample(
        source='stereo',
        gt_range_m=1.8,
        error_m=0.4,
    )
    analyzer.add_sample(
        source='monocular',
        gt_range_m=1.1,
        error_m=0.5,
        predicted_class_id=0,
        ground_truth_class_id=1,
    )

    stats = analyzer.compute_binned_rmse()

    assert stats.total_counts.tolist() == [0, 3, 0, 0]
    assert math.isclose(float(stats.source_rmse['monocular'][1]), math.sqrt((0.3 ** 2 + 0.5 ** 2) / 2.0), rel_tol=1e-6)
    assert math.isclose(float(stats.source_rmse['stereo'][1]), 0.4, rel_tol=1e-6)
    assert stats.correct_class_count == 1
    assert stats.incorrect_class_count == 1


def test_range_rmse_live_plot_total_rmse_percent_averages_finite_binned_values():
    analyzer = RangeRMSEAnalyzer(range_min_m=0.0, range_max_m=4.0, bin_width_m=1.0)
    analyzer.add_sample(source='monocular', gt_range_m=0.2, error_m=0.1)
    analyzer.add_sample(source='monocular', gt_range_m=1.2, error_m=0.2)
    analyzer.add_sample(source='stereo', gt_range_m=1.7, error_m=0.4)

    stats = analyzer.compute_binned_rmse()
    total_rmse_percent = RangeRMSELivePlot._compute_total_rmse_percent(stats)

    assert total_rmse_percent is not None
    expected = ((0.1 / 0.5) + (0.2 / 1.5) + (0.4 / 1.5)) * 100.0 / 3.0
    assert math.isclose(total_rmse_percent, expected, rel_tol=1e-6)


def test_range_rmse_live_plot_source_rmse_percent_matches_binwise_definition():
    analyzer = RangeRMSEAnalyzer(range_min_m=0.0, range_max_m=4.0, bin_width_m=1.0)
    analyzer.add_sample(source='monocular', gt_range_m=0.2, error_m=0.1)
    analyzer.add_sample(source='monocular', gt_range_m=1.2, error_m=0.2)
    analyzer.add_sample(source='stereo', gt_range_m=1.7, error_m=0.4)

    stats = analyzer.compute_binned_rmse()
    source_rmse_pct = RangeRMSELivePlot._compute_source_rmse_percent(stats.bin_centers, stats.source_rmse)

    assert math.isclose(float(source_rmse_pct['monocular'][0]), 20.0, rel_tol=1e-6)
    assert math.isclose(float(source_rmse_pct['monocular'][1]), (0.2 / 1.5) * 100.0, rel_tol=1e-6)
    assert math.isclose(float(source_rmse_pct['stereo'][1]), (0.4 / 1.5) * 100.0, rel_tol=1e-6)


def test_monocular_fit_tool_recovers_synthetic_offset():
    spec = importlib.util.spec_from_file_location('fit_monocular_depth', FIT_MONO_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    cone_height = 0.3034
    true_offset = 6.0
    rows = []
    for gt_axis in (3.0, 6.0, 10.0, 14.0):
        fy = 600.0
        bbox_height = ((fy * cone_height) / gt_axis) + true_offset
        est_axis = (fy * cone_height) / bbox_height
        rows.append(
            module.FitRow(
                timestamp=0.0,
                session_source='monocular',
                u_center_px=640.0,
                v_center_px=360.0,
                bbox_height_px=bbox_height,
                bbox_width_px=20.0,
                fy_px=fy,
                fx_px=600.0,
                cx_px=640.0,
                cy_px=360.0,
                est_axis_depth_m=est_axis,
                gt_axis_depth_m=gt_axis,
                axis_error_m=est_axis - gt_axis,
                gt_range_m=gt_axis,
                gt_x_cam_m=0.0,
                gt_y_cam_m=0.0,
                gt_z_cam_m=gt_axis,
                est_x_cam_m=0.0,
                est_y_cam_m=0.0,
                est_z_cam_m=est_axis,
                error_xy_m=0.0,
                cone_color='blue',
                predicted_class_id=0,
                ground_truth_class_id=0,
                cone_id='blue_000',
                projection_model='optical_z',
            )
        )

    fit = module.fit_offset(rows, cone_height_m=cone_height, allow_scale=False)
    assert math.isclose(float(fit['offset_px']), true_offset, abs_tol=0.05)
