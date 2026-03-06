from __future__ import annotations

import math
import pathlib
import sys

import numpy as np


TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from vehicle_plotter.plotting.offline_cone_plotter import OfflineConePlotter


def test_compute_total_rmse_percent_matches_live_definition():
    bin_centers = np.asarray([0.5, 1.5, 2.5], dtype=np.float32)
    rmse_by_source = {
        'monocular': np.asarray([0.1, 0.2, np.nan], dtype=np.float32),
        'stereo': np.asarray([np.nan, 0.4, np.nan], dtype=np.float32),
    }

    value = OfflineConePlotter._compute_total_rmse_percent(bin_centers, rmse_by_source)

    assert value is not None
    expected = ((0.1 / 0.5) + (0.2 / 1.5) + (0.4 / 1.5)) * 100.0 / 3.0
    assert math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_compute_source_rmse_percent_matches_binwise_definition():
    bin_centers = np.asarray([0.5, 1.5, 2.5], dtype=np.float32)
    rmse_by_source = {
        'monocular': np.asarray([0.1, 0.2, np.nan], dtype=np.float32),
        'stereo': np.asarray([np.nan, 0.4, np.nan], dtype=np.float32),
    }

    source_rmse_pct = OfflineConePlotter._compute_source_rmse_percent(bin_centers, rmse_by_source)

    assert math.isclose(float(source_rmse_pct['monocular'][0]), 20.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(source_rmse_pct['monocular'][1]), (0.2 / 1.5) * 100.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(source_rmse_pct['stereo'][1]), (0.4 / 1.5) * 100.0, rel_tol=1e-6, abs_tol=1e-6)


def test_compute_classification_counts_ignores_missing_values():
    predicted = np.asarray([1.0, 2.0, np.nan, 0.0], dtype=np.float32)
    ground_truth = np.asarray([1.0, 0.0, 3.0, np.nan], dtype=np.float32)

    correct_count, incorrect_count = OfflineConePlotter._compute_classification_counts(
        predicted,
        ground_truth,
    )

    assert correct_count == 1
    assert incorrect_count == 1


def test_generate_range_rmse_plot_writes_png(tmp_path):
    session_path = tmp_path / 'session'
    logs_path = session_path / 'logs'
    logs_path.mkdir(parents=True)
    (session_path / 'plots').mkdir(parents=True)

    csv_path = logs_path / 'cone_range_rmse_samples.csv'
    csv_path.write_text(
        '\n'.join(
            [
                'timestamp,source,gt_range_m,error_m,predicted_class_id,ground_truth_class_id',
                '1.0,monocular,0.2,0.1,1,1',
                '1.1,monocular,1.2,0.2,0,1',
                '1.2,stereo,1.7,0.4,,',
            ]
        ),
        encoding='utf-8',
    )

    output_path = OfflineConePlotter(session_path).generate_range_rmse_plot()

    assert output_path is not None
    assert output_path.exists()
    assert output_path.suffix == '.png'


def test_generate_combined_range_rmse_plot_writes_png(tmp_path):
    session_path = tmp_path / 'session'
    logs_path = session_path / 'logs'
    logs_path.mkdir(parents=True)
    (session_path / 'plots').mkdir(parents=True)

    (logs_path / 'cone_range_rmse_samples.csv').write_text(
        '\n'.join(
            [
                'timestamp,source,gt_range_m,error_m,predicted_class_id,ground_truth_class_id',
                '1.0,stereo,2.2,0.1,1,1',
                '1.1,stereo,4.2,0.2,0,1',
            ]
        ),
        encoding='utf-8',
    )
    (logs_path / 'cone_range_rmse_samples_lidar.csv').write_text(
        '\n'.join(
            [
                'timestamp,source,gt_range_m,error_m,predicted_class_id,ground_truth_class_id',
                '1.0,lidar,2.2,0.05,1,1',
                '1.1,lidar,4.2,0.08,0,0',
            ]
        ),
        encoding='utf-8',
    )

    output_path = OfflineConePlotter(session_path).generate_combined_range_rmse_plot()

    assert output_path is not None
    assert output_path.exists()
    assert output_path.name == 'cone_range_binned_rmse_camera_lidar.png'
