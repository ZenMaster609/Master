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


def test_compute_total_rmse_percent_matches_binwise_definition():
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


def test_generate_all_range_rmse_plots_writes_expected_pngs(tmp_path):
    session_path = tmp_path / 'session'
    logs_path = session_path / 'logs'
    logs_path.mkdir(parents=True)
    (session_path / 'plots').mkdir(parents=True)

    (logs_path / 'cone_range_rmse_samples_mono.csv').write_text(
        '\n'.join(
            [
                'timestamp,source,gt_range_m,error_m,predicted_class_id,ground_truth_class_id',
                '1.0,monocular,0.2,0.1,1,1',
                '1.1,monocular,1.2,0.2,0,1',
            ]
        ),
        encoding='utf-8',
    )
    (logs_path / 'cone_range_rmse_samples_stereo.csv').write_text(
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

    output_paths = OfflineConePlotter(session_path).generate_all_range_rmse_plots()

    assert [path.name for path in output_paths] == [
        'cone_range_rmse_mono.png',
        'cone_range_rmse_stereo.png',
        'cone_range_rmse_lidar.png',
    ]
    assert all(path.exists() for path in output_paths)


def test_generate_all_range_rmse_plots_creates_empty_pngs_for_missing_data(tmp_path):
    session_path = tmp_path / 'session'
    (session_path / 'logs').mkdir(parents=True)
    (session_path / 'plots').mkdir(parents=True)

    output_paths = OfflineConePlotter(session_path).generate_all_range_rmse_plots()

    assert [path.name for path in output_paths] == [
        'cone_range_rmse_mono.png',
        'cone_range_rmse_stereo.png',
        'cone_range_rmse_lidar.png',
    ]
    assert all(path.exists() for path in output_paths)


def test_all_sources_use_identical_binning_when_loaded(tmp_path):
    session_path = tmp_path / 'session'
    logs_path = session_path / 'logs'
    logs_path.mkdir(parents=True)

    for filename, source in (
        ('cone_range_rmse_samples_mono.csv', 'monocular'),
        ('cone_range_rmse_samples_stereo.csv', 'stereo'),
        ('cone_range_rmse_samples_lidar.csv', 'lidar'),
    ):
        (logs_path / filename).write_text(
            '\n'.join(
                [
                    'timestamp,source,gt_range_m,error_m,predicted_class_id,ground_truth_class_id',
                    f'1.0,{source},2.2,0.1,1,1',
                ]
            ),
            encoding='utf-8',
        )

    plotter = OfflineConePlotter(session_path)
    mono_stats = plotter._load_source_stats('cone_range_rmse_samples_mono.csv', expected_source='monocular')
    stereo_stats = plotter._load_source_stats('cone_range_rmse_samples_stereo.csv', expected_source='stereo')
    lidar_stats = plotter._load_source_stats('cone_range_rmse_samples_lidar.csv', expected_source='lidar')

    assert np.array_equal(mono_stats.bin_centers, stereo_stats.bin_centers)
    assert np.array_equal(stereo_stats.bin_centers, lidar_stats.bin_centers)
    assert mono_stats.total_counts.shape == stereo_stats.total_counts.shape == lidar_stats.total_counts.shape
