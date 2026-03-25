from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from vehicle_plotter.logging.path_tracking_eval import (  # noqa: E402
    analyze_path_tracking_csv,
    build_gt_midline_from_cones,
    build_stitched_reference_trace,
    compare_planner_path_to_gt,
    should_assume_identity_transform,
)
from vehicle_plotter.logging.path_tracking_eval_plots import (  # noqa: E402
    compute_path_tracking_overlay_average_distances,
    generate_path_tracking_cte_plot,
    generate_path_tracking_overlay_plot,
)


def test_build_gt_midline_open_track_stays_centered():
    blue = np.asarray([[0.0, 1.5], [5.0, 1.5], [10.0, 1.5], [15.0, 1.5]], dtype=np.float64)
    yellow = np.asarray([[0.0, -1.5], [5.0, -1.5], [10.0, -1.5], [15.0, -1.5]], dtype=np.float64)

    midline = build_gt_midline_from_cones(
        blue_xy=blue,
        yellow_xy=yellow,
        start_xy=np.asarray([0.0, 0.0], dtype=np.float64),
        heading_xy=np.asarray([1.0, 0.0], dtype=np.float64),
        frame_id='map',
        resolution_m=0.5,
    )

    assert midline.frame_id == 'map'
    assert midline.midline_xy.shape[0] >= 10
    assert np.allclose(midline.midline_xy[:, 1], 0.0, atol=1e-6)
    assert np.all(np.diff(midline.midline_xy[:, 0]) >= -1e-6)
    assert np.allclose(midline.left_xy[:4, 1], 1.5, atol=1e-6)
    assert np.allclose(midline.right_xy[:4, 1], -1.5, atol=1e-6)


def test_build_gt_midline_from_unordered_unequal_cones_uses_centered_pairs():
    blue = np.asarray(
        [[15.0, 1.5], [5.0, 1.5], [10.0, 1.5], [0.0, 1.5], [20.0, 1.5]],
        dtype=np.float64,
    )
    yellow = np.asarray(
        [[10.0, -1.5], [0.0, -1.5], [15.0, -1.5], [5.0, -1.5]],
        dtype=np.float64,
    )

    midline = build_gt_midline_from_cones(
        blue_xy=blue,
        yellow_xy=yellow,
        start_xy=np.asarray([0.0, 0.0], dtype=np.float64),
        heading_xy=np.asarray([1.0, 0.0], dtype=np.float64),
        frame_id='map',
        resolution_m=0.5,
    )

    assert midline.midline_xy.shape[0] >= 10
    assert np.allclose(midline.midline_xy[:, 1], 0.0, atol=1e-6)
    assert float(np.max(midline.midline_xy[:, 0])) >= 14.5


def test_build_gt_midline_loop_like_track_anchors_near_start():
    theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    blue = np.column_stack((8.0 * np.cos(theta), 5.0 * np.sin(theta) + 1.5))
    yellow = np.column_stack((6.0 * np.cos(theta), 3.0 * np.sin(theta) - 1.5))

    midline = build_gt_midline_from_cones(
        blue_xy=blue,
        yellow_xy=yellow,
        start_xy=np.asarray([7.0, 0.0], dtype=np.float64),
        heading_xy=np.asarray([0.0, 1.0], dtype=np.float64),
        frame_id='map',
        resolution_m=0.5,
    )

    assert midline.midline_xy.shape[0] >= 8
    first = midline.midline_xy[0]
    assert np.hypot(first[0] - 7.0, first[1] - 0.0) < 4.0


def test_compare_planner_path_to_gt_reports_expected_offset():
    gt = np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    planner = np.asarray([[0.0, 0.4], [5.0, 0.4], [10.0, 0.4]], dtype=np.float64)

    metrics = compare_planner_path_to_gt(
        planner_xy=planner,
        gt_midline_xy=gt,
        vehicle_xy=np.asarray([0.0, 0.0], dtype=np.float64),
        resolution_m=0.25,
    )

    assert abs(metrics['planner_vs_gt_cte_rms_m'] - 0.4) < 0.05
    assert abs(metrics['planner_vs_gt_cte_p95_m'] - 0.4) < 0.05
    assert abs(metrics['planner_vs_gt_cte_max_m'] - 0.4) < 0.05


def test_build_stitched_reference_trace_collapses_dense_duplicates():
    trace = np.asarray(
        [
            [0.0, 0.0],
            [0.02, 0.0],
            [0.04, 0.0],
            [0.3, 0.0],
            [0.32, 0.0],
            [0.7, 0.0],
        ],
        dtype=np.float64,
    )

    stitched = build_stitched_reference_trace(trace, min_spacing_m=0.1)
    assert stitched.shape[0] == 3
    assert np.allclose(stitched[0], [0.0, 0.0])
    assert np.allclose(stitched[-1], [0.7, 0.0])


def test_identity_transform_fallback_only_applies_to_map_odom_pair():
    assert should_assume_identity_transform('map', 'odom')
    assert should_assume_identity_transform('odom', 'map')
    assert not should_assume_identity_transform('map', 'base_link')
    assert not should_assume_identity_transform('track', 'odom')
    assert not should_assume_identity_transform('map', 'map')


def test_analyze_path_tracking_csv_and_plot_smoke(tmp_path):
    csv_path = tmp_path / 'path_tracking_eval.csv'

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'timestamp_sec',
                'sample_valid_flag',
                'status',
                'frame_id',
                'gt_source_frame',
                'vehicle_x_m',
                'vehicle_y_m',
                'planner_reference_x_m',
                'planner_reference_y_m',
                'gt_reference_x_m',
                'gt_reference_y_m',
                'controller_vs_planner_cte_m',
                'controller_vs_gt_cte_m',
                'planner_vs_gt_cte_rms_m',
                'planner_vs_gt_cte_p95_m',
                'planner_vs_gt_cte_max_m',
            ],
        )
        writer.writeheader()
        for idx in range(150):
            t = idx * 0.05
            writer.writerow({
                'timestamp_sec': t,
                'sample_valid_flag': 1.0,
                'status': 'ok',
                'frame_id': 'odom',
                'gt_source_frame': 'map',
                'vehicle_x_m': 0.2 * idx,
                'vehicle_y_m': 0.08 * np.sin(0.1 * idx),
                'planner_reference_x_m': 0.2 * idx,
                'planner_reference_y_m': 0.0,
                'gt_reference_x_m': 0.2 * idx,
                'gt_reference_y_m': 0.0,
                'controller_vs_planner_cte_m': 0.08 * np.sin(0.1 * idx),
                'controller_vs_gt_cte_m': 0.12 * np.sin(0.1 * idx + 0.2),
                'planner_vs_gt_cte_rms_m': 0.18,
                'planner_vs_gt_cte_p95_m': 0.2,
                'planner_vs_gt_cte_max_m': 0.22,
            })

    summary = analyze_path_tracking_csv(csv_path)
    assert summary['sample_count'] == 150.0
    assert summary['valid_sample_count'] == 150.0
    assert np.isfinite(summary['planner_vs_gt_cte_rms_m'])
    assert np.isfinite(summary['controller_vs_planner_cte_p95_m'])
    assert np.isfinite(summary['controller_vs_gt_cte_max_m'])

    cte_plot = tmp_path / 'path_tracking_eval_cte.png'
    generated_cte = generate_path_tracking_cte_plot(csv_path, cte_plot)
    assert generated_cte == cte_plot
    assert cte_plot.exists()
    assert cte_plot.stat().st_size > 0

    overlay_plot = tmp_path / 'path_tracking_eval_overlay.png'
    gt_midline = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float64)
    planner_trace = np.asarray([[0.0, 0.1], [10.0, 0.1], [20.0, 0.1], [30.0, 0.1]], dtype=np.float64)
    generated_overlay = generate_path_tracking_overlay_plot(
        csv_path,
        overlay_plot,
        gt_midline_xy=gt_midline,
        gt_left_xy=gt_midline + np.asarray([0.0, 1.5]),
        gt_right_xy=gt_midline + np.asarray([0.0, -1.5]),
        planner_trace_xy=planner_trace,
    )
    assert generated_overlay == overlay_plot
    assert overlay_plot.exists()
    assert overlay_plot.stat().st_size > 0


def test_compute_path_tracking_overlay_average_distances_uses_sample_means(tmp_path):
    csv_path = tmp_path / 'path_tracking_eval.csv'

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'timestamp_sec',
                'sample_valid_flag',
                'status',
                'frame_id',
                'gt_source_frame',
                'vehicle_x_m',
                'vehicle_y_m',
                'planner_reference_x_m',
                'planner_reference_y_m',
                'gt_reference_x_m',
                'gt_reference_y_m',
                'controller_vs_planner_cte_m',
                'controller_vs_gt_cte_m',
                'planner_vs_gt_cte_rms_m',
                'planner_vs_gt_cte_p95_m',
                'planner_vs_gt_cte_max_m',
            ],
        )
        writer.writeheader()
        rows = [
            (0.0, 0.20, 0.30, 0.1),
            (1.0, 0.20, 0.30, 0.1),
            (2.0, 0.20, 0.30, 0.1),
        ]
        for t, vehicle_y, planner_y, ctrl_vs_planner in rows:
            writer.writerow({
                'timestamp_sec': t,
                'sample_valid_flag': 1.0,
                'status': 'ok',
                'frame_id': 'odom',
                'gt_source_frame': 'map',
                'vehicle_x_m': t,
                'vehicle_y_m': vehicle_y,
                'planner_reference_x_m': t,
                'planner_reference_y_m': planner_y,
                'gt_reference_x_m': t,
                'gt_reference_y_m': 0.0,
                'controller_vs_planner_cte_m': ctrl_vs_planner,
                'controller_vs_gt_cte_m': vehicle_y,
                'planner_vs_gt_cte_rms_m': planner_y,
                'planner_vs_gt_cte_p95_m': planner_y,
                'planner_vs_gt_cte_max_m': planner_y,
            })

    averages = compute_path_tracking_overlay_average_distances(
        csv_path,
        gt_midline_xy=np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64),
        planner_trace_xy=np.asarray([[0.0, 0.3], [2.0, 0.3]], dtype=np.float64),
    )

    assert abs(averages['planner_vs_gt_avg_dist_m'] - 0.3) < 1e-9
    assert abs(averages['controller_vs_gt_avg_dist_m'] - 0.2) < 1e-9
    assert abs(averages['controller_vs_planner_avg_dist_m'] - 0.1) < 1e-9
