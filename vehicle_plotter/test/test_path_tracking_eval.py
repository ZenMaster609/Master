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
    GateLapCounter,
    analyze_path_tracking_csv,
    build_smalltrack_lap_gate,
    build_gt_midline_from_cones,
    build_stitched_reference_trace,
    compare_planner_path_to_gt,
    should_assume_identity_transform,
)
from vehicle_plotter.logging.path_tracking_eval_plots import (  # noqa: E402
    _estimate_average_track_width_m,
    _format_distance_with_half_width_percent,
    build_skidpad_gt_color_borders,
    build_skidpad_gt_overlay_segments,
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


def test_build_smalltrack_lap_gate_uses_big_orange_marker_pairs():
    big_orange = np.asarray(
        [
            [-9.61314, 13.0],
            [-9.99934, 12.989],
            [-9.62148, 8.39323],
            [-9.98667, 8.39348],
        ],
        dtype=np.float64,
    )

    gate = build_smalltrack_lap_gate(big_orange_xy=big_orange, frame_id='map')

    assert gate is not None
    assert gate.frame_id == 'map'
    assert gate.segment_xy.shape == (2, 2)
    assert np.allclose(gate.segment_xy[0], np.asarray([-9.804075, 8.393355], dtype=np.float64), atol=1e-5)
    assert np.allclose(gate.segment_xy[1], np.asarray([-9.80624, 12.9945], dtype=np.float64), atol=1e-5)


def test_gate_lap_counter_ignores_start_crossing_then_counts_next_full_crossing():
    counter = GateLapCounter(
        np.asarray([[0.0, -1.0], [0.0, 1.0]], dtype=np.float64),
        min_lap_travel_m=10.0,
        min_lap_time_sec=0.0,
        near_gate_distance_m=4.0,
    )

    assert counter.update(np.asarray([-3.0, 0.0], dtype=np.float64), 0.0).completed_laps == 0
    assert counter.update(np.asarray([3.0, 0.0], dtype=np.float64), 1.0).completed_laps == 0
    counter.update(np.asarray([3.0, 12.0], dtype=np.float64), 2.0)
    counter.update(np.asarray([-3.0, 12.0], dtype=np.float64), 3.0)
    counter.update(np.asarray([-3.0, 0.0], dtype=np.float64), 4.0)
    snapshot = counter.update(np.asarray([3.0, 0.0], dtype=np.float64), 5.0)

    assert snapshot.completed_laps == 1
    assert snapshot.just_completed_lap


def test_track_width_and_half_width_percentage_formatting():
    left = np.asarray([[0.0, 1.5], [5.0, 1.5], [10.0, 1.5]], dtype=np.float64)
    right = np.asarray([[0.0, -1.5], [5.0, -1.5], [10.0, -1.5]], dtype=np.float64)

    track_width_m = _estimate_average_track_width_m(left, right)

    assert abs(track_width_m - 3.0) < 1e-9
    assert _format_distance_with_half_width_percent(0.0, track_width_m) == '0.00 cm (0.00%)'
    assert _format_distance_with_half_width_percent(0.75, track_width_m) == '75.00 cm (50.00%)'
    assert _format_distance_with_half_width_percent(1.5, track_width_m) == '150.00 cm (100.00%)'


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
                'odom_child_frame_id',
                'resolved_control_frame',
                'vehicle_x_m',
                'vehicle_y_m',
                'body_center_x_m',
                'body_center_y_m',
                'front_axle_x_m',
                'front_axle_y_m',
                'planner_reference_x_m',
                'planner_reference_y_m',
                'gt_reference_x_m',
                'gt_reference_y_m',
                'planner_reference_vs_gt_cte_m',
                'body_center_vs_planner_cte_m',
                'front_axle_vs_planner_cte_m',
                'controller_vs_planner_cte_m',
                'body_center_vs_gt_cte_m',
                'front_axle_vs_gt_cte_m',
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
                'odom_child_frame_id': 'base_footprint',
                'resolved_control_frame': 'front_axle',
                'vehicle_x_m': 0.2 * idx,
                'vehicle_y_m': 0.08 * np.sin(0.1 * idx),
                'body_center_x_m': 0.2 * idx,
                'body_center_y_m': 0.08 * np.sin(0.1 * idx),
                'front_axle_x_m': 0.2 * idx + 0.825,
                'front_axle_y_m': 0.03 * np.sin(0.1 * idx),
                'planner_reference_x_m': 0.2 * idx,
                'planner_reference_y_m': 0.0,
                'gt_reference_x_m': 0.2 * idx,
                'gt_reference_y_m': 0.0,
                'planner_reference_vs_gt_cte_m': 0.02 * np.sin(0.1 * idx + 0.05),
                'body_center_vs_planner_cte_m': 0.08 * np.sin(0.1 * idx),
                'front_axle_vs_planner_cte_m': 0.03 * np.sin(0.1 * idx),
                'controller_vs_planner_cte_m': 0.08 * np.sin(0.1 * idx),
                'body_center_vs_gt_cte_m': 0.12 * np.sin(0.1 * idx + 0.2),
                'front_axle_vs_gt_cte_m': 0.05 * np.sin(0.1 * idx + 0.1),
                'controller_vs_gt_cte_m': 0.12 * np.sin(0.1 * idx + 0.2),
                'planner_vs_gt_cte_rms_m': 0.18,
                'planner_vs_gt_cte_p95_m': 0.2,
                'planner_vs_gt_cte_max_m': 0.22,
            })

    summary = analyze_path_tracking_csv(csv_path)
    assert summary['sample_count'] == 150.0
    assert summary['valid_sample_count'] == 150.0
    assert np.isfinite(summary['planner_vs_gt_cte_rms_m'])
    assert np.isfinite(summary['planner_reference_vs_gt_cte_rms_m'])
    assert np.isfinite(summary['controller_vs_planner_cte_p95_m'])
    assert np.isfinite(summary['controller_vs_gt_cte_max_m'])
    assert np.isfinite(summary['front_axle_vs_planner_cte_p95_m'])
    assert np.isfinite(summary['front_axle_vs_gt_cte_max_m'])
    assert summary['front_axle_vs_planner_cte_rms_m'] < summary['controller_vs_planner_cte_rms_m']

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
        lap_count=2,
        lap_target=3,
    )
    assert generated_overlay == overlay_plot
    assert overlay_plot.exists()
    assert overlay_plot.stat().st_size > 0


def test_generate_path_tracking_cte_plot_supports_planner_vs_gt_series_only(tmp_path):
    csv_path = tmp_path / 'path_tracking_eval.csv'

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
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
            'planner_reference_vs_gt_cte_m',
            'controller_vs_planner_cte_m',
            'controller_vs_gt_cte_m',
            'planner_vs_gt_cte_rms_m',
            'planner_vs_gt_cte_p95_m',
            'planner_vs_gt_cte_max_m',
        ])
        writer.writeheader()
        for idx, planner_vs_gt_m in enumerate((0.15, 0.20, 0.25)):
            writer.writerow({
                'timestamp_sec': float(idx),
                'sample_valid_flag': 1.0,
                'status': 'ok',
                'frame_id': 'odom',
                'gt_source_frame': 'map',
                'vehicle_x_m': float(idx),
                'vehicle_y_m': 0.0,
                'planner_reference_x_m': float(idx),
                'planner_reference_y_m': 0.0,
                'gt_reference_x_m': float(idx),
                'gt_reference_y_m': 0.0,
                'planner_reference_vs_gt_cte_m': planner_vs_gt_m,
                'controller_vs_planner_cte_m': float('nan'),
                'controller_vs_gt_cte_m': float('nan'),
                'planner_vs_gt_cte_rms_m': planner_vs_gt_m,
                'planner_vs_gt_cte_p95_m': planner_vs_gt_m,
                'planner_vs_gt_cte_max_m': planner_vs_gt_m,
            })

    output_path = tmp_path / 'path_tracking_eval_cte.png'
    generated = generate_path_tracking_cte_plot(csv_path, output_path)

    assert generated == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_skidpad_gt_overlay_segments_matches_router_geometry():
    segments = build_skidpad_gt_overlay_segments()

    assert len(segments) == 3
    assert segments[0].shape == (33, 2)
    assert segments[1].shape == (33, 2)
    assert np.allclose(segments[2], np.asarray([[0.0, -11.0], [0.0, 21.0]], dtype=np.float64))
    assert np.allclose(segments[0][0], np.asarray([-0.25, 0.0], dtype=np.float64), atol=1e-6)
    assert np.allclose(segments[1][0], np.asarray([18.25, 0.0], dtype=np.float64), atol=1e-6)


def test_build_skidpad_gt_color_borders_match_track_model_geometry():
    blue_xy, yellow_xy = build_skidpad_gt_color_borders()

    assert blue_xy.shape == (30, 2)
    assert yellow_xy.shape == (30, 2)
    assert np.allclose(blue_xy[0], np.asarray([-1.637, 0.0], dtype=np.float64))
    assert np.allclose(blue_xy[-1], np.asarray([-1.637, 0.0], dtype=np.float64))
    assert np.allclose(yellow_xy[0], np.asarray([16.862, 0.0], dtype=np.float64))
    assert np.allclose(yellow_xy[-1], np.asarray([16.862, 0.0], dtype=np.float64))
    assert np.allclose(blue_xy[18], np.asarray([9.25, 10.612], dtype=np.float64))
    assert np.allclose(yellow_xy[18], np.asarray([-9.25, 10.612], dtype=np.float64))


def test_compute_path_tracking_overlay_average_distances_supports_multi_segment_gt(tmp_path):
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
                'planner_reference_vs_gt_cte_m',
                'controller_vs_planner_cte_m',
                'controller_vs_gt_cte_m',
                'planner_vs_gt_cte_rms_m',
                'planner_vs_gt_cte_p95_m',
                'planner_vs_gt_cte_max_m',
            ],
        )
        writer.writeheader()
        for y_m in (-10.0, 0.0, 10.0):
            writer.writerow({
                'timestamp_sec': y_m,
                'sample_valid_flag': 1.0,
                'status': 'ok',
                'frame_id': 'odom',
                'gt_source_frame': 'map',
                'vehicle_x_m': 0.0,
                'vehicle_y_m': y_m,
                'planner_reference_x_m': 0.0,
                'planner_reference_y_m': y_m,
                'gt_reference_x_m': 0.0,
                'gt_reference_y_m': y_m,
                'planner_reference_vs_gt_cte_m': 0.0,
                'controller_vs_planner_cte_m': 0.0,
                'controller_vs_gt_cte_m': 5.0,
                'planner_vs_gt_cte_rms_m': 5.0,
                'planner_vs_gt_cte_p95_m': 5.0,
                'planner_vs_gt_cte_max_m': 5.0,
            })

    averages = compute_path_tracking_overlay_average_distances(
        csv_path,
        gt_midline_xy=np.empty((0, 2), dtype=np.float64),
        planner_trace_xy=np.asarray([[0.0, -10.0], [0.0, 10.0]], dtype=np.float64),
        gt_reference_segments_xy=build_skidpad_gt_overlay_segments(),
    )

    assert abs(averages['planner_vs_gt_avg_dist_m']) < 1e-9
    assert abs(averages['controller_vs_gt_avg_dist_m']) < 1e-9
    assert abs(averages['controller_vs_planner_avg_dist_m']) < 1e-9


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
                'planner_reference_vs_gt_cte_m',
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
                'planner_reference_vs_gt_cte_m': planner_y,
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
