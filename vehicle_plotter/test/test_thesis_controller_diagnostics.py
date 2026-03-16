from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from vehicle_plotter.logging.thesis_controller_diagnostics import (  # noqa: E402
    analyze_thesis_csv,
)
from vehicle_plotter.logging.thesis_controller_plots import (  # noqa: E402
    generate_thesis_controller_plot,
)


def test_analyze_thesis_csv_reports_expected_metrics(tmp_path):
    csv_path = tmp_path / 'thesis_controller_diagnostics.csv'
    fieldnames = [
        'timestamp_sec',
        'final_steering_cmd_rad',
        'actual_steering_rad',
        'steering_error_rad',
        'cte_m',
        'heading_error_rad',
        'steering_saturated_flag',
        'planner_centerline_jump_max_m',
        'planner_selected_edge_churn_ratio',
    ]
    dt = 0.02
    lag_samples = 2
    n = 300
    t = np.arange(n, dtype=np.float64) * dt
    final_cmd = 0.18 * np.sin(2.0 * np.pi * 0.8 * t)
    actual = np.roll(final_cmd, lag_samples)
    actual[:lag_samples] = final_cmd[0]
    cte = 0.15 * np.sin(2.0 * np.pi * 0.4 * t)
    heading = 0.08 * np.sin(2.0 * np.pi * 0.4 * t + 0.2)
    saturated = np.zeros((n,), dtype=np.float64)
    saturated[::10] = 1.0
    jump = 0.2 + (0.05 * np.sin(2.0 * np.pi * 0.25 * t))
    churn = 0.3 + (0.1 * np.cos(2.0 * np.pi * 0.25 * t))

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                'timestamp_sec': t[i],
                'final_steering_cmd_rad': final_cmd[i],
                'actual_steering_rad': actual[i],
                'steering_error_rad': actual[i] - final_cmd[i],
                'cte_m': cte[i],
                'heading_error_rad': heading[i],
                'steering_saturated_flag': saturated[i],
                'planner_centerline_jump_max_m': jump[i],
                'planner_selected_edge_churn_ratio': churn[i],
            })

    summary = analyze_thesis_csv(csv_path)
    assert summary['sample_count'] == float(n)
    assert abs(summary['lag_samples'] - lag_samples) <= 1.0
    assert np.isfinite(summary['cte_rms_m'])
    assert np.isfinite(summary['heading_error_p95_abs_rad'])
    assert np.isfinite(summary['steering_error_p95_abs_rad'])
    assert np.isfinite(summary['steering_saturation_ratio'])
    assert np.isfinite(summary['steering_activity_total_variation_rad'])
    assert np.isfinite(summary['steering_activity_rms_rate_radps'])
    assert np.isfinite(summary['IAE_cte'])
    assert np.isfinite(summary['ISE_cte'])


def test_generate_thesis_controller_plot_smoke(tmp_path):
    csv_path = tmp_path / 'thesis_controller_diagnostics.csv'
    fieldnames = [
        'timestamp_sec',
        'desired_steering_rad',
        'final_steering_cmd_rad',
        'actual_steering_rad',
        'cte_m',
        'heading_error_rad',
        'heading_contribution_rad',
        'cross_track_contribution_rad',
        'yaw_rate_damping_contribution_rad',
        'vehicle_speed_mps',
        'path_curvature_abs_p95_1pm',
        'planner_centerline_jump_max_m',
        'planner_selected_edge_churn_ratio',
        'plan_hold_active_flag',
        'plan_fallback_flag',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(180):
            t = i * 0.02
            writer.writerow({
                'timestamp_sec': t,
                'desired_steering_rad': 0.04 * np.sin(2.0 * t),
                'final_steering_cmd_rad': 0.05 * np.sin(2.0 * t + 0.1),
                'actual_steering_rad': 0.045 * np.sin(2.0 * t + 0.2),
                'cte_m': 0.15 * np.sin(0.7 * t),
                'heading_error_rad': 0.08 * np.cos(0.7 * t),
                'heading_contribution_rad': 0.03 * np.cos(0.9 * t),
                'cross_track_contribution_rad': 0.04 * np.sin(0.7 * t),
                'yaw_rate_damping_contribution_rad': 0.01 * np.cos(1.1 * t),
                'vehicle_speed_mps': 3.0 + (0.2 * np.sin(0.3 * t)),
                'path_curvature_abs_p95_1pm': 0.4 + (0.05 * np.cos(0.4 * t)),
                'planner_centerline_jump_max_m': 0.1 + (0.02 * np.sin(0.8 * t)),
                'planner_selected_edge_churn_ratio': 0.25 + (0.04 * np.cos(0.8 * t)),
                'plan_hold_active_flag': 1.0 if i > 120 else 0.0,
                'plan_fallback_flag': 1.0 if i > 150 else 0.0,
            })

    out_path = tmp_path / 'thesis_controller_diagnostics.png'
    generated = generate_thesis_controller_plot(csv_path, out_path)
    assert generated == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
