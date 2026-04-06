from __future__ import annotations

import csv
import pathlib
import sys

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from vehicle_plotter.logging.corridor_oscillation_analysis import (  # noqa: E402
    analyze_corridor_oscillation,
    generate_corridor_oscillation_plot,
    write_corridor_oscillation_summary_files,
)
from vehicle_plotter.logging.steering_diagnostics import (  # noqa: E402
    corridor_analysis_sample_metric_keys,
)


def test_corridor_oscillation_analysis_generates_summary_and_plot(tmp_path):
    thesis_csv = tmp_path / 'thesis_controller_diagnostics.csv'
    path_csv = tmp_path / 'path_tracking_eval.csv'

    thesis_fieldnames = [
        'timestamp_sec',
        'planner_centerline_jump_max_m',
        'planner_selected_edge_churn_ratio',
        'plan_hold_active_flag',
    ]
    thesis_fieldnames.extend(corridor_analysis_sample_metric_keys())

    with open(thesis_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=thesis_fieldnames)
        writer.writeheader()
        for idx in range(6):
            row = {
                'timestamp_sec': float(idx) * 0.1,
                'planner_centerline_jump_max_m': 0.05 + (0.01 * idx),
                'planner_selected_edge_churn_ratio': 0.20 + (0.02 * idx),
                'plan_hold_active_flag': 1.0 if idx >= 4 else 0.0,
            }
            for sample_idx in range(8):
                x_m = float(sample_idx)
                row[f'corridor_raw_anchor_p{sample_idx}_x_m'] = x_m
                row[f'corridor_raw_anchor_p{sample_idx}_y_m'] = 0.15
                row[f'corridor_prevalidation_centerline_p{sample_idx}_x_m'] = x_m
                row[f'corridor_prevalidation_centerline_p{sample_idx}_y_m'] = 0.10
                row[f'corridor_buffer_centerline_p{sample_idx}_x_m'] = x_m
                row[f'corridor_buffer_centerline_p{sample_idx}_y_m'] = 0.05
                row[f'corridor_control_path_p{sample_idx}_x_m'] = x_m
                row[f'corridor_control_path_p{sample_idx}_y_m'] = 0.02
            row['corridor_raw_anchor_point_count'] = 8.0
            row['corridor_prevalidation_centerline_point_count'] = 8.0
            row['corridor_buffer_centerline_point_count'] = 8.0
            row['corridor_control_path_point_count'] = 8.0
            writer.writerow(row)

    with open(path_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'sample_valid_flag',
            'planner_reference_s_m',
            'planner_reference_vs_gt_cte_m',
        ])
        writer.writeheader()
        for idx in range(6):
            writer.writerow({
                'sample_valid_flag': 1.0,
                'planner_reference_s_m': float(idx),
                'planner_reference_vs_gt_cte_m': 0.03 * (idx + 1),
            })

    summary = analyze_corridor_oscillation(thesis_csv, path_csv)
    assert summary['sample_count'] == 6.0
    assert summary['profile_row_count'] == 6.0
    assert summary['buffer_vs_prevalidation_abs_mean_m'] > 0.0
    assert summary['control_vs_buffer_abs_p95_m'] > 0.0
    assert summary['planner_reference_vs_gt_cte_rms_m'] > 0.0

    summary_json = tmp_path / 'corridor_oscillation_summary.json'
    summary_txt = tmp_path / 'corridor_oscillation_summary.txt'
    write_corridor_oscillation_summary_files(summary, summary_json, summary_txt)
    assert summary_json.exists()
    assert summary_txt.exists()

    plot_path = tmp_path / 'corridor_oscillation_analysis.png'
    generated = generate_corridor_oscillation_plot(thesis_csv, path_csv, plot_path)
    assert generated == plot_path
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
