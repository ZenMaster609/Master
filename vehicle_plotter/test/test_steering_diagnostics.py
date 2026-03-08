from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from vehicle_plotter.logging.steering_diagnostics import (  # noqa: E402
    analyze_csv,
    heading_error,
    nearest_point_on_polyline,
    parse_planner_diag,
    signed_cross_track_error,
)


def test_signed_cte_on_straight_line():
    path = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    cte_up, _yaw_up = signed_cross_track_error(2.0, 1.0, path)
    cte_down, _yaw_down = signed_cross_track_error(2.0, -1.0, path)
    assert cte_up > 0.0
    assert cte_down < 0.0


def test_heading_error_wrap_near_pi():
    err = heading_error(-3.13, 3.13)
    assert abs(err) < 0.05


def test_nearest_point_multi_segment():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    idx, pt = nearest_point_on_polyline(1.0, 1.0, path)
    assert idx in (0, 1)
    assert np.isfinite(pt).all()


class _KV:
    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value


class _Status:
    def __init__(self, name: str, values):
        self.name = name
        self.values = values


class _Diag:
    def __init__(self, statuses):
        self.status = statuses


def test_parse_planner_diag_with_keys():
    msg = _Diag([
        _Status(
            'delaunay_planner/stability',
            [
                _KV('centerline_jump_max_m', '0.7'),
                _KV('selected_edge_churn_ratio', '0.4'),
                _KV('tracked_cones_frame_delta_p95_m', '0.3'),
            ],
        )
    ])
    out = parse_planner_diag(msg)
    assert abs(out['centerline_jump_max_m'] - 0.7) < 1e-9
    assert abs(out['selected_edge_churn_ratio'] - 0.4) < 1e-9
    assert abs(out['tracked_cones_frame_delta_p95_m'] - 0.3) < 1e-9


def test_parse_planner_diag_missing_keys_returns_nan():
    msg = _Diag([_Status('other_status', [])])
    out = parse_planner_diag(msg)
    assert np.isnan(out['centerline_jump_max_m'])
    assert np.isnan(out['selected_edge_churn_ratio'])
    assert np.isnan(out['tracked_cones_frame_delta_p95_m'])


def test_analyze_csv_recovers_known_lag(tmp_path):
    csv_path = tmp_path / 'steering_tracking_diagnostics.csv'
    fieldnames = [
        'timestamp_sec',
        'desired_steering_rad',
        'actual_steering_rad',
        'steering_error_rad',
        'cte_m',
        'heading_error_rad',
        'planner_selected_edge_churn_ratio',
        'planner_centerline_jump_max_m',
    ]
    dt = 0.02
    lag_samples = 3
    n = 300
    t = np.arange(n, dtype=np.float64) * dt
    desired = np.sin(2.0 * np.pi * 1.2 * t)
    actual = np.roll(desired, lag_samples)
    actual[:lag_samples] = desired[0]
    cte = 0.1 * np.sin(2.0 * np.pi * 0.8 * t)

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                'timestamp_sec': t[i],
                'desired_steering_rad': desired[i],
                'actual_steering_rad': actual[i],
                'steering_error_rad': actual[i] - desired[i],
                'cte_m': cte[i],
                'heading_error_rad': 0.5 * cte[i],
                'planner_selected_edge_churn_ratio': 0.2,
                'planner_centerline_jump_max_m': 0.1,
            })

    summary = analyze_csv(csv_path)
    assert summary['sample_count'] == float(n)
    assert abs(summary['lag_samples'] - lag_samples) <= 1.0
    assert np.isfinite(summary['steering_error_rms_rad'])
