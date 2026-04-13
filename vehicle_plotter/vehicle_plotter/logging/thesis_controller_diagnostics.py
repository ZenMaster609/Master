"""Helpers for thesis-oriented controller diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .steering_diagnostics import (
    CONE_AUDIT_DIAG_KEYS,
    corridor_analysis_sample_metric_keys,
    heading_error,
    nearest_point_on_polyline,
    signed_cross_track_error,
)

THESIS_DIAG_FIELDNAMES = [
    'timestamp_sec',
    'cmd_stamp_sec',
    'cmd_age_sec',
    'desired_steering_rad',
    'desired_speed_mps',
    'actual_steering_deg',
    'actual_steering_rad',
    'steering_error_rad',
    'steering_error_abs_rad',
    'raw_steering_cmd_rad',
    'final_steering_cmd_rad',
    'vehicle_speed_mps',
    'vehicle_yaw_rate_rps',
    'physics_state_vx_mps',
    'physics_state_vy_mps',
    'physics_sideslip_rad',
    'physics_slip_angle_front_rad',
    'physics_slip_angle_rear_rad',
    'physics_kinematic_blend',
    'physics_kinematic_vy_ref_mps',
    'physics_kinematic_yaw_rate_ref_rps',
    'cte_m',
    'cte_abs_m',
    'heading_error_rad',
    'heading_error_abs_rad',
    'heading_contribution_rad',
    'cross_track_contribution_rad',
    'yaw_rate_damping_contribution_rad',
    'steering_saturated_flag',
    'centerline_point_count',
    'selected_edge_count',
    'plan_valid_flag',
    'plan_hold_active_flag',
    'plan_fallback_flag',
    'path_length_m',
    'path_curvature_abs_p95_1pm',
    'planner_centerline_jump_max_m',
    'planner_selected_edge_churn_ratio',
    'planner_selected_chain_churn_ratio',
    'planner_tracked_cones_frame_delta_p95_m',
    'planner_selected_chain_length',
    'planner_left_chain_length',
    'planner_right_chain_length',
    'planner_corridor_sample_count',
    'planner_corridor_width_min_m',
    'planner_corridor_width_median_m',
    'planner_corridor_width_max_m',
    'planner_publish_mode',
    'planner_hold_reason',
    'planner_candidate_source',
    'planner_midline_update_mode',
    'planner_mode',
]
THESIS_DIAG_FIELDNAMES.extend(f'planner_{key}' for key in CONE_AUDIT_DIAG_KEYS)
THESIS_DIAG_FIELDNAMES.extend(corridor_analysis_sample_metric_keys())


def build_thesis_sample_row(
    *,
    now_sec: float,
    cmd_stamp_sec: float,
    cmd_recv_sec: float,
    desired_steering_rad: float,
    desired_speed_mps: float,
    actual_steering_deg: float,
    vehicle_x_m: float,
    vehicle_y_m: float,
    vehicle_yaw_rad: float,
    vehicle_yaw_rate_rps: float,
    vehicle_speed_mps: float,
    centerline_xy: np.ndarray,
    planner_metrics: dict[str, float],
    planner_text_metrics: dict[str, str] | None = None,
) -> dict[str, float]:
    actual_rad = math.radians(actual_steering_deg) if math.isfinite(actual_steering_deg) else float('nan')
    cmd_age_sec = now_sec - cmd_recv_sec if math.isfinite(cmd_recv_sec) else float('nan')

    centerline_count = int(centerline_xy.shape[0])
    centerline_available = centerline_count >= 2
    cte_geom = float('nan')
    heading_geom = float('nan')
    if centerline_available and math.isfinite(vehicle_x_m) and math.isfinite(vehicle_y_m):
        _nearest_idx, _nearest_point = nearest_point_on_polyline(
            vehicle_x_m,
            vehicle_y_m,
            centerline_xy,
        )
        cte_geom, tangent_yaw = signed_cross_track_error(
            vehicle_x_m,
            vehicle_y_m,
            centerline_xy,
        )
        if math.isfinite(tangent_yaw) and math.isfinite(vehicle_yaw_rad):
            heading_geom = heading_error(vehicle_yaw_rad, tangent_yaw)

    cte_ctrl = float(planner_metrics.get('cross_track_error_m', float('nan')))
    heading_ctrl = float(planner_metrics.get('heading_error_rad', float('nan')))
    cte_m = cte_ctrl if math.isfinite(cte_ctrl) else cte_geom
    heading_err = heading_ctrl if math.isfinite(heading_ctrl) else heading_geom

    raw_cmd = float(planner_metrics.get('raw_steering_cmd_rad', float('nan')))
    final_cmd = float(planner_metrics.get('final_steering_cmd_rad', float('nan')))
    if not math.isfinite(final_cmd):
        final_cmd = desired_steering_rad
    steering_ref = final_cmd if math.isfinite(final_cmd) else desired_steering_rad
    steering_error = (
        actual_rad - steering_ref
        if math.isfinite(actual_rad) and math.isfinite(steering_ref)
        else float('nan')
    )

    vehicle_speed_dbg = float(planner_metrics.get('vehicle_speed_mps', float('nan')))
    speed_mps = vehicle_speed_dbg if math.isfinite(vehicle_speed_dbg) else vehicle_speed_mps

    path_length_m = float(planner_metrics.get('path_length_m', float('nan')))
    if not math.isfinite(path_length_m):
        path_length_m = compute_path_length_m(centerline_xy)
    path_curvature_abs_p95 = float(planner_metrics.get('path_curvature_abs_p95_1pm', float('nan')))
    if not math.isfinite(path_curvature_abs_p95):
        path_curvature_abs_p95 = compute_path_curvature_abs_p95(centerline_xy)

    centerline_point_count = float(planner_metrics.get('centerline_point_count', float('nan')))
    if not math.isfinite(centerline_point_count):
        centerline_point_count = float(centerline_count)

    text_metrics = planner_text_metrics or {}

    row: dict[str, float | str] = {
        'timestamp_sec': now_sec,
        'cmd_stamp_sec': cmd_stamp_sec,
        'cmd_age_sec': cmd_age_sec,
        'desired_steering_rad': desired_steering_rad,
        'desired_speed_mps': desired_speed_mps,
        'actual_steering_deg': actual_steering_deg,
        'actual_steering_rad': actual_rad,
        'steering_error_rad': steering_error,
        'steering_error_abs_rad': abs(steering_error) if math.isfinite(steering_error) else float('nan'),
        'raw_steering_cmd_rad': raw_cmd,
        'final_steering_cmd_rad': final_cmd,
        'vehicle_speed_mps': speed_mps,
        'vehicle_yaw_rate_rps': vehicle_yaw_rate_rps,
        'physics_state_vx_mps': float(planner_metrics.get('state_vx_mps', float('nan'))),
        'physics_state_vy_mps': float(planner_metrics.get('state_vy_mps', float('nan'))),
        'physics_sideslip_rad': float(planner_metrics.get('sideslip_rad', float('nan'))),
        'physics_slip_angle_front_rad': float(
            planner_metrics.get('slip_angle_front_rad', float('nan'))
        ),
        'physics_slip_angle_rear_rad': float(
            planner_metrics.get('slip_angle_rear_rad', float('nan'))
        ),
        'physics_kinematic_blend': float(planner_metrics.get('kinematic_blend', float('nan'))),
        'physics_kinematic_vy_ref_mps': float(
            planner_metrics.get('kinematic_vy_ref_mps', float('nan'))
        ),
        'physics_kinematic_yaw_rate_ref_rps': float(
            planner_metrics.get('kinematic_yaw_rate_ref_rps', float('nan'))
        ),
        'cte_m': cte_m,
        'cte_abs_m': abs(cte_m) if math.isfinite(cte_m) else float('nan'),
        'heading_error_rad': heading_err,
        'heading_error_abs_rad': abs(heading_err) if math.isfinite(heading_err) else float('nan'),
        'heading_contribution_rad': float(
            planner_metrics.get('heading_contribution_rad', float('nan'))
        ),
        'cross_track_contribution_rad': float(
            planner_metrics.get('cross_track_contribution_rad', float('nan'))
        ),
        'yaw_rate_damping_contribution_rad': float(
            planner_metrics.get('yaw_rate_damping_contribution_rad', float('nan'))
        ),
        'steering_saturated_flag': float(
            planner_metrics.get('steering_saturated_flag', float('nan'))
        ),
        'centerline_point_count': centerline_point_count,
        'selected_edge_count': float(planner_metrics.get('selected_edge_count', float('nan'))),
        'plan_valid_flag': float(planner_metrics.get('plan_valid_flag', float('nan'))),
        'plan_hold_active_flag': float(planner_metrics.get('plan_hold_active_flag', float('nan'))),
        'plan_fallback_flag': float(planner_metrics.get('plan_fallback_flag', float('nan'))),
        'path_length_m': path_length_m,
        'path_curvature_abs_p95_1pm': path_curvature_abs_p95,
        'planner_centerline_jump_max_m': float(
            planner_metrics.get('centerline_jump_max_m', float('nan'))
        ),
        'planner_selected_edge_churn_ratio': float(
            planner_metrics.get('selected_edge_churn_ratio', float('nan'))
        ),
        'planner_selected_chain_churn_ratio': float(
            planner_metrics.get(
                'selected_chain_churn_ratio',
                planner_metrics.get('selected_edge_churn_ratio', float('nan')),
            )
        ),
        'planner_tracked_cones_frame_delta_p95_m': float(
            planner_metrics.get('tracked_cones_frame_delta_p95_m', float('nan'))
        ),
        'planner_selected_chain_length': float(
            planner_metrics.get('selected_chain_length', float('nan'))
        ),
        'planner_left_chain_length': float(
            planner_metrics.get('left_chain_length', float('nan'))
        ),
        'planner_right_chain_length': float(
            planner_metrics.get('right_chain_length', float('nan'))
        ),
        'planner_corridor_sample_count': float(
            planner_metrics.get('corridor_sample_count', float('nan'))
        ),
        'planner_corridor_width_min_m': float(
            planner_metrics.get('corridor_width_min_m', float('nan'))
        ),
        'planner_corridor_width_median_m': float(
            planner_metrics.get('corridor_width_median_m', float('nan'))
        ),
        'planner_corridor_width_max_m': float(
            planner_metrics.get('corridor_width_max_m', float('nan'))
        ),
        'planner_publish_mode': str(text_metrics.get('publish_mode', '')).strip(),
        'planner_hold_reason': str(text_metrics.get('hold_reason', '')).strip(),
        'planner_candidate_source': str(text_metrics.get('candidate_source', '')).strip(),
        'planner_midline_update_mode': str(text_metrics.get('midline_update_mode', '')).strip(),
        'planner_mode': str(text_metrics.get('planner_mode', '')).strip(),
    }
    for key in corridor_analysis_sample_metric_keys():
        row[key] = float(planner_metrics.get(key, float('nan')))
    for key in CONE_AUDIT_DIAG_KEYS:
        row[f'planner_{key}'] = float(planner_metrics.get(key, float('nan')))
    return row


def analyze_thesis_csv(csv_path: Path) -> dict[str, float]:
    rows = _read_rows(csv_path)
    if not rows:
        return {'sample_count': 0.0}

    t = _series(rows, 'timestamp_sec')
    cte = _series(rows, 'cte_m')
    heading = _series(rows, 'heading_error_rad')
    desired = _series(rows, 'final_steering_cmd_rad')
    actual = _series(rows, 'actual_steering_rad')
    steering_error = _series(rows, 'steering_error_rad')
    steering_saturated = _series(rows, 'steering_saturated_flag')
    jump = _series(rows, 'planner_centerline_jump_max_m')
    churn = _series(rows, 'planner_selected_chain_churn_ratio')
    if not np.any(np.isfinite(churn)):
        churn = _series(rows, 'planner_selected_edge_churn_ratio')

    dt = _median_dt(t)
    lag_samples, lag_sec = _estimate_lag(desired, actual, dt)
    duration_sec = float(np.nanmax(t) - np.nanmin(t)) if np.any(np.isfinite(t)) else float('nan')

    summary = {
        'sample_count': float(len(rows)),
        'duration_sec': duration_sec,
        'cte_rms_m': _nanrms(cte),
        'cte_p95_abs_m': _nanpercentile_abs(cte, 95.0),
        'cte_max_abs_m': _nanmaxabs(cte),
        'heading_error_rms_rad': _nanrms(heading),
        'heading_error_p95_abs_rad': _nanpercentile_abs(heading, 95.0),
        'steering_error_rms_rad': _nanrms(steering_error),
        'steering_error_p95_abs_rad': _nanpercentile_abs(steering_error, 95.0),
        'lag_samples': float(lag_samples),
        'lag_sec': float(lag_sec),
        'steering_saturation_ratio': _nanmean(steering_saturated),
        'steering_activity_total_variation_rad': _total_variation(desired),
        'steering_activity_rms_rate_radps': _rms_rate(desired, dt),
        'IAE_cte': _integral_abs(cte, dt),
        'ISE_cte': _integral_square(cte, dt),
        'corr_abs_cte_vs_planner_jump': _nancorr(np.abs(cte), jump),
        'corr_abs_cte_vs_planner_churn': _nancorr(np.abs(cte), churn),
    }
    return summary


def write_thesis_summary_files(summary: dict[str, float], json_path: Path, txt_path: Path) -> None:
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    lines = [
        'Thesis Controller Diagnostics Summary',
        f"sample_count: {summary.get('sample_count', float('nan')):.0f}",
        f"duration_sec: {summary.get('duration_sec', float('nan')):.3f}",
        f"cte_rms_m: {summary.get('cte_rms_m', float('nan')):.6f}",
        f"cte_p95_abs_m: {summary.get('cte_p95_abs_m', float('nan')):.6f}",
        f"cte_max_abs_m: {summary.get('cte_max_abs_m', float('nan')):.6f}",
        f"heading_error_rms_rad: {summary.get('heading_error_rms_rad', float('nan')):.6f}",
        f"heading_error_p95_abs_rad: {summary.get('heading_error_p95_abs_rad', float('nan')):.6f}",
        f"steering_error_rms_rad: {summary.get('steering_error_rms_rad', float('nan')):.6f}",
        f"steering_error_p95_abs_rad: {summary.get('steering_error_p95_abs_rad', float('nan')):.6f}",
        f"lag_sec: {summary.get('lag_sec', float('nan')):.6f}",
        f"steering_saturation_ratio: {summary.get('steering_saturation_ratio', float('nan')):.6f}",
        f"steering_activity_total_variation_rad: {summary.get('steering_activity_total_variation_rad', float('nan')):.6f}",
        f"steering_activity_rms_rate_radps: {summary.get('steering_activity_rms_rate_radps', float('nan')):.6f}",
        f"IAE_cte: {summary.get('IAE_cte', float('nan')):.6f}",
        f"ISE_cte: {summary.get('ISE_cte', float('nan')):.6f}",
        f"corr_abs_cte_vs_planner_jump: {summary.get('corr_abs_cte_vs_planner_jump', float('nan')):.6f}",
        f"corr_abs_cte_vs_planner_churn: {summary.get('corr_abs_cte_vs_planner_churn', float('nan')):.6f}",
    ]
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def compute_path_length_m(path_xy: np.ndarray) -> float:
    if path_xy.shape[0] < 2:
        return float('nan')
    diffs = np.diff(path_xy, axis=0)
    lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    if lengths.size == 0:
        return float('nan')
    return float(np.sum(lengths))


def compute_path_curvature_abs_p95(path_xy: np.ndarray) -> float:
    if path_xy.shape[0] < 3:
        return float('nan')
    diffs = np.diff(path_xy, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    valid = seg_len > 1e-6
    if np.count_nonzero(valid) < 2:
        return float('nan')
    headings = np.arctan2(diffs[valid, 1], diffs[valid, 0])
    if headings.size < 2:
        return float('nan')
    delta_heading = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    ds = 0.5 * (seg_len[valid][1:] + seg_len[valid][:-1])
    valid_ds = ds > 1e-6
    if not np.any(valid_ds):
        return float('nan')
    curvature = np.abs(delta_heading[valid_ds] / ds[valid_ds])
    if curvature.size == 0:
        return float('nan')
    return float(np.percentile(curvature, 95.0))


def _read_rows(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, float]] = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                row[str(key)] = _safe_float(value)
            rows.append(row)
    return rows


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out if math.isfinite(out) else float('nan')


def _series(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([_safe_float(row.get(key, float('nan'))) for row in rows], dtype=np.float64)


def _median_dt(timestamps: np.ndarray) -> float:
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size < 2:
        return float('nan')
    dt = np.diff(np.sort(finite))
    dt = dt[dt > 1e-6]
    if dt.size == 0:
        return float('nan')
    return float(np.median(dt))


def _estimate_lag(desired: np.ndarray, actual: np.ndarray, dt: float) -> tuple[int, float]:
    mask = np.isfinite(desired) & np.isfinite(actual)
    if np.count_nonzero(mask) < 8:
        return 0, float('nan')
    x = desired[mask] - np.mean(desired[mask])
    y = actual[mask] - np.mean(actual[mask])
    corr = np.correlate(y, x, mode='full')
    lags = np.arange(-len(x) + 1, len(x), dtype=np.int64)
    best_idx = int(np.argmax(corr))
    lag_samples = int(lags[best_idx])
    lag_sec = float(lag_samples * dt) if math.isfinite(dt) else float('nan')
    return lag_samples, lag_sec


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float('nan')
    return float(np.mean(finite))


def _nanrms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean(np.square(finite))))


def _nanmaxabs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float('nan')
    return float(np.max(np.abs(finite)))


def _nanpercentile_abs(values: np.ndarray, percentile: float) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return float('nan')
    return float(np.percentile(finite, percentile))


def _integral_abs(values: np.ndarray, dt: float) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0 or not math.isfinite(dt):
        return float('nan')
    return float(np.sum(finite) * dt)


def _integral_square(values: np.ndarray, dt: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not math.isfinite(dt):
        return float('nan')
    return float(np.sum(np.square(finite)) * dt)


def _total_variation(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float('nan')
    return float(np.sum(np.abs(np.diff(finite))))


def _rms_rate(values: np.ndarray, dt: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2 or not math.isfinite(dt) or dt <= 1e-6:
        return float('nan')
    rates = np.diff(finite) / dt
    return float(np.sqrt(np.mean(np.square(rates))))


def _nancorr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return float('nan')
    a_masked = a[mask]
    b_masked = b[mask]
    if np.std(a_masked) <= 1e-12 or np.std(b_masked) <= 1e-12:
        return float('nan')
    return float(np.corrcoef(a_masked, b_masked)[0, 1])
