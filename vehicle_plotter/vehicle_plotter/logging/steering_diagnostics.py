"""Steering/path diagnostics helpers for logger_node."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ._stats import (
    safe_float as _safe_float,
    median_dt as _median_dt,
    estimate_lag as _estimate_lag,
    nanmean as _nanmean,
    nanrms as _nanrms,
    nanmaxabs as _nanmaxabs,
    nancorr as _nancorr,
)

CORRIDOR_ANALYSIS_SAMPLE_PREFIXES = (
    "corridor_raw_anchor",
    "corridor_prevalidation_centerline",
    "corridor_buffer_centerline",
    "corridor_control_path",
)
CORRIDOR_ANALYSIS_SAMPLE_COUNT = 8
CORRIDOR_ANALYSIS_SAMPLE_SPACING_M = 1.0
CONE_AUDIT_DIAG_KEYS = [
    "cone_audit_received_count",
    "cone_audit_used_left_count",
    "cone_audit_used_right_count",
    "cone_audit_stale_count",
    "cone_audit_live_count",
    "cone_audit_used_left_chain_count",
    "cone_audit_used_right_chain_count",
    "cone_audit_chain_step_too_close_count",
    "cone_audit_chain_step_too_far_count",
    "cone_audit_chain_no_forward_progress_count",
    "cone_audit_chain_radial_regression_count",
    "cone_audit_chain_forward_projection_count",
    "cone_audit_chain_heading_change_count",
    "cone_audit_chain_shadowed_count",
    "cone_audit_chain_not_best_next_step_count",
    "cone_audit_chain_no_forward_seed_count",
    "cone_audit_chain_unreached_count",
    "cone_audit_rejected_geometry_range_count",
    "cone_audit_rejected_geometry_behind_count",
    "cone_audit_rejected_geometry_horizon_count",
    "cone_audit_rejected_geometry_lateral_count",
    "cone_audit_rejected_confidence_count",
    "cone_audit_rejected_tentative_count",
    "cone_audit_rejected_color_count",
    "cone_audit_rejected_nonfinite_count",
]
CORRIDOR_PAIR_AUDIT_DIAG_KEYS = [
    "corridor_pair_audit_total_count",
    "corridor_pair_audit_rejected_count",
    "corridor_pair_audit_pair_valid_count",
    "corridor_pair_audit_pair_width_too_narrow_count",
    "corridor_pair_audit_pair_width_too_wide_count",
    "corridor_pair_audit_pair_left_behind_count",
    "corridor_pair_audit_pair_right_behind_count",
    "corridor_pair_audit_pair_left_beyond_horizon_count",
    "corridor_pair_audit_pair_right_beyond_horizon_count",
    "corridor_pair_audit_pair_not_in_longest_valid_slice_count",
    "corridor_pair_audit_pair_nonfinite_count",
    "corridor_pair_audit_pair_rejected_unknown_count",
]


def corridor_analysis_sample_metric_keys() -> list[str]:
    keys: list[str] = []
    for prefix in CORRIDOR_ANALYSIS_SAMPLE_PREFIXES:
        keys.append(f"{prefix}_point_count")
        for idx in range(CORRIDOR_ANALYSIS_SAMPLE_COUNT):
            keys.append(f"{prefix}_p{idx}_x_m")
            keys.append(f"{prefix}_p{idx}_y_m")
    return keys


PLANNER_DIAG_TEXT_DEFAULTS: dict[str, str] = {
    "candidate_source": "",
    "midline_estimation_mode": "",
    "midline_update_mode": "",
    "publish_mode": "",
    "hold_reason": "",
    "operator_reason": "",
    "planner_mode": "",
}

PLANNER_DIAG_DEFAULTS: dict[str, float] = {
    "centerline_jump_max_m": float("nan"),
    "selected_edge_churn_ratio": float("nan"),
    "selected_chain_churn_ratio": float("nan"),
    "tracked_cones_frame_delta_p95_m": float("nan"),
    "heading_error_rad": float("nan"),
    "cross_track_error_m": float("nan"),
    "vehicle_speed_mps": float("nan"),
    "speed_term_mps": float("nan"),
    "heading_contribution_rad": float("nan"),
    "cross_track_contribution_rad": float("nan"),
    "yaw_rate_damping_contribution_rad": float("nan"),
    "yaw_rate_rps": float("nan"),
    "raw_steering_cmd_rad": float("nan"),
    "steering_after_clamp_rad": float("nan"),
    "steering_after_filter_rad": float("nan"),
    "steering_after_rate_limit_rad": float("nan"),
    "final_steering_cmd_rad": float("nan"),
    "steering_saturated_flag": float("nan"),
    "nearest_path_index": float("nan"),
    "heading_path_index": float("nan"),
    "target_point_x_base_m": float("nan"),
    "target_point_y_base_m": float("nan"),
    "target_point_x_frame_m": float("nan"),
    "target_point_y_frame_m": float("nan"),
    "plan_valid_flag": float("nan"),
    "plan_hold_active_flag": float("nan"),
    "plan_fallback_flag": float("nan"),
    "centerline_point_count": float("nan"),
    "midline_estimated_point_count": float("nan"),
    "midline_estimated_extent_m": float("nan"),
    "midline_live_prefix_extent_m": float("nan"),
    "midline_estimation_join_lateral_m": float("nan"),
    "midline_estimation_join_heading_rad": float("nan"),
    "selected_edge_count": float("nan"),
    "selected_chain_length": float("nan"),
    "selected_chain_median_width_m": float("nan"),
    "expected_width_prior_m": float("nan"),
    "corridor_sample_count": float("nan"),
    "corridor_width_min_m": float("nan"),
    "corridor_width_median_m": float("nan"),
    "corridor_width_max_m": float("nan"),
    "left_chain_length": float("nan"),
    "right_chain_length": float("nan"),
    "path_length_m": float("nan"),
    "path_curvature_abs_p95_1pm": float("nan"),
    "planner_state_code": float("nan"),
    "fresh_publish_flag": float("nan"),
    "held_publish_flag": float("nan"),
    "stopped_flag": float("nan"),
    "waiting_flag": float("nan"),
    "operator_reason_code": float("nan"),
    "hold_remaining_s": float("nan"),
    "control_path_point_count": float("nan"),
    "zero_cmd_sent_flag": float("nan"),
    "state_vx_mps": float("nan"),
    "state_vy_mps": float("nan"),
    "state_speed_mps": float("nan"),
    "state_yaw_rad": float("nan"),
    "state_yaw_rate_rps": float("nan"),
    "state_ax_mps2": float("nan"),
    "state_ay_mps2": float("nan"),
    "desired_accel_mps2": float("nan"),
    "actual_accel_mps2": float("nan"),
    "desired_steering_rad": float("nan"),
    "actual_steering_rad": float("nan"),
    "steering_tracking_error_rad": float("nan"),
    "sideslip_rad": float("nan"),
    "slip_angle_front_rad": float("nan"),
    "slip_angle_rear_rad": float("nan"),
    "kinematic_blend": float("nan"),
    "kinematic_vy_ref_mps": float("nan"),
    "kinematic_yaw_rate_ref_rps": float("nan"),
}
for _key in corridor_analysis_sample_metric_keys():
    PLANNER_DIAG_DEFAULTS[_key] = float("nan")
for _key in CONE_AUDIT_DIAG_KEYS:
    PLANNER_DIAG_DEFAULTS[_key] = float("nan")
for _key in CORRIDOR_PAIR_AUDIT_DIAG_KEYS:
    PLANNER_DIAG_DEFAULTS[_key] = float("nan")

STEERING_DIAG_FIELDNAMES: list[str] = [
    "timestamp_sec",
    "cmd_stamp_sec",
    "cmd_age_sec",
    "desired_steering_rad",
    "desired_speed_mps",
    "actual_steering_deg",
    "actual_steering_rad",
    "steering_error_rad",
    "steering_error_abs_rad",
    "raw_steering_cmd_rad",
    "final_steering_cmd_rad",
    "steering_after_clamp_rad",
    "steering_after_filter_rad",
    "steering_after_rate_limit_rad",
    "steering_saturated_flag",
    "vehicle_x_m",
    "vehicle_y_m",
    "vehicle_yaw_rad",
    "vehicle_yaw_rate_rps",
    "vehicle_speed_mps",
    "physics_state_vx_mps",
    "physics_state_vy_mps",
    "physics_state_speed_mps",
    "physics_state_yaw_rad",
    "physics_state_yaw_rate_rps",
    "physics_state_ax_mps2",
    "physics_state_ay_mps2",
    "physics_desired_accel_mps2",
    "physics_actual_accel_mps2",
    "physics_desired_steering_rad",
    "physics_actual_steering_rad",
    "physics_steering_tracking_error_rad",
    "physics_sideslip_rad",
    "physics_slip_angle_front_rad",
    "physics_slip_angle_rear_rad",
    "physics_kinematic_blend",
    "physics_kinematic_vy_ref_mps",
    "physics_kinematic_yaw_rate_ref_rps",
    "speed_term_mps",
    "centerline_available",
    "centerline_point_count",
    "cte_m",
    "cte_abs_m",
    "heading_error_rad",
    "heading_error_abs_rad",
    "heading_contribution_rad",
    "cross_track_contribution_rad",
    "yaw_rate_damping_contribution_rad",
    "nearest_path_index",
    "heading_path_index",
    "target_point_x_base_m",
    "target_point_y_base_m",
    "target_point_x_frame_m",
    "target_point_y_frame_m",
    "nearest_path_point_x_m",
    "nearest_path_point_y_m",
    "planner_centerline_jump_max_m",
    "planner_selected_edge_churn_ratio",
    "planner_selected_chain_churn_ratio",
    "planner_tracked_cones_frame_delta_p95_m",
    "planner_state_code",
    "planner_fresh_publish_flag",
    "planner_held_publish_flag",
    "planner_stopped_flag",
    "planner_waiting_flag",
    "planner_operator_reason_code",
    "planner_hold_remaining_s",
    "planner_control_path_point_count",
    "planner_zero_cmd_sent_flag",
]
STEERING_DIAG_FIELDNAMES.extend(f"planner_{key}" for key in CONE_AUDIT_DIAG_KEYS)


def normalize_angle(angle_rad: float) -> float:
    """Wrap angle into [-pi, pi]."""
    out = float(angle_rad)
    while out > math.pi:
        out -= 2.0 * math.pi
    while out < -math.pi:
        out += 2.0 * math.pi
    return out


def heading_error(vehicle_yaw: float, tangent_yaw: float) -> float:
    """Signed heading error (vehicle minus path tangent) in radians."""
    return normalize_angle(float(vehicle_yaw) - float(tangent_yaw))


def nearest_point_on_polyline(
    x: float,
    y: float,
    path_xy: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Return nearest segment index and nearest point on that segment/polyline."""
    if path_xy.shape[0] == 0:
        return -1, np.array([float("nan"), float("nan")], dtype=np.float64)
    if path_xy.shape[0] == 1:
        return 0, np.array(path_xy[0], dtype=np.float64)

    p = np.array([float(x), float(y)], dtype=np.float64)
    best_idx = -1
    best_point = np.array([float("nan"), float("nan")], dtype=np.float64)
    best_dist_sq = float("inf")

    for idx in range(path_xy.shape[0] - 1):
        a = path_xy[idx]
        b = path_xy[idx + 1]
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq <= 1e-12:
            cand = np.array(a, dtype=np.float64)
        else:
            t = float(np.dot(p - a, ab) / ab_len_sq)
            t = float(np.clip(t, 0.0, 1.0))
            cand = a + (t * ab)
        dxy = p - cand
        d2 = float(np.dot(dxy, dxy))
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best_idx = idx
            best_point = cand

    return best_idx, best_point


def signed_cross_track_error(
    x: float,
    y: float,
    path_xy: np.ndarray,
) -> tuple[float, float]:
    """Return signed CTE and local path tangent yaw at nearest segment."""
    seg_idx, nearest = nearest_point_on_polyline(x, y, path_xy)
    if seg_idx < 0 or path_xy.shape[0] < 2:
        return float("nan"), float("nan")

    a = path_xy[seg_idx]
    b = path_xy[min(seg_idx + 1, path_xy.shape[0] - 1)]
    tangent = b - a
    tan_norm = float(np.hypot(tangent[0], tangent[1]))
    if tan_norm <= 1e-9:
        return float("nan"), float("nan")
    tx = float(tangent[0] / tan_norm)
    ty = float(tangent[1] / tan_norm)
    nx = -ty
    ny = tx
    vx = float(x) - float(nearest[0])
    vy = float(y) - float(nearest[1])
    cte = (vx * nx) + (vy * ny)
    tangent_yaw = math.atan2(ty, tx)
    return float(cte), float(tangent_yaw)


def parse_planner_diag(diag_msg: Any) -> dict[str, float]:
    """Extract planner stability + control-debug values from DiagnosticArray-like message."""
    defaults = dict(PLANNER_DIAG_DEFAULTS)
    statuses = getattr(diag_msg, "status", None)
    if statuses is None:
        return defaults

    for status in statuses:
        name = str(getattr(status, "name", ""))
        if not (
            name.endswith("/stability")
            or name.endswith("/control_debug")
            or name.endswith("/thesis_context")
            or name.endswith("/physics_debug")
        ):
            continue
        values = getattr(status, "values", [])
        for item in values:
            key = str(getattr(item, "key", ""))
            value = _safe_float(getattr(item, "value", "nan"))
            if key in defaults:
                defaults[key] = value
    if not math.isfinite(defaults["selected_chain_churn_ratio"]):
        defaults["selected_chain_churn_ratio"] = float(defaults["selected_edge_churn_ratio"])
    if not math.isfinite(defaults["selected_edge_churn_ratio"]):
        defaults["selected_edge_churn_ratio"] = float(defaults["selected_chain_churn_ratio"])
    return defaults


def parse_planner_diag_text(diag_msg: Any) -> dict[str, str]:
    """Extract planner text metrics from DiagnosticArray-like message."""
    defaults = dict(PLANNER_DIAG_TEXT_DEFAULTS)
    statuses = getattr(diag_msg, "status", None)
    if statuses is None:
        return defaults

    for status in statuses:
        name = str(getattr(status, "name", ""))
        if not name.endswith("/stability"):
            continue
        values = getattr(status, "values", [])
        for item in values:
            key = str(getattr(item, "key", ""))
            if key not in defaults:
                continue
            defaults[key] = str(getattr(item, "value", "")).strip()
    return defaults


def analyze_csv(csv_path: Path) -> dict[str, float]:
    """Compute summary metrics from steering tracking diagnostics CSV."""
    rows = _read_rows(csv_path)
    if not rows:
        return {"sample_count": 0.0}

    t = np.asarray([_safe_float(r.get("timestamp_sec", "nan")) for r in rows], dtype=np.float64)
    steering_error = np.asarray([_safe_float(r.get("steering_error_rad", "nan")) for r in rows], dtype=np.float64)
    cte = np.asarray([_safe_float(r.get("cte_m", "nan")) for r in rows], dtype=np.float64)
    heading = np.asarray([_safe_float(r.get("heading_error_rad", "nan")) for r in rows], dtype=np.float64)
    desired = np.asarray([_safe_float(r.get("desired_steering_rad", "nan")) for r in rows], dtype=np.float64)
    actual = np.asarray([_safe_float(r.get("actual_steering_rad", "nan")) for r in rows], dtype=np.float64)
    churn = np.asarray(
        [
            _safe_float(
                r.get(
                    "planner_selected_chain_churn_ratio",
                    r.get("planner_selected_edge_churn_ratio", "nan"),
                )
            )
            for r in rows
        ],
        dtype=np.float64,
    )
    jump = np.asarray([_safe_float(r.get("planner_centerline_jump_max_m", "nan")) for r in rows], dtype=np.float64)

    dt = _median_dt(t)
    lag_samples, lag_sec = _estimate_lag(desired, actual, dt)
    dom_hz, dom_ratio = _dominant_frequency_metrics(steering_error, dt)

    summary = {
        "sample_count": float(len(rows)),
        "duration_sec": float(np.nanmax(t) - np.nanmin(t)) if np.any(np.isfinite(t)) else float("nan"),
        "steering_error_mean_rad": _nanmean(steering_error),
        "steering_error_std_rad": _nanstd(steering_error),
        "steering_error_rms_rad": _nanrms(steering_error),
        "steering_error_max_abs_rad": _nanmaxabs(steering_error),
        "cte_mean_m": _nanmean(cte),
        "cte_std_m": _nanstd(cte),
        "cte_rms_m": _nanrms(cte),
        "cte_max_abs_m": _nanmaxabs(cte),
        "heading_error_rms_rad": _nanrms(heading),
        "heading_error_max_abs_rad": _nanmaxabs(heading),
        "lag_samples": float(lag_samples),
        "lag_sec": float(lag_sec),
        "dominant_steering_error_hz": float(dom_hz),
        "dominant_steering_error_ratio": float(dom_ratio),
        "corr_abs_steering_error_vs_churn": _nancorr(np.abs(steering_error), churn),
        "corr_abs_cte_vs_jump": _nancorr(np.abs(cte), jump),
    }
    return summary


def write_summary_files(summary: dict[str, float], json_path: Path, txt_path: Path) -> None:
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "Steering Tracking Diagnostics Summary",
        f"sample_count: {summary.get('sample_count', float('nan')):.0f}",
        f"duration_sec: {summary.get('duration_sec', float('nan')):.3f}",
        f"steering_error_rms_rad: {summary.get('steering_error_rms_rad', float('nan')):.6f}",
        f"steering_error_max_abs_rad: {summary.get('steering_error_max_abs_rad', float('nan')):.6f}",
        f"cte_rms_m: {summary.get('cte_rms_m', float('nan')):.6f}",
        f"cte_max_abs_m: {summary.get('cte_max_abs_m', float('nan')):.6f}",
        f"heading_error_rms_rad: {summary.get('heading_error_rms_rad', float('nan')):.6f}",
        f"lag_sec: {summary.get('lag_sec', float('nan')):.6f}",
        f"lag_samples: {summary.get('lag_samples', float('nan')):.1f}",
        f"dominant_steering_error_hz: {summary.get('dominant_steering_error_hz', float('nan')):.6f}",
        f"dominant_steering_error_ratio: {summary.get('dominant_steering_error_ratio', float('nan')):.6f}",
        f"corr_abs_steering_error_vs_churn: {summary.get('corr_abs_steering_error_vs_churn', float('nan')):.6f}",
        f"corr_abs_cte_vs_jump: {summary.get('corr_abs_cte_vs_jump', float('nan')):.6f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row]


def _nanstd(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.std(valid))


def _dominant_frequency_metrics(signal: np.ndarray, dt: float) -> tuple[float, float]:
    if not math.isfinite(dt) or dt <= 1e-6:
        return float("nan"), float("nan")
    data = signal[np.isfinite(signal)]
    if data.size < 16:
        return float("nan"), float("nan")
    detrended = data - np.mean(data)
    spec = np.fft.rfft(detrended)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(detrended.size, d=dt)
    if mag.size <= 1:
        return float("nan"), float("nan")
    mag[0] = 0.0
    peak_idx = int(np.argmax(mag))
    peak = float(mag[peak_idx])
    mean_mag = float(np.mean(mag[1:])) if mag.size > 1 else float("nan")
    ratio = float(peak / mean_mag) if mean_mag > 1e-12 else float("nan")
    return float(freqs[peak_idx]), ratio
