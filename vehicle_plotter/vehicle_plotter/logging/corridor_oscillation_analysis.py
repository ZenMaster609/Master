"""Offline analysis helpers for corridor planner path decomposition."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .steering_diagnostics import (
    CORRIDOR_ANALYSIS_SAMPLE_COUNT,
    CORRIDOR_ANALYSIS_SAMPLE_PREFIXES,
    CORRIDOR_ANALYSIS_SAMPLE_SPACING_M,
)


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _read_rows(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
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


def _series(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([_safe_float(row.get(key, float("nan"))) for row in rows], dtype=np.float64)


def _sample_matrix(rows: list[dict[str, float]], prefix: str, axis: str) -> np.ndarray:
    cols = [f"{prefix}_p{idx}_{axis}_m" for idx in range(CORRIDOR_ANALYSIS_SAMPLE_COUNT)]
    return np.asarray(
        [[_safe_float(row.get(col, float("nan"))) for col in cols] for row in rows],
        dtype=np.float64,
    )


def _count_finite_rows(matrix: np.ndarray) -> int:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return 0
    return int(np.count_nonzero(np.any(np.isfinite(matrix), axis=1)))


def _nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _nanrms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def _nanpercentile_abs(values: np.ndarray, percentile: float) -> float:
    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def analyze_corridor_oscillation(
    thesis_csv_path: Path,
    path_tracking_csv_path: Optional[Path] = None,
) -> dict[str, float]:
    thesis_rows = _read_rows(thesis_csv_path)
    path_rows = _read_rows(path_tracking_csv_path) if path_tracking_csv_path is not None else []
    if not thesis_rows:
        return {"sample_count": 0.0}

    buffer_y = _sample_matrix(thesis_rows, "corridor_buffer_centerline", "y")
    prevalidation_y = _sample_matrix(thesis_rows, "corridor_prevalidation_centerline", "y")
    control_y = _sample_matrix(thesis_rows, "corridor_control_path", "y")
    jump = _series(thesis_rows, "planner_centerline_jump_max_m")
    churn = _series(thesis_rows, "planner_selected_chain_churn_ratio")
    if not np.any(np.isfinite(churn)):
        churn = _series(thesis_rows, "planner_selected_edge_churn_ratio")

    buffer_minus_prevalidation = np.abs(buffer_y - prevalidation_y)
    control_minus_buffer = np.abs(control_y - buffer_y)

    summary = {
        "sample_count": float(len(thesis_rows)),
        "profile_row_count": float(_count_finite_rows(control_y)),
        "planner_jump_mean_m": _nanmean(jump),
        "planner_jump_rms_m": _nanrms(jump),
        "planner_churn_mean": _nanmean(churn),
        "buffer_vs_prevalidation_abs_mean_m": _nanmean(buffer_minus_prevalidation),
        "buffer_vs_prevalidation_abs_p95_m": _nanpercentile_abs(buffer_minus_prevalidation, 95.0),
        "control_vs_buffer_abs_mean_m": _nanmean(control_minus_buffer),
        "control_vs_buffer_abs_p95_m": _nanpercentile_abs(control_minus_buffer, 95.0),
    }

    if path_rows:
        valid_mask = _series(path_rows, "sample_valid_flag") > 0.5
        planner_cte = _series(path_rows, "planner_reference_vs_gt_cte_m")[valid_mask]
        summary["planner_reference_vs_gt_cte_rms_m"] = _nanrms(planner_cte)
        summary["planner_reference_vs_gt_cte_p95_abs_m"] = _nanpercentile_abs(planner_cte, 95.0)
    return summary


def write_corridor_oscillation_summary_files(
    summary: dict[str, float],
    json_path: Path,
    txt_path: Path,
) -> None:
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "Corridor Oscillation Analysis Summary",
        f"sample_count: {summary.get('sample_count', float('nan')):.0f}",
        f"profile_row_count: {summary.get('profile_row_count', float('nan')):.0f}",
        f"planner_jump_mean_m: {summary.get('planner_jump_mean_m', float('nan')):.6f}",
        f"planner_jump_rms_m: {summary.get('planner_jump_rms_m', float('nan')):.6f}",
        f"planner_churn_mean: {summary.get('planner_churn_mean', float('nan')):.6f}",
        f"buffer_vs_prevalidation_abs_mean_m: {summary.get('buffer_vs_prevalidation_abs_mean_m', float('nan')):.6f}",
        f"buffer_vs_prevalidation_abs_p95_m: {summary.get('buffer_vs_prevalidation_abs_p95_m', float('nan')):.6f}",
        f"control_vs_buffer_abs_mean_m: {summary.get('control_vs_buffer_abs_mean_m', float('nan')):.6f}",
        f"control_vs_buffer_abs_p95_m: {summary.get('control_vs_buffer_abs_p95_m', float('nan')):.6f}",
        f"planner_reference_vs_gt_cte_rms_m: {summary.get('planner_reference_vs_gt_cte_rms_m', float('nan')):.6f}",
        f"planner_reference_vs_gt_cte_p95_abs_m: {summary.get('planner_reference_vs_gt_cte_p95_abs_m', float('nan')):.6f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_percentile_band(ax, x: np.ndarray, matrix: np.ndarray, *, label: str, color: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return
    med = np.nanmedian(matrix, axis=0)
    lo = np.nanpercentile(matrix, 10.0, axis=0)
    hi = np.nanpercentile(matrix, 90.0, axis=0)
    valid = np.isfinite(med)
    if not np.any(valid):
        return
    ax.plot(x[valid], med[valid], label=label, color=color, linewidth=1.8)
    ax.fill_between(x[valid], lo[valid], hi[valid], color=color, alpha=0.15)


def _plot_path_error_over_arclength(ax, rows: list[dict[str, float]]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "path tracking eval unavailable", ha="center", va="center", transform=ax.transAxes)
        return
    sample_valid = _series(rows, "sample_valid_flag") > 0.5
    s = _series(rows, "planner_reference_s_m")[sample_valid]
    cte = _series(rows, "planner_reference_vs_gt_cte_m")[sample_valid]
    finite = np.isfinite(s) & np.isfinite(cte)
    if not np.any(finite):
        ax.text(0.5, 0.5, "planner arclength unavailable", ha="center", va="center", transform=ax.transAxes)
        return
    s = s[finite]
    cte = cte[finite]
    if s.size < 4:
        ax.plot(s, cte, color="#2ca02c", linewidth=1.5)
        return
    bins = np.linspace(float(np.min(s)), float(np.max(s)), min(32, max(8, int(np.sqrt(s.size)))))
    if bins.size < 3:
        ax.plot(s, cte, color="#2ca02c", linewidth=1.5)
        return
    centers: list[float] = []
    mean_signed: list[float] = []
    p95_abs: list[float] = []
    for idx in range(bins.size - 1):
        lo = bins[idx]
        hi = bins[idx + 1]
        if idx == bins.size - 2:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        if not np.any(mask):
            continue
        centers.append(0.5 * (lo + hi))
        segment = cte[mask]
        mean_signed.append(float(np.mean(segment)))
        p95_abs.append(float(np.percentile(np.abs(segment), 95.0)))
    if not centers:
        ax.plot(s, cte, color="#2ca02c", linewidth=1.5)
        return
    x = np.asarray(centers, dtype=np.float64)
    mean_signed_arr = np.asarray(mean_signed, dtype=np.float64)
    p95_abs_arr = np.asarray(p95_abs, dtype=np.float64)
    ax.plot(x, mean_signed_arr, color="#2ca02c", linewidth=1.8, label="Mean signed CTE")
    ax.plot(x, p95_abs_arr, color="#d62728", linewidth=1.5, label="P95 abs CTE")
    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0, alpha=0.5)
    ax.legend(loc="upper right", fontsize="small")


def generate_corridor_oscillation_plot(
    thesis_csv_path: Path,
    path_tracking_csv_path: Optional[Path],
    output_path: Path,
    *,
    title: str = "Corridor Planner Oscillation Analysis",
    dpi: int = 150,
) -> Optional[Path]:
    thesis_rows = _read_rows(thesis_csv_path)
    if not thesis_rows:
        return None
    path_rows = _read_rows(path_tracking_csv_path) if path_tracking_csv_path is not None else []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    station_x = (
        np.arange(CORRIDOR_ANALYSIS_SAMPLE_COUNT, dtype=np.float64)
        * float(CORRIDOR_ANALYSIS_SAMPLE_SPACING_M)
    )
    raw_anchor_y = _sample_matrix(thesis_rows, "corridor_raw_anchor", "y")
    prevalidation_y = _sample_matrix(thesis_rows, "corridor_prevalidation_centerline", "y")
    buffer_y = _sample_matrix(thesis_rows, "corridor_buffer_centerline", "y")
    control_y = _sample_matrix(thesis_rows, "corridor_control_path", "y")
    t = _series(thesis_rows, "timestamp_sec")
    finite_t = t[np.isfinite(t)]
    if finite_t.size:
        t = t - float(finite_t[0])
    else:
        t = np.arange(len(thesis_rows), dtype=np.float64)
    jump = _series(thesis_rows, "planner_centerline_jump_max_m")
    churn = _series(thesis_rows, "planner_selected_chain_churn_ratio")
    if not np.any(np.isfinite(churn)):
        churn = _series(thesis_rows, "planner_selected_edge_churn_ratio")
    hold_flag = _series(thesis_rows, "plan_hold_active_flag")

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0))

    ax = axes[0, 0]
    _plot_percentile_band(ax, station_x, raw_anchor_y, label="Raw anchors", color="#7f7f7f")
    _plot_percentile_band(ax, station_x, prevalidation_y, label="Prevalidation", color="#ff7f0e")
    _plot_percentile_band(ax, station_x, buffer_y, label="Post-buffer", color="#1f77b4")
    _plot_percentile_band(ax, station_x, control_y, label="Control path", color="#d62728")
    ax.set_title("Near-Field Lateral Profile")
    ax.set_xlabel("Distance Ahead (m)")
    ax.set_ylabel("Lateral Y in vehicle frame (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")

    ax = axes[0, 1]
    _plot_percentile_band(
        ax,
        station_x,
        np.abs(buffer_y - prevalidation_y),
        label="|buffer - prevalidation|",
        color="#1f77b4",
    )
    _plot_percentile_band(
        ax,
        station_x,
        np.abs(control_y - buffer_y),
        label="|control - buffer|",
        color="#d62728",
    )
    ax.set_title("Decomposition Error by Station")
    ax.set_xlabel("Distance Ahead (m)")
    ax.set_ylabel("Absolute lateral delta (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")

    ax = axes[1, 0]
    _plot_path_error_over_arclength(ax, path_rows)
    ax.set_title("Planner GT Error Over Arclength")
    ax.set_xlabel("Planner reference arclength (m)")
    ax.set_ylabel("Planner ref vs GT CTE (m)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    valid_jump = np.isfinite(t) & np.isfinite(jump)
    valid_churn = np.isfinite(t) & np.isfinite(churn)
    valid_hold = np.isfinite(t) & np.isfinite(hold_flag)
    if np.any(valid_jump):
        ax.plot(t[valid_jump], jump[valid_jump], color="#d62728", linewidth=1.5, label="Centerline jump")
    if np.any(valid_churn):
        ax.plot(t[valid_churn], churn[valid_churn], color="#ff7f0e", linewidth=1.5, label="Edge churn")
    if np.any(valid_hold):
        ax.plot(t[valid_hold], hold_flag[valid_hold], color="#9467bd", linewidth=1.2, linestyle=":", label="Hold flag")
    ax.set_title("Planner Stability Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude / flag")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize="small")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
