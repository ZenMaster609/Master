"""Offline plots for GT midline path-tracking evaluation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .path_tracking_eval import signed_cross_track_error


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _read_rows(csv_path: Path) -> list[dict[str, float | str]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, float | str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            row: dict[str, float | str] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                if key in {"status", "frame_id", "gt_source_frame"}:
                    row[str(key)] = "" if value is None else str(value)
                else:
                    row[str(key)] = _safe_float(value)
            rows.append(row)
    return rows


def _series(rows: list[dict[str, float | str]], key: str) -> np.ndarray:
    return np.asarray([_safe_float(row.get(key, float("nan"))) for row in rows], dtype=np.float64)


def _xy_series(rows: list[dict[str, float | str]], x_key: str, y_key: str) -> np.ndarray:
    x = _series(rows, x_key)
    y = _series(rows, y_key)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack((x[valid], y[valid])).astype(np.float64)


def _relative_time(rows: list[dict[str, float | str]]) -> np.ndarray:
    t = _series(rows, "timestamp_sec")
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return np.arange(len(rows), dtype=np.float64)
    return t - float(finite[0])


def _legend_if_labeled(ax, **kwargs) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def _point_to_polyline_distance_m(point_xy: np.ndarray, path_xy: np.ndarray) -> float:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    point_xy = np.asarray(point_xy, dtype=np.float64)
    if path_xy.ndim != 2 or path_xy.shape[0] == 0:
        return float("nan")
    if path_xy.shape[0] == 1:
        return float(np.hypot(*(point_xy - path_xy[0])))

    best_dist_sq = float("inf")
    for idx in range(path_xy.shape[0] - 1):
        start = path_xy[idx]
        end = path_xy[idx + 1]
        delta = end - start
        denom = float(np.dot(delta, delta))
        if denom <= 1e-12:
            nearest = start
        else:
            t = float(np.clip(np.dot(point_xy - start, delta) / denom, 0.0, 1.0))
            nearest = start + (t * delta)
        dist_sq = float(np.dot(point_xy - nearest, point_xy - nearest))
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
    return float(math.sqrt(best_dist_sq)) if math.isfinite(best_dist_sq) else float("nan")


def _mean_distance_to_polyline_m(points_xy: np.ndarray, path_xy: np.ndarray) -> float:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[0] == 0 or path_xy.ndim != 2 or path_xy.shape[0] == 0:
        return float("nan")
    distances = np.asarray(
        [_point_to_polyline_distance_m(point_xy, path_xy) for point_xy in points_xy],
        dtype=np.float64,
    )
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _mean_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _mean_abs_cross_track_to_polyline_m(points_xy: np.ndarray, path_xy: np.ndarray) -> float:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[0] == 0 or path_xy.ndim != 2 or path_xy.shape[0] < 2:
        return float("nan")
    errors = []
    for point_xy in points_xy:
        cte_m, _ = signed_cross_track_error(float(point_xy[0]), float(point_xy[1]), path_xy)
        if math.isfinite(cte_m):
            errors.append(abs(float(cte_m)))
    if not errors:
        return float("nan")
    return float(np.mean(np.asarray(errors, dtype=np.float64)))


def compute_path_tracking_overlay_average_distances(
    csv_path: Path,
    *,
    gt_midline_xy: np.ndarray,
    planner_trace_xy: np.ndarray,
) -> dict[str, float]:
    rows = _read_rows(csv_path)
    gt_midline_xy = np.asarray(gt_midline_xy, dtype=np.float64)
    planner_trace_xy = np.asarray(planner_trace_xy, dtype=np.float64)
    if not rows:
        return {
            "planner_vs_gt_avg_dist_m": float("nan"),
            "controller_vs_gt_avg_dist_m": float("nan"),
            "controller_vs_planner_avg_dist_m": float("nan"),
        }

    planner_reference_xy = _xy_series(rows, "planner_reference_x_m", "planner_reference_y_m")
    controller_vs_planner_cte = np.abs(_series(rows, "controller_vs_planner_cte_m"))
    controller_vs_gt_cte = np.abs(_series(rows, "controller_vs_gt_cte_m"))
    return {
        "planner_vs_gt_avg_dist_m": _mean_abs_cross_track_to_polyline_m(planner_reference_xy, gt_midline_xy),
        "controller_vs_gt_avg_dist_m": _mean_finite(controller_vs_gt_cte),
        "controller_vs_planner_avg_dist_m": _mean_finite(controller_vs_planner_cte),
    }


def _format_distance_cm(value_m: float) -> str:
    if not math.isfinite(value_m):
        return "n/a"
    return f"{value_m * 100.0:.2f} cm"


def generate_path_tracking_cte_plot(
    csv_path: Path,
    output_path: Path,
    *,
    title: str = "Path Tracking Evaluation",
    dpi: int = 150,
) -> Optional[Path]:
    rows = _read_rows(csv_path)
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = _relative_time(rows)
    cte_planner = _series(rows, "controller_vs_planner_cte_m")
    cte_gt = _series(rows, "controller_vs_gt_cte_m")

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    valid_planner = np.isfinite(t) & np.isfinite(cte_planner)
    valid_gt = np.isfinite(t) & np.isfinite(cte_gt)
    if np.any(valid_planner):
        ax.plot(t[valid_planner], cte_planner[valid_planner], color="#d62728", linewidth=1.8, label="Controller vs Planner")
    if np.any(valid_gt):
        ax.plot(t[valid_gt], cte_gt[valid_gt], color="#1f77b4", linewidth=1.8, label="Controller vs GT")
    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signed CTE (m)")
    ax.grid(True, alpha=0.3)
    _legend_if_labeled(ax, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_path_tracking_overlay_plot(
    csv_path: Path,
    output_path: Path,
    *,
    gt_midline_xy: np.ndarray,
    gt_left_xy: np.ndarray,
    gt_right_xy: np.ndarray,
    planner_trace_xy: np.ndarray,
    title: str = "GT Midline vs Planner vs Driven Trajectory",
    dpi: int = 150,
) -> Optional[Path]:
    rows = _read_rows(csv_path)
    gt_midline_xy = np.asarray(gt_midline_xy, dtype=np.float64)
    gt_left_xy = np.asarray(gt_left_xy, dtype=np.float64)
    gt_right_xy = np.asarray(gt_right_xy, dtype=np.float64)
    planner_trace_xy = np.asarray(planner_trace_xy, dtype=np.float64)
    if not rows or gt_midline_xy.size == 0:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vehicle_x = _series(rows, "vehicle_x_m")
    vehicle_y = _series(rows, "vehicle_y_m")
    valid_vehicle = np.isfinite(vehicle_x) & np.isfinite(vehicle_y)
    avg_distances = compute_path_tracking_overlay_average_distances(
        csv_path,
        gt_midline_xy=gt_midline_xy,
        planner_trace_xy=planner_trace_xy,
    )

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    if gt_left_xy.ndim == 2 and gt_left_xy.shape[0] > 0:
        ax.plot(gt_left_xy[:, 0], gt_left_xy[:, 1], color="#1f77b4", linewidth=1.4, linestyle=":", label="GT blue border")
    if gt_right_xy.ndim == 2 and gt_right_xy.shape[0] > 0:
        ax.plot(gt_right_xy[:, 0], gt_right_xy[:, 1], color="#f1c40f", linewidth=1.4, linestyle=":", label="GT yellow border")
    ax.plot(gt_midline_xy[:, 0], gt_midline_xy[:, 1], color="#111111", linewidth=2.2, label="GT midline")
    if planner_trace_xy.ndim == 2 and planner_trace_xy.shape[0] > 0:
        ax.plot(
            planner_trace_xy[:, 0],
            planner_trace_xy[:, 1],
            color="#ff7f0e",
            linewidth=2.0,
            linestyle="--",
            label="Planner trace",
        )
    if np.any(valid_vehicle):
        ax.plot(vehicle_x[valid_vehicle], vehicle_y[valid_vehicle], color="#1f77b4", linewidth=1.8, label="Driven trajectory")
        ax.scatter(vehicle_x[np.where(valid_vehicle)[0][0]], vehicle_y[np.where(valid_vehicle)[0][0]], color="green", s=45, zorder=5, label="Start")
        ax.scatter(vehicle_x[np.where(valid_vehicle)[0][-1]], vehicle_y[np.where(valid_vehicle)[0][-1]], color="red", s=45, zorder=5, label="End")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    _legend_if_labeled(ax, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    avg_text = "\n".join(
        (
            f"Planner avg dist to GT midline: {_format_distance_cm(avg_distances['planner_vs_gt_avg_dist_m'])}",
            f"Controller avg dist to GT midline: {_format_distance_cm(avg_distances['controller_vs_gt_avg_dist_m'])}",
            f"Controller avg dist to planner trace: {_format_distance_cm(avg_distances['controller_vs_planner_avg_dist_m'])}",
        )
    )
    ax.text(
        0.02,
        0.98,
        avg_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#333333",
            "alpha": 0.92,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
