"""Offline plots for GT midline path-tracking evaluation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .path_tracking_eval import signed_cross_track_error
from ..plotting.matplotlib_fonts import (
    DEFAULT_TITLE_FONTSIZE,
    LEGEND_FONTSIZE,
    apply_axis_label_fontsize,
    apply_tick_label_fontsize,
)

OVERLAY_LEGEND_FONTSIZE = LEGEND_FONTSIZE * 1.2


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
                if key in {"status", "frame_id", "gt_source_frame", "odom_child_frame_id", "resolved_control_frame"}:
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


def _series_prefer(rows: list[dict[str, float | str]], preferred_key: str, fallback_key: str) -> np.ndarray:
    preferred = _series(rows, preferred_key)
    fallback = _series(rows, fallback_key)
    return np.where(np.isfinite(preferred), preferred, fallback)


def _xy_series_prefer(
    rows: list[dict[str, float | str]],
    preferred_x_key: str,
    preferred_y_key: str,
    fallback_x_key: str,
    fallback_y_key: str,
) -> np.ndarray:
    x = _series_prefer(rows, preferred_x_key, fallback_x_key)
    y = _series_prefer(rows, preferred_y_key, fallback_y_key)
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
        kwargs.setdefault("fontsize", LEGEND_FONTSIZE)
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


def _normalize_polyline_set(paths_xy: Optional[list[np.ndarray]]) -> list[np.ndarray]:
    if not paths_xy:
        return []
    normalized: list[np.ndarray] = []
    for path_xy in paths_xy:
        arr = np.asarray(path_xy, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > 0:
            normalized.append(arr)
    return normalized


def _point_to_polyline_set_distance_m(point_xy: np.ndarray, paths_xy: list[np.ndarray]) -> float:
    if not paths_xy:
        return float("nan")
    best = float("inf")
    for path_xy in paths_xy:
        dist_m = _point_to_polyline_distance_m(point_xy, path_xy)
        if math.isfinite(dist_m):
            best = min(best, dist_m)
    return best if math.isfinite(best) else float("nan")


def _mean_distance_to_polyline_set_m(points_xy: np.ndarray, paths_xy: list[np.ndarray]) -> float:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[0] == 0 or not paths_xy:
        return float("nan")
    distances = np.asarray(
        [_point_to_polyline_set_distance_m(point_xy, paths_xy) for point_xy in points_xy],
        dtype=np.float64,
    )
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _estimate_average_track_width_m(left_xy: np.ndarray, right_xy: np.ndarray) -> float:
    left_xy = np.asarray(left_xy, dtype=np.float64)
    right_xy = np.asarray(right_xy, dtype=np.float64)
    if left_xy.ndim != 2 or right_xy.ndim != 2 or left_xy.shape[0] == 0 or right_xy.shape[0] == 0:
        return float("nan")
    pair_cost = np.linalg.norm(left_xy[:, None, :] - right_xy[None, :, :], axis=2)
    nearest = np.concatenate((np.min(pair_cost, axis=1), np.min(pair_cost, axis=0)))
    finite = nearest[np.isfinite(nearest) & (nearest > 1e-6)]
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


def _circle_polyline(
    center_xy: tuple[float, float],
    radius_m: float,
    *,
    sample_count: int = 33,
) -> np.ndarray:
    cx, cy = center_xy
    theta = np.linspace(0.0, 2.0 * math.pi, max(8, int(sample_count)), dtype=np.float64)
    return np.column_stack(
        (
            float(cx) + (float(radius_m) * np.cos(theta)),
            float(cy) + (float(radius_m) * np.sin(theta)),
        )
    ).astype(np.float64)


def build_skidpad_gt_overlay_segments(
    *,
    left_circle_center_xy: tuple[float, float] = (-9.25, 0.0),
    right_circle_center_xy: tuple[float, float] = (9.25, 0.0),
    circle_radius_m: float = 9.0,
    straight_x_m: float = 0.0,
    straight_start_y_m: float = -11.0,
    straight_end_y_m: float = 21.0,
) -> list[np.ndarray]:
    return [
        _circle_polyline(left_circle_center_xy, circle_radius_m),
        _circle_polyline(right_circle_center_xy, circle_radius_m),
        np.asarray(
            [
                [float(straight_x_m), float(straight_start_y_m)],
                [float(straight_x_m), float(straight_end_y_m)],
            ],
            dtype=np.float64,
        ),
    ]


def build_skidpad_gt_color_borders() -> tuple[np.ndarray, np.ndarray]:
    blue_xy = np.asarray(
        [
            [-1.637, 0.0],
            [-2.216, -2.913],
            [-3.867, -5.382],
            [-6.336, -7.033],
            [-9.25, -7.612],
            [-12.16, -7.033],
            [-14.63, -5.382],
            [-16.28, -2.913],
            [-16.86, 0.0],
            [-16.28, 2.9131],
            [-14.63, 5.3828],
            [-12.16, 7.033],
            [-9.25, 7.6125],
            [-6.336, 7.033],
            [-3.867, 5.3828],
            [-2.216, 2.9131],
            [1.7458, 7.5041],
            [5.1887, 9.8046],
            [9.25, 10.612],
            [13.311, 9.8046],
            [16.754, 7.5041],
            [19.054, 4.0612],
            [19.862, 0.0],
            [19.054, -4.061],
            [16.754, -7.504],
            [13.311, -9.804],
            [9.25, -10.61],
            [5.1887, -9.804],
            [1.7458, -7.504],
            [-1.637, 0.0],
        ],
        dtype=np.float64,
    )
    yellow_xy = np.asarray(
        [
            [16.862, 0.0],
            [16.283, -2.913],
            [14.632, -5.382],
            [12.163, -7.033],
            [9.25, -7.612],
            [6.3368, -7.033],
            [3.8671, -5.382],
            [2.2169, -2.913],
            [1.6375, 0.0],
            [2.2169, 2.9131],
            [3.8671, 5.3828],
            [6.3368, 7.033],
            [9.25, 7.6125],
            [12.163, 7.033],
            [14.632, 5.3828],
            [16.283, 2.9131],
            [-1.745, 7.5041],
            [-5.188, 9.8046],
            [-9.25, 10.612],
            [-13.31, 9.8046],
            [-16.75, 7.5041],
            [-19.05, 4.0612],
            [-19.86, 0.0],
            [-19.05, -4.061],
            [-16.75, -7.504],
            [-13.31, -9.8],
            [-9.25, -10.61],
            [-5.188, -9.804],
            [-1.745, -7.504],
            [16.862, 0.0],
        ],
        dtype=np.float64,
    )
    return blue_xy, yellow_xy


def compute_path_tracking_overlay_average_distances(
    csv_path: Path,
    *,
    gt_midline_xy: np.ndarray,
    planner_trace_xy: np.ndarray,
    gt_reference_segments_xy: Optional[list[np.ndarray]] = None,
) -> dict[str, float]:
    rows = _read_rows(csv_path)
    gt_midline_xy = np.asarray(gt_midline_xy, dtype=np.float64)
    planner_trace_xy = np.asarray(planner_trace_xy, dtype=np.float64)
    gt_reference_segments_xy = _normalize_polyline_set(gt_reference_segments_xy)
    if not rows:
        return {
            "planner_vs_gt_avg_dist_m": float("nan"),
            "controller_vs_gt_avg_dist_m": float("nan"),
            "controller_vs_planner_avg_dist_m": float("nan"),
        }

    planner_reference_xy = _xy_series(rows, "planner_reference_x_m", "planner_reference_y_m")
    vehicle_xy = _xy_series_prefer(
        rows,
        "front_axle_x_m",
        "front_axle_y_m",
        "vehicle_x_m",
        "vehicle_y_m",
    )
    controller_vs_planner_cte = np.abs(
        _series_prefer(rows, "front_axle_vs_planner_cte_m", "controller_vs_planner_cte_m")
    )
    controller_vs_gt_cte = np.abs(
        _series_prefer(rows, "front_axle_vs_gt_cte_m", "controller_vs_gt_cte_m")
    )
    if gt_reference_segments_xy:
        planner_vs_gt_avg_dist_m = _mean_distance_to_polyline_set_m(planner_reference_xy, gt_reference_segments_xy)
        controller_vs_gt_avg_dist_m = _mean_distance_to_polyline_set_m(vehicle_xy, gt_reference_segments_xy)
    else:
        planner_vs_gt_avg_dist_m = _mean_abs_cross_track_to_polyline_m(planner_reference_xy, gt_midline_xy)
        controller_vs_gt_avg_dist_m = _mean_finite(controller_vs_gt_cte)
    return {
        "planner_vs_gt_avg_dist_m": planner_vs_gt_avg_dist_m,
        "controller_vs_gt_avg_dist_m": controller_vs_gt_avg_dist_m,
        "controller_vs_planner_avg_dist_m": _mean_finite(controller_vs_planner_cte),
    }


def _format_distance_cm(value_m: float) -> str:
    if not math.isfinite(value_m):
        return "n/a"
    return f"{value_m * 100.0:.2f} cm"


def _format_error_percent_of_half_width(value_m: float, track_width_m: float) -> str:
    if not math.isfinite(value_m) or not math.isfinite(track_width_m):
        return "n/a"
    half_width_m = 0.5 * float(track_width_m)
    if half_width_m <= 1e-9:
        return "n/a"
    return f"{(100.0 * float(value_m) / half_width_m):.2f}%"


def _format_distance_with_half_width_percent(value_m: float, track_width_m: float) -> str:
    distance_text = _format_distance_cm(value_m)
    percent_text = _format_error_percent_of_half_width(value_m, track_width_m)
    if distance_text == "n/a" or percent_text == "n/a":
        return distance_text
    return f"{distance_text} ({percent_text})"


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
    planner_vs_gt = _series_prefer(rows, "planner_reference_vs_gt_cte_m", "planner_vs_gt_cte_rms_m")
    cte_planner = _series_prefer(rows, "front_axle_vs_planner_cte_m", "controller_vs_planner_cte_m")
    cte_gt = _series_prefer(rows, "front_axle_vs_gt_cte_m", "controller_vs_gt_cte_m")

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    valid_planner_vs_gt = np.isfinite(t) & np.isfinite(planner_vs_gt)
    valid_planner = np.isfinite(t) & np.isfinite(cte_planner)
    valid_gt = np.isfinite(t) & np.isfinite(cte_gt)
    if np.any(valid_planner_vs_gt):
        ax.plot(
            t[valid_planner_vs_gt],
            planner_vs_gt[valid_planner_vs_gt],
            color="#2ca02c",
            linewidth=1.8,
            label="Planner ref vs GT midline",
        )
    if np.any(valid_planner):
        ax.plot(t[valid_planner], cte_planner[valid_planner], color="#d62728", linewidth=1.8, label="Front axle vs Planner")
    if np.any(valid_gt):
        ax.plot(t[valid_gt], cte_gt[valid_gt], color="#1f77b4", linewidth=1.8, label="Front axle vs GT midline")
    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0, alpha=0.6)
    ax.set_title(title, fontsize=DEFAULT_TITLE_FONTSIZE)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CTE (m)")
    apply_axis_label_fontsize(ax)
    apply_tick_label_fontsize(ax)
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
    gt_overlay_segments_xy: Optional[list[np.ndarray]] = None,
    lap_count: Optional[int] = None,
    lap_target: Optional[int] = None,
    title: str = "GT Midline vs Planner vs Driven Trajectory",
    dpi: int = 150,
) -> Optional[Path]:
    rows = _read_rows(csv_path)
    gt_midline_xy = np.asarray(gt_midline_xy, dtype=np.float64)
    gt_left_xy = np.asarray(gt_left_xy, dtype=np.float64)
    gt_right_xy = np.asarray(gt_right_xy, dtype=np.float64)
    planner_trace_xy = np.asarray(planner_trace_xy, dtype=np.float64)
    gt_overlay_segments_xy = _normalize_polyline_set(gt_overlay_segments_xy)
    if not rows or (gt_midline_xy.size == 0 and not gt_overlay_segments_xy):
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vehicle_x = _series_prefer(rows, "front_axle_x_m", "vehicle_x_m")
    vehicle_y = _series_prefer(rows, "front_axle_y_m", "vehicle_y_m")
    valid_vehicle = np.isfinite(vehicle_x) & np.isfinite(vehicle_y)
    avg_distances = compute_path_tracking_overlay_average_distances(
        csv_path,
        gt_midline_xy=gt_midline_xy,
        planner_trace_xy=planner_trace_xy,
        gt_reference_segments_xy=gt_overlay_segments_xy,
    )
    track_width_m = _estimate_average_track_width_m(gt_left_xy, gt_right_xy)

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    if gt_left_xy.ndim == 2 and gt_left_xy.shape[0] > 0:
        ax.plot(gt_left_xy[:, 0], gt_left_xy[:, 1], color="#1f77b4", linewidth=1.4, linestyle=":", label="GT blue border")
    if gt_right_xy.ndim == 2 and gt_right_xy.shape[0] > 0:
        ax.plot(gt_right_xy[:, 0], gt_right_xy[:, 1], color="#f1c40f", linewidth=1.4, linestyle=":", label="GT yellow border")
    if gt_overlay_segments_xy:
        for idx, segment_xy in enumerate(gt_overlay_segments_xy):
            ax.plot(
                segment_xy[:, 0],
                segment_xy[:, 1],
                color="#111111",
                linewidth=2.2,
                label="GT reference" if idx == 0 else None,
            )
    else:
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
    ax.set_title(title, fontsize=DEFAULT_TITLE_FONTSIZE)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    apply_axis_label_fontsize(ax)
    apply_tick_label_fontsize(ax)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    _legend_if_labeled(
        ax,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        framealpha=0.9,
        columnspacing=1.0,
        handlelength=1.6,
        borderaxespad=0.2,
        fontsize=OVERLAY_LEGEND_FONTSIZE,
    )
    text_lines = [
        f"Avg track width: {_format_distance_cm(track_width_m)}",
        f"Planner avg dist to GT midline: {_format_distance_with_half_width_percent(avg_distances['planner_vs_gt_avg_dist_m'], track_width_m)}",
        f"Front axle avg dist to GT midline: {_format_distance_with_half_width_percent(avg_distances['controller_vs_gt_avg_dist_m'], track_width_m)}",
        f"Front axle avg dist to planner trace: {_format_distance_with_half_width_percent(avg_distances['controller_vs_planner_avg_dist_m'], track_width_m)}",
    ]
    if lap_count is not None:
        text_lines.append(f"Completed laps: {int(lap_count)}")
    if lap_target is not None and int(lap_target) > 0:
        text_lines.append(f"Auto-stop target: {int(lap_target)}")
    avg_text = "\n".join(text_lines)
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
