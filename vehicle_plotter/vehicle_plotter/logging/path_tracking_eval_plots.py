"""Offline plots for GT midline path-tracking evaluation."""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .path_tracking_eval import (
    GateLapCounter,
    build_gt_midline_from_cones,
    build_smalltrack_lap_gate,
    build_stitched_reference_trace,
    path_cumulative_lengths,
    signed_cross_track_error,
)
from ..plotting.matplotlib_fonts import (
    DEFAULT_TITLE_FONTSIZE,
    LEGEND_FONTSIZE,
    apply_serif_font_preferences,
    apply_axis_label_fontsize,
    apply_tick_label_fontsize,
)

OVERLAY_LEGEND_FONTSIZE = 13.0
OVERLAY_TEXT_FONTSIZE = 15.0
CTE_LEGEND_FONTSIZE = 15.0
CTE_TEXT_FONTSIZE = 15.0
MAX_PLOT_SAMPLES = 6000
TRACK_MODEL_BY_TRACK = {
    "acceleration": "acceleration",
    "skidpad": "skidpad",
    "smalltrack": "small_track",
}


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _read_rows(csv_path: Path, columns: Optional[set[str]] = None) -> list[dict[str, float | str]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, float | str]] = []
    string_columns = {"status", "frame_id", "gt_source_frame", "odom_child_frame_id", "resolved_control_frame"}
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            row: dict[str, float | str] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                key_str = str(key)
                if columns is not None and key_str not in columns:
                    continue
                if key_str in string_columns:
                    row[str(key)] = "" if value is None else str(value)
                else:
                    row[key_str] = _safe_float(value)
            rows.append(row)
    return rows


def _limited_indices(valid: np.ndarray, max_samples: int = MAX_PLOT_SAMPLES) -> np.ndarray:
    indices = np.flatnonzero(valid)
    if indices.size <= max_samples:
        return indices
    selected = np.linspace(0, indices.size - 1, int(max_samples), dtype=np.int64)
    return indices[selected]


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
    rows = _read_rows(
        csv_path,
        {
            "planner_reference_x_m",
            "planner_reference_y_m",
            "front_axle_x_m",
            "front_axle_y_m",
            "vehicle_x_m",
            "vehicle_y_m",
            "front_axle_vs_planner_cte_m",
            "controller_vs_planner_cte_m",
            "front_axle_vs_gt_cte_m",
            "controller_vs_gt_cte_m",
        },
    )
    return _compute_path_tracking_overlay_average_distances_from_rows(
        rows,
        gt_midline_xy=gt_midline_xy,
        planner_trace_xy=planner_trace_xy,
        gt_reference_segments_xy=gt_reference_segments_xy,
    )


def _compute_path_tracking_overlay_average_distances_from_rows(
    rows: list[dict[str, float | str]],
    *,
    gt_midline_xy: np.ndarray,
    planner_trace_xy: np.ndarray,
    gt_reference_segments_xy: Optional[list[np.ndarray]] = None,
) -> dict[str, float]:
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
    rows = _read_rows(
        csv_path,
        {
            "timestamp_sec",
            "planner_reference_vs_gt_cte_m",
            "planner_vs_gt_cte_rms_m",
            "front_axle_vs_planner_cte_m",
            "controller_vs_planner_cte_m",
            "front_axle_vs_gt_cte_m",
            "controller_vs_gt_cte_m",
        },
    )
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    apply_serif_font_preferences()
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
        plot_idx = _limited_indices(valid_planner_vs_gt)
        ax.plot(
            t[plot_idx],
            planner_vs_gt[plot_idx],
            color="#2ca02c",
            linewidth=1.8,
            label="Planner ref vs GT midline",
        )
    if np.any(valid_planner):
        plot_idx = _limited_indices(valid_planner)
        ax.plot(t[plot_idx], cte_planner[plot_idx], color="#d62728", linewidth=1.8, label="Front axle vs Planner")
    if np.any(valid_gt):
        plot_idx = _limited_indices(valid_gt)
        ax.plot(t[plot_idx], cte_gt[plot_idx], color="#1f77b4", linewidth=1.8, label="Front axle vs GT midline")
    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CTE (m)")
    apply_axis_label_fontsize(ax)
    apply_tick_label_fontsize(ax)
    ax.xaxis.label.set_size(CTE_TEXT_FONTSIZE)
    ax.yaxis.label.set_size(CTE_TEXT_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=CTE_TEXT_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=CTE_TEXT_FONTSIZE)
    ax.set_ylim(-2.0, 2.0)
    ax.grid(True, alpha=0.3)
    _legend_if_labeled(ax, loc="upper right", fontsize=CTE_LEGEND_FONTSIZE)

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
    average_lap_time_sec: Optional[float] = None,
    skidpad_circle_times_sec: Optional[dict[int, float]] = None,
    title: str = "GT Midline vs Planner vs Driven Trajectory",
    dpi: int = 150,
) -> Optional[Path]:
    rows = _read_rows(
        csv_path,
        {
            "planner_reference_x_m",
            "planner_reference_y_m",
            "front_axle_x_m",
            "front_axle_y_m",
            "vehicle_x_m",
            "vehicle_y_m",
            "front_axle_vs_planner_cte_m",
            "controller_vs_planner_cte_m",
            "front_axle_vs_gt_cte_m",
            "controller_vs_gt_cte_m",
        },
    )
    gt_midline_xy = np.asarray(gt_midline_xy, dtype=np.float64)
    gt_left_xy = np.asarray(gt_left_xy, dtype=np.float64)
    gt_right_xy = np.asarray(gt_right_xy, dtype=np.float64)
    planner_trace_xy = np.asarray(planner_trace_xy, dtype=np.float64)
    gt_overlay_segments_xy = _normalize_polyline_set(gt_overlay_segments_xy)
    if not rows or (gt_midline_xy.size == 0 and not gt_overlay_segments_xy):
        return None

    import matplotlib

    matplotlib.use("Agg")
    apply_serif_font_preferences()
    import matplotlib.pyplot as plt

    vehicle_x = _series_prefer(rows, "front_axle_x_m", "vehicle_x_m")
    vehicle_y = _series_prefer(rows, "front_axle_y_m", "vehicle_y_m")
    valid_vehicle = np.isfinite(vehicle_x) & np.isfinite(vehicle_y)
    avg_distances = _compute_path_tracking_overlay_average_distances_from_rows(
        rows,
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
        valid_indices = np.flatnonzero(valid_vehicle)
        plot_idx = _limited_indices(valid_vehicle)
        ax.plot(vehicle_x[plot_idx], vehicle_y[plot_idx], color="#1f77b4", linewidth=1.8, label="Driven trajectory")
        ax.scatter(vehicle_x[valid_indices[0]], vehicle_y[valid_indices[0]], color="green", s=45, zorder=5, label="Start")
        ax.scatter(vehicle_x[valid_indices[-1]], vehicle_y[valid_indices[-1]], color="red", s=45, zorder=5, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    apply_axis_label_fontsize(ax)
    apply_tick_label_fontsize(ax)
    ax.xaxis.label.set_size(OVERLAY_TEXT_FONTSIZE)
    ax.yaxis.label.set_size(OVERLAY_TEXT_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=OVERLAY_TEXT_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=OVERLAY_TEXT_FONTSIZE)
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
    if average_lap_time_sec is not None and np.isfinite(float(average_lap_time_sec)):
        text_lines.append(f"Avg lap time: {_format_duration_minutes_seconds(float(average_lap_time_sec))}")
    if skidpad_circle_times_sec:
        t2 = skidpad_circle_times_sec.get(2)
        t4 = skidpad_circle_times_sec.get(4)
        if t2 is not None and math.isfinite(t2):
            text_lines.append(f"Right circle time: {_format_duration_minutes_seconds(t2)}")
        if t4 is not None and math.isfinite(t4):
            text_lines.append(f"Left circle time: {_format_duration_minutes_seconds(t4)}")
    avg_text = "\n".join(text_lines)
    ax.text(
        0.02,
        0.98,
        avg_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=OVERLAY_TEXT_FONTSIZE,
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


def _format_duration_minutes_seconds(duration_sec: float) -> str:
    if not np.isfinite(duration_sec):
        return "n/a"
    total_seconds = max(0, int(round(float(duration_sec))))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def compute_skidpad_circle_times_sec(csv_path: Path) -> dict[int, float]:
    """Return elapsed time (seconds) per circle for the skidpad sequence right-right-left-left.

    Keys are 1-indexed circle numbers (1=first right, 2=second right,
    3=first left, 4=second left).  Only circles for which a full lap was
    detected will appear in the result.
    """
    rows = _read_rows(csv_path)
    if not rows:
        return {}

    _CROSSROADS_CENTER = (0.0, 0.0)
    _CROSSROADS_HALF_XY = (3.0, 2.5)
    _RIGHT_CENTER = (9.25, 0.0)
    _LEFT_CENTER = (-9.25, 0.0)
    _RADIUS_MIN, _RADIUS_MAX = 6.5, 11.5
    _LAP_COMPLETE_ANGLE = 5.5
    _SEQUENCE = ("right", "right", "left", "left")

    def _in_crossroads(x: float, y: float) -> bool:
        cx, cy = _CROSSROADS_CENTER
        hx, hy = _CROSSROADS_HALF_XY
        return abs(x - cx) <= hx and abs(y - cy) <= hy

    approach_complete = False
    route_index = 0
    completed_laps = 0
    lap_angle_accum = 0.0
    lap_armed = False
    last_theta: Optional[float] = None
    prev_in_crossroads: Optional[bool] = None

    circle_times: dict[int, float] = {}
    circle_start_sec: Optional[float] = None

    for row in rows:
        t = _safe_float(row.get("timestamp_sec", float("nan")))
        x = _safe_float(row.get("front_axle_x_m", row.get("vehicle_x_m", float("nan"))))
        y = _safe_float(row.get("front_axle_y_m", row.get("vehicle_y_m", float("nan"))))
        if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y)):
            continue

        xroads = _in_crossroads(x, y)
        just_entered = prev_in_crossroads is not None and not prev_in_crossroads and xroads

        branch = _SEQUENCE[route_index] if route_index < len(_SEQUENCE) else "straight"

        if branch in ("right", "left") and approach_complete:
            cx, cy = _RIGHT_CENTER if branch == "right" else _LEFT_CENTER
            r = math.hypot(x - cx, y - cy)
            if not xroads and _RADIUS_MIN <= r <= _RADIUS_MAX:
                theta = math.atan2(y - cy, x - cx)
                if last_theta is not None:
                    delta = (theta - last_theta + math.pi) % (2 * math.pi) - math.pi
                    lap_angle_accum += delta
                    if abs(lap_angle_accum) >= _LAP_COMPLETE_ANGLE:
                        lap_armed = True
                last_theta = theta
            else:
                last_theta = None
        else:
            last_theta = None
            if branch not in ("right", "left"):
                lap_angle_accum = 0.0
                lap_armed = False

        if just_entered:
            if not approach_complete:
                approach_complete = True
                circle_start_sec = t
            elif branch in ("right", "left") and lap_armed:
                completed_laps += 1
                if circle_start_sec is not None:
                    circle_times[completed_laps] = t - circle_start_sec
                circle_start_sec = t
                if route_index < len(_SEQUENCE) - 1:
                    route_index += 1
                lap_angle_accum = 0.0
                lap_armed = False
                last_theta = None

        prev_in_crossroads = xroads

    return circle_times


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _clean_string(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _parse_pose_xy_yaw(pose_text: object) -> tuple[float, float, float]:
    parts = str(pose_text or "").split()
    if len(parts) < 2:
        return 0.0, 0.0, 0.0
    x_m = _safe_float(parts[0])
    y_m = _safe_float(parts[1])
    yaw_rad = _safe_float(parts[5]) if len(parts) >= 6 else 0.0
    return (
        x_m if math.isfinite(x_m) else 0.0,
        y_m if math.isfinite(y_m) else 0.0,
        yaw_rad if math.isfinite(yaw_rad) else 0.0,
    )


def _apply_planar_transform(points_xy: np.ndarray, *, tx_m: float, ty_m: float, yaw_rad: float) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    rot = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    transformed = points_xy @ rot.T
    transformed[:, 0] += float(tx_m)
    transformed[:, 1] += float(ty_m)
    return transformed.astype(np.float64)


def _resolve_track_model_from_world(world_path: Path) -> tuple[str, tuple[float, float, float]] | None:
    try:
        root = ET.parse(world_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return None

    for include in root.findall(".//include"):
        uri = _clean_string(include.findtext("uri"))
        if not uri.startswith("model://"):
            continue
        model_name = uri.removeprefix("model://")
        if model_name in {"asphalt_plane", "ground_plane"} or "backdrop" in model_name:
            continue
        return model_name, _parse_pose_xy_yaw(include.findtext("pose"))
    return None


def _resolve_track_model_path(
    session_path: Path,
    launch_params: dict[str, object],
) -> tuple[Path, tuple[float, float, float]]:
    world_text = _clean_string(launch_params.get("world"))
    if world_text:
        world_path = Path(world_text)
        resolved = _resolve_track_model_from_world(world_path)
        if resolved is not None:
            model_name, model_pose = resolved
            world_share_dir = world_path.parent.parent
            candidates = [
                world_share_dir / "models" / model_name / "model.sdf",
                Path(__file__).resolve().parents[3] / "sim_car" / "models" / model_name / "model.sdf",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate, model_pose

    track_name = _clean_string(launch_params.get("track")).lower()
    model_name = TRACK_MODEL_BY_TRACK.get(track_name, track_name)
    candidate = Path(__file__).resolve().parents[3] / "sim_car" / "models" / model_name / "model.sdf"
    if candidate.exists():
        return candidate, (0.0, 0.0, 0.0)
    raise FileNotFoundError(f"Could not resolve a track model for session {session_path}")


def _classify_cone_uri(uri: str) -> str:
    model_name = _clean_string(uri).removeprefix("model://").lower()
    if model_name == "blue_cone":
        return "blue"
    if model_name == "yellow_cone":
        return "yellow"
    if model_name == "big_cone":
        return "big_orange"
    return ""


def _load_track_model_cones(
    model_path: Path,
    *,
    model_pose_xy_yaw: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    root = ET.parse(model_path).getroot()
    tx_m, ty_m, yaw_rad = model_pose_xy_yaw
    cones_by_color: dict[str, list[list[float]]] = {
        "blue": [],
        "yellow": [],
        "big_orange": [],
    }

    for include in root.findall(".//include"):
        color = _classify_cone_uri(_clean_string(include.findtext("uri")))
        if not color:
            continue
        x_m, y_m, _include_yaw = _parse_pose_xy_yaw(include.findtext("pose"))
        transformed = _apply_planar_transform(
            np.asarray([[x_m, y_m]], dtype=np.float64),
            tx_m=tx_m,
            ty_m=ty_m,
            yaw_rad=yaw_rad,
        )
        cones_by_color[color].append(transformed[0].tolist())

    return {
        color: np.asarray(points, dtype=np.float64).reshape((-1, 2)) if points else np.empty((0, 2), dtype=np.float64)
        for color, points in cones_by_color.items()
    }


def _infer_start_heading_xy_from_rows(rows: list[dict[str, float | str]]) -> tuple[np.ndarray, np.ndarray]:
    vehicle_xy = _xy_series_prefer(
        rows,
        "front_axle_x_m",
        "front_axle_y_m",
        "vehicle_x_m",
        "vehicle_y_m",
    )
    if vehicle_xy.shape[0] == 0:
        return np.asarray([0.0, 0.0], dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)

    start_xy = np.asarray(vehicle_xy[0], dtype=np.float64)
    heading_xy = np.asarray([1.0, 0.0], dtype=np.float64)
    for point_xy in vehicle_xy[1:]:
        delta = np.asarray(point_xy, dtype=np.float64) - start_xy
        norm = float(np.hypot(delta[0], delta[1]))
        if norm > 1e-6:
            heading_xy = (delta / norm).astype(np.float64)
            break
    return start_xy, heading_xy


def _compute_smalltrack_lap_metrics_from_rows(
    rows: list[dict[str, float | str]],
    *,
    gt_midline_xy: np.ndarray,
    big_orange_xy: np.ndarray,
) -> tuple[int | None, float | None]:
    gate = build_smalltrack_lap_gate(big_orange_xy=np.asarray(big_orange_xy, dtype=np.float64), frame_id="map")
    gt_midline_xy = np.asarray(gt_midline_xy, dtype=np.float64)
    if gate is None or gt_midline_xy.ndim != 2 or gt_midline_xy.shape[0] < 2:
        return None, None

    track_length_m = float(path_cumulative_lengths(gt_midline_xy)[-1])
    min_lap_travel_m = max(15.0, 0.6 * track_length_m) if math.isfinite(track_length_m) else 25.0
    gate_length_m = float(np.hypot(*(gate.segment_xy[1] - gate.segment_xy[0])))
    counter = GateLapCounter(
        gate.segment_xy,
        min_lap_travel_m=min_lap_travel_m,
        min_lap_time_sec=5.0,
        near_gate_distance_m=max(4.0, 2.0 * gate_length_m),
    )

    timestamps_sec = _series(rows, "timestamp_sec")
    vehicle_x = _series_prefer(rows, "front_axle_x_m", "vehicle_x_m")
    vehicle_y = _series_prefer(rows, "front_axle_y_m", "vehicle_y_m")
    lap_times_sec: list[float] = []
    completed_laps = 0

    for idx, timestamp_sec in enumerate(timestamps_sec):
        if not (math.isfinite(timestamp_sec) and math.isfinite(vehicle_x[idx]) and math.isfinite(vehicle_y[idx])):
            continue
        snapshot = counter.update(
            np.asarray([vehicle_x[idx], vehicle_y[idx]], dtype=np.float64),
            float(timestamp_sec),
        )
        completed_laps = int(snapshot.completed_laps)
        if snapshot.just_completed_lap and snapshot.last_lap_time_sec is not None:
            lap_times_sec.append(float(snapshot.last_lap_time_sec))

    average_lap_time_sec = float(np.mean(lap_times_sec)) if lap_times_sec else None
    return completed_laps, average_lap_time_sec


def _load_lap_target(session_path: Path, track_name: str) -> int | None:
    spawn_path = session_path / "configs" / "sim_car_config" / str(track_name) / "spawn.yaml"
    config = _load_yaml_mapping(spawn_path)
    lap_tracking = config.get("lap_tracking")
    if not isinstance(lap_tracking, dict):
        return None
    raw_value = lap_tracking.get("auto_suspend_after_laps")
    try:
        return int(raw_value) if raw_value is not None else None
    except (TypeError, ValueError):
        return None


def _compute_session_duration_sec(rows: list[dict[str, float | str]]) -> float | None:
    timestamps_sec = _series(rows, "timestamp_sec")
    finite = timestamps_sec[np.isfinite(timestamps_sec)]
    if finite.size < 2:
        return None
    duration_sec = float(finite[-1] - finite[0])
    return duration_sec if duration_sec > 0.0 else None


def regenerate_path_tracking_overlay_for_session(
    session_path: Path,
    *,
    overlay_format: str = "pdf",
    dpi: int = 150,
    delete_legacy_png: bool = False,
) -> Optional[Path]:
    session_path = Path(session_path)
    csv_path = session_path / "logs" / "path_tracking_eval.csv"
    rows = _read_rows(
        csv_path,
        {
            "timestamp_sec",
            "planner_reference_x_m",
            "planner_reference_y_m",
            "front_axle_x_m",
            "front_axle_y_m",
            "vehicle_x_m",
            "vehicle_y_m",
            "front_axle_vs_planner_cte_m",
            "controller_vs_planner_cte_m",
            "front_axle_vs_gt_cte_m",
            "controller_vs_gt_cte_m",
        },
    )
    if not rows:
        return None

    launch_params = _load_yaml_mapping(session_path / "configs" / "launch_parameters.yaml")
    track_name = _clean_string(launch_params.get("track")).lower() or "unknown"
    planner_trace_xy = build_stitched_reference_trace(
        _xy_series(rows, "planner_reference_x_m", "planner_reference_y_m"),
        min_spacing_m=0.1,
    )

    overlay_segments_xy: Optional[list[np.ndarray]] = None
    lap_count: int | None = None
    average_lap_time_sec: float | None = None
    skidpad_circle_times_sec: Optional[dict[int, float]] = None

    if track_name == "skidpad":
        gt_midline_xy = np.empty((0, 2), dtype=np.float64)
        gt_left_xy, gt_right_xy = build_skidpad_gt_color_borders()
        overlay_segments_xy = build_skidpad_gt_overlay_segments()
        skidpad_circle_times_sec = compute_skidpad_circle_times_sec(csv_path)
        average_lap_time_sec = _compute_session_duration_sec(rows)
    else:
        model_path, model_pose_xy_yaw = _resolve_track_model_path(session_path, launch_params)
        cone_sets = _load_track_model_cones(model_path, model_pose_xy_yaw=model_pose_xy_yaw)
        start_xy, heading_xy = _infer_start_heading_xy_from_rows(rows)
        gt_midline = build_gt_midline_from_cones(
            blue_xy=cone_sets["blue"],
            yellow_xy=cone_sets["yellow"],
            start_xy=start_xy,
            heading_xy=heading_xy,
            frame_id="map",
            resolution_m=0.5,
        )
        gt_midline_xy = gt_midline.midline_xy
        gt_left_xy = gt_midline.left_xy
        gt_right_xy = gt_midline.right_xy
        if track_name == "smalltrack":
            lap_count, average_lap_time_sec = _compute_smalltrack_lap_metrics_from_rows(
                rows,
                gt_midline_xy=gt_midline_xy,
                big_orange_xy=cone_sets["big_orange"],
            )
        else:
            average_lap_time_sec = _compute_session_duration_sec(rows)

    overlay_path = session_path / "plots" / f"path_tracking_eval_overlay.{overlay_format}"
    generated_path = generate_path_tracking_overlay_plot(
        csv_path,
        overlay_path,
        gt_midline_xy=gt_midline_xy,
        gt_left_xy=gt_left_xy,
        gt_right_xy=gt_right_xy,
        planner_trace_xy=planner_trace_xy,
        gt_overlay_segments_xy=overlay_segments_xy,
        lap_count=lap_count,
        lap_target=_load_lap_target(session_path, track_name),
        average_lap_time_sec=average_lap_time_sec,
        skidpad_circle_times_sec=skidpad_circle_times_sec,
        dpi=dpi,
    )
    if generated_path is not None and delete_legacy_png:
        legacy_png = session_path / "plots" / "path_tracking_eval_overlay.png"
        if legacy_png.exists() and legacy_png != generated_path:
            legacy_png.unlink()
    return generated_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate path tracking overlay plots for an existing session")
    parser.add_argument("session_path", help="Session directory under multidata/")
    parser.add_argument(
        "--overlay-format",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output format for the regenerated overlay plot",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for raster output formats")
    parser.add_argument(
        "--delete-legacy-png",
        action="store_true",
        help="Delete an existing legacy PNG overlay after successfully generating the new artifact",
    )
    args = parser.parse_args()

    generated_path = regenerate_path_tracking_overlay_for_session(
        Path(args.session_path),
        overlay_format=args.overlay_format,
        dpi=args.dpi,
        delete_legacy_png=args.delete_legacy_png,
    )
    if generated_path is None:
        raise SystemExit("No path tracking overlay could be generated for the supplied session")
    print(generated_path)


if __name__ == "__main__":
    main()
