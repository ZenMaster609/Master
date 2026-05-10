"""Shared CSV and path-analysis utilities."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row]


def read_float_csv_rows(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for raw_row in read_csv_rows(csv_path):
        row: dict[str, float] = {}
        for key, value in raw_row.items():
            if key is None:
                continue
            row[str(key)] = safe_float(value)
        rows.append(row)
    return rows


def series(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray(
        [safe_float(row.get(key, float("nan"))) for row in rows],
        dtype=np.float64,
    )


def relative_time(rows: list[dict[str, float]]) -> np.ndarray:
    t = series(rows, "timestamp_sec")
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return np.arange(len(rows), dtype=np.float64)
    return t - float(finite[0])


def path_cumulative_lengths(points_xy: np.ndarray) -> np.ndarray:
    points_xy = _as_xy(points_xy)
    if points_xy.shape[0] <= 1:
        return np.asarray([0.0], dtype=np.float64)
    diffs = points_xy[1:] - points_xy[:-1]
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float64)


def nearest_point_on_polyline(x: float, y: float, path_xy: np.ndarray) -> tuple[int, np.ndarray]:
    seg_idx, nearest, _progress_m = nearest_point_on_polyline_with_progress(x, y, path_xy)
    return seg_idx, nearest


def nearest_point_on_polyline_with_progress(
    x: float,
    y: float,
    path_xy: np.ndarray,
) -> tuple[int, np.ndarray, float]:
    path_xy = _as_xy(path_xy)
    if path_xy.shape[0] == 0:
        return -1, np.asarray([float("nan"), float("nan")], dtype=np.float64), float("nan")
    if path_xy.shape[0] == 1:
        return 0, np.asarray(path_xy[0], dtype=np.float64), 0.0

    p = np.asarray([float(x), float(y)], dtype=np.float64)
    cumulative = path_cumulative_lengths(path_xy)
    best_idx = -1
    best_point = np.asarray([float("nan"), float("nan")], dtype=np.float64)
    best_dist_sq = float("inf")
    best_progress_m = float("nan")

    for idx in range(path_xy.shape[0] - 1):
        a = path_xy[idx]
        b = path_xy[idx + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            t = 0.0
            cand = np.asarray(a, dtype=np.float64)
        else:
            t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
            cand = a + (t * ab)
        delta = p - cand
        dist_sq = float(np.dot(delta, delta))
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_idx = idx
            best_point = cand
            seg_len = float(np.hypot(*(b - a)))
            best_progress_m = float(cumulative[idx] + (t * seg_len))

    return best_idx, best_point, best_progress_m


def signed_cross_track_error(x: float, y: float, path_xy: np.ndarray) -> tuple[float, float]:
    path_xy = _as_xy(path_xy)
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
    dx = float(x) - float(nearest[0])
    dy = float(y) - float(nearest[1])
    cte = (dx * nx) + (dy * ny)
    tangent_yaw = math.atan2(ty, tx)
    return float(cte), float(tangent_yaw)


def _as_xy(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.reshape(arr, (-1, 2))
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[valid]
