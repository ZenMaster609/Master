"""Helpers for GT midline path-tracking evaluation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PATH_TRACKING_EVAL_FIELDNAMES = [
    "timestamp_sec",
    "sample_valid_flag",
    "status",
    "frame_id",
    "gt_source_frame",
    "vehicle_x_m",
    "vehicle_y_m",
    "planner_reference_x_m",
    "planner_reference_y_m",
    "gt_reference_x_m",
    "gt_reference_y_m",
    "controller_vs_planner_cte_m",
    "controller_vs_gt_cte_m",
    "planner_vs_gt_cte_rms_m",
    "planner_vs_gt_cte_p95_m",
    "planner_vs_gt_cte_max_m",
]

IDENTITY_WORLD_FRAMES = frozenset({"map", "odom"})


@dataclass(frozen=True)
class GTMidline:
    frame_id: str
    midline_xy: np.ndarray
    left_xy: np.ndarray
    right_xy: np.ndarray


def _as_xy(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.reshape(arr, (-1, 2))
    return arr[np.all(np.isfinite(arr), axis=1)]


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] * b[1]) - (a[1] * b[0]))


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.hypot(vec[0], vec[1]))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return (vec / norm).astype(np.float64)


def should_assume_identity_transform(source_frame: str, target_frame: str) -> bool:
    source = str(source_frame).strip().lower()
    target = str(target_frame).strip().lower()
    if not source or not target or source == target:
        return False
    return source in IDENTITY_WORLD_FRAMES and target in IDENTITY_WORLD_FRAMES


def _dedupe_points(points_xy: np.ndarray, *, tolerance_m: float = 1e-4) -> np.ndarray:
    if points_xy.shape[0] <= 1:
        return points_xy.astype(np.float64)
    keep = [points_xy[0]]
    for idx in range(1, points_xy.shape[0]):
        if float(np.hypot(*(points_xy[idx] - keep[-1]))) > tolerance_m:
            keep.append(points_xy[idx])
    return np.asarray(keep, dtype=np.float64)


def _median_segment_length(points_xy: np.ndarray) -> float:
    points_xy = _as_xy(points_xy)
    if points_xy.shape[0] < 2:
        return float("nan")
    diffs = np.diff(points_xy, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    finite = seg_len[np.isfinite(seg_len) & (seg_len > 1e-6)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _close_loop_if_needed(points_xy: np.ndarray) -> np.ndarray:
    points_xy = _as_xy(points_xy)
    if points_xy.shape[0] < 3:
        return points_xy
    if float(np.hypot(*(points_xy[0] - points_xy[-1]))) <= 1e-6:
        return points_xy
    median_spacing = _median_segment_length(points_xy)
    if not math.isfinite(median_spacing):
        return points_xy
    closure_gap = float(np.hypot(*(points_xy[0] - points_xy[-1])))
    if closure_gap <= max(1.5, 2.5 * median_spacing):
        return np.vstack((points_xy, points_xy[:1]))
    return points_xy


def order_boundary_points(
    points_xy: np.ndarray,
    *,
    start_xy: np.ndarray,
    heading_xy: np.ndarray,
) -> np.ndarray:
    points_xy = _dedupe_points(_as_xy(points_xy))
    if points_xy.shape[0] <= 1:
        return points_xy

    start_xy = np.asarray(start_xy, dtype=np.float64)
    heading_xy = _normalize(np.asarray(heading_xy, dtype=np.float64))

    start_dists = np.hypot(points_xy[:, 0] - start_xy[0], points_xy[:, 1] - start_xy[1])
    start_idx = int(np.argmin(start_dists))

    ordered = [points_xy[start_idx]]
    used = np.zeros((points_xy.shape[0],), dtype=bool)
    used[start_idx] = True
    travel_dir = heading_xy

    while np.count_nonzero(~used) > 0:
        current = ordered[-1]
        remaining_idx = np.where(~used)[0]
        remaining = points_xy[remaining_idx]
        deltas = remaining - current
        dists = np.hypot(deltas[:, 0], deltas[:, 1])
        valid = dists > 1e-9
        if not np.any(valid):
            break
        unit = np.zeros_like(deltas)
        unit[valid] = deltas[valid] / dists[valid, None]
        forward_cos = unit @ travel_dir
        heading_proj = deltas @ heading_xy
        lateral_penalty = np.abs(unit[:, 0] * (-travel_dir[1]) + unit[:, 1] * travel_dir[0])
        score = dists * (
            1.0
            + (1.5 * np.maximum(0.0, -forward_cos))
            + (0.15 * lateral_penalty)
            + (0.05 * np.maximum(0.0, -heading_proj))
        )
        score[~valid] = float("inf")
        best_local = int(np.argmin(score))
        next_idx = int(remaining_idx[best_local])
        next_point = points_xy[next_idx]
        used[next_idx] = True
        ordered.append(next_point)

        step_vec = next_point - current
        if float(np.hypot(step_vec[0], step_vec[1])) > 1e-9:
            step_dir = _normalize(step_vec)
            if float(np.dot(step_dir, travel_dir)) < -0.8:
                step_dir = -step_dir
            travel_dir = _normalize((0.65 * travel_dir) + (0.35 * step_dir))

    return _dedupe_points(np.asarray(ordered, dtype=np.float64))


def path_cumulative_lengths(points_xy: np.ndarray) -> np.ndarray:
    points_xy = _as_xy(points_xy)
    if points_xy.shape[0] <= 1:
        return np.asarray([0.0], dtype=np.float64)
    diffs = points_xy[1:] - points_xy[:-1]
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float64)


def sample_path_at_lengths(points_xy: np.ndarray, cum_lengths: np.ndarray, samples: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    capped = np.clip(np.asarray(samples, dtype=np.float64), 0.0, float(cum_lengths[-1]))
    x = np.interp(capped, cum_lengths, points_xy[:, 0])
    y = np.interp(capped, cum_lengths, points_xy[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


def build_midline_from_boundaries(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    *,
    resolution_m: float = 0.5,
) -> np.ndarray:
    left_xy = _dedupe_points(_as_xy(left_xy))
    right_xy = _dedupe_points(_as_xy(right_xy))
    if left_xy.shape[0] < 2 or right_xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    left_closed = left_xy.shape[0] >= 3 and float(np.hypot(*(left_xy[0] - left_xy[-1]))) <= 1e-6
    right_closed = right_xy.shape[0] >= 3 and float(np.hypot(*(right_xy[0] - right_xy[-1]))) <= 1e-6
    if left_closed:
        left_xy = np.asarray(left_xy[:-1], dtype=np.float64)
    if right_closed:
        right_xy = np.asarray(right_xy[:-1], dtype=np.float64)
    if left_xy.shape[0] < 2 or right_xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    midpoint_pairs = _build_midpoint_chain_from_matched_boundaries(left_xy, right_xy)
    midpoint_pairs = _dedupe_points(midpoint_pairs)
    if left_closed and right_closed:
        midpoint_pairs = _close_loop_if_needed(midpoint_pairs)

    if midpoint_pairs.shape[0] < 2:
        return midpoint_pairs

    cum_lengths = path_cumulative_lengths(midpoint_pairs)
    total_m = float(cum_lengths[-1])
    if total_m <= 1e-6:
        return midpoint_pairs

    step_m = max(0.05, float(resolution_m))
    samples = np.arange(0.0, total_m + 1e-9, step_m, dtype=np.float64)
    if samples.size == 0 or samples[-1] < total_m:
        samples = np.concatenate((samples, [total_m]))
    return sample_path_at_lengths(midpoint_pairs, cum_lengths, samples)


def _estimate_track_width(left_xy: np.ndarray, right_xy: np.ndarray) -> float:
    left_xy = _as_xy(left_xy)
    right_xy = _as_xy(right_xy)
    if left_xy.shape[0] == 0 or right_xy.shape[0] == 0:
        return float("nan")
    pair_cost = np.linalg.norm(left_xy[:, None, :] - right_xy[None, :, :], axis=2)
    nearest = np.concatenate((np.min(pair_cost, axis=1), np.min(pair_cost, axis=0)))
    finite = nearest[np.isfinite(nearest) & (nearest > 1e-6)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _build_midline_from_cone_pairs(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    *,
    start_xy: np.ndarray,
    heading_xy: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    left_xy = _dedupe_points(_as_xy(left_xy))
    right_xy = _dedupe_points(_as_xy(right_xy))
    if left_xy.shape[0] < 2 or right_xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    pair_cost = np.linalg.norm(left_xy[:, None, :] - right_xy[None, :, :], axis=2)
    width_m = _estimate_track_width(left_xy, right_xy)
    if not math.isfinite(width_m):
        return np.empty((0, 2), dtype=np.float64)

    # Accept plausible cross-track pairs, then build the centerline from matched midpoints.
    max_pair_distance_m = max(2.0 * width_m, width_m + 2.0)
    candidates: list[tuple[float, int, int]] = []
    for left_idx in range(left_xy.shape[0]):
        for right_idx in range(right_xy.shape[0]):
            dist_m = float(pair_cost[left_idx, right_idx])
            if dist_m <= max_pair_distance_m:
                candidates.append((dist_m, left_idx, right_idx))
    candidates.sort(key=lambda item: item[0])

    used_left: set[int] = set()
    used_right: set[int] = set()
    midpoint_pairs: list[np.ndarray] = []
    for _, left_idx, right_idx in candidates:
        if left_idx in used_left or right_idx in used_right:
            continue
        used_left.add(left_idx)
        used_right.add(right_idx)
        midpoint_pairs.append(0.5 * (left_xy[left_idx] + right_xy[right_idx]))

    midpoint_xy = _as_xy(midpoint_pairs)
    if midpoint_xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    midpoint_xy = order_boundary_points(midpoint_xy, start_xy=start_xy, heading_xy=heading_xy)
    midpoint_xy = _close_loop_if_needed(midpoint_xy)
    if midpoint_xy.shape[0] < 2:
        return midpoint_xy

    cum_lengths = path_cumulative_lengths(midpoint_xy)
    total_m = float(cum_lengths[-1])
    if total_m <= 1e-6:
        return midpoint_xy

    step_m = max(0.05, float(resolution_m))
    samples = np.arange(0.0, total_m + 1e-9, step_m, dtype=np.float64)
    if samples.size == 0 or samples[-1] < total_m:
        samples = np.concatenate((samples, [total_m]))
    return sample_path_at_lengths(midpoint_xy, cum_lengths, samples)


def _build_midpoint_chain_from_matched_boundaries(left_xy: np.ndarray, right_xy: np.ndarray) -> np.ndarray:
    n_left = int(left_xy.shape[0])
    n_right = int(right_xy.shape[0])
    if n_left == 0 or n_right == 0:
        return np.empty((0, 2), dtype=np.float64)

    pair_cost = np.linalg.norm(left_xy[:, None, :] - right_xy[None, :, :], axis=2)
    dp = np.full((n_left, n_right), float("inf"), dtype=np.float64)
    parent = np.full((n_left, n_right, 2), -1, dtype=np.int64)

    for i in range(n_left):
        for j in range(n_right):
            base = float(pair_cost[i, j])
            stall_penalty = 0.25 * base
            if i == 0 and j == 0:
                dp[i, j] = base
                continue

            candidates: list[tuple[float, tuple[int, int]]] = []
            if i > 0 and j > 0:
                candidates.append((dp[i - 1, j - 1], (i - 1, j - 1)))
            if i > 0:
                candidates.append((dp[i - 1, j] + stall_penalty, (i - 1, j)))
            if j > 0:
                candidates.append((dp[i, j - 1] + stall_penalty, (i, j - 1)))

            best_cost, best_parent = min(candidates, key=lambda item: item[0])
            dp[i, j] = base + float(best_cost)
            parent[i, j] = np.asarray(best_parent, dtype=np.int64)

    pairs: list[tuple[int, int]] = []
    i = n_left - 1
    j = n_right - 1
    while i >= 0 and j >= 0:
        pairs.append((i, j))
        prev_i = int(parent[i, j, 0])
        prev_j = int(parent[i, j, 1])
        if prev_i < 0 or prev_j < 0:
            break
        i, j = prev_i, prev_j
    pairs.reverse()

    if not pairs:
        return np.empty((0, 2), dtype=np.float64)

    midpoint_chain = np.empty((len(pairs), 2), dtype=np.float64)
    for idx, (left_idx, right_idx) in enumerate(pairs):
        midpoint_chain[idx] = 0.5 * (left_xy[left_idx] + right_xy[right_idx])
    return midpoint_chain


def build_gt_midline_from_cones(
    *,
    blue_xy: np.ndarray,
    yellow_xy: np.ndarray,
    start_xy: np.ndarray,
    heading_xy: np.ndarray,
    frame_id: str,
    resolution_m: float = 0.5,
) -> GTMidline:
    left_xy = _close_loop_if_needed(
        order_boundary_points(blue_xy, start_xy=start_xy, heading_xy=heading_xy)
    )
    right_xy = _close_loop_if_needed(
        order_boundary_points(yellow_xy, start_xy=start_xy, heading_xy=heading_xy)
    )
    midline_xy = _build_midline_from_cone_pairs(
        _as_xy(blue_xy),
        _as_xy(yellow_xy),
        start_xy=np.asarray(start_xy, dtype=np.float64),
        heading_xy=np.asarray(heading_xy, dtype=np.float64),
        resolution_m=resolution_m,
    )
    if midline_xy.shape[0] < 2:
        midline_xy = build_midline_from_boundaries(left_xy, right_xy, resolution_m=resolution_m)
    return GTMidline(
        frame_id=str(frame_id).strip(),
        midline_xy=midline_xy,
        left_xy=left_xy,
        right_xy=right_xy,
    )


def nearest_point_on_polyline(x: float, y: float, path_xy: np.ndarray) -> tuple[int, np.ndarray]:
    path_xy = _as_xy(path_xy)
    if path_xy.shape[0] == 0:
        return -1, np.asarray([float("nan"), float("nan")], dtype=np.float64)
    if path_xy.shape[0] == 1:
        return 0, np.asarray(path_xy[0], dtype=np.float64)

    p = np.asarray([float(x), float(y)], dtype=np.float64)
    best_idx = -1
    best_point = np.asarray([float("nan"), float("nan")], dtype=np.float64)
    best_dist_sq = float("inf")

    for idx in range(path_xy.shape[0] - 1):
        a = path_xy[idx]
        b = path_xy[idx + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
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

    return best_idx, best_point


def normalize_angle(angle_rad: float) -> float:
    out = float(angle_rad)
    while out > math.pi:
        out -= 2.0 * math.pi
    while out < -math.pi:
        out += 2.0 * math.pi
    return out


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


def project_point_to_path_s(path_xy: np.ndarray, cum_lengths: np.ndarray, point_xy: np.ndarray) -> float:
    path_xy = _as_xy(path_xy)
    point_xy = np.asarray(point_xy, dtype=np.float64)
    best_s = 0.0
    best_dist = float("inf")
    for idx in range(path_xy.shape[0] - 1):
        a = path_xy[idx]
        b = path_xy[idx + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            t = 0.0
            projected = a
        else:
            t = float(np.clip(np.dot(point_xy - a, ab) / denom, 0.0, 1.0))
            projected = a + (t * ab)
        dist = float(np.hypot(*(point_xy - projected)))
        if dist < best_dist:
            best_dist = dist
            best_s = float(cum_lengths[idx] + (t * np.hypot(ab[0], ab[1])))
    return best_s


def extract_forward_path_from_pose(path_xy: np.ndarray, vehicle_xy: np.ndarray, *, resolution_m: float) -> np.ndarray:
    path_xy = _as_xy(path_xy)
    if path_xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)
    cum_lengths = path_cumulative_lengths(path_xy)
    s_vehicle = project_point_to_path_s(path_xy, cum_lengths, np.asarray(vehicle_xy, dtype=np.float64))
    total = float(cum_lengths[-1])
    step = max(0.05, float(resolution_m))
    if (total - s_vehicle) <= step:
        return np.empty((0, 2), dtype=np.float64)
    samples = np.arange(s_vehicle, total + 1e-9, step, dtype=np.float64)
    if samples.size == 0 or samples[-1] < total:
        samples = np.concatenate((samples, [total]))
    return sample_path_at_lengths(path_xy, cum_lengths, samples)


def compare_planner_path_to_gt(
    *,
    planner_xy: np.ndarray,
    gt_midline_xy: np.ndarray,
    vehicle_xy: np.ndarray,
    resolution_m: float = 0.5,
) -> dict[str, float]:
    planner_forward = extract_forward_path_from_pose(
        planner_xy,
        vehicle_xy,
        resolution_m=resolution_m,
    )
    gt_forward = extract_forward_path_from_pose(
        gt_midline_xy,
        vehicle_xy,
        resolution_m=resolution_m,
    )
    if planner_forward.shape[0] < 2 or gt_forward.shape[0] < 2:
        return {
            "planner_vs_gt_cte_rms_m": float("nan"),
            "planner_vs_gt_cte_p95_m": float("nan"),
            "planner_vs_gt_cte_max_m": float("nan"),
            "planner_vs_gt_sample_count": 0.0,
        }

    planner_cum = path_cumulative_lengths(planner_forward)
    overlap_m = min(float(planner_cum[-1]), float(path_cumulative_lengths(gt_forward)[-1]))
    if overlap_m <= 1e-6:
        return {
            "planner_vs_gt_cte_rms_m": float("nan"),
            "planner_vs_gt_cte_p95_m": float("nan"),
            "planner_vs_gt_cte_max_m": float("nan"),
            "planner_vs_gt_sample_count": 0.0,
        }

    step_m = max(0.05, float(resolution_m))
    samples = np.arange(0.0, overlap_m + 1e-9, step_m, dtype=np.float64)
    if samples.size == 0 or samples[-1] < overlap_m:
        samples = np.concatenate((samples, [overlap_m]))
    planner_samples = sample_path_at_lengths(planner_forward, planner_cum, samples)

    errors = []
    for point_xy in planner_samples:
        cte, _ = signed_cross_track_error(float(point_xy[0]), float(point_xy[1]), gt_forward)
        if math.isfinite(cte):
            errors.append(abs(float(cte)))
    if not errors:
        return {
            "planner_vs_gt_cte_rms_m": float("nan"),
            "planner_vs_gt_cte_p95_m": float("nan"),
            "planner_vs_gt_cte_max_m": float("nan"),
            "planner_vs_gt_sample_count": 0.0,
        }

    errors_arr = np.asarray(errors, dtype=np.float64)
    return {
        "planner_vs_gt_cte_rms_m": _nanrms(errors_arr),
        "planner_vs_gt_cte_p95_m": _nanpercentile(errors_arr, 95.0),
        "planner_vs_gt_cte_max_m": float(np.nanmax(errors_arr)),
        "planner_vs_gt_sample_count": float(errors_arr.size),
    }


def build_stitched_reference_trace(reference_points_xy: np.ndarray, *, min_spacing_m: float = 0.1) -> np.ndarray:
    points_xy = _as_xy(reference_points_xy)
    if points_xy.shape[0] == 0:
        return points_xy

    stitched = [points_xy[0]]
    for idx in range(1, points_xy.shape[0]):
        step = float(np.hypot(*(points_xy[idx] - stitched[-1])))
        if step >= max(1e-4, float(min_spacing_m)):
            stitched.append(points_xy[idx])
    return np.asarray(stitched, dtype=np.float64)


def transform_xy_points(points_xy: np.ndarray, *, translation_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    points_xy = _as_xy(points_xy)
    if points_xy.shape[0] == 0:
        return points_xy
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return ((points_xy @ rot.T) + np.asarray(translation_xy, dtype=np.float64)).astype(np.float64)


def transform_xy_point(point_xy: np.ndarray, *, translation_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    out = transform_xy_points(np.asarray([point_xy], dtype=np.float64), translation_xy=translation_xy, yaw_rad=yaw_rad)
    if out.shape[0] == 0:
        return np.asarray([float("nan"), float("nan")], dtype=np.float64)
    return out[0]


def analyze_path_tracking_csv(csv_path: Path) -> dict[str, float]:
    rows = _read_rows(csv_path)
    if not rows:
        return {"sample_count": 0.0, "valid_sample_count": 0.0}

    t = _series(rows, "timestamp_sec")
    controller_vs_planner = _series(rows, "controller_vs_planner_cte_m")
    controller_vs_gt = _series(rows, "controller_vs_gt_cte_m")
    planner_vs_gt_rms = _series(rows, "planner_vs_gt_cte_rms_m")
    planner_vs_gt_p95 = _series(rows, "planner_vs_gt_cte_p95_m")
    planner_vs_gt_max = _series(rows, "planner_vs_gt_cte_max_m")
    valid_flags = _series(rows, "sample_valid_flag")

    duration_sec = float(np.nanmax(t) - np.nanmin(t)) if np.any(np.isfinite(t)) else float("nan")
    valid_sample_count = float(np.count_nonzero(valid_flags > 0.5))

    summary = {
        "sample_count": float(len(rows)),
        "valid_sample_count": valid_sample_count,
        "duration_sec": duration_sec,
        "controller_vs_planner_cte_rms_m": _nanrms(np.abs(controller_vs_planner)),
        "controller_vs_planner_cte_p95_m": _nanpercentile(np.abs(controller_vs_planner), 95.0),
        "controller_vs_planner_cte_max_m": _nanmax(np.abs(controller_vs_planner)),
        "controller_vs_gt_cte_rms_m": _nanrms(np.abs(controller_vs_gt)),
        "controller_vs_gt_cte_p95_m": _nanpercentile(np.abs(controller_vs_gt), 95.0),
        "controller_vs_gt_cte_max_m": _nanmax(np.abs(controller_vs_gt)),
        "planner_vs_gt_cte_rms_m": _nanrms(planner_vs_gt_rms),
        "planner_vs_gt_cte_p95_m": _nanpercentile(planner_vs_gt_p95, 95.0),
        "planner_vs_gt_cte_max_m": _nanmax(planner_vs_gt_max),
    }

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "")).strip() or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
    for key, value in sorted(status_counts.items()):
        summary[f"status_count::{key}"] = float(value)

    return summary


def write_path_tracking_summary_files(summary: dict[str, float], json_path: Path, txt_path: Path) -> None:
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "Path Tracking Evaluation Summary",
        f"sample_count: {summary.get('sample_count', float('nan')):.0f}",
        f"valid_sample_count: {summary.get('valid_sample_count', float('nan')):.0f}",
        f"duration_sec: {summary.get('duration_sec', float('nan')):.3f}",
        f"planner_vs_gt_cte_rms_m: {summary.get('planner_vs_gt_cte_rms_m', float('nan')):.6f}",
        f"planner_vs_gt_cte_p95_m: {summary.get('planner_vs_gt_cte_p95_m', float('nan')):.6f}",
        f"planner_vs_gt_cte_max_m: {summary.get('planner_vs_gt_cte_max_m', float('nan')):.6f}",
        f"controller_vs_planner_cte_rms_m: {summary.get('controller_vs_planner_cte_rms_m', float('nan')):.6f}",
        f"controller_vs_planner_cte_p95_m: {summary.get('controller_vs_planner_cte_p95_m', float('nan')):.6f}",
        f"controller_vs_planner_cte_max_m: {summary.get('controller_vs_planner_cte_max_m', float('nan')):.6f}",
        f"controller_vs_gt_cte_rms_m: {summary.get('controller_vs_gt_cte_rms_m', float('nan')):.6f}",
        f"controller_vs_gt_cte_p95_m: {summary.get('controller_vs_gt_cte_p95_m', float('nan')):.6f}",
        f"controller_vs_gt_cte_max_m: {summary.get('controller_vs_gt_cte_max_m', float('nan')):.6f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row]


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([_safe_float(row.get(key, float("nan"))) for row in rows], dtype=np.float64)


def _nanrms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def _nanpercentile(values: np.ndarray, percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def _nanmax(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))
