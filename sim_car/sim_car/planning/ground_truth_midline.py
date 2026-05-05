"""Ground-truth cone midline helpers for planner debug modes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GroundTruthMidline:
    frame_id: str
    midline_xy: np.ndarray
    left_xy: np.ndarray
    right_xy: np.ndarray


def as_xy(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.reshape(arr, (-1, 2))
    return arr[np.all(np.isfinite(arr), axis=1)]


def build_gt_midline_from_cones(
    *,
    blue_xy: np.ndarray,
    yellow_xy: np.ndarray,
    start_xy: np.ndarray,
    heading_xy: np.ndarray,
    frame_id: str,
    resolution_m: float = 0.5,
) -> GroundTruthMidline:
    left_xy = _close_loop_if_needed(
        order_boundary_points(blue_xy, start_xy=start_xy, heading_xy=heading_xy)
    )
    right_xy = _close_loop_if_needed(
        order_boundary_points(yellow_xy, start_xy=start_xy, heading_xy=heading_xy)
    )
    midline_xy = _build_midline_from_cone_pairs(
        as_xy(blue_xy),
        as_xy(yellow_xy),
        start_xy=np.asarray(start_xy, dtype=np.float64),
        heading_xy=np.asarray(heading_xy, dtype=np.float64),
        resolution_m=resolution_m,
    )
    if midline_xy.shape[0] < 2:
        midline_xy = build_midline_from_boundaries(left_xy, right_xy, resolution_m=resolution_m)
    return GroundTruthMidline(
        frame_id=str(frame_id).strip(),
        midline_xy=midline_xy,
        left_xy=left_xy,
        right_xy=right_xy,
    )


def build_forward_path_from_loop(
    *,
    path_xy: np.ndarray,
    vehicle_xy: np.ndarray,
    resolution_m: float,
    horizon_m: float,
) -> np.ndarray:
    path_xy = _dedupe_points(as_xy(path_xy))
    vehicle_xy = np.asarray(vehicle_xy, dtype=np.float64).reshape(2,)
    if path_xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    closed = _is_closed_path(path_xy)
    sample_path = np.asarray(path_xy[:-1] if closed else path_xy, dtype=np.float64)
    if sample_path.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    cumulative, total_m = _path_cumulative_lengths(sample_path, closed=closed)
    if total_m <= 1e-6:
        return np.empty((0, 2), dtype=np.float64)

    start_s = project_point_to_path_s(sample_path, vehicle_xy, closed=closed, cumulative=cumulative)
    if not math.isfinite(start_s):
        return np.empty((0, 2), dtype=np.float64)

    step_m = max(0.05, float(resolution_m))
    if closed:
        forward_m = total_m if float(horizon_m) <= 0.0 else min(float(horizon_m), total_m)
    else:
        forward_m = max(0.0, total_m - start_s)
        if float(horizon_m) > 0.0:
            forward_m = min(forward_m, float(horizon_m))
    if forward_m <= step_m:
        return np.empty((0, 2), dtype=np.float64)

    offsets = np.arange(0.0, forward_m + 1e-9, step_m, dtype=np.float64)
    if offsets.size == 0 or offsets[-1] < forward_m:
        offsets = np.concatenate((offsets, [forward_m]))
    samples = start_s + offsets
    return sample_path_at_lengths(sample_path, cumulative, samples, closed=closed)


def project_point_to_path_s(
    path_xy: np.ndarray,
    point_xy: np.ndarray,
    *,
    closed: bool,
    cumulative: np.ndarray | None = None,
) -> float:
    path_xy = as_xy(path_xy)
    point_xy = np.asarray(point_xy, dtype=np.float64).reshape(2,)
    if path_xy.shape[0] < 2:
        return float("nan")
    if cumulative is None:
        cumulative, _ = _path_cumulative_lengths(path_xy, closed=closed)

    best_s = float("nan")
    best_dist = float("inf")
    n_segments = path_xy.shape[0] if closed else path_xy.shape[0] - 1
    for idx in range(n_segments):
        a = path_xy[idx]
        b = path_xy[(idx + 1) % path_xy.shape[0]]
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
            best_s = float(cumulative[idx] + (t * np.hypot(ab[0], ab[1])))
    return best_s


def sample_path_at_lengths(
    points_xy: np.ndarray,
    cum_lengths: np.ndarray,
    samples: np.ndarray,
    *,
    closed: bool,
) -> np.ndarray:
    points_xy = as_xy(points_xy)
    if points_xy.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    sample_values = np.asarray(samples, dtype=np.float64)
    if closed:
        total_m = float(cum_lengths[-1])
        if total_m <= 1e-9:
            return np.empty((0, 2), dtype=np.float64)
        interp_path = np.vstack((points_xy, points_xy[:1]))
        capped = np.mod(sample_values, total_m)
        return _interp_path(interp_path, cum_lengths, capped)
    capped = np.clip(sample_values, 0.0, float(cum_lengths[-1]))
    return _interp_path(points_xy, cum_lengths, capped)


def order_boundary_points(
    points_xy: np.ndarray,
    *,
    start_xy: np.ndarray,
    heading_xy: np.ndarray,
) -> np.ndarray:
    points_xy = _dedupe_points(as_xy(points_xy))
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


def build_midline_from_boundaries(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    *,
    resolution_m: float = 0.5,
) -> np.ndarray:
    left_xy = _dedupe_points(as_xy(left_xy))
    right_xy = _dedupe_points(as_xy(right_xy))
    if left_xy.shape[0] < 2 or right_xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    left_closed = _is_closed_path(left_xy)
    right_closed = _is_closed_path(right_xy)
    if left_closed:
        left_xy = np.asarray(left_xy[:-1], dtype=np.float64)
    if right_closed:
        right_xy = np.asarray(right_xy[:-1], dtype=np.float64)
    if left_xy.shape[0] < 2 or right_xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    midpoint_pairs = _dedupe_points(_build_midpoint_chain_from_matched_boundaries(left_xy, right_xy))
    if left_closed and right_closed:
        midpoint_pairs = _close_loop_if_needed(midpoint_pairs)
    return _resample_midline_path(midpoint_pairs, resolution_m=resolution_m, closed=_is_closed_path(midpoint_pairs))


def _build_midline_from_cone_pairs(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    *,
    start_xy: np.ndarray,
    heading_xy: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    left_xy = _dedupe_points(as_xy(left_xy))
    right_xy = _dedupe_points(as_xy(right_xy))
    if left_xy.shape[0] < 2 or right_xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    pair_cost = np.linalg.norm(left_xy[:, None, :] - right_xy[None, :, :], axis=2)
    width_m = _estimate_track_width(left_xy, right_xy)
    if not math.isfinite(width_m):
        return np.empty((0, 2), dtype=np.float64)

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

    midpoint_xy = as_xy(midpoint_pairs)
    if midpoint_xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    midpoint_xy = order_boundary_points(midpoint_xy, start_xy=start_xy, heading_xy=heading_xy)
    midpoint_xy = _close_loop_if_needed(midpoint_xy)
    return _resample_midline_path(midpoint_xy, resolution_m=resolution_m, closed=_is_closed_path(midpoint_xy))


def _resample_midline_path(points_xy: np.ndarray, *, resolution_m: float, closed: bool) -> np.ndarray:
    points_xy = as_xy(points_xy)
    if points_xy.shape[0] < 2:
        return points_xy
    sample_path = np.asarray(points_xy[:-1] if closed else points_xy, dtype=np.float64)
    cumulative, total_m = _path_cumulative_lengths(sample_path, closed=closed)
    if total_m <= 1e-6:
        return sample_path
    step_m = max(0.05, float(resolution_m))
    samples = np.arange(0.0, total_m + 1e-9, step_m, dtype=np.float64)
    if samples.size == 0 or samples[-1] < total_m:
        samples = np.concatenate((samples, [total_m]))
    out = sample_path_at_lengths(sample_path, cumulative, samples, closed=closed)
    if closed and out.shape[0] >= 2 and float(np.hypot(*(out[0] - out[-1]))) > 1e-6:
        out = np.vstack((out, out[:1]))
    return out


def _path_cumulative_lengths(points_xy: np.ndarray, *, closed: bool) -> tuple[np.ndarray, float]:
    points_xy = as_xy(points_xy)
    if points_xy.shape[0] <= 1:
        return np.asarray([0.0], dtype=np.float64), 0.0
    diffs = points_xy[1:] - points_xy[:-1]
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    if closed:
        closing = points_xy[0] - points_xy[-1]
        seg_len = np.concatenate((seg_len, [float(np.hypot(closing[0], closing[1]))]))
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float64)
    return cumulative, float(cumulative[-1])


def _interp_path(points_xy: np.ndarray, cum_lengths: np.ndarray, samples: np.ndarray) -> np.ndarray:
    x = np.interp(samples, cum_lengths, points_xy[:, 0])
    y = np.interp(samples, cum_lengths, points_xy[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


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

    midpoint_chain = np.empty((len(pairs), 2), dtype=np.float64)
    for idx, (left_idx, right_idx) in enumerate(pairs):
        midpoint_chain[idx] = 0.5 * (left_xy[left_idx] + right_xy[right_idx])
    return midpoint_chain


def _close_loop_if_needed(points_xy: np.ndarray) -> np.ndarray:
    points_xy = as_xy(points_xy)
    if points_xy.shape[0] < 3:
        return points_xy
    if _is_closed_path(points_xy):
        return points_xy
    median_spacing = _median_segment_length(points_xy)
    if not math.isfinite(median_spacing):
        return points_xy
    closure_gap = float(np.hypot(*(points_xy[0] - points_xy[-1])))
    if closure_gap <= max(1.5, 2.5 * median_spacing):
        return np.vstack((points_xy, points_xy[:1]))
    return points_xy


def _is_closed_path(points_xy: np.ndarray) -> bool:
    points_xy = as_xy(points_xy)
    return points_xy.shape[0] >= 3 and float(np.hypot(*(points_xy[0] - points_xy[-1]))) <= 1e-6


def _dedupe_points(points_xy: np.ndarray, *, min_distance_m: float = 1e-6) -> np.ndarray:
    points_xy = as_xy(points_xy)
    if points_xy.shape[0] <= 1:
        return points_xy
    out = [points_xy[0]]
    for point in points_xy[1:]:
        if float(np.hypot(*(point - out[-1]))) > min_distance_m:
            out.append(point)
    return np.asarray(out, dtype=np.float64)


def _median_segment_length(points_xy: np.ndarray) -> float:
    points_xy = as_xy(points_xy)
    if points_xy.shape[0] < 2:
        return float("nan")
    diffs = points_xy[1:] - points_xy[:-1]
    lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    finite = lengths[np.isfinite(lengths) & (lengths > 1e-6)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _estimate_track_width(left_xy: np.ndarray, right_xy: np.ndarray) -> float:
    left_xy = as_xy(left_xy)
    right_xy = as_xy(right_xy)
    if left_xy.shape[0] == 0 or right_xy.shape[0] == 0:
        return float("nan")
    pair_cost = np.linalg.norm(left_xy[:, None, :] - right_xy[None, :, :], axis=2)
    nearest = np.concatenate((np.min(pair_cost, axis=1), np.min(pair_cost, axis=0)))
    finite = nearest[np.isfinite(nearest) & (nearest > 1e-6)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.hypot(vec[0], vec[1]))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / norm
