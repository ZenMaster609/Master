from __future__ import annotations

import numpy as np


def path_cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.asarray([0.0], dtype=np.float64)
    seg = points[1:] - points[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    return np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float64)


def sample_path_at_lengths(points: np.ndarray, cum_lengths: np.ndarray, samples: np.ndarray) -> np.ndarray:
    capped = np.clip(samples, 0.0, float(cum_lengths[-1]))
    x = np.interp(capped, cum_lengths, points[:, 0])
    y = np.interp(capped, cum_lengths, points[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


def project_point_to_path_s(path: np.ndarray, cum_lengths: np.ndarray, point_xy: np.ndarray) -> float:
    best_s = 0.0
    best_dist = float("inf")
    for idx in range(path.shape[0] - 1):
        a = path[idx]
        b = path[idx + 1]
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


def extract_forward_path_from_pose(
    *,
    path: np.ndarray,
    vehicle_xy: tuple[float, float],
    resolution_m: float,
) -> np.ndarray | None:
    if path.shape[0] < 2:
        return None
    cum = path_cumulative_lengths(path)
    s_vehicle = project_point_to_path_s(path, cum, np.asarray(vehicle_xy, dtype=np.float64))
    total = float(cum[-1])
    step = max(0.05, float(resolution_m))
    if (total - s_vehicle) <= step:
        return None
    samples = np.arange(s_vehicle, total + 1e-9, step, dtype=np.float64)
    if samples.size == 0 or samples[-1] < total:
        samples = np.concatenate((samples, [total]))
    return sample_path_at_lengths(path, cum, samples)


def resample_to_count(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] == count:
        return points
    if points.shape[0] <= 1 or count <= 1:
        return np.repeat(points[:1], count, axis=0)

    seg = points[1:] - points[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = max(float(cum[-1]), 1e-6)
    samples = np.linspace(0.0, total, count)
    x = np.interp(samples, cum, points[:, 0])
    y = np.interp(samples, cum, points[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


def splice_frozen_near_field(
    *,
    previous: np.ndarray,
    current: np.ndarray,
    freeze_horizon_m: float,
    blend_length_m: float,
    resolution_m: float,
) -> tuple[np.ndarray, bool]:
    if previous.shape[0] < 2 or current.shape[0] < 2:
        return current, False

    prev_cum = path_cumulative_lengths(previous)
    curr_cum = path_cumulative_lengths(current)
    prev_total = float(prev_cum[-1])
    curr_total = float(curr_cum[-1])
    if prev_total <= 1e-6 or curr_total <= 1e-6:
        return current, False

    overlap_m = min(prev_total, curr_total)
    freeze_m = min(max(0.0, float(freeze_horizon_m)), overlap_m)
    if freeze_m <= 0.0:
        return current, False

    blend_m = max(0.0, float(blend_length_m))
    blend_end_m = min(overlap_m, freeze_m + blend_m)
    step_m = max(0.05, float(resolution_m))
    samples = np.arange(0.0, curr_total + 1e-9, step_m, dtype=np.float64)
    if samples.size == 0 or samples[-1] < curr_total:
        samples = np.concatenate((samples, [curr_total]))

    current_rs = sample_path_at_lengths(current, curr_cum, samples)
    previous_rs = sample_path_at_lengths(previous, prev_cum, samples)
    out = np.array(current_rs, copy=True)
    out[samples <= freeze_m + 1e-9] = previous_rs[samples <= freeze_m + 1e-9]
    if blend_end_m > freeze_m + 1e-6:
        blend_mask = (samples > freeze_m) & (samples < blend_end_m)
        if np.any(blend_mask):
            w = ((samples[blend_mask] - freeze_m) / (blend_end_m - freeze_m)).reshape(-1, 1)
            out[blend_mask] = ((1.0 - w) * previous_rs[blend_mask]) + (w * current_rs[blend_mask])

    return out.astype(np.float64), True


def path_forward_extent_local(path_local: np.ndarray) -> float:
    if path_local.shape[0] == 0:
        return 0.0
    if path_local.shape[0] == 1:
        return max(0.0, float(path_local[0, 0]))
    diffs = np.diff(path_local, axis=0)
    path_length = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    x_span = float(np.max(path_local[:, 0]) - np.min(path_local[:, 0]))
    return max(path_length, x_span)

