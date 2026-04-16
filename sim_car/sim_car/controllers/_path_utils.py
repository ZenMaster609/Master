"""Shared path geometry utilities used by steering controllers.

These are pure functions with no state. The path coordinate system assumes
the vehicle is at the origin facing +x, with +y to the left.
"""

from __future__ import annotations

import numpy as np

from sim_car.controllers.base import FloatArray


def validate_control_path(control_path: FloatArray) -> None:
    """Raise ValueError if *control_path* is not a valid (N, 2) array."""
    if control_path.ndim != 2 or control_path.shape[1] != 2:
        raise ValueError("control_path must have shape (N, 2)")
    if control_path.shape[0] == 0:
        raise ValueError("control_path cannot be empty")


def nearest_projection_on_path(
    control_path: FloatArray,
) -> tuple[np.ndarray, int]:
    """Return the closest point on *control_path* to the vehicle (origin) and
    its segment index.

    The path is expressed in vehicle frame, so the vehicle is at (0, 0).
    Projection uses perpendicular foot of the origin onto each segment, clamped
    to the segment endpoints.
    """
    if control_path.shape[0] == 1:
        return np.asarray(control_path[0], dtype=np.float64), 0

    best_distance_sq = float("inf")
    best_point = np.asarray(control_path[0], dtype=np.float64)
    best_segment_idx = 0

    for seg_idx in range(control_path.shape[0] - 1):
        p0 = control_path[seg_idx]
        p1 = control_path[seg_idx + 1]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-12:
            projected = p0
        else:
            t = float(np.clip(-np.dot(p0, seg) / seg_len_sq, 0.0, 1.0))
            projected = p0 + (t * seg)

        distance_sq = float(np.dot(projected, projected))
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_point = np.asarray(projected, dtype=np.float64)
            best_segment_idx = seg_idx

    return best_point, best_segment_idx


def target_point_from_projection(
    *,
    control_path: FloatArray,
    projected_point: np.ndarray,
    nearest_segment_idx: int,
    lookahead_m: float,
) -> np.ndarray:
    """Walk *lookahead_m* along *control_path* starting from *projected_point*.

    Returns the resulting point, or the last path point if the lookahead
    distance exceeds the remaining path length.
    """
    if control_path.shape[0] == 1:
        return np.asarray(control_path[0], dtype=np.float64)

    remaining = max(0.0, float(lookahead_m))
    current_point = np.asarray(projected_point, dtype=np.float64)

    for seg_idx in range(nearest_segment_idx, control_path.shape[0] - 1):
        seg_end = np.asarray(control_path[seg_idx + 1], dtype=np.float64)
        seg_vec = seg_end - current_point
        seg_len = float(np.hypot(seg_vec[0], seg_vec[1]))
        if seg_len <= 1e-9:
            current_point = seg_end
            continue
        if remaining <= seg_len:
            return current_point + ((remaining / seg_len) * seg_vec)
        remaining -= seg_len
        current_point = seg_end

    return np.asarray(control_path[-1], dtype=np.float64)
