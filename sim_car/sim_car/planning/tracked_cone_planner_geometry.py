"""Shared geometry helpers for migrated tracked-cone planners."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Protocol

import numpy as np

from sim_car.planning.planner_utils import _clamp

_QUATERNION_DOUBLE = 2.0  # Quaternion rotation formulas double cross-product terms.
_ROTATION_IDENTITY_SCALE = 1.0  # Rotation matrices start from an identity diagonal.


class BoundaryChainConfig(Protocol):
    min_step_m: float
    max_step_m: float
    max_heading_change_rad: float
    min_forward_progress_m: float


class TrackWidthFilterConfig(Protocol):
    initial_width_m: float
    min_width_m: float
    max_width_m: float
    width_filter_alpha: float
    max_width_delta_per_update_m: float


@dataclass
class BoundaryChainData:
    filtered_indices: np.ndarray
    global_points: np.ndarray
    local_points: np.ndarray
    tangents_local: np.ndarray
    mean_heading_change_rad: float
    forward_extent_m: float
    rejected_reasons_by_filtered_index: dict[int, str] = field(default_factory=dict)


@dataclass
class BoundaryChainGrowth:
    chain_positions: np.ndarray
    heading_changes: list[float]
    rejected_reasons_by_filtered_index: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class UnknownPartnerCheck:
    longitudinal_error_m: float
    width_error_m: float
    radial_error_m: float

    @property
    def cost(self) -> float:
        return float(self.longitudinal_error_m + self.width_error_m + self.radial_error_m)


def empty_boundary_chain_data(
    *,
    rejected_reasons_by_filtered_index: Optional[dict[int, str]] = None,
) -> BoundaryChainData:
    return BoundaryChainData(
        filtered_indices=np.empty((0,), dtype=np.int64),
        global_points=np.empty((0, 2), dtype=np.float64),
        local_points=np.empty((0, 2), dtype=np.float64),
        tangents_local=np.empty((0, 2), dtype=np.float64),
        mean_heading_change_rad=float("inf"),
        forward_extent_m=0.0,
        rejected_reasons_by_filtered_index=dict(rejected_reasons_by_filtered_index or {}),
    )


def update_track_width_estimate(
    previous_width_m: Optional[float],
    measured_width_m: Optional[float],
    config: TrackWidthFilterConfig,
) -> float:
    width = (
        float(config.initial_width_m)
        if previous_width_m is None or not math.isfinite(float(previous_width_m))
        else float(previous_width_m)
    )
    width = _clamp(width, config.min_width_m, config.max_width_m)
    if measured_width_m is None or not math.isfinite(float(measured_width_m)):
        return width

    measured = _clamp(float(measured_width_m), config.min_width_m, config.max_width_m)
    delta = _clamp(
        measured - width,
        -float(config.max_width_delta_per_update_m),
        float(config.max_width_delta_per_update_m),
    )
    alpha = _clamp(float(config.width_filter_alpha), 0.0, 1.0)
    updated = width + (alpha * delta)
    return _clamp(updated, config.min_width_m, config.max_width_m)


def build_boundary_chain_data(
    *,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    side_indices: np.ndarray,
    config: BoundaryChainConfig,
    collect_rejection_reasons: bool = False,
    min_step_m: Optional[float] = None,
) -> BoundaryChainData:
    side_indices = np.asarray(side_indices, dtype=np.int64)
    if side_indices.size == 0:
        return empty_boundary_chain_data()

    side_local = filtered_local[side_indices]
    seed_pos = select_seed_index(side_local)
    if seed_pos < 0:
        rejected = (
            {int(filtered_idx): "chain_no_forward_seed" for filtered_idx in side_indices}
            if collect_rejection_reasons
            else {}
        )
        return empty_boundary_chain_data(rejected_reasons_by_filtered_index=rejected)

    growth = grow_boundary_chain_positions(
        side_local=side_local,
        config=config,
        seed_pos=seed_pos,
        collect_rejection_reasons=collect_rejection_reasons,
        min_step_m=min_step_m,
    )
    return materialize_boundary_chain_data(
        filtered_points=filtered_points,
        filtered_local=filtered_local,
        side_indices=side_indices,
        chain_positions=growth.chain_positions,
        heading_changes=growth.heading_changes,
        rejected_reasons_by_filtered_index=growth.rejected_reasons_by_filtered_index,
    )


def grow_boundary_chain_positions(
    *,
    side_local: np.ndarray,
    config: BoundaryChainConfig,
    seed_pos: int,
    collect_rejection_reasons: bool = False,
    min_step_m: Optional[float] = None,
) -> BoundaryChainGrowth:
    chain_positions = [int(seed_pos)]
    remaining = [idx for idx in range(side_local.shape[0]) if idx != int(seed_pos)]
    heading = np.asarray([1.0, 0.0], dtype=np.float64)
    heading_changes: list[float] = []
    rejection_reasons_by_side_pos: dict[int, str] = {}
    step_min = float(config.min_step_m) if min_step_m is None else float(min_step_m)

    while remaining:
        current_local = side_local[chain_positions[-1]]
        current_range = float(np.hypot(current_local[0], current_local[1]))
        best_pos = None
        best_score: Optional[tuple[float, float, float, float, int]] = None
        best_heading = heading
        best_heading_change = 0.0
        iteration_reasons: dict[int, str] = {}

        for candidate_pos in remaining:
            candidate_local = side_local[candidate_pos]
            candidate_range = float(np.hypot(candidate_local[0], candidate_local[1]))
            radial_progress = candidate_range - current_range
            min_radial_progress = max(0.05, 0.5 * float(config.min_forward_progress_m))
            delta = candidate_local - current_local
            distance = float(np.hypot(delta[0], delta[1]))

            if distance < step_min:
                iteration_reasons[candidate_pos] = "chain_step_too_close"
                continue
            if distance > float(config.max_step_m):
                iteration_reasons[candidate_pos] = "chain_step_too_far"
                continue
            if not candidate_progresses_from_vehicle(
                current_local=current_local,
                candidate_local=candidate_local,
                min_progress_m=float(config.min_forward_progress_m),
            ):
                iteration_reasons[candidate_pos] = "chain_no_forward_progress"
                continue
            if radial_progress < min_radial_progress:
                iteration_reasons[candidate_pos] = "chain_radial_regression"
                continue

            step_heading = delta / distance
            forward = float(np.dot(delta, heading))
            if forward < float(config.min_forward_progress_m):
                iteration_reasons[candidate_pos] = "chain_forward_projection"
                continue

            heading_change = abs(angle_between(heading, step_heading))
            if heading_change > float(config.max_heading_change_rad):
                iteration_reasons[candidate_pos] = "chain_heading_change"
                continue
            if candidate_is_shadowed(
                current_local=current_local,
                candidate_pos=candidate_pos,
                side_local=side_local,
                remaining=remaining,
            ):
                iteration_reasons[candidate_pos] = "chain_shadowed"
                continue

            iteration_reasons[candidate_pos] = "chain_not_best_next_step"
            score = (
                distance,
                heading_change,
                radial_progress,
                -forward,
                candidate_pos,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_pos = candidate_pos
                best_heading = step_heading
                best_heading_change = heading_change

        if best_pos is None:
            rejection_reasons_by_side_pos.update(iteration_reasons)
            break

        if collect_rejection_reasons:
            rejection_reasons_by_side_pos.update(
                {
                    pos: reason
                    for pos, reason in iteration_reasons.items()
                    if pos != best_pos
                }
            )
        chain_positions.append(best_pos)
        remaining.remove(best_pos)
        heading = best_heading
        heading_changes.append(best_heading_change)

    rejected: dict[int, str] = {}
    if collect_rejection_reasons:
        selected_positions = {int(pos) for pos in chain_positions}
        rejected = {
            int(pos): rejection_reasons_by_side_pos.get(pos, "chain_unreached")
            for pos in range(side_local.shape[0])
            if pos not in selected_positions
        }

    return BoundaryChainGrowth(
        chain_positions=np.asarray(chain_positions, dtype=np.int64),
        heading_changes=heading_changes,
        rejected_reasons_by_filtered_index=rejected,
    )


def materialize_boundary_chain_data(
    *,
    filtered_points: np.ndarray,
    filtered_local: np.ndarray,
    side_indices: np.ndarray,
    chain_positions: np.ndarray,
    heading_changes: list[float],
    rejected_reasons_by_filtered_index: Optional[dict[int, str]] = None,
) -> BoundaryChainData:
    chain_positions = np.asarray(chain_positions, dtype=np.int64)
    if chain_positions.size == 0:
        return empty_boundary_chain_data(
            rejected_reasons_by_filtered_index=rejected_reasons_by_filtered_index
        )

    filtered_indices = np.asarray(side_indices, dtype=np.int64)[chain_positions]
    global_points = filtered_points[filtered_indices]
    local_points = filtered_local[filtered_indices]
    tangents_local = estimate_tangents(local_points)
    mean_heading_change = float(np.mean(heading_changes)) if heading_changes else 0.0
    forward_extent = (
        float(np.max(local_points[:, 0]) - np.min(local_points[:, 0]))
        if local_points.shape[0] > 0
        else 0.0
    )
    rejected_by_filtered_idx = {
        int(np.asarray(side_indices, dtype=np.int64)[pos]): reason
        for pos, reason in dict(rejected_reasons_by_filtered_index or {}).items()
    }
    return BoundaryChainData(
        filtered_indices=filtered_indices,
        global_points=global_points,
        local_points=local_points,
        tangents_local=tangents_local,
        mean_heading_change_rad=mean_heading_change,
        forward_extent_m=forward_extent,
        rejected_reasons_by_filtered_index=rejected_by_filtered_idx,
    )


def select_seed_index(side_local: np.ndarray) -> int:
    candidates = np.flatnonzero(side_local[:, 0] >= 0.0)
    if candidates.size == 0:
        return -1

    best_pos = -1
    best_score = None
    for pos in candidates:
        x = float(side_local[pos, 0])
        y = float(side_local[pos, 1])
        score = (x, abs(y), math.hypot(x, y), int(pos))
        if best_score is None or score < best_score:
            best_score = score
            best_pos = int(pos)
    return best_pos


def candidate_progresses_from_vehicle(
    *,
    current_local: np.ndarray,
    candidate_local: np.ndarray,
    min_progress_m: float,
) -> bool:
    x_margin = max(0.05, 0.5 * float(min_progress_m))
    if float(candidate_local[0]) >= float(current_local[0]) - x_margin:
        return True

    current_y = float(current_local[1])
    candidate_y = float(candidate_local[1])
    if abs(current_y) <= 0.05:
        return False
    same_side = current_y * candidate_y >= 0.0
    outboard_progress = abs(candidate_y) >= abs(current_y) + (0.5 * float(min_progress_m))
    return bool(same_side and outboard_progress)


def candidate_is_shadowed(
    *,
    current_local: np.ndarray,
    candidate_pos: int,
    side_local: np.ndarray,
    remaining: list[int],
) -> bool:
    candidate_delta = side_local[candidate_pos] - current_local
    candidate_distance = float(np.hypot(candidate_delta[0], candidate_delta[1]))
    if candidate_distance <= 1e-9:
        return True
    candidate_dir = candidate_delta / candidate_distance

    for other_pos in remaining:
        if other_pos == candidate_pos:
            continue
        other_delta = side_local[other_pos] - current_local
        other_distance = float(np.hypot(other_delta[0], other_delta[1]))
        if other_distance <= 1e-9 or other_distance >= candidate_distance:
            continue
        other_dir = other_delta / other_distance
        if abs(angle_between(candidate_dir, other_dir)) > 0.30:
            continue
        if float(np.dot(other_delta, candidate_dir)) <= 0.0:
            continue
        return True
    return False


def estimate_tangents(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.shape[0] == 1:
        return np.asarray([[1.0, 0.0]], dtype=np.float64)

    tangents = np.empty_like(points)
    for idx in range(points.shape[0]):
        if idx == 0:
            delta = points[1] - points[0]
        elif idx == points.shape[0] - 1:
            delta = points[-1] - points[-2]
        else:
            delta = points[idx + 1] - points[idx - 1]
        norm = float(np.hypot(delta[0], delta[1]))
        tangents[idx] = (
            np.asarray([1.0, 0.0], dtype=np.float64)
            if norm <= 1e-9
            else (delta / norm)
        )
    return tangents


def moving_average(points: np.ndarray, window: int) -> np.ndarray:
    if points.shape[0] <= 2 or window <= 1:
        return np.asarray(points, dtype=np.float64)

    radius = max(1, int(window)) // 2
    out = np.empty_like(points)
    for idx in range(points.shape[0]):
        start = max(0, idx - radius)
        stop = min(points.shape[0], idx + radius + 1)
        out[idx] = np.mean(points[start:stop], axis=0)
    return out


def path_cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.asarray([0.0], dtype=np.float64)
    diffs = np.diff(points, axis=0)
    lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate(([0.0], np.cumsum(lengths))).astype(np.float64)


def path_heading_delta_max(path_local: np.ndarray) -> float:
    if path_local.shape[0] < 3:
        return 0.0
    diffs = np.diff(path_local, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    delta = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    return float(np.max(np.abs(delta))) if delta.size else 0.0


def path_curvature_abs_max(path_local: np.ndarray) -> float:
    if path_local.shape[0] < 3:
        return 0.0
    diffs = np.diff(path_local, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    valid = seg_len > 1e-6
    if np.count_nonzero(valid) < 2:
        return 0.0
    headings = np.arctan2(diffs[valid, 1], diffs[valid, 0])
    if headings.size < 2:
        return 0.0
    delta_heading = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    ds = 0.5 * (seg_len[valid][1:] + seg_len[valid][:-1])
    valid_ds = ds > 1e-6
    if not np.any(valid_ds):
        return 0.0
    curvature = np.abs(delta_heading[valid_ds] / ds[valid_ds])
    return float(np.max(curvature)) if curvature.size else 0.0


def path_self_intersects(
    points: np.ndarray,
    *,
    skip_closing_segment: bool = True,
) -> bool:
    if points.shape[0] < 4:
        return False
    for idx in range(points.shape[0] - 1):
        a0 = points[idx]
        a1 = points[idx + 1]
        for jdx in range(idx + 2, points.shape[0] - 1):
            if skip_closing_segment and idx == 0 and jdx == points.shape[0] - 2:
                continue
            b0 = points[jdx]
            b1 = points[jdx + 1]
            if segments_intersect(a0, a1, b0, b1):
                return True
    return False


def segments_intersect(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a0, a1, b0)
    o2 = orient(a0, a1, b1)
    o3 = orient(b0, b1, a0)
    o4 = orient(b0, b1, a1)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def to_vehicle_frame(
    points_xy: np.ndarray,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
) -> np.ndarray:
    dx = points_xy[:, 0] - float(vehicle_xy[0])
    dy = points_xy[:, 1] - float(vehicle_xy[1])
    cos_yaw = math.cos(vehicle_yaw)
    sin_yaw = math.sin(vehicle_yaw)
    x_local = (cos_yaw * dx) + (sin_yaw * dy)
    y_local = (-sin_yaw * dx) + (cos_yaw * dy)
    return np.column_stack((x_local, y_local)).astype(np.float64)


def inward_normal(tangent: np.ndarray, side: str) -> np.ndarray:
    if side == "blue":
        normal = np.asarray([tangent[1], -tangent[0]], dtype=np.float64)
    else:
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    norm = float(np.hypot(normal[0], normal[1]))
    if norm <= 1e-9:
        return np.asarray([0.0, 0.0], dtype=np.float64)
    return normal / norm


def angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    angle_a = math.atan2(float(vec_a[1]), float(vec_a[0]))
    angle_b = math.atan2(float(vec_b[1]), float(vec_b[0]))
    return math.atan2(math.sin(angle_b - angle_a), math.cos(angle_b - angle_a))


def pair_width_in_range(width_m: float, min_width_m: float, max_width_m: float) -> bool:
    return float(min_width_m) <= float(width_m) <= float(max_width_m)


def midpoint_outside_pair_span(midpoint_local: np.ndarray, width_m: float) -> bool:
    return bool(abs(float(midpoint_local[1])) > (0.5 * float(width_m)))


def inward_distance(delta: np.ndarray, normal: np.ndarray) -> float:
    return float(np.dot(delta, normal))


def width_jump_exceeds(
    previous_width_m: Optional[float],
    current_width_m: float,
    max_width_jump_m: float,
) -> bool:
    return (
        previous_width_m is not None
        and abs(float(current_width_m) - float(previous_width_m)) > float(max_width_jump_m)
    )


def unknown_partner_check(
    *,
    partner_local: np.ndarray,
    expected_partner_local: np.ndarray,
    anchor_tangent: np.ndarray,
    width_m: float,
    expected_width_m: float,
) -> UnknownPartnerCheck:
    return UnknownPartnerCheck(
        longitudinal_error_m=abs(
            float(np.dot(partner_local - expected_partner_local, anchor_tangent))
        ),
        width_error_m=abs(float(width_m) - float(expected_width_m)),
        radial_error_m=float(np.hypot(*(partner_local - expected_partner_local))),
    )


def unknown_partner_within_limits(
    check: UnknownPartnerCheck,
    *,
    max_longitudinal_error_m: float,
    max_width_error_m: float,
    search_radius_m: float,
) -> bool:
    return bool(
        check.longitudinal_error_m <= float(max_longitudinal_error_m)
        and check.width_error_m <= float(max_width_error_m)
        and check.radial_error_m <= float(search_radius_m)
    )


def prefer_previous_partner_option(
    *,
    options: list[dict[str, object]],
    current_option: dict[str, object],
    preferred_partner_track_id: Optional[int],
    reassignment_margin: float,
) -> dict[str, object]:
    if preferred_partner_track_id is None:
        return current_option
    preferred = next(
        (
            option
            for option in options
            if int(option["partner_track_id"]) == int(preferred_partner_track_id)
        ),
        None,
    )
    if preferred is None:
        return current_option
    if float(preferred["cost"]) <= (
        float(current_option["cost"]) + float(reassignment_margin)
    ):
        return preferred
    return current_option


def _transform_point(transform, x: float, y: float, z: float) -> tuple[float, float, float]:
    t = transform.transform.translation
    q = transform.transform.rotation
    qx = float(q.x)
    qy = float(q.y)
    qz = float(q.z)
    qw = float(q.w)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    r00 = _ROTATION_IDENTITY_SCALE - _QUATERNION_DOUBLE * (yy + zz)
    r01 = _QUATERNION_DOUBLE * (xy - wz)
    r02 = _QUATERNION_DOUBLE * (xz + wy)
    r10 = _QUATERNION_DOUBLE * (xy + wz)
    r11 = _ROTATION_IDENTITY_SCALE - _QUATERNION_DOUBLE * (xx + zz)
    r12 = _QUATERNION_DOUBLE * (yz - wx)
    r20 = _QUATERNION_DOUBLE * (xz - wy)
    r21 = _QUATERNION_DOUBLE * (yz + wx)
    r22 = _ROTATION_IDENTITY_SCALE - _QUATERNION_DOUBLE * (xx + yy)

    tx = float(t.x)
    ty = float(t.y)
    tz = float(t.z)

    px = (r00 * x) + (r01 * y) + (r02 * z) + tx
    py = (r10 * x) + (r11 * y) + (r12 * z) + ty
    pz = (r20 * x) + (r21 * y) + (r22 * z) + tz
    return px, py, pz


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = _QUATERNION_DOUBLE * (qw * qz + qx * qy)
    cosy_cosp = _ROTATION_IDENTITY_SCALE - _QUATERNION_DOUBLE * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _odom_point_to_base(
    x_odom: float,
    y_odom: float,
    tx: float,
    ty: float,
    yaw: float,
) -> tuple[float, float]:
    dx = x_odom - tx
    dy = y_odom - ty
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x_base = cos_y * dx + sin_y * dy
    y_base = -sin_y * dx + cos_y * dy
    return x_base, y_base


def _base_point_to_odom(
    x_base: float,
    y_base: float,
    tx: float,
    ty: float,
    yaw: float,
) -> tuple[float, float]:
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x_odom = tx + (cos_y * x_base) - (sin_y * y_base)
    y_odom = ty + (sin_y * x_base) + (cos_y * y_base)
    return x_odom, y_odom
