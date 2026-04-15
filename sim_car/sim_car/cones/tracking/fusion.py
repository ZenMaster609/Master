"""Fusion helpers for local cone memory tracking."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


CLASS_NAMES = ("unknown", "blue", "yellow", "orange")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def normalize_color(label: str) -> str:
    token = str(label).strip().lower().replace('-', '_').replace(' ', '_')
    if (
        'big_orange' in token
        or 'large_orange' in token
        or (('big' in token or 'large' in token) and 'orange' in token)
    ):
        return 'orange'
    if 'orange' in token:
        return 'orange'
    if 'yellow' in token:
        return 'yellow'
    if 'blue' in token:
        return 'blue'
    return 'unknown'


def resolve_boundary_color_by_lateral_position(
    label: str,
    lateral_y: float,
    *,
    infer_unknown: bool = True,
    infer_orange: bool = True,
) -> str:
    """Resolve ambiguous cone colors to a boundary side using lateral position."""

    normalized = normalize_color(label)
    if normalized == 'unknown':
        if not infer_unknown:
            return normalized
        return 'blue' if float(lateral_y) >= 0.0 else 'yellow'
    if normalized == 'orange':
        if not infer_orange:
            return normalized
        return 'blue' if float(lateral_y) >= 0.0 else 'yellow'
    return normalized


def normalize_boundary_color(label: Optional[str]) -> str:
    normalized = normalize_color(label or '')
    if normalized in {'blue', 'yellow'}:
        return normalized
    return ''


def resolve_boundary_colors_for_planning(
    *,
    points_xy: np.ndarray,
    raw_colors: list[str],
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
    boundary_hints: Optional[list[str]] = None,
    infer_unknown: bool = True,
    infer_orange: bool = True,
    orange_min_lateral_m: float = 0.9,
    orange_neighbor_radius_m: float = 3.5,
    orange_neighbor_margin_m: float = 0.75,
) -> list[str]:
    """Resolve raw cone classes to planner-facing boundary colors.

    Output colors are limited to blue/yellow/unknown. Any provided boundary hint
    takes precedence over raw color so `/tracked_cones` can carry both raw class
    and planner-facing side at the same time.
    """

    if points_xy.size == 0 or not raw_colors:
        return []

    normalized_raw = [normalize_color(color) for color in raw_colors]
    resolved = ['unknown'] * len(normalized_raw)
    normalized_hints = (
        [normalize_boundary_color(color) for color in boundary_hints]
        if boundary_hints is not None
        else [''] * len(normalized_raw)
    )
    if len(normalized_hints) != len(normalized_raw):
        normalized_hints = [''] * len(normalized_raw)

    rel = np.asarray(points_xy, dtype=np.float64) - np.asarray(vehicle_xy, dtype=np.float64).reshape(1, 2)
    vx, vy = _rotate_into_vehicle(rel, float(vehicle_yaw))

    for idx, raw_color in enumerate(normalized_raw):
        hint = normalized_hints[idx]
        if hint:
            resolved[idx] = hint
            continue
        if raw_color in {'blue', 'yellow'}:
            resolved[idx] = raw_color

    blue_idx = np.asarray([idx for idx, color in enumerate(resolved) if color == 'blue'], dtype=np.int64)
    yellow_idx = np.asarray([idx for idx, color in enumerate(resolved) if color == 'yellow'], dtype=np.int64)

    for idx, raw_color in enumerate(normalized_raw):
        if resolved[idx] in {'blue', 'yellow'}:
            continue
        lateral_y = float(vy[idx])
        if raw_color == 'unknown':
            if infer_unknown:
                resolved[idx] = resolve_boundary_color_by_lateral_position(
                    raw_color,
                    lateral_y,
                    infer_unknown=True,
                    infer_orange=False,
                )
            continue
        if raw_color == 'orange':
            if not infer_orange:
                continue
            resolved[idx] = _infer_orange_boundary_color(
                idx=idx,
                vx=vx,
                vy=vy,
                blue_idx=blue_idx,
                yellow_idx=yellow_idx,
                min_lateral_m=orange_min_lateral_m,
                neighbor_radius_m=orange_neighbor_radius_m,
                neighbor_margin_m=orange_neighbor_margin_m,
            )

    return [color if color in {'blue', 'yellow'} else 'unknown' for color in resolved]


def _rotate_into_vehicle(points_xy: np.ndarray, vehicle_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    cos_yaw = math.cos(float(vehicle_yaw))
    sin_yaw = math.sin(float(vehicle_yaw))
    vx = (cos_yaw * points_xy[:, 0]) + (sin_yaw * points_xy[:, 1])
    vy = (-sin_yaw * points_xy[:, 0]) + (cos_yaw * points_xy[:, 1])
    return vx, vy


def _infer_orange_boundary_color(
    *,
    idx: int,
    vx: np.ndarray,
    vy: np.ndarray,
    blue_idx: np.ndarray,
    yellow_idx: np.ndarray,
    min_lateral_m: float,
    neighbor_radius_m: float,
    neighbor_margin_m: float,
) -> str:
    d_blue = _nearest_neighbor_distance(idx=idx, vx=vx, vy=vy, candidate_idx=blue_idx)
    d_yellow = _nearest_neighbor_distance(idx=idx, vx=vx, vy=vy, candidate_idx=yellow_idx)
    neighbor_radius_m = float(max(0.0, neighbor_radius_m))
    neighbor_margin_m = float(max(0.0, neighbor_margin_m))

    if math.isfinite(d_blue) and d_blue <= neighbor_radius_m and (d_yellow - d_blue) >= neighbor_margin_m:
        return 'blue'
    if math.isfinite(d_yellow) and d_yellow <= neighbor_radius_m and (d_blue - d_yellow) >= neighbor_margin_m:
        return 'yellow'

    lateral_y = float(vy[idx])
    if abs(lateral_y) >= float(max(0.0, min_lateral_m)):
        return 'blue' if lateral_y >= 0.0 else 'yellow'
    return 'blue' if lateral_y >= 0.0 else 'yellow'


def _nearest_neighbor_distance(
    *,
    idx: int,
    vx: np.ndarray,
    vy: np.ndarray,
    candidate_idx: np.ndarray,
) -> float:
    if candidate_idx.size == 0:
        return float('inf')
    dx = vx[candidate_idx] - float(vx[idx])
    dy = vy[candidate_idx] - float(vy[idx])
    return float(np.min(np.hypot(dx, dy)))


def clamp_camera_range(camera_range_m: float) -> float:
    return max(0.0, min(20.0, float(camera_range_m)))


def near_band_limit(camera_range_m: float) -> float:
    return 20.0 - clamp_camera_range(camera_range_m)


def choose_position_source(
    *,
    range_m: float,
    camera_range_m: float,
    has_lidar_position: bool,
    has_camera_position: bool,
    prefer_lidar_if_camera_missing_far: bool,
    allow_camera_fallback_near: bool,
) -> Optional[str]:
    """Choose position source according to enforced near/far split policy."""

    near_limit = near_band_limit(camera_range_m)
    if range_m <= near_limit:
        if has_lidar_position:
            return 'lidar'
        if allow_camera_fallback_near and has_camera_position:
            return 'camera'
        return None

    if has_camera_position:
        return 'camera'
    if prefer_lidar_if_camera_missing_far and has_lidar_position:
        return 'lidar'
    return None


def update_class_probs(
    probs: list[float],
    *,
    label: Optional[str],
    confidence: float,
    decay: float = 0.0,
    gain: float = 1.0,
) -> list[float]:
    """Simple Bayesian-ish running update from camera label observations."""

    if not probs or len(probs) != len(CLASS_NAMES):
        probs = [1.0, 0.0, 0.0, 0.0]

    decay = max(0.0, min(0.99, float(decay)))
    gain = max(0.0, float(gain))
    if decay > 0.0:
        probs = [max(0.0, p * (1.0 - decay)) for p in probs]

    if label is None:
        total = sum(probs)
        if total <= 1e-9:
            return [1.0, 0.0, 0.0, 0.0]
        return [p / total for p in probs]

    normalized = normalize_color(label)
    idx = CLASS_TO_INDEX.get(normalized, 0)
    conf = max(0.0, min(1.0, float(confidence)))
    obs_strength = gain * (0.10 + 0.90 * conf)

    one_hot = [0.0] * len(CLASS_NAMES)
    one_hot[idx] = 1.0
    for i in range(len(probs)):
        probs[i] = (1.0 - obs_strength) * probs[i] + obs_strength * one_hot[i]

    total = sum(probs)
    if total <= 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    return [p / total for p in probs]


def class_from_probs(probs: list[float]) -> tuple[str, float]:
    if not probs or len(probs) != len(CLASS_NAMES):
        return 'unknown', 0.0
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return CLASS_NAMES[best_idx], float(max(0.0, min(1.0, probs[best_idx])))


def update_color_belief(
    scores: list[float],
    *,
    label: Optional[str],
    confidence: float,
    current_label: str,
    color_decay: float,
    color_switch_margin: float,
) -> tuple[list[float], str, float]:
    """Update per-class scores with decay and hysteresis."""

    updated = update_class_probs(
        scores,
        label=label,
        confidence=confidence,
        decay=color_decay,
        gain=1.0,
    )
    proposed_label, proposed_conf = class_from_probs(updated)
    current_norm = normalize_color(current_label)
    if current_norm == 'unknown':
        return updated, proposed_label, proposed_conf

    current_idx = CLASS_TO_INDEX.get(current_norm, 0)
    proposed_idx = CLASS_TO_INDEX.get(proposed_label, 0)
    if proposed_idx == current_idx:
        return updated, current_norm, proposed_conf

    current_score = updated[current_idx] if 0 <= current_idx < len(updated) else 0.0
    proposed_score = updated[proposed_idx] if 0 <= proposed_idx < len(updated) else 0.0
    margin = max(0.0, float(color_switch_margin))
    if proposed_score < (current_score + margin):
        return updated, current_norm, current_score
    return updated, proposed_label, proposed_conf


def blend_track_confidence(
    current_confidence: float,
    *,
    observation_confidence: float,
    track_confidence_gain: float,
    track_confidence_decay: float,
    seen: bool,
) -> float:
    """Bounded running confidence update for track validity."""

    current = max(0.0, min(1.0, float(current_confidence)))
    gain = max(0.0, float(track_confidence_gain))
    decay = max(0.0, min(1.0, float(track_confidence_decay)))
    if not seen:
        return max(0.0, current * (1.0 - decay))

    obs = max(0.0, min(1.0, float(observation_confidence)))
    reinforced = current + (gain * (0.25 + (0.75 * obs)))
    return max(0.0, min(1.0, reinforced))


def age_to_decay(age_sec: float, time_constant_sec: float) -> float:
    if time_constant_sec <= 1e-6 or age_sec <= 0.0:
        return 0.0
    return max(0.0, min(0.99, 1.0 - math.exp(-float(age_sec) / float(time_constant_sec))))
