"""Fusion helpers for local cone memory tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


CLASS_NAMES = ("unknown", "blue", "yellow", "orange")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class FusedObservation:
    """A fused per-cone observation from lidar/camera detections."""

    assoc_x: float
    assoc_y: float
    update_x: Optional[float]
    update_y: Optional[float]
    update_z: Optional[float]
    update_source: Optional[str]
    has_lidar: bool
    has_camera: bool
    camera_label: Optional[str]
    camera_confidence: float
    range_m: float


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
) -> list[float]:
    """Simple Bayesian-ish running update from camera label observations."""

    if not probs or len(probs) != len(CLASS_NAMES):
        probs = [1.0, 0.0, 0.0, 0.0]

    if label is None:
        return probs

    normalized = normalize_color(label)
    idx = CLASS_TO_INDEX.get(normalized, 0)
    conf = max(0.0, min(1.0, float(confidence)))
    obs_strength = 0.10 + 0.90 * conf

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
