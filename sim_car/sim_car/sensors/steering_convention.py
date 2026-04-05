"""Shared helpers for steering-angle sign conventions."""

from __future__ import annotations

import math


def steering_joint_mean_to_deg(
    left_rad: float,
    right_rad: float,
    *,
    sign: float = -1.0,
) -> float:
    """Convert steering joint positions to controller-sign steering degrees."""

    joint_mean_rad = 0.5 * (float(left_rad) + float(right_rad))
    steering_sign = -1.0 if float(sign) < 0.0 else 1.0
    return float(steering_sign * math.degrees(joint_mean_rad))
