"""Shared types for the cone memory node."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SensorDetection:
    x_odom: float
    y_odom: float
    z_odom: float
    x_base: float
    y_base: float
    z_base: float
    range_m: float
    color: str
    confidence: float
