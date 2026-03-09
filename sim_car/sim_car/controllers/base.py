"""Common steering controller interfaces used by planner nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ControllerOutput:
    """Result from a steering controller step."""

    steering_rad: float
    kappa: float
    lookahead_m: float
    target_point_base: FloatArray
    stanley_debug: Optional['StanleyDebugInfo'] = None


@dataclass(frozen=True)
class StanleyDebugInfo:
    """Stanley-specific debug values captured during one control step."""

    heading_error_rad: float
    cross_track_error_m: float
    vehicle_speed_mps: float
    speed_term_mps: float
    heading_contribution_rad: float
    cross_track_contribution_rad: float
    yaw_rate_damping_contribution_rad: float
    raw_steering_cmd_rad: float
    steering_after_clamp_rad: float
    steering_after_filter_rad: float
    steering_after_rate_limit_rad: float
    final_steering_cmd_rad: float
    steering_saturated_flag: bool
    nearest_path_index: int
    heading_path_index: int
    target_point_x_base_m: float
    target_point_y_base_m: float


class SteeringController(Protocol):
    """Minimal steering controller API."""

    def compute(
        self,
        control_path: FloatArray,
        speed_mps: float,
        yaw_rate_rps: float,
    ) -> ControllerOutput:
        """Return steering command and control diagnostics for the current path."""
