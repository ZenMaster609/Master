"""Common steering controller interfaces used by planner nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


class SteeringController(Protocol):
    """Minimal steering controller API."""

    def compute(
        self,
        control_path: FloatArray,
        speed_mps: float,
        yaw_rate_rps: float,
    ) -> ControllerOutput:
        """Return steering command and control diagnostics for the current path."""
