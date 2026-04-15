"""Pure pursuit steering controller used by migrated planners."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from sim_car.controllers.base import ControllerOutput, FloatArray
from sim_car.controllers._path_utils import (
    nearest_projection_on_path,
    target_point_from_projection,
    validate_control_path,
)


@dataclass(frozen=True)
class PurePursuitConfig:
    """Pure pursuit steering configuration."""

    lookahead_m: float = 3.0
    min_lookahead_m: float = 1.5
    max_lookahead_m: float = 8.0
    lookahead_gain: float = 0.0
    steering_limit_rad: float = 0.52
    steering_lowpass_alpha: float = 1.0
    steering_rate_limit_rad_s: float = 10.0
    wheelbase_m: float = 1.65


class PurePursuitController:
    """Pure pursuit controller with optional steering filtering."""

    def __init__(self, *, config: PurePursuitConfig, publish_rate_hz: float) -> None:
        self._config = config
        self._publish_rate_hz = max(1.0, float(publish_rate_hz))
        self._last_steering_cmd: Optional[float] = None

    def compute(
        self,
        control_path: FloatArray,
        speed_mps: float,
        yaw_rate_rps: float,
    ) -> ControllerOutput:
        del yaw_rate_rps

        validate_control_path(control_path)

        projected_point, nearest_segment_idx = nearest_projection_on_path(control_path)
        commanded_lookahead = self._compute_commanded_lookahead(float(speed_mps))
        target_point = target_point_from_projection(
            control_path=control_path,
            projected_point=projected_point,
            nearest_segment_idx=nearest_segment_idx,
            lookahead_m=commanded_lookahead,
        )

        target_x = float(target_point[0])
        target_y = float(target_point[1])
        lookahead_sq = (target_x * target_x) + (target_y * target_y)
        if lookahead_sq <= 1e-9:
            raw_curvature = 0.0
        else:
            raw_curvature = (2.0 * target_y) / lookahead_sq

        wheelbase = max(1e-3, self._config.wheelbase_m)
        steering_raw = float(math.atan(wheelbase * raw_curvature))
        steering_after_clamp = float(
            np.clip(steering_raw, -self._config.steering_limit_rad, self._config.steering_limit_rad)
        )
        steering_after_filter = steering_after_clamp
        steering_after_rate_limit = steering_after_filter

        if self._last_steering_cmd is not None:
            previous = float(self._last_steering_cmd)
            alpha = self._config.steering_lowpass_alpha
            steering_after_filter = (alpha * steering_after_clamp) + ((1.0 - alpha) * previous)
            if self._config.steering_rate_limit_rad_s > 0.0:
                max_step = self._config.steering_rate_limit_rad_s / self._publish_rate_hz
                steering_after_rate_limit = float(
                    np.clip(steering_after_filter, previous - max_step, previous + max_step)
                )
            else:
                steering_after_rate_limit = steering_after_filter

        steering = float(steering_after_rate_limit)
        self._last_steering_cmd = steering

        return ControllerOutput(
            steering_rad=steering,
            kappa=float(math.tan(steering) / wheelbase),
            lookahead_m=float(math.hypot(target_x, target_y)),
            target_point_base=np.asarray(target_point, dtype=np.float64),
        )

    def _compute_commanded_lookahead(self, speed_mps: float) -> float:
        speed_term = max(0.0, float(speed_mps))
        raw = self._config.lookahead_m + (self._config.lookahead_gain * speed_term)
        return float(
            np.clip(raw, self._config.min_lookahead_m, self._config.max_lookahead_m)
        )
