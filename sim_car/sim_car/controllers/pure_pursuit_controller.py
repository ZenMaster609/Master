"""Pure-pursuit steering controller used by the Delaunay planner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from sim_car.controllers.base import ControllerOutput, FloatArray


@dataclass(frozen=True)
class PurePursuitConfig:
    """Pure-pursuit steering configuration."""

    lookahead_min_m: float = 3.0
    lookahead_gain: float = 0.35
    steering_limit_rad: float = 0.52
    steering_lowpass_alpha: float = 0.6
    steering_rate_limit_rad_s: float = 10.0
    yaw_rate_damping_gain: float = 0.0
    cross_track_deadband_m: float = 0.0
    feedback_k_cte_rad_per_m: float = 0.04
    feedback_k_heading: float = 0.35
    wheelbase_m: float = 1.65


class PurePursuitController:
    """Pure-pursuit controller with optional feedback and steering filtering."""

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
        self._validate_path(control_path)

        speed = max(0.0, float(speed_mps))
        lookahead = max(
            self._config.lookahead_min_m,
            self._config.lookahead_min_m + (self._config.lookahead_gain * speed),
        )

        target, tangent_yaw = self._lookahead_target(control_path, lookahead)
        target_y = float(target[1])
        if abs(target_y) < self._config.cross_track_deadband_m:
            target_y = 0.0

        target_distance = max(float(np.hypot(target[0], target[1])), 1e-3)
        kappa = 2.0 * target_y / max(target_distance * target_distance, 1e-6)
        steering_ff = math.atan(self._config.wheelbase_m * kappa)
        steering_fb = (
            self._config.feedback_k_cte_rad_per_m * target_y
        ) + (self._config.feedback_k_heading * tangent_yaw)
        steering = steering_ff + steering_fb - (self._config.yaw_rate_damping_gain * float(yaw_rate_rps))
        steering = float(np.clip(steering, -self._config.steering_limit_rad, self._config.steering_limit_rad))

        if self._last_steering_cmd is not None:
            alpha = self._config.steering_lowpass_alpha
            previous = float(self._last_steering_cmd)
            steering = (alpha * steering) + ((1.0 - alpha) * previous)
            if self._config.steering_rate_limit_rad_s > 0.0:
                max_step = self._config.steering_rate_limit_rad_s / self._publish_rate_hz
                steering = float(np.clip(steering, previous - max_step, previous + max_step))

        self._last_steering_cmd = steering
        return ControllerOutput(
            steering_rad=float(steering),
            kappa=float(kappa),
            lookahead_m=float(lookahead),
            target_point_base=np.asarray(target, dtype=np.float64),
        )

    @staticmethod
    def _validate_path(control_path: FloatArray) -> None:
        if control_path.ndim != 2 or control_path.shape[1] != 2:
            raise ValueError('control_path must have shape (N, 2)')
        if control_path.shape[0] == 0:
            raise ValueError('control_path cannot be empty')

    @staticmethod
    def _lookahead_target(control_path: FloatArray, lookahead: float) -> tuple[np.ndarray, float]:
        if control_path.shape[0] <= 1:
            return np.asarray(control_path[0], dtype=np.float64), 0.0

        segment_lengths = np.hypot(np.diff(control_path[:, 0]), np.diff(control_path[:, 1]))
        arc_lengths = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths)))
        idx = int(np.searchsorted(arc_lengths, lookahead, side='left'))
        idx = min(max(idx, 0), control_path.shape[0] - 1)
        if idx == 0:
            dx = float(control_path[1, 0] - control_path[0, 0]) if control_path.shape[0] > 1 else 1.0
            dy = float(control_path[1, 1] - control_path[0, 1]) if control_path.shape[0] > 1 else 0.0
            return np.asarray(control_path[0], dtype=np.float64), float(math.atan2(dy, dx))

        a0 = float(arc_lengths[idx - 1])
        a1 = float(arc_lengths[idx])
        if a1 <= a0 + 1e-9:
            prev_idx = max(0, idx - 1)
            dx = float(control_path[idx, 0] - control_path[prev_idx, 0])
            dy = float(control_path[idx, 1] - control_path[prev_idx, 1])
            return np.asarray(control_path[idx], dtype=np.float64), float(math.atan2(dy, dx))

        ratio = float(np.clip((lookahead - a0) / (a1 - a0), 0.0, 1.0))
        target = (1.0 - ratio) * control_path[idx - 1] + ratio * control_path[idx]
        dx = float(control_path[idx, 0] - control_path[idx - 1, 0])
        dy = float(control_path[idx, 1] - control_path[idx - 1, 1])
        tangent_yaw = float(math.atan2(dy, dx))
        return np.asarray(target, dtype=np.float64), tangent_yaw
