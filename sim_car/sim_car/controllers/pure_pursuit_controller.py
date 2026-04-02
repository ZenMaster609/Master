"""Pure pursuit steering controller used by migrated planners."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from sim_car.controllers.base import ControllerOutput, FloatArray


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

        self._validate_path(control_path)

        projected_point, nearest_segment_idx = self._nearest_projection(control_path)
        commanded_lookahead = self._compute_commanded_lookahead(float(speed_mps))
        target_point = self._target_point_from_projection(
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

    @staticmethod
    def _validate_path(control_path: FloatArray) -> None:
        if control_path.ndim != 2 or control_path.shape[1] != 2:
            raise ValueError("control_path must have shape (N, 2)")
        if control_path.shape[0] == 0:
            raise ValueError("control_path cannot be empty")

    @staticmethod
    def _nearest_projection(control_path: FloatArray) -> tuple[np.ndarray, int]:
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

    @staticmethod
    def _target_point_from_projection(
        *,
        control_path: FloatArray,
        projected_point: np.ndarray,
        nearest_segment_idx: int,
        lookahead_m: float,
    ) -> np.ndarray:
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
