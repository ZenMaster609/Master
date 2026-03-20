"""Stanley steering controller used by the Delaunay planner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from sim_car.controllers.base import ControllerOutput, FloatArray, StanleyDebugInfo


@dataclass(frozen=True)
class StanleyConfig:
    """Stanley steering configuration."""

    k_gain: float = 1.2
    softening_speed_mps: float = 0.0
    heading_gain: float = 1.0
    lookahead_idx_offset: int = 0
    steering_limit_rad: float = 0.52
    steering_lowpass_alpha: float = 1.0
    steering_rate_limit_rad_s: float = 10.0
    use_yaw_rate_damping: bool = True
    yaw_rate_damping_gain: float = 0.0
    wheelbase_m: float = 1.65
    cross_track_deadband_m: float = 0.0


class StanleyController:
    """Stanley controller with optional steering filtering and yaw-rate damping."""

    def __init__(self, *, config: StanleyConfig, publish_rate_hz: float) -> None:
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

        target_point, nearest_segment_idx = self._nearest_projection(control_path)
        heading_segment_idx = int(
            np.clip(
                nearest_segment_idx + self._config.lookahead_idx_offset,
                0,
                control_path.shape[0] - 2,
            )
        )

        seg_vec = control_path[heading_segment_idx + 1] - control_path[heading_segment_idx]
        if float(np.hypot(seg_vec[0], seg_vec[1])) <= 1e-9:
            path_heading = 0.0
        else:
            path_heading = float(math.atan2(float(seg_vec[1]), float(seg_vec[0])))
        heading_error = self._normalize_angle(path_heading)

        cross_track_error = float(target_point[1])
        if abs(cross_track_error) < self._config.cross_track_deadband_m:
            cross_track_error = 0.0

        vehicle_speed = float(speed_mps)
        speed_term = max(1e-6, self._config.softening_speed_mps + max(0.0, vehicle_speed))
        cross_track_term = float(math.atan2(self._config.k_gain * cross_track_error, speed_term))
        heading_contribution = self._config.heading_gain * heading_error
        yaw_rate_damping_contribution = 0.0
        if self._config.use_yaw_rate_damping:
            yaw_rate_damping_contribution = -self._config.yaw_rate_damping_gain * float(yaw_rate_rps)

        steering_raw = heading_contribution + cross_track_term + yaw_rate_damping_contribution
        steering_after_clamp = float(
            np.clip(steering_raw, -self._config.steering_limit_rad, self._config.steering_limit_rad)
        )
        steering_after_filter = steering_after_clamp
        steering_after_rate_limit = steering_after_filter
        steering_saturated = bool(abs(steering_after_clamp - steering_raw) > 1e-9)

        if self._last_steering_cmd is not None:
            alpha = self._config.steering_lowpass_alpha
            previous = float(self._last_steering_cmd)
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
        wheelbase = max(1e-3, self._config.wheelbase_m)
        kappa = float(math.tan(steering) / wheelbase)
        lookahead_m = float(np.hypot(target_point[0], target_point[1]))

        return ControllerOutput(
            steering_rad=float(steering),
            kappa=kappa,
            lookahead_m=lookahead_m,
            target_point_base=np.asarray(target_point, dtype=np.float64),
            stanley_debug=StanleyDebugInfo(
                heading_error_rad=float(heading_error),
                cross_track_error_m=float(cross_track_error),
                vehicle_speed_mps=float(vehicle_speed),
                speed_term_mps=float(speed_term),
                heading_contribution_rad=float(heading_contribution),
                cross_track_contribution_rad=float(cross_track_term),
                yaw_rate_damping_contribution_rad=float(yaw_rate_damping_contribution),
                raw_steering_cmd_rad=float(steering_raw),
                steering_after_clamp_rad=float(steering_after_clamp),
                steering_after_filter_rad=float(steering_after_filter),
                steering_after_rate_limit_rad=float(steering_after_rate_limit),
                final_steering_cmd_rad=float(steering),
                steering_saturated_flag=steering_saturated,
                nearest_path_index=int(nearest_segment_idx),
                heading_path_index=int(heading_segment_idx),
                target_point_x_base_m=float(target_point[0]),
                target_point_y_base_m=float(target_point[1]),
            ),
        )

    @staticmethod
    def _validate_path(control_path: FloatArray) -> None:
        if control_path.ndim != 2 or control_path.shape[1] != 2:
            raise ValueError('control_path must have shape (N, 2)')
        if control_path.shape[0] == 0:
            raise ValueError('control_path cannot be empty')

    @staticmethod
    def _normalize_angle(angle_rad: float) -> float:
        return float(math.atan2(math.sin(angle_rad), math.cos(angle_rad)))

    @staticmethod
    def _nearest_projection(control_path: FloatArray) -> tuple[np.ndarray, int]:
        if control_path.shape[0] == 1:
            return np.asarray(control_path[0], dtype=np.float64), 0

        best_distance_sq = float('inf')
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
