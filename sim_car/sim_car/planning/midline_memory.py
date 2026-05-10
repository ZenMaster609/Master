from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from sim_car.planning.tracked_cone_planner_geometry import (
    extract_forward_path_from_pose,
    path_cumulative_lengths,
    sample_path_at_lengths,
)


@dataclass(frozen=True)
class MidlineMemoryConfig:
    horizon_m: float = 30.0
    station_spacing_m: float = 0.5
    near_distance_m: float = 4.0
    mid_distance_m: float = 12.0
    near_alpha: float = 0.04
    mid_alpha: float = 0.12
    far_alpha: float = 0.30
    near_max_lateral_shift_m: float = 0.07
    mid_max_lateral_shift_m: float = 0.18
    far_max_lateral_shift_m: float = 0.35
    min_buffer_confidence: float = 0.2
    hold_last_valid_duration_s: float = 3.0
    candidate_min_points: int = 2
    candidate_min_extent_m: float = 1.0
    candidate_jump_reject_threshold_m: float = 0.45
    candidate_jump_recover_frames: int = 3
    jump_check_horizon_m: float = 8.0
    recovery_near_horizon_m: float = 3.0
    recovery_near_lateral_max_m: float = 0.25
    min_estimated_extent_m: float = 6.0
    max_estimation_extension_m: float = 4.0
    max_estimation_join_lateral_m: float = 0.5
    max_estimation_join_heading_rad: float = 0.45
    allow_tangent_estimate_without_memory: bool = True
    max_tangent_estimation_extension_m: float = 2.0


@dataclass(frozen=True)
class MidlineCandidate:
    centerline: np.ndarray
    source: str
    updateable: bool
    update_reason: str = "ok"
    support_path: np.ndarray | None = None
    direct_commit: bool = False
    # Compatibility-only. Midline memory no longer estimates or extends
    # planner candidates; planners must provide their own publishable midline.
    allow_estimation: bool = False


@dataclass(frozen=True)
class MidlineUpdateResult:
    centerline: np.ndarray
    candidate_accepted: bool
    update_mode: str
    reason: str
    candidate_jump_m: float
    near_field_lateral_delta_max_m: float
    near_field_lateral_delta_mean_m: float
    buffer_confidence: float
    recovery_count: int
    estimation_mode: str
    estimated_point_count: int
    estimated_extent_m: float
    live_prefix_extent_m: float
    estimation_join_lateral_m: float
    estimation_join_heading_rad: float


class CommittedMidlineMemory:
    """Online committed centerline memory shared by tracked-cone planners."""

    def __init__(self, config: MidlineMemoryConfig) -> None:
        self.config = config
        self.path: np.ndarray | None = None
        self.confidence: float = 0.0
        self.last_update_sec: float = -1.0
        self.rejected_jump_streak: int = 0
        self.recovery_count: int = 0

    def clear(self) -> None:
        self.path = None
        self.confidence = 0.0
        self.last_update_sec = -1.0
        self.rejected_jump_streak = 0

    def update(
        self,
        *,
        candidate: MidlineCandidate,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
        now_sec: float,
    ) -> MidlineUpdateResult:
        cfg = self.config
        candidate_path = _as_path(candidate.centerline)
        candidate_ok, candidate_reason = self._candidate_is_usable(
            candidate_path=candidate_path,
            vehicle_xy=vehicle_xy,
        )
        candidate_valid = bool(candidate.updateable) and candidate_ok
        if not candidate.updateable:
            candidate_reason = candidate.update_reason or candidate_reason
        direct_commit = bool(candidate.direct_commit)
        update_reason = str(candidate.update_reason or "ok")

        stored_forward = self._stored_forward(vehicle_xy=vehicle_xy)
        candidate_forward = (
            self._candidate_forward(
                candidate_path=candidate_path,
                vehicle_xy=vehicle_xy,
            )
            if candidate_ok
            else np.empty((0, 2), dtype=np.float64)
        )
        metrics = self._transition_metrics(
            stored_forward=stored_forward,
            candidate_forward=candidate_forward,
            vehicle_xy=vehicle_xy,
            vehicle_yaw=vehicle_yaw,
        )
        del candidate

        if candidate_valid and direct_commit:
            estimation = _estimation_metrics(
                mode="none",
                live_prefix_extent_m=_path_extent(candidate_forward),
            )
            committed = self._resample_stations(candidate_forward)
            self.path = np.array(committed, copy=True)
            self.confidence = 1.0
            self.last_update_sec = float(now_sec)
            self.rejected_jump_streak = 0
            return self._result(
                centerline=committed,
                accepted=True,
                mode="direct",
                reason=update_reason or "ok",
                metrics=metrics,
                estimation=estimation,
            )

        if candidate_valid and (
            stored_forward is None
            or stored_forward.shape[0] < 2
            or self._stored_expired(now_sec)
        ):
            estimation = _estimation_metrics(
                mode="none",
                live_prefix_extent_m=_path_extent(candidate_forward),
            )
            seeded = self._resample_stations(candidate_forward)
            self.path = np.array(seeded, copy=True)
            self.confidence = 1.0
            self.last_update_sec = float(now_sec)
            self.rejected_jump_streak = 0
            return self._result(
                centerline=seeded,
                accepted=True,
                mode="seed",
                reason=update_reason or "ok",
                metrics=metrics,
                estimation=estimation,
            )

        if (
            candidate_valid
            and metrics["near_lateral_max_m"] > cfg.candidate_jump_reject_threshold_m
        ):
            self.rejected_jump_streak += 1
            if self._candidate_can_recover_from_jump(metrics):
                updated = self._resample_stations(candidate_forward)
                self.path = np.array(updated, copy=True)
                self.confidence = 1.0
                self.last_update_sec = float(now_sec)
                self.rejected_jump_streak = 0
                self.recovery_count += 1
                return self._result(
                    centerline=updated,
                    accepted=True,
                    mode="recovery",
                    reason="candidate_jump_recovery",
                    metrics=metrics,
                )
            held = self._held_path(
                stored_forward=stored_forward,
                now_sec=now_sec,
                reason="candidate_jump_rejected",
                metrics=metrics,
            )
            return held

        if candidate_valid:
            estimation = _estimation_metrics(
                mode="none",
                live_prefix_extent_m=_path_extent(candidate_forward),
            )
            candidate_samples = self._resample_stations(candidate_forward)
            if stored_forward is None or stored_forward.shape[0] < 2:
                updated = candidate_samples
                mode = "seed"
            else:
                stored_samples = self._resample_stations(stored_forward)
                updated = self._blend_samples_path_relative(
                    stored_samples=stored_samples,
                    candidate_samples=candidate_samples,
                    vehicle_xy=vehicle_xy,
                    vehicle_yaw=vehicle_yaw,
                )
                mode = "blend"
            self.path = np.array(updated, copy=True)
            self.confidence = min(1.0, max(self.confidence, 0.0) + 0.25)
            self.last_update_sec = float(now_sec)
            self.rejected_jump_streak = 0
            return self._result(
                centerline=updated,
                accepted=True,
                mode=mode,
                reason=update_reason or "ok",
                metrics=metrics,
                estimation=estimation,
            )

        self.rejected_jump_streak = 0
        return self._held_path(
            stored_forward=stored_forward,
            now_sec=now_sec,
            reason=candidate_reason,
            metrics=metrics,
        )

    def _candidate_is_usable(
        self,
        *,
        candidate_path: np.ndarray,
        vehicle_xy: tuple[float, float],
    ) -> tuple[bool, str]:
        cfg = self.config
        if candidate_path.shape[0] < int(cfg.candidate_min_points):
            return False, "candidate_too_short"
        if not np.all(np.isfinite(candidate_path)):
            return False, "candidate_non_finite"
        forward = self._candidate_forward(candidate_path=candidate_path, vehicle_xy=vehicle_xy)
        if forward.shape[0] < int(cfg.candidate_min_points):
            return False, "candidate_no_forward_path"
        if _path_extent(forward) < float(cfg.candidate_min_extent_m):
            return False, "candidate_extent_too_short"
        return True, "ok"

    def _stored_forward(self, *, vehicle_xy: tuple[float, float]) -> np.ndarray | None:
        if self.path is None or self.path.shape[0] < 2:
            return None
        forward = extract_forward_path_from_pose(
            path=self.path,
            vehicle_xy=vehicle_xy,
            resolution_m=self.config.station_spacing_m,
        )
        if forward is None or forward.shape[0] < 2:
            return np.array(self.path, copy=True)
        return np.asarray(forward, dtype=np.float64)

    def _candidate_forward(
        self,
        *,
        candidate_path: np.ndarray,
        vehicle_xy: tuple[float, float],
    ) -> np.ndarray:
        if candidate_path.shape[0] < 2:
            return np.array(candidate_path, copy=True)
        forward = extract_forward_path_from_pose(
            path=candidate_path,
            vehicle_xy=vehicle_xy,
            resolution_m=self.config.station_spacing_m,
        )
        if forward is None or forward.shape[0] < 2:
            return np.array(candidate_path, copy=True)
        return np.asarray(forward, dtype=np.float64)

    def _resample_stations(self, path: np.ndarray) -> np.ndarray:
        path = _as_path(path)
        if path.shape[0] < 2:
            return np.array(path, copy=True)
        cumulative = path_cumulative_lengths(path)
        total = min(float(cumulative[-1]), float(self.config.horizon_m))
        if total <= 1e-6:
            return np.asarray(path[:1], dtype=np.float64)
        step = max(0.05, float(self.config.station_spacing_m))
        samples = np.arange(0.0, total + 1e-9, step, dtype=np.float64)
        if samples.size == 0 or samples[-1] < total:
            samples = np.concatenate((samples, [total]))
        return sample_path_at_lengths(path, cumulative, samples)

    def _transition_metrics(
        self,
        *,
        stored_forward: np.ndarray | None,
        candidate_forward: np.ndarray,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> dict[str, float]:
        empty = {
            "jump_m": 0.0,
            "near_lateral_max_m": 0.0,
            "near_lateral_mean_m": 0.0,
        }
        if stored_forward is None or stored_forward.shape[0] < 2 or candidate_forward.shape[0] < 2:
            return empty
        stored = self._resample_stations(stored_forward)
        candidate = self._resample_stations(candidate_forward)
        count = min(stored.shape[0], candidate.shape[0])
        if count < 2:
            return empty
        stored = stored[:count]
        candidate = candidate[:count]
        cum = path_cumulative_lengths(candidate)
        jump_horizon = max(
            0.25,
            min(float(self.config.jump_check_horizon_m), float(self.config.horizon_m)),
        )
        jump_limit = max(2, int(np.searchsorted(cum, jump_horizon, side="right")))
        jump_limit = min(jump_limit, count)
        deltas = candidate[:jump_limit] - stored[:jump_limit]
        displacement = np.hypot(deltas[:, 0], deltas[:, 1])
        near_horizon = max(
            0.25,
            min(float(self.config.recovery_near_horizon_m), float(self.config.horizon_m)),
        )
        near_limit = max(2, int(np.searchsorted(cum, near_horizon, side="right")))
        near_limit = min(near_limit, count)
        stored_local = _to_vehicle_frame(stored[:near_limit], vehicle_xy=vehicle_xy, vehicle_yaw=vehicle_yaw)
        candidate_local = _to_vehicle_frame(candidate[:near_limit], vehicle_xy=vehicle_xy, vehicle_yaw=vehicle_yaw)
        lateral = np.abs(candidate_local[:, 1] - stored_local[:, 1])
        return {
            "jump_m": float(np.max(displacement)) if displacement.size else 0.0,
            "near_lateral_max_m": float(np.max(lateral)) if lateral.size else 0.0,
            "near_lateral_mean_m": float(np.mean(lateral)) if lateral.size else 0.0,
        }

    def _candidate_can_recover_from_jump(self, metrics: dict[str, float]) -> bool:
        if self.rejected_jump_streak < max(1, int(self.config.candidate_jump_recover_frames)):
            return False
        return float(metrics["near_lateral_max_m"]) <= float(self.config.recovery_near_lateral_max_m)

    def _stored_expired(self, now_sec: float) -> bool:
        if self.last_update_sec < 0.0:
            return True
        return (float(now_sec) - float(self.last_update_sec)) > float(
            self.config.hold_last_valid_duration_s
        )

    def _held_path(
        self,
        *,
        stored_forward: np.ndarray | None,
        now_sec: float,
        reason: str,
        metrics: dict[str, float],
    ) -> MidlineUpdateResult:
        self.confidence = max(0.0, float(self.confidence) - 0.10)
        if (
            self.path is None
            or self.path.shape[0] < 2
            or self.last_update_sec < 0.0
            or self._stored_expired(now_sec)
            or self.confidence < float(self.config.min_buffer_confidence)
        ):
            if self._stored_expired(now_sec):
                self.clear()
            return self._result(
                centerline=np.empty((0, 2), dtype=np.float64),
                accepted=False,
                mode="reject",
                reason=reason,
                metrics=metrics,
                estimation=_estimation_metrics(mode="none"),
            )
        held = (
            self._resample_stations(stored_forward)
            if stored_forward is not None and stored_forward.shape[0] >= 2
            else np.array(self.path, copy=True)
        )
        return self._result(
            centerline=held,
            accepted=False,
            mode="hold",
            reason=reason,
            metrics=metrics,
            estimation=_estimation_metrics(
                mode="none",
            ),
        )

    def _blend_samples_path_relative(
        self,
        *,
        stored_samples: np.ndarray,
        candidate_samples: np.ndarray,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> np.ndarray:
        if stored_samples.shape[0] == 0:
            return np.array(candidate_samples, copy=True)
        if candidate_samples.shape[0] == 0:
            return np.array(stored_samples, copy=True)
        count = min(stored_samples.shape[0], candidate_samples.shape[0])
        updated = np.empty((count, 2), dtype=np.float64)
        step = max(0.05, float(self.config.station_spacing_m))
        for idx in range(count):
            distance_ahead = float(idx) * step
            alpha, max_shift = self._blend_params(distance_ahead)
            tangent = _tangent_at(candidate_samples[:count], idx)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            stored_to_candidate = candidate_samples[idx] - stored_samples[idx]
            lateral_delta = float(np.dot(stored_to_candidate, normal))
            candidate_longitudinal = candidate_samples[idx] + (
                float(np.dot(stored_samples[idx] - candidate_samples[idx], normal)) * normal
            )
            lateral_step = float(np.clip(alpha * lateral_delta, -max_shift, max_shift))
            updated[idx] = candidate_longitudinal + (lateral_step * normal)

        if count > 0:
            first_local = _to_vehicle_frame(
                updated[:1],
                vehicle_xy=vehicle_xy,
                vehicle_yaw=vehicle_yaw,
            )[0]
            if float(first_local[0]) < -0.1:
                updated[0] = candidate_samples[0]
        if candidate_samples.shape[0] > count:
            updated = np.vstack((updated, candidate_samples[count:]))
        return updated

    def _blend_params(self, distance_ahead: float) -> tuple[float, float]:
        if distance_ahead <= float(self.config.near_distance_m):
            return float(self.config.near_alpha), float(self.config.near_max_lateral_shift_m)
        if distance_ahead <= float(self.config.mid_distance_m):
            return float(self.config.mid_alpha), float(self.config.mid_max_lateral_shift_m)
        return float(self.config.far_alpha), float(self.config.far_max_lateral_shift_m)

    def _result(
        self,
        *,
        centerline: np.ndarray,
        accepted: bool,
        mode: str,
        reason: str,
        metrics: dict[str, float],
        estimation: dict[str, float | str] | None = None,
    ) -> MidlineUpdateResult:
        estimation = _estimation_metrics() if estimation is None else estimation
        return MidlineUpdateResult(
            centerline=np.asarray(centerline, dtype=np.float64),
            candidate_accepted=bool(accepted),
            update_mode=str(mode),
            reason=str(reason),
            candidate_jump_m=float(metrics["jump_m"]),
            near_field_lateral_delta_max_m=float(metrics["near_lateral_max_m"]),
            near_field_lateral_delta_mean_m=float(metrics["near_lateral_mean_m"]),
            buffer_confidence=float(self.confidence),
            recovery_count=int(self.recovery_count),
            estimation_mode=str(estimation["mode"]),
            estimated_point_count=int(estimation["estimated_point_count"]),
            estimated_extent_m=float(estimation["estimated_extent_m"]),
            live_prefix_extent_m=float(estimation["live_prefix_extent_m"]),
            estimation_join_lateral_m=float(estimation["join_lateral_m"]),
            estimation_join_heading_rad=float(estimation["join_heading_rad"]),
        )


def _as_path(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.reshape(arr, (-1, 2))


def _path_extent(path: np.ndarray) -> float:
    if path.shape[0] < 2:
        return 0.0
    cumulative = path_cumulative_lengths(path)
    return float(cumulative[-1])


def _estimation_metrics(
    *,
    mode: str = "none",
    estimated_point_count: int = 0,
    estimated_extent_m: float = 0.0,
    live_prefix_extent_m: float = 0.0,
    join_lateral_m: float = float("nan"),
    join_heading_rad: float = float("nan"),
) -> dict[str, float | str]:
    return {
        "mode": str(mode),
        "estimated_point_count": int(estimated_point_count),
        "estimated_extent_m": float(estimated_extent_m),
        "live_prefix_extent_m": float(live_prefix_extent_m),
        "join_lateral_m": float(join_lateral_m),
        "join_heading_rad": float(join_heading_rad),
    }


def _join_metrics(
    *,
    join_lateral_m: float = float("nan"),
    join_heading_rad: float = float("nan"),
) -> dict[str, float]:
    return {
        "join_lateral_m": float(join_lateral_m),
        "join_heading_rad": float(join_heading_rad),
    }


def _to_vehicle_frame(
    points: np.ndarray,
    *,
    vehicle_xy: tuple[float, float],
    vehicle_yaw: float,
) -> np.ndarray:
    pts = _as_path(points)
    out = np.empty_like(pts)
    cos_y = math.cos(float(vehicle_yaw))
    sin_y = math.sin(float(vehicle_yaw))
    tx, ty = float(vehicle_xy[0]), float(vehicle_xy[1])
    for idx, point in enumerate(pts):
        dx = float(point[0]) - tx
        dy = float(point[1]) - ty
        out[idx, 0] = (cos_y * dx) + (sin_y * dy)
        out[idx, 1] = (-sin_y * dx) + (cos_y * dy)
    return out


def _tangent_at(path: np.ndarray, idx: int) -> np.ndarray:
    if path.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    if idx <= 0:
        tangent = path[1] - path[0]
    elif idx >= path.shape[0] - 1:
        tangent = path[-1] - path[-2]
    else:
        tangent = path[idx + 1] - path[idx - 1]
    norm = float(np.hypot(tangent[0], tangent[1]))
    if norm <= 1e-9 or not math.isfinite(norm):
        return np.array([1.0, 0.0], dtype=np.float64)
    return np.asarray(tangent / norm, dtype=np.float64)
