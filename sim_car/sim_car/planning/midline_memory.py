from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from sim_car.planning.path_stability import (
    extract_forward_path_from_pose,
    path_cumulative_lengths,
    project_point_to_path_s,
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
        support_forward = (
            self._candidate_forward(
                candidate_path=_as_path(candidate.support_path),
                vehicle_xy=vehicle_xy,
            )
            if candidate.support_path is not None
            else np.empty((0, 2), dtype=np.float64)
        )

        if candidate_valid and bool(candidate.direct_commit):
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
                reason=candidate.update_reason or "ok",
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
            if bool(candidate.allow_estimation):
                candidate_forward, estimation = self._estimate_short_candidate(
                    candidate_forward=candidate_forward,
                    stored_forward=None if self._stored_expired(now_sec) else stored_forward,
                    support_forward=support_forward,
                    vehicle_xy=vehicle_xy,
                    vehicle_yaw=vehicle_yaw,
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
                reason=candidate.update_reason or "ok",
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
            if bool(candidate.allow_estimation):
                candidate_forward, estimation = self._estimate_short_candidate(
                    candidate_forward=candidate_forward,
                    stored_forward=stored_forward,
                    support_forward=support_forward,
                    vehicle_xy=vehicle_xy,
                    vehicle_yaw=vehicle_yaw,
                )
            candidate_samples = self._resample_stations(candidate_forward)
            if stored_forward is None or stored_forward.shape[0] < 2:
                updated = candidate_samples
                mode = "seed"
            elif str(estimation["mode"]) == "tangent_tail":
                updated = candidate_samples
                mode = "estimate"
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
                reason=candidate.update_reason or "ok",
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
                mode="hold",
                estimated_point_count=int(held.shape[0]),
                estimated_extent_m=_path_extent(held),
            ),
        )

    def _estimate_short_candidate(
        self,
        *,
        candidate_forward: np.ndarray,
        stored_forward: np.ndarray | None,
        support_forward: np.ndarray,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
        prefer_stored_tail: bool = True,
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        candidate_forward = _as_path(candidate_forward)
        live_extent = _path_extent(candidate_forward)
        target_extent = min(
            float(self.config.min_estimated_extent_m),
            live_extent + float(self.config.max_estimation_extension_m),
            float(self.config.horizon_m),
        )
        if (
            candidate_forward.shape[0] < 2
            or live_extent >= target_extent - 1e-9
            or live_extent >= float(self.config.min_estimated_extent_m) - 1e-9
        ):
            return np.array(candidate_forward, copy=True), _estimation_metrics(
                mode="none",
                live_prefix_extent_m=live_extent,
            )

        if prefer_stored_tail and stored_forward is not None and stored_forward.shape[0] >= 2:
            joined, join_metrics = self._append_stored_tail(
                candidate_forward=candidate_forward,
                stored_forward=stored_forward,
                target_extent_m=target_extent,
            )
            if joined is not None and _path_extent(joined) > live_extent + 1e-6:
                return joined, _estimation_metrics(
                    mode="stored_tail",
                    estimated_point_count=max(0, int(joined.shape[0] - candidate_forward.shape[0])),
                    estimated_extent_m=max(0.0, _path_extent(joined) - live_extent),
                    live_prefix_extent_m=live_extent,
                    join_lateral_m=join_metrics["join_lateral_m"],
                    join_heading_rad=join_metrics["join_heading_rad"],
                )

        tangent = self._append_tangent_tail(
            candidate_forward=candidate_forward,
            support_forward=support_forward,
            target_extent_m=target_extent,
            vehicle_xy=vehicle_xy,
            vehicle_yaw=vehicle_yaw,
        )
        if tangent is not None and _path_extent(tangent) > live_extent + 1e-6:
            return tangent, _estimation_metrics(
                mode="tangent_tail",
                estimated_point_count=max(0, int(tangent.shape[0] - candidate_forward.shape[0])),
                estimated_extent_m=max(0.0, _path_extent(tangent) - live_extent),
                live_prefix_extent_m=live_extent,
            )
        return np.array(candidate_forward, copy=True), _estimation_metrics(
            mode="none",
            live_prefix_extent_m=live_extent,
        )

    def _append_stored_tail(
        self,
        *,
        candidate_forward: np.ndarray,
        stored_forward: np.ndarray,
        target_extent_m: float,
    ) -> tuple[np.ndarray | None, dict[str, float]]:
        live_extent = _path_extent(candidate_forward)
        if stored_forward.shape[0] < 2 or live_extent >= float(target_extent_m):
            return None, _join_metrics()
        stored_cum = path_cumulative_lengths(stored_forward)
        stored_total = float(stored_cum[-1])
        if stored_total <= 1e-6:
            return None, _join_metrics()
        live_end = np.asarray(candidate_forward[-1], dtype=np.float64)
        join_s = project_point_to_path_s(
            stored_forward,
            stored_cum,
            live_end,
        )
        join_point = sample_path_at_lengths(
            stored_forward,
            stored_cum,
            np.asarray([join_s], dtype=np.float64),
        )[0]
        live_tangent = _tangent_at(candidate_forward, candidate_forward.shape[0] - 1)
        stored_tangent = _tangent_at_s(stored_forward, stored_cum, join_s)
        join_delta = join_point - live_end
        live_normal = np.array([-live_tangent[1], live_tangent[0]], dtype=np.float64)
        join_lateral = abs(float(np.dot(join_delta, live_normal)))
        join_heading = _heading_delta(live_tangent, stored_tangent)
        metrics = _join_metrics(
            join_lateral_m=join_lateral,
            join_heading_rad=join_heading,
        )
        if join_lateral > float(self.config.max_estimation_join_lateral_m):
            return None, metrics
        if join_heading > float(self.config.max_estimation_join_heading_rad):
            return None, metrics

        needed_tail_m = max(0.0, float(target_extent_m) - live_extent)
        final_s = min(stored_total, join_s + needed_tail_m)
        if final_s <= join_s + 1e-6:
            return None, metrics
        step = max(0.05, float(self.config.station_spacing_m))
        samples = np.arange(join_s, final_s + 1e-9, step, dtype=np.float64)
        if samples.size == 0 or abs(float(samples[0]) - join_s) > 1e-9:
            samples = np.concatenate(([join_s], samples))
        if samples[-1] < final_s:
            samples = np.concatenate((samples, [final_s]))
        tail = sample_path_at_lengths(stored_forward, stored_cum, samples)
        if tail.shape[0] == 0:
            return None, metrics
        if float(np.hypot(*(tail[0] - live_end))) <= step * 0.5:
            tail = tail[1:]
        if tail.shape[0] == 0:
            return None, metrics
        return np.vstack((candidate_forward, tail)), metrics

    def _append_tangent_tail(
        self,
        *,
        candidate_forward: np.ndarray,
        support_forward: np.ndarray,
        target_extent_m: float,
        vehicle_xy: tuple[float, float],
        vehicle_yaw: float,
    ) -> np.ndarray | None:
        if not bool(self.config.allow_tangent_estimate_without_memory):
            return None
        live_extent = _path_extent(candidate_forward)
        extension_m = min(
            max(0.0, float(target_extent_m) - live_extent),
            float(self.config.max_tangent_estimation_extension_m),
            float(self.config.max_estimation_extension_m),
        )
        if candidate_forward.shape[0] < 2 or extension_m <= 1e-6:
            return None
        direction = _tangent_at(candidate_forward, candidate_forward.shape[0] - 1)
        support_direction = _support_tail_direction(
            candidate_end=candidate_forward[-1],
            support_forward=support_forward,
        )
        if support_direction is not None:
            direction = direction + support_direction
            norm = float(np.hypot(direction[0], direction[1]))
            if norm > 1e-9:
                direction = direction / norm
        if _direction_local_x(direction, vehicle_yaw=vehicle_yaw) <= 1e-6:
            direction = np.array([math.cos(float(vehicle_yaw)), math.sin(float(vehicle_yaw))], dtype=np.float64)
        step = max(0.05, float(self.config.station_spacing_m))
        samples = np.arange(step, extension_m + 1e-9, step, dtype=np.float64)
        if samples.size == 0 or samples[-1] < extension_m:
            samples = np.concatenate((samples, [extension_m]))
        tail = candidate_forward[-1][None, :] + (samples[:, None] * direction[None, :])
        return np.vstack((candidate_forward, tail))

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


def _tangent_at_s(path: np.ndarray, cumulative: np.ndarray, station_m: float) -> np.ndarray:
    if path.shape[0] < 2 or cumulative.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    total = float(cumulative[-1])
    eps = max(0.05, min(0.25, total * 0.05))
    s0 = max(0.0, float(station_m) - eps)
    s1 = min(total, float(station_m) + eps)
    if s1 <= s0 + 1e-9:
        return _tangent_at(path, min(path.shape[0] - 1, int(np.searchsorted(cumulative, station_m))))
    points = sample_path_at_lengths(
        path,
        cumulative,
        np.asarray([s0, s1], dtype=np.float64),
    )
    tangent = points[1] - points[0]
    norm = float(np.hypot(tangent[0], tangent[1]))
    if norm <= 1e-9 or not math.isfinite(norm):
        return np.array([1.0, 0.0], dtype=np.float64)
    return np.asarray(tangent / norm, dtype=np.float64)


def _heading_delta(a: np.ndarray, b: np.ndarray) -> float:
    yaw_a = math.atan2(float(a[1]), float(a[0]))
    yaw_b = math.atan2(float(b[1]), float(b[0]))
    return abs(float(math.atan2(math.sin(yaw_b - yaw_a), math.cos(yaw_b - yaw_a))))


def _direction_local_x(direction: np.ndarray, *, vehicle_yaw: float) -> float:
    cos_y = math.cos(float(vehicle_yaw))
    sin_y = math.sin(float(vehicle_yaw))
    return float((cos_y * float(direction[0])) + (sin_y * float(direction[1])))


def _support_tail_direction(
    *,
    candidate_end: np.ndarray,
    support_forward: np.ndarray,
) -> np.ndarray | None:
    support_forward = _as_path(support_forward)
    if support_forward.shape[0] < 2:
        return None
    cumulative = path_cumulative_lengths(support_forward)
    total = float(cumulative[-1])
    if total <= 1e-6:
        return None
    join_s = project_point_to_path_s(
        support_forward,
        cumulative,
        np.asarray(candidate_end, dtype=np.float64),
    )
    if join_s >= total - 1e-6:
        return None
    direction = _tangent_at_s(support_forward, cumulative, join_s)
    norm = float(np.hypot(direction[0], direction[1]))
    if norm <= 1e-9 or not math.isfinite(norm):
        return None
    return np.asarray(direction / norm, dtype=np.float64)
