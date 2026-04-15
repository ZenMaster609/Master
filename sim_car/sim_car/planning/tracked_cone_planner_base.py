from __future__ import annotations

from typing import Optional

import numpy as np
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.controller_config import build_steering_controller
from sim_car.planning.midline_memory import (
    CommittedMidlineMemory,
    MidlineCandidate,
    MidlineMemoryConfig,
)
from sim_car.planning.planner_runtime_types import TrackedConePlanningFrame, TrackedConePlanningMetadata
from sim_car.planning.tracked_cone_planner_runtime import TrackedConePlannerRuntime

MSG_TRACK_STATE_TENTATIVE = int(getattr(ConeDetection, "TRACK_STATE_TENTATIVE", 0))
MSG_TRACK_STATE_CONFIRMED = int(getattr(ConeDetection, "TRACK_STATE_CONFIRMED", 1))
MSG_TRACK_STATE_STALE = int(getattr(ConeDetection, "TRACK_STATE_STALE", 2))


class TrackedConePlannerBase(TrackedConePlannerRuntime):
    """Shared tracked-cone planner runtime used by midpoint/single-boundary planners."""

    def _build_steering_controller(self):
        return build_steering_controller(
            node=self,
            controller_type=self.controller_type,
            publish_rate_hz=self.publish_rate_hz,
        )

    @staticmethod
    def _extract_cone_metadata(msg: ConeDetectionArray) -> TrackedConePlanningMetadata:
        track_ids: list[int] = []
        track_states: list[int] = []
        track_confidences: list[float] = []
        for idx, cone in enumerate(msg.cones):
            track_id = int(getattr(cone, "track_id", 0))
            track_state = int(getattr(cone, "track_state", MSG_TRACK_STATE_CONFIRMED))
            track_conf = float(getattr(cone, "track_confidence", 0.0))
            track_ids.append(track_id if track_id > 0 else idx)
            track_states.append(track_state)
            track_confidences.append(max(0.0, min(1.0, track_conf)))
        return TrackedConePlanningMetadata(
            track_ids=np.asarray(track_ids, dtype=np.int64),
            track_states=np.asarray(track_states, dtype=np.int64),
            track_confidences=np.asarray(track_confidences, dtype=np.float64),
        )

    def _tentative_cone_is_usable_for_planning(
        self,
        *,
        raw_color: str,
        boundary_color: str,
    ) -> bool:
        del raw_color
        del boundary_color
        return False

    def _planner_confidences_for_msg(
        self,
        *,
        msg: ConeDetectionArray,
        confidences: np.ndarray,
        metadata: TrackedConePlanningMetadata,
    ) -> np.ndarray:
        planner_confidences = np.asarray(confidences, dtype=np.float64).copy()
        valid_track_conf_mask = metadata.track_confidences > 1e-6
        planner_confidences[valid_track_conf_mask] = metadata.track_confidences[valid_track_conf_mask]
        tentative_keep_mask = np.asarray(
            [
                self._tentative_cone_is_usable_for_planning(
                    raw_color=getattr(cone, "color", ""),
                    boundary_color=getattr(cone, "boundary_color", ""),
                )
                for cone in msg.cones
            ],
            dtype=bool,
        )
        planner_confidences[
            (metadata.track_ids > 0)
            & (metadata.track_states == MSG_TRACK_STATE_TENTATIVE)
            & (~tentative_keep_mask)
        ] = 0.0
        return planner_confidences

    def _tracked_cone_planning_frame(
        self,
        *,
        msg: ConeDetectionArray,
        points_xy: np.ndarray,
        colors: list[str],
        confidences: np.ndarray,
    ) -> TrackedConePlanningFrame:
        metadata = self._extract_cone_metadata(msg)
        raw_colors = [normalize_color(getattr(cone, "color", "")) for cone in msg.cones]
        boundary_hints = [str(getattr(cone, "boundary_color", "")).strip().lower() for cone in msg.cones]
        return TrackedConePlanningFrame(
            points_xy=np.asarray(points_xy, dtype=np.float64),
            colors=list(colors),
            raw_confidences=np.asarray(confidences, dtype=np.float64),
            planner_confidences=self._planner_confidences_for_msg(
                msg=msg,
                confidences=confidences,
                metadata=metadata,
            ),
            raw_colors=raw_colors,
            boundary_hints=boundary_hints,
            metadata=metadata,
        )

    def _clear_midline_buffer(self) -> None:
        self._midline_buffer_path = None
        self._midline_buffer_confidence = 0.0
        self._midline_buffer_last_update_sec = -1.0
        memory = getattr(self, "_midline_memory", None)
        if memory is not None:
            memory.clear()

    def _update_candidate_jump_reject_streak(
        self,
        *,
        candidate_update_ok: bool,
        candidate_update_reason: str,
    ) -> tuple[bool, str]:
        if not candidate_update_ok and candidate_update_reason == "candidate_jump_rejected":
            self._candidate_jump_reject_streak += 1
            if self._candidate_jump_reject_streak >= self.candidate_jump_recover_frames:
                self._clear_midline_buffer()
                return True, "candidate_jump_recovery"
        else:
            self._candidate_jump_reject_streak = 0
        return candidate_update_ok, candidate_update_reason

    def _midline_memory_config(self) -> MidlineMemoryConfig:
        near_shift = float(getattr(self, "midline_near_max_shift_m", 0.07))
        mid_shift = float(getattr(self, "midline_mid_max_shift_m", 0.18))
        far_shift = float(getattr(self, "midline_far_max_shift_m", 0.35))
        return MidlineMemoryConfig(
            horizon_m=max(1.0, float(getattr(self, "midline_horizon_m", 30.0))),
            station_spacing_m=max(0.05, float(getattr(self, "midline_station_spacing_m", 0.5))),
            near_distance_m=max(0.0, float(getattr(self, "midline_near_distance_m", 4.0))),
            mid_distance_m=max(
                float(getattr(self, "midline_near_distance_m", 4.0)),
                float(getattr(self, "midline_mid_distance_m", 12.0)),
            ),
            near_alpha=float(np.clip(float(getattr(self, "midline_near_alpha", 0.04)), 0.0, 1.0)),
            mid_alpha=float(np.clip(float(getattr(self, "midline_mid_alpha", 0.12)), 0.0, 1.0)),
            far_alpha=float(np.clip(float(getattr(self, "midline_far_alpha", 0.30)), 0.0, 1.0)),
            near_max_lateral_shift_m=max(0.0, near_shift),
            mid_max_lateral_shift_m=max(near_shift, mid_shift),
            far_max_lateral_shift_m=max(mid_shift, far_shift),
            min_buffer_confidence=float(
                np.clip(float(getattr(self, "midline_min_buffer_confidence", 0.2)), 0.0, 1.0)
            ),
            hold_last_valid_duration_s=max(
                0.0,
                float(getattr(self, "midline_hold_last_valid_duration_s", getattr(self, "hold_last_valid_s", 3.0))),
            ),
            candidate_min_points=max(2, int(getattr(self, "candidate_min_points", 2))),
            candidate_min_extent_m=max(0.5, float(getattr(self, "candidate_min_extent_m", 1.0))),
            candidate_jump_reject_threshold_m=max(
                0.0,
                float(getattr(self, "candidate_jump_reject_threshold_m", 0.45)),
            ),
            candidate_jump_recover_frames=max(1, int(getattr(self, "candidate_jump_recover_frames", 3))),
            jump_check_horizon_m=max(0.5, float(getattr(self, "centerline_jump_horizon_m", 8.0))),
            min_estimated_extent_m=max(
                0.5,
                float(getattr(self, "midline_min_estimated_extent_m", 6.0)),
            ),
            max_estimation_extension_m=max(
                0.0,
                float(getattr(self, "midline_max_estimation_extension_m", 4.0)),
            ),
            max_estimation_join_lateral_m=max(
                0.0,
                float(getattr(self, "midline_max_estimation_join_lateral_m", 0.5)),
            ),
            max_estimation_join_heading_rad=max(
                0.0,
                float(getattr(self, "midline_max_estimation_join_heading_rad", 0.45)),
            ),
            allow_tangent_estimate_without_memory=bool(
                getattr(self, "midline_allow_tangent_estimate_without_memory", True)
            ),
            max_tangent_estimation_extension_m=max(
                0.0,
                float(getattr(self, "midline_max_tangent_estimation_extension_m", 2.0)),
            ),
        )

    def _ensure_midline_memory(self) -> CommittedMidlineMemory:
        config = self._midline_memory_config()
        memory = getattr(self, "_midline_memory", None)
        if memory is None:
            memory = CommittedMidlineMemory(config)
            existing_path = getattr(self, "_midline_buffer_path", None)
            if existing_path is not None and np.asarray(existing_path).shape[0] >= 2:
                memory.path = np.array(existing_path, copy=True)
                memory.confidence = float(getattr(self, "_midline_buffer_confidence", 1.0))
                memory.last_update_sec = float(getattr(self, "_midline_buffer_last_update_sec", -1.0))
            self._midline_memory = memory
        else:
            memory.config = config
        return memory

    def _update_midline_memory_common(
        self,
        *,
        candidate_centerline: np.ndarray,
        candidate_source: str,
        candidate_update_ok: bool | None,
        candidate_update_reason: str,
        frame_id: str,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        now_sec: float,
        support_centerline: np.ndarray | None = None,
        direct_commit: bool = False,
        allow_estimation: bool = False,
    ) -> np.ndarray:
        self._last_midline_update_mode = "hold"
        self._last_midline_candidate_update_ok = False
        self._last_midline_candidate_update_reason = str(candidate_update_reason or "ok")
        self._last_midline_candidate_jump_m = float("nan")
        self._last_midline_near_lateral_delta_max_m = float("nan")
        self._last_midline_buffer_confidence = float(getattr(self, "_midline_buffer_confidence", 0.0))
        self._midline_recovery_count = int(getattr(self, "_midline_recovery_count", 0))
        self._last_midline_estimation_mode = "none"
        self._last_midline_estimated_point_count = 0
        self._last_midline_estimated_extent_m = 0.0
        self._last_midline_live_prefix_extent_m = 0.0
        self._last_midline_estimation_join_lateral_m = float("nan")
        self._last_midline_estimation_join_heading_rad = float("nan")

        candidate_path = np.asarray(candidate_centerline, dtype=np.float64)
        support_path = (
            np.asarray(support_centerline, dtype=np.float64)
            if support_centerline is not None
            else None
        )
        if not self._is_alias(frame_id, self.odom_frame):
            if candidate_path.shape[0] > 0:
                self._midline_buffer_path = np.array(candidate_path, copy=True)
                self._midline_buffer_last_update_sec = now_sec
                self._midline_buffer_confidence = 1.0
                self._last_midline_buffer_confidence = 1.0
                self._last_midline_candidate_update_ok = True
                self._last_midline_update_mode = "direct"
            return candidate_path

        memory = self._ensure_midline_memory()
        updateable = bool(candidate_update_ok) if candidate_update_ok is not None else True
        result = memory.update(
            candidate=MidlineCandidate(
                centerline=candidate_path,
                source=str(candidate_source),
                updateable=updateable,
                update_reason=str(candidate_update_reason or "ok"),
                support_path=support_path,
                direct_commit=bool(direct_commit),
                allow_estimation=bool(allow_estimation),
            ),
            vehicle_xy=(float(vehicle_x), float(vehicle_y)),
            vehicle_yaw=float(vehicle_yaw),
            now_sec=float(now_sec),
        )
        self._midline_buffer_path = (
            np.array(memory.path, copy=True)
            if memory.path is not None and memory.path.shape[0] >= 2
            else None
        )
        self._midline_buffer_last_update_sec = float(memory.last_update_sec)
        self._midline_buffer_confidence = float(memory.confidence)
        self._last_midline_update_mode = str(result.update_mode)
        self._last_midline_candidate_update_ok = bool(result.candidate_accepted)
        self._last_midline_candidate_update_reason = str(result.reason)
        self._last_midline_candidate_jump_m = float(result.candidate_jump_m)
        self._last_midline_near_lateral_delta_max_m = float(
            result.near_field_lateral_delta_max_m
        )
        self._last_midline_buffer_confidence = float(result.buffer_confidence)
        self._midline_recovery_count = int(result.recovery_count)
        self._last_midline_estimation_mode = str(result.estimation_mode)
        self._last_midline_estimated_point_count = int(result.estimated_point_count)
        self._last_midline_estimated_extent_m = float(result.estimated_extent_m)
        self._last_midline_live_prefix_extent_m = float(result.live_prefix_extent_m)
        self._last_midline_estimation_join_lateral_m = float(result.estimation_join_lateral_m)
        self._last_midline_estimation_join_heading_rad = float(result.estimation_join_heading_rad)
        return np.array(result.centerline, copy=True)

    def _midline_estimation_metrics_for_diagnostics(self) -> dict[str, object]:
        return {
            "midline_estimation_mode": getattr(self, "_last_midline_estimation_mode", "none"),
            "midline_estimated_point_count": int(
                getattr(self, "_last_midline_estimated_point_count", 0)
            ),
            "midline_estimated_extent_m": float(
                getattr(self, "_last_midline_estimated_extent_m", 0.0)
            ),
            "midline_live_prefix_extent_m": float(
                getattr(self, "_last_midline_live_prefix_extent_m", 0.0)
            ),
            "midline_estimation_join_lateral_m": float(
                getattr(self, "_last_midline_estimation_join_lateral_m", float("nan"))
            ),
            "midline_estimation_join_heading_rad": float(
                getattr(self, "_last_midline_estimation_join_heading_rad", float("nan"))
            ),
        }
