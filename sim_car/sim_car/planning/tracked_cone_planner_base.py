from __future__ import annotations

from typing import Optional

import numpy as np
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.controllers.factory import create_steering_controller
from sim_car.controllers.pure_pursuit_controller import PurePursuitConfig
from sim_car.controllers.stanley_controller import StanleyConfig
from sim_car.planning.planner_runtime_types import TrackedConePlanningFrame, TrackedConePlanningMetadata
from sim_car.planning.tracked_cone_planner_runtime import TrackedConePlannerRuntime

MSG_TRACK_STATE_TENTATIVE = int(getattr(ConeDetection, "TRACK_STATE_TENTATIVE", 0))
MSG_TRACK_STATE_CONFIRMED = int(getattr(ConeDetection, "TRACK_STATE_CONFIRMED", 1))
MSG_TRACK_STATE_STALE = int(getattr(ConeDetection, "TRACK_STATE_STALE", 2))


class TrackedConePlannerBase(TrackedConePlannerRuntime):
    """Shared tracked-cone planner runtime used by midpoint/single-boundary planners."""

    def _build_steering_controller(self):
        stanley_config = StanleyConfig(
            k_gain=max(0.0, float(self.get_parameter("stanley.k_gain").value)),
            softening_speed_mps=max(0.0, float(self.get_parameter("stanley.softening_speed_mps").value)),
            heading_gain=float(self.get_parameter("stanley.heading_gain").value),
            lookahead_idx_offset=max(0, int(self.get_parameter("stanley.lookahead_idx_offset").value)),
            steering_limit_rad=max(0.01, float(self.get_parameter("stanley.steering_limit_rad").value)),
            steering_lowpass_alpha=float(
                np.clip(float(self.get_parameter("stanley.steering_lowpass_alpha").value), 0.0, 1.0)
            ),
            steering_rate_limit_rad_s=max(
                0.0,
                float(self.get_parameter("stanley.steering_rate_limit_rad_s").value),
            ),
            use_yaw_rate_damping=bool(self.get_parameter("stanley.use_yaw_rate_damping").value),
            yaw_rate_damping_gain=max(
                0.0,
                float(self.get_parameter("stanley.yaw_rate_damping_gain").value),
            ),
            wheelbase_m=max(0.1, float(self.get_parameter("stanley.wheelbase_m").value)),
            cross_track_deadband_m=max(
                0.0,
                float(self.get_parameter("stanley.cross_track_deadband_m").value),
            ),
        )
        pure_pursuit_config = PurePursuitConfig(
            lookahead_m=max(0.0, float(self.get_parameter("pure_pursuit.lookahead_m").value)),
            min_lookahead_m=max(0.01, float(self.get_parameter("pure_pursuit.min_lookahead_m").value)),
            max_lookahead_m=max(
                0.01,
                float(self.get_parameter("pure_pursuit.max_lookahead_m").value),
            ),
            lookahead_gain=max(0.0, float(self.get_parameter("pure_pursuit.lookahead_gain").value)),
            steering_limit_rad=max(
                0.01,
                float(self.get_parameter("pure_pursuit.steering_limit_rad").value),
            ),
            steering_lowpass_alpha=float(
                np.clip(float(self.get_parameter("pure_pursuit.steering_lowpass_alpha").value), 0.0, 1.0)
            ),
            steering_rate_limit_rad_s=max(
                0.0,
                float(self.get_parameter("pure_pursuit.steering_rate_limit_rad_s").value),
            ),
            wheelbase_m=max(0.1, float(self.get_parameter("pure_pursuit.wheelbase_m").value)),
        )
        if pure_pursuit_config.max_lookahead_m < pure_pursuit_config.min_lookahead_m:
            pure_pursuit_config = PurePursuitConfig(
                lookahead_m=pure_pursuit_config.lookahead_m,
                min_lookahead_m=pure_pursuit_config.min_lookahead_m,
                max_lookahead_m=pure_pursuit_config.min_lookahead_m,
                lookahead_gain=pure_pursuit_config.lookahead_gain,
                steering_limit_rad=pure_pursuit_config.steering_limit_rad,
                steering_lowpass_alpha=pure_pursuit_config.steering_lowpass_alpha,
                steering_rate_limit_rad_s=pure_pursuit_config.steering_rate_limit_rad_s,
                wheelbase_m=pure_pursuit_config.wheelbase_m,
            )
        return create_steering_controller(
            controller_type=self.controller_type,
            stanley_config=stanley_config,
            pure_pursuit_config=pure_pursuit_config,
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
