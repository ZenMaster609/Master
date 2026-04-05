"""Skidpad-specific mission routing and lap detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SkidpadRouterConfig:
    crossroads_center_xy: tuple[float, float] = (0.0, 0.0)
    crossroads_half_extents_xy: tuple[float, float] = (3.0, 2.5)
    left_circle_center_xy: tuple[float, float] = (-9.25, 0.0)
    right_circle_center_xy: tuple[float, float] = (9.25, 0.0)
    circle_radius_window_m: tuple[float, float] = (6.5, 11.5)
    lap_complete_angle_rad: float = 5.5
    route_sequence: tuple[str, ...] = ("right", "right", "left", "left", "straight")
    route_laps: int = 1
    lobe_radius_m: float = 12.5
    right_lobe_min_x_m: float = -1.0
    left_lobe_max_x_m: float = 1.0
    straight_corridor_half_width_m: float = 2.0
    straight_min_y_m: float = -1.0
    parking_corridor_half_width_m: float = 2.0
    parking_start_y_m: float = 8.5
    parked_speed_threshold_mps: float = 0.5
    parked_hold_time_s: float = 1.0
    test_park_only: bool = False
    synthetic_pair_half_width_m: float = 1.6375
    synthetic_pair_y_m: tuple[float, ...] = (3.5, 5.5, 7.5)


@dataclass(frozen=True)
class SkidpadRouterSnapshot:
    stage_name: str
    active_branch: str
    completed_laps: int
    route_index: int
    current_route_pass: int
    total_route_passes: int
    completed_route_passes: int
    in_crossroads: bool
    lap_angle_accum_rad: float
    lap_armed: bool
    parked: bool
    just_completed_lap: bool
    just_entered_crossroads: bool


class SkidpadStateMachine:
    """Mission state machine for the fixed skidpad route."""

    def __init__(self, config: SkidpadRouterConfig) -> None:
        self._config = config
        self._repeatable_route_sequence = _normalize_repeatable_route_sequence(config.route_sequence)
        self._repeatable_route_length = len(self._repeatable_route_sequence)
        self._route_passes_total = 0 if config.test_park_only or self._repeatable_route_length == 0 else max(
            1, int(config.route_laps)
        )
        self._expanded_route_sequence = self._build_expanded_route_sequence()
        self.reset()

    def reset(self) -> None:
        self.completed_laps = 0
        self.route_index = 0
        self.approach_complete = False
        self.parked = False
        self._prev_in_crossroads: bool | None = None
        self._lap_angle_accum_rad = 0.0
        self._lap_armed = False
        self._last_theta_rad: float | None = None
        self._parked_since_sec: float | None = None

    @property
    def config(self) -> SkidpadRouterConfig:
        return self._config

    def active_branch(self) -> str:
        if self._config.test_park_only:
            return "straight"
        idx = min(max(self.route_index, 0), len(self._expanded_route_sequence) - 1)
        return str(self._expanded_route_sequence[idx]).strip().lower() or "straight"

    def stage_name(self) -> str:
        branch = self.active_branch()
        if self.parked:
            return "parked"
        if not self.approach_complete:
            return "approach"
        if branch == "straight":
            return "straight"
        if self._repeatable_route_length <= 0:
            return branch
        route_index_in_pass = self.route_index % self._repeatable_route_length
        lap_number = 1 + sum(
            1
            for prev_branch in self._repeatable_route_sequence[:route_index_in_pass]
            if str(prev_branch).strip().lower() == branch
        )
        return f"{branch}_{lap_number}"

    def current_route_pass(self) -> int:
        if self._route_passes_total <= 0:
            return 0
        if not self.approach_complete:
            return 1
        if self.active_branch() == "straight":
            return self._route_passes_total
        return min(self.completed_route_passes() + 1, self._route_passes_total)

    def total_route_passes(self) -> int:
        return self._route_passes_total

    def completed_route_passes(self) -> int:
        if self._route_passes_total <= 0 or self._repeatable_route_length <= 0:
            return 0
        return min(self.route_index // self._repeatable_route_length, self._route_passes_total)

    def update(self, x_m: float, y_m: float, speed_mps: float, now_sec: float) -> SkidpadRouterSnapshot:
        in_crossroads = self.is_in_crossroads(x_m, y_m)
        just_entered_crossroads = bool(self._prev_in_crossroads is False and in_crossroads)
        branch = self.active_branch()
        just_completed_lap = False

        if branch in {"right", "left"}:
            self._update_circle_progress(x_m=x_m, y_m=y_m, in_crossroads=in_crossroads, branch=branch)
        else:
            self._reset_lap_accumulator()

        if just_entered_crossroads:
            if not self.approach_complete:
                self.approach_complete = True
            elif branch in {"right", "left"} and self._lap_armed:
                self.completed_laps += 1
                just_completed_lap = True
                if self.route_index < len(self._expanded_route_sequence) - 1:
                    self.route_index += 1
                self._reset_lap_accumulator()
                branch = self.active_branch()

        if branch == "straight":
            self._update_parked_state(x_m=x_m, y_m=y_m, speed_mps=speed_mps, now_sec=now_sec)
        else:
            self._parked_since_sec = None

        self._prev_in_crossroads = in_crossroads
        return self._snapshot(
            in_crossroads=in_crossroads,
            just_completed_lap=just_completed_lap,
            just_entered_crossroads=just_entered_crossroads,
        )

    def route_mask(self, points_xy: np.ndarray) -> np.ndarray:
        if points_xy.size == 0:
            return np.empty((0,), dtype=bool)

        points = np.asarray(points_xy, dtype=np.float64)
        branch = self.active_branch()

        crossroads_mask = np.asarray(
            [self.is_in_crossroads(float(point[0]), float(point[1])) for point in points],
            dtype=bool,
        )
        if self.stage_name() == "approach":
            approach_mask = np.asarray(
                [self.is_in_approach_corridor(float(point[0]), float(point[1])) for point in points],
                dtype=bool,
            )
            if branch == "straight":
                straight_mask = np.asarray(
                    [self.is_in_straight_corridor(float(point[0]), float(point[1])) for point in points],
                    dtype=bool,
                )
                return crossroads_mask | approach_mask | straight_mask
            right_mask = np.asarray(
                [self.is_in_right_lobe(float(point[0]), float(point[1])) for point in points],
                dtype=bool,
            )
            return crossroads_mask | approach_mask | right_mask

        if branch == "right":
            right_mask = np.asarray(
                [self.is_in_right_lobe(float(point[0]), float(point[1])) for point in points],
                dtype=bool,
            )
            return crossroads_mask | right_mask

        if branch == "left":
            left_mask = np.asarray(
                [self.is_in_left_lobe(float(point[0]), float(point[1])) for point in points],
                dtype=bool,
            )
            return crossroads_mask | left_mask

        straight_mask = np.asarray(
            [self.is_in_straight_corridor(float(point[0]), float(point[1])) for point in points],
            dtype=bool,
        )
        return crossroads_mask | straight_mask

    def synthetic_cone_pairs(self) -> list[np.ndarray]:
        if self.active_branch() != "straight":
            return []
        pairs: list[np.ndarray] = []
        half_width = float(max(0.0, self._config.synthetic_pair_half_width_m))
        cx, _cy = self._config.crossroads_center_xy
        for y_m in self._config.synthetic_pair_y_m:
            y_val = float(y_m)
            pairs.append(np.asarray([cx - half_width, y_val], dtype=np.float64))
            pairs.append(np.asarray([cx + half_width, y_val], dtype=np.float64))
        return pairs

    def is_in_crossroads(self, x_m: float, y_m: float) -> bool:
        cx, cy = self._config.crossroads_center_xy
        hx, hy = self._config.crossroads_half_extents_xy
        return abs(float(x_m) - cx) <= hx and abs(float(y_m) - cy) <= hy

    def is_in_approach_corridor(self, x_m: float, y_m: float) -> bool:
        cx, cy = self._config.crossroads_center_xy
        hy = self._config.crossroads_half_extents_xy[1]
        return abs(float(x_m) - cx) <= float(self._config.straight_corridor_half_width_m) and float(y_m) <= (cy + hy)

    def is_in_right_lobe(self, x_m: float, y_m: float) -> bool:
        center = np.asarray(self._config.right_circle_center_xy, dtype=np.float64)
        point = np.asarray([x_m, y_m], dtype=np.float64)
        return float(np.linalg.norm(point - center)) <= float(self._config.lobe_radius_m) and float(x_m) >= float(
            self._config.right_lobe_min_x_m
        )

    def is_in_left_lobe(self, x_m: float, y_m: float) -> bool:
        center = np.asarray(self._config.left_circle_center_xy, dtype=np.float64)
        point = np.asarray([x_m, y_m], dtype=np.float64)
        return float(np.linalg.norm(point - center)) <= float(self._config.lobe_radius_m) and float(x_m) <= float(
            self._config.left_lobe_max_x_m
        )

    def is_in_straight_corridor(self, x_m: float, y_m: float) -> bool:
        cx, _cy = self._config.crossroads_center_xy
        return abs(float(x_m) - cx) <= float(self._config.straight_corridor_half_width_m) and float(y_m) >= float(
            self._config.straight_min_y_m
        )

    def is_in_parking_corridor(self, x_m: float, y_m: float) -> bool:
        cx, _cy = self._config.crossroads_center_xy
        return abs(float(x_m) - cx) <= float(self._config.parking_corridor_half_width_m) and float(y_m) >= float(
            self._config.parking_start_y_m
        )

    def _update_circle_progress(self, *, x_m: float, y_m: float, in_crossroads: bool, branch: str) -> None:
        center_xy = self._config.right_circle_center_xy if branch == "right" else self._config.left_circle_center_xy
        cx, cy = center_xy
        radius_m = math.hypot(float(x_m) - cx, float(y_m) - cy)
        radius_min_m, radius_max_m = self._config.circle_radius_window_m
        valid_sample = (not in_crossroads) and radius_min_m <= radius_m <= radius_max_m
        if not valid_sample:
            self._last_theta_rad = None
            return

        theta_rad = math.atan2(float(y_m) - cy, float(x_m) - cx)
        if self._last_theta_rad is not None:
            delta_rad = _wrap_to_pi(theta_rad - self._last_theta_rad)
            self._lap_angle_accum_rad += delta_rad
            if abs(self._lap_angle_accum_rad) >= float(self._config.lap_complete_angle_rad):
                self._lap_armed = True
        self._last_theta_rad = theta_rad

    def _reset_lap_accumulator(self) -> None:
        self._lap_angle_accum_rad = 0.0
        self._lap_armed = False
        self._last_theta_rad = None

    def _update_parked_state(self, *, x_m: float, y_m: float, speed_mps: float, now_sec: float) -> None:
        if not self.is_in_parking_corridor(x_m, y_m) or float(speed_mps) > float(self._config.parked_speed_threshold_mps):
            self._parked_since_sec = None
            return
        if self._parked_since_sec is None:
            self._parked_since_sec = float(now_sec)
            return
        if (float(now_sec) - self._parked_since_sec) >= float(self._config.parked_hold_time_s):
            self.parked = True

    def _build_expanded_route_sequence(self) -> tuple[str, ...]:
        if self._config.test_park_only:
            return ("straight",)
        if self._repeatable_route_length <= 0 or self._route_passes_total <= 0:
            return ("straight",)
        return (self._repeatable_route_sequence * self._route_passes_total) + ("straight",)

    def _snapshot(
        self,
        *,
        in_crossroads: bool,
        just_completed_lap: bool,
        just_entered_crossroads: bool,
    ) -> SkidpadRouterSnapshot:
        return SkidpadRouterSnapshot(
            stage_name=self.stage_name(),
            active_branch=self.active_branch(),
            completed_laps=self.completed_laps,
            route_index=self.route_index,
            current_route_pass=self.current_route_pass(),
            total_route_passes=self.total_route_passes(),
            completed_route_passes=self.completed_route_passes(),
            in_crossroads=in_crossroads,
            lap_angle_accum_rad=self._lap_angle_accum_rad,
            lap_armed=self._lap_armed,
            parked=self.parked,
            just_completed_lap=just_completed_lap,
            just_entered_crossroads=just_entered_crossroads,
        )


def config_from_parameters(
    *,
    crossroads_center_xy: Sequence[float],
    crossroads_half_extents_xy: Sequence[float],
    left_circle_center_xy: Sequence[float],
    right_circle_center_xy: Sequence[float],
    circle_radius_window_m: Sequence[float],
    route_sequence: Sequence[str],
    route_laps: int,
    synthetic_pair_y_m: Sequence[float],
    lap_complete_angle_rad: float,
    parking_corridor_half_width_m: float,
    parking_start_y_m: float,
    straight_corridor_half_width_m: float,
    straight_min_y_m: float,
    parked_speed_threshold_mps: float,
    parked_hold_time_s: float,
    test_park_only: bool,
    synthetic_pair_half_width_m: float,
    lobe_radius_m: float,
    right_lobe_min_x_m: float,
    left_lobe_max_x_m: float,
) -> SkidpadRouterConfig:
    return SkidpadRouterConfig(
        crossroads_center_xy=_pair_of_floats(crossroads_center_xy),
        crossroads_half_extents_xy=_pair_of_floats(crossroads_half_extents_xy),
        left_circle_center_xy=_pair_of_floats(left_circle_center_xy),
        right_circle_center_xy=_pair_of_floats(right_circle_center_xy),
        circle_radius_window_m=_pair_of_floats(circle_radius_window_m),
        lap_complete_angle_rad=float(lap_complete_angle_rad),
        route_sequence=(
            ("straight",)
            if bool(test_park_only)
            else tuple(str(item).strip().lower() for item in route_sequence)
        ),
        route_laps=max(1, int(route_laps)),
        lobe_radius_m=float(lobe_radius_m),
        right_lobe_min_x_m=float(right_lobe_min_x_m),
        left_lobe_max_x_m=float(left_lobe_max_x_m),
        straight_corridor_half_width_m=float(straight_corridor_half_width_m),
        straight_min_y_m=float(straight_min_y_m),
        parking_corridor_half_width_m=float(parking_corridor_half_width_m),
        parking_start_y_m=float(parking_start_y_m),
        parked_speed_threshold_mps=float(parked_speed_threshold_mps),
        parked_hold_time_s=float(parked_hold_time_s),
        test_park_only=bool(test_park_only),
        synthetic_pair_half_width_m=float(synthetic_pair_half_width_m),
        synthetic_pair_y_m=tuple(float(item) for item in synthetic_pair_y_m),
    )


def _pair_of_floats(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"expected 2 values, got {len(values)}")
    return (float(values[0]), float(values[1]))


def _normalize_repeatable_route_sequence(route_sequence: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(item).strip().lower() for item in route_sequence if str(item).strip())
    if normalized and normalized[-1] == "straight":
        return normalized[:-1]
    return normalized


def _wrap_to_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def boundary_color_from_lateral_y(lateral_y_m: float) -> str:
    return "blue" if float(lateral_y_m) >= 0.0 else "yellow"


def detect_stop_line_forward_distance_m(
    vehicle_frame_orange_points_xy: np.ndarray,
    *,
    cluster_depth_m: float = 0.75,
    min_lateral_span_m: float = 1.0,
    min_cluster_count: int = 4,
    min_points_per_side: int = 1,
) -> Optional[float]:
    points = np.asarray(vehicle_frame_orange_points_xy, dtype=np.float64)
    if points.size == 0 or points.ndim != 2 or points.shape[1] != 2:
        return None

    ahead = points[np.isfinite(points).all(axis=1) & (points[:, 0] > 0.0)]
    if ahead.shape[0] < int(max(2, min_cluster_count)):
        return None

    max_forward_m = float(np.max(ahead[:, 0]))
    depth_m = max(0.0, float(cluster_depth_m))
    cluster = ahead[ahead[:, 0] >= (max_forward_m - depth_m)]
    if cluster.shape[0] < int(max(2, min_cluster_count)):
        return None

    lateral_span_m = float(np.max(cluster[:, 1]) - np.min(cluster[:, 1]))
    if lateral_span_m < float(max(0.0, min_lateral_span_m)):
        return None

    min_side_count = int(max(0, min_points_per_side))
    if min_side_count > 0:
        left_count = int(np.count_nonzero(cluster[:, 1] >= 0.0))
        right_count = int(np.count_nonzero(cluster[:, 1] < 0.0))
        if left_count < min_side_count or right_count < min_side_count:
            return None

    return float(np.median(cluster[:, 0]))


def detect_stop_line_pair(
    vehicle_frame_orange_points_xy: np.ndarray,
    *,
    max_pair_distance_m: float = 1.0,
) -> Optional[tuple[int, int, float]]:
    points = np.asarray(vehicle_frame_orange_points_xy, dtype=np.float64)
    if points.size == 0 or points.ndim != 2 or points.shape[1] != 2:
        return None

    valid_indices = np.flatnonzero(np.isfinite(points).all(axis=1) & (points[:, 0] > 0.0))
    if valid_indices.size < 2:
        return None

    best_pair: Optional[tuple[int, int, float]] = None
    best_score: Optional[tuple[float, float]] = None
    max_distance_m = max(0.0, float(max_pair_distance_m))
    for lhs in range(valid_indices.size - 1):
        idx_a = int(valid_indices[lhs])
        point_a = points[idx_a]
        for rhs in range(lhs + 1, valid_indices.size):
            idx_b = int(valid_indices[rhs])
            point_b = points[idx_b]
            pair_distance_m = float(np.linalg.norm(point_b - point_a))
            if pair_distance_m > max_distance_m:
                continue

            forward_distance_m = 0.5 * float(point_a[0] + point_b[0])
            score = (forward_distance_m, -pair_distance_m)
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (idx_a, idx_b, forward_distance_m)

    return best_pair
