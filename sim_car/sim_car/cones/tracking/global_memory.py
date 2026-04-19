"""Long-lived global cone store used for track-belief plotting and visualization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim_car.cones.tracking.fusion import resolve_boundary_color_by_lateral_position

if TYPE_CHECKING:
    # Imported only for type-checker annotations; not at runtime to avoid circular imports.
    from sim_car.cones.tracking.tracker import ConeTrack


@dataclass
class GlobalCone:
    """A single entry in the global cone memory store."""

    cone_id: int
    x: float
    y: float
    z: float
    class_label: str
    confidence: float
    hits: int
    last_update_sec: float


class GlobalConeMemory:
    """Long-lived global cone store used for track-belief plotting."""

    def __init__(self) -> None:
        self._next_id = 1
        self.cones: list[GlobalCone] = []

    def update_from_tracks(
        self,
        *,
        tracks: list[ConeTrack],
        now_sec: float,
        merge_radius_m: float,
        max_cones: int,
    ) -> None:
        """Merge confirmed local tracks into the global cone store."""
        merge_sq = merge_radius_m * merge_radius_m
        for track in tracks:
            label, conf = track.class_label()
            best_idx = -1
            best_dist_sq = float('inf')
            for idx, cone in enumerate(self.cones):
                dx = cone.x - track.x
                dy = cone.y - track.y
                d2 = dx * dx + dy * dy
                if d2 <= merge_sq and d2 < best_dist_sq:
                    best_dist_sq = d2
                    best_idx = idx
            if best_idx >= 0:
                cone = self.cones[best_idx]
                alpha = 1.0 / float(max(2, cone.hits + 1))
                cone.x = (1.0 - alpha) * cone.x + alpha * track.x
                cone.y = (1.0 - alpha) * cone.y + alpha * track.y
                cone.z = (1.0 - alpha) * cone.z + alpha * track.z
                cone.hits += 1
                cone.last_update_sec = now_sec
                if label != 'unknown':
                    cone.class_label = label
                cone.confidence = max(cone.confidence, conf)
            else:
                self.cones.append(
                    GlobalCone(
                        cone_id=self._next_id,
                        x=track.x,
                        y=track.y,
                        z=track.z,
                        class_label=label,
                        confidence=conf,
                        hits=1,
                        last_update_sec=now_sec,
                    )
                )
                self._next_id += 1

        if len(self.cones) > max_cones:
            self.cones.sort(key=lambda c: (c.hits, c.last_update_sec), reverse=True)
            self.cones = self.cones[:max_cones]

    def infer_boundaries_and_centerline(
        self,
        *,
        min_hits: int,
        min_confidence: float = 0.0,
        vehicle_x: float,
        vehicle_y: float,
        heading_x: float,
        heading_y: float,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
        """Return (left, right, center) boundary point lists from the global store."""
        cones = [
            c for c in self.cones
            if c.hits >= int(min_hits) and float(c.confidence) >= float(min_confidence)
        ]
        left: list[tuple[float, float]] = []
        right: list[tuple[float, float]] = []
        for cone in cones:
            dx = cone.x - vehicle_x
            dy = cone.y - vehicle_y
            lateral_y = (-heading_y * dx) + (heading_x * dy)
            resolved_label = resolve_boundary_color_by_lateral_position(
                cone.class_label,
                lateral_y,
                infer_unknown=True,
                infer_orange=True,
            )
            if resolved_label == 'blue':
                left.append((cone.x, cone.y))
            elif resolved_label == 'yellow':
                right.append((cone.x, cone.y))

        left_sorted = _order_side(left, vehicle_x, vehicle_y, heading_x, heading_y)
        right_sorted = _order_side(right, vehicle_x, vehicle_y, heading_x, heading_y)
        center = _build_centerline(left_sorted, right_sorted)
        return left_sorted, right_sorted, center


def _order_side(
    points: list[tuple[float, float]],
    vehicle_x: float,
    vehicle_y: float,
    heading_x: float,
    heading_y: float,
) -> list[tuple[float, float]]:
    if not points:
        return []
    return sorted(
        points,
        key=lambda p: (p[0] - vehicle_x) * heading_x + (p[1] - vehicle_y) * heading_y,
    )


def _build_centerline(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not left or not right:
        return []
    center: list[tuple[float, float]] = []
    for lx, ly in left:
        best_idx = min(range(len(right)), key=lambda i: (right[i][0] - lx) ** 2 + (right[i][1] - ly) ** 2)
        rx, ry = right[best_idx]
        center.append(((lx + rx) * 0.5, (ly + ry) * 0.5))
    center.sort(key=lambda p: p[0])
    return center


# ---------------------------------------------------------------------------
# Permanent (cross-lap) cone memory
# ---------------------------------------------------------------------------


@dataclass
class PermaCone:
    """A high-confidence cone that has been passed and permanently remembered."""

    x: float
    y: float
    z: float
    label: str            # 'blue', 'yellow', or 'orange' — never 'unknown'
    confidence: float
    seen_count: int
    confirmed_by_live: bool = field(default=True)
    # True  → a live local track appeared near this cone on the most recent pass
    # False → reset when the car is far ahead; becomes True again if sensor sees it


class PermanentConeStore:
    """Stores high-confidence cones that have been passed so they can be re-injected
    into the planner feed on subsequent laps before sensors detect them again."""

    def __init__(self) -> None:
        self.cones: list[PermaCone] = []

    def harvest(
        self,
        tracks: list[ConeTrack],
        track_positions_in_base: list[tuple[float, float]],
        now_sec: float,
        *,
        behind_harvest_m: float,
        confirmed_ttl_sec: float,
        min_confidence: float,
        min_seen: int,
        confirm_hits: int,
        merge_radius_m: float,
    ) -> int:
        """Permanently store confirmed tracks that are behind the car or about to expire.

        A track qualifies when it is either:
        - behind the car by at least ``behind_harvest_m`` (well past), OR
        - older than ``confirmed_ttl_sec`` since last seen (handles slow/stopped car).

        Returns the number of newly added cones.
        """
        merge_sq = merge_radius_m * merge_radius_m
        added = 0
        for track, (x_base, _y_base) in zip(tracks, track_positions_in_base):
            is_behind = x_base < -behind_harvest_m
            is_expiring = (now_sec - track.last_seen_sec) >= confirmed_ttl_sec * 0.75
            if not (is_behind or is_expiring):
                continue
            if not track.is_confirmed(confirm_hits):
                continue
            if track.track_confidence < min_confidence:
                continue
            if track.seen_count < min_seen:
                continue
            label, _color_conf = track.class_label()
            if label == 'unknown':
                continue
            # Skip if a permanent cone already exists nearby (dedup)
            if any(
                (c.x - track.x) ** 2 + (c.y - track.y) ** 2 <= merge_sq
                for c in self.cones
            ):
                continue
            self.cones.append(
                PermaCone(
                    x=track.x,
                    y=track.y,
                    z=track.z,
                    label=label,
                    confidence=track.track_confidence,
                    seen_count=track.seen_count,
                )
            )
            added += 1
        return added

    def nearby_without_live_track(
        self,
        vehicle_x: float,
        vehicle_y: float,
        planning_range_m: float,
        live_tracks: list[ConeTrack],
        dedup_radius_m: float,
    ) -> list[PermaCone]:
        """Return permanent cones within planning range that have no nearby live local track.

        This prevents double-publishing a cone that the sensor already sees again.
        """
        range_sq = planning_range_m * planning_range_m
        dedup_sq = dedup_radius_m * dedup_radius_m
        live_pos = [(t.x, t.y) for t in live_tracks]
        result: list[PermaCone] = []
        for cone in self.cones:
            dx = cone.x - vehicle_x
            dy = cone.y - vehicle_y
            if dx * dx + dy * dy > range_sq:
                continue
            if any(
                (cone.x - lx) ** 2 + (cone.y - ly) ** 2 <= dedup_sq
                for lx, ly in live_pos
            ):
                continue
            result.append(cone)
        return result

    def update_and_prune(
        self,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        live_tracks: list[ConeTrack],
        *,
        prune_behind_m: float,
        confirm_range_m: float,
        dedup_radius_m: float,
    ) -> int:
        """Update live-detection flags and prune cones the car passed without seeing.

        Each permanent cone's ``confirmed_by_live`` flag is:
        - reset to False when the cone is more than (confirm_range_m + 5 m) ahead
          of the car (arms the check for the next approach),
        - set to True when a live local track appears within ``dedup_radius_m`` while
          the car is within ``confirm_range_m`` of the cone.

        A cone is removed when it is more than ``prune_behind_m`` behind the car
        AND ``confirmed_by_live`` is still False — meaning the car passed close
        enough for sensors to have detected it but nothing was found.
        """
        cos_yaw = math.cos(vehicle_yaw)
        sin_yaw = math.sin(vehicle_yaw)
        dedup_sq = dedup_radius_m * dedup_radius_m
        reset_arm_m = confirm_range_m + 5.0
        keep: list[PermaCone] = []
        pruned = 0
        for cone in self.cones:
            dx = cone.x - vehicle_x
            dy = cone.y - vehicle_y
            dist_sq = dx * dx + dy * dy
            x_base = cos_yaw * dx + sin_yaw * dy

            # Arm for next pass: cone is far ahead, reset the confirmation flag
            if x_base > reset_arm_m:
                cone.confirmed_by_live = False

            # Confirm: live track detected near this cone while within sensor range
            if dist_sq <= confirm_range_m * confirm_range_m and not cone.confirmed_by_live:
                if any(
                    (cone.x - t.x) ** 2 + (cone.y - t.y) ** 2 <= dedup_sq
                    for t in live_tracks
                ):
                    cone.confirmed_by_live = True

            # Prune: passed by prune_behind_m with no live confirmation this pass
            if x_base < -prune_behind_m and not cone.confirmed_by_live:
                pruned += 1
                continue

            keep.append(cone)
        self.cones = keep
        return pruned
