"""Long-lived global cone store used for track-belief plotting and visualization."""

from __future__ import annotations

from dataclasses import dataclass
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
