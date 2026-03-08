"""Track containers and data association for cone memory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from sim_car.cone_fusion import class_from_probs, normalize_color, update_class_probs


@dataclass
class TrackUpdate:
    assoc_x: float
    assoc_y: float
    update_x: Optional[float]
    update_y: Optional[float]
    update_z: Optional[float]
    update_source: Optional[str]
    has_lidar: bool
    has_camera: bool
    camera_label: Optional[str]
    camera_confidence: float
    range_m: float


@dataclass
class ConeTrack:
    track_id: int
    x: float
    y: float
    z: float
    variance: float
    class_probs: list[float]
    last_seen_sec: float
    seen_count: int = 1
    lidar_seen_recently: bool = False
    camera_seen_recently: bool = False
    unknown_streak: int = 0
    camera_class_confirmed: bool = False

    def is_confirmed(self, min_seen_count: int) -> bool:
        return self.seen_count >= int(min_seen_count)

    def class_label(self) -> tuple[str, float]:
        return class_from_probs(self.class_probs)


@dataclass
class GlobalCone:
    cone_id: int
    x: float
    y: float
    z: float
    class_label: str
    confidence: float
    hits: int
    last_update_sec: float


@dataclass
class TrackFrameStats:
    total_tracks: int = 0
    confirmed_tracks: int = 0
    pruned_tracks: int = 0
    matched_updates: int = 0
    new_tracks: int = 0
    suppressed_new_tracks: int = 0
    merged_tracks: int = 0


class LocalConeTracker:
    """Local short-term tracker for planner-facing fused cones."""

    def __init__(self) -> None:
        self._next_track_id = 1
        self.tracks: list[ConeTrack] = []

    def update(
        self,
        *,
        updates: list[TrackUpdate],
        now_sec: float,
        gate_radius_m: float,
        spawn_radius_m: float,
        alpha_lidar: float,
        alpha_camera: float,
        min_seen_count: int,
    ) -> TrackFrameStats:
        stats = TrackFrameStats()

        assignments, unmatched_update_idx = self._associate(updates, gate_radius_m)
        matched_tracks: set[int] = set()

        for track_idx, update_idx in assignments:
            update = updates[update_idx]
            track = self.tracks[track_idx]
            matched_tracks.add(track_idx)
            self._apply_update(
                track=track,
                update=update,
                now_sec=now_sec,
                alpha_lidar=alpha_lidar,
                alpha_camera=alpha_camera,
            )
            stats.matched_updates += 1

        for update_idx in unmatched_update_idx:
            update = updates[update_idx]
            if update.update_x is None or update.update_y is None:
                continue
            fallback_track_idx = self._find_nearby_track_for_update(
                update=update,
                radius_m=max(gate_radius_m, spawn_radius_m),
            )
            if fallback_track_idx is not None:
                self._apply_update(
                    track=self.tracks[fallback_track_idx],
                    update=update,
                    now_sec=now_sec,
                    alpha_lidar=alpha_lidar,
                    alpha_camera=alpha_camera,
                )
                stats.matched_updates += 1
                stats.suppressed_new_tracks += 1
                continue
            self.tracks.append(
                ConeTrack(
                    track_id=self._next_track_id,
                    x=float(update.update_x),
                    y=float(update.update_y),
                    z=float(update.update_z or 0.0),
                    variance=0.25,
                    class_probs=update_class_probs(
                        [1.0, 0.0, 0.0, 0.0],
                        label=update.camera_label,
                        confidence=update.camera_confidence,
                    ),
                    last_seen_sec=now_sec,
                    seen_count=1,
                    lidar_seen_recently=bool(update.has_lidar),
                    camera_seen_recently=bool(update.has_camera),
                    unknown_streak=1 if normalize_color(update.camera_label or 'unknown') == 'unknown' else 0,
                    camera_class_confirmed=normalize_color(update.camera_label or 'unknown') != 'unknown',
                )
            )
            self._next_track_id += 1
            stats.new_tracks += 1

        stats.total_tracks = len(self.tracks)
        stats.confirmed_tracks = sum(1 for t in self.tracks if t.is_confirmed(min_seen_count))
        return stats

    def merge_nearby_tracks(
        self,
        *,
        merge_radius_m: float,
        track_positions_in_base: Optional[list[tuple[float, float]]] = None,
        longitudinal_tolerance_m: float = 0.0,
        lateral_tolerance_m: float = 0.0,
    ) -> int:
        if len(self.tracks) <= 1:
            return 0

        merge_sq = merge_radius_m * merge_radius_m
        merged_count = 0
        keep: list[ConeTrack] = []
        base_positions_by_track_id = {
            track.track_id: track_positions_in_base[idx]
            for idx, track in enumerate(self.tracks)
        } if track_positions_in_base is not None and len(track_positions_in_base) == len(self.tracks) else {}

        for candidate in sorted(
            self.tracks,
            key=lambda track: (track.last_seen_sec, track.seen_count, track.camera_class_confirmed),
            reverse=True,
        ):
            best_idx = None
            best_dist_sq = float('inf')
            candidate_label, candidate_conf = candidate.class_label()

            for idx, existing in enumerate(keep):
                if not self._track_labels_compatible(existing, candidate_label):
                    continue
                dx = candidate.x - existing.x
                dy = candidate.y - existing.y
                dist_sq = dx * dx + dy * dy
                if not self._tracks_are_merge_compatible(
                    candidate=candidate,
                    existing=existing,
                    dist_sq=dist_sq,
                    merge_sq=merge_sq,
                    base_positions_by_track_id=base_positions_by_track_id,
                    longitudinal_tolerance_m=longitudinal_tolerance_m,
                    lateral_tolerance_m=lateral_tolerance_m,
                ):
                    continue
                if dist_sq < best_dist_sq:
                    best_idx = idx
                    best_dist_sq = dist_sq

            if best_idx is None:
                keep.append(candidate)
                continue

            existing = keep[best_idx]
            existing_label, existing_conf = existing.class_label()
            existing_weight = max(1.0, float(existing.seen_count))
            candidate_weight = max(1.0, float(candidate.seen_count))
            weight_sum = existing_weight + candidate_weight

            existing.x = ((existing.x * existing_weight) + (candidate.x * candidate_weight)) / weight_sum
            existing.y = ((existing.y * existing_weight) + (candidate.y * candidate_weight)) / weight_sum
            existing.z = ((existing.z * existing_weight) + (candidate.z * candidate_weight)) / weight_sum
            existing.variance = min(existing.variance, candidate.variance)
            existing.last_seen_sec = max(existing.last_seen_sec, candidate.last_seen_sec)
            existing.seen_count = max(existing.seen_count, candidate.seen_count)
            existing.lidar_seen_recently = existing.lidar_seen_recently or candidate.lidar_seen_recently
            existing.camera_seen_recently = existing.camera_seen_recently or candidate.camera_seen_recently
            existing.camera_class_confirmed = existing.camera_class_confirmed or candidate.camera_class_confirmed
            existing.unknown_streak = min(existing.unknown_streak, candidate.unknown_streak)
            if candidate_conf > existing_conf and candidate_label != 'unknown':
                existing.class_probs = list(candidate.class_probs)
            elif existing_label == 'unknown' and candidate_label == 'unknown':
                existing.class_probs = [
                    (a + b) * 0.5 for a, b in zip(existing.class_probs, candidate.class_probs)
                ]
            merged_count += 1

        self.tracks = keep
        return merged_count

    @staticmethod
    def _tracks_are_merge_compatible(
        *,
        candidate: ConeTrack,
        existing: ConeTrack,
        dist_sq: float,
        merge_sq: float,
        base_positions_by_track_id: dict[int, tuple[float, float]],
        longitudinal_tolerance_m: float,
        lateral_tolerance_m: float,
    ) -> bool:
        if dist_sq <= merge_sq:
            return True

        if longitudinal_tolerance_m <= 0.0 or lateral_tolerance_m <= 0.0:
            return False

        candidate_base = base_positions_by_track_id.get(candidate.track_id)
        existing_base = base_positions_by_track_id.get(existing.track_id)
        if candidate_base is None or existing_base is None:
            return False

        dx_base = abs(candidate_base[0] - existing_base[0])
        dy_base = abs(candidate_base[1] - existing_base[1])
        if dy_base > lateral_tolerance_m or dx_base > longitudinal_tolerance_m:
            return False

        # Prevent same-color cones on opposite sides of the car from collapsing
        # if the color estimate is still uncertain.
        if candidate_base[1] * existing_base[1] < 0.0 and max(abs(candidate_base[1]), abs(existing_base[1])) > 0.2:
            return False

        return True

    def prune(
        self,
        *,
        now_sec: float,
        ttl_sec: float,
        max_range_m: float,
        behind_drop_m: float,
        unknown_drop_frames: int,
        track_positions_in_base: list[tuple[float, float]],
    ) -> int:
        keep: list[ConeTrack] = []
        pruned = 0
        for idx, track in enumerate(self.tracks):
            x_base, y_base = track_positions_in_base[idx]
            age = now_sec - track.last_seen_sec
            too_old = age > ttl_sec
            too_far = (x_base * x_base + y_base * y_base) ** 0.5 > max_range_m
            behind = x_base < -behind_drop_m
            unknown_filtered = (
                int(unknown_drop_frames) > 0
                and not track.camera_class_confirmed
                and track.unknown_streak >= int(unknown_drop_frames)
            )
            if too_old or too_far or behind or unknown_filtered:
                pruned += 1
                continue
            keep.append(track)
        self.tracks = keep
        return pruned

    def confirmed_tracks(self, min_seen_count: int) -> list[ConeTrack]:
        return [t for t in self.tracks if t.is_confirmed(min_seen_count)]

    def tentative_tracks(self, min_seen_count: int) -> list[ConeTrack]:
        return [t for t in self.tracks if not t.is_confirmed(min_seen_count)]

    def _associate(self, updates: list[TrackUpdate], gate_radius_m: float):
        candidates: list[tuple[float, int, int]] = []
        gate_sq = gate_radius_m * gate_radius_m
        for t_idx, track in enumerate(self.tracks):
            for u_idx, update in enumerate(updates):
                dx = update.assoc_x - track.x
                dy = update.assoc_y - track.y
                dist_sq = dx * dx + dy * dy
                if dist_sq <= gate_sq:
                    candidates.append((dist_sq ** 0.5, t_idx, u_idx))

        candidates.sort(key=lambda item: item[0])
        taken_tracks: set[int] = set()
        taken_updates: set[int] = set()
        assignments: list[tuple[int, int]] = []
        for _dist, t_idx, u_idx in candidates:
            if t_idx in taken_tracks or u_idx in taken_updates:
                continue
            taken_tracks.add(t_idx)
            taken_updates.add(u_idx)
            assignments.append((t_idx, u_idx))

        unmatched_update_idx = [i for i in range(len(updates)) if i not in taken_updates]
        return assignments, unmatched_update_idx

    def _find_nearby_track_for_update(
        self,
        *,
        update: TrackUpdate,
        radius_m: float,
    ) -> Optional[int]:
        if not self.tracks:
            return None

        radius_sq = radius_m * radius_m
        update_label = normalize_color(update.camera_label or 'unknown')
        best_idx: Optional[int] = None
        best_score = float('inf')

        for idx, track in enumerate(self.tracks):
            if not self._track_labels_compatible(track, update_label):
                continue
            dx = update.assoc_x - track.x
            dy = update.assoc_y - track.y
            dist_sq = dx * dx + dy * dy
            if dist_sq > radius_sq:
                continue
            # Prefer the closest track; break ties toward more stable tracks.
            score = math.sqrt(dist_sq) - (0.02 * min(track.seen_count, 20))
            if score < best_score:
                best_score = score
                best_idx = idx

        return best_idx

    @staticmethod
    def _track_labels_compatible(track: ConeTrack, update_label: str) -> bool:
        track_label, _track_conf = track.class_label()
        track_norm = normalize_color(track_label)
        update_norm = normalize_color(update_label)
        if track_norm == 'unknown' or update_norm == 'unknown':
            return True
        return track_norm == update_norm

    @staticmethod
    def _apply_update(
        *,
        track: ConeTrack,
        update: TrackUpdate,
        now_sec: float,
        alpha_lidar: float,
        alpha_camera: float,
    ) -> None:
        if update.update_x is not None and update.update_y is not None:
            if update.update_source == 'lidar':
                alpha = max(0.0, min(1.0, alpha_lidar))
            elif update.update_source == 'camera':
                alpha = max(0.0, min(1.0, alpha_camera))
            else:
                alpha = 0.25
            old_x = track.x
            old_y = track.y
            old_z = track.z
            track.x = (1.0 - alpha) * old_x + alpha * float(update.update_x)
            track.y = (1.0 - alpha) * old_y + alpha * float(update.update_y)
            track.z = (1.0 - alpha) * old_z + alpha * float(update.update_z or 0.0)

            residual_sq = ((track.x - old_x) ** 2 + (track.y - old_y) ** 2)
            track.variance = (1.0 - alpha) * track.variance + alpha * residual_sq

        if update.camera_label is not None:
            track.class_probs = update_class_probs(
                track.class_probs,
                label=update.camera_label,
                confidence=update.camera_confidence,
            )
            if normalize_color(update.camera_label) != 'unknown':
                track.unknown_streak = 0
                track.camera_class_confirmed = True
            elif not track.camera_class_confirmed:
                track.unknown_streak += 1
        elif not track.camera_class_confirmed:
            track.unknown_streak += 1

        track.last_seen_sec = now_sec
        track.seen_count += 1
        track.lidar_seen_recently = bool(update.has_lidar)
        track.camera_seen_recently = bool(update.has_camera)


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
        vehicle_x: float,
        vehicle_y: float,
        heading_x: float,
        heading_y: float,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
        cones = [c for c in self.cones if c.hits >= int(min_hits)]
        left = [(c.x, c.y) for c in cones if c.class_label == 'blue']
        right = [(c.x, c.y) for c in cones if c.class_label == 'yellow']

        left_sorted = self._order_side(left, vehicle_x, vehicle_y, heading_x, heading_y)
        right_sorted = self._order_side(right, vehicle_x, vehicle_y, heading_x, heading_y)
        center = self._build_centerline(left_sorted, right_sorted)
        return left_sorted, right_sorted, center

    @staticmethod
    def _order_side(
        points: list[tuple[float, float]],
        vehicle_x: float,
        vehicle_y: float,
        heading_x: float,
        heading_y: float,
    ) -> list[tuple[float, float]]:
        if not points:
            return []

        def projection(p: tuple[float, float]) -> float:
            dx = p[0] - vehicle_x
            dy = p[1] - vehicle_y
            return dx * heading_x + dy * heading_y

        return sorted(points, key=projection)

    @staticmethod
    def _build_centerline(
        left: list[tuple[float, float]],
        right: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if not left or not right:
            return []

        center: list[tuple[float, float]] = []
        for lx, ly in left:
            best_idx = -1
            best_dist_sq = float('inf')
            for idx, (rx, ry) in enumerate(right):
                d2 = (rx - lx) ** 2 + (ry - ly) ** 2
                if d2 < best_dist_sq:
                    best_dist_sq = d2
                    best_idx = idx
            if best_idx < 0:
                continue
            rx, ry = right[best_idx]
            center.append(((lx + rx) * 0.5, (ly + ry) * 0.5))

        center.sort(key=lambda p: p[0])
        return center
