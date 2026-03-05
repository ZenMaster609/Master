"""Utilities for extracting cone-like clusters from 2D LiDAR points."""

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple


PointXYR = Tuple[float, float, float]


@dataclass(frozen=True)
class ClusterDetection:
    """Cone-like cluster summary in the LiDAR frame."""

    x_m: float
    y_m: float
    width_m: float
    depth_m: float
    point_count: int
    min_range_m: float
    max_range_m: float


def cluster_points(points: Sequence[PointXYR], jump_threshold_m: float) -> List[List[PointXYR]]:
    """Split scan-ordered points into contiguous Euclidean clusters."""
    if not points:
        return []
    threshold = max(1e-6, float(jump_threshold_m))
    clusters: List[List[PointXYR]] = []
    current: List[PointXYR] = [points[0]]
    prev_x, prev_y, _ = points[0]
    for x, y, r in points[1:]:
        jump = math.hypot(x - prev_x, y - prev_y)
        if jump > threshold:
            if current:
                clusters.append(current)
            current = [(x, y, r)]
        else:
            current.append((x, y, r))
        prev_x, prev_y = x, y
    if current:
        clusters.append(current)
    return clusters


def detect_cone_candidates(
    points: Sequence[PointXYR],
    *,
    jump_threshold_m: float,
    min_cluster_points: int,
    max_cluster_points: int,
    min_cluster_width_m: float,
    max_cluster_width_m: float,
    max_cluster_depth_m: float,
) -> List[ClusterDetection]:
    """Filter contiguous clusters into cone-like detections."""
    detections: List[ClusterDetection] = []
    min_pts = max(1, int(min_cluster_points))
    max_pts = max(min_pts, int(max_cluster_points))
    min_w = max(0.0, float(min_cluster_width_m))
    max_w = max(min_w, float(max_cluster_width_m))
    max_d = max(0.0, float(max_cluster_depth_m))
    for cluster in cluster_points(points, jump_threshold_m=jump_threshold_m):
        n = len(cluster)
        if n < min_pts or n > max_pts:
            continue
        x0, y0, _ = cluster[0]
        x1, y1, _ = cluster[-1]
        width = math.hypot(x1 - x0, y1 - y0)
        if width < min_w or width > max_w:
            continue
        ranges = [p[2] for p in cluster]
        r_min = min(ranges)
        r_max = max(ranges)
        depth = r_max - r_min
        if depth > max_d:
            continue
        cx = sum(p[0] for p in cluster) / float(n)
        cy = sum(p[1] for p in cluster) / float(n)
        detections.append(
            ClusterDetection(
                x_m=float(cx),
                y_m=float(cy),
                width_m=float(width),
                depth_m=float(depth),
                point_count=n,
                min_range_m=float(r_min),
                max_range_m=float(r_max),
            )
        )
    return detections


def points_from_ranges(
    ranges: Iterable[float],
    *,
    angle_min_rad: float,
    angle_increment_rad: float,
    range_min_m: float,
    range_max_m: float,
    min_detection_range_m: float,
    max_detection_range_m: float,
) -> List[PointXYR]:
    """Convert LiDAR polar ranges into finite XY points for clustering."""
    points: List[PointXYR] = []
    angle = float(angle_min_rad)
    lo = max(float(range_min_m), float(min_detection_range_m))
    hi = min(float(range_max_m), float(max_detection_range_m))
    if not math.isfinite(lo):
        lo = float(min_detection_range_m)
    if not math.isfinite(hi):
        hi = float(max_detection_range_m)
    for r in ranges:
        rr = float(r)
        if math.isfinite(rr) and lo <= rr <= hi:
            points.append((rr * math.cos(angle), rr * math.sin(angle), rr))
        angle += float(angle_increment_rad)
    return points

