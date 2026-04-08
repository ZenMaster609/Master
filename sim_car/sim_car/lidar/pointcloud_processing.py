"""Point-cloud processing helpers for simulated lidar cone extraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


_POINTFIELD_DTYPES: dict[int, np.dtype] = {
    PointField.INT8: np.dtype(np.int8),
    PointField.UINT8: np.dtype(np.uint8),
    PointField.INT16: np.dtype(np.int16),
    PointField.UINT16: np.dtype(np.uint16),
    PointField.INT32: np.dtype(np.int32),
    PointField.UINT32: np.dtype(np.uint32),
    PointField.FLOAT32: np.dtype(np.float32),
    PointField.FLOAT64: np.dtype(np.float64),
}


@dataclass(frozen=True)
class PointClusterDetection:
    """Cone-like cluster summary in the target frame."""

    x_m: float
    y_m: float
    z_m: float
    width_m: float
    depth_m: float
    height_m: float
    point_count: int
    min_range_m: float
    max_range_m: float
    accepted: bool = True
    reason: str = ''


def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """Decode XYZ fields from a PointCloud2 message into an ``Nx3`` float32 array."""
    if msg.width == 0 or msg.height == 0 or not msg.data:
        return np.empty((0, 3), dtype=np.float32)

    dtype = _pointcloud_dtype(msg.fields, msg.point_step, msg.is_bigendian)
    if dtype is None or not {'x', 'y', 'z'}.issubset(dtype.names or ()):
        return np.empty((0, 3), dtype=np.float32)

    rows: list[np.ndarray] = []
    for row_idx in range(int(msg.height)):
        start = row_idx * int(msg.row_step)
        stop = start + (int(msg.width) * int(msg.point_step))
        row = np.frombuffer(msg.data[start:stop], dtype=dtype, count=int(msg.width))
        rows.append(row)

    structured = rows[0] if len(rows) == 1 else np.concatenate(rows, axis=0)
    xyz = np.stack(
        (
            np.asarray(structured['x'], dtype=np.float32),
            np.asarray(structured['y'], dtype=np.float32),
            np.asarray(structured['z'], dtype=np.float32),
        ),
        axis=-1,
    )
    finite_mask = np.isfinite(xyz).all(axis=1)
    return xyz[finite_mask]


def xyz_array_to_pointcloud2(points_xyz: np.ndarray, *, frame_id: str, stamp) -> PointCloud2:
    """Encode an ``Nx3`` float array as a PointCloud2 message."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height = 1
    msg.width = int(points.shape[0])
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = False
    msg.data = points.tobytes()
    return msg


def downsample_points(points_xyz: np.ndarray, stride: int) -> np.ndarray:
    """Keep every ``stride``-th point."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    step = max(1, int(stride))
    if step == 1 or points.shape[0] <= 1:
        return points
    return points[::step]


def apply_azimuth_masks(points_xyz: np.ndarray, masked_ranges_deg: Sequence[float]) -> np.ndarray:
    """Remove points whose azimuth falls within any masked angular interval."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return points
    ranges = [float(v) for v in masked_ranges_deg]
    if len(ranges) < 2:
        return points

    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    keep_mask = np.ones(points.shape[0], dtype=bool)
    pair_count = len(ranges) // 2
    for idx in range(pair_count):
        start_deg = _normalize_angle_deg(ranges[2 * idx])
        end_deg = _normalize_angle_deg(ranges[(2 * idx) + 1])
        if start_deg <= end_deg:
            keep_mask &= ~((azimuth >= start_deg) & (azimuth <= end_deg))
        else:
            keep_mask &= ~((azimuth >= start_deg) | (azimuth <= end_deg))
    return points[keep_mask]


def apply_range_thinning(
    points_xyz: np.ndarray,
    *,
    thinning_start_range_m: float,
    max_range_m: float,
    keep_ratio_at_max_range: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly thin far points to emulate less-than-ideal returns."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return points

    thinning_start = max(0.0, float(thinning_start_range_m))
    far_range = max(thinning_start + 1e-6, float(max_range_m))
    min_keep = min(1.0, max(0.0, float(keep_ratio_at_max_range)))
    ranges = np.linalg.norm(points[:, :2], axis=1)
    keep_prob = np.ones(points.shape[0], dtype=np.float32)
    far_mask = ranges > thinning_start
    if np.any(far_mask):
        alpha = np.clip((ranges[far_mask] - thinning_start) / (far_range - thinning_start), 0.0, 1.0)
        keep_prob[far_mask] = 1.0 - ((1.0 - min_keep) * alpha)
    keep_mask = rng.random(points.shape[0]) <= keep_prob
    return points[keep_mask]


def crop_points_to_roi(
    points_xyz: np.ndarray,
    *,
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    y_max_m: float,
    z_min_m: float,
    z_max_m: float,
) -> np.ndarray:
    """Crop points to a rectangular ROI."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return points
    mask = (
        (points[:, 0] >= float(x_min_m)) &
        (points[:, 0] <= float(x_max_m)) &
        (points[:, 1] >= float(y_min_m)) &
        (points[:, 1] <= float(y_max_m)) &
        (points[:, 2] >= float(z_min_m)) &
        (points[:, 2] <= float(z_max_m))
    )
    return points[mask]


def suppress_ground_points(
    points_xyz: np.ndarray,
    *,
    base_cutoff_m: float,
    range_slope_m_per_m: float,
    range_bias_m: float,
    z_max_m: float,
) -> np.ndarray:
    """Reject points that lie too close to the flat ground plane."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return points

    ranges = np.linalg.norm(points[:, :2], axis=1)
    adaptive_floor = np.maximum(
        float(base_cutoff_m),
        float(range_bias_m) + (float(range_slope_m_per_m) * np.maximum(0.0, ranges - 3.0)),
    )
    keep_mask = (points[:, 2] >= adaptive_floor) & (points[:, 2] <= float(z_max_m))
    return points[keep_mask]


def cluster_xy_points_adaptive(points_xyz: np.ndarray, max_cluster_radius_m: float) -> list[np.ndarray]:
    """Cluster points in the XY plane using range-adaptive Euclidean thresholds."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return []

    max_radius = max(1e-6, float(max_cluster_radius_m))
    cells = np.floor(points[:, :2] / max_radius).astype(np.int32)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, (cell_x, cell_y) in enumerate(cells):
        buckets.setdefault((int(cell_x), int(cell_y)), []).append(idx)

    visited = np.zeros(points.shape[0], dtype=bool)
    clusters: list[np.ndarray] = []
    for seed_idx in range(points.shape[0]):
        if visited[seed_idx]:
            continue
        visited[seed_idx] = True
        queue = [seed_idx]
        cluster = [seed_idx]
        while queue:
            point_idx = queue.pop()
            cell_x, cell_y = cells[point_idx]
            px = float(points[point_idx, 0])
            py = float(points[point_idx, 1])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor_indices = buckets.get((int(cell_x + dx), int(cell_y + dy)))
                    if not neighbor_indices:
                        continue
                    for candidate_idx in neighbor_indices:
                        if visited[candidate_idx]:
                            continue
                        qx = float(points[candidate_idx, 0])
                        qy = float(points[candidate_idx, 1])
                        mean_range = 0.5 * (
                            math.hypot(px, py) +
                            math.hypot(qx, qy)
                        )
                        radius = _cluster_radius_for_range(mean_range)
                        radius_sq = radius * radius
                        if ((px - qx) * (px - qx)) + ((py - qy) * (py - qy)) > radius_sq:
                            continue
                        visited[candidate_idx] = True
                        queue.append(candidate_idx)
                        cluster.append(candidate_idx)
        clusters.append(np.asarray(cluster, dtype=np.int32))
    return clusters


def detect_cone_like_clusters(
    points_xyz: np.ndarray,
    *,
    max_cluster_radius_m: float,
    min_cluster_points: int,
    max_cluster_points: int,
    min_cluster_width_m: float,
    max_cluster_width_m: float,
    min_cluster_depth_m: float,
    max_cluster_depth_m: float,
    min_cluster_height_m: float,
    max_cluster_height_m: float,
) -> list[PointClusterDetection]:
    """Filter XY clusters into cone-like detections with range-adaptive acceptance."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    detections: list[PointClusterDetection] = []
    min_pts = max(1, int(min_cluster_points))
    max_pts = max(min_pts, int(max_cluster_points))

    for cluster_indices in cluster_xy_points_adaptive(points, max_cluster_radius_m=max_cluster_radius_m):
        cluster = points[cluster_indices]
        count = int(cluster.shape[0])

        mins = np.min(cluster, axis=0)
        maxs = np.max(cluster, axis=0)
        x_span = float(maxs[0] - mins[0])
        y_span = float(maxs[1] - mins[1])
        width = max(x_span, y_span)
        depth = min(x_span, y_span)
        height = float(maxs[2] - mins[2])

        centroid = np.mean(cluster, axis=0)
        ranges = np.linalg.norm(cluster[:, :2], axis=1)
        centroid_range = float(math.hypot(float(centroid[0]), float(centroid[1])))
        thresholds = _acceptance_thresholds_for_range(
            centroid_range,
            min_cluster_points=min_pts,
            max_cluster_points=max_pts,
            min_cluster_width_m=float(min_cluster_width_m),
            max_cluster_width_m=float(max_cluster_width_m),
            min_cluster_depth_m=float(min_cluster_depth_m),
            max_cluster_depth_m=float(max_cluster_depth_m),
            min_cluster_height_m=float(min_cluster_height_m),
            max_cluster_height_m=float(max_cluster_height_m),
        )

        accepted, reason = _cluster_acceptance_reason(
            count=count,
            width=width,
            depth=depth,
            height=height,
            thresholds=thresholds,
        )
        detections.append(
            PointClusterDetection(
                x_m=float(centroid[0]),
                y_m=float(centroid[1]),
                z_m=float(centroid[2]),
                width_m=float(width),
                depth_m=float(depth),
                height_m=float(height),
                point_count=count,
                min_range_m=float(np.min(ranges)),
                max_range_m=float(np.max(ranges)),
                accepted=accepted,
                reason=reason,
            )
        )
    return [detection for detection in detections if detection.accepted]


def summarize_clusters_for_debug(
    points_xyz: np.ndarray,
    *,
    max_cluster_radius_m: float,
    min_cluster_points: int,
    max_cluster_points: int,
    min_cluster_width_m: float,
    max_cluster_width_m: float,
    min_cluster_depth_m: float,
    max_cluster_depth_m: float,
    min_cluster_height_m: float,
    max_cluster_height_m: float,
) -> list[PointClusterDetection]:
    """Return accepted and rejected cluster summaries for debugging."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    detections: list[PointClusterDetection] = []
    min_pts = max(1, int(min_cluster_points))
    max_pts = max(min_pts, int(max_cluster_points))
    for cluster_indices in cluster_xy_points_adaptive(points, max_cluster_radius_m=max_cluster_radius_m):
        cluster = points[cluster_indices]
        count = int(cluster.shape[0])
        mins = np.min(cluster, axis=0)
        maxs = np.max(cluster, axis=0)
        x_span = float(maxs[0] - mins[0])
        y_span = float(maxs[1] - mins[1])
        width = max(x_span, y_span)
        depth = min(x_span, y_span)
        height = float(maxs[2] - mins[2])
        centroid = np.mean(cluster, axis=0)
        ranges = np.linalg.norm(cluster[:, :2], axis=1)
        centroid_range = float(math.hypot(float(centroid[0]), float(centroid[1])))
        thresholds = _acceptance_thresholds_for_range(
            centroid_range,
            min_cluster_points=min_pts,
            max_cluster_points=max_pts,
            min_cluster_width_m=float(min_cluster_width_m),
            max_cluster_width_m=float(max_cluster_width_m),
            min_cluster_depth_m=float(min_cluster_depth_m),
            max_cluster_depth_m=float(max_cluster_depth_m),
            min_cluster_height_m=float(min_cluster_height_m),
            max_cluster_height_m=float(max_cluster_height_m),
        )
        accepted, reason = _cluster_acceptance_reason(
            count=count,
            width=width,
            depth=depth,
            height=height,
            thresholds=thresholds,
        )
        detections.append(
            PointClusterDetection(
                x_m=float(centroid[0]),
                y_m=float(centroid[1]),
                z_m=float(centroid[2]),
                width_m=float(width),
                depth_m=float(depth),
                height_m=float(height),
                point_count=count,
                min_range_m=float(np.min(ranges)),
                max_range_m=float(np.max(ranges)),
                accepted=accepted,
                reason=reason,
            )
        )
    return detections


def _pointcloud_dtype(fields: Iterable[PointField], point_step: int, is_bigendian: bool) -> np.dtype | None:
    sorted_fields = sorted(fields, key=lambda field: int(field.offset))
    if not sorted_fields:
        return None

    dtype_fields: list[tuple] = []
    offset = 0
    byteorder = '>' if is_bigendian else '<'
    for field in sorted_fields:
        base_dtype = _POINTFIELD_DTYPES.get(int(field.datatype))
        if base_dtype is None:
            continue
        if int(field.offset) > offset:
            dtype_fields.append((f'__pad{offset}', f'V{int(field.offset) - offset}'))
            offset = int(field.offset)
        typed = base_dtype.newbyteorder(byteorder)
        if int(field.count) > 1:
            dtype_fields.append((field.name, typed, (int(field.count),)))
            offset += typed.itemsize * int(field.count)
        else:
            dtype_fields.append((field.name, typed))
            offset += typed.itemsize
    if int(point_step) > offset:
        dtype_fields.append((f'__pad{offset}', f'V{int(point_step) - offset}'))
    if not dtype_fields:
        return None
    return np.dtype(dtype_fields)


def _normalize_angle_deg(angle_deg: float) -> float:
    normalized = math.fmod(float(angle_deg), 360.0)
    if normalized <= -180.0:
        normalized += 360.0
    elif normalized > 180.0:
        normalized -= 360.0
    return normalized


def _cluster_radius_for_range(range_m: float) -> float:
    if range_m < 6.0:
        return 0.18
    if range_m < 12.0:
        return 0.24
    return 0.30


def _acceptance_thresholds_for_range(
    range_m: float,
    *,
    min_cluster_points: int,
    max_cluster_points: int,
    min_cluster_width_m: float,
    max_cluster_width_m: float,
    min_cluster_depth_m: float,
    max_cluster_depth_m: float,
    min_cluster_height_m: float,
    max_cluster_height_m: float,
) -> dict[str, float]:
    if range_m < 5.0:
        return {
            'min_points': max(3, min_cluster_points if min_cluster_points < 3 else 3),
            'max_points': max_cluster_points,
            'min_width': min_cluster_width_m,
            'max_width': max_cluster_width_m,
            'min_depth': min_cluster_depth_m,
            'max_depth': max_cluster_depth_m,
            'min_height': min_cluster_height_m,
            'max_height': max_cluster_height_m,
        }
    if range_m < 10.0:
        return {
            'min_points': max(2, min_cluster_points if min_cluster_points < 2 else 2),
            'max_points': max_cluster_points,
            'min_width': 0.02,
            'max_width': max(max_cluster_width_m, 0.70),
            'min_depth': 0.01,
            'max_depth': max_cluster_depth_m,
            'min_height': 0.08,
            'max_height': max(max_cluster_height_m, 0.75),
        }
    return {
        'min_points': max(2, min_cluster_points if min_cluster_points < 2 else 2),
        'max_points': max_cluster_points,
        'min_width': 0.01,
        'max_width': max(max_cluster_width_m, 0.85),
        'min_depth': 0.005,
        'max_depth': max_cluster_depth_m,
        'min_height': 0.05,
        'max_height': max(max_cluster_height_m, 0.90),
    }


def _cluster_acceptance_reason(
    *,
    count: int,
    width: float,
    depth: float,
    height: float,
    thresholds: dict[str, float],
) -> tuple[bool, str]:
    if count < int(thresholds['min_points']):
        return False, 'too_few_points'
    if count > int(thresholds['max_points']):
        return False, 'too_many_points'
    if width < float(thresholds['min_width']):
        return False, 'too_narrow'
    if width > float(thresholds['max_width']):
        return False, 'too_wide'
    if depth < float(thresholds['min_depth']):
        return False, 'too_shallow'
    if depth > float(thresholds['max_depth']):
        return False, 'too_deep'
    if height < float(thresholds['min_height']):
        return False, 'too_short'
    if height > float(thresholds['max_height']):
        return False, 'too_tall'
    return True, ''


def summarize_rejection_reasons(clusters: Sequence[PointClusterDetection]) -> dict[str, int]:
    """Count cluster rejection reasons for low-rate logging."""
    return dict(Counter(cluster.reason for cluster in clusters if not cluster.accepted and cluster.reason))
