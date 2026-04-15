"""Visualization marker builders for the cone memory node.

All functions here are pure (no Node/self dependency). They accept the data
they need explicitly so they can be tested and reused independently.
"""

from __future__ import annotations

from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cones.nodes.memory_types import SensorDetection
from sim_car.cones.tracking.tracker import (
    TRACK_STATE_CONFIRMED,
    TRACK_STATE_STALE,
    TRACK_STATE_TENTATIVE,
    ConeTrack,
)


# ---------------------------------------------------------------------------
# Color / state helpers
# ---------------------------------------------------------------------------

def cone_color_rgb(label: str) -> tuple[float, float, float]:
    """Return (r, g, b) in [0, 1] for a cone color label."""
    if label == 'blue':
        return 0.15, 0.45, 1.0
    if label == 'yellow':
        return 1.0, 0.92, 0.15
    if label == 'orange':
        return 1.0, 0.45, 0.05
    return 0.75, 0.75, 0.75


def track_state_namespace(track_state: int) -> str:
    """Return the rviz namespace string for a track state."""
    if int(track_state) == TRACK_STATE_CONFIRMED:
        return 'confirmed'
    if int(track_state) == TRACK_STATE_STALE:
        return 'stale'
    return 'tentative'


def track_state_alpha(track_state: int) -> float:
    """Return the marker alpha value for a track state."""
    if int(track_state) == TRACK_STATE_CONFIRMED:
        return 0.95
    if int(track_state) == TRACK_STATE_STALE:
        return 0.55
    return 0.35


def track_state_id_offset(track_state: int) -> int:
    """Return the marker ID offset for a track state (avoids id collisions)."""
    if int(track_state) == TRACK_STATE_CONFIRMED:
        return 50_000
    if int(track_state) == TRACK_STATE_STALE:
        return 75_000
    return 100_000


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def split_polyline_by_gap(
    points: list[tuple[float, float]],
    max_gap_m: float,
) -> list[list[tuple[float, float]]]:
    """Split *points* into contiguous sub-lists wherever the gap exceeds *max_gap_m*."""
    if len(points) <= 1:
        return [list(points)] if points else []
    if max_gap_m <= 0.0:
        return [list(points)]

    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [points[0]]
    max_gap_sq = float(max_gap_m) * float(max_gap_m)

    for prev, curr in zip(points, points[1:]):
        dx = float(curr[0] - prev[0])
        dy = float(curr[1] - prev[1])
        if (dx * dx) + (dy * dy) > max_gap_sq:
            if current:
                segments.append(current)
            current = [curr]
            continue
        current.append(curr)

    if current:
        segments.append(current)
    return segments


# ---------------------------------------------------------------------------
# Individual marker builders
# ---------------------------------------------------------------------------

def make_cone_marker(
    track: ConeTrack,
    now: Time,
    *,
    odom_frame: str,
    publish_rate_hz: float,
) -> Marker:
    """Build a CYLINDER marker for a single cone track."""
    label, _conf = track.class_label()
    r, g, b = cone_color_rgb(label)
    marker = Marker()
    marker.header.frame_id = odom_frame
    marker.header.stamp = now.to_msg()
    marker.ns = track_state_namespace(track.track_state)
    marker.id = track.track_id
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD
    marker.pose.position.x = float(track.x)
    marker.pose.position.y = float(track.y)
    marker.pose.position.z = float(track.z) + 0.15
    marker.pose.orientation.w = 1.0
    scale = 0.22 if track.track_state == TRACK_STATE_TENTATIVE else 0.30
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = 0.30
    marker.color.a = track_state_alpha(track.track_state)
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.lifetime = Duration(seconds=1.0 / max(1.0, publish_rate_hz) * 2.0).to_msg()
    return marker


def make_id_marker(
    track: ConeTrack,
    now: Time,
    *,
    odom_frame: str,
    publish_rate_hz: float,
) -> Marker:
    """Build a TEXT_VIEW_FACING marker showing the track id."""
    marker = Marker()
    marker.header.frame_id = odom_frame
    marker.header.stamp = now.to_msg()
    marker.ns = f'{track_state_namespace(track.track_state)}_id'
    marker.id = track.track_id + track_state_id_offset(track.track_state)
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position.x = float(track.x)
    marker.pose.position.y = float(track.y)
    marker.pose.position.z = float(track.z) + 0.55
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.18
    marker.color.a = min(1.0, track_state_alpha(track.track_state) + 0.15)
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.text = (
        f'{track.track_id} s={track.seen_count} m={track.missed_count}'
        if track.track_state != TRACK_STATE_CONFIRMED
        else str(track.track_id)
    )
    marker.lifetime = Duration(seconds=1.0 / max(1.0, publish_rate_hz) * 2.0).to_msg()
    return marker


def make_line_marker(
    now: Time,
    *,
    marker_id: int,
    ns: str,
    points: list[tuple[float, float]],
    odom_frame: str,
    rgb: tuple[float, float, float],
) -> Marker:
    """Build a LINE_STRIP marker from a list of (x, y) points."""
    marker = Marker()
    marker.header.frame_id = odom_frame
    marker.header.stamp = now.to_msg()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.10
    marker.color.a = 0.95
    marker.color.r = float(rgb[0])
    marker.color.g = float(rgb[1])
    marker.color.b = float(rgb[2])
    marker.pose.orientation.w = 1.0
    for x, y in points:
        pt = Point()
        pt.x = float(x)
        pt.y = float(y)
        pt.z = 0.02
        marker.points.append(pt)
    return marker


def make_raw_sensor_markers(
    detections: list[SensorDetection],
    now: Time,
    *,
    odom_frame: str,
    ns: str,
    scale: float,
    rgba: tuple[float, float, float, float],
) -> MarkerArray:
    """Build a MarkerArray of SPHERE markers for raw sensor detections."""
    arr = MarkerArray()
    clear = Marker()
    clear.header.frame_id = odom_frame
    clear.header.stamp = now.to_msg()
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)

    r, g, b, a = rgba
    for idx, det in enumerate(detections):
        marker = Marker()
        marker.header.frame_id = odom_frame
        marker.header.stamp = now.to_msg()
        marker.ns = ns
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = det.x_odom
        marker.pose.position.y = det.y_odom
        marker.pose.position.z = det.z_odom + 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = a
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        arr.markers.append(marker)

    return arr
