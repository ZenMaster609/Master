"""
Coordinate transforms and conversions for vehicle_plotter.

Includes:
- Quaternion to Euler angle conversion
- GPS to local coordinate conversion
- Angle normalization utilities
"""

import math
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GPSOrigin:
    """
    GPS origin for local coordinate conversion.

    Used to convert GPS (lat, lon) to local (x, y) coordinates
    using a simple equirectangular projection.
    """
    latitude: float   # degrees
    longitude: float  # degrees
    altitude: float = 0.0  # meters (optional)

    # Earth radius in meters (WGS84 equatorial)
    EARTH_RADIUS_M = 6378137.0


# Module-level GPS origin (set once at startup)
_gps_origin: Optional[GPSOrigin] = None


def set_gps_origin(lat: float, lon: float, alt: float = 0.0) -> None:
    """
    Set the GPS origin for local coordinate conversion.

    Args:
        lat: Origin latitude in degrees
        lon: Origin longitude in degrees
        alt: Origin altitude in meters (optional)
    """
    global _gps_origin
    _gps_origin = GPSOrigin(latitude=lat, longitude=lon, altitude=alt)


def get_gps_origin() -> Optional[GPSOrigin]:
    """Get the current GPS origin."""
    return _gps_origin


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """
    Convert quaternion to yaw angle (rotation around Z-axis).

    Assumes standard ROS convention:
    - Quaternion (x, y, z, w) represents rotation
    - Z-axis points up
    - Yaw is rotation around Z-axis

    Args:
        x, y, z, w: Quaternion components

    Returns:
        Yaw angle in radians, range [-pi, pi]
    """
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Uses ZYX convention (yaw-pitch-roll).

    Args:
        x, y, z, w: Quaternion components

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def gps_to_local(lat: float, lon: float, origin: Optional[GPSOrigin] = None) -> Tuple[float, float]:
    """
    Convert GPS coordinates to local (x, y) coordinates.

    Uses equirectangular projection centered at the origin.
    Accurate for small areas (< 10 km from origin).

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        origin: GPS origin (uses module-level origin if None)

    Returns:
        Tuple of (x, y) in meters (East, North)

    Raises:
        ValueError: If no origin is set
    """
    if origin is None:
        origin = _gps_origin

    if origin is None:
        raise ValueError("GPS origin not set. Call set_gps_origin() first.")

    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    origin_lat_rad = math.radians(origin.latitude)
    origin_lon_rad = math.radians(origin.longitude)

    # Equirectangular projection
    # x = East, y = North
    x = GPSOrigin.EARTH_RADIUS_M * (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad)
    y = GPSOrigin.EARTH_RADIUS_M * (lat_rad - origin_lat_rad)

    return x, y


def local_to_gps(x: float, y: float, origin: Optional[GPSOrigin] = None) -> Tuple[float, float]:
    """
    Convert local (x, y) coordinates to GPS.

    Inverse of gps_to_local().

    Args:
        x: East coordinate in meters
        y: North coordinate in meters
        origin: GPS origin (uses module-level origin if None)

    Returns:
        Tuple of (latitude, longitude) in degrees

    Raises:
        ValueError: If no origin is set
    """
    if origin is None:
        origin = _gps_origin

    if origin is None:
        raise ValueError("GPS origin not set. Call set_gps_origin() first.")

    origin_lat_rad = math.radians(origin.latitude)
    origin_lon_rad = math.radians(origin.longitude)

    # Inverse equirectangular projection
    lat_rad = y / GPSOrigin.EARTH_RADIUS_M + origin_lat_rad
    lon_rad = x / (GPSOrigin.EARTH_RADIUS_M * math.cos(origin_lat_rad)) + origin_lon_rad

    return math.degrees(lat_rad), math.degrees(lon_rad)


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to range [-pi, pi].

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in radians
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Compute the shortest angular difference between two angles.

    Args:
        angle1: First angle in radians
        angle2: Second angle in radians

    Returns:
        Shortest angular difference in radians [-pi, pi]
    """
    diff = angle1 - angle2
    return normalize_angle(diff)


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi
