"""
VehicleState dataclass - canonical vehicle state representation.

This mirrors the VehicleState.msg ROS message and provides convenient
Python methods for state manipulation, logging, and message conversion.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import math


@dataclass
class VehicleState:
    """
    Canonical vehicle state representation.

    This is the unified state that all adapters produce and all
    consumers (plotter, logger) expect. Designed for easy extension
    when EKF/UKF is added later.

    Coordinate Frames:
        - Position (x, y): Local frame, origin at GPS origin or sim start
        - Velocity (vx, vy): Body frame, vx = forward, vy = lateral
        - Yaw: Counter-clockwise from local X-axis (East)
    """

    # Timestamp (ROS time in seconds since epoch)
    timestamp: float = 0.0

    # Position (local frame, meters)
    x: float = 0.0
    y: float = 0.0

    # Velocity (body frame, m/s)
    vx: float = 0.0
    vy: float = 0.0

    # Orientation
    yaw: float = 0.0          # radians, -pi to pi
    yaw_rate: float = 0.0     # rad/s

    # Derived quantities
    speed: float = 0.0                          # sqrt(vx^2 + vy^2)
    distance_traveled: float = 0.0              # cumulative (meters)

    # Slip (optional, requires wheel velocity vs body velocity)
    slip_longitudinal: float = 0.0   # (wheel_vel - body_vel) / body_vel
    slip_lateral: float = 0.0        # atan(vy / vx) - sideslip angle

    # Wheel encoder velocities [FL, FR, RL, RR] in m/s
    encoder_velocities: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # Raw GPS (for reference, optional)
    gps_latitude: float = 0.0
    gps_longitude: float = 0.0
    gps_altitude: float = 0.0
    gps_valid: bool = False

    # Source information (for debugging)
    source_adapter: str = "unknown"

    # Reserved for future EKF integration
    covariance: List[float] = field(default_factory=lambda: [0.0] * 36)  # 6x6 state covariance
    estimation_status: str = "raw"  # "raw", "filtered", "predicted"

    def __post_init__(self):
        """Compute derived quantities after initialization."""
        self.speed = math.sqrt(self.vx**2 + self.vy**2)

    def update_speed(self) -> None:
        """Recompute speed from velocity components."""
        self.speed = math.sqrt(self.vx**2 + self.vy**2)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for logging.

        Flattens encoder arrays for easier DataFrame handling.
        """
        return {
            'timestamp': self.timestamp,
            'x': self.x,
            'y': self.y,
            'vx': self.vx,
            'vy': self.vy,
            'yaw': self.yaw,
            'yaw_rate': self.yaw_rate,
            'speed': self.speed,
            'distance_traveled': self.distance_traveled,
            'slip_longitudinal': self.slip_longitudinal,
            'slip_lateral': self.slip_lateral,
            'encoder_fl': self.encoder_velocities[0],
            'encoder_fr': self.encoder_velocities[1],
            'encoder_rl': self.encoder_velocities[2],
            'encoder_rr': self.encoder_velocities[3],
            'gps_latitude': self.gps_latitude,
            'gps_longitude': self.gps_longitude,
            'gps_altitude': self.gps_altitude,
            'gps_valid': self.gps_valid,
            'estimation_status': self.estimation_status,
            'source_adapter': self.source_adapter,
        }

    def to_msg(self):
        """
        Convert to ROS VehicleState message.

        Note: Import inside function to avoid circular imports
        and allow use without ROS.
        """
        from vehicle_plotter_msgs.msg import VehicleState as VehicleStateMsg
        from std_msgs.msg import Header
        from builtin_interfaces.msg import Time

        msg = VehicleStateMsg()

        # Header
        msg.header = Header()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        msg.header.stamp = Time(sec=sec, nanosec=nanosec)
        msg.header.frame_id = 'odom'

        # Position
        msg.x = self.x
        msg.y = self.y

        # Velocity
        msg.vx = self.vx
        msg.vy = self.vy

        # Orientation
        msg.yaw = self.yaw
        msg.yaw_rate = self.yaw_rate

        # Derived
        msg.speed = self.speed
        msg.distance_traveled = self.distance_traveled

        # Slip
        msg.slip_longitudinal = self.slip_longitudinal
        msg.slip_lateral = self.slip_lateral

        # Encoders
        msg.encoder_velocities = [float(v) for v in self.encoder_velocities]

        # GPS
        msg.gps_latitude = self.gps_latitude
        msg.gps_longitude = self.gps_longitude
        msg.gps_altitude = self.gps_altitude
        msg.gps_valid = self.gps_valid

        # EKF fields
        msg.covariance = list(self.covariance)
        msg.estimation_status = self.estimation_status
        msg.source_adapter = self.source_adapter

        return msg

    @classmethod
    def from_msg(cls, msg) -> 'VehicleState':
        """
        Create VehicleState from ROS VehicleState message.
        """
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        return cls(
            timestamp=timestamp,
            x=msg.x,
            y=msg.y,
            vx=msg.vx,
            vy=msg.vy,
            yaw=msg.yaw,
            yaw_rate=msg.yaw_rate,
            speed=msg.speed,
            distance_traveled=msg.distance_traveled,
            slip_longitudinal=msg.slip_longitudinal,
            slip_lateral=msg.slip_lateral,
            encoder_velocities=list(msg.encoder_velocities),
            gps_latitude=msg.gps_latitude,
            gps_longitude=msg.gps_longitude,
            gps_altitude=msg.gps_altitude,
            gps_valid=msg.gps_valid,
            covariance=list(msg.covariance),
            estimation_status=msg.estimation_status,
            source_adapter=msg.source_adapter,
        )

    def copy(self) -> 'VehicleState':
        """Create a deep copy of this state."""
        return VehicleState(
            timestamp=self.timestamp,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            yaw=self.yaw,
            yaw_rate=self.yaw_rate,
            speed=self.speed,
            distance_traveled=self.distance_traveled,
            slip_longitudinal=self.slip_longitudinal,
            slip_lateral=self.slip_lateral,
            encoder_velocities=list(self.encoder_velocities),
            gps_latitude=self.gps_latitude,
            gps_longitude=self.gps_longitude,
            gps_altitude=self.gps_altitude,
            gps_valid=self.gps_valid,
            covariance=list(self.covariance),
            estimation_status=self.estimation_status,
            source_adapter=self.source_adapter,
        )
