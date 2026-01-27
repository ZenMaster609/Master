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

    # Wheel encoder speeds [FL, FR, RL, RR] in RPM
    encoder_velocities: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # Wheel encoder angle accumulation [FL, FR, RL, RR] in radians (per RPM window)
    encoder_angle_accum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # Raw GPS (for reference, optional)
    gps_latitude: float = 0.0
    gps_longitude: float = 0.0
    gps_altitude: float = 0.0
    gps_valid: bool = False

    # GPS converted to local coordinates
    gps_local_x: float = 0.0
    gps_local_y: float = 0.0

    # INS fused position (explicit naming)
    ins_x: float = 0.0
    ins_y: float = 0.0

    # Dead-reckoning position (IMU-only integration)
    dr_x: float = 0.0
    dr_y: float = 0.0

    # Suspension displacements [FL, FR, RL, RR] in meters
    suspension: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # Steering angle (degrees)
    steering_angle: float = 0.0
    steering_valid: bool = False

    # Cooling system
    water_pressure: float = 0.0        # bar
    water_flow: float = 0.0            # L/min
    water_temp_in: float = 0.0         # celsius
    water_temp_out: float = 0.0        # celsius
    water_temp_radiator: float = 0.0   # celsius

    # Brake temperatures
    brake_temp_fr: float = 0.0         # celsius
    brake_temp_rl: float = 0.0         # celsius

    # Pitot tube
    pitot_dynamic_pressure: float = 0.0  # Pa

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

        Flattens encoder and suspension arrays for easier DataFrame handling.
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
            'encoder_angle_fl': self.encoder_angle_accum[0],
            'encoder_angle_fr': self.encoder_angle_accum[1],
            'encoder_angle_rl': self.encoder_angle_accum[2],
            'encoder_angle_rr': self.encoder_angle_accum[3],
            'gps_latitude': self.gps_latitude,
            'gps_longitude': self.gps_longitude,
            'gps_altitude': self.gps_altitude,
            'gps_valid': self.gps_valid,
            'gps_local_x': self.gps_local_x,
            'gps_local_y': self.gps_local_y,
            'ins_x': self.ins_x,
            'ins_y': self.ins_y,
            'dr_x': self.dr_x,
            'dr_y': self.dr_y,
            'suspension_fl': self.suspension[0],
            'suspension_fr': self.suspension[1],
            'suspension_rl': self.suspension[2],
            'suspension_rr': self.suspension[3],
            'steering_angle': self.steering_angle,
            'steering_valid': self.steering_valid,
            'water_pressure': self.water_pressure,
            'water_flow': self.water_flow,
            'water_temp_in': self.water_temp_in,
            'water_temp_out': self.water_temp_out,
            'water_temp_radiator': self.water_temp_radiator,
            'brake_temp_fr': self.brake_temp_fr,
            'brake_temp_rl': self.brake_temp_rl,
            'pitot_dynamic_pressure': self.pitot_dynamic_pressure,
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
        msg.encoder_angle_accum = [float(v) for v in self.encoder_angle_accum]

        # GPS
        msg.gps_latitude = self.gps_latitude
        msg.gps_longitude = self.gps_longitude
        msg.gps_altitude = self.gps_altitude
        msg.gps_valid = self.gps_valid

        # GPS local coordinates
        msg.gps_local_x = self.gps_local_x
        msg.gps_local_y = self.gps_local_y

        # INS position
        msg.ins_x = self.ins_x
        msg.ins_y = self.ins_y

        # Dead-reckoning position
        msg.dr_x = self.dr_x
        msg.dr_y = self.dr_y

        # Suspension
        msg.suspension = [float(v) for v in self.suspension]

        # Steering
        msg.steering_angle = float(self.steering_angle)
        msg.steering_valid = self.steering_valid

        # Cooling system
        msg.water_pressure = float(self.water_pressure)
        msg.water_flow = float(self.water_flow)
        msg.water_temp_in = float(self.water_temp_in)
        msg.water_temp_out = float(self.water_temp_out)
        msg.water_temp_radiator = float(self.water_temp_radiator)

        # Brake temperatures
        msg.brake_temp_fr = float(self.brake_temp_fr)
        msg.brake_temp_rl = float(self.brake_temp_rl)

        # Pitot
        msg.pitot_dynamic_pressure = float(self.pitot_dynamic_pressure)

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
            encoder_angle_accum=list(msg.encoder_angle_accum),
            gps_latitude=msg.gps_latitude,
            gps_longitude=msg.gps_longitude,
            gps_altitude=msg.gps_altitude,
            gps_valid=msg.gps_valid,
            gps_local_x=msg.gps_local_x,
            gps_local_y=msg.gps_local_y,
            ins_x=msg.ins_x,
            ins_y=msg.ins_y,
            dr_x=msg.dr_x,
            dr_y=msg.dr_y,
            suspension=list(msg.suspension),
            steering_angle=msg.steering_angle,
            steering_valid=msg.steering_valid,
            water_pressure=msg.water_pressure,
            water_flow=msg.water_flow,
            water_temp_in=msg.water_temp_in,
            water_temp_out=msg.water_temp_out,
            water_temp_radiator=msg.water_temp_radiator,
            brake_temp_fr=msg.brake_temp_fr,
            brake_temp_rl=msg.brake_temp_rl,
            pitot_dynamic_pressure=msg.pitot_dynamic_pressure,
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
            encoder_angle_accum=list(self.encoder_angle_accum),
            gps_latitude=self.gps_latitude,
            gps_longitude=self.gps_longitude,
            gps_altitude=self.gps_altitude,
            gps_valid=self.gps_valid,
            gps_local_x=self.gps_local_x,
            gps_local_y=self.gps_local_y,
            ins_x=self.ins_x,
            ins_y=self.ins_y,
            dr_x=self.dr_x,
            dr_y=self.dr_y,
            suspension=list(self.suspension),
            steering_angle=self.steering_angle,
            steering_valid=self.steering_valid,
            water_pressure=self.water_pressure,
            water_flow=self.water_flow,
            water_temp_in=self.water_temp_in,
            water_temp_out=self.water_temp_out,
            water_temp_radiator=self.water_temp_radiator,
            brake_temp_fr=self.brake_temp_fr,
            brake_temp_rl=self.brake_temp_rl,
            pitot_dynamic_pressure=self.pitot_dynamic_pressure,
            covariance=list(self.covariance),
            estimation_status=self.estimation_status,
            source_adapter=self.source_adapter,
        )
