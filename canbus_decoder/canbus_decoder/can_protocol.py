#!/usr/bin/env python3
"""
CAN Protocol Definitions and Decoders

Defines the CAN message format for vehicle wheel encoders and
provides decoding functions.

CAN IDs:
- 0xB9 (185): Front axle (FL, FR wheels)
- 0xBA (186): Rear axle (RL, RR wheels)

Byte Layout (8 bytes):
- [0-1]: Wheel 1 velocity (16-bit unsigned, mm/s, little-endian)
- [2-3]: Wheel 2 velocity (16-bit unsigned, mm/s, little-endian)
- [4]: Suspension WSS 1 (8-bit unsigned, mm)
- [5]: Suspension WSS 2 (8-bit unsigned, mm)
- [6-7]: Steering angle (16-bit signed, little-endian, 0.1 deg/LSB)
         Only valid for front axle (ID 185)
"""

import struct
from typing import Optional
from dataclasses import dataclass
import math

# CAN ID definitions
CAN_ID_FRONT_AXLE = 0xB9  # 185 decimal
CAN_ID_REAR_AXLE = 0xBA   # 186 decimal

# Expected DLC (Data Length Code)
EXPECTED_DLC = 8


@dataclass
class WheelPairData:
    """Decoded data for a wheel pair (front or rear axle)."""
    wheel_1_velocity_mms: int      # mm/s
    wheel_2_velocity_mms: int      # mm/s
    suspension_1_mm: int           # mm
    suspension_2_mm: int           # mm
    steering_angle_decideg: Optional[int] = None  # 0.1 deg, None for rear

    @property
    def wheel_1_velocity_ms(self) -> float:
        """Wheel 1 velocity in m/s."""
        return self.wheel_1_velocity_mms / 1000.0

    @property
    def wheel_2_velocity_ms(self) -> float:
        """Wheel 2 velocity in m/s."""
        return self.wheel_2_velocity_mms / 1000.0

    @property
    def suspension_1_m(self) -> float:
        """Suspension 1 displacement in meters."""
        return self.suspension_1_mm / 1000.0

    @property
    def suspension_2_m(self) -> float:
        """Suspension 2 displacement in meters."""
        return self.suspension_2_mm / 1000.0

    @property
    def steering_angle_rad(self) -> Optional[float]:
        """Steering angle in radians."""
        if self.steering_angle_decideg is None:
            return None
        # Convert decidegrees to degrees, then to radians
        degrees = self.steering_angle_decideg * 0.1
        return math.radians(degrees)

    @property
    def steering_angle_deg(self) -> Optional[float]:
        """Steering angle in degrees."""
        if self.steering_angle_decideg is None:
            return None
        return self.steering_angle_decideg * 0.1


def decode_wheel_pair(can_id: int, data: bytes) -> Optional[WheelPairData]:
    """
    Decode CAN frame for wheel pair.

    Args:
        can_id: CAN ID (185 or 186)
        data: 8-byte CAN data payload

    Returns:
        WheelPairData if valid, None if invalid
    """
    # Validate CAN ID
    if can_id not in [CAN_ID_FRONT_AXLE, CAN_ID_REAR_AXLE]:
        return None

    # Validate data length
    if len(data) < EXPECTED_DLC:
        return None

    try:
        # Decode using struct (little-endian format)
        # H = unsigned short (16-bit), B = unsigned char (8-bit), h = signed short
        wheel_1_vel, wheel_2_vel, susp_1, susp_2, steer_raw = struct.unpack(
            '<HHBBh', data[:8]
        )

        # Steering angle only valid for front axle
        steering = steer_raw if can_id == CAN_ID_FRONT_AXLE else None

        return WheelPairData(
            wheel_1_velocity_mms=wheel_1_vel,
            wheel_2_velocity_mms=wheel_2_vel,
            suspension_1_mm=susp_1,
            suspension_2_mm=susp_2,
            steering_angle_decideg=steering,
        )

    except struct.error:
        # Handle unpacking errors
        return None


@dataclass
class VehicleCANData:
    """Complete vehicle CAN data from both axles."""
    # Wheel velocities [FL, FR, RL, RR] in m/s
    wheel_velocities: list[float]

    # Suspension displacements [FL, FR, RL, RR] in meters
    suspension_displacements: list[float]

    # Steering angle in radians (None if not available)
    steering_angle: Optional[float] = None

    # Flags for data validity
    front_valid: bool = False
    rear_valid: bool = False

    @property
    def all_valid(self) -> bool:
        """Check if both front and rear axle data are valid."""
        return self.front_valid and self.rear_valid


def combine_axle_data(
    front_data: Optional[WheelPairData],
    rear_data: Optional[WheelPairData],
) -> VehicleCANData:
    """
    Combine front and rear axle data into complete vehicle data.

    Args:
        front_data: Front axle wheel pair data (FL, FR)
        rear_data: Rear axle wheel pair data (RL, RR)

    Returns:
        VehicleCANData with combined information
    """
    # Initialize with zeros
    wheel_vels = [0.0, 0.0, 0.0, 0.0]  # [FL, FR, RL, RR]
    suspensions = [0.0, 0.0, 0.0, 0.0]  # [FL, FR, RL, RR]
    steering = None

    front_valid = front_data is not None
    rear_valid = rear_data is not None

    # Fill in front axle data (FL, FR)
    if front_data is not None:
        wheel_vels[0] = front_data.wheel_1_velocity_ms  # FL
        wheel_vels[1] = front_data.wheel_2_velocity_ms  # FR
        suspensions[0] = front_data.suspension_1_m      # FL
        suspensions[1] = front_data.suspension_2_m      # FR
        steering = front_data.steering_angle_rad

    # Fill in rear axle data (RL, RR)
    if rear_data is not None:
        wheel_vels[2] = rear_data.wheel_1_velocity_ms  # RL
        wheel_vels[3] = rear_data.wheel_2_velocity_ms  # RR
        suspensions[2] = rear_data.suspension_1_m      # RL
        suspensions[3] = rear_data.suspension_2_m      # RR

    return VehicleCANData(
        wheel_velocities=wheel_vels,
        suspension_displacements=suspensions,
        steering_angle=steering,
        front_valid=front_valid,
        rear_valid=rear_valid,
    )
