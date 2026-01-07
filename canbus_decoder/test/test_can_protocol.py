#!/usr/bin/env python3
"""
Unit tests for CAN protocol decoding.

Tests the can_protocol module's ability to decode wheel encoder CAN messages.
"""

import pytest
import struct
import math
from canbus_decoder.can_protocol import (
    decode_wheel_pair,
    combine_axle_data,
    WheelPairData,
    CAN_ID_FRONT_AXLE,
    CAN_ID_REAR_AXLE,
    EXPECTED_DLC,
)


class TestWheelPairData:
    """Test WheelPairData dataclass and its properties."""

    def test_unit_conversions(self):
        """Test unit conversion properties."""
        data = WheelPairData(
            wheel_1_velocity_mms=1000,  # 1000 mm/s = 1.0 m/s
            wheel_2_velocity_mms=2500,  # 2500 mm/s = 2.5 m/s
            suspension_1_mm=100,        # 100 mm = 0.1 m
            suspension_2_mm=50,         # 50 mm = 0.05 m
            steering_angle_decideg=450, # 450 decideg = 45.0 deg
        )

        # Velocity conversions
        assert data.wheel_1_velocity_ms == pytest.approx(1.0)
        assert data.wheel_2_velocity_ms == pytest.approx(2.5)

        # Suspension conversions
        assert data.suspension_1_m == pytest.approx(0.1)
        assert data.suspension_2_m == pytest.approx(0.05)

        # Steering angle conversions
        assert data.steering_angle_deg == pytest.approx(45.0)
        assert data.steering_angle_rad == pytest.approx(math.radians(45.0))

    def test_none_steering_angle(self):
        """Test that rear axle has None steering angle."""
        data = WheelPairData(
            wheel_1_velocity_mms=1000,
            wheel_2_velocity_mms=1000,
            suspension_1_mm=50,
            suspension_2_mm=50,
            steering_angle_decideg=None,
        )

        assert data.steering_angle_decideg is None
        assert data.steering_angle_deg is None
        assert data.steering_angle_rad is None


class TestDecodeWheelPair:
    """Test decode_wheel_pair function."""

    def test_decode_front_axle_valid(self):
        """Test decoding valid front axle message."""
        # Create test data: FL=10mm/s, FR=20mm/s, susp=5mm/3mm, steer=100 decideg
        data = struct.pack('<HHBBh', 10, 20, 5, 3, 100)

        result = decode_wheel_pair(CAN_ID_FRONT_AXLE, data)

        assert result is not None
        assert result.wheel_1_velocity_mms == 10
        assert result.wheel_2_velocity_mms == 20
        assert result.suspension_1_mm == 5
        assert result.suspension_2_mm == 3
        assert result.steering_angle_decideg == 100

    def test_decode_rear_axle_valid(self):
        """Test decoding valid rear axle message."""
        # Create test data: RL=30mm/s, RR=40mm/s, susp=10mm/8mm, steer ignored
        data = struct.pack('<HHBBh', 30, 40, 10, 8, 999)  # steer value ignored

        result = decode_wheel_pair(CAN_ID_REAR_AXLE, data)

        assert result is not None
        assert result.wheel_1_velocity_mms == 30
        assert result.wheel_2_velocity_mms == 40
        assert result.suspension_1_mm == 10
        assert result.suspension_2_mm == 8
        assert result.steering_angle_decideg is None  # Rear axle, should be None

    def test_decode_negative_steering_angle(self):
        """Test decoding negative steering angle (left turn)."""
        # Negative steering angle (signed 16-bit)
        data = struct.pack('<HHBBh', 100, 100, 50, 50, -300)

        result = decode_wheel_pair(CAN_ID_FRONT_AXLE, data)

        assert result is not None
        assert result.steering_angle_decideg == -300
        assert result.steering_angle_deg == pytest.approx(-30.0)
        assert result.steering_angle_rad == pytest.approx(math.radians(-30.0))

    def test_decode_invalid_can_id(self):
        """Test decoding with invalid CAN ID."""
        data = struct.pack('<HHBBh', 10, 20, 5, 3, 100)

        result = decode_wheel_pair(0x123, data)  # Invalid ID

        assert result is None

    def test_decode_malformed_data_short(self):
        """Test decoding with insufficient data length."""
        data = b'\x0A\x00\x14\x00'  # Only 4 bytes, need 8

        result = decode_wheel_pair(CAN_ID_FRONT_AXLE, data)

        assert result is None

    def test_decode_malformed_data_empty(self):
        """Test decoding with empty data."""
        data = b''

        result = decode_wheel_pair(CAN_ID_FRONT_AXLE, data)

        assert result is None

    def test_decode_max_values(self):
        """Test decoding maximum unsigned values."""
        # Max unsigned 16-bit: 65535, max unsigned 8-bit: 255
        data = struct.pack('<HHBBh', 65535, 65535, 255, 255, 32767)

        result = decode_wheel_pair(CAN_ID_FRONT_AXLE, data)

        assert result is not None
        assert result.wheel_1_velocity_mms == 65535
        assert result.wheel_2_velocity_mms == 65535
        assert result.suspension_1_mm == 255
        assert result.suspension_2_mm == 255
        assert result.steering_angle_decideg == 32767

    def test_decode_zero_values(self):
        """Test decoding all-zero message."""
        data = struct.pack('<HHBBh', 0, 0, 0, 0, 0)

        result = decode_wheel_pair(CAN_ID_FRONT_AXLE, data)

        assert result is not None
        assert result.wheel_1_velocity_mms == 0
        assert result.wheel_2_velocity_mms == 0
        assert result.suspension_1_mm == 0
        assert result.suspension_2_mm == 0
        assert result.steering_angle_decideg == 0


class TestCombineAxleData:
    """Test combine_axle_data function."""

    def test_combine_both_valid(self):
        """Test combining valid front and rear data."""
        front = WheelPairData(
            wheel_1_velocity_mms=1000,
            wheel_2_velocity_mms=1500,
            suspension_1_mm=50,
            suspension_2_mm=60,
            steering_angle_decideg=100,
        )

        rear = WheelPairData(
            wheel_1_velocity_mms=2000,
            wheel_2_velocity_mms=2500,
            suspension_1_mm=70,
            suspension_2_mm=80,
            steering_angle_decideg=None,
        )

        vehicle = combine_axle_data(front, rear)

        # Check wheel velocities [FL, FR, RL, RR]
        assert vehicle.wheel_velocities == pytest.approx([1.0, 1.5, 2.0, 2.5])

        # Check suspension [FL, FR, RL, RR]
        assert vehicle.suspension_displacements == pytest.approx([0.05, 0.06, 0.07, 0.08])

        # Check steering angle
        assert vehicle.steering_angle == pytest.approx(math.radians(10.0))

        # Check validity flags
        assert vehicle.front_valid is True
        assert vehicle.rear_valid is True
        assert vehicle.all_valid is True

    def test_combine_front_only(self):
        """Test with only front axle data."""
        front = WheelPairData(
            wheel_1_velocity_mms=1000,
            wheel_2_velocity_mms=1500,
            suspension_1_mm=50,
            suspension_2_mm=60,
            steering_angle_decideg=100,
        )

        vehicle = combine_axle_data(front, None)

        # Front wheels populated, rear wheels zero
        assert vehicle.wheel_velocities[0] == pytest.approx(1.0)  # FL
        assert vehicle.wheel_velocities[1] == pytest.approx(1.5)  # FR
        assert vehicle.wheel_velocities[2] == pytest.approx(0.0)  # RL
        assert vehicle.wheel_velocities[3] == pytest.approx(0.0)  # RR

        assert vehicle.front_valid is True
        assert vehicle.rear_valid is False
        assert vehicle.all_valid is False

    def test_combine_rear_only(self):
        """Test with only rear axle data."""
        rear = WheelPairData(
            wheel_1_velocity_mms=2000,
            wheel_2_velocity_mms=2500,
            suspension_1_mm=70,
            suspension_2_mm=80,
            steering_angle_decideg=None,
        )

        vehicle = combine_axle_data(None, rear)

        # Front wheels zero, rear wheels populated
        assert vehicle.wheel_velocities[0] == pytest.approx(0.0)  # FL
        assert vehicle.wheel_velocities[1] == pytest.approx(0.0)  # FR
        assert vehicle.wheel_velocities[2] == pytest.approx(2.0)  # RL
        assert vehicle.wheel_velocities[3] == pytest.approx(2.5)  # RR

        # Steering should be None (no front data)
        assert vehicle.steering_angle is None

        assert vehicle.front_valid is False
        assert vehicle.rear_valid is True
        assert vehicle.all_valid is False

    def test_combine_neither_valid(self):
        """Test with no valid data."""
        vehicle = combine_axle_data(None, None)

        # All zeros
        assert vehicle.wheel_velocities == pytest.approx([0.0, 0.0, 0.0, 0.0])
        assert vehicle.suspension_displacements == pytest.approx([0.0, 0.0, 0.0, 0.0])
        assert vehicle.steering_angle is None

        assert vehicle.front_valid is False
        assert vehicle.rear_valid is False
        assert vehicle.all_valid is False


class TestRealWorldScenarios:
    """Test realistic CAN message scenarios."""

    def test_vehicle_moving_straight(self):
        """Test vehicle moving straight with equal wheel speeds."""
        # Front: 5000 mm/s (5 m/s), steering straight (0 deg)
        front_data = struct.pack('<HHBBh', 5000, 5000, 50, 50, 0)
        front = decode_wheel_pair(CAN_ID_FRONT_AXLE, front_data)

        # Rear: 5000 mm/s (5 m/s)
        rear_data = struct.pack('<HHBBh', 5000, 5000, 50, 50, 0)
        rear = decode_wheel_pair(CAN_ID_REAR_AXLE, rear_data)

        vehicle = combine_axle_data(front, rear)

        # All wheels should be 5.0 m/s
        assert all(v == pytest.approx(5.0) for v in vehicle.wheel_velocities)
        assert vehicle.steering_angle == pytest.approx(0.0)

    def test_vehicle_turning_right(self):
        """Test vehicle turning right with differential wheel speeds."""
        # Front: Inner wheel slower, outer wheel faster, steering right (-30 deg)
        front_data = struct.pack('<HHBBh', 4500, 5500, 45, 55, -300)
        front = decode_wheel_pair(CAN_ID_FRONT_AXLE, front_data)

        # Rear: Similar differential
        rear_data = struct.pack('<HHBBh', 4600, 5400, 48, 52, 0)
        rear = decode_wheel_pair(CAN_ID_REAR_AXLE, rear_data)

        vehicle = combine_axle_data(front, rear)

        # Inner wheels (left) slower than outer wheels (right)
        assert vehicle.wheel_velocities[0] < vehicle.wheel_velocities[1]  # FL < FR
        assert vehicle.wheel_velocities[2] < vehicle.wheel_velocities[3]  # RL < RR

        # Steering angle negative (right turn)
        assert vehicle.steering_angle < 0

    def test_vehicle_stationary(self):
        """Test vehicle stationary (all zeros)."""
        front_data = struct.pack('<HHBBh', 0, 0, 50, 50, 0)
        front = decode_wheel_pair(CAN_ID_FRONT_AXLE, front_data)

        rear_data = struct.pack('<HHBBh', 0, 0, 50, 50, 0)
        rear = decode_wheel_pair(CAN_ID_REAR_AXLE, rear_data)

        vehicle = combine_axle_data(front, rear)

        assert all(v == pytest.approx(0.0) for v in vehicle.wheel_velocities)
        assert vehicle.steering_angle == pytest.approx(0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
