#!/usr/bin/env python3
"""
VN-200 Binary Protocol Parser

Pure Python implementation for parsing VectorNav VN-200 binary packets.
No ROS dependencies - can be used standalone or in non-ROS applications.

Binary Packet Structure:
    [Sync (1)] [Groups (1)] [Group1 Fields (2)]...[GroupN Fields (2)] [Payload] [CRC (2)]

    Sync byte: 0xFA
    Groups byte: Bits 0-5 indicate which groups are present
    CRC: 16-bit CRC over entire packet excluding sync byte

Reference: VN-200 User Manual, Section 7.2 - Binary Output Protocol
"""

import struct
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import IntEnum, IntFlag
import math


# ============================================================================
# Constants
# ============================================================================

SYNC_BYTE = 0xFA


# ============================================================================
# Enums for Group and Field Selection
# ============================================================================

class GroupBit(IntFlag):
    """Binary output group selection bits (Groups byte)."""
    COMMON = 0x01      # Group 1: Common outputs
    TIME = 0x02        # Group 2: Time outputs
    IMU = 0x04         # Group 3: IMU outputs
    GPS = 0x08         # Group 4: GPS outputs
    ATTITUDE = 0x10    # Group 5: Attitude outputs
    INS = 0x20         # Group 6: INS outputs


class CommonField(IntFlag):
    """Group 1 (Common) field selection bits."""
    TIME_STARTUP = 0x0001      # uint64 - Time since startup (ns)
    TIME_GPS = 0x0002          # uint64 - GPS time (ns)
    TIME_SYNC_IN = 0x0004      # uint64 - SyncIn time (ns)
    YAW_PITCH_ROLL = 0x0008    # 3x float32 - Euler angles (deg)
    QUATERNION = 0x0010        # 4x float32 - Attitude quaternion
    ANGULAR_RATE = 0x0020      # 3x float32 - Angular velocity (rad/s)
    POSITION = 0x0040          # 3x float64 - Position (lat, lon, alt)
    VELOCITY = 0x0080          # 3x float32 - Velocity NED (m/s)
    ACCEL = 0x0100             # 3x float32 - Acceleration (m/s^2)
    IMU = 0x0200               # 6x float32 - Uncompensated IMU
    MAG_PRES = 0x0400          # 4x float32 - Mag + pressure
    DELTA_THETA = 0x0800       # float32 + 6x float32 - Coning/sculling
    INS_STATUS = 0x1000        # uint16 - INS status
    SYNC_IN_CNT = 0x2000       # uint32 - SyncIn count
    TIME_GPS_PPS = 0x4000      # uint64 - GPS PPS time (ns)


class IMUField(IntFlag):
    """Group 3 (IMU) field selection bits."""
    IMU_STATUS = 0x0001        # uint16 - IMU status
    UNCOMP_MAG = 0x0002        # 3x float32 - Uncompensated mag (Gauss)
    UNCOMP_ACCEL = 0x0004      # 3x float32 - Uncompensated accel (m/s^2)
    UNCOMP_GYRO = 0x0008       # 3x float32 - Uncompensated gyro (rad/s)
    TEMP = 0x0010              # float32 - Temperature (C)
    PRES = 0x0020              # float32 - Pressure (kPa)
    DELTA_THETA = 0x0040       # float32 + 3x float32 - Delta theta
    DELTA_VEL = 0x0080         # float32 + 3x float32 - Delta velocity
    MAG = 0x0100               # 3x float32 - Compensated mag (Gauss)
    ACCEL = 0x0200             # 3x float32 - Compensated accel (m/s^2)
    ANGULAR_RATE = 0x0400      # 3x float32 - Compensated gyro (rad/s)
    SENS_SAT = 0x0800          # uint16 - Sensor saturation flags


class GPSField(IntFlag):
    """Group 4 (GPS) field selection bits."""
    UTC = 0x0001               # 8 bytes - UTC time
    TOW = 0x0002               # uint64 - Time of week (ns)
    WEEK = 0x0004              # uint16 - GPS week
    NUM_SATS = 0x0008          # uint8 - Number of satellites
    FIX = 0x0010               # uint8 - GPS fix type
    POS_LLA = 0x0020           # 3x float64 - Position (lat, lon, alt)
    POS_ECEF = 0x0040          # 3x float64 - Position ECEF (m)
    VEL_NED = 0x0080           # 3x float32 - Velocity NED (m/s)
    VEL_ECEF = 0x0100          # 3x float32 - Velocity ECEF (m/s)
    POS_U = 0x0200             # 3x float32 - Position uncertainty NED (m)
    VEL_U = 0x0400             # float32 - Velocity uncertainty (m/s)
    TIME_U = 0x0800            # uint32 - Time uncertainty (ns)
    TIME_INFO = 0x1000         # 2 bytes - Time info status
    DOP = 0x2000               # 7x float32 - Dilution of precision
    SAT_INFO = 0x4000          # Variable - Satellite info


class AttitudeField(IntFlag):
    """Group 5 (Attitude) field selection bits."""
    VPE_STATUS = 0x0001        # uint16 - VPE status
    YAW_PITCH_ROLL = 0x0002    # 3x float32 - Euler angles (deg)
    QUATERNION = 0x0004        # 4x float32 - Quaternion
    DCM = 0x0008               # 9x float32 - Direction cosine matrix
    MAG_NED = 0x0010           # 3x float32 - Magnetic NED (Gauss)
    ACCEL_NED = 0x0020         # 3x float32 - Acceleration NED (m/s^2)
    LINEAR_ACCEL_BODY = 0x0040 # 3x float32 - Linear accel body (m/s^2)
    LINEAR_ACCEL_NED = 0x0080  # 3x float32 - Linear accel NED (m/s^2)
    YPR_U = 0x0100             # 3x float32 - YPR uncertainty (deg)
    # Reserved bits 0x0200 - 0x8000


class INSField(IntFlag):
    """Group 6 (INS) field selection bits."""
    INS_STATUS = 0x0001        # uint16 - INS status
    POS_LLA = 0x0002           # 3x float64 - Position (lat, lon, alt)
    POS_ECEF = 0x0004          # 3x float64 - Position ECEF (m)
    VEL_BODY = 0x0008          # 3x float32 - Velocity body (m/s)
    VEL_NED = 0x0010           # 3x float32 - Velocity NED (m/s)
    VEL_ECEF = 0x0020          # 3x float32 - Velocity ECEF (m/s)
    MAG_ECEF = 0x0040          # 3x float32 - Magnetic ECEF (Gauss)
    ACCEL_ECEF = 0x0080        # 3x float32 - Acceleration ECEF (m/s^2)
    LINEAR_ACCEL_ECEF = 0x0100 # 3x float32 - Linear accel ECEF (m/s^2)
    POS_U = 0x0200             # float32 - Position uncertainty (m)
    VEL_U = 0x0400             # float32 - Velocity uncertainty (m/s)


class GPSFix(IntEnum):
    """GPS fix types."""
    NO_FIX = 0
    TIME_ONLY = 1
    FIX_2D = 2
    FIX_3D = 3
    SBAS = 4
    RTK_FLOAT = 5
    RTK_FIXED = 6


class INSMode(IntEnum):
    """INS operating modes (from InsStatus bits 0-1)."""
    NOT_TRACKING = 0
    DEGRADED = 1
    HEALTHY = 2


# ============================================================================
# Field Size Definitions (bytes)
# ============================================================================

COMMON_FIELD_SIZES = {
    CommonField.TIME_STARTUP: 8,
    CommonField.TIME_GPS: 8,
    CommonField.TIME_SYNC_IN: 8,
    CommonField.YAW_PITCH_ROLL: 12,
    CommonField.QUATERNION: 16,
    CommonField.ANGULAR_RATE: 12,
    CommonField.POSITION: 24,
    CommonField.VELOCITY: 12,
    CommonField.ACCEL: 12,
    CommonField.IMU: 24,
    CommonField.MAG_PRES: 16,
    CommonField.DELTA_THETA: 28,
    CommonField.INS_STATUS: 2,
    CommonField.SYNC_IN_CNT: 4,
    CommonField.TIME_GPS_PPS: 8,
}

IMU_FIELD_SIZES = {
    IMUField.IMU_STATUS: 2,
    IMUField.UNCOMP_MAG: 12,
    IMUField.UNCOMP_ACCEL: 12,
    IMUField.UNCOMP_GYRO: 12,
    IMUField.TEMP: 4,
    IMUField.PRES: 4,
    IMUField.DELTA_THETA: 16,
    IMUField.DELTA_VEL: 16,
    IMUField.MAG: 12,
    IMUField.ACCEL: 12,
    IMUField.ANGULAR_RATE: 12,
    IMUField.SENS_SAT: 2,
}

GPS_FIELD_SIZES = {
    GPSField.UTC: 8,
    GPSField.TOW: 8,
    GPSField.WEEK: 2,
    GPSField.NUM_SATS: 1,
    GPSField.FIX: 1,
    GPSField.POS_LLA: 24,
    GPSField.POS_ECEF: 24,
    GPSField.VEL_NED: 12,
    GPSField.VEL_ECEF: 12,
    GPSField.POS_U: 12,
    GPSField.VEL_U: 4,
    GPSField.TIME_U: 4,
    GPSField.TIME_INFO: 2,
    GPSField.DOP: 28,
    # SAT_INFO is variable length
}

ATTITUDE_FIELD_SIZES = {
    AttitudeField.VPE_STATUS: 2,
    AttitudeField.YAW_PITCH_ROLL: 12,
    AttitudeField.QUATERNION: 16,
    AttitudeField.DCM: 36,
    AttitudeField.MAG_NED: 12,
    AttitudeField.ACCEL_NED: 12,
    AttitudeField.LINEAR_ACCEL_BODY: 12,
    AttitudeField.LINEAR_ACCEL_NED: 12,
    AttitudeField.YPR_U: 12,
}

INS_FIELD_SIZES = {
    INSField.INS_STATUS: 2,
    INSField.POS_LLA: 24,
    INSField.POS_ECEF: 24,
    INSField.VEL_BODY: 12,
    INSField.VEL_NED: 12,
    INSField.VEL_ECEF: 12,
    INSField.MAG_ECEF: 12,
    INSField.ACCEL_ECEF: 12,
    INSField.LINEAR_ACCEL_ECEF: 12,
    INSField.POS_U: 4,
    INSField.VEL_U: 4,
}


# ============================================================================
# Data Classes for Parsed Data
# ============================================================================

@dataclass
class CommonData:
    """Parsed data from Group 1 (Common)."""
    time_startup_ns: Optional[int] = None
    time_gps_ns: Optional[int] = None
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    quat_w: Optional[float] = None
    quat_x: Optional[float] = None
    quat_y: Optional[float] = None
    quat_z: Optional[float] = None
    angular_rate_x: Optional[float] = None  # rad/s
    angular_rate_y: Optional[float] = None
    angular_rate_z: Optional[float] = None
    position_lat: Optional[float] = None    # deg
    position_lon: Optional[float] = None    # deg
    position_alt: Optional[float] = None    # m
    velocity_n: Optional[float] = None      # m/s
    velocity_e: Optional[float] = None
    velocity_d: Optional[float] = None
    accel_x: Optional[float] = None         # m/s^2
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    ins_status: Optional[int] = None

    @property
    def yaw_rad(self) -> Optional[float]:
        """Yaw angle in radians."""
        return math.radians(self.yaw_deg) if self.yaw_deg is not None else None

    @property
    def pitch_rad(self) -> Optional[float]:
        """Pitch angle in radians."""
        return math.radians(self.pitch_deg) if self.pitch_deg is not None else None

    @property
    def roll_rad(self) -> Optional[float]:
        """Roll angle in radians."""
        return math.radians(self.roll_deg) if self.roll_deg is not None else None


@dataclass
class IMUData:
    """Parsed data from Group 3 (IMU)."""
    imu_status: Optional[int] = None
    uncomp_mag_x: Optional[float] = None    # Gauss
    uncomp_mag_y: Optional[float] = None
    uncomp_mag_z: Optional[float] = None
    uncomp_accel_x: Optional[float] = None  # m/s^2
    uncomp_accel_y: Optional[float] = None
    uncomp_accel_z: Optional[float] = None
    uncomp_gyro_x: Optional[float] = None   # rad/s
    uncomp_gyro_y: Optional[float] = None
    uncomp_gyro_z: Optional[float] = None
    temperature_c: Optional[float] = None
    pressure_kpa: Optional[float] = None
    mag_x: Optional[float] = None           # Compensated, Gauss
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    accel_x: Optional[float] = None         # Compensated, m/s^2
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    angular_rate_x: Optional[float] = None  # Compensated, rad/s
    angular_rate_y: Optional[float] = None
    angular_rate_z: Optional[float] = None


@dataclass
class GPSData:
    """Parsed data from Group 4 (GPS)."""
    tow_ns: Optional[int] = None            # Time of week (ns)
    week: Optional[int] = None              # GPS week
    num_sats: Optional[int] = None
    fix: Optional[GPSFix] = None
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    altitude_m: Optional[float] = None
    vel_north_mps: Optional[float] = None
    vel_east_mps: Optional[float] = None
    vel_down_mps: Optional[float] = None
    pos_uncertainty_n: Optional[float] = None  # m
    pos_uncertainty_e: Optional[float] = None
    pos_uncertainty_d: Optional[float] = None
    vel_uncertainty_mps: Optional[float] = None

    @property
    def has_fix(self) -> bool:
        """Check if GPS has a valid fix."""
        return self.fix is not None and self.fix >= GPSFix.FIX_2D

    @property
    def pos_uncertainty_m(self) -> Optional[Tuple[float, float, float]]:
        """Position uncertainty as (N, E, D) tuple in meters."""
        if self.pos_uncertainty_n is not None:
            return (self.pos_uncertainty_n, self.pos_uncertainty_e, self.pos_uncertainty_d)
        return None


@dataclass
class AttitudeData:
    """Parsed data from Group 5 (Attitude)."""
    vpe_status: Optional[int] = None
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    quat_w: Optional[float] = None
    quat_x: Optional[float] = None
    quat_y: Optional[float] = None
    quat_z: Optional[float] = None
    linear_accel_body_x: Optional[float] = None  # m/s^2
    linear_accel_body_y: Optional[float] = None
    linear_accel_body_z: Optional[float] = None
    yaw_uncertainty_deg: Optional[float] = None
    pitch_uncertainty_deg: Optional[float] = None
    roll_uncertainty_deg: Optional[float] = None


@dataclass
class INSData:
    """Parsed data from Group 6 (INS)."""
    ins_status: Optional[int] = None
    mode: Optional[INSMode] = None
    gps_fix: Optional[bool] = None
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    altitude_m: Optional[float] = None
    vel_body_x: Optional[float] = None      # m/s, forward
    vel_body_y: Optional[float] = None      # m/s, right
    vel_body_z: Optional[float] = None      # m/s, down
    vel_ned_n: Optional[float] = None       # m/s
    vel_ned_e: Optional[float] = None
    vel_ned_d: Optional[float] = None
    pos_uncertainty_m: Optional[float] = None
    vel_uncertainty_mps: Optional[float] = None


@dataclass
class VN200Packet:
    """Complete parsed VN-200 binary packet."""
    timestamp_ns: int = 0                   # Host receive time
    common: CommonData = field(default_factory=CommonData)
    imu: IMUData = field(default_factory=IMUData)
    gps: GPSData = field(default_factory=GPSData)
    attitude: AttitudeData = field(default_factory=AttitudeData)
    ins: INSData = field(default_factory=INSData)
    groups_present: int = 0                 # Bitmask of groups in packet
    raw_bytes: bytes = b''                  # Original packet for debugging

    @property
    def has_common(self) -> bool:
        """Check if Common group data is present."""
        return bool(self.groups_present & GroupBit.COMMON)

    @property
    def has_time(self) -> bool:
        """Check if Time group data is present."""
        return bool(self.groups_present & GroupBit.TIME)

    @property
    def has_imu(self) -> bool:
        """Check if IMU group data is present."""
        return bool(self.groups_present & GroupBit.IMU)

    @property
    def has_gps(self) -> bool:
        """Check if GPS group data is present."""
        return bool(self.groups_present & GroupBit.GPS)

    @property
    def has_attitude(self) -> bool:
        """Check if Attitude group data is present."""
        return bool(self.groups_present & GroupBit.ATTITUDE)

    @property
    def has_ins(self) -> bool:
        """Check if INS group data is present."""
        return bool(self.groups_present & GroupBit.INS)


# ============================================================================
# CRC-16 Calculation
# ============================================================================

def calculate_crc16(data: bytes) -> int:
    """
    Calculate CRC-16 for VN-200 packet validation.

    Uses CRC-16-CCITT polynomial.

    Args:
        data: Bytes to calculate CRC over (excludes sync byte)

    Returns:
        16-bit CRC value
    """
    crc = 0
    for byte in data:
        crc = ((crc >> 8) | (crc << 8)) & 0xFFFF
        crc ^= byte
        crc ^= (crc & 0xFF) >> 4
        crc ^= (crc << 12) & 0xFFFF
        crc ^= ((crc & 0xFF) << 5) & 0xFFFF
    return crc


# ============================================================================
# Packet Parser
# ============================================================================

class VN200Parser:
    """
    VN-200 binary packet parser.

    Stateful parser that handles:
    - Byte synchronization (finding 0xFA sync byte)
    - CRC validation
    - Field extraction based on group/field selectors

    Usage:
        parser = VN200Parser()

        # Feed raw bytes from serial port
        packets = parser.feed(serial_data)

        for packet in packets:
            print(f"Yaw: {packet.common.yaw_deg}")
    """

    def __init__(self):
        self._buffer = bytearray()
        self._packet_count = 0
        self._crc_errors = 0
        self._sync_losses = 0

    def feed(self, data: bytes) -> List[VN200Packet]:
        """
        Feed raw bytes and extract complete packets.

        Args:
            data: Raw bytes from serial port

        Returns:
            List of parsed VN200Packet objects (may be empty)
        """
        self._buffer.extend(data)
        packets = []

        while True:
            packet = self._try_parse_packet()
            if packet is None:
                break
            packets.append(packet)
            self._packet_count += 1

        return packets

    def _try_parse_packet(self) -> Optional[VN200Packet]:
        """Attempt to parse one complete packet from buffer."""
        # Find sync byte
        while len(self._buffer) > 0 and self._buffer[0] != SYNC_BYTE:
            self._buffer.pop(0)
            self._sync_losses += 1

        # Need at least sync + groups byte
        if len(self._buffer) < 2:
            return None

        groups_byte = self._buffer[1]

        # Calculate header size (sync + groups + field selectors)
        num_groups = bin(groups_byte & 0x3F).count('1')
        header_size = 2 + (num_groups * 2)

        if len(self._buffer) < header_size:
            return None

        # Parse field selectors and calculate payload size
        field_selectors = {}
        payload_size = 0
        offset = 2

        for group_bit in [GroupBit.COMMON, GroupBit.TIME, GroupBit.IMU,
                          GroupBit.GPS, GroupBit.ATTITUDE, GroupBit.INS]:
            if groups_byte & group_bit:
                field_sel = struct.unpack_from('<H', self._buffer, offset)[0]
                field_selectors[group_bit] = field_sel
                payload_size += self._calculate_payload_size(group_bit, field_sel)
                offset += 2

        # Total packet size = header + payload + CRC(2)
        total_size = header_size + payload_size + 2

        if len(self._buffer) < total_size:
            return None

        # Extract packet bytes
        packet_bytes = bytes(self._buffer[:total_size])

        # Validate CRC (over everything except sync byte, including CRC itself)
        crc_data = packet_bytes[1:]
        if calculate_crc16(crc_data) != 0:
            self._crc_errors += 1
            # Remove sync byte and continue searching
            self._buffer.pop(0)
            return None

        # CRC valid - remove packet from buffer
        del self._buffer[:total_size]

        # Parse payload
        return self._parse_payload(packet_bytes, groups_byte, field_selectors, header_size)

    def _calculate_payload_size(self, group: GroupBit, fields: int) -> int:
        """Calculate payload size for a group based on field selector."""
        size_map = {
            GroupBit.COMMON: COMMON_FIELD_SIZES,
            GroupBit.TIME: {},  # Time group not fully implemented
            GroupBit.IMU: IMU_FIELD_SIZES,
            GroupBit.GPS: GPS_FIELD_SIZES,
            GroupBit.ATTITUDE: ATTITUDE_FIELD_SIZES,
            GroupBit.INS: INS_FIELD_SIZES,
        }

        sizes = size_map.get(group, {})
        total = 0

        for field_enum, field_size in sizes.items():
            if fields & field_enum:
                total += field_size

        return total

    def _parse_payload(
        self,
        packet: bytes,
        groups: int,
        field_selectors: Dict[GroupBit, int],
        header_size: int
    ) -> VN200Packet:
        """Parse payload data into VN200Packet."""
        result = VN200Packet(
            groups_present=groups,
            raw_bytes=packet,
        )

        offset = header_size

        # Parse each present group in order
        if GroupBit.COMMON in field_selectors:
            offset = self._parse_common(packet, offset, field_selectors[GroupBit.COMMON], result.common)

        if GroupBit.TIME in field_selectors:
            # Skip Time group for now (not commonly used)
            pass

        if GroupBit.IMU in field_selectors:
            offset = self._parse_imu(packet, offset, field_selectors[GroupBit.IMU], result.imu)

        if GroupBit.GPS in field_selectors:
            offset = self._parse_gps(packet, offset, field_selectors[GroupBit.GPS], result.gps)

        if GroupBit.ATTITUDE in field_selectors:
            offset = self._parse_attitude(packet, offset, field_selectors[GroupBit.ATTITUDE], result.attitude)

        if GroupBit.INS in field_selectors:
            offset = self._parse_ins(packet, offset, field_selectors[GroupBit.INS], result.ins)

        return result

    def _parse_common(self, packet: bytes, offset: int, fields: int, data: CommonData) -> int:
        """Parse Common group fields."""
        if fields & CommonField.TIME_STARTUP:
            data.time_startup_ns = struct.unpack_from('<Q', packet, offset)[0]
            offset += 8

        if fields & CommonField.TIME_GPS:
            data.time_gps_ns = struct.unpack_from('<Q', packet, offset)[0]
            offset += 8

        if fields & CommonField.TIME_SYNC_IN:
            offset += 8  # Skip

        if fields & CommonField.YAW_PITCH_ROLL:
            data.yaw_deg, data.pitch_deg, data.roll_deg = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & CommonField.QUATERNION:
            data.quat_w, data.quat_x, data.quat_y, data.quat_z = struct.unpack_from('<ffff', packet, offset)
            offset += 16

        if fields & CommonField.ANGULAR_RATE:
            data.angular_rate_x, data.angular_rate_y, data.angular_rate_z = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & CommonField.POSITION:
            data.position_lat, data.position_lon, data.position_alt = struct.unpack_from('<ddd', packet, offset)
            offset += 24

        if fields & CommonField.VELOCITY:
            data.velocity_n, data.velocity_e, data.velocity_d = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & CommonField.ACCEL:
            data.accel_x, data.accel_y, data.accel_z = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & CommonField.IMU:
            offset += 24  # Skip uncompensated IMU

        if fields & CommonField.MAG_PRES:
            offset += 16  # Skip mag/pressure

        if fields & CommonField.DELTA_THETA:
            offset += 28  # Skip coning/sculling

        if fields & CommonField.INS_STATUS:
            data.ins_status = struct.unpack_from('<H', packet, offset)[0]
            offset += 2

        if fields & CommonField.SYNC_IN_CNT:
            offset += 4  # Skip

        if fields & CommonField.TIME_GPS_PPS:
            offset += 8  # Skip

        return offset

    def _parse_imu(self, packet: bytes, offset: int, fields: int, data: IMUData) -> int:
        """Parse IMU group fields."""
        if fields & IMUField.IMU_STATUS:
            data.imu_status = struct.unpack_from('<H', packet, offset)[0]
            offset += 2

        if fields & IMUField.UNCOMP_MAG:
            data.uncomp_mag_x, data.uncomp_mag_y, data.uncomp_mag_z = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & IMUField.UNCOMP_ACCEL:
            data.uncomp_accel_x, data.uncomp_accel_y, data.uncomp_accel_z = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & IMUField.UNCOMP_GYRO:
            data.uncomp_gyro_x, data.uncomp_gyro_y, data.uncomp_gyro_z = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & IMUField.TEMP:
            data.temperature_c = struct.unpack_from('<f', packet, offset)[0]
            offset += 4

        if fields & IMUField.PRES:
            data.pressure_kpa = struct.unpack_from('<f', packet, offset)[0]
            offset += 4

        if fields & IMUField.DELTA_THETA:
            offset += 16  # Skip

        if fields & IMUField.DELTA_VEL:
            offset += 16  # Skip

        if fields & IMUField.MAG:
            data.mag_x, data.mag_y, data.mag_z = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & IMUField.ACCEL:
            data.accel_x, data.accel_y, data.accel_z = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & IMUField.ANGULAR_RATE:
            data.angular_rate_x, data.angular_rate_y, data.angular_rate_z = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & IMUField.SENS_SAT:
            offset += 2  # Skip

        return offset

    def _parse_gps(self, packet: bytes, offset: int, fields: int, data: GPSData) -> int:
        """Parse GPS group fields."""
        if fields & GPSField.UTC:
            offset += 8  # Skip UTC struct

        if fields & GPSField.TOW:
            data.tow_ns = struct.unpack_from('<Q', packet, offset)[0]
            offset += 8

        if fields & GPSField.WEEK:
            data.week = struct.unpack_from('<H', packet, offset)[0]
            offset += 2

        if fields & GPSField.NUM_SATS:
            data.num_sats = packet[offset]
            offset += 1

        if fields & GPSField.FIX:
            try:
                data.fix = GPSFix(packet[offset])
            except ValueError:
                data.fix = GPSFix.NO_FIX
            offset += 1

        if fields & GPSField.POS_LLA:
            data.latitude_deg, data.longitude_deg, data.altitude_m = struct.unpack_from('<ddd', packet, offset)
            offset += 24

        if fields & GPSField.POS_ECEF:
            offset += 24  # Skip ECEF

        if fields & GPSField.VEL_NED:
            data.vel_north_mps, data.vel_east_mps, data.vel_down_mps = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & GPSField.VEL_ECEF:
            offset += 12  # Skip ECEF velocity

        if fields & GPSField.POS_U:
            data.pos_uncertainty_n, data.pos_uncertainty_e, data.pos_uncertainty_d = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & GPSField.VEL_U:
            data.vel_uncertainty_mps = struct.unpack_from('<f', packet, offset)[0]
            offset += 4

        if fields & GPSField.TIME_U:
            offset += 4  # Skip

        if fields & GPSField.TIME_INFO:
            offset += 2  # Skip

        if fields & GPSField.DOP:
            offset += 28  # Skip DOP values

        return offset

    def _parse_attitude(self, packet: bytes, offset: int, fields: int, data: AttitudeData) -> int:
        """Parse Attitude group fields."""
        if fields & AttitudeField.VPE_STATUS:
            data.vpe_status = struct.unpack_from('<H', packet, offset)[0]
            offset += 2

        if fields & AttitudeField.YAW_PITCH_ROLL:
            data.yaw_deg, data.pitch_deg, data.roll_deg = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & AttitudeField.QUATERNION:
            data.quat_w, data.quat_x, data.quat_y, data.quat_z = struct.unpack_from('<ffff', packet, offset)
            offset += 16

        if fields & AttitudeField.DCM:
            offset += 36  # Skip DCM

        if fields & AttitudeField.MAG_NED:
            offset += 12  # Skip

        if fields & AttitudeField.ACCEL_NED:
            offset += 12  # Skip

        if fields & AttitudeField.LINEAR_ACCEL_BODY:
            data.linear_accel_body_x, data.linear_accel_body_y, data.linear_accel_body_z = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & AttitudeField.LINEAR_ACCEL_NED:
            offset += 12  # Skip

        if fields & AttitudeField.YPR_U:
            data.yaw_uncertainty_deg, data.pitch_uncertainty_deg, data.roll_uncertainty_deg = struct.unpack_from('<fff', packet, offset)
            offset += 12

        return offset

    def _parse_ins(self, packet: bytes, offset: int, fields: int, data: INSData) -> int:
        """Parse INS group fields."""
        if fields & INSField.INS_STATUS:
            data.ins_status = struct.unpack_from('<H', packet, offset)[0]
            data.mode = INSMode(data.ins_status & 0x03)
            data.gps_fix = bool((data.ins_status >> 2) & 0x01)
            offset += 2

        if fields & INSField.POS_LLA:
            data.latitude_deg, data.longitude_deg, data.altitude_m = struct.unpack_from('<ddd', packet, offset)
            offset += 24

        if fields & INSField.POS_ECEF:
            offset += 24  # Skip ECEF

        if fields & INSField.VEL_BODY:
            data.vel_body_x, data.vel_body_y, data.vel_body_z = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & INSField.VEL_NED:
            data.vel_ned_n, data.vel_ned_e, data.vel_ned_d = struct.unpack_from('<fff', packet, offset)
            offset += 12

        if fields & INSField.VEL_ECEF:
            offset += 12  # Skip ECEF velocity

        if fields & INSField.MAG_ECEF:
            offset += 12  # Skip

        if fields & INSField.ACCEL_ECEF:
            offset += 12  # Skip

        if fields & INSField.LINEAR_ACCEL_ECEF:
            offset += 12  # Skip

        if fields & INSField.POS_U:
            data.pos_uncertainty_m = struct.unpack_from('<f', packet, offset)[0]
            offset += 4

        if fields & INSField.VEL_U:
            data.vel_uncertainty_mps = struct.unpack_from('<f', packet, offset)[0]
            offset += 4

        return offset

    @property
    def stats(self) -> Dict[str, int]:
        """Return parser statistics."""
        return {
            'packets_parsed': self._packet_count,
            'crc_errors': self._crc_errors,
            'sync_losses': self._sync_losses,
        }

    def reset(self) -> None:
        """Reset parser state and statistics."""
        self._buffer.clear()
        self._packet_count = 0
        self._crc_errors = 0
        self._sync_losses = 0
