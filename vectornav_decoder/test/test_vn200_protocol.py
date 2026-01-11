#!/usr/bin/env python3
"""Unit tests for VN-200 binary protocol parser."""

import pytest
import struct
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectornav_decoder.vn200_protocol import (
    VN200Parser,
    VN200Packet,
    CommonData,
    IMUData,
    GPSData,
    INSData,
    calculate_crc16,
    GroupBit,
    CommonField,
    IMUField,
    GPSField,
    INSField,
    GPSFix,
    INSMode,
    SYNC_BYTE,
)


class TestCRC16:
    """Test CRC-16 calculation."""

    def test_crc_empty(self):
        """Empty data should produce consistent CRC."""
        crc = calculate_crc16(b'')
        assert crc == 0

    def test_crc_single_byte(self):
        """Single byte CRC calculation."""
        crc = calculate_crc16(b'\x01')
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_crc_known_values(self):
        """CRC with known input produces valid 16-bit result."""
        data = b'\x01\x02\x03\x04\x05\x06\x07\x08'
        crc = calculate_crc16(data)
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_crc_different_inputs(self):
        """Different inputs produce different CRCs."""
        crc1 = calculate_crc16(b'\x01\x02\x03\x04')
        crc2 = calculate_crc16(b'\x04\x03\x02\x01')
        assert crc1 != crc2


class TestVN200Parser:
    """Test binary packet parsing."""

    def _build_packet(self, groups: int, field_selectors: list, payload: bytes) -> bytes:
        """
        Helper to build test packets with correct CRC.

        Args:
            groups: Groups byte value
            field_selectors: List of 16-bit field selector values
            payload: Payload bytes

        Returns:
            Complete packet bytes including sync, header, payload, and CRC
        """
        # Header: sync + groups + field selectors
        header = bytes([SYNC_BYTE, groups])
        for fs in field_selectors:
            header += struct.pack('<H', fs)

        # Combine for CRC calculation (excludes sync byte)
        crc_data = header[1:] + payload

        # Calculate CRC
        crc = calculate_crc16(crc_data)

        # Build complete packet
        return header + payload + struct.pack('<H', crc)

    def test_parser_init(self):
        """Parser initializes with empty state."""
        parser = VN200Parser()
        assert parser.stats['packets_parsed'] == 0
        assert parser.stats['crc_errors'] == 0
        assert parser.stats['sync_losses'] == 0

    def test_parse_empty_buffer(self):
        """Parser returns empty list for empty input."""
        parser = VN200Parser()
        packets = parser.feed(b'')
        assert packets == []

    def test_parse_incomplete_packet(self):
        """Parser waits for complete packet."""
        parser = VN200Parser()

        # Just sync byte
        packets = parser.feed(bytes([SYNC_BYTE]))
        assert packets == []

        # Sync + groups (still incomplete)
        parser.reset()
        packets = parser.feed(bytes([SYNC_BYTE, 0x00]))
        assert packets == []

    def test_parse_sync_search(self):
        """Parser finds sync byte in noisy data."""
        parser = VN200Parser()

        # Build valid minimal packet (no groups enabled)
        groups = 0x00
        payload = b''
        packet = self._build_packet(groups, [], payload)

        # Add noise before packet
        noisy_data = b'\x00\x01\x02\x03' + packet
        packets = parser.feed(noisy_data)

        assert len(packets) == 1
        assert parser.stats['sync_losses'] == 4

    def test_parse_no_groups(self):
        """Parse packet with no groups enabled."""
        parser = VN200Parser()

        groups = 0x00
        packet = self._build_packet(groups, [], b'')
        packets = parser.feed(packet)

        assert len(packets) == 1
        pkt = packets[0]
        assert pkt.groups_present == 0
        assert not pkt.has_common
        assert not pkt.has_imu
        assert not pkt.has_gps
        assert not pkt.has_ins

    def test_parse_common_ypr(self):
        """Parse packet with Common group YawPitchRoll field."""
        parser = VN200Parser()

        groups = GroupBit.COMMON
        fields = CommonField.YAW_PITCH_ROLL

        # Payload: YPR (3 floats)
        yaw, pitch, roll = 45.0, 10.0, -5.0
        payload = struct.pack('<fff', yaw, pitch, roll)

        packet = self._build_packet(groups, [fields], payload)
        packets = parser.feed(packet)

        assert len(packets) == 1
        pkt = packets[0]
        assert pkt.has_common
        assert pkt.common.yaw_deg == pytest.approx(45.0)
        assert pkt.common.pitch_deg == pytest.approx(10.0)
        assert pkt.common.roll_deg == pytest.approx(-5.0)

    def test_parse_common_ypr_radians(self):
        """YPR values convert to radians correctly."""
        parser = VN200Parser()

        groups = GroupBit.COMMON
        fields = CommonField.YAW_PITCH_ROLL

        yaw, pitch, roll = 90.0, 0.0, 0.0
        payload = struct.pack('<fff', yaw, pitch, roll)

        packet = self._build_packet(groups, [fields], payload)
        packets = parser.feed(packet)

        assert len(packets) == 1
        pkt = packets[0]
        assert pkt.common.yaw_rad == pytest.approx(math.pi / 2)

    def test_parse_common_angular_rate(self):
        """Parse packet with Common group AngularRate field."""
        parser = VN200Parser()

        groups = GroupBit.COMMON
        fields = CommonField.ANGULAR_RATE

        wx, wy, wz = 0.1, 0.2, 0.3
        payload = struct.pack('<fff', wx, wy, wz)

        packet = self._build_packet(groups, [fields], payload)
        packets = parser.feed(packet)

        assert len(packets) == 1
        pkt = packets[0]
        assert pkt.common.angular_rate_x == pytest.approx(0.1)
        assert pkt.common.angular_rate_y == pytest.approx(0.2)
        assert pkt.common.angular_rate_z == pytest.approx(0.3)

    def test_parse_common_accel(self):
        """Parse packet with Common group Accel field."""
        parser = VN200Parser()

        groups = GroupBit.COMMON
        fields = CommonField.ACCEL

        ax, ay, az = 0.0, 0.0, -9.81
        payload = struct.pack('<fff', ax, ay, az)

        packet = self._build_packet(groups, [fields], payload)
        packets = parser.feed(packet)

        assert len(packets) == 1
        pkt = packets[0]
        assert pkt.common.accel_x == pytest.approx(0.0)
        assert pkt.common.accel_y == pytest.approx(0.0)
        assert pkt.common.accel_z == pytest.approx(-9.81)

    def test_parse_common_multiple_fields(self):
        """Parse packet with multiple Common group fields."""
        parser = VN200Parser()

        groups = GroupBit.COMMON
        fields = CommonField.YAW_PITCH_ROLL | CommonField.ANGULAR_RATE | CommonField.ACCEL

        # Payload: YPR + AngRate + Accel
        yaw, pitch, roll = 45.0, 10.0, -5.0
        wx, wy, wz = 0.1, 0.2, 0.3
        ax, ay, az = 0.0, 0.0, -9.81

        payload = struct.pack('<fff', yaw, pitch, roll)
        payload += struct.pack('<fff', wx, wy, wz)
        payload += struct.pack('<fff', ax, ay, az)

        packet = self._build_packet(groups, [fields], payload)
        packets = parser.feed(packet)

        assert len(packets) == 1
        pkt = packets[0]
        assert pkt.common.yaw_deg == pytest.approx(45.0)
        assert pkt.common.angular_rate_z == pytest.approx(0.3)
        assert pkt.common.accel_z == pytest.approx(-9.81)

    def test_parse_imu_group(self):
        """Parse packet with IMU group fields."""
        parser = VN200Parser()

        groups = GroupBit.IMU
        fields = IMUField.UNCOMP_ACCEL | IMUField.UNCOMP_GYRO | IMUField.TEMP

        # Payload: UncompAccel + UncompGyro + Temp
        ax, ay, az = 0.1, 0.2, -9.8
        wx, wy, wz = 0.01, 0.02, 0.03
        temp = 25.5

        payload = struct.pack('<fff', ax, ay, az)
        payload += struct.pack('<fff', wx, wy, wz)
        payload += struct.pack('<f', temp)

        packet = self._build_packet(groups, [fields], payload)
        packets = parser.feed(packet)

        assert len(packets) == 1
        pkt = packets[0]
        assert pkt.has_imu
        assert pkt.imu.uncomp_accel_x == pytest.approx(0.1)
        assert pkt.imu.uncomp_gyro_z == pytest.approx(0.03)
        assert pkt.imu.temperature_c == pytest.approx(25.5)

    def test_parse_gps_group(self):
        """Parse packet with GPS group fields."""
        parser = VN200Parser()

        groups = GroupBit.GPS
        fields = GPSField.NUM_SATS | GPSField.FIX | GPSField.POS_LLA

        # Payload: NumSats + Fix + PosLla
        num_sats = 12
        fix = GPSFix.FIX_3D
        lat, lon, alt = 59.123456, 10.654321, 100.5

        payload = bytes([num_sats, fix])
        payload += struct.pack('<ddd', lat, lon, alt)

        packet = self._build_packet(groups, [fields], payload)
        packets = parser.feed(packet)

        assert len(packets) == 1
        pkt = packets[0]
        assert pkt.has_gps
        assert pkt.gps.num_sats == 12
        assert pkt.gps.fix == GPSFix.FIX_3D
        assert pkt.gps.has_fix
        assert pkt.gps.latitude_deg == pytest.approx(59.123456)
        assert pkt.gps.longitude_deg == pytest.approx(10.654321)
        assert pkt.gps.altitude_m == pytest.approx(100.5)

    def test_parse_ins_group(self):
        """Parse packet with INS group fields."""
        parser = VN200Parser()

        groups = GroupBit.INS
        fields = INSField.INS_STATUS | INSField.VEL_BODY

        # Payload: InsStatus + VelBody
        # Status: mode=2 (healthy), gps_fix=1
        ins_status = 0x02 | (1 << 2)
        vx, vy, vz = 5.0, 0.1, 0.0

        payload = struct.pack('<H', ins_status)
        payload += struct.pack('<fff', vx, vy, vz)

        packet = self._build_packet(groups, [fields], payload)
        packets = parser.feed(packet)

        assert len(packets) == 1
        pkt = packets[0]
        assert pkt.has_ins
        assert pkt.ins.mode == INSMode.HEALTHY
        assert pkt.ins.gps_fix == True
        assert pkt.ins.vel_body_x == pytest.approx(5.0)

    def test_parse_multiple_groups(self):
        """Parse packet with multiple groups."""
        parser = VN200Parser()

        groups = GroupBit.COMMON | GroupBit.GPS
        common_fields = CommonField.YAW_PITCH_ROLL
        gps_fields = GPSField.NUM_SATS | GPSField.FIX

        # Payload: YPR + NumSats + Fix
        yaw, pitch, roll = 90.0, 0.0, 0.0
        num_sats = 8
        fix = GPSFix.FIX_2D

        payload = struct.pack('<fff', yaw, pitch, roll)
        payload += bytes([num_sats, fix])

        packet = self._build_packet(groups, [common_fields, gps_fields], payload)
        packets = parser.feed(packet)

        assert len(packets) == 1
        pkt = packets[0]
        assert pkt.has_common
        assert pkt.has_gps
        assert pkt.common.yaw_deg == pytest.approx(90.0)
        assert pkt.gps.num_sats == 8

    def test_crc_error_handling(self):
        """Parser rejects packets with bad CRC."""
        parser = VN200Parser()

        # Build packet with wrong CRC
        bad_packet = bytes([SYNC_BYTE, 0x00, 0xFF, 0xFF])

        packets = parser.feed(bad_packet)
        assert len(packets) == 0
        assert parser.stats['crc_errors'] > 0

    def test_incremental_parsing(self):
        """Parser handles data arriving in chunks."""
        parser = VN200Parser()

        groups = 0x00
        packet = self._build_packet(groups, [], b'')

        # Feed one byte at a time
        for i, byte in enumerate(packet):
            packets = parser.feed(bytes([byte]))
            if i < len(packet) - 1:
                assert len(packets) == 0
            else:
                assert len(packets) == 1

    def test_multiple_packets(self):
        """Parser handles multiple consecutive packets."""
        parser = VN200Parser()

        groups = GroupBit.COMMON
        fields = CommonField.YAW_PITCH_ROLL

        # Build 3 packets
        packets_data = b''
        for yaw in [0.0, 45.0, 90.0]:
            payload = struct.pack('<fff', yaw, 0.0, 0.0)
            packets_data += self._build_packet(groups, [fields], payload)

        packets = parser.feed(packets_data)

        assert len(packets) == 3
        assert packets[0].common.yaw_deg == pytest.approx(0.0)
        assert packets[1].common.yaw_deg == pytest.approx(45.0)
        assert packets[2].common.yaw_deg == pytest.approx(90.0)

    def test_parser_reset(self):
        """Parser reset clears state and statistics."""
        parser = VN200Parser()

        # Feed some data to generate stats
        parser.feed(b'\x00\x01\x02')

        # Reset
        parser.reset()

        assert parser.stats['packets_parsed'] == 0
        assert parser.stats['crc_errors'] == 0
        assert parser.stats['sync_losses'] == 0


class TestDataClasses:
    """Test data class properties and methods."""

    def test_common_data_yaw_rad(self):
        """CommonData yaw_rad property converts correctly."""
        data = CommonData(yaw_deg=180.0)
        assert data.yaw_rad == pytest.approx(math.pi)

    def test_common_data_yaw_rad_none(self):
        """CommonData yaw_rad returns None when yaw_deg is None."""
        data = CommonData()
        assert data.yaw_rad is None

    def test_gps_data_has_fix(self):
        """GPSData has_fix property works correctly."""
        data = GPSData(fix=GPSFix.NO_FIX)
        assert not data.has_fix

        data = GPSData(fix=GPSFix.FIX_2D)
        assert data.has_fix

        data = GPSData(fix=GPSFix.FIX_3D)
        assert data.has_fix

    def test_vn200_packet_has_properties(self):
        """VN200Packet has_* properties work correctly."""
        pkt = VN200Packet(groups_present=GroupBit.COMMON | GroupBit.GPS)

        assert pkt.has_common
        assert not pkt.has_imu
        assert pkt.has_gps
        assert not pkt.has_ins


class TestEnums:
    """Test enum values."""

    def test_group_bits(self):
        """GroupBit values are correct."""
        assert GroupBit.COMMON == 0x01
        assert GroupBit.TIME == 0x02
        assert GroupBit.IMU == 0x04
        assert GroupBit.GPS == 0x08
        assert GroupBit.ATTITUDE == 0x10
        assert GroupBit.INS == 0x20

    def test_gps_fix_values(self):
        """GPSFix enum values are correct."""
        assert GPSFix.NO_FIX == 0
        assert GPSFix.FIX_3D == 3
        assert GPSFix.RTK_FIXED == 6

    def test_ins_mode_values(self):
        """INSMode enum values are correct."""
        assert INSMode.NOT_TRACKING == 0
        assert INSMode.DEGRADED == 1
        assert INSMode.HEALTHY == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
