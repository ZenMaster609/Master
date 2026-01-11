#!/usr/bin/env python3
"""Unit tests for VN-200 configuration handling."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectornav_decoder.vn200_config import (
    SerialConfig,
    GroupConfig,
    BinaryOutputConfig,
    VectorNavConfig,
    parse_config_dict,
    build_register_command,
    build_full_ascii_command,
    create_default_config,
    COMMON_FIELD_BITS,
    IMU_FIELD_BITS,
    GPS_FIELD_BITS,
    INS_FIELD_BITS,
)


class TestSerialConfig:
    """Test SerialConfig dataclass."""

    def test_defaults(self):
        """SerialConfig has sensible defaults."""
        config = SerialConfig()
        assert config.port == '/dev/ttyUSB0'
        assert config.baudrate == 921600
        assert config.timeout_sec == 0.1

    def test_custom_values(self):
        """SerialConfig accepts custom values."""
        config = SerialConfig(
            port='/dev/ttyACM0',
            baudrate=115200,
            timeout_sec=0.5,
        )
        assert config.port == '/dev/ttyACM0'
        assert config.baudrate == 115200
        assert config.timeout_sec == 0.5


class TestGroupConfig:
    """Test GroupConfig dataclass."""

    def test_defaults(self):
        """GroupConfig defaults to disabled."""
        config = GroupConfig()
        assert config.enabled == False
        assert config.fields == []
        assert config.field_bits == 0

    def test_enabled_with_fields(self):
        """GroupConfig can be enabled with fields."""
        config = GroupConfig(
            enabled=True,
            fields=['YawPitchRoll', 'AngularRate'],
            field_bits=0x0028,
        )
        assert config.enabled == True
        assert len(config.fields) == 2
        assert config.field_bits == 0x0028


class TestBinaryOutputConfig:
    """Test BinaryOutputConfig dataclass."""

    def test_defaults(self):
        """BinaryOutputConfig has sensible defaults."""
        config = BinaryOutputConfig()
        assert config.register == 75
        assert config.async_mode == 2
        assert config.rate_divisor == 4

    def test_groups_byte_empty(self):
        """groups_byte is 0 when no groups enabled."""
        config = BinaryOutputConfig()
        assert config.groups_byte == 0

    def test_groups_byte_single(self):
        """groups_byte correct for single group."""
        config = BinaryOutputConfig()
        config.group_common.enabled = True
        assert config.groups_byte == 0x01

    def test_groups_byte_multiple(self):
        """groups_byte correct for multiple groups."""
        config = BinaryOutputConfig()
        config.group_common.enabled = True
        config.group_imu.enabled = True
        config.group_gps.enabled = True
        config.group_ins.enabled = True
        assert config.groups_byte == 0x01 | 0x04 | 0x08 | 0x20

    def test_output_rate_hz(self):
        """output_rate_hz calculates correctly."""
        config = BinaryOutputConfig()
        config.rate_divisor = 4
        assert config.output_rate_hz == 200.0

        config.rate_divisor = 2
        assert config.output_rate_hz == 400.0

        config.rate_divisor = 8
        assert config.output_rate_hz == 100.0


class TestParseConfigDict:
    """Test configuration parsing from dictionaries."""

    def test_empty_dict(self):
        """Empty dict produces default config."""
        config = parse_config_dict({})
        assert config.serial.port == '/dev/ttyUSB0'
        assert config.binary_output.rate_divisor == 4

    def test_none_input(self):
        """None input produces default config."""
        config = parse_config_dict(None)
        assert config.serial.port == '/dev/ttyUSB0'

    def test_serial_config(self):
        """Serial config parsed correctly."""
        data = {
            'serial': {
                'port': '/dev/ttyUSB1',
                'baudrate': 115200,
                'timeout_sec': 0.2,
            }
        }
        config = parse_config_dict(data)
        assert config.serial.port == '/dev/ttyUSB1'
        assert config.serial.baudrate == 115200
        assert config.serial.timeout_sec == 0.2

    def test_output_config(self):
        """Output rate config parsed correctly."""
        data = {
            'output': {
                'rate_divisor': 2,
                'async_mode': 1,
            }
        }
        config = parse_config_dict(data)
        assert config.binary_output.rate_divisor == 2
        assert config.binary_output.async_mode == 1

    def test_group_config(self):
        """Group configuration parsed correctly."""
        data = {
            'binary_output_1': {
                'group_common': {
                    'enabled': True,
                    'fields': ['YawPitchRoll', 'AngularRate'],
                },
                'group_gps': {
                    'enabled': True,
                    'fields': ['NumSats', 'Fix', 'PosLla'],
                },
            }
        }
        config = parse_config_dict(data)
        assert config.binary_output.group_common.enabled == True
        assert 'YawPitchRoll' in config.binary_output.group_common.fields
        assert config.binary_output.group_common.field_bits == (
            COMMON_FIELD_BITS['YawPitchRoll'] |
            COMMON_FIELD_BITS['AngularRate']
        )
        assert config.binary_output.group_gps.enabled == True
        assert config.binary_output.group_gps.field_bits == (
            GPS_FIELD_BITS['NumSats'] |
            GPS_FIELD_BITS['Fix'] |
            GPS_FIELD_BITS['PosLla']
        )

    def test_topics_config(self):
        """Topics configuration parsed correctly."""
        data = {
            'topics': {
                'imu': '/custom/imu',
                'gps': '/custom/gps',
            }
        }
        config = parse_config_dict(data)
        assert config.topics.imu == '/custom/imu'
        assert config.topics.gps == '/custom/gps'


class TestBuildRegisterCommand:
    """Test ASCII command generation."""

    def test_minimal_command(self):
        """Minimal command with no groups."""
        config = BinaryOutputConfig()
        cmd = build_register_command(config)
        assert 'VNWRG,75' in cmd
        assert ',2,' in cmd  # async_mode
        assert ',4,' in cmd  # rate_divisor
        assert ',0x00' in cmd  # groups byte

    def test_single_group_command(self):
        """Command with single group enabled."""
        config = BinaryOutputConfig()
        config.group_common.enabled = True
        config.group_common.field_bits = 0x0028

        cmd = build_register_command(config)
        assert 'VNWRG,75' in cmd
        assert ',0x01,' in cmd  # groups byte
        assert '0x0028' in cmd  # field bits

    def test_multiple_groups_command(self):
        """Command with multiple groups enabled."""
        config = BinaryOutputConfig()
        config.group_common.enabled = True
        config.group_common.field_bits = 0x0028
        config.group_gps.enabled = True
        config.group_gps.field_bits = 0x0038

        cmd = build_register_command(config)
        assert 'VNWRG,75' in cmd
        # Groups byte should be 0x01 | 0x08 = 0x09
        assert ',0x09,' in cmd


class TestBuildFullAsciiCommand:
    """Test complete ASCII command with checksum."""

    def test_command_format(self):
        """Command has correct format."""
        cmd = build_full_ascii_command("VNASY,1")
        assert cmd.startswith(b'$')
        assert b'VNASY,1' in cmd
        assert b'*' in cmd
        assert cmd.endswith(b'\r\n')

    def test_checksum_present(self):
        """Command includes checksum."""
        cmd = build_full_ascii_command("VNRRG,1")
        # Should be $VNRRG,1*XX\r\n where XX is checksum
        decoded = cmd.decode('ascii')
        assert '*' in decoded
        # Checksum is 2 hex digits
        checksum_part = decoded.split('*')[1][:2]
        assert len(checksum_part) == 2
        assert all(c in '0123456789ABCDEF' for c in checksum_part)


class TestCreateDefaultConfig:
    """Test default configuration creation."""

    def test_default_config_valid(self):
        """Default config is valid."""
        config = create_default_config()
        assert config.register == 75
        assert config.rate_divisor == 4
        assert config.output_rate_hz == 200.0

    def test_default_groups_enabled(self):
        """Default config has expected groups enabled."""
        config = create_default_config()
        assert config.group_common.enabled == True
        assert config.group_imu.enabled == True
        assert config.group_gps.enabled == True
        assert config.group_ins.enabled == True
        assert config.group_time.enabled == False
        assert config.group_attitude.enabled == False

    def test_default_fields_populated(self):
        """Default config has fields populated."""
        config = create_default_config()
        assert 'YawPitchRoll' in config.group_common.fields
        assert 'UncompAccel' in config.group_imu.fields
        assert 'PosLla' in config.group_gps.fields
        assert 'VelBody' in config.group_ins.fields

    def test_default_field_bits_nonzero(self):
        """Default config has non-zero field bits."""
        config = create_default_config()
        assert config.group_common.field_bits > 0
        assert config.group_imu.field_bits > 0
        assert config.group_gps.field_bits > 0
        assert config.group_ins.field_bits > 0


class TestFieldBitMappings:
    """Test field name to bit value mappings."""

    def test_common_field_bits(self):
        """Common field bits are correct."""
        assert COMMON_FIELD_BITS['TimeStartup'] == 0x0001
        assert COMMON_FIELD_BITS['YawPitchRoll'] == 0x0008
        assert COMMON_FIELD_BITS['AngularRate'] == 0x0020
        assert COMMON_FIELD_BITS['Accel'] == 0x0100

    def test_imu_field_bits(self):
        """IMU field bits are correct."""
        assert IMU_FIELD_BITS['UncompAccel'] == 0x0004
        assert IMU_FIELD_BITS['UncompGyro'] == 0x0008
        assert IMU_FIELD_BITS['Temp'] == 0x0010

    def test_gps_field_bits(self):
        """GPS field bits are correct."""
        assert GPS_FIELD_BITS['NumSats'] == 0x0008
        assert GPS_FIELD_BITS['Fix'] == 0x0010
        assert GPS_FIELD_BITS['PosLla'] == 0x0020

    def test_ins_field_bits(self):
        """INS field bits are correct."""
        assert INS_FIELD_BITS['InsStatus'] == 0x0001
        assert INS_FIELD_BITS['PosLla'] == 0x0002
        assert INS_FIELD_BITS['VelBody'] == 0x0008


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
