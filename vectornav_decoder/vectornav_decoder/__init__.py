"""
VectorNav Decoder Package

Decodes VectorNav VN-200 serial data (binary protocol) into structured
ROS 2 messages for IMU, GPS, and INS data.
"""

from .vn200_protocol import (
    VN200Parser,
    VN200Packet,
    CommonData,
    IMUData,
    GPSData,
    INSData,
    GroupBit,
    CommonField,
    IMUField,
    GPSField,
    INSField,
    GPSFix,
    INSMode,
    calculate_crc16,
)

from .vn200_config import (
    VectorNavConfig,
    BinaryOutputConfig,
    GroupConfig,
    SerialConfig,
    load_config_from_yaml,
    build_register_command,
)

__all__ = [
    # Protocol
    'VN200Parser',
    'VN200Packet',
    'CommonData',
    'IMUData',
    'GPSData',
    'INSData',
    'GroupBit',
    'CommonField',
    'IMUField',
    'GPSField',
    'INSField',
    'GPSFix',
    'INSMode',
    'calculate_crc16',
    # Config
    'VectorNavConfig',
    'BinaryOutputConfig',
    'GroupConfig',
    'SerialConfig',
    'load_config_from_yaml',
    'build_register_command',
]
