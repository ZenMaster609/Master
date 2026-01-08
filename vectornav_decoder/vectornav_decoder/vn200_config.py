#!/usr/bin/env python3
"""
VN-200 Configuration Handling

Parses YAML configuration and builds ASCII commands for VN-200 register setup.
No ROS dependencies - can be used standalone.

Configuration controls:
- Serial port settings
- Binary output rate
- Which data groups and fields to output

Reference: VN-200 User Manual, Section 7.3 - Binary Output Registers (75-79)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml


# ============================================================================
# Field Name to Bit Value Mappings
# ============================================================================

COMMON_FIELD_BITS = {
    'TimeStartup': 0x0001,
    'TimeGps': 0x0002,
    'TimeSyncIn': 0x0004,
    'YawPitchRoll': 0x0008,
    'Quaternion': 0x0010,
    'AngularRate': 0x0020,
    'Position': 0x0040,
    'Velocity': 0x0080,
    'Accel': 0x0100,
    'Imu': 0x0200,
    'MagPres': 0x0400,
    'DeltaTheta': 0x0800,
    'InsStatus': 0x1000,
    'SyncInCnt': 0x2000,
    'TimeGpsPps': 0x4000,
}

TIME_FIELD_BITS = {
    'TimeStartup': 0x0001,
    'TimeGps': 0x0002,
    'GpsTow': 0x0004,
    'GpsWeek': 0x0008,
    'TimeSyncIn': 0x0010,
    'TimeGpsPps': 0x0020,
    'TimeUtc': 0x0040,
    'SyncInCnt': 0x0080,
    'SyncOutCnt': 0x0100,
    'TimeStatus': 0x0200,
}

IMU_FIELD_BITS = {
    'ImuStatus': 0x0001,
    'UncompMag': 0x0002,
    'UncompAccel': 0x0004,
    'UncompGyro': 0x0008,
    'Temp': 0x0010,
    'Pres': 0x0020,
    'DeltaTheta': 0x0040,
    'DeltaVel': 0x0080,
    'Mag': 0x0100,
    'Accel': 0x0200,
    'AngularRate': 0x0400,
    'SensSat': 0x0800,
}

GPS_FIELD_BITS = {
    'Utc': 0x0001,
    'Tow': 0x0002,
    'Week': 0x0004,
    'NumSats': 0x0008,
    'Fix': 0x0010,
    'PosLla': 0x0020,
    'PosEcef': 0x0040,
    'VelNed': 0x0080,
    'VelEcef': 0x0100,
    'PosU': 0x0200,
    'VelU': 0x0400,
    'TimeU': 0x0800,
    'TimeInfo': 0x1000,
    'Dop': 0x2000,
    'SatInfo': 0x4000,
}

ATTITUDE_FIELD_BITS = {
    'VpeStatus': 0x0001,
    'YawPitchRoll': 0x0002,
    'Quaternion': 0x0004,
    'Dcm': 0x0008,
    'MagNed': 0x0010,
    'AccelNed': 0x0020,
    'LinearAccelBody': 0x0040,
    'LinearAccelNed': 0x0080,
    'YprU': 0x0100,
}

INS_FIELD_BITS = {
    'InsStatus': 0x0001,
    'PosLla': 0x0002,
    'PosEcef': 0x0004,
    'VelBody': 0x0008,
    'VelNed': 0x0010,
    'VelEcef': 0x0020,
    'MagEcef': 0x0040,
    'AccelEcef': 0x0080,
    'LinearAccelEcef': 0x0100,
    'PosU': 0x0200,
    'VelU': 0x0400,
}


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class SerialConfig:
    """Serial port configuration."""
    port: str = '/dev/ttyUSB0'
    baudrate: int = 921600
    timeout_sec: float = 0.1
    reconnect_delay_sec: float = 1.0
    max_reconnect_attempts: int = 10


@dataclass
class GroupConfig:
    """Configuration for a single binary output group."""
    enabled: bool = False
    fields: List[str] = field(default_factory=list)
    field_bits: int = 0  # Computed from fields


@dataclass
class BinaryOutputConfig:
    """Complete binary output configuration for one register."""
    register: int = 75              # Register 75, 76, or 77
    async_mode: int = 2             # 0=None, 1=Port1, 2=Port2, 3=Both
    rate_divisor: int = 4           # 800 Hz / divisor = output rate

    group_common: GroupConfig = field(default_factory=GroupConfig)
    group_time: GroupConfig = field(default_factory=GroupConfig)
    group_imu: GroupConfig = field(default_factory=GroupConfig)
    group_gps: GroupConfig = field(default_factory=GroupConfig)
    group_attitude: GroupConfig = field(default_factory=GroupConfig)
    group_ins: GroupConfig = field(default_factory=GroupConfig)

    @property
    def groups_byte(self) -> int:
        """Compute groups byte from enabled groups."""
        byte = 0
        if self.group_common.enabled:
            byte |= 0x01
        if self.group_time.enabled:
            byte |= 0x02
        if self.group_imu.enabled:
            byte |= 0x04
        if self.group_gps.enabled:
            byte |= 0x08
        if self.group_attitude.enabled:
            byte |= 0x10
        if self.group_ins.enabled:
            byte |= 0x20
        return byte

    @property
    def output_rate_hz(self) -> float:
        """Compute output rate in Hz."""
        return 800.0 / self.rate_divisor


@dataclass
class TopicConfig:
    """ROS topic configuration."""
    imu: str = '/vectornav/imu'
    gps: str = '/vectornav/gps'
    ins: str = '/vectornav/ins'
    raw: str = '/vectornav/raw'


@dataclass
class VectorNavConfig:
    """Complete VectorNav configuration from YAML."""
    serial: SerialConfig = field(default_factory=SerialConfig)
    binary_output: BinaryOutputConfig = field(default_factory=BinaryOutputConfig)
    topics: TopicConfig = field(default_factory=TopicConfig)
    diagnostics_rate_hz: float = 1.0
    stale_timeout_ms: int = 50
    frame_id: str = 'vectornav'


# ============================================================================
# YAML Loading Functions
# ============================================================================

def load_config_from_yaml(yaml_path: str) -> VectorNavConfig:
    """
    Load VectorNav configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        VectorNavConfig with parsed settings
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    return parse_config_dict(data)


def parse_config_dict(data: Dict[str, Any]) -> VectorNavConfig:
    """
    Parse configuration from dictionary.

    Args:
        data: Configuration dictionary (from YAML or elsewhere)

    Returns:
        VectorNavConfig with parsed settings
    """
    config = VectorNavConfig()

    if data is None:
        return config

    # Serial config
    if 'serial' in data:
        serial = data['serial']
        config.serial = SerialConfig(
            port=serial.get('port', '/dev/ttyUSB0'),
            baudrate=serial.get('baudrate', 921600),
            timeout_sec=serial.get('timeout_sec', 0.1),
            reconnect_delay_sec=serial.get('reconnect_delay_sec', 1.0),
            max_reconnect_attempts=serial.get('max_reconnect_attempts', 10),
        )

    # Output rate configuration
    if 'output' in data:
        output = data['output']
        config.binary_output.rate_divisor = output.get('rate_divisor', 4)
        config.binary_output.async_mode = output.get('async_mode', 2)

    # Binary output groups configuration
    if 'binary_output_1' in data:
        bo = data['binary_output_1']
        config.binary_output.register = 75

        _parse_group(bo, 'group_common', config.binary_output.group_common, COMMON_FIELD_BITS)
        _parse_group(bo, 'group_time', config.binary_output.group_time, TIME_FIELD_BITS)
        _parse_group(bo, 'group_imu', config.binary_output.group_imu, IMU_FIELD_BITS)
        _parse_group(bo, 'group_gps', config.binary_output.group_gps, GPS_FIELD_BITS)
        _parse_group(bo, 'group_attitude', config.binary_output.group_attitude, ATTITUDE_FIELD_BITS)
        _parse_group(bo, 'group_ins', config.binary_output.group_ins, INS_FIELD_BITS)

    # Topics config
    if 'topics' in data:
        topics = data['topics']
        config.topics = TopicConfig(
            imu=topics.get('imu', '/vectornav/imu'),
            gps=topics.get('gps', '/vectornav/gps'),
            ins=topics.get('ins', '/vectornav/ins'),
            raw=topics.get('raw', '/vectornav/raw'),
        )

    # Diagnostics
    if 'diagnostics' in data:
        diag = data['diagnostics']
        config.diagnostics_rate_hz = diag.get('publish_rate_hz', 1.0)
        config.stale_timeout_ms = diag.get('stale_timeout_ms', 50)

    # Frame ID
    if 'frame_id' in data:
        config.frame_id = data['frame_id']

    return config


def _parse_group(data: dict, key: str, group: GroupConfig, field_map: dict) -> None:
    """
    Parse a single group configuration.

    Args:
        data: Parent dictionary containing group config
        key: Key name for this group (e.g., 'group_common')
        group: GroupConfig object to populate
        field_map: Mapping of field names to bit values
    """
    if key not in data:
        return

    gdata = data[key]
    group.enabled = gdata.get('enabled', False)
    group.fields = gdata.get('fields', [])

    # Compute field bits from field names
    group.field_bits = 0
    for field_name in group.fields:
        if field_name in field_map:
            group.field_bits |= field_map[field_name]


# ============================================================================
# ASCII Command Building
# ============================================================================

def build_register_command(config: BinaryOutputConfig) -> str:
    """
    Build ASCII command to configure binary output register.

    Register 75/76/77 format:
        VNWRG,<reg>,<asyncMode>,<rateDivisor>,<groups>,<group1Fields>,...

    Args:
        config: Binary output configuration

    Returns:
        Command string (without $ prefix or checksum)

    Example:
        VNWRG,75,2,4,0x25,0x0129,0x0006,0x0038,0x000F
    """
    parts = [
        f"VNWRG,{config.register}",
        str(config.async_mode),
        str(config.rate_divisor),
        f"0x{config.groups_byte:02X}",
    ]

    # Add field selectors for each enabled group (in order)
    if config.group_common.enabled:
        parts.append(f"0x{config.group_common.field_bits:04X}")
    if config.group_time.enabled:
        parts.append(f"0x{config.group_time.field_bits:04X}")
    if config.group_imu.enabled:
        parts.append(f"0x{config.group_imu.field_bits:04X}")
    if config.group_gps.enabled:
        parts.append(f"0x{config.group_gps.field_bits:04X}")
    if config.group_attitude.enabled:
        parts.append(f"0x{config.group_attitude.field_bits:04X}")
    if config.group_ins.enabled:
        parts.append(f"0x{config.group_ins.field_bits:04X}")

    return ','.join(parts)


def build_full_ascii_command(command: str) -> bytes:
    """
    Build complete ASCII command with header and checksum.

    Args:
        command: Command string without $ or checksum

    Returns:
        Complete command bytes including $, checksum, and CRLF
    """
    # Calculate checksum (XOR of all bytes)
    checksum = 0
    for c in command:
        checksum ^= ord(c)

    full_cmd = f"${command}*{checksum:02X}\r\n"
    return full_cmd.encode('ascii')


def create_default_config() -> BinaryOutputConfig:
    """
    Create a default binary output configuration.

    Configures typical fields for vehicle state estimation:
    - Common: TimeStartup, YawPitchRoll, AngularRate, Accel
    - IMU: UncompAccel, UncompGyro, Temp
    - GPS: NumSats, Fix, PosLla, VelNed, PosU, VelU
    - INS: InsStatus, PosLla, VelBody, VelNed

    Returns:
        BinaryOutputConfig with sensible defaults
    """
    config = BinaryOutputConfig()
    config.register = 75
    config.async_mode = 2  # Port 2
    config.rate_divisor = 4  # 200 Hz

    # Common group
    config.group_common.enabled = True
    config.group_common.fields = ['TimeStartup', 'YawPitchRoll', 'AngularRate', 'Accel']
    config.group_common.field_bits = (
        COMMON_FIELD_BITS['TimeStartup'] |
        COMMON_FIELD_BITS['YawPitchRoll'] |
        COMMON_FIELD_BITS['AngularRate'] |
        COMMON_FIELD_BITS['Accel']
    )

    # IMU group
    config.group_imu.enabled = True
    config.group_imu.fields = ['UncompAccel', 'UncompGyro', 'Temp']
    config.group_imu.field_bits = (
        IMU_FIELD_BITS['UncompAccel'] |
        IMU_FIELD_BITS['UncompGyro'] |
        IMU_FIELD_BITS['Temp']
    )

    # GPS group
    config.group_gps.enabled = True
    config.group_gps.fields = ['NumSats', 'Fix', 'PosLla', 'VelNed', 'PosU', 'VelU']
    config.group_gps.field_bits = (
        GPS_FIELD_BITS['NumSats'] |
        GPS_FIELD_BITS['Fix'] |
        GPS_FIELD_BITS['PosLla'] |
        GPS_FIELD_BITS['VelNed'] |
        GPS_FIELD_BITS['PosU'] |
        GPS_FIELD_BITS['VelU']
    )

    # INS group
    config.group_ins.enabled = True
    config.group_ins.fields = ['InsStatus', 'PosLla', 'VelBody', 'VelNed']
    config.group_ins.field_bits = (
        INS_FIELD_BITS['InsStatus'] |
        INS_FIELD_BITS['PosLla'] |
        INS_FIELD_BITS['VelBody'] |
        INS_FIELD_BITS['VelNed']
    )

    return config
