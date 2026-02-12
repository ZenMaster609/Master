"""
Log configuration dataclass.

Provides configuration for the logging subsystem including
output format, signals to log, and performance settings.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
from pathlib import Path


@dataclass
class LogConfig:
    """
    Configuration for the logging subsystem.

    Attributes:
        base_path: Root directory for log files
        format: Output format ('parquet', 'csv', 'hdf5')
        compression: Compression algorithm for parquet/hdf5
        signals: List of VehicleState fields to log
        flush_interval_sec: Seconds between disk flushes
        buffer_size: Max samples before forced flush
        max_file_size_mb: Split files when exceeded
        auto_start: Start logging on node startup
        session_name: Custom session name (auto if None)
    """

    # Output settings
    base_path: Path = field(default_factory=lambda: Path.home() / ".ros" / "vehicle_logs")
    format: Literal["parquet", "csv", "hdf5"] = "parquet"
    compression: Literal["snappy", "gzip", "zstd", "none"] = "snappy"

    # Signals to log (VehicleState attribute names)
    signals: List[str] = field(default_factory=lambda: [
        "timestamp",
        "x", "y",
        "vx", "vy",
        "yaw", "yaw_rate",
        "raw_x", "raw_y", "raw_vx", "raw_vy", "raw_yaw", "raw_speed",
        "imu_vx", "imu_vy", "imu_yaw",
        "speed", "distance_traveled",
        "slip_longitudinal", "slip_lateral",
        "encoder_fl", "encoder_fr", "encoder_rl", "encoder_rr",
        "vel_fl", "vel_fr", "vel_rl", "vel_rr",
        "gps_latitude", "gps_longitude", "gps_valid",
        "gps_local_x", "gps_local_y",
        "ins_x", "ins_y",
        "dr_x", "dr_y",
        "suspension_fl", "suspension_fr", "suspension_rl", "suspension_rr",
        "steering_angle", "steering_valid",
        "water_pressure", "water_flow",
        "water_temp_in", "water_temp_out", "water_temp_radiator",
        "brake_temp_fr", "brake_temp_rl",
        "pitot_dynamic_pressure",
        "estimation_status", "source_adapter",
    ])

    # Optional raw sensor logging (at native rates)
    log_raw_imu: bool = False
    log_raw_gps: bool = False
    log_raw_odom: bool = False

    # Performance settings
    flush_interval_sec: float = 5.0        # Flush to disk every N seconds
    buffer_size: int = 10000               # Max samples before forced flush
    max_file_size_mb: int = 100            # Split files if exceeded

    # Session settings
    auto_start: bool = True                # Start logging on node startup
    session_name: Optional[str] = None     # Custom session name (auto-generated if None)

    def get_signal_schema(self) -> dict:
        """
        Get the schema (column name -> type) for configured signals.

        Returns:
            Dictionary mapping signal names to Python types
        """
        type_map = {
            'timestamp': float,
            'x': float,
            'y': float,
            'vx': float,
            'vy': float,
            'yaw': float,
            'yaw_rate': float,
            'raw_x': float,
            'raw_y': float,
            'raw_vx': float,
            'raw_vy': float,
            'raw_yaw': float,
            'raw_speed': float,
            'imu_vx': float,
            'imu_vy': float,
            'imu_yaw': float,
            'speed': float,
            'distance_traveled': float,
            'slip_longitudinal': float,
            'slip_lateral': float,
            'encoder_fl': float,
            'encoder_fr': float,
            'encoder_rl': float,
            'encoder_rr': float,
            'vel_fl': float,
            'vel_fr': float,
            'vel_rl': float,
            'vel_rr': float,
            'gps_latitude': float,
            'gps_longitude': float,
            'gps_altitude': float,
            'gps_valid': bool,
            'gps_local_x': float,
            'gps_local_y': float,
            'ins_x': float,
            'ins_y': float,
            'dr_x': float,
            'dr_y': float,
            'suspension_fl': float,
            'suspension_fr': float,
            'suspension_rl': float,
            'suspension_rr': float,
            'steering_angle': float,
            'steering_valid': bool,
            'water_pressure': float,
            'water_flow': float,
            'water_temp_in': float,
            'water_temp_out': float,
            'water_temp_radiator': float,
            'brake_temp_fr': float,
            'brake_temp_rl': float,
            'pitot_dynamic_pressure': float,
            'estimation_status': str,
            'source_adapter': str,
        }

        return {signal: type_map.get(signal, float) for signal in self.signals}


def get_minimal_config() -> LogConfig:
    """Get minimal logging config (position and velocity only)."""
    return LogConfig(
        signals=[
            "timestamp",
            "x", "y",
            "vx", "vy",
            "yaw",
            "raw_x", "raw_y", "raw_vx", "raw_vy", "raw_yaw", "raw_speed",
            "imu_vx", "imu_vy", "imu_yaw",
            "speed",
        ],
    )


def get_full_config() -> LogConfig:
    """Get full logging config (all available signals)."""
    return LogConfig(
        signals=[
            "timestamp",
            "x", "y",
            "vx", "vy",
            "yaw", "yaw_rate",
            "raw_x", "raw_y", "raw_vx", "raw_vy", "raw_yaw", "raw_speed",
            "imu_vx", "imu_vy", "imu_yaw",
            "speed", "distance_traveled",
            "slip_longitudinal", "slip_lateral",
            "encoder_fl", "encoder_fr", "encoder_rl", "encoder_rr",
            "vel_fl", "vel_fr", "vel_rl", "vel_rr",
            "gps_latitude", "gps_longitude", "gps_altitude", "gps_valid",
            "gps_local_x", "gps_local_y",
            "ins_x", "ins_y",
            "dr_x", "dr_y",
            "suspension_fl", "suspension_fr", "suspension_rl", "suspension_rr",
            "steering_angle", "steering_valid",
            "water_pressure", "water_flow",
            "water_temp_in", "water_temp_out", "water_temp_radiator",
            "brake_temp_fr", "brake_temp_rl",
            "pitot_dynamic_pressure",
            "estimation_status", "source_adapter",
        ],
        log_raw_imu=True,
        log_raw_gps=True,
    )
