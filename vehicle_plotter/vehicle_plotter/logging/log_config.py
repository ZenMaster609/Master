"""
Log configuration dataclass.

Provides configuration for the logging subsystem including
output format, signals to log, and performance settings.
"""

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional
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
        "encoder_speed_fl", "encoder_speed_fr", "encoder_speed_rl", "encoder_speed_rr",
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
            'encoder_speed_fl': float,
            'encoder_speed_fr': float,
            'encoder_speed_rl': float,
            'encoder_speed_rr': float,
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


@dataclass(frozen=True)
class LoggerNodeConfig:
    log_format: str
    compression: str
    base_path_str: str
    session_name: str
    flush_interval_sec: float
    buffer_size: int
    state_topic: str
    enable_logging: bool
    enable_state_logging: bool
    wait_for_session: bool
    session_timeout_sec: float
    adapter_type: str
    auto_plot_on_shutdown: bool
    camera_cone_eval_topic: str
    lidar_cone_eval_topic: str
    path_tracking_eval_enabled: bool
    path_tracking_eval_rate_hz: float
    path_tracking_eval_gt_track_topic: str
    path_tracking_eval_odom_topic: str
    path_tracking_eval_planner_path_topic: str
    path_tracking_eval_track_name: str
    path_tracking_eval_tf_timeout_sec: float
    path_tracking_eval_filename: str
    path_tracking_eval_summary_json: str
    path_tracking_eval_summary_txt: str
    path_tracking_eval_autostop_laps: int
    off_track_autostop_enabled: bool
    off_track_autostop_timeout_s: float
    off_track_autostop_planner_diag_topic: str
    control_reference_wheelbase_m: float


LOGGER_NODE_PARAMETER_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ('format', 'parquet'),
    ('compression', 'snappy'),
    ('base_path', ''),
    ('session_name', ''),
    ('flush_interval_sec', 5.0),
    ('buffer_size', 1000),
    ('state_topic', 'vehicle_plotter/state'),
    ('enable_logging', True),
    ('enable_state_logging', True),
    ('wait_for_session', True),
    ('session_timeout_sec', 5.0),
    ('adapter', 'gazebo'),
    ('auto_plot_on_shutdown', True),
    ('camera_cone_eval_topic', '/sim/stereo/eval'),
    ('lidar_cone_eval_topic', '/sim/lidar/eval'),
    ('path_tracking_eval_enabled', False),
    ('path_tracking_eval_rate_hz', 20.0),
    ('path_tracking_eval_gt_track_topic', '/ground_truth/track'),
    ('path_tracking_eval_odom_topic', '/sim/odom'),
    ('path_tracking_eval_planner_path_topic', '/planned_centerline'),
    ('path_tracking_eval_track_name', ''),
    ('path_tracking_eval_tf_timeout_sec', 0.05),
    ('path_tracking_eval_filename', 'path_tracking_eval.csv'),
    ('path_tracking_eval_summary_json', 'path_tracking_eval_summary.json'),
    ('path_tracking_eval_summary_txt', 'path_tracking_eval_summary.txt'),
    ('path_tracking_eval_autostop_laps', 0),
    ('control_reference_wheelbase_m', 1.65),
    ('off_track_autostop_enabled', True),
    ('off_track_autostop_timeout_s', 5.0),
    ('off_track_autostop_planner_diag_topic', '/midpoint_planner/diagnostics'),
)


def _declare_and_read_parameters(node, defaults: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, default in defaults:
        node.declare_parameter(name, default)
        values[name] = node.get_parameter(name).value
    return values


def _nonempty_string(value: Any, fallback: str) -> str:
    return str(value).strip() or fallback


def declare_and_load_config(node) -> LoggerNodeConfig:
    values = _declare_and_read_parameters(node, LOGGER_NODE_PARAMETER_DEFAULTS)
    return LoggerNodeConfig(
        log_format=values['format'],
        compression=values['compression'],
        base_path_str=values['base_path'],
        session_name=values['session_name'],
        flush_interval_sec=values['flush_interval_sec'],
        buffer_size=values['buffer_size'],
        state_topic=values['state_topic'],
        enable_logging=values['enable_logging'],
        enable_state_logging=values['enable_state_logging'],
        wait_for_session=values['wait_for_session'],
        session_timeout_sec=values['session_timeout_sec'],
        adapter_type=values['adapter'],
        auto_plot_on_shutdown=values['auto_plot_on_shutdown'],
        camera_cone_eval_topic=str(values['camera_cone_eval_topic']).strip(),
        lidar_cone_eval_topic=str(values['lidar_cone_eval_topic']).strip(),
        path_tracking_eval_enabled=bool(values['path_tracking_eval_enabled']),
        path_tracking_eval_rate_hz=max(1.0, float(values['path_tracking_eval_rate_hz'])),
        path_tracking_eval_gt_track_topic=_nonempty_string(
            values['path_tracking_eval_gt_track_topic'], '/ground_truth/track'
        ),
        path_tracking_eval_odom_topic=_nonempty_string(
            values['path_tracking_eval_odom_topic'], '/sim/odom'
        ),
        path_tracking_eval_planner_path_topic=_nonempty_string(
            values['path_tracking_eval_planner_path_topic'], '/planned_centerline'
        ),
        path_tracking_eval_track_name=str(values['path_tracking_eval_track_name']).strip().lower(),
        path_tracking_eval_tf_timeout_sec=max(
            0.0, float(values['path_tracking_eval_tf_timeout_sec'])
        ),
        path_tracking_eval_filename=_nonempty_string(
            values['path_tracking_eval_filename'], 'path_tracking_eval.csv'
        ),
        path_tracking_eval_summary_json=_nonempty_string(
            values['path_tracking_eval_summary_json'], 'path_tracking_eval_summary.json'
        ),
        path_tracking_eval_summary_txt=_nonempty_string(
            values['path_tracking_eval_summary_txt'], 'path_tracking_eval_summary.txt'
        ),
        path_tracking_eval_autostop_laps=max(
            0, int(values['path_tracking_eval_autostop_laps'])
        ),
        off_track_autostop_enabled=bool(values['off_track_autostop_enabled']),
        off_track_autostop_timeout_s=max(1.0, float(values['off_track_autostop_timeout_s'])),
        off_track_autostop_planner_diag_topic=_nonempty_string(
            values['off_track_autostop_planner_diag_topic'], '/midpoint_planner/diagnostics'
        ),
        control_reference_wheelbase_m=max(0.1, float(values['control_reference_wheelbase_m'])),
    )


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
            "encoder_speed_fl", "encoder_speed_fr", "encoder_speed_rl", "encoder_speed_rr",
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
