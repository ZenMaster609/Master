# Launch Files and Arguments

This file lists each launch file in the workspace and the arguments it accepts. Defaults are the current values in the launch files.

## sim_car

### `sim_car/launch/gazebo_sim.launch.py`

- `use_sim_time` (default: `true`)
- `world` (default: `sim_car/worlds/small_track.world`)
- `headless` (default: `false`)
- `update_rate_hz` (default: `100.0`)
- `topic_prefix` (default: `/sim/raw`)

### `sim_car/launch/nodes.launch.py`

- `topic_prefix` (default: `/sim/raw`)

### `sim_car/launch/full_sim_launch.launch.py`

- `headless` (default: `true`)
- `update_rate_hz` (default: `100.0`)
- `world` (default: `sim_car/worlds/small_track.world`)
- `plotting` (default: `true`)
- `logging` (default: `true`)
- `close_plots` (default: `true`)
- `rosbagging` (default: `true`)
- `steering` (default: `false`)
- `use_sim_time` (default: `true`)
- `measure` (default: `true`)
- `measurement_config` (default: `sim_car/config/sensor_config.yaml`)

## measurement_node

### `measurement_node/launch/measurement.launch.py`

- `headless` (default: `false`)
- `use_sim_time` (default: `true`)
- `config_path` (default: `sim_car/config/sensor_config.yaml`)

## vehicle_plotter

### `vehicle_plotter/launch/plotter.launch.py`

- `adapter` (default: `gazebo`)
- `output_rate_hz` (default: `50.0`)
- `enable_plot` (default: `true`)
- `plot_rate_hz` (default: `30.0`)
- `dark_mode` (default: `true`)
- `save_plots_on_exit` (default: `false`)
- `save_plot_data_on_exit` (default: `true`)
- `close_plots` (default: `true`)
- `enable_log` (default: `true`)
- `log_format` (default: `parquet`)
- `log_path` (default: empty string)
- `gps_origin_lat` (default: `0.0`)
- `gps_origin_lon` (default: `0.0`)
- `use_sim_time` (default: `true`)
- `enable_rosbag` (default: `true`)

### `vehicle_plotter/launch/offline_replay.launch.py`

- `bag_path` (required)
- `rate` (default: `1.0`)
- `loop` (default: `false`)
- `enable_plot` (default: `true`)
- `enable_log` (default: `false`)
- `log_path` (default: `~/.ros/vehicle_logs`)

### `vehicle_plotter/launch/replay.launch.py`

- `bag_path` (required)
- `rate` (default: `1.0`)
- `enable_plot` (default: `true`)
- `enable_log` (default: `false`)
- `adapter` (default: `can`)
- `qos_override` (default: `vehicle_plotter/config/replay_qos_override.yaml`)

### `vehicle_plotter/launch/vcan_test.launch.py`

- `mode` (default: `circle`)
- `base_velocity_mps` (default: `5.0`)
- `enable_plot` (default: `true`)
- `enable_log` (default: `true`)
- `enable_rosbag` (default: `true`)

## canbus_decoder

### `canbus_decoder/launch/can_decoder.launch.py`

- `can_device` (default: `can0`)
- `bitrate` (default: `500000`)
- `stale_timeout_ms` (default: `100`)
- `publish_rate_hz` (default: `50.0`)
- `wheel_radius` (default: `0.23`)
- `stats_interval` (default: `5.0`)
- `show_stats` (default: `true`)

### `canbus_decoder/launch/can_monitor.launch.py`

- `can_device` (default: `can0`)
- `bitrate` (default: `500000`)
- `verbose` (default: `false`)
- `stats_interval` (default: `5.0`)

## vectornav_decoder

### `vectornav_decoder/launch/vectornav_decoder.launch.py`

- `config_file` (default: `vectornav_decoder/config/default_output.yaml`)
- `serial_port` (default: `/dev/ttyUSB0`)
- `baudrate` (default: `921600`)
- `publish_rate_hz` (default: `200.0`)
- `frame_id` (default: `vectornav`)
- `imu_topic` (default: `/vectornav/imu`)
- `gps_topic` (default: `/vectornav/gps`)
- `ins_topic` (default: `/vectornav/ins`)
- `show_stats` (default: `true`)
- `stats_interval` (default: `5.0`)

### `vectornav_decoder/launch/vectornav_monitor.launch.py`

- `config_file` (default: `vectornav_decoder/config/default_output.yaml`)
- `serial_port` (default: `/dev/ttyUSB0`)
- `baudrate` (default: `921600`)
- `verbose` (default: `false`)
- `stats_interval` (default: `5.0`)
