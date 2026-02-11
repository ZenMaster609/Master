# Nodes and Topics

This file lists the ROS2 nodes in the workspace and their key inputs/outputs. Topic names are shown as defaults; many are configurable via parameters or remappings.

## sim_car

### `ackermann_cmd_bridge`

| Item | Value |
| --- | --- |
| Publishes | `/cmd_vel` (`geometry_msgs/Twist`) |
| Subscribes | `/cmd` (`ackermann_msgs/AckermannDriveStamped`), `/sim/brake_cmd` (`std_msgs/Float32`) |
| Services | `/race_car_model/command_mode` (`std_srvs/Trigger`) |
| Key parameters | `input_topic`, `output_topic`, `wheelbase`, `command_mode`, `max_speed`, `accel_limit`, `brake_decel_limit`, `control_rate`, `brake_cmd_topic`, `steering_from_desired_speed`, `steering_speed_floor` |

### `wheel_encoder_node`

| Item | Value |
| --- | --- |
| Publishes | `/sim/raw/wheel_encoder/rpm`, `/sim/raw/wheel_encoder/angle_accum`, `/sim/raw/wheel_encoder/speed_mm_s` (`std_msgs/Float32MultiArray`) |
| Subscribes | `/sim/raw/joint_states` (`sensor_msgs/JointState`) |
| Key parameters | `publish_rate`, `publish_rpm`, `publish_angle_accum`, `publish_speed_mm_s`, `wheel_radius`, `min_dt`, `min_window_sec`, `min_delta`, `topic_prefix` |

### `suspension_sensor_node`

| Item | Value |
| --- | --- |
| Publishes | `/sim/raw/suspension` (`std_msgs/Float32MultiArray`) |
| Subscribes | `/sim/raw/joint_states` (`sensor_msgs/JointState`), `/sim/raw/odom` (`nav_msgs/Odometry`) |
| Key parameters | `mode`, `noise_stddev`, `bias_fl`, `bias_fr`, `bias_rl`, `bias_rr`, `publish_rate`, `dropout_probability`, `static_mm`, `pitch_gain`, `roll_gain`, `filter_tau_sec`, `topic_prefix` |

### `steering_sensor_node`

| Item | Value |
| --- | --- |
| Publishes | `/sim/raw/steering_angle` (`std_msgs/Float32`) |
| Subscribes | `/sim/raw/joint_states` (`sensor_msgs/JointState`) |
| Key parameters | `noise_stddev`, `latency_ms`, `publish_rate`, `bias`, `dropout_probability`, `topic_prefix` |

### `virtual_sensors_node`

| Item | Value |
| --- | --- |
| Publishes | `/sim/raw/cooling/water_pressure`, `/sim/raw/cooling/water_flow`, `/sim/raw/cooling/water_temp_in`, `/sim/raw/cooling/water_temp_out`, `/sim/raw/cooling/water_temp_radiator`, `/sim/raw/brakes/temp_fr`, `/sim/raw/brakes/temp_rl`, `/sim/raw/pitot/dynamic_pressure` (`std_msgs/Float32`) |
| Subscribes | `/sim/raw/odom` (`nav_msgs/Odometry`), `/cmd_vel` (`geometry_msgs/Twist`), `/sim/brake_cmd` (`std_msgs/Float32`) |
| Key parameters | `publish_rate`, per-signal `publish_rate_*`, `ambient_temp`, `noise_pressure`, `noise_flow`, `noise_temp`, `noise_brake_temp`, `noise_pitot`, `brake_cmd_topic`, `topic_prefix` |

### Single-topic virtual sensors

These nodes share the same input model and parameters, but each publishes a single topic.

| Node | Publishes |
| --- | --- |
| `water_pressure_node` | `/sim/raw/cooling/water_pressure` |
| `water_flow_node` | `/sim/raw/cooling/water_flow` |
| `water_temp_in_node` | `/sim/raw/cooling/water_temp_in` |
| `water_temp_out_node` | `/sim/raw/cooling/water_temp_out` |
| `water_temp_radiator_node` | `/sim/raw/cooling/water_temp_radiator` |
| `brake_temp_fr_node` | `/sim/raw/brakes/temp_fr` |
| `brake_temp_rl_node` | `/sim/raw/brakes/temp_rl` |
| `pitot_dynamic_pressure_node` | `/sim/raw/pitot/dynamic_pressure` |

Common subscriptions and parameters for these nodes:

- Subscribes: `/sim/raw/odom` (`nav_msgs/Odometry`), `/cmd_vel` (`geometry_msgs/Twist`), and `/sim/brake_cmd` (`std_msgs/Float32`) for brake temperature nodes.
- Parameters: `publish_rate`, `ambient_temp`, `topic_prefix`, plus one noise parameter (`noise_pressure`, `noise_flow`, `noise_temp`, `noise_brake_temp`, or `noise_pitot`).

## measurement_node

### `measurement_node`

| Item | Value |
| --- | --- |
| Publishes | Config-defined output topics (default `/sim/*` topics) |
| Subscribes | Config-defined input topics (default `/sim/raw/*` topics) |
| Key parameters | `config_path` (defaults to `sim_car/config/sensor_config.yaml`) |

## canbus_decoder

### `can_decoder_node`

| Item | Value |
| --- | --- |
| Publishes | `/can/wheel_rpm` (`std_msgs/Float32MultiArray`), `/can/suspension` (`std_msgs/Float32MultiArray`), `/can/steering_angle` (`std_msgs/Float32`) |
| Subscribes | `/can/rx` (`can_msgs/Frame`) by default |
| Key parameters | `can_topic`, `stale_timeout_ms`, `publish_rate_hz`, `wheel_radius`, `show_stats`, `stats_interval` |

### `can_monitor_node`

| Item | Value |
| --- | --- |
| Publishes | none |
| Subscribes | `/can/rx` (`can_msgs/Frame`) |
| Key parameters | `can_topic`, `filter_ids`, `show_stats`, `stats_interval`, `verbose` |

### `vcan_publisher_node`

| Item | Value |
| --- | --- |
| Publishes | `/to_can_bus` (`can_msgs/Frame`) |
| Subscribes | none |
| Key parameters | `publish_rate_hz`, `mode`, `base_velocity_mps` |

## vectornav_decoder

### `vectornav_decoder_node`

| Item | Value |
| --- | --- |
| Publishes | `/vectornav/imu` (`sensor_msgs/Imu`), `/vectornav/gps` (`sensor_msgs/NavSatFix`), `/vectornav/ins` (`nav_msgs/Odometry`) |
| Subscribes | none (reads directly from serial port) |
| Key parameters | `config_file`, `serial_port`, `baudrate`, `publish_rate_hz`, `frame_id`, `imu_topic`, `gps_topic`, `ins_topic`, `show_stats`, `stats_interval` |

### `vectornav_monitor_node`

| Item | Value |
| --- | --- |
| Publishes | none |
| Subscribes | none (reads directly from serial port) |
| Key parameters | `config_file`, `serial_port`, `baudrate`, `verbose`, `stats_interval` |

## vehicle_plotter

### `data_collector_node`

| Item | Value |
| --- | --- |
| Publishes | `/vehicle_plotter/state` (`vehicle_plotter_msgs/VehicleState`) |
| Subscribes | Adapter-defined topics (Gazebo, CAN, or VectorNav) |
| Key parameters | `adapter`, `output_rate_hz`, `gps_origin_lat`, `gps_origin_lon`, `sync_buffer_sec` |

### `plotter_node`

| Item | Value |
| --- | --- |
| Publishes | none |
| Subscribes | `/vehicle_plotter/state`, `/run_session` |
| Key parameters | `backend`, `update_rate_hz`, `window_title`, `plot_layout`, `dark_mode`, `enable_gui`, `state_topic`, `save_plots_on_exit`, `save_plot_data_on_exit`, `close_plots_on_shutdown`, `wait_for_session`, `session_timeout_sec`, `base_path` |

### `logger_node`

| Item | Value |
| --- | --- |
| Publishes | none |
| Subscribes | `/vehicle_plotter/state`, `/run_session` |
| Key parameters | `format`, `compression`, `base_path`, `session_name`, `flush_interval_sec`, `buffer_size`, `state_topic`, `enable_logging`, `wait_for_session`, `session_timeout_sec`, `adapter`, `auto_plot_on_shutdown` |

### `session_manager_node`

| Item | Value |
| --- | --- |
| Publishes | `/run_session` (`vehicle_plotter_msgs/RunSession`) |
| Subscribes | none |
| Key parameters | `base_path`, `broadcast_rate_hz` |

### `rosbag_controller_node`

| Item | Value |
| --- | --- |
| Publishes | none |
| Subscribes | `/run_session` (`vehicle_plotter_msgs/RunSession`) when `wait_for_session` is true |
| Key parameters | `enable_recording`, `topics`, `mode`, `compression`, `max_bag_size`, `wait_for_session`, `session_timeout_sec`, `base_path`, `config_file` |

## steering_gui

### `EUFSRobotSteeringGUI` (RQT plugin)

| Item | Value |
| --- | --- |
| Publishes | Ackermann commands to a user-selected topic, `/sim/brake_cmd` (`std_msgs/Float32`) |
| Subscribes | none |
| Key parameters | `brake_cmd_topic` |
