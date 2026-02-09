# Packages

This file summarizes each ROS2 package in the workspace, with its nodes, launch files, configs, and key assets.

## canbus_decoder (`canbus_decoder/`)

- Purpose: Decode raw CAN frames into wheel RPM, suspension, and steering topics; provide monitor and VCAN test publisher.
- Nodes (console scripts):
  - `can_decoder_node`
  - `can_monitor_node`
  - `vcan_publisher_node`
- Launch files:
  - `canbus_decoder/launch/can_decoder.launch.py`
  - `canbus_decoder/launch/can_monitor.launch.py`
- Configs: none (parameters are set via launch or CLI).

## eufs_gz_dynamics (`eufs_gz_dynamics/`)

- Purpose: Gazebo dynamics plugin library for EUFS models.
- Nodes: none (builds a shared library in `lib/`).
- Launch files: none.
- Configs: none in this package.

## eufs_msgs (`eufs_msgs/`)

- Purpose: EUFS messages and actions used by EUFS sim tooling.
- Messages: numerous `.msg` files under `eufs_msgs/msg/`.
- Actions: `eufs_msgs/action/CheckForObjects.action`.
- Nodes: none.
- Launch files: none.

## eufs_launcher (`eufs_sim/eufs_launcher/`)

- Purpose: EUFS sim launcher UI and launch files.
- Nodes: none (ships a `scripts/eufs_launcher` helper).
- Launch files:
  - `eufs_sim/eufs_launcher/launch/eufs_launcher.launch.py`
  - `eufs_sim/eufs_launcher/launch/simulation.launch.py`
- Configs:
  - `eufs_sim/eufs_launcher/config/eufs_launcher.yaml`
  - `eufs_sim/eufs_launcher/config/default.rviz`

## eufs_models (`eufs_sim/eufs_models/`)

- Purpose: EUFS vehicle model utilities and libraries (C++).
- Nodes: none (library built from `src/`).
- Launch files: none.
- Configs:
  - `eufs_sim/eufs_models/config/noise.yaml`

## eufs_plugins (`eufs_sim/eufs_plugins/`)

- Purpose: Gazebo plugins and URDF snippets for EUFS models.
- Nodes: none (plugins and URDF resources).
- Launch files: none.
- Assets: `gazebo_cone_plugins/`, `gazebo_race_car_model/`, `gazebo_simulate_bounding_boxes/`, `urdf/`.

## eufs_racecar (`eufs_sim/eufs_racecar/`)

- Purpose: EUFS racecar model assets (URDF, meshes, materials) and a load launch file.
- Nodes: none.
- Launch files:
  - `eufs_sim/eufs_racecar/launch/load_car.launch.py`
- Assets: `urdf/`, `meshes/`, `materials/`, `robots/`.

## eufs_rqt (`eufs_sim/eufs_rqt/`)

- Purpose: EUFS RQT GUI plugins and perspective configuration.
- Nodes: none (ships `scripts/eufs_robot_steering_gui` and `scripts/mission_control_gui`).
- Launch files: none.
- Configs:
  - `eufs_sim/eufs_rqt/config/eufs_sim.perspective`

## eufs_sensors (`eufs_sim/eufs_sensors/`)

- Purpose: EUFS sensor URDFs and meshes (IMU, GPS, lidar, etc.).
- Nodes: none.
- Launch files: none.
- Assets: `urdf/`, `meshes/`.

## eufs_tracks (`eufs_sim/eufs_tracks/`)

- Purpose: Track generation and format conversion tools.
- Nodes: a launch file references an executable named `eufs_tracks`.
- Scripts: `scripts/eufs_tracks_generator`, `scripts/eufs_tracks_converter`.
- Launch files:
  - `eufs_sim/eufs_tracks/launch/eufs_track_generator.launch.py`
- Assets: `worlds/`, `models/`, `materials/`, `meshes/`, `csv/`, `image/`.

## measurement_node (`measurement_node/`)

- Purpose: Apply latency, dropout, noise, bias, and saturation to raw sensor topics.
- Nodes (console scripts):
  - `measurement_node`
- Launch files:
  - `measurement_node/launch/measurement.launch.py`
- Configs:
  - Consumes `sim_car/config/sensor_config.yaml` (path passed via `config_path`).

## sim_car (`sim_car/`)

- Purpose: Main Gazebo Fortress simulation, control bridge, and virtual sensors.
- Nodes (console scripts):
  - `ackermann_cmd_bridge`
  - `wheel_encoder_node`
  - `suspension_sensor_node`
  - `steering_sensor_node`
  - `virtual_sensors_node`
  - `water_pressure_node`
  - `water_flow_node`
  - `water_temp_in_node`
  - `water_temp_out_node`
  - `water_temp_radiator_node`
  - `brake_temp_fr_node`
  - `brake_temp_rl_node`
  - `pitot_dynamic_pressure_node`
- Launch files:
  - `sim_car/launch/gazebo_sim.launch.py`
  - `sim_car/launch/nodes.launch.py`
  - `sim_car/launch/full_sim_launch.launch.py`
- Configs:
  - `sim_car/config/sensor_config.yaml`
  - `sim_car/config/eufs_config.yaml`
- Assets: `urdf/`, `worlds/`, `models/`, `meshes/`, `materials/`.

## steering_gui (`steering_gui/`)

- Purpose: RQT steering GUI for sending Ackermann commands and brake commands.
- Executable script:
  - `scripts/eufs_robot_steering_gui`
- Nodes: GUI plugin runs inside RQT; publishes to an Ackermann topic and brake topic.
- Launch files: none (launched as an RQT plugin or via `ros2 run`).
- Assets: `resource/EUFSRobotSteeringGUI.ui`, `plugin.xml`.

## vectornav_decoder (`vectornav_decoder/`)

- Purpose: VN-200 serial decoder for IMU/GPS/INS data.
- Nodes (console scripts):
  - `vectornav_decoder_node`
  - `vectornav_monitor_node`
- Launch files:
  - `vectornav_decoder/launch/vectornav_decoder.launch.py`
  - `vectornav_decoder/launch/vectornav_monitor.launch.py`
- Configs:
  - `vectornav_decoder/config/default_output.yaml`
  - `vectornav_decoder/config/high_rate_imu.yaml`

## vehicle_plotter (`vehicle_plotter/`)

- Purpose: Aggregate sensor data into `VehicleState`, plot in real time, log to disk, and manage rosbag recording.
- Nodes (console scripts):
  - `data_collector_node`
  - `plotter_node`
  - `logger_node`
  - `rosbag_controller_node`
  - `session_manager_node`
- Launch files:
  - `vehicle_plotter/launch/plotter.launch.py`
  - `vehicle_plotter/launch/offline_replay.launch.py`
  - `vehicle_plotter/launch/replay.launch.py`
  - `vehicle_plotter/launch/vcan_test.launch.py`
- Configs:
  - `vehicle_plotter/config/default_plots.yaml`
  - `vehicle_plotter/config/gazebo_topics.yaml`
  - `vehicle_plotter/config/can_topics.yaml`
  - `vehicle_plotter/config/vectornav_topics.yaml`
  - `vehicle_plotter/config/rosbag_topics.yaml`
  - `vehicle_plotter/config/qos_overrides.yaml`
  - `vehicle_plotter/config/replay_qos_override.yaml`

## vehicle_plotter_msgs (`vehicle_plotter_msgs/`)

- Purpose: Custom messages used by vehicle_plotter.
- Messages:
  - `vehicle_plotter_msgs/msg/VehicleState.msg`
  - `vehicle_plotter_msgs/msg/RunSession.msg`
- Nodes: none.
- Launch files: none.
