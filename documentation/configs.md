# Configuration Files

This file lists the main configuration files used by the packages in this workspace.

## sim_car

### `sim_car/config/sensor_config.yaml`

- Used by `sim_car`'s `measurement_node` to convert `/sim/raw/*` topics into `/sim/*` topics with optional noise/latency/dropout.
- Top-level keys:
  - `seed`: RNG seed for repeatability.
  - `signals`: per-signal settings including `input_topic`, `output_topic`, `msg_type`, `rate_hz`, `latency_ms`, `dropout_prob`, `noise_std`, `bias_init`, `bias_rw_std`, `saturation_min`, `saturation_max`, `apply_orientation`.

### `sim_car/config/eufs_config.yaml`

- Vehicle model and control settings consumed by `sim_car` launch files.
- Sections: `inertia`, `kinematics`, `tire`, `aero`, `dynamics`, `control`, `input_ranges`.

## vehicle_plotter

### `vehicle_plotter/config/default_plots.yaml`

- Defines plot layout and which fields appear in each plot panel.
- Used by `plotter_node` for UI layout.

### `vehicle_plotter/config/gazebo_topics.yaml`

- Topic mappings and sensor rates for the Gazebo adapter.
- Keys: `adapter`, `topics`, `sensor_rates`, `gps_origin`.

### `vehicle_plotter/config/rosbag_topics.yaml`

- Topic lists for rosbag recording, grouped by mode.
- Modes: `common`, `simulation`, `windows_plotter`, `minimal`.

### `vehicle_plotter/config/qos_overrides.yaml`

- QoS overrides for `ros2 bag record` to handle high-rate topics.

### `vehicle_plotter/config/replay_qos_override.yaml`

- QoS overrides for `ros2 bag play` to match recorded topic QoS.

## eufs_models

### `eufs_remastered/eufs_models/config/noise.yaml`

- Noise parameters for EUFS vehicle model dynamics (required by `eufs_gz_dynamics`).
