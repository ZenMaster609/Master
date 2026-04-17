# Sensors In `sim_car`

The `sim_car` sensor package creates a two-layer virtual sensor pipeline:

`simulated state or simple model -> /sim/raw/... -> measurement_node -> /sim/...`

Raw topics are idealized sensor values. Measured topics are the same values after configurable rate limiting, latency, noise, dropout, bias, bias drift, and saturation.

## Launch Modes

The full sim launch has three relevant switches:

- `sensor_nodes:=true`: start the raw virtual sensor nodes.
- `measure:=true`: start `measurement_node` and make the stack use `/sim/raw` inputs where applicable.
- `sensor_pipeline:=true`: enable both raw sensor nodes and measurement together, and start the vehicle plotter state dashboard.

`sensor_pipeline:=true` is the normal full path. It implies the raw sensor nodes and measurement layer, and it also makes `plotter_node` publish `/vehicle_plotter/state`. `logging:=true` can also enable the state dashboard/logging path, but it does not by itself start the raw sensor nodes or `measurement_node`.

In normal sensor-pipeline runs, raw nodes publish under `/sim/raw/...`, `measurement_node` republishes under `/sim/...`, and downstream logging/plotting reads the measured topics.

## Enabled Sensors

Sensor enablement and measurement behavior come from:

`sim_car/config/sensor_config.yaml`

The launch file reads this config before creating raw sensor nodes. If all signals associated with a sensor are disabled, that raw sensor node is not started.

The current sensor groups are:

- odometry measurement pass-through from `/sim/raw/odom` to `/sim/odom`
- wheel encoder RPM and wheel speed
- suspension travel
- steering angle
- cooling water pressure, flow, inlet temperature, and outlet temperature
- front-right and rear-left brake temperatures
- pitot dynamic pressure
- disabled placeholders for IMU and NavSat-style signals

The IMU and NavSat entries in `sensor_config.yaml` are currently configuration placeholders. They are not backed by raw sensor nodes in the current `nodes.launch.py`.

## Topic Pattern

The package uses a consistent topic convention:

- `/sim/raw/...`: idealized value from Gazebo or a simplified model.
- `/sim/...`: measured value after `measurement_node`.

Examples:

- `/sim/raw/wheel_encoder/rpm` -> `/sim/wheel_encoder/rpm`
- `/sim/raw/wheel_encoder/speed_mm_s` -> `/sim/wheel_encoder/speed_mm_s`
- `/sim/raw/suspension` -> `/sim/suspension`
- `/sim/raw/steering_angle` -> `/sim/steering_angle`
- `/sim/raw/cooling/water_pressure` -> `/sim/cooling/water_pressure`
- `/sim/raw/brakes/temp_fr` -> `/sim/brakes/temp_fr`
- `/sim/raw/pitot/dynamic_pressure` -> `/sim/pitot/dynamic_pressure`

## Raw Sensor Nodes

Raw sensor nodes are launched by `sim_car/launch/nodes.launch.py`.

### Wheel Encoder

`wheel_encoder_node.py` reads Gazebo joint states and converts wheel rotation into:

- accumulated wheel angle
- wheel RPM
- wheel linear speed in mm/s

The wheel outputs are four-element arrays ordered front-left, front-right, rear-left, rear-right.

### Suspension

`suspension_sensor_node.py` publishes four suspension travel values. The current launch uses synthetic mode, which estimates suspension movement from vehicle motion rather than reading real suspension joints.

Synthetic mode is intentionally simple. It reacts to acceleration, braking, and cornering through pitch and roll gains so the signal behaves plausibly in plots without being a detailed suspension simulation.

### Steering

`steering_sensor_node.py` publishes steering angle on the raw steering topic. Its node-level parameters include publish rate, noise, latency, bias, and dropout, then `measurement_node` can also apply the configured measurement effects.

### Cooling, Brake, And Pitot Sensors

The small virtual sensor nodes are:

- `water_pressure_node.py`
- `water_flow_node.py`
- `water_temp_in_node.py`
- `water_temp_out_node.py`
- `brake_temp_fr_node.py`
- `brake_temp_rl_node.py`
- `pitot_dynamic_pressure_node.py`

They share model logic from `virtual_sensors_model.py` through `virtual_sensors_base.py`.

The model is physics-inspired, not high fidelity:

- cooling pressure and flow increase with speed and throttle-like demand
- water temperatures change gradually through heating and cooling terms
- brake temperatures rise during braking and cool with airflow/ambient cooling
- pitot dynamic pressure follows vehicle speed

The current package uses the dedicated node list above rather than removed legacy aggregate sensor nodes.

## Measurement Node

`measurement_node.py` is the generic measurement-effect layer. It reads each enabled non-`plot_only` signal from `sensor_config.yaml`, subscribes to the configured input topic, and publishes to the configured output topic.

For each signal, the config can set:

- `msg_type`
- `rate_hz`
- `latency_ms`
- `dropout_prob`
- `noise_std`
- `bias_init`
- `bias_rw_std`
- `saturation_min`
- `saturation_max`
- message-field-specific noise for structured messages

This lets the raw sensor node stay focused on the physical value while the YAML config controls how realistic or imperfect the measured value should be.

## Plot-Only Signals

Some entries in `sensor_config.yaml` are marked `plot_only: true`. These do not create measurement processors. They tell the plotting layer how to derive extra plot channels from existing messages, such as position, velocity, and yaw from odometry.

## Practical Data Flow

With `sensor_pipeline:=true`:

1. Gazebo and raw sensor nodes publish idealized `/sim/raw/...` topics.
2. `measurement_node` applies configured imperfections.
3. Measured `/sim/...` topics are consumed by `vehicle_plotter`.
4. `plotter_node` publishes `/vehicle_plotter/state`.
5. `logger_node` writes enabled run artifacts.

The full launch always starts the main `logger_node` for diagnostics and path evaluation, but vehicle-state log chunks require `/vehicle_plotter/state`, which is produced by `plotter_node` when `sensor_pipeline:=true` or `logging:=true`.

This split is useful for debugging:

- wrong raw value: inspect the raw sensor node or model
- wrong delay/noise/dropout: inspect `sensor_config.yaml` or `measurement_node`
- wrong dashboard/log value: inspect `vehicle_plotter` adapters and logging

## Useful Commands

Build:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car vehicle_plotter vehicle_plotter_msgs
```

Source:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Launch the full sensor pipeline:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py sensor_pipeline:=true
```
