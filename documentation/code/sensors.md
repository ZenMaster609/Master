# Sensors Code Map

This page maps the `documentation/concepts/sensors.md` behavior to the raw sensor nodes, measurement layer, and launch/config wiring.

## Primary Files

- `sim_car/sim_car/sensors/measurement_node.py`
- `sim_car/launch/nodes.launch.py`
- `sim_car/config/sensor_config.yaml`
- `vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`

## Function Map

### Launch Modes And Enablement

- `generate_launch_description` in `sim_car/launch/nodes.launch.py`: starts the raw sensor nodes and measurement node used by the sensor pipeline.
- `_build_sensor_nodes` in `sim_car/launch/nodes.launch.py`: decides which raw sensor nodes to launch from `sensor_config.yaml`.
- `_load_sensor_config`, `_get_signal_rate`, and `_any_signal_enabled` in `sim_car/launch/nodes.launch.py`: read the config and decide whether a given sensor group is active.

### Measurement Layer

- `MeasurementNode` in `sim_car/sim_car/sensors/measurement_node.py`: loads the measurement config, constructs per-signal processors, and republishes measured `/sim/...` topics.
- `MeasurementNode._load_config` and `MeasurementNode._parse_signal` in `sim_car/sim_car/sensors/measurement_node.py`: parse `sensor_config.yaml` into runtime signal definitions.
- `MeasurementNode._process_all` in `sim_car/sim_car/sensors/measurement_node.py`: drives every configured `SignalProcessor` each timer cycle.
- `SignalProcessor.process` in `sim_car/sim_car/sensors/measurement_node.py`: applies latency, dropout, rate limiting, and noise timing for one signal.
- `SignalProcessor._apply_measurement`, `_apply_generic`, and `_apply_scalar` in `sim_car/sim_car/sensors/measurement_node.py`: mutate each message into its measured form.

### Wheel Encoder

- `WheelEncoderNode` in `sim_car/sim_car/sensors/wheel_encoder_node.py`: converts joint state wheel motion into RPM, accumulated angle, and wheel speed outputs.
- `WheelEncoderNode.joint_state_callback` in `sim_car/sim_car/sensors/wheel_encoder_node.py`: reads wheel joint motion from Gazebo.
- `WheelEncoderNode.publish_wheel_velocities` in `sim_car/sim_car/sensors/wheel_encoder_node.py`: publishes the four-wheel encoder arrays.
- `WheelEncoderNode._rpm_to_mm_s` in `sim_car/sim_car/sensors/wheel_encoder_node.py`: implements the RPM-to-linear-speed conversion.

### Suspension

- `SuspensionSensorNode` in `sim_car/sim_car/sensors/suspension_sensor_node.py`: publishes suspension travel values.
- `SuspensionSensorNode.publish_suspension` in `sim_car/sim_car/sensors/suspension_sensor_node.py`: emits the four-value suspension output.
- `SuspensionSensorNode._compute_synthetic_mm` in `sim_car/sim_car/sensors/suspension_sensor_node.py`: generates the current synthetic suspension estimate from motion cues.
- `SuspensionSensorNode.odom_callback` in `sim_car/sim_car/sensors/suspension_sensor_node.py`: updates the motion state that feeds the synthetic model.

### Steering

- `SteeringSensorNode` in `sim_car/sim_car/sensors/steering_sensor_node.py`: turns steering-joint state into the published steering angle signal.
- `SteeringSensorNode.joint_state_callback` in `sim_car/sim_car/sensors/steering_sensor_node.py`: reads the steering joint angle.
- `SteeringSensorNode.publish_steering` in `sim_car/sim_car/sensors/steering_sensor_node.py`: republishes the steering signal at the configured rate.

### Downstream Vehicle State Consumption

- `GazeboAdapter.setup_subscriptions` in `vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`: subscribes to the measured `/sim/...` outputs.
- `GazeboAdapter.compute_state` in `vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`: converts the measured topics into `VehicleState` data for logging and plotting.

## Related Entry Points

- `sim_car/config/sensor_config.yaml`: source of truth for enable flags, message types, topic names, and measurement behavior.
- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: enables the sensor pipeline with `sensor_nodes:=true`, `measure:=true`, and `sensor_pipeline:=true`.
