# Sensors Math

## Scope

This page documents the math used by the simulated and measured sensor pipeline in `sim_car/sim_car/sensors`. It covers generic measurement corruption, virtual cooling/brake/pitot models, wheel and steering conversions, synthetic suspension, and odometry delay.

## Pipeline Map

1. Raw Gazebo or virtual sensor topics are published under `/sim/raw/...`.
2. `sim_car/sim_car/sensors/measurement_node.py::MeasurementNode` applies configurable rate limiting, latency, dropout, noise, bias, bias random walk, and saturation to measured topics.
3. Specialized sensor nodes convert joint states, odometry, commands, or virtual state into sensor-like signals.
4. Downstream packages consume measured `/sim/...` topics or plot-only raw/virtual topics depending on launch configuration.

## Mathematical Building Blocks

### Generic Measurement Model

`sim_car/sim_car/sensors/measurement_node.py::SignalProcessor.process` releases buffered messages only after their timestamp plus configured latency has passed. It also enforces a minimum publish period from `rate_hz` and drops messages by a Bernoulli test using `dropout_prob`.

`SignalProcessor._apply_scalar` applies the scalar measurement model:

```text
bias(t + dt) = bias(t) + Normal(0, bias_rw_std * sqrt(dt))
measured = true + bias + Normal(0, noise_std)
measured = clip(measured, saturation_min, saturation_max)
```

Parameters can be scalar, per-index lists, or per-field dictionaries, so the same math applies to `Float32`, vectors, IMU fields, odometry fields, and generic numeric message fields.

### Virtual Cooling And Fluid Signals

`sim_car/sim_car/sensors/virtual_sensors_model.py::VirtualSensorsModel` updates a simple thermal state model. The water and radiator temperatures integrate first-order heating/cooling terms using explicit Euler steps:

```text
state += dt * rate(state, speed, throttle, ambient)
```

`compute_water_pressure` and `compute_water_flow` are affine proxy models in vehicle speed and throttle. They are not fluid simulations; they provide plausible changing signals for plotting and measurement-pipeline testing.

### Pitot Dynamic Pressure

`VirtualSensorsModel.compute_pitot_pressure` computes:

```text
q = 0.5 * AIR_DENSITY * vehicle_speed^2
```

At very low speed it adds small random variation and clamps the result nonnegative. This gives the plotter a realistic quadratic speed signal without coupling to an aerodynamic model.

### Brake Temperature Model

`VirtualSensorsModel._update_brake_temps` estimates deceleration from speed change, gates heating on active braking and positive deceleration, then applies separate front/rear heating gains plus airflow and ambient cooling terms.

The model is intentionally asymmetric: front brakes heat faster than rear brakes. The output is clamped to a plausible display range.

### Wheel Encoder Conversion

`sim_car/sim_car/sensors/wheel_encoder_node.py::WheelEncoderNode.joint_state_callback` unwraps wheel joint deltas into `[-pi, pi]` before accumulating absolute rotation. `publish_wheel_velocities` computes average angular velocity over the accumulation window:

```text
omega_avg = angle_accum / time_accum
rpm = omega_avg * 60 / (2 * pi)
speed_mm_s = rpm * wheel_radius * 2 * pi / 60 * 1000
```

Windowed averaging reduces sensitivity to individual joint-state timing jitter.

### Steering Angle Conversion

`sim_car/sim_car/sensors/steering_sensor_node.py::SteeringSensorNode.publish_steering` averages front-left and front-right steering joint positions through `sim_car/sim_car/sensors/steering_convention.py::steering_joint_mean_to_deg`. The sign parameter maps Gazebo joint convention to controller/display convention. The node then applies latency, dropout, bias, and Gaussian noise.

### Synthetic Suspension

`sim_car/sim_car/sensors/suspension_sensor_node.py::SuspensionSensorNode._compute_synthetic_mm` estimates:

```text
a_long = (speed - last_speed) / dt
a_lat = speed * yaw_rate
```

It maps longitudinal acceleration to pitch displacement and lateral acceleration to roll displacement:

```text
front = static - pitch_gain * a_long +/- roll_gain * a_lat
rear  = static + pitch_gain * a_long +/- roll_gain * a_lat
```

An optional first-order low-pass filter uses:

```text
alpha = dt / (filter_tau_sec + dt)
filtered = (1 - alpha) * previous + alpha * raw
```

This gives repeatable suspension-like signals even when physical suspension joints are not the desired source.

### Odometry Delay

`sim_car/sim_car/sensors/odom_delay_node.py::OdomDelayNode` stores odometry messages with release time:

```text
release_time = message_stamp + delay_sec
```

The timer publishes buffered messages whose release time is no longer in the future. This creates a controlled lag for testing planners/controllers under delayed odometry.

## Function Reference

| Math operation | Function | Runtime use |
| --- | --- | --- |
| Latency and rate gate | `sim_car/sim_car/sensors/measurement_node.py::SignalProcessor.process` | Delays, downsamples, and drops measured messages. |
| Noise/bias/saturation | `sim_car/sim_car/sensors/measurement_node.py::SignalProcessor._apply_scalar` | Applies configurable measurement corruption to scalar fields. |
| Field parameter resolution | `sim_car/sim_car/sensors/measurement_node.py::SignalProcessor._resolve_param` | Selects scalar, indexed, or field-specific noise/bias settings. |
| Virtual state step | `sim_car/sim_car/sensors/virtual_sensors_model.py::VirtualSensorsModel.step` | Advances cooling and brake temperature state. |
| Pitot pressure | `sim_car/sim_car/sensors/virtual_sensors_model.py::VirtualSensorsModel.compute_pitot_pressure` | Generates dynamic-pressure proxy from speed. |
| Brake temperature | `sim_car/sim_car/sensors/virtual_sensors_model.py::VirtualSensorsModel._update_brake_temps` | Generates front/rear brake thermal signals. |
| Wheel RPM | `sim_car/sim_car/sensors/wheel_encoder_node.py::WheelEncoderNode.publish_wheel_velocities` | Converts accumulated wheel angle to RPM and speed. |
| Wheel speed conversion | `sim_car/sim_car/sensors/wheel_encoder_node.py::WheelEncoderNode._rpm_to_mm_s` | Converts RPM to linear mm/s from wheel radius. |
| Steering conversion | `sim_car/sim_car/sensors/steering_sensor_node.py::SteeringSensorNode.publish_steering` | Publishes averaged, signed, noisy steering angle. |
| Steering convention | `sim_car/sim_car/sensors/steering_convention.py::steering_joint_mean_to_deg` | Central conversion from joint radians to displayed/controller degrees. |
| Synthetic suspension | `sim_car/sim_car/sensors/suspension_sensor_node.py::SuspensionSensorNode._compute_synthetic_mm` | Maps acceleration proxies to four suspension displacements. |
| Odometry release delay | `sim_car/sim_car/sensors/odom_delay_node.py::OdomDelayNode._flush_ready_messages` | Republishes odometry after a fixed time lag. |

## Notes / Limits

- The generic measurement node corrupts values independently per scalar field. It does not model cross-axis covariance.
- Bias random walk scales with `sqrt(dt)`, so it behaves like a discrete Brownian increment under variable update intervals.
- Virtual cooling, brake, pressure, and flow models are signal-generation models, not validated physical component models.
- The wheel encoder accumulates absolute wheel rotation, so it measures speed magnitude rather than signed wheel travel.
- Synthetic suspension uses simple acceleration proxies from odometry. It is suitable for dashboard/logging behavior, not suspension dynamics validation.
