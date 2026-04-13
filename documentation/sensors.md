# Sensors in `sim_car`

This document explains the purpose and structure of the code in `sim_car/sim_car/sensors`.

The goal of this package is to create a simple virtual sensor pipeline for the simulator:

`Gazebo / simulated vehicle state -> raw sensor topics -> measurement node -> measured topics`

The raw topics represent the idealized sensor outputs produced directly from simulation or simple physical models. The measurement node then converts those raw signals into more realistic measured signals by adding effects such as latency and noise.

## Overall idea

The sensor package contains two main kinds of components:

- sensor nodes that generate or derive raw sensor values on `/sim/raw/...`
- a measurement node that republishes those signals on `/sim/...` with configurable imperfections

This separation is useful because it keeps the physical meaning of the raw signals clear, while still making it possible to simulate measurement quality separately.

## Main groups of sensors

The package currently covers these groups of signals:

- wheel encoder signals
- suspension travel
- steering angle
- cooling system signals
- brake temperatures
- pitot dynamic pressure

Some of these are derived almost directly from simulator topics, while others are generated from simplified physical models.

## Topic structure

The package follows a consistent topic naming pattern:

- `/sim/raw/...` means the raw, idealized signal
- `/sim/...` means the measured signal after the measurement node has processed it

Examples:

- `/sim/raw/wheel_encoder/rpm` -> `/sim/wheel_encoder/rpm`
- `/sim/raw/suspension` -> `/sim/suspension`
- `/sim/raw/cooling/water_pressure` -> `/sim/cooling/water_pressure`

This makes it easy to distinguish between the clean simulated value and the value that the rest of the stack should treat as a sensor measurement.

## Two layers in the package

### 1. Raw sensor generation

The raw sensor nodes publish the idealized signal. Different nodes do this in different ways:

- some use Gazebo outputs directly and convert them into a more useful form
- some use a simplified model of the physical system and publish a plausible sensor value

For example:

- `wheel_encoder_node.py` reads wheel joint states and converts wheel rotation into RPM and linear wheel speed
- `suspension_sensor_node.py` can either read suspension joint positions directly or synthesize suspension travel from vehicle dynamics
- the cooling, brake, and pitot sensors use a small internal model to produce values that change with vehicle motion and control input

### 2. Measurement effects

`measurement_node.py` sits after the raw sensor nodes. It subscribes to configured `/sim/raw/...` topics and republishes them to `/sim/...`.

Its job is to apply measurement effects such as:

- latency
- output rate limiting
- random dropout
- Gaussian noise
- constant bias
- bias random walk
- saturation limits

This means the raw sensor node is responsible for the physical value, while the measurement node is responsible for the quality of that value as a sensor reading.

## Important files

### `measurement_node.py`

This is the central node for measurement modeling.

It loads a YAML configuration and creates one processor per signal. Each processor:

- subscribes to the raw topic
- buffers incoming messages
- delays them according to the configured latency
- optionally drops samples
- modifies the message fields according to the configured noise and bias model
- republishes the result on the measured topic

The design is generic enough to support both simple scalar topics and structured ROS messages such as odometry or IMU messages.

### `virtual_sensors_model.py`

This file contains the simplified physical model used by several virtual sensor nodes.

It is not a ROS node itself. Instead, it stores internal state and updates values such as:

- water temperatures
- water pressure
- water flow
- brake temperatures
- pitot pressure

The model is driven mainly by:

- vehicle speed
- commanded acceleration or braking
- ambient temperature

This file provides the shared logic so that multiple nodes can use the same underlying sensor model.

### `virtual_sensors_base.py`

This is a helper base class for simple single-output virtual sensor nodes.

It reduces duplication for nodes that:

- subscribe to odometry and command topics
- update the shared virtual sensor model
- publish one scalar sensor output

In practice, this makes the individual sensor nodes shorter and easier to maintain.

### `wheel_encoder_node.py`

This node derives wheel encoder signals from the wheel joint states.

Its main steps are:

- subscribe to raw joint states from Gazebo
- track wheel rotation over time
- unwrap wheel angle changes
- compute average rotational speed
- publish RPM and linear wheel speed

It publishes one value per wheel, so the output is a four-element array for front-left, front-right, rear-left, and rear-right.

### `suspension_sensor_node.py`

This node publishes four suspension travel values.

It supports two modes:

- `joint_states`: use the suspension joint positions directly
- `synthetic`: estimate suspension movement from longitudinal and lateral vehicle dynamics

The synthetic mode is useful when the goal is not exact suspension physics, but a plausible signal that reacts to braking, acceleration, and cornering.

### Single-sensor nodes

The package also contains small dedicated nodes such as:

- `water_pressure_node.py`
- `water_flow_node.py`
- `water_temp_in_node.py`
- `water_temp_out_node.py`
- `brake_temp_fr_node.py`
- `brake_temp_rl_node.py`
- `pitot_dynamic_pressure_node.py`
- `steering_sensor_node.py`

These nodes follow the same general pattern:

- read the relevant simulator or model inputs
- compute one physical quantity
- publish that quantity on a raw topic

Splitting them into separate nodes makes the pipeline easier to inspect and launch.

## How the modeled sensors work

The modeled sensors are not intended to be high-fidelity physical simulations. They are simplified, physics-inspired approximations.

For example:

- water pressure and flow increase with speed and throttle
- water temperatures evolve gradually based on heating and cooling dynamics
- brake temperatures rise under braking and cool down with airflow and ambient cooling
- pitot pressure is computed from dynamic pressure, which depends on vehicle speed

This is a practical compromise: the values react in a believable way without requiring a full thermal or fluid simulation.

## How measurement effects are configured

The measurement behavior is configured in `sim_car/config/sensor_config.yaml`.

For each signal, the config can define:

- input topic
- output topic
- message type
- publish rate
- latency
- dropout probability
- noise standard deviation
- bias and bias random walk
- saturation limits

Because this configuration is external, the same raw sensor generation code can be reused while easily changing how realistic or noisy the measured signals should be.

## Practical data flow in the project

In the current simulation setup, the intended path is:

1. Gazebo or a simple internal model produces the raw quantity.
2. A sensor node publishes it on `/sim/raw/...`.
3. `measurement_node` applies imperfections and republishes on `/sim/...`.
4. `vehicle_plotter` and the rest of the stack consume the measured topics.

This keeps the architecture easy to understand:

- sensor nodes create the value
- measurement node degrades the value
- downstream nodes use the degraded value as the sensor reading

## Why this structure is useful

From a system-design point of view, the package separates three different concerns:

- physical signal generation
- sensor measurement quality
- downstream use of the signal

That makes it easier to explain and reason about the system:

- if the value itself is wrong, the issue is likely in the raw sensor node or model
- if the value is delayed or noisy, the issue is likely in the measurement configuration
- if the signal is plotted or consumed incorrectly, the issue is downstream

This separation is one of the main strengths of the current sensor package, even though the implementation has been simplified compared to the earlier, more modular version.
