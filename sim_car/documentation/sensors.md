# Sensors and Measurement Pipelines (sim_car)

This document describes every **simulated sensor** in the `sim_car` package and how each measurement is produced. It covers both:

- **Real sensors (simulated via Gazebo/URDF)**: IMU, GPS, joint states, odometry.
- **Virtual sensors (purely simulated nodes)**: wheel RPM, suspension displacement, steering angle, cooling system, brake temps, pitot dynamic pressure.

Where relevant, equations are shown for the internal models used in the nodes.

---

## Global Data Flow Overview

**Dynamics and joints**
- The Gazebo model is driven by the `eufs_gz_dynamics` plugin (see `sim_car/urdf/eufs_car.urdf.xacro` and `eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`).
- It updates wheel/steering joints and vehicle motion, which then feed the joint state and odometry publishers.

**Simulation topics (key producers)**
- `/sim/joint_states`: published by Gazebo joint state publisher plugin.
- `/sim/odom`: published by Gazebo odometry plugin.
- `/sim/imu`: published by Gazebo IMU sensor.
- `/sim/navsat`: published by Gazebo NavSat/GPS sensor.
- `/cmd_vel`: input to `eufs_gz_dynamics` (control).

**Derived/virtual topics (nodes in `sim_car`)**
- `/sim/wheel_encoder/rpm`: wheel encoder node.
- `/sim/suspension`: suspension sensor node.
- `/sim/steering_angle`: steering sensor node.
- `/sim/cooling/*`, `/sim/brakes/*`, `/sim/pitot/*`: virtual sensors node.

---

## Real Sensors (Gazebo/URDF)

### 1) IMU (`/sim/imu`)
**Source**: Gazebo IMU sensor attached to `imu_link` in `sim_car/urdf/eufs_car.urdf.xacro`.

**Pipeline**
1. `eufs_gz_dynamics` updates vehicle state and link motion.
2. Gazebo IMU sensor samples angular velocity and linear acceleration from the link.
3. Gaussian noise is injected per axis in the URDF.
4. Message is published on `/sim/imu`.

**Noise model (per axis)**
- Angular velocity noise: `N(0, 0.001)` rad/s
- Linear acceleration noise: `N(0, 0.01)` m/s^2

**Update rate**: 50 Hz (defined in the URDF).

---

### 2) GPS / NavSat (`/sim/navsat`)
**Source**: Gazebo NavSat sensor attached to `gps_link` in `sim_car/urdf/eufs_car.urdf.xacro`.

**Pipeline**
1. Gazebo computes world position of `gps_link`.
2. NavSat sensor converts position to latitude/longitude/altitude.
3. Gaussian noise is injected in horizontal and vertical position components.
4. Message is published on `/sim/navsat`.

**Noise model**
- Horizontal position noise: `N(0, 0.5)` meters
- Vertical position noise: `N(0, 1.0)` meters

**Update rate**: 50 Hz (defined in the URDF).

---

### 3) Joint States (`/sim/joint_states`)
**Source**: Gazebo joint state publisher plugin in `sim_car/urdf/eufs_car.urdf.xacro`.

**Pipeline**
1. `eufs_gz_dynamics` updates steering and wheel joints.
2. Joint state publisher reads the joint positions/velocities.
3. Publishes to `/sim/joint_states`.

**Joints published**
- Steering hinge joints
- All four wheel joints

**Update rate**: `config['dynamics']['update_rate_hz']` from the YAML config.

---

### 4) Odometry (`/sim/odom`)
**Source**: Gazebo odometry publisher plugin in `sim_car/urdf/eufs_car.urdf.xacro`.

**Pipeline**
1. Gazebo computes chassis pose/velocity in world frame.
2. Odometry plugin publishes `/sim/odom` with `odom` frame and `base_footprint` child.

**Update rate**: 50 Hz (fixed in URDF).

---

## Virtual Sensors (sim_car nodes)

These sensors are simulated in Python nodes under `sim_car/sim_car/`.

---

### 5) Wheel Encoder RPM (`/sim/wheel_encoder/rpm`)
**Node**: `sim_car/wheel_encoder_node.py`

**Inputs**
- `/sim/joint_states` (wheel joint positions)

**Pipeline**
1. Read wheel joint position (radians) from `/sim/joint_states`.
2. Compute delta angle and unwrap across `[-pi, pi]`.
3. Accumulate absolute rotation and elapsed time.
4. Compute average angular speed over a minimum time window.
5. Convert to RPM and publish.

**Equations**
- Angle unwrap per sample:
  - while `delta > pi`, `delta -= 2*pi`
  - while `delta < -pi`, `delta += 2*pi`
- Accumulated angular velocity:
  - `omega_avg = sum(|delta|) / sum(dt)`
- RPM conversion:
  - `rpm = omega_avg * 60 / (2*pi)`

**Stability rules**
- Ignore tiny deltas below `min_delta`.
- Ignore tiny dt below `min_dt`.
- Only update RPM if the time window >= `min_window_sec`.
- Otherwise, reuse last RPM value for stability.

**Update rate**: configurable (default 50 Hz).

---

### 6) Suspension Displacement (`/sim/suspension`)
**Node**: `sim_car/suspension_sensor_node.py`

Two modes are supported:

#### a) Joint-state mode (`mode: joint_states`)
**Inputs**
- `/sim/joint_states` (suspension prismatic joints)

**Pipeline**
1. Read suspension joint positions in meters.
2. Convert to millimeters: `mm = meters * 1000`.
3. Add bias and Gaussian noise.

**Equation**
- `disp_mm = pos_m * 1000 + bias_mm + N(0, sigma_mm)`

#### b) Synthetic mode (`mode: synthetic`)
**Inputs**
- `/sim/odom` (speed and yaw rate)

**Pipeline**
1. Estimate longitudinal acceleration:
   - `a_long = (v - v_prev) / dt`
2. Estimate lateral acceleration:
   - `a_lat = v * yaw_rate`
3. Compute per-corner travel based on pitch/roll gains.
4. Optional 1st-order low-pass filter.
5. Add bias and noise.

**Equations**
- Corner displacements (mm):
  - `FL = static_mm + (-pitch_gain * a_long) + (roll_gain * a_lat)`
  - `FR = static_mm + (-pitch_gain * a_long) - (roll_gain * a_lat)`
  - `RL = static_mm + (pitch_gain * a_long) + (roll_gain * a_lat)`
  - `RR = static_mm + (pitch_gain * a_long) - (roll_gain * a_lat)`

- Optional low-pass filter:
  - `alpha = dt / (tau + dt)`
  - `y_k = (1 - alpha) * y_{k-1} + alpha * x_k`

**Dropout**
- With probability `dropout_probability`, a publish cycle is skipped.

**Update rate**: configurable (default 100 Hz).

---

### 7) Steering Angle (`/sim/steering_angle`)
**Node**: `sim_car/steering_sensor_node.py`

**Inputs**
- `/sim/joint_states` (front steering joints)

**Pipeline**
1. Read front left/right steering joint positions (radians).
2. Average and convert to degrees.
3. Push into a latency buffer.
4. Output the value closest to `now - latency_ms`.
5. Add bias and Gaussian noise.

**Equations**
- Average steering (rad): `delta = (delta_fl + delta_fr) / 2`
- Convert to degrees: `deg = rad * 180 / pi`
- Output (with bias/noise):
  - `deg_out = deg_delayed + bias + N(0, sigma_deg)`

**Dropout**
- With probability `dropout_probability`, a publish cycle is skipped.

**Update rate**: configurable (default 100 Hz).

---

### 8) Cooling System Sensors (`/sim/cooling/*`)
**Node**: `sim_car/virtual_sensors_node.py`

**Inputs**
- `/sim/odom` for vehicle speed `v`
- `/cmd_vel` for throttle proxy

#### a) Water pressure (`/sim/cooling/water_pressure`)
**Model**
- `pressure_bar = 1.5 + 0.05 * v + 0.3 * throttle + N(0, sigma)`

#### b) Water flow (`/sim/cooling/water_flow`)
**Model**
- `flow_Lmin = 30 + 2.0 * v + 15 * throttle + N(0, sigma)`

#### c) Water temperatures (`/sim/cooling/water_temp_*`)
**Model**: simple first-order heating/cooling with airflow cooling.

State variables:
- `T_out` (after engine, hottest)
- `T_rad` (after radiator)
- `T_in` (coolest, before engine)

Discrete update (per tick `dt`):
- Heat generation: `heat = throttle * v * 2.0`
- Radiator cooling: `cool_rad = 0.1 * v * (T_rad - T_amb)`
- Ambient cooling: `cool_amb = 0.01 * (T - T_amb)`

Applied as:
- `T_out += dt * (heat - (T_out - T_in) / tau_cool)`
- `T_rad += dt * ((T_out - T_rad) / tau_heat - cool_rad / 10 - cool_amb)`
- `T_in  += dt * ((T_rad - T_in) / tau_heat - cool_amb)`

Noise is added at publish time.

---

### 9) Brake Temperatures (`/sim/brakes/temp_fr`, `/sim/brakes/temp_rl`)
**Node**: `sim_car/virtual_sensors_node.py`

**Inputs**
- `/sim/odom` for vehicle speed
- `/cmd_vel` and `/sim/brake_cmd` for braking proxy

**Pipeline**
1. Estimate deceleration: `decel = max(0, (v_prev - v) / dt)`.
2. Determine effective braking input.
3. Compute heat input (front gets more than rear).
4. Cool via airflow and ambient convection.

**Equations**
- Effective brake input:
  - `b = max(brake_cmd, braking_from_cmd_vel)`
- Heat generation (simplified):
  - `heat_fr = v * decel * 0.5 * b + 30 * b`
  - `heat_rl = v * decel * 0.3 * b + 20 * b`
- Cooling terms:
  - `cool_air = 0.02 * v * (T - T_amb)`
  - `cool_amb = 0.005 * (T - T_amb)`
- Temperature update:
  - `T += dt * (heat - cool_air - cool_amb)`

Noise is added at publish time.

---

### 10) Pitot Dynamic Pressure (`/sim/pitot/dynamic_pressure`)
**Node**: `sim_car/virtual_sensors_node.py`

**Inputs**
- `/sim/odom` for vehicle speed `v`

**Equation**
- `q = 0.5 * rho * v^2`
- `rho = 1.225 kg/m^3` (sea-level air density)
- Add low-speed noise and clamp to `q >= 0`.

---

## Relevant Files

- Gazebo sensors and plugins: `sim_car/urdf/eufs_car.urdf.xacro`
- Vehicle URDF (full suspension and wheel joints): `sim_car/urdf/car.urdf`
- Sensor nodes:
  - `sim_car/sim_car/wheel_encoder_node.py`
  - `sim_car/sim_car/suspension_sensor_node.py`
  - `sim_car/sim_car/steering_sensor_node.py`
  - `sim_car/sim_car/virtual_sensors_node.py`
- Sensor config: `sim_car/config/sensor_config.yaml`
- Dynamics plugin: `eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`
