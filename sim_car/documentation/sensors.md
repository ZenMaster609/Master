# Sensors (sim_car)

This document describes all **sensor topics** used by the sim stack:
- where they get data from,
- how the value is computed,
- and where the node publishes the result.

All sensor nodes are always launched in `sim_car/launch/nodes.launch.py`.

---

## Data Flow Summary

**Gazebo sources** (from `sim_car/urdf/eufs_car.urdf.xacro`):
- `/sim/imu` (IMU)
- `/sim/navsat` (GPS)
- `/sim/odom` (odometry)
- `/sim/joint_states` (wheel + steering joints)

**Derived sensors** (Python nodes in `sim_car/sim_car/`):
- `/sim/wheel_encoder/rpm`
- `/sim/wheel_encoder/angle_accum`
- `/sim/suspension`
- `/sim/steering_angle`
- `/sim/cooling/*`
- `/sim/brakes/*`
- `/sim/pitot/*`

These topics are consumed by `vehicle_plotter` via the Gazebo adapter to build
`/vehicle_plotter/state` for plotting and logging.

---

## Sensor Details

### IMU (`/sim/imu`)
- **Inputs**: Gazebo IMU sensor attached to `imu_link`.
- **Computation**: Gazebo reports angular velocity + linear acceleration with URDF-configured noise.
- **Output**: `/sim/imu` (sensor_msgs/Imu).

### GPS / NavSat (`/sim/navsat`)
- **Inputs**: Gazebo NavSat sensor attached to `gps_link`.
- **Computation**: Gazebo converts world position to LLA and applies URDF-configured noise.
- **Output**: `/sim/navsat` (sensor_msgs/NavSatFix).

### Odometry (`/sim/odom`)
- **Inputs**: Gazebo odometry plugin on the chassis.
- **Computation**: Gazebo publishes pose + twist in the world frame.
- **Output**: `/sim/odom` (nav_msgs/Odometry).

### Joint States (`/sim/joint_states`)
- **Inputs**: Gazebo joint state publisher plugin.
- **Computation**: Joint positions/velocities for steering and wheels.
- **Output**: `/sim/joint_states` (sensor_msgs/JointState).

---

### Wheel Encoder RPM (`/sim/wheel_encoder/rpm`)
- **Node**: `sim_car/sim_car/wheel_encoder_node.py`
- **Inputs**: `/sim/joint_states` (wheel joint positions)
- **Computation**:
  - Unwrap joint angle deltas across `[-pi, pi]`
  - Accumulate angle over time
  - Compute average angular speed over a minimum window
  - Convert to RPM
- **Output**: `/sim/wheel_encoder/rpm` (std_msgs/Float32MultiArray)

### Wheel Encoder Angle Accum (`/sim/wheel_encoder/angle_accum`)
- **Node**: `sim_car/sim_car/wheel_encoder_node.py`
- **Inputs**: `/sim/joint_states`
- **Computation**: Accumulated wheel rotation in radians per wheel.
- **Output**: `/sim/wheel_encoder/angle_accum` (std_msgs/Float32MultiArray)

---

### Suspension Displacement (`/sim/suspension`)
- **Node**: `sim_car/sim_car/suspension_sensor_node.py`
- **Inputs**: `/sim/joint_states` (or `/sim/odom` in synthetic mode)
- **Computation**:
  - **Joint-state mode**: use suspension joint positions (meters → mm), add bias + noise
  - **Synthetic mode**: estimate accel from odom, apply pitch/roll gains, optional low-pass filter
- **Output**: `/sim/suspension` (std_msgs/Float32MultiArray)

---

### Steering Angle (`/sim/steering_angle`)
- **Node**: `sim_car/sim_car/steering_sensor_node.py`
- **Inputs**: `/sim/joint_states` (front steering joints)
- **Computation**:
  - Average FL/FR steering joint angles (rad)
  - Convert to degrees
  - Apply latency buffer, bias, and noise
- **Output**: `/sim/steering_angle` (std_msgs/Float32)

---

### Cooling Sensors (`/sim/cooling/*`)
- **Nodes**:
  - `sim_car/sim_car/water_pressure_node.py`
  - `sim_car/sim_car/water_flow_node.py`
  - `sim_car/sim_car/water_temp_in_node.py`
  - `sim_car/sim_car/water_temp_out_node.py`
  - `sim_car/sim_car/water_temp_radiator_node.py`
- **Inputs**: `/sim/odom` (speed), `/cmd_vel` (throttle proxy)
- **Computation**:
  - **Pressure**: `1.5 + 0.05*v + 0.3*throttle` + noise
  - **Flow**: `30 + 2.0*v + 15*throttle` + noise
  - **Temps**: first-order thermal model with radiator + ambient cooling
- **Outputs**:
  - `/sim/cooling/water_pressure`
  - `/sim/cooling/water_flow`
  - `/sim/cooling/water_temp_in`
  - `/sim/cooling/water_temp_out`
  - `/sim/cooling/water_temp_radiator`

---

### Brake Temperatures (`/sim/brakes/temp_fr`, `/sim/brakes/temp_rl`)
- **Nodes**:
  - `sim_car/sim_car/brake_temp_fr_node.py`
  - `sim_car/sim_car/brake_temp_rl_node.py`
- **Inputs**: `/sim/odom`, `/cmd_vel`, `/sim/brake_cmd`
- **Computation**:
  - Estimate deceleration from odom
  - Use explicit brake command if present
  - Heat from braking, cool via airflow + ambient
- **Outputs**:
  - `/sim/brakes/temp_fr`
  - `/sim/brakes/temp_rl`

---

### Pitot Dynamic Pressure (`/sim/pitot/dynamic_pressure`)
- **Node**: `sim_car/sim_car/pitot_dynamic_pressure_node.py`
- **Inputs**: `/sim/odom` (speed)
- **Computation**: `q = 0.5 * rho * v^2` with low-speed noise
- **Output**: `/sim/pitot/dynamic_pressure`

---

## Configuration References

- Sensor rates/limits: `sim_car/config/sensor_config.yaml`
- Gazebo sensors and plugins: `sim_car/urdf/eufs_car.urdf.xacro`
- Derived sensor nodes: `sim_car/sim_car/*.py`
- Plotter adapter (topic mapping): `vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`

---

## Adding a New Sensor

### 1) Decide the sensor type
- **Gazebo-native**: add a `<sensor>` in `sim_car/urdf/eufs_car.urdf.xacro`.
- **Derived**: create a Python node in `sim_car/sim_car/` that subscribes to existing topics and publishes a new `/sim/*` topic.

### 2) Add the topic
- **Gazebo-native**: set the sensor topic in the URDF.
- **Derived**: publish under `/sim/...` and add the node to `sim_car/launch/nodes.launch.py`.

### 3) Add config (optional)
- Add rates/limits to `sim_car/config/sensor_config.yaml`.
- Read them in `sim_car/launch/nodes.launch.py` and pass as node parameters.

### 4) Wire into the plotter
- Register the new topic in `vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`:
  - Add to `DEFAULT_TOPICS`
  - Add to `SENSOR_RATES` if it should be time-synchronized
  - Create a subscription and store the last message
  - Map the value into `VehicleState` in `compute_state()`
- If you need a new field, add it to:
  - `vehicle_plotter_msgs/msg/VehicleState.msg`
  - `vehicle_plotter/vehicle_plotter/core/vehicle_state.py`

### 5) Add plots (optional)
- Add a plot in `vehicle_plotter/vehicle_plotter/plotting/plot_config.py` (`get_all_plots()`).

### 6) Rebuild
```
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car vehicle_plotter vehicle_plotter_msgs
cd ~/ros2_ws && source install/setup.bash
```
