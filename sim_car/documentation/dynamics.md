# sim_car Dynamics

This document explains how the EUFS dynamics stack is integrated in this project, what each model means, and where to edit parameters, meshes, and wiring.

## What “bikemode” and “point mode” mean

- **DynamicBicycle ("bikemode")**: A single-track vehicle model with tire slip, lateral forces, yaw dynamics, and simple aero/drag. It uses slip angle + a Pacejka-style tire curve (B/C/D/E) to compute lateral forces. This is the realistic handling model.
- **PointMass ("point mode")**: A simple point mass with acceleration applied in the steering direction. No tire slip, no yaw inertia model. Yaw is derived from velocity direction. This is the fast, simplified model.

## Where the dynamics live

- **EUFS vehicle models (Dynamics)**:
  - `eufs_sim/eufs_models/src/dynamic_bicycle.cpp`
  - `eufs_sim/eufs_models/src/point_mass.cpp`
  - `eufs_sim/eufs_models/include/eufs_models/*.hpp`
- **Gazebo system wrapper (plugin)**:
  - `eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`

The plugin loads one of the EUFS models and drives the Gazebo entity using the model state.

## How everything is wired together

1. **URDF/Xacro defines the car and plugin**
   - `sim_car/urdf/eufs_car.urdf.xacro` builds the vehicle and embeds the EUFS plugin:
     - `libeufs_gz_dynamics.so`
     - `vehicle_model` (DynamicBicycle or PointMass)
     - `yaml_config` path
     - control and joint names

2. **Launch builds a robot description and injects config**
   - `sim_car/launch/gazebo_sim.launch.py` runs Xacro, patches sensor rates, and forces the `yaml_config` path.

3. **Plugin reads commands and updates the model**
   - `eufs_gz_dynamics` subscribes to `/cmd_vel` by default and converts Twist into steering + velocity/accel.
   - The selected EUFS model updates `state` every tick.
   - The plugin applies pose, velocity, and wheel/steering joint positions in Gazebo.

4. **Meshes and resources**
   - Meshes are under `sim_car/meshes/` (default chassis: `sisu-20d.dae`).
   - `sim_car/launch/gazebo_sim.launch.py` sets `GZ_SIM_RESOURCE_PATH` and `IGN_GAZEBO_RESOURCE_PATH` so Gazebo can find meshes/materials.

## Changing dynamics parameters

Edit the YAML config passed to the model:

- `sim_car/config/eufs_config.yaml`

This file controls:

- **inertia**: mass and yaw inertia
- **kinematics**: wheelbase, axle distances, weight distribution
- **tire**: slip curve (B/C/D/E) and wheel radius
- **aero**: downforce and drag coefficients
- **input_ranges**: limits for acceleration, velocity, steering

The plugin reads this on startup and uses it to size the dynamics (wheel radius, wheelbase, steering rate limit, etc).

## Switching between DynamicBicycle and PointMass

Update the `<vehicle_model>` tag in `sim_car/urdf/eufs_car.urdf.xacro`:

- `DynamicBicycle` for realistic handling
- `PointMass` for simplified behavior

This is the "bikemode" vs "point mode" switch.

## Key plugin controls you can tune

In `sim_car/urdf/eufs_car.urdf.xacro`:

- `update_rate`: dynamics update rate (Hz)
- `control_delay`: command delay to simulate latency
- `steering_lock_time`: time to sweep full steering range
- `command_mode`: `velocity` or `acceleration` input interpretation
- `use_cmd_vel` + `cmd_vel_topic`: command wiring
- Joint name tags: must match URDF joints for steering and wheels

## EUFS components included in this project

- **EUFS model library**: `eufs_sim/eufs_models` provides the vehicle dynamics core.
- **EUFS racecar config examples** (optional): `eufs_sim/eufs_racecar/robots/*/config*.yaml` can be used by pointing `yaml_config` at one of these files.

This project uses its own default config (`sim_car/config/eufs_config.yaml`) and a custom Gazebo system wrapper (`eufs_gz_dynamics`).

## Common changes

- **Change the chassis mesh**: edit `chassis_mesh` in `sim_car/urdf/eufs_car.urdf.xacro`.
- **Change wheelbase/track**: update `sim_car/config/eufs_config.yaml` and ensure the wheel geometry in `sim_car/urdf/eufs_wheels.urdf.xacro` matches.
- **Change command topic**: set `cmd_vel_topic` and/or `use_cmd_vel` in the plugin block.
- **Swap dynamics model**: set `vehicle_model` to `DynamicBicycle` or `PointMass`.
