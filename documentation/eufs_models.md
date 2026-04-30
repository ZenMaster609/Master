# EUFS Vehicle Models

`eufs_models` is a pure C++ library providing vehicle dynamics models for use in simulation. It has no ROS or Gazebo dependency and can be linked by any component that needs to integrate vehicle physics.

## Purpose

The library defines a common interface for vehicle dynamics models and provides two concrete implementations: a dynamic bicycle model and a simpler point-mass model. Both models consume a control input and produce an updated vehicle state by integrating equations of motion over a timestep.

The library also provides:

- a vehicle parameter loader that reads YAML configuration files
- a vehicle state container
- a control input struct
- a noise model for realistic sensor simulation

## Vehicle Models

### Dynamic Bicycle Model

The dynamic bicycle model (`dynamic_bicycle`) uses Ackermann steering kinematics and tire force equations to simulate realistic lateral and longitudinal vehicle dynamics. It integrates a set of ordinary differential equations (ODEs) over each simulation timestep using the vehicle parameter set.

Parameters that shape bicycle model behavior include:

- inertia: vehicle mass, yaw moment of inertia, center-of-gravity position
- kinematics: wheelbase, track width
- tire: cornering stiffness coefficients for front and rear
- aerodynamics: drag and downforce coefficients
- input ranges: limits on throttle, braking, and steering

### Point-Mass Model

The point-mass model (`point_mass`) is a simplified dynamics model that treats the vehicle as a single mass. It is suitable for low-fidelity simulations or for testing planners and controllers where precise lateral dynamics are not required.

## Vehicle State

`vehicle_state.hpp` defines the state container shared by all models. The state includes:

- position (x, y, z)
- orientation (roll, pitch, yaw)
- linear velocity (vx, vy, vz)
- angular velocity
- wheel speeds and steering angle

The state is updated in-place by each model's integration step.

## Control Input

`vehicle_input.hpp` defines the input struct consumed by the models. Inputs include throttle, braking force, and steering angle. Input ranges are validated against the configured parameter limits.

## Vehicle Parameters

Vehicle parameters are loaded from a YAML file using `vehicle_param.hpp`. The loader populates a flat parameter struct covering all model sections. The parameter file is pointed to by the Gazebo plugin via SDF configuration and can be swapped to change vehicle behavior without recompiling.

## Noise Model

`noise.hpp` provides a configurable noise model used to add realistic imperfections to the simulated state before it is published as sensor output. Standard deviations for each state component are set in `config/noise.yaml`. All values are currently zero, meaning the simulation publishes clean state by default. Non-zero values introduce Gaussian noise to position, orientation, and velocity outputs.

## Useful Commands

Build the models library and the Gazebo plugin that uses it:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select eufs_models eufs_gz_dynamics
```

Source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```
