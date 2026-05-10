# EUFS Simulation Infrastructure And Steering GUI

This page covers the EUFS-remastered simulation packages that support vehicle dynamics, custom messages, and manual steering control.

## Vehicle Models

`eufs_models` is a pure C++ library providing vehicle dynamics models for use in simulation. It has no ROS or Gazebo dependency and can be linked by any component that needs to integrate vehicle physics.

The library defines a common interface for vehicle dynamics models and provides two concrete implementations: a dynamic bicycle model and a simpler point-mass model. Both models consume a control input and produce an updated vehicle state by integrating equations of motion over a timestep.

The library also provides:

- a vehicle parameter loader that reads YAML configuration files
- a vehicle state container
- a control input struct
- a noise model for realistic sensor simulation

### Dynamic Bicycle Model

The dynamic bicycle model (`dynamic_bicycle`) uses Ackermann steering kinematics and tire force equations to simulate realistic lateral and longitudinal vehicle dynamics. It integrates a set of ordinary differential equations over each simulation timestep using the vehicle parameter set.

Parameters that shape bicycle model behavior include:

- inertia: vehicle mass, yaw moment of inertia, center-of-gravity position
- kinematics: wheelbase, track width
- tire: cornering stiffness coefficients for front and rear
- aerodynamics: drag and downforce coefficients
- input ranges: limits on throttle, braking, and steering

### Point-Mass Model

The point-mass model (`point_mass`) is a simplified dynamics model that treats the vehicle as a single mass. It is suitable for low-fidelity simulations or for testing planners and controllers where precise lateral dynamics are not required.

### Vehicle State And Input

`vehicle_state.hpp` defines the state container shared by all models. The state includes position, orientation, linear velocity, angular velocity, wheel speeds, and steering angle. The state is updated in-place by each model's integration step.

`vehicle_input.hpp` defines the input struct consumed by the models. Inputs include throttle, braking force, and steering angle. Input ranges are validated against the configured parameter limits.

### Vehicle Parameters And Noise

Vehicle parameters are loaded from a YAML file using `vehicle_param.hpp`. The loader populates a parameter struct covering all model sections. The parameter file is pointed to by the Gazebo plugin via SDF configuration and can be swapped to change vehicle behavior without recompiling.

`noise.hpp` provides a configurable noise model used to add realistic imperfections to the simulated state before it is published as sensor output. Standard deviations for each state component are set in `config/noise.yaml`. All values are currently zero, meaning the simulation publishes clean state by default. Non-zero values introduce Gaussian noise to position, orientation, and velocity outputs.

## Gazebo Dynamics Plugin

`eufs_gz_dynamics` is a Gazebo Ignition system plugin that connects the `eufs_models` vehicle dynamics library to the Gazebo simulation. It replaces the default Gazebo physics for the race car model with a custom dynamics integration step, allowing realistic vehicle behavior to be driven by a configurable dynamics model.

By default, Gazebo simulates rigid-body physics. The plugin bypasses this for the vehicle and instead:

1. Receives control commands for throttle, braking, and steering from ROS 2 topics.
2. Feeds them into the configured `eufs_models` dynamics model each simulation step.
3. Sets the resulting vehicle pose and velocity directly on the Gazebo model entity.
4. Publishes vehicle state and diagnostics back to ROS 2.

This allows the vehicle behavior to be defined entirely by the mathematical model in `eufs_models` rather than by Gazebo's contact dynamics.

### Plugin Lifecycle

The plugin implements two Gazebo system interfaces:

- **ISystemConfigure**: called once at startup to initialize the ROS 2 node, parse SDF parameters, and load the vehicle dynamics model from its YAML config.
- **ISystemPreUpdate**: called every simulation step to process the latest control command and advance the dynamics model state.

At configure time, the plugin creates a dedicated ROS 2 executor, reads the `<yaml_config>` SDF element, instantiates the selected dynamics model, and resolves the canonical link of the Gazebo model for state injection.

At each simulation step, the plugin takes the most recent command from the command queue, calls the dynamics model integration with the current timestep, writes the resulting pose back to the Gazebo entity, and publishes diagnostic information for monitoring.

### Control Input And State Output

The plugin subscribes to control command topics that carry throttle, steering, and braking values. Commands are timestamped and queued so that the pre-update step always uses the most recently received input.

After each integration step the plugin publishes the updated vehicle state. This includes position, velocity, and orientation derived from the dynamics model, not from Gazebo's own physics integration.

Diagnostic messages are also published with model health and timing information, compatible with standard ROS 2 diagnostic tooling.

### Configuration

The plugin is configured via SDF in the robot model definition. The key SDF element is:

```xml
<yaml_config>/path/to/vehicle_params.yaml</yaml_config>
```

The YAML file pointed to here is the `eufs_models` vehicle parameter file, which controls all dynamics model constants including mass, inertia, tire stiffness, and input limits.

## EUFS Messages

`eufs_msgs` is a small ROS 2 package that defines the custom message types used across the EUFS simulation stack. Other packages in `eufs_remastered` depend on these messages for cone positions, wheel speeds, and related simulation data.

### ConeArrayWithCovariance

The primary cone position message carries arrays of cones separated by color:

- `blue_cones`
- `yellow_cones`
- `orange_cones`
- `big_orange_cones`
- `unknown_color_cones`

Each element in these arrays is a `ConeWithCovariance`. The message includes a standard `std_msgs/Header` for frame and timestamp information.

This message is published by `gazebo_ground_truth_cones` and consumed by any node that needs the raw simulated cone positions from Gazebo.

### ConeWithCovariance

A single cone position with uncertainty. It contains:

- a `geometry_msgs/Point` for the cone's 3D position
- a flat array representing the upper triangle of a 2D position covariance matrix

The covariance allows downstream nodes to weight cone observations by their expected positional uncertainty, which is relevant when the confusion matrix simulation is active in `gazebo_cone_plugins`.

### WheelSpeeds

`WheelSpeeds` carries per-wheel velocity information between simulation and control nodes. It has four floating-point fields:

- `lf`: left front
- `rf`: right front
- `lb`: left back
- `rb`: right back

## Steering GUI

`steering_gui` is an RQT plugin that provides a manual steering control panel for the simulated vehicle. It is intended for development and debugging: a developer can drive the simulated car by hand to test perception, planning, or control behavior without writing a custom command publisher.

During simulation development it is useful to manually steer the vehicle to put it in specific configurations or to quickly verify that the control stack is receiving and responding to commands. The steering GUI provides a graphical slider interface for this purpose inside the standard RQT tool window.

### Interface

The plugin adds a dockable RQT panel containing:

- **Steering slider**: controls the steering angle sent to the vehicle. The range and resolution match the configured Ackermann steering limits.
- **Keyboard shortcuts**: allow quick control from the keyboard without interacting with the slider directly.
- **Topic configuration**: the command topic can be changed at runtime to target different namespaces or vehicle instances.

### Output Topics

The GUI publishes two topics:

- **Ackermann drive command** (`AckermannDriveStamped`): the primary steering and speed command. Published to the configured command topic, which defaults to the `eufs_gz_dynamics` subscription.
- **Brake command** (`Float32`): brake value published to a separate brake topic. Allows the GUI to command braking independently of the drive command.

### Feedback

The plugin subscribes to the vehicle state topic to display current speed or steering angle as feedback. This lets the operator confirm that commands are being received and that the vehicle is responding.

## Useful Commands

Build the related packages:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select eufs_models eufs_gz_dynamics eufs_msgs steering_gui
```

Source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Inspect a message definition:

```bash
cd ~/ros2_ws && source install/setup.bash && ros2 interface show eufs_msgs/msg/ConeArrayWithCovariance
```

Launch the steering GUI as a standalone RQT panel:

```bash
cd ~/ros2_ws && source install/setup.bash && rqt --standalone steering_gui
```
