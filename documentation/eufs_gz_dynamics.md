# EUFS Gazebo Dynamics Plugin

`eufs_gz_dynamics` is a Gazebo Ignition system plugin that connects the `eufs_models` vehicle dynamics library to the Gazebo simulation. It replaces the default Gazebo physics for the race car model with a custom dynamics integration step, allowing realistic vehicle behavior to be driven by a configurable dynamics model.

## Purpose

By default, Gazebo simulates rigid-body physics. The `eufs_gz_dynamics` plugin bypasses this for the vehicle and instead:

1. Receives control commands (throttle, braking, steering) from ROS 2 topics.
2. Feeds them into the configured `eufs_models` dynamics model each simulation step.
3. Sets the resulting vehicle pose and velocity directly on the Gazebo model entity.
4. Publishes vehicle state and diagnostics back to ROS 2.

This allows the vehicle behavior to be defined entirely by the mathematical model in `eufs_models` rather than by Gazebo's contact dynamics.

## Plugin Lifecycle

The plugin implements two Gazebo system interfaces:

- **ISystemConfigure**: called once at startup to initialize the ROS 2 node, parse SDF parameters, and load the vehicle dynamics model from its YAML config.
- **ISystemPreUpdate**: called every simulation step to process the latest control command and advance the dynamics model state.

At configure time, the plugin:

- Creates a dedicated ROS 2 executor and spins it on a background thread.
- Reads the `<yaml_config>` SDF element to find the vehicle parameter file.
- Instantiates the selected dynamics model from `eufs_models`.
- Resolves the canonical link of the Gazebo model for state injection.

At each simulation step, the plugin:

- Takes the most recent command from the command queue.
- Calls the dynamics model integration with the current timestep.
- Writes the resulting pose back to the Gazebo entity via joint or link control.
- Publishes diagnostic information for monitoring.

## Control Input

The plugin subscribes to control command topics that carry throttle, steering, and braking values. Commands are timestamped and queued so that the pre-update step always uses the most recently received input.

## State Output

After each integration step the plugin publishes the updated vehicle state. This includes position, velocity, and orientation derived from the dynamics model, not from Gazebo's own physics integration.

Diagnostic messages are also published with model health and timing information, compatible with standard ROS 2 diagnostic tooling.

## Configuration

The plugin is configured via SDF in the robot model definition. The key SDF element is:

```xml
<yaml_config>/path/to/vehicle_params.yaml</yaml_config>
```

The YAML file pointed to here is the `eufs_models` vehicle parameter file, which controls all dynamics model constants including mass, inertia, tire stiffness, and input limits.

## Useful Commands

Build the plugin:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select eufs_gz_dynamics eufs_models
```

Source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```

The plugin is loaded automatically when Gazebo launches the model that includes it in its SDF definition.
