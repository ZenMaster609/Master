# EUFS Messages

`eufs_msgs` is a small ROS 2 package that defines the custom message types used across the EUFS simulation stack. Other packages in `eufs_remastered` depend on these messages for cone positions, wheel speeds, and related simulation data.

## Messages

### ConeArrayWithCovariance

The primary cone position message. It carries arrays of cones separated by color:

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

A four-element wheel speed message with fields:

- `lf`: left front
- `rf`: right front
- `lb`: left back
- `rb`: right back

All values are floating-point. This message is used to carry per-wheel velocity information between simulation and control nodes.

## Usage

These messages are imported by any package in the stack that handles cone ground truth or wheel speed data. The `gazebo_cone_plugins` ground-truth publisher uses `ConeArrayWithCovariance` directly. Packages that bridge simulation state to downstream consumers may use `WheelSpeeds` for odometry or traction control inputs.

## Useful Commands

Build the messages package:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select eufs_msgs
```

Source the workspace and inspect a message definition:

```bash
cd ~/ros2_ws && source install/setup.bash
ros2 interface show eufs_msgs/msg/ConeArrayWithCovariance
```
