# EUFS Messages Code Map

This page maps the `documentation/eufs_msgs.md` behavior to the message definition files in the `eufs_msgs` package.

## Primary Files

- `eufs_remastered/eufs_msgs/msg/ConeArrayWithCovariance.msg`
- `eufs_remastered/eufs_msgs/msg/ConeWithCovariance.msg`
- `eufs_remastered/eufs_msgs/msg/WheelSpeeds.msg`

## Message Definitions

### ConeArrayWithCovariance

- `eufs_remastered/eufs_msgs/msg/ConeArrayWithCovariance.msg`: top-level cone array message; contains a `std_msgs/Header` and five named arrays of `ConeWithCovariance` (one per cone color: blue, yellow, orange, big_orange, unknown_color).

### ConeWithCovariance

- `eufs_remastered/eufs_msgs/msg/ConeWithCovariance.msg`: single-cone message holding a `geometry_msgs/Point` for position and a flat float array for the 2D positional covariance upper triangle.

### WheelSpeeds

- `eufs_remastered/eufs_msgs/msg/WheelSpeeds.msg`: four-element wheel speed message with float fields `lf`, `rf`, `lb`, `rb` for left-front, right-front, left-back, right-back wheel speeds.

## Related Entry Points

- `eufs_remastered/eufs_msgs/CMakeLists.txt`: registers the `.msg` files for code generation and exports the package as an ament dependency.
- `eufs_remastered/eufs_msgs/package.xml`: declares `rosidl_default_generators` build dependency for message generation.
- `eufs_remastered/gazebo_cone_plugins/src/gazebo_ground_truth_cones.cpp`: primary publisher of `ConeArrayWithCovariance`; depends on `eufs_msgs` at link time.
