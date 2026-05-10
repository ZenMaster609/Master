# Gazebo Cone Plugins Code Map

This page maps the `documentation/concepts/gazebo_cone_plugins.md` behavior to the Gazebo ground-truth cone plugin implementation.

## Primary Files

- `eufs_remastered/gazebo_cone_plugins/src/gazebo_ground_truth_cones.cpp`
- `eufs_remastered/gazebo_cone_plugins/config/cone_confusion_matrix.yaml`

## Function Map

### Plugin Lifecycle

- Ground-truth cone plugin class in `eufs_remastered/gazebo_cone_plugins/src/gazebo_ground_truth_cones.cpp`: Gazebo `ISystemConfigure` and `ISystemPreUpdate` implementation; initializes the ROS 2 publisher, enumerates cone entities at configure time, and publishes ground-truth positions every simulation step.
- Configure implementation in `eufs_remastered/gazebo_cone_plugins/src/gazebo_ground_truth_cones.cpp`: scans the Gazebo `EntityComponentManager` for entities whose names match `blue_cone_*`, `yellow_cone_*`, `orange_cone_*`, and `big_cone_*`; caches entity handles and loads the confusion matrix YAML.
- PreUpdate implementation in `eufs_remastered/gazebo_cone_plugins/src/gazebo_ground_truth_cones.cpp`: reads the world pose of each cached cone entity, applies the confusion matrix probability distribution, assembles a `ConeArrayWithCovariance`, and publishes it.

### Cone Classification

- Entity name parsing in `eufs_remastered/gazebo_cone_plugins/src/gazebo_ground_truth_cones.cpp`: determines the true cone color from the Gazebo entity name prefix before confusion matrix application.
- Confusion matrix application in `eufs_remastered/gazebo_cone_plugins/src/gazebo_ground_truth_cones.cpp`: samples from the per-color output distribution to decide the published color for each cone; uses the loaded YAML probability rows.

### Configuration

- `eufs_remastered/gazebo_cone_plugins/config/cone_confusion_matrix.yaml`: color misclassification probability table; currently an identity matrix (all cones published with their true color); modify non-diagonal values to introduce simulated color noise.

## Related Entry Points

- `eufs_remastered/eufs_msgs/msg/ConeArrayWithCovariance.msg`: message type published by the plugin; each of the five color arrays contains `ConeWithCovariance` entries.
- `eufs_remastered/gazebo_cone_plugins/CMakeLists.txt`: links the plugin against `eufs_msgs`, Gazebo Ignition, and `rclcpp`; registers the plugin with the Gazebo plugin system.
- Vehicle or world SDF file: contains the `<plugin>` block that loads the ground-truth cone plugin and configures the output topic name and confusion matrix path.
- `sim_car/sim_car/cones/nodes/evaluator_node.py`: subscribes to `/ground_truth/cones` (the plugin's output topic) to score cone perception predictions.
