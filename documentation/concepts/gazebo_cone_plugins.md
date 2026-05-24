# Gazebo Cone Plugins

`gazebo_cone_plugins` is a Gazebo Ignition system plugin that publishes the ground-truth positions of all cones in the simulated world. It is the authoritative source of cone position data used for evaluating perception quality during simulation runs.

## Purpose

During simulation, cones are placed as Gazebo entities in the world. The plugin scans the entity list at each step, finds all cone models by name, reads their poses, and publishes them as a `eufs_msgs/ConeArrayWithCovariance` message. This gives the rest of the stack access to perfect ground-truth cone positions without querying Gazebo through a service call each time.

## Runtime Flow

At configure time the plugin:

1. Initializes a ROS 2 node and publisher.
2. Enumerates all entities in the Gazebo world that match cone name patterns (`blue_cone_*`, `yellow_cone_*`, `orange_cone_*`, `big_cone_*`).
3. Loads the confusion matrix from `config/cone_confusion_matrix.yaml` if present.
4. Reads SDF options such as `publishTrack`, `groundTruthTrackTopicName`, `visibleConesTopicName`, and the legacy `groundTruthConesTopicName`.

At each simulation step the plugin:

1. Reads the current world pose of each cone entity.
2. Publishes the full map-frame track on `/ground_truth/track` when `publishTrack` is enabled.
3. Filters cones by the simulated LiDAR-visible region, optionally applies the cone confusion matrix to simulate misclassification, and publishes visible cones on `/ground_truth/cones`.

## Cone Color Detection

Cone color is determined from the entity name prefix. The naming convention in the Gazebo world SDF must match the expected prefixes for cones to be detected correctly.

## Cone Confusion Matrix

The confusion matrix (`config/cone_confusion_matrix.yaml`) is a probability table that maps each true cone color to a distribution over output colors. This allows the plugin to simulate realistic sensor color errors, such as occasionally misclassifying a blue cone as yellow, unknown, or undetected.

The current configuration is an identity matrix, meaning all cones are published with their true color. Non-identity values can be set to test perception robustness under color noise.

Matrix format:

```yaml
blue:
  blue: 1.0
  yellow: 0.0
  orange: 0.0
  big_orange: 0.0
  unknown_color: 0.0
  undetected: 0.0
yellow:
  blue: 0.0
  yellow: 1.0
  ...
```

## Output

The plugin publishes `eufs_msgs/ConeArrayWithCovariance` on two configurable topics:

- `/ground_truth/track`: the full map-frame track, used by path tracking evaluation and ground-truth midline helpers.
- `/ground_truth/cones`: visible cones around the vehicle, used by `cone_evaluator_node` to score perception predictions.

## Useful Commands

Build the plugin:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select gazebo_cone_plugins eufs_msgs
```

Source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Inspect ground-truth cone output during a simulation run:

```bash
cd ~/ros2_ws && ros2 topic echo /ground_truth/cones
```
