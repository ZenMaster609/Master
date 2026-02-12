# gazebo_cone_plugins (Fortress remaster)

Remastered cone plugin for ROS 2 Humble + Ignition Gazebo Fortress.

Included features only:

- Ground-truth track cones topic.
- Per-update visible cones topic (camera/lidar FOV model).
- Optional YAML confusion-matrix color misclassification.

## Plugin library and class

- `filename`: `libgazebo_ground_truth_cones.so`
- `name`: `gazebo_plugins::eufs_plugins::GazeboGroundTruthConesRemastered`

Legacy alias supported:

- `gazebo_plugins::eufs_plugins::GazeboGroundTruthCones`

## Core SDF params

- `groundTruthTrackTopicName` (default `/ground_truth/track`)
- `visibleConesTopicName` (default `/ground_truth/cones`)
- `groundTruthConesTopicName` (legacy alias of `visibleConesTopicName`)
- `updateRate` (default `25.0`)
- `trackModelName` (default `track`)
- `carFrameLink` (default `base_footprint`)
- `trackFrame` (`map` or `base_footprint`, default `map`)
- `visibleFrame` (default `base_footprint`)
- `publishTrack` (default `true`)

Visibility params:

- `cameraViewDistance`, `cameraMinViewDistance`, `cameraFOV`
- `lidarOn`, `lidarViewDistance`, `lidarMinViewDistance`, `lidarXViewDistance`, `lidarYViewDistance`, `lidarFOV`

Miscoloring params:

- `enableConeMiscoloring` (default `false`)
- `confusionMatrixYaml` (path to YAML)
- `recolor_config` (legacy alias; also enables miscoloring)

## Confusion matrix

Example file: `config/cone_confusion_matrix.yaml`

Rows are source colors, columns are destination colors.
`undetected` drops cones.
