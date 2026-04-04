# Perception

`sim_car` uses a single camera-perception node to detect cones and estimate their distance. The same downstream topic contract is used in both camera modes, so cone memory, planner, and evaluator nodes do not need mode-specific logic.

## Shared architecture

Runtime flow:

`stereo cameras -> perception_node -> ConeDetectionArray -> cone_evaluator_node -> cone_memory_node / planner`

Shared responsibilities:

- YOLO predicts cone bounding boxes and class labels from the left camera image.
- The perception node converts each detection into a 3D cone position in the configured output frame.
- The node publishes `vehicle_plotter_msgs/ConeDetectionArray` on `/.../stereo/perception/cones_3d`.
- `cone_evaluator_node` compares those detections against simulation ground truth and publishes cone RMSE samples for plotting/logging.

The camera mode is selected only by `stereo:=true/false` in [`full_sim_launch.launch.py`](/home/aleks/ros2_ws/src/Master/sim_car/launch/full_sim_launch.launch.py).

## Stereo

Stereo mode uses both camera streams:

1. The node time-pairs the left and right frames.
2. The stereo pipeline rectifies both images using the stereo calibration file.
3. Disparity is computed from the rectified pair.
4. Depth is recovered from disparity using focal length and baseline.
5. YOLO runs on the rectified left image.
6. For each cone bbox, the node samples the lower-central region of the depth map and uses the median valid depth as the cone axis depth.
7. The 2D bbox center and estimated axis depth are reconstructed into a 3D cone point and transformed into the output frame.

Why this structure is useful:

- YOLO handles semantic detection and cone class.
- Stereo provides direct geometric distance estimates.
- The lower-central bbox crop is more stable than using the full box, because it biases the depth sample toward the cone body/base instead of background pixels around the edges.

## Monocular

Monocular mode uses only the left camera stream:

1. YOLO runs on the left image.
2. The bbox height is converted to depth with a pinhole-camera model:

`Z = fy * H / (h - delta)`

Where:

- `fy` is the vertical focal length in pixels
- `H` is the assumed real cone height
- `h` is the detected bbox height in pixels
- `delta` is `monocular_bbox_height_offset_px`

Cone height assumptions:

- standard cones use `monocular_cone_height_m`
- big orange cones use `monocular_big_cone_height_m`

`monocular_bbox_height_offset_px` is treated as an empirical correction term. It compensates for systematic bbox-height bias from the detector and image geometry. In the current codebase it is a manually maintained runtime parameter, not a separate fitting subsystem.

Why this structure is useful:

- It gives a simple, explainable distance estimate when stereo is disabled.
- It keeps the mono pipeline cheap and easy to describe in a thesis.
- It shares the same downstream output interface as stereo mode.

## Output and downstream use

Both modes publish:

- cone class (`blue`, `yellow`, `orange`, `big_orange`, `unknown`)
- confidence
- 3D position in the configured output frame

That output is consumed by:

- `cone_evaluator_node` for RMSE/classification logging
- `cone_memory_node` for camera/lidar fusion into `/tracked_cones`
- the active tracked-cone planner (`midpoint`, `single_boundary`, or `corridor`) either through `/tracked_cones` or directly from the camera cone topic when cone memory is disabled

This common interface is intentional: only the distance-estimation method changes between stereo and monocular modes.

## Design rationale

The camera perception stack is deliberately split into two separable parts:

- semantic detection: YOLO finds cones and predicts their class
- geometric ranging: stereo disparity or monocular bbox-height geometry estimates distance

That separation is useful for explanation and maintenance:

- it makes it clear which part is responsible for classification errors versus range errors
- it lets mono and stereo share the same detector and downstream integration
- it keeps the planner and fusion stack independent of the camera ranging method

## Minimal commands

Build:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car vehicle_plotter
```

Source:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Launch stereo:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py stereo:=true
```

Launch monocular:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py stereo:=false
```
