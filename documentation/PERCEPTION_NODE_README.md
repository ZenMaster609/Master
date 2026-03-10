# Perception Node Overview

This document explains what `sim_car/perception_node` does, which topics it uses, and how the data flows from stereo images to cone detections and evaluation metrics.

It is intentionally system-level: it describes behavior and interfaces, not implementation details.

## Goal

`perception_node` has two jobs:

1. Produce 3D cone detections from stereo + YOLO.
2. Evaluate depth/detection quality against ground-truth cones and publish metrics.

The main planning/control consumer of its output is `delaunay_planner_node`, which subscribes to the cone detections topic.

## Where It Runs

In `sim_car/launch/full_sim_launch.launch.py`, `perception_node` is launched by default.

Its stereo/eval topics are driven by `topic_prefix`:
- if `measure:=false` then prefix is `/sim`
- if `measure:=true` then prefix is `/sim/raw`

So in full sim, the cone output is typically:
- `/sim/stereo/perception/cones_3d` (no measurement node)
- `/sim/raw/stereo/perception/cones_3d` (with measurement node in front)

## Input Topics

### Core perception inputs

1. Left image
- Parameter: `left_image_topic`
- Default in node: `/sim/raw/stereo/left/image_raw`
- Full-sim override: `<topic_prefix>/stereo/left/image_raw`
- Type: `sensor_msgs/msg/Image`

2. Right image
- Parameter: `right_image_topic`
- Default in node: `/sim/raw/stereo/right/image_raw`
- Full-sim override: `<topic_prefix>/stereo/right/image_raw`
- Type: `sensor_msgs/msg/Image`

3. Left camera info
- Parameter: `left_camera_info_topic`
- Default in node: `/sim/raw/stereo/left/camera_info`
- Full-sim override: `<topic_prefix>/stereo/left/camera_info`
- Type: `sensor_msgs/msg/CameraInfo`

4. Right camera info
- Parameter: `right_camera_info_topic`
- Default in node: `/sim/raw/stereo/right/camera_info`
- Full-sim override: `<topic_prefix>/stereo/right/camera_info`
- Type: `sensor_msgs/msg/CameraInfo`

### Ground-truth and tracking inputs (for evaluation)

5. Ground-truth visible cones
- Parameter: `ground_truth_cones_topic`
- Default: `/ground_truth/cones`
- Type: `eufs_msgs/msg/ConeArrayWithCovariance`
- Used to compute depth/detection error metrics.

6. Ground-truth track cones
- Parameter: `ground_truth_track_topic`
- Default: `/ground_truth/track`
- Type: `eufs_msgs/msg/ConeArrayWithCovariance`
- Used to assign stable cone IDs and per-cone statistics.

7. Odometry (evaluation fallback)
- Parameter: `cone_eval_odom_topic`
- Default: `/sim/odom`
- Type: `nav_msgs/msg/Odometry`
- Used only when TF mapping to track frame is unavailable and source frame is base-like.

## Output Topics

### Main output used by planners/controllers

1. Cone detections (3D)
- Parameter: `cone_detections_topic`
- Default in node: `/sim/raw/stereo/perception/cones_3d`
- Full-sim override: `<topic_prefix>/stereo/perception/cones_3d`
- Type: `vehicle_plotter_msgs/msg/ConeDetectionArray`
- Message contains:
  - `header.stamp` and `header.frame_id`
  - array of cones with:
    - `color`
    - `position` (x, y, z)
    - `confidence`

### Stereo and cone evaluation metrics

All under `eval_topic_prefix`:
- Default in node: `/sim/raw/stereo/eval`
- Full-sim override: `<topic_prefix>/stereo/eval`

Published metrics include:
- `<eval>/epipolar_mean_px` (`std_msgs/msg/Float32`)
- `<eval>/epipolar_median_px` (`std_msgs/msg/Float32`)
- `<eval>/disparity_valid_ratio` (`std_msgs/msg/Float32`)
- `<eval>/depth_valid_ratio` (`std_msgs/msg/Float32`)
- `<eval>/depth_mean_m` (`std_msgs/msg/Float32`)
- `<eval>/yolo/detection_count` (`std_msgs/msg/Int32`)
- `<eval>/yolo/inference_ms` (`std_msgs/msg/Float32`)
- `<eval>/cone_depth_pairs` (`std_msgs/msg/Int32`)
- `<eval>/cone_depth_axis_mae_m` (`std_msgs/msg/Float32`)
- `<eval>/cone_depth_axis_rmse_m` (`std_msgs/msg/Float32`)
- `<eval>/cone_depth_axis_bias_m` (`std_msgs/msg/Float32`)
- `<eval>/cone_depth_range_mae_m` (`std_msgs/msg/Float32`)
- `<eval>/cone_depth_range_rmse_m` (`std_msgs/msg/Float32`)
- `<eval>/cone_depth_sync_dt_ms` (`std_msgs/msg/Float32`)
- `<eval>/cone_depth_yolo_detections` (`std_msgs/msg/Int32`)
- `<eval>/cone_depth_yolo_depth_valid` (`std_msgs/msg/Int32`)
- `<eval>/cone_depth_gt_projected` (`std_msgs/msg/Int32`)
- `<eval>/cone_depth_bbox_matches` (`std_msgs/msg/Int32`)
- `<eval>/cone_depth_cone_id_matches` (`std_msgs/msg/Int32`)
- `<eval>/cone_depth_per_cone` (`std_msgs/msg/String`, table-like text)
- `<eval>/cone_depth_samples` (`std_msgs/msg/String`, range-RMSE samples CSV payload)

### Debug image stream

- Parameter: `camera_debug_topic`
- Default in node: `/sim/raw/stereo/camera_debug`
- Full-sim override: `<topic_prefix>/stereo/camera_debug`
- Type: `sensor_msgs/msg/Image`
- Mode selected by `camera_debug` parameter:
  - `none`, `disparity`, `depth`, `left_rect`, `yolo`

## End-to-End Data Flow

### 1) Stereo synchronization and pairing

Left and right images are buffered separately, time-paired with a max skew (`max_time_diff_sec`), and processed in a worker thread.

This prevents mismatched stereo pairs from contaminating disparity/depth estimates.

### 2) Stereo depth pipeline

For each paired frame:
- decode + rectify
- disparity computation (CPU SGBM or CUDA path based on config)
- depth conversion

Basic stereo quality metrics (epipolar error, valid disparity/depth ratios, mean depth) are published at the perf timer rate.

### 3) YOLO detection on rectified left image

If `yolo_enabled=true`, detections are produced on the rectified left image.

For each detection box, depth is sampled near the center pixel, giving a per-detection depth estimate (`depth_m`).

### 4) 3D cone reconstruction and publish

Each YOLO detection with valid depth is reconstructed to a 3D point in camera coordinates.

Then the node tries to publish in `cone_detections_frame` (default `base_footprint`):
- if transform exists: point is transformed into output frame
- if not: namespaced frame fallback is attempted
- if still not available: message is published in source camera frame with a warning

Color labels are normalized to `blue`, `yellow`, `orange`, `big_orange`, or `unknown`.

### 5) Ground-truth matching and error metrics

The node finds the nearest ground-truth cone packet by timestamp and rejects it if sync delta exceeds `cone_eval_sync_slop_sec`.

Ground-truth cones are projected into image coordinates, then matched with YOLO detections to compute:
- axis depth error metrics (MAE/RMSE/bias)
- range error metrics
- counts: projected cones, valid YOLO depth estimates, bbox matches, cone-ID matches

Per-cone running stats are also tracked using the track reference topic (`/ground_truth/track`).

### 6) Debug/performance loop

At `perf_log_hz`, the node:
- logs and publishes stereo + cone-depth metrics
- publishes per-cone summary table text
- optionally updates live range-binned RMSE plot (`cone_plotting_2`)

## Why These Outputs Matter for the Goal

The planning stack needs consistent 3D cones with color and confidence.

`perception_node` provides that as a single, planner-friendly message (`ConeDetectionArray`), while also exporting detailed health metrics so you can tune:
- stereo quality
- YOLO confidence/IOU settings
- frame/timestamp alignment
- transform correctness

In short:
- `.../cones_3d` is the control input product
- `.../stereo/eval/*` is the quality/tuning telemetry

## Common Parameters to Tune First

1. Detection and reconstruction
- `yolo_enabled`
- `yolo_model_path`
- `yolo_conf_threshold`
- `yolo_iou_threshold`
- `cone_detections_frame`

2. Sync and robustness
- `max_time_diff_sec`
- `queue_size`
- `cone_eval_sync_slop_sec`
- `cone_eval_tf_timeout_sec`

3. Stereo depth quality
- `num_disparities`
- `block_size`
- `uniqueness_ratio`
- `min_depth_m`
- `max_depth_m`
- `disparity_valid_threshold`

4. Debug and observability
- `camera_debug`
- `camera_debug_n_frames`
- `perf_log_hz`
- `cone_plotting_2`

## Run / Build Commands

Build:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car
```

Source:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Run full sim (includes `perception_node`):

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py
```

Run `perception_node` directly (example):

```bash
cd ~/ros2_ws && ros2 run sim_car perception_node
```

Inspect key outputs:

```bash
cd ~/ros2_ws && ros2 topic echo /sim/stereo/perception/cones_3d
cd ~/ros2_ws && ros2 topic list | rg stereo/eval
cd ~/ros2_ws && ros2 topic echo /sim/stereo/eval/cone_depth_axis_rmse_m
```

## Notes

- If your launch uses `measure:=true`, replace `/sim/...` with `/sim/raw/...` for perception topics.
- Cone output frame can become namespaced (for example `sim_car/base_footprint`) depending on TF availability and namespace context.
- Ground-truth inputs are for evaluation only; cone output can still be produced even when GT topics are missing.
