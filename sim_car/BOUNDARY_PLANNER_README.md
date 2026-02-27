# Boundary Planner (Option 2) Overview

This document explains how `boundary_planner_node` drives the car from cone detections using:
- boundary fitting (left/right cone lines)
- centerline generation
- Pure Pursuit steering
- curvature-aware speed control

It focuses on what the node consumes and publishes, and how those signals are used to produce control.

## Goal

Track the center of a cone-defined corridor in real time, using only:
- perceived cones (`sim/stereo/perception/cones_3d`)
- odometry (`sim/odom`)

The node computes steering and speed commands on `/cmd` for the existing sim control bridge.

## Node and Launch Integration

- Node executable: `sim_car boundary_planner_node`
- Main config file: `sim_car/config/boundary_planner.yaml`
- Standalone launch: `sim_car/launch/boundary_planner.launch.py`
- Integrated launch: `sim_car/launch/full_sim_launch.launch.py` with argument:
  - `boundary_planner` (default: `true`)

## Topic Interfaces

### Inputs

1. Cones topic
- Default: `sim/stereo/perception/cones_3d`
- Type: `vehicle_plotter_msgs/msg/ConeDetectionArray`
- Used fields:
  - `header.stamp`: freshness/staleness checks
  - `header.frame_id`: frame handling / TF transform
  - Per cone:
    - `position.x`, `position.y` (and `z` for transform math)
    - `color` (blue/yellow/etc. to split left vs right boundary)
    - `confidence` (filter low-confidence detections)

2. Odometry topic
- Default: `sim/odom`
- Type: `nav_msgs/msg/Odometry`
- Used for current speed estimate:
  - prefers `twist.twist.linear.x` when odom frame is base-like
  - otherwise uses planar magnitude `sqrt(vx^2 + vy^2)`

### Outputs

1. Drive command
- Default: `/cmd`
- Type: `ackermann_msgs/msg/AckermannDriveStamped`
- Fields set:
  - `drive.steering_angle` from Pure Pursuit
  - `drive.speed` from curvature-based speed planner

2. Debug centerline path
- Default: `sim/planner/centerline_path`
- Type: `nav_msgs/msg/Path`
- Frame: active base frame used by the planner for this cycle

3. Debug markers
- Default: `sim/planner/markers`
- Type: `visualization_msgs/msg/MarkerArray`
- Contains:
  - filtered raw cone points by side
  - fitted left/right boundary curves
  - generated centerline
  - selected Pure Pursuit target point
  - text marker (speed, lookahead, steering, curvature)

## End-to-End Data Flow

### 1) Data freshness gating

At each planner tick (default 30 Hz):
- If latest cone message age <= `max_cone_age_s`, use it.
- If stale:
  - reuse the last valid plan only for `hold_last_valid_s`
  - after that:
    - if `stop_if_no_path=true`, publish zero speed and zero steering
    - otherwise hold last command

This prevents driving on old perception for too long.

### 2) Cone filtering and side split

Incoming cones are filtered by:
- confidence (`min_confidence`)
- forward range (`x_min_m` to `x_max_m`)
- lateral range (`abs(y) <= y_abs_max_m`)

Remaining cones are split into:
- left boundary colors (default `blue`)
- right boundary colors (default `yellow`)

Color mapping is parameterized in YAML.

### 3) Frame handling

Planner expects cones in a base-style car frame (`base_footprint` by default).

If cone `header.frame_id` differs:
- tries TF transform into configured base frame
- if that fails and frames are namespaced (example `sim_car/base_footprint/stereo_left_camera`), it auto-resolves a namespaced base frame like `sim_car/base_footprint` and retries

This avoids failures caused only by namespace differences.

### 4) Boundary fitting

For each side independently:
- sort by `x` and keep closest `max_points_per_side`
- fit `y(x)` polynomial:
  - `poly2` (default) or `poly1`
  - optional RANSAC to reject outliers
  - least-squares fallback

A side is considered fit only if it has at least `min_points_per_side`.

### 5) Centerline generation

Sample `x` over a forward horizon and compute centerline:
- both sides available:
  - `y_center = 0.5 * (y_left + y_right) + side_offset_m`
- only left:
  - `y_center = y_left - track_width_m/2 + side_offset_m`
- only right:
  - `y_center = y_right + track_width_m/2 + side_offset_m`
- neither:
  - no valid path

### 6) Pure Pursuit steering

Given current speed `v`:
- lookahead distance:
  - `Ld = lookahead_min_m + lookahead_gain * v`
- choose centerline point whose distance from origin is closest to `Ld`
- compute curvature:
  - `kappa = 2 * y_target / Ld^2`
- steering command:
  - `delta = atan(wheelbase_m * kappa)`
  - clamp to `[-steering_limit_rad, +steering_limit_rad]`

### 7) Speed command

Speed decreases with curvature magnitude:
- `v_des = speed_max_mps / (1 + curvature_speed_gain * abs(kappa))`
- clamp to `[speed_min_mps, speed_max_mps]`
- optional smoothing:
  - `v_cmd = alpha * v_des + (1 - alpha) * v_prev`

This naturally slows in tighter turns and speeds up on straights.

## Practical Behavior Summary

When detections are good:
- boundaries are fit
- centerline is stable
- steering follows lookahead point
- speed adapts to turn tightness

When detections degrade temporarily:
- planner can hold last valid path briefly

When detections are missing too long:
- planner safely commands stop (default behavior)

## Key Parameters to Tune First

1. Geometry and control
- `pure_pursuit.wheelbase_m`
- `pure_pursuit.steering_limit_rad`
- `pure_pursuit.lookahead_min_m`
- `pure_pursuit.lookahead_gain`

2. Path shape stability
- `fitting.max_points_per_side`
- `fitting.ransac_inlier_thresh_m`
- `path_generation.horizon_m`
- `path_generation.sample_count`

3. Speed aggressiveness
- `speed_control.speed_min_mps`
- `speed_control.speed_max_mps`
- `speed_control.curvature_speed_gain`
- `speed_control.lowpass_speed_alpha`

4. Robustness to perception dropouts
- `gating.max_cone_age_s`
- `gating.hold_last_valid_s`
- `pure_pursuit.stop_if_no_path`

## Run / Test Commands

Build:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car
```

Source:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Run full sim with boundary planner enabled (default):

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py
```

Run full sim with boundary planner disabled:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py boundary_planner:=false
```

Run planner standalone with parameter file:

```bash
cd ~/ros2_ws && ros2 run sim_car boundary_planner_node --ros-args --params-file /home/aleks/ros2_ws/src/Master/sim_car/config/boundary_planner.yaml
```

## Notes

- The planner uses only cone detections and odometry; it does not require a global map.
- Output `/cmd` is intended to be consumed by the existing command bridge in this workspace.
- If you see frame transform warnings, inspect cone message `header.frame_id` and TF tree namespaces.
