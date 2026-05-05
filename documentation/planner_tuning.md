# Planner Tuning

This document explains where planner and controller parameters live, what the main parameter groups affect, and how to tune them without changing runtime code.

## Where Parameters Come From

Planner parameters are layered in this order:

1. Node-declared defaults in the planner/controller Python code.
2. Shared tracked-cone defaults in `tracked_cone_planner_contract.py`.
3. Planner-specific defaults in each planner node.
4. Track and controller YAML overlays selected by `full_sim_launch.launch.py`.
5. Launch-time overrides such as `planner_rate_hz`, `planner_odom_delay_ms`, and `planner_odom_lag_compensation_ms`.

The most important files are:

- `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: shared defaults for midpoint, single-boundary, and corridor planners.
- `sim_car/sim_car/planning/*_planner_node.py`: planner-specific defaults and parameter reading.
- `sim_car/sim_car/controllers/*.py`: controller behavior and config dataclasses.
- `sim_car/config/<track>/stanley_controller.yaml`: Stanley tuning overlay per track.
- `sim_car/config/<track>/pure_pursuit_controller.yaml`: pure-pursuit tuning overlay per track.
- `sim_car/config/<track>/spawn.yaml`: spawn pose, speed limits, and some track-level planner limits.
- `sim_car/config/skidpad/skidpad_router.yaml`: skidpad and acceleration routing/parking behavior.

## Launch-Time Selection

Main launch arguments:

- `planner`: `midpoint`, `single_boundary`, `corridor`, `linetest`, or `none`.
- `controller`: `stanley`, `pure_pursuit`, or `none`.
- `track`: `smalltrack`, `skidpad`, or `acceleration`.
- `cone_memory_enabled`: whether planners consume `/tracked_cones` or direct camera detections.
- `planner_rate_hz`: planner/controller timer rate.
- `planner_odom_delay_ms`: optional fixed delay on the odometry feed.
- `planner_odom_lag_compensation_ms`: forward pose projection inside planner/controller transforms.
- `logging`: enable vehicle-state logging/dashboard support.
- `controller_diagnostics`, `thesis_controller_diagnostics`, and `path_tracking_eval`: enable logger-side evaluation artifacts.

Example:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=corridor controller:=stanley
```

## Track Overlays

Track folders under `sim_car/config/` provide controller and speed overlays:

- `smalltrack/`
- `skidpad/`
- `acceleration/`

The selected `spawn.yaml` provides:

- spawn pose
- `speed_control.speed_min_mps`
- `speed_control.speed_max_mps`
- `speed_control.curvature_speed_gain`
- `speed_control.lowpass_speed_alpha`
- optional `planner_limits.max_planner_length_m`

`planner_limits.max_planner_length_m` is converted by launch into:

- `filtering.max_cone_range_m`
- `centerline.max_path_length_m`
- `filtering.planning_horizon_m` for the corridor planner

This is how skidpad shortens the planning horizon without requiring a separate planner YAML.

Planner config selection is registry-based. `midpoint`, `single_boundary`, and `corridor` are the migrated tracked-cone planners and use the shared tracked-cone runtime. `linetest` is configured separately and is allowed on `acceleration` and `smalltrack`. `none` disables planner node launch while keeping the rest of the stack available for sensor/perception checks.

## Shared Planner Groups

### `filtering.*`

Controls which cones are allowed into planning.

Useful parameters:

- `filtering.max_cone_range_m`: farthest usable cone range.
- `filtering.behind_drop_m`: how far behind the car cones may remain usable.
- `filtering.min_confidence`: minimum planner-facing confidence.
- `filtering.min_required_cones`: minimum usable colored cones.
- `filtering.infer_unknown_by_side`: convert unknown cones to blue/yellow from lateral side.
- `filtering.infer_orange_by_side`: convert orange cones to blue/yellow for planning.

Tuning effect:

- Raising `min_confidence` makes planning cleaner but more likely to drop sparse detections.
- Lowering it helps sparse detections but can admit false positives.
- Shortening `max_cone_range_m` reduces unstable far-field geometry.

### `boundary_chain.*`

Controls how cones on one side are chained forward.

Useful parameters:

- `min_step_m` and `max_step_m`: valid distance between neighboring cones.
- `max_heading_change_rad`: maximum turn allowed between chain steps.
- `min_forward_progress_m`: required forward movement.
- `min_chain_length`: minimum accepted side-chain length.

Tuning effect:

- Smaller `max_step_m` rejects gaps but can break chains.
- Larger `max_heading_change_rad` helps tight turns but can connect wrong cones.

### `pairing.*`

Used mainly by midpoint and single-boundary planning.

Useful parameters:

- `min_pair_width_m` and `max_pair_width_m`: accepted track-width window.
- `max_width_jump_m`: maximum width change between neighboring pairs.
- `min_pair_count`: minimum number of accepted left/right pairs.
- `pair_reassignment_margin`: how strongly a new pairing must beat the old choice.
- `enforce_opposite_color_pairing`: require blue/yellow pairs when enabled.

Tuning effect:

- A narrow width window reduces wrong pairings.
- A wide window helps when perception is noisy or cone color is missing.

### `width_estimation.*`

Maintains the expected track width.

Useful parameters:

- `initial_width_m`: starting width prior.
- `min_width_m` and `max_width_m`: allowed width bounds.
- `alpha`: update speed for the width estimate.
- `max_delta_per_update_m`: maximum width change per update.
- `min_trustworthy_pairs`: minimum pairs needed before trusting a measured width.

Tuning effect:

- Higher `alpha` adapts faster but can chase bad pairings.
- Lower `alpha` is more stable but slower to adapt.

### `centerline.*`

Controls final path shape.

Useful parameters:

- `path_resolution_m`: spacing between published path points.
- `max_path_length_m`: maximum planned centerline length.
- planner-specific smoothing window or validation limits.

For migrated tracked-cone planners, `centerline.temporal_alpha` is compatibility-only. Path stability is now mainly handled by midline memory.

### `midline_memory.*`

Shared path-stability layer for migrated planners.

Useful parameters:

- `horizon_m`: stored midline horizon.
- `station_spacing_m`: spacing of memory samples.
- `near_distance_m` and `mid_distance_m`: split near, mid, and far update zones.
- `near_alpha`, `mid_alpha`, `far_alpha`: blend rates by distance.
- `near_max_lateral_shift_m`, `mid_max_lateral_shift_m`, `far_max_lateral_shift_m`: per-update shift caps.
- `control_handoff_distance_m`: distance near the car protected for controller continuity.
- `hold_last_valid_duration_s`: how long a valid midline can be reused.
- `min_buffer_confidence`: minimum confidence before the buffer is discarded.

Tuning effect:

- Lower near alpha and shift caps make steering calmer.
- Higher far alpha lets the plan react earlier to new geometry.
- Too much hold time can hide planner failures; too little can create command dropouts.

### `validation.*`

Rejects unsafe candidate paths.

Useful parameters:

- `min_path_points`
- `min_forward_extent_m`
- `max_near_field_lateral_jump_m`
- `candidate_jump_reject_threshold_m`
- `candidate_jump_recover_frames`
- planner-specific heading/curvature limits

Tuning effect:

- Stricter validation avoids jumps but increases held-path or no-path operation.
- Looser validation can keep moving through sparse detections but risks steering toward bad geometry.

### `speed_control.*`

Converts controller curvature into a speed command.

Useful parameters:

- `speed_min_mps`
- `speed_max_mps`
- `curvature_speed_gain`
- `lowpass_speed_alpha`

The planner commands lower speed for higher curvature. Track overlays set the current main speeds: acceleration fixes min and max at the same high value, while smalltrack and skidpad allow lower speed in corners.

When `controller:=none`, the launch overlays `control.controller_type: none` and the selected planner publishes paths/diagnostics without issuing normal controller commands.

## Controller Groups

### `stanley.*`

Key parameters:

- `k_gain`: cross-track correction strength.
- `heading_gain`: path-heading correction strength.
- `softening_speed_mps`: prevents excessive cross-track correction at low speed.
- `lookahead_idx_offset`: use a later segment for heading.
- `steering_limit_rad`: final steering clamp.
- `steering_lowpass_alpha`: steering smoothing.
- `steering_rate_limit_rad_s`: maximum steering change rate.
- `use_yaw_rate_damping` and `yaw_rate_damping_gain`: yaw-rate damping.
- `cross_track_deadband_m`: ignore small lateral errors.

### `pure_pursuit.*`

Key parameters:

- `lookahead_m`: base lookahead distance.
- `lookahead_gain`: speed-based lookahead increase.
- `min_lookahead_m` and `max_lookahead_m`: lookahead clamp.
- `steering_limit_rad`: final steering clamp.
- `steering_lowpass_alpha`: steering smoothing.
- `steering_rate_limit_rad_s`: maximum steering change rate.

## Useful Commands

Build after code changes, not required for docs-only edits:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car vehicle_plotter vehicle_plotter_msgs
```

Source:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Run smalltrack with corridor and Stanley:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=corridor controller:=stanley
```

Run skidpad with midpoint and pure pursuit:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=skidpad planner:=midpoint controller:=pure_pursuit
```

Run acceleration line test:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=acceleration planner:=linetest controller:=stanley
```
