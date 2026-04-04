# Migrated Planner Tuning

This document covers the active tracked-cone planners:

- `midpoint`
- `single_boundary`
- `corridor`

The tuning model is intentionally split in two layers:

1. Code defaults inside each planner node define the shared baseline for that planner.
2. Track-specific YAML files under [`sim_car/config/<track>/`](/home/aleks/ros2_ws/src/Master/sim_car/config) contain only real overrides for that planner/track.

The active planner config files are:

- [`sim_car/config/acceleration/midpoint_planner.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/acceleration/midpoint_planner.yaml)
- [`sim_car/config/acceleration/single_boundary_planner.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/acceleration/single_boundary_planner.yaml)
- [`sim_car/config/acceleration/corridor_planner.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/acceleration/corridor_planner.yaml)
- [`sim_car/config/smalltrack/midpoint_planner.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/smalltrack/midpoint_planner.yaml)
- [`sim_car/config/smalltrack/single_boundary_planner.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/smalltrack/single_boundary_planner.yaml)
- [`sim_car/config/smalltrack/corridor_planner.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/smalltrack/corridor_planner.yaml)
- [`sim_car/config/skidpad/midpoint_planner.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/skidpad/midpoint_planner.yaml)
- [`sim_car/config/skidpad/single_boundary_planner.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/skidpad/single_boundary_planner.yaml)
- [`sim_car/config/skidpad/corridor_planner.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/skidpad/corridor_planner.yaml)

## What stays in YAML

Only keep parameters in a track YAML if they are both:

- actually read by that planner, and
- intentionally different from that planner's built-in baseline.

This keeps the files short and makes the remaining knobs high signal.

## Controller tuning

Both controllers remain available for all migrated planners:

- `control.controller_type`
- `stanley.*`
- `pure_pursuit.*`

Controller tuning now lives inline in the planner track YAML when that track really overrides the planner baseline. There is no separate controller config directory anymore.

## Launch-owned values

These are normally supplied by launch and should stay out of the planner YAMLs unless you intentionally need a standalone-node override:

- `topics.tracked_cones_topic`
- `topics.odom_topic`
- `runtime.publish_rate_hz`
- `lap_tracking.*`
- `diagnostics.publish_thesis_context`

## Minimal commands

Rebuild after planner/config changes:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car vehicle_plotter
```

Source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Launch the sim:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py
```
