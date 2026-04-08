# Migrated Planner Tuning

This document covers the active tracked-cone planners:

- `midpoint`
- `single_boundary`
- `corridor`

For `full_sim_launch`, the common planner pipeline is:

1. cone input selection by launch and the skidpad router
2. frame transform and vehicle pose resolution
3. cone confidence and color gating
4. boundary-chain construction
5. planner-specific line generation
6. candidate acceptance and recovery
7. stored-midline blending and holding
8. controller handoff

The planner-specific parts are intentionally different:

- `midpoint` builds the line from paired left/right cone midpoints.
- `single_boundary` offsets inward from one trusted boundary.
- `corridor` fits the line inside a valid left/right overlap corridor.

## Live Configuration Model

For the migrated planners, `full_sim_launch` does not load per-planner YAML files. The active tests assert that `midpoint_planner.yaml`, `single_boundary_planner.yaml`, and `corridor_planner.yaml` are not part of the launch contract.

The live configuration split is:

1. Planner node defaults define the baseline planner behavior.
2. Track `spawn.yaml` files contribute:
   - spawn pose
   - speed-control defaults
   - optional planner-length overrides such as `planner_limits.max_planner_length_m`
3. Track controller YAML files contribute controller tuning only:
   - [`sim_car/config/acceleration/stanley_controller.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/acceleration/stanley_controller.yaml)
   - [`sim_car/config/acceleration/pure_pursuit_controller.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/acceleration/pure_pursuit_controller.yaml)
   - [`sim_car/config/smalltrack/stanley_controller.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/smalltrack/stanley_controller.yaml)
   - [`sim_car/config/smalltrack/pure_pursuit_controller.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/smalltrack/pure_pursuit_controller.yaml)
   - [`sim_car/config/skidpad/stanley_controller.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/skidpad/stanley_controller.yaml)
   - [`sim_car/config/skidpad/pure_pursuit_controller.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/skidpad/pure_pursuit_controller.yaml)

## Shared Cone Contract

The common pre-planning cone contract for the live planners is:

- transformed `points_xy`
- resolved planner `colors`
- raw detection confidence
- planner confidence after track-state policy
- track id, track state, and track confidence
- boundary-color hint when present
- raw normalized cone color

The only intentional live difference at this stage is tentative-track policy:

- `midpoint` may keep tentative cones when color or boundary hints are strong.
- `single_boundary` rejects tentative cones for planning.
- `corridor` rejects tentative cones for planning.

## Launch-Owned Values

These are supplied by launch for the migrated planners:

- `topics.tracked_cones_topic`
- `topics.odom_topic`
- `runtime.publish_rate_hz`
- `diagnostics.publish_thesis_context`
- `speed_control.*`
- controller selection via `resolved_controller_config`

Only `single_boundary` also consumes launch-provided lap tracking:

- `lap_tracking.gt_track_topic`
- `lap_tracking.target_laps`

Track routing also matters:

- `smalltrack` uses `/tracked_cones`
- `skidpad` and `acceleration` use `/tracked_cones/skidpad_routed` for the migrated planners

## Controller Tuning

Both controllers remain available for all migrated planners:

- `control.controller_type`
- `stanley.*`
- `pure_pursuit.*`

The controller YAMLs under [`sim_car/config`](/home/aleks/ros2_ws/src/Master/sim_car/config) are the live source of per-track controller tuning.

## Minimal Commands

Rebuild after planner/runtime changes:

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
