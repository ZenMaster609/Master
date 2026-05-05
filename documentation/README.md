# Documentation Index

This folder documents the current `sim_car`, `vehicle_plotter`, and `vehicle_plotter_msgs` setup. The source of truth is the ROS 2 code, launch files, package entry points, and YAML config in this repository.

## Main Runtime

- [Autonomous Stack Flowcharts](flowcharts/README.md): layered Draw.io system diagrams and thesis-ready exports.
- [Refactored Architecture](refactored_architecture.md): current module boundaries after the planner and logger refactors.
- [Vehicle Plotter](vehicle_plotter.md): run sessions, `/vehicle_plotter/state`, logging, diagnostics, and generated artifacts.
- [Sensors](sensors.md): raw `/sim/raw/...` virtual sensors, `measurement_node`, measured `/sim/...` topics, and plot-only signals.
- [Perception](perception.md): camera perception, YOLO, monocular/stereo ranging, cone evaluation, and cone memory.
- [2D LiDAR](2d_lidar.md): `LaserScan` cone clustering.
- [Reproducible Environment](reproducible_environment.md): native ROS 2/Gazebo/perception runtime assumptions.

## Planning And Control

- [Planner Tuning](planner_tuning.md): launch selections, config overlays, shared planner groups, and controller groups.
- [Planner Geometry Comparison](planner_geometry_comparison.md): shared and planner-specific geometry for midpoint, single-boundary, and corridor centerline construction.
- [Midpoint Planner](midpoint_planner.md): left/right pairing and midpoint centerline generation.
- [Single-Boundary Planner](single_boundary_planner.md): one-boundary fallback planning through inward offsets.
- [Corridor Planner](corridor_planner.md): strict two-boundary corridor sampling.
- [Line Test Planner](linetest_planner.md): fixed straight-line controller test planner.
- [Skidpad Routing](skidpad_routing.md): skidpad/acceleration cone filtering, route state, and parking stop behavior.
- [Stanley Controller](stanley_controller.md): heading/cross-track controller behavior.
- [Pure Pursuit Controller](pure_pursuit_controller.md): lookahead target controller behavior.

## Code Reference

- [Code Reference Index](code/README.md): companion pages that map each documentation topic to the source files, classes, and helper functions that implement it.

## Common Launch Recipes

Build the current packages:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car vehicle_plotter vehicle_plotter_msgs
```

Source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Run the default smalltrack stack:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py
```

Run with the full measured sensor pipeline and live state dashboard:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py sensor_pipeline:=true
```

Run with explicit logging:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py logging:=true
```

Run a planner/controller combination:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=corridor controller:=stanley
```

Run focused package tests:

```bash
cd ~/ros2_ws && colcon test --packages-select sim_car vehicle_plotter vehicle_plotter_msgs
cd ~/ros2_ws && colcon test-result --verbose
```
