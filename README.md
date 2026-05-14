# Documentation Index

This folder documents the current `sim_car`, `vehicle_plotter`, and `vehicle_plotter_msgs` setup. The source of truth is the ROS 2 code, launch files, package entry points, and YAML config in this repository.

## Main Runtime


- [Vehicle Plotter](concepts/vehicle_plotter.md): run sessions, `/vehicle_plotter/state`, logging, diagnostics, and generated artifacts.
- [Sensors](concepts/sensors.md): raw `/sim/raw/...` virtual sensors, `measurement_node`, measured `/sim/...` topics, and plot-only signals.
- [Perception](concepts/perception.md): camera perception, YOLO, monocular/stereo ranging, cone evaluation, and cone memory.
- [2D LiDAR](concepts/2d_lidar.md): `LaserScan` cone clustering.
- [Reproducible Environment](concepts/reproducible_environment.md): native ROS 2/Gazebo/perception runtime assumptions.

## Planning And Control

- [Planning System](concepts/planning_system.md): tracked-cone planners, linetest, skidpad routing, runtime states, diagnostics, and tuning model.
- [Steering Controllers](concepts/steering_controllers.md): Stanley and pure-pursuit steering behavior and tuning.

## Simulation Infrastructure

- [EUFS Simulation Infrastructure And Steering GUI](concepts/eufs_and_steering_gui.md): vehicle models, Gazebo dynamics, EUFS messages, and the RQT steering panel.

## Code Reference

- [Code Reference Index](code/README.md): companion pages that map each documentation topic to the source files, classes, and helper functions that implement it.

## Math Reference

- [Cones Math](math/cones.md): frame transforms, detection pairing, track updates, color belief, and boundary-color inference.
- [Controllers Math](math/controllers.md): control-path projection, lookahead selection, Stanley control, pure pursuit, and steering filtering.
- [2D LiDAR Math](math/lidar.md): polar projection, scan-order clustering, and cone-candidate geometry filters.
- [Perception Math](math/perception.md): monocular/stereo depth, camera reconstruction, TF transforms, and candidate deduplication.
- [Planning Math](math/planning.md): boundary chains, cone pairing, centerline construction, path validation, midline memory, and skidpad geometry.
- [Sensors Math](math/sensors.md): measurement corruption, virtual sensor models, wheel/steering conversions, suspension proxies, and odometry delay.

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
