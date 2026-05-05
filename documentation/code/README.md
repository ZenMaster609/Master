# Code Reference Index

This folder maps the behavior-oriented docs in `documentation/` to the source files, classes, and helper functions that implement them.

Use these pages as companions to the main docs:

- `documentation/*.md`: what the system does
- `documentation/code/*.md`: where that behavior lives in code

Most pages focus on runtime symbols, but some topics are mainly implemented through launch/config/scripts instead of one core algorithm. In those cases the companion page points to launch entry points, parameter loaders, and setup scripts.

## Runtime

- [Vehicle Plotter](vehicle_plotter.md)
- [Sensors](sensors.md)
- [Perception](perception.md)
- [Cones](cones.md)
- [2D LiDAR](2d_lidar.md)

## Planning And Control

- [Planner Tuning](planner_tuning.md)
- [Tracked-Cone Planner Geometry](tracked_cone_planner_geometry.md)
- [Midpoint Planner](midpoint_planner.md)
- [Single-Boundary Planner](single_boundary_planner.md)
- [Corridor Planner](corridor_planner.md)
- [Line Test Planner](linetest_planner.md)
- [Skidpad Routing](skidpad_routing.md)
- [Stanley Controller](stanley_controller.md)
- [Pure Pursuit Controller](pure_pursuit_controller.md)

## Environment

- [Reproducible Environment](reproducible_environment.md)

## Simulation Infrastructure (eufs_remastered)

- [EUFS Vehicle Models](eufs_models.md)
- [EUFS Gazebo Dynamics Plugin](eufs_gz_dynamics.md)
- [EUFS Messages](eufs_msgs.md)
- [Gazebo Cone Plugins](gazebo_cone_plugins.md)
- [Steering GUI](steering_gui.md)
