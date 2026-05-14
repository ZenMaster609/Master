# Codebase Navigation Guide

A quick-reference map of where things live and where to add new things.

---

## Folder Structure

```
Master/
├── sim_car/                          # Simulation, sensors, perception, planning, control
│   ├── sim_car/
│   │   ├── cones/                    # Cone detection, tracking, and memory
│   │   │   ├── nodes/                #   ROS2 nodes (evaluator, memory)
│   │   │   ├── tracking/             #   Algorithms (fusion, global memory, tracker)
│   │   │   └── plotting/             #   Runtime visualisation
│   │   ├── controllers/              # Steering controllers (pure pursuit, stanley)
│   │   ├── lidar/                    # 2D and 3D LiDAR processing
│   │   ├── perception/               # Camera + YOLO cone detection
│   │   ├── planning/                 # Path planners and state machine
│   │   └── sensors/                  # Virtual sensor simulation nodes
│   ├── config/                       # YAML config files (one sub-folder per track type)
│   ├── launch/                       # ROS2 launch files
│   ├── test/                         # Unit and integration tests
│   ├── models/                       # Gazebo 3D models (cones, tracks)
│   ├── worlds/                       # Gazebo world files
│   └── urdf/                         # Robot description
│
├── vehicle_plotter/                  # Logging, live plotting, and offline analysis
│   ├── vehicle_plotter/
│   │   ├── core/                     # Shared contracts: VehicleState, QoS, time sync
│   │   ├── nodes/                    # ROS2 nodes (logger, plotter, session manager)
│   │   ├── logging/                  # Log writers (CSV, Parquet) and path evaluation
│   │   ├── plotting/                 # Live and offline plot definitions
│   │   ├── analysis/                 # Shared post-run analysis utilities
│   │   ├── adapters/                 # External data adapters (Gazebo)
│   │   └── utils/                    # Ring buffer, transforms
│   ├── launch/                       # plotter.launch.py
│   └── test/
│
├── vehicle_plotter_msgs/             # Custom ROS2 message definitions
│   └── msg/                          # VehicleState.msg, ConeDetection.msg, RunSession.msg
│
├── eufs_remastered/                  # Third-party dynamics + steering GUI
│   ├── eufs_gz_dynamics/             # Gazebo vehicle dynamics plugin (C++)
│   ├── eufs_models/                  # Vehicle model implementations (C++)
│   ├── eufs_msgs/                    # EUFS-specific ROS2 messages
│   ├── gazebo_cone_plugins/          # Ground-truth cone provider plugin (C++)
│   └── steering_gui/                 # RQT steering control GUI
│
├── documentation/
│   ├── code/                         # Implementation docs (← you are here)
│   ├── concepts/                     # High-level concept explanations
│   └── math/                         # Mathematical derivations
│
├── tools/                            # Standalone scripts (plot_parquet.py, etc.)
├── utilities/                        # plot_logs.py
├── docker/                           # Docker and container setup
└── scripts/                          # Build helpers
```

---

## Where to Add Things

### New ROS2 node
| Sub-system | Add node file | Register in |
|---|---|---|
| Cone pipeline | `sim_car/sim_car/cones/nodes/` | `sim_car/setup.py` `console_scripts` + relevant launch file |
| Planning | `sim_car/sim_car/planning/` | same as above |
| Sensors | `sim_car/sim_car/sensors/` | same as above |
| Plotter / logger | `vehicle_plotter/vehicle_plotter/nodes/` | `vehicle_plotter/setup.py` + `launch/plotter.launch.py` |

### New configuration parameter
1. Declare the param in the relevant `config/<track>/xxx.yaml`.
2. Add the field to the matching config dataclass (e.g. `planner_config_base.py` for planners).
3. Add a one-line comment with units and reason for the default value.
4. If it's used in more than one file, put it in the shared constants file (see below).

### New shared constant
- **Planning constants** → `sim_car/sim_car/planning/planner_constants.py`
- **Steering/convention constants** → `sim_car/sim_car/sensors/steering_convention.py`
- **Topic name utilities** → `sim_car/sim_car/sensors/topic_utils.py`
- **Analysis utilities** → `vehicle_plotter/vehicle_plotter/analysis/analysis_utils.py`

Never define the same constant in two files — import from one of the above.

### New geometry / math helper
- Pure geometry (no ROS) → `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`
- Algorithm helpers shared across `*_core.py` files → `sim_car/sim_car/planning/planner_utils.py` (create if absent)
- Post-run analysis math → `vehicle_plotter/vehicle_plotter/analysis/analysis_utils.py`

### New ROS2 message type
- Add a `.msg` file to `vehicle_plotter_msgs/msg/` (or `eufs_remastered/eufs_msgs/msg/` if EUFS-specific).
- Register it in the package's `CMakeLists.txt`.
- Rebuild: `cd ~/ros2_ws && colcon build --symlink-install --packages-select vehicle_plotter_msgs`

### New Gazebo model (cone, track element, etc.)
- Add a model directory under `sim_car/models/<model_name>/` with a `model.sdf` and `model.config`.
- Reference it in the relevant `.world` file in `sim_car/worlds/`.

### New test
- `sim_car/test/` for anything in the `sim_car` package.
- `vehicle_plotter/test/` for anything in `vehicle_plotter`.
- Run tests: `cd ~/ros2_ws && colcon test --packages-select sim_car && colcon test-result --verbose`

### New plot or analysis view
- **Live plotting**: add a panel/series in `vehicle_plotter/vehicle_plotter/plotting/plot_definitions.py`.
- **Offline plotting**: extend `vehicle_plotter/vehicle_plotter/plotting/offline_plotter.py` or `offline_cone_plotter.py`.

---

## Key Config Files

| File | What it controls |
|---|---|
| `sim_car/config/eufs_config.yaml` | Vehicle dynamics (mass, wheelbase, tyre params) |
| `sim_car/config/sensor_config.yaml` | Which sensors are active and their noise model |
| `sim_car/config/cone_memory.yaml` | Cone fusion thresholds and memory parameters |
| `sim_car/config/stereo_calibration.yaml` | Stereo camera intrinsics and extrinsics |
| `sim_car/config/<track>/spawn.yaml` | Car spawn pose for each track type |
| `sim_car/config/<track>/pure_pursuit_controller.yaml` | Pure-pursuit gains per track |
| `sim_car/config/<track>/stanley_controller.yaml` | Stanley gains per track |
| `sim_car/config/<track>/linetest.yaml` | Line-test planner params per track |
| `sim_car/config/skidpad/skidpad_router.yaml` | Skidpad router geometry params |

---

## Launch Files

| File | What it starts |
|---|---|
| `sim_car/launch/full_sim_launch.launch.py` | Everything: Gazebo, all nodes, plotter |
| `sim_car/launch/gazebo_sim.launch.py` | Gazebo environment only |
| `sim_car/launch/nodes.launch.py` | All non-Gazebo ROS2 nodes |
| `sim_car/launch/clock_only.launch.py` | Sim clock without physics (useful for replay) |
| `vehicle_plotter/launch/plotter.launch.py` | Logger + plotter nodes only |

---

## Deeper Documentation

Each sub-system has a dedicated doc in this folder:

| Doc | Covers |
|---|---|
| [planning_system.md](planning_system.md) | Planner architecture, state machine, config fields |
| [cones.md](cones.md) | Cone tracking pipeline and memory node |
| [perception.md](perception.md) | YOLO detection, stereo depth, monocular depth |
| [2d_lidar.md](2d_lidar.md) | 2D LiDAR clustering and cone extraction |
| [sensors.md](sensors.md) | Virtual sensor model and noise layer |
| [steering_controllers.md](steering_controllers.md) | Pure pursuit and Stanley controller |
| [vehicle_plotter.md](vehicle_plotter.md) | Logging, live plotting, session management |
| [gazebo_cone_plugins.md](gazebo_cone_plugins.md) | Ground-truth cone provider plugin |
| [eufs_and_steering_gui.md](eufs_and_steering_gui.md) | EUFS dynamics and steering GUI |
| [reproducible_environment.md](reproducible_environment.md) | Docker and dependency setup |
