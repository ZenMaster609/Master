# EUFS Simulation Infrastructure And Steering GUI Code Map

This page maps the `documentation/concepts/eufs_and_steering_gui.md` behavior to the EUFS-remastered source files that implement vehicle dynamics, Gazebo integration, custom messages, and manual steering control.

## Vehicle Models

### Primary Files

- `eufs_remastered/eufs_models/include/eufs_models/eufs_models.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/vehicle_model.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/dynamic_bicycle.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/point_mass.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/vehicle_param.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/vehicle_state.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/vehicle_input.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/noise.hpp`

### Function Map

- `VehicleModelBase` in `eufs_remastered/eufs_models/include/eufs_models/vehicle_model.hpp`: abstract base class defining the shared interface for all vehicle dynamics models; any concrete model must implement the integration step.
- `VehicleModelBase::updateState` in `eufs_remastered/eufs_models/include/eufs_models/vehicle_model.hpp`: overridden by each model to advance the vehicle state by one timestep given a control input.
- `DynamicBicycle` in `eufs_remastered/eufs_models/include/eufs_models/dynamic_bicycle.hpp`: concrete bicycle model class using Ackermann kinematics and tire force equations.
- `DynamicBicycle` implementation in `eufs_remastered/eufs_models/src/dynamic_bicycle.cpp`: ODE integration loop; reads vehicle parameters to compute lateral and longitudinal forces and advances position, velocity, and orientation.
- `PointMass` in `eufs_remastered/eufs_models/include/eufs_models/point_mass.hpp`: simplified dynamics model treating the vehicle as a single mass.
- `PointMass` implementation in `eufs_remastered/eufs_models/src/point_mass.cpp`: integration logic for the point-mass equations of motion.
- `VehicleParam` in `eufs_remastered/eufs_models/include/eufs_models/vehicle_param.hpp`: YAML-based parameter struct; loading function populates inertia, kinematics, tire, aerodynamics, dynamics, and input-range sections from the configured file path.
- `VehicleState` in `eufs_remastered/eufs_models/include/eufs_models/vehicle_state.hpp`: container for full vehicle state, updated in-place by each integration step.
- `Input` in `eufs_remastered/eufs_models/include/eufs_models/vehicle_input.hpp`: control input struct holding throttle, braking, and steering values consumed by model integration.
- `NoiseModel` in `eufs_remastered/eufs_models/include/eufs_models/noise.hpp`: applies Gaussian noise to vehicle state components before publication; standard deviations are loaded from `eufs_remastered/eufs_models/config/noise.yaml`.
- `eufs_remastered/eufs_models/include/eufs_models/eufs_models.hpp`: convenience header that includes all model types; `eufs_gz_dynamics` includes this single header to access the full library.

### Related Entry Points

- `eufs_remastered/eufs_models/config/noise.yaml`: noise standard deviation values for each state component.
- `eufs_remastered/eufs_models/CMakeLists.txt`: build configuration; exports the library for use by `eufs_gz_dynamics`.

## Gazebo Dynamics Plugin

### Primary Files

- `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`

### Function Map

- `EufsRaceCarModel` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: top-level Gazebo system plugin class implementing `ISystemConfigure` and `ISystemPreUpdate`; owns the ROS 2 node, executor thread, dynamics model, and command queue.
- `EufsRaceCarModel::Configure` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: called once at Gazebo startup; calls `InitRos`, `ParseSdf`, and `InitVehicleModel`, then resolves the canonical model link.
- `EufsRaceCarModel::PreUpdate` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: called every simulation step; dequeues the latest command, advances the dynamics model, and writes the resulting state to the Gazebo entity.
- `EufsRaceCarModel::InitRos` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: creates the `rclcpp::Node`, sets up the executor, starts the background spin thread, subscribes to `/cmd_vel` `geometry_msgs/Twist` input, and sets up diagnostic publishers.
- `EufsRaceCarModel::ParseSdf` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: reads the `<yaml_config>` SDF element and any other plugin parameters from the model SDF.
- `EufsRaceCarModel::InitVehicleModel` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: loads the vehicle parameter YAML and instantiates the selected `eufs::models` dynamics model.
- `EufsRaceCarModel` command callback in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: ROS 2 subscription callback that converts incoming `/cmd_vel` velocity and yaw-rate commands into the internal `Command` queue.
- `Command` struct in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: simple POD holding a `eufs::models::Input` and a timestamp; the `PreUpdate` step pops the most recent command from this queue.
- Diagnostic publisher in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: publishes `diagnostic_msgs/DiagnosticArray` with plugin health and timing data.

### Related Entry Points

- `eufs_remastered/eufs_models/include/eufs_models/eufs_models.hpp`: umbrella header included by the plugin to access all dynamics model types.
- Vehicle SDF model file in the simulation assets: contains the `<plugin>` block that loads `eufs_gz_dynamics` and the `<yaml_config>` element pointing to the parameter file.
- `eufs_remastered/eufs_gz_dynamics/CMakeLists.txt`: links the plugin against `eufs_models` and the Gazebo and ROS 2 libraries.

## EUFS Messages

### Primary Files

- `eufs_remastered/eufs_msgs/msg/ConeArrayWithCovariance.msg`
- `eufs_remastered/eufs_msgs/msg/ConeWithCovariance.msg`
- `eufs_remastered/eufs_msgs/msg/WheelSpeeds.msg`

### Message Definitions

- `eufs_remastered/eufs_msgs/msg/ConeArrayWithCovariance.msg`: top-level cone array message; contains a `std_msgs/Header` and five named arrays of `ConeWithCovariance`.
- `eufs_remastered/eufs_msgs/msg/ConeWithCovariance.msg`: single-cone message holding a `geometry_msgs/Point` for position and a flat float array for the 2D positional covariance upper triangle.
- `eufs_remastered/eufs_msgs/msg/WheelSpeeds.msg`: four-element wheel speed message with float fields `lf`, `rf`, `lb`, `rb` for left-front, right-front, left-back, and right-back wheel speeds.

### Related Entry Points

- `eufs_remastered/eufs_msgs/CMakeLists.txt`: registers the `.msg` files for code generation and exports the package as an ament dependency.
- `eufs_remastered/eufs_msgs/package.xml`: declares `rosidl_default_generators` build dependency for message generation.
- `eufs_remastered/gazebo_cone_plugins/src/gazebo_ground_truth_cones.cpp`: primary publisher of `ConeArrayWithCovariance`; depends on `eufs_msgs` at link time.

## Steering GUI

### Primary Files

- `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`

### Function Map

- `EUFSRobotSteeringGUI` in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: main RQT plugin class; sets up the Qt widget, creates publishers and subscriptions, and wires slider signals to command publishing.
- `EUFSRobotSteeringGUI.__init__` in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: initializes the plugin, loads the UI, and sets default parameter values for command topic and steering range.
- `EUFSRobotSteeringGUI._send_ackermann_drive_stamped` in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: builds an `AckermannDriveStamped` message from the current slider value and publishes it to the configured drive command topic.
- `EUFSRobotSteeringGUI._publish_brake_cmd` in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: publishes a `Float32` brake value to the separate brake command topic.
- Keyboard event handlers in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: intercept key presses to allow steering adjustment without using the slider directly.
- Topic name handling in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: allows the operator to change the drive command topic at runtime so the GUI can target different vehicle namespaces.

### Related Entry Points

- `eufs_remastered/steering_gui/package.xml` and `setup.py`: register the plugin with the RQT plugin system so it appears in the Plugins menu.
- `eufs_remastered/steering_gui/plugin.xml`: RQT plugin descriptor file mapping the GUI class to the plugin entry point.
- `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: subscribes to `/cmd_vel`; use `ackermann_cmd_bridge` when the GUI or planners publish `AckermannDriveStamped` on `/cmd`.
