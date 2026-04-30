# EUFS Gazebo Dynamics Code Map

This page maps the `documentation/eufs_gz_dynamics.md` behavior to the Gazebo system plugin implementation.

## Primary Files

- `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`

## Function Map

### Plugin Lifecycle

- `EufsRaceCarModel` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: top-level Gazebo system plugin class implementing `ISystemConfigure` and `ISystemPreUpdate`; owns the ROS 2 node, executor thread, dynamics model, and command queue.
- `EufsRaceCarModel::Configure` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: called once at Gazebo startup; calls `InitRos`, `ParseSdf`, and `InitVehicleModel`, then resolves the canonical model link.
- `EufsRaceCarModel::PreUpdate` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: called every simulation step; dequeues the latest command, advances the dynamics model, and writes the resulting state to the Gazebo entity.

### ROS 2 Integration

- `EufsRaceCarModel::InitRos` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: creates the `rclcpp::Node`, sets up the executor, starts the background spin thread, subscribes to control command topics, and sets up state and diagnostic publishers.
- `EufsRaceCarModel::ParseSdf` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: reads the `<yaml_config>` SDF element and any other plugin parameters from the model SDF.
- `EufsRaceCarModel::InitVehicleModel` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: loads the vehicle parameter YAML and instantiates the selected `eufs::models` dynamics model.

### Command Handling

- `EufsRaceCarModel` command callback in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: ROS 2 subscription callback that pushes incoming control commands into the internal `Command` queue, stamped with the receive time.
- `Command` struct in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: simple POD holding a `eufs::models::Input` and a timestamp; the `PreUpdate` step pops the most recent command from this queue.

### State And Diagnostics

- State publisher setup in `EufsRaceCarModel::InitRos` in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: creates the publisher used to broadcast the post-integration vehicle state each simulation step.
- Diagnostic publisher in `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: publishes `diagnostic_msgs/DiagnosticArray` with plugin health and timing data.

## Related Entry Points

- `eufs_remastered/eufs_models/include/eufs_models/eufs_models.hpp`: umbrella header included by the plugin to access all dynamics model types.
- Vehicle SDF model file (in the simulation assets): contains the `<plugin>` block that loads `eufs_gz_dynamics` and the `<yaml_config>` element pointing to the parameter file.
- `eufs_remastered/eufs_gz_dynamics/CMakeLists.txt`: links the plugin against `eufs_models` and the Gazebo and ROS 2 libraries.
