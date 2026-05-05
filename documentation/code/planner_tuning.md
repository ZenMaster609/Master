# Planner Tuning Code Map

This page maps the `documentation/planner_tuning.md` behavior to the launch/config code that selects planners, controllers, and parameter overlays.

## Primary Files

- `sim_car/launch/full_sim_launch.launch.py`
- `sim_car/sim_car/planning/tracked_cone_planner_contract.py`
- `sim_car/sim_car/planning/controller_config.py`
- `sim_car/config/`

## Function Map

### Launch-Time Selection

- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: main launch entry point that wires planner, controller, track, LiDAR, and logging selections.
- `LaunchSelection` in `sim_car/launch/full_sim_launch.launch.py`: carries the resolved planner/controller/track bundle.
- `_resolve_launch_selection` in `sim_car/launch/full_sim_launch.launch.py`: resolves the selected planner, controller, track overlays, and bundle paths.
- `_validate_planner_and_controller_args` in `sim_car/launch/full_sim_launch.launch.py`: rejects invalid planner/controller combinations before node startup.

### Track And Overlay Resolution

- `_resolve_track_bundle` in `sim_car/launch/full_sim_launch.launch.py`: maps track, planner, and controller names to YAML overlay files.
- `_load_spawn_defaults`, `_load_speed_control_defaults`, and `_load_planner_limit_overrides` in `sim_car/launch/full_sim_launch.launch.py`: pull track-level defaults from `spawn.yaml`.
- `_write_parameter_overlay` in `sim_car/launch/full_sim_launch.launch.py`: materializes merged parameter overlays for launched nodes.

### Planner/Controller Shared Defaults

- `read_migrated_tracked_cone_planner_common_config` in `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: reads the shared tracked-cone planner parameters.
- `apply_common_config_to_node` in `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: applies those shared values to each planner node instance.
- `MigratedTrackedConePlannerCommonConfig` in `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: defines the common tracked-cone planner parameter surface.
- `normalize_tracked_cone_controller_type` and `build_tracked_cone_controller` in `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: normalize controller choice and create the active steering controller.

### Controller Config Builders

- `build_stanley_config` in `sim_car/sim_car/planning/controller_config.py`: reads the Stanley-specific parameter group into a config object.
- `build_pure_pursuit_config` in `sim_car/sim_car/planning/controller_config.py`: reads the pure-pursuit parameter group into a config object.
- `build_steering_controller` in `sim_car/sim_car/planning/controller_config.py`: constructs the selected controller from the loaded config.

### LiDAR And Measurement Wiring

- `_lidar_enabled_condition` in `sim_car/launch/full_sim_launch.launch.py`: enables the scan LiDAR node when LiDAR or cone memory is enabled.
- `_configure_measurement_config` and `_write_rate_adjusted_measurement_config` in `sim_car/launch/full_sim_launch.launch.py`: prepare the measurement config used when planner rate affects sensor timing.

## Related Entry Points

- `sim_car/config/<track>/stanley_controller.yaml`: Stanley track overlays.
- `sim_car/config/<track>/pure_pursuit_controller.yaml`: pure-pursuit track overlays.
- `sim_car/config/<track>/spawn.yaml`: track-level speed and planner-limit overlays.
- `sim_car/config/skidpad/skidpad_router.yaml`: routing and parking behavior for skidpad and acceleration.
