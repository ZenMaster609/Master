# Stanley Controller Code Map

This page maps the `documentation/stanley_controller.md` behavior to the Stanley controller implementation.

## Primary Files

- `sim_car/sim_car/controllers/stanley_controller.py`
- `sim_car/sim_car/planning/controller_config.py`

## Function Map

### Inputs And Configuration

- `StanleyConfig` in `sim_car/sim_car/controllers/stanley_controller.py`: parameter object for gains, damping, deadband, and steering limits.
- `build_stanley_config` in `sim_car/sim_car/planning/controller_config.py`: reads ROS parameters into `StanleyConfig`.

### Nearest Path Projection

- `StanleyController.compute` in `sim_car/sim_car/controllers/stanley_controller.py`: full control-step implementation, including nearest path projection, heading/cross-track terms, damping, and filtering.

### Heading Term

- `StanleyController.compute` in `sim_car/sim_car/controllers/stanley_controller.py`: computes path heading, heading error, and the heading contribution.
- `_normalize_angle` in `sim_car/sim_car/controllers/stanley_controller.py`: wraps heading error into `[-pi, pi]`.

### Cross-Track Term

- `StanleyController.compute` in `sim_car/sim_car/controllers/stanley_controller.py`: computes cross-track error, applies deadband, and evaluates the Stanley `atan2` cross-track term.

### Yaw-Rate Damping

- `StanleyController.compute` in `sim_car/sim_car/controllers/stanley_controller.py`: subtracts the optional yaw-rate damping contribution from the steering command.

### Steering Command

- `StanleyController.compute` in `sim_car/sim_car/controllers/stanley_controller.py`: clamps, low-pass filters, and rate-limits the final steering output before returning it to the planner node.

## Related Entry Points

- `build_steering_controller` in `sim_car/sim_car/planning/controller_config.py`: shared controller factory that instantiates `StanleyController` when selected.
- `normalize_tracked_cone_controller_type` in `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: normalizes the planner-side controller selection string before construction.
