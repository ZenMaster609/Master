# Pure Pursuit Controller Code Map

This page maps the `documentation/pure_pursuit_controller.md` behavior to the pure-pursuit controller implementation.

## Primary Files

- `sim_car/sim_car/controllers/pure_pursuit_controller.py`
- `sim_car/sim_car/planning/controller_config.py`

## Function Map

### Inputs And Configuration

- `PurePursuitConfig` in `sim_car/sim_car/controllers/pure_pursuit_controller.py`: parameter object for lookahead, steering limits, and output filtering.
- `build_pure_pursuit_config` in `sim_car/sim_car/planning/controller_config.py`: reads ROS parameters into `PurePursuitConfig`.

### Nearest Projection

- `PurePursuitController.compute` in `sim_car/sim_car/controllers/pure_pursuit_controller.py`: full control-step implementation, including nearest-path projection, target-point selection, curvature, and filtered steering output.

### Lookahead Distance

- `PurePursuitController._compute_commanded_lookahead` in `sim_car/sim_car/controllers/pure_pursuit_controller.py`: computes and clamps the active lookahead distance from speed.

### Target Point Selection

- `PurePursuitController.compute` in `sim_car/sim_car/controllers/pure_pursuit_controller.py`: walks forward from the nearest projection until the commanded lookahead distance is reached.

### Curvature And Steering

- `PurePursuitController.compute` in `sim_car/sim_car/controllers/pure_pursuit_controller.py`: converts the selected target point into curvature and then steering angle.

### Filtering And Rate Limiting

- `PurePursuitController.compute` in `sim_car/sim_car/controllers/pure_pursuit_controller.py`: clamps steering, applies low-pass filtering, and rate-limits the final command before returning it.

## Related Entry Points

- `build_steering_controller` in `sim_car/sim_car/planning/controller_config.py`: shared controller factory that instantiates `PurePursuitController` when selected.
- `normalize_tracked_cone_controller_type` in `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: normalizes the planner-side controller selection string before construction.
