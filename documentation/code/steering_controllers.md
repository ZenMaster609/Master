# Steering Controllers Code Map

This page maps `documentation/concepts/steering_controllers.md` to the controller implementation.

## Primary Files

- `sim_car/sim_car/controllers/base.py`
- `sim_car/sim_car/controllers/_path_utils.py`
- `sim_car/sim_car/controllers/factory.py`
- `sim_car/sim_car/controllers/stanley_controller.py`
- `sim_car/sim_car/controllers/pure_pursuit_controller.py`
- `sim_car/sim_car/planning/tracked_cone_planner_contract.py`
- `sim_car/sim_car/planning/tracked_cone_planner_base.py`
- `sim_car/sim_car/planning/linetest_planner_node.py`

## Shared Controller Interface

- `ControllerOutput` in `controllers/base.py`: common return object containing steering angle, curvature, lookahead distance, target point, and optional Stanley debug data.
- `SteeringController` in `controllers/base.py`: protocol implemented by both controller classes.
- `validate_control_path` in `controllers/_path_utils.py`: shared path validation before controller computation.
- `nearest_projection_on_path` in `controllers/_path_utils.py`: shared nearest-segment projection helper.
- `target_point_from_projection` in `controllers/_path_utils.py`: pure-pursuit target lookup along the path from a projection.

## Controller Selection

- `create_steering_controller` in `controllers/factory.py`: instantiates `StanleyController` or `PurePursuitController` from the normalized `control.controller_type`.
- `build_stanley_config` in `planning/tracked_cone_planner_contract.py`: reads `stanley.*` ROS parameters into `StanleyConfig`.
- `build_pure_pursuit_config` in `planning/tracked_cone_planner_contract.py`: reads `pure_pursuit.*` ROS parameters into `PurePursuitConfig`.
- `build_steering_controller` in `planning/tracked_cone_planner_contract.py`: creates both config objects and passes them to `create_steering_controller`.
- `normalize_tracked_cone_controller_type` in `planning/tracked_cone_planner_contract.py`: accepts `stanley`, `pure_pursuit`, or `none` for tracked-cone planners.

## Planner Integration

- `TrackedConePlannerBase._build_steering_controller` in `tracked_cone_planner_base.py`: creates the selected steering controller for midpoint, single-boundary, and corridor planners.
- `TrackedConePlannerBase._centerline_to_vehicle_frame`: converts a centerline from planning frame into the controller frame.
- `GenericTrackedConePlannerNode._dispatch_controller` in `tracked_cone_planner_node.py`: calls the active controller for tracked-cone planners after the planner-specific `_run_controller*` method extracts the forward control path.
- `TrackedConePlannerBase._publish_cmd`: publishes the resulting Ackermann command.
- `LineTestPlannerNode._build_steering_controller` in `linetest_planner_node.py`: reuses the same controller factory for fixed-line controller tests.
- `LineTestPlannerNode._on_timer`: runs line projection, controller invocation, command publication, diagnostics, and brake behavior.

## Stanley Implementation

- `StanleyConfig` in `stanley_controller.py`: configuration dataclass for lateral gain, softening, heading gain, heading segment offset, steering limits, filtering, yaw-rate damping, wheelbase, and cross-track deadband.
- `StanleyController.compute`: validates the path, projects the vehicle origin onto the nearest path segment, computes heading error, cross-track error, yaw-rate damping, steering clamp/filter/rate limit, curvature, lookahead distance, and debug metrics.
- `_normalize_angle` in `stanley_controller.py`: wraps heading error into `[-pi, pi]`.
- `StanleyDebugInfo` in `controllers/base.py`: stores diagnostic fields such as heading contribution, cross-track contribution, yaw-rate damping contribution, raw steering, clamped steering, filtered steering, final steering, saturation flag, nearest path index, heading path index, and target point.

## Pure Pursuit Implementation

- `PurePursuitConfig` in `pure_pursuit_controller.py`: configuration dataclass for lookahead bounds, speed-based lookahead gain, steering limits, filtering, rate limiting, and wheelbase.
- `PurePursuitController.compute`: validates the path, projects the vehicle origin onto the path, computes the commanded lookahead, selects a target point ahead of the projection, converts target geometry into curvature, clamps/filters/rate-limits steering, and returns `ControllerOutput`.
- `PurePursuitController._compute_commanded_lookahead`: applies `lookahead_m + lookahead_gain * speed` and clamps between `min_lookahead_m` and `max_lookahead_m`.

## Where To Edit

- Add a shared controller output/debug field: `controllers/base.py`.
- Change path projection or target sampling: `controllers/_path_utils.py`.
- Change controller selection behavior: `controllers/factory.py`.
- Change Stanley math or defaults: `controllers/stanley_controller.py`.
- Change pure-pursuit math or defaults: `controllers/pure_pursuit_controller.py`.
- Change ROS parameter loading for either controller: `planning/tracked_cone_planner_contract.py`.
- Change how tracked-cone planners call controllers: `planning/tracked_cone_planner_base.py`.
- Change fixed-line controller test behavior: `planning/linetest_planner_node.py`.
