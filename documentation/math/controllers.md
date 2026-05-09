# Controllers Math

## Scope

This page documents the steering-controller math in `sim_car/sim_car/controllers`. It covers the path geometry shared by Stanley and pure pursuit, then the controller-specific steering laws and post-processing applied before publishing an Ackermann command.

The controller input path is expressed in vehicle frame: the vehicle reference point is at `(0, 0)`, forward is `+x`, and left is `+y`.

## Pipeline Map

1. A planner publishes or internally produces a vehicle-frame control path.
2. `sim_car/sim_car/controllers/_path_utils.py::validate_control_path` checks that the path has shape `(N, 2)`.
3. `nearest_projection_on_path` finds the closest point on the path to the vehicle reference point.
4. Stanley uses the nearest projection and a heading segment to compute heading and cross-track steering.
5. Pure pursuit walks forward by a commanded lookahead distance and computes target curvature from the selected point.
6. Both controllers clamp steering, optionally low-pass filter it, optionally rate-limit it, and return `ControllerOutput`.
7. `sim_car/sim_car/controllers/ackermann_cmd_bridge.py` and the tracked-cone planner nodes publish the final steering command into the actuation pipeline.

## Mathematical Building Blocks

### Closest Projection Onto A Polyline

`sim_car/sim_car/controllers/_path_utils.py::nearest_projection_on_path` projects the vehicle origin onto every path segment. For segment endpoints `p0` and `p1`, it computes the clamped scalar projection

```text
t = clip(-dot(p0, p1 - p0) / ||p1 - p0||^2, 0, 1)
projected = p0 + t * (p1 - p0)
```

The selected projection minimizes squared distance to the origin. This gives both controllers a stable nearest path reference even when the path is discretized.

### Arc-Length Target Selection

`sim_car/sim_car/controllers/_path_utils.py::target_point_from_projection` starts at the nearest projection and walks forward along segment lengths until the requested lookahead distance is consumed. If the path ends first, it returns the final path point.

This keeps lookahead distance path-relative, not index-relative. That matters because planner output can be resampled at different resolutions.

### Stanley Heading And Cross-Track Terms

`sim_car/sim_car/controllers/stanley_controller.py::StanleyController.compute` selects a heading segment near the projection and computes the path heading with `atan2`. Because the path is already in vehicle frame, the vehicle heading is zero, so the heading error is the normalized path heading.

The lateral error is the `y` coordinate of the nearest projection. A configurable deadband can zero small cross-track errors. The Stanley cross-track term is:

```text
atan2(k_gain * cross_track_error, softening_speed_mps + max(0, speed_mps))
```

The softening speed prevents excessive steering at low speed. Optional yaw-rate damping subtracts `yaw_rate_damping_gain * yaw_rate_rps`, reducing steering when the vehicle is already rotating in the commanded direction.

### Pure-Pursuit Curvature

`sim_car/sim_car/controllers/pure_pursuit_controller.py::PurePursuitController.compute` chooses a commanded lookahead:

```text
lookahead = clip(base_lookahead + lookahead_gain * max(0, speed), min_lookahead, max_lookahead)
```

Given target point `(x, y)` in vehicle frame, it computes:

```text
raw_curvature = 2 * y / (x^2 + y^2)
steering = atan(wheelbase_m * raw_curvature)
```

This converts a geometric target point into a bicycle-model steering command. The controller returns both steering and the final curvature `tan(steering) / wheelbase_m`.

### Steering Saturation, Filtering, And Rate Limiting

Both controllers clamp raw steering to `[-steering_limit_rad, steering_limit_rad]`. If a previous command exists, they optionally apply:

```text
filtered = alpha * clamped + (1 - alpha) * previous
max_step = steering_rate_limit_rad_s / publish_rate_hz
limited = clip(filtered, previous - max_step, previous + max_step)
```

This makes the command stream less sensitive to one-frame path jumps and avoids unrealistically fast steering changes.

## Function Reference

| Math operation | Function | Runtime use |
| --- | --- | --- |
| Path shape validation | `sim_car/sim_car/controllers/_path_utils.py::validate_control_path` | Rejects malformed controller input before geometry is computed. |
| Closest path point | `sim_car/sim_car/controllers/_path_utils.py::nearest_projection_on_path` | Shared nearest-path reference for Stanley and pure pursuit. |
| Lookahead target | `sim_car/sim_car/controllers/_path_utils.py::target_point_from_projection` | Pure-pursuit target point at path-relative distance. |
| Stanley steering | `sim_car/sim_car/controllers/stanley_controller.py::StanleyController.compute` | Heading, cross-track, yaw-rate damping, clamp, filter, rate limit. |
| Angle wrapping | `sim_car/sim_car/controllers/stanley_controller.py::_normalize_angle` | Keeps heading errors in `[-pi, pi]`. |
| Pure-pursuit steering | `sim_car/sim_car/controllers/pure_pursuit_controller.py::PurePursuitController.compute` | Lookahead curvature and steering conversion. |
| Adaptive lookahead | `sim_car/sim_car/controllers/pure_pursuit_controller.py::PurePursuitController._compute_commanded_lookahead` | Scales lookahead with speed inside configured bounds. |

## Notes / Limits

- Both controllers assume the input path is already in the vehicle frame. Frame conversion belongs to the planner/runtime layer.
- The Stanley implementation uses the projected point's lateral coordinate as cross-track error, which is appropriate for the vehicle-frame path representation.
- The pure-pursuit curvature is undefined for a zero-distance target; the code explicitly returns zero curvature for that degenerate case.
- Filtering and rate limiting are command post-processing steps. They do not change the planner path or controller geometry.
