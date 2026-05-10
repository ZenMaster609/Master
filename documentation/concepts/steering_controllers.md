# Steering Controllers

The planners can run either Stanley or pure pursuit after they have produced a forward path in the vehicle frame. Both controllers receive the same shape of input and return the same output type, so the planner can switch between them with `controller:=stanley`, `controller:=pure_pursuit`, or disable steering with `controller:=none`.

## Shared Input

Each control step receives:

- `control_path`: an `N x 2` path in the vehicle or base frame.
- `speed_mps`: current vehicle speed.
- `yaw_rate_rps`: current yaw rate.

The vehicle is treated as being at the origin of the control frame and facing positive X. Positive Y is lateral left. The planner extracts this path from the published centerline, transforms it into the vehicle frame, and then calls the selected controller.

Both controllers validate that the path has usable points, compute a target or projection on the path, produce a steering angle in radians, estimate curvature as `tan(steering) / wheelbase`, and return a target point for diagnostics and RViz.

## Stanley

Stanley combines path heading error with cross-track error. It is usually more explicit about lateral correction than pure pursuit.

The controller first projects the vehicle origin onto the nearest path segment. That projection gives:

- the nearest target point;
- the nearest segment index;
- the cross-track error from the target point's lateral position.

The heading term uses a path segment heading. `stanley.lookahead_idx_offset` can shift the segment used for heading so the controller points along a later part of the path. Because the path is already in the vehicle frame, the vehicle heading is zero.

The cross-track term is:

`atan2(k_gain * cross_track_error, softening_speed_mps + max(speed_mps, 0))`

Useful Stanley parameters:

- `stanley.k_gain`: lateral-error correction strength.
- `stanley.heading_gain`: heading-error correction strength.
- `stanley.lookahead_idx_offset`: segment offset for heading calculation.
- `stanley.softening_speed_mps`: reduces low-speed cross-track spikes.
- `stanley.cross_track_deadband_m`: suppresses very small lateral errors.
- `stanley.use_yaw_rate_damping` and `stanley.yaw_rate_damping_gain`: resist rapid yaw motion.

Stanley tends to be useful when you want direct control over how strongly the car corrects lateral offset and heading error. Too much gain can create oscillation; too little can make the car lag or cut corners.

## Pure Pursuit

Pure pursuit chooses a target point ahead of the vehicle and steers toward it. It is usually easier to reason about than Stanley, but large lookahead values can cut tight corners.

The controller first projects the vehicle origin onto the path. From that projection it walks forward along the path until it reaches the commanded lookahead distance. If the path ends first, the final point is used.

The commanded lookahead is:

`pure_pursuit.lookahead_m + pure_pursuit.lookahead_gain * max(speed_mps, 0)`

Then it is clamped between:

- `pure_pursuit.min_lookahead_m`
- `pure_pursuit.max_lookahead_m`

The selected target point is converted into curvature:

`curvature = 2 * target_y / (target_x^2 + target_y^2)`

Then steering is:

`atan(wheelbase_m * curvature)`

Useful pure-pursuit parameters:

- `pure_pursuit.lookahead_m`: base target distance.
- `pure_pursuit.min_lookahead_m`: lower bound for slow or tight motion.
- `pure_pursuit.max_lookahead_m`: upper bound for high speed.
- `pure_pursuit.lookahead_gain`: speed-based lookahead growth.

Shorter lookahead tracks more tightly but can oscillate. Longer lookahead is smoother but reacts later and can cut inside curves.

## Shared Steering Limits

Both controllers apply the same final shaping pattern:

1. Clamp steering to the configured steering limit.
2. Apply first-order low-pass filtering.
3. Apply steering rate limiting.

Stanley parameters:

- `stanley.steering_limit_rad`
- `stanley.steering_lowpass_alpha`
- `stanley.steering_rate_limit_rad_s`

Pure-pursuit parameters:

- `pure_pursuit.steering_limit_rad`
- `pure_pursuit.steering_lowpass_alpha`
- `pure_pursuit.steering_rate_limit_rad_s`

For both controllers, `steering_lowpass_alpha` is clipped to `[0, 1]`. A value of `1.0` means no smoothing; lower values retain more of the previous command. Rate limiting is applied per control step using the planner publish rate.

## Tuning Guide

Use Stanley when:

- lateral offset correction needs to be explicit;
- heading and cross-track contributions need separate tuning;
- yaw-rate damping is useful for oscillation control.

Use pure pursuit when:

- a target-point controller is easier to tune;
- the path is smooth and lookahead behavior is enough;
- you want fewer interacting terms.

Common adjustments:

- Car cuts inside corners with Stanley: increase `stanley.heading_gain` or `stanley.k_gain`.
- Car oscillates with Stanley: reduce `stanley.k_gain`, add `stanley.softening_speed_mps`, lower rate limit, or add yaw-rate damping.
- Stanley reacts too early to noisy path segments: increase `stanley.lookahead_idx_offset` or add filtering.
- Car oscillates with pure pursuit: increase lookahead, lower steering rate limit, or add filtering.
- Pure pursuit cuts corners: reduce lookahead or reduce speed in curves.
- Pure pursuit reacts too slowly: reduce lookahead or increase speed-based lookahead only when speed scaling is needed.

## Launch Examples

Run Stanley on smalltrack:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=midpoint controller:=stanley
```

Run pure pursuit on smalltrack:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=midpoint controller:=pure_pursuit
```

Run without steering output:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=corridor controller:=none
```
