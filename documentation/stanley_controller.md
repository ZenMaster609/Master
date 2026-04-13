# Stanley Controller

The Stanley controller turns a planned path in the vehicle frame into a steering command. It combines path heading error with cross-track error, then applies optional yaw-rate damping and steering filtering.

It is used by all planner nodes when `controller:=stanley`.

## Inputs

Each control step receives:

- `control_path`: an `N x 2` path in the vehicle/base frame
- `speed_mps`: current vehicle speed
- `yaw_rate_rps`: current yaw rate

The vehicle is assumed to be at the origin of the control frame and facing along positive X. Positive Y is lateral left.

## Nearest Path Projection

The controller first projects the vehicle origin onto the nearest path segment. This gives:

- target point on the path
- nearest segment index
- cross-track error from the target point's lateral `y`

The nearest segment is also used as the starting point for heading selection.

## Heading Term

The path heading is computed from a path segment. Normally this is the nearest segment, but `stanley.lookahead_idx_offset` can shift the heading calculation to a later segment.

The heading error is:

`heading_error = path_heading - vehicle_heading`

Because the path is already in the vehicle frame, vehicle heading is zero. The value is normalized to `[-pi, pi]`.

The heading contribution is:

`heading_gain * heading_error`

Tuning effect:

- Higher `heading_gain` points the car more aggressively along the path tangent.
- Too high can cause twitchy steering or oscillation.
- Too low can make the car cut corners or lag path heading.

## Cross-Track Term

The cross-track error is the lateral offset from the car to the nearest point on the path.

Small errors can be suppressed with:

`stanley.cross_track_deadband_m`

The cross-track contribution is:

`atan2(k_gain * cross_track_error, softening_speed_mps + max(speed_mps, 0))`

Tuning effect:

- Higher `k_gain` corrects lateral offset harder.
- Higher `softening_speed_mps` reduces low-speed steering spikes.
- A small deadband can reduce steering chatter around the centerline.

## Yaw-Rate Damping

When enabled, yaw-rate damping subtracts:

`yaw_rate_damping_gain * yaw_rate_rps`

This resists rapid yaw motion. It can help damp oscillations, but too much damping makes the controller reluctant to rotate into a turn.

Relevant parameters:

- `stanley.use_yaw_rate_damping`
- `stanley.yaw_rate_damping_gain`

## Steering Command

The raw steering command is:

`heading contribution + cross-track contribution + yaw-rate damping contribution`

Then the controller applies:

1. clamp to `stanley.steering_limit_rad`
2. first-order low-pass filtering with `stanley.steering_lowpass_alpha`
3. rate limit with `stanley.steering_rate_limit_rad_s`

`steering_lowpass_alpha` is clipped to `[0, 1]`:

- `1.0`: no smoothing
- lower values: more smoothing from the previous command

Rate limit is applied per control step using the planner publish rate.

## Output

The controller returns:

- steering angle in radians
- curvature estimate from `tan(steering) / wheelbase`
- lookahead distance to the selected target point
- target point in the vehicle frame
- Stanley debug fields

The planner uses the curvature to choose speed through `speed_control.*`.

## Tuning Guide

Common adjustments:

- Car cuts inside corners: increase `heading_gain` or `k_gain`.
- Car oscillates around the path: reduce `k_gain`, add softening, lower steering rate limit, or add yaw-rate damping.
- Steering reacts too late: reduce `lookahead_idx_offset` or increase `heading_gain`.
- Steering reacts too early to noisy path segments: increase `lookahead_idx_offset` or add low-pass/rate limiting.
- Low-speed steering is too aggressive: increase `softening_speed_mps`.

Track overlays currently tune Stanley per track:

- smalltrack: moderate heading gain with one-segment heading lookahead
- skidpad: higher heading gain and gain for tight geometry
- acceleration: heading gain near zero with segment lookahead for straight-line control

## Useful Commands

Run Stanley on smalltrack:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=midpoint controller:=stanley
```

Run Stanley diagnostics:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py controller:=stanley controller_diagnostics:=true logging:=true
```
