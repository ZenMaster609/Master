# Pure Pursuit Controller

The pure pursuit controller turns a planned path in the vehicle frame into a steering command by choosing a target point ahead of the car and steering toward it.

It is used by planner nodes when `controller:=pure_pursuit`.

## Inputs

Each control step receives:

- `control_path`: an `N x 2` path in the vehicle/base frame
- `speed_mps`: current vehicle speed
- `yaw_rate_rps`: ignored by pure pursuit

The vehicle is at the origin of the control frame and points along positive X.

## Nearest Projection

The controller first finds the nearest projection of the vehicle origin onto the path. This gives a stable starting point for measuring distance along the path.

If the path has only one point, that point becomes the target.

## Lookahead Distance

The commanded lookahead is:

`lookahead_m + lookahead_gain * max(speed_mps, 0)`

Then it is clamped between:

- `pure_pursuit.min_lookahead_m`
- `pure_pursuit.max_lookahead_m`

Tuning effect:

- Shorter lookahead tracks tighter but can oscillate.
- Longer lookahead is smoother but cuts corners and responds later.
- `lookahead_gain` makes the car look farther ahead at higher speed.

## Target Point Selection

Starting from the nearest projection, the controller walks forward along the path until it has traveled the lookahead distance. That point becomes the pursuit target.

If the path ends before the lookahead distance is reached, the final path point becomes the target.

## Curvature And Steering

The controller computes curvature from the target point:

`curvature = 2 * target_y / (target_x^2 + target_y^2)`

Then it converts curvature to steering:

`steering = atan(wheelbase_m * curvature)`

The result is clamped to `pure_pursuit.steering_limit_rad`.

## Filtering And Rate Limiting

After clamping, the controller applies:

1. low-pass filtering with `pure_pursuit.steering_lowpass_alpha`
2. rate limiting with `pure_pursuit.steering_rate_limit_rad_s`

`steering_lowpass_alpha` is clipped to `[0, 1]`:

- `1.0`: no smoothing
- lower values: more smoothing from the previous steering command

The rate limit is applied per control step using the planner publish rate.

## Output

The controller returns:

- steering angle in radians
- curvature estimate from `tan(steering) / wheelbase`
- actual distance to the selected target point
- target point in the vehicle frame

The planner uses the curvature to choose speed through `speed_control.*`.

## Tuning Guide

Common adjustments:

- Car oscillates: increase lookahead, lower steering rate limit, or add filtering.
- Car cuts corners: reduce lookahead or reduce speed in curves.
- Car reacts too slowly: reduce lookahead or increase `lookahead_gain` only if speed scaling is needed.
- Low-speed steering is too sharp: increase `min_lookahead_m`.
- High-speed steering is too nervous: increase `lookahead_gain` or `max_lookahead_m`.

Pure pursuit is usually easier to reason about than Stanley, but it can cut tight corners when lookahead is large.

## Useful Commands

Run pure pursuit on smalltrack:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=midpoint controller:=pure_pursuit
```

Run pure pursuit on skidpad:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=skidpad planner:=single_boundary controller:=pure_pursuit
```
