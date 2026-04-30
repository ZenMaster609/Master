# Skidpad Routing

The skidpad router is a deterministic mission filter for skidpad and acceleration runs. It republishes cones for the active route branch so normal planners can drive the correct part of the course without understanding skidpad mission state.

It runs as `skidpad_router_node` when the selected track is `skidpad` or `acceleration` and the selected planner is `midpoint`, `single_boundary`, or `corridor`.

## Core Idea

The router sits between cone memory and the planner:

`/tracked_cones -> skidpad_router_node -> /tracked_cones/skidpad_routed -> planner`

If cone memory is disabled, the router uses the direct camera cone topic as input. The planner still consumes the routed output on skidpad and acceleration. `linetest` does not use the router because it does not consume cones.

## State Machine

The skidpad route state machine tracks:

- approach
- right circle passes
- left circle passes
- straight exit/parking
- parked

The default route sequence is:

`right, right, left, left, straight`

The route can be repeated with `routing.route_laps`, but the current default is one pass through that sequence.

## Geometry Regions

The router uses simple fixed regions:

- crossroads rectangle around the center
- right lobe around the right circle
- left lobe around the left circle
- straight corridor for approach/exit
- parking corridor after the straight exit

Each incoming cone is converted into the odom frame, then tested against the active route mask.

During approach, the mask includes the approach corridor, crossroads, and the first active branch. During circle branches, it keeps crossroads plus only the active lobe. During straight, it keeps crossroads plus the straight corridor.

## Lap Arming

For circular branches, the router tracks angular progress around the active circle center. Samples count only when the vehicle is outside the crossroads and inside the configured circle radius window.

As the vehicle moves around the circle, the router accumulates wrapped angle changes. Once the accumulated angle magnitude exceeds `routing.lap_complete_angle_rad`, the lap is armed.

The branch advances when the vehicle next enters the crossroads while armed. This prevents a partial circle from being counted just because the car crossed the center area.

## Cone Filtering

In normal skidpad routing, the router:

1. Converts cones to odom geometry.
2. Builds a route mask from the current state.
3. Keeps cones inside the active route region.
4. Publishes the filtered cones on `/tracked_cones/skidpad_routed`.

For non-parking skidpad branches, cone colors and track metadata are preserved.

## Parking Mode

Parking mode is active:

- on skidpad when the active branch is `straight`
- on acceleration after the finish/parking latch triggers

In parking mode, the router focuses on orange cones. It removes detected stop-line cones from the routed set and assigns planner-facing `boundary_color` to the remaining orange cones based on their lateral side relative to the vehicle.

This lets normal boundary planners treat parking-lane orange cones as temporary blue/yellow boundaries.

## Stop-Line Detection

For skidpad parking, the router looks for a close pair of orange cones ahead of the vehicle and treats it as the stop line.

For acceleration, the router detects a stop row from the farthest cluster of orange cones ahead of the vehicle. It checks:

- minimum cluster count
- cluster depth
- lateral span
- minimum points per side

If the row-style detector fails, acceleration mode can infer a stop row from the farthest orange frontier and the median left/right lateral positions.

Once a stop line is found, it is latched in odom. The router continues using that line for stop override even if the original cones leave the routed set.

## Stop Override

When the latched stop line is close enough, the router can override `/cmd` directly:

- approach speed is limited by distance to the stop target
- steering override is zero
- brake command is published near the target

Relevant parameters:

- `parking.stop_margin_m`
- `parking.target_margin_m`
- `parking.stop_override_publish_rate_hz`
- `parking.stop_approach_speed_gain`
- `parking.brake_activation_margin_m`
- `parking.brake_command`

The override keeps publishing at a fixed rate while active so the vehicle remains stopped even if planner commands continue.

## Diagnostics And Markers

The router publishes diagnostics with current stage, branch, lap state, route index, parking status, and stop-line state.

It also publishes markers for:

- active stage text
- circles
- crossroads
- stop line
- stop target
- acceleration parking status

For acceleration, non-relevant skidpad circle/crossroads markers are deleted and parking markers are used instead.

## Useful Commands

Run skidpad with routing:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=skidpad planner:=midpoint controller:=stanley
```

Run acceleration with tracked-cone planner and router-managed stop behavior:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=acceleration planner:=midpoint controller:=stanley
```
