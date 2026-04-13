# Line Test Planner

The line test planner publishes a fixed straight centerline and optionally runs the selected steering controller against it. It is mainly used on the acceleration track and for controller sanity checks.

Unlike the tracked-cone planners, it does not consume cone detections.

## Core Idea

The planner creates a static line between configured start and end points:

`line.start -> line.end -> fixed centerline -> controller -> /cmd`

Default acceleration config:

- start: `(-38.5, 0.0)`
- end: `(46.5, 0.0)`
- point spacing: `0.5 m`

The full centerline is published in the odom frame.

## Control Path

Each cycle, the node:

1. Reads the latest odometry.
2. Projects the vehicle onto the fixed line.
3. Builds the remaining forward segment of the line.
4. Converts that segment into the vehicle frame.
5. Runs Stanley or pure pursuit unless `controller:=none`.
6. Publishes Ackermann command and planner diagnostics.

Once the car passes the end of the line, there is no forward control path. If `control.stop_if_no_path` is true, the planner sends a zero command.

## Parking And Brake Behavior

The line test planner has simple end-of-line brake behavior:

- `parking.brake_activation_distance_m`: distance from line end where braking activates.
- `parking.brake_command`: brake command value.

When the remaining line distance is inside the brake activation distance, the planner commands zero speed and publishes brake command.

Acceleration parking/final-stop behavior can also be handled by the skidpad router when acceleration runs use tracked-cone planners. The line test planner's own brake behavior is specific to its fixed-line mode.

## Controller Testing Use

Because the path is fixed and simple, line test runs are useful for separating controller behavior from perception and planning behavior.

Good uses:

- verify steering sign convention
- check Ackermann command bridge behavior
- tune Stanley or pure-pursuit response on a straight path
- test speed command and braking behavior
- debug odometry delay or lag compensation without cone noise

Bad uses:

- evaluating cone perception
- testing boundary pairing
- validating skidpad routing

## Tuning Parameters

Important groups:

- `line.*`: start/end geometry and spacing.
- `runtime.publish_rate_hz`: planner/controller loop rate.
- `control.controller_type`: `stanley`, `pure_pursuit`, or `none`.
- `control.odom_lag_compensation_ms`: pose projection for control.
- `speed_control.*`: speed command limits.
- `parking.*`: end-of-line braking.

The selected controller overlay still applies, so `controller:=stanley` or `controller:=pure_pursuit` changes steering behavior without changing the line planner.

## Useful Commands

Run acceleration line test:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=acceleration planner:=linetest controller:=stanley
```

Run the same fixed line with pure pursuit:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py track:=acceleration planner:=linetest controller:=pure_pursuit
```
