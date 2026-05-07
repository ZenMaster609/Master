# Planning System

The planning system turns remembered cone tracks and odometry into a centerline, a steering command, diagnostics, and RViz markers. It is built around one shared tracked-cone runtime with three interchangeable centerline algorithms: midpoint, single-boundary, and corridor. A separate linetest planner exists for fixed-line controller checks, and a skidpad router can filter `/tracked_cones` before the tracked-cone planner sees them.

## Runtime Shape

The normal data path is:

`cone memory -> /tracked_cones -> optional skidpad router -> tracked-cone planner -> /planned_centerline + /cmd + diagnostics + markers`

Cone memory publishes `ConeDetectionArray` messages with positions, raw colors, boundary color hints, track IDs, track states, and confidence values. The planner also consumes `/sim/odom` so it can resolve the vehicle pose, transform cones into the planning frame, discard unusable cones, and convert the final path into the controller frame.

The planner starts in `waiting` until it has tracked cones and a vehicle pose. Once a usable path is produced it publishes in `fresh` mode. If the current frame is weak or fails validation, the runtime can publish a remembered path in `held` mode. If no usable or held path is available, the runtime either sends zero command or repeats the last command depending on `control.stop_if_no_path`.

## Shared Planning Cycle

Each tracked-cone planner timer does the same high-level work:

1. Snapshot the latest tracked cones and odometry.
2. Resolve the requested planning frame and the vehicle pose.
3. Transform cones into that frame and normalize raw colors into planner colors.
4. Filter cones by range, confidence, behind-car distance, and event-specific routing.
5. Run the selected centerline algorithm.
6. Validate the candidate path for enough points, forward extent, heading, jumps, and shape.
7. Update width estimates, pair memory, and committed midline memory.
8. Select fresh or held publication mode.
9. Convert the control path into the vehicle frame and run Stanley or pure pursuit.
10. Publish the path, command, diagnostics, and RViz markers.

The shared runtime owns the mechanics around transforms, hold behavior, controller invocation, diagnostics, and visualization. The individual algorithms own only the geometric decision of how to build a candidate path from cones.

## Cone Meaning

The planner sees two color layers:

- Raw color: what perception or cone memory believes the cone is.
- Boundary color: the planning-side left/right hint after color normalization and inference.

Blue and yellow cones are used as left/right boundaries. Unknown and orange cones can be inferred by side when enabled, which helps sparse or ambiguous perception. Track state matters too: confirmed tracks are normally trusted; tentative tracks are gated per planner; stale cones are counted and visualized but are not treated as strong live evidence.

## Midpoint Planner

The midpoint planner tries to build left and right boundary chains, pair opposite-side cones, and connect the pair midpoints into a centerline.

It works best when both sides of the track are visible. Its important decisions are:

- grow plausible left and right chains forward from the vehicle;
- reject pairs with impossible width, bad orientation, poor progress, or unstable reassignment;
- keep pair memory so pair choices do not flap every frame;
- smooth and resample the midpoint chain into the published centerline.

Midpoint is the most direct interpretation of a normal two-boundary track.

## Single-Boundary Planner

The single-boundary planner can publish a path when only one side is reliable. It selects the best boundary chain, estimates the inward normal, offsets the boundary by the current width estimate, and uses pair evidence when available to update that width estimate.

It is useful when one side of the track is missing, sparse, or temporarily occluded. Because it invents the opposite side through an offset, it validates the near-field path aggressively and relies on width memory to avoid sudden lateral jumps.

## Corridor Planner

The corridor planner is stricter. It builds both boundary chains, samples a drivable corridor between them, validates corridor width and membership, and fits the centerline through the accepted corridor anchors.

It is useful when the planner should prefer a conservative two-boundary channel over a looser pairing strategy. It rejects more situations than single-boundary because it requires enough corridor samples and a consistent channel.

## Midline Memory And Holding

Every tracked-cone planner feeds candidate paths into committed midline memory. Memory stores a station-sampled path ahead of the vehicle. Fresh candidates can seed, blend into, or directly replace the stored path. Candidates with large near-field jumps are held back unless they pass recovery checks over repeated frames.

There are two related stabilizers:

- Last-valid hold: keeps a recently valid path for a configured duration when the current frame fails.
- Midline memory: blends valid candidates into a committed path so the controller sees stable near-field geometry.

This is why planner diagnostics distinguish `fresh`, `held`, and `stopped` states instead of only saying whether the core algorithm succeeded.

## Controller Output

The planner publishes a centerline in the planning frame, then extracts the forward portion needed for control and transforms it into the vehicle frame. The selected controller receives that local path, speed, and yaw-rate data.

Stanley combines heading error and cross-track error. Pure pursuit selects a lookahead point and steers toward it. Both return steering, curvature, target/lookahead information, and debug metrics. The planner uses curvature to choose a speed between `speed_control.speed_min_mps` and `speed_control.speed_max_mps`.

If `controller:=none`, the planner can still publish the centerline and diagnostics without steering the vehicle.

## Linetest Planner

The linetest planner does not consume cones. It publishes a fixed line or an optional ground-truth-derived path and runs the same steering-controller factory as the tracked-cone planners.

Use it to debug controller signs, odometry lag, Ackermann command behavior, and end-of-line braking without involving perception or cone pairing.

## Skidpad And Acceleration Routing

The skidpad router sits between cone memory and the planner for skidpad and acceleration-style runs. It subscribes to `/tracked_cones`, tracks mission stage from odometry, masks cones to the active branch, republishes routed cones, and publishes route diagnostics and markers.

For skidpad, it selects the active lobe or straight section. For acceleration parking behavior, it can detect the final stop row, override boundary colors for the parking corridor, and publish stop/brake commands near the target.

## Diagnostics And RViz

Planner diagnostics are the main operator view of why the planner is behaving a certain way. The important fields are:

- planner state: `waiting`, `fresh`, `held`, or `stopped`;
- reason: waiting for cones, missing pose, transform failure, no safe chain, continuity rejection, controller failure, and similar;
- path metrics: point count, chain length, pair count, width estimate, candidate counts, near-field jump, and hold time;
- control metrics: command speed, steering, curvature, lookahead, and zero-command flag.

RViz markers show remembered cones, boundary chains, accepted pairs or corridor rungs, raw candidate paths, final centerline, lookahead target, and the same operator status text. If cones disappear from RViz and the planner says it is waiting for tracked cones, check cone memory and the selected planner input topic first.

## Tuning Model

Most planner tuning is grouped by responsibility:

- `filtering.*`: which cones are allowed into planning.
- `boundary_chain.*`: how side chains grow.
- `pairing.*`: how left/right candidates are paired.
- `width_estimation.*`: how the expected track width updates.
- `centerline.*`: path resolution, smoothing, and max length.
- `validation.*`: minimum path quality and jump limits.
- `midline_memory.*`: committed-path blending and hold behavior.
- `control.*`, `stanley.*`, `pure_pursuit.*`, and `speed_control.*`: controller and speed behavior.

Track YAML files provide overlays for controller gains, spawn pose, speed limits, and event-specific planning horizon. Launch arguments choose the active planner, controller, track, and whether cone memory or routing is used.
