# Midpoint Planner

The midpoint planner builds a centerline by pairing cones from the left and right track boundaries, then driving through the midpoint of each accepted pair.

It is best when both sides of the track are visible often enough to form stable blue/yellow pairs.

## Core Idea

The planner turns a cloud of tracked cones into a path:

`tracked cones -> filtered boundary cones -> left/right chains -> paired cones -> midpoint chain -> centerline`

The planner does not need every cone. It needs enough reliable, forward-visible cones to build a sequence of plausible track-width pairs.

The low-level geometry helpers are shared with the single-boundary and corridor planners through `tracked_cone_planner_geometry.py`, while shared filtering, ordering, boundary-chain wrapping, and path finalization live in `planner_utils.py`. The midpoint core still owns the midpoint-specific pairing and ordering decisions.

## Cone Filtering

Each cycle, the node:

1. Receives cone detections from the configured tracked-cones topic.
2. Resolves the vehicle pose in the planning frame.
3. Converts cones into the planning frame.
4. Converts raw colors into planner-facing boundary colors.
5. Filters cones by geometry, confidence, and color.

Geometry filtering removes cones that are too far away or too far behind the car. Confidence filtering uses planner-facing confidence, which can come from cone memory track confidence rather than raw detector confidence.

Unknown and orange cones can be inferred to blue/yellow by lateral side. Boundary hints from routed cones take priority over raw color.

## Boundary Chains

After filtering, blue cones are treated as the left boundary and yellow cones as the right boundary. The planner orders candidates deterministically, then builds a chain for each side.

The chain builder rejects steps that are:

- too short
- too long
- not making enough forward progress
- changing heading too sharply

This avoids connecting cones that are close in space but implausible as neighboring track-boundary cones.

## Pair Creation

The midpoint planner then searches for left/right pairs. A pair is accepted when its geometry looks like a track-width cross-section.

Important checks include:

- pair width inside the configured minimum and maximum
- width not jumping too much from neighboring pairs
- pair orientation consistent with local boundary tangent
- optional opposite-color pairing
- optional unknown-pair completion

Pair memory helps avoid rapid reassignment when two possible pairings are close. A new pairing needs to be meaningfully better before replacing a previous one.

## Midpoint Ordering

Each accepted pair produces one midpoint. The planner orders midpoint candidates into a forward chain and trims segments that are too long.

The ordering logic is deliberately conservative near the vehicle because early path points dominate steering. It uses recent path/pair history to avoid backtracking or swapping the order of close midpoints between frames.

## Width Estimate

The planner maintains a filtered track-width estimate:

- It starts from `width_estimation.initial_width_m`.
- It is clamped between `width_estimation.min_width_m` and `width_estimation.max_width_m`.
- It only trusts measured widths when enough pairs are available.
- It updates gradually through `width_estimation.alpha` and `width_estimation.max_delta_per_update_m`.

This width estimate is used as the expected width prior for future pairing.

## Centerline Finalization

The raw midpoint chain is resampled into a centerline with the configured path resolution and maximum length. The planner can smooth local geometry with its smoothing window, then validates the final path.

Validation rejects paths that:

- have too few points
- have too little forward extent
- contain non-finite geometry
- start with a heading flip near the vehicle
- jump too far laterally from the previous valid path
- bend too sharply near the car
- self-intersect

When validation fails, the planner does not immediately publish arbitrary new geometry.

## Path Memory And Hold Behavior

The migrated tracked-cone planners share a midline memory layer. For midpoint, this memory is important because left/right pairings can change frame to frame as cone visibility changes.

The memory layer:

- keeps a sampled path buffer
- blends fresh candidates into the buffer
- updates near, mid, and far path sections at different rates
- caps lateral movement per update
- protects the near-field segment used by the controller
- holds the last valid path briefly when fresh planning fails

If a candidate path jumps too far from the buffered path, it can be rejected. After repeated jump rejections, the buffer is cleared so the planner can recover instead of staying locked to stale geometry.

## Controller Coupling

The midpoint planner publishes the centerline and, unless `controller:=none`, computes Ackermann commands using the selected controller. Speed is chosen from curvature through the shared `speed_control.*` parameters.

The controller sees the path in the vehicle frame. The planner can apply odometry lag compensation before transforming the path for control.

## Strengths And Weaknesses

Strengths:

- straightforward to explain
- strong path when both boundaries are visible
- centerline follows actual left/right track width

Weaknesses:

- sensitive to wrong cone pairing
- can struggle when one boundary disappears
- needs path memory to avoid near-field jitter

For one-sided visibility, the single-boundary planner is usually more appropriate. For a stricter left/right corridor interpretation, use the corridor planner.
