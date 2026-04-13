# Single-Boundary Planner

The single-boundary planner builds a driveable centerline from whichever track boundary is reliable enough. Instead of requiring left/right pairs, it can offset one visible boundary inward by the estimated track width.

This makes it useful when one side of the track is missing, occluded, or unreliable.

## Core Idea

The planner turns tracked cones into a path:

`tracked cones -> filtered boundary cones -> best boundary chain -> inward offset path -> centerline`

It still reads both blue and yellow cones, and it can evaluate possible pairs, but its main planned path comes from a selected single boundary chain.

## Cone Filtering

Filtering is similar to the midpoint planner:

- remove cones too far away
- remove cones too far behind the car
- remove low-confidence cones
- keep blue/yellow cones
- optionally use unknown cones for pair completion

Unknown and orange cones can be resolved by side or by boundary hints from routing/fusion.

## Boundary Chains

The planner builds candidate chains for the blue side and yellow side. A chain is a forward sequence of cones that looks like one physical boundary.

Chain gates reject:

- steps that are too close
- steps that are too far
- steps with too little forward progress
- sharp heading changes

The planner then chooses a fallback chain. "Fallback" here means one-boundary planning, not failure. If either boundary has enough cones and enough forward extent, the planner can generate a centerline.

## Offset Path Generation

Once a boundary is selected, the planner estimates the inward direction and offsets the chain by the expected track width. The side determines the offset direction:

- blue boundary is offset toward the right side of the track
- yellow boundary is offset toward the left side of the track

The expected width comes from the filtered width estimate. If good left/right pairs are visible, they can update the width estimate. If not, the planner keeps using the prior width.

This is the key difference from the midpoint planner: single-boundary planning can keep driving from one clean boundary instead of waiting for full cross-track pairs.

## Track-Width Use

The planner keeps a track-width estimate with:

- starting width
- min/max bounds
- slow update alpha
- maximum per-update width shift
- minimum trustworthy pair count

When both sides are visible, the width estimate can improve. When only one side is visible, the estimate holds steady and the planner offsets the visible chain by that width.

## Validation

After the offset path is finalized, the planner validates it before publishing as a fresh path.

Validation checks include:

- minimum point count
- minimum forward extent
- finite geometry
- near-vehicle heading sanity
- near-field lateral jump against previous path
- heading/kink limit
- self-intersection

Single-boundary mode allows a larger near-field lateral jump than normal paired planning. This is intentional because switching from a poor two-sided estimate to a one-sided offset can otherwise look like a large jump even when it is the better plan.

## Path Memory And Hold Behavior

Like midpoint and corridor, the single-boundary planner uses the shared midline memory layer.

The memory layer:

- stores a sampled midline
- blends candidates by distance ahead of the vehicle
- limits lateral changes near the controller handoff region
- holds a recent valid path when fresh planning fails
- clears after repeated rejected jumps so the planner can recover

This is especially important for one-boundary planning because the selected side can change as cone visibility changes.

## Lap-Gate Support

The node also contains smalltrack lap-gate support. It can build a gate from big orange cones in ground truth and count crossings when lap tracking is enabled. This is used for automated run stopping/evaluation behavior, not for creating the path itself.

## Strengths And Weaknesses

Strengths:

- works with one visible boundary
- less dependent on left/right pairing
- useful when cone colors or detections are incomplete

Weaknesses:

- depends heavily on the track-width prior
- can be biased if the visible boundary is noisy
- has less geometric confirmation than midpoint or corridor planning

Use this planner when maintaining continuity through sparse detections matters more than strict two-boundary geometry.
