# Corridor Planner

The corridor planner builds a centerline by first constructing explicit left and right boundary chains, then sampling the drivable corridor between them.

It is stricter than the midpoint planner: it wants both boundaries to be reliable enough to describe a corridor, not just individual left/right cone pairs.

## Core Idea

The planner turns tracked cones into a corridor:

`tracked cones -> filtered colored cones -> left chain + right chain -> corridor samples -> fitted centerline`

Instead of selecting independent midpoint pairs, the corridor planner reasons about a continuous overlap between the two boundaries.

The low-level geometry helpers are shared with the midpoint and single-boundary planners through `tracked_cone_planner_geometry.py`, while common filtering, ordering, boundary-chain wrapping, and path finalization live in `planner_utils.py`. The corridor core keeps the corridor-specific sampling, fitting, and audit behavior local.

## Cone Filtering

The corridor planner keeps only planner-facing blue/yellow cones after geometry and confidence filtering. Unlike midpoint and single-boundary planning, it does not use unknown cones for pair completion in the core corridor algorithm.

The main geometry filters are:

- maximum range
- planning horizon
- maximum lateral range
- behind-car drop distance
- finite coordinates

This makes corridor planning cleaner, but it can reject more data when perception is sparse.

## Boundary Chain Construction

The planner builds one chain for blue cones and one chain for yellow cones. Each chain is a plausible sequence of boundary cones moving forward from the vehicle.

Chain construction checks:

- step distance
- forward projection along the current chain heading
- heading change
- shadowing by better candidates

## Corridor Sampling

Once both chains are long enough, the planner samples possible corridor cross-sections. A valid cross-section connects the left and right boundary at a reasonable width and forward position.

Corridor samples are rejected when they are:

- too narrow
- too wide
- behind the useful corridor
- beyond the planning horizon
- not part of the longest valid overlap
- non-finite

The accepted samples define the corridor center anchors, widths, boundary points, and rung segments.

## Centerline Fitting

The raw center anchors are smoothed and resampled into a centerline. The output path uses:

- `corridor.path_fit_smoothing_window`
- `centerline.path_resolution_m`
- `centerline.max_path_length_m`

The result is a continuous path that stays within the sampled corridor instead of jumping between isolated pairs.

## Corridor Membership And Validation

The planner validates the centerline before treating it as fresh.

Validation checks include:

- enough corridor samples
- enough path points
- enough forward extent
- finite geometry
- path remains inside the corridor with `corridor.membership_margin_m`
- initial heading sanity
- near-field lateral jump limit
- heading delta limit
- curvature limit
- self-intersection

If the path exits the corridor or bends too sharply, it is rejected instead of being handed directly to the controller.

## Path Stabilization

The corridor planner uses the same midline memory layer as the other migrated tracked-cone planners.

It additionally keeps pair memory for corridor rungs so temporary missed detections do not immediately destroy the local corridor. The memory is still bounded by retention and validation; it is not meant to invent a long track from stale data.

The near-field path is anchored near the vehicle so the controller does not chase a laterally offset path origin.

## Strengths And Weaknesses

Strengths:

- strongest geometric interpretation when both boundaries are visible
- good debug/audit visibility
- avoids many wrong isolated pairings

Weaknesses:

- requires enough reliable cones on both sides
- more parameters affect rejection
- sparse or one-sided perception can lead to held paths

Use corridor when you want the planner to obey a continuous drivable channel and have enough cone quality to support that stricter model.
