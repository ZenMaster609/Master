# Tracked-Cone Planner Geometry Code Map

This page maps the shared geometry helpers used by the migrated tracked-cone planners: midpoint, single-boundary, and corridor.

## Primary Files

- `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`
- `sim_car/sim_car/planning/planner_utils.py`

## Function Map

### Boundary Chains

- `build_boundary_chain_data`: shared boundary-chain construction used by the midpoint, single-boundary, and corridor cores.
- `grow_boundary_chain_positions`: grows a one-side cone sequence from the nearest useful forward seed.
- `select_seed_index`: chooses the first forward cone for a boundary chain.
- `candidate_progresses_from_vehicle`: forward/outboard progress gate for candidate chain steps.
- `candidate_is_shadowed`: rejects a farther candidate when a closer cone sits in nearly the same direction.
- `materialize_boundary_chain_data`: converts selected chain positions into points, tangents, forward extent, and optional rejection reasons.
- `empty_boundary_chain_data`: creates a consistent empty chain result.

### Boundary Math

- `estimate_tangents`: estimates the local chain direction at each boundary point.
- `inward_normal`: rotates a boundary tangent toward the track interior for blue/yellow sides.
- `angle_between`: signed 2D angle helper used by chain growth and shadowing.

### Track Width

- `update_track_width_estimate`: shared rate-limited exponential update for the filtered track-width prior.

### Path Geometry

- `to_vehicle_frame`: converts odom-frame path points into the local vehicle frame.
- `path_cumulative_lengths`: computes station distances along a path.
- `moving_average`: shared smoothing primitive used by final path shaping.
- `path_heading_delta_max`: measures the largest heading change along a path.
- `path_curvature_abs_max`: measures maximum absolute curvature along a path.
- `path_self_intersects` and `segments_intersect`: shared self-intersection checks.

### Pairing Predicates

- `pair_width_in_range`: common width-range gate for pair/corridor samples.
- `inward_distance`: projection of a candidate partner onto the anchor's inward normal.
- `midpoint_outside_pair_span`: midpoint-planner geometry gate for implausible pair spans.
- `width_jump_exceeds`: continuity gate for pair-width jumps.
- `unknown_partner_check` and `unknown_partner_within_limits`: shared unknown-cone partner plausibility checks.
- `prefer_previous_partner_option`: chooses a previous track-ID partner when its cost stays within reassignment margin.

## Planner Usage

- `planner_utils.py`: owns shared cone filtering/order extraction, `_build_boundary_chain`, `_finalize_path`, `_resample_path`, path-length helpers, and empty-result field filling.
- `midpoint_planner_core.py`: uses shared filtering, boundary-chain, path finalization, width/tangent/normal helpers, and pair predicates inside midpoint-specific pair creation and ordering.
- `single_boundary_planner_core.py`: uses shared filtering, boundary-chain, path finalization, offset math, width update, and pair predicates while keeping one-boundary selection local.
- `corridor_planner_core.py`: uses shared filtering, boundary-chain, path finalization, width-range, width-update, and path-shape helpers while preserving corridor audit and membership logic locally.
