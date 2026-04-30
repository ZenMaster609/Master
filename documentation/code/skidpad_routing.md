# Skidpad Routing Code Map

This page maps the `documentation/skidpad_routing.md` behavior to the skidpad router node and state-machine helpers.

## Primary Files

- `sim_car/sim_car/planning/skidpad_router_node.py`
- `sim_car/sim_car/planning/skidpad_router_core.py`
- `sim_car/config/skidpad/skidpad_router.yaml`

## Function Map

### Core Idea

- `SkidpadRouterNode._cones_cb` in `sim_car/sim_car/planning/skidpad_router_node.py`: receives cone detections, routes them through the current mission state, and republishes the filtered output.
- `SkidpadRouterNode._route_non_parking_cones` in `sim_car/sim_car/planning/skidpad_router_node.py`: normal branch-routing path for approach and circle phases.
- `SkidpadRouterNode._route_parking_cones` in `sim_car/sim_car/planning/skidpad_router_node.py`: parking-mode path that reinterprets orange cones as planner-facing boundaries.

### State Machine

- `SkidpadStateMachine` in `sim_car/sim_car/planning/skidpad_router_core.py`: mission-state object that tracks approach, laps, straight exit, and parked state.
- `SkidpadRouterNode._odom_cb` in `sim_car/sim_car/planning/skidpad_router_node.py`: feeds vehicle pose updates into routing and state progression.
- `SkidpadRouterNode._make_initial_snapshot` in `sim_car/sim_car/planning/skidpad_router_node.py`: creates the initial route snapshot used by the node and diagnostics.

### Geometry Regions

- `config_from_parameters` in `sim_car/sim_car/planning/skidpad_router_core.py`: loads the geometric region parameters and routing sequence.
- `SkidpadRouterNode._convert_cones_to_odom` in `sim_car/sim_car/planning/skidpad_router_node.py`: converts incoming cone detections into odom-frame geometry before region tests.
- `SkidpadRouterNode._point_to_odom` and `_vehicle_point_to_odom` in `sim_car/sim_car/planning/skidpad_router_node.py`: low-level helpers for geometry-frame conversion.

### Lap Arming

- `SkidpadRouterNode._update_acceleration_parking_latch` in `sim_car/sim_car/planning/skidpad_router_node.py`: updates the acceleration finish/parking latch.
- `SkidpadRouterNode._build_status_text` in `sim_car/sim_car/planning/skidpad_router_node.py`: summarizes the current route state, including armed/complete transitions, for diagnostics.

### Parking Mode

- `SkidpadRouterNode._parking_mode_is_active` in `sim_car/sim_car/planning/skidpad_router_node.py`: decides when the router enters parking logic.
- `boundary_color_from_lateral_y` in `sim_car/sim_car/planning/skidpad_router_core.py`: converts lateral side into planner-facing boundary color during parking.
- `SkidpadRouterNode._boundary_color_for_odom_point` in `sim_car/sim_car/planning/skidpad_router_node.py`: node-side wrapper that applies the same lateral-side color logic to routed odom points.

### Stop-Line And Acceleration Finish Detection

- `detect_stop_line_forward_distance_m` in `sim_car/sim_car/planning/skidpad_router_core.py`: computes forward stop-line distance from candidate orange cone geometry.
- `detect_stop_line_pair` in `sim_car/sim_car/planning/skidpad_router_core.py`: detects the stop-line cone pair used by skidpad parking.
- `detect_acceleration_stop_row` in `sim_car/sim_car/planning/skidpad_router_core.py`: identifies the acceleration stop row from orange-cone geometry.
- `SkidpadRouterNode._detect_stop_line_cluster_mask`, `_detect_acceleration_stop_row_mask`, and `_apply_acceleration_stop_row_detection` in `sim_car/sim_car/planning/skidpad_router_node.py`: apply the core detection logic to the routed cone set.

### Stop Override

- `SkidpadRouterNode._publish_parking_override_cmd` and `_parking_override_brake_cmd` in `sim_car/sim_car/planning/skidpad_router_node.py`: command the final parking stop override.

## Related Entry Points

- `SkidpadRouterNode._publish_diagnostics` and `_publish_markers` in `sim_car/sim_car/planning/skidpad_router_node.py`: expose route state, masks, and stop targets for debugging.
- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: launches this node on `skidpad` and `acceleration` for tracked-cone planners.
