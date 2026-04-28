# Line Test Planner Code Map

This page maps the `documentation/linetest_planner.md` behavior to the fixed-line planner node and its controller/braking helpers.

## Primary Files

- `sim_car/sim_car/planning/linetest_planner_node.py`
- `sim_car/sim_car/planning/controller_config.py`
- `sim_car/sim_car/controllers/stanley_controller.py`
- `sim_car/sim_car/controllers/pure_pursuit_controller.py`

## Function Map

### Core Idea

- `LineTestPlannerNode.__init__` in `sim_car/sim_car/planning/linetest_planner_node.py`: initializes the fixed-line planner state and generated centerline.
- `LineTestPlannerNode._generate_line_path` in `sim_car/sim_car/planning/linetest_planner_node.py`: constructs the fixed straight centerline from configured start/end points and spacing.
- `LineTestPlannerNode._current_centerline` in `sim_car/sim_car/planning/linetest_planner_node.py`: returns the active line or GT-derived centerline depending on mode.

### Control Path

- `LineTestPlannerNode._on_timer` in `sim_car/sim_car/planning/linetest_planner_node.py`: main planning/control loop for fixed-line tracking.
- `LineTestPlannerNode._resolve_vehicle_pose` in `sim_car/sim_car/planning/linetest_planner_node.py`: resolves the current vehicle pose used for line projection and control.
- `LineTestPlannerNode._build_forward_control_path` in `sim_car/sim_car/planning/linetest_planner_node.py`: extracts the remaining forward segment of the fixed line.
- `LineTestPlannerNode._centerline_to_vehicle_frame` in `sim_car/sim_car/planning/linetest_planner_node.py`: converts the selected forward path into the controller frame.
- `LineTestPlannerNode._build_steering_controller` in `sim_car/sim_car/planning/linetest_planner_node.py`: creates the configured Stanley or pure-pursuit controller.

### Controller Invocation

- `build_steering_controller` in `sim_car/sim_car/planning/controller_config.py`: shared controller factory used by the node.
- `StanleyController.compute` in `sim_car/sim_car/controllers/stanley_controller.py`: controller path when `controller_type` is `stanley`.
- `PurePursuitController.compute` in `sim_car/sim_car/controllers/pure_pursuit_controller.py`: controller path when `controller_type` is `pure_pursuit`.
- `LineTestPlannerNode._publish_cmd` in `sim_car/sim_car/planning/linetest_planner_node.py`: publishes the resulting Ackermann command.

### Parking And Brake Behavior

- `LineTestPlannerNode._line_remaining_distance_m` in `sim_car/sim_car/planning/linetest_planner_node.py`: computes distance remaining to the end of the line.
- `LineTestPlannerNode._linetest_brake_cmd` in `sim_car/sim_car/planning/linetest_planner_node.py`: implements the configured end-of-line brake command.
- `LineTestPlannerNode._publish_brake_cmd` in `sim_car/sim_car/planning/linetest_planner_node.py`: publishes the brake command topic.
- `LineTestPlannerNode._apply_no_path_behavior` in `sim_car/sim_car/planning/linetest_planner_node.py`: handles the no-forward-path stop behavior.

### Diagnostics And Debugging

- `LineTestPlannerNode._publish_outputs` in `sim_car/sim_car/planning/linetest_planner_node.py`: publishes the path, markers, and control command for the current cycle.
- `LineTestPlannerNode._publish_diagnostics` in `sim_car/sim_car/planning/linetest_planner_node.py`: emits planner diagnostics.
- `LineTestPlannerNode._control_debug_metrics` in `sim_car/sim_car/planning/linetest_planner_node.py`: packages controller debug metrics for diagnostics and logging.

## Related Entry Points

- `LineTestPlannerNode._build_forward_gt_control_path` and `_maybe_build_gt_midline` in `sim_car/sim_car/planning/linetest_planner_node.py`: support the GT midline mode used for some controller tests.
- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: launches this planner when `planner:=linetest`.
