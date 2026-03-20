from __future__ import annotations

import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'full_sim_launch.launch.py'
STANLEY_CONFIG = REPO_ROOT / 'sim_car' / 'config' / 'controllers' / 'stanley.yaml'


def test_full_launch_uses_single_perception_node_with_stereo_toggle():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'stereo'" in content
    assert "DeclareLaunchArgument(\n        'mono'" not in content
    assert "executable='perception_node'" in content
    assert "stereo_enabled': ParameterValue(" in content
    assert "mono_perception_node" not in content
    assert "stereo_perception_node" not in content


def test_full_launch_declares_and_passes_thesis_controller_diagnostics():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'thesis_controller_diagnostics'" in content
    assert "DeclareLaunchArgument(\n        'thesis_controller_diagnostics_live_plot_enabled'" in content
    assert "'thesis_controller_diagnostics'" in content
    assert "'thesis_controller_diagnostics_enabled': LaunchConfiguration('thesis_controller_diagnostics')" in content
    assert (
        "'enable_thesis_controller_diagnostics_plot': LaunchConfiguration(\n"
        "                'thesis_controller_diagnostics_live_plot_enabled'\n"
        "            )"
    ) in content
    assert "'diagnostics.publish_thesis_context': ParameterValue(" in content


def test_full_launch_declares_camera_cone_rmse_plotting_and_wires_it():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'camera_cone_rmse_plotting'" in content
    assert content.count("LaunchConfiguration('camera_cone_rmse_plotting')") >= 4
    assert "'camera_cone_eval_topic': PythonExpression([" in content


def test_full_launch_supports_midpoint_and_single_boundary_planners():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "Planner to launch: 'delaunay', 'midpoint', 'single_boundary', or 'none'" in content
    assert "executable='midpoint_planner_node'" in content
    assert "executable='single_boundary_planner_node'" in content
    assert "\"'.lower() == 'midpoint'\"" in content
    assert "\"'.lower() == 'single_boundary'\"" in content
    assert "supported_planners = {'delaunay', 'midpoint', 'single_boundary', 'none'}" in content
    assert "hybrid_force_single_boundary" not in content


def test_full_launch_supports_controller_selection_via_controller_yaml():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "Controller to use: 'stanley', 'pure_pursuit', or 'none'" in content
    assert "supported_controllers = {'stanley', 'pure_pursuit', 'none'}" in content
    assert "'controllers'," in content
    assert "'stanley.yaml' if '" in content
    assert "'pure_pursuit.yaml' if '" in content
    assert "'none.yaml'" in content


def test_full_launch_loads_controller_yaml_after_planner_yaml():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    delaunay_block = content.split("delaunay_planner_node = Node(", 1)[1].split("midpoint_planner_node = Node(", 1)[0]
    midpoint_block = content.split("midpoint_planner_node = Node(", 1)[1].split("single_boundary_planner_node = Node(", 1)[0]
    single_boundary_block = content.split("single_boundary_planner_node = Node(", 1)[1].split("rviz_node = Node(", 1)[0]

    assert delaunay_block.index("PathJoinSubstitution([sim_car_share, 'config', 'delaunay_planner.yaml'])") < delaunay_block.index("controller_config")
    assert midpoint_block.index("PathJoinSubstitution([sim_car_share, 'config', 'midpoint_planner.yaml'])") < midpoint_block.index("controller_config")
    assert single_boundary_block.index("PathJoinSubstitution([sim_car_share, 'config', 'single_boundary_planner.yaml'])") < single_boundary_block.index("controller_config")


def test_stanley_controller_yaml_matches_pre_split_baseline():
    content = STANLEY_CONFIG.read_text(encoding='utf-8')

    assert "k_gain: 1.2" in content
    assert "softening_speed_mps: 0.0" in content
    assert "heading_gain: 1.0" in content
    assert "lookahead_idx_offset: 0" in content
    assert "steering_lowpass_alpha: 1.0" in content
