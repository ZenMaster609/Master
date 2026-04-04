from __future__ import annotations

import importlib.util
import pathlib

import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'full_sim_launch.launch.py'
STANLEY_CONFIG = REPO_ROOT / 'sim_car' / 'config' / 'controllers' / 'stanley.yaml'
SIM_CAR_SHARE = REPO_ROOT / 'sim_car'
TRACKS = {
    'acceleration': 'acceleration.world',
    'skidpad': 'skidpad.world',
    'smalltrack': 'small_track.world',
}
MIGRATED_PLANNERS = ('midpoint', 'single_boundary', 'corridor')

spec = importlib.util.spec_from_file_location('full_sim_launch', FULL_LAUNCH)
assert spec is not None
assert spec.loader is not None
full_sim_launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_sim_launch)


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


def test_full_launch_declares_and_passes_path_tracking_eval():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'path_tracking_eval'" in content
    assert "'path_tracking_eval'" in content
    assert "'path_tracking_eval_enabled': LaunchConfiguration('path_tracking_eval')" in content
    assert "'path_tracking_eval_gt_track_topic': '/ground_truth/track'" in content
    assert "'path_tracking_eval_odom_topic': '/sim/odom'" in content
    assert "'path_tracking_eval_planner_path_topic': '/planned_centerline'" in content
    assert "'path_tracking_eval_track_name': LaunchConfiguration('track')" in content


def test_full_launch_declares_camera_cone_rmse_plotting_and_wires_it():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'camera_cone_rmse_plotting'" in content
    assert content.count("LaunchConfiguration('camera_cone_rmse_plotting')") >= 4
    assert "'camera_cone_eval_topic': PythonExpression([" in content


def test_full_launch_supports_midpoint_single_boundary_and_corridor_planners():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "Planner to launch: 'delaunay', 'midpoint', 'single_boundary', 'corridor', or 'none'" in content
    assert "executable='skidpad_router_node'" in content
    assert "executable='midpoint_planner_node'" in content
    assert "executable='single_boundary_planner_node'" in content
    assert "executable='corridor_planner_node'" in content
    assert "\"'.lower() == 'midpoint'\"" in content
    assert "\"'.lower() == 'single_boundary'\"" in content
    assert "\"'.lower() == 'corridor'\"" in content
    assert "SUPPORTED_PLANNERS = {'delaunay', 'midpoint', 'single_boundary', 'corridor', 'none'}" in content
    assert "hybrid_force_single_boundary" not in content


def test_full_launch_declares_track_arg_and_optional_controller_override():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'track'" in content
    assert "Track preset to load: 'acceleration', 'skidpad', or 'smalltrack'" in content
    assert "Optional controller override: 'stanley', 'pure_pursuit', or 'none'" in content
    assert "default_value=''" in content


def test_full_launch_supports_planner_specific_rviz_profiles():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "default_value='planner'" in content
    assert "'planner', 'clean', 'planner_debug', 'midpoint'," in content
    assert "'single_boundary', or 'corridor'" in content
    assert "'midpoint': 'midpoint_planner.rviz'" in content
    assert "'single_boundary': 'single_boundary_planner.rviz'" in content
    assert "'corridor': 'corridor_planner.rviz'" in content
    assert "if rviz_profile in {'planner', 'auto'}:" in content
    assert "resolved_filename = profile_to_filename.get(planner, 'driving_clean.rviz')" in content


def test_full_launch_uses_resolved_planner_configs_for_migrated_planners():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    midpoint_block = content.split("midpoint_planner_node = Node(", 1)[1].split("single_boundary_planner_node = Node(", 1)[0]
    single_boundary_block = content.split("single_boundary_planner_node = Node(", 1)[1].split("corridor_planner_node = Node(", 1)[0]
    corridor_block = content.split("corridor_planner_node = Node(", 1)[1].split("camera_debug_viewer_node = Node(", 1)[0]

    assert "resolved_planner_config" in midpoint_block
    assert "resolved_planner_config" in single_boundary_block
    assert "resolved_planner_config" in corridor_block
    assert "PathJoinSubstitution([sim_car_share, 'config', 'midpoint_planner.yaml'])" not in midpoint_block
    assert "PathJoinSubstitution([sim_car_share, 'config', 'single_boundary_planner.yaml'])" not in single_boundary_block
    assert "PathJoinSubstitution([sim_car_share, 'config', 'corridor_planner.yaml'])" not in corridor_block


def test_full_launch_wires_skidpad_router_output_into_migrated_planners() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "skidpad_router_node = Node(" in content
    assert "PathJoinSubstitution([sim_car_share, 'config', 'skidpad', 'skidpad_router.yaml'])" in content
    assert "'topics.output_topic': '/tracked_cones/skidpad_routed'" in content
    assert "'topics.viz_topic': router_viz_topic" in content
    assert "'/tracked_cones/skidpad_routed' if '" in content
    assert "'.lower() in ('skidpad', 'acceleration') else ('/tracked_cones' if '" in content
    assert "'routing.event_mode': LaunchConfiguration('track')" in content


def test_full_launch_only_shows_skidpad_router_markers_for_skidpad() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "router_viz_topic = PythonExpression([" in content
    assert "\"'.lower() == 'skidpad' else '/skidpad_router/markers_hidden'\"" in content


def test_track_bundle_resolves_world_and_planner_config_paths():
    for track, world_name in TRACKS.items():
        for planner in MIGRATED_PLANNERS:
            selection = full_sim_launch._resolve_launch_selection(
                SIM_CAR_SHARE,
                track=track,
                planner=planner,
            )
            assert selection['world'] == str(SIM_CAR_SHARE / 'worlds' / world_name)
            assert selection['planner_config'] == str(SIM_CAR_SHARE / 'config' / track / f'{planner}_planner.yaml')
            assert selection['spawn_config'] == str(SIM_CAR_SHARE / 'config' / track / 'spawn.yaml')


def test_track_bundle_loads_track_spawn_defaults():
    expected = {
        'acceleration': ('-47.5', '0.0', '0.0'),
        'skidpad': ('0.0', '-10.0', '1.5708'),
        'smalltrack': ('9.58', '-5.2', '3.75'),
    }

    for track, (spawn_x, spawn_y, spawn_yaw) in expected.items():
        selection = full_sim_launch._resolve_launch_selection(
            SIM_CAR_SHARE,
            track=track,
            planner='midpoint',
        )
        assert selection['spawn_x'] == spawn_x
        assert selection['spawn_y'] == spawn_y
        assert selection['spawn_yaw'] == spawn_yaw


def test_track_bundle_overrides_world_spawn_and_controller_when_requested():
    selection = full_sim_launch._resolve_launch_selection(
        SIM_CAR_SHARE,
        track='smalltrack',
        planner='corridor',
        world_override='/tmp/custom.world',
        spawn_x_override='1.0',
        spawn_y_override='2.0',
        spawn_yaw_override='3.0',
        controller_override='pure_pursuit',
    )

    assert selection['world'] == '/tmp/custom.world'
    assert selection['spawn_x'] == '1.0'
    assert selection['spawn_y'] == '2.0'
    assert selection['spawn_yaw'] == '3.0'
    assert selection['controller_override'] == 'pure_pursuit'


def test_delaunay_keeps_controller_yaml_selection():
    selection = full_sim_launch._resolve_launch_selection(
        SIM_CAR_SHARE,
        track='smalltrack',
        planner='delaunay',
    )
    assert selection['planner_config'] == str(SIM_CAR_SHARE / 'config' / 'delaunay_planner.yaml')
    assert selection['delaunay_controller'] == 'stanley'


def test_track_specific_planner_configs_embed_control_sections():
    for track in TRACKS:
        for planner in MIGRATED_PLANNERS:
            config_path = SIM_CAR_SHARE / 'config' / track / f'{planner}_planner.yaml'
            with config_path.open('r', encoding='utf-8') as config_file:
                config = yaml.safe_load(config_file) or {}

            node_name = f'{planner}_planner_node'
            params = config[node_name]['ros__parameters']
            assert params['control']['controller_type'] == 'stanley'
            assert 'stanley' in params
            assert 'pure_pursuit' in params


def test_track_specific_spawn_configs_define_pose():
    for track in TRACKS:
        config_path = SIM_CAR_SHARE / 'config' / track / 'spawn.yaml'
        with config_path.open('r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file) or {}

        assert set(config['spawn']) == {'spawn_x', 'spawn_y', 'spawn_yaw'}


def test_stanley_controller_yaml_matches_pre_split_baseline():
    content = STANLEY_CONFIG.read_text(encoding='utf-8')

    assert "k_gain: 1.2" in content
    assert "softening_speed_mps: 0.0" in content
    assert "heading_gain: 1.6" in content
    assert "lookahead_idx_offset: 0" in content
    assert "steering_lowpass_alpha: 1.0" in content
