from __future__ import annotations

import ast
import importlib.util
import pathlib

import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'full_sim_launch.launch.py'
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


def _flatten(data, prefix: str = '') -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in data.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, name))
        else:
            out[name] = value
    return out


def _planner_node_contract(planner: str) -> tuple[set[str], set[str]]:
    planner_path = SIM_CAR_SHARE / 'sim_car' / 'planning' / f'{planner}_planner_node.py'
    module = ast.parse(planner_path.read_text(encoding='utf-8'))

    declared_defaults: dict[str, object] | None = None
    read_parameters: set[str] = set()

    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get_parameter':
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                read_parameters.add(node.args[0].value)
        if isinstance(node, ast.FunctionDef) and node.name == '_declare_parameters':
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == 'defaults':
                        declared_defaults = ast.literal_eval(stmt.value)
                        break
                if declared_defaults is not None:
                    break

    assert declared_defaults is not None, planner
    return set(declared_defaults), read_parameters


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
    assert "thesis_controller_diagnostics_live_plot_enabled" not in content
    assert "'thesis_controller_diagnostics'" in content
    assert "'thesis_controller_diagnostics_enabled': LaunchConfiguration('thesis_controller_diagnostics')" in content
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


def test_full_launch_always_wires_offline_cone_eval_topics():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'camera_cone_rmse_plotting'" not in content
    assert "DeclareLaunchArgument(\n        'lidar_cone_rmse_plotting'" not in content
    assert "DeclareLaunchArgument(\n        'cone_rmse_plotting'" not in content
    assert "'camera_cone_eval_topic': PythonExpression([" in content
    assert "'enable_log': 'true'" in content
    assert "'source_name': camera_source_name" in content
    assert "'source_name': 'lidar'" in content


def test_full_launch_supports_only_migrated_planners():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "Planner to launch: 'midpoint', 'single_boundary', 'corridor', or 'none'" in content
    assert "executable='midpoint_planner_node'" in content
    assert "executable='single_boundary_planner_node'" in content
    assert "executable='corridor_planner_node'" in content
    assert "\"'.lower() == 'midpoint'\"" in content
    assert "\"'.lower() == 'single_boundary'\"" in content
    assert "\"'.lower() == 'corridor'\"" in content
    assert "SUPPORTED_PLANNERS = {'midpoint', 'single_boundary', 'corridor', 'none'}" in content


def test_full_launch_declares_track_arg_and_optional_controller_override():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'track'" in content
    assert "Track preset to load: 'acceleration', 'skidpad', or 'smalltrack'" in content
    assert "Optional controller override: 'stanley', 'pure_pursuit', or 'none'" in content
    assert "default_value=''" in content
    assert "config', 'controllers'" not in content


def test_full_launch_supports_planner_specific_rviz_profiles():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "default_value='planner'" in content
    assert "'planner', 'clean', 'planner_debug', 'midpoint'," in content
    assert "'single_boundary', or 'corridor'" in content
    assert "'midpoint': 'midpoint_planner.rviz'" in content
    assert "'single_boundary': 'single_boundary_planner.rviz'" in content
    assert "'corridor': 'corridor_planner.rviz'" in content


def test_full_launch_uses_resolved_track_specific_planner_configs():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    midpoint_block = content.split("midpoint_planner_node = Node(", 1)[1].split("single_boundary_planner_node = Node(", 1)[0]
    single_boundary_block = content.split("single_boundary_planner_node = Node(", 1)[1].split("corridor_planner_node = Node(", 1)[0]
    corridor_block = content.split("corridor_planner_node = Node(", 1)[1].split("camera_debug_viewer_node = Node(", 1)[0]

    assert "resolved_planner_config" in midpoint_block
    assert "resolved_planner_config" in single_boundary_block
    assert "resolved_planner_config" in corridor_block
    assert "config', 'midpoint_planner.yaml'" not in content
    assert "config', 'single_boundary_planner.yaml'" not in content
    assert "config', 'corridor_planner.yaml'" not in content


def test_full_launch_wires_skidpad_router_output_into_migrated_planners() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "skidpad_router_node = Node(" in content
    assert "PathJoinSubstitution([sim_car_share, 'config', 'skidpad', 'skidpad_router.yaml'])" in content
    assert "'topics.output_topic': '/tracked_cones/skidpad_routed'" in content
    assert "'topics.viz_topic': router_viz_topic" in content
    assert "router passes normal cones through unchanged until that" in content
    assert "'/tracked_cones/skidpad_routed' if '" in content
    assert "'.lower() in ('skidpad', 'acceleration') else ('/tracked_cones' if '" in content
    assert "'routing.event_mode': LaunchConfiguration('track')" in content
    assert "'.lower() in ('skidpad', 'acceleration') else '/skidpad_router/markers_hidden'" in content


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
        'acceleration': ('-38.5', '0.0', '0.0'),
        'skidpad': ('0.0', '-10.0', '1.5708'),
        'smalltrack': ('-13.6758', '10.3753', '0'),
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


def test_track_specific_planner_configs_only_use_declared_and_read_parameters():
    for planner in MIGRATED_PLANNERS:
        declared, read_params = _planner_node_contract(planner)
        for track in TRACKS:
            config_path = SIM_CAR_SHARE / 'config' / track / f'{planner}_planner.yaml'
            config = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
            node_name = f'{planner}_planner_node'
            params = _flatten(config[node_name]['ros__parameters'])

            assert set(params).issubset(declared)
            assert set(params).issubset(read_params)


def test_track_specific_planner_configs_match_expected_sparse_overrides():
    expected = {
        ('acceleration', 'midpoint'): {
            'filtering.max_cone_range_m': 20.0,
            'stanley.lookahead_idx_offset': 4,
        },
        ('acceleration', 'single_boundary'): {},
        ('acceleration', 'corridor'): {},
        ('smalltrack', 'midpoint'): {},
        ('smalltrack', 'single_boundary'): {
            'filtering.max_cone_range_m': 20.0,
            'midline_memory.horizon_m': 20.0,
            'stanley.lookahead_idx_offset': 1,
            'pure_pursuit.lookahead_m': 2.0,
            'pure_pursuit.min_lookahead_m': 0.5,
            'pure_pursuit.max_lookahead_m': 10.0,
            'speed_control.speed_max_mps': 4.17,
        },
        ('smalltrack', 'corridor'): {},
        ('skidpad', 'midpoint'): {},
        ('skidpad', 'single_boundary'): {
            'filtering.max_cone_range_m': 14.0,
            'centerline.max_path_length_m': 14.0,
            'stanley.k_gain': 1.4,
            'stanley.heading_gain': 1.8,
            'speed_control.speed_max_mps': 4.17,
        },
        ('skidpad', 'corridor'): {},
    }

    for (track, planner), expected_flat in expected.items():
        config_path = SIM_CAR_SHARE / 'config' / track / f'{planner}_planner.yaml'
        config = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
        node_name = f'{planner}_planner_node'
        params = _flatten(config[node_name]['ros__parameters'])
        assert params == expected_flat


def test_track_specific_spawn_configs_define_pose():
    for track in TRACKS:
        config_path = SIM_CAR_SHARE / 'config' / track / 'spawn.yaml'
        config = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
        assert set(config['spawn']) == {'spawn_x', 'spawn_y', 'spawn_yaw'}


def test_removed_legacy_configs_are_absent():
    assert not (SIM_CAR_SHARE / 'config' / 'midpoint_planner.yaml').exists()
    assert not (SIM_CAR_SHARE / 'config' / 'single_boundary_planner.yaml').exists()
    assert not (SIM_CAR_SHARE / 'config' / 'corridor_planner.yaml').exists()
    assert not (SIM_CAR_SHARE / 'config' / 'controllers').exists()
