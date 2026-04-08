from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest
import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'full_sim_launch.launch.py'
SIM_CAR_SHARE = REPO_ROOT / 'sim_car'
TRACKED_CONE_PLANNER_BASE = SIM_CAR_SHARE / 'sim_car' / 'planning' / 'tracked_cone_planner_base.py'
SHARED_CONTROLLER_CONFIG = SIM_CAR_SHARE / 'sim_car' / 'planning' / 'controller_config.py'
TRACKS = {
    'acceleration': 'acceleration.world',
    'skidpad': 'skidpad.world',
    'smalltrack': 'small_track.world',
}
MIGRATED_PLANNERS = ('midpoint', 'single_boundary', 'corridor')
CONFIGURED_PLANNERS = MIGRATED_PLANNERS + ('linetest',)
CONTROLLERS = ('stanley', 'pure_pursuit')

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
    def _parameter_reads(module: ast.AST) -> set[str]:
        read_parameters: set[str] = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get_parameter':
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    read_parameters.add(node.args[0].value)
        return read_parameters

    planner_path = SIM_CAR_SHARE / 'sim_car' / 'planning' / f'{planner}_planner_node.py'
    module = ast.parse(planner_path.read_text(encoding='utf-8'))

    declared_defaults: dict[str, object] | None = None
    read_parameters = _parameter_reads(module)

    for node in ast.walk(module):
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
    if planner in MIGRATED_PLANNERS:
        read_parameters |= _parameter_reads(ast.parse(TRACKED_CONE_PLANNER_BASE.read_text(encoding='utf-8')))
    if planner in CONFIGURED_PLANNERS:
        read_parameters |= _parameter_reads(ast.parse(SHARED_CONTROLLER_CONFIG.read_text(encoding='utf-8')))
    return set(declared_defaults), read_parameters


def _load_yaml(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


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

    assert "Planner to launch: 'midpoint', 'single_boundary', 'corridor', 'linetest', or 'none'" in content
    assert "from sim_car.planning.planner_registry import (" in content
    assert "get_planner_spec('midpoint').executable" in content
    assert "get_planner_spec('single_boundary').executable" in content
    assert "get_planner_spec('corridor').executable" in content
    assert "get_planner_spec('linetest').executable" in content
    assert "_planner_selected_condition('midpoint')" in content
    assert "_planner_selected_condition('single_boundary')" in content
    assert "_planner_selected_condition('corridor')" in content
    assert "_planner_selected_condition('linetest')" in content


def test_full_launch_declares_track_arg_and_controller_selection():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'track'" in content
    assert "Track preset to load: 'acceleration', 'skidpad', or 'smalltrack'" in content
    assert "Controller to launch: 'stanley', 'pure_pursuit', or 'none'" in content
    assert "default_value='stanley'" in content
    assert "config', 'controllers'" not in content


def test_full_launch_supports_planner_specific_rviz_profiles():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "default_value='planner'" in content
    assert "'planner', 'clean', 'planner_debug', 'midpoint'," in content
    assert "'single_boundary', 'corridor', or 'linetest'" in content
    assert "get_planner_spec(planner).default_rviz_profile" in content
    assert "'linetest': 'planner_debug.rviz'" in content


def test_full_launch_uses_resolved_controller_and_speed_control_configs():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    midpoint_block = content.split("midpoint_planner_node = Node(", 1)[1].split("single_boundary_planner_node = Node(", 1)[0]
    single_boundary_block = content.split("single_boundary_planner_node = Node(", 1)[1].split("corridor_planner_node = Node(", 1)[0]
    corridor_block = content.split("corridor_planner_node = Node(", 1)[1].split("camera_debug_viewer_node = Node(", 1)[0]
    linetest_block = content.split("linetest_planner_node = Node(", 1)[1].split("camera_debug_viewer_node = Node(", 1)[0]

    for block in (midpoint_block, single_boundary_block, corridor_block, linetest_block):
        assert "resolved_controller_config" in block
        assert "resolved_speed_control_config" in block

    assert "config', 'midpoint_planner.yaml'" not in content
    assert "config', 'single_boundary_planner.yaml'" not in content
    assert "config', 'corridor_planner.yaml'" not in content


def test_full_launch_wires_skidpad_router_output_into_migrated_planners() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "skidpad_router_node = Node(" in content
    assert "PathJoinSubstitution([sim_car_share, 'config', 'skidpad', 'skidpad_router.yaml'])" in content
    assert "'topics.output_topic': '/tracked_cones/skidpad_routed'" in content
    assert "'topics.viz_topic': router_viz_topic" in content
    assert "'/tracked_cones/skidpad_routed' if '" in content
    assert "'.lower() in ('skidpad', 'acceleration') else ('/tracked_cones' if '" in content
    assert "'routing.event_mode': LaunchConfiguration('track')" in content
    assert "'.lower() in ('skidpad', 'acceleration') else '/skidpad_router/markers_hidden'" in content


def test_track_bundle_resolves_world_spawn_and_default_controller_config_paths():
    for track, world_name in TRACKS.items():
        for planner in MIGRATED_PLANNERS:
            selection = full_sim_launch._resolve_launch_selection(
                SIM_CAR_SHARE,
                track=track,
                planner=planner,
            )
            assert selection.world == str(SIM_CAR_SHARE / 'worlds' / world_name)
            assert selection.planner_config == ''
            assert selection.controller == 'stanley'
            assert selection.controller_config == str(SIM_CAR_SHARE / 'config' / track / 'stanley_controller.yaml')
            assert selection.spawn_config == str(SIM_CAR_SHARE / 'config' / track / 'spawn.yaml')


def test_track_bundle_loads_track_spawn_defaults():
    expected = {
        'acceleration': ('-38.5', '0.0', '0.0'),
        'skidpad': ('0.0', '-10.0', '1.5708'),
        'smalltrack': ('-13.6758', '10.4753', '0'),
    }

    for track, (spawn_x, spawn_y, spawn_yaw) in expected.items():
        selection = full_sim_launch._resolve_launch_selection(
            SIM_CAR_SHARE,
            track=track,
            planner='midpoint',
        )
        assert selection.spawn_x == spawn_x
        assert selection.spawn_y == spawn_y
        assert selection.spawn_yaw == spawn_yaw


def test_track_bundle_loads_track_speed_control_defaults():
    expected = {
        'acceleration': {
            'speed_min_mps': 4.17,
            'speed_max_mps': 4.17,
            'curvature_speed_gain': 4.0,
            'lowpass_speed_alpha': 0.15,
        },
        'skidpad': {
            'speed_min_mps': 1.0,
            'speed_max_mps': 4.17,
            'curvature_speed_gain': 4.0,
            'lowpass_speed_alpha': 0.15,
        },
        'smalltrack': {
            'speed_min_mps': 1.0,
            'speed_max_mps': 2.0,
            'curvature_speed_gain': 4.0,
            'lowpass_speed_alpha': 0.15,
        },
    }

    for track, speed_control in expected.items():
        selection = full_sim_launch._resolve_launch_selection(
            SIM_CAR_SHARE,
            track=track,
            planner='midpoint',
        )
        assert selection.speed_control == speed_control


def test_track_bundle_loads_optional_track_planner_limit_defaults():
    expected = {
        'acceleration': {},
        'smalltrack': {},
        'skidpad': {
            'filtering.max_cone_range_m': 14.0,
            'centerline.max_path_length_m': 14.0,
            'filtering.planning_horizon_m': 14.0,
        },
    }

    for track, planner_limits in expected.items():
        selection = full_sim_launch._resolve_launch_selection(
            SIM_CAR_SHARE,
            track=track,
            planner='midpoint',
        )
        assert selection.planner_limits == planner_limits


def test_track_bundle_resolves_acceleration_linetest_and_default_controller_config_paths():
    selection = full_sim_launch._resolve_launch_selection(
        SIM_CAR_SHARE,
        track='acceleration',
        planner='linetest',
    )

    assert selection.planner_config == str(SIM_CAR_SHARE / 'config' / 'acceleration' / 'linetest.yaml')
    assert selection.controller_config == str(SIM_CAR_SHARE / 'config' / 'acceleration' / 'stanley_controller.yaml')
    assert selection.spawn_config == str(SIM_CAR_SHARE / 'config' / 'acceleration' / 'spawn.yaml')


def test_track_bundle_rejects_linetest_for_non_acceleration_tracks():
    for track in ('skidpad', 'smalltrack'):
        try:
            full_sim_launch._resolve_launch_selection(
                SIM_CAR_SHARE,
                track=track,
                planner='linetest',
            )
        except RuntimeError as exc:
            assert "planner='linetest' is only supported with track='acceleration'" in str(exc)
        else:
            raise AssertionError(f'linetest unexpectedly allowed for track={track}')


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

    assert selection.world == '/tmp/custom.world'
    assert selection.spawn_x == '1.0'
    assert selection.spawn_y == '2.0'
    assert selection.spawn_yaw == '3.0'
    assert selection.controller == 'pure_pursuit'
    assert selection.controller_config == str(SIM_CAR_SHARE / 'config' / 'smalltrack' / 'pure_pursuit_controller.yaml')


def test_track_bundle_supports_controller_none_without_controller_config_file():
    selection = full_sim_launch._resolve_launch_selection(
        SIM_CAR_SHARE,
        track='smalltrack',
        planner='corridor',
        controller_override='none',
    )

    assert selection.controller == 'none'
    assert selection.controller_config == ''


@pytest.mark.parametrize(
    (
        'track',
        'planner',
        'controller_override',
        'expected_controller',
        'expected_diag_topic',
        'expected_rviz_profile',
        'has_planner_limits',
    ),
    [
        ('smalltrack', 'midpoint', '', 'stanley', '/midpoint_planner/diagnostics', 'midpoint', False),
        ('skidpad', 'corridor', 'none', 'none', '/corridor_planner/diagnostics', 'corridor', True),
        ('acceleration', 'linetest', 'pure_pursuit', 'pure_pursuit', '/linetest_planner/diagnostics', 'linetest', False),
    ],
)
def test_launch_selection_matrix_uses_registry_backed_metadata(
    track: str,
    planner: str,
    controller_override: str,
    expected_controller: str,
    expected_diag_topic: str,
    expected_rviz_profile: str,
    has_planner_limits: bool,
):
    selection = full_sim_launch._resolve_launch_selection(
        SIM_CAR_SHARE,
        track=track,
        planner=planner,
        controller_override=controller_override,
    )

    assert selection.controller == expected_controller
    assert selection.planner_diagnostics_topic == expected_diag_topic
    assert selection.planner_default_rviz_profile == expected_rviz_profile
    assert bool(selection.planner_overlay_parameters()) is has_planner_limits


def test_launch_selection_controller_none_uses_overlay_without_file_path():
    selection = full_sim_launch._resolve_launch_selection(
        SIM_CAR_SHARE,
        track='smalltrack',
        planner='corridor',
        controller_override='none',
    )

    assert selection.controller_config == ''
    assert selection.controller_overlay_parameters() == {
        '/**': {
            'ros__parameters': {
                'control': {
                    'controller_type': 'none',
                },
            },
        },
    }


def test_controller_configs_only_use_declared_and_read_parameters():
    for track in TRACKS:
        for controller in CONTROLLERS:
            config_path = SIM_CAR_SHARE / 'config' / track / f'{controller}_controller.yaml'
            config = _load_yaml(config_path)
            params = _flatten(config['/**']['ros__parameters'])

            for planner in CONFIGURED_PLANNERS:
                declared, read_params = _planner_node_contract(planner)
                assert set(params).issubset(declared)
                assert set(params).issubset(read_params)


def test_controller_configs_match_expected_sparse_overrides():
    expected = {
        ('acceleration', 'stanley'): {
            'control.controller_type': 'stanley',
            'stanley.k_gain': 1.2,
            'stanley.heading_gain': 1.6,
            'stanley.lookahead_idx_offset': 4,
        },
        ('acceleration', 'pure_pursuit'): {
            'control.controller_type': 'pure_pursuit',
            'pure_pursuit.lookahead_m': 3.0,
            'pure_pursuit.min_lookahead_m': 1.5,
            'pure_pursuit.max_lookahead_m': 8.0,
        },
        ('skidpad', 'stanley'): {
            'control.controller_type': 'stanley',
            'stanley.k_gain': 1.4,
            'stanley.heading_gain': 1.8,
            'stanley.lookahead_idx_offset': 0,
        },
        ('skidpad', 'pure_pursuit'): {
            'control.controller_type': 'pure_pursuit',
            'pure_pursuit.lookahead_m': 3.0,
            'pure_pursuit.min_lookahead_m': 1.5,
            'pure_pursuit.max_lookahead_m': 8.0,
        },
        ('smalltrack', 'stanley'): {
            'control.controller_type': 'stanley',
            'stanley.k_gain': 1.2,
            'stanley.heading_gain': 1.6,
            'stanley.lookahead_idx_offset': 1,
        },
        ('smalltrack', 'pure_pursuit'): {
            'control.controller_type': 'pure_pursuit',
            'pure_pursuit.lookahead_m': 2.0,
            'pure_pursuit.min_lookahead_m': 0.5,
            'pure_pursuit.max_lookahead_m': 10.0,
        },
    }

    for (track, controller), expected_flat in expected.items():
        config_path = SIM_CAR_SHARE / 'config' / track / f'{controller}_controller.yaml'
        config = _load_yaml(config_path)
        params = _flatten(config['/**']['ros__parameters'])
        assert params == expected_flat


def test_track_specific_spawn_configs_define_pose_and_speed_control():
    for track in TRACKS:
        config_path = SIM_CAR_SHARE / 'config' / track / 'spawn.yaml'
        config = _load_yaml(config_path)
        assert set(config['spawn']) == {'spawn_x', 'spawn_y', 'spawn_yaw'}
        assert set(config['speed_control']) == {
            'speed_min_mps',
            'speed_max_mps',
            'curvature_speed_gain',
            'lowpass_speed_alpha',
        }


def test_skidpad_spawn_config_defines_max_planner_length():
    config_path = SIM_CAR_SHARE / 'config' / 'skidpad' / 'spawn.yaml'
    config = _load_yaml(config_path)
    assert config['planner_limits'] == {'max_planner_length_m': 14.0}


def test_speed_control_spawn_configs_only_use_declared_and_read_parameters():
    for track in TRACKS:
        config_path = SIM_CAR_SHARE / 'config' / track / 'spawn.yaml'
        config = _load_yaml(config_path)
        params = _flatten({'speed_control': config['speed_control']})

        for planner in CONFIGURED_PLANNERS:
            declared, read_params = _planner_node_contract(planner)
            assert set(params).issubset(declared)
            assert set(params).issubset(read_params)


def test_planner_limit_spawn_configs_only_use_declared_and_read_parameters():
    config_path = SIM_CAR_SHARE / 'config' / 'skidpad' / 'spawn.yaml'
    config = _load_yaml(config_path)
    params = {
        'filtering.max_cone_range_m': config['planner_limits']['max_planner_length_m'],
        'centerline.max_path_length_m': config['planner_limits']['max_planner_length_m'],
        'filtering.planning_horizon_m': config['planner_limits']['max_planner_length_m'],
    }

    for planner in MIGRATED_PLANNERS:
        declared, read_params = _planner_node_contract(planner)
        if planner == 'corridor':
            expected_keys = set(params)
        else:
            expected_keys = {
                'filtering.max_cone_range_m',
                'centerline.max_path_length_m',
            }
        assert expected_keys.issubset(declared)
        assert expected_keys.issubset(read_params)


def test_smalltrack_spawn_config_keeps_lap_tracking():
    config_path = SIM_CAR_SHARE / 'config' / 'smalltrack' / 'spawn.yaml'
    config = _load_yaml(config_path)
    assert config['lap_tracking'] == {'auto_suspend_after_laps': 1}


def test_removed_legacy_configs_are_absent_and_controller_configs_exist():
    assert not (SIM_CAR_SHARE / 'config' / 'midpoint_planner.yaml').exists()
    assert not (SIM_CAR_SHARE / 'config' / 'single_boundary_planner.yaml').exists()
    assert not (SIM_CAR_SHARE / 'config' / 'corridor_planner.yaml').exists()
    assert not (SIM_CAR_SHARE / 'config' / 'controllers').exists()

    for track in TRACKS:
        for planner in MIGRATED_PLANNERS:
            assert not (SIM_CAR_SHARE / 'config' / track / f'{planner}_planner.yaml').exists()
        for controller in CONTROLLERS:
            assert (SIM_CAR_SHARE / 'config' / track / f'{controller}_controller.yaml').exists()


def test_linetest_config_only_uses_declared_and_read_parameters():
    declared, read_params = _planner_node_contract('linetest')
    config_path = SIM_CAR_SHARE / 'config' / 'acceleration' / 'linetest.yaml'
    config = _load_yaml(config_path)
    params = _flatten(config['linetest_planner_node']['ros__parameters'])

    assert set(params).issubset(declared)
    assert set(params).issubset(read_params)


def test_linetest_config_no_longer_contains_controller_or_speed_control_blocks():
    config_path = SIM_CAR_SHARE / 'config' / 'acceleration' / 'linetest.yaml'
    config = _load_yaml(config_path)
    params = config['linetest_planner_node']['ros__parameters']

    assert 'control' not in params
    assert 'stanley' not in params
    assert 'pure_pursuit' not in params
    assert 'speed_control' not in params
