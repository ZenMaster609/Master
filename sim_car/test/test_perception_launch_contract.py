from __future__ import annotations

import ast
import importlib.util
import math
import pathlib
import sys

import pytest
import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'full_sim_launch.launch.py'
SIM_CAR_SHARE = REPO_ROOT / 'sim_car'
TRACKED_CONE_PLANNER_BASE = SIM_CAR_SHARE / 'sim_car' / 'planning' / 'tracked_cone_planner_base.py'
SHARED_CONTROLLER_CONFIG = SIM_CAR_SHARE / 'sim_car' / 'planning' / 'controller_config.py'
TRACKED_CONE_PLANNER_CONTRACT = SIM_CAR_SHARE / 'sim_car' / 'planning' / 'tracked_cone_planner_contract.py'
TRACKS = {
    'acceleration': 'acceleration.world',
    'skidpad': 'skidpad.world',
    'smalltrack': 'small_track.world',
}
MIGRATED_PLANNERS = ('midpoint', 'single_boundary', 'corridor')
CONFIGURED_PLANNERS = MIGRATED_PLANNERS + ('linetest',)
CONTROLLERS = ('stanley', 'pure_pursuit')

if str(SIM_CAR_SHARE) not in sys.path:
    sys.path.insert(0, str(SIM_CAR_SHARE))

spec = importlib.util.spec_from_file_location('full_sim_launch', FULL_LAUNCH)
assert spec is not None
assert spec.loader is not None
full_sim_launch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = full_sim_launch
spec.loader.exec_module(full_sim_launch)

contract_spec = importlib.util.spec_from_file_location(
    'tracked_cone_planner_contract',
    TRACKED_CONE_PLANNER_CONTRACT,
)
assert contract_spec is not None
assert contract_spec.loader is not None
tracked_cone_planner_contract = importlib.util.module_from_spec(contract_spec)
sys.modules[contract_spec.name] = tracked_cone_planner_contract
contract_spec.loader.exec_module(tracked_cone_planner_contract)


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
    def _literal_eval_parameter_default(node: ast.AST) -> object:
        if isinstance(node, ast.Dict):
            return {
                _literal_eval_parameter_default(key): _literal_eval_parameter_default(value)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.UnaryOp):
            value = _literal_eval_parameter_default(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp):
            left = _literal_eval_parameter_default(node.left)
            right = _literal_eval_parameter_default(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == 'math'
            and node.attr == 'pi'
        ):
            return math.pi
        return ast.literal_eval(node)

    def _parameter_reads(module: ast.AST) -> set[str]:
        read_parameters: set[str] = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get_parameter':
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    read_parameters.add(node.args[0].value)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {'_read_float_parameter', '_read_int_parameter', '_read_bool_parameter'}:
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        if isinstance(node.args[1].value, str):
                            read_parameters.add(node.args[1].value)
                if node.func.id == '_read_param':
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        if isinstance(node.args[1].value, str):
                            read_parameters.add(node.args[1].value)
        return read_parameters

    def _merge_declared_defaults(module: ast.AST, planner_name: str) -> dict[str, object]:
        defaults = (
            dict(tracked_cone_planner_contract.PUBLIC_TRACKED_CONE_PLANNER_DEFAULTS)
            if planner_name in MIGRATED_PLANNERS
            else {}
        )
        found_defaults_assignment = False

        for node in ast.walk(module):
            if not isinstance(node, ast.FunctionDef) or node.name != '_declare_parameters':
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if not isinstance(target, ast.Name) or target.id != 'defaults':
                            continue
                        if isinstance(stmt.value, ast.Dict):
                            defaults.update(_literal_eval_parameter_default(stmt.value))
                            found_defaults_assignment = True
                            break
                        if planner_name not in MIGRATED_PLANNERS:
                            defaults.update(_literal_eval_parameter_default(stmt.value))
                            found_defaults_assignment = True
                            break
                    if found_defaults_assignment and planner_name not in MIGRATED_PLANNERS:
                        break
                if (
                    planner_name in MIGRATED_PLANNERS
                    and isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and isinstance(stmt.value.func.value, ast.Name)
                    and stmt.value.func.value.id == 'defaults'
                    and stmt.value.func.attr == 'update'
                    and stmt.value.args
                ):
                    defaults.update(_literal_eval_parameter_default(stmt.value.args[0]))
                    found_defaults_assignment = True
            break

        if planner_name not in MIGRATED_PLANNERS:
            assert found_defaults_assignment, planner_name
        return defaults

    planner_path = SIM_CAR_SHARE / 'sim_car' / 'planning' / f'{planner}_planner_node.py'
    module = ast.parse(planner_path.read_text(encoding='utf-8'))

    declared_defaults = _merge_declared_defaults(module, planner)
    read_parameters = _parameter_reads(module)
    if planner in MIGRATED_PLANNERS:
        read_parameters |= _parameter_reads(ast.parse(TRACKED_CONE_PLANNER_BASE.read_text(encoding='utf-8')))
        read_parameters |= _parameter_reads(ast.parse(TRACKED_CONE_PLANNER_CONTRACT.read_text(encoding='utf-8')))
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


def test_full_launch_always_enables_path_tracking_eval():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "logger_node = Node(" in content
    assert "DeclareLaunchArgument(\n        'path_tracking_eval'" not in content
    assert "LaunchConfiguration('path_tracking_eval')" not in content
    assert "'path_tracking_eval_enabled': True" in content
    assert "'path_tracking_eval_gt_track_topic': '/ground_truth/track'" in content
    assert "'path_tracking_eval_odom_topic': '/sim/odom'" in content
    assert "'path_tracking_eval_planner_path_topic': '/planned_centerline'" in content
    assert "'path_tracking_eval_track_name': LaunchConfiguration('track')" in content
    assert "shutdown_on_logger_exit = RegisterEventHandler(" in content
    assert "logger autostop reached lap target" in content


def test_full_launch_passes_lap_tracking_to_all_smalltrack_capable_planners():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    corridor_block = content.split("corridor_planner_node = Node(", 1)[1].split("linetest_planner_node = Node(", 1)[0]
    linetest_block = content.split("linetest_planner_node = Node(", 1)[1].split("camera_debug_viewer_node = Node(", 1)[0]

    assert "'lap_tracking.target_laps': resolved_path_tracking_autostop_laps" in corridor_block
    assert "'topics.tracked_cones_topic': planner_input_topic" in linetest_block
    assert "'lap_tracking.target_laps': resolved_path_tracking_autostop_laps" in linetest_block


def test_full_launch_always_wires_offline_cone_eval_topics():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'camera_cone_rmse_plotting'" not in content
    assert "DeclareLaunchArgument(\n        'lidar_cone_rmse_plotting'" not in content
    assert "DeclareLaunchArgument(\n        'cone_rmse_plotting'" not in content
    assert "'camera_cone_eval_topic': PythonExpression([" in content
    assert "'enable_log': 'false'" in content
    assert "'source_name': camera_source_name" in content
    assert "'source_name': 'lidar'" in content


def test_full_launch_supports_only_migrated_planners():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "Planner to launch: 'midpoint', 'single_boundary', 'corridor', 'linetest', or 'none'" in content
    assert "from sim_car.planning.planner_constants import (" in content
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
    assert "'routing.shutdown_on_route_complete': False" in content
    assert "'routing.shutdown_on_parking_complete': ParameterValue(" in content
    assert "'.lower() in ('skidpad', 'acceleration')" in content
    assert "shutdown_on_skidpad_router_exit = RegisterEventHandler(" in content
    assert "skidpad/acceleration router mission target reached" in content
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
        'smalltrack': ('-13.6758', '10.3753', '0'),
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
            'speed_min_mps': 6.17,
            'speed_max_mps': 6.17,
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
            'speed_max_mps': 7.0,
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


def test_track_bundle_resolves_smalltrack_linetest_config_path():
    selection = full_sim_launch._resolve_launch_selection(
        SIM_CAR_SHARE,
        track='smalltrack',
        planner='linetest',
    )

    assert selection.planner_config == str(SIM_CAR_SHARE / 'config' / 'smalltrack' / 'linetest.yaml')
    assert selection.controller_config == str(SIM_CAR_SHARE / 'config' / 'smalltrack' / 'stanley_controller.yaml')
    assert selection.spawn_config == str(SIM_CAR_SHARE / 'config' / 'smalltrack' / 'spawn.yaml')


def test_track_bundle_rejects_linetest_for_skidpad():
    try:
        full_sim_launch._resolve_launch_selection(
            SIM_CAR_SHARE,
            track='skidpad',
            planner='linetest',
        )
    except RuntimeError as exc:
        assert "planner='linetest' is only supported with track='acceleration, smalltrack'" in str(exc)
    else:
        raise AssertionError('linetest unexpectedly allowed for track=skidpad')


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
        'controller',
        'speed_max_mps',
        'lidar_pipeline',
        'stereo_enabled',
        'camera_range_m',
        'expected_prefix',
    ),
    [
        (
            'smalltrack', 'midpoint', 'stanley', 7.0, 'scan2d', 'false', '20.0',
            'mono_20_small_7_mid_stan_2d',
        ),
        (
            'smalltrack', 'linetest', 'stanley', 7.0, 'scan2d', 'true', '10',
            'stereo_10_small_7_line_stan_2d',
        ),
        (
            'acceleration', 'single_boundary', 'pure_pursuit', 6.17, 'scan2d', 'false', '7.5',
            'mono_7p5_acc_6_SB_pp_2d',
        ),
        (
            'skidpad', 'corridor', 'stanley', 7.0, 'pointcloud3d', 'true', '0.0',
            'stereo_0_skid_7_cor_stan_3d',
        ),
    ],
)
def test_run_id_prefix_uses_abbreviated_launch_selection(
    track: str,
    planner: str,
    controller: str,
    speed_max_mps: float,
    lidar_pipeline: str,
    stereo_enabled: str,
    camera_range_m: str,
    expected_prefix: str,
):
    assert (
        full_sim_launch._abbreviated_run_id_prefix(
            track,
            planner,
            controller,
            speed_max_mps,
            lidar_pipeline,
            stereo_enabled,
            camera_range_m,
        )
        == expected_prefix
    )


def test_full_launch_passes_resolved_run_id_prefix_to_plotter_launch():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "resolved_run_id_prefix = LaunchConfiguration('resolved_run_id_prefix')" in content
    assert "'run_id_prefix': resolved_run_id_prefix" in content
    assert "SetLaunchConfiguration('resolved_run_id_prefix', run_id_prefix)" in content


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
        ('smalltrack', 'linetest', '', 'stanley', '/linetest_planner/diagnostics', 'linetest', False),
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
            'stanley.heading_gain': 0.0,
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
        expected_keys = set(params)
        assert expected_keys.issubset(declared)
        assert expected_keys.issubset(read_params)


def test_migrated_planners_expose_restored_low_level_algorithm_parameters():
    expected_shared = {
        'filtering.max_cone_range_m',
        'filtering.min_confidence',
        'boundary_chain.min_step_m',
        'boundary_chain.max_step_m',
        'width_estimation.initial_width_m',
        'centerline.max_path_length_m',
        'validation.hold_last_valid_s',
        'midline_memory.horizon_m',
        'debug.show_pair_lines',
    }
    for planner in MIGRATED_PLANNERS:
        declared, read_params = _planner_node_contract(planner)
        assert expected_shared.issubset(declared)
        assert expected_shared.issubset(read_params)


def test_cone_memory_config_uses_namespaced_public_parameters():
    config = _load_yaml(SIM_CAR_SHARE / 'config' / 'cone_memory.yaml')
    params = _flatten(config['cone_memory_node']['ros__parameters'])
    allowed_prefixes = (
        'frames.',
        'vehicle.',
        'memory.',
        'fusion.',
        'runtime.',
        'debug.',
        'permanent_memory.',
        'topics.',
    )

    assert all(name.startswith(allowed_prefixes) for name in params)
    assert 'memory.confirm_hits' in params
    assert 'memory.min_seen_count' in params


def test_smalltrack_spawn_config_keeps_lap_tracking():
    config_path = SIM_CAR_SHARE / 'config' / 'smalltrack' / 'spawn.yaml'
    config = _load_yaml(config_path)
    assert config['lap_tracking'] == {'auto_suspend_after_laps': 10}


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


@pytest.mark.parametrize(
    'config_path',
    [
        SIM_CAR_SHARE / 'config' / 'acceleration' / 'linetest.yaml',
        SIM_CAR_SHARE / 'config' / 'smalltrack' / 'linetest.yaml',
    ],
)
def test_linetest_config_only_uses_declared_and_read_parameters(config_path: pathlib.Path):
    declared, read_params = _planner_node_contract('linetest')
    config = _load_yaml(config_path)
    params = _flatten(config['linetest_planner_node']['ros__parameters'])

    assert set(params).issubset(declared)
    assert set(params).issubset(read_params)


def test_linetest_config_stops_before_acceleration_parking_line():
    config_path = SIM_CAR_SHARE / 'config' / 'acceleration' / 'linetest.yaml'
    config = _load_yaml(config_path)
    line = config['linetest_planner_node']['ros__parameters']['line']

    assert line['end_x_m'] == 46.5


@pytest.mark.parametrize(
    'config_path',
    [
        SIM_CAR_SHARE / 'config' / 'acceleration' / 'linetest.yaml',
        SIM_CAR_SHARE / 'config' / 'smalltrack' / 'linetest.yaml',
    ],
)
def test_linetest_config_no_longer_contains_controller_or_speed_control_blocks(config_path: pathlib.Path):
    config = _load_yaml(config_path)
    params = config['linetest_planner_node']['ros__parameters']

    assert 'control' not in params
    assert 'stanley' not in params
    assert 'pure_pursuit' not in params
    assert 'speed_control' not in params
