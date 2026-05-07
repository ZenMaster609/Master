from __future__ import annotations

import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'full_sim_launch.launch.py'


def test_sensor_pipeline_enables_vehicle_state_logging():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "'enable_state_logging': LaunchConfiguration('sensor_pipeline')" in content
    assert "LaunchConfiguration('sensor_pipeline'),\n                value_type=bool" in content
    assert "LaunchConfiguration('logging')" not in content
    assert "DeclareLaunchArgument(\n        'logging'" not in content


def test_sensor_pipeline_enables_sensor_nodes_without_sensor_nodes_arg():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'sensor_nodes'" not in content
    assert "LaunchConfiguration('sensor_nodes')" not in content
    assert "condition=IfCondition(LaunchConfiguration('sensor_pipeline'))" in content


def test_rviz_argument_is_the_only_rviz_launch_switch():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'use_rviz'" not in content
    assert "LaunchConfiguration('use_rviz')" not in content
    assert "LaunchConfiguration('rviz'), \"'.lower() == 'true' and '\"" in content


def test_full_launch_uses_fixed_bridge_and_sim_time_defaults():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    for removed_arg in ('bridge', 'ackermann_steering_sign', 'use_sim_time'):
        assert f"DeclareLaunchArgument(\n        '{removed_arg}'" not in content
        assert f"LaunchConfiguration('{removed_arg}')" not in content

    assert "DEFAULT_ACKERMANN_STEERING_SIGN = 1.0" in content
    assert "DEFAULT_USE_SIM_TIME = True" in content
    assert "DEFAULT_USE_SIM_TIME_LAUNCH = 'true'" in content
    assert "executable='ackermann_cmd_bridge'" in content
    assert "DEFAULT_ACKERMANN_STEERING_SIGN,\n                value_type=float" in content
    assert "'use_sim_time': DEFAULT_USE_SIM_TIME" in content
    assert "'use_sim_time': ParameterValue(\n                DEFAULT_USE_SIM_TIME" in content
    assert "'use_sim_time': DEFAULT_USE_SIM_TIME_LAUNCH" in content
