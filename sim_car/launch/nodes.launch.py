#!/usr/bin/env python3

"""
Launch file for control and sensor processing nodes.

Includes cmd_vel control, wheel encoder, suspension sensor,
steering sensor, and virtual sensor nodes.
"""

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'topic_prefix',
            default_value='/sim/raw',
            description='Topic prefix for sim sensors (/sim or /sim/raw)',
        ),
        DeclareLaunchArgument(
            'sensor_config',
            default_value=PathJoinSubstitution([FindPackageShare('sim_car'), 'config', 'sensor_config.yaml']),
            description='Path to sensor config YAML for sim_car sensor node enable/rates',
        ),
        OpaqueFunction(function=_build_sensor_nodes),
    ])


def _build_sensor_nodes(context):
    sensor_config = _load_sensor_config(LaunchConfiguration('sensor_config').perform(context))
    topic_prefix = LaunchConfiguration('topic_prefix')
    wheel_rate = _get_signal_rate(sensor_config, '/sim/wheel_encoder/rpm', 50.0)
    suspension_rate = _get_signal_rate(sensor_config, '/sim/suspension', 50.0)
    steering_rate = _get_signal_rate(sensor_config, '/sim/steering_angle', 50.0)
    virtual_rates = {
        'water_pressure': _get_signal_rate(sensor_config, '/sim/cooling/water_pressure', 50.0),
        'water_flow': _get_signal_rate(sensor_config, '/sim/cooling/water_flow', 50.0),
        'water_temp_in': _get_signal_rate(sensor_config, '/sim/cooling/water_temp_in', 50.0),
        'water_temp_out': _get_signal_rate(sensor_config, '/sim/cooling/water_temp_out', 50.0),
        'water_temp_radiator': _get_signal_rate(sensor_config, '/sim/cooling/water_temp_radiator', 50.0),
        'brake_temp_fr': _get_signal_rate(sensor_config, '/sim/brakes/temp_fr', 50.0),
        'brake_temp_rl': _get_signal_rate(sensor_config, '/sim/brakes/temp_rl', 50.0),
        'pitot_dynamic_pressure': _get_signal_rate(sensor_config, '/sim/pitot/dynamic_pressure', 50.0),
    }

    wheel_encoder_enabled = _any_signal_enabled(
        sensor_config,
        topics=[
            '/sim/raw/wheel_encoder/rpm', '/sim/wheel_encoder/rpm',
            '/sim/raw/wheel_encoder/angle_accum', '/sim/wheel_encoder/angle_accum',
            '/sim/raw/wheel_encoder/speed_mm_s', '/sim/wheel_encoder/speed_mm_s',
        ],
    )
    suspension_enabled = _any_signal_enabled(
        sensor_config,
        topics=['/sim/raw/suspension', '/sim/suspension'],
    )
    steering_enabled = _any_signal_enabled(
        sensor_config,
        topics=['/sim/raw/steering_angle', '/sim/steering_angle'],
    )

    water_pressure_enabled = _any_signal_enabled(
        sensor_config,
        topics=['/sim/raw/cooling/water_pressure', '/sim/cooling/water_pressure'],
    )
    water_flow_enabled = _any_signal_enabled(
        sensor_config,
        topics=['/sim/raw/cooling/water_flow', '/sim/cooling/water_flow'],
    )
    water_temp_in_enabled = _any_signal_enabled(
        sensor_config,
        topics=['/sim/raw/cooling/water_temp_in', '/sim/cooling/water_temp_in'],
    )
    water_temp_out_enabled = _any_signal_enabled(
        sensor_config,
        topics=['/sim/raw/cooling/water_temp_out', '/sim/cooling/water_temp_out'],
    )
    water_temp_radiator_enabled = _any_signal_enabled(
        sensor_config,
        topics=['/sim/raw/cooling/water_temp_radiator', '/sim/cooling/water_temp_radiator'],
    )
    brake_temp_fr_enabled = _any_signal_enabled(
        sensor_config,
        topics=['/sim/raw/brakes/temp_fr', '/sim/brakes/temp_fr'],
    )
    brake_temp_rl_enabled = _any_signal_enabled(
        sensor_config,
        topics=['/sim/raw/brakes/temp_rl', '/sim/brakes/temp_rl'],
    )
    pitot_enabled = _any_signal_enabled(
        sensor_config,
        topics=['/sim/raw/pitot/dynamic_pressure', '/sim/pitot/dynamic_pressure'],
    )

    # Wheel encoder node
    wheel_encoder_node = None
    if wheel_encoder_enabled:
        wheel_encoder_node = Node(
            package='sim_car',
            executable='wheel_encoder_node',
            name='wheel_encoder_node',
            output='screen',
            parameters=[{
                'publish_rate': wheel_rate,
                'wheel_radius': 0.23,
                'topic_prefix': topic_prefix,
            }],
        )

    # Suspension sensor node
    suspension_sensor_node = None
    if suspension_enabled:
        suspension_sensor_node = Node(
            package='sim_car',
            executable='suspension_sensor_node',
            name='suspension_sensor_node',
            output='screen',
            parameters=[{
                'mode': 'synthetic',
                'noise_stddev': 0.5,     # mm
                'bias_fl': 0.0,
                'bias_fr': 0.0,
                'bias_rl': 0.0,
                'bias_rr': 0.0,
                'publish_rate': suspension_rate,
                'dropout_probability': 0.0,
                'static_mm': 20.0,
                'pitch_gain': 4.0,       # mm per m/s^2
                'roll_gain': 3.0,        # mm per m/s^2
                'filter_tau_sec': 0.0,   # 0 disables low-pass filter
                'topic_prefix': topic_prefix,
            }],
        )

    # Steering angle sensor node
    steering_sensor_node = None
    if steering_enabled:
        steering_sensor_node = Node(
            package='sim_car',
            executable='steering_sensor_node',
            name='steering_sensor_node',
            output='screen',
            parameters=[{
                'noise_stddev': 0.1,     # degrees
                'latency_ms': 5.0,
                'bias': 0.0,
                'publish_rate': steering_rate,
                'dropout_probability': 0.0,
                'topic_prefix': topic_prefix,
            }],
        )

    # Virtual sensors nodes (cooling, brakes, pitot)
    water_pressure_node = None
    if water_pressure_enabled:
        water_pressure_node = Node(
            package='sim_car',
            executable='water_pressure_node',
            name='water_pressure_node',
            output='screen',
            parameters=[{
                'publish_rate': virtual_rates['water_pressure'],
                'ambient_temp': 25.0,
                'noise_pressure': 0.02,  # bar
                'topic_prefix': topic_prefix,
            }],
        )

    water_flow_node = None
    if water_flow_enabled:
        water_flow_node = Node(
            package='sim_car',
            executable='water_flow_node',
            name='water_flow_node',
            output='screen',
            parameters=[{
                'publish_rate': virtual_rates['water_flow'],
                'ambient_temp': 25.0,
                'noise_flow': 0.5,       # L/min
                'topic_prefix': topic_prefix,
            }],
        )

    water_temp_in_node = None
    if water_temp_in_enabled:
        water_temp_in_node = Node(
            package='sim_car',
            executable='water_temp_in_node',
            name='water_temp_in_node',
            output='screen',
            parameters=[{
                'publish_rate': virtual_rates['water_temp_in'],
                'ambient_temp': 25.0,
                'noise_temp': 0.3,       # Celsius
                'topic_prefix': topic_prefix,
            }],
        )

    water_temp_out_node = None
    if water_temp_out_enabled:
        water_temp_out_node = Node(
            package='sim_car',
            executable='water_temp_out_node',
            name='water_temp_out_node',
            output='screen',
            parameters=[{
                'publish_rate': virtual_rates['water_temp_out'],
                'ambient_temp': 25.0,
                'noise_temp': 0.3,       # Celsius
                'topic_prefix': topic_prefix,
            }],
        )

    water_temp_radiator_node = None
    if water_temp_radiator_enabled:
        water_temp_radiator_node = Node(
            package='sim_car',
            executable='water_temp_radiator_node',
            name='water_temp_radiator_node',
            output='screen',
            parameters=[{
                'publish_rate': virtual_rates['water_temp_radiator'],
                'ambient_temp': 25.0,
                'noise_temp': 0.3,       # Celsius
                'topic_prefix': topic_prefix,
            }],
        )

    brake_temp_fr_node = None
    if brake_temp_fr_enabled:
        brake_temp_fr_node = Node(
            package='sim_car',
            executable='brake_temp_fr_node',
            name='brake_temp_fr_node',
            output='screen',
            parameters=[{
                'publish_rate': virtual_rates['brake_temp_fr'],
                'ambient_temp': 25.0,
                'noise_brake_temp': 1.0, # Celsius
                'topic_prefix': topic_prefix,
            }],
        )

    brake_temp_rl_node = None
    if brake_temp_rl_enabled:
        brake_temp_rl_node = Node(
            package='sim_car',
            executable='brake_temp_rl_node',
            name='brake_temp_rl_node',
            output='screen',
            parameters=[{
                'publish_rate': virtual_rates['brake_temp_rl'],
                'ambient_temp': 25.0,
                'noise_brake_temp': 1.0, # Celsius
                'topic_prefix': topic_prefix,
            }],
        )

    pitot_dynamic_pressure_node = None
    if pitot_enabled:
        pitot_dynamic_pressure_node = Node(
            package='sim_car',
            executable='pitot_dynamic_pressure_node',
            name='pitot_dynamic_pressure_node',
            output='screen',
            parameters=[{
                'publish_rate': virtual_rates['pitot_dynamic_pressure'],
                'ambient_temp': 25.0,
                'noise_pitot': 2.0,      # Pa
                'topic_prefix': topic_prefix,
            }],
        )

    nodes = [
        wheel_encoder_node,
        suspension_sensor_node,
        steering_sensor_node,
        water_pressure_node,
        water_flow_node,
        water_temp_in_node,
        water_temp_out_node,
        water_temp_radiator_node,
        brake_temp_fr_node,
        brake_temp_rl_node,
        pitot_dynamic_pressure_node,
    ]
    return [node for node in nodes if node is not None]


def _load_sensor_config(config_path=None):
    if not config_path:
        config_path = os.path.join(
            get_package_share_directory('sim_car'),
            'config',
            'sensor_config.yaml',
        )
    try:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError):
        return {}



def _get_config_value(config, keys, default):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value if value is not None else default


def _get_signal_rate(config, output_topic, default):
    signals = config.get('signals')
    if not isinstance(signals, dict):
        return default
    for signal in signals.values():
        if not isinstance(signal, dict):
            continue
        if signal.get('plot_only', False):
            continue
        if signal.get('output_topic') != output_topic:
            continue
        rate = signal.get('rate_hz')
        if rate is None:
            return default
        try:
            return float(rate)
        except (TypeError, ValueError):
            return default
    return default


def _any_signal_enabled(config, topics, default=True):
    signals = config.get('signals')
    if not isinstance(signals, dict):
        return default
    found = False
    for signal in signals.values():
        if not isinstance(signal, dict):
            continue
        if signal.get('plot_only', False):
            continue
        input_topic = signal.get('input_topic')
        output_topic = signal.get('output_topic')
        if input_topic in topics or output_topic in topics:
            found = True
            if bool(signal.get('enabled', True)):
                return True
    if not found:
        return default
    return False
