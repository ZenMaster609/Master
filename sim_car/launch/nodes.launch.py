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
from launch_ros.actions import Node


def generate_launch_description():
    sensor_config = _load_sensor_config()
    wheel_rate = _get_config_value(sensor_config, ['sensors', 'real', 'encoders', 'frequency_hz'], 50.0)
    suspension_rate = _get_config_value(sensor_config, ['sensors', 'real', 'suspension', 'frequency_hz'], 50.0)
    steering_rate = _get_config_value(sensor_config, ['sensors', 'virtual', 'steering_angle', 'frequency_hz'], 50.0)
    virtual_rates = {
        'water_pressure': _get_config_value(
            sensor_config, ['sensors', 'virtual', 'water_pressure', 'frequency_hz'], 50.0
        ),
        'water_flow': _get_config_value(
            sensor_config, ['sensors', 'virtual', 'water_flow', 'frequency_hz'], 50.0
        ),
        'water_temp_in': _get_config_value(
            sensor_config, ['sensors', 'virtual', 'water_temp_in', 'frequency_hz'], 50.0
        ),
        'water_temp_out': _get_config_value(
            sensor_config, ['sensors', 'virtual', 'water_temp_out', 'frequency_hz'], 50.0
        ),
        'water_temp_radiator': _get_config_value(
            sensor_config, ['sensors', 'virtual', 'water_temp_radiator', 'frequency_hz'], 50.0
        ),
        'brake_temp_fr': _get_config_value(
            sensor_config, ['sensors', 'virtual', 'brake_temp_fr', 'frequency_hz'], 50.0
        ),
        'brake_temp_rl': _get_config_value(
            sensor_config, ['sensors', 'virtual', 'brake_temp_rl', 'frequency_hz'], 50.0
        ),
        'pitot_dynamic_pressure': _get_config_value(
            sensor_config, ['sensors', 'virtual', 'pitot_dynamic_pressure', 'frequency_hz'], 50.0
        ),
    }

    # Wheel encoder node
    wheel_encoder_node = Node(
        package='sim_car',
        executable='wheel_encoder_node',
        name='wheel_encoder_node',
        output='screen',
        parameters=[{
            'publish_rate': wheel_rate,
            'wheel_radius': 0.23,
        }],
    )

    # Suspension sensor node
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
        }],
    )

    # Steering angle sensor node
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
        }],
    )

    # Virtual sensors nodes (cooling, brakes, pitot)
    water_pressure_node = Node(
        package='sim_car',
        executable='water_pressure_node',
        name='water_pressure_node',
        output='screen',
        parameters=[{
            'publish_rate': virtual_rates['water_pressure'],
            'ambient_temp': 25.0,
            'noise_pressure': 0.02,  # bar
        }],
    )

    water_flow_node = Node(
        package='sim_car',
        executable='water_flow_node',
        name='water_flow_node',
        output='screen',
        parameters=[{
            'publish_rate': virtual_rates['water_flow'],
            'ambient_temp': 25.0,
            'noise_flow': 0.5,       # L/min
        }],
    )

    water_temp_in_node = Node(
        package='sim_car',
        executable='water_temp_in_node',
        name='water_temp_in_node',
        output='screen',
        parameters=[{
            'publish_rate': virtual_rates['water_temp_in'],
            'ambient_temp': 25.0,
            'noise_temp': 0.3,       # Celsius
        }],
    )

    water_temp_out_node = Node(
        package='sim_car',
        executable='water_temp_out_node',
        name='water_temp_out_node',
        output='screen',
        parameters=[{
            'publish_rate': virtual_rates['water_temp_out'],
            'ambient_temp': 25.0,
            'noise_temp': 0.3,       # Celsius
        }],
    )

    water_temp_radiator_node = Node(
        package='sim_car',
        executable='water_temp_radiator_node',
        name='water_temp_radiator_node',
        output='screen',
        parameters=[{
            'publish_rate': virtual_rates['water_temp_radiator'],
            'ambient_temp': 25.0,
            'noise_temp': 0.3,       # Celsius
        }],
    )

    brake_temp_fr_node = Node(
        package='sim_car',
        executable='brake_temp_fr_node',
        name='brake_temp_fr_node',
        output='screen',
        parameters=[{
            'publish_rate': virtual_rates['brake_temp_fr'],
            'ambient_temp': 25.0,
            'noise_brake_temp': 1.0, # Celsius
        }],
    )

    brake_temp_rl_node = Node(
        package='sim_car',
        executable='brake_temp_rl_node',
        name='brake_temp_rl_node',
        output='screen',
        parameters=[{
            'publish_rate': virtual_rates['brake_temp_rl'],
            'ambient_temp': 25.0,
            'noise_brake_temp': 1.0, # Celsius
        }],
    )

    pitot_dynamic_pressure_node = Node(
        package='sim_car',
        executable='pitot_dynamic_pressure_node',
        name='pitot_dynamic_pressure_node',
        output='screen',
        parameters=[{
            'publish_rate': virtual_rates['pitot_dynamic_pressure'],
            'ambient_temp': 25.0,
            'noise_pitot': 2.0,      # Pa
        }],
    )

    return LaunchDescription([
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
    ])


def _load_sensor_config():
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
