#!/usr/bin/env python3

"""
Launch file for control and sensor processing nodes.

Includes cmd_vel control, wheel encoder, suspension sensor,
steering sensor, and virtual sensors nodes.
"""

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
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
    virtual_update_rate = max(virtual_rates.values()) if virtual_rates else 50.0

    # Declare launch arguments
    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='auto',
        description='Control mode: auto, keyboard, or none (skip control_node for manual keyboard in separate terminal)'
    )

    linear_speed_arg = DeclareLaunchArgument(
        'linear_speed',
        default_value='3.0',
        description='Default linear speed for auto mode (m/s)'
    )

    angular_speed_arg = DeclareLaunchArgument(
        'angular_speed',
        default_value='0.5',
        description='Default angular speed for auto mode (rad/s)'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='1.0',
        description='Sensor status publish rate (Hz)'
    )

    enable_real_sensors_arg = DeclareLaunchArgument(
        'enable_real_sensors',
        default_value='true',
        description='Enable real sensor nodes (wheel/suspension/steering)'
    )

    enable_virtual_sensors_arg = DeclareLaunchArgument(
        'enable_virtual_sensors',
        default_value='true',
        description='Enable virtual sensor nodes (cooling/brakes/pitot)'
    )

    # Get launch configurations
    control_mode = LaunchConfiguration('control_mode')
    linear_speed = LaunchConfiguration('linear_speed')
    angular_speed = LaunchConfiguration('angular_speed')
    publish_rate = LaunchConfiguration('publish_rate')
    enable_real_sensors = LaunchConfiguration('enable_real_sensors')
    enable_virtual_sensors = LaunchConfiguration('enable_virtual_sensors')

    # Control node (handles both keyboard and auto modes)
    # Publishes /cmd_vel based on configured mode
    # Skipped when control_mode='none' to allow manual keyboard control in separate terminal
    control_node = Node(
        package='sim_car',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{
            'mode': control_mode,
            'linear_speed': linear_speed,
            'angular_speed': angular_speed,
        }],
        condition=IfCondition(PythonExpression(["'", control_mode, "' != 'none'"]))
    )

    # Sensor processor node
    sensor_processor_node = Node(
        package='sim_car',
        executable='sensor_processor',
        name='sensor_processor',
        output='screen',
        parameters=[{
            'publish_rate': publish_rate
        }],
        condition=IfCondition(enable_real_sensors)
    )

    # Wheel encoder node
    wheel_encoder_node = Node(
        package='sim_car',
        executable='wheel_encoder_node',
        name='wheel_encoder_node',
        output='screen',
        parameters=[{
            'wheel_radius': 0.23,   # Match URDF
            'publish_rate': wheel_rate,
        }],
        condition=IfCondition(enable_real_sensors)
    )

    # Suspension sensor node
    suspension_sensor_node = Node(
        package='sim_car',
        executable='suspension_sensor_node',
        name='suspension_sensor_node',
        output='screen',
        parameters=[{
            'noise_stddev': 0.5,     # mm
            'bias_fl': 0.0,
            'bias_fr': 0.0,
            'bias_rl': 0.0,
            'bias_rr': 0.0,
            'publish_rate': suspension_rate,
            'dropout_probability': 0.0,
        }],
        condition=IfCondition(enable_real_sensors)
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
        condition=IfCondition(enable_real_sensors)
    )

    # Virtual sensors node (cooling, brakes, pitot)
    virtual_sensors_node = Node(
        package='sim_car',
        executable='virtual_sensors_node',
        name='virtual_sensors_node',
        output='screen',
        parameters=[{
            'publish_rate': virtual_update_rate,
            'publish_rate_water_pressure': virtual_rates['water_pressure'],
            'publish_rate_water_flow': virtual_rates['water_flow'],
            'publish_rate_water_temp_in': virtual_rates['water_temp_in'],
            'publish_rate_water_temp_out': virtual_rates['water_temp_out'],
            'publish_rate_water_temp_radiator': virtual_rates['water_temp_radiator'],
            'publish_rate_brake_temp_fr': virtual_rates['brake_temp_fr'],
            'publish_rate_brake_temp_rl': virtual_rates['brake_temp_rl'],
            'publish_rate_pitot_dynamic_pressure': virtual_rates['pitot_dynamic_pressure'],
            'ambient_temp': 25.0,
            'noise_pressure': 0.02,  # bar
            'noise_flow': 0.5,       # L/min
            'noise_temp': 0.3,       # Celsius
            'noise_brake_temp': 1.0, # Celsius
            'noise_pitot': 2.0,      # Pa
        }],
        condition=IfCondition(enable_virtual_sensors)
    )

    return LaunchDescription([
        control_mode_arg,
        linear_speed_arg,
        angular_speed_arg,
        publish_rate_arg,
        enable_real_sensors_arg,
        enable_virtual_sensors_arg,
        control_node,
        sensor_processor_node,
        wheel_encoder_node,
        suspension_sensor_node,
        steering_sensor_node,
        virtual_sensors_node,
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
