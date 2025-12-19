#!/usr/bin/env python3

"""
Launch file for control and sensor processing nodes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # Declare launch arguments
    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='keyboard',
        description='Control mode: keyboard or auto'
    )

    linear_speed_arg = DeclareLaunchArgument(
        'linear_speed',
        default_value='0.5',
        description='Default linear speed (m/s)'
    )

    angular_speed_arg = DeclareLaunchArgument(
        'angular_speed',
        default_value='1.0',
        description='Default angular speed (rad/s)'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='1.0',
        description='Sensor status publish rate (Hz)'
    )

    ticks_per_rev_arg = DeclareLaunchArgument(
        'ticks_per_revolution',
        default_value='2048',
        description='Wheel encoder ticks per revolution'
    )

    # Get launch configurations
    control_mode = LaunchConfiguration('control_mode')
    linear_speed = LaunchConfiguration('linear_speed')
    angular_speed = LaunchConfiguration('angular_speed')
    publish_rate = LaunchConfiguration('publish_rate')
    ticks_per_rev = LaunchConfiguration('ticks_per_revolution')

    # Control node
    control_node = Node(
        package='sim_car',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{
            'mode': control_mode,
            'linear_speed': linear_speed,
            'angular_speed': angular_speed
        }]
    )

    # Sensor processor node
    sensor_processor_node = Node(
        package='sim_car',
        executable='sensor_processor',
        name='sensor_processor',
        output='screen',
        parameters=[{
            'publish_rate': publish_rate
        }]
    )

    # Wheel encoder node
    wheel_encoder_node = Node(
        package='sim_car',
        executable='wheel_encoder_node',
        name='wheel_encoder_node',
        output='screen',
        parameters=[{
            'ticks_per_revolution': ticks_per_rev,
            'wheel_radius': 0.1,
            'publish_rate': 50.0
        }]
    )

    return LaunchDescription([
        control_mode_arg,
        linear_speed_arg,
        angular_speed_arg,
        publish_rate_arg,
        ticks_per_rev_arg,
        control_node,
        sensor_processor_node,
        wheel_encoder_node,
    ])
