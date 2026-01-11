#!/usr/bin/env python3
"""
Launch file for VectorNav VN-200 monitor (debugging).

Usage:
    ros2 launch vectornav_decoder vectornav_monitor.launch.py
    ros2 launch vectornav_decoder vectornav_monitor.launch.py verbose:=true
    ros2 launch vectornav_decoder vectornav_monitor.launch.py serial_port:=/dev/ttyUSB1
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get default config path
    pkg_dir = get_package_share_directory('vectornav_decoder')
    default_config = os.path.join(pkg_dir, 'config', 'default_output.yaml')

    # Arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to YAML configuration file'
    )

    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial port device'
    )

    baudrate_arg = DeclareLaunchArgument(
        'baudrate',
        default_value='921600',
        description='Serial baud rate'
    )

    verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='false',
        description='Show detailed field values for each packet'
    )

    stats_interval_arg = DeclareLaunchArgument(
        'stats_interval',
        default_value='5.0',
        description='Statistics interval in seconds'
    )

    # VectorNav monitor node
    vectornav_monitor = Node(
        package='vectornav_decoder',
        executable='vectornav_monitor_node',
        name='vectornav_monitor',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'serial_port': LaunchConfiguration('serial_port'),
            'baudrate': LaunchConfiguration('baudrate'),
            'verbose': LaunchConfiguration('verbose'),
            'stats_interval': LaunchConfiguration('stats_interval'),
        }],
    )

    return LaunchDescription([
        # Arguments
        config_file_arg,
        serial_port_arg,
        baudrate_arg,
        verbose_arg,
        stats_interval_arg,

        # Info
        LogInfo(msg=['Starting VectorNav monitor on: ', LaunchConfiguration('serial_port')]),

        # Node
        vectornav_monitor,
    ])
