#!/usr/bin/env python3
"""
IRL Jetson Source Launch - Jetson as data source for remote plotting.

Usage:
    ros2 launch vehicle_plotter irl_jetson_source.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LifecycleNode


def generate_launch_description():
    enable_can_arg = DeclareLaunchArgument('enable_can', default_value='true')
    enable_vectornav_arg = DeclareLaunchArgument('enable_vectornav', default_value='true')
    enable_log_arg = DeclareLaunchArgument('enable_log', default_value='true')
    enable_rosbag_arg = DeclareLaunchArgument('enable_rosbag', default_value='true')
    can_device_arg = DeclareLaunchArgument('can_device', default_value='can0')
    serial_port_arg = DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB0')
    output_rate_arg = DeclareLaunchArgument('output_rate_hz', default_value='50.0')

    session_manager = Node(
        package='vehicle_plotter',
        executable='session_manager_node',
        name='session_manager',
        parameters=[{'broadcast_rate_hz': 1.0}],
        output='screen',
    )

    socketcan_receiver = LifecycleNode(
        package='ros2_socketcan',
        executable='socket_can_receiver_node_exe',
        name='socket_can_receiver',
        namespace='',
        parameters=[{'interface': LaunchConfiguration('can_device')}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_can')),
    )

    can_decoder = Node(
        package='canbus_decoder',
        executable='can_decoder_node',
        name='can_decoder',
        parameters=[{'publish_rate_hz': 50.0, 'show_stats': True, 'stats_interval': 10.0}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_can')),
    )

    vectornav_decoder = Node(
        package='vectornav_decoder',
        executable='vectornav_decoder_node',
        name='vectornav_decoder',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baudrate': 921600,
            'publish_rate_hz': 200.0,
            'show_stats': True,
            'stats_interval': 10.0,
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_vectornav')),
    )

    data_collector = Node(
        package='vehicle_plotter',
        executable='data_collector_node',
        name='data_collector',
        parameters=[{'adapter': 'can', 'output_rate_hz': LaunchConfiguration('output_rate_hz')}],
        output='screen',
    )

    logger = Node(
        package='vehicle_plotter',
        executable='logger_node',
        name='logger',
        parameters=[{'format': 'parquet', 'compression': 'snappy', 'wait_for_session': True, 'session_timeout_sec': 2.0}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_log')),
    )

    rosbag_controller = Node(
        package='vehicle_plotter',
        executable='rosbag_controller_node',
        name='rosbag_controller',
        parameters=[{'mode': 'jetson_hardware', 'compression': 'zstd', 'wait_for_session': True, 'session_timeout_sec': 2.0}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_rosbag')),
    )

    return LaunchDescription([
        enable_can_arg, enable_vectornav_arg, enable_log_arg, enable_rosbag_arg,
        can_device_arg, serial_port_arg, output_rate_arg,
        session_manager, socketcan_receiver, can_decoder, vectornav_decoder,
        data_collector, logger, rosbag_controller,
    ])
