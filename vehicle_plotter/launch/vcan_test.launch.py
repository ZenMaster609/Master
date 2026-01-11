#!/usr/bin/env python3
"""
Virtual CAN Test Launch - Test full pipeline with fake CAN data.

Usage:
    ros2 launch vehicle_plotter vcan_test.launch.py
    ros2 launch vehicle_plotter vcan_test.launch.py mode:=circle
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mode_arg = DeclareLaunchArgument('mode', default_value='circle')
    base_velocity_arg = DeclareLaunchArgument('base_velocity_mps', default_value='5.0')
    enable_plot_arg = DeclareLaunchArgument('enable_plot', default_value='true')
    enable_log_arg = DeclareLaunchArgument('enable_log', default_value='true')
    enable_rosbag_arg = DeclareLaunchArgument('enable_rosbag', default_value='true')

    session_manager = Node(
        package='vehicle_plotter',
        executable='session_manager_node',
        name='session_manager',
        parameters=[{'broadcast_rate_hz': 1.0}],
        output='screen',
    )

    vcan_publisher = Node(
        package='canbus_decoder',
        executable='vcan_publisher_node',
        name='vcan_publisher',
        parameters=[{
            'publish_rate_hz': 100.0,
            'mode': LaunchConfiguration('mode'),
            'base_velocity_mps': LaunchConfiguration('base_velocity_mps'),
        }],
        output='screen',
    )

    can_decoder = Node(
        package='canbus_decoder',
        executable='can_decoder_node',
        name='can_decoder',
        parameters=[{'publish_rate_hz': 50.0, 'show_stats': True, 'stats_interval': 5.0}],
        remappings=[('/can/rx', '/to_can_bus')],
        output='screen',
    )

    data_collector = Node(
        package='vehicle_plotter',
        executable='data_collector_node',
        name='data_collector',
        parameters=[{'adapter': 'can', 'output_rate_hz': 50.0}],
        output='screen',
    )

    plotter = Node(
        package='vehicle_plotter',
        executable='plotter_node',
        name='plotter',
        parameters=[{
            'backend': 'pyqtgraph',
            'update_rate_hz': 30.0,
            'window_title': 'Vehicle Plotter (VCAN Test)',
            'save_plots_on_exit': True,
            'save_plot_data_on_exit': True,
            'wait_for_session': True,
            'session_timeout_sec': 2.0,
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_plot')),
    )

    logger = Node(
        package='vehicle_plotter',
        executable='logger_node',
        name='logger',
        parameters=[{'format': 'parquet', 'wait_for_session': True, 'session_timeout_sec': 2.0}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_log')),
    )

    rosbag_controller = Node(
        package='vehicle_plotter',
        executable='rosbag_controller_node',
        name='rosbag_controller',
        parameters=[{'mode': 'vcan_test', 'compression': 'zstd', 'wait_for_session': True, 'session_timeout_sec': 2.0}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_rosbag')),
    )

    return LaunchDescription([
        mode_arg, base_velocity_arg, enable_plot_arg, enable_log_arg, enable_rosbag_arg,
        session_manager, vcan_publisher, can_decoder, data_collector, plotter, logger, rosbag_controller,
    ])
