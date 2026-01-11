#!/usr/bin/env python3
"""
Replay Mode Launch - Playback rosbag with plotting.

Usage:
    ros2 launch vehicle_plotter replay.launch.py bag_path:=/path/to/bag
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bag_path_arg = DeclareLaunchArgument('bag_path', description='Path to rosbag')
    rate_arg = DeclareLaunchArgument('rate', default_value='1.0')
    enable_plot_arg = DeclareLaunchArgument('enable_plot', default_value='true')
    enable_log_arg = DeclareLaunchArgument('enable_log', default_value='false')
    adapter_arg = DeclareLaunchArgument('adapter', default_value='can')

    bag_player = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_path'),
             '--rate', LaunchConfiguration('rate'), '--clock'],
        output='screen',
    )

    session_manager = Node(
        package='vehicle_plotter',
        executable='session_manager_node',
        name='session_manager',
        parameters=[{'broadcast_rate_hz': 1.0}],
        output='screen',
    )

    data_collector = Node(
        package='vehicle_plotter',
        executable='data_collector_node',
        name='data_collector',
        parameters=[{'adapter': LaunchConfiguration('adapter'), 'output_rate_hz': 50.0, 'use_sim_time': True}],
        output='screen',
    )

    plotter = Node(
        package='vehicle_plotter',
        executable='plotter_node',
        name='plotter',
        parameters=[{
            'backend': 'pyqtgraph',
            'update_rate_hz': 30.0,
            'window_title': 'Vehicle Plotter (Replay)',
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

    return LaunchDescription([
        bag_path_arg, rate_arg, enable_plot_arg, enable_log_arg, adapter_arg,
        session_manager, bag_player, data_collector, plotter, logger,
    ])
