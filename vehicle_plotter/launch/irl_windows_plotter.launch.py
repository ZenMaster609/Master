#!/usr/bin/env python3
"""
IRL Windows Plotter Launch - Windows subscriber/plotter for remote data.

Usage:
    ros2 launch vehicle_plotter irl_windows_plotter.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enable_log_arg = DeclareLaunchArgument('enable_log', default_value='true')
    enable_rosbag_arg = DeclareLaunchArgument('enable_rosbag', default_value='true')
    enable_plot_arg = DeclareLaunchArgument('enable_plot', default_value='true')
    save_plots_arg = DeclareLaunchArgument('save_plots_on_exit', default_value='true')
    save_plot_data_arg = DeclareLaunchArgument('save_plot_data_on_exit', default_value='true')
    update_rate_arg = DeclareLaunchArgument('update_rate_hz', default_value='30.0')

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
            'update_rate_hz': LaunchConfiguration('update_rate_hz'),
            'window_title': 'Vehicle Plotter (Windows)',
            'dark_mode': True,
            'save_plots_on_exit': LaunchConfiguration('save_plots_on_exit'),
            'save_plot_data_on_exit': LaunchConfiguration('save_plot_data_on_exit'),
            'wait_for_session': True,
            'session_timeout_sec': 10.0,
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_plot')),
    )

    logger = Node(
        package='vehicle_plotter',
        executable='logger_node',
        name='logger',
        parameters=[{'format': 'parquet', 'compression': 'snappy', 'wait_for_session': True, 'session_timeout_sec': 10.0}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_log')),
    )

    rosbag_controller = Node(
        package='vehicle_plotter',
        executable='rosbag_controller_node',
        name='rosbag_controller',
        parameters=[{'mode': 'windows_plotter', 'compression': 'zstd', 'wait_for_session': True, 'session_timeout_sec': 10.0}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_rosbag')),
    )

    return LaunchDescription([
        enable_log_arg, enable_rosbag_arg, enable_plot_arg, save_plots_arg, save_plot_data_arg, update_rate_arg,
        data_collector, plotter, logger, rosbag_controller,
    ])
