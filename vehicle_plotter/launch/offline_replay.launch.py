#!/usr/bin/env python3
"""
Offline replay launch file.

Replays a rosbag and runs the plotter for visualization.

Usage:
    ros2 launch vehicle_plotter offline_replay.launch.py bag_path:=/path/to/rosbag

    # With specific log path:
    ros2 launch vehicle_plotter offline_replay.launch.py \
        bag_path:=/path/to/rosbag \
        log_path:=/path/to/output
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        description='Path to rosbag directory for replay'
    )

    rate_arg = DeclareLaunchArgument(
        'rate',
        default_value='1.0',
        description='Playback rate multiplier'
    )

    loop_arg = DeclareLaunchArgument(
        'loop',
        default_value='false',
        description='Loop playback'
    )

    enable_plot_arg = DeclareLaunchArgument(
        'enable_plot',
        default_value='true',
        description='Enable plotting'
    )

    enable_log_arg = DeclareLaunchArgument(
        'enable_log',
        default_value='false',
        description='Enable logging (creates new log from replay)'
    )

    log_path_arg = DeclareLaunchArgument(
        'log_path',
        default_value='~/.ros/vehicle_logs',
        description='Path for output logs'
    )

    # Rosbag play command
    rosbag_play = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play',
            LaunchConfiguration('bag_path'),
            '--rate', LaunchConfiguration('rate'),
            '--clock',  # Publish clock for sim time
        ],
        output='screen',
    )

    # Plotter node
    plotter_node = Node(
        package='vehicle_plotter',
        executable='plotter_node',
        name='plotter',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_plot')),
        parameters=[{
            'update_rate_hz': 30.0,
            'direct_from_sensors': False,
            'use_sim_time': True,
        }],
    )

    # Logger node (optional, for re-processing)
    logger_node = Node(
        package='vehicle_plotter',
        executable='logger_node',
        name='logger',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_log')),
        parameters=[{
            'format': 'parquet',
            'base_path': LaunchConfiguration('log_path'),
            'session_name': 'replay',
            'use_sim_time': True,
        }],
    )

    return LaunchDescription([
        # Arguments
        bag_path_arg,
        rate_arg,
        loop_arg,
        enable_plot_arg,
        enable_log_arg,
        log_path_arg,

        # Nodes
        rosbag_play,
        plotter_node,
        logger_node,
    ])
