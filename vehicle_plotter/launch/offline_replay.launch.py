#!/usr/bin/env python3
"""
Offline replay launch file.

Replays a rosbag and optionally runs the logger for offline artifacts.

Usage:
    ros2 launch vehicle_plotter offline_replay.launch.py bag_path:=/path/to/rosbag

    # With specific log path:
    ros2 launch vehicle_plotter offline_replay.launch.py \
        bag_path:=/path/to/rosbag \
        log_path:=/path/to/output
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
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

    # Logger node (optional, for re-processing)
    logger_node = Node(
        package='vehicle_plotter',
        executable='logger_node',
        name='logger',
        output='screen',
        parameters=[{
            'format': 'parquet',
            'base_path': LaunchConfiguration('log_path'),
            'session_name': 'replay',
            'use_sim_time': True,
        }],
        condition=IfCondition(LaunchConfiguration('enable_log')),
    )

    return LaunchDescription([
        # Arguments
        bag_path_arg,
        rate_arg,
        loop_arg,
        enable_log_arg,
        log_path_arg,

        # Nodes
        rosbag_play,
        logger_node,
    ])
