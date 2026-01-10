#!/usr/bin/env python3
"""Wrapper for modes/sim.launch.py - Full Gazebo simulation"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='true'),
        DeclareLaunchArgument('control_mode', default_value='auto'),
        DeclareLaunchArgument('enable_plot', default_value='true'),
        DeclareLaunchArgument('enable_log', default_value='true'),
        DeclareLaunchArgument('enable_rosbag', default_value='true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('vehicle_plotter'),
                    'launch', 'modes', 'sim.launch.py'
                ])
            ]),
            launch_arguments={
                'headless': LaunchConfiguration('headless'),
                'control_mode': LaunchConfiguration('control_mode'),
                'enable_plot': LaunchConfiguration('enable_plot'),
                'enable_log': LaunchConfiguration('enable_log'),
                'enable_rosbag': LaunchConfiguration('enable_rosbag'),
            }.items()
        )
    ])
