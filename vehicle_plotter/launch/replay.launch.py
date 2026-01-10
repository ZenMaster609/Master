#!/usr/bin/env python3
"""Wrapper for modes/replay.launch.py"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('bag_path', description='Path to rosbag'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('adapter', default_value='can'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('vehicle_plotter'),
                    'launch', 'modes', 'replay.launch.py'
                ])
            ]),
            launch_arguments={
                'bag_path': LaunchConfiguration('bag_path'),
                'rate': LaunchConfiguration('rate'),
                'adapter': LaunchConfiguration('adapter'),
            }.items()
        )
    ])
