#!/usr/bin/env python3
"""
Minimal launch: Gazebo + /clock bridge only (no robot, no sensors).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    sim_car_share = get_package_share_directory('sim_car')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    resource_path = sim_car_share
    resource_paths = [
        resource_path,
        os.path.join(resource_path, 'models'),
        os.path.join(resource_path, 'meshes'),
        os.path.join(resource_path, 'materials'),
    ]
    resource_path_value = ':'.join(resource_paths)

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([sim_car_share, 'worlds', 'small_track.world']),
        description='World file to load'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='true',
        description='Run simulation without GUI (server only)'
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s ', LaunchConfiguration('world')]}.items(),
        condition=IfCondition(LaunchConfiguration('headless'))
    )

    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', LaunchConfiguration('world')]}.items(),
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path_value),
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', resource_path_value),
        world_arg,
        headless_arg,
        gazebo,
        gazebo_gui,
        clock_bridge,
    ])
