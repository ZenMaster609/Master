#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([FindPackageShare('sim_car'), 'config', 'pair_midpoint_planner.yaml']),
        description='Path to pair midpoint planner parameter file',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock',
    )

    node = Node(
        package='sim_car',
        executable='pair_midpoint_planner_node',
        name='pair_midpoint_planner_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'),
                    value_type=bool,
                ),
            },
        ],
    )

    return LaunchDescription([
        params_file_arg,
        use_sim_time_arg,
        node,
    ])
