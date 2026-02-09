#!/usr/bin/env python3
"""
Launch file for measurement pipeline:
- Gazebo sim
- sim_car sensor nodes (publish /sim/raw/*)
- measurement_node (publishes /sim/*)
- data_collector_node (vehicle_plotter)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_car_share = FindPackageShare('sim_car')
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo headless (no GUI)'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from /clock topic'
    )

    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value=PathJoinSubstitution([sim_car_share, 'config', 'sensor_config.yaml']),
        description='Measurement config YAML path'
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'gazebo_sim.launch.py'])
        ),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
        }.items(),
    )

    sim_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'nodes.launch.py'])
        ),
    )

    measurement_node = Node(
        package='measurement_node',
        executable='measurement_node',
        name='measurement_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'config_path': LaunchConfiguration('config_path'),
        }],
    )

    data_collector_node = Node(
        package='vehicle_plotter',
        executable='data_collector_node',
        name='data_collector',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'adapter': 'gazebo',
        }],
    )

    return LaunchDescription([
        headless_arg,
        use_sim_time_arg,
        config_path_arg,
        gazebo_launch,
        sim_nodes_launch,
        measurement_node,
        data_collector_node,
    ])
