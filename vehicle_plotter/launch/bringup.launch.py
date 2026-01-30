#!/usr/bin/env python3
"""
Full system bringup launch file.

Launches:
1. Gazebo simulation with car (from sim_car)
2. Control and sensor nodes (from sim_car)
3. Vehicle plotter nodes (data collector, plotter, logger)

Usage:
    # Launch sim + sensors + plotter
    ros2 launch vehicle_plotter bringup.launch.py

    # Headless (no Gazebo GUI, but plotter still shows):
    ros2 launch vehicle_plotter bringup.launch.py headless:=true

    # No plotting:
    ros2 launch vehicle_plotter bringup.launch.py enable_plot:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Find package shares
    sim_car_share = FindPackageShare('sim_car')
    vehicle_plotter_share = FindPackageShare('vehicle_plotter')

    # Note: session_manager is included via plotter.launch.py

    # Declare arguments

    # Gazebo options
    # Default to headless in Docker (set headless:=false if you have X11 working)
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo headless (no GUI)'
    )

    # Plotter options
    enable_plot_arg = DeclareLaunchArgument(
        'enable_plot',
        default_value='true',
        description='Enable live plotting'
    )

    enable_log_arg = DeclareLaunchArgument(
        'enable_log',
        default_value='true',
        description='Enable data logging'
    )

    log_format_arg = DeclareLaunchArgument(
        'log_format',
        default_value='parquet',
        description='Log format: parquet or csv'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from /clock topic'
    )

    # Include Gazebo simulation launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'gazebo_sim.launch.py'])
        ),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
        }.items(),
    )

    # Include sim_car nodes launch
    sim_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'nodes.launch.py'])
        ),
    )

    # Include vehicle_plotter launch
    plotter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([vehicle_plotter_share, 'launch', 'plotter.launch.py'])
        ),
        launch_arguments={
            'adapter': 'gazebo',
            'enable_plot': LaunchConfiguration('enable_plot'),
            'enable_log': LaunchConfiguration('enable_log'),
            'log_format': LaunchConfiguration('log_format'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    return LaunchDescription([
        # Arguments
        headless_arg,
        enable_plot_arg,
        enable_log_arg,
        log_format_arg,
        use_sim_time_arg,

        # Included launches (plotter.launch.py includes session_manager)
        gazebo_launch,
        sim_nodes_launch,
        plotter_launch,
    ])
