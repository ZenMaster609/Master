#!/usr/bin/env python3
"""
Full sim bringup for the EUFS car with sensors and live plotting.

Launches:
1. Gazebo sim with the EUFS car in small_track.world (sim_car)
2. Control + sensor nodes (sim_car)
3. Vehicle plotter nodes (vehicle_plotter)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_car_share = FindPackageShare('sim_car')
    vehicle_plotter_share = FindPackageShare('vehicle_plotter')

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo headless (no GUI)'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([sim_car_share, 'worlds', 'small_track.world']),
        description='Full path to world file to load'
    )

    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='auto',
        description='Control mode: auto, keyboard, or none'
    )

    linear_speed_arg = DeclareLaunchArgument(
        'linear_speed',
        default_value='0.5',
        description='Linear speed in m/s (for auto mode)'
    )

    angular_speed_arg = DeclareLaunchArgument(
        'angular_speed',
        default_value='1.0',
        description='Angular speed in rad/s (for auto mode)'
    )

    sensor_mode_arg = DeclareLaunchArgument(
        'sensor_mode',
        default_value='real',
        description='Sensor mode: real, virtual, or both'
    )

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

    enable_rosbag_arg = DeclareLaunchArgument(
        'enable_rosbag',
        default_value='true',
        description='Enable rosbag recording'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from /clock topic'
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'gazebo_sim.launch.py'])
        ),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'world': LaunchConfiguration('world'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    sensor_mode = LaunchConfiguration('sensor_mode')
    enable_real_sensors = PythonExpression([
        "'true' if '", sensor_mode, "' in ['real', 'both'] else 'false'"
    ])
    enable_virtual_sensors = PythonExpression([
        "'true' if '", sensor_mode, "' in ['virtual', 'both'] else 'false'"
    ])

    sim_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'nodes.launch.py'])
        ),
        launch_arguments={
            'control_mode': LaunchConfiguration('control_mode'),
            'linear_speed': LaunchConfiguration('linear_speed'),
            'angular_speed': LaunchConfiguration('angular_speed'),
            'enable_real_sensors': enable_real_sensors,
            'enable_virtual_sensors': enable_virtual_sensors,
        }.items(),
    )

    plotter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([vehicle_plotter_share, 'launch', 'plotter.launch.py'])
        ),
        launch_arguments={
            'adapter': 'gazebo',
            'enable_plot': LaunchConfiguration('enable_plot'),
            'enable_real_plot': enable_real_sensors,
            'enable_virtual_plot': enable_virtual_sensors,
            'enable_log': LaunchConfiguration('enable_log'),
            'enable_rosbag': LaunchConfiguration('enable_rosbag'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    return LaunchDescription([
        headless_arg,
        world_arg,
        control_mode_arg,
        linear_speed_arg,
        angular_speed_arg,
        sensor_mode_arg,
        enable_plot_arg,
        enable_log_arg,
        enable_rosbag_arg,
        use_sim_time_arg,
        gazebo_launch,
        sim_nodes_launch,
        plotter_launch,
    ])
