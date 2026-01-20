#!/usr/bin/env python3
"""
Full system bringup launch file.

Launches:
1. Gazebo simulation with car (from sim_car)
2. Control and sensor nodes (from sim_car)
3. Vehicle plotter nodes (data collector, plotter, logger)

Usage:
    # Auto mode (car drives in circles) - default
    ros2 launch vehicle_plotter bringup.launch.py

    # For KEYBOARD control, run bringup WITHOUT control_node, then run control separately:
    ros2 launch vehicle_plotter bringup.launch.py control_mode:=none
    ros2 run sim_car control_node --ros-args -p mode:=keyboard  # in separate terminal

    # Headless (no Gazebo GUI, but plotter still shows):
    ros2 launch vehicle_plotter bringup.launch.py headless:=true

    # No plotting:
    ros2 launch vehicle_plotter bringup.launch.py enable_plot:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
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

    # Control options
    # Use 'auto' for autonomous driving, 'none' to run keyboard control separately
    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='auto',
        description='Control mode: auto (default), or none (for keyboard control in separate terminal)'
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

    enable_ackermann_arg = DeclareLaunchArgument(
        'enable_ackermann',
        default_value='true',
        description='Enable Ackermann steering control node'
    )

    sensor_mode_arg = DeclareLaunchArgument(
        'sensor_mode',
        default_value='real',
        description='Sensor mode: real, virtual, or both'
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

    sensor_mode = LaunchConfiguration('sensor_mode')
    enable_real_sensors = PythonExpression([
        "'true' if '", sensor_mode, "' in ['real', 'both'] else 'false'"
    ])
    enable_virtual_sensors = PythonExpression([
        "'true' if '", sensor_mode, "' in ['virtual', 'both'] else 'false'"
    ])

    # Include sim_car nodes launch
    sim_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'nodes.launch.py'])
        ),
        launch_arguments={
            'control_mode': LaunchConfiguration('control_mode'),
            'linear_speed': LaunchConfiguration('linear_speed'),
            'angular_speed': LaunchConfiguration('angular_speed'),
            'enable_ackermann': LaunchConfiguration('enable_ackermann'),
            'enable_real_sensors': enable_real_sensors,
            'enable_virtual_sensors': enable_virtual_sensors,
        }.items(),
    )

    # Include vehicle_plotter launch
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
            'log_format': LaunchConfiguration('log_format'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    return LaunchDescription([
        # Arguments
        headless_arg,
        control_mode_arg,
        linear_speed_arg,
        angular_speed_arg,
        enable_ackermann_arg,
        sensor_mode_arg,
        enable_plot_arg,
        enable_log_arg,
        log_format_arg,
        use_sim_time_arg,

        # Included launches (plotter.launch.py includes session_manager)
        gazebo_launch,
        sim_nodes_launch,
        plotter_launch,
    ])
