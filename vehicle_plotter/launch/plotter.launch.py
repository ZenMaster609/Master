#!/usr/bin/env python3
"""
Launch file for vehicle_plotter nodes.

Launches data collector, plotter, and logger nodes with configurable options.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments

    # Adapter configuration
    adapter_arg = DeclareLaunchArgument(
        'adapter',
        default_value='gazebo',
        description='Sensor adapter type: gazebo or vectornav'
    )

    # Output rate
    output_rate_arg = DeclareLaunchArgument(
        'output_rate_hz',
        default_value='50.0',
        description='Data output rate in Hz'
    )

    # Plotting options
    enable_plot_arg = DeclareLaunchArgument(
        'enable_plot',
        default_value='true',
        description='Enable real-time plotting GUI'
    )

    plot_rate_arg = DeclareLaunchArgument(
        'plot_rate_hz',
        default_value='30.0',
        description='Plot refresh rate in Hz'
    )

    dark_mode_arg = DeclareLaunchArgument(
        'dark_mode',
        default_value='true',
        description='Use dark theme for plots'
    )

    # Logging options
    enable_log_arg = DeclareLaunchArgument(
        'enable_log',
        default_value='true',
        description='Enable data logging to file'
    )

    log_format_arg = DeclareLaunchArgument(
        'log_format',
        default_value='parquet',
        description='Log format: parquet or csv'
    )

    log_path_arg = DeclareLaunchArgument(
        'log_path',
        default_value='/home/developer/ros2_ws/src/logs',
        description='Base path for log files (maps to E:/master/logs on Windows)'
    )

    # GPS origin (0,0 = auto-detect from first GPS message)
    gps_origin_lat_arg = DeclareLaunchArgument(
        'gps_origin_lat',
        default_value='0.0',
        description='GPS origin latitude (0 = auto-detect)'
    )

    gps_origin_lon_arg = DeclareLaunchArgument(
        'gps_origin_lon',
        default_value='0.0',
        description='GPS origin longitude (0 = auto-detect)'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from /clock topic'
    )

    # Data collector node (always runs)
    # Note: We use wall clock for timers but sensor timestamps for sync,
    # so use_sim_time is set to false to ensure timers fire reliably
    data_collector_node = Node(
        package='vehicle_plotter',
        executable='data_collector_node',
        name='data_collector',
        output='screen',
        parameters=[{
            'adapter': LaunchConfiguration('adapter'),
            'output_rate_hz': LaunchConfiguration('output_rate_hz'),
            'gps_origin_lat': LaunchConfiguration('gps_origin_lat'),
            'gps_origin_lon': LaunchConfiguration('gps_origin_lon'),
            'use_sim_time': False,  # Use wall clock for timers
        }],
    )

    # Plotter node (conditional)
    # Use wall clock for refresh timer
    plotter_node = Node(
        package='vehicle_plotter',
        executable='plotter_node',
        name='plotter',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_plot')),
        parameters=[{
            'backend': 'pyqtgraph',
            'update_rate_hz': LaunchConfiguration('plot_rate_hz'),
            'dark_mode': LaunchConfiguration('dark_mode'),
            'enable_gui': True,
            'use_sim_time': False,  # Use wall clock for timers
        }],
    )

    # Logger node (conditional)
    # Use wall clock for flush timer
    logger_node = Node(
        package='vehicle_plotter',
        executable='logger_node',
        name='logger',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_log')),
        parameters=[{
            'format': LaunchConfiguration('log_format'),
            'base_path': LaunchConfiguration('log_path'),
            'flush_interval_sec': 5.0,
            'buffer_size': 1000,
            'adapter': LaunchConfiguration('adapter'),  # For directory prefix (sim_ or jetson_)
            'auto_plot_on_shutdown': True,  # Generate plots when logger shuts down
            'use_sim_time': False,  # Use wall clock for timers
        }],
    )

    return LaunchDescription([
        # Arguments
        adapter_arg,
        output_rate_arg,
        enable_plot_arg,
        plot_rate_arg,
        dark_mode_arg,
        enable_log_arg,
        log_format_arg,
        log_path_arg,
        gps_origin_lat_arg,
        gps_origin_lon_arg,
        use_sim_time_arg,

        # Nodes
        data_collector_node,
        plotter_node,
        logger_node,
    ])
