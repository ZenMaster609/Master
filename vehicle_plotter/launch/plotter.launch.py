#!/usr/bin/env python3
"""
Launch file for vehicle_plotter nodes.

Launches data collector, plotter, and logger nodes with configurable options.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    # Session manager node - creates unified session for plotter and logger
    session_manager_node = Node(
        package='vehicle_plotter',
        executable='session_manager_node',
        name='session_manager',
        output='screen',
        parameters=[{'broadcast_rate_hz': 1.0}],
    )

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
        description='Enable live plotting'
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

    save_plots_on_exit_arg = DeclareLaunchArgument(
        'save_plots_on_exit',
        default_value='false',
        description='Save plot window images on shutdown'
    )

    save_plot_data_on_exit_arg = DeclareLaunchArgument(
        'save_plot_data_on_exit',
        default_value='true',
        description='Save plotted data CSVs on shutdown'
    )

    close_plots_on_shutdown_arg = DeclareLaunchArgument(
        'close_plots',
        default_value='true',
        description='Close plot windows when the plotter node shuts down'
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
        default_value='',
        description='Base path for log files (empty = auto-detect ./multidata)'
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

    sensor_config_arg = DeclareLaunchArgument(
        'sensor_config',
        default_value='',
        description='Path to sim_car sensor_config.yaml (empty = auto-detect)'
    )

    # Rosbag options
    enable_rosbag_arg = DeclareLaunchArgument(
        'enable_rosbag',
        default_value='true',
        description='Enable rosbag recording'
    )

    enable_data_collector_arg = DeclareLaunchArgument(
        'enable_data_collector',
        default_value='true',
        description='Enable data collector node'
    )

    # Data collector node (always runs)
    # Note: We use wall clock for timers but sensor timestamps for sync,
    # so use_sim_time is set to false to ensure timers fire reliably
    data_collector_node = Node(
        package='vehicle_plotter',
        executable='data_collector_node',
        name='data_collector',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_data_collector')),
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
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('enable_plot'), "'.lower() == 'true'"
        ])),
        parameters=[{
            'backend': 'pyqtgraph',
            'update_rate_hz': LaunchConfiguration('plot_rate_hz'),
            'dark_mode': LaunchConfiguration('dark_mode'),
            'enable_gui': True,
            'plot_layout': 'all',
            'window_title': 'Vehicle Plotter',
            'save_plots_on_exit': LaunchConfiguration('save_plots_on_exit'),
            'save_plot_data_on_exit': LaunchConfiguration('save_plot_data_on_exit'),
            'close_plots_on_shutdown': LaunchConfiguration('close_plots'),
            'sensor_config_path': LaunchConfiguration('sensor_config'),
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

    # Rosbag controller node (conditional)
    rosbag_controller_node = Node(
        package='vehicle_plotter',
        executable='rosbag_controller_node',
        name='rosbag_controller',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_rosbag')),
        parameters=[{
            'mode': 'simulation',
            'compression': 'zstd',
            'wait_for_session': True,
            'session_timeout_sec': 5.0,
        }],
    )

    return LaunchDescription([
        # Arguments
        adapter_arg,
        output_rate_arg,
        enable_plot_arg,
        plot_rate_arg,
        dark_mode_arg,
        save_plots_on_exit_arg,
        save_plot_data_on_exit_arg,
        close_plots_on_shutdown_arg,
        enable_log_arg,
        log_format_arg,
        log_path_arg,
        gps_origin_lat_arg,
        gps_origin_lon_arg,
        use_sim_time_arg,
        sensor_config_arg,
        enable_rosbag_arg,
        enable_data_collector_arg,

        # Nodes (session_manager must start first)
        session_manager_node,
        data_collector_node,
        plotter_node,
        logger_node,
        rosbag_controller_node,
    ])
