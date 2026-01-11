#!/usr/bin/env python3
"""
Simulation Mode Launch - Full Gazebo simulation with session management.

Usage:
    ros2 launch vehicle_plotter sim.launch.py
    ros2 launch vehicle_plotter sim.launch.py headless:=false
    ros2 launch vehicle_plotter sim.launch.py control_mode:=keyboard
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_car_share = FindPackageShare('sim_car')

    headless_arg = DeclareLaunchArgument('headless', default_value='true')
    control_mode_arg = DeclareLaunchArgument('control_mode', default_value='auto')
    linear_speed_arg = DeclareLaunchArgument('linear_speed', default_value='0.5')
    angular_speed_arg = DeclareLaunchArgument('angular_speed', default_value='1.0')
    enable_plot_arg = DeclareLaunchArgument('enable_plot', default_value='true')
    enable_log_arg = DeclareLaunchArgument('enable_log', default_value='true')
    enable_rosbag_arg = DeclareLaunchArgument('enable_rosbag', default_value='true')

    session_manager = Node(
        package='vehicle_plotter',
        executable='session_manager_node',
        name='session_manager',
        parameters=[{'broadcast_rate_hz': 1.0}],
        output='screen',
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'gazebo_sim.launch.py'])
        ),
        launch_arguments={'headless': LaunchConfiguration('headless')}.items(),
    )

    sim_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'nodes.launch.py'])
        ),
        launch_arguments={
            'control_mode': LaunchConfiguration('control_mode'),
            'linear_speed': LaunchConfiguration('linear_speed'),
            'angular_speed': LaunchConfiguration('angular_speed'),
        }.items(),
    )

    data_collector = Node(
        package='vehicle_plotter',
        executable='data_collector_node',
        name='data_collector',
        parameters=[{'adapter': 'gazebo', 'output_rate_hz': 50.0, 'use_sim_time': True}],
        output='screen',
    )

    plotter = Node(
        package='vehicle_plotter',
        executable='plotter_node',
        name='plotter',
        parameters=[{
            'backend': 'pyqtgraph',
            'update_rate_hz': 30.0,
            'window_title': 'Vehicle Plotter (Simulation)',
            'save_plots_on_exit': True,
            'save_plot_data_on_exit': True,
            'wait_for_session': True,
            'session_timeout_sec': 2.0,
            'use_sim_time': True,
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_plot')),
    )

    logger = Node(
        package='vehicle_plotter',
        executable='logger_node',
        name='logger',
        parameters=[{
            'format': 'parquet',
            'wait_for_session': True,
            'session_timeout_sec': 2.0,
            'use_sim_time': True,
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_log')),
    )

    rosbag_controller = Node(
        package='vehicle_plotter',
        executable='rosbag_controller_node',
        name='rosbag_controller',
        parameters=[{
            'mode': 'simulation',
            'compression': 'zstd',
            'wait_for_session': True,
            'session_timeout_sec': 2.0,
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_rosbag')),
    )

    return LaunchDescription([
        headless_arg, control_mode_arg, linear_speed_arg, angular_speed_arg,
        enable_plot_arg, enable_log_arg, enable_rosbag_arg,
        session_manager, gazebo_launch, sim_nodes_launch,
        data_collector, plotter, logger, rosbag_controller,
    ])
