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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import yaml


def generate_launch_description():
    sim_car_share = FindPackageShare('sim_car')
    vehicle_plotter_share = FindPackageShare('vehicle_plotter')

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='true',
        description='Run Gazebo headless (no GUI)'
    )

    update_rate_arg = DeclareLaunchArgument(
        'update_rate_hz',
        default_value='100.0',
        description='Dynamics + joint state update rate (Hz)'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([sim_car_share, 'worlds', 'small_track.world']),
        description='Full path to world file to load'
    )

    plotting_arg = DeclareLaunchArgument(
        'plotting',
        default_value='true',
        description='Enable live plotting'
    )

    logging_arg = DeclareLaunchArgument(
        'logging',
        default_value='true',
        description='Enable data logging'
    )

    close_plots_on_shutdown_arg = DeclareLaunchArgument(
        'close_plots',
        default_value='true',
        description='Close live plot windows when the plotter node shuts down'
    )

    rosbagging_arg = DeclareLaunchArgument(
        'rosbagging',
        default_value='true',
        description='Enable rosbag recording'
    )

    steering_arg = DeclareLaunchArgument(
        'steering',
        default_value='false',
        description='Enable the EUFS steering GUI'
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
            'update_rate_hz': LaunchConfiguration('update_rate_hz'),
        }.items(),
    )

    enable_steering_gui = LaunchConfiguration('steering')

    sim_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'nodes.launch.py'])
        ),
    )

    plotter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([vehicle_plotter_share, 'launch', 'plotter.launch.py'])
        ),
        launch_arguments={
            'adapter': 'gazebo',
            'enable_plot': LaunchConfiguration('plotting'),
            'enable_log': LaunchConfiguration('logging'),
            'enable_rosbag': LaunchConfiguration('rosbagging'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'close_plots_on_shutdown': LaunchConfiguration('close_plots'),
        }.items(),
    )

    steering_gui_node = Node(
        name='eufs_robot_steering_gui',
        package='steering_gui',
        executable='eufs_robot_steering_gui',
        arguments=['--force-discover'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('steering'))
    )

    control_config = _load_control_config()

    steering_bridge_node = Node(
        name='ackermann_cmd_bridge',
        package='sim_car',
        executable='ackermann_cmd_bridge',
        output='screen',
        parameters=[{
            'input_topic': '/cmd',
            'output_topic': '/cmd_vel',
            'wheelbase': 1.6,
            'command_mode': 'acceleration',
            'max_speed': control_config['max_speed'],
            'accel_limit': control_config['accel_limit'],
            'brake_decel_limit': control_config['brake_decel_limit'],
        }],
        condition=IfCondition(LaunchConfiguration('steering'))
    )

    return LaunchDescription([
        headless_arg,
        update_rate_arg,
        world_arg,
        plotting_arg,
        logging_arg,
        close_plots_on_shutdown_arg,
        rosbagging_arg,
        steering_arg,
        use_sim_time_arg,
        gazebo_launch,
        sim_nodes_launch,
        plotter_launch,
        steering_bridge_node,
        steering_gui_node,
    ])


def _load_control_config():
    try:
        config_path = get_package_share_directory('sim_car')
    except Exception:
        return _default_control_config()
    eufs_config = f"{config_path}/config/eufs_config.yaml"
    try:
        with open(eufs_config, 'r') as config_file:
            config = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError):
        return _default_control_config()

    control = config.get('control')
    if not isinstance(control, dict):
        return _default_control_config()
    return {
        'max_speed': float(control.get('max_speed', 75.0)),
        'accel_limit': float(control.get('accel_limit', 12.5)),
        'brake_decel_limit': float(control.get('brake_decel_limit', 25.0)),
    }


def _default_control_config():
    return {
        'max_speed': 75.0,
        'accel_limit': 12.5,
        'brake_decel_limit': 25.0,
    }
