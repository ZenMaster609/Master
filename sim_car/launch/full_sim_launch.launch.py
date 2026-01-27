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
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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

    sensor_mode = LaunchConfiguration('sensor_mode')
    enable_steering_gui = LaunchConfiguration('steering')
    control_mode = PythonExpression([
        "'none' if '", enable_steering_gui, "' == 'true' else 'auto'"
    ])
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
            'control_mode': control_mode,
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
            'enable_plot': LaunchConfiguration('plotting'),
            'enable_real_plot': enable_real_sensors,
            'enable_virtual_plot': enable_virtual_sensors,
            'enable_virtual_sensors': enable_virtual_sensors,
            'enable_log': LaunchConfiguration('logging'),
            'enable_rosbag': LaunchConfiguration('rosbagging'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    steering_gui_node = Node(
        name='eufs_robot_steering_gui',
        package='eufs_rqt',
        executable='eufs_robot_steering_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration('steering'))
    )

    steering_bridge_node = Node(
        name='ackermann_cmd_bridge',
        package='sim_car',
        executable='ackermann_cmd_bridge',
        output='screen',
        parameters=[{
            'input_topic': '/cmd',
            'output_topic': '/cmd_vel',
            'wheelbase': 1.6,
            'command_mode': 'velocity',
        }],
        condition=IfCondition(LaunchConfiguration('steering'))
    )

    return LaunchDescription([
        headless_arg,
        update_rate_arg,
        world_arg,
        linear_speed_arg,
        angular_speed_arg,
        sensor_mode_arg,
        plotting_arg,
        logging_arg,
        rosbagging_arg,
        steering_arg,
        use_sim_time_arg,
        gazebo_launch,
        sim_nodes_launch,
        plotter_launch,
        steering_bridge_node,
        steering_gui_node,
    ])
