#!/usr/bin/env python3
"""
Launch file for VectorNav VN-200 decoder.

Usage:
    ros2 launch vectornav_decoder vectornav_decoder.launch.py
    ros2 launch vectornav_decoder vectornav_decoder.launch.py serial_port:=/dev/ttyUSB1
    ros2 launch vectornav_decoder vectornav_decoder.launch.py config_file:=/path/to/config.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get default config path
    pkg_dir = get_package_share_directory('vectornav_decoder')
    default_config = os.path.join(pkg_dir, 'config', 'default_output.yaml')

    # Arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to YAML configuration file'
    )

    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial port device'
    )

    baudrate_arg = DeclareLaunchArgument(
        'baudrate',
        default_value='921600',
        description='Serial baud rate'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate_hz',
        default_value='200.0',
        description='Maximum publish rate in Hz'
    )

    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='vectornav',
        description='TF frame ID for published messages'
    )

    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/vectornav/imu',
        description='IMU topic name'
    )

    gps_topic_arg = DeclareLaunchArgument(
        'gps_topic',
        default_value='/vectornav/gps',
        description='GPS topic name'
    )

    ins_topic_arg = DeclareLaunchArgument(
        'ins_topic',
        default_value='/vectornav/ins',
        description='INS odometry topic name'
    )

    show_stats_arg = DeclareLaunchArgument(
        'show_stats',
        default_value='true',
        description='Show periodic statistics'
    )

    stats_interval_arg = DeclareLaunchArgument(
        'stats_interval',
        default_value='5.0',
        description='Statistics interval in seconds'
    )

    # VectorNav decoder node
    vectornav_decoder = Node(
        package='vectornav_decoder',
        executable='vectornav_decoder_node',
        name='vectornav_decoder',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'serial_port': LaunchConfiguration('serial_port'),
            'baudrate': LaunchConfiguration('baudrate'),
            'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
            'frame_id': LaunchConfiguration('frame_id'),
            'imu_topic': LaunchConfiguration('imu_topic'),
            'gps_topic': LaunchConfiguration('gps_topic'),
            'ins_topic': LaunchConfiguration('ins_topic'),
            'show_stats': LaunchConfiguration('show_stats'),
            'stats_interval': LaunchConfiguration('stats_interval'),
        }],
    )

    return LaunchDescription([
        # Arguments
        config_file_arg,
        serial_port_arg,
        baudrate_arg,
        publish_rate_arg,
        frame_id_arg,
        imu_topic_arg,
        gps_topic_arg,
        ins_topic_arg,
        show_stats_arg,
        stats_interval_arg,

        # Info
        LogInfo(msg=['Starting VectorNav decoder on: ', LaunchConfiguration('serial_port')]),
        LogInfo(msg=['Using config: ', LaunchConfiguration('config_file')]),

        # Node
        vectornav_decoder,
    ])
