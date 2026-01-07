#!/usr/bin/env python3
"""
Launch file for CAN bus wheel encoder decoder.

Launches:
1. ros2_socketcan bridge (raw CAN <-> ROS 2)
2. can_decoder_node (decodes wheel encoder messages)

Usage:
    ros2 launch canbus_decoder can_decoder.launch.py
    ros2 launch canbus_decoder can_decoder.launch.py can_device:=can1
    ros2 launch canbus_decoder can_decoder.launch.py publish_rate_hz:=100.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    can_device_arg = DeclareLaunchArgument(
        'can_device',
        default_value='can0',
        description='CAN device name (can0, can1, vcan0, etc.)'
    )

    bitrate_arg = DeclareLaunchArgument(
        'bitrate',
        default_value='500000',
        description='CAN bitrate in bps (500000, 1000000, etc.)'
    )

    stale_timeout_arg = DeclareLaunchArgument(
        'stale_timeout_ms',
        default_value='100',
        description='Timeout for stale data in milliseconds'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate_hz',
        default_value='50.0',
        description='Publishing rate in Hz'
    )

    stats_interval_arg = DeclareLaunchArgument(
        'stats_interval',
        default_value='5.0',
        description='Interval between statistics printouts (seconds)'
    )

    show_stats_arg = DeclareLaunchArgument(
        'show_stats',
        default_value='true',
        description='Show periodic statistics'
    )

    # ros2_socketcan bridge node
    # This converts raw SocketCAN to ROS 2 can_msgs/Frame
    socketcan_bridge = Node(
        package='ros2_socketcan',
        executable='socket_can_bridge',
        name='socket_can_bridge',
        output='screen',
        parameters=[{
            'interface': LaunchConfiguration('can_device'),
        }],
    )

    # CAN decoder node
    can_decoder = Node(
        package='canbus_decoder',
        executable='can_decoder_node',
        name='can_decoder',
        output='screen',
        parameters=[{
            'can_topic': '/can/rx',
            'stale_timeout_ms': LaunchConfiguration('stale_timeout_ms'),
            'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
            'show_stats': LaunchConfiguration('show_stats'),
            'stats_interval': LaunchConfiguration('stats_interval'),
        }],
    )

    return LaunchDescription([
        # Arguments
        can_device_arg,
        bitrate_arg,
        stale_timeout_arg,
        publish_rate_arg,
        stats_interval_arg,
        show_stats_arg,

        # Info
        LogInfo(msg=['Starting CAN decoder on device: ', LaunchConfiguration('can_device')]),
        LogInfo(msg=['Publishing rate: ', LaunchConfiguration('publish_rate_hz'), ' Hz']),

        # Nodes
        socketcan_bridge,
        can_decoder,
    ])
