#!/usr/bin/env python3
"""
Launch file for CAN bus monitoring.

Launches:
1. ros2_socketcan bridge (raw CAN <-> ROS 2)
2. can_monitor_node (displays CAN traffic)

Usage:
    ros2 launch canbus_decoder can_monitor.launch.py
    ros2 launch canbus_decoder can_monitor.launch.py can_device:=can1 bitrate:=1000000
    ros2 launch canbus_decoder can_monitor.launch.py verbose:=true
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

    verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='false',
        description='Show every CAN frame (true) or just new IDs and changes (false)'
    )

    stats_interval_arg = DeclareLaunchArgument(
        'stats_interval',
        default_value='5.0',
        description='Interval between statistics printouts (seconds)'
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

    # CAN monitor node
    can_monitor = Node(
        package='canbus_decoder',
        executable='can_monitor_node',
        name='can_monitor',
        output='screen',
        parameters=[{
            'can_topic': '/can/rx',
            'verbose': LaunchConfiguration('verbose'),
            'stats_interval': LaunchConfiguration('stats_interval'),
            'show_stats': True,
        }],
    )

    return LaunchDescription([
        # Arguments
        can_device_arg,
        bitrate_arg,
        verbose_arg,
        stats_interval_arg,

        # Info
        LogInfo(msg=['Starting CAN monitor on device: ', LaunchConfiguration('can_device')]),

        # Nodes
        socketcan_bridge,
        can_monitor,
    ])
