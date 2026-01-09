#!/usr/bin/env python3
"""
Launch file for Jetson hardware bringup (headless mode).

Launches all hardware nodes for real vehicle operation:
- CAN bus decoder (wheel encoders)
- VectorNav VN-200 decoder (IMU/GPS/INS)
- Data collector and logger (no plotting GUI)

Usage:
    # Full system (CAN + VectorNav + logging)
    ros2 launch vehicle_plotter jetson_bringup.launch.py

    # CAN only
    ros2 launch vehicle_plotter jetson_bringup.launch.py enable_vectornav:=false

    # VectorNav only
    ros2 launch vehicle_plotter jetson_bringup.launch.py enable_can:=false

    # Custom devices
    ros2 launch vehicle_plotter jetson_bringup.launch.py can_device:=can1 serial_port:=/dev/ttyUSB1
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, GroupAction, EmitEvent, RegisterEventHandler
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.conditions import IfCondition
from launch.events import matches_action
from launch_ros.actions import Node, LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition
from ament_index_python.packages import get_package_share_directory
import lifecycle_msgs.msg


def generate_launch_description():
    # Get package directories
    vectornav_pkg = get_package_share_directory('vectornav_decoder')
    default_vn_config = os.path.join(vectornav_pkg, 'config', 'default_output.yaml')

    # Get home directory for log path
    home_dir = os.path.expanduser('~')
    default_log_path = os.path.join(home_dir, '.ros', 'logs')

    # =========================================================================
    # Launch Arguments
    # =========================================================================

    # --- Enable/Disable Sensors ---
    enable_can_arg = DeclareLaunchArgument(
        'enable_can',
        default_value='true',
        description='Enable CAN bus decoder (wheel encoders)'
    )

    enable_vectornav_arg = DeclareLaunchArgument(
        'enable_vectornav',
        default_value='true',
        description='Enable VectorNav VN-200 decoder (IMU/GPS/INS)'
    )

    enable_log_arg = DeclareLaunchArgument(
        'enable_log',
        default_value='true',
        description='Enable data logging'
    )

    # --- CAN Configuration ---
    can_device_arg = DeclareLaunchArgument(
        'can_device',
        default_value='can0',
        description='CAN device name (can0, can1, vcan0)'
    )

    can_publish_rate_arg = DeclareLaunchArgument(
        'can_publish_rate_hz',
        default_value='50.0',
        description='CAN decoder publish rate in Hz'
    )

    # --- VectorNav Configuration ---
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='VectorNav serial port (/dev/ttyUSB0 or /dev/vectornav)'
    )

    baudrate_arg = DeclareLaunchArgument(
        'baudrate',
        default_value='921600',
        description='VectorNav serial baud rate'
    )

    vn_config_arg = DeclareLaunchArgument(
        'vn_config_file',
        default_value=default_vn_config,
        description='VectorNav output configuration YAML'
    )

    vn_publish_rate_arg = DeclareLaunchArgument(
        'vn_publish_rate_hz',
        default_value='200.0',
        description='VectorNav publish rate in Hz'
    )

    # --- Data Collector Configuration ---
    output_rate_arg = DeclareLaunchArgument(
        'output_rate_hz',
        default_value='50.0',
        description='Vehicle state output rate in Hz'
    )

    # --- Logger Configuration ---
    log_path_arg = DeclareLaunchArgument(
        'log_path',
        default_value=default_log_path,
        description='Base path for log files'
    )

    log_format_arg = DeclareLaunchArgument(
        'log_format',
        default_value='parquet',
        description='Log format: parquet or csv'
    )

    # --- Adapter Selection ---
    adapter_arg = DeclareLaunchArgument(
        'adapter',
        default_value='can',
        description='Sensor adapter: can, vectornav, or gazebo'
    )

    # --- Common ---
    show_stats_arg = DeclareLaunchArgument(
        'show_stats',
        default_value='true',
        description='Show periodic statistics from decoders'
    )

    stats_interval_arg = DeclareLaunchArgument(
        'stats_interval',
        default_value='10.0',
        description='Statistics logging interval in seconds'
    )

    # =========================================================================
    # CAN Bus Nodes (conditional)
    # =========================================================================

    # ros2_socketcan receiver node (lifecycle node)
    # Converts raw SocketCAN to ROS 2 can_msgs/Frame on /from_can_bus
    socketcan_receiver = LifecycleNode(
        package='ros2_socketcan',
        executable='socket_can_receiver_node_exe',
        name='socket_can_receiver',
        namespace='',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_can')),
        parameters=[{
            'interface': LaunchConfiguration('can_device'),
        }],
    )

    # Auto-configure the lifecycle node
    socketcan_configure = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(socketcan_receiver),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    # Auto-activate after configure succeeds
    socketcan_activate_handler = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=socketcan_receiver,
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(socketcan_receiver),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                )
            ],
        )
    )

    # CAN decoder node - decodes wheel encoder CAN messages
    can_decoder = Node(
        package='canbus_decoder',
        executable='can_decoder_node',
        name='can_decoder',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_can')),
        parameters=[{
            'can_topic': '/from_can_bus',
            'stale_timeout_ms': 100,
            'publish_rate_hz': LaunchConfiguration('can_publish_rate_hz'),
            'show_stats': LaunchConfiguration('show_stats'),
            'stats_interval': LaunchConfiguration('stats_interval'),
        }],
    )

    # =========================================================================
    # VectorNav Nodes (conditional)
    # =========================================================================

    vectornav_decoder = Node(
        package='vectornav_decoder',
        executable='vectornav_decoder_node',
        name='vectornav_decoder',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_vectornav')),
        parameters=[{
            'config_file': LaunchConfiguration('vn_config_file'),
            'serial_port': LaunchConfiguration('serial_port'),
            'baudrate': LaunchConfiguration('baudrate'),
            'publish_rate_hz': LaunchConfiguration('vn_publish_rate_hz'),
            'frame_id': 'vectornav',
            'imu_topic': '/vectornav/imu',
            'gps_topic': '/vectornav/gps',
            'ins_topic': '/vectornav/ins',
            'show_stats': LaunchConfiguration('show_stats'),
            'stats_interval': LaunchConfiguration('stats_interval'),
        }],
    )

    # =========================================================================
    # Vehicle Plotter Nodes (always run)
    # =========================================================================

    # Data collector - aggregates sensor data into VehicleState
    data_collector = Node(
        package='vehicle_plotter',
        executable='data_collector_node',
        name='data_collector',
        output='screen',
        parameters=[{
            'adapter': LaunchConfiguration('adapter'),
            'output_rate_hz': LaunchConfiguration('output_rate_hz'),
            'gps_origin_lat': 0.0,  # Auto-detect from first GPS fix
            'gps_origin_lon': 0.0,
            'use_sim_time': False,  # Real hardware = wall clock
        }],
    )

    # Logger node - writes vehicle state to disk (no GUI plotter)
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
            'use_sim_time': False,
        }],
    )

    # =========================================================================
    # Launch Description
    # =========================================================================

    return LaunchDescription([
        # Arguments - Enable/Disable
        enable_can_arg,
        enable_vectornav_arg,
        enable_log_arg,

        # Arguments - CAN
        can_device_arg,
        can_publish_rate_arg,

        # Arguments - VectorNav
        serial_port_arg,
        baudrate_arg,
        vn_config_arg,
        vn_publish_rate_arg,

        # Arguments - Data Collector
        output_rate_arg,

        # Arguments - Logger
        log_path_arg,
        log_format_arg,

        # Arguments - Adapter
        adapter_arg,

        # Arguments - Common
        show_stats_arg,
        stats_interval_arg,

        # Startup info
        LogInfo(msg=['============================================']),
        LogInfo(msg=['  Jetson Hardware Bringup (Headless)']),
        LogInfo(msg=['============================================']),
        LogInfo(msg=['CAN enabled:       ', LaunchConfiguration('enable_can')]),
        LogInfo(msg=['VectorNav enabled: ', LaunchConfiguration('enable_vectornav')]),
        LogInfo(msg=['Logging enabled:   ', LaunchConfiguration('enable_log')]),
        LogInfo(msg=['Log path:          ', LaunchConfiguration('log_path')]),
        LogInfo(msg=['============================================']),

        # CAN nodes (with lifecycle management)
        socketcan_activate_handler,
        socketcan_receiver,
        socketcan_configure,
        can_decoder,

        # VectorNav node
        vectornav_decoder,

        # Vehicle plotter nodes
        data_collector,
        logger_node,
    ])
