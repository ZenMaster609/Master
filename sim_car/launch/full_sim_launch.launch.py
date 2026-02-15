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
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import yaml


def generate_launch_description():
    sim_car_share = FindPackageShare('sim_car')
    vehicle_plotter_share = FindPackageShare('vehicle_plotter')

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo headless (no GUI)'
    )

    update_rate_arg = DeclareLaunchArgument(
        'update_rate_hz',
        default_value='100.0',
        description='Dynamics + joint state update rate (Hz)'
    )

    sensors_render_engine_arg = DeclareLaunchArgument(
        'sensors_render_engine',
        default_value='ogre',
        description='Render engine for injected Gazebo sensors plugin (ogre or ogre2)'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([sim_car_share, 'worlds', 'acceleration.world']),
        description='Full path to world file to load'
    )

    plotting_arg = DeclareLaunchArgument(
        'plotting',
        default_value='false',
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
        default_value='false',
        description='Enable rosbag recording'
    )

    steering_arg = DeclareLaunchArgument(
        'steering',
        default_value='true',
        description='Enable the EUFS steering GUI'
    )

    control_bridge_arg = DeclareLaunchArgument(
        'bridge',
        default_value='throttle',
        description="Control bridge to use: 'throttle' or 'ackermann'"
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from /clock topic'
    )

    measure_arg = DeclareLaunchArgument(
        'measure',
        default_value='false',
        description='Enable measurement_node and use /sim/raw topics'
    )

    camera_stream_arg = DeclareLaunchArgument(
        'camera_stream',
        default_value='none',
        description="OpenCV stereo stream window mode: 'raw', 'rect', or 'none'"
    )

    stereo_depth_arg = DeclareLaunchArgument(
        'stereo_depth',
        default_value='true',
        description='Enable stereo depth estimation node'
    )

    stereo_compute_depth_arg = DeclareLaunchArgument(
        'stereo_compute_depth',
        default_value='true',
        description='(Deprecated) Compute disparity/depth in stereo_depth_node'
    )

    disparity_toggle_arg = DeclareLaunchArgument(
        'disparity_toggle',
        default_value='true',
        description='Master enable for disparity + depth computation'
    )

    disparity_sampling_arg = DeclareLaunchArgument(
        'disparity_sampling',
        default_value='1',
        description='Compute disparity+depth every N rectified pairs (1=every frame, 5=every 5 frames)'
    )

    stereo_eval_arg = DeclareLaunchArgument(
        'stereo_eval',
        default_value='true',
        description='Enable stereo calibration/rectification evaluation node'
    )

    measurement_config_arg = DeclareLaunchArgument(
        'measurement_config',
        default_value=PathJoinSubstitution([sim_car_share, 'config', 'sensor_config.yaml']),
        description='Measurement config YAML path'
    )

    topic_prefix = PythonExpression([
        "'/sim/raw' if '",
        LaunchConfiguration('measure'),
        "'.lower() == 'true' else '/sim'"
    ])

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'gazebo_sim.launch.py'])
        ),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'world': LaunchConfiguration('world'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'update_rate_hz': LaunchConfiguration('update_rate_hz'),
            'sensors_render_engine': LaunchConfiguration('sensors_render_engine'),
            'topic_prefix': topic_prefix,
        }.items(),
    )

    enable_steering_gui = LaunchConfiguration('steering')

    sim_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'nodes.launch.py'])
        ),
        launch_arguments={
            'topic_prefix': topic_prefix,
        }.items(),
    )

    measurement_node = Node(
        package='measurement_node',
        executable='measurement_node',
        name='measurement_node',
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('measure'), "'.lower() == 'true'"
        ])),
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'config_path': LaunchConfiguration('measurement_config'),
        }],
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
            'sensor_config': LaunchConfiguration('measurement_config'),
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

    throttle_bridge_node = Node(
        name='throttle_cmd_bridge',
        package='sim_car',
        executable='throttle_cmd_bridge',
        output='screen',
        parameters=[{
            'input_topic': '/cmd',
            'output_topic': '/cmd_vel',
            'wheelbase': 1.6,
            'input_mode': 'throttle',
            'max_speed': control_config['max_speed'],
            'accel_limit': control_config['accel_limit'],
            'brake_decel_limit': control_config['brake_decel_limit'],
        }],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('steering'), "'.lower() == 'true' and '",
            LaunchConfiguration('bridge'), "'.lower() == 'throttle'"
        ]))
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
            'command_mode': 'acceleration',
            'max_speed': control_config['max_speed'],
            'accel_limit': control_config['accel_limit'],
            'brake_decel_limit': control_config['brake_decel_limit'],
        }],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('steering'), "'.lower() == 'true' and '",
            LaunchConfiguration('bridge'), "'.lower() == 'ackermann'"
        ]))
    )

    camera_stream_node = Node(
        package='sim_car',
        executable='camera_stream_node',
        name='camera_stream_node',
        output='screen',
        parameters=[{
            'left_topic': PythonExpression([
                "'", topic_prefix, "' + ("
                "'/stereo/left/image_raw' if '", LaunchConfiguration('camera_stream'),
                "'.lower() == 'raw' else '/stereo/left/image_rect'"
                ")"
            ]),
            'right_topic': PythonExpression([
                "'", topic_prefix, "' + ("
                "'/stereo/right/image_raw' if '", LaunchConfiguration('camera_stream'),
                "'.lower() == 'raw' else '/stereo/right/image_rect'"
                ")"
            ]),
            'window_name': PythonExpression([
                "'Stereo Raw Stream' if '", LaunchConfiguration('camera_stream'),
                "'.lower() == 'raw' else 'Stereo Rectified Stream'"
            ]),
            'show_right': True,
            'downsampling': 0.3,
            'n_frames': 3,
        }],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('camera_stream'), "'.lower() != 'none'"
        ]))
    )

    stereo_depth_node = Node(
        package='sim_car',
        executable='stereo_depth_node',
        name='stereo_depth_node',
        output='screen',
        parameters=[{
            'left_image_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/left/image_raw'"]),
            'right_image_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/right/image_raw'"]),
            'left_camera_info_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/left/camera_info'"]),
            'right_camera_info_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/right/camera_info'"]),
            'left_rect_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/left/image_rect'"]),
            'right_rect_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/right/image_rect'"]),
            'disparity_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/disparity'"]),
            'depth_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/depth'"]),
            'depth_preview_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/depth_preview'"]),
            'calibration_file': PathJoinSubstitution([sim_car_share, 'config', 'stereo_calibration.yaml']),
            'baseline_m': 0.12,
            'publish_rectified': True,
            'publish_preview': True,
            'rectify_rate_hz': 15.0,
            'disparity_sampling': ParameterValue(
                LaunchConfiguration('disparity_sampling'),
                value_type=int,
            ),
            # Gazebo cameras can be phase-shifted; allow a bit more slack so rectified outputs stay live.
            'max_time_diff_sec': 0.08,
            # disparity_toggle is the master switch; stereo_compute_depth kept for backward compat.
            'compute_disparity': ParameterValue(PythonExpression([
                "'", LaunchConfiguration('disparity_toggle'), "'.lower() == 'true' and '",
                LaunchConfiguration('stereo_compute_depth'), "'.lower() == 'true'"
            ]), value_type=bool),
        }],
        condition=IfCondition(LaunchConfiguration('stereo_depth'))
    )

    stereo_eval_node = Node(
        package='sim_car',
        executable='stereo_eval_node',
        name='stereo_eval_node',
        output='screen',
        parameters=[{
            'left_rect_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/left/image_rect'"]),
            'right_rect_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/right/image_rect'"]),
            'disparity_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/disparity'"]),
            'depth_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/depth'"]),
            'min_depth_m': 0.3,
            'max_depth_m': 30.0,
            'report_period_sec': 1.0,
        }],
        condition=IfCondition(LaunchConfiguration('stereo_eval'))
    )

    return LaunchDescription([
        headless_arg,
        update_rate_arg,
        sensors_render_engine_arg,
        world_arg,
        plotting_arg,
        logging_arg,
        close_plots_on_shutdown_arg,
        rosbagging_arg,
        steering_arg,
        control_bridge_arg,
        use_sim_time_arg,
        measure_arg,
        camera_stream_arg,
        stereo_depth_arg,
        stereo_compute_depth_arg,
        disparity_toggle_arg,
        disparity_sampling_arg,
        stereo_eval_arg,
        measurement_config_arg,
        gazebo_launch,
        sim_nodes_launch,
        measurement_node,
        plotter_launch,
        throttle_bridge_node,
        steering_bridge_node,
        camera_stream_node,
        stereo_depth_node,
        stereo_eval_node,
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
