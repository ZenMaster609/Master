#!/usr/bin/env python3
"""
Full sim bringup for the EUFS car.

This launch file keeps the virtual sensor pipeline separate from the
autonomy stack:

    sensor nodes -> measurement_node -> vehicle_plotter

The top-level ``sensor_pipeline`` argument is the simple on/off switch for
that path. Per-node toggles are kept for compatibility, but are no longer
needed for normal use.
"""

from launch import LaunchDescription
import tempfile
from pathlib import Path

from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
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
        default_value='180.0',
        description='Dynamics + joint state update rate (Hz)'
    )

    camera_rate_arg = DeclareLaunchArgument(
        'camera_rate_hz',
        default_value='15.0',
        description='Camera sensor update rate in Hz'
    )

    perception_rate_arg = DeclareLaunchArgument(
        'perception_rate_hz',
        default_value='60.0',
        description='LiDAR + cone-memory target rate in Hz'
    )

    planner_rate_arg = DeclareLaunchArgument(
        'planner_rate_hz',
        default_value='60.0',
        description='Planner/controller/odom target rate in Hz'
    )

    sensors_render_engine_arg = DeclareLaunchArgument(
        'sensors_render_engine',
        default_value='ogre2',
        description='Render engine for injected Gazebo sensors plugin (ogre or ogre2)'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([sim_car_share, 'worlds', 'small_track.world']),
        description='Full path to world file to load'
    )

    spawn_x_arg = DeclareLaunchArgument(
        'spawn_x',
        default_value='9.58',
        description='Initial world X position for the car model (meters)'
    )

    spawn_y_arg = DeclareLaunchArgument(
        'spawn_y',
        default_value='-5.2',
        description='Initial world Y position for the car model (meters)'
    )

    spawn_z_arg = DeclareLaunchArgument(
        'spawn_z',
        default_value='0.0',
        description='Initial world Z position for the car model (meters)'
    )

    spawn_yaw_arg = DeclareLaunchArgument(
        'spawn_yaw',
        default_value='3.75',
        description='Initial world yaw for the car model (radians)'
    )

    cone_rmse_plotting_arg = DeclareLaunchArgument(
        'cone_rmse_plotting',
        default_value='true',
        description='Enable aggregated range-binned RMSE plotting'
    )

    controller_diagnostics_arg = DeclareLaunchArgument(
        'controller_diagnostics',
        default_value='true',
        description='Enable controller diagnostics CSV logging in the main logger'
    )

    controller_diagnostics_live_plot_enabled_arg = DeclareLaunchArgument(
        'controller_diagnostics_live_plot_enabled',
        default_value='true',
        description='Enable live controller diagnostics plot window'
    )

    controller_diagnostics_live_plot_rate_hz_arg = DeclareLaunchArgument(
        'controller_diagnostics_live_plot_rate_hz',
        default_value='10.0',
        description='Refresh rate for live controller diagnostics plot'
    )

    controller_diagnostics_live_buffer_sec_arg = DeclareLaunchArgument(
        'controller_diagnostics_live_buffer_sec',
        default_value='30.0',
        description='History window in seconds for live controller diagnostics plot'
    )

    logging_arg = DeclareLaunchArgument(
        'logging',
        default_value='false',
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
        default_value='false',
        description='Enable the EUFS steering GUI'
    )

    control_bridge_arg = DeclareLaunchArgument(
        'bridge',
        default_value='ackermann',
        description="Control bridge to use. Supported value today: 'ackermann'"
    )

    ackermann_steering_sign_arg = DeclareLaunchArgument(
        'ackermann_steering_sign',
        default_value='1.0',
        description='Sign applied to Ackermann steering before cmd_vel conversion (+1 or -1)'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from /clock topic'
    )

    sensor_pipeline_arg = DeclareLaunchArgument(
        'sensor_pipeline',
        default_value='false',
        description='Enable sim sensor nodes, measurement_node, and vehicle_plotter together'
    )

    measure_arg = DeclareLaunchArgument(
        'measure',
        default_value='false',
        description='Enable measurement_node and use /sim/raw topics'
    )

    sensor_nodes_arg = DeclareLaunchArgument(
        'sensor_nodes',
        default_value='false',
        description='Enable sim_car sensor nodes launch include'
    )

    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value='delaunay',
        description="Planner to launch: 'delaunay' or 'none'"
    )

    controller_arg = DeclareLaunchArgument(
        'controller',
        default_value='stanley',
        description="Controller to use: 'stanley' only"
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz (disabled when headless=true)'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Alias switch for RViz launch (must also satisfy rviz:=true)'
    )

    rviz_profile_arg = DeclareLaunchArgument(
        'rviz_profile',
        default_value='clean',
        description="RViz profile to load when rviz_config is not provided: 'clean' or 'planner_debug'"
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value='',
        description='Explicit path to RViz display config file (overrides rviz_profile when set)'
    )

    perception_queue_size_arg = DeclareLaunchArgument(
        'perception_queue_size',
        default_value='8',
        description='Max buffered frames per stereo side before dropping old frames'
    )

    cuda_arg = DeclareLaunchArgument(
        'cuda',
        default_value='true',
        description='Enable CUDA disparity backend (false forces CPU StereoSGBM)'
    )

    stereo_arg = DeclareLaunchArgument(
        'stereo',
        default_value='false',
        description='Enable stereo depth processing for RMSE plotting'
    )
    mono_arg = DeclareLaunchArgument(
        'mono',
        default_value='false',
        description='Alias to force monocular camera source naming for cone plotting'
    )

    lidar_enabled_arg = DeclareLaunchArgument(
        'lidar_enabled',
        default_value='true',
        description='Enable LiDAR cone detection/evaluation node'
    )

    cone_memory_enabled_arg = DeclareLaunchArgument(
        'cone_memory_enabled',
        default_value='true',
        description='Enable cone memory fusion node and publish /tracked_cones'
    )

    camera_range_arg = DeclareLaunchArgument(
        'camera_range_m',
        default_value='0.0',
        description='Far-band range (m) where camera overrides lidar for position (0..20)'
    )

    prefer_lidar_if_camera_missing_far_arg = DeclareLaunchArgument(
        'prefer_lidar_if_camera_missing_far',
        default_value='true',
        description='In far band, use lidar position if camera position is missing'
    )

    allow_camera_fallback_near_arg = DeclareLaunchArgument(
        'allow_camera_fallback_near',
        default_value='false',
        description='In near band, allow camera position if lidar position is missing'
    )

    enable_track_live_plot_arg = DeclareLaunchArgument(
        'enable_track_live_plot',
        default_value='false',
        description='Enable live global track-belief plot in cone_memory_node'
    )

    lidar_cone_rmse_plotting_arg = DeclareLaunchArgument(
        'lidar_cone_rmse_plotting',
        default_value='true',
        description='Enable aggregated range-binned RMSE plotting for LiDAR eval stream'
    )

    camera_debug_arg = DeclareLaunchArgument(
        'camera_debug',
        default_value='true',
        description='Enable perception debug image stream'
    )

    camera_debug_n_frames_arg = DeclareLaunchArgument(
        'camera_debug_n_frames',
        default_value='3',
        description='Publish camera debug image every N frames (1=every frame)'
    )

    monocular_bbox_height_offset_px_arg = DeclareLaunchArgument(
        'monocular_bbox_height_offset_px',
        default_value='1.3075',
        description='Pixel offset subtracted from YOLO bbox height for monocular depth correction'
    )

    yolo_enabled_arg = DeclareLaunchArgument(
        'yolo_enabled',
        default_value='true',
        description='Enable YOLO detection in perception_node'
    )

    yolo_model_path_arg = DeclareLaunchArgument(
        'yolo_model_path',
        default_value=PathJoinSubstitution([sim_car_share, 'yolo', 'weights', 'best.pt']),
        description='Path to YOLO model (.pt or .onnx)'
    )

    yolo_input_size_arg = DeclareLaunchArgument(
        'yolo_input_size',
        default_value='960',
        description='YOLO square inference input size'
    )

    yolo_conf_threshold_arg = DeclareLaunchArgument(
        'yolo_conf_threshold',
        default_value='0.25',
        description='YOLO confidence threshold'
    )

    yolo_iou_threshold_arg = DeclareLaunchArgument(
        'yolo_iou_threshold',
        default_value='0.45',
        description='YOLO NMS IoU threshold'
    )

    yolo_prefer_cuda_arg = DeclareLaunchArgument(
        'yolo_prefer_cuda',
        default_value='true',
        description='Prefer OpenCV DNN CUDA backend for YOLO ONNX'
    )

    opencv_pythonpath_arg = DeclareLaunchArgument(
        'opencv_pythonpath',
        default_value=PathJoinSubstitution(
            [EnvironmentVariable('HOME'), 'ros2_ws', 'opencv_local', 'lib', 'python3.10', 'dist-packages']
        ),
        description='OpenCV python package path prepended for perception_node'
    )

    opencv_ld_library_path_arg = DeclareLaunchArgument(
        'opencv_ld_library_path',
        default_value=PathJoinSubstitution([EnvironmentVariable('HOME'), 'ros2_ws', 'opencv_local', 'lib']),
        description='OpenCV shared library path prepended for perception_node'
    )

    cudnn_ld_library_path_arg = DeclareLaunchArgument(
        'cudnn_ld_library_path',
        default_value=PathJoinSubstitution([EnvironmentVariable('HOME'), 'ros2_ws', 'cudnn_py', 'nvidia', 'cudnn', 'lib']),
        description='cuDNN shared library path prepended for perception_node'
    )

    yolo_ultralytics_pythonpath_arg = DeclareLaunchArgument(
        'yolo_ultralytics_pythonpath',
        default_value=PathJoinSubstitution(
            [EnvironmentVariable('HOME'), 'ros2_ws', 'yolo_pt_venv', 'lib', 'python3.10', 'site-packages']
        ),
        description='Ultralytics/torch site-packages path for YOLO .pt backend'
    )

    measurement_config_arg = DeclareLaunchArgument(
        'measurement_config',
        default_value=PathJoinSubstitution([sim_car_share, 'config', 'sensor_config.yaml']),
        description='Measurement config YAML path'
    )

    launch_argument_names = [
        'headless',
        'update_rate_hz',
        'camera_rate_hz',
        'perception_rate_hz',
        'planner_rate_hz',
        'sensors_render_engine',
        'world',
        'spawn_x',
        'spawn_y',
        'spawn_z',
        'spawn_yaw',
        'cone_rmse_plotting',
        'controller_diagnostics',
        'controller_diagnostics_live_plot_enabled',
        'controller_diagnostics_live_plot_rate_hz',
        'controller_diagnostics_live_buffer_sec',
        'logging',
        'close_plots',
        'rosbagging',
        'steering',
        'bridge',
        'ackermann_steering_sign',
        'use_sim_time',
        'sensor_pipeline',
        'measure',
        'sensor_nodes',
        'planner',
        'controller',
        'rviz',
        'use_rviz',
        'rviz_profile',
        'rviz_config',
        'perception_queue_size',
        'cuda',
        'stereo',
        'mono',
        'lidar_enabled',
        'cone_memory_enabled',
        'camera_range_m',
        'prefer_lidar_if_camera_missing_far',
        'allow_camera_fallback_near',
        'enable_track_live_plot',
        'lidar_cone_rmse_plotting',
        'camera_debug',
        'camera_debug_n_frames',
        'monocular_bbox_height_offset_px',
        'yolo_enabled',
        'yolo_model_path',
        'yolo_input_size',
        'yolo_conf_threshold',
        'yolo_iou_threshold',
        'yolo_prefer_cuda',
        'opencv_pythonpath',
        'opencv_ld_library_path',
        'cudnn_ld_library_path',
        'yolo_ultralytics_pythonpath',
        'measurement_config',
    ]
    launch_parameters_snapshot = {
        name: LaunchConfiguration(name) for name in launch_argument_names
    }

    resolved_measurement_config = LaunchConfiguration('resolved_measurement_config')
    resolved_rviz_config = LaunchConfiguration('resolved_rviz_config')
    launch_args_validation = OpaqueFunction(function=_validate_planner_and_controller_args)
    measurement_config_setup = OpaqueFunction(function=_configure_measurement_config)
    rviz_config_setup = OpaqueFunction(function=_configure_rviz_config)

    sensor_nodes_enabled = PythonExpression([
        "('", LaunchConfiguration('sensor_pipeline'), "'.lower() == 'true') or ('",
        LaunchConfiguration('sensor_nodes'), "'.lower() == 'true')"
    ])
    measurement_enabled = PythonExpression([
        "('", LaunchConfiguration('sensor_pipeline'), "'.lower() == 'true') or ('",
        LaunchConfiguration('measure'), "'.lower() == 'true')"
    ])
    sensor_plot_enabled = LaunchConfiguration('sensor_pipeline')
    plotter_stack_enabled = PythonExpression([
        "'true' if ('",
        LaunchConfiguration('sensor_pipeline'),
        "'.lower() == 'true' or '",
        LaunchConfiguration('logging'),
        "'.lower() == 'true' or '",
        LaunchConfiguration('rosbagging'),
        "'.lower() == 'true' or '",
        LaunchConfiguration('cone_rmse_plotting'),
        "'.lower() == 'true' or '",
        LaunchConfiguration('lidar_cone_rmse_plotting'),
        "'.lower() == 'true' or '",
        LaunchConfiguration('controller_diagnostics'),
        "'.lower() == 'true') else 'false'"
    ])
    topic_prefix = PythonExpression([
        "'/sim/raw' if (('",
        LaunchConfiguration('sensor_pipeline'),
        "'.lower() == 'true') or ('",
        LaunchConfiguration('measure'),
        "'.lower() == 'true')) else '/sim'"
    ])
    camera_source_name = PythonExpression([
        "'stereo' if '",
        LaunchConfiguration('stereo'),
        "'.lower() == 'true' else ('monocular' if '",
        LaunchConfiguration('mono'),
        "'.lower() == 'true' else 'monocular')"
    ])
    camera_debug_enabled_expr = PythonExpression([
        "'", LaunchConfiguration('camera_debug'),
        "'.strip().lower() in ['true', '1', 'on', 'yes', 'depth', 'disparity', 'left_rect', 'rect_left', 'yolo']"
    ])
    stereo_pair_slack_sec = PythonExpression([
        "max(0.02, 1.2 / float(",
        LaunchConfiguration('camera_rate_hz'),
        "))"
    ])
    delaunay_history_frames = PythonExpression([
        "max(8, int((float(",
        LaunchConfiguration('planner_rate_hz'),
        ") * 0.2) + 0.999))"
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
            'camera_rate_hz': LaunchConfiguration('camera_rate_hz'),
            'perception_rate_hz': LaunchConfiguration('perception_rate_hz'),
            'planner_rate_hz': LaunchConfiguration('planner_rate_hz'),
            'sensors_render_engine': LaunchConfiguration('sensors_render_engine'),
            'topic_prefix': topic_prefix,
            'spawn_x': LaunchConfiguration('spawn_x'),
            'spawn_y': LaunchConfiguration('spawn_y'),
            'spawn_z': LaunchConfiguration('spawn_z'),
            'spawn_yaw': LaunchConfiguration('spawn_yaw'),
        }.items(),
    )

    sim_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'nodes.launch.py'])
        ),
        launch_arguments={
            'topic_prefix': topic_prefix,
            'sensor_config': resolved_measurement_config,
        }.items(),
        condition=IfCondition(sensor_nodes_enabled),
    )

    measurement_node = Node(
        package='sim_car',
        executable='measurement_node',
        name='measurement_node',
        output='screen',
        condition=IfCondition(measurement_enabled),
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'config_path': resolved_measurement_config,
        }],
    )

    plotter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([vehicle_plotter_share, 'launch', 'plotter.launch.py'])
        ),
        condition=IfCondition(plotter_stack_enabled),
        launch_arguments={
            'enable_sensor_plot': sensor_plot_enabled,
            'enable_cone_rmse_plot': PythonExpression([
                "'true' if ('",
                LaunchConfiguration('cone_rmse_plotting'),
                "'.lower() == 'true' or '",
                LaunchConfiguration('lidar_cone_rmse_plotting'),
                "'.lower() == 'true') else 'false'"
            ]),
            'enable_controller_diagnostics_plot': LaunchConfiguration(
                'controller_diagnostics_live_plot_enabled'
            ),
            'enable_log': PythonExpression([
                "'true' if ('",
                LaunchConfiguration('logging'),
                "'.lower() == 'true' or '",
                LaunchConfiguration('cone_rmse_plotting'),
                "'.lower() == 'true' or '",
                LaunchConfiguration('lidar_cone_rmse_plotting'),
                "'.lower() == 'true' or '",
                LaunchConfiguration('controller_diagnostics'),
                "'.lower() == 'true') else 'false'"
            ]),
            'enable_state_logging': PythonExpression([
                "'true' if ('",
                LaunchConfiguration('sensor_pipeline'),
                "'.lower() == 'true' and '",
                LaunchConfiguration('logging'),
                "'.lower() == 'true') else 'false'"
            ]),
            'enable_rosbag': LaunchConfiguration('rosbagging'),
            'sensor_config': resolved_measurement_config,
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'close_plots': LaunchConfiguration('close_plots'),
            'camera_cone_eval_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/eval'"]),
            'lidar_cone_eval_topic': PythonExpression([
                "'",
                topic_prefix,
                "' + '/lidar/eval' if '",
                LaunchConfiguration('lidar_enabled'),
                "'.lower() == 'true' else ''"
            ]),
            'camera_source': camera_source_name,
            'controller_diagnostics_enabled': LaunchConfiguration('controller_diagnostics'),
            'controller_diagnostics_rate_hz': '50.0',
            'controller_diagnostics_cmd_topic': '/cmd',
            'controller_diagnostics_steering_topic': '/sim/steering_angle',
            'controller_diagnostics_joint_states_topic': '/sim/joint_states',
            'controller_diagnostics_odom_topic': '/sim/odom',
            'controller_diagnostics_path_topic': '/planned_centerline',
            'controller_diagnostics_planner_diag_topic': '/delaunay_planner/diagnostics',
            'controller_diagnostics_live_plot_enabled': LaunchConfiguration('controller_diagnostics_live_plot_enabled'),
            'controller_diagnostics_live_plot_rate_hz': LaunchConfiguration('controller_diagnostics_live_plot_rate_hz'),
            'controller_diagnostics_live_buffer_sec': LaunchConfiguration('controller_diagnostics_live_buffer_sec'),
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
            'wheelbase': 1.65,
            'command_mode': 'velocity',
            'steering_sign': ParameterValue(
                LaunchConfiguration('ackermann_steering_sign'),
                value_type=float,
            ),
            'max_speed': control_config['max_speed'],
            'accel_limit': control_config['accel_limit'],
            'brake_decel_limit': control_config['brake_decel_limit'],
            'control_rate': ParameterValue(
                LaunchConfiguration('planner_rate_hz'),
                value_type=float,
            ),
        }],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('bridge'), "'.lower() == 'ackermann'"
        ]))
    )

    stereo_perception_node = Node(
        package='sim_car',
        executable='stereo_perception_node',
        name='perception_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('stereo'), "'.lower() == 'true'"])),
        additional_env={
            'PYTHONNOUSERSITE': '1',
            'PYTHONPATH': [
                LaunchConfiguration('opencv_pythonpath'),
                ':',
                LaunchConfiguration('yolo_ultralytics_pythonpath'),
                ':',
                EnvironmentVariable('PYTHONPATH', default_value=''),
            ],
            'LD_LIBRARY_PATH': [
                LaunchConfiguration('opencv_ld_library_path'),
                ':',
                LaunchConfiguration('cudnn_ld_library_path'),
                ':',
                EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
            ],
        },
        parameters=[{
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'),
                value_type=bool,
            ),
            'left_image_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/left/image_raw'"]),
            'right_image_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/right/image_raw'"]),
            'left_camera_info_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/left/camera_info'"]),
            'right_camera_info_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/right/camera_info'"]),
            'monocular_cone_height_m': 0.3034,
            'monocular_big_cone_height_m': 0.51,
            'monocular_bbox_height_offset_px': ParameterValue(
                LaunchConfiguration('monocular_bbox_height_offset_px'),
                value_type=float,
            ),
            'eval_topic_prefix': PythonExpression(["'", topic_prefix, "' + '/stereo/eval'"]),
            'camera_debug_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/camera_debug'"]),
            'cone_detections_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/perception/cones_3d'"]),
            'cone_detections_frame': 'front_axle',
            'camera_debug': ParameterValue(
                LaunchConfiguration('camera_debug'),
                value_type=bool,
            ),
            'camera_debug_n_frames': ParameterValue(
                LaunchConfiguration('camera_debug_n_frames'),
                value_type=int,
            ),
            'camera_debug_scale': 0.5,
            'camera_debug_mono': True,
            'calibration_file': PathJoinSubstitution([sim_car_share, 'config', 'stereo_calibration.yaml']),
            'baseline_m': 0.12,
            'yolo_enabled': ParameterValue(
                LaunchConfiguration('yolo_enabled'),
                value_type=bool,
            ),
            'yolo_model_path': LaunchConfiguration('yolo_model_path'),
            'yolo_input_size': ParameterValue(
                LaunchConfiguration('yolo_input_size'),
                value_type=int,
            ),
            'yolo_conf_threshold': ParameterValue(
                LaunchConfiguration('yolo_conf_threshold'),
                value_type=float,
            ),
            'yolo_iou_threshold': ParameterValue(
                LaunchConfiguration('yolo_iou_threshold'),
                value_type=float,
            ),
            'yolo_prefer_cuda': ParameterValue(
                LaunchConfiguration('yolo_prefer_cuda'),
                value_type=bool,
            ),
            'queue_size': ParameterValue(
                LaunchConfiguration('perception_queue_size'),
                value_type=int,
            ),
            'prefer_cuda': ParameterValue(
                LaunchConfiguration('cuda'),
                value_type=bool,
            ),
            # Keep stereo matching within roughly one frame period at the selected perception rate.
            'max_time_diff_sec': ParameterValue(
                stereo_pair_slack_sec,
                value_type=float,
            ),
        }],
    )

    mono_perception_node = Node(
        package='sim_car',
        executable='mono_perception_node',
        name='perception_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('stereo'), "'.lower() != 'true'"])),
        additional_env={
            'PYTHONNOUSERSITE': '1',
            'PYTHONPATH': [
                LaunchConfiguration('opencv_pythonpath'),
                ':',
                LaunchConfiguration('yolo_ultralytics_pythonpath'),
                ':',
                EnvironmentVariable('PYTHONPATH', default_value=''),
            ],
            'LD_LIBRARY_PATH': [
                LaunchConfiguration('opencv_ld_library_path'),
                ':',
                LaunchConfiguration('cudnn_ld_library_path'),
                ':',
                EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
            ],
        },
        parameters=[{
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'),
                value_type=bool,
            ),
            'left_image_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/left/image_raw'"]),
            'right_image_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/right/image_raw'"]),
            'left_camera_info_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/left/camera_info'"]),
            'right_camera_info_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/right/camera_info'"]),
            'monocular_cone_height_m': 0.3034,
            'monocular_big_cone_height_m': 0.51,
            'monocular_bbox_height_offset_px': ParameterValue(
                LaunchConfiguration('monocular_bbox_height_offset_px'),
                value_type=float,
            ),
            'eval_topic_prefix': PythonExpression(["'", topic_prefix, "' + '/stereo/eval'"]),
            'camera_debug_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/camera_debug'"]),
            'cone_detections_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/perception/cones_3d'"]),
            'cone_detections_frame': 'front_axle',
            'camera_debug': ParameterValue(
                LaunchConfiguration('camera_debug'),
                value_type=bool,
            ),
            'camera_debug_n_frames': ParameterValue(
                LaunchConfiguration('camera_debug_n_frames'),
                value_type=int,
            ),
            'camera_debug_scale': 0.5,
            'camera_debug_mono': True,
            'calibration_file': PathJoinSubstitution([sim_car_share, 'config', 'stereo_calibration.yaml']),
            'baseline_m': 0.12,
            'yolo_enabled': ParameterValue(
                LaunchConfiguration('yolo_enabled'),
                value_type=bool,
            ),
            'yolo_model_path': LaunchConfiguration('yolo_model_path'),
            'yolo_input_size': ParameterValue(
                LaunchConfiguration('yolo_input_size'),
                value_type=int,
            ),
            'yolo_conf_threshold': ParameterValue(
                LaunchConfiguration('yolo_conf_threshold'),
                value_type=float,
            ),
            'yolo_iou_threshold': ParameterValue(
                LaunchConfiguration('yolo_iou_threshold'),
                value_type=float,
            ),
            'yolo_prefer_cuda': ParameterValue(
                LaunchConfiguration('yolo_prefer_cuda'),
                value_type=bool,
            ),
            'queue_size': ParameterValue(
                LaunchConfiguration('perception_queue_size'),
                value_type=int,
            ),
            'prefer_cuda': ParameterValue(
                LaunchConfiguration('cuda'),
                value_type=bool,
            ),
            'max_time_diff_sec': ParameterValue(
                stereo_pair_slack_sec,
                value_type=float,
            ),
        }],
    )

    camera_cone_evaluator_node = Node(
        package='sim_car',
        executable='cone_evaluator_node',
        name='camera_cone_evaluator_node',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'),
                value_type=bool,
            ),
            'predicted_cones_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/perception/cones_3d'"]),
            'ground_truth_cones_topic': '/ground_truth/cones',
            'eval_topic_prefix': PythonExpression(["'", topic_prefix, "' + '/stereo/eval'"]),
            'source_name': camera_source_name,
            'cone_rmse_plotting': ParameterValue(PythonExpression([
                "'true' if ('",
                LaunchConfiguration('logging'),
                "'.lower() == 'true' or '",
                LaunchConfiguration('cone_rmse_plotting'),
                "'.lower() == 'true') else 'false'"
            ]),
                value_type=bool,
            ),
        }],
    )

    lidar_node = Node(
        package='sim_car',
        executable='lidar_node',
        name='lidar_node',
        output='screen',
        condition=IfCondition(PythonExpression([
            "'",
            LaunchConfiguration('lidar_enabled'),
            "'.lower() == 'true' or '",
            LaunchConfiguration('cone_memory_enabled'),
            "'.lower() == 'true'",
        ])),
        parameters=[{
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'),
                value_type=bool,
            ),
            'scan_topic': PythonExpression(["'", topic_prefix, "' + '/lidar'"]),
            'cone_detections_topic': PythonExpression(["'", topic_prefix, "' + '/lidar/perception/cones_3d'"]),
            'cone_detections_frame': 'front_axle',
        }],
    )

    lidar_cone_evaluator_node = Node(
        package='sim_car',
        executable='cone_evaluator_node',
        name='lidar_cone_evaluator_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('lidar_enabled')),
        parameters=[{
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'),
                value_type=bool,
            ),
            'predicted_cones_topic': PythonExpression(["'", topic_prefix, "' + '/lidar/perception/cones_3d'"]),
            'ground_truth_cones_topic': '/ground_truth/cones',
            'eval_topic_prefix': PythonExpression(["'", topic_prefix, "' + '/lidar/eval'"]),
            'source_name': 'lidar',
            'cone_rmse_plotting': ParameterValue(PythonExpression([
                "'true' if ('",
                LaunchConfiguration('logging'),
                "'.lower() == 'true' or '",
                LaunchConfiguration('lidar_cone_rmse_plotting'),
                "'.lower() == 'true' or '",
                LaunchConfiguration('cone_rmse_plotting'),
                "'.lower() == 'true') else 'false'"
            ]), value_type=bool),
        }],
    )

    cone_memory_node = Node(
        package='sim_car',
        executable='cone_memory_node',
        name='cone_memory_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('cone_memory_enabled')),
        parameters=[
            PathJoinSubstitution([sim_car_share, 'config', 'cone_memory.yaml']),
            {
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'),
                    value_type=bool,
                ),
                'lidar_cones_topic': PythonExpression(["'", topic_prefix, "' + '/lidar/perception/cones_3d'"]),
                'camera_cones_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/perception/cones_3d'"]),
                'base_frame': 'front_axle',
                'camera_range_m': ParameterValue(
                    LaunchConfiguration('camera_range_m'),
                    value_type=float,
                ),
                'prefer_lidar_if_camera_missing_far': ParameterValue(
                    LaunchConfiguration('prefer_lidar_if_camera_missing_far'),
                    value_type=bool,
                ),
                'allow_camera_fallback_near': ParameterValue(
                    LaunchConfiguration('allow_camera_fallback_near'),
                    value_type=bool,
                ),
                'publish_rate_hz': ParameterValue(
                    LaunchConfiguration('perception_rate_hz'),
                    value_type=float,
                ),
                'enable_track_live_plot': ParameterValue(
                    LaunchConfiguration('enable_track_live_plot'),
                    value_type=bool,
                ),
            },
        ],
    )

    delaunay_planner_node = Node(
        package='sim_car',
        executable='delaunay_planner_node',
        name='delaunay_planner_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([sim_car_share, 'config', 'delaunay_planner.yaml']),
            {
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'),
                    value_type=bool,
                ),
                'topics.tracked_cones_topic': PythonExpression([
                    "'/tracked_cones' if '",
                    LaunchConfiguration('cone_memory_enabled'),
                    "'.lower() == 'true' else '",
                    topic_prefix,
                    "' + '/stereo/perception/cones_3d'",
                ]),
                'topics.odom_topic': '/sim/odom',
                'control.controller_type': LaunchConfiguration('controller'),
                'runtime.publish_rate_hz': ParameterValue(
                    LaunchConfiguration('planner_rate_hz'),
                    value_type=float,
                ),
                'validation.max_history_frames': ParameterValue(
                    delaunay_history_frames,
                    value_type=int,
                ),
            },
        ],
        condition=IfCondition(PythonExpression([
            "'",
            LaunchConfiguration('planner'),
            "'.lower() == 'delaunay'"
        ])),
    )

    camera_debug_viewer_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='camera_debug_viewer',
        output='screen',
        arguments=[PythonExpression(["'", topic_prefix, "' + '/stereo/camera_debug'"])],
        condition=IfCondition(camera_debug_enabled_expr),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='planner_debug_rviz',
        output='screen',
        arguments=['-d', resolved_rviz_config],
        parameters=[{
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'),
                value_type=bool,
            ),
        }],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('use_rviz'), "'.lower() == 'true' and '",
            LaunchConfiguration('rviz'), "'.lower() == 'true' and '",
            LaunchConfiguration('headless'), "'.lower() != 'true'"
        ])),
    )

    run_artifacts_node = Node(
        package='sim_car',
        executable='run_artifacts_node',
        name='run_artifacts_node',
        output='screen',
        parameters=[{
            'run_session_topic': '/run_session',
            'config_source_dir': PathJoinSubstitution([sim_car_share, 'config']),
            'copy_glob': '*.yaml',
            'write_once': True,
            'session_timeout_sec': 10.0,
            'launch_parameters': launch_parameters_snapshot,
        }],
    )

    return LaunchDescription([
        headless_arg,
        update_rate_arg,
        camera_rate_arg,
        perception_rate_arg,
        planner_rate_arg,
        sensors_render_engine_arg,
        world_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_z_arg,
        spawn_yaw_arg,
        cone_rmse_plotting_arg,
        controller_diagnostics_arg,
        controller_diagnostics_live_plot_enabled_arg,
        controller_diagnostics_live_plot_rate_hz_arg,
        controller_diagnostics_live_buffer_sec_arg,
        logging_arg,
        close_plots_on_shutdown_arg,
        rosbagging_arg,
        steering_arg,
        control_bridge_arg,
        ackermann_steering_sign_arg,
        use_sim_time_arg,
        sensor_pipeline_arg,
        measure_arg,
        sensor_nodes_arg,
        planner_arg,
        controller_arg,
        rviz_arg,
        use_rviz_arg,
        rviz_profile_arg,
        rviz_config_arg,
        perception_queue_size_arg,
        cuda_arg,
        stereo_arg,
        mono_arg,
        lidar_enabled_arg,
        cone_memory_enabled_arg,
        camera_range_arg,
        prefer_lidar_if_camera_missing_far_arg,
        allow_camera_fallback_near_arg,
        enable_track_live_plot_arg,
        lidar_cone_rmse_plotting_arg,
        camera_debug_arg,
        camera_debug_n_frames_arg,
        monocular_bbox_height_offset_px_arg,
        yolo_enabled_arg,
        yolo_model_path_arg,
        yolo_input_size_arg,
        yolo_conf_threshold_arg,
        yolo_iou_threshold_arg,
        yolo_prefer_cuda_arg,
        opencv_pythonpath_arg,
        opencv_ld_library_path_arg,
        cudnn_ld_library_path_arg,
        yolo_ultralytics_pythonpath_arg,
        measurement_config_arg,
        launch_args_validation,
        measurement_config_setup,
        rviz_config_setup,
        gazebo_launch,
        sim_nodes_launch,
        measurement_node,
        plotter_launch,
        steering_bridge_node,
        stereo_perception_node,
        mono_perception_node,
        camera_cone_evaluator_node,
        lidar_node,
        lidar_cone_evaluator_node,
        cone_memory_node,
        delaunay_planner_node,
        camera_debug_viewer_node,
        rviz_node,
        steering_gui_node,
        run_artifacts_node,
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


def _configure_measurement_config(context, *_args, **_kwargs):
    config_path = LaunchConfiguration('measurement_config').perform(context)
    planner_rate_hz = float(LaunchConfiguration('planner_rate_hz').perform(context))
    resolved_config = _write_rate_adjusted_measurement_config(config_path, planner_rate_hz)
    return [SetLaunchConfiguration('resolved_measurement_config', resolved_config)]


def _configure_rviz_config(context, *_args, **_kwargs):
    explicit_config = LaunchConfiguration('rviz_config').perform(context).strip()
    if explicit_config:
        resolved_config = explicit_config
    else:
        sim_car_share = get_package_share_directory('sim_car')
        rviz_profile = LaunchConfiguration('rviz_profile').perform(context).strip().lower()
        profile_to_filename = {
            'clean': 'driving_clean.rviz',
            'planner_debug': 'planner_debug.rviz',
            'debug': 'planner_debug.rviz',
        }
        resolved_filename = profile_to_filename.get(rviz_profile, 'driving_clean.rviz')
        resolved_config = str(Path(sim_car_share) / 'rviz' / resolved_filename)

    return [SetLaunchConfiguration('resolved_rviz_config', resolved_config)]


def _validate_planner_and_controller_args(context, *_args, **_kwargs):
    bridge = LaunchConfiguration('bridge').perform(context).strip().lower()
    planner = LaunchConfiguration('planner').perform(context).strip().lower()
    controller = LaunchConfiguration('controller').perform(context).strip().lower()

    supported_bridges = {'ackermann'}
    supported_planners = {'delaunay', 'none'}
    supported_controllers = {'stanley'}

    if bridge not in supported_bridges:
        raise RuntimeError(
            "Unsupported launch argument bridge='%s'. Supported value today: ackermann"
            % bridge
        )
    if planner not in supported_planners:
        raise RuntimeError(
            "Unsupported launch argument planner='%s'. Supported values: delaunay, none"
            % planner
        )
    if controller not in supported_controllers:
        raise RuntimeError(
            "Unsupported launch argument controller='%s'. Supported value: stanley"
            % controller
        )
    return []


def _write_rate_adjusted_measurement_config(config_path: str, planner_rate_hz: float) -> str:
    if not config_path:
        return config_path

    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError):
        return config_path

    signals = config.get('signals')
    if not isinstance(signals, dict):
        return config_path

    changed = False
    target_rate_hz = max(1.0, float(planner_rate_hz))
    for signal_name, signal_cfg in signals.items():
        if not isinstance(signal_cfg, dict):
            continue

        input_topic = str(signal_cfg.get('input_topic', '')).strip()
        output_topic = str(signal_cfg.get('output_topic', '')).strip()
        msg_type = str(signal_cfg.get('msg_type', '')).strip()
        if (
            signal_name == 'odom'
            or input_topic == '/sim/raw/odom'
            or output_topic == '/sim/odom'
            or msg_type == 'nav_msgs/msg/Odometry'
        ):
            signal_cfg['rate_hz'] = target_rate_hz
            changed = True
            break

    if not changed:
        return config_path

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml.safe_dump(config, tmp, default_flow_style=False, sort_keys=False)
        return tmp.name


def _default_control_config():
    return {
        'max_speed': 75.0,
        'accel_limit': 12.5,
        'brake_decel_limit': 25.0,
    }
