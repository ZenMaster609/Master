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

from dataclasses import dataclass
from launch import LaunchDescription
import sys
import tempfile
from pathlib import Path

from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import yaml

try:
    from sim_car.planning.planner_constants import (
        CONFIGURED_PLANNERS,
        MIGRATED_PLANNERS,
        SUPPORTED_CONTROLLERS,
        SUPPORTED_PLANNERS,
        get_planner_spec,
        planner_allowed_for_track,
    )
except ModuleNotFoundError:
    source_package_root = Path(__file__).resolve().parents[1]
    if str(source_package_root) not in sys.path:
        sys.path.insert(0, str(source_package_root))
    from sim_car.planning.planner_constants import (
        CONFIGURED_PLANNERS,
        MIGRATED_PLANNERS,
        SUPPORTED_CONTROLLERS,
        SUPPORTED_PLANNERS,
        get_planner_spec,
        planner_allowed_for_track,
    )


SUPPORTED_TRACKS = {
    'acceleration': 'acceleration.world',
    'skidpad': 'skidpad.world',
    'smalltrack': 'small_track.world',
}
RUN_ID_TRACK_ABBREVIATIONS = {
    'smalltrack': 'small',
    'acceleration': 'acc',
    'skidpad': 'skid',
}
RUN_ID_PLANNER_ABBREVIATIONS = {
    'midpoint': 'mid',
    'single_boundary': 'SB',
    'corridor': 'cor',
    'linetest': 'line',
    'none': 'none',
}
RUN_ID_CONTROLLER_ABBREVIATIONS = {
    'pure_pursuit': 'pp',
    'stanley': 'stan',
    'none': 'none',
}
SCAN2D_LIDAR_PIPELINE_NAME = 'scan2d'
SCAN2D_RUN_ID_SUFFIX = '2d'
DEFAULT_ACKERMANN_STEERING_SIGN = 1.0  # Existing default preserves the configured steering direction.
DEFAULT_USE_SIM_TIME = True  # Full sim nodes use the Gazebo /clock timeline by default.
DEFAULT_USE_SIM_TIME_LAUNCH = 'true'  # Included launch files receive launch arguments as strings.


@dataclass(frozen=True)
class LaunchSelection:
    track: str
    planner: str
    controller: str
    world: str
    planner_config: str
    controller_config: str
    spawn_config: str
    spawn_x: str
    spawn_y: str
    spawn_yaw: str
    path_tracking_autostop_laps: str
    speed_control: dict[str, float]
    planner_limits: dict[str, float]
    planner_diagnostics_topic: str
    planner_default_rviz_profile: str

    def controller_overlay_parameters(self) -> dict:
        if self.controller != 'none':
            return {}
        return {
            '/**': {
                'ros__parameters': {
                    'control': {
                        'controller_type': 'none',
                    },
                },
            },
        }

    def speed_control_overlay_parameters(self) -> dict:
        return {
            '/**': {
                'ros__parameters': {
                    'speed_control': self.speed_control,
                },
            },
        }

    def planner_overlay_parameters(self) -> dict:
        if not self.planner_limits:
            return {}
        return {
            '/**': {
                'ros__parameters': self.planner_limits,
            },
        }


def _lidar_enabled_condition():
    return IfCondition(PythonExpression([
        "(('",
        LaunchConfiguration('lidar_enabled'),
        "'.lower() == 'true') or ('",
        LaunchConfiguration('cone_memory_enabled'),
        "'.lower() == 'true'))",
    ]))


def _planner_selected_condition(planner_name: str):
    return IfCondition(PythonExpression([
        "'",
        LaunchConfiguration('planner'),
        f"'.lower() == '{planner_name}'",
    ]))


def _planner_odom_delay_enabled_expr():
    return PythonExpression([
        "float('",
        LaunchConfiguration('planner_odom_delay_ms'),
        "') > 0.0",
    ])


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
        description='Planner/controller/odom t  arget rate in Hz'
    )

    planner_odom_delay_ms_arg = DeclareLaunchArgument(
        'planner_odom_delay_ms',
        default_value='20.0',
        description='Optional fixed delay applied to planner/controller odometry feed (ms)'
    )

    planner_odom_lag_compensation_ms_arg = DeclareLaunchArgument(
        'planner_odom_lag_compensation_ms',
        default_value='20.0',
        description='Optional forward pose compensation applied inside planner/controller path transform (ms)'
    )

    sensors_render_engine_arg = DeclareLaunchArgument(
        'sensors_render_engine',
        default_value='ogre2',
        description='Render engine for injected Gazebo sensors plugin (ogre or ogre2)'
    )

    track_arg = DeclareLaunchArgument(
        'track',
        default_value='smalltrack',
        description="Track preset to load: 'acceleration', 'skidpad', or 'smalltrack'"
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='',
        description='Optional full path to world file override'
    )

    spawn_x_arg = DeclareLaunchArgument(
        'spawn_x',
        default_value='',
        description='Optional initial world X position override for the car model (meters)'
    )

    spawn_y_arg = DeclareLaunchArgument(
        'spawn_y',
        default_value='',
        description='Optional initial world Y position override for the car model (meters)'
    )

    spawn_z_arg = DeclareLaunchArgument(
        'spawn_z',
        default_value='0.0',
        description='Initial world Z position for the car model (meters)'
    )

    spawn_yaw_arg = DeclareLaunchArgument(
        'spawn_yaw',
        default_value='',
        description='Optional initial world yaw override for the car model (radians)'
    )

    steering_arg = DeclareLaunchArgument(
        'steering',
        default_value='false',
        description='Enable the EUFS steering GUI'
    )

    physics_model_arg = DeclareLaunchArgument(
        'physics_model',
        default_value='bicycle',
        description="Physics model for Gazebo dynamics: 'pointmass' or 'bicycle'"
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

    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value='midpoint',
        description="Planner to launch: 'midpoint', 'single_boundary', 'corridor', 'linetest', or 'none'"
    )

    controller_arg = DeclareLaunchArgument(
        'controller',
        default_value='stanley',
        description="Controller to launch: 'stanley', 'pure_pursuit', or 'none'"
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz (disabled when headless=true)'
    )

    rviz_profile_arg = DeclareLaunchArgument(
        'rviz_profile',
        default_value='planner',
        description=(
            "RViz profile to load when rviz_config is not provided: "
            "'planner', 'clean', 'planner_debug', 'midpoint', "
            "'single_boundary', 'corridor', or 'linetest'"
        )
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
        description='Enable stereo depth processing (false selects monocular camera mode)'
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

    camera_debug_arg = DeclareLaunchArgument(
        'camera_debug',
        default_value='false',
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

    cuda_runtime_ld_library_path_arg = DeclareLaunchArgument(
        'cuda_runtime_ld_library_path',
        default_value=PathJoinSubstitution([EnvironmentVariable('HOME'), 'ros2_ws', 'cuda_runtime', 'lib']),
        description='CUDA runtime shared library path prepended for perception_node'
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
        'planner_odom_delay_ms',
        'planner_odom_lag_compensation_ms',
        'sensors_render_engine',
        'track',
        'world',
        'spawn_x',
        'spawn_y',
        'spawn_z',
        'spawn_yaw',
        'steering',
        'sensor_pipeline',
        'measure',
        'planner',
        'controller',
        'rviz',
        'rviz_profile',
        'rviz_config',
        'perception_queue_size',
        'cuda',
        'stereo',
        'lidar_enabled',
        'cone_memory_enabled',
        'camera_range_m',
        'prefer_lidar_if_camera_missing_far',
        'allow_camera_fallback_near',
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
        'cuda_runtime_ld_library_path',
        'yolo_ultralytics_pythonpath',
        'measurement_config',
    ]
    launch_parameters_snapshot = {
        name: LaunchConfiguration(name) for name in launch_argument_names
    }
    launch_parameters_snapshot['lidar_pipeline'] = SCAN2D_LIDAR_PIPELINE_NAME

    resolved_measurement_config = LaunchConfiguration('resolved_measurement_config')
    resolved_rviz_config = LaunchConfiguration('resolved_rviz_config')
    resolved_world = LaunchConfiguration('resolved_world')
    resolved_spawn_x = LaunchConfiguration('resolved_spawn_x')
    resolved_spawn_y = LaunchConfiguration('resolved_spawn_y')
    resolved_spawn_yaw = LaunchConfiguration('resolved_spawn_yaw')
    resolved_planner_config = LaunchConfiguration('resolved_planner_config')
    resolved_controller_config = LaunchConfiguration('resolved_controller_config')
    resolved_speed_control_config = LaunchConfiguration('resolved_speed_control_config')
    resolved_planner_diagnostics_topic = LaunchConfiguration('resolved_planner_diagnostics_topic')
    resolved_path_tracking_autostop_laps = LaunchConfiguration('resolved_path_tracking_autostop_laps')
    resolved_shutdown_on_logger_exit = LaunchConfiguration('resolved_shutdown_on_logger_exit')
    resolved_run_id_prefix = LaunchConfiguration('resolved_run_id_prefix')
    launch_args_validation = OpaqueFunction(function=_validate_planner_and_controller_args)
    track_selection_setup = OpaqueFunction(function=_configure_track_selection)
    measurement_config_setup = OpaqueFunction(function=_configure_measurement_config)
    rviz_config_setup = OpaqueFunction(function=_configure_rviz_config)

    measurement_enabled = PythonExpression([
        "('", LaunchConfiguration('sensor_pipeline'), "'.lower() == 'true') or ('",
        LaunchConfiguration('measure'), "'.lower() == 'true')"
    ])
    topic_prefix = PythonExpression([
        "'/sim/raw' if (('",
        LaunchConfiguration('sensor_pipeline'),
        "'.lower() == 'true') or ('",
        LaunchConfiguration('measure'),
        "'.lower() == 'true')) else '/sim'"
    ])
    delayed_planner_odom_topic = '/sim/odom_delayed'
    planner_odom_topic = PythonExpression([
        "'/sim/odom_delayed' if ",
        _planner_odom_delay_enabled_expr(),
        " else '/sim/odom'",
    ])
    camera_source_name = PythonExpression([
        "'stereo' if '",
        LaunchConfiguration('stereo'),
        "'.lower() == 'true' else 'monocular'"
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
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'gazebo_sim.launch.py'])
        ),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'world': resolved_world,
            'use_sim_time': DEFAULT_USE_SIM_TIME_LAUNCH,
            'update_rate_hz': LaunchConfiguration('update_rate_hz'),
            'camera_rate_hz': LaunchConfiguration('camera_rate_hz'),
            'perception_rate_hz': LaunchConfiguration('perception_rate_hz'),
            'planner_rate_hz': LaunchConfiguration('planner_rate_hz'),
            'physics_model': LaunchConfiguration('physics_model'),
            'sensors_render_engine': LaunchConfiguration('sensors_render_engine'),
            'topic_prefix': topic_prefix,
            'spawn_x': resolved_spawn_x,
            'spawn_y': resolved_spawn_y,
            'spawn_z': LaunchConfiguration('spawn_z'),
            'spawn_yaw': resolved_spawn_yaw,
        }.items(),
    )

    sim_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_car_share, 'launch', 'nodes.launch.py'])
        ),
        launch_arguments={
            'topic_prefix': topic_prefix,
            'sensor_config': resolved_measurement_config,
            'use_sim_time': DEFAULT_USE_SIM_TIME_LAUNCH,
        }.items(),
        condition=IfCondition(LaunchConfiguration('sensor_pipeline')),
    )

    measurement_node = Node(
        package='sim_car',
        executable='measurement_node',
        name='measurement_node',
        output='screen',
        condition=IfCondition(measurement_enabled),
        parameters=[{
            'use_sim_time': DEFAULT_USE_SIM_TIME,
            'config_path': resolved_measurement_config,
        }],
    )

    odom_delay_node = Node(
        package='sim_car',
        executable='odom_delay_node',
        name='odom_delay_node',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(
                DEFAULT_USE_SIM_TIME,
                value_type=bool,
            ),
            'input_topic': '/sim/odom',
            'output_topic': delayed_planner_odom_topic,
            'delay_ms': ParameterValue(
                LaunchConfiguration('planner_odom_delay_ms'),
                value_type=float,
            ),
            'flush_rate_hz': 200.0,
        }],
        condition=IfCondition(_planner_odom_delay_enabled_expr()),
    )

    plotter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([vehicle_plotter_share, 'launch', 'plotter.launch.py'])
        ),
        launch_arguments={
            'enable_log': 'false',
            'enable_state_logging': LaunchConfiguration('sensor_pipeline'),
            'use_sim_time': DEFAULT_USE_SIM_TIME_LAUNCH,
            'run_id_prefix': resolved_run_id_prefix,
        }.items(),
    )

    logger_node = Node(
        package='vehicle_plotter',
        executable='logger_node',
        name='logger',
        output='screen',
        parameters=[{
            'format': 'parquet',
            'flush_interval_sec': 5.0,
            'buffer_size': 1000,
            'adapter': 'gazebo',
            'enable_logging': True,
            'state_topic': '/vehicle_plotter/state',
            'enable_state_logging': ParameterValue(
                LaunchConfiguration('sensor_pipeline'),
                value_type=bool,
            ),
            'auto_plot_on_shutdown': True,
            'camera_cone_eval_topic': PythonExpression([
                "'",
                topic_prefix,
                "' + '/stereo/eval'"
            ]),
            'lidar_cone_eval_topic': PythonExpression([
                "'",
                topic_prefix,
                "' + '/lidar/eval' if '",
                LaunchConfiguration('lidar_enabled'),
                "'.lower() == 'true' else ''"
            ]),
            'off_track_autostop_planner_diag_topic': resolved_planner_diagnostics_topic,
            'path_tracking_eval_enabled': True,
            'path_tracking_eval_gt_track_topic': '/ground_truth/track',
            'path_tracking_eval_odom_topic': '/sim/odom',
            'path_tracking_eval_planner_path_topic': '/planned_centerline',
            'path_tracking_eval_track_name': LaunchConfiguration('track'),
            'path_tracking_eval_autostop_laps': ParameterValue(
                resolved_path_tracking_autostop_laps,
                value_type=int,
            ),
            'use_sim_time': False,
        }],
    )

    shutdown_on_logger_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=logger_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason='logger autostop reached lap target'),
                    condition=IfCondition(resolved_shutdown_on_logger_exit),
                ),
            ],
        )
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
                DEFAULT_ACKERMANN_STEERING_SIGN,
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
    )

    perception_node = Node(
        package='sim_car',
        executable='perception_node',
        name='perception_node',
        output='screen',
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
                LaunchConfiguration('cuda_runtime_ld_library_path'),
                ':',
                EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
            ],
        },
        parameters=[{
            'use_sim_time': ParameterValue(
                DEFAULT_USE_SIM_TIME,
                value_type=bool,
            ),
            'stereo_enabled': ParameterValue(
                LaunchConfiguration('stereo'),
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

    camera_cone_evaluator_node = Node(
        package='sim_car',
        executable='cone_evaluator_node',
        name='camera_cone_evaluator_node',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(
                DEFAULT_USE_SIM_TIME,
                value_type=bool,
            ),
            'predicted_cones_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/perception/cones_3d'"]),
            'ground_truth_cones_topic': '/ground_truth/cones',
            'eval_topic_prefix': PythonExpression(["'", topic_prefix, "' + '/stereo/eval'"]),
            'source_name': camera_source_name,
        }],
    )

    lidar_node = Node(
        package='sim_car',
        executable='lidar_node',
        name='lidar_node',
        output='screen',
        condition=_lidar_enabled_condition(),
        parameters=[{
            'use_sim_time': ParameterValue(
                DEFAULT_USE_SIM_TIME,
                value_type=bool,
            ),
            'scan_topic': PythonExpression(["'", topic_prefix, "' + '/lidar'"]),
            'cone_detections_topic': PythonExpression(["'", topic_prefix, "' + '/lidar/perception/cones_3d'"]),
            'lidar_frame': 'lidar_scan_link',
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
                DEFAULT_USE_SIM_TIME,
                value_type=bool,
            ),
            'predicted_cones_topic': PythonExpression(["'", topic_prefix, "' + '/lidar/perception/cones_3d'"]),
            'ground_truth_cones_topic': '/ground_truth/cones',
            'eval_topic_prefix': PythonExpression(["'", topic_prefix, "' + '/lidar/eval'"]),
            'source_name': 'lidar',
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
                    DEFAULT_USE_SIM_TIME,
                    value_type=bool,
                ),
                'topics.lidar_cones_topic': PythonExpression(["'", topic_prefix, "' + '/lidar/perception/cones_3d'"]),
                'topics.camera_cones_topic': PythonExpression(["'", topic_prefix, "' + '/stereo/perception/cones_3d'"]),
                'frames.base_frame': 'front_axle',
                'fusion.camera_range_m': ParameterValue(
                    LaunchConfiguration('camera_range_m'),
                    value_type=float,
                ),
                'fusion.prefer_lidar_if_camera_missing_far': ParameterValue(
                    LaunchConfiguration('prefer_lidar_if_camera_missing_far'),
                    value_type=bool,
                ),
                'fusion.allow_camera_fallback_near': ParameterValue(
                    LaunchConfiguration('allow_camera_fallback_near'),
                    value_type=bool,
                ),
                'runtime.publish_rate_hz': ParameterValue(
                    LaunchConfiguration('perception_rate_hz'),
                    value_type=float,
                ),
            },
        ],
    )

    # Acceleration shares the routed topic so finish-line orange/stop handling stays
    # centralized, but the router passes normal cones through unchanged until that
    # finish latch is triggered.
    planner_input_topic = PythonExpression([
        "'/tracked_cones/skidpad_routed' if '",
        LaunchConfiguration('track'),
        "'.lower() in ('skidpad', 'acceleration') else ('/tracked_cones' if '",
        LaunchConfiguration('cone_memory_enabled'),
        "'.lower() == 'true' else '",
        topic_prefix,
        "' + '/stereo/perception/cones_3d')",
    ])
    router_input_topic = PythonExpression([
        "'/tracked_cones' if '",
        LaunchConfiguration('cone_memory_enabled'),
        "'.lower() == 'true' else '",
        topic_prefix,
        "' + '/stereo/perception/cones_3d'",
    ])
    router_viz_topic = PythonExpression([
        "'/skidpad_router/markers' if '",
        LaunchConfiguration('track'),
        "'.lower() in ('skidpad', 'acceleration') else '/skidpad_router/markers_hidden'",
    ])

    skidpad_router_node = Node(
        package='sim_car',
        executable='skidpad_router_node',
        name='skidpad_router_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([sim_car_share, 'config', 'skidpad', 'skidpad_router.yaml']),
            {
                'use_sim_time': ParameterValue(
                    DEFAULT_USE_SIM_TIME,
                    value_type=bool,
                ),
                'topics.input_topic': router_input_topic,
                'topics.output_topic': '/tracked_cones/skidpad_routed',
                'topics.odom_topic': planner_odom_topic,
                'topics.cmd_topic': '/cmd',
                'topics.viz_topic': router_viz_topic,
                'routing.event_mode': LaunchConfiguration('track'),
                'routing.shutdown_on_route_complete': False,
                'routing.shutdown_on_parking_complete': ParameterValue(
                    PythonExpression([
                        "'",
                        LaunchConfiguration('track'),
                        "'.lower() in ('skidpad', 'acceleration')",
                    ]),
                    value_type=bool,
                ),
            },
        ],
        condition=IfCondition(PythonExpression([
            "'",
            LaunchConfiguration('track'),
            "'.lower() in ('skidpad', 'acceleration') and '",
            LaunchConfiguration('planner'),
            "'.lower() in ('midpoint', 'single_boundary', 'corridor')"
        ])),
    )

    shutdown_on_skidpad_router_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=skidpad_router_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason='skidpad/acceleration router mission target reached'),
                    condition=IfCondition(PythonExpression([
                        "'",
                        LaunchConfiguration('track'),
                        "'.lower() in ('skidpad', 'acceleration')",
                    ])),
                ),
            ],
        )
    )

    midpoint_planner_node = Node(
        package='sim_car',
        executable=get_planner_spec('midpoint').executable,
        name='midpoint_planner_node',
        output='screen',
        parameters=[
            resolved_planner_config,
            resolved_controller_config,
            resolved_speed_control_config,
            {
                'use_sim_time': ParameterValue(
                    DEFAULT_USE_SIM_TIME,
                    value_type=bool,
                ),
                'topics.tracked_cones_topic': planner_input_topic,
                'topics.odom_topic': planner_odom_topic,
                'control.odom_lag_compensation_ms': ParameterValue(
                    LaunchConfiguration('planner_odom_lag_compensation_ms'),
                    value_type=float,
                ),
                'runtime.publish_rate_hz': ParameterValue(
                    LaunchConfiguration('planner_rate_hz'),
                    value_type=float,
                ),
                'lap_tracking.target_laps': resolved_path_tracking_autostop_laps,
            },
        ],
        condition=_planner_selected_condition('midpoint'),
    )

    single_boundary_planner_node = Node(
        package='sim_car',
        executable=get_planner_spec('single_boundary').executable,
        name='single_boundary_planner_node',
        output='screen',
        parameters=[
            resolved_planner_config,
            resolved_controller_config,
            resolved_speed_control_config,
            {
                'use_sim_time': ParameterValue(
                    DEFAULT_USE_SIM_TIME,
                    value_type=bool,
                ),
                'topics.tracked_cones_topic': planner_input_topic,
                'topics.odom_topic': planner_odom_topic,
                'control.odom_lag_compensation_ms': ParameterValue(
                    LaunchConfiguration('planner_odom_lag_compensation_ms'),
                    value_type=float,
                ),
                'runtime.publish_rate_hz': ParameterValue(
                    LaunchConfiguration('planner_rate_hz'),
                    value_type=float,
                ),
                'lap_tracking.target_laps': resolved_path_tracking_autostop_laps,
            },
        ],
        condition=_planner_selected_condition('single_boundary'),
    )

    corridor_planner_node = Node(
        package='sim_car',
        executable=get_planner_spec('corridor').executable,
        name='corridor_planner_node',
        output='screen',
        parameters=[
            resolved_planner_config,
            resolved_controller_config,
            resolved_speed_control_config,
            {
                'use_sim_time': ParameterValue(
                    DEFAULT_USE_SIM_TIME,
                    value_type=bool,
                ),
                'topics.tracked_cones_topic': planner_input_topic,
                'topics.odom_topic': planner_odom_topic,
                'control.odom_lag_compensation_ms': ParameterValue(
                    LaunchConfiguration('planner_odom_lag_compensation_ms'),
                    value_type=float,
                ),
                'runtime.publish_rate_hz': ParameterValue(
                    LaunchConfiguration('planner_rate_hz'),
                    value_type=float,
                ),
                'lap_tracking.target_laps': resolved_path_tracking_autostop_laps,
            },
        ],
        condition=_planner_selected_condition('corridor'),
    )

    linetest_planner_node = Node(
        package='sim_car',
        executable=get_planner_spec('linetest').executable,
        name='linetest_planner_node',
        output='screen',
        parameters=[
            resolved_planner_config,
            resolved_controller_config,
            resolved_speed_control_config,
            {
                'use_sim_time': ParameterValue(
                    DEFAULT_USE_SIM_TIME,
                    value_type=bool,
                ),
                'topics.odom_topic': planner_odom_topic,
                'topics.tracked_cones_topic': planner_input_topic,
                'lap_tracking.target_laps': resolved_path_tracking_autostop_laps,
                'control.odom_lag_compensation_ms': ParameterValue(
                    LaunchConfiguration('planner_odom_lag_compensation_ms'),
                    value_type=float,
                ),
                'runtime.publish_rate_hz': ParameterValue(
                    LaunchConfiguration('planner_rate_hz'),
                    value_type=float,
                ),
            },
        ],
        condition=_planner_selected_condition('linetest'),
    )

    camera_debug_viewer_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='camera_debug_viewer',
        output='screen',
        arguments=[PythonExpression(["'", topic_prefix, "' + '/stereo/camera_debug'"])],
        condition=IfCondition(camera_debug_enabled_expr),
    )

    odom_tf_broadcaster_node = Node(
        package='sim_car',
        executable='odom_tf_broadcaster_node',
        name='odom_tf_broadcaster_node',
        output='screen',
        parameters=[{
                'use_sim_time': ParameterValue(
                    DEFAULT_USE_SIM_TIME,
                    value_type=bool,
                ),
                'odom_topic': planner_odom_topic,
                'default_frame_id': 'odom',
                'default_child_frame_id': 'base_footprint',
            }],
        condition=IfCondition(_planner_odom_delay_enabled_expr()),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='planner_debug_rviz',
        output='screen',
        arguments=['-d', resolved_rviz_config],
        parameters=[{
            'use_sim_time': ParameterValue(
                DEFAULT_USE_SIM_TIME,
                value_type=bool,
            ),
        }],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('rviz'), "'.lower() == 'true' and '",
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
        planner_odom_delay_ms_arg,
        planner_odom_lag_compensation_ms_arg,
        sensors_render_engine_arg,
        track_arg,
        world_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_z_arg,
        spawn_yaw_arg,
        steering_arg,
        physics_model_arg,
        sensor_pipeline_arg,
        measure_arg,
        planner_arg,
        controller_arg,
        rviz_arg,
        rviz_profile_arg,
        rviz_config_arg,
        perception_queue_size_arg,
        cuda_arg,
        stereo_arg,
        lidar_enabled_arg,
        cone_memory_enabled_arg,
        camera_range_arg,
        prefer_lidar_if_camera_missing_far_arg,
        allow_camera_fallback_near_arg,
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
        cuda_runtime_ld_library_path_arg,
        yolo_ultralytics_pythonpath_arg,
        measurement_config_arg,
        launch_args_validation,
        track_selection_setup,
        measurement_config_setup,
        rviz_config_setup,
        gazebo_launch,
        sim_nodes_launch,
        measurement_node,
        odom_delay_node,
        plotter_launch,
        logger_node,
        shutdown_on_logger_exit,
        steering_bridge_node,
        perception_node,
        camera_cone_evaluator_node,
        lidar_node,
        lidar_cone_evaluator_node,
        cone_memory_node,
        skidpad_router_node,
        shutdown_on_skidpad_router_exit,
        midpoint_planner_node,
        single_boundary_planner_node,
        corridor_planner_node,
        linetest_planner_node,
        camera_debug_viewer_node,
        odom_tf_broadcaster_node,
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


def _write_parameter_overlay(parameters: dict) -> str:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.safe_dump(parameters, temp_file, default_flow_style=False, sort_keys=True)
        return temp_file.name


def _resolve_track_bundle(sim_car_share: Path, track: str, planner: str, controller: str) -> dict[str, str]:
    normalized_track = str(track).strip().lower() or 'smalltrack'
    normalized_planner = str(planner).strip().lower()
    normalized_controller = str(controller).strip().lower() or 'stanley'
    planner_spec = get_planner_spec(normalized_planner)

    if normalized_track not in SUPPORTED_TRACKS:
        raise RuntimeError(
            "Unsupported launch argument track='%s'. Supported values: acceleration, skidpad, smalltrack"
            % track
        )
    if normalized_planner not in SUPPORTED_PLANNERS:
        raise RuntimeError(
            "Unsupported launch argument planner='%s'. Supported values: midpoint, single_boundary, corridor, linetest, none"
            % planner
        )
    if not planner_allowed_for_track(planner_name=normalized_planner, track_name=normalized_track):
        allowed_tracks = sorted(planner_spec.allowed_tracks or ())
        allowed_text = ', '.join(allowed_tracks)
        raise RuntimeError(
            "planner='%s' is only supported with track='%s'" % (normalized_planner, allowed_text)
        )
    if normalized_controller not in SUPPORTED_CONTROLLERS:
        raise RuntimeError(
            "Unsupported launch argument controller='%s'. Supported values: stanley, pure_pursuit, none"
            % controller
        )

    config_dir = sim_car_share / 'config' / normalized_track
    bundle = {
        'track': normalized_track,
        'planner': normalized_planner,
        'controller': normalized_controller,
        'planner_spec': planner_spec,
        'world': str(sim_car_share / 'worlds' / SUPPORTED_TRACKS[normalized_track]),
        'spawn_config': str(config_dir / 'spawn.yaml'),
        'planner_config': '',
        'controller_config': '',
    }
    if normalized_planner == 'linetest':
        bundle['planner_config'] = str(config_dir / 'linetest.yaml')
    if normalized_controller in {'stanley', 'pure_pursuit'}:
        bundle['controller_config'] = str(config_dir / f'{normalized_controller}_controller.yaml')
    return bundle


def _load_spawn_defaults(spawn_config_path: str) -> dict[str, str]:
    config = _load_yaml_file(spawn_config_path)
    values = config.get('spawn')
    if not isinstance(values, dict):
        raise RuntimeError(f"Spawn config missing 'spawn' mapping: {spawn_config_path}")

    resolved = {}
    for key in ('spawn_x', 'spawn_y', 'spawn_yaw'):
        if key not in values:
            raise RuntimeError(f"Spawn config missing '{key}': {spawn_config_path}")
        resolved[key] = str(values[key])
    return resolved


def _load_lap_tracking_defaults(spawn_config_path: str) -> dict[str, str]:
    config = _load_yaml_file(spawn_config_path)
    lap_tracking = config.get('lap_tracking')
    if not isinstance(lap_tracking, dict):
        return {
            'autostop_laps': '0',
        }
    autostop_laps = max(0, int(lap_tracking.get('auto_suspend_after_laps', 0)))
    return {
        'autostop_laps': str(autostop_laps),
    }


def _load_speed_control_defaults(spawn_config_path: str) -> dict[str, float]:
    config = _load_yaml_file(spawn_config_path)
    speed_control = config.get('speed_control')
    if not isinstance(speed_control, dict):
        raise RuntimeError(f"Spawn config missing 'speed_control' mapping: {spawn_config_path}")

    resolved = {}
    for key in (
        'speed_min_mps',
        'speed_max_mps',
        'curvature_speed_gain',
        'lowpass_speed_alpha',
    ):
        if key not in speed_control:
            raise RuntimeError(f"Spawn config missing 'speed_control.{key}': {spawn_config_path}")
        resolved[key] = float(speed_control[key])
    return resolved


def _load_planner_limit_overrides(spawn_config_path: str) -> dict[str, float]:
    config = _load_yaml_file(spawn_config_path)
    planner_limits = config.get('planner_limits')
    if not isinstance(planner_limits, dict):
        return {}

    max_planner_length = planner_limits.get('max_planner_length_m')
    if max_planner_length is None:
        return {}

    max_planner_length_m = float(max_planner_length)
    return {
        'filtering.max_cone_range_m': max_planner_length_m,
        'centerline.max_path_length_m': max_planner_length_m,
        'filtering.planning_horizon_m': max_planner_length_m,
    }


def _load_yaml_file(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f'Failed reading YAML config {config_path}: {exc}') from exc
    if not isinstance(config, dict):
        raise RuntimeError(f'YAML config must contain a mapping at the top level: {config_path}')
    return config


def _abbreviated_run_id_prefix(track: str, planner: str, controller: str, speed_max_mps: float) -> str:
    normalized_track = str(track).strip().lower() or 'smalltrack'
    normalized_planner = str(planner).strip().lower() or 'midpoint'
    normalized_controller = str(controller).strip().lower() or 'stanley'

    return '_'.join([
        RUN_ID_TRACK_ABBREVIATIONS.get(normalized_track, normalized_track),
        str(int(round(float(speed_max_mps)))),
        RUN_ID_PLANNER_ABBREVIATIONS.get(normalized_planner, normalized_planner),
        RUN_ID_CONTROLLER_ABBREVIATIONS.get(normalized_controller, normalized_controller),
        SCAN2D_RUN_ID_SUFFIX,
    ])


def _resolve_launch_selection(
    sim_car_share: Path,
    *,
    track: str,
    planner: str,
    world_override: str = '',
    spawn_x_override: str = '',
    spawn_y_override: str = '',
    spawn_yaw_override: str = '',
    controller_override: str = '',
) -> LaunchSelection:
    normalized_controller = str(controller_override).strip().lower() or 'stanley'
    bundle = _resolve_track_bundle(sim_car_share, track, planner, normalized_controller)
    spawn_defaults = _load_spawn_defaults(bundle['spawn_config'])
    lap_tracking_defaults = _load_lap_tracking_defaults(bundle['spawn_config'])
    speed_control_defaults = _load_speed_control_defaults(bundle['spawn_config'])
    planner_limit_overrides = _load_planner_limit_overrides(bundle['spawn_config'])
    selection = LaunchSelection(
        track=bundle['track'],
        planner=bundle['planner'],
        controller=bundle['controller'],
        world=str(world_override).strip() or bundle['world'],
        planner_config=bundle['planner_config'],
        controller_config=bundle['controller_config'],
        spawn_config=bundle['spawn_config'],
        spawn_x=str(spawn_x_override).strip() or spawn_defaults['spawn_x'],
        spawn_y=str(spawn_y_override).strip() or spawn_defaults['spawn_y'],
        spawn_yaw=str(spawn_yaw_override).strip() or spawn_defaults['spawn_yaw'],
        path_tracking_autostop_laps=lap_tracking_defaults['autostop_laps'],
        speed_control=speed_control_defaults,
        planner_limits=planner_limit_overrides,
        planner_diagnostics_topic=bundle['planner_spec'].diagnostics_topic,
        planner_default_rviz_profile=bundle['planner_spec'].default_rviz_profile,
    )

    if selection.planner == 'linetest' and not Path(selection.planner_config).exists():
        raise RuntimeError(f'Planner config does not exist: {selection.planner_config}')
    if selection.controller in {'stanley', 'pure_pursuit'} and not Path(selection.controller_config).exists():
        raise RuntimeError(f'Controller config does not exist: {selection.controller_config}')
    if not Path(selection.spawn_config).exists():
        raise RuntimeError(f'Spawn config does not exist: {selection.spawn_config}')

    return selection


def _configure_track_selection(context, *_args, **_kwargs):
    sim_car_share = Path(get_package_share_directory('sim_car'))
    selection = _resolve_launch_selection(
        sim_car_share,
        track=LaunchConfiguration('track').perform(context),
        planner=LaunchConfiguration('planner').perform(context),
        world_override=LaunchConfiguration('world').perform(context),
        spawn_x_override=LaunchConfiguration('spawn_x').perform(context),
        spawn_y_override=LaunchConfiguration('spawn_y').perform(context),
        spawn_yaw_override=LaunchConfiguration('spawn_yaw').perform(context),
        controller_override=LaunchConfiguration('controller').perform(context),
    )
    run_id_prefix = _abbreviated_run_id_prefix(
        selection.track,
        selection.planner,
        selection.controller,
        selection.speed_control['speed_max_mps'],
    )

    planner_config_path = selection.planner_config if selection.planner == 'linetest' else _write_parameter_overlay({})

    if selection.planner in CONFIGURED_PLANNERS:
        if selection.controller == 'none':
            controller_config_path = _write_parameter_overlay(selection.controller_overlay_parameters())
        else:
            controller_config_path = selection.controller_config
        speed_control_config_path = _write_parameter_overlay(selection.speed_control_overlay_parameters())
        planner_overlay_parameters = selection.planner_overlay_parameters()
        if planner_overlay_parameters:
            planner_config_path = _write_parameter_overlay(planner_overlay_parameters)
    else:
        controller_config_path = _write_parameter_overlay({})
        speed_control_config_path = _write_parameter_overlay({})

    return [
        SetLaunchConfiguration('resolved_world', selection.world),
        SetLaunchConfiguration('resolved_spawn_x', selection.spawn_x),
        SetLaunchConfiguration('resolved_spawn_y', selection.spawn_y),
        SetLaunchConfiguration('resolved_spawn_yaw', selection.spawn_yaw),
        SetLaunchConfiguration('resolved_planner_config', planner_config_path),
        SetLaunchConfiguration('resolved_controller_config', controller_config_path),
        SetLaunchConfiguration('resolved_speed_control_config', speed_control_config_path),
        SetLaunchConfiguration('resolved_planner_diagnostics_topic', selection.planner_diagnostics_topic),
        SetLaunchConfiguration(
            'resolved_path_tracking_autostop_laps',
            selection.path_tracking_autostop_laps,
        ),
        SetLaunchConfiguration(
            'resolved_shutdown_on_logger_exit',
            'true'
            if (
                selection.track == 'smalltrack'
                and int(selection.path_tracking_autostop_laps) > 0
            )
            else 'false',
        ),
        SetLaunchConfiguration('resolved_run_id_prefix', run_id_prefix),
    ]


def _configure_measurement_config(context, *_args, **_kwargs):
    config_path = LaunchConfiguration('measurement_config').perform(context)
    planner_rate_hz = float(LaunchConfiguration('planner_rate_hz').perform(context))
    resolved_config = _write_rate_adjusted_measurement_config(config_path, planner_rate_hz)
    return [SetLaunchConfiguration('resolved_measurement_config', resolved_config)]


def _configure_rviz_config(context, *_args, **_kwargs):
    explicit_config = LaunchConfiguration('rviz_config').perform(context).strip()
    if explicit_config:
        base_config = explicit_config
    else:
        sim_car_share = get_package_share_directory('sim_car')
        rviz_profile = LaunchConfiguration('rviz_profile').perform(context).strip().lower()
        planner = LaunchConfiguration('planner').perform(context).strip().lower()
        profile_to_filename = {
            'clean': 'driving_clean.rviz',
            'planner_debug': 'planner_debug.rviz',
            'debug': 'planner_debug.rviz',
            'midpoint': 'midpoint_planner.rviz',
            'single_boundary': 'single_boundary_planner.rviz',
            'corridor': 'corridor_planner.rviz',
            'linetest': 'planner_debug.rviz',
            'none': 'driving_clean.rviz',
        }
        if rviz_profile in {'planner', 'auto'}:
            planner_profile = get_planner_spec(planner).default_rviz_profile if planner in SUPPORTED_PLANNERS else 'clean'
            resolved_filename = profile_to_filename.get(planner_profile, 'driving_clean.rviz')
        else:
            resolved_filename = profile_to_filename.get(rviz_profile, 'driving_clean.rviz')
        base_config = str(Path(sim_car_share) / 'rviz' / resolved_filename)

    return [SetLaunchConfiguration('resolved_rviz_config', base_config)]


def _validate_planner_and_controller_args(context, *_args, **_kwargs):
    track = LaunchConfiguration('track').perform(context).strip().lower()
    planner = LaunchConfiguration('planner').perform(context).strip().lower()
    controller = LaunchConfiguration('controller').perform(context).strip().lower() or 'stanley'

    if track not in SUPPORTED_TRACKS:
        raise RuntimeError(
            "Unsupported launch argument track='%s'. Supported values: acceleration, skidpad, smalltrack"
            % track
        )
    if planner not in SUPPORTED_PLANNERS:
        raise RuntimeError(
            "Unsupported launch argument planner='%s'. Supported values: midpoint, single_boundary, corridor, linetest, none"
            % planner
        )
    if not planner_allowed_for_track(planner_name=planner, track_name=track):
        planner_spec = get_planner_spec(planner)
        allowed_tracks = sorted(planner_spec.allowed_tracks or ())
        allowed_text = ', '.join(allowed_tracks)
        raise RuntimeError("planner='%s' is only supported with track='%s'" % (planner, allowed_text))
    if controller not in SUPPORTED_CONTROLLERS:
        raise RuntimeError(
            "Unsupported launch argument controller='%s'. Supported values: stanley, pure_pursuit, none"
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
