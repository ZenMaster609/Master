#!/usr/bin/env python3
"""Launch the vehicle_plotter live windows, logger, and rosbag controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    output_rate_arg = DeclareLaunchArgument(
        "output_rate_hz",
        default_value="50.0",
        description="Vehicle state output rate in Hz",
    )
    enable_sensor_plot_arg = DeclareLaunchArgument(
        "enable_sensor_plot",
        default_value="true",
        description="Enable the main sensor dashboard",
    )
    enable_cone_rmse_plot_arg = DeclareLaunchArgument(
        "enable_cone_rmse_plot",
        default_value="false",
        description="Enable the live cone RMSE window",
    )
    enable_controller_diagnostics_plot_arg = DeclareLaunchArgument(
        "enable_controller_diagnostics_plot",
        default_value="false",
        description="Enable the live controller diagnostics window",
    )
    enable_thesis_controller_diagnostics_plot_arg = DeclareLaunchArgument(
        "enable_thesis_controller_diagnostics_plot",
        default_value="false",
        description="Enable the live thesis controller diagnostics window",
    )
    plot_rate_arg = DeclareLaunchArgument(
        "plot_rate_hz",
        default_value="30.0",
        description="Sensor dashboard refresh rate in Hz",
    )
    dark_mode_arg = DeclareLaunchArgument(
        "dark_mode",
        default_value="true",
        description="Use dark theme for plots",
    )
    save_plots_on_exit_arg = DeclareLaunchArgument(
        "save_plots_on_exit",
        default_value="false",
        description="Save plot window images on shutdown",
    )
    save_plot_data_on_exit_arg = DeclareLaunchArgument(
        "save_plot_data_on_exit",
        default_value="true",
        description="Save plotted data CSVs on shutdown",
    )
    close_plots_on_shutdown_arg = DeclareLaunchArgument(
        "close_plots",
        default_value="true",
        description="Close plot windows when plotter nodes shut down",
    )
    camera_cone_eval_topic_arg = DeclareLaunchArgument(
        "camera_cone_eval_topic",
        default_value="/sim/stereo/eval",
        description="Camera cone evaluator topic prefix",
    )
    lidar_cone_eval_topic_arg = DeclareLaunchArgument(
        "lidar_cone_eval_topic",
        default_value="/sim/lidar/eval",
        description="LiDAR cone evaluator topic prefix",
    )
    camera_source_arg = DeclareLaunchArgument(
        "camera_source",
        default_value="monocular",
        description="Camera source label for cone RMSE window",
    )
    controller_diagnostics_enabled_arg = DeclareLaunchArgument(
        "controller_diagnostics_enabled",
        default_value="false",
        description="Enable controller diagnostics CSV and summaries in the logger",
    )
    thesis_controller_diagnostics_enabled_arg = DeclareLaunchArgument(
        "thesis_controller_diagnostics_enabled",
        default_value="false",
        description="Enable thesis controller diagnostics CSV and summaries in the logger",
    )
    path_tracking_eval_enabled_arg = DeclareLaunchArgument(
        "path_tracking_eval_enabled",
        default_value="false",
        description="Enable GT midline path-tracking evaluation in the logger",
    )
    path_tracking_eval_gt_track_topic_arg = DeclareLaunchArgument(
        "path_tracking_eval_gt_track_topic",
        default_value="/ground_truth/track",
        description="Ground-truth track cones topic for path-tracking evaluation",
    )
    path_tracking_eval_odom_topic_arg = DeclareLaunchArgument(
        "path_tracking_eval_odom_topic",
        default_value="/sim/odom",
        description="Odometry topic for path-tracking evaluation",
    )
    path_tracking_eval_planner_path_topic_arg = DeclareLaunchArgument(
        "path_tracking_eval_planner_path_topic",
        default_value="/planned_centerline",
        description="Planner path topic for path-tracking evaluation",
    )
    controller_diagnostics_rate_hz_arg = DeclareLaunchArgument(
        "controller_diagnostics_rate_hz",
        default_value="50.0",
        description="Controller diagnostics sampling rate in Hz",
    )
    controller_diagnostics_cmd_topic_arg = DeclareLaunchArgument(
        "controller_diagnostics_cmd_topic",
        default_value="/cmd",
        description="Ackermann command topic for controller diagnostics",
    )
    controller_diagnostics_steering_topic_arg = DeclareLaunchArgument(
        "controller_diagnostics_steering_topic",
        default_value="/sim/steering_angle",
        description="Measured steering angle topic for controller diagnostics",
    )
    controller_diagnostics_joint_states_topic_arg = DeclareLaunchArgument(
        "controller_diagnostics_joint_states_topic",
        default_value="/sim/raw/joint_states",
        description="Joint states topic used as fallback steering source",
    )
    controller_diagnostics_odom_topic_arg = DeclareLaunchArgument(
        "controller_diagnostics_odom_topic",
        default_value="/sim/odom",
        description="Odometry topic for controller diagnostics",
    )
    controller_diagnostics_path_topic_arg = DeclareLaunchArgument(
        "controller_diagnostics_path_topic",
        default_value="/planned_centerline",
        description="Reference path topic for controller diagnostics",
    )
    controller_diagnostics_planner_diag_topic_arg = DeclareLaunchArgument(
        "controller_diagnostics_planner_diag_topic",
        default_value="/delaunay_planner/diagnostics",
        description="Planner diagnostics topic for controller diagnostics",
    )
    controller_diagnostics_live_plot_enabled_arg = DeclareLaunchArgument(
        "controller_diagnostics_live_plot_enabled",
        default_value="false",
        description="Enable the live controller diagnostics window",
    )
    controller_diagnostics_live_plot_rate_hz_arg = DeclareLaunchArgument(
        "controller_diagnostics_live_plot_rate_hz",
        default_value="10.0",
        description="Refresh rate for the live controller diagnostics window",
    )
    controller_diagnostics_live_buffer_sec_arg = DeclareLaunchArgument(
        "controller_diagnostics_live_buffer_sec",
        default_value="30.0",
        description="History window in seconds for the live controller diagnostics window",
    )
    enable_log_arg = DeclareLaunchArgument(
        "enable_log",
        default_value="true",
        description="Enable data logging to file",
    )
    enable_state_logging_arg = DeclareLaunchArgument(
        "enable_state_logging",
        default_value="true",
        description="Subscribe to vehicle_plotter/state for vehicle-state logging",
    )
    log_format_arg = DeclareLaunchArgument(
        "log_format",
        default_value="parquet",
        description="Log format: parquet or csv",
    )
    log_path_arg = DeclareLaunchArgument(
        "log_path",
        default_value="",
        description="Base path for log files (empty = auto-detect ./multidata)",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time from /clock topic",
    )
    sensor_config_arg = DeclareLaunchArgument(
        "sensor_config",
        default_value="",
        description="Path to sim_car sensor_config.yaml (empty = auto-detect)",
    )
    enable_rosbag_arg = DeclareLaunchArgument(
        "enable_rosbag",
        default_value="true",
        description="Enable rosbag recording",
    )

    session_manager_node = Node(
        package="vehicle_plotter",
        executable="session_manager_node",
        name="session_manager",
        output="screen",
        parameters=[{"broadcast_rate_hz": 1.0}],
    )

    plotter_node = Node(
        package="vehicle_plotter",
        executable="plotter_node",
        name="plotter",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_sensor_plot")),
        parameters=[{
            "backend": "pyqtgraph",
            "update_rate_hz": ParameterValue(
                LaunchConfiguration("plot_rate_hz"),
                value_type=float,
            ),
            "state_output_rate_hz": ParameterValue(
                LaunchConfiguration("output_rate_hz"),
                value_type=float,
            ),
            "dark_mode": LaunchConfiguration("dark_mode"),
            "enable_gui": True,
            "direct_from_sensors": True,
            "plot_layout": "all",
            "window_title": "Vehicle Plotter",
            "save_plots_on_exit": ParameterValue(
                LaunchConfiguration("save_plots_on_exit"),
                value_type=bool,
            ),
            "save_plot_data_on_exit": ParameterValue(
                LaunchConfiguration("save_plot_data_on_exit"),
                value_type=bool,
            ),
            "close_plots_on_shutdown": ParameterValue(
                LaunchConfiguration("close_plots"),
                value_type=bool,
            ),
            "sensor_config_path": LaunchConfiguration("sensor_config"),
            "use_sim_time": False,
        }],
    )

    cone_rmse_plot_node = Node(
        package="vehicle_plotter",
        executable="cone_rmse_plot_node",
        name="cone_rmse_plot",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_cone_rmse_plot")),
        parameters=[{
            "camera_eval_topic": LaunchConfiguration("camera_cone_eval_topic"),
            "lidar_eval_topic": LaunchConfiguration("lidar_cone_eval_topic"),
            "camera_source": LaunchConfiguration("camera_source"),
            "update_period_sec": 0.2,
            "use_sim_time": False,
        }],
    )

    controller_diagnostics_plot_node = Node(
        package="vehicle_plotter",
        executable="controller_diagnostics_plot_node",
        name="controller_diagnostics_plot",
        output="screen",
        condition=IfCondition(
            PythonExpression([
                "('", LaunchConfiguration("controller_diagnostics_enabled"),
                "'.lower() == 'true') and ('",
                LaunchConfiguration("enable_controller_diagnostics_plot"),
                "'.lower() == 'true')",
            ])
        ),
        parameters=[{
            "controller_diagnostics_rate_hz": ParameterValue(
                LaunchConfiguration("controller_diagnostics_rate_hz"),
                value_type=float,
            ),
            "controller_diagnostics_cmd_topic": LaunchConfiguration(
                "controller_diagnostics_cmd_topic"
            ),
            "controller_diagnostics_steering_topic": LaunchConfiguration(
                "controller_diagnostics_steering_topic"
            ),
            "controller_diagnostics_joint_states_topic": LaunchConfiguration(
                "controller_diagnostics_joint_states_topic"
            ),
            "controller_diagnostics_odom_topic": LaunchConfiguration(
                "controller_diagnostics_odom_topic"
            ),
            "controller_diagnostics_path_topic": LaunchConfiguration(
                "controller_diagnostics_path_topic"
            ),
            "controller_diagnostics_planner_diag_topic": LaunchConfiguration(
                "controller_diagnostics_planner_diag_topic"
            ),
            "controller_diagnostics_live_plot_rate_hz": ParameterValue(
                LaunchConfiguration("controller_diagnostics_live_plot_rate_hz"),
                value_type=float,
            ),
            "controller_diagnostics_live_buffer_sec": ParameterValue(
                LaunchConfiguration("controller_diagnostics_live_buffer_sec"),
                value_type=float,
            ),
            "use_sim_time": False,
        }],
    )

    thesis_controller_diagnostics_plot_node = Node(
        package="vehicle_plotter",
        executable="thesis_controller_diagnostics_plot_node",
        name="thesis_controller_diagnostics_plot",
        output="screen",
        condition=IfCondition(
            PythonExpression([
                "('", LaunchConfiguration("thesis_controller_diagnostics_enabled"),
                "'.lower() == 'true') and ('",
                LaunchConfiguration("enable_thesis_controller_diagnostics_plot"),
                "'.lower() == 'true')",
            ])
        ),
        parameters=[{
            "controller_diagnostics_rate_hz": ParameterValue(
                LaunchConfiguration("controller_diagnostics_rate_hz"),
                value_type=float,
            ),
            "controller_diagnostics_cmd_topic": LaunchConfiguration(
                "controller_diagnostics_cmd_topic"
            ),
            "controller_diagnostics_steering_topic": LaunchConfiguration(
                "controller_diagnostics_steering_topic"
            ),
            "controller_diagnostics_joint_states_topic": LaunchConfiguration(
                "controller_diagnostics_joint_states_topic"
            ),
            "controller_diagnostics_odom_topic": LaunchConfiguration(
                "controller_diagnostics_odom_topic"
            ),
            "controller_diagnostics_path_topic": LaunchConfiguration(
                "controller_diagnostics_path_topic"
            ),
            "controller_diagnostics_planner_diag_topic": LaunchConfiguration(
                "controller_diagnostics_planner_diag_topic"
            ),
            "controller_diagnostics_live_plot_rate_hz": ParameterValue(
                LaunchConfiguration("controller_diagnostics_live_plot_rate_hz"),
                value_type=float,
            ),
            "controller_diagnostics_live_buffer_sec": ParameterValue(
                LaunchConfiguration("controller_diagnostics_live_buffer_sec"),
                value_type=float,
            ),
            "use_sim_time": False,
        }],
    )

    logger_node = Node(
        package="vehicle_plotter",
        executable="logger_node",
        name="logger",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_log")),
        parameters=[{
            "format": LaunchConfiguration("log_format"),
            "base_path": LaunchConfiguration("log_path"),
            "flush_interval_sec": 5.0,
            "buffer_size": 1000,
            "adapter": "gazebo",
            "enable_state_logging": ParameterValue(
                LaunchConfiguration("enable_state_logging"),
                value_type=bool,
            ),
            "auto_plot_on_shutdown": True,
            "camera_cone_eval_topic": LaunchConfiguration("camera_cone_eval_topic"),
            "lidar_cone_eval_topic": LaunchConfiguration("lidar_cone_eval_topic"),
            "controller_diagnostics_enabled": ParameterValue(
                LaunchConfiguration("controller_diagnostics_enabled"),
                value_type=bool,
            ),
            "thesis_controller_diagnostics_enabled": ParameterValue(
                LaunchConfiguration("thesis_controller_diagnostics_enabled"),
                value_type=bool,
            ),
            "path_tracking_eval_enabled": ParameterValue(
                LaunchConfiguration("path_tracking_eval_enabled"),
                value_type=bool,
            ),
            "path_tracking_eval_gt_track_topic": LaunchConfiguration(
                "path_tracking_eval_gt_track_topic"
            ),
            "path_tracking_eval_odom_topic": LaunchConfiguration(
                "path_tracking_eval_odom_topic"
            ),
            "path_tracking_eval_planner_path_topic": LaunchConfiguration(
                "path_tracking_eval_planner_path_topic"
            ),
            "controller_diagnostics_rate_hz": ParameterValue(
                LaunchConfiguration("controller_diagnostics_rate_hz"),
                value_type=float,
            ),
            "controller_diagnostics_cmd_topic": LaunchConfiguration(
                "controller_diagnostics_cmd_topic"
            ),
            "controller_diagnostics_steering_topic": LaunchConfiguration(
                "controller_diagnostics_steering_topic"
            ),
            "controller_diagnostics_joint_states_topic": LaunchConfiguration(
                "controller_diagnostics_joint_states_topic"
            ),
            "controller_diagnostics_odom_topic": LaunchConfiguration(
                "controller_diagnostics_odom_topic"
            ),
            "controller_diagnostics_path_topic": LaunchConfiguration(
                "controller_diagnostics_path_topic"
            ),
            "controller_diagnostics_planner_diag_topic": LaunchConfiguration(
                "controller_diagnostics_planner_diag_topic"
            ),
            "use_sim_time": False,
        }],
    )

    rosbag_controller_node = Node(
        package="vehicle_plotter",
        executable="rosbag_controller_node",
        name="rosbag_controller",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_rosbag")),
        parameters=[{
            "mode": "simulation",
            "compression": "zstd",
            "wait_for_session": True,
            "session_timeout_sec": 5.0,
        }],
    )

    return LaunchDescription([
        output_rate_arg,
        enable_sensor_plot_arg,
        enable_cone_rmse_plot_arg,
        enable_controller_diagnostics_plot_arg,
        enable_thesis_controller_diagnostics_plot_arg,
        plot_rate_arg,
        dark_mode_arg,
        save_plots_on_exit_arg,
        save_plot_data_on_exit_arg,
        close_plots_on_shutdown_arg,
        camera_cone_eval_topic_arg,
        lidar_cone_eval_topic_arg,
        camera_source_arg,
        controller_diagnostics_enabled_arg,
        thesis_controller_diagnostics_enabled_arg,
        path_tracking_eval_enabled_arg,
        path_tracking_eval_gt_track_topic_arg,
        path_tracking_eval_odom_topic_arg,
        path_tracking_eval_planner_path_topic_arg,
        controller_diagnostics_rate_hz_arg,
        controller_diagnostics_cmd_topic_arg,
        controller_diagnostics_steering_topic_arg,
        controller_diagnostics_joint_states_topic_arg,
        controller_diagnostics_odom_topic_arg,
        controller_diagnostics_path_topic_arg,
        controller_diagnostics_planner_diag_topic_arg,
        controller_diagnostics_live_plot_enabled_arg,
        controller_diagnostics_live_plot_rate_hz_arg,
        controller_diagnostics_live_buffer_sec_arg,
        enable_log_arg,
        enable_state_logging_arg,
        log_format_arg,
        log_path_arg,
        use_sim_time_arg,
        sensor_config_arg,
        enable_rosbag_arg,
        session_manager_node,
        plotter_node,
        cone_rmse_plot_node,
        controller_diagnostics_plot_node,
        thesis_controller_diagnostics_plot_node,
        logger_node,
        rosbag_controller_node,
    ])
