#!/usr/bin/env python3
"""Launch the vehicle_plotter logger and session manager."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch.conditions import IfCondition
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
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
    path_tracking_eval_track_name_arg = DeclareLaunchArgument(
        "path_tracking_eval_track_name",
        default_value="",
        description="Track name used for path-tracking overlay specialization",
    )
    path_tracking_eval_autostop_laps_arg = DeclareLaunchArgument(
        "path_tracking_eval_autostop_laps",
        default_value="0",
        description="Automatically end the run after this many counted laps (0 disables)",
    )
    off_track_autostop_planner_diag_topic_arg = DeclareLaunchArgument(
        "off_track_autostop_planner_diag_topic",
        default_value="/midpoint_planner/diagnostics",
        description="Planner diagnostics topic used by off-track autostop",
    )
    shutdown_on_logger_exit_arg = DeclareLaunchArgument(
        "shutdown_on_logger_exit",
        default_value="false",
        description="Shut down the launch when the logger process exits",
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
    state_topic_arg = DeclareLaunchArgument(
        "state_topic",
        default_value="/vehicle_plotter/state",
        description="VehicleState topic published by plotter_node and consumed by logger",
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
    run_id_prefix_arg = DeclareLaunchArgument(
        "run_id_prefix",
        default_value="",
        description="Optional run id prefix before the timestamp",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time from /clock topic",
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

    session_manager_node = Node(
        package="vehicle_plotter",
        executable="session_manager_node",
        name="session_manager",
        output="screen",
        parameters=[{
            "broadcast_rate_hz": 1.0,
            "run_id_prefix": LaunchConfiguration("run_id_prefix"),
        }],
    )

    plotter_node = Node(
        package="vehicle_plotter",
        executable="plotter_node",
        name="plotter_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_state_logging")),
        parameters=[{
            "adapter": "gazebo",
            "state_topic": LaunchConfiguration("state_topic"),
            "output_rate_hz": 50.0,
            "use_sim_time": ParameterValue(
                LaunchConfiguration("use_sim_time"),
                value_type=bool,
            ),
            "auto_export_dashboard": True,
            "dashboard_filename": "virtual_sensors.png",
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
            "state_topic": LaunchConfiguration("state_topic"),
            "enable_state_logging": ParameterValue(
                LaunchConfiguration("enable_state_logging"),
                value_type=bool,
            ),
            "auto_plot_on_shutdown": True,
            "camera_cone_eval_topic": LaunchConfiguration("camera_cone_eval_topic"),
            "lidar_cone_eval_topic": LaunchConfiguration("lidar_cone_eval_topic"),
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
            "path_tracking_eval_track_name": LaunchConfiguration(
                "path_tracking_eval_track_name"
            ),
            "path_tracking_eval_autostop_laps": ParameterValue(
                LaunchConfiguration("path_tracking_eval_autostop_laps"),
                value_type=int,
            ),
            "off_track_autostop_planner_diag_topic": LaunchConfiguration(
                "off_track_autostop_planner_diag_topic"
            ),
            "use_sim_time": False,
        }],
        on_exit=[
            EmitEvent(
                event=Shutdown(reason="logger exited"),
                condition=IfCondition(LaunchConfiguration("shutdown_on_logger_exit")),
            ),
        ],
    )

    return LaunchDescription([
        path_tracking_eval_enabled_arg,
        path_tracking_eval_gt_track_topic_arg,
        path_tracking_eval_odom_topic_arg,
        path_tracking_eval_planner_path_topic_arg,
        path_tracking_eval_track_name_arg,
        path_tracking_eval_autostop_laps_arg,
        off_track_autostop_planner_diag_topic_arg,
        shutdown_on_logger_exit_arg,
        enable_log_arg,
        enable_state_logging_arg,
        state_topic_arg,
        log_format_arg,
        log_path_arg,
        run_id_prefix_arg,
        use_sim_time_arg,
        camera_cone_eval_topic_arg,
        lidar_cone_eval_topic_arg,
        session_manager_node,
        plotter_node,
        logger_node,
    ])
