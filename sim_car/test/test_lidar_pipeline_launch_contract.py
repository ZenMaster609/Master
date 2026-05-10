from __future__ import annotations

import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'full_sim_launch.launch.py'
GAZEBO_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'gazebo_sim.launch.py'
URDF = REPO_ROOT / 'sim_car' / 'urdf' / 'eufs_car.urdf.xacro'


def test_full_launch_uses_scan_lidar_only() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'lidar_pipeline'" not in content
    assert "executable='lidar_node'" in content
    assert "executable='pointcloud_lidar_node'" not in content
    assert "def _lidar_enabled_condition():" in content
    assert "SCAN2D_LIDAR_PIPELINE_NAME = 'scan2d'" in content
    assert "launch_parameters_snapshot['lidar_pipeline'] = SCAN2D_LIDAR_PIPELINE_NAME" in content


def test_full_launch_has_no_pointcloud_rviz_debug_injection() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "rviz_raw_pointcloud_debug" not in content
    assert "rviz_filtered_pointcloud_debug" not in content
    assert "_write_rviz_config_with_pointcloud_debugs" not in content
    assert "_upsert_rviz_pointcloud_display" not in content
    assert "Raw PointCloud Debug" not in content
    assert "Filtered PointCloud Debug" not in content


def test_full_launch_uses_config_defaults_without_pointcloud_overlays() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "def _cone_memory_pipeline_parameters():" not in content
    assert "def _planner_pipeline_parameters():" not in content
    assert "POINTCLOUD3D_CONE_MEMORY_CONFIRM_HITS" not in content
    assert "POINTCLOUD3D_PLANNER_MIN_CONFIDENCE" not in content
    assert "**_cone_memory_pipeline_parameters()" not in content
    assert "**_planner_pipeline_parameters()" not in content


def test_planner_odom_delay_is_opt_in_for_closed_loop_stability() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "'planner_odom_delay_ms',\n        default_value='0.0'" in content
    assert "'planner_odom_lag_compensation_ms',\n        default_value='0.0'" in content
    assert "def _planner_odom_delay_enabled_expr():" in content
    assert "planner_odom_topic = PythonExpression([" in content
    assert "'/sim/odom_delayed' if " in content
    assert " else '/sim/odom'" in content
    assert "'control.odom_lag_compensation_ms': ParameterValue(" in content
    assert "LaunchConfiguration('planner_odom_lag_compensation_ms')" in content
    assert content.count("'control.odom_lag_compensation_ms': ParameterValue(") == 4
    assert content.count("condition=IfCondition(_planner_odom_delay_enabled_expr())") == 2


def test_cone_memory_defaults_remain_scan_lidar_defaults() -> None:
    config_content = (REPO_ROOT / 'sim_car' / 'config' / 'cone_memory.yaml').read_text(encoding='utf-8')
    launch_content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "min_seen_count: 3" in config_content
    assert "confirm_hits: 3" in config_content
    assert "'memory.confirm_hits': ParameterValue(" not in launch_content
    assert "'memory.min_seen_count': ParameterValue(" not in launch_content


def test_gazebo_launch_bridges_only_laserscan() -> None:
    content = GAZEBO_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n            'lidar_pipeline'" not in content
    assert "PointCloud2" not in content
    assert "lidar/points" not in content
    assert "f'{topic_prefix}/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'" in content
    assert "f'lidar_pipeline:=" not in content


def test_urdf_contains_scan_lidar_profile_only() -> None:
    content = URDF.read_text(encoding='utf-8')

    assert "<xacro:arg name=\"lidar_pipeline\"" not in content
    assert "lidar_pipeline_mode" not in content
    assert "lidar_os1" not in content
    assert "os1_lidar" not in content
    assert "<xacro:property name=\"lidar_scan_mount_xyz\" value=\"1.45 0 -0.09\"/>" in content
    assert "<topic>$(arg topic_prefix)/lidar</topic>" in content
    assert "<samples>${legacy_lidar_horizontal_samples}</samples>" in content
