from __future__ import annotations

import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'full_sim_launch.launch.py'
GAZEBO_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'gazebo_sim.launch.py'
URDF = REPO_ROOT / 'sim_car' / 'urdf' / 'eufs_car.urdf.xacro'


def test_full_launch_declares_lidar_pipeline_and_pointcloud_node() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'lidar_pipeline'" in content
    assert "DeclareLaunchArgument(\n        'rviz_raw_pointcloud_debug'" in content
    assert "DeclareLaunchArgument(\n        'rviz_filtered_pointcloud_debug'" in content
    assert "default_value='pointcloud3d'" in content
    assert "executable='pointcloud_lidar_node'" in content
    assert "def _pointcloud3d_lidar_parameters(topic_prefix):" in content
    assert "def _lidar_pipeline_match_expr(pipeline: str):" in content
    assert "def _lidar_pipeline_enabled_condition(pipeline: str):" in content
    assert "_lidar_pipeline_match_expr('pointcloud3d')" in content
    assert "_lidar_pipeline_enabled_condition('scan2d')" in content


def test_full_launch_generates_rviz_config_with_pointcloud_debug_toggle() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "_write_rviz_config_with_pointcloud_debugs(" in content
    assert "def _pointcloud3d_debug_topics(topic_prefix) -> dict[str, str]:" in content
    assert "'raw_pointcloud_topic': f'{topic_prefix}/lidar/points'" in content
    assert "'filtered_pointcloud_topic': f'{topic_prefix}/lidar/points_filtered'" in content
    assert "name='Raw PointCloud Debug'" in content
    assert "name='Filtered PointCloud Debug'" in content


def test_full_launch_scopes_pointcloud3d_only_tuning_away_from_scan2d() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "def _lidar_pipeline_enabled_condition(pipeline: str):" in content
    assert "def _cone_memory_pipeline_parameters():" in content
    assert "def _planner_pipeline_parameters():" in content
    assert "def _pointcloud3d_debug_topics(topic_prefix) -> dict[str, str]:" in content
    assert "POINTCLOUD3D_CONE_MEMORY_CONFIRM_HITS = 1" in content
    assert "LEGACY_CONE_MEMORY_CONFIRM_HITS = 3" in content
    assert "POINTCLOUD3D_PLANNER_MIN_CONFIDENCE = 0.15" in content
    assert "LEGACY_PLANNER_MIN_CONFIDENCE = 0.3" in content
    assert "**_cone_memory_pipeline_parameters()" in content
    assert "**_planner_pipeline_parameters()" in content


def test_planner_odom_delay_is_opt_in_for_closed_loop_stability() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "'planner_odom_delay_ms',\n        default_value='0.0'" in content
    assert "def _planner_odom_delay_enabled_expr():" in content
    assert "planner_odom_topic = PythonExpression([" in content
    assert "'/sim/odom_delayed' if " in content
    assert " else '/sim/odom'" in content
    assert content.count("condition=IfCondition(_planner_odom_delay_enabled_expr())") == 2


def test_cone_memory_defaults_remain_legacy_and_launch_applies_pointcloud3d_overrides() -> None:
    config_content = (REPO_ROOT / 'sim_car' / 'config' / 'cone_memory.yaml').read_text(encoding='utf-8')
    launch_content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "min_seen_count: 3" in config_content
    assert "confirm_hits: 3" in config_content
    assert "POINTCLOUD3D_CONE_MEMORY_CONFIRM_HITS = 1" in launch_content


def test_gazebo_launch_bridges_pointcloud_when_requested() -> None:
    content = GAZEBO_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n            'lidar_pipeline'" in content
    assert "f'{topic_prefix}/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked'" in content
    assert "f'{topic_prefix}/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'" in content
    assert "f'lidar_pipeline:={lidar_pipeline}'" in content


def test_urdf_contains_dual_lidar_profiles() -> None:
    content = URDF.read_text(encoding='utf-8')

    assert "<xacro:arg name=\"lidar_pipeline\" default=\"pointcloud3d\"/>" in content
    assert "<xacro:property name=\"lidar_pipeline_mode\" value=\"$(arg lidar_pipeline)\"/>" in content
    assert "<xacro:if value=\"${lidar_pipeline_mode == 'scan2d'}\">" in content
    assert "<xacro:unless value=\"${lidar_pipeline_mode == 'scan2d'}\">" in content
    assert "<xacro:property name=\"os1_lidar_horizontal_samples\" value=\"2048\"/>" in content
    assert "<xacro:property name=\"os1_lidar_vertical_samples\" value=\"32\"/>" in content
    assert "<xacro:property name=\"lidar_scan_mount_xyz\" value=\"1.45 0 -0.09\"/>" in content
    assert "<xacro:property name=\"lidar_os1_mount_xyz\" value=\"1.855 0 0.32\"/>" in content
    assert "<topic>$(arg topic_prefix)/lidar</topic>" in content
