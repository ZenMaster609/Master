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
    assert "'pointcloud_topic': PythonExpression([\"'\", topic_prefix, \"' + '/lidar/points'\"])" in content
    assert "'.lower() == 'pointcloud3d'" in content
    assert "'.lower() == 'scan2d'" in content


def test_full_launch_generates_rviz_config_with_pointcloud_debug_toggle() -> None:
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "_write_rviz_config_with_pointcloud_debugs(" in content
    assert "pointcloud_topic = f'{topic_prefix}/lidar/points'" in content
    assert "filtered_pointcloud_topic = f'{topic_prefix}/lidar/points_filtered'" in content
    assert "name='Raw PointCloud Debug'" in content
    assert "name='Filtered PointCloud Debug'" in content


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
    assert "<samples>32</samples>" in content
    assert "<max_angle>3.14159265</max_angle>" in content
    assert "<topic>$(arg topic_prefix)/lidar</topic>" in content
