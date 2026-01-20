#!/usr/bin/env python3
"""
Launch file for Gazebo Fortress simulation with ros_gz bridge.
Updated from Gazebo Classic to modern Gazebo (Fortress).
"""

import os
import tempfile
import xml.etree.ElementTree as ET
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_sim_car = get_package_share_directory('sim_car')
    world_file = os.path.join(pkg_sim_car, 'worlds', 'test_world.sdf')

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default=world_file)
    headless = LaunchConfiguration('headless', default='false')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'world',
            default_value=world_file,
            description='Full path to world file to load'
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run simulation without GUI (server only)'
        ),
        OpaqueFunction(function=_launch_simulation),
    ])


def _launch_simulation(context, *args, **kwargs):
    pkg_sim_car = get_package_share_directory('sim_car')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    config_path = os.path.join(pkg_sim_car, 'config', 'sensor_config.yaml')
    sensor_config = _load_sensor_config(config_path)

    step_rate = _get_config_value(sensor_config, ['simulation', 'step_rate_hz'], 50.0)
    max_step_size = 1.0 / step_rate if step_rate > 0 else 0.02
    imu_rate = _get_config_value(sensor_config, ['sensors', 'real', 'imu', 'frequency_hz'], 50.0)
    gnss_rate = _get_config_value(sensor_config, ['sensors', 'real', 'gnss', 'frequency_hz'], 50.0)

    world_path = LaunchConfiguration('world').perform(context)
    urdf_path = os.path.join(pkg_sim_car, 'urdf', 'car.urdf')

    updated_world = _write_updated_world(world_path, max_step_size)
    robot_desc = _build_robot_description(urdf_path, imu_rate, gnss_rate)

    use_sim_time = LaunchConfiguration('use_sim_time')
    headless = LaunchConfiguration('headless')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s ', updated_world]}.items(),
        condition=IfCondition(headless)
    )

    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', updated_world]}.items(),
        condition=UnlessCondition(headless)
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'sim_car',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_desc
        }]
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/sim/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/sim/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
            '/sim/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat',
            '/sim/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/sim/rear_left_wheel_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/sim/rear_right_wheel_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/sim/steering_fl_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/sim/steering_fr_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/sim/suspension_fl_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/sim/suspension_fr_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/sim/suspension_rl_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/sim/suspension_rr_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
        ],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return [
        gazebo,
        gazebo_gui,
        robot_state_publisher,
        joint_state_publisher,
        spawn_entity,
        bridge,
    ]


def _load_sensor_config(config_path):
    try:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _get_config_value(config, keys, default):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value if value is not None else default


def _write_updated_world(world_path, max_step_size):
    try:
        tree = ET.parse(world_path)
        root = tree.getroot()
    except ET.ParseError:
        return world_path

    physics = root.find('.//world/physics')
    if physics is None:
        world_elem = root.find('.//world')
        if world_elem is None:
            return world_path
        physics = ET.SubElement(world_elem, 'physics', attrib={'name': 'configured', 'type': 'ignored'})
    max_step_elem = physics.find('max_step_size')
    if max_step_elem is None:
        max_step_elem = ET.SubElement(physics, 'max_step_size')
    max_step_elem.text = f'{max_step_size:.6f}'

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sdf')
    tree.write(tmp.name, encoding='utf-8', xml_declaration=True)
    return tmp.name


def _build_robot_description(urdf_path, imu_rate, gnss_rate):
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except ET.ParseError:
        with open(urdf_path, 'r') as urdf_file:
            return urdf_file.read()

    for sensor in root.findall(".//gazebo/sensor[@name='imu']"):
        rate = sensor.find('update_rate')
        if rate is None:
            rate = ET.SubElement(sensor, 'update_rate')
        rate.text = str(imu_rate)

    for sensor in root.findall(".//gazebo/sensor[@name='navsat']"):
        rate = sensor.find('update_rate')
        if rate is None:
            rate = ET.SubElement(sensor, 'update_rate')
        rate.text = str(gnss_rate)

    return ET.tostring(root, encoding='unicode')
