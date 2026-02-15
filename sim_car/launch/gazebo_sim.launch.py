#!/usr/bin/env python3
"""
Launch file for Gazebo Fortress simulation with ros_gz bridge.
Updated from Gazebo Classic to modern Gazebo (Fortress).
"""

import os
import tempfile
import xml.etree.ElementTree as ET
import subprocess
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_sim_car = get_package_share_directory('sim_car')
    world_file = os.path.join(pkg_sim_car, 'worlds', 'small_track.world')

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default=world_file)
    headless = LaunchConfiguration('headless', default='false')
    update_rate_hz = LaunchConfiguration('update_rate_hz', default='100.0')
    topic_prefix = LaunchConfiguration('topic_prefix', default='/sim/raw')
    sensors_render_engine = LaunchConfiguration('sensors_render_engine', default='ogre')

    resource_path = os.path.join(pkg_sim_car)
    resource_paths = [
        resource_path,
        os.path.join(resource_path, 'models'),
        os.path.join(resource_path, 'meshes'),
        os.path.join(resource_path, 'materials'),
    ]
    resource_path_value = ':'.join(resource_paths)

    return LaunchDescription([
        # Prefer discrete NVIDIA GPU rendering when available.
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
        SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '0'),
        SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', '0'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path_value),
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', resource_path_value),
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
        DeclareLaunchArgument(
            'update_rate_hz',
            default_value='100.0',
            description='Dynamics + joint state update rate (Hz)'
        ),
        DeclareLaunchArgument(
            'topic_prefix',
            default_value='/sim/raw',
            description='Topic prefix for sim sensors (/sim or /sim/raw)'
        ),
        DeclareLaunchArgument(
            'sensors_render_engine',
            default_value='ogre',
            description='Render engine for injected sensors plugin (ogre or ogre2)'
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
    urdf_path = os.path.join(pkg_sim_car, 'urdf', 'eufs_car.urdf.xacro')
    topic_prefix = LaunchConfiguration('topic_prefix').perform(context)
    sensors_render_engine = LaunchConfiguration('sensors_render_engine').perform(context)

    updated_world = _write_updated_world(world_path, max_step_size, sensors_render_engine)
    eufs_config_path = os.path.join(pkg_sim_car, 'config', 'eufs_config.yaml')
    update_rate_value = float(LaunchConfiguration('update_rate_hz').perform(context))
    updated_eufs_config = _write_updated_eufs_config(eufs_config_path, update_rate_value)
    robot_desc = _build_robot_description(
        urdf_path, imu_rate, gnss_rate, updated_eufs_config, topic_prefix
    )

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

    imu_enabled = _any_signal_enabled(
        sensor_config,
        topics=[f'{topic_prefix}/imu', '/sim/imu', '/sim/raw/imu'],
    )
    navsat_enabled = _any_signal_enabled(
        sensor_config,
        topics=[f'{topic_prefix}/navsat', '/sim/navsat', '/sim/raw/navsat'],
    )
    odom_enabled = _any_signal_enabled(
        sensor_config,
        topics=[f'{topic_prefix}/odom', '/sim/odom', '/sim/raw/odom'],
    )

    bridge_args = [
        f'{topic_prefix}/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        f'{topic_prefix}/stereo/left/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        f'{topic_prefix}/stereo/right/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        f'{topic_prefix}/stereo/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        f'{topic_prefix}/stereo/right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
    ]
    if odom_enabled:
        bridge_args.append(f'{topic_prefix}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry')
    if imu_enabled:
        bridge_args.append(f'{topic_prefix}/imu@sensor_msgs/msg/Imu[gz.msgs.IMU')
    if navsat_enabled:
        bridge_args.append(f'{topic_prefix}/navsat@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat')

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_args,
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


def _any_signal_enabled(config, topics, default=True):
    signals = config.get('signals')
    if not isinstance(signals, dict):
        return default
    found = False
    for signal in signals.values():
        if not isinstance(signal, dict):
            continue
        if signal.get('plot_only', False):
            continue
        input_topic = signal.get('input_topic')
        output_topic = signal.get('output_topic')
        if input_topic in topics or output_topic in topics:
            found = True
            if bool(signal.get('enabled', True)):
                return True
    if not found:
        return default
    return False


def _write_updated_world(world_path, max_step_size, sensors_render_engine):
    try:
        tree = ET.parse(world_path)
        root = tree.getroot()
    except ET.ParseError:
        return world_path

    world_elem = root.find('.//world')
    if world_elem is None:
        return world_path

    _ensure_required_world_systems(world_elem, sensors_render_engine)

    physics = world_elem.find('physics')
    if physics is None:
        physics = ET.SubElement(world_elem, 'physics', attrib={'name': 'configured', 'type': 'ignored'})
    max_step_elem = physics.find('max_step_size')
    if max_step_elem is None:
        max_step_elem = ET.SubElement(physics, 'max_step_size')
    max_step_elem.text = f'{max_step_size:.6f}'

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sdf')
    tree.write(tmp.name, encoding='utf-8', xml_declaration=True)
    return tmp.name


def _ensure_required_world_systems(world_elem, sensors_render_engine):
    required_plugins = [
        ('ignition-gazebo-physics-system', 'ignition::gazebo::systems::Physics'),
        ('ignition-gazebo-user-commands-system', 'ignition::gazebo::systems::UserCommands'),
        ('ignition-gazebo-scene-broadcaster-system', 'ignition::gazebo::systems::SceneBroadcaster'),
        ('ignition-gazebo-sensors-system', 'ignition::gazebo::systems::Sensors'),
    ]

    existing_plugins = world_elem.findall('plugin')

    for filename, plugin_name in required_plugins:
        plugin_elem = _find_world_plugin(existing_plugins, filename, plugin_name)
        if plugin_elem is None:
            plugin_elem = ET.SubElement(world_elem, 'plugin', attrib={
                'filename': filename,
                'name': plugin_name,
            })
            existing_plugins.append(plugin_elem)

        if plugin_name == 'ignition::gazebo::systems::Sensors' and plugin_elem.find('render_engine') is None:
            render_engine = plugin_elem.find('render_engine')
            if render_engine is None:
                render_engine = ET.SubElement(plugin_elem, 'render_engine')
            render_engine.text = sensors_render_engine


def _find_world_plugin(plugins, filename, plugin_name):
    for plugin in plugins:
        if plugin.get('filename') == filename:
            return plugin
        if plugin.get('name') == plugin_name:
            return plugin
    return None


def _write_updated_eufs_config(config_path, update_rate_hz):
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError):
        return config_path

    dynamics = config.get('dynamics')
    if not isinstance(dynamics, dict):
        dynamics = {}
        config['dynamics'] = dynamics
    dynamics['update_rate_hz'] = float(update_rate_hz)

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml.safe_dump(config, tmp, default_flow_style=False, sort_keys=False)
        return tmp.name


def _build_robot_description(urdf_path, imu_rate, gnss_rate, eufs_config_path, topic_prefix):
    robot_xml = _load_robot_xml(urdf_path, eufs_config_path, topic_prefix)
    try:
        root = ET.fromstring(robot_xml)
    except ET.ParseError:
        return robot_xml

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

    for plugin in root.findall(".//gazebo/plugin"):
        if plugin.get('filename') == 'libeufs_gz_dynamics.so':
            yaml_elem = plugin.find('yaml_config')
            if yaml_elem is None:
                yaml_elem = ET.SubElement(plugin, 'yaml_config')
            yaml_elem.text = eufs_config_path

    return ET.tostring(root, encoding='unicode')


def _load_robot_xml(urdf_path, eufs_config_path, topic_prefix):
    if urdf_path.endswith('.xacro'):
        return _run_xacro(urdf_path, eufs_config_path, topic_prefix)
    with open(urdf_path, 'r') as urdf_file:
        return urdf_file.read()


def _run_xacro(urdf_path, eufs_config_path, topic_prefix):
    cmd = [
        'xacro',
        urdf_path,
        f'config_file:={eufs_config_path}',
        f'topic_prefix:={topic_prefix}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout
