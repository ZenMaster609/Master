from setuptools import find_packages, setup
from glob import glob
from pathlib import Path
import os


def walk_data_files(source_dir, install_dir):
    data = []
    for root, _, files in os.walk(source_dir):
        if not files:
            continue
        rel_dir = os.path.relpath(root, source_dir)
        dest_dir = install_dir if rel_dir == '.' else os.path.join(install_dir, rel_dir)
        data.append((dest_dir, [os.path.join(root, f) for f in files]))
    return data


def _workspace_root_from_source(source_path: Path) -> Path | None:
    parts = source_path.parts
    if 'src' not in parts:
        return None
    src_index = parts.index('src')
    if src_index == 0:
        return None
    return Path(*parts[:src_index])


def prune_stale_build_config(source_dir: str, package_name: str) -> None:
    source_root = Path(source_dir).resolve()
    workspace_root = _workspace_root_from_source(source_root)
    if workspace_root is None:
        return

    build_config_root = workspace_root / 'build' / package_name / 'config'
    if not build_config_root.exists():
        return

    expected_relpaths = {
        path.relative_to(source_root)
        for path in source_root.rglob('*')
        if path.is_file()
    }

    for path in sorted(build_config_root.rglob('*'), reverse=True):
        try:
            relpath = path.relative_to(build_config_root)
        except ValueError:
            continue

        if path.is_dir() and not path.is_symlink():
            if not any(path.iterdir()):
                path.rmdir()
            continue

        if relpath in expected_relpaths:
            continue

        if path.exists() or path.is_symlink():
            path.unlink()

package_name = 'sim_car'
prune_stale_build_config('config', package_name)

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ]
    + walk_data_files('config', os.path.join('share', package_name, 'config'))
    + walk_data_files('meshes', os.path.join('share', package_name, 'meshes'))
    + walk_data_files('materials', os.path.join('share', package_name, 'materials'))
    + walk_data_files('models', os.path.join('share', package_name, 'models'))
    + walk_data_files('rviz', os.path.join('share', package_name, 'rviz'))
    + walk_data_files('yolo', os.path.join('share', package_name, 'yolo')),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='ROS2 Gazebo car simulation with virtual sensors',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wheel_encoder_node = sim_car.sensors.wheel_encoder_node:main',
            'ackermann_cmd_bridge = sim_car.controllers.ackermann_cmd_bridge:main',
            'suspension_sensor_node = sim_car.sensors.suspension_sensor_node:main',
            'steering_sensor_node = sim_car.sensors.steering_sensor_node:main',
            'water_pressure_node = sim_car.sensors.simple_sensor_node:main_water_pressure',
            'water_flow_node = sim_car.sensors.simple_sensor_node:main_water_flow',
            'water_temp_in_node = sim_car.sensors.simple_sensor_node:main_water_temp_in',
            'water_temp_out_node = sim_car.sensors.simple_sensor_node:main_water_temp_out',
            'brake_temp_fr_node = sim_car.sensors.simple_sensor_node:main_brake_temp_fr',
            'brake_temp_rl_node = sim_car.sensors.simple_sensor_node:main_brake_temp_rl',
            'pitot_dynamic_pressure_node = sim_car.sensors.simple_sensor_node:main_pitot_dynamic_pressure',
            'measurement_node = sim_car.sensors.measurement_node:main',
            'odom_delay_node = sim_car.sensors.odom_delay_node:main',
            'odom_tf_broadcaster_node = sim_car.sensors.odom_tf_broadcaster_node:main',
            'perception_node = sim_car.perception.perception_node:main',
            'cone_evaluator_node = sim_car.cones.nodes.evaluator_node:main',
            'lidar_node = sim_car.lidar.lidar_node:main',
            'midpoint_planner_node = sim_car.planning.tracked_cone_planner_node:main_midpoint',
            'single_boundary_planner_node = sim_car.planning.tracked_cone_planner_node:main_single_boundary',
            'corridor_planner_node = sim_car.planning.tracked_cone_planner_node:main_corridor',
            'linetest_planner_node = sim_car.planning.linetest_planner_node:main',
            'skidpad_router_node = sim_car.planning.skidpad_router_node:main',
            'cone_memory_node = sim_car.cones.nodes.memory_node:main',
            'run_artifacts_node = sim_car.run_artifacts_node:main',
        ],
    },
)
