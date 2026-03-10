from setuptools import find_packages, setup
from glob import glob
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

package_name = 'sim_car'

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
            'virtual_sensors_node = sim_car.sensors.virtual_sensors_node:main',
            'water_pressure_node = sim_car.sensors.water_pressure_node:main',
            'water_flow_node = sim_car.sensors.water_flow_node:main',
            'water_temp_in_node = sim_car.sensors.water_temp_in_node:main',
            'water_temp_out_node = sim_car.sensors.water_temp_out_node:main',
            'water_temp_radiator_node = sim_car.sensors.water_temp_radiator_node:main',
            'brake_temp_fr_node = sim_car.sensors.brake_temp_fr_node:main',
            'brake_temp_rl_node = sim_car.sensors.brake_temp_rl_node:main',
            'pitot_dynamic_pressure_node = sim_car.sensors.pitot_dynamic_pressure_node:main',
            'measurement_node = sim_car.sensors.measurement_node:main',
            'stereo_perception_node = sim_car.perception.stereo_perception_node:main',
            'mono_perception_node = sim_car.perception.mono_perception_node:main',
            'cone_evaluator_node = sim_car.cones.nodes.evaluator_node:main',
            'cone_plotting2_node = sim_car.cones.nodes.plotting2_node:main',
            'lidar_node = sim_car.lidar.lidar_node:main',
            'delaunay_planner_node = sim_car.planning.delaunay_planner_node:main',
            'cone_memory_node = sim_car.cones.nodes.memory_node:main',
            'run_artifacts_node = sim_car.run_artifacts_node:main',
        ],
    },
)
