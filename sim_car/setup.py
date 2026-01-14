from setuptools import setup
from glob import glob
import os

package_name = 'sim_car'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='ROS2 Gazebo car simulation with virtual sensors',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = sim_car.control_node:main',
            'sensor_processor = sim_car.sensor_processor:main',
            'wheel_encoder_node = sim_car.wheel_encoder_node:main',
            'ackermann_control_node = sim_car.ackermann_control_node:main',
            'suspension_sensor_node = sim_car.suspension_sensor_node:main',
            'steering_sensor_node = sim_car.steering_sensor_node:main',
            'virtual_sensors_node = sim_car.virtual_sensors_node:main',
        ],
    },
)
