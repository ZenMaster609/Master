from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'vehicle_plotter'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
    ],
    extras_require={
        'plotting': ['pyqtgraph', 'PyQt5'],
        'logging': ['pyarrow'],
        'all': ['pyqtgraph', 'PyQt5', 'pyarrow'],
    },
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Real-time plotting and logging for vehicle state data',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_collector_node = vehicle_plotter.nodes.data_collector_node:main',
            'plotter_node = vehicle_plotter.nodes.plotter_node:main',
            'logger_node = vehicle_plotter.nodes.logger_node:main',
            'rosbag_controller_node = vehicle_plotter.nodes.rosbag_controller_node:main',
            'session_manager_node = vehicle_plotter.nodes.session_manager_node:main',
        ],
    },
)
