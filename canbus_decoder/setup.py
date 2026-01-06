from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'canbus_decoder'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='CAN bus decoder for vehicle data',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'can_monitor_node = canbus_decoder.can_monitor_node:main',
        ],
    },
)
