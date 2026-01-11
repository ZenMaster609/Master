from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vectornav_decoder'

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
    install_requires=['setuptools', 'pyserial', 'pyyaml'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='VectorNav VN-200 serial decoder for IMU/GPS/INS data',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vectornav_decoder_node = vectornav_decoder.vectornav_decoder_node:main',
            'vectornav_monitor_node = vectornav_decoder.vectornav_monitor_node:main',
        ],
    },
)
