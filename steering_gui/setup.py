from setuptools import setup

package_name = 'steering_gui'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/' + package_name + '/resource', [
            'resource/EUFSRobotSteeringGUI.ui'
        ]),
        ('share/' + package_name, ['plugin.xml']),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Robot steering RQT GUI for sim_car',
    license='MIT',
    tests_require=['pytest'],
    scripts=['scripts/eufs_robot_steering_gui'],
)
