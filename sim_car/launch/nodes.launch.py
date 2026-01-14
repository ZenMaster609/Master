#!/usr/bin/env python3

"""
Launch file for control and sensor processing nodes.

Includes Ackermann control, wheel encoder, suspension sensor,
steering sensor, and virtual sensors nodes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # Declare launch arguments
    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='keyboard',
        description='Control mode: keyboard or auto'
    )

    linear_speed_arg = DeclareLaunchArgument(
        'linear_speed',
        default_value='3.0',
        description='Default linear speed for auto mode (m/s)'
    )

    angular_speed_arg = DeclareLaunchArgument(
        'angular_speed',
        default_value='0.5',
        description='Default angular speed for auto mode (rad/s)'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='1.0',
        description='Sensor status publish rate (Hz)'
    )

    # Get launch configurations
    control_mode = LaunchConfiguration('control_mode')
    linear_speed = LaunchConfiguration('linear_speed')
    angular_speed = LaunchConfiguration('angular_speed')
    publish_rate = LaunchConfiguration('publish_rate')

    # Ackermann control node (RWD + steering)
    ackermann_control_node = Node(
        package='sim_car',
        executable='ackermann_control_node',
        name='ackermann_control_node',
        output='screen',
        parameters=[{
            'mode': control_mode,
            'auto_linear_speed': linear_speed,
            'auto_angular_speed': angular_speed,
            'wheelbase': 1.6,       # meters
            'track_width': 1.1,     # meters
            'wheel_radius': 0.23,   # meters
            'max_steering_angle': 0.52,  # radians (~30 deg)
            'max_speed': 15.0,      # m/s
        }]
    )

    # Sensor processor node
    sensor_processor_node = Node(
        package='sim_car',
        executable='sensor_processor',
        name='sensor_processor',
        output='screen',
        parameters=[{
            'publish_rate': publish_rate
        }]
    )

    # Wheel encoder node
    wheel_encoder_node = Node(
        package='sim_car',
        executable='wheel_encoder_node',
        name='wheel_encoder_node',
        output='screen',
        parameters=[{
            'wheel_radius': 0.23,   # Match URDF
            'publish_rate': 50.0
        }]
    )

    # Suspension sensor node
    suspension_sensor_node = Node(
        package='sim_car',
        executable='suspension_sensor_node',
        name='suspension_sensor_node',
        output='screen',
        parameters=[{
            'noise_stddev': 0.5,     # mm
            'bias_fl': 0.0,
            'bias_fr': 0.0,
            'bias_rl': 0.0,
            'bias_rr': 0.0,
            'publish_rate': 100.0,
            'dropout_probability': 0.0,
        }]
    )

    # Steering angle sensor node
    steering_sensor_node = Node(
        package='sim_car',
        executable='steering_sensor_node',
        name='steering_sensor_node',
        output='screen',
        parameters=[{
            'noise_stddev': 0.1,     # degrees
            'latency_ms': 5.0,
            'bias': 0.0,
            'publish_rate': 100.0,
            'dropout_probability': 0.0,
        }]
    )

    # Virtual sensors node (cooling, brakes, pitot)
    virtual_sensors_node = Node(
        package='sim_car',
        executable='virtual_sensors_node',
        name='virtual_sensors_node',
        output='screen',
        parameters=[{
            'publish_rate': 10.0,
            'ambient_temp': 25.0,
            'noise_pressure': 0.02,  # bar
            'noise_flow': 0.5,       # L/min
            'noise_temp': 0.3,       # Celsius
            'noise_brake_temp': 1.0, # Celsius
            'noise_pitot': 2.0,      # Pa
        }]
    )

    return LaunchDescription([
        control_mode_arg,
        linear_speed_arg,
        angular_speed_arg,
        publish_rate_arg,
        ackermann_control_node,
        sensor_processor_node,
        wheel_encoder_node,
        suspension_sensor_node,
        steering_sensor_node,
        virtual_sensors_node,
    ])
