"""
Ackermann Control Node for RWD Formula Student Car.

Converts /cmd_vel Twist messages to:
- Ackermann steering angles for front wheels
- Velocity commands for rear drive wheels

Implements proper Ackermann geometry where inner wheel turns
more than outer wheel for correct turn radius.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class AckermannControlNode(Node):
    """
    Converts Twist commands to Ackermann steering + RWD drive commands.

    Parameters:
        wheelbase (float): Distance between front and rear axles (m)
        track_width (float): Distance between left and right wheels (m)
        wheel_radius (float): Wheel radius for velocity conversion (m)
        max_steering_angle (float): Maximum steering angle (radians)
        max_speed (float): Maximum vehicle speed (m/s)
        mode (str): Control mode - 'keyboard' for manual, 'auto' for autonomous

    Subscribes:
        /cmd_vel (geometry_msgs/Twist): Velocity commands

    Publishes:
        /sim/steering_fl_cmd (std_msgs/Float64): Front left steering position (rad)
        /sim/steering_fr_cmd (std_msgs/Float64): Front right steering position (rad)
        /sim/rear_left_wheel_cmd (std_msgs/Float64): Rear left wheel velocity (rad/s)
        /sim/rear_right_wheel_cmd (std_msgs/Float64): Rear right wheel velocity (rad/s)
    """

    def __init__(self):
        super().__init__('ackermann_control_node')

        # Declare parameters with defaults matching the URDF
        self.declare_parameter('wheelbase', 1.6)  # meters (front to rear axle)
        self.declare_parameter('track_width', 1.1)  # meters (left to right wheel)
        self.declare_parameter('wheel_radius', 0.23)  # meters
        self.declare_parameter('max_steering_angle', 0.52)  # radians (~30 degrees)
        self.declare_parameter('max_speed', 15.0)  # m/s
        self.declare_parameter('mode', 'keyboard')  # 'keyboard' or 'auto'
        self.declare_parameter('auto_linear_speed', 3.0)  # m/s for auto mode
        self.declare_parameter('auto_angular_speed', 0.5)  # rad/s for auto mode

        # Get parameters
        self.wheelbase = self.get_parameter('wheelbase').value
        self.track_width = self.get_parameter('track_width').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.max_speed = self.get_parameter('max_speed').value
        self.mode = self.get_parameter('mode').value
        self.auto_linear_speed = self.get_parameter('auto_linear_speed').value
        self.auto_angular_speed = self.get_parameter('auto_angular_speed').value

        # Publishers for steering joints (position control)
        self.steering_fl_pub = self.create_publisher(
            Float64, '/sim/steering_fl_cmd', 10)
        self.steering_fr_pub = self.create_publisher(
            Float64, '/sim/steering_fr_cmd', 10)

        # Publishers for rear drive wheels (velocity control)
        self.rear_left_pub = self.create_publisher(
            Float64, '/sim/rear_left_wheel_cmd', 10)
        self.rear_right_pub = self.create_publisher(
            Float64, '/sim/rear_right_wheel_cmd', 10)

        # Subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Auto mode timer
        if self.mode == 'auto':
            self.auto_timer = self.create_timer(0.02, self.auto_mode_callback)
            self.get_logger().info('Running in AUTO mode - circular motion')
        else:
            self.get_logger().info('Running in KEYBOARD mode - waiting for /cmd_vel')

        self.get_logger().info(
            f'Ackermann Control initialized: '
            f'wheelbase={self.wheelbase}m, track={self.track_width}m, '
            f'wheel_r={self.wheel_radius}m, max_steer={math.degrees(self.max_steering_angle):.1f}deg'
        )

    def cmd_vel_callback(self, msg: Twist):
        """
        Convert Twist to Ackermann steering angles and wheel velocities.

        For a car-like robot:
        - linear.x = desired forward velocity (m/s)
        - angular.z = desired turning rate (rad/s)

        The Ackermann geometry ensures:
        - Inner wheel turns more than outer wheel
        - All wheels rotate around a common point (instant center)
        """
        linear_vel = msg.linear.x
        angular_vel = msg.angular.z

        # Clamp speed
        linear_vel = max(-self.max_speed, min(self.max_speed, linear_vel))

        # Compute steering angles using Ackermann geometry
        steering_fl, steering_fr = self.compute_ackermann_angles(linear_vel, angular_vel)

        # Compute rear wheel velocities
        rear_left_vel, rear_right_vel = self.compute_wheel_velocities(
            linear_vel, angular_vel)

        # Publish steering commands
        fl_msg = Float64()
        fl_msg.data = steering_fl
        self.steering_fl_pub.publish(fl_msg)

        fr_msg = Float64()
        fr_msg.data = steering_fr
        self.steering_fr_pub.publish(fr_msg)

        # Publish drive wheel velocities
        rl_msg = Float64()
        rl_msg.data = rear_left_vel
        self.rear_left_pub.publish(rl_msg)

        rr_msg = Float64()
        rr_msg.data = rear_right_vel
        self.rear_right_pub.publish(rr_msg)

    def compute_ackermann_angles(self, linear_vel: float, angular_vel: float):
        """
        Compute Ackermann steering angles for front wheels.

        The Ackermann geometry ensures proper turning:
        - Turn radius R = linear_vel / angular_vel
        - Inner wheel angle: atan(L / (R - T/2))
        - Outer wheel angle: atan(L / (R + T/2))

        Where L = wheelbase, T = track width

        Args:
            linear_vel: Forward velocity (m/s)
            angular_vel: Yaw rate (rad/s)

        Returns:
            Tuple of (left_angle, right_angle) in radians
        """
        # Handle straight driving
        if abs(angular_vel) < 0.001 or abs(linear_vel) < 0.01:
            return 0.0, 0.0

        # Turn radius (positive = turning left, negative = turning right)
        turn_radius = linear_vel / angular_vel

        # Avoid division by zero for very tight turns
        if abs(turn_radius) < 0.1:
            turn_radius = 0.1 if turn_radius >= 0 else -0.1

        # Ackermann geometry
        # For left turn (positive radius), left wheel is inner
        # For right turn (negative radius), right wheel is inner
        L = self.wheelbase
        T = self.track_width

        # Inner and outer radii
        inner_radius = abs(turn_radius) - T / 2
        outer_radius = abs(turn_radius) + T / 2

        # Clamp inner radius to avoid extreme angles
        inner_radius = max(inner_radius, 0.3)

        # Compute angles
        inner_angle = math.atan(L / inner_radius)
        outer_angle = math.atan(L / outer_radius)

        # Clamp to max steering angle
        inner_angle = min(inner_angle, self.max_steering_angle)
        outer_angle = min(outer_angle, self.max_steering_angle)

        # Assign to left/right based on turn direction
        if turn_radius > 0:  # Turning left
            steering_fl = inner_angle   # Left wheel is inner
            steering_fr = outer_angle   # Right wheel is outer
        else:  # Turning right
            steering_fl = -outer_angle  # Left wheel is outer
            steering_fr = -inner_angle  # Right wheel is inner

        return steering_fl, steering_fr

    def compute_wheel_velocities(self, linear_vel: float, angular_vel: float):
        """
        Compute rear wheel angular velocities for RWD.

        For differential steering effect on rear wheels:
        - When turning, outer wheel spins faster
        - v_left = v - (angular_vel * track_width / 2)
        - v_right = v + (angular_vel * track_width / 2)

        Args:
            linear_vel: Forward velocity (m/s)
            angular_vel: Yaw rate (rad/s)

        Returns:
            Tuple of (left_wheel_vel, right_wheel_vel) in rad/s
        """
        # Differential effect for rear wheels
        half_track = self.track_width / 2

        # Left wheel is inner when turning left (slower)
        # Right wheel is inner when turning right (slower)
        left_linear_vel = linear_vel - angular_vel * half_track
        right_linear_vel = linear_vel + angular_vel * half_track

        # Convert linear to angular velocity
        left_angular_vel = left_linear_vel / self.wheel_radius
        right_angular_vel = right_linear_vel / self.wheel_radius

        return left_angular_vel, right_angular_vel

    def auto_mode_callback(self):
        """
        Auto mode: drive in circles for testing.
        """
        # Create a Twist message for circular motion
        twist = Twist()
        twist.linear.x = self.auto_linear_speed
        twist.angular.z = self.auto_angular_speed

        # Process through the normal callback
        self.cmd_vel_callback(twist)


def main(args=None):
    rclpy.init(args=args)
    node = AckermannControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
