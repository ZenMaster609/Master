#!/usr/bin/env python3
"""
SessionManagerNode - Manages run sessions and broadcasts run_id.

Creates a RunSession on startup and periodically publishes the session
info to /run_session for other nodes to synchronize.
"""

import rclpy
from rclpy.node import Node
from pathlib import Path
from typing import Optional

from vehicle_plotter_msgs.msg import RunSession as RunSessionMsg

from ..core.run_session import RunSession
from ..core.qos_profiles import RELIABLE_SENSOR_QOS


class SessionManagerNode(Node):
    """
    Manages run session lifecycle and broadcasts run_id.

    This node should run on the primary machine (e.g., Jetson) to
    establish a session that other nodes (logger, plotter)
    can synchronize with.

    Parameters:
        base_path (str): Base path for multidata storage
        broadcast_rate_hz (float): How often to publish session info
        run_id_prefix (str): Optional run id prefix before the timestamp
    """

    def __init__(self):
        super().__init__('session_manager_node')

        # Declare parameters
        self.declare_parameter('base_path', '')
        self.declare_parameter('broadcast_rate_hz', 1.0)
        self.declare_parameter('run_id_prefix', '')

        # Get parameters
        base_path_str = self.get_parameter('base_path').value
        broadcast_rate = self.get_parameter('broadcast_rate_hz').value
        run_id_prefix = self.get_parameter('run_id_prefix').value

        # Parse base path
        if base_path_str:
            base_path = Path(base_path_str).expanduser()
        else:
            base_path = None

        # Create session
        self._run_session = RunSession.create_new(base_path, run_id_prefix=str(run_id_prefix))
        self._run_session.ensure_directories()
        self._run_session.save_session_info()

        self.get_logger().info(f'SessionManagerNode started')
        self.get_logger().info(f'  Run ID: {self._run_session.run_id}')
        self.get_logger().info(f'  Session path: {self._run_session.session_path}')

        # Create publisher
        self.session_pub = self.create_publisher(
            RunSessionMsg,
            '/run_session',
            RELIABLE_SENSOR_QOS,
        )

        # Publish immediately
        self._publish_session()

        # Create timer for periodic broadcast
        self.broadcast_timer = self.create_timer(
            1.0 / broadcast_rate,
            self._publish_session,
        )

    def _publish_session(self) -> None:
        """Publish current session info."""
        msg = self._run_session.to_msg()
        self.session_pub.publish(msg)

    @property
    def run_session(self) -> RunSession:
        """Get the current run session."""
        return self._run_session


def main(args=None):
    rclpy.init(args=args)

    node = SessionManagerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'SessionManagerNode shutting down, run_id: {node.run_session.run_id}'
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
