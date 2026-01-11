#!/usr/bin/env python3
"""
RosbagControllerNode - Manages rosbag2 recording with session integration.

Wraps `ros2 bag record` subprocess to automatically record topics
during runs, with proper session directory integration.
"""

import rclpy
from rclpy.node import Node
from pathlib import Path
from typing import Optional, List
import subprocess
import signal
import os
import yaml

from vehicle_plotter_msgs.msg import RunSession as RunSessionMsg

from ..core.run_session import RunSession, get_default_base_path
from ..core.qos_profiles import RELIABLE_SENSOR_QOS


class RosbagControllerNode(Node):
    """
    Controls rosbag2 recording lifecycle.

    Integrates with RunSession for proper directory placement and
    multi-machine synchronization.

    Parameters:
        enable_recording (bool): Start recording on startup
        topics (list): Topics to record (empty = use mode config)
        mode (str): Recording mode for topic selection
        compression (str): Compression mode ('zstd', 'none')
        max_bag_size (int): Max bag size in MB before splitting
        wait_for_session (bool): Wait for /run_session before recording
        session_timeout_sec (float): Timeout to wait for /run_session
    """

    def __init__(self):
        super().__init__('rosbag_controller_node')

        # Declare parameters
        self.declare_parameter('enable_recording', True)
        self.declare_parameter('topics', [])
        self.declare_parameter('mode', 'common')  # common, jetson_hardware, simulation, etc.
        self.declare_parameter('compression', 'zstd')
        self.declare_parameter('max_bag_size', 0)  # 0 = no limit
        self.declare_parameter('wait_for_session', True)
        self.declare_parameter('session_timeout_sec', 5.0)
        self.declare_parameter('base_path', '')
        self.declare_parameter('config_file', '')

        # Get parameters
        self._enable_recording = self.get_parameter('enable_recording').value
        topics_param = self.get_parameter('topics').value
        self._mode = self.get_parameter('mode').value
        self._compression = self.get_parameter('compression').value
        self._max_bag_size = self.get_parameter('max_bag_size').value
        self._wait_for_session = self.get_parameter('wait_for_session').value
        self._session_timeout = self.get_parameter('session_timeout_sec').value
        base_path_str = self.get_parameter('base_path').value
        config_file = self.get_parameter('config_file').value

        # Parse base path
        if base_path_str:
            self._base_path = Path(base_path_str).expanduser()
        else:
            self._base_path = None

        # Load topics from config if not specified
        if topics_param:
            self._topics = list(topics_param)
        else:
            self._topics = self._load_topics_from_config(config_file)

        self.get_logger().info(f'RosbagControllerNode starting...')
        self.get_logger().info(f'  Recording enabled: {self._enable_recording}')
        self.get_logger().info(f'  Mode: {self._mode}')
        self.get_logger().info(f'  Topics: {len(self._topics)} configured')
        self.get_logger().info(f'  Compression: {self._compression}')

        self._run_session: Optional[RunSession] = None
        self._recording_process: Optional[subprocess.Popen] = None
        self._session_initialized = False

        if self._enable_recording:
            # Subscribe to run session
            self.session_sub = self.create_subscription(
                RunSessionMsg,
                '/run_session',
                self.session_callback,
                RELIABLE_SENSOR_QOS,
            )

            if self._wait_for_session:
                self.session_timer = self.create_timer(
                    self._session_timeout,
                    self.session_timeout_callback
                )
                self.get_logger().info(
                    f'  Waiting up to {self._session_timeout}s for /run_session...'
                )
            else:
                self._initialize_recording(None)

        # Status timer
        self.status_timer = self.create_timer(30.0, self.status_callback)

    def _load_topics_from_config(self, config_file: str) -> List[str]:
        """Load topics from YAML config file based on mode."""
        # Try to find config file
        if config_file:
            config_path = Path(config_file)
        else:
            # Default location
            config_path = Path(__file__).parent.parent.parent / 'config' / 'rosbag_topics.yaml'

        if not config_path.exists():
            self.get_logger().warning(f'Config file not found: {config_path}')
            return ['/vehicle_plotter/state']

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Get common topics + mode-specific topics
            topics = list(config.get('common', []))

            if self._mode in config:
                topics.extend(config[self._mode])

            # Remove duplicates while preserving order
            seen = set()
            unique_topics = []
            for t in topics:
                if t not in seen:
                    seen.add(t)
                    unique_topics.append(t)

            return unique_topics

        except Exception as e:
            self.get_logger().error(f'Error loading config: {e}')
            return ['/vehicle_plotter/state']

    def session_callback(self, msg: RunSessionMsg) -> None:
        """Handle incoming run session message."""
        if self._session_initialized:
            return

        self.get_logger().info(
            f'Received run_session: {msg.run_id} from {msg.originator_hostname}'
        )

        if hasattr(self, 'session_timer') and self.session_timer:
            self.session_timer.cancel()
            self.session_timer = None

        self._initialize_recording(msg)

    def session_timeout_callback(self) -> None:
        """Called when session timeout expires."""
        if self._session_initialized:
            return

        self.get_logger().info('No /run_session received, creating own session')

        if self.session_timer:
            self.session_timer.cancel()
            self.session_timer = None

        self._initialize_recording(None)

    def _initialize_recording(self, msg: Optional[RunSessionMsg]) -> None:
        """Initialize recording with session info."""
        if self._session_initialized:
            return

        # Create or adopt run session
        if msg:
            self._run_session = RunSession.from_msg(msg, self._base_path)
        else:
            self._run_session = RunSession.create_new(self._base_path)

        self._run_session.ensure_directories()

        self.get_logger().info(f'  Run ID: {self._run_session.run_id}')
        self.get_logger().info(f'  Rosbags path: {self._run_session.rosbags_path}')

        # Start recording
        self._start_recording()
        self._session_initialized = True

    def _start_recording(self) -> None:
        """Start the ros2 bag record subprocess."""
        if not self._topics:
            self.get_logger().warning('No topics to record')
            return

        bag_path = self._run_session.rosbags_path / 'bag'

        # Build command
        cmd = [
            'ros2', 'bag', 'record',
            '-o', str(bag_path),
        ]

        # Add compression if enabled
        if self._compression and self._compression != 'none':
            cmd.extend(['--compression-mode', 'message'])
            cmd.extend(['--compression-format', self._compression])

        # Add max bag size if specified
        if self._max_bag_size > 0:
            cmd.extend(['--max-bag-size', str(self._max_bag_size * 1024 * 1024)])

        # Add topics
        cmd.extend(self._topics)

        self.get_logger().info(f'Starting rosbag recording: {len(self._topics)} topics')
        self.get_logger().debug(f'Command: {" ".join(cmd)}')

        try:
            self._recording_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # Create new process group for clean shutdown
            )
            self.get_logger().info(f'Recording started (PID: {self._recording_process.pid})')
        except Exception as e:
            self.get_logger().error(f'Failed to start recording: {e}')
            self._recording_process = None

    def _stop_recording(self) -> None:
        """Stop the ros2 bag record subprocess."""
        if self._recording_process is None:
            return

        self.get_logger().info('Stopping rosbag recording...')

        try:
            # Send SIGINT to the process group for graceful shutdown
            os.killpg(os.getpgid(self._recording_process.pid), signal.SIGINT)

            # Wait for process to finish (with timeout)
            try:
                self._recording_process.wait(timeout=10.0)
                self.get_logger().info('Recording stopped gracefully')
            except subprocess.TimeoutExpired:
                self.get_logger().warning('Recording did not stop gracefully, forcing...')
                os.killpg(os.getpgid(self._recording_process.pid), signal.SIGKILL)
                self._recording_process.wait()

        except ProcessLookupError:
            self.get_logger().info('Recording process already terminated')
        except Exception as e:
            self.get_logger().error(f'Error stopping recording: {e}')

        self._recording_process = None

    def status_callback(self) -> None:
        """Log periodic status update."""
        if self._recording_process is None:
            if not self._session_initialized:
                self.get_logger().info('Waiting for session initialization...')
            else:
                self.get_logger().warning('Recording process not running')
            return

        # Check if process is still running
        poll = self._recording_process.poll()
        if poll is not None:
            self.get_logger().warning(f'Recording process exited with code {poll}')
            self._recording_process = None
        else:
            self.get_logger().info('Recording in progress...')

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._stop_recording()

        if self._run_session:
            self.get_logger().info(
                f'Rosbag saved to: {self._run_session.rosbags_path}'
            )

    @property
    def run_session(self) -> Optional[RunSession]:
        """Get the current run session."""
        return self._run_session

    @property
    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._recording_process is not None and self._recording_process.poll() is None


def main(args=None):
    rclpy.init(args=args)

    node = RosbagControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
