"""
Run session management for multi-machine data collection.

Handles run ID generation, storage layout, and session synchronization
across Jetson and Windows machines.

Storage Layout:
    ./multidata/<run_id>/
      linux/
        logs/           # Parquet/CSV vehicle state data
        rosbags/        # ros2 bag recordings
        plots/          # PNG exports from plotter
        plot_data/      # CSV of plotted data points
        ground_truth/   # ros2 node/topic snapshots
      windows/
        logs/
        rosbags/
        plots/
        plot_data/
        ground_truth/
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import platform
import socket
import os


def detect_os_folder() -> str:
    """Detect the appropriate OS folder name."""
    system = platform.system().lower()
    if system == 'linux':
        return 'linux'
    elif system == 'windows':
        return 'windows'
    elif system == 'darwin':
        return 'macos'
    else:
        return system


def get_default_base_path() -> Path:
    """Get the default base path for multidata storage."""
    # Try to find the Master folder (repo root)
    # Look for common markers
    #
    # NOTE: Use .resolve() on __file__ to follow symlinks from colcon build
    # directory to the actual source location in Master/
    search_paths = [
        Path(__file__).resolve().parent.parent.parent.parent,  # From this file up to Master
        Path.home() / 'ros2_ws' / 'src' / 'Master',
        Path('/home') / os.getenv('USER', 'user') / 'ros2_ws' / 'src' / 'Master',
        Path.cwd(),  # Last resort: current working directory
    ]

    for p in search_paths:
        # Check for both vehicle_plotter and CLAUDE.md to confirm this is the Master folder
        if p.exists() and (p / 'vehicle_plotter').exists() and (p / 'CLAUDE.md').exists():
            return p / 'multidata'

    # Fallback to current working directory
    return Path.cwd() / 'multidata'


@dataclass
class RunSession:
    """
    Manages a data collection run session.

    Handles run ID generation, directory creation, and path management
    for logs, rosbags, plots, and ground truth files.

    Attributes:
        run_id: Unique session identifier (format: YYYY-MM-DD_HH-MM-SS)
        base_path: Root path for multidata storage
        os_folder: OS-specific subfolder ('linux' or 'windows')
        originator_hostname: Machine that generated the run_id
        ros_domain_id: ROS_DOMAIN_ID for verification
        start_time: Session start timestamp
    """

    run_id: str
    base_path: Path
    os_folder: str = field(default_factory=detect_os_folder)
    originator_hostname: str = field(default_factory=socket.gethostname)
    ros_domain_id: int = field(default_factory=lambda: int(os.getenv('ROS_DOMAIN_ID', '0')))
    start_time: datetime = field(default_factory=datetime.now)

    @classmethod
    def create_new(cls, base_path: Optional[Path] = None) -> 'RunSession':
        """
        Create a new run session with a fresh run_id.

        Args:
            base_path: Optional base path. Uses default if not specified.

        Returns:
            New RunSession instance with generated run_id
        """
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = Path(base_path) if base_path else get_default_base_path()
        return cls(run_id=run_id, base_path=base)

    @classmethod
    def from_run_id(cls, run_id: str, base_path: Optional[Path] = None) -> 'RunSession':
        """
        Create a session from an existing run_id (e.g., received from another machine).

        Args:
            run_id: Existing run ID string
            base_path: Optional base path. Uses default if not specified.

        Returns:
            RunSession instance with the provided run_id
        """
        base = Path(base_path) if base_path else get_default_base_path()
        return cls(run_id=run_id, base_path=base)

    @classmethod
    def from_msg(cls, msg, base_path: Optional[Path] = None) -> 'RunSession':
        """
        Create a session from a ROS RunSession message.

        Args:
            msg: vehicle_plotter_msgs/RunSession message
            base_path: Optional base path override

        Returns:
            RunSession instance populated from message
        """
        base = Path(base_path) if base_path else Path(msg.base_path) if msg.base_path else get_default_base_path()

        return cls(
            run_id=msg.run_id,
            base_path=base,
            originator_hostname=msg.originator_hostname,
            ros_domain_id=msg.ros_domain_id,
        )

    @property
    def session_path(self) -> Path:
        """Full path to this machine's session folder."""
        return self.base_path / self.run_id / self.os_folder

    @property
    def logs_path(self) -> Path:
        """Path for Parquet/CSV log files."""
        return self.session_path / 'logs'

    @property
    def rosbags_path(self) -> Path:
        """Path for rosbag recordings."""
        return self.session_path / 'rosbags'

    @property
    def plots_path(self) -> Path:
        """Path for plot PNG exports."""
        return self.session_path / 'plots'

    @property
    def plot_data_path(self) -> Path:
        """Path for plot data CSV exports."""
        return self.session_path / 'plot_data'

    @property
    def ground_truth_path(self) -> Path:
        """Path for runtime ground truth snapshots."""
        return self.session_path / 'ground_truth'

    def ensure_directories(self) -> None:
        """Create all session directories if they don't exist."""
        for path in [
            self.logs_path,
            self.rosbags_path,
            self.plots_path,
            self.plot_data_path,
            self.ground_truth_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def save_session_info(self) -> Path:
        """
        Save session metadata to JSON file.

        Returns:
            Path to the saved session_info.json file
        """
        self.ensure_directories()

        info = {
            'run_id': self.run_id,
            'os_folder': self.os_folder,
            'originator_hostname': self.originator_hostname,
            'local_hostname': socket.gethostname(),
            'ros_domain_id': self.ros_domain_id,
            'start_time': self.start_time.isoformat(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
                'python_version': platform.python_version(),
            },
            'paths': {
                'base_path': str(self.base_path),
                'session_path': str(self.session_path),
                'logs_path': str(self.logs_path),
                'rosbags_path': str(self.rosbags_path),
                'plots_path': str(self.plots_path),
                'plot_data_path': str(self.plot_data_path),
                'ground_truth_path': str(self.ground_truth_path),
            },
        }

        info_path = self.session_path / 'session_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        return info_path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'base_path': str(self.base_path),
            'os_folder': self.os_folder,
            'originator_hostname': self.originator_hostname,
            'ros_domain_id': self.ros_domain_id,
            'start_time': self.start_time.isoformat(),
        }

    def to_msg(self):
        """
        Convert to ROS RunSession message.

        Returns:
            vehicle_plotter_msgs/RunSession message
        """
        from vehicle_plotter_msgs.msg import RunSession as RunSessionMsg
        from builtin_interfaces.msg import Time

        msg = RunSessionMsg()
        msg.run_id = self.run_id
        msg.base_path = str(self.base_path)
        msg.originator_hostname = self.originator_hostname
        msg.ros_domain_id = self.ros_domain_id

        # Convert datetime to ROS Time
        timestamp = self.start_time.timestamp()
        msg.start_time = Time(
            sec=int(timestamp),
            nanosec=int((timestamp - int(timestamp)) * 1e9)
        )

        return msg

    def __str__(self) -> str:
        return f"RunSession(run_id={self.run_id}, path={self.session_path})"

    def __repr__(self) -> str:
        return (f"RunSession(run_id='{self.run_id}', base_path='{self.base_path}', "
                f"os_folder='{self.os_folder}', originator='{self.originator_hostname}')")
