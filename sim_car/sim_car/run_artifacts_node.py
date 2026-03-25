#!/usr/bin/env python3
"""
RunArtifactsNode - Writes per-run config snapshots under multidata/<run_id>/configs.

Subscribes to /run_session and saves:
1) A recursive copy of sim_car config YAML files.
2) launch_parameters.yaml with resolved full_sim launch argument values.
"""

from pathlib import Path
import shutil
from typing import Dict, Any

import rclpy
from rclpy.node import Node

import yaml

from vehicle_plotter_msgs.msg import RunSession


def copy_config_snapshot(source_dir: Path, target_dir: Path, copy_glob: str) -> int:
    """Copy matching config files recursively while preserving relative paths."""
    copied = 0
    for src in sorted(source_dir.rglob(copy_glob)):
        if not src.is_file():
            continue
        relative_path = src.relative_to(source_dir)
        destination = target_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)
        copied += 1
    return copied


class RunArtifactsNode(Node):
    """Persist config snapshots tied to the active run session."""

    def __init__(self) -> None:
        super().__init__(
            'run_artifacts_node',
            automatically_declare_parameters_from_overrides=True,
        )

        self._run_session_topic = str(self._get_param_value('run_session_topic', '/run_session')).strip() or '/run_session'
        self._copy_glob = str(self._get_param_value('copy_glob', '*.yaml')).strip() or '*.yaml'
        self._write_once = bool(self._get_param_value('write_once', True))
        self._session_timeout_sec = float(self._get_param_value('session_timeout_sec', 10.0))

        source_dir_param = str(self._get_param_value('config_source_dir', '')).strip()
        self._config_source_dir = Path(source_dir_param).expanduser() if source_dir_param else None
        self._launch_parameters = self._read_launch_parameters()
        self._wrote_artifacts = False

        self._session_sub = self.create_subscription(
            RunSession,
            self._run_session_topic,
            self._run_session_cb,
            10,
        )

        self._timeout_timer = None
        if self._session_timeout_sec > 0.0:
            self._timeout_timer = self.create_timer(self._session_timeout_sec, self._timeout_cb)

        self.get_logger().info(f'RunArtifactsNode waiting on {self._run_session_topic}')

    def _get_param_value(self, name: str, default: Any) -> Any:
        if self.has_parameter(name):
            return self.get_parameter(name).value
        return default

    def _read_launch_parameters(self) -> Dict[str, Any]:
        params = self.get_parameters_by_prefix('launch_parameters')
        values: Dict[str, Any] = {}
        for key, param in params.items():
            values[key] = param.value
        return dict(sorted(values.items()))

    def _timeout_cb(self) -> None:
        if self._wrote_artifacts:
            return
        self.get_logger().warn('No /run_session received before timeout; skipping run artifact snapshot')
        self._shutdown_self()

    def _run_session_cb(self, msg: RunSession) -> None:
        if self._wrote_artifacts and self._write_once:
            return

        run_id = str(msg.run_id).strip()
        base_path_str = str(msg.base_path).strip()
        if not run_id:
            self.get_logger().warn('Received /run_session without run_id, skipping')
            return

        base_path = Path(base_path_str).expanduser() if base_path_str else Path.cwd() / 'multidata'
        session_path = base_path / run_id
        configs_path = session_path / 'configs'
        config_snapshot_path = configs_path / 'sim_car_config'
        launch_params_path = configs_path / 'launch_parameters.yaml'

        try:
            config_snapshot_path.mkdir(parents=True, exist_ok=True)
            self._copy_config_files(config_snapshot_path)
            self._write_launch_parameters(launch_params_path)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f'Failed writing run artifact snapshot: {exc}')
            self._shutdown_self()
            return

        self._wrote_artifacts = True
        self.get_logger().info(f'Wrote run config snapshot: {configs_path}')
        if self._write_once:
            self._shutdown_self()

    def _copy_config_files(self, target_dir: Path) -> None:
        if self._config_source_dir is None:
            self.get_logger().warn('config_source_dir is empty; skipping config YAML copy')
            return
        if not self._config_source_dir.exists():
            self.get_logger().warn(f'config_source_dir does not exist: {self._config_source_dir}')
            return

        copied = copy_config_snapshot(self._config_source_dir, target_dir, self._copy_glob)
        self.get_logger().info(f'Copied {copied} config file(s) from {self._config_source_dir}')

    def _write_launch_parameters(self, launch_params_path: Path) -> None:
        launch_params_path.parent.mkdir(parents=True, exist_ok=True)
        with launch_params_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(
                self._launch_parameters,
                f,
                default_flow_style=False,
                sort_keys=True,
            )

    def _shutdown_self(self) -> None:
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RunArtifactsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
