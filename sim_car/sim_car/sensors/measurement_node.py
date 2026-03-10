#!/usr/bin/env python3
"""
MeasurementNode - applies latency, dropout, noise, bias, and saturation to raw sensor topics.
"""

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rosidl_runtime_py.utilities import get_message


@dataclass
class SignalConfig:
    name: str
    enabled: bool
    input_topic: str
    output_topic: str
    msg_type: str
    rate_hz: float
    latency_ms: float
    dropout_prob: float
    noise_std: Any
    bias_init: Any
    bias_rw_std: Any
    saturation_min: Any
    saturation_max: Any
    apply_orientation: bool
    fields: Optional[list]


class SignalProcessor:
    def __init__(self, node: Node, cfg: SignalConfig, seed: int):
        self.node = node
        self.cfg = cfg
        self.msg_class = get_message(cfg.msg_type)
        self.publisher = self.node.create_publisher(self.msg_class, cfg.output_topic, 10)
        self.subscription = self.node.create_subscription(
            self.msg_class, cfg.input_topic, self._callback, 10
        )

        self.latency_sec = max(0.0, float(cfg.latency_ms) / 1000.0)
        self.min_period = 0.0
        if cfg.rate_hz and cfg.rate_hz > 0.0:
            self.min_period = 1.0 / float(cfg.rate_hz)

        self.dropout_prob = float(cfg.dropout_prob) if cfg.dropout_prob is not None else 0.0
        self.buffer = deque()
        self.last_pub_stamp: Optional[float] = None
        self.last_bias_time: Optional[float] = None
        self.bias_state: Dict[str, float] = {}

        self.rng = np.random.default_rng(seed)

    def _callback(self, msg):
        stamp = self._get_msg_stamp(msg)
        self.buffer.append((stamp, msg))

    def _get_msg_stamp(self, msg) -> float:
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = msg.header.stamp
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return self.node.get_clock().now().nanoseconds * 1e-9

    def process(self, now_sec: float) -> None:
        while self.buffer:
            stamp, msg = self.buffer[0]
            if (stamp + self.latency_sec) > now_sec:
                break
            self.buffer.popleft()

            if self.min_period > 0.0 and self.last_pub_stamp is not None:
                if (stamp - self.last_pub_stamp) < self.min_period:
                    continue

            if self.dropout_prob > 0.0 and self.rng.random() < self.dropout_prob:
                continue

            out = copy.deepcopy(msg)
            self._apply_measurement(out, stamp)
            self.publisher.publish(out)
            self.last_pub_stamp = stamp

    def _apply_measurement(self, msg, stamp: float) -> None:
        dt = 0.0
        if self.last_bias_time is not None:
            dt = max(0.0, stamp - self.last_bias_time)
        self.last_bias_time = stamp

        msg_type = self.cfg.msg_type
        apply_orientation = bool(self.cfg.apply_orientation)

        if msg_type in ('std_msgs/msg/Float32', 'std_msgs/msg/Float64'):
            msg.data = float(self._apply_scalar('data', float(msg.data), dt))
            return

        if msg_type in ('std_msgs/msg/Float32MultiArray', 'std_msgs/msg/Float64MultiArray'):
            msg.data = [
                float(self._apply_scalar(f'data[{i}]', float(v), dt, index=i, base='data'))
                for i, v in enumerate(msg.data)
            ]
            return

        if msg_type == 'geometry_msgs/msg/Vector3':
            for axis in ('x', 'y', 'z'):
                setattr(
                    msg,
                    axis,
                    float(self._apply_scalar(f'{axis}', float(getattr(msg, axis)), dt, base=axis))
                )
            return

        if msg_type == 'geometry_msgs/msg/Twist':
            for part_name in ('linear', 'angular'):
                part = getattr(msg, part_name)
                for axis in ('x', 'y', 'z'):
                    path = f'{part_name}.{axis}'
                    setattr(
                        part,
                        axis,
                        float(self._apply_scalar(path, float(getattr(part, axis)), dt, base=part_name))
                    )
            return

        if msg_type == 'sensor_msgs/msg/Imu':
            for part_name in ('angular_velocity', 'linear_acceleration'):
                part = getattr(msg, part_name)
                for axis in ('x', 'y', 'z'):
                    path = f'{part_name}.{axis}'
                    setattr(
                        part,
                        axis,
                        float(self._apply_scalar(path, float(getattr(part, axis)), dt, base=part_name))
                    )
            if apply_orientation:
                for axis in ('x', 'y', 'z', 'w'):
                    path = f'orientation.{axis}'
                    setattr(
                        msg.orientation,
                        axis,
                        float(
                            self._apply_scalar(
                                path,
                                float(getattr(msg.orientation, axis)),
                                dt,
                                base='orientation',
                            )
                        ),
                    )
            return

        if msg_type == 'nav_msgs/msg/Odometry':
            position = msg.pose.pose.position
            for axis in ('x', 'y', 'z'):
                path = f'pose.position.{axis}'
                setattr(
                    position,
                    axis,
                    float(self._apply_scalar(path, float(getattr(position, axis)), dt, base='pose'))
                )
            if apply_orientation:
                for axis in ('x', 'y', 'z', 'w'):
                    path = f'pose.orientation.{axis}'
                    setattr(
                        msg.pose.pose.orientation,
                        axis,
                        float(
                            self._apply_scalar(
                                path,
                                float(getattr(msg.pose.pose.orientation, axis)),
                                dt,
                                base='pose',
                            )
                        ),
                    )
            for part_name in ('linear', 'angular'):
                part = getattr(msg.twist.twist, part_name)
                for axis in ('x', 'y', 'z'):
                    path = f'twist.{part_name}.{axis}'
                    setattr(
                        part,
                        axis,
                        float(self._apply_scalar(path, float(getattr(part, axis)), dt, base='twist'))
                    )
            return

        if msg_type == 'sensor_msgs/msg/NavSatFix':
            msg.latitude = float(self._apply_scalar('latitude', float(msg.latitude), dt, base='latitude'))
            msg.longitude = float(self._apply_scalar('longitude', float(msg.longitude), dt, base='longitude'))
            msg.altitude = float(self._apply_scalar('altitude', float(msg.altitude), dt, base='altitude'))
            return

        self._apply_generic(msg, dt, apply_orientation)

    def _apply_generic(self, msg, dt: float, apply_orientation: bool) -> None:
        if not hasattr(msg, '__slots__'):
            return
        for field in msg.__slots__:
            if field in ('header',):
                continue
            if 'covariance' in field:
                continue
            if field == 'orientation' and not apply_orientation:
                continue
            value = getattr(msg, field)
            if isinstance(value, (float, int)):
                if isinstance(value, bool) or isinstance(value, int):
                    continue
                setattr(msg, field, float(self._apply_scalar(field, float(value), dt, base=field)))
                continue
            if isinstance(value, (list, tuple)):
                if not value or not isinstance(value[0], (float, int)):
                    continue
                new_list = []
                for i, val in enumerate(value):
                    if isinstance(val, bool) or isinstance(val, int):
                        new_list.append(val)
                        continue
                    new_list.append(
                        float(self._apply_scalar(f'{field}[{i}]', float(val), dt, index=i, base=field))
                    )
                setattr(msg, field, type(value)(new_list))
                continue
            if hasattr(value, '__slots__'):
                self._apply_generic(value, dt, apply_orientation)

    def _apply_scalar(
        self,
        path: str,
        value: float,
        dt: float,
        *,
        index: Optional[int] = None,
        base: Optional[str] = None,
    ) -> float:
        if not self._should_apply(path):
            return value

        noise_std = self._resolve_param(self.cfg.noise_std, path, index=index, base=base, default=0.0)
        bias_init = self._resolve_param(self.cfg.bias_init, path, index=index, base=base, default=0.0)
        bias_rw_std = self._resolve_param(self.cfg.bias_rw_std, path, index=index, base=base, default=0.0)
        sat_min = self._resolve_param(self.cfg.saturation_min, path, index=index, base=base, default=None)
        sat_max = self._resolve_param(self.cfg.saturation_max, path, index=index, base=base, default=None)

        bias_key = path
        if bias_key not in self.bias_state:
            self.bias_state[bias_key] = float(bias_init)

        if bias_rw_std and dt > 0.0:
            self.bias_state[bias_key] += float(self.rng.normal(0.0, bias_rw_std * math.sqrt(dt)))

        value = float(value) + self.bias_state[bias_key]
        if noise_std and noise_std > 0.0:
            value += float(self.rng.normal(0.0, noise_std))

        if sat_min is not None:
            value = max(float(sat_min), value)
        if sat_max is not None:
            value = min(float(sat_max), value)

        return value

    def _resolve_param(self, param, path: str, *, index: Optional[int], base: Optional[str], default):
        if param is None:
            return default
        if isinstance(param, (float, int)):
            return float(param)
        if isinstance(param, list):
            if index is None or index >= len(param):
                return default
            return float(param[index])
        if isinstance(param, dict):
            if path in param:
                return float(param[path])
            if base and base in param:
                return self._resolve_param(param[base], path, index=index, base=base, default=default)
            return default
        return default

    def _should_apply(self, path: str) -> bool:
        fields = self.cfg.fields
        if not fields:
            return True
        for field in fields:
            if path == field:
                return True
            if path.startswith(f'{field}.'):
                return True
            if path.startswith(f'{field}['):
                return True
        return False


class MeasurementNode(Node):
    def __init__(self):
        super().__init__('measurement_node')

        default_config = self._default_config_path()
        self.declare_parameter('config_path', default_config)
        config_path = self.get_parameter('config_path').value

        config = self._load_config(config_path)
        seed = int(config.get('seed', 0) or 0)

        self.processors = []
        signals = config.get('signals', {})
        if not isinstance(signals, dict):
            signals = {}

        for name, raw_cfg in signals.items():
            if not isinstance(raw_cfg, dict):
                continue
            if raw_cfg.get('plot_only', False):
                continue
            cfg = self._parse_signal(name, raw_cfg)
            if not cfg.enabled:
                continue
            if not cfg.input_topic or not cfg.output_topic or not cfg.msg_type:
                self.get_logger().warn(
                    f'Skipping signal {name}: missing input_topic/output_topic/msg_type'
                )
                continue
            signal_seed = self._derive_seed(seed, name)
            self.processors.append(SignalProcessor(self, cfg, signal_seed))

        self.get_logger().info('MeasurementNode started')
        self.get_logger().info('Measurement mapping table:')
        for proc in self.processors:
            self.get_logger().info(
                f"  {proc.cfg.input_topic} -> {proc.cfg.output_topic} "
                f"({proc.cfg.msg_type}), enabled={proc.cfg.enabled}, "
                f"rate={proc.cfg.rate_hz}Hz, latency={proc.cfg.latency_ms}ms"
            )

        self.process_timer = self.create_timer(0.01, self._process_all)

    def _process_all(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        for proc in self.processors:
            proc.process(now_sec)

    def _default_config_path(self) -> str:
        try:
            sim_car_share = get_package_share_directory('sim_car')
            return f"{sim_car_share}/config/sensor_config.yaml"
        except Exception:
            return ''

    def _load_config(self, path: str) -> Dict[str, Any]:
        if not path:
            self.get_logger().warn('No config_path provided; starting with empty config')
            return {}
        try:
            with open(path, 'r') as config_file:
                return yaml.safe_load(config_file) or {}
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().error(f'Failed to load config {path}: {exc}')
            return {}

    def _parse_signal(self, name: str, raw: Dict[str, Any]) -> SignalConfig:
        return SignalConfig(
            name=name,
            enabled=bool(raw.get('enabled', True)),
            input_topic=str(raw.get('input_topic', '')),
            output_topic=str(raw.get('output_topic', '')),
            msg_type=str(raw.get('msg_type', '')),
            rate_hz=float(raw.get('rate_hz', 0.0) or 0.0),
            latency_ms=float(raw.get('latency_ms', 0.0) or 0.0),
            dropout_prob=float(raw.get('dropout_prob', 0.0) or 0.0),
            noise_std=raw.get('noise_std', 0.0),
            bias_init=raw.get('bias_init', 0.0),
            bias_rw_std=raw.get('bias_rw_std', 0.0),
            saturation_min=raw.get('saturation_min', None),
            saturation_max=raw.get('saturation_max', None),
            apply_orientation=bool(raw.get('apply_orientation', False)),
            fields=raw.get('fields', None),
        )

    def _derive_seed(self, base: int, name: str) -> int:
        name_sum = sum(bytearray(name.encode('utf-8')))
        return (int(base) + name_sum) % (2**32)


def main(args=None):
    rclpy.init(args=args)
    node = MeasurementNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
