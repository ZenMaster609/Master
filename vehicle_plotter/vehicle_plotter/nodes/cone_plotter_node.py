#!/usr/bin/env python3
"""
ConePlotterNode - Live plots for stereo cone depth evaluation metrics.

Consumes CSV-style per-cone metrics from std_msgs/String and renders a
YAML-configured 2x5 layout using the existing PlotManager backend.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String
import yaml
from ament_index_python.packages import get_package_share_directory

from ..plotting.plot_manager import PlotManager
from ..plotting.plot_config import (
    AxisConfig,
    PlotConfig,
    PlotLayoutConfig,
    SeriesConfig,
    XAxisType,
)


@dataclass
class ConeRow:
    """Parsed row from /cone_depth_per_cone CSV text."""

    cone_id: str
    samples: int
    mae: Optional[float]
    rmse: Optional[float]
    rmse_x: Optional[float]
    rmse_y: Optional[float]
    dcam_inst: Optional[float]
    dgt_inst: Optional[float]
    dcam: Optional[float]
    dgt: Optional[float]


@dataclass
class ConePlotRuntimeConfig:
    """Runtime behavior loaded from YAML."""

    cone_topic: str
    cone_metrics_prefix: str
    top_cone_count: int
    total_average_stride: int


class ConePlotterNode(Node):
    """Plot per-cone depth metrics and aggregate error trends."""

    def __init__(self):
        super().__init__('cone_plotter_node')

        self.declare_parameter('backend', 'pyqtgraph')
        self.declare_parameter('enable_gui', True)
        self.declare_parameter('close_plots_on_shutdown', True)
        self.declare_parameter('config_path', '')
        self.declare_parameter('cone_topic', '')

        backend = str(self.get_parameter('backend').value)
        enable_gui = bool(self.get_parameter('enable_gui').value)
        self._close_plots_on_shutdown = bool(
            self.get_parameter('close_plots_on_shutdown').value
        )
        config_path = str(self.get_parameter('config_path').value).strip()
        cone_topic_override = str(self.get_parameter('cone_topic').value).strip()

        yaml_config = self._load_yaml_config(config_path or self._default_config_path())
        layout_config = self._build_layout(yaml_config)
        runtime_config = self._build_runtime_config(yaml_config, cone_topic_override)

        self._top_cone_count = runtime_config.top_cone_count
        self._total_average_stride = runtime_config.total_average_stride

        self.plot_manager = PlotManager(
            layout_config=layout_config,
            backend=backend,
            enable_gui=enable_gui,
        )
        self._window_open = True

        # Running aggregate of the per-frame all-cone averages, decimated.
        self._all_avg_update_count = 0
        self._total_avg_rmse_sum = 0.0
        self._total_avg_mae_sum = 0.0
        self._total_avg_rmse_count = 0
        self._total_avg_mae_count = 0
        self._avg_total_rmse: Optional[float] = None
        self._avg_total_mae: Optional[float] = None

        self._msg_count = 0
        self._latest_rows: List[ConeRow] = []
        self._slot_rows: List[Optional[ConeRow]] = [None] * self._top_cone_count
        self._latest_cone_metrics: Dict[str, Optional[float]] = {
            'cone_depth_pairs': None,
            'cone_depth_axis_mae_m': None,
            'cone_depth_axis_rmse_m': None,
            'cone_depth_axis_bias_m': None,
            'cone_depth_range_mae_m': None,
            'cone_depth_range_rmse_m': None,
            'cone_depth_sync_dt_ms': None,
        }

        self._cone_sub = self.create_subscription(
            String,
            runtime_config.cone_topic,
            self._cone_table_callback,
            10,
        )
        prefix = runtime_config.cone_metrics_prefix.rstrip('/')
        self._cone_pairs_sub = self.create_subscription(
            Int32,
            f'{prefix}/cone_depth_pairs',
            self._cone_pairs_callback,
            10,
        )
        self._cone_axis_mae_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_axis_mae_m',
            self._cone_axis_mae_callback,
            10,
        )
        self._cone_axis_rmse_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_axis_rmse_m',
            self._cone_axis_rmse_callback,
            10,
        )
        self._cone_axis_bias_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_axis_bias_m',
            self._cone_axis_bias_callback,
            10,
        )
        self._cone_range_mae_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_range_mae_m',
            self._cone_range_mae_callback,
            10,
        )
        self._cone_range_rmse_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_range_rmse_m',
            self._cone_range_rmse_callback,
            10,
        )
        self._cone_sync_dt_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_sync_dt_ms',
            self._cone_sync_dt_callback,
            10,
        )

        self.refresh_timer = self.create_timer(
            1.0 / layout_config.update_rate_hz,
            self._refresh_callback,
        )

        self.get_logger().info(
            'ConePlotterNode started '
            f'topic={runtime_config.cone_topic} '
            f'metrics_prefix={runtime_config.cone_metrics_prefix} '
            f'top_cones={self._top_cone_count} '
            f'total_avg_stride={self._total_average_stride}'
        )

    def _default_config_path(self) -> str:
        try:
            share = get_package_share_directory('vehicle_plotter')
            return str(Path(share) / 'config' / 'cone_plots.yaml')
        except Exception:
            return ''

    def _load_yaml_config(self, path: str) -> Dict:
        if not path:
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                config = yaml.safe_load(handle) or {}
            if not isinstance(config, dict):
                return {}
            return config
        except (OSError, yaml.YAMLError):
            self.get_logger().warn(f'Failed to load cone plot config at: {path}')
            return {}

    def _build_runtime_config(
        self,
        config: Dict,
        cone_topic_override: str,
    ) -> ConePlotRuntimeConfig:
        settings = config.get('settings', {}) if isinstance(config.get('settings'), dict) else {}
        topic = cone_topic_override or str(
            settings.get('cone_topic', '/sim/stereo/eval/cone_depth_per_cone')
        )
        metrics_prefix = topic
        if metrics_prefix.endswith('/cone_depth_per_cone'):
            metrics_prefix = metrics_prefix[: -len('/cone_depth_per_cone')]
        return ConePlotRuntimeConfig(
            cone_topic=topic,
            cone_metrics_prefix=metrics_prefix,
            top_cone_count=max(1, int(settings.get('top_cone_count', 4))),
            total_average_stride=max(1, int(settings.get('total_average_stride', 15))),
        )

    def _build_layout(self, config: Dict) -> PlotLayoutConfig:
        window_cfg = config.get('window', {}) if isinstance(config.get('window'), dict) else {}
        layout_cfg = config.get('layout', {}) if isinstance(config.get('layout'), dict) else {}
        plots_cfg = config.get('plots', []) if isinstance(config.get('plots'), list) else []

        rows = max(1, int(layout_cfg.get('rows', 2)))
        cols = max(1, int(layout_cfg.get('cols', 5)))
        width = max(640, int(window_cfg.get('width', 2200)))
        height = max(480, int(window_cfg.get('height', 950)))
        update_rate_hz = max(1.0, float(window_cfg.get('update_rate_hz', 15.0)))
        default_buffer_size = max(10, int(layout_cfg.get('buffer_size', 900)))

        plots: List[PlotConfig] = []
        for plot_entry in plots_cfg:
            if not isinstance(plot_entry, dict):
                continue

            name = str(plot_entry.get('name', f'plot_{len(plots)}'))
            series_cfg = plot_entry.get('series', [])
            if not isinstance(series_cfg, list):
                continue

            series: List[SeriesConfig] = []
            for series_entry in series_cfg:
                if not isinstance(series_entry, dict):
                    continue
                variable = str(series_entry.get('variable', '')).strip()
                if not variable:
                    continue
                line_style = str(series_entry.get('line_style', 'solid')).strip().lower()
                if line_style not in {'solid', 'dashed', 'dotted'}:
                    line_style = 'solid'
                series.append(
                    SeriesConfig(
                        name=str(series_entry.get('name', variable)),
                        variable=variable,
                        color=str(series_entry.get('color', 'auto')),
                        line_width=float(series_entry.get('line_width', 1.5)),
                        line_style=line_style,  # type: ignore[arg-type]
                        scale=float(series_entry.get('scale', 1.0)),
                        offset=float(series_entry.get('offset', 0.0)),
                    )
                )
            if not series:
                continue

            y_axis = self._parse_axis_config(plot_entry.get('y_axis'))
            x_axis = self._parse_axis_config(plot_entry.get('x_axis'))

            plot_type = str(plot_entry.get('type', 'timeseries')).strip().lower()
            if plot_type not in {'timeseries', 'xy', 'histogram'}:
                plot_type = 'timeseries'

            x_axis_type_name = str(plot_entry.get('x_axis_type', 'time')).strip().lower()
            x_axis_type = self._x_axis_type_from_text(x_axis_type_name)

            plots.append(
                PlotConfig(
                    name=name,
                    series=series,
                    x_axis_type=x_axis_type,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    plot_type=plot_type,  # type: ignore[arg-type]
                    buffer_size=max(10, int(plot_entry.get('buffer_size', default_buffer_size))),
                    show_legend=bool(plot_entry.get('show_legend', True)),
                    show_grid=bool(plot_entry.get('show_grid', True)),
                    row=max(0, int(plot_entry.get('row', 0))),
                    col=max(0, int(plot_entry.get('col', 0))),
                    row_span=max(1, int(plot_entry.get('row_span', 1))),
                    col_span=max(1, int(plot_entry.get('col_span', 1))),
                )
            )

        return PlotLayoutConfig(
            plots=plots,
            rows=rows,
            cols=cols,
            window_title=str(window_cfg.get('title', 'Cone Depth Validation')),
            window_size=(width, height),
            dark_mode=bool(window_cfg.get('dark_mode', True)),
            update_rate_hz=update_rate_hz,
        )

    def _parse_axis_config(self, raw_axis) -> Optional[AxisConfig]:
        if not isinstance(raw_axis, dict):
            return None
        label = str(raw_axis.get('label', '')).strip()
        unit = str(raw_axis.get('unit', '')).strip()
        variable = str(raw_axis.get('variable', '')).strip()
        limits = None
        raw_limits = raw_axis.get('limits')
        if isinstance(raw_limits, list) and len(raw_limits) == 2:
            limits = (float(raw_limits[0]), float(raw_limits[1]))
        return AxisConfig(
            label=label,
            variable=variable,
            unit=unit,
            scale=float(raw_axis.get('scale', 1.0)),
            offset=float(raw_axis.get('offset', 0.0)),
            limits=limits,
            auto_scale=bool(raw_axis.get('auto_scale', True)),
        )

    @staticmethod
    def _x_axis_type_from_text(value: str) -> XAxisType:
        mapping = {
            'time': XAxisType.TIME,
            'distance': XAxisType.DISTANCE,
            'yaw': XAxisType.YAW,
            'encoder_total': XAxisType.ENCODER_TOTAL,
            'encoder_fl': XAxisType.ENCODER_FL,
            'custom': XAxisType.CUSTOM,
        }
        return mapping.get(value, XAxisType.TIME)

    def _cone_table_callback(self, msg: String) -> None:
        rows = self._parse_cone_rows(msg.data)
        self._latest_rows = rows

        selected = self._select_slot_rows(rows)

        avg_rmse_now = self._mean([row.rmse for row in rows if row.rmse is not None])
        avg_mae_now = self._mean([row.mae for row in rows if row.mae is not None])

        self._all_avg_update_count += 1
        if self._all_avg_update_count % self._total_average_stride == 0:
            if avg_rmse_now is not None:
                self._total_avg_rmse_sum += avg_rmse_now
                self._total_avg_rmse_count += 1
                self._avg_total_rmse = self._total_avg_rmse_sum / self._total_avg_rmse_count
            if avg_mae_now is not None:
                self._total_avg_mae_sum += avg_mae_now
                self._total_avg_mae_count += 1
                self._avg_total_mae = self._total_avg_mae_sum / self._total_avg_mae_count

        state_values: Dict[str, float] = {
            'timestamp': self._now_sec(),
            'avg_rmse': self._to_plot_value(avg_rmse_now),
            'avg_mae': self._to_plot_value(avg_mae_now),
            'avg_total_rmse': self._to_plot_value(self._avg_total_rmse),
            'avg_total_mae': self._to_plot_value(self._avg_total_mae),
        }

        for idx in range(self._top_cone_count):
            prefix = f'cone_{idx + 1}'
            row = selected[idx] if idx < len(selected) else None
            dcam_plot = (
                row.dcam_inst if row and row.dcam_inst is not None
                else (row.dcam if row else None)
            )
            dgt_plot = (
                row.dgt_inst if row and row.dgt_inst is not None
                else (row.dgt if row else None)
            )
            state_values[f'{prefix}_rmse'] = self._to_plot_value(row.rmse if row else None)
            state_values[f'{prefix}_mae'] = self._to_plot_value(row.mae if row else None)
            state_values[f'{prefix}_rmse_x'] = self._to_plot_value(
                row.rmse_x if row and row.rmse_x is not None else (row.rmse if row else None)
            )
            state_values[f'{prefix}_rmse_y'] = self._to_plot_value(
                row.rmse_y if row and row.rmse_y is not None else (row.rmse if row else None)
            )
            state_values[f'{prefix}_dcam'] = self._to_plot_value(dcam_plot)
            state_values[f'{prefix}_dgt'] = self._to_plot_value(dgt_plot)
            state_values[f'{prefix}_dcam_inst'] = self._to_plot_value(row.dcam_inst if row else None)
            state_values[f'{prefix}_dgt_inst'] = self._to_plot_value(row.dgt_inst if row else None)
            state_values[f'{prefix}_dcam_avg'] = self._to_plot_value(row.dcam if row else None)
            state_values[f'{prefix}_dgt_avg'] = self._to_plot_value(row.dgt if row else None)

        for key, value in self._latest_cone_metrics.items():
            state_values[key] = self._to_plot_value(value)

        self.plot_manager.push_state(SimpleNamespace(**state_values))
        self._msg_count += 1

    def _push_state_from_latest(self) -> None:
        rows = self._latest_rows
        selected = list(self._slot_rows)
        avg_rmse_now = self._mean([row.rmse for row in rows if row.rmse is not None])
        avg_mae_now = self._mean([row.mae for row in rows if row.mae is not None])

        state_values: Dict[str, float] = {
            'timestamp': self._now_sec(),
            'avg_rmse': self._to_plot_value(avg_rmse_now),
            'avg_mae': self._to_plot_value(avg_mae_now),
            'avg_total_rmse': self._to_plot_value(self._avg_total_rmse),
            'avg_total_mae': self._to_plot_value(self._avg_total_mae),
        }
        for idx in range(self._top_cone_count):
            prefix = f'cone_{idx + 1}'
            row = selected[idx] if idx < len(selected) else None
            dcam_plot = (
                row.dcam_inst if row and row.dcam_inst is not None
                else (row.dcam if row else None)
            )
            dgt_plot = (
                row.dgt_inst if row and row.dgt_inst is not None
                else (row.dgt if row else None)
            )
            state_values[f'{prefix}_rmse'] = self._to_plot_value(row.rmse if row else None)
            state_values[f'{prefix}_mae'] = self._to_plot_value(row.mae if row else None)
            state_values[f'{prefix}_rmse_x'] = self._to_plot_value(
                row.rmse_x if row and row.rmse_x is not None else (row.rmse if row else None)
            )
            state_values[f'{prefix}_rmse_y'] = self._to_plot_value(
                row.rmse_y if row and row.rmse_y is not None else (row.rmse if row else None)
            )
            state_values[f'{prefix}_dcam'] = self._to_plot_value(dcam_plot)
            state_values[f'{prefix}_dgt'] = self._to_plot_value(dgt_plot)
            state_values[f'{prefix}_dcam_inst'] = self._to_plot_value(row.dcam_inst if row else None)
            state_values[f'{prefix}_dgt_inst'] = self._to_plot_value(row.dgt_inst if row else None)
            state_values[f'{prefix}_dcam_avg'] = self._to_plot_value(row.dcam if row else None)
            state_values[f'{prefix}_dgt_avg'] = self._to_plot_value(row.dgt if row else None)
        for key, value in self._latest_cone_metrics.items():
            state_values[key] = self._to_plot_value(value)

        self.plot_manager.push_state(SimpleNamespace(**state_values))
        self._msg_count += 1

    def _set_metric(self, key: str, value: Optional[float]) -> None:
        self._latest_cone_metrics[key] = value
        self._push_state_from_latest()

    def _cone_pairs_callback(self, msg: Int32) -> None:
        self._set_metric('cone_depth_pairs', float(msg.data))

    def _cone_axis_mae_callback(self, msg: Float32) -> None:
        self._set_metric('cone_depth_axis_mae_m', float(msg.data))

    def _cone_axis_rmse_callback(self, msg: Float32) -> None:
        self._set_metric('cone_depth_axis_rmse_m', float(msg.data))

    def _cone_axis_bias_callback(self, msg: Float32) -> None:
        self._set_metric('cone_depth_axis_bias_m', float(msg.data))

    def _cone_range_mae_callback(self, msg: Float32) -> None:
        self._set_metric('cone_depth_range_mae_m', float(msg.data))

    def _cone_range_rmse_callback(self, msg: Float32) -> None:
        self._set_metric('cone_depth_range_rmse_m', float(msg.data))

    def _cone_sync_dt_callback(self, msg: Float32) -> None:
        self._set_metric('cone_depth_sync_dt_ms', float(msg.data))

    def _refresh_callback(self) -> None:
        if not self._window_open:
            return
        self._window_open = self.plot_manager.refresh()
        if not self._window_open:
            self.get_logger().info('Cone plot window closed, shutting down')
            self.shutdown()

    def shutdown(self) -> None:
        if self._close_plots_on_shutdown:
            self.plot_manager.close()
        rclpy.shutdown()

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    @staticmethod
    def _to_plot_value(value: Optional[float]) -> float:
        if value is None:
            return float('nan')
        if not math.isfinite(value):
            return float('nan')
        return float(value)

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return float(sum(values) / len(values))

    @staticmethod
    def _cone_distance_sort_key(row: ConeRow):
        dgt = (
            row.dgt_inst if row.dgt_inst is not None
            else (row.dgt if row.dgt is not None else math.inf)
        )
        dcam = (
            row.dcam_inst if row.dcam_inst is not None
            else (row.dcam if row.dcam is not None else math.inf)
        )
        return (dgt, dcam, row.cone_id)

    def _select_slot_rows(self, rows: List[ConeRow]) -> List[Optional[ConeRow]]:
        ranked_rows = sorted(rows, key=self._cone_distance_sort_key)
        if not ranked_rows:
            return list(self._slot_rows)

        by_id: Dict[str, ConeRow] = {}
        for row in ranked_rows:
            if row.cone_id not in by_id:
                by_id[row.cone_id] = row

        selected: List[Optional[ConeRow]] = [None] * self._top_cone_count
        used_ids = set()

        # Prefer keeping prior cone-to-column assignment stable.
        for idx, prev in enumerate(self._slot_rows):
            if prev is None:
                continue
            current = by_id.get(prev.cone_id)
            if current is None:
                continue
            selected[idx] = current
            used_ids.add(prev.cone_id)

        for row in ranked_rows:
            if row.cone_id in used_ids:
                continue
            try:
                target_idx = selected.index(None)
            except ValueError:
                break
            selected[target_idx] = row
            used_ids.add(row.cone_id)

        # Keep previous values for temporarily missing cone slots.
        for idx, row in enumerate(selected):
            if row is None:
                selected[idx] = self._slot_rows[idx]

        self._slot_rows = selected
        return selected

    @staticmethod
    def _parse_cone_rows(text: str) -> List[ConeRow]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []
        if lines[0].startswith('no per-cone depth samples'):
            return []

        header_map: Dict[str, int] = {}
        start_idx = 0
        if lines[0].startswith('cone_id,'):
            start_idx = 1
            header_parts = [part.strip() for part in lines[0].split(',')]
            header_map = {name: idx for idx, name in enumerate(header_parts)}
        parsed_rows: List[ConeRow] = []

        for line in lines[start_idx:]:
            if line.startswith('...'):
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) < 7:
                continue

            def get_value(key: str, fallback_idx: Optional[int] = None) -> Optional[str]:
                idx = header_map.get(key)
                if idx is None:
                    idx = fallback_idx
                if idx is None or idx < 0 or idx >= len(parts):
                    return None
                return parts[idx]

            parsed_rows.append(
                ConeRow(
                    cone_id=parts[0],
                    samples=ConePlotterNode._parse_int(get_value('samples', 2) or '0'),
                    mae=ConePlotterNode._parse_float(get_value('axis_mae_m', 3) or ''),
                    rmse=ConePlotterNode._parse_float(get_value('axis_rmse_m', 4) or ''),
                    rmse_x=ConePlotterNode._parse_float(get_value('axis_rmse_x_m', None) or ''),
                    rmse_y=ConePlotterNode._parse_float(get_value('axis_rmse_y_m', None) or ''),
                    dcam_inst=ConePlotterNode._parse_float(get_value('dcam_inst', None) or ''),
                    dgt_inst=ConePlotterNode._parse_float(get_value('dgt_inst', None) or ''),
                    dcam=ConePlotterNode._parse_float(get_value('dcam', 5) or ''),
                    dgt=ConePlotterNode._parse_float(get_value('dgt', 6) or ''),
                )
            )

        return parsed_rows

    @staticmethod
    def _parse_float(value: str) -> Optional[float]:
        lowered = value.strip().lower()
        if lowered in {'', 'n/a', 'nan', 'none'}:
            return None
        try:
            number = float(lowered)
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        return number

    @staticmethod
    def _parse_int(value: str) -> int:
        try:
            return max(0, int(value))
        except ValueError:
            return 0


def main(args=None):
    rclpy.init(args=args)
    node = ConePlotterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.get_logger().info(
                f'ConePlotterNode shutting down, received {node._msg_count} updates'
            )
        if node._close_plots_on_shutdown:
            node.plot_manager.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
