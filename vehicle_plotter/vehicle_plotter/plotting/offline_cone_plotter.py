"""
Offline headless cone plotter for cone depth evaluation metrics.

Reads logs/cone_metrics.csv and renders a static plot using the same layout
defined in config/cone_plots.yaml.
"""

from pathlib import Path
from typing import Dict, List, Optional
import csv
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory


class OfflineConePlotter:
    """Generate static cone-depth plots from recorded cone metric CSV data."""

    def __init__(self, session_path: Path):
        self.session_path = Path(session_path)

    def _default_config_path(self) -> Path:
        share = Path(get_package_share_directory('vehicle_plotter'))
        return share / 'config' / 'cone_plots.yaml'

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        path = config_path if config_path is not None else self._default_config_path()
        with open(path, 'r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            return {}
        return config

    def _load_data(self) -> Optional[List[Dict[str, float]]]:
        csv_path = self.session_path / 'logs' / 'cone_metrics.csv'
        if not csv_path.exists():
            return None
        rows: List[Dict[str, float]] = []
        with open(csv_path, 'r', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed: Dict[str, float] = {}
                for key, value in row.items():
                    if value is None:
                        parsed[key] = float('nan')
                        continue
                    text = str(value).strip()
                    if text == '':
                        parsed[key] = float('nan')
                        continue
                    try:
                        number = float(text)
                    except ValueError:
                        parsed[key] = float('nan')
                    else:
                        parsed[key] = number if math.isfinite(number) else float('nan')
                rows.append(parsed)
        if not rows:
            return None
        return rows

    def _load_range_rmse_samples(self) -> Optional[Dict[str, np.ndarray]]:
        csv_path = self.session_path / 'logs' / 'cone_range_rmse_samples.csv'
        if not csv_path.exists():
            return None

        gt_ranges: List[float] = []
        ex_values: List[float] = []
        ey_values: List[float] = []
        with open(csv_path, 'r', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                gt_raw = row.get('gt_range_m', '')
                ex_raw = row.get('ex_m', '')
                ey_raw = row.get('ey_m', '')
                try:
                    gt = float(str(gt_raw).strip())
                    ex = float(str(ex_raw).strip())
                    ey = float(str(ey_raw).strip())
                except ValueError:
                    continue
                if not (math.isfinite(gt) and math.isfinite(ex) and math.isfinite(ey)):
                    continue
                gt_ranges.append(gt)
                ex_values.append(ex)
                ey_values.append(ey)

        if not gt_ranges:
            return None
        return {
            'gt_range_m': np.asarray(gt_ranges, dtype=np.float32),
            'ex_m': np.asarray(ex_values, dtype=np.float32),
            'ey_m': np.asarray(ey_values, dtype=np.float32),
        }

    @staticmethod
    def _series_present(rows: List[Dict[str, float]], name: str) -> bool:
        for row in rows:
            value = row.get(name, float('nan'))
            if math.isfinite(value):
                return True
        return False

    def generate_plot(
        self,
        config_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        dpi: int = 150,
    ) -> Optional[Path]:
        rows_data = self._load_data()
        if rows_data is None:
            return None

        config = self._load_config(config_path)
        window_cfg = config.get('window', {}) if isinstance(config.get('window'), dict) else {}
        layout_cfg = config.get('layout', {}) if isinstance(config.get('layout'), dict) else {}
        plots_cfg = config.get('plots', []) if isinstance(config.get('plots'), list) else []

        rows = max(1, int(layout_cfg.get('rows', 2)))
        cols = max(1, int(layout_cfg.get('cols', 5)))
        width_px = max(800, int(window_cfg.get('width', 2200)))
        height_px = max(500, int(window_cfg.get('height', 950)))
        fig_w = width_px / 110.0
        fig_h = height_px / 110.0

        time_sec = [row.get('timestamp', float('nan')) for row in rows_data]
        valid_times = [t for t in time_sec if math.isfinite(t)]
        if not valid_times:
            return None
        t0 = min(valid_times)
        rel_t = [t - t0 if math.isfinite(t) else float('nan') for t in time_sec]

        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
        fig.suptitle(str(window_cfg.get('title', 'Cone Depth Validation')), fontsize=14)

        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time (s)')

        for plot in plots_cfg:
            if not isinstance(plot, dict):
                continue
            row = max(0, min(rows - 1, int(plot.get('row', 0))))
            col = max(0, min(cols - 1, int(plot.get('col', 0))))
            ax = axes[row][col]
            ax.set_title(str(plot.get('name', 'plot')))

            y_axis = plot.get('y_axis', {})
            if isinstance(y_axis, dict):
                label = str(y_axis.get('label', '')).strip()
                unit = str(y_axis.get('unit', '')).strip()
                if label and unit:
                    ax.set_ylabel(f'{label} ({unit})')
                elif label:
                    ax.set_ylabel(label)
                raw_limits = y_axis.get('limits')
                if isinstance(raw_limits, list) and len(raw_limits) == 2:
                    try:
                        y_min = float(raw_limits[0])
                        y_max = float(raw_limits[1])
                        ax.set_ylim(y_min, y_max)
                    except (TypeError, ValueError):
                        pass

            series_cfg = plot.get('series', [])
            if not isinstance(series_cfg, list):
                continue

            any_series = False
            for series in series_cfg:
                if not isinstance(series, dict):
                    continue
                variable = str(series.get('variable', '')).strip()
                if not variable or not self._series_present(rows_data, variable):
                    continue
                y = [row.get(variable, float('nan')) for row in rows_data]
                color = str(series.get('color', '#1f77b4'))
                label = str(series.get('name', variable))
                ax.plot(rel_t, y, color=color, linewidth=1.3, label=label)
                any_series = True

            if any_series:
                ax.legend(fontsize='small')
            else:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)

        fig.tight_layout()
        plots_path = self.session_path / 'plots'
        plots_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path if output_path is not None else plots_path / 'cone_depth_validation.png'
        fig.savefig(final_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return final_path

    def generate_range_rmse_plot(
        self,
        output_path: Optional[Path] = None,
        dpi: int = 150,
    ) -> Optional[Path]:
        sample_data = self._load_range_rmse_samples()
        if sample_data is None:
            return None

        range_min = 0.0
        range_max = 20.0
        bin_width = 1.0
        num_bins = 20
        bin_centers = range_min + (np.arange(num_bins, dtype=np.float32) + 0.5) * bin_width

        gt = sample_data['gt_range_m']
        ex = sample_data['ex_m']
        ey = sample_data['ey_m']

        valid = (
            np.isfinite(gt)
            & np.isfinite(ex)
            & np.isfinite(ey)
            & (gt >= range_min)
            & (gt <= range_max)
        )
        if not np.any(valid):
            return None

        gt_valid = gt[valid]
        ex_valid = ex[valid]
        ey_valid = ey[valid]

        bin_indices = np.floor((gt_valid - range_min) / bin_width).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        counts = np.bincount(bin_indices, minlength=num_bins).astype(np.int32)
        ex_sq_sum = np.bincount(bin_indices, weights=np.square(ex_valid), minlength=num_bins)
        ey_sq_sum = np.bincount(bin_indices, weights=np.square(ey_valid), minlength=num_bins)

        rmse_x = np.full(num_bins, np.nan, dtype=np.float32)
        rmse_y = np.full(num_bins, np.nan, dtype=np.float32)
        non_empty = counts > 0
        rmse_x[non_empty] = np.sqrt(ex_sq_sum[non_empty] / counts[non_empty]).astype(np.float32)
        rmse_y[non_empty] = np.sqrt(ey_sq_sum[non_empty] / counts[non_empty]).astype(np.float32)

        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10.0, 7.0),
            gridspec_kw={'height_ratios': [3, 1]},
        )
        fig.suptitle('Cone Range-Binned RMSE', fontsize=14)

        ax_top.plot(bin_centers, rmse_x, label='RMSE_x', linewidth=2.0)
        ax_top.plot(bin_centers, rmse_y, label='RMSE_y', linewidth=2.0)
        ax_top.set_xlim(range_min, range_max)
        ax_top.set_ylabel('RMSE (m)')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend()

        ymax = 1.0
        finite_vals = np.concatenate((rmse_x[np.isfinite(rmse_x)], rmse_y[np.isfinite(rmse_y)]))
        if finite_vals.size > 0:
            ymax = max(0.1, float(np.max(finite_vals)) * 1.2)
        ax_top.set_ylim(0.0, ymax)

        ax_bottom.bar(
            bin_centers,
            counts.astype(np.float32),
            width=0.9 * bin_width,
            align='center',
            color='tab:gray',
        )
        ax_bottom.set_xlim(range_min, range_max)
        ax_bottom.set_xlabel('Ground-truth range (m)')
        ax_bottom.set_ylabel('Samples')
        ax_bottom.grid(True, alpha=0.3)

        max_count = int(np.max(counts)) if counts.size > 0 else 0
        ax_bottom.set_ylim(0.0, max(1.0, float(max_count) * 1.2))

        fig.tight_layout()
        plots_path = self.session_path / 'plots'
        plots_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path if output_path is not None else plots_path / 'cone_range_binned_rmse.png'
        fig.savefig(final_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return final_path
