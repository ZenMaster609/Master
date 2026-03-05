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

    def __init__(
        self,
        session_path: Path,
        *,
        metrics_filename: str = 'cone_metrics.csv',
        range_rmse_filename: str = 'cone_range_rmse_samples.csv',
        output_suffix: str = '',
    ):
        self.session_path = Path(session_path)
        self.metrics_filename = str(metrics_filename).strip() or 'cone_metrics.csv'
        self.range_rmse_filename = str(range_rmse_filename).strip() or 'cone_range_rmse_samples.csv'
        self.output_suffix = str(output_suffix).strip()

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
        csv_path = self.session_path / 'logs' / self.metrics_filename
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
        csv_path = self.session_path / 'logs' / self.range_rmse_filename
        if not csv_path.exists():
            return None

        sources: List[str] = []
        gt_ranges: List[float] = []
        error_values: List[float] = []
        predicted_class_ids: List[float] = []
        ground_truth_class_ids: List[float] = []
        with open(csv_path, 'r', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                gt_raw = row.get('gt_range_m', '')
                error_raw = row.get('error_m', '')
                source_raw = row.get('source', 'unknown')
                source = str(source_raw).strip().lower() or 'unknown'
                gt = float('nan')
                error = float('nan')
                try:
                    gt = float(str(gt_raw).strip())
                except ValueError:
                    gt = float('nan')
                if error_raw is not None and str(error_raw).strip() != '':
                    try:
                        error = float(str(error_raw).strip())
                    except ValueError:
                        error = float('nan')
                else:
                    ex_raw = row.get('ex_m', '')
                    ey_raw = row.get('ey_m', '')
                    try:
                        ex = float(str(ex_raw).strip())
                        ey = float(str(ey_raw).strip())
                    except ValueError:
                        ex = float('nan')
                        ey = float('nan')
                    if math.isfinite(ex) and math.isfinite(ey):
                        error = math.hypot(ex, ey)
                        source = 'stereo'
                if not (math.isfinite(gt) and math.isfinite(error)):
                    continue
                predicted_class_id = self._parse_optional_float(row.get('predicted_class_id', ''))
                ground_truth_class_id = self._parse_optional_float(row.get('ground_truth_class_id', ''))
                sources.append(source)
                gt_ranges.append(gt)
                error_values.append(error)
                predicted_class_ids.append(predicted_class_id)
                ground_truth_class_ids.append(ground_truth_class_id)

        if not gt_ranges:
            return None
        return {
            'source': np.asarray(sources, dtype=object),
            'gt_range_m': np.asarray(gt_ranges, dtype=np.float32),
            'error_m': np.asarray(error_values, dtype=np.float32),
            'predicted_class_id': np.asarray(predicted_class_ids, dtype=np.float32),
            'ground_truth_class_id': np.asarray(ground_truth_class_ids, dtype=np.float32),
        }

    @staticmethod
    def _parse_optional_float(value: object) -> float:
        text = '' if value is None else str(value).strip()
        if text == '':
            return float('nan')
        try:
            parsed = float(text)
        except ValueError:
            return float('nan')
        return parsed if math.isfinite(parsed) else float('nan')

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
        if output_path is not None:
            final_path = output_path
        else:
            suffix = f'_{self.output_suffix}' if self.output_suffix else ''
            final_path = plots_path / f'cone_depth_validation{suffix}.png'
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
        error = sample_data['error_m']
        source = sample_data['source']
        predicted_class_id = sample_data.get('predicted_class_id')
        ground_truth_class_id = sample_data.get('ground_truth_class_id')

        valid = (
            np.isfinite(gt)
            & np.isfinite(error)
            & (gt >= range_min)
            & (gt <= range_max)
        )
        if not np.any(valid):
            return None

        gt_valid = gt[valid]
        error_valid = error[valid]
        source_valid = source[valid]
        predicted_valid = (
            np.asarray(predicted_class_id, dtype=np.float32)[valid]
            if predicted_class_id is not None
            else np.full_like(gt_valid, np.nan, dtype=np.float32)
        )
        ground_truth_valid = (
            np.asarray(ground_truth_class_id, dtype=np.float32)[valid]
            if ground_truth_class_id is not None
            else np.full_like(gt_valid, np.nan, dtype=np.float32)
        )

        bin_indices = np.floor((gt_valid - range_min) / bin_width).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        counts = np.bincount(bin_indices, minlength=num_bins).astype(np.int32)
        source_order = ['monocular', 'stereo', 'lidar']
        source_colors = {
            'monocular': 'tab:blue',
            'stereo': 'tab:orange',
            'lidar': 'tab:green',
        }
        source_labels = {
            'monocular': 'mono_rmse',
            'stereo': 'stereo_rmse',
            'lidar': 'lidar_rmse',
        }
        rmse_by_source: Dict[str, np.ndarray] = {}
        present_sources = set(str(item) for item in source_valid.tolist())
        ordered_sources = source_order + sorted(s for s in present_sources if s not in source_order)
        ordered_sources = [s for s in ordered_sources if s in present_sources]
        for source_name in ordered_sources:
            source_mask = source_valid == source_name
            if not np.any(source_mask):
                continue
            source_bins = bin_indices[source_mask]
            source_error = error_valid[source_mask]
            source_counts = np.bincount(source_bins, minlength=num_bins).astype(np.int32)
            source_sq_sum = np.bincount(source_bins, weights=np.square(source_error), minlength=num_bins)
            rmse = np.full(num_bins, np.nan, dtype=np.float32)
            non_empty = source_counts > 0
            rmse[non_empty] = np.sqrt(source_sq_sum[non_empty] / source_counts[non_empty]).astype(np.float32)
            rmse_by_source[source_name] = rmse

        rmse_pct_by_source = self._compute_source_rmse_percent(bin_centers, rmse_by_source)
        total_rmse_pct = self._compute_total_rmse_percent(bin_centers, rmse_by_source)
        correct_count, incorrect_count = self._compute_classification_counts(predicted_valid, ground_truth_valid)

        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10.0, 7.0),
            gridspec_kw={'height_ratios': [3, 1]},
        )
        fig.suptitle('Cone Range-Binned RMSE', fontsize=14)
        ax_top_pct = ax_top.twinx()

        finite_parts = []
        finite_pct_parts = []
        for source_name in ordered_sources:
            rmse = rmse_by_source.get(source_name)
            if rmse is None or not np.any(np.isfinite(rmse)):
                continue
            ax_top.plot(
                bin_centers,
                rmse,
                label=source_labels.get(source_name, source_name),
                linewidth=2.0,
                color=source_colors.get(source_name, None),
            )
            finite_parts.append(rmse[np.isfinite(rmse)])
            rmse_pct = rmse_pct_by_source.get(source_name)
            if rmse_pct is not None and np.any(np.isfinite(rmse_pct)):
                ax_top_pct.plot(
                    bin_centers,
                    rmse_pct,
                    label=f"{source_labels.get(source_name, source_name)}_pct",
                    linewidth=1.8,
                    linestyle='--',
                    color=source_colors.get(source_name, None),
                    alpha=0.85,
                )
                finite_pct_parts.append(rmse_pct[np.isfinite(rmse_pct)])
        ax_top.set_xlim(range_min, range_max)
        ax_top.set_ylabel('RMSE (m)')
        ax_top_pct.set_ylabel('RMSE (%)')
        ax_top.grid(True, alpha=0.3)
        if finite_parts:
            handles = ax_top.get_lines() + ax_top_pct.get_lines()
            ax_top.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 0.88))

        total_rmse_text = 'n/a' if total_rmse_pct is None else f'{total_rmse_pct:.2f}%'
        ax_top.text(
            0.02,
            0.98,
            f'rmse_total={total_rmse_text}',
            transform=ax_top.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox={
                'boxstyle': 'round,pad=0.3',
                'facecolor': 'white',
                'edgecolor': '0.6',
                'alpha': 0.85,
            },
        )

        total_classified = correct_count + incorrect_count
        accuracy_text = (
            f'{(100.0 * float(correct_count) / float(total_classified)):.1f}%'
            if total_classified > 0
            else 'n/a'
        )
        ax_top.text(
            0.98,
            0.98,
            'Cone classification',
            transform=ax_top.transAxes,
            ha='right',
            va='top',
            fontweight='semibold',
            fontsize=10,
        )
        ax_top.text(
            0.98,
            0.90,
            f'Correct: {correct_count}\n'
            f'Incorrect: {incorrect_count}\n'
            f'Accuracy: {accuracy_text}',
            transform=ax_top.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            bbox={
                'boxstyle': 'round,pad=0.3',
                'facecolor': 'white',
                'edgecolor': '0.6',
                'alpha': 0.85,
            },
        )

        ymax = 1.0
        if finite_parts:
            finite_vals = np.concatenate(finite_parts)
            if finite_vals.size > 0:
                ymax = max(0.1, float(np.max(finite_vals)) * 1.2)
        ax_top.set_ylim(0.0, ymax)
        pct_ymax = 1.0
        if finite_pct_parts:
            finite_pct_vals = np.concatenate(finite_pct_parts)
            if finite_pct_vals.size > 0:
                pct_ymax = max(0.1, float(np.max(finite_pct_vals)) * 1.2)
        ax_top_pct.set_ylim(0.0, pct_ymax)

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
        if output_path is not None:
            final_path = output_path
        else:
            suffix = f'_{self.output_suffix}' if self.output_suffix else ''
            final_path = plots_path / f'cone_range_binned_rmse{suffix}.png'
        fig.savefig(final_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return final_path

    @staticmethod
    def _compute_total_rmse_percent(
        bin_centers: np.ndarray,
        rmse_by_source: Dict[str, np.ndarray],
    ) -> Optional[float]:
        percent_values = [
            arr[np.isfinite(arr)]
            for arr in OfflineConePlotter._compute_source_rmse_percent(bin_centers, rmse_by_source).values()
            if np.any(np.isfinite(arr))
        ]

        if not percent_values:
            return None
        return float(np.mean(np.concatenate(percent_values)))

    @staticmethod
    def _compute_source_rmse_percent(
        bin_centers: np.ndarray,
        rmse_by_source: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        centers = np.asarray(bin_centers, dtype=np.float32)
        if centers.size == 0:
            return {}

        source_rmse_pct: Dict[str, np.ndarray] = {}
        for source, rmse in rmse_by_source.items():
            rmse_arr = np.asarray(rmse, dtype=np.float32)
            if rmse_arr.shape != centers.shape:
                continue
            rmse_pct = np.full_like(rmse_arr, np.nan, dtype=np.float32)
            valid = np.isfinite(rmse_arr) & np.isfinite(centers) & (centers > 0.0)
            if np.any(valid):
                rmse_pct[valid] = (rmse_arr[valid] / centers[valid]) * 100.0
            source_rmse_pct[source] = rmse_pct
        return source_rmse_pct

    @staticmethod
    def _compute_classification_counts(
        predicted_class_id: np.ndarray,
        ground_truth_class_id: np.ndarray,
    ) -> tuple[int, int]:
        predicted = np.asarray(predicted_class_id, dtype=np.float32)
        ground_truth = np.asarray(ground_truth_class_id, dtype=np.float32)
        if predicted.shape != ground_truth.shape or predicted.size == 0:
            return 0, 0

        valid = np.isfinite(predicted) & np.isfinite(ground_truth)
        if not np.any(valid):
            return 0, 0

        matches = predicted[valid].astype(np.int32) == ground_truth[valid].astype(np.int32)
        correct_count = int(np.count_nonzero(matches))
        incorrect_count = int(np.count_nonzero(~matches))
        return correct_count, incorrect_count
