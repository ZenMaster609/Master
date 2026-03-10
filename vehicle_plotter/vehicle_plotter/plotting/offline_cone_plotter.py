"""Offline headless cone plotter for range-binned cone RMSE metrics."""

from pathlib import Path
from typing import Dict, List, Optional
import csv
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class OfflineConePlotter:
    """Generate static cone range-RMSE plots from recorded CSV data."""

    def __init__(
        self,
        session_path: Path,
        *,
        range_rmse_filename: str = 'cone_range_rmse_samples.csv',
        output_suffix: str = '',
    ):
        self.session_path = Path(session_path)
        self.range_rmse_filename = str(range_rmse_filename).strip() or 'cone_range_rmse_samples.csv'
        self.output_suffix = str(output_suffix).strip()

    def _load_range_rmse_samples(self) -> Optional[Dict[str, np.ndarray]]:
        return self._load_range_rmse_samples_from_filename(self.range_rmse_filename)

    def _load_range_rmse_samples_from_filename(self, filename: str) -> Optional[Dict[str, np.ndarray]]:
        csv_path = self.session_path / 'logs' / filename
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
    def _summarize_range_rmse_samples(sample_data: Dict[str, np.ndarray]) -> Optional[Dict[str, object]]:
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

        valid = np.isfinite(gt) & np.isfinite(error) & (gt >= range_min) & (gt <= range_max)
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

        rmse_pct_by_source = OfflineConePlotter._compute_source_rmse_percent(bin_centers, rmse_by_source)
        total_rmse_pct = OfflineConePlotter._compute_total_rmse_percent(bin_centers, rmse_by_source)
        correct_count, incorrect_count = OfflineConePlotter._compute_classification_counts(predicted_valid, ground_truth_valid)
        return {
            'range_min': range_min,
            'range_max': range_max,
            'bin_width': bin_width,
            'bin_centers': bin_centers,
            'counts': counts,
            'ordered_sources': ordered_sources,
            'source_colors': source_colors,
            'source_labels': source_labels,
            'rmse_by_source': rmse_by_source,
            'rmse_pct_by_source': rmse_pct_by_source,
            'total_rmse_pct': total_rmse_pct,
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
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

    def generate_range_rmse_plot(
        self,
        output_path: Optional[Path] = None,
        dpi: int = 150,
    ) -> Optional[Path]:
        sample_data = self._load_range_rmse_samples()
        if sample_data is None:
            return None
        summary = self._summarize_range_rmse_samples(sample_data)
        if summary is None:
            return None

        range_min = float(summary['range_min'])
        range_max = float(summary['range_max'])
        bin_width = float(summary['bin_width'])
        bin_centers = np.asarray(summary['bin_centers'], dtype=np.float32)
        counts = np.asarray(summary['counts'], dtype=np.int32)
        ordered_sources = list(summary['ordered_sources'])
        source_colors = dict(summary['source_colors'])
        source_labels = dict(summary['source_labels'])
        rmse_by_source = dict(summary['rmse_by_source'])
        rmse_pct_by_source = dict(summary['rmse_pct_by_source'])
        total_rmse_pct = summary['total_rmse_pct']
        correct_count = int(summary['correct_count'])
        incorrect_count = int(summary['incorrect_count'])

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

    def generate_combined_range_rmse_plot(
        self,
        *,
        left_range_rmse_filename: Optional[str] = None,
        right_range_rmse_filename: str = 'cone_range_rmse_samples_lidar.csv',
        output_path: Optional[Path] = None,
        dpi: int = 150,
    ) -> Optional[Path]:
        left_filename = str(left_range_rmse_filename).strip() if left_range_rmse_filename is not None else self.range_rmse_filename
        left_data = self._load_range_rmse_samples_from_filename(left_filename)
        right_data = self._load_range_rmse_samples_from_filename(right_range_rmse_filename)
        left_summary = self._summarize_range_rmse_samples(left_data) if left_data is not None else None
        right_summary = self._summarize_range_rmse_samples(right_data) if right_data is not None else None
        if left_summary is None and right_summary is None:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0), squeeze=False)
        fig.suptitle('Cone Range-Binned RMSE (Camera + LiDAR)', fontsize=14)

        for idx, summary in enumerate([left_summary, right_summary]):
            ax = axes[0][idx]
            ax_pct = ax.twinx()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Ground-truth range (m)')
            ax.set_ylabel('RMSE (m)')
            ax_pct.set_ylabel('RMSE (%)')
            if summary is None:
                ax.set_title('no data')
                ax.text(0.5, 0.5, 'no samples', ha='center', va='center', transform=ax.transAxes)
                continue

            bin_centers = np.asarray(summary['bin_centers'], dtype=np.float32)
            ordered_sources = list(summary['ordered_sources'])
            source_colors = dict(summary['source_colors'])
            source_labels = dict(summary['source_labels'])
            rmse_by_source = dict(summary['rmse_by_source'])
            rmse_pct_by_source = dict(summary['rmse_pct_by_source'])
            total_rmse_pct = summary['total_rmse_pct']
            correct_count = int(summary['correct_count'])
            incorrect_count = int(summary['incorrect_count'])
            counts = np.asarray(summary['counts'], dtype=np.int32)

            if idx == 0:
                title = 'Stereo/Mono'
                if 'stereo' in ordered_sources and 'monocular' not in ordered_sources:
                    title = 'Stereo'
                elif 'monocular' in ordered_sources and 'stereo' not in ordered_sources:
                    title = 'Monocular'
            else:
                title = 'LiDAR'
            ax.set_title(title)

            finite_parts = []
            finite_pct_parts = []
            for source_name in ordered_sources:
                rmse = rmse_by_source.get(source_name)
                if rmse is None or not np.any(np.isfinite(rmse)):
                    continue
                ax.plot(
                    bin_centers,
                    rmse,
                    label=source_labels.get(source_name, source_name),
                    linewidth=2.0,
                    color=source_colors.get(source_name, None),
                )
                finite_parts.append(rmse[np.isfinite(rmse)])
                rmse_pct = rmse_pct_by_source.get(source_name)
                if rmse_pct is not None and np.any(np.isfinite(rmse_pct)):
                    ax_pct.plot(
                        bin_centers,
                        rmse_pct,
                        label=f"{source_labels.get(source_name, source_name)}_pct",
                        linewidth=1.8,
                        linestyle='--',
                        color=source_colors.get(source_name, None),
                        alpha=0.85,
                    )
                    finite_pct_parts.append(rmse_pct[np.isfinite(rmse_pct)])

            if ax.get_lines() or ax_pct.get_lines():
                ax.legend(handles=ax.get_lines() + ax_pct.get_lines(), loc='upper left', bbox_to_anchor=(0.0, 0.88))

            ymax = 1.0
            if finite_parts:
                finite_vals = np.concatenate(finite_parts)
                if finite_vals.size > 0:
                    ymax = max(0.1, float(np.max(finite_vals)) * 1.2)
            ax.set_ylim(0.0, ymax)
            pct_ymax = 1.0
            if finite_pct_parts:
                finite_pct_vals = np.concatenate(finite_pct_parts)
                if finite_pct_vals.size > 0:
                    pct_ymax = max(0.1, float(np.max(finite_pct_vals)) * 1.2)
            ax_pct.set_ylim(0.0, pct_ymax)
            ax.set_xlim(float(summary['range_min']), float(summary['range_max']))

            total_rmse_text = 'n/a' if total_rmse_pct is None else f'{float(total_rmse_pct):.2f}%'
            total_classified = correct_count + incorrect_count
            accuracy_text = f'{(100.0 * float(correct_count) / float(total_classified)):.1f}%' if total_classified > 0 else 'n/a'
            ax.text(
                0.98,
                0.98,
                f'rmse_total={total_rmse_text}\n'
                f'samples={int(np.sum(counts))}\n'
                f'accuracy={accuracy_text}',
                transform=ax.transAxes,
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

        fig.tight_layout()
        plots_path = self.session_path / 'plots'
        plots_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path if output_path is not None else (plots_path / 'cone_range_binned_rmse_camera_lidar.png')
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
