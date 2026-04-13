"""Offline headless cone plotter for range-binned cone RMSE metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import csv
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .matplotlib_fonts import (
    CONE_STATS_FONTSIZE,
    LEGEND_FONTSIZE,
    SUPTITLE_FONTSIZE,
    apply_axis_label_fontsize,
    apply_tick_label_fontsize,
)
from .range_rmse_analyzer import RangeRMSEAnalyzer, RangeRMSEBinStats


class OfflineConePlotter:
    """Generate static cone range-RMSE plots from recorded CSV data."""

    _SOURCE_SPECS = {
        'monocular': {
            'csv': 'cone_range_rmse_samples_mono.csv',
            'png': 'cone_range_rmse_mono.png',
            'title': 'Monocular Camera',
            'label': 'mono_rmse',
            'color': 'tab:blue',
            'show_classification': True,
        },
        'stereo': {
            'csv': 'cone_range_rmse_samples_stereo.csv',
            'png': 'cone_range_rmse_stereo.png',
            'title': 'Stereo Camera',
            'label': 'stereo_rmse',
            'color': 'tab:orange',
            'show_classification': True,
        },
        'lidar': {
            'csv': 'cone_range_rmse_samples_lidar.csv',
            'png': 'cone_range_rmse_lidar.png',
            'title': 'LiDAR',
            'label': 'lidar_rmse',
            'color': 'tab:green',
            'show_classification': False,
        },
    }

    def __init__(self, session_path: Path):
        self.session_path = Path(session_path)

    def generate_all_range_rmse_plots(self, dpi: int = 150) -> list[Path]:
        output_paths: list[Path] = []
        for source_name, spec in self._SOURCE_SPECS.items():
            stats = self._load_source_stats(spec['csv'], expected_source=source_name)
            output_paths.append(
                self._generate_source_plot(
                    source_name=source_name,
                    stats=stats,
                    title=str(spec['title']),
                    series_label=str(spec['label']),
                    color=str(spec['color']),
                    output_filename=str(spec['png']),
                    show_classification=bool(spec['show_classification']),
                    dpi=dpi,
                )
            )
        return output_paths

    def _load_source_stats(self, filename: str, *, expected_source: str) -> RangeRMSEBinStats:
        analyzer = RangeRMSEAnalyzer(range_min_m=0.0, range_max_m=20.0, bin_width_m=1.0)
        csv_path = self.session_path / 'logs' / filename
        if not csv_path.exists():
            return analyzer.compute_binned_rmse()

        with open(csv_path, 'r', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source = self._normalize_source_name(row.get('source', '')) or expected_source
                if source != expected_source:
                    continue
                gt_range_m = self._parse_optional_float(row.get('gt_range_m', ''))
                error_m = self._parse_optional_float(row.get('error_m', ''))
                if not (math.isfinite(gt_range_m) and math.isfinite(error_m)):
                    continue
                analyzer.add_sample(
                    source=source,
                    gt_range_m=gt_range_m,
                    error_m=error_m,
                    predicted_class_id=self._parse_optional_int(row.get('predicted_class_id', '')),
                    ground_truth_class_id=self._parse_optional_int(row.get('ground_truth_class_id', '')),
                )
        return analyzer.compute_binned_rmse()

    @staticmethod
    def _normalize_source_name(value: object) -> str:
        source = '' if value is None else str(value).strip().lower()
        if source in {'mono', 'monocular'}:
            return 'monocular'
        if source in {'stereo', 'camera'}:
            return 'stereo'
        if source == 'lidar':
            return 'lidar'
        return ''

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
    def _parse_optional_int(value: object) -> Optional[int]:
        parsed = OfflineConePlotter._parse_optional_float(value)
        if not math.isfinite(parsed):
            return None
        return int(parsed)

    def _generate_source_plot(
        self,
        *,
        source_name: str,
        stats: RangeRMSEBinStats,
        title: str,
        series_label: str,
        color: str,
        output_filename: str,
        show_classification: bool,
        dpi: int,
    ) -> Path:
        bin_centers = np.asarray(stats.bin_centers, dtype=np.float32)
        counts = np.asarray(stats.total_counts, dtype=np.int32)
        rmse = np.asarray(
            stats.source_rmse.get(source_name, np.full_like(bin_centers, np.nan)),
            dtype=np.float32,
        )
        rmse_pct = np.asarray(
            self._compute_source_rmse_percent(bin_centers, {source_name: rmse}).get(
                source_name,
                np.full_like(bin_centers, np.nan),
            ),
            dtype=np.float32,
        )
        total_rmse_pct = self._compute_total_rmse_percent(bin_centers, {source_name: rmse})
        has_data = bool(np.any(counts > 0) and np.any(np.isfinite(rmse)))

        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10.0, 7.0),
            gridspec_kw={'height_ratios': [3, 1]},
        )
        fig.suptitle(f'{title} Cone Range-Binned RMSE', fontsize=SUPTITLE_FONTSIZE)
        ax_top_pct = ax_top.twinx()

        if has_data:
            ax_top.plot(
                bin_centers,
                rmse,
                label=series_label,
                linewidth=2.0,
                color=color,
            )
            if np.any(np.isfinite(rmse_pct)):
                ax_top_pct.plot(
                    bin_centers,
                    rmse_pct,
                    label=f'{series_label}_pct',
                    linewidth=1.8,
                    linestyle='--',
                    color=color,
                    alpha=0.85,
                )
            handles = ax_top.get_lines() + ax_top_pct.get_lines()
            if handles:
                ax_top.legend(
                    handles=handles,
                    loc='upper left',
                    bbox_to_anchor=(0.0, 0.88),
                    fontsize=LEGEND_FONTSIZE,
                )
        else:
            ax_top.text(
                0.5,
                0.5,
                'no data',
                transform=ax_top.transAxes,
                ha='center',
                va='center',
                fontsize=12,
                bbox={
                    'boxstyle': 'round,pad=0.3',
                    'facecolor': 'white',
                    'edgecolor': '0.6',
                    'alpha': 0.85,
                },
            )

        ax_top.set_xlim(float(bin_centers[0] - 0.5), float(bin_centers[-1] + 0.5))
        ax_top.set_ylabel('RMSE (m)')
        ax_top_pct.set_ylabel('RMSE (%)')
        apply_axis_label_fontsize(ax_top)
        apply_axis_label_fontsize(ax_top_pct)
        apply_tick_label_fontsize(ax_top)
        apply_tick_label_fontsize(ax_top_pct)
        ax_top.grid(True, alpha=0.3)

        finite_rmse = rmse[np.isfinite(rmse)]
        finite_rmse_pct = rmse_pct[np.isfinite(rmse_pct)]
        ax_top.set_ylim(0.0, max(0.1, float(np.max(finite_rmse)) * 1.2) if finite_rmse.size > 0 else 1.0)
        ax_top_pct.set_ylim(
            0.0,
            max(0.1, float(np.max(finite_rmse_pct)) * 1.2) if finite_rmse_pct.size > 0 else 1.0,
        )

        if show_classification:
            total_rmse_text = 'n/a' if total_rmse_pct is None else f'{float(total_rmse_pct):.2f}%'
            sample_count = int(np.sum(counts))
            correct_count = int(stats.correct_class_count)
            incorrect_count = int(stats.incorrect_class_count)
            total_classified = correct_count + incorrect_count
            accuracy_text = (
                f'{(100.0 * float(correct_count) / float(total_classified)):.1f}%'
                if total_classified > 0 else 'n/a'
            )
            top_lines = [
                f'rmse_total={total_rmse_text}',
                f'samples={sample_count}',
                f'accuracy={accuracy_text}',
            ]
            ax_top.text(
                0.98,
                0.98,
                '\n'.join(top_lines),
                transform=ax_top.transAxes,
                ha='right',
                va='top',
                fontsize=CONE_STATS_FONTSIZE,
                bbox={
                    'boxstyle': 'round,pad=0.3',
                    'facecolor': 'white',
                    'edgecolor': '0.6',
                    'alpha': 0.85,
                },
            )

        ax_bottom.bar(
            bin_centers,
            counts.astype(np.float32),
            width=0.9,
            align='center',
            color='tab:gray',
        )
        ax_bottom.set_xlim(float(bin_centers[0] - 0.5), float(bin_centers[-1] + 0.5))
        ax_bottom.set_xlabel('Ground-truth range (m)')
        ax_bottom.set_ylabel('Samples')
        apply_axis_label_fontsize(ax_bottom)
        apply_tick_label_fontsize(ax_bottom)
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.set_ylim(0.0, max(1.0, float(np.max(counts)) * 1.2) if counts.size > 0 else 1.0)

        fig.tight_layout()
        plots_path = self.session_path / 'plots'
        plots_path.mkdir(parents=True, exist_ok=True)
        final_path = plots_path / output_filename
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
