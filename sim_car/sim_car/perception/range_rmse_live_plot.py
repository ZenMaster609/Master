"""Live matplotlib plotter for range-binned cone RMSE metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .range_rmse_analyzer import RangeRMSEAnalyzer, RangeRMSEBinStats


class RangeRMSELivePlot:
    """Render live combined-RMSE curves and per-bin sample counts."""

    _SOURCE_COLORS = {
        'monocular': 'tab:blue',
        'stereo': 'tab:orange',
        'lidar': 'tab:green',
    }
    _SOURCE_LABELS = {
        'monocular': 'mono_rmse',
        'stereo': 'stereo_rmse',
        'lidar': 'lidar_rmse',
    }

    def __init__(
        self,
        range_min_m: float = 0.0,
        range_max_m: float = 20.0,
        bin_width_m: float = 1.0,
        title: str = 'Cone Range-Binned RMSE',
    ):
        import matplotlib.pyplot as plt

        self._plt = plt
        self._range_min_m = float(range_min_m)
        self._range_max_m = float(range_max_m)
        self._bin_width_m = float(bin_width_m)

        num_bins = max(1, int(round((self._range_max_m - self._range_min_m) / self._bin_width_m)))
        self._bin_centers = self._range_min_m + (np.arange(num_bins, dtype=np.float32) + 0.5) * self._bin_width_m

        self._plt.ion()
        self._fig, (self._ax_top, self._ax_bottom) = self._plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10.0, 7.0),
            gridspec_kw={'height_ratios': [3, 1]},
        )
        self._fig.canvas.manager.set_window_title(title)
        self._configure_window_focus_behavior()
        self._ax_top_pct = self._ax_top.twinx()

        nan_series = np.full(num_bins, np.nan, dtype=np.float32)
        self._source_lines = {}
        self._source_pct_lines = {}
        for source in RangeRMSEAnalyzer.SOURCE_ORDER:
            line, = self._ax_top.plot(
                self._bin_centers,
                nan_series,
                label=self._SOURCE_LABELS.get(source, source),
                linewidth=2.0,
                color=self._SOURCE_COLORS.get(source, None),
            )
            line.set_visible(False)
            self._source_lines[source] = line
            pct_line, = self._ax_top_pct.plot(
                self._bin_centers,
                nan_series,
                label=f"{self._SOURCE_LABELS.get(source, source)}_pct",
                linewidth=1.8,
                linestyle='--',
                color=self._SOURCE_COLORS.get(source, None),
                alpha=0.85,
            )
            pct_line.set_visible(False)
            self._source_pct_lines[source] = pct_line

        self._ax_top.set_xlim(self._range_min_m, self._range_max_m)
        self._ax_top.set_ylabel('RMSE (m)')
        self._ax_top.grid(True)
        self._ax_top_pct.set_ylabel('RMSE (%)')
        self._rmse_total_text = self._ax_top.text(
            0.02,
            0.98,
            'rmse_total=n/a',
            transform=self._ax_top.transAxes,
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
        self._class_header_text = self._ax_top.text(
            0.98,
            0.98,
            'Cone classification',
            transform=self._ax_top.transAxes,
            ha='right',
            va='top',
            fontweight='semibold',
            fontsize=10,
        )
        self._class_body_text = self._ax_top.text(
            0.98,
            0.90,
            'Correct: 0\nIncorrect: 0\nAccuracy: n/a',
            transform=self._ax_top.transAxes,
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

        zeros = np.zeros(num_bins, dtype=np.float32)
        self._bars = self._ax_bottom.bar(
            self._bin_centers,
            zeros,
            width=self._bin_width_m * 0.9,
            align='center',
            color='tab:gray',
        )
        self._ax_bottom.set_xlim(self._range_min_m, self._range_max_m)
        self._ax_bottom.set_xlabel('Ground-truth range (m)')
        self._ax_bottom.set_ylabel('Samples')
        self._ax_bottom.grid(True)

        self._update_legend()
        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def update(self, stats: RangeRMSEBinStats) -> bool:
        """Update plot from latest binned stats. Returns False if figure is closed."""
        if not self._is_open():
            return False

        centers = np.asarray(stats.bin_centers, dtype=np.float32)
        counts = np.asarray(stats.total_counts, dtype=np.float32)
        if centers.shape != self._bin_centers.shape:
            return True

        finite_series = []
        finite_pct_series = []
        visible_sources = []
        source_rmse_pct = self._compute_source_rmse_percent(centers, stats.source_rmse)
        for source, line in self._source_lines.items():
            rmse = np.asarray(stats.source_rmse.get(source, np.full_like(self._bin_centers, np.nan)), dtype=np.float32)
            rmse_pct = np.asarray(source_rmse_pct.get(source, np.full_like(self._bin_centers, np.nan)), dtype=np.float32)
            line.set_xdata(centers)
            line.set_ydata(rmse)
            pct_line = self._source_pct_lines[source]
            pct_line.set_xdata(centers)
            pct_line.set_ydata(rmse_pct)
            has_data = bool(np.any(np.isfinite(rmse)))
            line.set_visible(has_data)
            pct_line.set_visible(bool(np.any(np.isfinite(rmse_pct))))
            if has_data:
                visible_sources.append(source)
                finite_series.append(rmse[np.isfinite(rmse)])
            if np.any(np.isfinite(rmse_pct)):
                finite_pct_series.append(rmse_pct[np.isfinite(rmse_pct)])

        self._update_legend(visible_sources)

        if finite_series:
            finite_vals = np.concatenate(finite_series)
            ymax = max(0.1, float(np.max(finite_vals)) * 1.2)
            self._ax_top.set_ylim(0.0, ymax)
        else:
            self._ax_top.set_ylim(0.0, 1.0)
        if finite_pct_series:
            finite_pct_vals = np.concatenate(finite_pct_series)
            pct_ymax = max(0.1, float(np.max(finite_pct_vals)) * 1.2)
            self._ax_top_pct.set_ylim(0.0, pct_ymax)
        else:
            self._ax_top_pct.set_ylim(0.0, 1.0)

        total_rmse_pct = self._compute_total_rmse_percent(stats)
        if total_rmse_pct is None:
            total_rmse_text = 'n/a'
        else:
            total_rmse_text = f'{total_rmse_pct:.2f}%'
        self._rmse_total_text.set_text(f'rmse_total={total_rmse_text}')

        max_count = 0.0
        for bar, count in zip(self._bars, counts):
            height = float(count)
            bar.set_height(height)
            if height > max_count:
                max_count = height
        self._ax_bottom.set_ylim(0.0, max(1.0, max_count * 1.2))

        correct_count = int(stats.correct_class_count)
        incorrect_count = int(stats.incorrect_class_count)
        total_count = correct_count + incorrect_count
        if total_count > 0:
            accuracy_text = f'{(100.0 * float(correct_count) / float(total_count)):.1f}%'
        else:
            accuracy_text = 'n/a'
        self._class_body_text.set_text(
            f'Correct: {correct_count}\n'
            f'Incorrect: {incorrect_count}\n'
            f'Accuracy: {accuracy_text}'
        )

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        return self._is_open()

    def close(self) -> None:
        if self._is_open():
            self._plt.close(self._fig)

    def _is_open(self) -> bool:
        if self._fig is None:
            return False
        return bool(self._plt.fignum_exists(self._fig.number))

    def _configure_window_focus_behavior(self) -> None:
        """Best-effort: avoid stealing keyboard focus while still showing updates."""
        manager = getattr(self._fig.canvas, 'manager', None)
        window = getattr(manager, 'window', None)
        if window is None:
            return
        try:
            from PyQt5 import QtCore

            window.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
            window.setFocusPolicy(QtCore.Qt.NoFocus)
        except Exception:
            return

    def _update_legend(self, visible_sources: list[str] | None = None) -> None:
        if visible_sources is None:
            visible_sources = []
        legend = self._ax_top.get_legend()
        if legend is not None:
            legend.remove()
        handles = []
        for source in RangeRMSEAnalyzer.SOURCE_ORDER:
            if source not in visible_sources:
                continue
            handles.append(self._source_lines[source])
            handles.append(self._source_pct_lines[source])
        if handles:
            self._ax_top.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 0.88))

    @staticmethod
    def _compute_total_rmse_percent(stats: RangeRMSEBinStats) -> Optional[float]:
        percent_values = [
            arr[np.isfinite(arr)]
            for arr in RangeRMSELivePlot._compute_source_rmse_percent(stats.bin_centers, stats.source_rmse).values()
            if np.any(np.isfinite(arr))
        ]

        if not percent_values:
            return None

        return float(np.mean(np.concatenate(percent_values)))

    @staticmethod
    def _compute_source_rmse_percent(
        bin_centers: np.ndarray,
        source_rmse: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        centers = np.asarray(bin_centers, dtype=np.float32)
        if centers.size == 0:
            return {}

        source_rmse_pct: dict[str, np.ndarray] = {}
        for source, rmse in source_rmse.items():
            rmse_arr = np.asarray(rmse, dtype=np.float32)
            if rmse_arr.shape != centers.shape:
                continue
            rmse_pct = np.full_like(rmse_arr, np.nan, dtype=np.float32)
            valid = np.isfinite(rmse_arr) & np.isfinite(centers) & (centers > 0.0)
            if np.any(valid):
                rmse_pct[valid] = (rmse_arr[valid] / centers[valid]) * 100.0
            source_rmse_pct[source] = rmse_pct
        return source_rmse_pct
