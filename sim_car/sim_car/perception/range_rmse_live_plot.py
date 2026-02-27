"""Live matplotlib plotter for range-binned cone RMSE metrics."""

from __future__ import annotations

import numpy as np

from .range_rmse_analyzer import RangeRMSEBinStats


class RangeRMSELivePlot:
    """Render live RMSE_x/RMSE_y curves and per-bin sample counts."""

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

        nan_series = np.full(num_bins, np.nan, dtype=np.float32)
        self._line_rmse_x, = self._ax_top.plot(self._bin_centers, nan_series, label='RMSE_x', linewidth=2.0)
        self._line_rmse_y, = self._ax_top.plot(self._bin_centers, nan_series, label='RMSE_y', linewidth=2.0)

        self._ax_top.set_xlim(self._range_min_m, self._range_max_m)
        self._ax_top.set_ylabel('RMSE (m)')
        self._ax_top.grid(True)
        self._ax_top.legend(['RMSE_x', 'RMSE_y'])

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

        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def update(self, stats: RangeRMSEBinStats) -> bool:
        """Update plot from latest binned stats. Returns False if figure is closed."""
        if not self._is_open():
            return False

        centers = np.asarray(stats.bin_centers, dtype=np.float32)
        rmse_x = np.asarray(stats.rmse_x, dtype=np.float32)
        rmse_y = np.asarray(stats.rmse_y, dtype=np.float32)
        counts = np.asarray(stats.counts, dtype=np.float32)

        if centers.shape != self._bin_centers.shape:
            return True

        self._line_rmse_x.set_xdata(centers)
        self._line_rmse_x.set_ydata(rmse_x)
        self._line_rmse_y.set_xdata(centers)
        self._line_rmse_y.set_ydata(rmse_y)

        finite_vals = np.concatenate((rmse_x[np.isfinite(rmse_x)], rmse_y[np.isfinite(rmse_y)]))
        if finite_vals.size > 0:
            ymax = max(0.1, float(np.max(finite_vals)) * 1.2)
            self._ax_top.set_ylim(0.0, ymax)
        else:
            self._ax_top.set_ylim(0.0, 1.0)

        max_count = 0.0
        for bar, count in zip(self._bars, counts):
            height = float(count)
            bar.set_height(height)
            if height > max_count:
                max_count = height
        self._ax_bottom.set_ylim(0.0, max(1.0, max_count * 1.2))

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
