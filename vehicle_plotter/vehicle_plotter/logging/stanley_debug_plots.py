"""Stanley steering debug plots (live + offline)."""

from __future__ import annotations

from collections import deque
import csv
import math
from pathlib import Path
from typing import Deque, Optional

import numpy as np


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out if math.isfinite(out) else float('nan')


def _read_rows(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, float]] = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                row[str(key)] = _safe_float(value)
            rows.append(row)
    return rows


def _series(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([_safe_float(row.get(key, float('nan'))) for row in rows], dtype=np.float64)


_PLOT_STATE_KEYS = (
    'raw_steering_cmd_rad',
    'final_steering_cmd_rad',
    'actual_steering_rad',
    'heading_error_rad',
    'heading_contribution_rad',
    'cte_m',
    'cross_track_contribution_rad',
    'vehicle_speed_mps',
    'speed_term_mps',
    'vehicle_x_m',
    'vehicle_y_m',
    'nearest_path_point_x_m',
    'nearest_path_point_y_m',
    'target_point_x_frame_m',
    'target_point_y_frame_m',
)


def _relative_time(rows: list[dict[str, float]]) -> np.ndarray:
    t = _series(rows, 'timestamp_sec')
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return np.arange(len(rows), dtype=np.float64)
    return t - float(finite[0])


def _auto_zero_limits(*signals: np.ndarray) -> tuple[float, float]:
    finite_parts = [sig[np.isfinite(sig)] for sig in signals if sig.size > 0]
    finite_parts = [part for part in finite_parts if part.size > 0]
    if not finite_parts:
        return -0.01, 0.01
    merged = np.concatenate(finite_parts)
    max_abs = float(np.max(np.abs(merged)))
    span = max(5e-4, 1.25 * max_abs)
    return -span, span


def _plot_if_finite(ax, x: np.ndarray, y: np.ndarray, *, label: str, color: str, style: str = '-') -> None:
    if y.size == 0:
        return
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return
    ax.plot(x[mask], y[mask], style, label=label, color=color, linewidth=1.5)


def _legend_if_labeled(ax, **kwargs) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def _rows_for_plotting(rows: list[dict[str, float]], *, collapse_holds: bool) -> list[dict[str, float]]:
    if not collapse_holds or len(rows) <= 2:
        return rows

    filtered: list[dict[str, float]] = [rows[0]]
    last_kept = rows[0]

    def _same_value(a: float, b: float) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        if not math.isfinite(a) or not math.isfinite(b):
            return False
        return abs(a - b) <= 1e-9

    for idx in range(1, len(rows) - 1):
        row = rows[idx]
        next_row = rows[idx + 1]
        same_as_last = all(
            _same_value(_safe_float(row.get(key, float('nan'))), _safe_float(last_kept.get(key, float('nan'))))
            for key in _PLOT_STATE_KEYS
        )
        same_as_next = all(
            _same_value(_safe_float(row.get(key, float('nan'))), _safe_float(next_row.get(key, float('nan'))))
            for key in _PLOT_STATE_KEYS
        )
        if same_as_last and same_as_next:
            continue
        filtered.append(row)
        last_kept = row

    filtered.append(rows[-1])
    return filtered


def _draw_stanley_debug_figure(
    fig,
    rows: list[dict[str, float]],
    title: str,
    *,
    collapse_holds: bool = False,
) -> None:
    fig.clf()
    rows = _rows_for_plotting(rows, collapse_holds=collapse_holds)
    axes = fig.subplots(3, 2)
    t = _relative_time(rows)

    raw_steer = _series(rows, 'raw_steering_cmd_rad')
    final_steer = _series(rows, 'final_steering_cmd_rad')
    actual_steer = _series(rows, 'actual_steering_rad')
    heading_err = _series(rows, 'heading_error_rad')
    heading_contrib = _series(rows, 'heading_contribution_rad')
    cte = _series(rows, 'cte_m')
    cross_contrib = _series(rows, 'cross_track_contribution_rad')
    speed = _series(rows, 'vehicle_speed_mps')
    speed_term = _series(rows, 'speed_term_mps')

    # Row 1, Col 1: Steering overview
    ax = axes[0, 0]
    _plot_if_finite(ax, t, raw_steer, label='Raw Stanley cmd', color='#1f77b4')
    _plot_if_finite(ax, t, final_steer, label='Final steering cmd', color='#d62728')
    _plot_if_finite(ax, t, actual_steer, label='Actual steering', color='#2ca02c')
    ax.set_title('Steering Command Overview')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Steering (rad)')
    ax.grid(True, alpha=0.3)
    _legend_if_labeled(ax, loc='upper right', fontsize='small')

    # Row 1, Col 2: Near-zero zoom
    ax = axes[0, 1]
    _plot_if_finite(ax, t, raw_steer, label='Raw Stanley cmd', color='#1f77b4')
    _plot_if_finite(ax, t, final_steer, label='Final steering cmd', color='#d62728')
    y_min, y_max = _auto_zero_limits(raw_steer, final_steer)
    ax.set_ylim(y_min, y_max)
    ax.set_title('Steering Zoom Near Zero')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Steering (rad)')
    ax.grid(True, alpha=0.3)
    _legend_if_labeled(ax, loc='upper right', fontsize='small')

    # Row 2, Col 1: Heading error + contribution
    ax = axes[1, 0]
    _plot_if_finite(ax, t, heading_err, label='Heading error', color='#9467bd')
    _plot_if_finite(ax, t, heading_contrib, label='Heading contribution', color='#ff7f0e')
    ax.set_title('Heading Error and Contribution')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Radians')
    ax.grid(True, alpha=0.3)
    _legend_if_labeled(ax, loc='upper right', fontsize='small')

    # Row 2, Col 2: Cross-track error + contribution
    ax = axes[1, 1]
    ax2 = ax.twinx()
    _plot_if_finite(ax, t, cte, label='Cross-track error', color='#2ca02c')
    _plot_if_finite(ax2, t, cross_contrib, label='Cross-track contribution', color='#d62728')
    ax.set_title('Cross-Track Error and Contribution')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CTE (m)')
    ax2.set_ylabel('Steering contribution (rad)')
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize='small')

    # Row 3, Col 1: Speed + softened denominator (+ optional cross-track contribution)
    ax = axes[2, 0]
    ax2 = ax.twinx()
    _plot_if_finite(ax, t, speed, label='Vehicle speed', color='#1f77b4')
    _plot_if_finite(ax, t, speed_term, label='Softened speed term', color='#17becf')
    _plot_if_finite(ax2, t, cross_contrib, label='Cross-track contribution', color='#d62728')
    ax.set_title('Speed and Stanley Speed Term')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax2.set_ylabel('Contribution (rad)')
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize='small')

    # Row 3, Col 2: XY tracking
    ax = axes[2, 1]
    vehicle_x = _series(rows, 'vehicle_x_m')
    vehicle_y = _series(rows, 'vehicle_y_m')
    nearest_x = _series(rows, 'nearest_path_point_x_m')
    nearest_y = _series(rows, 'nearest_path_point_y_m')
    target_x = _series(rows, 'target_point_x_frame_m')
    target_y = _series(rows, 'target_point_y_frame_m')
    _plot_if_finite(ax, nearest_x, nearest_y, label='Reference (nearest path)', color='#7f7f7f', style='--')
    _plot_if_finite(ax, vehicle_x, vehicle_y, label='Vehicle path', color='#1f77b4')
    _plot_if_finite(ax, target_x, target_y, label='Target point trace', color='#ff7f0e', style=':')
    if np.any(np.isfinite(vehicle_x) & np.isfinite(vehicle_y)):
        idx = np.where(np.isfinite(vehicle_x) & np.isfinite(vehicle_y))[0][-1]
        ax.scatter(vehicle_x[idx], vehicle_y[idx], color='#1f77b4', s=45, marker='o', label='Current vehicle')
    if np.any(np.isfinite(target_x) & np.isfinite(target_y)):
        idx_t = np.where(np.isfinite(target_x) & np.isfinite(target_y))[0][-1]
        ax.scatter(target_x[idx_t], target_y[idx_t], color='#ff7f0e', s=45, marker='x', label='Current target')
        if np.any(np.isfinite(vehicle_x) & np.isfinite(vehicle_y)):
            idx_v = np.where(np.isfinite(vehicle_x) & np.isfinite(vehicle_y))[0][-1]
            ax.plot(
                [vehicle_x[idx_v], target_x[idx_t]],
                [vehicle_y[idx_v], target_y[idx_t]],
                color='#ff7f0e',
                linewidth=1.2,
                alpha=0.8,
            )
    ax.set_title('XY Path Tracking View')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    _legend_if_labeled(ax, loc='best', fontsize='small')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()


def generate_stanley_debug_plot(
    csv_path: Path,
    output_path: Path,
    *,
    title: str = 'Stanley Oscillation Debug',
    dpi: int = 150,
) -> Optional[Path]:
    """Render a 3x2 Stanley debug plot from the steering diagnostics CSV."""
    rows = _read_rows(csv_path)
    if not rows:
        return None

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.0, 11.0))
    _draw_stanley_debug_figure(fig, rows, title=title, collapse_holds=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return output_path


class StanleyDebugLivePlot:
    """Interactive live plot for Stanley debug rows."""

    def __init__(
        self,
        *,
        buffer_sec: float = 30.0,
        sample_rate_hz: float = 50.0,
        title: str = 'Stanley Oscillation Debug (Live)',
    ) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._plt.ion()
        self._fig = self._plt.figure(figsize=(16.0, 11.0))
        manager = getattr(self._fig.canvas, 'manager', None)
        if manager is not None and hasattr(manager, 'set_window_title'):
            manager.set_window_title(title)
        self._configure_window_focus_behavior()
        max_rows = max(200, int(max(1.0, float(buffer_sec)) * max(1.0, float(sample_rate_hz))))
        self._rows: Deque[dict[str, float]] = deque(maxlen=max_rows)
        self._title = title
        self._draw()

    def update(self, row: dict[str, float]) -> bool:
        if not self._is_open():
            return False
        self._rows.append(dict(row))
        self._draw()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        return self._is_open()

    def close(self) -> None:
        if self._is_open():
            self._plt.close(self._fig)

    def _draw(self) -> None:
        _draw_stanley_debug_figure(self._fig, list(self._rows), title=self._title)

    def _is_open(self) -> bool:
        return bool(self._plt.fignum_exists(self._fig.number))

    def _configure_window_focus_behavior(self) -> None:
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
