"""Thesis-oriented controller diagnostics plots."""

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


def _plot_if_finite(ax, x: np.ndarray, y: np.ndarray, *, label: str, color: str, style: str = '-') -> None:
    if y.size == 0:
        return
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return
    ax.plot(x[mask], y[mask], style, label=label, color=color, linewidth=1.5)


def _relative_time(rows: list[dict[str, float]]) -> np.ndarray:
    t = _series(rows, 'timestamp_sec')
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return np.arange(len(rows), dtype=np.float64)
    return t - float(finite[0])


def _legend_if_labeled(ax, **kwargs) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def _draw_thesis_figure(fig, rows: list[dict[str, float]], title: str) -> None:
    fig.clf()
    axes = fig.subplots(2, 2)
    t = _relative_time(rows)

    cte = _series(rows, 'cte_m')
    heading = _series(rows, 'heading_error_rad')
    desired = _series(rows, 'desired_steering_rad')
    final_cmd = _series(rows, 'final_steering_cmd_rad')
    actual = _series(rows, 'actual_steering_rad')
    heading_contrib = _series(rows, 'heading_contribution_rad')
    cross_track_contrib = _series(rows, 'cross_track_contribution_rad')
    yaw_damping = _series(rows, 'yaw_rate_damping_contribution_rad')
    speed = _series(rows, 'vehicle_speed_mps')
    curvature = _series(rows, 'path_curvature_abs_p95_1pm')
    jump = _series(rows, 'planner_centerline_jump_max_m')
    churn = _series(rows, 'planner_selected_edge_churn_ratio')
    hold_flag = _series(rows, 'plan_hold_active_flag')
    fallback_flag = _series(rows, 'plan_fallback_flag')

    ax = axes[0, 0]
    ax2 = ax.twinx()
    _plot_if_finite(ax, t, cte, label='CTE', color='#1f77b4')
    _plot_if_finite(ax2, t, heading, label='Heading error', color='#d62728')
    ax.set_title('Tracking Performance')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CTE (m)')
    ax2.set_ylabel('Heading error (rad)')
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize='small')

    ax = axes[0, 1]
    _plot_if_finite(ax, t, desired, label='Desired steering', color='#7f7f7f', style='--')
    _plot_if_finite(ax, t, final_cmd, label='Final steering cmd', color='#d62728')
    _plot_if_finite(ax, t, actual, label='Actual steering', color='#2ca02c')
    ax.set_title('Steering Tracking')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Steering (rad)')
    ax.grid(True, alpha=0.3)
    _legend_if_labeled(ax, loc='upper right', fontsize='small')

    ax = axes[1, 0]
    _plot_if_finite(ax, t, heading_contrib, label='Heading contribution', color='#9467bd')
    _plot_if_finite(ax, t, cross_track_contrib, label='Cross-track contribution', color='#ff7f0e')
    _plot_if_finite(ax, t, yaw_damping, label='Yaw-rate damping', color='#17becf')
    _plot_if_finite(ax, t, final_cmd, label='Final steering cmd', color='#d62728')
    ax.set_title('Controller Decomposition')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Radians')
    ax.grid(True, alpha=0.3)
    _legend_if_labeled(ax, loc='upper right', fontsize='small')

    ax = axes[1, 1]
    ax2 = ax.twinx()
    _plot_if_finite(ax, t, speed, label='Speed', color='#1f77b4')
    _plot_if_finite(ax, t, curvature, label='Path curvature p95', color='#2ca02c')
    _plot_if_finite(ax2, t, jump, label='Planner jump', color='#d62728')
    _plot_if_finite(ax2, t, churn, label='Planner churn', color='#ff7f0e')
    _plot_if_finite(ax2, t, hold_flag, label='Hold flag', color='#9467bd', style=':')
    _plot_if_finite(ax2, t, fallback_flag, label='Fallback flag', color='#8c564b', style='-.')
    ax.set_title('Operating Context')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed / curvature')
    ax2.set_ylabel('Planner health / flags')
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize='small')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()


def generate_thesis_controller_plot(
    csv_path: Path,
    output_path: Path,
    *,
    title: str = 'Thesis Controller Diagnostics',
    dpi: int = 150,
) -> Optional[Path]:
    rows = _read_rows(csv_path)
    if not rows:
        return None

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.0, 10.0))
    _draw_thesis_figure(fig, rows, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return output_path


class ThesisControllerDiagnosticsLivePlot:
    """Interactive live plot for thesis controller diagnostics."""

    def __init__(
        self,
        *,
        buffer_sec: float = 30.0,
        sample_rate_hz: float = 50.0,
        title: str = 'Thesis Controller Diagnostics (Live)',
    ) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._plt.ion()
        self._fig = self._plt.figure(figsize=(16.0, 10.0))
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
        _draw_thesis_figure(self._fig, list(self._rows), self._title)

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
