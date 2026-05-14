from __future__ import annotations

import math
import pathlib
import sys

import pandas as pd
import pytest


TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from vehicle_plotter.plotting import offline_plotter as offline_plotter_module
from vehicle_plotter.plotting.offline_plotter import (
    MILLIMETERS_PER_SECOND_TO_METERS_PER_SECOND_SCALE,
    MOVEMENT_VELOCITY_PLOT_INDEX,
    MOVEMENT_WHEEL_SPEED_PLOT_INDEX,
    OfflinePlotter,
    WHEEL_SPEED_MAX_MPS,
    WHEEL_SPEED_MIN_MPS,
)
from vehicle_plotter.plotting.plot_definitions import get_all_plot_definitions
from vehicle_plotter.plotting.plot_config import VELOCITY_AXIS_LIMITS_MPS, get_all_plots, get_default_plots


SESSION_ROW_COUNT = 1205  # Exceeds the live dashboard buffer that previously clipped export history.
SAMPLE_PERIOD_SEC = 0.1  # Keeps test timestamps simple while preserving time-series ordering.
FORWARD_SPEED_MPS = 3.0  # Forms a 3-4-5 speed triangle for scalar speed assertions.
LATERAL_SPEED_MPS = 4.0  # Forms a 3-4-5 speed triangle for scalar speed assertions.
STALE_LOGGED_SPEED_MPS = 999.0  # Proves export rows recompute speed instead of trusting stale logs.
EXPECTED_SCALAR_SPEED_MPS = 5.0  # Hypotenuse of the configured velocity components.
EXPECTED_VELOCITY_SERIES_COUNT = 2  # Movement exports should show Vx and Vy only.


def test_movement_layout_uses_velocity_components_with_export_limits(tmp_path):
    plotter = OfflinePlotter(tmp_path)

    layout = plotter._build_movement_layout()

    velocity_plot = layout.plots[MOVEMENT_VELOCITY_PLOT_INDEX]
    assert velocity_plot.name == "Velocity"
    assert len(velocity_plot.series) == EXPECTED_VELOCITY_SERIES_COUNT
    assert [series.variable for series in velocity_plot.series] == ["vx", "vy"]
    assert velocity_plot.y_axis.label == "Velocity"
    assert velocity_plot.y_axis.unit == "m/s"
    assert velocity_plot.y_axis.limits == VELOCITY_AXIS_LIMITS_MPS
    assert velocity_plot.y_axis.auto_scale is False

    wheel_speed_plot = layout.plots[MOVEMENT_WHEEL_SPEED_PLOT_INDEX]
    assert wheel_speed_plot.y_axis.limits == (WHEEL_SPEED_MIN_MPS, WHEEL_SPEED_MAX_MPS)
    assert all(
        series.scale == MILLIMETERS_PER_SECOND_TO_METERS_PER_SECOND_SCALE
        for series in wheel_speed_plot.series
    )


def test_live_dashboard_velocity_axis_is_permanently_clamped():
    for layout in (get_all_plots(), get_default_plots()):
        velocity_plot = next(plot for plot in layout.plots if plot.name == "Velocity")

        assert velocity_plot.y_axis.limits == VELOCITY_AXIS_LIMITS_MPS
        assert velocity_plot.y_axis.auto_scale is False


def test_standalone_velocity_plot_is_permanently_clamped():
    velocity_plot = next(plot for plot in get_all_plot_definitions() if plot.name == "Velocity")

    assert [series[0] for series in velocity_plot.series] == ["vx", "vy"]
    assert velocity_plot.required_columns == ["vx", "vy"]
    assert velocity_plot.y_limits == VELOCITY_AXIS_LIMITS_MPS


def test_dashboard_export_expands_buffers_to_include_full_session(tmp_path, monkeypatch):
    captured = {}
    plotter = OfflinePlotter(tmp_path)
    plotter.data = pd.DataFrame(
        {
            "timestamp": [index * SAMPLE_PERIOD_SEC for index in range(SESSION_ROW_COUNT)],
            "vx": [FORWARD_SPEED_MPS] * SESSION_ROW_COUNT,
            "vy": [LATERAL_SPEED_MPS] * SESSION_ROW_COUNT,
        }
    )

    class CapturingPlotManager:
        def __init__(self, *, layout_config, enable_gui):
            captured["buffer_sizes"] = [plot.buffer_size for plot in layout_config.plots]
            captured["enable_gui"] = enable_gui
            self.pushed_states = []

        def push_state(self, state):
            self.pushed_states.append(state)

        def export_static_dashboard(self, path, *, dpi):
            captured["export_path"] = path
            captured["dpi"] = dpi
            captured["pushed_count"] = len(self.pushed_states)

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(offline_plotter_module, "PlotManager", CapturingPlotManager)

    plotter._export_dashboard(tmp_path / "movement.pdf", layout_config=plotter._build_movement_layout())

    assert all(buffer_size >= SESSION_ROW_COUNT for buffer_size in captured["buffer_sizes"])
    assert captured["pushed_count"] == SESSION_ROW_COUNT
    assert captured["closed"] is True


def test_exported_vehicle_state_recomputes_scalar_speed_from_components(tmp_path):
    plotter = OfflinePlotter(tmp_path)

    state = plotter._vehicle_state_from_row(
        {
            "vx": FORWARD_SPEED_MPS,
            "vy": LATERAL_SPEED_MPS,
            "speed": STALE_LOGGED_SPEED_MPS,
        }
    )

    assert state.speed == pytest.approx(math.hypot(FORWARD_SPEED_MPS, LATERAL_SPEED_MPS))
    assert state.speed == pytest.approx(EXPECTED_SCALAR_SPEED_MPS)
