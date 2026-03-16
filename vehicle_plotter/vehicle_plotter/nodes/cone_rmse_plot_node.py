#!/usr/bin/env python3
"""Combined live cone RMSE window for camera and LiDAR evaluator streams."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from ..plotting.range_rmse_analyzer import RangeRMSEAnalyzer, RangeRMSEBinStats

# This node does not use 3D plots; suppress non-fatal mixed-site-packages warning.
warnings.filterwarnings(
    "ignore",
    message=r"Unable to import Axes3D\..*",
    category=UserWarning,
    module=r"matplotlib\.projections",
)
warnings.filterwarnings(
    "ignore",
    message=r"The value of the smallest subnormal for <class '.*'?> type is zero\.",
    category=UserWarning,
    module=r"numpy\.core\.getlimits",
)


@dataclass
class _PanelArtists:
    ax_top: object
    ax_top_pct: object
    ax_bottom: object
    source_lines: Dict[str, object]
    source_pct_lines: Dict[str, object]
    bars: object
    rmse_total_text: object
    class_header_text: object
    class_body_text: object


class _SideBySideRMSEFigure:
    _SOURCE_COLORS = {
        "monocular": "tab:blue",
        "stereo": "tab:orange",
        "lidar": "tab:green",
    }
    _SOURCE_LABELS = {
        "monocular": "mono_rmse",
        "stereo": "stereo_rmse",
        "lidar": "lidar_rmse",
    }

    def __init__(
        self,
        *,
        range_min_m: float = 0.0,
        range_max_m: float = 20.0,
        bin_width_m: float = 1.0,
        left_title: str = "Camera",
        right_title: str = "LiDAR",
    ):
        import matplotlib.pyplot as plt

        self._plt = plt
        self._range_min_m = float(range_min_m)
        self._range_max_m = float(range_max_m)
        self._bin_width_m = float(bin_width_m)
        num_bins = max(
            1, int(round((self._range_max_m - self._range_min_m) / self._bin_width_m))
        )
        self._bin_centers = self._range_min_m + (
            (np.arange(num_bins, dtype=np.float32) + 0.5) * self._bin_width_m
        )

        self._plt.ion()
        self._fig = self._plt.figure(figsize=(18.0, 7.0))
        self._fig.canvas.manager.set_window_title("Cone RMSE: Camera + LiDAR")
        self._configure_window_focus_behavior()
        grid = self._fig.add_gridspec(
            2,
            2,
            height_ratios=[3.0, 1.0],
            width_ratios=[1.0, 1.0],
            hspace=0.15,
            wspace=0.25,
        )
        self._left_panel = self._create_panel(grid, col=0, title=left_title)
        self._right_panel = self._create_panel(grid, col=1, title=right_title)
        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def update(self, left_stats: RangeRMSEBinStats, right_stats: RangeRMSEBinStats) -> bool:
        if not self._is_open():
            return False
        self._update_panel(self._left_panel, left_stats)
        self._update_panel(self._right_panel, right_stats)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        return self._is_open()

    def close(self) -> None:
        if self._is_open():
            self._plt.close(self._fig)

    def _create_panel(self, grid, *, col: int, title: str) -> _PanelArtists:
        ax_top = self._fig.add_subplot(grid[0, col])
        ax_bottom = self._fig.add_subplot(grid[1, col], sharex=ax_top)
        ax_top_pct = ax_top.twinx()

        ax_top.set_title(title)
        ax_top.set_xlim(self._range_min_m, self._range_max_m)
        ax_top.set_ylabel("RMSE (m)")
        ax_top.grid(True, alpha=0.3)
        ax_top_pct.set_ylabel("RMSE (%)")

        source_lines: Dict[str, object] = {}
        source_pct_lines: Dict[str, object] = {}
        nan_series = np.full_like(self._bin_centers, np.nan, dtype=np.float32)
        for source in RangeRMSEAnalyzer.SOURCE_ORDER:
            line, = ax_top.plot(
                self._bin_centers,
                nan_series,
                label=self._SOURCE_LABELS.get(source, source),
                linewidth=2.0,
                color=self._SOURCE_COLORS.get(source, None),
            )
            line.set_visible(False)
            source_lines[source] = line

            pct_line, = ax_top_pct.plot(
                self._bin_centers,
                nan_series,
                label=f"{self._SOURCE_LABELS.get(source, source)}_pct",
                linewidth=1.8,
                linestyle="--",
                color=self._SOURCE_COLORS.get(source, None),
                alpha=0.85,
            )
            pct_line.set_visible(False)
            source_pct_lines[source] = pct_line

        rmse_total_text = ax_top.text(
            0.02,
            0.98,
            "rmse_total=n/a",
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.6",
                "alpha": 0.85,
            },
        )
        class_header_text = ax_top.text(
            0.98,
            0.98,
            "Cone classification",
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            fontweight="semibold",
            fontsize=10,
        )
        class_body_text = ax_top.text(
            0.98,
            0.90,
            "Correct: 0\nIncorrect: 0\nAccuracy: n/a",
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.6",
                "alpha": 0.85,
            },
        )

        zeros = np.zeros_like(self._bin_centers, dtype=np.float32)
        bars = ax_bottom.bar(
            self._bin_centers,
            zeros,
            width=self._bin_width_m * 0.9,
            align="center",
            color="tab:gray",
        )
        ax_bottom.set_xlim(self._range_min_m, self._range_max_m)
        ax_bottom.set_xlabel("Ground-truth range (m)")
        ax_bottom.set_ylabel("Samples")
        ax_bottom.grid(True, alpha=0.3)

        return _PanelArtists(
            ax_top=ax_top,
            ax_top_pct=ax_top_pct,
            ax_bottom=ax_bottom,
            source_lines=source_lines,
            source_pct_lines=source_pct_lines,
            bars=bars,
            rmse_total_text=rmse_total_text,
            class_header_text=class_header_text,
            class_body_text=class_body_text,
        )

    def _update_panel(self, panel: _PanelArtists, stats: RangeRMSEBinStats) -> None:
        centers = np.asarray(stats.bin_centers, dtype=np.float32)
        counts = np.asarray(stats.total_counts, dtype=np.float32)
        if centers.shape != self._bin_centers.shape:
            return

        finite_series = []
        finite_pct_series = []
        visible_sources = []
        source_rmse_pct = self._compute_source_rmse_percent(centers, stats.source_rmse)
        for source, line in panel.source_lines.items():
            rmse = np.asarray(
                stats.source_rmse.get(source, np.full_like(self._bin_centers, np.nan)),
                dtype=np.float32,
            )
            rmse_pct = np.asarray(
                source_rmse_pct.get(source, np.full_like(self._bin_centers, np.nan)),
                dtype=np.float32,
            )
            line.set_xdata(centers)
            line.set_ydata(rmse)
            pct_line = panel.source_pct_lines[source]
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

        self._update_legend(panel, visible_sources)

        if finite_series:
            panel.ax_top.set_ylim(
                0.0,
                max(0.1, float(np.max(np.concatenate(finite_series))) * 1.2),
            )
        else:
            panel.ax_top.set_ylim(0.0, 1.0)

        if finite_pct_series:
            panel.ax_top_pct.set_ylim(
                0.0,
                max(0.1, float(np.max(np.concatenate(finite_pct_series))) * 1.2),
            )
        else:
            panel.ax_top_pct.set_ylim(0.0, 1.0)

        total_rmse_pct = self._compute_total_rmse_percent(stats)
        panel.rmse_total_text.set_text(
            f"rmse_total={'n/a' if total_rmse_pct is None else f'{total_rmse_pct:.2f}%'}"
        )

        max_count = 0.0
        for bar, count in zip(panel.bars, counts):
            height = float(count)
            bar.set_height(height)
            if height > max_count:
                max_count = height
        panel.ax_bottom.set_ylim(0.0, max(1.0, max_count * 1.2))

        correct_count = int(stats.correct_class_count)
        incorrect_count = int(stats.incorrect_class_count)
        total_count = correct_count + incorrect_count
        accuracy_text = (
            f"{(100.0 * float(correct_count) / float(total_count)):.1f}%"
            if total_count > 0
            else "n/a"
        )
        panel.class_body_text.set_text(
            f"Correct: {correct_count}\n"
            f"Incorrect: {incorrect_count}\n"
            f"Accuracy: {accuracy_text}"
        )

    @staticmethod
    def _compute_source_rmse_percent(
        bin_centers: np.ndarray,
        source_rmse: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        denom = np.asarray(bin_centers, dtype=np.float32)
        valid = np.isfinite(denom) & (np.abs(denom) > 1e-6)
        for source, rmse in source_rmse.items():
            arr = np.asarray(rmse, dtype=np.float32)
            pct = np.full_like(arr, np.nan, dtype=np.float32)
            if arr.shape == denom.shape:
                pct[valid] = (100.0 * arr[valid]) / denom[valid]
            out[source] = pct
        return out

    @staticmethod
    def _compute_total_rmse_percent(stats: RangeRMSEBinStats) -> Optional[float]:
        rmse_arrays = [
            np.asarray(arr, dtype=np.float32)
            for arr in stats.source_rmse.values()
            if np.asarray(arr).size == np.asarray(stats.bin_centers).size
        ]
        if not rmse_arrays:
            return None
        stacked = np.vstack(rmse_arrays)
        if not np.any(np.isfinite(stacked)):
            return None
        with np.errstate(invalid="ignore"):
            finite_rmse = np.nanmean(stacked, axis=0)
        centers = np.asarray(stats.bin_centers, dtype=np.float32)
        valid = np.isfinite(finite_rmse) & np.isfinite(centers) & (np.abs(centers) > 1e-6)
        if not np.any(valid):
            return None
        return float(np.nanmean((100.0 * finite_rmse[valid]) / centers[valid]))

    @staticmethod
    def _update_legend(panel: _PanelArtists, visible_sources: list[str]) -> None:
        legend = panel.ax_top.get_legend()
        if legend is not None:
            legend.remove()
        handles = []
        for source in RangeRMSEAnalyzer.SOURCE_ORDER:
            if source not in visible_sources:
                continue
            handles.append(panel.source_lines[source])
            handles.append(panel.source_pct_lines[source])
        if handles:
            panel.ax_top.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.88))

    def _is_open(self) -> bool:
        if self._fig is None:
            return False
        return bool(self._plt.fignum_exists(self._fig.number))

    def _configure_window_focus_behavior(self) -> None:
        manager = getattr(self._fig.canvas, "manager", None)
        window = getattr(manager, "window", None)
        if window is None:
            return
        try:
            from PyQt5 import QtCore

            window.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
            window.setFocusPolicy(QtCore.Qt.NoFocus)
        except Exception:
            return


class ConeRMSEPlotNode(Node):
    """Subscribe to evaluator sample topics and render a combined cone RMSE window."""

    def __init__(self) -> None:
        super().__init__("cone_rmse_plot_node")
        self.declare_parameter("camera_eval_topic", "/sim/raw/stereo/eval")
        self.declare_parameter("lidar_eval_topic", "/sim/raw/lidar/eval")
        self.declare_parameter("camera_source", "stereo")
        self.declare_parameter("update_period_sec", 0.2)

        camera_eval_topic = str(self.get_parameter("camera_eval_topic").value).strip().rstrip("/")
        lidar_eval_topic = str(self.get_parameter("lidar_eval_topic").value).strip().rstrip("/")
        camera_enabled = bool(camera_eval_topic)
        lidar_enabled = bool(lidar_eval_topic)
        self._camera_source = (
            str(self.get_parameter("camera_source").value).strip().lower() or "stereo"
        )
        update_period_sec = max(0.05, float(self.get_parameter("update_period_sec").value))

        self._camera_analyzer = RangeRMSEAnalyzer(range_min_m=0.0, range_max_m=20.0, bin_width_m=1.0)
        self._lidar_analyzer = RangeRMSEAnalyzer(range_min_m=0.0, range_max_m=20.0, bin_width_m=1.0)

        left_title = "Stereo" if self._camera_source == "stereo" else "Monocular"
        if not camera_enabled:
            left_title = f"{left_title} (disabled)"
        right_title = "LiDAR" if lidar_enabled else "LiDAR (disabled)"
        self._figure: Optional[_SideBySideRMSEFigure] = None
        try:
            self._figure = _SideBySideRMSEFigure(left_title=left_title, right_title=right_title)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(
                f"Failed to initialize cone RMSE window ({exc}); disabling node."
            )

        if camera_eval_topic:
            self.create_subscription(
                String,
                f"{camera_eval_topic}/cone_depth_samples",
                self._camera_samples_cb,
                10,
            )
        if lidar_eval_topic:
            self.create_subscription(
                String,
                f"{lidar_eval_topic}/cone_depth_samples",
                self._lidar_samples_cb,
                10,
            )
        self._timer = self.create_timer(update_period_sec, self._update_plot)
        rclpy.get_default_context().on_shutdown(self.shutdown)

        self.get_logger().info(
            "ConeRMSEPlotNode ready: "
            f"camera_samples={(camera_eval_topic + '/cone_depth_samples') if camera_enabled else 'disabled'} "
            f"lidar_samples={(lidar_eval_topic + '/cone_depth_samples') if lidar_enabled else 'disabled'} "
            f"camera_source={self._camera_source}"
        )

    def _camera_samples_cb(self, msg: String) -> None:
        self._ingest_payload(msg.data, self._camera_analyzer, default_source=self._camera_source)

    def _lidar_samples_cb(self, msg: String) -> None:
        self._ingest_payload(msg.data, self._lidar_analyzer, default_source="lidar")

    def _ingest_payload(
        self,
        payload: str,
        analyzer: RangeRMSEAnalyzer,
        *,
        default_source: str,
    ) -> None:
        lines = [line.strip() for line in str(payload).splitlines() if line.strip()]
        if not lines:
            return
        start_idx = (
            1 if lines[0].lower().replace(" ", "").startswith("source,gt_range_m,error_m") else 0
        )

        for line in lines[start_idx:]:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            source = parts[0].lower() if parts[0] else default_source
            try:
                gt_range = float(parts[1])
                error_m = float(parts[2])
            except ValueError:
                continue
            if not (math.isfinite(gt_range) and math.isfinite(error_m)):
                continue

            predicted_class_id = self._parse_optional_int(parts[3]) if len(parts) > 3 else None
            ground_truth_class_id = self._parse_optional_int(parts[4]) if len(parts) > 4 else None
            analyzer.add_sample(
                source=source,
                gt_range_m=gt_range,
                error_m=error_m,
                predicted_class_id=predicted_class_id,
                ground_truth_class_id=ground_truth_class_id,
            )

    @staticmethod
    def _parse_optional_int(value: str) -> Optional[int]:
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None

    def _update_plot(self) -> None:
        if self._figure is None:
            return
        left_stats = self._camera_analyzer.compute_binned_rmse()
        right_stats = self._lidar_analyzer.compute_binned_rmse()
        if not self._figure.update(left_stats, right_stats):
            self.get_logger().warn("Cone RMSE plot window closed")
            self._figure = None

    def shutdown(self) -> None:
        if self._figure is not None:
            self._figure.close()
            self._figure = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConeRMSEPlotNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
