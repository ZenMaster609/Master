"""Shared runtime helper for cone_plotting_2 sample publishing and live plotting."""

from __future__ import annotations

import math
from typing import Optional

from std_msgs.msg import String

from sim_car.perception.range_rmse_analyzer import RangeRMSEAnalyzer
from sim_car.perception.range_rmse_live_plot import RangeRMSELivePlot


class ConePlotting2Runtime:
    """Holds analyzer, live plot, and sample CSV publisher state."""

    def __init__(self, node, *, eval_topic_prefix: str, enabled: bool, enable_live_plot: bool = True):
        self._node = node
        self.enabled = bool(enabled)
        self._enable_live_plot = bool(enable_live_plot)
        self._sample_pub = node.create_publisher(String, f"{eval_topic_prefix.rstrip('/')}/cone_depth_samples", 10)
        self._analyzer: Optional[RangeRMSEAnalyzer] = None
        self._plot: Optional[RangeRMSELivePlot] = None
        self._timer = None

        if not self.enabled:
            return

        self._analyzer = RangeRMSEAnalyzer(range_min_m=0.0, range_max_m=20.0, bin_width_m=1.0)
        if not self._enable_live_plot:
            return
        try:
            self._plot = RangeRMSELivePlot(range_min_m=0.0, range_max_m=20.0, bin_width_m=1.0)
            self._timer = node.create_timer(0.2, self._update_plot)
        except Exception as exc:  # pylint: disable=broad-except
            self._node.get_logger().warn(f'Failed to initialize cone_plotting_2 window ({exc}); disabling live plot.')
            self._plot = None

    def record_sample(
        self,
        *,
        source: str,
        gt_range_m: float,
        error_m: float,
        predicted_class_id: Optional[int] = None,
        ground_truth_class_id: Optional[int] = None,
    ) -> None:
        if not self.enabled or self._analyzer is None:
            return
        self._analyzer.add_sample(
            source=source,
            gt_range_m=float(gt_range_m),
            error_m=float(error_m),
            predicted_class_id=predicted_class_id,
            ground_truth_class_id=ground_truth_class_id,
        )

    def publish_sample_rows(self, samples: list[tuple[str, float, float, Optional[int], Optional[int]]]) -> None:
        if not self.enabled or not samples:
            return
        lines = ['source,gt_range_m,error_m,predicted_class_id,ground_truth_class_id']
        for source, gt_range_m, error_m, predicted_class_id, ground_truth_class_id in samples:
            if not source:
                continue
            if not (math.isfinite(gt_range_m) and math.isfinite(error_m)):
                continue
            predicted_str = '' if predicted_class_id is None else str(int(predicted_class_id))
            gt_str = '' if ground_truth_class_id is None else str(int(ground_truth_class_id))
            lines.append(f'{source},{gt_range_m:.6f},{error_m:.6f},{predicted_str},{gt_str}')
        if len(lines) > 1:
            self._sample_pub.publish(String(data='\n'.join(lines)))

    def close(self) -> None:
        if self._plot is not None:
            try:
                self._plot.close()
            except Exception:  # pylint: disable=broad-except
                pass
            self._plot = None

    def _update_plot(self) -> None:
        if self._analyzer is None or self._plot is None:
            return
        stats = self._analyzer.compute_binned_rmse()
        if not self._plot.update(stats):
            self._plot = None
