"""Shared runtime helper for cone RMSE sample publishing."""

from __future__ import annotations

import math
from typing import Optional

from std_msgs.msg import String


class ConePlotting2Runtime:
    """Publishes evaluator sample rows for downstream cone RMSE logging/plotting."""

    def __init__(self, node, *, eval_topic_prefix: str, enabled: bool):
        self._node = node
        self.enabled = bool(enabled)
        self._sample_pub = node.create_publisher(String, f"{eval_topic_prefix.rstrip('/')}/cone_depth_samples", 10)

    def record_sample(
        self,
        *,
        source: str,
        gt_range_m: float,
        error_m: float,
        predicted_class_id: Optional[int] = None,
        ground_truth_class_id: Optional[int] = None,
    ) -> None:
        del source, gt_range_m, error_m, predicted_class_id, ground_truth_class_id

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
        return
