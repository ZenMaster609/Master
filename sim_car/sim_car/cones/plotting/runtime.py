"""Shared runtime helper for cone RMSE sample publishing."""

from __future__ import annotations

import math
from typing import Optional

from std_msgs.msg import String


def format_sample_rows(samples: list[tuple[str, float, float, Optional[int], Optional[int]]]) -> str:
    lines = ['source,gt_range_m,error_m,predicted_class_id,ground_truth_class_id']
    for source, gt_range_m, error_m, predicted_class_id, ground_truth_class_id in samples:
        if not source:
            continue
        if not (math.isfinite(gt_range_m) and math.isfinite(error_m)):
            continue
        predicted_str = '' if predicted_class_id is None else str(int(predicted_class_id))
        gt_str = '' if ground_truth_class_id is None else str(int(ground_truth_class_id))
        lines.append(f'{source},{gt_range_m:.6f},{error_m:.6f},{predicted_str},{gt_str}')
    return '\n'.join(lines)


class ConePlotting2Runtime:
    """Publishes evaluator sample rows for downstream cone RMSE logging/plotting."""

    def __init__(self, node, *, eval_topic_prefix: str, enabled: bool):
        self._node = node
        self.enabled = bool(enabled)
        self._sample_pub = node.create_publisher(String, f"{eval_topic_prefix.rstrip('/')}/cone_depth_samples", 10)

    def publish_sample_rows(self, samples: list[tuple[str, float, float, Optional[int], Optional[int]]]) -> None:
        if not self.enabled or not samples:
            return
        payload = format_sample_rows(samples)
        if payload.count('\n') > 0:
            self._sample_pub.publish(String(data=payload))

    def close(self) -> None:
        return
