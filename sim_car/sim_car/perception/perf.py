"""Performance accounting and periodic stereo debug publishing."""

from dataclasses import dataclass
import threading
import time

from std_msgs.msg import Float32

from .eval_metrics import StereoEvalMetrics


@dataclass
class PerfSnapshot:
    """Interval snapshot of pair/process throughput and timing."""

    elapsed_sec: float
    incoming_left: int
    incoming_right: int
    paired: int
    processed: int
    dropped_left: int
    dropped_right: int
    queue_peak_left: int
    queue_peak_right: int
    pair_dt_sum: float
    decode_ms_sum: float
    rectify_ms_sum: float
    disparity_ms_sum: float
    depth_ms_sum: float
    total_ms_sum: float
    backend: str


class PerfLogger:
    """Tracks perception rates/timings and publishes eval metrics each perf tick."""

    def __init__(self, node, eval_topic_prefix: str):
        self._node = node
        self._lock = threading.Lock()
        self._last_log_time = time.monotonic()

        self._incoming_left = 0
        self._incoming_right = 0
        self._paired = 0
        self._processed = 0
        self._dropped_left = 0
        self._dropped_right = 0
        self._queue_peak_left = 0
        self._queue_peak_right = 0
        self._pair_dt_sum = 0.0
        self._decode_ms_sum = 0.0
        self._rectify_ms_sum = 0.0
        self._disparity_ms_sum = 0.0
        self._depth_ms_sum = 0.0
        self._total_ms_sum = 0.0
        self._backend = 'cpu'

        prefix = eval_topic_prefix.rstrip('/')
        self._epi_mean_pub = node.create_publisher(Float32, f'{prefix}/epipolar_mean_px', 10)
        self._epi_median_pub = node.create_publisher(Float32, f'{prefix}/epipolar_median_px', 10)
        self._disp_valid_pub = node.create_publisher(Float32, f'{prefix}/disparity_valid_ratio', 10)
        self._depth_valid_pub = node.create_publisher(Float32, f'{prefix}/depth_valid_ratio', 10)
        self._depth_mean_pub = node.create_publisher(Float32, f'{prefix}/depth_mean_m', 10)

    def count_incoming(self, side: str, queue_len: int):
        with self._lock:
            if side == 'left':
                self._incoming_left += 1
                self._queue_peak_left = max(self._queue_peak_left, int(queue_len))
            else:
                self._incoming_right += 1
                self._queue_peak_right = max(self._queue_peak_right, int(queue_len))

    def count_pair(self, pair_dt_sec: float):
        with self._lock:
            self._paired += 1
            self._pair_dt_sum += float(pair_dt_sec)

    def count_drop(self, side: str):
        with self._lock:
            if side == 'left':
                self._dropped_left += 1
            else:
                self._dropped_right += 1

    def record_processed(self, timings_ms: dict, backend: str):
        with self._lock:
            self._processed += 1
            self._decode_ms_sum += float(timings_ms.get('decode', 0.0))
            self._rectify_ms_sum += float(timings_ms.get('rectify', 0.0))
            self._disparity_ms_sum += float(timings_ms.get('disparity', 0.0))
            self._depth_ms_sum += float(timings_ms.get('depth', 0.0))
            self._total_ms_sum += float(timings_ms.get('total', 0.0))
            self._backend = str(backend)

    def log_and_publish(self, eval_metrics: StereoEvalMetrics):
        snapshot = self._snapshot_and_reset()
        if snapshot is None:
            return

        self._publish_eval(eval_metrics)

        in_left_hz = snapshot.incoming_left / snapshot.elapsed_sec
        in_right_hz = snapshot.incoming_right / snapshot.elapsed_sec
        paired_hz = snapshot.paired / snapshot.elapsed_sec
        processed_hz = snapshot.processed / snapshot.elapsed_sec
        pair_dt_ms = (snapshot.pair_dt_sum / max(1, snapshot.paired)) * 1000.0
        decode_ms = snapshot.decode_ms_sum / max(1, snapshot.processed)
        rectify_ms = snapshot.rectify_ms_sum / max(1, snapshot.processed)
        disparity_ms = snapshot.disparity_ms_sum / max(1, snapshot.processed)
        depth_ms = snapshot.depth_ms_sum / max(1, snapshot.processed)
        total_ms = snapshot.total_ms_sum / max(1, snapshot.processed)

        self._node.get_logger().info(
            'perception perf '
            f'in_hz L/R={in_left_hz:.2f}/{in_right_hz:.2f} '
            f'paired_hz={paired_hz:.2f} processed_hz={processed_hz:.2f} '
            f'avg_ms pair_dt={pair_dt_ms:.2f} decode={decode_ms:.2f} rectify={rectify_ms:.2f} '
            f'disparity={disparity_ms:.2f} depth={depth_ms:.2f} total={total_ms:.2f} '
            f'dropped L/R={snapshot.dropped_left}/{snapshot.dropped_right} '
            f'q_peak L/R={snapshot.queue_peak_left}/{snapshot.queue_peak_right} '
            f'backend={snapshot.backend} '
            f'eval epi_mean={self._fmt(eval_metrics.epipolar_mean_px)} '
            f'epi_median={self._fmt(eval_metrics.epipolar_median_px)} '
            f'matches={eval_metrics.epipolar_matches} '
            f'disp_valid={self._fmt(eval_metrics.disparity_valid_ratio)} '
            f'depth_valid={self._fmt(eval_metrics.depth_valid_ratio)} '
            f'depth_mean={self._fmt(eval_metrics.depth_mean_m)}'
        )

    def _snapshot_and_reset(self):
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_log_time
            if elapsed <= 0.0:
                return None
            snapshot = PerfSnapshot(
                elapsed_sec=elapsed,
                incoming_left=self._incoming_left,
                incoming_right=self._incoming_right,
                paired=self._paired,
                processed=self._processed,
                dropped_left=self._dropped_left,
                dropped_right=self._dropped_right,
                queue_peak_left=self._queue_peak_left,
                queue_peak_right=self._queue_peak_right,
                pair_dt_sum=self._pair_dt_sum,
                decode_ms_sum=self._decode_ms_sum,
                rectify_ms_sum=self._rectify_ms_sum,
                disparity_ms_sum=self._disparity_ms_sum,
                depth_ms_sum=self._depth_ms_sum,
                total_ms_sum=self._total_ms_sum,
                backend=self._backend,
            )
            self._last_log_time = now
            self._incoming_left = 0
            self._incoming_right = 0
            self._paired = 0
            self._processed = 0
            self._dropped_left = 0
            self._dropped_right = 0
            self._queue_peak_left = 0
            self._queue_peak_right = 0
            self._pair_dt_sum = 0.0
            self._decode_ms_sum = 0.0
            self._rectify_ms_sum = 0.0
            self._disparity_ms_sum = 0.0
            self._depth_ms_sum = 0.0
            self._total_ms_sum = 0.0
            return snapshot

    def _publish_eval(self, metrics: StereoEvalMetrics):
        if metrics.epipolar_mean_px is not None:
            self._epi_mean_pub.publish(Float32(data=float(metrics.epipolar_mean_px)))
        if metrics.epipolar_median_px is not None:
            self._epi_median_pub.publish(Float32(data=float(metrics.epipolar_median_px)))
        if metrics.disparity_valid_ratio is not None:
            self._disp_valid_pub.publish(Float32(data=float(metrics.disparity_valid_ratio)))
        if metrics.depth_valid_ratio is not None:
            self._depth_valid_pub.publish(Float32(data=float(metrics.depth_valid_ratio)))
        if metrics.depth_mean_m is not None:
            self._depth_mean_pub.publish(Float32(data=float(metrics.depth_mean_m)))

    @staticmethod
    def _fmt(value):
        return 'n/a' if value is None else f'{value:.4f}'
