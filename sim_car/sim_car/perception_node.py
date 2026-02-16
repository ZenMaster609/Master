"""ROS2 node that runs stereo perception and evaluation in one process."""

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Deque, Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from sim_car.perception import PerfLogger, StereoEvaluator, StereoPipeline, StereoPipelineConfig


@dataclass
class FramePacket:
    """Queued frame packet used by pairer worker."""

    msg: Image
    pair_time_sec: float


class PerceptionNode(Node):
    """Subscribes stereo topics, computes disparity/depth, and publishes eval metrics."""

    def __init__(self):
        super().__init__('perception_node')

        self._declare_parameters()
        self._read_parameters()

        self._left_info: Optional[CameraInfo] = None
        self._right_info: Optional[CameraInfo] = None

        self._left_queue: Deque[FramePacket] = deque()
        self._right_queue: Deque[FramePacket] = deque()
        self._queue_lock = threading.Lock()
        self._queue_cv = threading.Condition(self._queue_lock)
        self._running = True

        self._pipeline = StereoPipeline(
            logger=self.get_logger(),
            config=StereoPipelineConfig(
                calibration_file=self.calibration_file,
                prefer_cuda=self.prefer_cuda,
                min_disparity=self.min_disparity,
                num_disparities=self.num_disparities,
                block_size=self.block_size,
                uniqueness_ratio=self.uniqueness_ratio,
                speckle_window_size=self.speckle_window_size,
                speckle_range=self.speckle_range,
                disp12_max_diff=self.disp12_max_diff,
                pre_filter_cap=self.pre_filter_cap,
                baseline_m=self.baseline_m,
                focal_length_px=self.focal_length_px,
                disparity_valid_threshold=self.disparity_valid_threshold,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
            ),
        )
        self._evaluator = StereoEvaluator(
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            disparity_valid_threshold=self.disparity_valid_threshold,
            orb_features=self.eval_orb_features,
            max_matches=self.eval_max_matches,
            match_ratio_test=self.eval_match_ratio_test,
        )
        self._perf = PerfLogger(self, eval_topic_prefix=self.eval_topic_prefix)

        self.create_subscription(Image, self.left_image_topic, self._left_image_cb, 10)
        self.create_subscription(Image, self.right_image_topic, self._right_image_cb, 10)
        self.create_subscription(CameraInfo, self.left_camera_info_topic, self._left_info_cb, 10)
        self.create_subscription(CameraInfo, self.right_camera_info_topic, self._right_info_cb, 10)

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        if self.perf_log_hz > 0.0:
            self.create_timer(1.0 / self.perf_log_hz, self._perf_timer_cb)

        self.get_logger().info(
            'perception_node ready: '
            f'left={self.left_image_topic} right={self.right_image_topic} '
            f'eval_prefix={self.eval_topic_prefix} perf_log_hz={self.perf_log_hz:.2f}'
        )

    def _declare_parameters(self):
        self.declare_parameter('left_image_topic', '/sim/raw/stereo/left/image_raw')
        self.declare_parameter('right_image_topic', '/sim/raw/stereo/right/image_raw')
        self.declare_parameter('left_camera_info_topic', '/sim/raw/stereo/left/camera_info')
        self.declare_parameter('right_camera_info_topic', '/sim/raw/stereo/right/camera_info')

        self.declare_parameter('calibration_file', '')
        self.declare_parameter('max_time_diff_sec', 0.08)
        self.declare_parameter('queue_size', 30)
        self.declare_parameter('perf_log_hz', 1.0)
        self.declare_parameter('prefer_cuda', True)

        self.declare_parameter('min_disparity', 0)
        self.declare_parameter('num_disparities', 192)
        self.declare_parameter('block_size', 7)
        self.declare_parameter('uniqueness_ratio', 10)
        self.declare_parameter('speckle_window_size', 100)
        self.declare_parameter('speckle_range', 2)
        self.declare_parameter('disp12_max_diff', 1)
        self.declare_parameter('pre_filter_cap', 31)

        self.declare_parameter('baseline_m', 0.12)
        self.declare_parameter('focal_length_px', 0.0)
        self.declare_parameter('disparity_valid_threshold', 0.1)
        self.declare_parameter('min_depth_m', 0.3)
        self.declare_parameter('max_depth_m', 30.0)

        self.declare_parameter('eval_topic_prefix', '/sim/raw/stereo/eval')
        self.declare_parameter('eval_orb_features', 700)
        self.declare_parameter('eval_max_matches', 200)
        self.declare_parameter('eval_match_ratio_test', 0.75)

    def _read_parameters(self):
        self.left_image_topic = str(self.get_parameter('left_image_topic').value)
        self.right_image_topic = str(self.get_parameter('right_image_topic').value)
        self.left_camera_info_topic = str(self.get_parameter('left_camera_info_topic').value)
        self.right_camera_info_topic = str(self.get_parameter('right_camera_info_topic').value)

        self.calibration_file = str(self.get_parameter('calibration_file').value)
        self.max_time_diff_sec = max(0.0, float(self.get_parameter('max_time_diff_sec').value))
        self.queue_size = max(5, int(self.get_parameter('queue_size').value))
        self.perf_log_hz = max(0.0, float(self.get_parameter('perf_log_hz').value))
        self.prefer_cuda = bool(self.get_parameter('prefer_cuda').value)

        self.min_disparity = int(self.get_parameter('min_disparity').value)
        self.num_disparities = int(self.get_parameter('num_disparities').value)
        self.block_size = int(self.get_parameter('block_size').value)
        self.uniqueness_ratio = int(self.get_parameter('uniqueness_ratio').value)
        self.speckle_window_size = int(self.get_parameter('speckle_window_size').value)
        self.speckle_range = int(self.get_parameter('speckle_range').value)
        self.disp12_max_diff = int(self.get_parameter('disp12_max_diff').value)
        self.pre_filter_cap = int(self.get_parameter('pre_filter_cap').value)

        self.baseline_m = float(self.get_parameter('baseline_m').value)
        self.focal_length_px = float(self.get_parameter('focal_length_px').value)
        self.disparity_valid_threshold = float(self.get_parameter('disparity_valid_threshold').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)

        self.eval_topic_prefix = str(self.get_parameter('eval_topic_prefix').value)
        self.eval_orb_features = int(self.get_parameter('eval_orb_features').value)
        self.eval_max_matches = int(self.get_parameter('eval_max_matches').value)
        self.eval_match_ratio_test = float(self.get_parameter('eval_match_ratio_test').value)

    def _left_info_cb(self, msg: CameraInfo):
        self._left_info = msg

    def _right_info_cb(self, msg: CameraInfo):
        self._right_info = msg

    def _left_image_cb(self, msg: Image):
        self._enqueue_frame(msg, side='left')

    def _right_image_cb(self, msg: Image):
        self._enqueue_frame(msg, side='right')

    def _enqueue_frame(self, msg: Image, side: str):
        pair_time_sec = self._stamp_to_sec(msg)
        if pair_time_sec <= 0.0:
            pair_time_sec = time.monotonic()

        packet = FramePacket(msg=msg, pair_time_sec=pair_time_sec)
        with self._queue_cv:
            queue = self._left_queue if side == 'left' else self._right_queue
            queue.append(packet)
            self._perf.count_incoming(side=side, queue_len=len(queue))

            while len(queue) > self.queue_size:
                queue.popleft()
                self._perf.count_drop(side=side)

            self._queue_cv.notify()

    def _worker_loop(self):
        while True:
            with self._queue_cv:
                pair = None
                while self._running and pair is None:
                    pair = self._pair_frames()
                    if pair is None:
                        self._queue_cv.wait(timeout=0.05)
                if not self._running:
                    return

            left_packet, right_packet = pair
            output = self._pipeline.process(
                left_msg=left_packet.msg,
                right_msg=right_packet.msg,
                left_info=self._left_info,
                right_info=self._right_info,
            )
            if output is None:
                continue

            self._evaluator.update(
                left_rect_gray=output.left_rect,
                right_rect_gray=output.right_rect,
                disparity=output.disparity,
                depth=output.depth,
            )
            self._perf.record_processed(timings_ms=output.timings_ms, backend=output.backend)

    def _pair_frames(self):
        while self._left_queue and self._right_queue:
            left_head = self._left_queue[0]
            best_idx = -1
            best_dt = None
            for idx, right_packet in enumerate(self._right_queue):
                dt = abs(left_head.pair_time_sec - right_packet.pair_time_sec)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_idx = idx

            if best_dt is not None and best_dt <= self.max_time_diff_sec:
                left_packet = self._left_queue.popleft()
                right_packet = self._right_queue[best_idx]
                del self._right_queue[best_idx]
                self._perf.count_pair(best_dt)
                return left_packet, right_packet

            left_t = left_head.pair_time_sec
            right_oldest_t = self._right_queue[0].pair_time_sec
            right_newest_t = self._right_queue[-1].pair_time_sec

            if right_newest_t < left_t - self.max_time_diff_sec:
                self._right_queue.popleft()
                self._perf.count_drop(side='right')
                continue
            if left_t < right_oldest_t - self.max_time_diff_sec:
                self._left_queue.popleft()
                self._perf.count_drop(side='left')
                continue

            if left_t <= right_oldest_t:
                self._left_queue.popleft()
                self._perf.count_drop(side='left')
            else:
                self._right_queue.popleft()
                self._perf.count_drop(side='right')
        return None

    def _perf_timer_cb(self):
        self._perf.log_and_publish(self._evaluator.snapshot())

    @staticmethod
    def _stamp_to_sec(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def destroy_node(self):
        with self._queue_cv:
            self._running = False
            self._queue_cv.notify_all()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
