"""
Stereo quality evaluation node.

Evaluates calibrated stereo output using:
- Epipolar row error on rectified images (feature-match |dy|)
- Valid disparity ratio
- Valid depth ratio and basic depth stats
"""

from collections import deque
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class StereoEvalNode(Node):
    """Evaluate stereo calibration/rectification quality from live topics."""

    def __init__(self):
        super().__init__('stereo_eval_node')

        self.declare_parameter('left_rect_topic', '/sim/raw/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic', '/sim/raw/stereo/right/image_rect')
        self.declare_parameter('disparity_topic', '/sim/raw/stereo/disparity')
        self.declare_parameter('depth_topic', '/sim/raw/stereo/depth')
        self.declare_parameter('max_time_diff_sec', 0.04)
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('report_period_sec', 1.0)
        self.declare_parameter('min_depth_m', 0.3)
        self.declare_parameter('max_depth_m', 30.0)
        self.declare_parameter('orb_features', 700)
        self.declare_parameter('max_matches', 200)
        self.declare_parameter('match_ratio_test', 0.75)

        left_rect_topic = str(self.get_parameter('left_rect_topic').value)
        right_rect_topic = str(self.get_parameter('right_rect_topic').value)
        disparity_topic = str(self.get_parameter('disparity_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)

        self.max_time_diff_sec = float(self.get_parameter('max_time_diff_sec').value)
        self.queue_size = max(2, int(self.get_parameter('queue_size').value))
        self.report_period_sec = float(self.get_parameter('report_period_sec').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.max_matches = int(self.get_parameter('max_matches').value)
        self.match_ratio_test = float(self.get_parameter('match_ratio_test').value)

        self.orb = cv2.ORB_create(nfeatures=int(self.get_parameter('orb_features').value))
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._left_q = deque(maxlen=self.queue_size)
        self._right_q = deque(maxlen=self.queue_size)
        self._disp_q = deque(maxlen=self.queue_size)
        self._depth_q = deque(maxlen=self.queue_size)

        self._last_epi_mean: Optional[float] = None
        self._last_epi_med: Optional[float] = None
        self._last_epi_matches: int = 0
        self._last_disp_valid: Optional[float] = None
        self._last_depth_valid: Optional[float] = None
        self._last_depth_mean: Optional[float] = None

        self.create_subscription(Image, left_rect_topic, self._left_cb, 10)
        self.create_subscription(Image, right_rect_topic, self._right_cb, 10)
        self.create_subscription(Image, disparity_topic, self._disp_cb, 10)
        self.create_subscription(Image, depth_topic, self._depth_cb, 10)

        self.epi_mean_pub = self.create_publisher(Float32, '/sim/raw/stereo/eval/epipolar_mean_px', 10)
        self.epi_med_pub = self.create_publisher(Float32, '/sim/raw/stereo/eval/epipolar_median_px', 10)
        self.disp_valid_pub = self.create_publisher(Float32, '/sim/raw/stereo/eval/disparity_valid_ratio', 10)
        self.depth_valid_pub = self.create_publisher(Float32, '/sim/raw/stereo/eval/depth_valid_ratio', 10)
        self.depth_mean_pub = self.create_publisher(Float32, '/sim/raw/stereo/eval/depth_mean_m', 10)

        self.create_timer(self.report_period_sec, self._report)

        self.get_logger().info(
            f'stereo_eval_node subscribed to left={left_rect_topic}, right={right_rect_topic}, '
            f'disparity={disparity_topic}, depth={depth_topic}'
        )

    def _left_cb(self, msg: Image):
        self._left_q.append(msg)
        self._try_epipolar_eval()

    def _right_cb(self, msg: Image):
        self._right_q.append(msg)
        self._try_epipolar_eval()

    def _disp_cb(self, msg: Image):
        self._disp_q.append(msg)
        arr = self._image_to_32fc1(msg)
        if arr is None:
            return
        valid = arr > 0.0
        self._last_disp_valid = float(np.mean(valid))

    def _depth_cb(self, msg: Image):
        self._depth_q.append(msg)
        arr = self._image_to_32fc1(msg)
        if arr is None:
            return
        finite = np.isfinite(arr)
        valid = finite & (arr >= self.min_depth_m) & (arr <= self.max_depth_m)
        self._last_depth_valid = float(np.mean(valid))
        if np.any(valid):
            self._last_depth_mean = float(np.mean(arr[valid]))

    def _try_epipolar_eval(self):
        if not self._left_q or not self._right_q:
            return
        left = self._left_q[0]
        left_t = self._stamp_to_sec(left)

        best_idx = -1
        best_dt = None
        for idx, right in enumerate(self._right_q):
            dt = abs(left_t - self._stamp_to_sec(right))
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_idx = idx

        if best_dt is None:
            return
        if best_dt > self.max_time_diff_sec:
            if self._stamp_to_sec(self._right_q[-1]) < left_t:
                self._left_q.popleft()
            return

        right = self._right_q[best_idx]
        self._left_q.popleft()
        del self._right_q[best_idx]
        self._compute_epipolar_error(left, right)

    def _compute_epipolar_error(self, left_msg: Image, right_msg: Image):
        left = self._image_to_bgr(left_msg)
        right = self._image_to_bgr(right_msg)
        if left is None or right is None:
            return

        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.orb.detectAndCompute(left_gray, None)
        kp2, des2 = self.orb.detectAndCompute(right_gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            self._last_epi_matches = 0
            return

        knn = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.match_ratio_test * n.distance:
                good.append(m)

        if not good:
            self._last_epi_matches = 0
            return

        good = sorted(good, key=lambda m: m.distance)[: self.max_matches]
        dy = []
        for m in good:
            y_l = kp1[m.queryIdx].pt[1]
            y_r = kp2[m.trainIdx].pt[1]
            dy.append(abs(y_l - y_r))

        if not dy:
            self._last_epi_matches = 0
            return

        dy_arr = np.array(dy, dtype=np.float32)
        self._last_epi_mean = float(np.mean(dy_arr))
        self._last_epi_med = float(np.median(dy_arr))
        self._last_epi_matches = int(len(dy_arr))

    def _report(self):
        if self._last_epi_mean is not None:
            self.epi_mean_pub.publish(Float32(data=float(self._last_epi_mean)))
        if self._last_epi_med is not None:
            self.epi_med_pub.publish(Float32(data=float(self._last_epi_med)))
        if self._last_disp_valid is not None:
            self.disp_valid_pub.publish(Float32(data=float(self._last_disp_valid)))
        if self._last_depth_valid is not None:
            self.depth_valid_pub.publish(Float32(data=float(self._last_depth_valid)))
        if self._last_depth_mean is not None:
            self.depth_mean_pub.publish(Float32(data=float(self._last_depth_mean)))

        self.get_logger().info(
            'stereo_eval: '
            f'epi_mean_px={self._fmt(self._last_epi_mean)} '
            f'epi_median_px={self._fmt(self._last_epi_med)} '
            f'matches={self._last_epi_matches} '
            f'disp_valid={self._fmt(self._last_disp_valid)} '
            f'depth_valid={self._fmt(self._last_depth_valid)} '
            f'depth_mean_m={self._fmt(self._last_depth_mean)}'
        )

    def _fmt(self, val: Optional[float]) -> str:
        return 'n/a' if val is None else f'{val:.4f}'

    def _image_to_32fc1(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding.lower() != '32fc1':
            return None
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
            return arr.reshape((msg.height, msg.width))
        except Exception:  # pylint: disable=broad-except
            return None

    def _image_to_bgr(self, msg: Image) -> Optional[np.ndarray]:
        try:
            if msg.height <= 0 or msg.width <= 0:
                return None
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            enc = msg.encoding.lower()
            if enc == 'bgr8':
                return data.reshape((msg.height, msg.width, 3))
            if enc == 'rgb8':
                rgb = data.reshape((msg.height, msg.width, 3))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if enc in ('mono8', '8uc1'):
                mono = data.reshape((msg.height, msg.width))
                return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
            return None
        except Exception:  # pylint: disable=broad-except
            return None

    def _stamp_to_sec(self, msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = StereoEvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
