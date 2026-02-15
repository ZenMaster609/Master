"""
Simple local camera stream viewer for sim stereo feeds.

Subscribes to left/right sensor_msgs/Image topics and displays them in OpenCV.
"""

from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraStreamNode(Node):
    """Display left/right camera feeds in a local OpenCV window."""

    def __init__(self):
        super().__init__('camera_stream_node')

        # Default to rectified streams (most common for debugging stereo pipelines).
        # Override topics to view raw streams if desired.
        self.declare_parameter('left_topic', '/sim/raw/stereo/left/image_rect')
        self.declare_parameter('right_topic', '/sim/raw/stereo/right/image_rect')
        self.declare_parameter('window_name', 'Sim Stereo Stream')
        self.declare_parameter('show_right', True)
        # Resize only for display (imshow) to reduce GUI/CPU load. Does not affect the stereo pipeline.
        self.declare_parameter('downsampling', 0.3)
        self.declare_parameter('n_frames', 3)
        # Backward-compat: old param name. If set (>0), overrides downsampling.
        self.declare_parameter('display_scale', -1.0)

        left_topic = str(self.get_parameter('left_topic').value)
        right_topic = str(self.get_parameter('right_topic').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.show_right = bool(self.get_parameter('show_right').value)
        self.downsampling = float(self.get_parameter('downsampling').value)
        self.n_frames = max(1, int(self.get_parameter('n_frames').value))
        display_scale = float(self.get_parameter('display_scale').value)
        if display_scale > 0.0:
            self.downsampling = display_scale

        self._left_frame: Optional[np.ndarray] = None
        self._right_frame: Optional[np.ndarray] = None
        self._left_stamp_ns: Optional[int] = None
        self._right_stamp_ns: Optional[int] = None
        # Small buffers to match left/right by timestamp before displaying.
        self._left_buf = deque(maxlen=10)   # entries: (stamp_ns, frame)
        self._right_buf = deque(maxlen=10)  # entries: (stamp_ns, frame)
        self._render_count = 0
        self._sync_tol_ns = int(0.05 * 1e9)  # 50ms default tolerance for display pairing

        # Keep subscription depth at 1 so stale frames are dropped under load.
        self._left_sub = self.create_subscription(
            Image, left_topic, self._left_callback, 1
        )
        self._right_sub = self.create_subscription(
            Image, right_topic, self._right_callback, 1
        )

        # Keep UI events progressing even when only one stream updates.
        self._ui_timer = self.create_timer(0.03, self._render)

        self.get_logger().info(
            f'camera_stream_node listening on left={left_topic}, '
            f'right={right_topic}, show_right={self.show_right}'
        )

    def _left_callback(self, msg: Image):
        frame = self._image_to_bgr(msg)
        if frame is not None:
            self._left_buf.append((self._stamp_ns(msg), frame))
            self._try_match()

    def _right_callback(self, msg: Image):
        frame = self._image_to_bgr(msg)
        if frame is not None:
            self._right_buf.append((self._stamp_ns(msg), frame))
            self._try_match()

    @staticmethod
    def _stamp_ns(msg: Image) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _try_match(self):
        if not self._left_buf or not self._right_buf:
            return

        # Find best (closest) pair; if multiple have same dt, prefer newest.
        best = None  # (dt_ns, -newest_ns, i, j)
        for i, (l_ns, _l_frame) in enumerate(self._left_buf):
            for j, (r_ns, _r_frame) in enumerate(self._right_buf):
                dt = abs(l_ns - r_ns)
                newest = max(l_ns, r_ns)
                cand = (dt, -newest, i, j)
                if best is None or cand < best:
                    best = cand

        if best is None:
            return
        dt_ns, _neg_newest, i, j = best
        if dt_ns > self._sync_tol_ns:
            # Drop oldest frame from the side that is lagging behind.
            if self._left_buf[0][0] < self._right_buf[0][0]:
                self._left_buf.popleft()
            else:
                self._right_buf.popleft()
            return

        l_ns, l_frame = self._left_buf[i]
        r_ns, r_frame = self._right_buf[j]

        # Remove consumed entries and anything older (to avoid latency buildup).
        for _ in range(i + 1):
            self._left_buf.popleft()
        for _ in range(j + 1):
            self._right_buf.popleft()

        self._left_frame = l_frame
        self._right_frame = r_frame
        self._left_stamp_ns = l_ns
        self._right_stamp_ns = r_ns

    def _image_to_bgr(self, msg: Image) -> Optional[np.ndarray]:
        """Convert sensor_msgs/Image to BGR OpenCV frame for common encodings."""
        try:
            if msg.height <= 0 or msg.width <= 0:
                return None

            data = np.frombuffer(msg.data, dtype=np.uint8)
            encoding = msg.encoding.lower()

            if encoding in ('rgb8',):
                frame = data.reshape((msg.height, msg.width, 3))
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if encoding in ('bgr8',):
                return data.reshape((msg.height, msg.width, 3))

            if encoding in ('mono8', '8uc1'):
                mono = data.reshape((msg.height, msg.width))
                return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

            if encoding in ('bgra8',):
                frame = data.reshape((msg.height, msg.width, 4))
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if encoding in ('rgba8',):
                frame = data.reshape((msg.height, msg.width, 4))
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            self.get_logger().warn(
                f'Unsupported image encoding "{msg.encoding}" on topic stream.'
            )
            return None
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f'Failed to decode image: {exc}')
            return None

    def _render(self):
        self._render_count += 1
        if self.n_frames > 1 and (self._render_count % self.n_frames) != 0:
            cv2.waitKey(1)
            return

        if self._left_frame is None and self._right_frame is None:
            cv2.waitKey(1)
            return

        if self.show_right and self._left_frame is not None and self._right_frame is not None:
            if self._left_frame.shape[:2] != self._right_frame.shape[:2]:
                right = cv2.resize(
                    self._right_frame,
                    (self._left_frame.shape[1], self._left_frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                right = self._right_frame
            frame = np.hstack((self._left_frame, right))
        elif self._left_frame is not None:
            frame = self._left_frame
        elif self._right_frame is not None:
            frame = self._right_frame
        else:
            cv2.waitKey(1)
            return

        if 0.0 < self.downsampling < 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=self.downsampling,
                fy=self.downsampling,
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamNode()

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
