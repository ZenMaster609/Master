"""
Simple local camera stream viewer for sim stereo feeds.

Subscribes to left/right sensor_msgs/Image topics and displays them in OpenCV.
"""

from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraStreamNode(Node):
    """Display left/right camera feeds in a local OpenCV window."""

    def __init__(self):
        super().__init__('camera_stream_node')

        self.declare_parameter('left_topic', '/sim/raw/stereo/left/image_raw')
        self.declare_parameter('right_topic', '/sim/raw/stereo/right/image_raw')
        self.declare_parameter('window_name', 'Sim Stereo Stream')
        self.declare_parameter('show_right', True)
        self.declare_parameter('display_scale', 0.5)

        left_topic = str(self.get_parameter('left_topic').value)
        right_topic = str(self.get_parameter('right_topic').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.show_right = bool(self.get_parameter('show_right').value)
        self.display_scale = float(self.get_parameter('display_scale').value)

        self._left_frame: Optional[np.ndarray] = None
        self._right_frame: Optional[np.ndarray] = None

        self._left_sub = self.create_subscription(
            Image, left_topic, self._left_callback, 10
        )
        self._right_sub = self.create_subscription(
            Image, right_topic, self._right_callback, 10
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
            self._left_frame = frame
            self._render()

    def _right_callback(self, msg: Image):
        frame = self._image_to_bgr(msg)
        if frame is not None:
            self._right_frame = frame
            self._render()

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

        if 0.0 < self.display_scale < 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
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
