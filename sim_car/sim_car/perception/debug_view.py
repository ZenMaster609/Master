"""Debug image publishing for disparity/depth/rectified-left inspection."""

import cv2
from sensor_msgs.msg import Image
import numpy as np


class CameraDebugPublisher:
    """Publishes a debug image from disparity, depth, or rectified-left every N frames."""

    VALID_MODES = {'none', 'disparity', 'depth', 'left_rect'}
    BOX_HALF_SIZE_PX = 14

    def __init__(
        self,
        node,
        mode: str,
        topic: str,
        publish_every_n: int,
        min_depth_m: float,
        max_depth_m: float,
        max_disparity: float,
        disparity_valid_threshold: float,
    ):
        self._node = node
        self._mode = self._sanitize_mode(mode)
        self._topic = str(topic)
        self._publish_every_n = max(1, int(publish_every_n))
        self._min_depth_m = float(min_depth_m)
        self._max_depth_m = float(max_depth_m)
        self._max_disparity = max(1.0, float(max_disparity))
        self._disparity_valid_threshold = float(disparity_valid_threshold)
        self._counter = 0

        self._publisher = None
        if self._mode != 'none':
            self._publisher = node.create_publisher(Image, self._topic, 10)
            self._node.get_logger().info(
                f'camera_debug enabled: mode={self._mode} topic={self._topic} '
                f'every_n={self._publish_every_n}'
            )

    @property
    def enabled(self) -> bool:
        return self._publisher is not None

    @classmethod
    def _sanitize_mode(cls, mode: str) -> str:
        value = str(mode).strip().lower()
        if value == 'rect_left':
            value = 'left_rect'
        if value in cls.VALID_MODES:
            return value
        return 'none'

    def maybe_publish(
        self,
        header,
        disparity: np.ndarray,
        depth: np.ndarray,
        left_rect: np.ndarray | None = None,
        cone_overlays: list[dict] | None = None,
    ):
        if self._publisher is None:
            return

        self._counter += 1
        if (self._counter % self._publish_every_n) != 0:
            return

        if self._mode == 'depth':
            image = self._depth_to_mono8(depth)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self._draw_overlays(image, cone_overlays)
            encoding = 'bgr8'
        elif self._mode == 'left_rect':
            if left_rect is None:
                return
            image = self._left_rect_to_bgr8(left_rect)
            self._draw_overlays(image, cone_overlays)
            encoding = 'bgr8'
        else:
            image = self._disparity_to_mono8(disparity)
            encoding = 'mono8'

        msg = Image()
        msg.header = header
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = False
        channels = 1 if image.ndim == 2 else int(image.shape[2])
        msg.step = int(image.shape[1] * channels)
        msg.data = image.tobytes()
        self._publisher.publish(msg)

    def _disparity_to_mono8(self, disparity: np.ndarray) -> np.ndarray:
        image = np.zeros(disparity.shape, dtype=np.uint8)
        valid = disparity > self._disparity_valid_threshold
        if not np.any(valid):
            return image

        clipped = np.clip(disparity[valid], 0.0, self._max_disparity)
        normalized = clipped / self._max_disparity
        image[valid] = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
        return image

    def _depth_to_mono8(self, depth: np.ndarray) -> np.ndarray:
        image = np.zeros(depth.shape, dtype=np.uint8)
        valid = np.isfinite(depth)
        if not np.any(valid):
            return image

        clipped = np.clip(depth[valid], self._min_depth_m, self._max_depth_m)
        denom = max(1e-6, self._max_depth_m - self._min_depth_m)
        normalized = (clipped - self._min_depth_m) / denom
        # Invert so closer objects are brighter.
        image[valid] = np.clip((1.0 - normalized) * 255.0, 0.0, 255.0).astype(np.uint8)
        return image

    @staticmethod
    def _left_rect_to_bgr8(left_rect: np.ndarray) -> np.ndarray:
        if left_rect.ndim == 2:
            return cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR)
        if left_rect.ndim == 3 and left_rect.shape[2] == 3:
            return left_rect.copy()
        if left_rect.ndim == 3 and left_rect.shape[2] == 1:
            return cv2.cvtColor(left_rect[:, :, 0], cv2.COLOR_GRAY2BGR)
        return np.zeros((left_rect.shape[0], left_rect.shape[1], 3), dtype=np.uint8)

    def _draw_overlays(self, image: np.ndarray, cone_overlays: list[dict] | None):
        if not cone_overlays:
            return
        h, w = image.shape[:2]
        for overlay in cone_overlays:
            u = int(round(float(overlay.get('u', -1))))
            v = int(round(float(overlay.get('v', -1))))
            if u < 0 or v < 0 or u >= w or v >= h:
                continue
            color = self._color_bgr(str(overlay.get('color', 'unknown')))
            label = str(overlay.get('label', '')).strip()

            x0 = max(0, u - self.BOX_HALF_SIZE_PX)
            y0 = max(0, v - self.BOX_HALF_SIZE_PX)
            x1 = min(w - 1, u + self.BOX_HALF_SIZE_PX)
            y1 = min(h - 1, v + self.BOX_HALF_SIZE_PX)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

            if label:
                lines = [ln for ln in label.split('\n') if ln]
                if not lines:
                    continue
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.40
                thickness = 1

                text_sizes = [cv2.getTextSize(ln, font, scale, thickness)[0] for ln in lines]
                max_line_w = max(size[0] for size in text_sizes)
                line_h = max(size[1] for size in text_sizes) + 4
                placement = str(overlay.get('placement', 'right')).strip().lower()
                if placement == 'left':
                    text_x = x0 - max_line_w - 6
                    text_y = y0 + line_h
                elif placement == 'top':
                    text_x = u - (max_line_w // 2)
                    text_y = y0 - 4 - ((len(lines) - 1) * line_h)
                elif placement == 'bottom':
                    text_x = u - (max_line_w // 2)
                    text_y = y1 + line_h + 2
                else:  # right
                    text_x = x1 + 4
                    text_y = y0 + line_h

                text_x = max(1, min(w - max_line_w - 1, text_x))
                max_first_baseline = h - 2 - ((len(lines) - 1) * line_h)
                if max_first_baseline < 12:
                    text_y = 12
                else:
                    text_y = max(12, min(max_first_baseline, text_y))

                for i, ln in enumerate(lines):
                    y = min(h - 2, text_y + (i * line_h))
                    cv2.putText(
                        image,
                        ln,
                        (text_x, y),
                        font,
                        scale,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )

    @staticmethod
    def _color_bgr(color: str) -> tuple[int, int, int]:
        if color == 'blue':
            return (255, 0, 0)
        if color == 'yellow':
            return (0, 255, 255)
        if color == 'orange' or color == 'big_orange':
            return (0, 165, 255)
        return (255, 255, 255)
