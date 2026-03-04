"""Low-overhead debug image publisher for annotated camera frames."""

from sensor_msgs.msg import Image


class CameraDebugPublisher:
    """Publishes a provided debug image every N frames."""

    def __init__(
        self,
        node,
        enabled: bool,
        topic: str,
        publish_every_n: int,
    ):
        self._node = node
        self._topic = str(topic)
        self._publish_every_n = max(1, int(publish_every_n))
        self._counter = 0

        self._publisher = None
        if enabled:
            self._publisher = node.create_publisher(Image, self._topic, 10)
            self._node.get_logger().info(
                f'camera_debug enabled: topic={self._topic} every_n={self._publish_every_n}'
            )

    @property
    def enabled(self) -> bool:
        return self._publisher is not None

    def should_publish(self) -> bool:
        if self._publisher is None:
            return False
        self._counter += 1
        return (self._counter % self._publish_every_n) == 0

    def publish_image(self, header, image, encoding: str) -> None:
        if self._publisher is None or image is None:
            return

        msg = Image()
        msg.header = header
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = str(encoding)
        msg.is_bigendian = False
        channels = 1 if image.ndim == 2 else int(image.shape[2])
        msg.step = int(image.shape[1] * channels)
        msg.data = image.tobytes()
        self._publisher.publish(msg)
