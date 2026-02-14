"""
Stereo depth estimation node for sim stereo cameras.

Subscribes to left/right image streams, computes disparity with StereoSGBM,
and publishes disparity/depth images for downstream perception (e.g. YOLO + 3D projection).
"""

from collections import deque
import os
from typing import Optional

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
import yaml


class StereoDepthNode(Node):
    """Compute stereo disparity/depth from left/right image topics."""

    def __init__(self):
        super().__init__('stereo_depth_node')

        # Inputs
        self.declare_parameter('left_image_topic', '/sim/raw/stereo/left/image_raw')
        self.declare_parameter('right_image_topic', '/sim/raw/stereo/right/image_raw')
        self.declare_parameter('left_camera_info_topic', '/sim/raw/stereo/left/camera_info')
        self.declare_parameter('right_camera_info_topic', '/sim/raw/stereo/right/camera_info')

        # Outputs
        self.declare_parameter('left_rect_topic', '/sim/raw/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic', '/sim/raw/stereo/right/image_rect')
        self.declare_parameter('disparity_topic', '/sim/raw/stereo/disparity')
        self.declare_parameter('depth_topic', '/sim/raw/stereo/depth')
        self.declare_parameter('depth_preview_topic', '/sim/raw/stereo/depth_preview')
        self.declare_parameter('publish_preview', True)
        self.declare_parameter('publish_rectified', True)
        self.declare_parameter('calibration_file', '')

        # Stereo matching parameters
        self.declare_parameter('max_time_diff_sec', 0.03)
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('min_disparity', 0)
        self.declare_parameter('num_disparities', 192)  # must be divisible by 16
        self.declare_parameter('block_size', 7)  # odd number
        self.declare_parameter('uniqueness_ratio', 10)
        self.declare_parameter('speckle_window_size', 100)
        self.declare_parameter('speckle_range', 2)
        self.declare_parameter('disp12_max_diff', 1)
        self.declare_parameter('pre_filter_cap', 31)
        self.declare_parameter('baseline_m', 0.12)  # used if calibration yaml is unavailable
        self.declare_parameter('focal_length_px', 0.0)  # used if calibration yaml is unavailable
        self.declare_parameter('min_depth_m', 0.3)
        self.declare_parameter('max_depth_m', 30.0)

        # Read parameters
        left_topic = str(self.get_parameter('left_image_topic').value)
        right_topic = str(self.get_parameter('right_image_topic').value)
        left_info_topic = str(self.get_parameter('left_camera_info_topic').value)
        right_info_topic = str(self.get_parameter('right_camera_info_topic').value)

        left_rect_topic = str(self.get_parameter('left_rect_topic').value)
        right_rect_topic = str(self.get_parameter('right_rect_topic').value)
        disparity_topic = str(self.get_parameter('disparity_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        depth_preview_topic = str(self.get_parameter('depth_preview_topic').value)
        self.publish_preview = bool(self.get_parameter('publish_preview').value)
        self.publish_rectified = bool(self.get_parameter('publish_rectified').value)
        self.calibration_file = str(self.get_parameter('calibration_file').value)

        self.max_time_diff_sec = float(self.get_parameter('max_time_diff_sec').value)
        self.queue_size = max(2, int(self.get_parameter('queue_size').value))

        self.min_disparity = int(self.get_parameter('min_disparity').value)
        self.num_disparities = self._sanitize_num_disparities(
            int(self.get_parameter('num_disparities').value)
        )
        self.block_size = self._sanitize_block_size(int(self.get_parameter('block_size').value))
        self.uniqueness_ratio = int(self.get_parameter('uniqueness_ratio').value)
        self.speckle_window_size = int(self.get_parameter('speckle_window_size').value)
        self.speckle_range = int(self.get_parameter('speckle_range').value)
        self.disp12_max_diff = int(self.get_parameter('disp12_max_diff').value)
        self.pre_filter_cap = int(self.get_parameter('pre_filter_cap').value)

        self.baseline_m = float(self.get_parameter('baseline_m').value)
        self.focal_length_px = float(self.get_parameter('focal_length_px').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)

        self._left_info: Optional[CameraInfo] = None
        self._right_info: Optional[CameraInfo] = None
        self._left_queue = deque(maxlen=self.queue_size)
        self._right_queue = deque(maxlen=self.queue_size)
        self._no_focal_warned = False
        self._warned_no_rectification = False

        # Calibration / rectification state
        self._rectify_ready = False
        self._calib_fx: Optional[float] = None
        self._calib_baseline_m: Optional[float] = None
        self._map_l1 = None
        self._map_l2 = None
        self._map_r1 = None
        self._map_r2 = None

        # Stereo matcher
        p1 = 8 * 1 * (self.block_size ** 2)
        p2 = 32 * 1 * (self.block_size ** 2)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=self.disp12_max_diff,
            preFilterCap=self.pre_filter_cap,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self._load_calibration_and_build_maps()

        # Subscribers
        self.create_subscription(Image, left_topic, self._left_image_cb, 10)
        self.create_subscription(Image, right_topic, self._right_image_cb, 10)
        self.create_subscription(CameraInfo, left_info_topic, self._left_info_cb, 10)
        self.create_subscription(CameraInfo, right_info_topic, self._right_info_cb, 10)

        # Publishers
        self.left_rect_pub = self.create_publisher(Image, left_rect_topic, 10)
        self.right_rect_pub = self.create_publisher(Image, right_rect_topic, 10)
        self.disparity_pub = self.create_publisher(Image, disparity_topic, 10)
        self.depth_pub = self.create_publisher(Image, depth_topic, 10)
        self.depth_preview_pub = self.create_publisher(Image, depth_preview_topic, 10)

        self.get_logger().info(
            f'stereo_depth_node ready. left={left_topic}, right={right_topic}, '
            f'depth={depth_topic}, disparity={disparity_topic}'
        )

    def _default_calibration_path(self) -> str:
        try:
            share = get_package_share_directory('sim_car')
            return os.path.join(share, 'config', 'stereo_calibration.yaml')
        except Exception:  # pylint: disable=broad-except
            return ''

    def _load_calibration_and_build_maps(self):
        calibration_path = self.calibration_file.strip() or self._default_calibration_path()
        if not calibration_path:
            self.get_logger().warn('No stereo calibration path available; rectification disabled.')
            return
        if not os.path.exists(calibration_path):
            self.get_logger().warn(
                f'Stereo calibration file not found at {calibration_path}; rectification disabled.'
            )
            return

        try:
            with open(calibration_path, 'r', encoding='utf-8') as calib_file:
                content = yaml.safe_load(calib_file) or {}
            cfg = content.get('stereo_calibration', {})

            width = int(cfg['image_width'])
            height = int(cfg['image_height'])
            image_size = (width, height)

            k_left = self._matrix_from_cfg(cfg['left']['camera_matrix'], (3, 3))
            d_left = self._matrix_from_cfg(cfg['left']['distortion_coefficients'], (-1,))
            k_right = self._matrix_from_cfg(cfg['right']['camera_matrix'], (3, 3))
            d_right = self._matrix_from_cfg(cfg['right']['distortion_coefficients'], (-1,))

            r_stereo = self._matrix_from_cfg(cfg['stereo']['rotation_matrix'], (3, 3))
            t_stereo = self._matrix_from_cfg(cfg['stereo']['translation'], (3, 1))

            r1, r2, p1, p2, _q, _, _ = cv2.stereoRectify(
                k_left,
                d_left,
                k_right,
                d_right,
                image_size,
                r_stereo,
                t_stereo,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0.0,
            )

            self._map_l1, self._map_l2 = cv2.initUndistortRectifyMap(
                k_left, d_left, r1, p1, image_size, cv2.CV_32FC1
            )
            self._map_r1, self._map_r2 = cv2.initUndistortRectifyMap(
                k_right, d_right, r2, p2, image_size, cv2.CV_32FC1
            )

            self._calib_fx = float(p1[0, 0])
            if abs(float(p2[0, 0])) > 1e-9:
                self._calib_baseline_m = abs(float(p2[0, 3]) / float(p2[0, 0]))
            else:
                self._calib_baseline_m = abs(float(t_stereo[0, 0]))

            self._rectify_ready = True
            self.get_logger().info(
                f'Loaded stereo calibration from {calibration_path}. '
                f'fx={self._calib_fx:.3f}px baseline={self._calib_baseline_m:.4f}m '
                f'size={width}x{height}'
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(
                f'Failed to load stereo calibration from {calibration_path}: {exc}. '
                'Rectification disabled.'
            )

    def _matrix_from_cfg(self, node, shape):
        arr = np.array(node['data'], dtype=np.float64)
        if len(shape) == 2:
            rows, cols = shape
            return arr.reshape((rows, cols))
        return arr.reshape((-1,))

    def _sanitize_num_disparities(self, value: int) -> int:
        if value < 16:
            return 16
        if value % 16 != 0:
            return (value // 16) * 16
        return value

    def _sanitize_block_size(self, value: int) -> int:
        if value < 3:
            value = 3
        if value % 2 == 0:
            value += 1
        return value

    def _left_info_cb(self, msg: CameraInfo):
        self._left_info = msg

    def _right_info_cb(self, msg: CameraInfo):
        self._right_info = msg

    def _left_image_cb(self, msg: Image):
        self._left_queue.append(msg)
        self._try_process()

    def _right_image_cb(self, msg: Image):
        self._right_queue.append(msg)
        self._try_process()

    def _try_process(self):
        if not self._left_queue or not self._right_queue:
            return

        left_msg = self._left_queue[0]
        left_t = self._stamp_to_sec(left_msg)

        best_idx = -1
        best_dt = None
        for idx, right_msg in enumerate(self._right_queue):
            dt = abs(left_t - self._stamp_to_sec(right_msg))
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_idx = idx

        if best_dt is None:
            return

        # If right queue is too old, drop left and continue.
        if best_dt > self.max_time_diff_sec:
            if self._stamp_to_sec(self._right_queue[-1]) < left_t:
                self._left_queue.popleft()
            return

        right_msg = self._right_queue[best_idx]
        self._left_queue.popleft()
        del self._right_queue[best_idx]
        self._process_pair(left_msg, right_msg)

    def _process_pair(self, left_msg: Image, right_msg: Image):
        left_color = self._image_to_bgr(left_msg)
        right_color = self._image_to_bgr(right_msg)
        if left_color is None or right_color is None:
            return

        if left_color.shape[:2] != right_color.shape[:2]:
            right_color = cv2.resize(
                right_color,
                (left_color.shape[1], left_color.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        if self._rectify_ready:
            left_rect = cv2.remap(left_color, self._map_l1, self._map_l2, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_color, self._map_r1, self._map_r2, cv2.INTER_LINEAR)
        else:
            left_rect = left_color
            right_rect = right_color
            if not self._warned_no_rectification:
                self.get_logger().warn(
                    'Stereo rectification maps are not ready; using raw images for disparity.'
                )
                self._warned_no_rectification = True

        if self.publish_rectified:
            self.left_rect_pub.publish(
                self._make_uint8_bgr_msg(header=left_msg.header, image=left_rect)
            )
            self.right_rect_pub.publish(
                self._make_uint8_bgr_msg(header=right_msg.header, image=right_rect)
            )

        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        disparity_msg = self._make_float_image_msg(
            header=left_msg.header,
            image=disparity,
            encoding='32FC1',
        )
        self.disparity_pub.publish(disparity_msg)

        focal_px = self._calib_fx if self._calib_fx is not None else self.focal_length_px
        baseline_m = self._calib_baseline_m if self._calib_baseline_m is not None else self.baseline_m
        if focal_px <= 0.0 and self._left_info is not None:
            focal_px = float(self._left_info.k[0])

        if focal_px <= 0.0:
            if not self._no_focal_warned:
                self.get_logger().warn(
                    'No valid focal length available yet (set focal_length_px or wait for CameraInfo).'
                )
                self._no_focal_warned = True
            return

        valid = disparity > 0.1
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        depth[valid] = (focal_px * baseline_m) / disparity[valid]

        finite = np.isfinite(depth)
        depth[(finite & (depth < self.min_depth_m)) | (finite & (depth > self.max_depth_m))] = np.nan

        depth_msg = self._make_float_image_msg(
            header=left_msg.header,
            image=depth,
            encoding='32FC1',
        )
        self.depth_pub.publish(depth_msg)

        if self.publish_preview:
            preview = self._depth_preview(depth)
            preview_msg = self._make_uint8_image_msg(
                header=left_msg.header,
                image=preview,
                encoding='mono8',
            )
            self.depth_preview_pub.publish(preview_msg)

    def _depth_preview(self, depth: np.ndarray) -> np.ndarray:
        out = np.zeros(depth.shape, dtype=np.uint8)
        finite = np.isfinite(depth)
        if not np.any(finite):
            return out

        d = np.clip(depth[finite], self.min_depth_m, self.max_depth_m)
        norm = (d - self.min_depth_m) / max(1e-6, (self.max_depth_m - self.min_depth_m))
        out_vals = (255.0 * (1.0 - norm)).astype(np.uint8)  # nearer = brighter
        out[finite] = out_vals
        return out

    def _image_to_bgr(self, msg: Image) -> Optional[np.ndarray]:
        try:
            if msg.height <= 0 or msg.width <= 0:
                return None

            data = np.frombuffer(msg.data, dtype=np.uint8)
            enc = msg.encoding.lower()

            if enc == 'mono8' or enc == '8uc1':
                gray = data.reshape((msg.height, msg.width))
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if enc == 'rgb8':
                rgb = data.reshape((msg.height, msg.width, 3))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if enc == 'bgr8':
                return data.reshape((msg.height, msg.width, 3))

            if enc == 'bgra8':
                bgra = data.reshape((msg.height, msg.width, 4))
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

            if enc == 'rgba8':
                rgba = data.reshape((msg.height, msg.width, 4))
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

            self.get_logger().warn(f'Unsupported encoding for stereo depth image: {msg.encoding}')
            return None
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f'Failed to decode image for stereo depth: {exc}')
            return None

    def _make_float_image_msg(self, header, image: np.ndarray, encoding: str) -> Image:
        msg = Image()
        msg.header = header
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = int(image.shape[1] * 4)
        msg.data = image.astype(np.float32).tobytes()
        return msg

    def _make_uint8_image_msg(self, header, image: np.ndarray, encoding: str) -> Image:
        msg = Image()
        msg.header = header
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = int(image.shape[1])
        msg.data = image.astype(np.uint8).tobytes()
        return msg

    def _make_uint8_bgr_msg(self, header, image: np.ndarray) -> Image:
        msg = Image()
        msg.header = header
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = int(image.shape[1] * 3)
        msg.data = image.astype(np.uint8).tobytes()
        return msg

    def _stamp_to_sec(self, msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()

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
