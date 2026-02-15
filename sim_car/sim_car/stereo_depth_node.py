"""
Stereo depth estimation node for sim stereo cameras.

Subscribes to left/right image streams, rectifies with known calibration, computes
disparity/depth, and publishes results for downstream perception.
"""

import os
import threading
from collections import deque
from typing import Optional

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
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
        # Rectified image publishing rate (independent of disparity/depth compute).
        self.declare_parameter('rectify_rate_hz', 15.0)
        # Disparity/depth sampling in rectified-frame units:
        # 1 = compute every rectified pair, 5 = compute every 5th rectified pair, etc.
        self.declare_parameter('disparity_sampling', 1)
        # Backward compat: previously used timer-based disparity compute.
        self.declare_parameter('processing_rate_hz', 10.0)
        self.declare_parameter('compute_disparity', True)
        self.declare_parameter('prefer_cuda', True)
        self.declare_parameter('require_cuda', False)
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
        self.rectify_rate_hz = float(self.get_parameter('rectify_rate_hz').value)
        self.disparity_sampling = max(1, int(self.get_parameter('disparity_sampling').value))
        self.processing_rate_hz = float(self.get_parameter('processing_rate_hz').value)
        self.compute_disparity = bool(self.get_parameter('compute_disparity').value)
        self.prefer_cuda = bool(self.get_parameter('prefer_cuda').value)
        self.require_cuda = bool(self.get_parameter('require_cuda').value)

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
        self._latest_left: Optional[Image] = None
        self._latest_right: Optional[Image] = None
        # Buffers used to pair left/right frames for rectification and disparity.
        # Keep small to avoid latency.
        self._left_buf = deque(maxlen=20)   # entries: (seq, msg, t_sec)
        self._right_buf = deque(maxlen=20)  # entries: (seq, msg, t_sec)
        # Some sim image sources publish constant header stamps (e.g. 0), so we
        # cannot rely on stamp-based de-duplication. Use monotonic counters.
        self._left_seq: int = 0
        self._right_seq: int = 0
        self._last_processed_left_seq: int = -1
        self._last_processed_right_seq: int = -1
        self._last_rect_left_seq: int = -1
        self._last_rect_right_seq: int = -1
        self._last_rect_dt_sec: Optional[float] = None
        self._last_rect_header: Optional[Header] = None
        self._last_left_gray: Optional[np.ndarray] = None
        self._last_right_gray: Optional[np.ndarray] = None
        self._no_focal_warned = False
        self._warned_no_rectification = False
        self._use_cuda_disparity = False
        self._cuda_stereo = None
        self._cuda_stream = None
        self._cuda_left = None
        self._cuda_right = None
        self._cuda_candidates = []
        self._cuda_candidate_idx = 0
        self._cuda_validated = False
        self._compute_inflight = False
        self._compute_lock = threading.Lock()
        self._rect_pair_count = 0

        # Calibration / rectification state
        self._rectify_ready = False
        self._calib_fx: Optional[float] = None
        self._calib_baseline_m: Optional[float] = None
        self._rectified_size = None  # (width, height) - always full-res (no pipeline downsampling)
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
        self._init_cuda_backend()

        self._load_calibration_and_build_maps()

        # Subscribers
        # Keep subscription depth low to avoid stale-frame latency under heavy load.
        self.create_subscription(Image, left_topic, self._left_image_cb, 1)
        self.create_subscription(Image, right_topic, self._right_image_cb, 1)
        self.create_subscription(CameraInfo, left_info_topic, self._left_info_cb, 1)
        self.create_subscription(CameraInfo, right_info_topic, self._right_info_cb, 1)

        if self.publish_rectified and self.rectify_rate_hz > 0.0:
            self._rectify_timer = self.create_timer(
                1.0 / max(1.0, self.rectify_rate_hz),
                self._publish_rectified_latest,
            )
        else:
            self._rectify_timer = None

        # Publishers
        self.left_rect_pub = self.create_publisher(Image, left_rect_topic, 10)
        self.right_rect_pub = self.create_publisher(Image, right_rect_topic, 10)
        self.disparity_pub = self.create_publisher(Image, disparity_topic, 10)
        self.depth_pub = self.create_publisher(Image, depth_topic, 10)
        self.depth_preview_pub = self.create_publisher(Image, depth_preview_topic, 10)

        self.get_logger().info(
            f'stereo_depth_node ready. left={left_topic}, right={right_topic}, '
            f'depth={depth_topic}, disparity={disparity_topic}, '
            f'backend={"cuda" if self._use_cuda_disparity else "cpu"}'
        )

    @staticmethod
    def _stamp_to_sec_header(header: Header) -> float:
        return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9

    @staticmethod
    def _max_stamp(a, b):
        if (int(a.sec), int(a.nanosec)) >= (int(b.sec), int(b.nanosec)):
            return a
        return b

    def _common_header(self, left_msg: Image, right_msg: Image) -> Header:
        hdr = Header()
        hdr.frame_id = left_msg.header.frame_id or right_msg.header.frame_id
        hdr.stamp = self._max_stamp(left_msg.header.stamp, right_msg.header.stamp)
        return hdr

    def _init_cuda_backend(self):
        if not self.prefer_cuda and not self.require_cuda:
            return
        try:
            has_cuda_mod = hasattr(cv2, 'cuda')
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if has_cuda_mod else 0
            has_bm = has_cuda_mod and hasattr(cv2.cuda, 'createStereoBM')
            if cuda_count > 0 and has_bm:
                # CUDA StereoBM path. Some OpenCV Python builds require a Stream arg and can
                # still fail at runtime depending on image size / params; we validate lazily.
                self._cuda_stream = cv2.cuda.Stream()
                self._cuda_left = cv2.cuda_GpuMat()
                self._cuda_right = cv2.cuda_GpuMat()
                self._cuda_candidates = self._build_cuda_candidates(
                    self.num_disparities, self.block_size
                )
                self._cuda_candidate_idx = 0
                self._cuda_stereo = None
                self._cuda_validated = False
                self._use_cuda_disparity = True
                self.get_logger().info(
                    f'OpenCV CUDA detected (devices={cuda_count}); enabling CUDA StereoBM backend (lazy-validated).'
                )
            else:
                msg = (
                    'CUDA disparity backend unavailable in this OpenCV build '
                    f'(cuda_devices={cuda_count}, has_cuda_bm={has_bm}).'
                )
                if self.require_cuda:
                    raise RuntimeError(msg)
                self.get_logger().warn(msg + ' Falling back to CPU SGBM.')
        except Exception as exc:  # pylint: disable=broad-except
            if self.require_cuda:
                raise
            self.get_logger().warn(f'CUDA backend probe failed: {exc}. Falling back to CPU SGBM.')

    @staticmethod
    def _build_cuda_candidates(num_disparities: int, block_size: int):
        def _sanitize_nd(v: int) -> int:
            v = max(16, int(v))
            return (v // 16) * 16

        def _sanitize_bs(v: int) -> int:
            v = max(5, int(v))
            if v % 2 == 0:
                v += 1
            return v

        nd0 = _sanitize_nd(num_disparities)
        bs0 = _sanitize_bs(block_size)
        candidates = [
            (nd0, bs0),
            (min(nd0, 128), bs0),
            (min(nd0, 128), 9),
            (64, 9),
        ]
        out = []
        seen = set()
        for nd, bs in candidates:
            nd = _sanitize_nd(nd)
            bs = _sanitize_bs(bs)
            if (nd, bs) in seen:
                continue
            seen.add((nd, bs))
            out.append((nd, bs))
        return out

    def _ensure_cuda_matcher(self) -> bool:
        if not self._use_cuda_disparity:
            return False
        if self._cuda_stream is None or self._cuda_left is None or self._cuda_right is None:
            return False

        if self._cuda_stereo is not None:
            return True

        if self._cuda_candidate_idx >= len(self._cuda_candidates):
            self._use_cuda_disparity = False
            return False

        nd, bs = self._cuda_candidates[self._cuda_candidate_idx]
        self._cuda_candidate_idx += 1
        try:
            self._cuda_stereo = cv2.cuda.createStereoBM(numDisparities=int(nd), blockSize=int(bs))
            self._cuda_validated = False
            self.get_logger().info(f'CUDA StereoBM configured: num_disparities={nd} block_size={bs}')
            return True
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f'Failed to configure CUDA StereoBM (nd={nd}, bs={bs}): {exc}')
            self._cuda_stereo = None
            return self._ensure_cuda_matcher()

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
            self._rectified_size = image_size

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
                k_left, d_left, r1, p1, self._rectified_size, cv2.CV_32FC1
            )
            self._map_r1, self._map_r2 = cv2.initUndistortRectifyMap(
                k_right, d_right, r2, p2, self._rectified_size, cv2.CV_32FC1
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
                f'size={width}x{height} rectified_out={self._rectified_size[0]}x{self._rectified_size[1]}'
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
        self._latest_left = msg
        self._left_seq += 1
        self._left_buf.append((self._left_seq, msg, self._stamp_to_sec(msg)))

    def _right_image_cb(self, msg: Image):
        self._latest_right = msg
        self._right_seq += 1
        self._right_buf.append((self._right_seq, msg, self._stamp_to_sec(msg)))

    def _select_synced_pair(self):
        """Pick and remove a time-synced (left,right) pair from buffers."""
        tol = float(self.max_time_diff_sec)
        while self._left_buf and self._right_buf:
            left_seq, left_msg, left_t = self._left_buf[0]

            # Find closest right frame in time to this left.
            best_idx = -1
            best_dt = None
            for idx, (_rseq, _rmsg, r_t) in enumerate(self._right_buf):
                dt = abs(left_t - r_t)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_idx = idx

            if best_dt is None:
                return None
            if best_dt <= tol:
                right_seq, right_msg, _right_t = self._right_buf[best_idx]
                self._left_buf.popleft()
                del self._right_buf[best_idx]
                return left_seq, left_msg, right_seq, right_msg, float(best_dt)

            # No good match yet: drop stale frames to converge.
            right_oldest_t = self._right_buf[0][2]
            right_newest_t = self._right_buf[-1][2]
            if right_newest_t < left_t:
                # Rights are behind; drop oldest right.
                self._right_buf.popleft()
                continue
            if left_t < right_oldest_t:
                # Left is behind; drop oldest left.
                self._left_buf.popleft()
                continue

            # Overlapping but not within tolerance; drop the older side.
            if left_t <= right_oldest_t:
                self._left_buf.popleft()
            else:
                self._right_buf.popleft()

        return None

    def _publish_rectified_latest(self):
        pair = self._select_synced_pair()
        if pair is None:
            return
        left_seq, left_msg, right_seq, right_msg, dt = pair
        self._publish_rectified_if_new(left_msg, right_msg, left_seq, right_seq, dt_sec=dt)

    def _compute_snapshot_worker(
        self,
        header: Header,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        left_seq: int,
        right_seq: int,
    ):
        try:
            self._compute_and_publish_from_rectified(header, left_gray, right_gray)
            self._last_processed_left_seq = left_seq
            self._last_processed_right_seq = right_seq
        finally:
            with self._compute_lock:
                self._compute_inflight = False

    def _publish_rectified_if_new(
        self,
        left_msg: Image,
        right_msg: Image,
        left_seq: int,
        right_seq: int,
        dt_sec: Optional[float] = None,
    ):
        if self._last_rect_left_seq == left_seq and self._last_rect_right_seq == right_seq:
            return

        left_color = self._image_to_bgr(left_msg)
        right_color = self._image_to_bgr(right_msg)
        if left_color is None or right_color is None:
            return

        # Keep sizes consistent for side-by-side viewing.
        if left_color.shape[:2] != right_color.shape[:2]:
            right_color = cv2.resize(
                right_color,
                (left_color.shape[1], left_color.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        header = self._common_header(left_msg, right_msg)
        self._last_rect_dt_sec = float(dt_sec) if dt_sec is not None else abs(
            self._stamp_to_sec(left_msg) - self._stamp_to_sec(right_msg)
        )

        if self._rectify_ready:
            left_rect = cv2.remap(left_color, self._map_l1, self._map_l2, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_color, self._map_r1, self._map_r2, cv2.INTER_LINEAR)
        else:
            left_rect = left_color
            right_rect = right_color

        # Publish rectified frames at the rectify timer rate regardless of whether disparity is computed.
        self.left_rect_pub.publish(self._make_uint8_bgr_msg(header=header, image=left_rect))
        self.right_rect_pub.publish(self._make_uint8_bgr_msg(header=header, image=right_rect))

        # Cache grayscale rectified frames for disparity/depth compute.
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        self._last_left_gray = left_gray
        self._last_right_gray = right_gray
        self._last_rect_header = header
        self._last_rect_left_seq = left_seq
        self._last_rect_right_seq = right_seq
        self._rect_pair_count += 1

        # Trigger disparity/depth compute based on rectified-pair sampling.
        if not self.compute_disparity:
            return
        if self.disparity_sampling > 1 and (self._rect_pair_count % self.disparity_sampling) != 0:
            return
        if self._last_processed_left_seq == left_seq and self._last_processed_right_seq == right_seq:
            return

        with self._compute_lock:
            if self._compute_inflight:
                return
            self._compute_inflight = True

        # Snapshot to ensure compute uses the same rectified pair as the publish.
        hdr = Header()
        hdr.frame_id = header.frame_id
        hdr.stamp = header.stamp
        t = threading.Thread(
            target=self._compute_snapshot_worker,
            # Arrays are immutable after remap; safe to pass without copying.
            args=(hdr, left_gray, right_gray, left_seq, right_seq),
            daemon=True,
        )
        t.start()

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

        # If a dedicated rectification timer is running, it handles `/image_rect`.
        if self.publish_rectified and self._rectify_timer is None:
            self.left_rect_pub.publish(
                self._make_uint8_bgr_msg(header=left_msg.header, image=left_rect)
            )
            self.right_rect_pub.publish(
                self._make_uint8_bgr_msg(header=right_msg.header, image=right_rect)
            )
        # Disparity/depth computation is done from the rectification timer cache.

    def _compute_and_publish_from_rectified(self, header: Header, left_gray: np.ndarray, right_gray: np.ndarray):
        if self._use_cuda_disparity and self._ensure_cuda_matcher():
            try:
                # Upload into persistent GPU mats to avoid per-frame allocations.
                self._cuda_left.upload(left_gray, stream=self._cuda_stream)
                self._cuda_right.upload(right_gray, stream=self._cuda_stream)

                disparity_gpu = self._cuda_stereo.compute(self._cuda_left, self._cuda_right, self._cuda_stream)
                self._cuda_stream.waitForCompletion()
                disparity = disparity_gpu.download().astype(np.float32) / 16.0
                if not self._cuda_validated:
                    self._cuda_validated = True
                    self.get_logger().info('CUDA StereoBM disparity path validated.')
            except Exception as exc:  # pylint: disable=broad-except
                # Never let CUDA backend kill the node. Try next config; if none works, fall back to CPU.
                self.get_logger().warn(f'CUDA disparity failed; trying fallback. Error: {exc}')
                self._cuda_stereo = None
                self._cuda_validated = False
                if not self._ensure_cuda_matcher():
                    self._use_cuda_disparity = False
                disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        else:
            disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        self.disparity_pub.publish(
            self._make_float_image_msg(header=header, image=disparity, encoding='32FC1')
        )

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

        self.depth_pub.publish(self._make_float_image_msg(header=header, image=depth, encoding='32FC1'))

        if self.publish_preview:
            preview = self._depth_preview(depth)
            self.depth_preview_pub.publish(
                self._make_uint8_image_msg(header=header, image=preview, encoding='mono8')
            )

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

    def _stamp_to_ns(self, msg: Image) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


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
