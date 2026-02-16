"""Minimal stereo depth node with CUDA-first disparity and CPU fallback."""

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
import yaml


@dataclass
class FramePacket:
    """Queued frame metadata used for left/right pairing."""

    msg: Image
    pair_time_sec: float
    arrival_time_sec: float
    stamp_ns: int


class StereoDepthNode(Node):
    """SUBSCRIBE -> SYNC PAIR -> RECTIFY -> DISPARITY -> DEPTH(optional) -> PUBLISH."""

    def __init__(self):
        super().__init__('stereo_depth_node')

        self._declare_parameters()
        self._read_parameters()

        # Runtime state
        self._left_info: Optional[CameraInfo] = None
        self._right_info: Optional[CameraInfo] = None

        self._left_queue: Deque[FramePacket] = deque()
        self._right_queue: Deque[FramePacket] = deque()
        self._queue_lock = threading.Lock()
        self._queue_cv = threading.Condition(self._queue_lock)
        self._running = True

        self._use_arrival_pairing = False
        self._stamp_state = {
            'left': {'last_ns': None, 'repeat_count': 0},
            'right': {'last_ns': None, 'repeat_count': 0},
        }
        self._pair_counter = 0

        # Calibration / rectification
        self._rectify_ready = False
        self._map_l1 = None
        self._map_l2 = None
        self._map_r1 = None
        self._map_r2 = None
        self._rectified_size: Optional[Tuple[int, int]] = None
        self._calib_fx: Optional[float] = None
        self._calib_baseline_m: Optional[float] = None
        self._no_rectification_warned = False

        # Backend state
        self._backend = 'cpu'
        self._cuda_enabled = False
        self._cuda_validated = False
        self._cuda_matcher = None
        self._cuda_stream = None
        self._cuda_left = None
        self._cuda_right = None

        self._cpu_matcher = self._create_cpu_matcher()

        # One-time warning flags
        self._warned_no_intrinsics = False

        # Perf counters
        self._perf_lock = threading.Lock()
        self._perf_last_log_sec = time.monotonic()
        self._perf = {
            'incoming_left': 0,
            'incoming_right': 0,
            'paired': 0,
            'computed': 0,
            'dropped_left': 0,
            'dropped_right': 0,
            'dropped_overflow_left': 0,
            'dropped_overflow_right': 0,
            'dropped_stale_left': 0,
            'dropped_stale_right': 0,
            'queue_peak_left': 0,
            'queue_peak_right': 0,
            'decode_ms_sum': 0.0,
            'decode_count': 0,
            'remap_ms_sum': 0.0,
            'remap_count': 0,
            'disparity_ms_sum': 0.0,
            'disparity_count': 0,
            'depth_ms_sum': 0.0,
            'depth_count': 0,
            'publish_ms_sum': 0.0,
            'publish_count': 0,
            'loop_ms_sum': 0.0,
            'loop_count': 0,
            'pair_dt_sum': 0.0,
        }
        self._dropped_left_total = 0
        self._dropped_right_total = 0
        self._dropped_overflow_left_total = 0
        self._dropped_overflow_right_total = 0
        self._dropped_stale_left_total = 0
        self._dropped_stale_right_total = 0

        self._load_calibration()
        self._init_cuda_backend()

        # Publishers: depth/disparity outputs + optional preview
        self._depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self._disparity_pub = self.create_publisher(Image, self.disparity_topic, 10)
        self._preview_pub = self.create_publisher(Image, self.preview_topic, 10)

        # Subscribers
        self.create_subscription(Image, self.left_image_topic, self._left_image_cb, 10)
        self.create_subscription(Image, self.right_image_topic, self._right_image_cb, 10)
        self.create_subscription(CameraInfo, self.left_camera_info_topic, self._left_info_cb, 10)
        self.create_subscription(CameraInfo, self.right_camera_info_topic, self._right_info_cb, 10)

        # Worker thread for pair + compute to keep callbacks lightweight
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        if self.perf_log_hz > 0.0:
            self.create_timer(1.0 / self.perf_log_hz, self._log_perf)

        self.get_logger().info(
            f'stereo_depth_node ready: left={self.left_image_topic} right={self.right_image_topic} '
            f'output_mode={self.output_mode} depth_topic={self.depth_topic} '
            f'disparity_topic={self.disparity_topic} backend={self._backend}'
        )

    def _declare_parameters(self):
        # Inputs
        self.declare_parameter('left_image_topic', '/sim/raw/stereo/left/image_raw')
        self.declare_parameter('right_image_topic', '/sim/raw/stereo/right/image_raw')
        self.declare_parameter('left_camera_info_topic', '/sim/raw/stereo/left/camera_info')
        self.declare_parameter('right_camera_info_topic', '/sim/raw/stereo/right/camera_info')

        # Outputs
        self.declare_parameter('disparity_topic', '/sim/raw/stereo/disparity')
        self.declare_parameter('depth_topic', '/sim/raw/stereo/depth')
        self.declare_parameter('output_mode', 'depth')  # {'depth','disparity','both','none'}

        self.declare_parameter('publish_preview', False)
        self.declare_parameter('preview_topic', '/sim/raw/stereo/depth_preview')
        # Backward compatibility alias
        self.declare_parameter('depth_preview_topic', '/sim/raw/stereo/depth_preview')
        self.declare_parameter('preview_type', 'depth')  # {'disparity','depth'}
        self.declare_parameter('preview_scale', 0.5)

        # Calibration
        self.declare_parameter('calibration_file', '')

        # Pairing / scheduling
        self.declare_parameter('max_time_diff_sec', 0.03)
        self.declare_parameter('queue_size', 30)
        self.declare_parameter('disparity_sampling', 1)
        self.declare_parameter('perf_log_hz', 1.0)

        # Disparity backends
        self.declare_parameter('prefer_cuda', True)
        self.declare_parameter('require_cuda', False)
        self.declare_parameter('min_disparity', 0)
        self.declare_parameter('num_disparities', 192)
        self.declare_parameter('block_size', 7)
        self.declare_parameter('uniqueness_ratio', 10)
        self.declare_parameter('speckle_window_size', 100)
        self.declare_parameter('speckle_range', 2)
        self.declare_parameter('disp12_max_diff', 1)
        self.declare_parameter('pre_filter_cap', 31)

        # Depth
        self.declare_parameter('baseline_m', 0.12)
        self.declare_parameter('focal_length_px', 0.0)
        self.declare_parameter('disparity_valid_threshold', 0.1)
        self.declare_parameter('min_depth_m', 0.3)
        self.declare_parameter('max_depth_m', 30.0)

        # Legacy compatibility parameters (kept, ignored or lightly used)
        self.declare_parameter('left_rect_topic', '/sim/raw/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic', '/sim/raw/stereo/right/image_rect')
        self.declare_parameter('publish_rectified', False)
        self.declare_parameter('processing_rate_hz', 10.0)
        self.declare_parameter('compute_disparity', True)

    def _read_parameters(self):
        self.left_image_topic = str(self.get_parameter('left_image_topic').value)
        self.right_image_topic = str(self.get_parameter('right_image_topic').value)
        self.left_camera_info_topic = str(self.get_parameter('left_camera_info_topic').value)
        self.right_camera_info_topic = str(self.get_parameter('right_camera_info_topic').value)

        self.disparity_topic = str(self.get_parameter('disparity_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.output_mode = self._sanitize_mode(str(self.get_parameter('output_mode').value), 'depth')

        self.publish_preview = bool(self.get_parameter('publish_preview').value)
        preview_topic = str(self.get_parameter('preview_topic').value).strip()
        legacy_preview_topic = str(self.get_parameter('depth_preview_topic').value).strip()
        self.preview_topic = preview_topic or legacy_preview_topic
        self.preview_type = self._sanitize_mode(str(self.get_parameter('preview_type').value), 'depth')
        self.preview_scale = max(0.05, float(self.get_parameter('preview_scale').value))

        self.calibration_file = str(self.get_parameter('calibration_file').value)

        self.max_time_diff_sec = max(0.0, float(self.get_parameter('max_time_diff_sec').value))
        self.queue_size = max(5, int(self.get_parameter('queue_size').value))
        self.disparity_sampling = max(1, int(self.get_parameter('disparity_sampling').value))
        self.perf_log_hz = max(0.0, float(self.get_parameter('perf_log_hz').value))

        self.prefer_cuda = bool(self.get_parameter('prefer_cuda').value)
        self.require_cuda = bool(self.get_parameter('require_cuda').value)

        self.min_disparity = int(self.get_parameter('min_disparity').value)
        self.num_disparities = self._sanitize_num_disparities(int(self.get_parameter('num_disparities').value))
        self.block_size = self._sanitize_block_size(int(self.get_parameter('block_size').value), minimum=3)
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

        self.compute_disparity = bool(self.get_parameter('compute_disparity').value)

    @staticmethod
    def _sanitize_mode(value: str, default_value: str) -> str:
        value = value.strip().lower()
        if value in {'disparity', 'depth', 'both', 'none'}:
            return value
        return default_value

    @staticmethod
    def _sanitize_num_disparities(value: int) -> int:
        value = max(16, int(value))
        return (value // 16) * 16

    @staticmethod
    def _sanitize_block_size(value: int, minimum: int = 3) -> int:
        value = max(minimum, int(value))
        if value % 2 == 0:
            value += 1
        return value

    @staticmethod
    def _stamp_to_ns(header: Header) -> int:
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    @staticmethod
    def _stamp_to_sec_from_ns(stamp_ns: int) -> float:
        return float(stamp_ns) * 1e-9

    @staticmethod
    def _max_stamp(a, b):
        if (int(a.sec), int(a.nanosec)) >= (int(b.sec), int(b.nanosec)):
            return a
        return b

    def _common_header(self, left_header: Header, right_header: Header) -> Header:
        header = Header()
        header.frame_id = left_header.frame_id or right_header.frame_id
        header.stamp = self._max_stamp(left_header.stamp, right_header.stamp)
        return header

    def _default_calibration_path(self) -> str:
        try:
            share_dir = get_package_share_directory('sim_car')
            return os.path.join(share_dir, 'config', 'stereo_calibration.yaml')
        except Exception:  # pylint: disable=broad-except
            return ''

    def _load_calibration(self):
        calibration_path = self.calibration_file.strip() or self._default_calibration_path()
        if not calibration_path:
            self.get_logger().warn('No calibration file provided; rectification disabled.')
            return
        if not os.path.exists(calibration_path):
            self.get_logger().warn(
                f'Calibration file not found at {calibration_path}; rectification disabled.'
            )
            return

        try:
            with open(calibration_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file) or {}
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

            p1, p2 = self._build_maps(
                k_left=k_left,
                d_left=d_left,
                k_right=k_right,
                d_right=d_right,
                r_stereo=r_stereo,
                t_stereo=t_stereo,
                image_size=image_size,
            )

            self._calib_fx = float(p1[0, 0])
            if abs(float(p2[0, 0])) > 1e-9:
                self._calib_baseline_m = abs(float(p2[0, 3]) / float(p2[0, 0]))
            else:
                self._calib_baseline_m = abs(float(t_stereo[0, 0]))

            self.get_logger().info(
                f'Loaded calibration: {calibration_path} '
                f'fx={self._calib_fx:.3f}px baseline={self._calib_baseline_m:.4f}m '
                f'size={width}x{height}'
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(
                f'Failed to load calibration from {calibration_path}: {exc}. Rectification disabled.'
            )

    def _build_maps(self, k_left, d_left, k_right, d_right, r_stereo, t_stereo, image_size):
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
            k_left,
            d_left,
            r1,
            p1,
            image_size,
            cv2.CV_16SC2,
        )
        self._map_r1, self._map_r2 = cv2.initUndistortRectifyMap(
            k_right,
            d_right,
            r2,
            p2,
            image_size,
            cv2.CV_16SC2,
        )

        self._rectified_size = image_size
        self._rectify_ready = True
        return p1, p2

    @staticmethod
    def _matrix_from_cfg(node, shape):
        arr = np.array(node['data'], dtype=np.float64)
        if len(shape) == 2:
            rows, cols = shape
            return arr.reshape((rows, cols))
        return arr.reshape((-1,))

    def _create_cpu_matcher(self):
        p1 = 8 * (self.block_size ** 2)
        p2 = 32 * (self.block_size ** 2)
        return cv2.StereoSGBM_create(
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

    def _init_cuda_backend(self):
        if not self.prefer_cuda and not self.require_cuda:
            return

        try:
            has_cuda_mod = hasattr(cv2, 'cuda')
            has_cuda_bm = has_cuda_mod and hasattr(cv2.cuda, 'createStereoBM')
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if has_cuda_mod else 0

            if not has_cuda_mod or not has_cuda_bm or cuda_count <= 0:
                msg = (
                    'CUDA disparity unavailable '
                    f'(has_cuda_mod={has_cuda_mod}, has_cuda_bm={has_cuda_bm}, devices={cuda_count}).'
                )
                if self.require_cuda:
                    raise RuntimeError(msg)
                self.get_logger().warn(msg + ' Falling back to CPU StereoSGBM.')
                return

            cuda_block_size = self._sanitize_block_size(self.block_size, minimum=5)
            self._cuda_stream = cv2.cuda.Stream()
            self._cuda_left = cv2.cuda_GpuMat()
            self._cuda_right = cv2.cuda_GpuMat()
            self._cuda_matcher = cv2.cuda.createStereoBM(
                numDisparities=int(self.num_disparities),
                blockSize=int(cuda_block_size),
            )

            self._cuda_enabled = True
            self._backend = 'cuda'
            self.get_logger().info(
                f'CUDA disparity backend enabled: StereoBM(num_disparities={self.num_disparities}, '
                f'block_size={cuda_block_size}, devices={cuda_count})'
            )
        except Exception as exc:  # pylint: disable=broad-except
            if self.require_cuda:
                raise
            self._cuda_enabled = False
            self._backend = 'cpu'
            self.get_logger().warn(f'CUDA init failed ({exc}); using CPU StereoSGBM.')

    def _left_info_cb(self, msg: CameraInfo):
        self._left_info = msg

    def _right_info_cb(self, msg: CameraInfo):
        self._right_info = msg

    def _left_image_cb(self, msg: Image):
        self._enqueue_frame(msg, side='left')

    def _right_image_cb(self, msg: Image):
        self._enqueue_frame(msg, side='right')

    def _enqueue_frame(self, msg: Image, side: str):
        arrival_sec = time.monotonic()
        stamp_ns = self._stamp_to_ns(msg.header)
        mode_switched = False

        with self._queue_cv:
            if self._should_use_arrival_locked(side, stamp_ns) and not self._use_arrival_pairing:
                self._use_arrival_pairing = True
                self._left_queue.clear()
                self._right_queue.clear()
                mode_switched = True

            pair_time_sec = (
                arrival_sec if self._use_arrival_pairing else self._stamp_to_sec_from_ns(stamp_ns)
            )
            packet = FramePacket(
                msg=msg,
                pair_time_sec=pair_time_sec,
                arrival_time_sec=arrival_sec,
                stamp_ns=stamp_ns,
            )

            queue = self._left_queue if side == 'left' else self._right_queue
            queue.append(packet)
            self._update_queue_peak(side, len(queue))
            self._update_incoming_metrics(side)

            dropped = 0
            while len(queue) > self.queue_size:
                queue.popleft()
                dropped += 1
            if dropped:
                self._count_dropped(side, dropped, reason='overflow')

            self._queue_cv.notify()

        if mode_switched:
            self.get_logger().warn(
                'Detected constant/zero image timestamps; using time.monotonic() arrival time for pairing.'
            )

    def _should_use_arrival_locked(self, side: str, stamp_ns: int) -> bool:
        state = self._stamp_state[side]

        if stamp_ns <= 0:
            state['last_ns'] = stamp_ns
            state['repeat_count'] = 0
            return True

        last_ns = state['last_ns']
        if last_ns is None:
            state['last_ns'] = stamp_ns
            state['repeat_count'] = 0
            return False

        if stamp_ns == last_ns:
            state['repeat_count'] += 1
        else:
            state['repeat_count'] = 0

        state['last_ns'] = stamp_ns
        return state['repeat_count'] >= 3

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

            left_packet, right_packet, pair_dt = pair
            self._pair_counter += 1

            if self.disparity_sampling > 1 and (self._pair_counter % self.disparity_sampling) != 0:
                continue
            if not self.compute_disparity:
                continue

            self._process_pair(left_packet, right_packet, pair_dt)

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
                self._count_pair(best_dt)
                return left_packet, right_packet, best_dt

            left_t = left_head.pair_time_sec
            right_oldest_t = self._right_queue[0].pair_time_sec
            right_newest_t = self._right_queue[-1].pair_time_sec

            if right_newest_t < left_t - self.max_time_diff_sec:
                self._right_queue.popleft()
                self._count_dropped('right', 1, reason='stale')
                continue
            if left_t < right_oldest_t - self.max_time_diff_sec:
                self._left_queue.popleft()
                self._count_dropped('left', 1, reason='stale')
                continue

            if left_t <= right_oldest_t:
                self._left_queue.popleft()
                self._count_dropped('left', 1, reason='stale')
            else:
                self._right_queue.popleft()
                self._count_dropped('right', 1, reason='stale')

        return None

    def _process_pair(self, left_packet: FramePacket, right_packet: FramePacket, _pair_dt: float):
        loop_t0 = time.perf_counter()
        header = self._common_header(left_packet.msg.header, right_packet.msg.header)

        t0 = time.perf_counter()
        left_gray = self._decode_to_gray(left_packet.msg)
        right_gray = self._decode_to_gray(right_packet.msg)
        decode_ms = (time.perf_counter() - t0) * 1000.0
        self._add_stage_time('decode', decode_ms, count=2)

        if left_gray is None or right_gray is None:
            return

        t1 = time.perf_counter()
        left_rect, right_rect = self._rectify_gray(left_gray, right_gray)
        remap_ms = (time.perf_counter() - t1) * 1000.0
        self._add_stage_time('remap', remap_ms)

        t2 = time.perf_counter()
        disparity = self._compute_disparity_cuda(left_rect, right_rect)
        if disparity is None:
            disparity = self._compute_disparity_cpu(left_rect, right_rect)
        disparity_ms = (time.perf_counter() - t2) * 1000.0
        self._add_stage_time('disparity', disparity_ms)

        need_depth = self.output_mode in {'depth', 'both'} or (
            self.publish_preview and self.preview_type == 'depth'
        )
        depth = None
        depth_ms = 0.0

        if need_depth:
            t3 = time.perf_counter()
            depth = self._compute_depth(disparity)
            depth_ms = (time.perf_counter() - t3) * 1000.0
            self._add_stage_time('depth', depth_ms)

        t4 = time.perf_counter()
        self._publish_output(header, disparity, depth)
        publish_ms = (time.perf_counter() - t4) * 1000.0
        self._add_stage_time('publish', publish_ms)
        loop_ms = (time.perf_counter() - loop_t0) * 1000.0
        self._add_stage_time('loop', loop_ms)
        self._count_compute()

    def _rectify_gray(self, left_gray: np.ndarray, right_gray: np.ndarray):
        if not self._rectify_ready:
            if not self._no_rectification_warned:
                self.get_logger().warn('Rectification maps unavailable; using unrectified grayscale frames.')
                self._no_rectification_warned = True
            return left_gray, right_gray

        target_w, target_h = self._rectified_size
        if left_gray.shape[:2] != (target_h, target_w):
            left_gray = cv2.resize(left_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if right_gray.shape[:2] != (target_h, target_w):
            right_gray = cv2.resize(right_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        left_rect = cv2.remap(left_gray, self._map_l1, self._map_l2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, self._map_r1, self._map_r2, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def _compute_disparity_cuda(self, left_gray: np.ndarray, right_gray: np.ndarray) -> Optional[np.ndarray]:
        if not self._cuda_enabled:
            return None

        try:
            self._cuda_left.upload(left_gray, stream=self._cuda_stream)
            self._cuda_right.upload(right_gray, stream=self._cuda_stream)
            disparity_gpu = self._cuda_matcher.compute(self._cuda_left, self._cuda_right, self._cuda_stream)
            self._cuda_stream.waitForCompletion()
            disparity_raw = disparity_gpu.download()
            disparity = self._disparity_to_float(disparity_raw)
            if not self._cuda_validated:
                self._cuda_validated = True
                self.get_logger().info('CUDA disparity compute validated; using CUDA backend.')
            return disparity
        except Exception as exc:  # pylint: disable=broad-except
            self._cuda_enabled = False
            self._backend = 'cpu'
            self.get_logger().warn(f'CUDA disparity compute failed ({exc}); switching to CPU StereoSGBM.')
            return None

    def _compute_disparity_cpu(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        disparity_raw = self._cpu_matcher.compute(left_gray, right_gray)
        return self._disparity_to_float(disparity_raw)

    @staticmethod
    def _disparity_to_float(disparity_raw: np.ndarray) -> np.ndarray:
        if disparity_raw.dtype in (np.int16, np.int32, np.uint16, np.uint32):
            return disparity_raw.astype(np.float32) / 16.0
        return disparity_raw.astype(np.float32)

    def _compute_depth(self, disparity: np.ndarray) -> Optional[np.ndarray]:
        fx, baseline = self._resolve_fx_baseline()
        if fx <= 0.0 or baseline <= 0.0:
            if not self._warned_no_intrinsics:
                self.get_logger().warn(
                    'Cannot compute depth: missing valid fx/baseline '
                    '(check calibration, focal_length_px, baseline_m, or CameraInfo).'
                )
                self._warned_no_intrinsics = True
            return None

        valid = disparity > self.disparity_valid_threshold
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        depth[valid] = (fx * baseline) / disparity[valid]

        finite = np.isfinite(depth)
        out_of_range = (depth < self.min_depth_m) | (depth > self.max_depth_m)
        depth[finite & out_of_range] = np.nan
        return depth

    def _resolve_fx_baseline(self) -> Tuple[float, float]:
        fx = self._calib_fx if self._calib_fx is not None else self.focal_length_px
        baseline = self._calib_baseline_m if self._calib_baseline_m is not None else self.baseline_m

        if fx <= 0.0 and self._left_info is not None:
            fx = float(self._left_info.k[0]) if len(self._left_info.k) >= 1 else fx
            if fx <= 0.0 and len(self._left_info.p) >= 1:
                fx = float(self._left_info.p[0])

        if baseline <= 0.0 and self._right_info is not None and len(self._right_info.p) >= 4:
            right_fx = float(self._right_info.p[0])
            tx = float(self._right_info.p[3])
            if abs(right_fx) > 1e-9:
                baseline = abs(tx / right_fx)

        if baseline <= 0.0 and self._left_info is not None and self._right_info is not None:
            if len(self._left_info.p) >= 4 and len(self._right_info.p) >= 4:
                left_fx = float(self._left_info.p[0])
                right_fx = float(self._right_info.p[0])
                if abs(left_fx) > 1e-9 and abs(right_fx) > 1e-9:
                    tx_left = float(self._left_info.p[3]) / left_fx
                    tx_right = float(self._right_info.p[3]) / right_fx
                    baseline = abs(tx_right - tx_left)

        return float(fx), float(baseline)

    def _publish_output(self, header: Header, disparity: np.ndarray, depth: Optional[np.ndarray]):
        if self.output_mode in {'disparity', 'both'}:
            self._disparity_pub.publish(self._make_float_image_msg(header, disparity))

        if self.output_mode in {'depth', 'both'}:
            if depth is None:
                if self.output_mode == 'depth':
                    return
            else:
                self._depth_pub.publish(self._make_float_image_msg(header, depth))

        if not self.publish_preview:
            return

        if self.preview_type == 'depth':
            preview_src = depth
            if preview_src is None:
                return
            preview = self._build_preview(preview_src, mode='depth')
        else:
            preview = self._build_preview(disparity, mode='disparity')

        self._preview_pub.publish(self._make_mono8_image_msg(header, preview))

    def _build_preview(self, image: np.ndarray, mode: str) -> np.ndarray:
        preview = np.zeros(image.shape, dtype=np.uint8)

        if mode == 'depth':
            valid = np.isfinite(image)
            if np.any(valid):
                clipped = np.clip(image[valid], self.min_depth_m, self.max_depth_m)
                denom = max(1e-6, self.max_depth_m - self.min_depth_m)
                normalized = (clipped - self.min_depth_m) / denom
                preview[valid] = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            valid = image > self.disparity_valid_threshold
            if np.any(valid):
                clipped = np.clip(image[valid], 0.0, float(self.num_disparities))
                denom = max(1e-6, float(self.num_disparities))
                normalized = clipped / denom
                preview[valid] = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)

        if abs(self.preview_scale - 1.0) > 1e-6:
            out_w = max(1, int(preview.shape[1] * self.preview_scale))
            out_h = max(1, int(preview.shape[0] * self.preview_scale))
            preview = cv2.resize(preview, (out_w, out_h), interpolation=cv2.INTER_AREA)

        return preview

    def _decode_to_gray(self, msg: Image) -> Optional[np.ndarray]:
        try:
            if msg.height <= 0 or msg.width <= 0:
                return None

            encoding = msg.encoding.lower()
            if encoding in {'mono8', '8uc1'}:
                return self._reshape_mono8(msg)
            if encoding == 'rgb8':
                rgb = self._reshape_color8(msg, channels=3)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if encoding == 'bgr8':
                bgr = self._reshape_color8(msg, channels=3)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if encoding == 'rgba8':
                rgba = self._reshape_color8(msg, channels=4)
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
            if encoding == 'bgra8':
                bgra = self._reshape_color8(msg, channels=4)
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)

            self.get_logger().warn(f'Unsupported image encoding for stereo depth: {msg.encoding}')
            return None
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f'Failed to decode image ({msg.encoding}): {exc}')
            return None

    @staticmethod
    def _reshape_mono8(msg: Image) -> np.ndarray:
        row_bytes = int(msg.step)
        needed = row_bytes * int(msg.height)
        data = np.frombuffer(msg.data, dtype=np.uint8, count=needed)
        rows = data.reshape((msg.height, row_bytes))
        return rows[:, : msg.width].copy()

    @staticmethod
    def _reshape_color8(msg: Image, channels: int) -> np.ndarray:
        row_bytes = int(msg.step)
        needed = row_bytes * int(msg.height)
        data = np.frombuffer(msg.data, dtype=np.uint8, count=needed)
        rows = data.reshape((msg.height, row_bytes))
        usable = rows[:, : msg.width * channels]
        return usable.reshape((msg.height, msg.width, channels)).copy()

    @staticmethod
    def _make_float_image_msg(header: Header, image: np.ndarray) -> Image:
        msg = Image()
        msg.header = header
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = '32FC1'
        msg.is_bigendian = False
        msg.step = int(image.shape[1] * 4)
        msg.data = image.astype(np.float32).tobytes()
        return msg

    @staticmethod
    def _make_mono8_image_msg(header: Header, image: np.ndarray) -> Image:
        msg = Image()
        msg.header = header
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = 'mono8'
        msg.is_bigendian = False
        msg.step = int(image.shape[1])
        msg.data = image.astype(np.uint8).tobytes()
        return msg

    def _update_incoming_metrics(self, side: str):
        key = 'incoming_left' if side == 'left' else 'incoming_right'
        with self._perf_lock:
            self._perf[key] += 1

    def _update_queue_peak(self, side: str, queue_len: int):
        key = 'queue_peak_left' if side == 'left' else 'queue_peak_right'
        with self._perf_lock:
            self._perf[key] = max(int(self._perf[key]), int(queue_len))

    def _count_pair(self, pair_dt_sec: float):
        with self._perf_lock:
            self._perf['paired'] += 1
            self._perf['pair_dt_sum'] += float(pair_dt_sec)

    def _count_compute(self):
        with self._perf_lock:
            self._perf['computed'] += 1

    def _count_dropped(self, side: str, count: int, reason: str):
        key = 'dropped_left' if side == 'left' else 'dropped_right'
        reason = reason.lower().strip()
        with self._perf_lock:
            self._perf[key] += int(count)
            if reason == 'overflow':
                reason_key = 'dropped_overflow_left' if side == 'left' else 'dropped_overflow_right'
                self._perf[reason_key] += int(count)
            else:
                reason_key = 'dropped_stale_left' if side == 'left' else 'dropped_stale_right'
                self._perf[reason_key] += int(count)

            if side == 'left':
                self._dropped_left_total += int(count)
                if reason == 'overflow':
                    self._dropped_overflow_left_total += int(count)
                else:
                    self._dropped_stale_left_total += int(count)
            else:
                self._dropped_right_total += int(count)
                if reason == 'overflow':
                    self._dropped_overflow_right_total += int(count)
                else:
                    self._dropped_stale_right_total += int(count)

    def _add_stage_time(self, stage: str, elapsed_ms: float, count: int = 1):
        with self._perf_lock:
            if stage == 'decode':
                self._perf['decode_ms_sum'] += elapsed_ms
                self._perf['decode_count'] += int(count)
            elif stage == 'remap':
                self._perf['remap_ms_sum'] += elapsed_ms
                self._perf['remap_count'] += int(count)
            elif stage == 'disparity':
                self._perf['disparity_ms_sum'] += elapsed_ms
                self._perf['disparity_count'] += int(count)
            elif stage == 'depth':
                self._perf['depth_ms_sum'] += elapsed_ms
                self._perf['depth_count'] += int(count)
            elif stage == 'publish':
                self._perf['publish_ms_sum'] += elapsed_ms
                self._perf['publish_count'] += int(count)
            elif stage == 'loop':
                self._perf['loop_ms_sum'] += elapsed_ms
                self._perf['loop_count'] += int(count)

    def _log_perf(self):
        now = time.monotonic()
        with self._perf_lock:
            elapsed = max(1e-6, now - self._perf_last_log_sec)
            snapshot = dict(self._perf)
            dropped_left_total = self._dropped_left_total
            dropped_right_total = self._dropped_right_total
            dropped_overflow_left_total = self._dropped_overflow_left_total
            dropped_overflow_right_total = self._dropped_overflow_right_total
            dropped_stale_left_total = self._dropped_stale_left_total
            dropped_stale_right_total = self._dropped_stale_right_total

            # reset interval counters
            for key in self._perf:
                self._perf[key] = 0 if isinstance(self._perf[key], int) else 0.0
            self._perf_last_log_sec = now

        in_left_hz = snapshot['incoming_left'] / elapsed
        in_right_hz = snapshot['incoming_right'] / elapsed
        paired_hz = snapshot['paired'] / elapsed
        compute_hz = snapshot['computed'] / elapsed

        decode_avg = snapshot['decode_ms_sum'] / max(1, snapshot['decode_count'])
        remap_avg = snapshot['remap_ms_sum'] / max(1, snapshot['remap_count'])
        disparity_avg = snapshot['disparity_ms_sum'] / max(1, snapshot['disparity_count'])
        depth_avg = snapshot['depth_ms_sum'] / max(1, snapshot['depth_count'])
        publish_avg = snapshot['publish_ms_sum'] / max(1, snapshot['publish_count'])
        loop_avg = snapshot['loop_ms_sum'] / max(1, snapshot['loop_count'])
        pair_dt_avg = snapshot['pair_dt_sum'] / max(1, snapshot['paired'])
        queue_peak_left = int(snapshot['queue_peak_left'])
        queue_peak_right = int(snapshot['queue_peak_right'])

        self.get_logger().info(
            'perf '
            f'in_hz L/R={in_left_hz:.2f}/{in_right_hz:.2f} '
            f'paired_hz={paired_hz:.2f} compute_hz={compute_hz:.2f} '
            f'avg_ms decode={decode_avg:.2f} remap={remap_avg:.2f} '
            f'disparity={disparity_avg:.2f} depth={depth_avg:.2f} '
            f'publish={publish_avg:.2f} loop={loop_avg:.2f} '
            f'pair_dt={pair_dt_avg:.4f}s '
            f'dropped L/R={snapshot["dropped_left"]}/{snapshot["dropped_right"]} '
            f'overflow L/R={snapshot["dropped_overflow_left"]}/{snapshot["dropped_overflow_right"]} '
            f'stale L/R={snapshot["dropped_stale_left"]}/{snapshot["dropped_stale_right"]} '
            f'q_peak L/R={queue_peak_left}/{queue_peak_right} '
            f'(total {dropped_left_total}/{dropped_right_total}, '
            f'overflow_total {dropped_overflow_left_total}/{dropped_overflow_right_total}, '
            f'stale_total {dropped_stale_left_total}/{dropped_stale_right_total}) '
            f'backend={self._backend}'
        )

    def destroy_node(self):
        with self._queue_cv:
            self._running = False
            self._queue_cv.notify_all()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        return super().destroy_node()


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
