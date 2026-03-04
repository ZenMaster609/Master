"""Stereo processing pipeline: decode, rectify, disparity, and depth."""

from dataclasses import dataclass
import os
import time
from typing import Dict, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
import yaml


@dataclass(frozen=True)
class StereoPipelineConfig:
    """Configuration for stereo processing and depth conversion."""

    calibration_file: str
    prefer_cuda: bool
    min_disparity: int
    num_disparities: int
    block_size: int
    uniqueness_ratio: int
    speckle_window_size: int
    speckle_range: int
    disp12_max_diff: int
    pre_filter_cap: int
    baseline_m: float
    focal_length_px: float
    disparity_valid_threshold: float
    min_depth_m: float
    max_depth_m: float


@dataclass
class StereoPipelineOutput:
    """Result from one stereo pair processing pass."""

    left_rect: np.ndarray
    left_rect_color: Optional[np.ndarray]
    right_rect: np.ndarray
    disparity: np.ndarray
    depth: np.ndarray
    timings_ms: Dict[str, float]
    backend: str


class StereoPipeline:
    """Processes a left/right pair into rectified disparity and depth arrays."""

    def __init__(self, logger, config: StereoPipelineConfig):
        self._logger = logger
        self._cfg = config

        self._num_disparities = self._sanitize_num_disparities(config.num_disparities)
        self._block_size = self._sanitize_block_size(config.block_size)

        self._cpu_matcher = self._create_cpu_matcher()

        self._backend = 'cpu'
        self._cuda_enabled = False
        self._cuda_validated = False
        self._cuda_matcher = None
        self._cuda_stream = None
        self._cuda_left = None
        self._cuda_right = None

        self._rectify_ready = False
        self._map_l1 = None
        self._map_l2 = None
        self._map_r1 = None
        self._map_r2 = None
        self._rectified_size: Optional[Tuple[int, int]] = None
        self._rectified_p1 = None
        self._rectify_r1 = None
        self._calib_fx: Optional[float] = None
        self._calib_baseline_m: Optional[float] = None
        self._warned_no_rectification = False
        self._warned_no_intrinsics = False

        self._load_calibration()
        self._init_cuda_backend()

    @property
    def backend(self) -> str:
        """Current disparity backend name."""
        return self._backend

    def build_rectified_left_camera_info(self, left_info: Optional[CameraInfo]) -> Optional[CameraInfo]:
        """Return left CameraInfo updated to match the rectified image geometry."""
        if left_info is None or not self._rectify_ready or self._rectified_p1 is None or self._rectified_size is None:
            return left_info

        rectified = CameraInfo()
        rectified.header = left_info.header
        rectified.height = int(self._rectified_size[1])
        rectified.width = int(self._rectified_size[0])
        rectified.distortion_model = left_info.distortion_model
        rectified.d = [0.0] * len(left_info.d)
        rectified.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        rectified.k = [
            float(self._rectified_p1[0, 0]), 0.0, float(self._rectified_p1[0, 2]),
            0.0, float(self._rectified_p1[1, 1]), float(self._rectified_p1[1, 2]),
            0.0, 0.0, 1.0,
        ]
        rectified.p = [
            float(self._rectified_p1[0, 0]), 0.0, float(self._rectified_p1[0, 2]), float(self._rectified_p1[0, 3]),
            0.0, float(self._rectified_p1[1, 1]), float(self._rectified_p1[1, 2]), float(self._rectified_p1[1, 3]),
            0.0, 0.0, 1.0, 0.0,
        ]
        rectified.binning_x = left_info.binning_x
        rectified.binning_y = left_info.binning_y
        rectified.roi = left_info.roi
        return rectified

    def left_rectification_rotation(self) -> Optional[np.ndarray]:
        """Return the left-camera rectification rotation matrix, if available."""
        if self._rectify_r1 is None:
            return None
        return self._rectify_r1.copy()

    def process(
        self,
        left_msg: Image,
        right_msg: Image,
        left_info: Optional[CameraInfo],
        right_info: Optional[CameraInfo],
    ) -> Optional[StereoPipelineOutput]:
        """Run decode + rectify + disparity + depth for one pair."""
        total_t0 = time.perf_counter()

        decode_t0 = time.perf_counter()
        left_bgr = self._decode_to_bgr(left_msg)
        left_gray = self._decode_to_gray(left_msg)
        right_gray = self._decode_to_gray(right_msg)
        decode_ms = (time.perf_counter() - decode_t0) * 1000.0
        if left_gray is None or right_gray is None:
            return None

        rectify_t0 = time.perf_counter()
        left_rect, right_rect = self._rectify_gray(left_gray, right_gray)
        left_rect_color = self._rectify_color(left_bgr) if left_bgr is not None else None
        rectify_ms = (time.perf_counter() - rectify_t0) * 1000.0

        disparity_t0 = time.perf_counter()
        disparity = self._compute_disparity_cuda(left_rect, right_rect)
        if disparity is None:
            disparity = self._compute_disparity_cpu(left_rect, right_rect)
        disparity_ms = (time.perf_counter() - disparity_t0) * 1000.0

        depth_t0 = time.perf_counter()
        depth = self._compute_depth(disparity, left_info, right_info)
        depth_ms = (time.perf_counter() - depth_t0) * 1000.0

        total_ms = (time.perf_counter() - total_t0) * 1000.0
        return StereoPipelineOutput(
            left_rect=left_rect,
            left_rect_color=left_rect_color,
            right_rect=right_rect,
            disparity=disparity,
            depth=depth,
            timings_ms={
                'decode': decode_ms,
                'rectify': rectify_ms,
                'disparity': disparity_ms,
                'depth': depth_ms,
                'total': total_ms,
            },
            backend=self._backend,
        )

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

    def _create_cpu_matcher(self):
        p1 = 8 * (self._block_size ** 2)
        p2 = 32 * (self._block_size ** 2)
        return cv2.StereoSGBM_create(
            minDisparity=int(self._cfg.min_disparity),
            numDisparities=int(self._num_disparities),
            blockSize=int(self._block_size),
            P1=p1,
            P2=p2,
            disp12MaxDiff=int(self._cfg.disp12_max_diff),
            preFilterCap=int(self._cfg.pre_filter_cap),
            uniquenessRatio=int(self._cfg.uniqueness_ratio),
            speckleWindowSize=int(self._cfg.speckle_window_size),
            speckleRange=int(self._cfg.speckle_range),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def _init_cuda_backend(self):
        if not self._cfg.prefer_cuda:
            return
        try:
            has_cuda_mod = hasattr(cv2, 'cuda')
            has_cuda_bm = has_cuda_mod and hasattr(cv2.cuda, 'createStereoBM')
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if has_cuda_mod else 0
            if not has_cuda_mod or not has_cuda_bm or cuda_count <= 0:
                self._logger.warn(
                    'CUDA disparity unavailable; using CPU StereoSGBM '
                    f'(has_cuda_mod={has_cuda_mod}, has_cuda_bm={has_cuda_bm}, devices={cuda_count})'
                )
                return

            cuda_block_size = self._sanitize_block_size(self._cfg.block_size, minimum=5)
            self._cuda_stream = cv2.cuda.Stream()
            self._cuda_left = cv2.cuda_GpuMat()
            self._cuda_right = cv2.cuda_GpuMat()
            self._cuda_matcher = cv2.cuda.createStereoBM(
                numDisparities=int(self._num_disparities),
                blockSize=int(cuda_block_size),
            )
            self._cuda_enabled = True
            self._backend = 'cuda'
            self._logger.info(
                'CUDA disparity backend enabled: '
                f'StereoBM(num_disparities={self._num_disparities}, block_size={cuda_block_size})'
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warn(f'CUDA init failed ({exc}); using CPU StereoSGBM.')
            self._cuda_enabled = False
            self._backend = 'cpu'

    def _compute_disparity_cuda(self, left_gray: np.ndarray, right_gray: np.ndarray) -> Optional[np.ndarray]:
        if not self._cuda_enabled:
            return None
        try:
            self._cuda_left.upload(left_gray, stream=self._cuda_stream)
            self._cuda_right.upload(right_gray, stream=self._cuda_stream)
            disparity_gpu = self._cuda_matcher.compute(self._cuda_left, self._cuda_right, self._cuda_stream)
            self._cuda_stream.waitForCompletion()
            disparity = self._disparity_to_float(disparity_gpu.download())
            if not self._cuda_validated:
                self._cuda_validated = True
                self._logger.info('CUDA disparity compute validated; using CUDA backend.')
            return disparity
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warn(f'CUDA disparity compute failed ({exc}); switching to CPU StereoSGBM.')
            self._cuda_enabled = False
            self._backend = 'cpu'
            return None

    def _compute_disparity_cpu(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        disparity_raw = self._cpu_matcher.compute(left_gray, right_gray)
        return self._disparity_to_float(disparity_raw)

    @staticmethod
    def _disparity_to_float(disparity_raw: np.ndarray) -> np.ndarray:
        if disparity_raw.dtype in (np.int16, np.int32, np.uint16, np.uint32):
            return disparity_raw.astype(np.float32) / 16.0
        return disparity_raw.astype(np.float32)

    def _compute_depth(
        self,
        disparity: np.ndarray,
        left_info: Optional[CameraInfo],
        right_info: Optional[CameraInfo],
    ) -> np.ndarray:
        fx, baseline = self._resolve_fx_baseline(left_info, right_info)
        if fx <= 0.0 or baseline <= 0.0:
            if not self._warned_no_intrinsics:
                self._logger.warn(
                    'Depth intrinsics unavailable; emitting NaN depth '
                    '(check calibration, focal_length_px, baseline_m, or CameraInfo).'
                )
                self._warned_no_intrinsics = True
            return np.full(disparity.shape, np.nan, dtype=np.float32)

        valid = disparity > float(self._cfg.disparity_valid_threshold)
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        depth[valid] = (fx * baseline) / disparity[valid]

        finite = np.isfinite(depth)
        out_of_range = (depth < float(self._cfg.min_depth_m)) | (depth > float(self._cfg.max_depth_m))
        depth[finite & out_of_range] = np.nan
        return depth

    def _resolve_fx_baseline(
        self,
        left_info: Optional[CameraInfo],
        right_info: Optional[CameraInfo],
    ) -> Tuple[float, float]:
        fx = self._calib_fx if self._calib_fx is not None else float(self._cfg.focal_length_px)
        baseline = (
            self._calib_baseline_m
            if self._calib_baseline_m is not None
            else float(self._cfg.baseline_m)
        )

        if fx <= 0.0 and left_info is not None:
            if len(left_info.k) >= 1:
                fx = float(left_info.k[0])
            if fx <= 0.0 and len(left_info.p) >= 1:
                fx = float(left_info.p[0])

        if baseline <= 0.0 and right_info is not None and len(right_info.p) >= 4:
            right_fx = float(right_info.p[0])
            tx = float(right_info.p[3])
            if abs(right_fx) > 1e-9:
                baseline = abs(tx / right_fx)

        if baseline <= 0.0 and left_info is not None and right_info is not None:
            if len(left_info.p) >= 4 and len(right_info.p) >= 4:
                left_fx = float(left_info.p[0])
                right_fx = float(right_info.p[0])
                if abs(left_fx) > 1e-9 and abs(right_fx) > 1e-9:
                    tx_left = float(left_info.p[3]) / left_fx
                    tx_right = float(right_info.p[3]) / right_fx
                    baseline = abs(tx_right - tx_left)

        return float(fx), float(baseline)

    def _rectify_gray(self, left_gray: np.ndarray, right_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._rectify_ready:
            if not self._warned_no_rectification:
                self._logger.warn('Rectification maps unavailable; using unrectified stereo frames.')
                self._warned_no_rectification = True
            return left_gray, right_gray

        target_w, target_h = self._rectified_size
        if left_gray.shape[:2] != (target_h, target_w):
            left_gray = cv2.resize(left_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if right_gray.shape[:2] != (target_h, target_w):
            right_gray = cv2.resize(right_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        left_rect = cv2.remap(left_gray, self._map_l1, self._map_l2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, self._map_r1, self._map_r2, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def _rectify_color(self, left_bgr: np.ndarray) -> np.ndarray:
        if not self._rectify_ready:
            return left_bgr

        target_w, target_h = self._rectified_size
        if left_bgr.shape[:2] != (target_h, target_w):
            left_bgr = cv2.resize(left_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return cv2.remap(left_bgr, self._map_l1, self._map_l2, cv2.INTER_LINEAR)

    def _load_calibration(self):
        calibration_path = self._cfg.calibration_file.strip() or self._default_calibration_path()
        if not calibration_path:
            self._logger.warn('No calibration file configured; rectification disabled.')
            return
        if not os.path.exists(calibration_path):
            self._logger.warn(f'Calibration file not found at {calibration_path}; rectification disabled.')
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

            self._logger.info(
                'Loaded stereo calibration: '
                f'{calibration_path} fx={self._calib_fx:.3f}px baseline={self._calib_baseline_m:.4f}m'
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warn(f'Failed to load calibration from {calibration_path}: {exc}')

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
            k_left, d_left, r1, p1, image_size, cv2.CV_16SC2
        )
        self._map_r1, self._map_r2 = cv2.initUndistortRectifyMap(
            k_right, d_right, r2, p2, image_size, cv2.CV_16SC2
        )
        self._rectified_size = image_size
        self._rectified_p1 = p1.copy()
        self._rectify_r1 = r1.copy()
        self._rectify_ready = True
        return p1, p2

    @staticmethod
    def _matrix_from_cfg(node, shape):
        arr = np.array(node['data'], dtype=np.float64)
        if len(shape) == 2:
            return arr.reshape(shape)
        return arr.reshape((-1,))

    @staticmethod
    def _default_calibration_path() -> str:
        try:
            share_dir = get_package_share_directory('sim_car')
            return os.path.join(share_dir, 'config', 'stereo_calibration.yaml')
        except Exception:  # pylint: disable=broad-except
            return ''

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

            self._logger.warn(f'Unsupported stereo encoding: {msg.encoding}')
            return None
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warn(f'Failed to decode image ({msg.encoding}): {exc}')
            return None

    @staticmethod
    def _decode_to_bgr(msg: Image) -> Optional[np.ndarray]:
        try:
            if msg.height <= 0 or msg.width <= 0:
                return None

            encoding = msg.encoding.lower()
            if encoding == 'bgr8':
                return StereoPipeline._reshape_color8(msg, channels=3)
            if encoding == 'rgb8':
                rgb = StereoPipeline._reshape_color8(msg, channels=3)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if encoding == 'bgra8':
                bgra = StereoPipeline._reshape_color8(msg, channels=4)
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            if encoding == 'rgba8':
                rgba = StereoPipeline._reshape_color8(msg, channels=4)
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            if encoding in {'mono8', '8uc1'}:
                gray = StereoPipeline._reshape_mono8(msg)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None
        return None

    @staticmethod
    def _reshape_mono8(msg: Image) -> np.ndarray:
        row_bytes = int(msg.step) if int(msg.step) > 0 else int(msg.width)
        needed = row_bytes * int(msg.height)
        data = np.frombuffer(msg.data, dtype=np.uint8, count=needed)
        rows = data.reshape((msg.height, row_bytes))
        return rows[:, : msg.width].copy()

    @staticmethod
    def _reshape_color8(msg: Image, channels: int) -> np.ndarray:
        min_row_bytes = int(msg.width) * channels
        row_bytes = int(msg.step) if int(msg.step) > 0 else min_row_bytes
        needed = row_bytes * int(msg.height)
        data = np.frombuffer(msg.data, dtype=np.uint8, count=needed)
        rows = data.reshape((msg.height, row_bytes))
        usable = rows[:, : min_row_bytes]
        return usable.reshape((msg.height, msg.width, channels)).copy()
