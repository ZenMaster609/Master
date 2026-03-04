"""ROS2 node that runs stereo perception and evaluation in one process."""

import csv
from collections import deque
from dataclasses import dataclass
import io
import math
import os
import threading
import time
import warnings
from typing import Deque, Optional

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from eufs_msgs.msg import ConeArrayWithCovariance
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Header, Int32, String
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.perception import (
    CameraDebugPublisher,
    PerfLogger,
    StereoEvaluator,
    StereoPipeline,
    StereoPipelineConfig,
    YoloOnnxDetector,
    YoloPtDetector,
    estimate_axis_depth_from_bbox_height,
)
from sim_car.perception.range_rmse_analyzer import RangeRMSEAnalyzer
from sim_car.perception.range_rmse_live_plot import RangeRMSELivePlot

# Matplotlib can warn about missing Axes3D when mixed site-packages are present.
# This node only uses 2D plotting, so suppress that specific non-fatal warning.
warnings.filterwarnings(
    'ignore',
    message=r'Unable to import Axes3D\..*',
    category=UserWarning,
    module=r'matplotlib\.projections',
)


@dataclass
class FramePacket:
    """Queued frame packet used by pairer worker."""

    msg: Image
    pair_time_sec: float


@dataclass
class ConePacket:
    """Queued cone packet used for time matching with stereo frames."""

    msg: ConeArrayWithCovariance
    stamp_sec: float


@dataclass
class ConeDepthMetrics:
    """Latest cone-vs-depth validation metrics."""

    pairs: int = 0
    axis_mae_m: Optional[float] = None
    axis_rmse_m: Optional[float] = None
    axis_bias_m: Optional[float] = None
    range_mae_m: Optional[float] = None
    range_rmse_m: Optional[float] = None
    sync_dt_ms: Optional[float] = None
    yolo_detections: int = 0
    yolo_depth_valid: int = 0
    gt_projected: int = 0
    bbox_matches: int = 0
    cone_id_matches: int = 0


@dataclass(frozen=True)
class TrackConeRef:
    """Track cone reference used to assign stable per-cone IDs."""

    cone_id: str
    color: str
    x: float
    y: float


@dataclass
class ConeRunningStats:
    """Running error statistics for one cone ID."""

    color: str
    samples: int = 0
    abs_err_sum: float = 0.0
    sq_err_sum: float = 0.0
    sq_err_x_sum: float = 0.0
    sq_err_y_sum: float = 0.0
    cam_depth_sum: float = 0.0
    gt_depth_sum: float = 0.0

    def add(self, err_m: float, err_x_m: float, err_y_m: float, cam_depth_m: float, gt_depth_m: float):
        self.samples += 1
        self.abs_err_sum += abs(err_m)
        self.sq_err_sum += err_m * err_m
        self.sq_err_x_sum += err_x_m * err_x_m
        self.sq_err_y_sum += err_y_m * err_y_m
        self.cam_depth_sum += cam_depth_m
        self.gt_depth_sum += gt_depth_m

    def mae(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return self.abs_err_sum / float(self.samples)

    def rmse(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return math.sqrt(self.sq_err_sum / float(self.samples))

    def rmse_x(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return math.sqrt(self.sq_err_x_sum / float(self.samples))

    def rmse_y(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return math.sqrt(self.sq_err_y_sum / float(self.samples))

    def dcam(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return self.cam_depth_sum / float(self.samples)

    def dgt(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return self.gt_depth_sum / float(self.samples)


@dataclass
class VisibleConeMetrics:
    """Per-frame cone metrics for currently visible/matched cones."""

    cone_id: str
    color: str
    samples: int
    mae: Optional[float]
    rmse: Optional[float]
    rmse_x: Optional[float]
    rmse_y: Optional[float]
    dcam_inst: float
    dgt_inst: float
    dcam: float
    dgt: float


@dataclass
class MonocularFitSample:
    """Per-match monocular sample used for offline depth fitting."""

    timestamp: float
    session_source: str
    u_center_px: float
    v_center_px: float
    bbox_height_px: float
    bbox_width_px: float
    fy_px: float
    fx_px: float
    cx_px: float
    cy_px: float
    est_axis_depth_m: float
    gt_axis_depth_m: float
    axis_error_m: float
    gt_range_m: float
    gt_x_cam_m: float
    gt_y_cam_m: float
    gt_z_cam_m: float
    est_x_cam_m: float
    est_y_cam_m: float
    est_z_cam_m: float
    error_xy_m: float
    cone_color: str
    predicted_class_id: Optional[int]
    ground_truth_class_id: Optional[int]
    cone_id: str
    projection_model: str


class PerceptionNode(Node):
    """Subscribes stereo topics, computes disparity/depth, and publishes eval metrics."""

    _CONE_CLASS_NAME_TO_ID = {
        'blue': 0,
        'yellow': 1,
        'orange': 2,
        'big_orange': 3,
        'unknown': 4,
    }

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

        self._cone_queue: Deque[ConePacket] = deque()
        self._cone_lock = threading.Lock()
        self._latest_cone_metrics = ConeDepthMetrics()
        self._track_lock = threading.Lock()
        self._track_cones = []
        self._track_frame_id = 'map'
        self._planner_path_lock = threading.Lock()
        self._latest_planner_path: Optional[Path] = None
        self._planner_markers_lock = threading.Lock()
        self._latest_pair_links_marker: Optional[Marker] = None
        self._latest_planner_status_text: str = ''
        self._odom_lock = threading.Lock()
        self._latest_odom = None
        self._odom_queue: Deque[tuple[float, Odometry]] = deque()
        self._cone_stats_lock = threading.Lock()
        self._cone_stats = {}
        self._visible_cones_lock = threading.Lock()
        self._latest_visible_cones: list[VisibleConeMetrics] = []
        self._range_rmse_analyzer: Optional[RangeRMSEAnalyzer] = None
        self._range_rmse_plot: Optional[RangeRMSELivePlot] = None
        self._warned_cone_frame_fallback = False
        self._warned_cone_source_alias = False
        self._warned_cone_projection_alias = False
        self._last_throttled_log_sec: dict[str, float] = {}

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_cache_lock = threading.Lock()
        self._tf_cache = {}

        self._pipeline: Optional[StereoPipeline] = None
        self._evaluator: Optional[StereoEvaluator] = None
        self._perf: Optional[PerfLogger] = None
        if self.stereo_enabled:
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
        self._camera_debug = CameraDebugPublisher(
            node=self,
            enabled=self.camera_debug,
            topic=self.camera_debug_topic,
            publish_every_n=self.camera_debug_n_frames,
        )
        self._yolo_detector = self._init_yolo_detector()
        self._yolo_backend = self._yolo_detector.backend if self._yolo_detector is not None else 'disabled'

        prefix = self.eval_topic_prefix.rstrip('/')
        self._cone_pairs_pub = self.create_publisher(Int32, f'{prefix}/cone_depth_pairs', 10)
        self._cone_axis_mae_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_axis_mae_m', 10)
        self._cone_axis_rmse_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_axis_rmse_m', 10)
        self._cone_axis_bias_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_axis_bias_m', 10)
        self._cone_range_mae_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_range_mae_m', 10)
        self._cone_range_rmse_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_range_rmse_m', 10)
        self._cone_sync_dt_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_sync_dt_ms', 10)
        self._cone_yolo_detections_pub = self.create_publisher(Int32, f'{prefix}/cone_depth_yolo_detections', 10)
        self._cone_yolo_depth_valid_pub = self.create_publisher(Int32, f'{prefix}/cone_depth_yolo_depth_valid', 10)
        self._cone_gt_projected_pub = self.create_publisher(Int32, f'{prefix}/cone_depth_gt_projected', 10)
        self._cone_bbox_matches_pub = self.create_publisher(Int32, f'{prefix}/cone_depth_bbox_matches', 10)
        self._cone_id_matches_pub = self.create_publisher(Int32, f'{prefix}/cone_depth_cone_id_matches', 10)
        self._cone_per_cone_pub = self.create_publisher(String, f'{prefix}/cone_depth_per_cone', 10)
        self._cone_sample_pub = self.create_publisher(String, f'{prefix}/cone_depth_samples', 10)
        self._cone_mono_fit_sample_pub = self.create_publisher(
            String,
            f'{prefix}/cone_depth_monocular_fit_samples',
            10,
        )
        self._yolo_count_pub = self.create_publisher(Int32, f'{prefix}/yolo/detection_count', 10)
        self._yolo_infer_ms_pub = self.create_publisher(Float32, f'{prefix}/yolo/inference_ms', 10)
        self._cone_detections_pub = self.create_publisher(ConeDetectionArray, self.cone_detections_topic, 10)

        self.create_subscription(Image, self.left_image_topic, self._left_image_cb, 10)
        self.create_subscription(CameraInfo, self.left_camera_info_topic, self._left_info_cb, 10)
        if self.stereo_enabled:
            self.create_subscription(Image, self.right_image_topic, self._right_image_cb, 10)
            self.create_subscription(CameraInfo, self.right_camera_info_topic, self._right_info_cb, 10)
        self.create_subscription(ConeArrayWithCovariance, self.ground_truth_cones_topic, self._cone_gt_cb, 10)
        self.create_subscription(ConeArrayWithCovariance, self.ground_truth_track_topic, self._track_gt_cb, 10)
        self.create_subscription(Path, self.planner_path_topic, self._planner_path_cb, 10)
        self.create_subscription(MarkerArray, self.planner_markers_topic, self._planner_markers_cb, 10)
        self.create_subscription(Odometry, self.cone_eval_odom_topic, self._odom_cb, 10)

        self._worker_thread: Optional[threading.Thread] = None
        if self.stereo_enabled:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

        if self.perf_log_hz > 0.0 and self._perf is not None and self._evaluator is not None:
            self.create_timer(1.0 / self.perf_log_hz, self._perf_timer_cb)
        if self.cone_plotting_2:
            self._range_rmse_analyzer = RangeRMSEAnalyzer(
                range_min_m=0.0,
                range_max_m=20.0,
                bin_width_m=1.0,
            )
            try:
                self._range_rmse_plot = RangeRMSELivePlot(
                    range_min_m=0.0,
                    range_max_m=20.0,
                    bin_width_m=1.0,
                )
                self.create_timer(0.2, self._update_range_rmse_plot)
            except Exception as exc:  # pylint: disable=broad-except
                self.cone_plotting_2 = False
                self._range_rmse_analyzer = None
                self._range_rmse_plot = None
                self.get_logger().warn(
                    f'Failed to initialize cone_plotting_2 window ({exc}); disabling cone_plotting_2.'
                )

        self.get_logger().info(
            'perception_node ready: '
            f'left={self.left_image_topic} right={self.right_image_topic} '
            f'stereo_enabled={self.stereo_enabled} '
            f'eval_prefix={self.eval_topic_prefix} perf_log_hz={self.perf_log_hz:.2f} '
            f'camera_debug={self.camera_debug} cones={self.ground_truth_cones_topic} '
            f'track={self.ground_truth_track_topic} odom={self.cone_eval_odom_topic} '
            f'yolo_enabled={self.yolo_enabled} yolo_backend={self._yolo_backend} '
            f'cone_plotting_2={self.cone_plotting_2} '
            f'monocular_cone_height_m={self.monocular_cone_height_m:.4f} '
            f'monocular_bbox_height_offset_px={self.monocular_bbox_height_offset_px:.3f} '
            f'cone_detections_topic={self.cone_detections_topic} '
            f'cone_detections_frame={self.cone_detections_frame} '
            f'planner_path_topic={self.planner_path_topic} '
            f'planner_markers_topic={self.planner_markers_topic} '
            f'camera_debug_scale={self.camera_debug_scale:.2f} '
            f'camera_debug_mono={self.camera_debug_mono}'
        )

    def _declare_parameters(self):
        self.declare_parameter('left_image_topic', '/sim/raw/stereo/left/image_raw')
        self.declare_parameter('right_image_topic', '/sim/raw/stereo/right/image_raw')
        self.declare_parameter('left_camera_info_topic', '/sim/raw/stereo/left/camera_info')
        self.declare_parameter('right_camera_info_topic', '/sim/raw/stereo/right/camera_info')
        self.declare_parameter('stereo_enabled', True)
        self.declare_parameter('monocular_cone_height_m', 0.3034)
        self.declare_parameter('monocular_bbox_height_offset_px', 0.0)

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
        self.declare_parameter('camera_debug', 'true')
        self.declare_parameter('camera_debug_topic', '/sim/raw/stereo/camera_debug')
        self.declare_parameter('camera_debug_n_frames', 30)
        self.declare_parameter('planner_path_topic', '/sim/planner/pair_midpoint_path')
        self.declare_parameter('planner_markers_topic', '/sim/planner/pair_midpoint_markers')
        self.declare_parameter('camera_debug_scale', 0.5)
        self.declare_parameter('camera_debug_mono', True)
        self.declare_parameter('yolo_enabled', False)
        self.declare_parameter('yolo_model_path', '')
        self.declare_parameter('yolo_input_size', 960)
        self.declare_parameter('yolo_conf_threshold', 0.25)
        self.declare_parameter('yolo_iou_threshold', 0.45)
        self.declare_parameter('yolo_max_detections', 100)
        self.declare_parameter('yolo_prefer_cuda', True)
        self.declare_parameter('yolo_class_names', [])
        self.declare_parameter('cone_detections_topic', '/sim/raw/stereo/perception/cones_3d')
        self.declare_parameter('cone_detections_frame', 'base_footprint')

        self.declare_parameter('ground_truth_cones_topic', '/ground_truth/cones')
        self.declare_parameter('ground_truth_track_topic', '/ground_truth/track')
        self.declare_parameter('cone_eval_sync_slop_sec', 0.10)
        self.declare_parameter('cone_eval_target_frame', '')
        self.declare_parameter('cone_eval_fallback_frame', 'stereo_left_link')
        self.declare_parameter('cone_eval_projection_model', 'auto')
        self.declare_parameter('cone_eval_pixel_radius', 2)
        self.declare_parameter('cone_eval_tf_timeout_sec', 0.0)
        self.declare_parameter('cone_eval_include_unknown', False)
        self.declare_parameter('cone_eval_track_match_threshold_m', 1.5)
        self.declare_parameter('cone_eval_track_match_relaxed_threshold_m', 3.0)
        self.declare_parameter('cone_eval_per_cone_max_rows', 120)
        self.declare_parameter('cone_eval_odom_topic', '/sim/odom')
        self.declare_parameter('cone_plotting_2', False)

    def _read_parameters(self):
        self.left_image_topic = str(self.get_parameter('left_image_topic').value)
        self.right_image_topic = str(self.get_parameter('right_image_topic').value)
        self.left_camera_info_topic = str(self.get_parameter('left_camera_info_topic').value)
        self.right_camera_info_topic = str(self.get_parameter('right_camera_info_topic').value)
        self.stereo_enabled = bool(self.get_parameter('stereo_enabled').value)
        self.monocular_cone_height_m = max(1e-6, float(self.get_parameter('monocular_cone_height_m').value))
        self.monocular_bbox_height_offset_px = float(self.get_parameter('monocular_bbox_height_offset_px').value)

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
        raw_camera_debug = self.get_parameter('camera_debug').value
        self.camera_debug = self._sanitize_camera_debug(raw_camera_debug)
        raw_camera_debug_token = str(raw_camera_debug).strip().lower()
        if raw_camera_debug_token in {'rect_left', 'left_rect', 'depth', 'disparity', 'yolo'}:
            self.get_logger().warn(
                f'camera_debug="{raw_camera_debug}" uses deprecated mode semantics; treating it as enabled'
            )
        self.camera_debug_topic = str(self.get_parameter('camera_debug_topic').value)
        self.camera_debug_n_frames = max(1, int(self.get_parameter('camera_debug_n_frames').value))
        self.planner_path_topic = str(self.get_parameter('planner_path_topic').value).strip()
        if not self.planner_path_topic:
            self.planner_path_topic = '/sim/planner/pair_midpoint_path'
        self.planner_markers_topic = str(self.get_parameter('planner_markers_topic').value).strip()
        if not self.planner_markers_topic:
            self.planner_markers_topic = '/sim/planner/pair_midpoint_markers'
        self.camera_debug_scale = float(np.clip(float(self.get_parameter('camera_debug_scale').value), 0.1, 1.0))
        self.camera_debug_mono = bool(self.get_parameter('camera_debug_mono').value)
        self.yolo_enabled = bool(self.get_parameter('yolo_enabled').value)
        self.yolo_model_path = str(self.get_parameter('yolo_model_path').value)
        self.yolo_input_size = max(64, int(self.get_parameter('yolo_input_size').value))
        self.yolo_conf_threshold = float(self.get_parameter('yolo_conf_threshold').value)
        self.yolo_iou_threshold = float(self.get_parameter('yolo_iou_threshold').value)
        self.yolo_max_detections = max(1, int(self.get_parameter('yolo_max_detections').value))
        self.yolo_prefer_cuda = bool(self.get_parameter('yolo_prefer_cuda').value)
        raw_class_names = self.get_parameter('yolo_class_names').value
        self.yolo_class_names = self._sanitize_yolo_class_names(raw_class_names)
        self.cone_detections_topic = str(self.get_parameter('cone_detections_topic').value)
        self.cone_detections_frame = str(self.get_parameter('cone_detections_frame').value).strip()
        if not self.cone_detections_frame:
            self.cone_detections_frame = 'base_footprint'

        self.ground_truth_cones_topic = str(self.get_parameter('ground_truth_cones_topic').value)
        self.ground_truth_track_topic = str(self.get_parameter('ground_truth_track_topic').value)
        self.cone_eval_sync_slop_sec = max(0.01, float(self.get_parameter('cone_eval_sync_slop_sec').value))
        self.cone_eval_target_frame = str(self.get_parameter('cone_eval_target_frame').value)
        self.cone_eval_fallback_frame = str(self.get_parameter('cone_eval_fallback_frame').value)
        self.cone_eval_projection_model = self._sanitize_projection_model(
            self.get_parameter('cone_eval_projection_model').value
        )
        self.cone_eval_pixel_radius = max(0, int(self.get_parameter('cone_eval_pixel_radius').value))
        self.cone_eval_tf_timeout_sec = max(0.0, float(self.get_parameter('cone_eval_tf_timeout_sec').value))
        self.cone_eval_include_unknown = bool(self.get_parameter('cone_eval_include_unknown').value)
        self.cone_eval_track_match_threshold_m = max(
            0.1, float(self.get_parameter('cone_eval_track_match_threshold_m').value)
        )
        self.cone_eval_track_match_relaxed_threshold_m = max(
            self.cone_eval_track_match_threshold_m,
            float(self.get_parameter('cone_eval_track_match_relaxed_threshold_m').value),
        )
        self.cone_eval_per_cone_max_rows = max(1, int(self.get_parameter('cone_eval_per_cone_max_rows').value))
        self.cone_eval_odom_topic = str(self.get_parameter('cone_eval_odom_topic').value)
        self.cone_plotting_2 = bool(self.get_parameter('cone_plotting_2').value)

    def _left_info_cb(self, msg: CameraInfo):
        self._left_info = msg

    def _right_info_cb(self, msg: CameraInfo):
        self._right_info = msg

    def _left_image_cb(self, msg: Image):
        if self.stereo_enabled:
            self._enqueue_frame(msg, side='left')
            return
        self._process_monocular_frame(msg)

    def _right_image_cb(self, msg: Image):
        self._enqueue_frame(msg, side='right')

    def _cone_gt_cb(self, msg: ConeArrayWithCovariance):
        stamp_sec = self._stamp_msg_to_sec(msg.header.stamp)
        with self._cone_lock:
            self._cone_queue.append(ConePacket(msg=msg, stamp_sec=stamp_sec))
            while len(self._cone_queue) > 200:
                self._cone_queue.popleft()

    def _track_gt_cb(self, msg: ConeArrayWithCovariance):
        refs = []
        refs.extend(self._build_track_refs(msg.blue_cones, 'blue'))
        refs.extend(self._build_track_refs(msg.yellow_cones, 'yellow'))
        refs.extend(self._build_track_refs(msg.orange_cones, 'orange'))
        refs.extend(self._build_track_refs(msg.big_orange_cones, 'big_orange'))
        refs.extend(self._build_track_refs(msg.unknown_color_cones, 'unknown'))
        frame_id = str(msg.header.frame_id).strip() or 'map'
        with self._track_lock:
            self._track_cones = refs
            self._track_frame_id = frame_id

    def _planner_path_cb(self, msg: Path):
        with self._planner_path_lock:
            self._latest_planner_path = msg

    def _planner_markers_cb(self, msg: MarkerArray):
        pair_links_marker = None
        status_text = ''
        for marker in msg.markers:
            marker_ns = str(marker.ns).strip()
            if marker.type == Marker.LINE_LIST and marker_ns == 'pair_links':
                pair_links_marker = marker
            elif marker.type == Marker.TEXT_VIEW_FACING and marker_ns == 'status':
                status_text = str(marker.text)
        with self._planner_markers_lock:
            self._latest_pair_links_marker = pair_links_marker
            self._latest_planner_status_text = status_text

    def _odom_cb(self, msg: Odometry):
        stamp_sec = self._stamp_msg_to_sec(msg.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = time.monotonic()
        with self._odom_lock:
            self._latest_odom = msg
            self._odom_queue.append((stamp_sec, msg))
            while len(self._odom_queue) > 400:
                self._odom_queue.popleft()

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
            if self._pipeline is None:
                continue
            output = self._pipeline.process(
                left_msg=left_packet.msg,
                right_msg=right_packet.msg,
                left_info=self._left_info,
                right_info=self._right_info,
            )
            if output is None:
                continue

            if self._evaluator is not None:
                self._evaluator.update(
                    left_rect_gray=output.left_rect,
                    right_rect_gray=output.right_rect,
                    disparity=output.disparity,
                    depth=output.depth,
                )
            if self._perf is not None:
                self._perf.record_processed(timings_ms=output.timings_ms, backend=output.backend)

            eval_header = self._common_header(left_packet.msg.header, right_packet.msg.header)
            stereo_left_info = self._pipeline.build_rectified_left_camera_info(self._left_info)
            rectification_rotation = self._pipeline.left_rectification_rotation()

            stereo_debug_image = output.left_rect_color
            if stereo_debug_image is None:
                stereo_debug_image = output.left_rect
            yolo_input_image = output.left_rect_color
            if yolo_input_image is None:
                yolo_input_image = cv2.cvtColor(output.left_rect, cv2.COLOR_GRAY2BGR)

            yolo_detections, infer_ms = self._run_yolo(yolo_input_image)
            self._yolo_count_pub.publish(Int32(data=len(yolo_detections)))
            self._yolo_infer_ms_pub.publish(Float32(data=float(infer_ms)))
            self._apply_depth_map_to_detections(output.depth, yolo_detections)
            self._latest_cone_metrics = self._evaluate_yolo_depth(
                yolo_detections=yolo_detections,
                left_info=stereo_left_info,
                eval_header=eval_header,
                sample_source='stereo',
                rectification_rotation=rectification_rotation,
                publish_range_rmse_samples=False,
            )

            if self.cone_plotting_2:
                self._evaluate_cone_depth(
                    depth=output.depth,
                    left_info=stereo_left_info,
                    eval_header=eval_header,
                    publish_results=False,
                    record_stats=False,
                    sample_source='stereo',
                    rectification_rotation=rectification_rotation,
                )

            if self._camera_debug.should_publish():
                debug_image = self._build_camera_debug_image(
                    left_rect=stereo_debug_image,
                    left_info=stereo_left_info,
                    eval_header=eval_header,
                    yolo_detections=yolo_detections,
                )
                if debug_image is not None:
                    encoding = 'mono8' if self.camera_debug_mono else 'bgr8'
                    self._camera_debug.publish_image(eval_header, debug_image, encoding)

    def _build_camera_debug_image(
        self,
        left_rect: np.ndarray | None,
        left_info: Optional[CameraInfo],
        eval_header: Header,
        yolo_detections: list[dict],
    ) -> Optional[np.ndarray]:
        if left_rect is None or left_rect.size == 0:
            return None

        if left_rect.ndim == 2:
            debug_image = left_rect.copy()
        elif left_rect.ndim == 3 and left_rect.shape[2] == 1:
            debug_image = left_rect[:, :, 0].copy()
        elif left_rect.ndim == 3 and left_rect.shape[2] >= 3:
            debug_image = cv2.cvtColor(left_rect[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            return None

        if self.camera_debug_scale < 0.999:
            debug_image = cv2.resize(
                debug_image,
                None,
                fx=self.camera_debug_scale,
                fy=self.camera_debug_scale,
                interpolation=cv2.INTER_AREA,
            )

        self._draw_yolo_debug_overlays(debug_image, yolo_detections)
        self._draw_pair_links_overlay(
            image=debug_image,
            left_info=left_info,
            eval_header=eval_header,
        )
        self._draw_planner_path_overlay(
            image=debug_image,
            left_info=left_info,
            eval_header=eval_header,
        )
        self._draw_debug_status_text(debug_image, yolo_detection_count=len(yolo_detections))

        if self.camera_debug_mono:
            return debug_image
        return cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)

    def _process_monocular_frame(self, msg: Image) -> None:
        left_bgr = StereoPipeline._decode_to_bgr(msg)
        if left_bgr is None:
            return

        yolo_input_image = left_bgr
        if yolo_input_image is None:
            return

        eval_header = Header()
        eval_header.stamp = msg.header.stamp
        eval_header.frame_id = msg.header.frame_id

        yolo_detections, infer_ms = self._run_yolo(yolo_input_image)
        self._yolo_count_pub.publish(Int32(data=len(yolo_detections)))
        self._yolo_infer_ms_pub.publish(Float32(data=float(infer_ms)))
        self._apply_monocular_depth_to_detections(yolo_detections, self._left_info)
        self._latest_cone_metrics = self._evaluate_yolo_depth(
            yolo_detections=yolo_detections,
            left_info=self._left_info,
            eval_header=eval_header,
        )

        if self._camera_debug.should_publish():
            debug_image = self._build_camera_debug_image(
                left_rect=left_bgr,
                left_info=self._left_info,
                eval_header=eval_header,
                yolo_detections=yolo_detections,
            )
            if debug_image is not None:
                encoding = 'mono8' if self.camera_debug_mono else 'bgr8'
                self._camera_debug.publish_image(eval_header, debug_image, encoding)

    def _apply_monocular_depth_to_detections(
        self,
        yolo_detections: list[dict],
        left_info: Optional[CameraInfo],
    ) -> None:
        if not yolo_detections:
            return

        _fx, fy, _cx, _cy = self._camera_intrinsics(left_info) if left_info is not None else (0.0, 0.0, 0.0, 0.0)
        for det in yolo_detections:
            x0 = float(det.get('x0', -1.0))
            y0 = float(det.get('y0', -1.0))
            x1 = float(det.get('x1', -1.0))
            y1 = float(det.get('y1', -1.0))
            if x1 <= x0 or y1 <= y0:
                det['depth_m'] = None
                continue

            det['u_center'] = 0.5 * (x0 + x1)
            det['v_center'] = 0.5 * (y0 + y1)
            bbox_height_px = y1 - y0
            depth_m = estimate_axis_depth_from_bbox_height(
                fy_px=fy,
                cone_height_m=self.monocular_cone_height_m,
                bbox_height_px=bbox_height_px,
                bbox_height_offset_px=self.monocular_bbox_height_offset_px,
            )
            det['depth_m'] = float(depth_m) if depth_m is not None else None

    def _apply_depth_map_to_detections(
        self,
        depth: np.ndarray,
        yolo_detections: list[dict],
    ) -> None:
        if depth is None or depth.size == 0 or not yolo_detections:
            return

        height, width = depth.shape[:2]
        for det in yolo_detections:
            x0 = int(det.get('x0', -1))
            y0 = int(det.get('y0', -1))
            x1 = int(det.get('x1', -1))
            y1 = int(det.get('y1', -1))
            if x1 <= x0 or y1 <= y0:
                det['depth_m'] = None
                continue

            u = max(0.0, min(float(width - 1), 0.5 * float(x0 + x1)))
            v = max(0.0, min(float(height - 1), 0.5 * float(y0 + y1)))
            det['u_center'] = float(u)
            det['v_center'] = float(v)
            est_axis = self._sample_depth_from_bbox(depth, x0, y0, x1, y1)
            if np.isfinite(est_axis):
                det['depth_m'] = float(est_axis)
            else:
                det['depth_m'] = None

    @staticmethod
    def _sample_depth_from_bbox(
        depth: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> float:
        """Sample depth from the lower-middle of a bbox, where cone pixels are more stable."""
        height, width = depth.shape[:2]
        if x1 <= x0 or y1 <= y0:
            return float('nan')

        box_w = x1 - x0
        box_h = y1 - y0
        crop_x0 = max(0, x0 + int(round(0.25 * box_w)))
        crop_x1 = min(width, x1 - int(round(0.25 * box_w)))
        crop_y0 = max(0, y0 + int(round(0.45 * box_h)))
        crop_y1 = min(height, y0 + int(round(0.95 * box_h)))
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            u = 0.5 * float(x0 + x1)
            v = 0.5 * float(y0 + y1)
            return PerceptionNode._sample_depth(depth, u, v, radius_px=2)

        patch = depth[crop_y0:crop_y1, crop_x0:crop_x1]
        valid = patch[np.isfinite(patch)]
        if valid.size > 0:
            return float(np.median(valid))

        u = 0.5 * float(x0 + x1)
        v = 0.5 * float(y0 + y1)
        return PerceptionNode._sample_depth(depth, u, v, radius_px=2)

    def _draw_yolo_debug_overlays(self, image: np.ndarray, yolo_detections: list[dict]) -> None:
        if image is None or not yolo_detections:
            return

        height, width = image.shape[:2]
        scale = self.camera_debug_scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.40
        thickness = 1

        for det in yolo_detections:
            x0 = int(round(float(det.get('x0', -1)) * scale))
            y0 = int(round(float(det.get('y0', -1)) * scale))
            x1 = int(round(float(det.get('x1', -1)) * scale))
            y1 = int(round(float(det.get('y1', -1)) * scale))
            if x1 <= x0 or y1 <= y0:
                continue

            x0 = max(0, min(width - 1, x0))
            y0 = max(0, min(height - 1, y0))
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            cv2.rectangle(image, (x0, y0), (x1, y1), 255, 1)

            label = str(det.get('label', '')).strip()
            if not label:
                continue

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = max(0, min(width - text_w - 3, x0))
            text_y = max(text_h + baseline + 2, y0 - 2)
            box_y0 = max(0, text_y - text_h - baseline - 2)
            box_y1 = min(height - 1, text_y + 1)
            box_x1 = min(width - 1, text_x + text_w + 2)
            cv2.rectangle(image, (text_x, box_y0), (box_x1, box_y1), 0, -1)
            cv2.putText(image, label, (text_x + 1, text_y), font, font_scale, 255, thickness, cv2.LINE_AA)

    def _draw_planner_path_overlay(
        self,
        image: np.ndarray,
        left_info: Optional[CameraInfo],
        eval_header: Header,
    ) -> None:
        projected_segments = self._project_planner_path_to_debug_image(
            image_shape=image.shape,
            left_info=left_info,
            eval_header=eval_header,
        )
        if not projected_segments:
            return

        for segment in projected_segments:
            if len(segment) >= 2:
                cv2.polylines(image, [np.array(segment, dtype=np.int32)], False, 180, 2, cv2.LINE_AA)
            for point in segment:
                cv2.circle(image, point, 2, 120, -1, cv2.LINE_AA)

    def _draw_pair_links_overlay(
        self,
        image: np.ndarray,
        left_info: Optional[CameraInfo],
        eval_header: Header,
    ) -> None:
        projected_segments = self._project_pair_links_to_debug_image(
            image_shape=image.shape,
            left_info=left_info,
            eval_header=eval_header,
        )
        if not projected_segments:
            return

        for start_point, end_point in projected_segments:
            cv2.line(image, start_point, end_point, 200, 1, cv2.LINE_AA)
            cv2.circle(image, start_point, 1, 160, -1, cv2.LINE_AA)
            cv2.circle(image, end_point, 1, 160, -1, cv2.LINE_AA)

    def _project_pair_links_to_debug_image(
        self,
        image_shape: tuple[int, ...],
        left_info: Optional[CameraInfo],
        eval_header: Header,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        if left_info is None:
            return []

        pair_links_marker = self._latest_pair_links_marker_copy()
        if pair_links_marker is None or len(pair_links_marker.points) < 2:
            return []

        image_stamp_sec = self._stamp_msg_to_sec(eval_header.stamp)
        marker_stamp_sec = self._stamp_msg_to_sec(pair_links_marker.header.stamp)
        if marker_stamp_sec > 0.0 and image_stamp_sec > 0.0 and abs(marker_stamp_sec - image_stamp_sec) > 0.5:
            return []

        camera_frame = str(left_info.header.frame_id).strip() or str(eval_header.frame_id).strip()
        marker_frame = str(pair_links_marker.header.frame_id).strip()
        if not camera_frame or not marker_frame:
            return []

        fx, fy, cx, cy = self._camera_intrinsics(left_info)
        if fx <= 0.0 or fy <= 0.0:
            return []

        transform = self._lookup_transform(
            target_frame=camera_frame,
            source_frame=marker_frame,
            stamp=eval_header.stamp,
        )
        if transform is None:
            return []

        projection_model = self._projection_model_for_frame(camera_frame)
        height, width = image_shape[:2]
        projected_segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
        point_count = len(pair_links_marker.points)
        for idx in range(0, point_count - 1, 2):
            start = pair_links_marker.points[idx]
            end = pair_links_marker.points[idx + 1]
            projected_pair = []
            for point in (start, end):
                x_cam, y_cam, z_cam = self._transform_point(transform, float(point.x), float(point.y), float(point.z))
                projection = self._project_to_pixel(
                    x_cam=x_cam,
                    y_cam=y_cam,
                    z_cam=z_cam,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    model=projection_model,
                )
                if projection is None:
                    projected_pair = []
                    break
                u, v, _ = projection
                u = int(round(u * self.camera_debug_scale))
                v = int(round(v * self.camera_debug_scale))
                if u < 0 or v < 0 or u >= width or v >= height:
                    projected_pair = []
                    break
                projected_pair.append((u, v))
            if len(projected_pair) == 2:
                projected_segments.append((projected_pair[0], projected_pair[1]))
        return projected_segments

    def _draw_debug_status_text(self, image: np.ndarray, yolo_detection_count: int) -> None:
        lines = [f'yolo={int(yolo_detection_count)}']
        planner_status = self._latest_planner_status_text_copy()
        if planner_status:
            lines.extend([line.strip() for line in planner_status.splitlines() if line.strip()])
        else:
            lines.append('planner_status=unavailable')

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        line_gap = 4
        text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
        max_width = max((size[0] for size in text_sizes), default=0)
        line_height = max((size[1] for size in text_sizes), default=10) + line_gap
        box_height = max(18, len(lines) * line_height + 8)
        box_width = max(30, max_width + 10)

        height, width = image.shape[:2]
        x0 = 6
        y0 = 6
        x1 = min(width - 1, x0 + box_width)
        y1 = min(height - 1, y0 + box_height)
        cv2.rectangle(image, (x0, y0), (x1, y1), 0, -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), 180, 1)

        baseline_y = y0 + 14
        for idx, line in enumerate(lines):
            y = min(height - 4, baseline_y + idx * line_height)
            cv2.putText(image, line, (x0 + 4, y), font, font_scale, 255, thickness, cv2.LINE_AA)

    def _project_planner_path_to_debug_image(
        self,
        image_shape: tuple[int, ...],
        left_info: Optional[CameraInfo],
        eval_header: Header,
    ) -> list[list[tuple[int, int]]]:
        if left_info is None:
            self._warn_throttled('camera_debug_no_info', 'camera_debug skipping planner overlay: no left camera info')
            return []

        planner_path = self._latest_planner_path_copy()
        if planner_path is None or not planner_path.poses:
            self._warn_throttled('camera_debug_no_path', 'camera_debug skipping planner overlay: no planner path')
            return []

        image_stamp_sec = self._stamp_msg_to_sec(eval_header.stamp)
        path_stamp_sec = self._stamp_msg_to_sec(planner_path.header.stamp)
        if path_stamp_sec > 0.0 and image_stamp_sec > 0.0 and abs(path_stamp_sec - image_stamp_sec) > 0.5:
            self._warn_throttled(
                'camera_debug_stale_path',
                f'camera_debug skipping stale planner path: dt={abs(path_stamp_sec - image_stamp_sec):.3f}s',
            )
            return []

        camera_frame = str(left_info.header.frame_id).strip() or str(eval_header.frame_id).strip()
        path_frame = str(planner_path.header.frame_id).strip()
        if not camera_frame or not path_frame:
            self._warn_throttled('camera_debug_path_frame', 'camera_debug skipping planner overlay: missing frame ids')
            return []

        fx, fy, cx, cy = self._camera_intrinsics(left_info)
        if fx <= 0.0 or fy <= 0.0:
            self._warn_throttled('camera_debug_intrinsics', 'camera_debug skipping planner overlay: invalid intrinsics')
            return []

        transform = self._lookup_transform(
            target_frame=camera_frame,
            source_frame=path_frame,
            stamp=eval_header.stamp,
        )
        if transform is None:
            self._warn_throttled(
                'camera_debug_path_tf',
                f'camera_debug skipping planner overlay: transform {path_frame}->{camera_frame} unavailable',
            )
            return []

        projection_model = self._projection_model_for_frame(camera_frame)
        height, width = image_shape[:2]
        projected_segments: list[list[tuple[int, int]]] = []
        current_segment: list[tuple[int, int]] = []
        for pose_stamped in planner_path.poses:
            px = float(pose_stamped.pose.position.x)
            py = float(pose_stamped.pose.position.y)
            pz = float(pose_stamped.pose.position.z)
            x_cam, y_cam, z_cam = self._transform_point(transform, px, py, pz)
            projection = self._project_to_pixel(
                x_cam=x_cam,
                y_cam=y_cam,
                z_cam=z_cam,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                model=projection_model,
            )
            if projection is None:
                if current_segment:
                    projected_segments.append(current_segment)
                    current_segment = []
                continue
            u, v, _ = projection
            u = int(round(u * self.camera_debug_scale))
            v = int(round(v * self.camera_debug_scale))
            if u < 0 or v < 0 or u >= width or v >= height:
                if current_segment:
                    projected_segments.append(current_segment)
                    current_segment = []
                continue
            current_segment.append((u, v))
        if current_segment:
            projected_segments.append(current_segment)
        return projected_segments

    def _latest_planner_path_copy(self) -> Optional[Path]:
        with self._planner_path_lock:
            return self._latest_planner_path

    def _latest_pair_links_marker_copy(self) -> Optional[Marker]:
        with self._planner_markers_lock:
            return self._latest_pair_links_marker

    def _latest_planner_status_text_copy(self) -> str:
        with self._planner_markers_lock:
            return self._latest_planner_status_text

    def _evaluate_yolo_depth(
        self,
        yolo_detections: list[dict],
        left_info: Optional[CameraInfo],
        eval_header: Header,
        publish_results: bool = True,
        publish_cone_detections: bool = True,
        record_stats: bool = True,
        sample_source: str = 'monocular',
        rectification_rotation: Optional[np.ndarray] = None,
        publish_range_rmse_samples: bool = True,
    ) -> ConeDepthMetrics:
        """Estimate monocular YOLO depth and compute GT depth MAE/RMSE."""
        metrics = ConeDepthMetrics()
        metrics.yolo_detections = len(yolo_detections) if yolo_detections is not None else 0
        visible_rows: list[VisibleConeMetrics] = []
        range_rmse_samples: list[tuple[str, float, float, Optional[int], Optional[int]]] = []
        monocular_fit_samples: list[MonocularFitSample] = []
        if not yolo_detections:
            if publish_cone_detections:
                self._publish_cone_detections(
                    yolo_detections=[],
                    left_info=left_info,
                    eval_header=eval_header,
                )
            return self._finalize_public_cone_eval(
                metrics=metrics,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                monocular_fit_samples=monocular_fit_samples,
                publish_results=publish_results,
                publish_range_rmse_samples=publish_range_rmse_samples,
            )

        yolo_depth_pairs = 0
        for det in yolo_detections:
            est_axis = det.get('depth_m', None)
            if est_axis is None or not np.isfinite(float(est_axis)):
                det['depth_m'] = None
                continue
            yolo_depth_pairs += 1
        metrics.yolo_depth_valid = yolo_depth_pairs
        if publish_cone_detections:
            self._publish_cone_detections(
                yolo_detections=yolo_detections,
                left_info=left_info,
                eval_header=eval_header,
            )

        # GT MAE/RMSE computed by matching YOLO boxes to projected GT cone points.
        if left_info is None:
            metrics.pairs = yolo_depth_pairs
            return self._finalize_public_cone_eval(
                metrics=metrics,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                monocular_fit_samples=monocular_fit_samples,
                publish_results=publish_results,
                publish_range_rmse_samples=publish_range_rmse_samples,
            )

        fx, fy, cx, cy = self._camera_intrinsics(left_info)
        if fx <= 0.0 or fy <= 0.0:
            metrics.pairs = yolo_depth_pairs
            return self._finalize_public_cone_eval(
                metrics=metrics,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                monocular_fit_samples=monocular_fit_samples,
                publish_results=publish_results,
                publish_range_rmse_samples=publish_range_rmse_samples,
            )

        image_height = int(left_info.height) if int(left_info.height) > 0 else 0
        image_width = int(left_info.width) if int(left_info.width) > 0 else 0

        frame_stamp_sec = self._stamp_msg_to_sec(eval_header.stamp)
        cone_packet = self._get_nearest_cone_packet(frame_stamp_sec)
        if cone_packet is None:
            metrics.pairs = yolo_depth_pairs
            return self._finalize_public_cone_eval(
                metrics=metrics,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                monocular_fit_samples=monocular_fit_samples,
                publish_results=publish_results,
                publish_range_rmse_samples=publish_range_rmse_samples,
            )

        metrics.sync_dt_ms = abs(frame_stamp_sec - cone_packet.stamp_sec) * 1000.0
        if metrics.sync_dt_ms > self.cone_eval_sync_slop_sec * 1000.0:
            metrics.pairs = yolo_depth_pairs
            return self._finalize_public_cone_eval(
                metrics=metrics,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                monocular_fit_samples=monocular_fit_samples,
                publish_results=publish_results,
                publish_range_rmse_samples=publish_range_rmse_samples,
            )

        src_frame = cone_packet.msg.header.frame_id
        if not src_frame:
            metrics.pairs = yolo_depth_pairs
            return self._finalize_public_cone_eval(
                metrics=metrics,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                monocular_fit_samples=monocular_fit_samples,
                publish_results=publish_results,
                publish_range_rmse_samples=publish_range_rmse_samples,
            )

        transform_bundle = self._resolve_transform_and_projection(
            source_frame=src_frame,
            left_info_frame=left_info.header.frame_id,
            stamp=eval_header.stamp,
        )
        if transform_bundle is None:
            metrics.pairs = yolo_depth_pairs
            return self._finalize_public_cone_eval(
                metrics=metrics,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                monocular_fit_samples=monocular_fit_samples,
                publish_results=publish_results,
            )
        transform, projection_model = transform_bundle

        with self._track_lock:
            track_frame_id = self._track_frame_id
        track_transform = None
        track_coord_mode = 'none'
        if track_frame_id == src_frame:
            track_coord_mode = 'identity'
        else:
            track_transform = self._lookup_transform(
                target_frame=track_frame_id,
                source_frame=src_frame,
                stamp=eval_header.stamp,
            )
            if track_transform is not None:
                track_coord_mode = 'transform'
            elif (
                track_frame_id in {'map', 'odom'}
                and self._is_base_footprint_frame(src_frame)
            ):
                track_coord_mode = 'odom_fallback'

        gt_points = []
        for cone_color, cone in self._iter_cones(cone_packet.msg):
            x_cam, y_cam, z_cam = self._transform_point(transform, cone.point.x, cone.point.y, cone.point.z)
            x_eval, y_eval, z_eval = x_cam, y_cam, z_cam
            if rectification_rotation is not None:
                x_eval, y_eval, z_eval = self._rotate_point(rectification_rotation, x_cam, y_cam, z_cam)
            projection = self._project_to_pixel(
                x_cam=x_eval,
                y_cam=y_eval,
                z_cam=z_eval,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                model=projection_model,
            )
            if projection is None:
                continue
            u_gt, v_gt, gt_axis = projection
            if (
                image_width > 0
                and image_height > 0
                and (u_gt < 0.0 or v_gt < 0.0 or u_gt >= float(image_width) or v_gt >= float(image_height))
            ):
                continue
            gt_range = math.sqrt((x_cam * x_cam) + (y_cam * y_cam) + (z_cam * z_cam))
            cone_id = None
            if track_coord_mode == 'transform':
                x_map, y_map, _ = self._transform_point(
                    track_transform,
                    cone.point.x,
                    cone.point.y,
                    cone.point.z,
                )
                cone_id = self._match_cone_id(
                    x_map=x_map,
                    y_map=y_map,
                    color=cone_color,
                )
            elif track_coord_mode == 'identity':
                cone_id = self._match_cone_id(
                    x_map=float(cone.point.x),
                    y_map=float(cone.point.y),
                    color=cone_color,
                )
            elif track_coord_mode == 'odom_fallback':
                fallback_point = self._point_to_track_with_odom_fallback(
                    x=float(cone.point.x),
                    y=float(cone.point.y),
                    z=float(cone.point.z),
                    target_stamp_sec=frame_stamp_sec,
                )
                if fallback_point is not None:
                    x_map, y_map, _ = fallback_point
                    cone_id = self._match_cone_id(
                        x_map=x_map,
                        y_map=y_map,
                        color=cone_color,
                    )
            gt_points.append(
                {
                    'u': float(u_gt),
                    'v': float(v_gt),
                    'gt_axis': float(gt_axis),
                    'gt_range': float(gt_range),
                    'x_cam': float(x_cam),
                    'y_cam': float(y_cam),
                    'z_cam': float(z_cam),
                    'cone_id': cone_id,
                    'color': cone_color,
                }
            )
        metrics.gt_projected = len(gt_points)

        errors_axis = []
        errors_range = []
        used_gt = set()
        bbox_matches = 0
        cone_id_matches = 0
        for det in yolo_detections:
            est_axis = det.get('depth_m', None)
            if est_axis is None or not np.isfinite(float(est_axis)):
                continue

            x0 = float(det.get('x0', -1))
            y0 = float(det.get('y0', -1))
            x1 = float(det.get('x1', -1))
            y1 = float(det.get('y1', -1))
            if x1 <= x0 or y1 <= y0:
                continue

            u_center = float(det.get('u_center', 0.5 * (x0 + x1)))
            v_center = float(det.get('v_center', 0.5 * (y0 + y1)))

            best_idx = None
            best_dist_sq = None
            for idx, point in enumerate(gt_points):
                if idx in used_gt:
                    continue
                u_gt = float(point['u'])
                v_gt = float(point['v'])
                if u_gt < x0 or u_gt > x1 or v_gt < y0 or v_gt > y1:
                    continue
                du = u_gt - u_center
                dv = v_gt - v_center
                dist_sq = (du * du) + (dv * dv)
                if best_dist_sq is None or dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_idx = idx

            if best_idx is None:
                continue
            bbox_matches += 1
            used_gt.add(best_idx)
            matched = gt_points[best_idx]
            gt_axis = float(matched['gt_axis'])
            gt_range = float(matched['gt_range'])
            axis_err = float(est_axis) - gt_axis
            est_cam = self._reconstruct_cam_point_from_axis(
                u=u_center,
                v=v_center,
                axis_depth=float(est_axis),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                model=projection_model,
            )
            if est_cam is None:
                continue
            est_x, est_y, est_z = est_cam
            if rectification_rotation is not None:
                est_x, est_y, est_z = self._rotate_point_transpose(rectification_rotation, est_x, est_y, est_z)
            err_x = est_x - float(matched['x_cam'])
            err_y = est_y - float(matched['y_cam'])
            cone_color = str(matched.get('color', 'unknown'))
            predicted_class_id = self._detection_to_class_id(det)
            ground_truth_class_id = self._cone_class_name_to_id(cone_color)
            error_xy_m = math.hypot(err_x, err_y)
            errors_axis.append(axis_err)
            ray_scale = math.sqrt(1.0 + ((u_center - cx) / fx) ** 2 + ((v_center - cy) / fy) ** 2)
            est_range = float(est_axis) * ray_scale
            errors_range.append(est_range - gt_range)
            self._record_range_rmse_sample(
                source=sample_source,
                gt_range_m=gt_range,
                error_m=error_xy_m,
                predicted_class_id=predicted_class_id,
                ground_truth_class_id=ground_truth_class_id,
            )
            range_rmse_samples.append(
                (
                    sample_source,
                    gt_range,
                    error_xy_m,
                    predicted_class_id,
                    ground_truth_class_id,
                )
            )

            cone_id = matched.get('cone_id')
            monocular_fit_samples.append(
                MonocularFitSample(
                    timestamp=float(frame_stamp_sec),
                    session_source='monocular',
                    u_center_px=float(u_center),
                    v_center_px=float(v_center),
                    bbox_height_px=float(y1 - y0),
                    bbox_width_px=float(x1 - x0),
                    fy_px=float(fy),
                    fx_px=float(fx),
                    cx_px=float(cx),
                    cy_px=float(cy),
                    est_axis_depth_m=float(est_axis),
                    gt_axis_depth_m=float(gt_axis),
                    axis_error_m=float(axis_err),
                    gt_range_m=float(gt_range),
                    gt_x_cam_m=float(matched['x_cam']),
                    gt_y_cam_m=float(matched['y_cam']),
                    gt_z_cam_m=float(matched['z_cam']),
                    est_x_cam_m=float(est_x),
                    est_y_cam_m=float(est_y),
                    est_z_cam_m=float(est_z),
                    error_xy_m=float(error_xy_m),
                    cone_color=cone_color,
                    predicted_class_id=predicted_class_id,
                    ground_truth_class_id=ground_truth_class_id,
                    cone_id=str(cone_id) if cone_id is not None else '',
                    projection_model=str(projection_model),
                )
            )
            if isinstance(cone_id, str) and record_stats:
                cone_id_matches += 1
                self._update_cone_stats(
                    cone_id=cone_id,
                    color=cone_color,
                    axis_err_m=axis_err,
                    err_x_m=err_x,
                    err_y_m=err_y,
                    dcam_m=est_range,
                    dgt_m=gt_range,
                )
                samples, mae_v, rmse_v, rmse_x_v, rmse_y_v, dcam_v, dgt_v = self._get_cone_stat_values(cone_id)
                visible_rows.append(
                    VisibleConeMetrics(
                        cone_id=cone_id,
                        color=cone_color,
                        samples=samples,
                        mae=mae_v,
                        rmse=rmse_v,
                        rmse_x=rmse_x_v,
                        rmse_y=rmse_y_v,
                        dcam_inst=est_range,
                        dgt_inst=gt_range,
                        dcam=dcam_v if dcam_v is not None else est_range,
                        dgt=dgt_v if dgt_v is not None else gt_range,
                    )
                )
        metrics.bbox_matches = bbox_matches
        metrics.cone_id_matches = cone_id_matches

        metrics.pairs = len(errors_axis)
        if metrics.pairs > 0:
            axis_arr = np.asarray(errors_axis, dtype=np.float32)
            metrics.axis_mae_m = float(np.mean(np.abs(axis_arr)))
            metrics.axis_rmse_m = float(np.sqrt(np.mean(np.square(axis_arr))))
            metrics.axis_bias_m = float(np.mean(axis_arr))
            range_arr = np.asarray(errors_range, dtype=np.float32)
            metrics.range_mae_m = float(np.mean(np.abs(range_arr)))
            metrics.range_rmse_m = float(np.sqrt(np.mean(np.square(range_arr))))
        return self._finalize_public_cone_eval(
            metrics=metrics,
            visible_rows=visible_rows,
            range_rmse_samples=range_rmse_samples,
            monocular_fit_samples=monocular_fit_samples,
            publish_results=publish_results,
            publish_range_rmse_samples=publish_range_rmse_samples,
        )

    def _init_yolo_detector(self):
        if not self.yolo_enabled:
            return None
        model_path = self._resolve_yolo_model_path(self.yolo_model_path)
        if not model_path or not os.path.isfile(model_path):
            self.get_logger().warn(
                f'YOLO model not found (yolo_model_path="{self.yolo_model_path}"); disabling YOLO.'
            )
            return None
        suffix = os.path.splitext(model_path)[1].lower()
        try:
            if suffix == '.pt':
                return YoloPtDetector(
                    logger=self.get_logger(),
                    model_path=model_path,
                    input_size=self.yolo_input_size,
                    conf_threshold=self.yolo_conf_threshold,
                    iou_threshold=self.yolo_iou_threshold,
                    max_detections=self.yolo_max_detections,
                    class_names=self.yolo_class_names,
                    prefer_cuda=self.yolo_prefer_cuda,
                )
            if suffix == '.onnx':
                return YoloOnnxDetector(
                    logger=self.get_logger(),
                    model_path=model_path,
                    input_size=self.yolo_input_size,
                    conf_threshold=self.yolo_conf_threshold,
                    iou_threshold=self.yolo_iou_threshold,
                    max_detections=self.yolo_max_detections,
                    class_names=self.yolo_class_names,
                    prefer_cuda=self.yolo_prefer_cuda,
                )
            self.get_logger().warn(
                f'Unsupported YOLO model extension "{suffix}" for "{model_path}"; expected .pt or .onnx.'
            )
            return None
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f'Failed to initialize YOLO detector ({exc}); disabling YOLO.')
            return None

    def _run_yolo(self, left_rect: np.ndarray) -> tuple[list[dict], float]:
        if self._yolo_detector is None:
            return [], 0.0
        try:
            detections, infer_ms = self._yolo_detector.detect(left_rect)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f'YOLO inference failed ({exc})')
            return [], 0.0

        overlays = []
        for det in detections:
            overlays.append(
                {
                    'x0': int(det.x0),
                    'y0': int(det.y0),
                    'x1': int(det.x1),
                    'y1': int(det.y1),
                    'confidence': float(det.confidence),
                    'class_id': int(det.class_id),
                    'label': str(det.label),
                }
            )
        return overlays, float(infer_ms)

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

    def _evaluate_cone_depth(
        self,
        depth: np.ndarray,
        left_info: Optional[CameraInfo],
        eval_header: Header,
        publish_results: bool = True,
        record_stats: bool = True,
        sample_source: str = 'stereo',
        rectification_rotation: Optional[np.ndarray] = None,
    ) -> tuple[ConeDepthMetrics, list[dict]]:
        metrics = ConeDepthMetrics()
        overlays = []
        visible_rows: list[VisibleConeMetrics] = []
        range_rmse_samples: list[tuple[str, float, float, Optional[int], Optional[int]]] = []
        if left_info is None:
            return self._finalize_optional_cone_eval(
                metrics=metrics,
                overlays=overlays,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                publish_results=publish_results,
            )

        height, width = depth.shape[:2]
        fx, fy, cx, cy = self._camera_intrinsics(left_info)
        if fx <= 0.0 or fy <= 0.0:
            return self._finalize_optional_cone_eval(
                metrics=metrics,
                overlays=overlays,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                publish_results=publish_results,
            )

        frame_stamp_sec = self._stamp_msg_to_sec(eval_header.stamp)
        cone_packet = self._get_nearest_cone_packet(frame_stamp_sec)
        if cone_packet is None:
            return self._finalize_optional_cone_eval(
                metrics=metrics,
                overlays=overlays,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                publish_results=publish_results,
            )

        metrics.sync_dt_ms = abs(frame_stamp_sec - cone_packet.stamp_sec) * 1000.0
        if metrics.sync_dt_ms > self.cone_eval_sync_slop_sec * 1000.0:
            return self._finalize_optional_cone_eval(
                metrics=metrics,
                overlays=overlays,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                publish_results=publish_results,
            )

        src_frame = cone_packet.msg.header.frame_id
        if not src_frame:
            return self._finalize_optional_cone_eval(
                metrics=metrics,
                overlays=overlays,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                publish_results=publish_results,
            )

        transform_bundle = self._resolve_transform_and_projection(
            source_frame=src_frame,
            left_info_frame=left_info.header.frame_id,
            stamp=eval_header.stamp,
        )
        if transform_bundle is None:
            return self._finalize_optional_cone_eval(
                metrics=metrics,
                overlays=overlays,
                visible_rows=visible_rows,
                range_rmse_samples=range_rmse_samples,
                publish_results=publish_results,
            )
        transform, projection_model = transform_bundle
        with self._track_lock:
            track_frame_id = self._track_frame_id
        track_transform = None
        track_coord_mode = 'none'
        if track_frame_id == src_frame:
            track_coord_mode = 'identity'
        else:
            track_transform = self._lookup_transform(
                target_frame=track_frame_id,
                source_frame=src_frame,
                stamp=eval_header.stamp,
            )
            if track_transform is not None:
                track_coord_mode = 'transform'
            elif (
                track_frame_id in {'map', 'odom'}
                and self._is_base_footprint_frame(src_frame)
            ):
                track_coord_mode = 'odom_fallback'

        errors_axis = []
        errors_range = []

        for cone_color, cone in self._iter_cones(cone_packet.msg):
            x_cam, y_cam, z_cam = self._transform_point(transform, cone.point.x, cone.point.y, cone.point.z)
            x_eval, y_eval, z_eval = x_cam, y_cam, z_cam
            if rectification_rotation is not None:
                x_eval, y_eval, z_eval = self._rotate_point(rectification_rotation, x_cam, y_cam, z_cam)
            projection = self._project_to_pixel(
                x_cam=x_eval,
                y_cam=y_eval,
                z_cam=z_eval,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                model=projection_model,
            )
            if projection is None:
                continue
            u, v, gt_axis = projection
            if u < 0.0 or v < 0.0 or u >= float(width) or v >= float(height):
                continue

            est_axis = self._sample_depth(depth, u, v, self.cone_eval_pixel_radius)
            if not np.isfinite(est_axis):
                continue

            err_axis = float(est_axis - gt_axis)
            est_cam = self._reconstruct_cam_point_from_axis(
                u=float(u),
                v=float(v),
                axis_depth=float(est_axis),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                model=projection_model,
            )
            if est_cam is None:
                continue
            est_x, est_y, est_z = est_cam
            if rectification_rotation is not None:
                est_x, est_y, est_z = self._rotate_point_transpose(rectification_rotation, est_x, est_y, est_z)
            err_x = est_x - float(x_cam)
            err_y = est_y - float(y_cam)
            predicted_class_id = self._cone_class_name_to_id(cone_color)
            ground_truth_class_id = predicted_class_id
            errors_axis.append(err_axis)
            gt_range = math.sqrt((x_cam * x_cam) + (y_cam * y_cam) + (z_cam * z_cam))
            ray_scale = math.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
            est_range = float(est_axis) * ray_scale
            self._record_range_rmse_sample(
                source=sample_source,
                gt_range_m=gt_range,
                error_m=math.hypot(err_x, err_y),
                predicted_class_id=predicted_class_id,
                ground_truth_class_id=ground_truth_class_id,
            )
            range_rmse_samples.append(
                (
                    sample_source,
                    gt_range,
                    math.hypot(err_x, err_y),
                    predicted_class_id,
                    ground_truth_class_id,
                )
            )
            cone_id = None
            if track_coord_mode == 'transform':
                x_map, y_map, _ = self._transform_point(
                    track_transform,
                    cone.point.x,
                    cone.point.y,
                    cone.point.z,
                )
                cone_id = self._match_cone_id(
                    x_map=x_map,
                    y_map=y_map,
                    color=cone_color,
                )
            elif track_coord_mode == 'identity':
                cone_id = self._match_cone_id(
                    x_map=float(cone.point.x),
                    y_map=float(cone.point.y),
                    color=cone_color,
                )
            elif track_coord_mode == 'odom_fallback':
                fallback_point = self._point_to_track_with_odom_fallback(
                    x=float(cone.point.x),
                    y=float(cone.point.y),
                    z=float(cone.point.z),
                    target_stamp_sec=frame_stamp_sec,
                )
                if fallback_point is not None:
                    x_map, y_map, _ = fallback_point
                    cone_id = self._match_cone_id(
                        x_map=x_map,
                        y_map=y_map,
                        color=cone_color,
                    )
            if cone_id is not None and record_stats:
                self._update_cone_stats(
                    cone_id=cone_id,
                    color=cone_color,
                    axis_err_m=err_axis,
                    err_x_m=err_x,
                    err_y_m=err_y,
                    dcam_m=est_range,
                    dgt_m=gt_range,
                )
                samples, mae_v, rmse_v, rmse_x_v, rmse_y_v, dcam_v, dgt_v = self._get_cone_stat_values(cone_id)
                mae = mae_v if mae_v is not None else abs(err_axis)
                rmse = rmse_v if rmse_v is not None else abs(err_axis)
                rmse_x = rmse_x_v if rmse_x_v is not None else abs(err_x)
                rmse_y = rmse_y_v if rmse_y_v is not None else abs(err_y)
                dcam = dcam_v if dcam_v is not None else est_range
                dgt = dgt_v if dgt_v is not None else gt_range
                overlays.append(
                    {
                        'u': float(u),
                        'v': float(v),
                        'color': cone_color,
                        'sort_depth': float(gt_axis),
                        'label': (
                            f'{cone_id} '
                            f'MAE:{mae:.2f}m '
                            f'RMSE:{rmse:.2f}m '
                            f'RMSE_x:{rmse_x:.2f}m '
                            f'RMSE_y:{rmse_y:.2f}m '
                            f'\n'
                            f'dcam:{dcam:.2f}m '
                            f'dgt:{dgt:.2f}m'
                        ),
                    }
                )
                visible_rows.append(
                    VisibleConeMetrics(
                        cone_id=cone_id,
                        color=cone_color,
                        samples=samples,
                        mae=mae,
                        rmse=rmse,
                        rmse_x=rmse_x,
                        rmse_y=rmse_y,
                        dcam_inst=est_range,
                        dgt_inst=gt_range,
                        dcam=dcam,
                        dgt=dgt,
                    )
                )

            errors_range.append(est_range - gt_range)

        metrics.pairs = len(errors_axis)
        if metrics.pairs > 0:
            axis_arr = np.asarray(errors_axis, dtype=np.float32)
            range_arr = np.asarray(errors_range, dtype=np.float32)
            metrics.axis_mae_m = float(np.mean(np.abs(axis_arr)))
            metrics.axis_rmse_m = float(np.sqrt(np.mean(np.square(axis_arr))))
            metrics.axis_bias_m = float(np.mean(axis_arr))
            metrics.range_mae_m = float(np.mean(np.abs(range_arr)))
            metrics.range_rmse_m = float(np.sqrt(np.mean(np.square(range_arr))))

        overlays = self._assign_overlay_placements(overlays, cx)
        return self._finalize_optional_cone_eval(
            metrics=metrics,
            overlays=overlays,
            visible_rows=visible_rows,
            range_rmse_samples=range_rmse_samples,
            publish_results=publish_results,
        )

    @staticmethod
    def _assign_overlay_placements(overlays: list[dict], image_cx: float) -> list[dict]:
        if not overlays:
            return overlays

        placement_cycle = ('bottom', 'left', 'right', 'top')
        left = []
        right = []
        for overlay in overlays:
            depth = float(overlay.get('sort_depth', float('inf')))
            u = float(overlay.get('u', 0.0))
            if u < image_cx:
                left.append((depth, overlay))
            else:
                right.append((depth, overlay))

        left.sort(key=lambda item: item[0])
        right.sort(key=lambda item: item[0])

        for idx in range(max(len(left), len(right))):
            placement = placement_cycle[idx % len(placement_cycle)]
            if idx < len(left):
                left[idx][1]['placement'] = PerceptionNode._side_safe_placement(
                    placement=placement,
                    side='left',
                )
            if idx < len(right):
                right[idx][1]['placement'] = PerceptionNode._side_safe_placement(
                    placement=placement,
                    side='right',
                )

        for overlay in overlays:
            overlay.pop('sort_depth', None)
            if 'placement' not in overlay:
                overlay['placement'] = 'right'
        return overlays

    def _finalize_public_cone_eval(
        self,
        metrics: ConeDepthMetrics,
        visible_rows: list[VisibleConeMetrics],
        range_rmse_samples: list[tuple[str, float, float, Optional[int], Optional[int]]],
        monocular_fit_samples: Optional[list[MonocularFitSample]] = None,
        publish_results: bool = True,
        publish_range_rmse_samples: bool = True,
    ) -> ConeDepthMetrics:
        if publish_range_rmse_samples:
            self._publish_range_rmse_samples(range_rmse_samples)
        self._publish_monocular_fit_samples(monocular_fit_samples or [])
        if publish_results:
            self._set_latest_visible_cones(visible_rows)
            self._publish_per_cone_table()
            self._publish_cone_metrics(metrics)
        return metrics

    def _finalize_optional_cone_eval(
        self,
        metrics: ConeDepthMetrics,
        overlays: list[dict],
        visible_rows: list[VisibleConeMetrics],
        range_rmse_samples: list[tuple[str, float, float, Optional[int], Optional[int]]],
        publish_results: bool,
    ) -> tuple[ConeDepthMetrics, list[dict]]:
        self._publish_range_rmse_samples(range_rmse_samples)
        if publish_results:
            self._set_latest_visible_cones(visible_rows)
            self._publish_per_cone_table()
            self._publish_cone_metrics(metrics)
        return metrics, overlays

    @staticmethod
    def _side_safe_placement(placement: str, side: str) -> str:
        p = str(placement).strip().lower()
        s = str(side).strip().lower()
        if s == 'left' and p == 'right':
            return 'top'
        if s == 'right' and p == 'left':
            return 'top'
        return p

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp):
        cache_key = (target_frame, source_frame)
        query_time = Time.from_msg(stamp)
        timeout = Duration(seconds=float(self.cone_eval_tf_timeout_sec))
        last_exc = None
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                query_time,
                timeout=timeout,
            )
            with self._tf_cache_lock:
                self._tf_cache[cache_key] = transform
            return transform
        except TransformException:
            pass
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=timeout,
            )
            with self._tf_cache_lock:
                self._tf_cache[cache_key] = transform
            return transform
        except TransformException as exc:
            last_exc = exc

        with self._tf_cache_lock:
            cached = self._tf_cache.get(cache_key)
        if cached is not None:
            return cached

        if last_exc is not None:
            self.get_logger().debug(
                f'cone depth eval transform failed {source_frame}->{target_frame}: {last_exc}'
            )
        return None

    def _resolve_transform_and_projection(self, source_frame: str, left_info_frame: str, stamp):
        explicit_target = self.cone_eval_target_frame.strip()
        fallback_target = self.cone_eval_fallback_frame.strip()

        candidates = []
        if explicit_target:
            candidates.append(explicit_target)
        if left_info_frame:
            candidates.append(left_info_frame)
        if fallback_target:
            candidates.append(fallback_target)

        seen = set()
        unique_candidates = []
        for frame in candidates:
            if frame in seen:
                continue
            seen.add(frame)
            unique_candidates.append(frame)

        for frame in unique_candidates:
            transform = self._lookup_transform(
                target_frame=frame,
                source_frame=source_frame,
                stamp=stamp,
            )
            if transform is None:
                continue

            model = self.cone_eval_projection_model
            if model == 'auto':
                frame_lower = frame.lower()
                if 'optical' in frame_lower or frame_lower.endswith('_camera'):
                    model = 'optical_z'
                elif frame_lower.endswith('_link'):
                    model = 'forward_x'
                else:
                    model = 'optical_z'
            return transform, model
        return None

    @staticmethod
    def _project_to_pixel(
        x_cam: float,
        y_cam: float,
        z_cam: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        model: str,
    ):
        if model == 'forward_x':
            if x_cam <= 0.0:
                return None
            u = (fx * (-y_cam / x_cam)) + cx
            v = (fy * (-z_cam / x_cam)) + cy
            return float(u), float(v), float(x_cam)

        # optical_z
        if z_cam <= 0.0:
            return None
        u = (fx * (x_cam / z_cam)) + cx
        v = (fy * (y_cam / z_cam)) + cy
        return float(u), float(v), float(z_cam)

    @staticmethod
    def _reconstruct_cam_point_from_axis(
        u: float,
        v: float,
        axis_depth: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        model: str,
    ):
        if not math.isfinite(axis_depth) or axis_depth <= 0.0 or fx == 0.0 or fy == 0.0:
            return None
        if model == 'forward_x':
            x_cam = axis_depth
            y_cam = -((u - cx) / fx) * x_cam
            z_cam = -((v - cy) / fy) * x_cam
            return float(x_cam), float(y_cam), float(z_cam)
        z_cam = axis_depth
        x_cam = ((u - cx) / fx) * z_cam
        y_cam = ((v - cy) / fy) * z_cam
        return float(x_cam), float(y_cam), float(z_cam)

    @staticmethod
    def _build_track_refs(cones, color: str):
        refs = []
        ordered = sorted(cones, key=lambda c: (float(c.point.x), float(c.point.y)))
        for idx, cone in enumerate(ordered):
            refs.append(
                TrackConeRef(
                    cone_id=f'{color}_{idx:03d}',
                    color=color,
                    x=float(cone.point.x),
                    y=float(cone.point.y),
                )
            )
        return refs

    def _match_cone_id(self, x_map: float, y_map: float, color: str) -> Optional[str]:
        with self._track_lock:
            track_refs = self._track_cones
        if not track_refs:
            return None

        # First try strict, color-consistent nearest-neighbor matching.
        strict = self._find_nearest_track_ref(
            track_refs=track_refs,
            x_map=x_map,
            y_map=y_map,
            color=color,
            threshold_m=self.cone_eval_track_match_threshold_m,
            allow_color_mismatch=False,
            require_unambiguous=False,
        )
        if strict is not None:
            return strict

        # When moving, fallback transforms can be noisier; use a relaxed but
        # ambiguity-guarded match before giving up.
        relaxed = self._find_nearest_track_ref(
            track_refs=track_refs,
            x_map=x_map,
            y_map=y_map,
            color=color,
            threshold_m=self.cone_eval_track_match_relaxed_threshold_m,
            allow_color_mismatch=False,
            require_unambiguous=True,
        )
        if relaxed is not None:
            return relaxed

        # Final fallback: allow cross-color only if it is still clearly unique.
        if color != 'unknown':
            return self._find_nearest_track_ref(
                track_refs=track_refs,
                x_map=x_map,
                y_map=y_map,
                color=color,
                threshold_m=self.cone_eval_track_match_relaxed_threshold_m,
                allow_color_mismatch=True,
                require_unambiguous=True,
            )
        return None

    @staticmethod
    def _find_nearest_track_ref(
        track_refs: list[TrackConeRef],
        x_map: float,
        y_map: float,
        color: str,
        threshold_m: float,
        allow_color_mismatch: bool,
        require_unambiguous: bool,
    ) -> Optional[str]:
        best_id = None
        best_dist_sq = None
        second_best_dist_sq = None
        threshold_sq = max(0.0, threshold_m) * max(0.0, threshold_m)
        for ref in track_refs:
            if (not allow_color_mismatch) and color != 'unknown' and ref.color != color:
                continue
            dx = ref.x - x_map
            dy = ref.y - y_map
            dist_sq = (dx * dx) + (dy * dy)
            if dist_sq > threshold_sq:
                continue
            if best_dist_sq is None or dist_sq < best_dist_sq:
                if best_dist_sq is not None:
                    second_best_dist_sq = best_dist_sq
                best_dist_sq = dist_sq
                best_id = ref.cone_id
            elif second_best_dist_sq is None or dist_sq < second_best_dist_sq:
                second_best_dist_sq = dist_sq

        if best_id is None:
            return None
        if not require_unambiguous:
            return best_id
        if second_best_dist_sq is None:
            return best_id

        # Accept only if the nearest candidate is clearly better than the next one.
        best_d = math.sqrt(best_dist_sq)
        second_d = math.sqrt(second_best_dist_sq)
        ratio = second_d / max(best_d, 1e-6)
        if ratio < 1.15 and (second_d - best_d) < 0.35:
            return None
        return best_id

    def _update_cone_stats(
        self,
        cone_id: str,
        color: str,
        axis_err_m: float,
        err_x_m: float,
        err_y_m: float,
        dcam_m: float,
        dgt_m: float,
    ):
        with self._cone_stats_lock:
            stats = self._cone_stats.get(cone_id)
            if stats is None:
                stats = ConeRunningStats(color=color)
                self._cone_stats[cone_id] = stats
            stats.add(
                err_m=axis_err_m,
                err_x_m=err_x_m,
                err_y_m=err_y_m,
                cam_depth_m=dcam_m,
                gt_depth_m=dgt_m,
            )

    def _get_cone_stat_values(self, cone_id: str):
        with self._cone_stats_lock:
            stats = self._cone_stats.get(cone_id)
            if stats is None:
                return 0, None, None, None, None, None, None
            return (
                stats.samples,
                stats.mae(),
                stats.rmse(),
                stats.rmse_x(),
                stats.rmse_y(),
                stats.dcam(),
                stats.dgt(),
            )

    @staticmethod
    def _is_base_footprint_frame(frame_id: str) -> bool:
        frame = frame_id.strip()
        return frame == 'base_footprint' or frame.endswith('/base_footprint')

    def _point_to_track_with_odom_fallback(
        self,
        x: float,
        y: float,
        z: float,
        target_stamp_sec: Optional[float] = None,
    ):
        with self._odom_lock:
            odom = self._nearest_odom_locked(target_stamp_sec)
        if odom is None:
            return None

        pose = odom.pose.pose
        return self._transform_point_from_pose(
            pose,
            x,
            y,
            z,
        )

    def _nearest_odom_locked(self, target_stamp_sec: Optional[float]) -> Optional[Odometry]:
        if self._latest_odom is None:
            return None
        if target_stamp_sec is None or not math.isfinite(target_stamp_sec):
            return self._latest_odom
        if not self._odom_queue:
            return self._latest_odom

        best_msg = None
        best_dt = None
        for stamp_sec, msg in self._odom_queue:
            dt = abs(stamp_sec - target_stamp_sec)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_msg = msg
        if best_msg is None:
            return self._latest_odom

        # If nearest odom sample is too far in time, prefer latest.
        if best_dt is not None and best_dt > 0.25:
            return self._latest_odom
        return best_msg

    def _per_cone_table_text(self) -> str:
        with self._visible_cones_lock:
            visible_items = list(self._latest_visible_cones)

        if not visible_items:
            return 'no per-cone depth samples yet'

        lines = [
            'cone_id,color,samples,axis_mae_m,axis_rmse_m,axis_rmse_x_m,axis_rmse_y_m,'
            'dcam_inst,dgt_inst,dcam,dgt'
        ]
        for idx, item in enumerate(visible_items):
            if idx >= self.cone_eval_per_cone_max_rows:
                remaining = len(visible_items) - idx
                lines.append(f'... ({remaining} more cones)')
                break
            mae_str = 'n/a' if item.mae is None else f'{item.mae:.4f}'
            rmse_str = 'n/a' if item.rmse is None else f'{item.rmse:.4f}'
            rmse_x_str = 'n/a' if item.rmse_x is None else f'{item.rmse_x:.4f}'
            rmse_y_str = 'n/a' if item.rmse_y is None else f'{item.rmse_y:.4f}'
            dcam_inst_str = f'{item.dcam_inst:.4f}'
            dgt_inst_str = f'{item.dgt_inst:.4f}'
            dcam_str = f'{item.dcam:.4f}'
            dgt_str = f'{item.dgt:.4f}'
            lines.append(
                f'{item.cone_id},{item.color},{item.samples},{mae_str},{rmse_str},'
                f'{rmse_x_str},{rmse_y_str},{dcam_inst_str},{dgt_inst_str},{dcam_str},{dgt_str}'
            )
        return '\n'.join(lines)

    def _set_latest_visible_cones(self, rows: list[VisibleConeMetrics]) -> None:
        def sort_dgt(row: VisibleConeMetrics) -> float:
            if math.isfinite(row.dgt_inst):
                return row.dgt_inst
            if math.isfinite(row.dgt):
                return row.dgt
            return math.inf

        def sort_dcam(row: VisibleConeMetrics) -> float:
            if math.isfinite(row.dcam_inst):
                return row.dcam_inst
            if math.isfinite(row.dcam):
                return row.dcam
            return math.inf

        best_by_id: dict[str, VisibleConeMetrics] = {}
        for row in rows:
            previous = best_by_id.get(row.cone_id)
            if previous is None or sort_dgt(row) < sort_dgt(previous):
                best_by_id[row.cone_id] = row
        ordered = sorted(best_by_id.values(), key=lambda row: (sort_dgt(row), sort_dcam(row), row.cone_id))
        with self._visible_cones_lock:
            self._latest_visible_cones = ordered

    def _publish_per_cone_table(self) -> None:
        self._cone_per_cone_pub.publish(String(data=self._per_cone_table_text()))

    def _get_nearest_cone_packet(self, target_stamp_sec: float) -> Optional[ConePacket]:
        with self._cone_lock:
            if not self._cone_queue:
                return None

            while len(self._cone_queue) > 2:
                if self._cone_queue[1].stamp_sec < target_stamp_sec - self.cone_eval_sync_slop_sec:
                    self._cone_queue.popleft()
                else:
                    break

            best_packet = None
            best_dt = None
            for packet in self._cone_queue:
                dt = abs(packet.stamp_sec - target_stamp_sec)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_packet = packet
            return best_packet

    def _publish_cone_metrics(self, metrics: ConeDepthMetrics):
        self._cone_pairs_pub.publish(Int32(data=int(metrics.pairs)))
        self._cone_yolo_detections_pub.publish(Int32(data=int(metrics.yolo_detections)))
        self._cone_yolo_depth_valid_pub.publish(Int32(data=int(metrics.yolo_depth_valid)))
        self._cone_gt_projected_pub.publish(Int32(data=int(metrics.gt_projected)))
        self._cone_bbox_matches_pub.publish(Int32(data=int(metrics.bbox_matches)))
        self._cone_id_matches_pub.publish(Int32(data=int(metrics.cone_id_matches)))
        if metrics.axis_mae_m is not None:
            self._cone_axis_mae_pub.publish(Float32(data=float(metrics.axis_mae_m)))
        if metrics.axis_rmse_m is not None:
            self._cone_axis_rmse_pub.publish(Float32(data=float(metrics.axis_rmse_m)))
        if metrics.axis_bias_m is not None:
            self._cone_axis_bias_pub.publish(Float32(data=float(metrics.axis_bias_m)))
        if metrics.range_mae_m is not None:
            self._cone_range_mae_pub.publish(Float32(data=float(metrics.range_mae_m)))
        if metrics.range_rmse_m is not None:
            self._cone_range_rmse_pub.publish(Float32(data=float(metrics.range_rmse_m)))
        if metrics.sync_dt_ms is not None:
            self._cone_sync_dt_pub.publish(Float32(data=float(metrics.sync_dt_ms)))

    def _publish_cone_detections(
        self,
        yolo_detections: list[dict],
        left_info: Optional[CameraInfo],
        eval_header: Header,
    ) -> None:
        msg = ConeDetectionArray()
        msg.header.stamp = eval_header.stamp
        msg.header.frame_id = self.cone_detections_frame

        if not yolo_detections or left_info is None:
            self._cone_detections_pub.publish(msg)
            return

        fx, fy, cx, cy = self._camera_intrinsics(left_info)
        if fx <= 0.0 or fy <= 0.0:
            self._cone_detections_pub.publish(msg)
            return

        camera_frame = str(left_info.header.frame_id).strip() or str(eval_header.frame_id).strip()
        if not camera_frame:
            self._cone_detections_pub.publish(msg)
            return

        projection_model = self._projection_model_for_frame(camera_frame)
        cam_to_output = None
        output_frame = self.cone_detections_frame
        transform_source_frame = camera_frame
        if camera_frame != self.cone_detections_frame:
            target_candidates = [self.cone_detections_frame]
            namespaced_frame = self._resolve_namespaced_output_frame(
                camera_frame=camera_frame,
                requested_frame=self.cone_detections_frame,
            )
            if namespaced_frame and namespaced_frame not in target_candidates:
                target_candidates.append(namespaced_frame)

            source_candidates = self._cone_output_source_frame_candidates(
                source_frame=camera_frame,
                requested_output_frame=self.cone_detections_frame,
            )

            for target_candidate in target_candidates:
                if cam_to_output is not None:
                    break
                for source_candidate in source_candidates:
                    candidate = self._lookup_transform(
                        target_frame=target_candidate,
                        source_frame=source_candidate,
                        stamp=eval_header.stamp,
                    )
                    if candidate is None:
                        continue
                    cam_to_output = candidate
                    output_frame = target_candidate
                    transform_source_frame = source_candidate
                    msg.header.frame_id = output_frame
                    break

            if cam_to_output is None:
                output_frame = camera_frame
                msg.header.frame_id = output_frame
                if not self._warned_cone_frame_fallback:
                    self.get_logger().warn(
                        'cone detections transform unavailable '
                        f'{camera_frame}->{self.cone_detections_frame}; '
                        f'publishing in source frame "{output_frame}"'
                    )
                    self._warned_cone_frame_fallback = True
            elif (transform_source_frame != camera_frame) and (not self._warned_cone_source_alias):
                self.get_logger().warn(
                    'cone detections using frame alias '
                    f'{transform_source_frame}->{output_frame} '
                    f'(original source {camera_frame})'
                )
                self._warned_cone_source_alias = True

        reconstruction_model = projection_model
        if transform_source_frame != camera_frame:
            aliased_model = self._projection_model_for_frame(transform_source_frame)
            if aliased_model != reconstruction_model:
                reconstruction_model = aliased_model
                if not self._warned_cone_projection_alias:
                    self.get_logger().warn(
                        'cone detections projection model override '
                        f'{projection_model}->{reconstruction_model} '
                        f'because transform source frame is "{transform_source_frame}" '
                        f'(original "{camera_frame}")'
                    )
                    self._warned_cone_projection_alias = True

        for det in yolo_detections:
            axis_depth = det.get('depth_m')
            if axis_depth is None or not np.isfinite(float(axis_depth)):
                continue

            u_center = det.get('u_center')
            v_center = det.get('v_center')
            if u_center is None or v_center is None:
                x0 = float(det.get('x0', -1))
                y0 = float(det.get('y0', -1))
                x1 = float(det.get('x1', -1))
                y1 = float(det.get('y1', -1))
                if x1 <= x0 or y1 <= y0:
                    continue
                u_center = 0.5 * (x0 + x1)
                v_center = 0.5 * (y0 + y1)

            cam_point = self._reconstruct_cam_point_from_axis(
                u=float(u_center),
                v=float(v_center),
                axis_depth=float(axis_depth),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                model=reconstruction_model,
            )
            if cam_point is None:
                continue

            x_out, y_out, z_out = cam_point
            if cam_to_output is not None:
                x_out, y_out, z_out = self._transform_point(
                    cam_to_output,
                    x_out,
                    y_out,
                    z_out,
                )

            cone = ConeDetection()
            cone.color = self._normalize_detection_color(str(det.get('label', '')))
            confidence = det.get('confidence')
            if confidence is None or not np.isfinite(float(confidence)):
                cone.confidence = 0.0
            else:
                cone.confidence = float(max(0.0, min(1.0, float(confidence))))
            cone.position.x = float(x_out)
            cone.position.y = float(y_out)
            cone.position.z = float(z_out)
            msg.cones.append(cone)

        self._cone_detections_pub.publish(msg)

    @staticmethod
    def _camera_intrinsics(camera_info: CameraInfo):
        fx = float(camera_info.k[0]) if len(camera_info.k) >= 1 else 0.0
        fy = float(camera_info.k[4]) if len(camera_info.k) >= 5 else 0.0
        cx = float(camera_info.k[2]) if len(camera_info.k) >= 3 else 0.0
        cy = float(camera_info.k[5]) if len(camera_info.k) >= 6 else 0.0
        return fx, fy, cx, cy

    @staticmethod
    def _sample_depth(depth: np.ndarray, u: float, v: float, radius_px: int) -> float:
        u_i = int(round(u))
        v_i = int(round(v))
        height, width = depth.shape[:2]
        if u_i < 0 or v_i < 0 or u_i >= width or v_i >= height:
            return float('nan')

        if radius_px <= 0:
            value = float(depth[v_i, u_i])
            return value if np.isfinite(value) else float('nan')

        u0 = max(0, u_i - radius_px)
        u1 = min(width, u_i + radius_px + 1)
        v0 = max(0, v_i - radius_px)
        v1 = min(height, v_i + radius_px + 1)
        patch = depth[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch)]
        if valid.size == 0:
            return float('nan')
        return float(np.median(valid))

    @staticmethod
    def _transform_point(transform, x: float, y: float, z: float):
        t = transform.transform.translation
        q = transform.transform.rotation

        qx = float(q.x)
        qy = float(q.y)
        qz = float(q.z)
        qw = float(q.w)

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)
        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)

        tx = float(t.x)
        ty = float(t.y)
        tz = float(t.z)

        px = (r00 * x) + (r01 * y) + (r02 * z) + tx
        py = (r10 * x) + (r11 * y) + (r12 * z) + ty
        pz = (r20 * x) + (r21 * y) + (r22 * z) + tz
        return px, py, pz

    @staticmethod
    def _rotate_point(rotation: np.ndarray, x: float, y: float, z: float) -> tuple[float, float, float]:
        point = np.asarray([x, y, z], dtype=np.float64)
        rotated = rotation @ point
        return float(rotated[0]), float(rotated[1]), float(rotated[2])

    @staticmethod
    def _rotate_point_transpose(rotation: np.ndarray, x: float, y: float, z: float) -> tuple[float, float, float]:
        point = np.asarray([x, y, z], dtype=np.float64)
        rotated = rotation.T @ point
        return float(rotated[0]), float(rotated[1]), float(rotated[2])

    def _warn_throttled(self, key: str, message: str) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get(key, -1.0)
        if (now_sec - last_sec) >= 1.0:
            self.get_logger().warn(message)
            self._last_throttled_log_sec[key] = now_sec

    @staticmethod
    def _transform_point_from_pose(pose, x: float, y: float, z: float):
        q = pose.orientation

        qx = float(q.x)
        qy = float(q.y)
        qz = float(q.z)
        qw = float(q.w)

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)
        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)

        t = pose.position
        tx = float(t.x)
        ty = float(t.y)
        tz = float(t.z)

        px = (r00 * x) + (r01 * y) + (r02 * z) + tx
        py = (r10 * x) + (r11 * y) + (r12 * z) + ty
        pz = (r20 * x) + (r21 * y) + (r22 * z) + tz
        return px, py, pz

    def _iter_cones(self, msg: ConeArrayWithCovariance):
        for cone in msg.blue_cones:
            yield 'blue', cone
        for cone in msg.yellow_cones:
            yield 'yellow', cone
        for cone in msg.orange_cones:
            yield 'orange', cone
        for cone in msg.big_orange_cones:
            yield 'big_orange', cone
        if self.cone_eval_include_unknown:
            for cone in msg.unknown_color_cones:
                yield 'unknown', cone

    def _perf_timer_cb(self):
        if self._perf is not None and self._evaluator is not None:
            self._perf.log_and_publish(self._evaluator.snapshot())
        self._publish_per_cone_table()
        m = self._latest_cone_metrics
        self.get_logger().info(
            'cone depth eval '
            f'pairs={m.pairs} '
            f'yolo_det={m.yolo_detections} '
            f'depth_valid={m.yolo_depth_valid} '
            f'gt_proj={m.gt_projected} '
            f'bbox_matches={m.bbox_matches}'
        )

    @staticmethod
    def _fmt_opt(value: Optional[float]) -> str:
        if value is None:
            return 'n/a'
        return f'{value:.4f}'

    def _record_range_rmse_sample(
        self,
        source: str,
        gt_range_m: float,
        error_m: float,
        predicted_class_id: Optional[int] = None,
        ground_truth_class_id: Optional[int] = None,
    ) -> None:
        if self._range_rmse_analyzer is None:
            return
        self._range_rmse_analyzer.add_sample(
            source=source,
            gt_range_m=gt_range_m,
            error_m=error_m,
            predicted_class_id=predicted_class_id,
            ground_truth_class_id=ground_truth_class_id,
        )

    def _publish_range_rmse_samples(
        self,
        samples: list[tuple[str, float, float, Optional[int], Optional[int]]],
    ) -> None:
        if not self.cone_plotting_2:
            return
        if not samples:
            return
        lines = ['source,gt_range_m,error_m,predicted_class_id,ground_truth_class_id']
        for source, gt_range_m, error_m, predicted_class_id, ground_truth_class_id in samples:
            if not source:
                continue
            if not (math.isfinite(gt_range_m) and math.isfinite(error_m)):
                continue
            predicted_str = '' if predicted_class_id is None else str(int(predicted_class_id))
            gt_str = '' if ground_truth_class_id is None else str(int(ground_truth_class_id))
            lines.append(f'{source},{gt_range_m:.6f},{error_m:.6f},{predicted_str},{gt_str}')
        if len(lines) <= 1:
            return
        self._cone_sample_pub.publish(String(data='\n'.join(lines)))

    def _publish_monocular_fit_samples(self, samples: list[MonocularFitSample]) -> None:
        if not samples:
            return
        fieldnames = list(MonocularFitSample.__dataclass_fields__.keys())
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = {}
            for key in fieldnames:
                value = getattr(sample, key)
                row[key] = '' if value is None else value
            writer.writerow(row)
        self._cone_mono_fit_sample_pub.publish(String(data=buffer.getvalue().strip()))

    def _update_range_rmse_plot(self) -> None:
        if self._range_rmse_analyzer is None or self._range_rmse_plot is None:
            return
        stats = self._range_rmse_analyzer.compute_binned_rmse()
        is_open = self._range_rmse_plot.update(stats)
        if not is_open:
            self._range_rmse_plot = None

    @staticmethod
    def _resolve_yolo_model_path(path: str) -> str:
        candidate = str(path).strip()
        if not candidate:
            return ''
        if os.path.isabs(candidate):
            return candidate
        direct = os.path.abspath(candidate)
        if os.path.exists(direct):
            return direct
        workspace_relative = os.path.abspath(os.path.join(os.path.expanduser('~/ros2_ws'), candidate))
        if os.path.exists(workspace_relative):
            return workspace_relative
        return direct

    @staticmethod
    def _sanitize_yolo_class_names(value) -> list[str]:
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        if not text:
            return []
        return [token.strip() for token in text.split(',') if token.strip()]

    @staticmethod
    def _stamp_to_sec(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _stamp_msg_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _sanitize_camera_debug(value) -> bool:
        token = str(value).strip().lower()
        if token in {'', 'false', '0', 'off', 'none', 'no'}:
            return False
        if token in {'rect_left', 'left_rect', 'depth', 'disparity', 'yolo'}:
            return True
        return token in {'true', '1', 'on', 'yes'}

    @staticmethod
    def _sanitize_projection_model(value) -> str:
        mode = str(value).strip().lower()
        if mode in {'auto', 'optical_z', 'forward_x'}:
            return mode
        return 'auto'

    @staticmethod
    def _projection_model_for_frame(frame_id: str) -> str:
        frame = str(frame_id).strip().lower()
        if 'optical' in frame or frame.endswith('_camera'):
            return 'optical_z'
        if frame.endswith('_link'):
            return 'forward_x'
        return 'optical_z'

    @staticmethod
    def _normalize_detection_color(label: str) -> str:
        token = str(label).strip().lower().replace('-', '_').replace(' ', '_')
        if 'big_orange' in token or ('big' in token and 'orange' in token):
            return 'big_orange'
        if 'orange' in token:
            return 'orange'
        if 'yellow' in token:
            return 'yellow'
        if 'blue' in token:
            return 'blue'
        return 'unknown'

    @classmethod
    def _cone_class_name_to_id(cls, value: str) -> int:
        normalized = cls._normalize_detection_color(value)
        return int(cls._CONE_CLASS_NAME_TO_ID.get(normalized, cls._CONE_CLASS_NAME_TO_ID['unknown']))

    def _detection_to_class_id(self, det: dict) -> int:
        class_id = det.get('class_id')
        if class_id is not None:
            try:
                idx = int(class_id)
            except (TypeError, ValueError):
                idx = -1
            if 0 <= idx < len(self.yolo_class_names):
                return self._cone_class_name_to_id(self.yolo_class_names[idx])
        return self._cone_class_name_to_id(str(det.get('label', '')))

    @staticmethod
    def _resolve_namespaced_output_frame(camera_frame: str, requested_frame: str) -> str:
        requested = str(requested_frame).strip().strip('/')
        source = str(camera_frame).strip().strip('/')
        if not requested or not source:
            return ''
        if '/' in requested:
            return ''
        marker = f'/{requested}/'
        source_with_slashes = f'/{source}/'
        idx = source_with_slashes.find(marker)
        if idx < 0:
            return ''
        prefix = source_with_slashes[1:idx].strip('/')
        if not prefix:
            return requested
        return f'{prefix}/{requested}'

    @classmethod
    def _cone_output_source_frame_candidates(
        cls,
        source_frame: str,
        requested_output_frame: str,
    ) -> list[str]:
        source = str(source_frame).strip().strip('/')
        if not source:
            return []

        candidates: list[str] = [source]
        if '/' in source:
            parts = [p for p in source.split('/') if p]
            if parts:
                leaf = parts[-1]
                if leaf not in candidates:
                    candidates.append(leaf)
                namespace_leaf = f'{parts[0]}/{leaf}'
                if namespace_leaf not in candidates:
                    candidates.append(namespace_leaf)

            requested = str(requested_output_frame).strip().strip('/')
            marker = f'/{requested}/' if requested else ''
            source_with_slashes = f'/{source}/'
            if marker and marker in source_with_slashes:
                idx = source_with_slashes.find(marker)
                prefix = source_with_slashes[1:idx].strip('/')
                suffix_start = idx + len(marker)
                suffix = source_with_slashes[suffix_start:-1].strip('/')
                if prefix and suffix:
                    prefixed_suffix = f'{prefix}/{suffix}'
                    if prefixed_suffix not in candidates:
                        candidates.append(prefixed_suffix)

        expanded = list(candidates)
        for token in candidates:
            if token.endswith('_camera'):
                link_token = token[:-7] + '_link'
                if link_token not in expanded:
                    expanded.append(link_token)
        return expanded

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

    def destroy_node(self):
        with self._queue_cv:
            self._running = False
            self._queue_cv.notify_all()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if self._range_rmse_plot is not None:
            self._range_rmse_plot.close()
            self._range_rmse_plot = None
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
