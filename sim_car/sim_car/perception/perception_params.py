"""Parameter declaration and loading for the camera perception node."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerceptionConfig:
    left_image_topic: str
    right_image_topic: str
    left_camera_info_topic: str
    right_camera_info_topic: str
    stereo_enabled: bool
    monocular_cone_height_m: float
    monocular_big_cone_height_m: float
    monocular_bbox_height_offset_px: float
    calibration_file: str
    max_time_diff_sec: float
    queue_size: int
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
    camera_debug: bool
    camera_debug_topic: str
    camera_debug_n_frames: int
    yolo_enabled: bool
    yolo_model_path: str
    yolo_input_size: int
    yolo_conf_threshold: float
    yolo_iou_threshold: float
    yolo_max_detections: int
    yolo_prefer_cuda: bool
    yolo_class_names: list[str]
    cone_detections_topic: str
    cone_detections_frame: str
    cone_dedup_radius_m: float
    cone_eval_tf_timeout_sec: float


def declare_parameters(node) -> None:
    node.declare_parameter('left_image_topic', '/sim/raw/stereo/left/image_raw')
    node.declare_parameter('right_image_topic', '/sim/raw/stereo/right/image_raw')
    node.declare_parameter('left_camera_info_topic', '/sim/raw/stereo/left/camera_info')
    node.declare_parameter('right_camera_info_topic', '/sim/raw/stereo/right/camera_info')
    node.declare_parameter('stereo_enabled', True)
    node.declare_parameter('monocular_cone_height_m', 0.3034)
    node.declare_parameter('monocular_big_cone_height_m', 0.51)
    node.declare_parameter('monocular_bbox_height_offset_px', 0.0)

    node.declare_parameter('calibration_file', '')
    node.declare_parameter('max_time_diff_sec', 0.08)
    node.declare_parameter('queue_size', 30)
    node.declare_parameter('prefer_cuda', True)

    node.declare_parameter('min_disparity', 0)
    node.declare_parameter('num_disparities', 192)
    node.declare_parameter('block_size', 7)
    node.declare_parameter('uniqueness_ratio', 10)
    node.declare_parameter('speckle_window_size', 100)
    node.declare_parameter('speckle_range', 2)
    node.declare_parameter('disp12_max_diff', 1)
    node.declare_parameter('pre_filter_cap', 31)

    node.declare_parameter('baseline_m', 0.12)
    node.declare_parameter('focal_length_px', 0.0)
    node.declare_parameter('disparity_valid_threshold', 0.1)
    node.declare_parameter('min_depth_m', 0.3)
    node.declare_parameter('max_depth_m', 30.0)

    node.declare_parameter('camera_debug', True)
    node.declare_parameter('camera_debug_topic', '/sim/raw/stereo/camera_debug')
    node.declare_parameter('camera_debug_n_frames', 30)

    node.declare_parameter('yolo_enabled', False)
    node.declare_parameter('yolo_model_path', '')
    node.declare_parameter('yolo_input_size', 960)
    node.declare_parameter('yolo_conf_threshold', 0.25)
    node.declare_parameter('yolo_iou_threshold', 0.45)
    node.declare_parameter('yolo_max_detections', 100)
    node.declare_parameter('yolo_prefer_cuda', True)
    node.declare_parameter('yolo_class_names', [])

    node.declare_parameter('cone_detections_topic', '/sim/raw/stereo/perception/cones_3d')
    node.declare_parameter('cone_detections_frame', 'base_footprint')
    node.declare_parameter('cone_dedup_radius_m', 0.85)
    node.declare_parameter('cone_eval_tf_timeout_sec', 0.0)


def load_parameters(node) -> PerceptionConfig:
    return PerceptionConfig(
        left_image_topic=str(node.get_parameter('left_image_topic').value),
        right_image_topic=str(node.get_parameter('right_image_topic').value),
        left_camera_info_topic=str(node.get_parameter('left_camera_info_topic').value),
        right_camera_info_topic=str(node.get_parameter('right_camera_info_topic').value),
        stereo_enabled=bool(node.get_parameter('stereo_enabled').value),
        monocular_cone_height_m=max(1e-6, float(node.get_parameter('monocular_cone_height_m').value)),
        monocular_big_cone_height_m=max(1e-6, float(node.get_parameter('monocular_big_cone_height_m').value)),
        monocular_bbox_height_offset_px=float(node.get_parameter('monocular_bbox_height_offset_px').value),
        calibration_file=str(node.get_parameter('calibration_file').value),
        max_time_diff_sec=max(0.0, float(node.get_parameter('max_time_diff_sec').value)),
        queue_size=max(5, int(node.get_parameter('queue_size').value)),
        prefer_cuda=bool(node.get_parameter('prefer_cuda').value),
        min_disparity=int(node.get_parameter('min_disparity').value),
        num_disparities=int(node.get_parameter('num_disparities').value),
        block_size=int(node.get_parameter('block_size').value),
        uniqueness_ratio=int(node.get_parameter('uniqueness_ratio').value),
        speckle_window_size=int(node.get_parameter('speckle_window_size').value),
        speckle_range=int(node.get_parameter('speckle_range').value),
        disp12_max_diff=int(node.get_parameter('disp12_max_diff').value),
        pre_filter_cap=int(node.get_parameter('pre_filter_cap').value),
        baseline_m=float(node.get_parameter('baseline_m').value),
        focal_length_px=float(node.get_parameter('focal_length_px').value),
        disparity_valid_threshold=float(node.get_parameter('disparity_valid_threshold').value),
        min_depth_m=float(node.get_parameter('min_depth_m').value),
        max_depth_m=float(node.get_parameter('max_depth_m').value),
        camera_debug=_sanitize_camera_debug(node.get_parameter('camera_debug').value),
        camera_debug_topic=str(node.get_parameter('camera_debug_topic').value),
        camera_debug_n_frames=max(1, int(node.get_parameter('camera_debug_n_frames').value)),
        yolo_enabled=bool(node.get_parameter('yolo_enabled').value),
        yolo_model_path=str(node.get_parameter('yolo_model_path').value),
        yolo_input_size=max(64, int(node.get_parameter('yolo_input_size').value)),
        yolo_conf_threshold=float(node.get_parameter('yolo_conf_threshold').value),
        yolo_iou_threshold=float(node.get_parameter('yolo_iou_threshold').value),
        yolo_max_detections=max(1, int(node.get_parameter('yolo_max_detections').value)),
        yolo_prefer_cuda=bool(node.get_parameter('yolo_prefer_cuda').value),
        yolo_class_names=_sanitize_yolo_class_names(node.get_parameter('yolo_class_names').value),
        cone_detections_topic=str(node.get_parameter('cone_detections_topic').value),
        cone_detections_frame=str(node.get_parameter('cone_detections_frame').value).strip() or 'base_footprint',
        cone_dedup_radius_m=max(0.01, float(node.get_parameter('cone_dedup_radius_m').value)),
        cone_eval_tf_timeout_sec=max(0.0, float(node.get_parameter('cone_eval_tf_timeout_sec').value)),
    )


def _sanitize_camera_debug(value) -> bool:
    token = str(value).strip().lower()
    if token in {'', 'false', '0', 'off', 'none', 'no'}:
        return False
    return token in {'true', '1', 'on', 'yes', 'rect_left', 'left_rect', 'depth', 'disparity', 'yolo'}


def _sanitize_yolo_class_names(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [token.strip() for token in text.split(',') if token.strip()]
