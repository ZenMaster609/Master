# Perception Code Map

This page maps the `documentation/concepts/perception.md` behavior to the camera perception node, depth helpers, and YOLO backend adapters.

## Primary Files

- `sim_car/sim_car/perception/perception_node.py`
- `sim_car/sim_car/perception/perception_params.py`
- `sim_car/sim_car/perception/detection_depth.py`
- `sim_car/sim_car/perception/cone_geometry.py`

## Function Map

### Runtime Flow

- `PerceptionNode` in `sim_car/sim_car/perception/perception_node.py`: owns the subscriptions, worker loop, YOLO calls, depth attachment, TF handling, and `ConeDetectionArray` publication.
- `PerceptionNode._publish_cone_detections` in `sim_car/sim_car/perception/perception_node.py`: converts detector outputs into planner-facing 3D cone detections.
- `declare_parameters` and `load_parameters` in `sim_car/sim_car/perception/perception_params.py`: declare and load the node’s runtime configuration.

### YOLO Detection

- `init_yolo_detector` in `sim_car/sim_car/perception/yolo_runtime.py`: selects the `.pt` or `.onnx` detector backend from the configured model path.
- `run_yolo` in `sim_car/sim_car/perception/yolo_runtime.py`: executes the active detector and normalizes backend output into the node’s expected detection dicts.
- `YoloPtDetector.detect` in `sim_car/sim_car/perception/yolo_pt.py`: Torch/Ultralytics detector path for `.pt` models.
- `YoloOnnxDetector.detect` in `sim_car/sim_car/perception/yolo_onnx.py`: OpenCV DNN detector path for `.onnx` models.

### Monocular Mode

- `PerceptionNode._process_monocular_frame` in `sim_car/sim_car/perception/perception_node.py`: runs the left-image monocular pipeline end to end.
- `estimate_axis_depth_from_bbox_height` in `sim_car/sim_car/perception/monocular_depth.py`: implements the bounding-box-height depth estimate used in monocular mode.
- `apply_monocular_depth_to_detections` in `sim_car/sim_car/perception/detection_depth.py`: attaches monocular range estimates to YOLO detections.
- `reconstruct_cam_point_from_axis` in `sim_car/sim_car/perception/cone_geometry.py`: converts image-space cone axis and depth into a camera-frame 3D point.

### Stereo Mode

- `PerceptionNode._enqueue_frame` and `PerceptionNode._pair_frames` in `sim_car/sim_car/perception/perception_node.py`: buffer and pair left/right frames by timestamp.
- `PerceptionNode._worker_loop` in `sim_car/sim_car/perception/perception_node.py`: background processing loop that runs stereo jobs without blocking the image callbacks.
- `StereoPipeline` in `sim_car/sim_car/perception/stereo_pipeline.py`: performs rectification, disparity generation, and depth-map production.
- `apply_depth_map_to_detections` in `sim_car/sim_car/perception/detection_depth.py`: samples stereo depth for each YOLO detection.

### Cone Geometry And Frame Output

- `camera_intrinsics` in `sim_car/sim_car/perception/cone_geometry.py`: extracts the camera model terms used by the geometry functions.
- `transform_point` in `sim_car/sim_car/perception/cone_geometry.py`: applies TF transforms to reconstructed cone points.
- `deduplicate_cone_candidates` in `sim_car/sim_car/perception/cone_geometry.py`: merges duplicate cone candidates before publication.
- `resolve_namespaced_output_frame` and `cone_output_source_frame_candidates` in `sim_car/sim_car/perception/cone_geometry.py`: help the node cope with namespaced camera frames and output-frame requests.
- `PerceptionNode._lookup_transform` in `sim_car/sim_car/perception/perception_node.py`: performs the TF lookup used during output-frame conversion.

### Debug Output

- `PerceptionNode._publish_debug_image` in `sim_car/sim_car/perception/perception_node.py`: publishes rendered detection overlays when camera debug is enabled.
- `_sanitize_camera_debug` in `sim_car/sim_car/perception/perception_params.py`: normalizes the camera-debug flag from parameters.

## Related Entry Points

- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: wires image topics, camera info topics, model path, overlay paths, and stereo/mono launch choices.
- `resolve_yolo_model_path` in `sim_car/sim_car/perception/yolo_runtime.py`: normalizes the configured model path before backend selection.
