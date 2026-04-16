#!/usr/bin/env python3
"""Local cone memory tracker with optional global track-belief export."""

from __future__ import annotations

from collections import deque
from builtin_interfaces.msg import Time as TimeMsg
import math
import time
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray, RunSession
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cones.nodes._memory_csv import resolve_logs_dir, save_track_data_csv
from sim_car.cones.nodes._memory_viz import (
    make_cone_marker,
    make_id_marker,
    make_line_marker,
    make_raw_sensor_markers,
    split_polyline_by_gap,
)
from sim_car.cones.nodes.memory_types import SensorDetection
from sim_car.cones.tracking.fusion import (
    choose_position_source,
    clamp_camera_range,
    normalize_color,
    resolve_boundary_colors_for_planning,
)
from sim_car.cones.tracking.pose import base_point_to_odom, convert_odom_child_pose_to_base_frame, odom_point_to_base
from sim_car.cones.tracking.tracker import (
    TRACK_STATE_CONFIRMED,
    ConeTrack,
    LocalConeTracker,
    TrackUpdate,
)


class ConeMemoryNode(Node):
    """Fuse lidar+camera cone detections into stable local tracks."""

    def __init__(self) -> None:
        super().__init__('cone_memory_node')
        self._declare_parameters()
        self._read_parameters()

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._last_throttled_log_sec: dict[str, float] = {}

        self._local_tracker = LocalConeTracker()
        self._latest_lidar: list[SensorDetection] = []
        self._latest_camera: list[SensorDetection] = []
        self._latest_lidar_stamp_ns: int = -1
        self._latest_camera_stamp_ns: int = -1
        self._last_processed_lidar_stamp_ns: int = -1
        self._last_processed_camera_stamp_ns: int = -1
        self._last_tracker_update_stamp: Optional[TimeMsg] = None
        self._latest_odom_msg: Optional[Odometry] = None
        self._recent_odom_msgs: deque[Odometry] = deque(maxlen=512)
        self._last_plot_save_attempted = False
        self._run_session_run_id = ''
        self._run_session_base_path = ''

        self._tracked_pub = self.create_publisher(ConeDetectionArray, self.tracked_cones_topic, 10)
        self._viz_pub = self.create_publisher(MarkerArray, self.viz_topic, 10)
        self._track_viz_pub = self.create_publisher(MarkerArray, self.believed_track_viz_topic, 10)
        self._raw_lidar_viz_pub = self.create_publisher(MarkerArray, f'{self.viz_topic}/raw_lidar', 10)
        self._raw_camera_viz_pub = self.create_publisher(MarkerArray, f'{self.viz_topic}/raw_camera', 10)

        self.create_subscription(
            ConeDetectionArray,
            self.lidar_cones_topic,
            self._lidar_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            ConeDetectionArray,
            self.camera_cones_topic,
            self._camera_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(RunSession, self.run_session_topic, self._run_session_cb, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos_profile_sensor_data)

        self.create_timer(1.0 / max(1.0, self.publish_rate_hz), self._on_timer)

        self.get_logger().info(
            'cone_memory_node ready '
            f'lidar={self.lidar_cones_topic} camera={self.camera_cones_topic} '
            f'tracked={self.tracked_cones_topic} viz={self.viz_topic} '
            f'track_viz={self.believed_track_viz_topic} '
            f'camera_range_m={self.camera_range_m:.2f}'
        )

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'front_axle')
        self.declare_parameter('tf_timeout_s', 0.03)
        self.declare_parameter('odom_pose_sync_tolerance_sec', 0.05)
        self.declare_parameter('base_pose_latest_fallback_tolerance_sec', 0.02)
        self.declare_parameter('wheelbase_m', 1.65)

        self.declare_parameter('max_range_m', 25.0)
        self.declare_parameter('behind_drop_m', 8.0)
        self.declare_parameter('ttl_sec', 2.0)
        self.declare_parameter('tentative_ttl_sec', 1.0)
        self.declare_parameter('stale_after_sec', 0.6)
        self.declare_parameter('confirmed_prune_after_sec', 3.0)
        self.declare_parameter('unknown_drop_frames', 15)
        self.declare_parameter('gate_radius_m', 0.5)
        self.declare_parameter('spawn_radius_m', 0.85)
        self.declare_parameter('dedup_radius_m', 0.85)
        self.declare_parameter('track_merge_radius_m', 0.85)
        self.declare_parameter('track_merge_longitudinal_tolerance_m', 2.50)
        self.declare_parameter('track_merge_lateral_tolerance_m', 0.55)
        self.declare_parameter('min_seen_count', 2)
        self.declare_parameter('confirm_hits', 0)
        self.declare_parameter('alpha_lidar', 0.4)
        self.declare_parameter('alpha_camera', 0.2)
        self.declare_parameter('track_confidence_gain', 0.25)
        self.declare_parameter('track_confidence_decay', 0.10)
        self.declare_parameter('color_decay', 0.05)
        self.declare_parameter('color_switch_margin', 0.20)
        self.declare_parameter('publish_stale_tracks', True)
        self.declare_parameter('stale_planner_ttl_sec', 1.5)

        self.declare_parameter('camera_range_m', 0.0)
        self.declare_parameter('prefer_lidar_if_camera_missing_far', True)
        self.declare_parameter('allow_camera_fallback_near', False)

        self.declare_parameter('publish_rate_hz', 60.0)
        self.declare_parameter('enable_id_text', False)
        self.declare_parameter('enable_tentative_viz', False)
        self.declare_parameter('enable_raw_debug_viz', False)

        self.declare_parameter('enable_global_track_memory', True)
        self.declare_parameter('global_merge_radius_m', 0.6)
        self.declare_parameter('global_min_hits_for_track', 2)
        self.declare_parameter('global_max_cones', 2000)
        self.declare_parameter('believed_track_viz_min_hits', 3)
        self.declare_parameter('believed_track_viz_min_confidence', 0.45)
        self.declare_parameter('believed_track_viz_max_gap_m', 6.0)
        self.declare_parameter('believed_track_viz_show_center', False)
        self.declare_parameter('believed_track_viz_show_cones', False)

        self.declare_parameter('save_track_plot_on_shutdown', True)
        self.declare_parameter('track_plot_data_filename', 'cone_memory_track_data.csv')

        self.declare_parameter('lidar_cones_topic', '/sim/lidar/perception/cones_3d')
        self.declare_parameter('camera_cones_topic', '/sim/stereo/perception/cones_3d')
        self.declare_parameter('tracked_cones_topic', '/tracked_cones')
        self.declare_parameter('viz_topic', '/local_cone_map_viz')
        self.declare_parameter('believed_track_viz_topic', '/cone_memory/believed_track_viz')
        self.declare_parameter('run_session_topic', '/run_session')
        self.declare_parameter('odom_topic', '/sim/odom')

    def _read_parameters(self) -> None:
        self.odom_frame = str(self.get_parameter('odom_frame').value).strip() or 'odom'
        self.base_frame = str(self.get_parameter('base_frame').value).strip() or 'front_axle'
        self.tf_timeout_s = max(0.0, float(self.get_parameter('tf_timeout_s').value))
        self.odom_pose_sync_tolerance_sec = max(
            0.0, float(self.get_parameter('odom_pose_sync_tolerance_sec').value)
        )
        self.base_pose_latest_fallback_tolerance_sec = max(
            0.0, float(self.get_parameter('base_pose_latest_fallback_tolerance_sec').value)
        )
        self.wheelbase_m = max(0.1, float(self.get_parameter('wheelbase_m').value))

        self.max_range_m = max(1.0, float(self.get_parameter('max_range_m').value))
        self.behind_drop_m = max(0.0, float(self.get_parameter('behind_drop_m').value))
        self.ttl_sec = max(0.05, float(self.get_parameter('ttl_sec').value))
        self.tentative_ttl_sec = max(0.05, float(self.get_parameter('tentative_ttl_sec').value))
        self.stale_after_sec = max(0.0, float(self.get_parameter('stale_after_sec').value))
        self.confirmed_prune_after_sec = max(
            self.ttl_sec, float(self.get_parameter('confirmed_prune_after_sec').value)
        )
        self.unknown_drop_frames = max(0, int(self.get_parameter('unknown_drop_frames').value))
        self.gate_radius_m = max(0.05, float(self.get_parameter('gate_radius_m').value))
        self.spawn_radius_m = max(self.gate_radius_m, float(self.get_parameter('spawn_radius_m').value))
        self.dedup_radius_m = max(0.01, float(self.get_parameter('dedup_radius_m').value))
        self.track_merge_radius_m = max(0.01, float(self.get_parameter('track_merge_radius_m').value))
        self.track_merge_longitudinal_tolerance_m = max(
            self.track_merge_radius_m,
            float(self.get_parameter('track_merge_longitudinal_tolerance_m').value),
        )
        self.track_merge_lateral_tolerance_m = max(
            0.01, float(self.get_parameter('track_merge_lateral_tolerance_m').value)
        )
        self.min_seen_count = max(1, int(self.get_parameter('min_seen_count').value))
        configured_confirm_hits = int(self.get_parameter('confirm_hits').value)
        self.confirm_hits = max(1, configured_confirm_hits if configured_confirm_hits > 0 else self.min_seen_count)
        self.alpha_lidar = max(0.0, min(1.0, float(self.get_parameter('alpha_lidar').value)))
        self.alpha_camera = max(0.0, min(1.0, float(self.get_parameter('alpha_camera').value)))
        self.track_confidence_gain = max(0.0, float(self.get_parameter('track_confidence_gain').value))
        self.track_confidence_decay = max(
            0.0, min(1.0, float(self.get_parameter('track_confidence_decay').value))
        )
        self.color_decay = max(0.0, min(0.99, float(self.get_parameter('color_decay').value)))
        self.color_switch_margin = max(0.0, float(self.get_parameter('color_switch_margin').value))
        self.publish_stale_tracks = bool(self.get_parameter('publish_stale_tracks').value)
        self.stale_planner_ttl_sec = max(0.0, float(self.get_parameter('stale_planner_ttl_sec').value))

        self.camera_range_m = clamp_camera_range(float(self.get_parameter('camera_range_m').value))
        self.prefer_lidar_if_camera_missing_far = bool(
            self.get_parameter('prefer_lidar_if_camera_missing_far').value
        )
        self.allow_camera_fallback_near = bool(self.get_parameter('allow_camera_fallback_near').value)

        self.publish_rate_hz = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.enable_id_text = bool(self.get_parameter('enable_id_text').value)
        self.enable_tentative_viz = bool(self.get_parameter('enable_tentative_viz').value)
        self.enable_raw_debug_viz = bool(self.get_parameter('enable_raw_debug_viz').value)

        self.enable_global_track_memory = bool(self.get_parameter('enable_global_track_memory').value)
        self.global_merge_radius_m = max(0.05, float(self.get_parameter('global_merge_radius_m').value))
        self.global_min_hits_for_track = max(1, int(self.get_parameter('global_min_hits_for_track').value))
        self.global_max_cones = max(10, int(self.get_parameter('global_max_cones').value))
        self.believed_track_viz_min_hits = max(1, int(self.get_parameter('believed_track_viz_min_hits').value))
        self.believed_track_viz_min_confidence = max(
            0.0, min(1.0, float(self.get_parameter('believed_track_viz_min_confidence').value))
        )
        self.believed_track_viz_max_gap_m = max(
            0.0, float(self.get_parameter('believed_track_viz_max_gap_m').value)
        )
        self.believed_track_viz_show_center = bool(self.get_parameter('believed_track_viz_show_center').value)
        self.believed_track_viz_show_cones = bool(self.get_parameter('believed_track_viz_show_cones').value)

        self.save_track_plot_on_shutdown = bool(self.get_parameter('save_track_plot_on_shutdown').value)
        self.track_plot_data_filename = str(self.get_parameter('track_plot_data_filename').value).strip()

        self.lidar_cones_topic = str(self.get_parameter('lidar_cones_topic').value)
        self.camera_cones_topic = str(self.get_parameter('camera_cones_topic').value)
        self.tracked_cones_topic = str(self.get_parameter('tracked_cones_topic').value)
        self.viz_topic = str(self.get_parameter('viz_topic').value)
        self.believed_track_viz_topic = str(self.get_parameter('believed_track_viz_topic').value)
        self.run_session_topic = str(self.get_parameter('run_session_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _lidar_cb(self, msg: ConeDetectionArray) -> None:
        self._latest_lidar = self._convert_msg_to_detections(msg, source_name='lidar')
        self._latest_lidar_stamp_ns = self._stamp_to_ns(msg.header.stamp)

    def _camera_cb(self, msg: ConeDetectionArray) -> None:
        self._latest_camera = self._convert_msg_to_detections(msg, source_name='camera')
        self._latest_camera_stamp_ns = self._stamp_to_ns(msg.header.stamp)

    def _run_session_cb(self, msg: RunSession) -> None:
        self._run_session_run_id = str(msg.run_id).strip()
        self._run_session_base_path = str(msg.base_path).strip()

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom_msg = msg
        self._recent_odom_msgs.append(msg)

    # ------------------------------------------------------------------
    # Sensor detection conversion
    # ------------------------------------------------------------------

    def _convert_msg_to_detections(
        self,
        msg: ConeDetectionArray,
        *,
        source_name: str,
    ) -> list[SensorDetection]:
        source_frame = str(msg.header.frame_id).strip()
        if not source_frame:
            return []

        to_odom, _resolved_odom, _resolved_source_odom = self._lookup_transform_with_alias(
            self.odom_frame, source_frame, msg.header.stamp, allow_latest=False
        )
        to_base, _resolved_base, _resolved_source_base = self._lookup_transform_with_alias(
            self.base_frame, source_frame, msg.header.stamp, allow_latest=True
        )
        source_is_base = self._is_alias(source_frame, self.base_frame)
        odom_pose_fallback = None
        nearest_odom_dt_ms = float('nan')
        if to_odom is None:
            odom_pose_fallback, nearest_odom_dt_ms = self._nearest_odom_pose(msg.header.stamp)
            if odom_pose_fallback is None:
                odom_pose_fallback = self._current_base_pose_if_stamp_is_fresh(msg.header.stamp)
        base_identity_ok = to_base is None and source_is_base
        if (to_odom is None and odom_pose_fallback is None) or (to_base is None and not base_identity_ok):
            stamp_age_ms = (self.get_clock().now().nanoseconds - self._stamp_to_ns(msg.header.stamp)) / 1e6
            self._warn_throttled(
                'cone_tf_missing',
                f'cone transform unavailable sensor={source_name} source={source_frame} '
                f'targets={self.odom_frame} and {self.base_frame} '
                f'nearest_odom_dt_ms={nearest_odom_dt_ms:.1f} '
                f'stamp_age_ms={stamp_age_ms:.1f}; dropping frame',
            )
            return []

        detections: list[SensorDetection] = []
        for cone in msg.cones:
            x_src = float(cone.position.x)
            y_src = float(cone.position.y)
            z_src = float(cone.position.z)
            if to_base is not None:
                x_base, y_base, z_base = self._transform_point(to_base, x_src, y_src, z_src)
            else:
                x_base, y_base, z_base = x_src, y_src, z_src
            if to_odom is not None:
                x_odom, y_odom, z_odom = self._transform_point(to_odom, x_src, y_src, z_src)
            else:
                x_odom, y_odom, z_odom = base_point_to_odom(x_base, y_base, z_base, odom_pose_fallback)
            detections.append(
                SensorDetection(
                    x_odom=x_odom,
                    y_odom=y_odom,
                    z_odom=z_odom,
                    x_base=x_base,
                    y_base=y_base,
                    z_base=z_base,
                    range_m=float(math.hypot(x_base, y_base)),
                    color=normalize_color(cone.color),
                    confidence=float(max(0.0, min(1.0, float(cone.confidence)))),
                )
            )
        return detections

    def _nearest_odom_pose(self, stamp) -> tuple[Optional[tuple[float, float, float]], float]:
        if not self._recent_odom_msgs:
            return None, float('nan')

        target_ns = self._stamp_to_ns(stamp)
        tolerance_ns = int(self.odom_pose_sync_tolerance_sec * 1e9)
        best_pose: Optional[tuple[float, float, float]] = None
        best_dt_ns: Optional[int] = None

        for odom in self._recent_odom_msgs:
            pose = self._base_pose_from_odom_msg(odom)
            if pose is None:
                continue
            odom_ns = self._stamp_to_ns(odom.header.stamp)
            dt_ns = abs(odom_ns - target_ns)
            if best_dt_ns is None or dt_ns < best_dt_ns:
                best_dt_ns = dt_ns
                best_pose = pose

        if best_pose is None:
            return None, float('nan')
        best_dt_ms = float(best_dt_ns) / 1e6 if best_dt_ns is not None else float('nan')
        if tolerance_ns > 0 and (best_dt_ns is None or best_dt_ns > tolerance_ns):
            return None, best_dt_ms
        return best_pose, best_dt_ms

    def _current_base_pose_if_stamp_is_fresh(self, stamp) -> Optional[tuple[float, float, float]]:
        tolerance_sec = float(self.base_pose_latest_fallback_tolerance_sec)
        if tolerance_sec <= 0.0:
            return None
        stamp_ns = self._stamp_to_ns(stamp)
        now_ns = int(self.get_clock().now().nanoseconds)
        if abs(now_ns - stamp_ns) > int(tolerance_sec * 1e9):
            return None
        base_in_odom, _resolved_odom, _resolved_base = self._lookup_transform_with_alias(
            self.odom_frame, self.base_frame, Time().to_msg(), allow_latest=True
        )
        return self._resolve_base_pose(Time().to_msg(), base_in_odom)

    # ------------------------------------------------------------------
    # Main timer loop
    # ------------------------------------------------------------------

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        now_sec = float(now.nanoseconds) * 1e-9

        base_in_odom, _resolved_odom, _resolved_base = self._lookup_transform_with_alias(
            self.odom_frame, self.base_frame, now.to_msg()
        )
        base_pose = self._resolve_base_pose(now.to_msg(), base_in_odom)
        if base_pose is None:
            self._warn_throttled('base_pose_tf', f'missing tf/odom pose for {self.odom_frame}<-{self.base_frame}')
            return
        veh_x, veh_y, veh_yaw = base_pose

        lidar_is_new = (
            self._latest_lidar_stamp_ns >= 0
            and self._latest_lidar_stamp_ns != self._last_processed_lidar_stamp_ns
        )
        camera_is_new = (
            self._latest_camera_stamp_ns >= 0
            and self._latest_camera_stamp_ns != self._last_processed_camera_stamp_ns
        )

        stats = None
        if lidar_is_new or camera_is_new:
            updates = self._build_fused_observations(
                self._latest_lidar if lidar_is_new else [],
                self._latest_camera if camera_is_new else [],
            )
            stats = self._local_tracker.update(
                updates=updates,
                now_sec=now_sec,
                gate_radius_m=self.gate_radius_m,
                spawn_radius_m=self.spawn_radius_m,
                alpha_lidar=self.alpha_lidar,
                alpha_camera=self.alpha_camera,
                confirm_hits=self.confirm_hits,
                track_confidence_gain=self.track_confidence_gain,
                track_confidence_decay=self.track_confidence_decay,
                color_decay=self.color_decay,
                color_switch_margin=self.color_switch_margin,
            )
            if lidar_is_new:
                self._last_processed_lidar_stamp_ns = self._latest_lidar_stamp_ns
            if camera_is_new:
                self._last_processed_camera_stamp_ns = self._latest_camera_stamp_ns
            latest_update_ns = max(
                self._latest_lidar_stamp_ns if lidar_is_new else -1,
                self._latest_camera_stamp_ns if camera_is_new else -1,
            )
            if latest_update_ns >= 0:
                self._last_tracker_update_stamp = self._ns_to_stamp(latest_update_ns)

        track_positions_in_base = [
            odom_point_to_base(track.x, track.y, (veh_x, veh_y, veh_yaw))
            for track in self._local_tracker.tracks
        ]
        pruned = self._local_tracker.prune(
            now_sec=now_sec,
            tentative_ttl_sec=self.tentative_ttl_sec,
            stale_after_sec=self.stale_after_sec,
            confirmed_prune_after_sec=self.confirmed_prune_after_sec,
            max_range_m=self.max_range_m,
            behind_drop_m=self.behind_drop_m,
            unknown_drop_frames=self.unknown_drop_frames,
            confirm_hits=self.confirm_hits,
            track_positions_in_base=track_positions_in_base,
        )
        track_positions_in_base = [
            odom_point_to_base(track.x, track.y, (veh_x, veh_y, veh_yaw))
            for track in self._local_tracker.tracks
        ]
        merged_tracks = self._local_tracker.merge_nearby_tracks(
            merge_radius_m=self.track_merge_radius_m,
            track_positions_in_base=track_positions_in_base,
            longitudinal_tolerance_m=self.track_merge_longitudinal_tolerance_m,
            lateral_tolerance_m=self.track_merge_lateral_tolerance_m,
        )
        if stats is not None:
            stats.merged_tracks += merged_tracks

        confirmed = self._local_tracker.confirmed_tracks(self.confirm_hits)
        stale = self._local_tracker.stale_tracks(self.confirm_hits)
        tentative = self._local_tracker.tentative_tracks(self.confirm_hits)
        planner_tracks = self._local_tracker.planner_tracks(
            now_sec=now_sec,
            confirm_hits=self.confirm_hits,
            publish_stale_tracks=self.publish_stale_tracks,
            stale_planner_ttl_sec=self.stale_planner_ttl_sec,
        )

        heading_x = math.cos(veh_yaw)
        heading_y = math.sin(veh_yaw)

        left, right, center = self._infer_track_boundaries_and_centerline(
            tracks=planner_tracks,
            min_track_confidence=0.0,
            min_color_confidence=0.0,
            vehicle_x=veh_x,
            vehicle_y=veh_y,
            heading_x=heading_x,
            heading_y=heading_y,
        )
        viz_left, viz_right, viz_center = self._infer_track_boundaries_and_centerline(
            tracks=planner_tracks,
            min_track_confidence=self.believed_track_viz_min_confidence,
            min_color_confidence=self.believed_track_viz_min_confidence,
            vehicle_x=veh_x,
            vehicle_y=veh_y,
            heading_x=heading_x,
            heading_y=heading_y,
        )

        self._publish_tracked_cones(
            planner_tracks, now,
            vehicle_x=veh_x, vehicle_y=veh_y, heading_x=heading_x, heading_y=heading_y,
        )
        self._publish_local_markers(confirmed=confirmed, stale=stale, tentative=tentative, now=now)
        self._publish_track_markers(tracks=planner_tracks, left=viz_left, right=viz_right, center=viz_center, now=now)
        if self.enable_raw_debug_viz:
            self._publish_raw_markers(now=now)

        status_suffix = '' if stats is not None else ' inputs=reused'
        self._info_throttled(
            'cone_memory_status',
            f'tracks total={len(self._local_tracker.tracks)} confirmed={len(confirmed)} '
            f'stale={len(stale)} tentative={len(tentative)} planner={len(planner_tracks)} '
            f'left={len(left)} right={len(right)} center={len(center)} '
            f'pruned={pruned} merged={merged_tracks}{status_suffix}',
        )

    # ------------------------------------------------------------------
    # Boundary / centerline inference
    # ------------------------------------------------------------------

    def _infer_track_boundaries_and_centerline(
        self,
        *,
        tracks: list[ConeTrack],
        min_track_confidence: float,
        min_color_confidence: float,
        vehicle_x: float,
        vehicle_y: float,
        heading_x: float,
        heading_y: float,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
        filtered_tracks = [
            t for t in tracks
            if float(t.track_confidence) >= float(min_track_confidence)
            and float(t.class_label()[1]) >= float(min_color_confidence)
        ]

        if filtered_tracks:
            points_xy = np.asarray(
                [[float(t.x), float(t.y)] for t in filtered_tracks], dtype=np.float64
            )
            resolved_colors = resolve_boundary_colors_for_planning(
                points_xy=points_xy,
                raw_colors=[t.class_label()[0] for t in filtered_tracks],
                vehicle_xy=(float(vehicle_x), float(vehicle_y)),
                vehicle_yaw=math.atan2(float(heading_y), float(heading_x)),
            )
        else:
            resolved_colors = []

        left: list[tuple[float, float]] = []
        right: list[tuple[float, float]] = []
        for track, resolved in zip(filtered_tracks, resolved_colors):
            if resolved == 'blue':
                left.append((float(track.x), float(track.y)))
            elif resolved == 'yellow':
                right.append((float(track.x), float(track.y)))

        def projection(p: tuple[float, float]) -> float:
            dx = p[0] - float(vehicle_x)
            dy = p[1] - float(vehicle_y)
            return (dx * float(heading_x)) + (dy * float(heading_y))

        left.sort(key=projection)
        right.sort(key=projection)
        center: list[tuple[float, float]] = []
        if left and right:
            for lx, ly in left:
                best_idx = min(
                    range(len(right)),
                    key=lambda i: (right[i][0] - lx) ** 2 + (right[i][1] - ly) ** 2,
                )
                rx, ry = right[best_idx]
                center.append(((lx + rx) * 0.5, (ly + ry) * 0.5))
            center.sort(key=projection)
        return left, right, center

    # ------------------------------------------------------------------
    # Sensor fusion helpers
    # ------------------------------------------------------------------

    def _build_fused_observations(
        self,
        lidar: list[SensorDetection],
        camera: list[SensorDetection],
    ) -> list[TrackUpdate]:
        lidar = _deduplicate_sensor_detections(lidar, self.dedup_radius_m)
        camera = _deduplicate_sensor_detections(camera, self.dedup_radius_m)
        pairs, unmatched_lidar_idx, unmatched_camera_idx = _pair_detections(lidar, camera, self.gate_radius_m)
        out: list[TrackUpdate] = []
        for l_idx, c_idx in pairs:
            out.append(self._make_observation(lidar_det=lidar[l_idx], camera_det=camera[c_idx]))
        for l_idx in unmatched_lidar_idx:
            out.append(self._make_observation(lidar_det=lidar[l_idx], camera_det=None))
        for c_idx in unmatched_camera_idx:
            out.append(self._make_observation(lidar_det=None, camera_det=camera[c_idx]))
        return out

    def _make_observation(
        self,
        *,
        lidar_det: Optional[SensorDetection],
        camera_det: Optional[SensorDetection],
    ) -> TrackUpdate:
        has_lidar = lidar_det is not None
        has_camera = camera_det is not None

        range_candidates = []
        if lidar_det is not None:
            range_candidates.append(lidar_det.range_m)
        if camera_det is not None:
            range_candidates.append(camera_det.range_m)
        range_m = min(range_candidates) if range_candidates else 999.0

        source = choose_position_source(
            range_m=range_m,
            camera_range_m=self.camera_range_m,
            has_lidar_position=has_lidar,
            has_camera_position=has_camera,
            prefer_lidar_if_camera_missing_far=self.prefer_lidar_if_camera_missing_far,
            allow_camera_fallback_near=self.allow_camera_fallback_near,
        )

        update_x = update_y = update_z = None
        if source == 'lidar' and lidar_det is not None:
            update_x, update_y, update_z = lidar_det.x_odom, lidar_det.y_odom, lidar_det.z_odom
        elif source == 'camera' and camera_det is not None:
            update_x, update_y, update_z = camera_det.x_odom, camera_det.y_odom, camera_det.z_odom

        if update_x is not None and update_y is not None:
            assoc_x, assoc_y = float(update_x), float(update_y)
        elif camera_det is not None:
            assoc_x, assoc_y = float(camera_det.x_odom), float(camera_det.y_odom)
        elif lidar_det is not None:
            assoc_x, assoc_y = float(lidar_det.x_odom), float(lidar_det.y_odom)
        else:
            assoc_x, assoc_y = 0.0, 0.0

        return TrackUpdate(
            assoc_x=assoc_x,
            assoc_y=assoc_y,
            update_x=update_x,
            update_y=update_y,
            update_z=update_z,
            update_source=source,
            has_lidar=has_lidar,
            has_camera=has_camera,
            camera_label=camera_det.color if camera_det is not None else None,
            camera_confidence=camera_det.confidence if camera_det is not None else 0.0,
            range_m=range_m,
        )

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_tracked_cones(
        self,
        tracks: list[ConeTrack],
        now: Time,
        *,
        vehicle_x: float,
        vehicle_y: float,
        heading_x: float,
        heading_y: float,
    ) -> None:
        msg = ConeDetectionArray()
        msg.header.stamp = (
            self._last_tracker_update_stamp if self._last_tracker_update_stamp is not None else now.to_msg()
        )
        msg.header.frame_id = self.odom_frame
        points_xy = (
            np.asarray([[float(t.x), float(t.y)] for t in tracks], dtype=np.float64)
            if tracks else np.empty((0, 2), dtype=np.float64)
        )
        planner_boundary_colors = resolve_boundary_colors_for_planning(
            points_xy=points_xy,
            raw_colors=[t.class_label()[0] for t in tracks],
            vehicle_xy=(float(vehicle_x), float(vehicle_y)),
            vehicle_yaw=math.atan2(float(heading_y), float(heading_x)),
        )
        for track, boundary_color in zip(tracks, planner_boundary_colors):
            label, class_conf = track.class_label()
            cone = ConeDetection()
            cone.color = label
            _set_cone_field(cone, 'boundary_color', boundary_color)
            cone.confidence = float(max(0.0, min(1.0, track.track_confidence)))
            cone.position.x = float(track.x)
            cone.position.y = float(track.y)
            cone.position.z = float(track.z)
            _set_cone_field(cone, 'track_id', int(track.track_id))
            _set_cone_field(cone, 'track_state', int(track.track_state))
            _set_cone_field(cone, 'track_confidence', float(max(0.0, min(1.0, track.track_confidence))))
            _set_cone_field(cone, 'color_confidence', float(max(0.0, min(1.0, class_conf))))
            _set_cone_field(cone, 'seen_count', int(track.seen_count))
            _set_cone_field(cone, 'consecutive_seen_count', int(track.consecutive_seen_count))
            _set_cone_field(cone, 'missed_count', int(track.missed_count))
            _set_cone_field(cone, 'first_seen', self._ns_to_stamp(int(max(0.0, track.first_seen_sec) * 1e9)))
            _set_cone_field(cone, 'last_seen', self._ns_to_stamp(int(max(0.0, track.last_seen_sec) * 1e9)))
            msg.cones.append(cone)
        self._tracked_pub.publish(msg)

    def _publish_local_markers(
        self,
        *,
        confirmed: list[ConeTrack],
        stale: list[ConeTrack],
        tentative: list[ConeTrack],
        now: Time,
    ) -> None:
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.odom_frame
        clear.header.stamp = now.to_msg()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for groups in (confirmed, stale, (tentative if self.enable_tentative_viz else [])):
            for track in groups:
                arr.markers.append(
                    make_cone_marker(track, now, odom_frame=self.odom_frame, publish_rate_hz=self.publish_rate_hz)
                )
                if self.enable_id_text:
                    arr.markers.append(
                        make_id_marker(track, now, odom_frame=self.odom_frame, publish_rate_hz=self.publish_rate_hz)
                    )
        self._viz_pub.publish(arr)

    def _publish_track_markers(
        self,
        *,
        tracks: list[ConeTrack],
        left: list[tuple[float, float]],
        right: list[tuple[float, float]],
        center: list[tuple[float, float]],
        now: Time,
    ) -> None:
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.odom_frame
        clear.header.stamp = now.to_msg()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        marker_id = 1
        marker_id = self._append_polyline_segments(arr, now, marker_id=marker_id, ns='track_left', points=left, rgb=(0.2, 0.4, 1.0))
        marker_id = self._append_polyline_segments(arr, now, marker_id=marker_id, ns='track_right', points=right, rgb=(1.0, 0.9, 0.2))
        if self.believed_track_viz_show_center:
            marker_id = self._append_polyline_segments(arr, now, marker_id=marker_id, ns='track_center', points=center, rgb=(0.0, 0.0, 0.0))

        if self.believed_track_viz_show_cones:
            from geometry_msgs.msg import Point as GPoint
            cones_marker = Marker()
            cones_marker.header.frame_id = self.odom_frame
            cones_marker.header.stamp = now.to_msg()
            cones_marker.ns = 'remembered_cones'
            cones_marker.id = marker_id
            cones_marker.type = Marker.SPHERE_LIST
            cones_marker.action = Marker.ADD
            cones_marker.scale.x = 0.20
            cones_marker.scale.y = 0.20
            cones_marker.scale.z = 0.20
            cones_marker.color.a = 0.7
            cones_marker.color.r = 0.8
            cones_marker.color.g = 0.8
            cones_marker.color.b = 0.8
            for track in tracks:
                if track.seen_count < self.believed_track_viz_min_hits:
                    continue
                if float(track.track_confidence) < self.believed_track_viz_min_confidence:
                    continue
                pt = GPoint()
                pt.x = float(track.x)
                pt.y = float(track.y)
                pt.z = float(track.z)
                cones_marker.points.append(pt)
            arr.markers.append(cones_marker)

        self._track_viz_pub.publish(arr)

    def _append_polyline_segments(
        self,
        arr: MarkerArray,
        now: Time,
        *,
        marker_id: int,
        ns: str,
        points: list[tuple[float, float]],
        rgb: tuple[float, float, float],
    ) -> int:
        for segment in split_polyline_by_gap(points, self.believed_track_viz_max_gap_m):
            if len(segment) < 2:
                continue
            arr.markers.append(
                make_line_marker(now, marker_id=marker_id, ns=ns, points=segment, odom_frame=self.odom_frame, rgb=rgb)
            )
            marker_id += 1
        return marker_id

    def _publish_raw_markers(self, *, now: Time) -> None:
        lidar_arr = make_raw_sensor_markers(
            self._latest_lidar, now,
            odom_frame=self.odom_frame,
            ns='raw_lidar',
            scale=0.14,
            rgba=(0.2, 0.9, 1.0, 0.7),
        )
        camera_arr = make_raw_sensor_markers(
            self._latest_camera, now,
            odom_frame=self.odom_frame,
            ns='raw_camera',
            scale=0.12,
            rgba=(1.0, 0.3, 0.8, 0.65),
        )
        self._raw_lidar_viz_pub.publish(lidar_arr)
        self._raw_camera_viz_pub.publish(camera_arr)

    # ------------------------------------------------------------------
    # Track data saving
    # ------------------------------------------------------------------

    def _save_track_data(self) -> None:
        if self._last_plot_save_attempted:
            return
        self._last_plot_save_attempted = True
        if not self.save_track_plot_on_shutdown:
            return

        base_in_odom, _resolved_odom, _resolved_base = self._lookup_transform_with_alias(
            self.odom_frame, self.base_frame, Time().to_msg()
        )
        pose = self._resolve_base_pose(Time().to_msg(), base_in_odom)
        veh_x, veh_y, veh_yaw = pose if pose is not None else (0.0, 0.0, 0.0)
        heading_x, heading_y = math.cos(veh_yaw), math.sin(veh_yaw)
        tracks = self._local_tracker.planner_tracks(
            now_sec=float(self.get_clock().now().nanoseconds) * 1e-9,
            confirm_hits=self.confirm_hits,
            publish_stale_tracks=self.publish_stale_tracks,
            stale_planner_ttl_sec=self.stale_planner_ttl_sec,
        )
        _left, _right, center = self._infer_track_boundaries_and_centerline(
            tracks=tracks,
            min_track_confidence=self.believed_track_viz_min_confidence,
            min_color_confidence=self.believed_track_viz_min_confidence,
            vehicle_x=veh_x,
            vehicle_y=veh_y,
            heading_x=heading_x,
            heading_y=heading_y,
        )
        if not self.track_plot_data_filename:
            return
        logs_dir = resolve_logs_dir(
            self._run_session_run_id, self._run_session_base_path, self.get_logger()
        )
        if logs_dir is None:
            self.get_logger().warn('could not resolve logs output directory for cone memory csv data')
            return
        logs_dir.mkdir(parents=True, exist_ok=True)
        save_track_data_csv(logs_dir / self.track_plot_data_filename, tracks, center, self.get_logger())

    # ------------------------------------------------------------------
    # TF / frame utilities
    # ------------------------------------------------------------------

    def _resolve_base_pose(self, stamp, tf_msg) -> Optional[tuple[float, float, float]]:
        if tf_msg is not None:
            tx = float(tf_msg.transform.translation.x)
            ty = float(tf_msg.transform.translation.y)
            q = tf_msg.transform.rotation
            yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
            return tx, ty, yaw
        odom = self._latest_odom_msg
        if odom is None:
            return None
        return self._base_pose_from_odom_msg(odom)

    def _base_pose_from_odom_msg(self, odom: Odometry) -> Optional[tuple[float, float, float]]:
        if not self._is_alias(odom.header.frame_id, self.odom_frame):
            return None
        pose = odom.pose.pose
        tx = float(pose.position.x)
        ty = float(pose.position.y)
        q = pose.orientation
        yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        return convert_odom_child_pose_to_base_frame(
            child_frame=str(odom.child_frame_id).strip(),
            base_frame=self.base_frame,
            tx=tx,
            ty=ty,
            yaw=yaw,
            wheelbase_m=self.wheelbase_m,
            is_alias=self._is_alias,
        )

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp, *, allow_latest: bool = True):
        timeout = Duration(seconds=self.tf_timeout_s)
        try:
            stamp_time = Time.from_msg(stamp)
            return self._tf_buffer.lookup_transform(target_frame, source_frame, stamp_time, timeout=timeout)
        except (TransformException, ValueError):
            pass
        if not allow_latest:
            return None
        try:
            return self._tf_buffer.lookup_transform(target_frame, source_frame, Time(), timeout=timeout)
        except TransformException:
            return None

    def _lookup_transform_with_alias(self, target_frame: str, source_frame: str, stamp, *, allow_latest: bool = True):
        for target_candidate in self._frame_aliases(target_frame):
            for source_candidate in self._frame_aliases(source_frame):
                tf_msg = self._lookup_transform(target_candidate, source_candidate, stamp, allow_latest=allow_latest)
                if tf_msg is not None:
                    return tf_msg, target_candidate, source_candidate
        return None, target_frame, source_frame

    def _frame_aliases(self, frame: str) -> list[str]:
        token = str(frame).strip()
        out: list[str] = []

        def add(candidate: str) -> None:
            c = candidate.strip()
            if c and c not in out:
                out.append(c)

        add(token)
        leaf = token.split('/')[-1] if token else ''
        add(leaf)

        if token in {'base_link', 'base_footprint'} or leaf in {'base_link', 'base_footprint'}:
            add(token.replace('base_link', 'base_footprint'))
            add(token.replace('base_footprint', 'base_link'))
            if '/' in token:
                prefix = token.rsplit('/', 1)[0]
                add(f'{prefix}/base_link')
                add(f'{prefix}/base_footprint')
            add('base_link')
            add('base_footprint')
        if token == 'front_axle' or leaf == 'front_axle':
            if '/' in token:
                prefix = token.rsplit('/', 1)[0]
                add(f'{prefix}/front_axle')
            add('front_axle')
        return out

    def _is_alias(self, frame_a: str, frame_b: str) -> bool:
        return bool(set(self._frame_aliases(frame_a)).intersection(self._frame_aliases(frame_b)))

    @staticmethod
    def _transform_point(transform, x: float, y: float, z: float) -> tuple[float, float, float]:
        t = transform.transform.translation
        q = transform.transform.rotation
        qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)
        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)
        tx, ty, tz = float(t.x), float(t.y), float(t.z)
        px = (r00 * x) + (r01 * y) + (r02 * z) + tx
        py = (r10 * x) + (r11 * y) + (r12 * z) + ty
        pz = (r20 * x) + (r21 * y) + (r22 * z) + tz
        return px, py, pz

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return (int(stamp.sec) * 1_000_000_000) + int(stamp.nanosec)

    @staticmethod
    def _ns_to_stamp(stamp_ns: int) -> TimeMsg:
        msg = TimeMsg()
        msg.sec = int(stamp_ns // 1_000_000_000)
        msg.nanosec = int(stamp_ns % 1_000_000_000)
        return msg

    def _warn_throttled(self, key: str, message: str, period_sec: float = 1.0) -> None:
        now_sec = time.monotonic()
        if (now_sec - self._last_throttled_log_sec.get(key, -1.0)) >= period_sec:
            self.get_logger().warn(message)
            self._last_throttled_log_sec[key] = now_sec

    def _info_throttled(self, key: str, message: str, period_sec: float = 1.0) -> None:
        now_sec = time.monotonic()
        if (now_sec - self._last_throttled_log_sec.get(key, -1.0)) >= period_sec:
            self.get_logger().info(message)
            self._last_throttled_log_sec[key] = now_sec

    def destroy_node(self):
        self._save_track_data()
        return super().destroy_node()


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no node dependency)
# ---------------------------------------------------------------------------

def _set_cone_field(cone: ConeDetection, field_name: str, value) -> None:
    if hasattr(cone, field_name):
        setattr(cone, field_name, value)


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _deduplicate_sensor_detections(
    detections: list[SensorDetection],
    dedup_radius_m: float,
) -> list[SensorDetection]:
    """Merge sensor detections that are within *dedup_radius_m* of each other."""
    if len(detections) <= 1:
        return list(detections)

    radius_sq = float(dedup_radius_m) ** 2
    merged: list[SensorDetection] = []
    total_weight: list[float] = []

    for det in sorted(detections, key=lambda d: (d.range_m, -d.confidence)):
        best_idx = -1
        best_dist_sq = float('inf')
        for idx, existing in enumerate(merged):
            if (
                det.color != existing.color
                and det.color != 'unknown'
                and existing.color != 'unknown'
            ):
                continue
            dx = det.x_base - existing.x_base
            dy = det.y_base - existing.y_base
            dist_sq = (dx * dx) + (dy * dy)
            if dist_sq <= radius_sq and dist_sq < best_dist_sq:
                best_idx = idx
                best_dist_sq = dist_sq

        if best_idx < 0:
            merged.append(det)
            total_weight.append(max(0.1, det.confidence))
            continue

        prev = merged[best_idx]
        prev_weight = total_weight[best_idx]
        add_weight = max(0.1, det.confidence)
        weight_sum = prev_weight + add_weight
        merged_det = SensorDetection(
            x_odom=((prev.x_odom * prev_weight) + (det.x_odom * add_weight)) / weight_sum,
            y_odom=((prev.y_odom * prev_weight) + (det.y_odom * add_weight)) / weight_sum,
            z_odom=((prev.z_odom * prev_weight) + (det.z_odom * add_weight)) / weight_sum,
            x_base=((prev.x_base * prev_weight) + (det.x_base * add_weight)) / weight_sum,
            y_base=((prev.y_base * prev_weight) + (det.y_base * add_weight)) / weight_sum,
            z_base=((prev.z_base * prev_weight) + (det.z_base * add_weight)) / weight_sum,
            range_m=float(math.hypot(
                ((prev.x_base * prev_weight) + (det.x_base * add_weight)) / weight_sum,
                ((prev.y_base * prev_weight) + (det.y_base * add_weight)) / weight_sum,
            )),
            color=det.color if det.confidence >= prev.confidence else prev.color,
            confidence=max(prev.confidence, det.confidence),
        )
        merged[best_idx] = merged_det
        total_weight[best_idx] = weight_sum

    return merged


def _pair_detections(
    lidar: list[SensorDetection],
    camera: list[SensorDetection],
    gate_radius_m: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy nearest-neighbour pairing of lidar and camera detections."""
    gate_sq = gate_radius_m * gate_radius_m
    candidates: list[tuple[float, int, int]] = []
    for l_idx, l_det in enumerate(lidar):
        for c_idx, c_det in enumerate(camera):
            dx = l_det.x_odom - c_det.x_odom
            dy = l_det.y_odom - c_det.y_odom
            d2 = dx * dx + dy * dy
            if d2 <= gate_sq:
                candidates.append((d2 ** 0.5, l_idx, c_idx))

    candidates.sort(key=lambda item: item[0])
    taken_lidar: set[int] = set()
    taken_camera: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _dist, l_idx, c_idx in candidates:
        if l_idx in taken_lidar or c_idx in taken_camera:
            continue
        taken_lidar.add(l_idx)
        taken_camera.add(c_idx)
        pairs.append((l_idx, c_idx))

    unmatched_lidar = [i for i in range(len(lidar)) if i not in taken_lidar]
    unmatched_camera = [i for i in range(len(camera)) if i not in taken_camera]
    return pairs, unmatched_lidar, unmatched_camera


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConeMemoryNode()
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
