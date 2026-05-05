"""RViz marker construction for the tracked-cone planner runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.planning_state_machine import _OPERATOR_STATE_COLORS
from sim_car.planning.triangulation_planner_core import CoreResult


CORRIDOR_PAIR_AUDIT_REASONS = (
    "pair_valid",
    "pair_width_too_narrow",
    "pair_width_too_wide",
    "pair_left_behind",
    "pair_right_behind",
    "pair_left_beyond_horizon",
    "pair_right_beyond_horizon",
    "pair_not_in_longest_valid_slice",
    "pair_nonfinite",
    "pair_rejected_unknown",
)
# Width matches the accepted-pair marker while using the corridor-specific namespace.
_CORRIDOR_RUNG_MARKER_WIDTH_M = 0.07
# Raw chain lines sit above the boundary lines so both can be inspected in RViz.
_CORRIDOR_RAW_CHAIN_LINE_WIDTH_M = 0.06
# Raw chain points are larger than fitted boundary points to show input samples.
_CORRIDOR_RAW_CHAIN_POINT_DIAMETER_M = 0.34
# Audit lines are intentionally slimmer than accepted-pair lines to read as diagnostics.
_CORRIDOR_AUDIT_MARKER_WIDTH_M = 0.05
# Audit labels float above cones so they do not cover the pair segment line.
_CORRIDOR_AUDIT_LABEL_Z_M = 0.85
# Audit label text is small enough to keep dense rejection labels readable.
_CORRIDOR_AUDIT_LABEL_TEXT_HEIGHT_M = 0.14
# Label color stays slightly transparent so overlapping labels remain distinguishable.
_CORRIDOR_AUDIT_LABEL_ALPHA = 0.95
# Segment midpoint is the average of its two endpoints.
_SEGMENT_MIDPOINT_WEIGHT = 0.5


@dataclass(frozen=True)
class _CorridorPairAuditVizData:
    segments: np.ndarray
    reasons: list[str]
    widths_m: np.ndarray
    anchors_local: np.ndarray
    count: int


class VisualizationMixin:
    """RViz marker construction and cone visualization helpers for TrackedConePlannerRuntime."""

    def _build_markers(
        self,
        *,
        now,
        frame_id: str,
        result: Optional[CoreResult],
        centerline: np.ndarray,
        raw_centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        status: str,
        operator_state: str,
        control_target_frame: Optional[np.ndarray],
    ) -> MarkerArray:
        arr = MarkerArray()
        marker_id = self._append_clear_marker(arr, frame_id, now)
        if result is None:
            self._append_status_marker(arr, marker_id, frame_id, now, status, operator_state=operator_state)
            return arr

        if getattr(self, 'show_raw_cones', False):
            marker_id = self._append_remembered_cone_marker(arr, marker_id, frame_id, now)
        marker_id = self._append_graph_edge_markers(arr, marker_id, frame_id, now, result)
        left_boundary, right_boundary = self._resolve_boundary_viz_paths(result)
        marker_id = self._append_boundary_markers(
            arr, marker_id, frame_id, now, result, left_boundary, right_boundary,
        )
        marker_id = self._append_single_boundary_marker(arr, marker_id, frame_id, now, result, left_boundary, right_boundary)
        marker_id = self._append_pair_markers(arr, marker_id, frame_id, now, result)
        marker_id = self._append_corridor_pair_audit_markers(arr, marker_id, frame_id, now, result)
        marker_id = self._append_raw_path_markers(arr, marker_id, frame_id, now, result, raw_centerline, raw_midpoint_chain)
        marker_id = self._append_centerline_marker(arr, marker_id, frame_id, now, centerline)
        marker_id = self._append_lookahead_marker(arr, marker_id, frame_id, now, control_target_frame)
        self._append_status_marker(arr, marker_id, frame_id, now, status, operator_state=operator_state)
        return arr

    @staticmethod
    def _append_clear_marker(markers: MarkerArray, frame_id: str, stamp) -> int:
        clear = Marker()
        clear.header.frame_id = frame_id
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        return 1

    def _append_graph_edge_markers(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: Any,
    ) -> int:
        if hasattr(self, 'show_boundary_chains'):
            return marker_id
        edges = (
            (getattr(self, 'show_triangulation_edges', False), 'triangulation_edges', result.triangulation_edges, (0.3, 0.7, 1.0, 0.35), 0.03),
            (getattr(self, 'show_candidate_edges', False), 'candidate_cross_edges', result.candidate_edges, (1.0, 0.6, 0.1, 0.8), 0.06),
            (getattr(self, 'show_selected_edges', False), 'selected_cross_edges', result.selected_edges, (0.2, 1.0, 0.3, 0.95), 0.08),
        )
        for enabled, ns, edge_data, color, width in edges:
            if not enabled:
                continue
            markers.markers.append(
                self._make_edge_list_marker(
                    frame_id=frame_id, stamp=stamp, marker_id=marker_id,
                    ns=ns, points=result.filtered_points, edges=edge_data,
                    color=color, width=width,
                )
            )
            marker_id += 1
        return marker_id

    def _resolve_boundary_viz_paths(self, result: Any) -> tuple[np.ndarray, np.ndarray]:
        left_boundary = np.array(getattr(result, 'left_boundary', np.empty((0, 2))), copy=True)
        right_boundary = np.array(getattr(result, 'right_boundary', np.empty((0, 2))), copy=True)
        if not hasattr(self, '_last_viz_left_boundary'):
            return left_boundary, right_boundary
        if left_boundary.shape[0] > 0:
            self._last_viz_left_boundary = np.array(left_boundary, copy=True)
        elif self._last_viz_left_boundary is not None:
            left_boundary = np.array(self._last_viz_left_boundary, copy=True)
        if right_boundary.shape[0] > 0:
            self._last_viz_right_boundary = np.array(right_boundary, copy=True)
        elif self._last_viz_right_boundary is not None:
            right_boundary = np.array(self._last_viz_right_boundary, copy=True)
        return left_boundary, right_boundary

    def _append_boundary_markers(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: Any,
        left_boundary: np.ndarray,
        right_boundary: np.ndarray,
    ) -> int:
        if hasattr(self, 'show_boundary_chains') and not self.show_boundary_chains:
            return marker_id
        if hasattr(self, 'show_boundary_chains'):
            return self._append_migrated_boundary_markers(
                markers, marker_id, frame_id, stamp, result, left_boundary, right_boundary,
            )
        return self._append_runtime_boundary_markers(markers, marker_id, frame_id, stamp, left_boundary, right_boundary)

    def _append_migrated_boundary_markers(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: Any,
        left_boundary: np.ndarray,
        right_boundary: np.ndarray,
    ) -> int:
        marker_id = self._append_line_and_points_markers(
            markers, marker_id, frame_id, stamp, 'boundary_left', 'boundary_left_points',
            left_boundary, (0.2, 0.45, 1.0, 0.95), (0.2, 0.55, 1.0, 1.0), 0.12, 0.13,
        )
        marker_id = self._append_corridor_raw_chain_marker(
            markers, marker_id, frame_id, stamp, result, side='left',
        )
        marker_id = self._append_line_and_points_markers(
            markers, marker_id, frame_id, stamp, 'boundary_right', 'boundary_right_points',
            right_boundary, (1.0, 0.9, 0.2, 0.95), (1.0, 0.92, 0.25, 1.0), 0.14, 0.15,
        )
        return self._append_corridor_raw_chain_marker(
            markers, marker_id, frame_id, stamp, result, side='right',
        )

    def _append_corridor_raw_chain_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: Any,
        *,
        side: str,
    ) -> int:
        points = self._corridor_raw_chain_points(result, side)
        if points.shape[0] == 0:
            return marker_id
        colors = self._corridor_raw_chain_colors(side)
        z_offsets = (0.22, 0.23) if side == 'left' else (0.24, 0.25)
        return self._append_line_and_points_markers(
            markers, marker_id, frame_id, stamp,
            f'raw_chain_{side}', f'raw_chain_{side}_points',
            points, colors[0], colors[1], z_offsets[0], z_offsets[1],
            line_width=_CORRIDOR_RAW_CHAIN_LINE_WIDTH_M,
            point_scale=_CORRIDOR_RAW_CHAIN_POINT_DIAMETER_M,
        )

    @staticmethod
    def _corridor_raw_chain_points(result: Any, side: str) -> np.ndarray:
        attr_name = f'raw_{side}_chain_points'
        return np.asarray(getattr(result, attr_name, np.empty((0, 2))), dtype=np.float64)

    @staticmethod
    def _corridor_raw_chain_colors(
        side: str,
    ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
        if side == 'left':
            return (0.05, 0.25, 1.0, 0.95), (0.05, 0.25, 1.0, 1.0)
        return (1.0, 0.82, 0.0, 0.95), (1.0, 0.82, 0.0, 1.0)

    def _append_line_and_points_markers(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        line_ns: str,
        points_ns: str,
        points: np.ndarray,
        line_color: tuple[float, float, float, float],
        points_color: tuple[float, float, float, float],
        line_z: float,
        points_z: float,
        line_width: float = 0.10,
        point_scale: float = 0.22,
    ) -> int:
        markers.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id, stamp=stamp, marker_id=marker_id,
                ns=line_ns, points=points, color=line_color, width=line_width, z_offset=line_z,
            )
        )
        marker_id += 1
        point_marker = self._make_points_marker(
            frame_id=frame_id, stamp=stamp, marker_id=marker_id,
            ns=points_ns, points=points, color=points_color, scale=point_scale,
        )
        for point in point_marker.points:
            point.z = points_z
        markers.markers.append(point_marker)
        return marker_id + 1

    def _append_runtime_boundary_markers(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        left_boundary: np.ndarray,
        right_boundary: np.ndarray,
    ) -> int:
        for ns, points, color in (
            ('boundary_left', left_boundary, (0.2, 0.45, 1.0, 0.95)),
            ('boundary_right', right_boundary, (1.0, 0.9, 0.2, 0.95)),
        ):
            markers.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id, stamp=stamp, marker_id=marker_id,
                    ns=ns, points=points, color=color, width=0.07, z_offset=0.02,
                )
            )
            marker_id += 1
        return marker_id

    def _append_single_boundary_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: Any,
        left_boundary: np.ndarray,
        right_boundary: np.ndarray,
    ) -> int:
        active_boundary = np.empty((0, 2), dtype=np.float64)
        if getattr(result, 'planner_mode', '') == 'single_boundary':
            if getattr(result, 'active_boundary_side', '') == 'blue':
                active_boundary = left_boundary
            elif getattr(result, 'active_boundary_side', '') == 'yellow':
                active_boundary = right_boundary
        if active_boundary.shape[0] == 0:
            return marker_id
        markers.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id, stamp=stamp, marker_id=marker_id,
                ns='single_boundary_active', points=active_boundary,
                color=(0.15, 1.0, 0.2, 1.0), width=0.16, z_offset=0.20,
            )
        )
        return marker_id + 1

    def _append_pair_markers(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: Any,
    ) -> int:
        if not getattr(self, 'show_pair_lines', False):
            return marker_id
        pair_segments = getattr(self, '_current_pair_segments_for_viz', None)
        if pair_segments is not None and pair_segments.size > 0:
            self._last_viz_pair_segments = np.array(pair_segments, copy=True)
        elif getattr(self, '_last_viz_pair_segments', None) is not None:
            pair_segments = self._last_viz_pair_segments
        namespace = 'corridor_rungs' if self._has_corridor_pair_audit(result) else 'accepted_pairs'
        markers.markers.append(
            self._make_pair_segment_marker(
                frame_id=frame_id, stamp=stamp, marker_id=marker_id,
                ns=namespace, pair_segments=pair_segments,
                color=(0.2, 1.0, 0.3, 0.95), width=_CORRIDOR_RUNG_MARKER_WIDTH_M,
            )
        )
        return marker_id + 1

    @staticmethod
    def _has_corridor_pair_audit(result: Any) -> bool:
        return hasattr(result, 'corridor_pair_audit_segments')

    def _append_corridor_pair_audit_markers(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: Any,
    ) -> int:
        if not getattr(self, 'show_corridor_pair_audit', False):
            return marker_id
        if not self._has_corridor_pair_audit(result):
            return marker_id
        audit = self._corridor_pair_audit_data(result)
        if audit.count <= 0:
            return marker_id
        marker_id = self._append_corridor_pair_audit_segments(
            markers, marker_id, frame_id, stamp, audit,
        )
        return self._append_corridor_pair_audit_labels(
            markers, marker_id, frame_id, stamp, audit,
        )

    @staticmethod
    def _corridor_pair_audit_data(result: Any) -> _CorridorPairAuditVizData:
        segments = np.asarray(result.corridor_pair_audit_segments, dtype=np.float64)
        reasons = list(result.corridor_pair_audit_reasons)
        widths_m = np.asarray(result.corridor_pair_audit_widths_m, dtype=np.float64)
        anchors_local = np.asarray(result.corridor_pair_audit_anchors_local, dtype=np.float64)
        count = min(len(reasons), segments.shape[0], widths_m.size, anchors_local.shape[0])
        return _CorridorPairAuditVizData(segments, reasons, widths_m, anchors_local, count)

    def _append_corridor_pair_audit_segments(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        audit: _CorridorPairAuditVizData,
    ) -> int:
        for reason in CORRIDOR_PAIR_AUDIT_REASONS:
            if reason == 'pair_valid':
                continue
            indices = self._corridor_pair_audit_indices(audit, reason)
            if not indices:
                continue
            markers.markers.append(
                self._make_pair_segment_marker(
                    frame_id=frame_id, stamp=stamp, marker_id=marker_id,
                    ns=f'corridor_pair_audit_{reason}',
                    pair_segments=audit.segments[indices],
                    color=self._corridor_pair_audit_rgba(reason),
                    width=_CORRIDOR_AUDIT_MARKER_WIDTH_M,
                )
            )
            marker_id += 1
        return marker_id

    @staticmethod
    def _corridor_pair_audit_indices(
        audit: _CorridorPairAuditVizData,
        reason: str,
    ) -> list[int]:
        return [
            idx for idx in range(audit.count)
            if audit.reasons[idx] == reason and np.all(np.isfinite(audit.segments[idx]))
        ]

    def _append_corridor_pair_audit_labels(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        audit: _CorridorPairAuditVizData,
    ) -> int:
        if not getattr(self, 'corridor_pair_audit_show_labels', False):
            return marker_id
        label_count = 0
        max_labels = int(getattr(self, 'corridor_pair_audit_max_labels', 0))
        for idx in range(audit.count):
            label = self._corridor_pair_audit_label(frame_id, stamp, marker_id, audit, idx)
            if label is None:
                continue
            markers.markers.append(label)
            marker_id += 1
            label_count += 1
            if label_count >= max_labels:
                break
        return marker_id

    def _corridor_pair_audit_label(
        self,
        frame_id: str,
        stamp,
        marker_id: int,
        audit: _CorridorPairAuditVizData,
        idx: int,
    ) -> Optional[Marker]:
        reason = audit.reasons[idx]
        if reason == 'pair_valid' or not np.all(np.isfinite(audit.segments[idx])):
            return None
        midpoint = _SEGMENT_MIDPOINT_WEIGHT * (audit.segments[idx, 0, :] + audit.segments[idx, 1, :])
        if not np.all(np.isfinite(midpoint)):
            return None
        label = self._base_corridor_pair_audit_label(frame_id, stamp, marker_id, midpoint)
        label.text = (
            f'pair[{idx}] {reason}\n'
            f'w={float(audit.widths_m[idx]):.2f}m '
            f'local=({float(audit.anchors_local[idx, 0]):.1f},'
            f'{float(audit.anchors_local[idx, 1]):.1f})'
        )
        return label

    @staticmethod
    def _base_corridor_pair_audit_label(
        frame_id: str,
        stamp,
        marker_id: int,
        midpoint: np.ndarray,
    ) -> Marker:
        label = Marker()
        label.header.frame_id = frame_id
        label.header.stamp = stamp
        label.ns = 'corridor_pair_audit_labels'
        label.id = marker_id
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(midpoint[0])
        label.pose.position.y = float(midpoint[1])
        label.pose.position.z = _CORRIDOR_AUDIT_LABEL_Z_M
        label.pose.orientation.w = 1.0
        label.scale.z = _CORRIDOR_AUDIT_LABEL_TEXT_HEIGHT_M
        label.color.r = label.color.g = label.color.b = 1.0
        label.color.a = _CORRIDOR_AUDIT_LABEL_ALPHA
        return label

    @staticmethod
    def _corridor_pair_audit_rgba(reason: str) -> tuple[float, float, float, float]:
        if reason == 'pair_width_too_narrow':
            return 1.0, 0.05, 0.05, 0.9
        if reason == 'pair_width_too_wide':
            return 1.0, 0.45, 0.05, 0.9
        if reason in {'pair_left_beyond_horizon', 'pair_right_beyond_horizon'}:
            return 0.0, 0.85, 1.0, 0.9
        if reason in {'pair_left_behind', 'pair_right_behind'}:
            return 1.0, 0.0, 0.85, 0.9
        if reason == 'pair_not_in_longest_valid_slice':
            return 0.85, 0.85, 0.85, 0.75
        return 0.65, 0.2, 1.0, 0.9

    def _append_raw_path_markers(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: Any,
        raw_centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
    ) -> int:
        marker_id = self._append_raw_midpoint_marker(
            markers, marker_id, frame_id, stamp, result, raw_midpoint_chain,
        )
        marker_id = self._append_raw_offset_path_markers(markers, marker_id, frame_id, stamp, result)
        return self._append_prevalidation_marker(markers, marker_id, frame_id, stamp, raw_centerline)

    def _append_raw_midpoint_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: Any,
        raw_midpoint_chain: np.ndarray,
    ) -> int:
        if not getattr(self, 'show_raw_midpoint_chain', False):
            return marker_id
        midpoint_chain = raw_midpoint_chain
        if hasattr(self, '_last_viz_raw_midpoint_chain'):
            midpoint_chain = self._cached_raw_midpoint_chain(raw_midpoint_chain)
        namespace = 'corridor_center_anchors' if self._has_corridor_pair_audit(result) else 'raw_midpoint_chain'
        markers.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id, stamp=stamp, marker_id=marker_id,
                ns=namespace, points=midpoint_chain,
                color=(1.0, 1.0, 1.0, 0.95), width=0.06, z_offset=0.03,
            )
        )
        return marker_id + 1

    def _cached_raw_midpoint_chain(self, raw_midpoint_chain: np.ndarray) -> np.ndarray:
        midpoint_chain = raw_midpoint_chain
        if midpoint_chain.size > 0:
            self._last_viz_raw_midpoint_chain = np.array(midpoint_chain, copy=True)
        elif self._last_viz_raw_midpoint_chain is not None:
            midpoint_chain = np.array(self._last_viz_raw_midpoint_chain, copy=True)
        return midpoint_chain

    def _append_raw_offset_path_markers(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        result: Any,
    ) -> int:
        if not getattr(self, 'show_raw_offset_path', False):
            return marker_id
        raw_offset_path = self._cached_raw_offset_path(result)
        markers.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id, stamp=stamp, marker_id=marker_id,
                ns='raw_offset_path', points=raw_offset_path,
                color=(0.2, 1.0, 1.0, 0.9), width=0.09, z_offset=0.18,
            )
        )
        marker_id += 1
        points_marker = self._make_points_marker(
            frame_id=frame_id, stamp=stamp, marker_id=marker_id,
            ns='raw_offset_path_points', points=raw_offset_path,
            color=(0.2, 1.0, 1.0, 1.0), scale=0.20,
        )
        for point in points_marker.points:
            point.z = 0.19
        markers.markers.append(points_marker)
        return marker_id + 1

    def _cached_raw_offset_path(self, result: Any) -> np.ndarray:
        raw_offset_path = np.array(getattr(result, 'raw_offset_path', np.empty((0, 2))), copy=True)
        if raw_offset_path.shape[0] > 0:
            self._last_viz_raw_offset_path = np.array(raw_offset_path, copy=True)
        elif getattr(self, '_last_viz_raw_offset_path', None) is not None:
            raw_offset_path = np.array(self._last_viz_raw_offset_path, copy=True)
        return raw_offset_path

    def _append_prevalidation_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        raw_centerline: np.ndarray,
    ) -> int:
        if not getattr(self, 'show_raw_prevalidation_centerline', False):
            return marker_id
        markers.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id, stamp=stamp, marker_id=marker_id,
                ns='raw_prevalidation_centerline', points=raw_centerline,
                color=(1.0, 0.15, 0.85, 0.9), width=0.07, z_offset=0.05,
            )
        )
        return marker_id + 1

    def _append_centerline_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        centerline: np.ndarray,
    ) -> int:
        markers.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id, stamp=stamp, marker_id=marker_id,
                ns='centerline', points=centerline, color=(0.95, 0.15, 0.15, 1.0),
                width=getattr(self, '_centerline_marker_width_m', 0.09), z_offset=0.07,
            )
        )
        return marker_id + 1

    def _append_lookahead_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        control_target_frame: Optional[np.ndarray],
    ) -> int:
        if not getattr(self, 'show_lookahead_point', False) or control_target_frame is None:
            return marker_id
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'lookahead'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.25
        marker.color.r = marker.color.a = 1.0
        marker.pose.position.x = float(control_target_frame[0])
        marker.pose.position.y = float(control_target_frame[1])
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0
        markers.markers.append(marker)
        return marker_id + 1

    @staticmethod
    def _path_point_yaw(path_xy: np.ndarray, idx: int) -> float:
        if path_xy.shape[0] < 2:
            return 0.0
        if idx == path_xy.shape[0] - 1:
            dx, dy = path_xy[idx] - path_xy[idx - 1]
        else:
            dx, dy = path_xy[idx + 1] - path_xy[idx]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return float(math.atan2(dy, dx))

    @staticmethod
    def _make_points_marker(
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        color: tuple[float, float, float, float],
        scale: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.pose.orientation.w = 1.0
        for x, y in points:
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = 0.03
            marker.points.append(pt)
        return marker

    def _update_remembered_cone_viz(
        self,
        *,
        points_xy: np.ndarray,
        colors: list[str],
    ) -> None:
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            self._remembered_cone_viz_points = np.empty((0, 2), dtype=np.float64)
            self._remembered_cone_viz_colors = []
            return

        normalized_colors = [normalize_color(color) for color in colors]
        if len(normalized_colors) < points.shape[0]:
            normalized_colors.extend(['unknown'] * (points.shape[0] - len(normalized_colors)))
        elif len(normalized_colors) > points.shape[0]:
            normalized_colors = normalized_colors[: points.shape[0]]

        finite_mask = np.all(np.isfinite(points), axis=1)
        self._remembered_cone_viz_points = np.array(points[finite_mask], copy=True)
        self._remembered_cone_viz_colors = [
            color
            for color, keep in zip(normalized_colors, finite_mask)
            if bool(keep)
        ]

    def _append_remembered_cone_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
    ) -> int:
        points = np.asarray(
            getattr(self, '_remembered_cone_viz_points', np.empty((0, 2), dtype=np.float64)),
            dtype=np.float64,
        )
        colors = list(getattr(self, '_remembered_cone_viz_colors', []))
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            return marker_id
        if len(colors) < points.shape[0]:
            colors.extend(['unknown'] * (points.shape[0] - len(colors)))
        elif len(colors) > points.shape[0]:
            colors = colors[: points.shape[0]]

        markers.markers.append(
            self._make_colored_points_marker(
                frame_id=frame_id,
                stamp=stamp,
                marker_id=marker_id,
                ns='remembered_cones',
                points=points,
                colors=colors,
                scale=0.18,
            )
        )
        return marker_id + 1

    @classmethod
    def _make_colored_points_marker(
        cls,
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        colors: list[str],
        scale: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        for (x, y), color_name in zip(np.asarray(points, dtype=np.float64), colors):
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = 0.03
            marker.points.append(pt)

            rgba = ColorRGBA()
            rgba.r, rgba.g, rgba.b, rgba.a = cls._cone_marker_rgba(color_name)
            marker.colors.append(rgba)
        return marker

    @staticmethod
    def _cone_marker_rgba(color_name: str) -> tuple[float, float, float, float]:
        color = normalize_color(color_name)
        if color == 'blue':
            return 0.2, 0.55, 1.0, 0.95
        if color == 'yellow':
            return 1.0, 0.92, 0.25, 0.95
        if color == 'orange':
            return 1.0, 0.55, 0.15, 0.95
        return 0.8, 0.8, 0.8, 0.80

    def _make_pair_segment_marker(
        self,
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        pair_segments: Optional[np.ndarray],
        color: tuple[float, float, float, float],
        width: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(width)
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        if pair_segments is None:
            return marker
        for segment in np.asarray(pair_segments, dtype=np.float64):
            if segment.shape != (2, 2):
                continue
            for x, y in segment:
                point = Point()
                point.x = float(x)
                point.y = float(y)
                point.z = 0.03
                marker.points.append(point)
        return marker

    @classmethod
    def _make_edge_list_marker(
        cls,
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        edges: np.ndarray,
        color: tuple[float, float, float, float],
        width: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.pose.orientation.w = 1.0
        cls._append_edge_marker_points(marker, points, edges)
        return marker

    @staticmethod
    def _append_edge_marker_points(
        marker: Marker,
        points: np.ndarray,
        edges: np.ndarray,
    ) -> None:
        for edge in edges:
            a = int(edge[0])
            b = int(edge[1])
            if a < 0 or b < 0 or a >= points.shape[0] or b >= points.shape[0]:
                continue
            p0 = Point()
            p0.x = float(points[a, 0])
            p0.y = float(points[a, 1])
            p0.z = 0.02
            p1 = Point()
            p1.x = float(points[b, 0])
            p1.y = float(points[b, 1])
            p1.z = 0.02
            marker.points.append(p0)
            marker.points.append(p1)

    @staticmethod
    def _make_line_strip_marker(
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        color: tuple[float, float, float, float],
        width: float,
        z_offset: float = 0.02,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.pose.orientation.w = 1.0
        for x, y in points:
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = float(z_offset)
            marker.points.append(pt)
        return marker

    @staticmethod
    def _append_status_marker(
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        status: str,
        *,
        operator_state: str,
    ) -> int:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'status'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 0.32
        marker.color.a = 1.0
        color = _OPERATOR_STATE_COLORS.get(operator_state, _OPERATOR_STATE_COLORS['waiting'])
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.pose.position.x = 1.5
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.35
        marker.pose.orientation.w = 1.0
        marker.text = status
        markers.markers.append(marker)
        return marker_id + 1
