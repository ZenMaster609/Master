"""Cone reconstruction and frame-handling helpers."""

from __future__ import annotations

import math


def camera_intrinsics(camera_info) -> tuple[float, float, float, float]:
    fx = float(camera_info.k[0]) if len(camera_info.k) >= 1 else 0.0
    fy = float(camera_info.k[4]) if len(camera_info.k) >= 5 else 0.0
    cx = float(camera_info.k[2]) if len(camera_info.k) >= 3 else 0.0
    cy = float(camera_info.k[5]) if len(camera_info.k) >= 6 else 0.0
    return fx, fy, cx, cy


def projection_model_for_frame(frame_id: str) -> str:
    frame = str(frame_id).strip().lower()
    if 'optical' in frame or frame.endswith('_camera'):
        return 'optical_z'
    if frame.endswith('_link'):
        return 'forward_x'
    return 'optical_z'


def normalize_detection_color(label: str) -> str:
    token = str(label).strip().lower().replace('-', '_').replace(' ', '_')
    if (
        'big_orange' in token
        or 'large_orange' in token
        or (('big' in token or 'large' in token) and 'orange' in token)
    ):
        return 'big_orange'
    if 'orange' in token:
        return 'orange'
    if 'yellow' in token:
        return 'yellow'
    if 'blue' in token:
        return 'blue'
    return 'unknown'


def reconstruct_cam_point_from_axis(
    *,
    u: float,
    v: float,
    axis_depth: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    model: str,
) -> tuple[float, float, float] | None:
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


def transform_point(transform, x: float, y: float, z: float) -> tuple[float, float, float]:
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


def deduplicate_cone_candidates(
    candidates: list[tuple[float, float, float, str, float]],
    dedup_radius_m: float,
) -> list[tuple[float, float, float, str, float]]:
    if len(candidates) <= 1:
        return list(candidates)

    radius_sq = float(dedup_radius_m) * float(dedup_radius_m)
    merged: list[tuple[float, float, float, str, float, float]] = []

    for x, y, z, color, confidence in sorted(
        candidates,
        key=lambda item: (-item[4], (item[0] * item[0]) + (item[1] * item[1])),
    ):
        best_idx = -1
        best_dist_sq = float('inf')
        for idx, (mx, my, _mz, mcolor, _mconf, _weight) in enumerate(merged):
            if color != mcolor and color != 'unknown' and mcolor != 'unknown':
                continue
            dx = x - mx
            dy = y - my
            dist_sq = (dx * dx) + (dy * dy)
            if dist_sq <= radius_sq and dist_sq < best_dist_sq:
                best_idx = idx
                best_dist_sq = dist_sq

        weight = max(0.1, confidence)
        if best_idx < 0:
            merged.append((x, y, z, color, confidence, weight))
            continue

        mx, my, mz, mcolor, mconf, mweight = merged[best_idx]
        total_weight = mweight + weight
        merged[best_idx] = (
            ((mx * mweight) + (x * weight)) / total_weight,
            ((my * mweight) + (y * weight)) / total_weight,
            ((mz * mweight) + (z * weight)) / total_weight,
            color if confidence >= mconf else mcolor,
            max(mconf, confidence),
            total_weight,
        )

    return [(x, y, z, color, confidence) for x, y, z, color, confidence, _weight in merged]


def resolve_namespaced_output_frame(camera_frame: str, requested_frame: str) -> str:
    requested = str(requested_frame).strip().strip('/')
    source = str(camera_frame).strip().strip('/')
    if not requested or not source or '/' in requested:
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


def cone_output_source_frame_candidates(source_frame: str, requested_output_frame: str) -> list[str]:
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
