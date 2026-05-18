"""Per-track path tracking metrics report writer."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .path_tracking_eval_plots import (
    _point_to_polyline_set_distance_m,
    _read_rows,
    build_skidpad_gt_overlay_segments,
)


TRACK_NAME_SKIDPAD = "skidpad"
STATUS_OK = "ok"
# Rows with the valid flag set to 1.0 are treated as evaluation-ready samples.
VALID_SAMPLE_FLAG_MIN = 0.5
# P95 is the thesis-facing tail metric already used elsewhere in the evaluator.
PERCENTILE_P95 = 95.0

PATH_TRACKING_ROW_COLUMNS = {
    "timestamp_sec",
    "sample_valid_flag",
    "status",
    "resolved_control_frame",
    "vehicle_x_m",
    "vehicle_y_m",
    "body_center_x_m",
    "body_center_y_m",
    "front_axle_x_m",
    "front_axle_y_m",
    "planner_reference_x_m",
    "planner_reference_y_m",
    "planner_reference_vs_gt_cte_m",
    "body_center_vs_planner_cte_m",
    "front_axle_vs_planner_cte_m",
    "controller_vs_planner_cte_m",
    "body_center_vs_gt_cte_m",
    "front_axle_vs_gt_cte_m",
    "controller_vs_gt_cte_m",
    "planner_vs_gt_cte_rms_m",
    "planner_vs_gt_cte_p95_m",
    "planner_vs_gt_cte_max_m",
}

SUMMARY_METRIC_KEYS = [
    "sample_count",
    "valid_sample_count",
    "duration_sec",
    "planner_reference_vs_gt_cte_rms_m",
    "planner_reference_vs_gt_cte_p95_m",
    "planner_reference_vs_gt_cte_max_m",
    "planner_vs_gt_cte_rms_m",
    "planner_vs_gt_cte_p95_m",
    "planner_vs_gt_cte_max_m",
    "body_center_vs_planner_cte_rms_m",
    "body_center_vs_planner_cte_p95_m",
    "body_center_vs_planner_cte_max_m",
    "front_axle_vs_planner_cte_rms_m",
    "front_axle_vs_planner_cte_p95_m",
    "front_axle_vs_planner_cte_max_m",
    "controller_vs_planner_cte_rms_m",
    "controller_vs_planner_cte_p95_m",
    "controller_vs_planner_cte_max_m",
    "body_center_vs_gt_cte_rms_m",
    "body_center_vs_gt_cte_p95_m",
    "body_center_vs_gt_cte_max_m",
    "front_axle_vs_gt_cte_rms_m",
    "front_axle_vs_gt_cte_p95_m",
    "front_axle_vs_gt_cte_max_m",
    "controller_vs_gt_cte_rms_m",
    "controller_vs_gt_cte_p95_m",
    "controller_vs_gt_cte_max_m",
]

STATUS_COUNT_COLUMNS = {
    "ok": "status_count_ok",
    "partial_metrics": "status_count_partial_metrics",
    "waiting_for_odom": "status_count_waiting_for_odom",
    "waiting_for_gt_track": "status_count_waiting_for_gt_track",
    "waiting_for_planner_path": "status_count_waiting_for_planner_path",
    "frame_transform_unavailable": "status_count_frame_transform_unavailable",
}

TRACK_METRICS_FIELDNAMES = [
    "run_id",
    "session_start_time",
    "track",
    "planner",
    "controller",
    "lidar_pipeline",
    "lidar_enabled",
    "camera_mode",
    "stereo",
    "completed_laps",
    "lap_target",
    "avg_lap_time_sec",
    "speed_min_mps",
    "speed_max_mps",
    "desired_speed_mps",
    "curvature_speed_gain",
    "lowpass_speed_alpha",
    "camera_range_m",
    "lidar_near_band_limit_m",
    "prefer_lidar_if_camera_missing_far",
    "allow_camera_fallback_near",
    "avg_track_width_m",
    "planner_vs_gt_avg_dist_m",
    "front_axle_vs_gt_avg_dist_m",
    "front_axle_vs_planner_avg_dist_m",
    *SUMMARY_METRIC_KEYS,
    *STATUS_COUNT_COLUMNS.values(),
    "status_counts_json",
    "path_tracking_csv",
    "path_tracking_summary_json",
    "path_tracking_overlay_pdf",
]


def build_track_metrics_record(
    *,
    session_path: Path,
    run_id: str | None = None,
    completed_laps: int | None = None,
    lap_target: int | None = None,
    avg_lap_time_sec: float | None = None,
    overlay_average_distances: dict[str, float] | None = None,
    avg_track_width_m: float | None = None,
) -> dict[str, Any]:
    """Build one per-run path tracking metrics record from a run folder."""
    session_path = Path(session_path)
    launch_params = _load_yaml_mapping(session_path / "configs" / "launch_parameters.yaml")
    session_info = _load_json_mapping(session_path / "session_info.json")
    summary_path = session_path / "logs" / "path_tracking_eval_summary.json"
    summary = _load_json_mapping(summary_path)

    track = _clean_string(launch_params.get("track")) or "unknown"
    planner = _clean_string(launch_params.get("planner"))
    controller = _clean_string(launch_params.get("controller"))
    lidar_pipeline = _clean_string(launch_params.get("lidar_pipeline"))
    stereo_enabled = _bool_or_none(launch_params.get("stereo"))
    camera_mode = "stereo" if stereo_enabled is True else "mono" if stereo_enabled is False else ""

    speed_control = _load_speed_control(session_path, track)
    speed_min_mps = _float_or_none(speed_control.get("speed_min_mps"))
    speed_max_mps = _float_or_none(speed_control.get("speed_max_mps"))
    desired_speed_mps = speed_max_mps

    if lap_target is None:
        lap_target = _load_lap_target(session_path, track)

    camera_range_m = _float_or_none(launch_params.get("camera_range_m"))
    if camera_range_m is None:
        camera_range_m = _load_cone_memory_float(session_path, "camera_range_m")
    lidar_near_band_limit_m = None
    if camera_range_m is not None:
        lidar_near_band_limit_m = 20.0 - float(camera_range_m)

    status_counts = _extract_status_counts(summary)
    overlay_average_distances = overlay_average_distances or {}
    run_id_value = _clean_string(run_id) or _clean_string(session_info.get("run_id")) or session_path.name

    record: dict[str, Any] = {
        "run_id": run_id_value,
        "session_start_time": _clean_string(session_info.get("start_time")),
        "track": track,
        "planner": planner,
        "controller": controller,
        "lidar_pipeline": lidar_pipeline,
        "lidar_enabled": _bool_or_none(launch_params.get("lidar_enabled")),
        "camera_mode": camera_mode,
        "stereo": stereo_enabled,
        "completed_laps": _int_or_none(completed_laps),
        "lap_target": _int_or_none(lap_target),
        "avg_lap_time_sec": _float_or_none(avg_lap_time_sec),
        "speed_min_mps": speed_min_mps,
        "speed_max_mps": speed_max_mps,
        "desired_speed_mps": desired_speed_mps,
        "curvature_speed_gain": _float_or_none(speed_control.get("curvature_speed_gain")),
        "lowpass_speed_alpha": _float_or_none(speed_control.get("lowpass_speed_alpha")),
        "camera_range_m": camera_range_m,
        "lidar_near_band_limit_m": lidar_near_band_limit_m,
        "prefer_lidar_if_camera_missing_far": _bool_or_none(
            launch_params.get("prefer_lidar_if_camera_missing_far")
        ),
        "allow_camera_fallback_near": _bool_or_none(launch_params.get("allow_camera_fallback_near")),
        "avg_track_width_m": _float_or_none(avg_track_width_m),
        "planner_vs_gt_avg_dist_m": _float_or_none(
            overlay_average_distances.get("planner_vs_gt_avg_dist_m")
        ),
        "front_axle_vs_gt_avg_dist_m": _float_or_none(
            overlay_average_distances.get("controller_vs_gt_avg_dist_m")
        ),
        "front_axle_vs_planner_avg_dist_m": _float_or_none(
            overlay_average_distances.get("controller_vs_planner_avg_dist_m")
        ),
        "status_counts_json": json.dumps(status_counts, sort_keys=True),
        "path_tracking_csv": _relative_to_session(session_path, session_path / "logs" / "path_tracking_eval.csv"),
        "path_tracking_summary_json": _relative_to_session(session_path, summary_path),
        "path_tracking_overlay_pdf": _relative_to_session(
            session_path,
            session_path / "plots" / "path_tracking_eval_overlay.pdf",
        ),
    }

    for key in SUMMARY_METRIC_KEYS:
        record[key] = _float_or_none(summary.get(key))
    for status, column in STATUS_COUNT_COLUMNS.items():
        record[column] = _int_or_none(status_counts.get(status))
    _apply_track_specific_metric_overrides(
        record=record,
        session_path=session_path,
        track=track,
    )

    return record


def write_track_metrics_report(
    *,
    session_path: Path,
    base_path: Path | None = None,
    run_id: str | None = None,
    completed_laps: int | None = None,
    lap_target: int | None = None,
    avg_lap_time_sec: float | None = None,
    overlay_average_distances: dict[str, float] | None = None,
    avg_track_width_m: float | None = None,
) -> tuple[Path, Path]:
    """Append or replace one run in the per-track CSV and JSONL reports."""
    session_path = Path(session_path)
    record = build_track_metrics_record(
        session_path=session_path,
        run_id=run_id,
        completed_laps=completed_laps,
        lap_target=lap_target,
        avg_lap_time_sec=avg_lap_time_sec,
        overlay_average_distances=overlay_average_distances,
        avg_track_width_m=avg_track_width_m,
    )

    output_base_path = Path(base_path) if base_path is not None else session_path.parent
    output_dir = output_base_path / "track_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    track = _sanitize_filename_part(str(record.get("track") or "unknown"))
    csv_path = output_dir / f"{track}_runs.csv"
    jsonl_path = output_dir / f"{track}_runs.jsonl"

    _upsert_csv_record(csv_path, record)
    _upsert_jsonl_record(jsonl_path, record)
    return csv_path, jsonl_path


def _upsert_csv_record(csv_path: Path, record: dict[str, Any]) -> None:
    existing_rows = _read_csv_records(csv_path)
    rows = [row for row in existing_rows if str(row.get("run_id", "")).strip() != str(record["run_id"])]
    rows.append({key: _csv_value(record.get(key)) for key in TRACK_METRICS_FIELDNAMES})

    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_METRICS_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(csv_path)


def _upsert_jsonl_record(jsonl_path: Path, record: dict[str, Any]) -> None:
    existing_records = _read_jsonl_records(jsonl_path)
    records = [
        item
        for item in existing_records
        if str(item.get("run_id", "")).strip() != str(record["run_id"])
    ]
    records.append(_json_ready_record(record))

    temp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    temp_path.replace(jsonl_path)


def _read_csv_records(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader if row]


def _read_jsonl_records(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_speed_control(session_path: Path, track: str) -> dict[str, Any]:
    spawn_config = _load_yaml_mapping(
        session_path / "configs" / "sim_car_config" / str(track) / "spawn.yaml"
    )
    speed_control = spawn_config.get("speed_control")
    return speed_control if isinstance(speed_control, dict) else {}


def _load_lap_target(session_path: Path, track: str) -> int | None:
    spawn_config = _load_yaml_mapping(
        session_path / "configs" / "sim_car_config" / str(track) / "spawn.yaml"
    )
    lap_tracking = spawn_config.get("lap_tracking")
    if not isinstance(lap_tracking, dict):
        return None
    return _int_or_none(lap_tracking.get("auto_suspend_after_laps"))


def _load_cone_memory_float(session_path: Path, key: str) -> float | None:
    config = _load_yaml_mapping(session_path / "configs" / "sim_car_config" / "cone_memory.yaml")
    params = config.get("cone_memory_node")
    if isinstance(params, dict):
        ros_params = params.get("ros__parameters")
        if isinstance(ros_params, dict):
            return _float_or_none(ros_params.get(key))
    return None


def _apply_track_specific_metric_overrides(
    *,
    record: dict[str, Any],
    session_path: Path,
    track: str,
) -> None:
    if str(track).strip().lower() != TRACK_NAME_SKIDPAD:
        return
    rows = _load_path_tracking_rows(session_path)
    valid_rows = _valid_evaluation_rows(rows)
    if not valid_rows:
        return
    _apply_skidpad_metric_overrides(record=record, valid_rows=valid_rows, sample_count=len(rows))


def _load_path_tracking_rows(session_path: Path) -> list[dict[str, float | str]]:
    csv_path = session_path / "logs" / "path_tracking_eval.csv"
    return _read_rows(csv_path, PATH_TRACKING_ROW_COLUMNS)


def _valid_evaluation_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    return [row for row in rows if _is_valid_evaluation_row(row)]


def _is_valid_evaluation_row(row: dict[str, float | str]) -> bool:
    sample_valid_flag = _float_or_none(row.get("sample_valid_flag"))
    status = str(row.get("status") or "").strip().lower()
    return bool(sample_valid_flag is not None and sample_valid_flag > VALID_SAMPLE_FLAG_MIN and status == STATUS_OK)


def _apply_skidpad_metric_overrides(
    *,
    record: dict[str, Any],
    valid_rows: list[dict[str, float | str]],
    sample_count: int,
) -> None:
    gt_segments_xy = build_skidpad_gt_overlay_segments()
    planner_vs_gt_stats = _point_distance_stats(valid_rows, gt_segments_xy, _planner_reference_point_xy)
    body_center_vs_gt_stats = _point_distance_stats(valid_rows, gt_segments_xy, _body_center_point_xy)
    front_axle_vs_gt_stats = _point_distance_stats(valid_rows, gt_segments_xy, _front_axle_point_xy)
    controller_vs_gt_stats = _point_distance_stats(valid_rows, gt_segments_xy, _controller_point_xy)
    body_center_vs_planner_stats = _absolute_series_stats(valid_rows, _body_center_vs_planner_error_m)
    front_axle_vs_planner_stats = _absolute_series_stats(valid_rows, _front_axle_vs_planner_error_m)
    controller_vs_planner_stats = _absolute_series_stats(valid_rows, _controller_vs_planner_error_m)
    record["sample_count"] = float(sample_count)
    record["valid_sample_count"] = float(len(valid_rows))
    record["planner_vs_gt_avg_dist_m"] = planner_vs_gt_stats["mean_m"]
    record["front_axle_vs_gt_avg_dist_m"] = front_axle_vs_gt_stats["mean_m"]
    record["front_axle_vs_planner_avg_dist_m"] = front_axle_vs_planner_stats["mean_m"]
    _apply_stats_to_record(record, "planner_reference_vs_gt_cte", planner_vs_gt_stats)
    _apply_stats_to_record(record, "planner_vs_gt_cte", planner_vs_gt_stats)
    _apply_stats_to_record(record, "body_center_vs_gt_cte", body_center_vs_gt_stats)
    _apply_stats_to_record(record, "front_axle_vs_gt_cte", front_axle_vs_gt_stats)
    _apply_stats_to_record(record, "controller_vs_gt_cte", controller_vs_gt_stats)
    _apply_stats_to_record(record, "body_center_vs_planner_cte", body_center_vs_planner_stats)
    _apply_stats_to_record(record, "front_axle_vs_planner_cte", front_axle_vs_planner_stats)
    _apply_stats_to_record(record, "controller_vs_planner_cte", controller_vs_planner_stats)


def _apply_stats_to_record(record: dict[str, Any], metric_prefix: str, stats: dict[str, float | None]) -> None:
    record[f"{metric_prefix}_rms_m"] = stats["rms_m"]
    record[f"{metric_prefix}_p95_m"] = stats["p95_m"]
    record[f"{metric_prefix}_max_m"] = stats["max_m"]


def _point_distance_stats(
    rows: list[dict[str, float | str]],
    gt_segments_xy: list[np.ndarray],
    point_getter,
) -> dict[str, float | None]:
    distances_m = [
        _distance_to_polyline_set_m(point_getter(row), gt_segments_xy)
        for row in rows
    ]
    return _finite_distance_stats(distances_m)


def _absolute_series_stats(
    rows: list[dict[str, float | str]],
    value_getter,
) -> dict[str, float | None]:
    values_m = [value_getter(row) for row in rows]
    return _finite_distance_stats(values_m)


def _finite_distance_stats(values_m: list[float]) -> dict[str, float | None]:
    finite_values_m = np.asarray(
        [abs(float(value_m)) for value_m in values_m if value_m is not None and math.isfinite(float(value_m))],
        dtype=np.float64,
    )
    if finite_values_m.size == 0:
        return {"mean_m": None, "rms_m": None, "p95_m": None, "max_m": None}
    return {
        "mean_m": float(np.mean(finite_values_m)),
        "rms_m": float(math.sqrt(float(np.mean(np.square(finite_values_m))))),
        "p95_m": float(np.percentile(finite_values_m, PERCENTILE_P95)),
        "max_m": float(np.max(finite_values_m)),
    }


def _distance_to_polyline_set_m(point_xy: np.ndarray | None, gt_segments_xy: list[np.ndarray]) -> float | None:
    if point_xy is None:
        return None
    distance_m = _point_to_polyline_set_distance_m(point_xy, gt_segments_xy)
    return distance_m if math.isfinite(distance_m) else None


def _planner_reference_point_xy(row: dict[str, float | str]) -> np.ndarray | None:
    return _row_point_xy(row, "planner_reference_x_m", "planner_reference_y_m")


def _body_center_point_xy(row: dict[str, float | str]) -> np.ndarray | None:
    return _row_point_xy(row, "body_center_x_m", "body_center_y_m")


def _front_axle_point_xy(row: dict[str, float | str]) -> np.ndarray | None:
    return _row_point_xy(row, "front_axle_x_m", "front_axle_y_m")


def _controller_point_xy(row: dict[str, float | str]) -> np.ndarray | None:
    if _resolved_control_frame_name(row) == "front_axle":
        front_axle_xy = _front_axle_point_xy(row)
        if front_axle_xy is not None:
            return front_axle_xy
    return _body_center_point_xy(row)


def _row_point_xy(row: dict[str, float | str], x_key: str, y_key: str) -> np.ndarray | None:
    x_m = _float_or_none(row.get(x_key))
    y_m = _float_or_none(row.get(y_key))
    if x_m is None or y_m is None:
        return None
    return np.asarray([x_m, y_m], dtype=np.float64)


def _body_center_vs_planner_error_m(row: dict[str, float | str]) -> float | None:
    return _float_or_none(row.get("body_center_vs_planner_cte_m"))


def _front_axle_vs_planner_error_m(row: dict[str, float | str]) -> float | None:
    return _float_or_none(row.get("front_axle_vs_planner_cte_m"))


def _controller_vs_planner_error_m(row: dict[str, float | str]) -> float | None:
    if _resolved_control_frame_name(row) == "front_axle":
        front_axle_error_m = _front_axle_vs_planner_error_m(row)
        if front_axle_error_m is not None:
            return front_axle_error_m
    return _float_or_none(row.get("controller_vs_planner_cte_m"))


def _resolved_control_frame_name(row: dict[str, float | str]) -> str:
    return str(row.get("resolved_control_frame") or "").strip().lower()


def _extract_status_counts(summary: dict[str, Any]) -> dict[str, int]:
    status_counts: dict[str, int] = {}
    for key, value in summary.items():
        if not str(key).startswith("status_count::"):
            continue
        status = str(key).split("::", 1)[1]
        count = _int_or_none(value)
        if count is not None:
            status_counts[status] = count
    return status_counts


def _clean_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return None
    return out


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _json_ready_record(record: dict[str, Any]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key in TRACK_METRICS_FIELDNAMES:
        value = record.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            ready[key] = None
        else:
            ready[key] = value
    try:
        ready["status_counts"] = json.loads(str(record.get("status_counts_json", "{}")))
    except json.JSONDecodeError:
        ready["status_counts"] = {}
    return ready


def _relative_to_session(session_path: Path, path: Path) -> str:
    try:
        return str(path.relative_to(session_path))
    except ValueError:
        return str(path)


def _sanitize_filename_part(value: str) -> str:
    safe = []
    for char in str(value).strip().lower():
        if char.isalnum() or char in {"_", "-"}:
            safe.append(char)
        else:
            safe.append("_")
    return "_".join(part for part in "".join(safe).split("_") if part) or "unknown"
