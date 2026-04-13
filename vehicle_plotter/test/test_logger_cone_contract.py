from __future__ import annotations

import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOGGER_NODE = REPO_ROOT / 'vehicle_plotter' / 'vehicle_plotter' / 'nodes' / 'logger_node.py'
PLOTTER_LAUNCH = REPO_ROOT / 'vehicle_plotter' / 'launch' / 'plotter.launch.py'


def test_logger_keeps_cone_depth_samples_and_drops_mono_fit_path():
    content = LOGGER_NODE.read_text(encoding='utf-8')

    assert 'cone_depth_samples' in content
    assert 'cone_depth_monocular_fit_samples' not in content
    assert 'monocular_fit_samples' not in content


def test_logger_routes_rmse_samples_by_source_and_uses_fixed_output_names():
    logger_content = LOGGER_NODE.read_text(encoding='utf-8')

    assert '_normalize_cone_source_name' in logger_content
    assert "self._cone_range_rmse_samples_by_source" in logger_content
    assert 'cone_range_rmse_samples_mono.csv' in logger_content
    assert 'cone_range_rmse_samples_stereo.csv' in logger_content
    assert 'cone_range_rmse_samples_lidar.csv' in logger_content
    assert 'cone_range_binned_rmse_camera_lidar.png' not in logger_content


def test_logger_and_plotter_launch_expose_path_tracking_eval():
    logger_content = LOGGER_NODE.read_text(encoding='utf-8')
    launch_content = PLOTTER_LAUNCH.read_text(encoding='utf-8')

    assert "self.declare_parameter('path_tracking_eval_enabled', False)" in logger_content
    assert "self.declare_parameter('path_tracking_eval_gt_track_topic', '/ground_truth/track')" in logger_content
    assert "self.declare_parameter('path_tracking_eval_odom_topic', '/sim/odom')" in logger_content
    assert "self.declare_parameter('path_tracking_eval_planner_path_topic', '/planned_centerline')" in logger_content
    assert "self.declare_parameter('path_tracking_eval_track_name', '')" in logger_content
    assert '"path_tracking_eval_enabled"' in launch_content
    assert '"path_tracking_eval_gt_track_topic"' in launch_content
    assert '"path_tracking_eval_odom_topic"' in launch_content
    assert '"path_tracking_eval_planner_path_topic"' in launch_content
    assert '"path_tracking_eval_track_name"' in launch_content


def test_logger_autostop_forces_process_exit_after_cleanup():
    logger_content = LOGGER_NODE.read_text(encoding='utf-8')

    assert 'Smalltrack lap target reached' in logger_content
    assert 'self.shutdown()' in logger_content
    assert 'self._request_process_exit(parent_delay_s=0.1, force_delay_s=5.0)' in logger_content
    assert 'os.kill(parent_pid, signal.SIGINT)' in logger_content
    assert 'os.killpg(process_group_id, signal.SIGINT)' in logger_content
    assert 'threading.Timer(max(0.0, float(parent_delay_s)), _signal_parent_launch)' in logger_content
    assert 'threading.Timer(max(0.0, float(force_delay_s)), _force_exit_process_group)' in logger_content
    assert 'os._exit(0)' in logger_content


def test_plotter_launch_contains_state_plotter_node_and_no_aux_live_plot_nodes():
    launch_content = PLOTTER_LAUNCH.read_text(encoding='utf-8')

    assert 'executable="plotter_node"' in launch_content
    assert 'cone_rmse_plot_node' not in launch_content
    assert 'controller_diagnostics_plot_node' not in launch_content
    assert 'thesis_controller_diagnostics_plot_node' not in launch_content
