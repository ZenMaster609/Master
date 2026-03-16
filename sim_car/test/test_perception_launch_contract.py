from __future__ import annotations

import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'full_sim_launch.launch.py'


def test_full_launch_uses_single_perception_node_with_stereo_toggle():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'stereo'" in content
    assert "DeclareLaunchArgument(\n        'mono'" not in content
    assert "executable='perception_node'" in content
    assert "stereo_enabled': ParameterValue(" in content
    assert "mono_perception_node" not in content
    assert "stereo_perception_node" not in content


def test_full_launch_declares_and_passes_thesis_controller_diagnostics():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'thesis_controller_diagnostics'" in content
    assert "DeclareLaunchArgument(\n        'thesis_controller_diagnostics_live_plot_enabled'" in content
    assert "'thesis_controller_diagnostics'" in content
    assert "'thesis_controller_diagnostics_enabled': LaunchConfiguration('thesis_controller_diagnostics')" in content
    assert (
        "'enable_thesis_controller_diagnostics_plot': LaunchConfiguration(\n"
        "                'thesis_controller_diagnostics_live_plot_enabled'\n"
        "            )"
    ) in content
    assert "'diagnostics.publish_thesis_context': ParameterValue(" in content


def test_full_launch_declares_camera_cone_rmse_plotting_and_wires_it():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument(\n        'camera_cone_rmse_plotting'" in content
    assert content.count("LaunchConfiguration('camera_cone_rmse_plotting')") >= 4
    assert "'camera_cone_eval_topic': PythonExpression([" in content
