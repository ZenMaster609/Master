from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / 'sim_car' / 'sim_car' / 'run_artifacts_node.py'

vehicle_plotter_msgs = types.ModuleType('vehicle_plotter_msgs')
vehicle_plotter_msgs_msg = types.ModuleType('vehicle_plotter_msgs.msg')
vehicle_plotter_msgs_msg.RunSession = object
vehicle_plotter_msgs.msg = vehicle_plotter_msgs_msg
sys.modules.setdefault('vehicle_plotter_msgs', vehicle_plotter_msgs)
sys.modules.setdefault('vehicle_plotter_msgs.msg', vehicle_plotter_msgs_msg)

spec = importlib.util.spec_from_file_location('run_artifacts_node', MODULE_PATH)
assert spec is not None
assert spec.loader is not None
run_artifacts_node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_artifacts_node)
copy_config_snapshot = run_artifacts_node.copy_config_snapshot


def test_copy_config_snapshot_recurses_and_preserves_subdirectories(tmp_path):
    source_dir = tmp_path / 'config'
    target_dir = tmp_path / 'snapshot'

    top_level_yaml = source_dir / 'sensor_config.yaml'
    nested_yaml = source_dir / 'controllers' / 'stanley.yaml'
    nested_non_yaml = source_dir / 'controllers' / 'notes.txt'

    top_level_yaml.parent.mkdir(parents=True, exist_ok=True)
    nested_yaml.parent.mkdir(parents=True, exist_ok=True)

    top_level_yaml.write_text('top: true\n', encoding='utf-8')
    nested_yaml.write_text('controller: stanley\n', encoding='utf-8')
    nested_non_yaml.write_text('ignore me\n', encoding='utf-8')

    copied = copy_config_snapshot(source_dir, target_dir, '*.yaml')

    assert copied == 2
    assert (target_dir / 'sensor_config.yaml').read_text(encoding='utf-8') == 'top: true\n'
    assert (target_dir / 'controllers' / 'stanley.yaml').read_text(encoding='utf-8') == 'controller: stanley\n'
    assert not (target_dir / 'controllers' / 'notes.txt').exists()
