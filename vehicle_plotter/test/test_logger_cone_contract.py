from __future__ import annotations

import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOGGER_NODE = REPO_ROOT / 'vehicle_plotter' / 'vehicle_plotter' / 'nodes' / 'logger_node.py'


def test_logger_keeps_cone_depth_samples_and_drops_mono_fit_path():
    content = LOGGER_NODE.read_text(encoding='utf-8')

    assert 'cone_depth_samples' in content
    assert 'cone_depth_monocular_fit_samples' not in content
    assert 'monocular_fit_samples' not in content
