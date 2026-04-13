from __future__ import annotations

import pathlib
import re
import sys


TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from vehicle_plotter.core.run_session import RunSession, sanitize_run_id_prefix


def test_create_new_uses_custom_run_id_prefix(tmp_path):
    session = RunSession.create_new(tmp_path, run_id_prefix='small_mid_stan_3d')

    assert re.fullmatch(r'small_mid_stan_3d_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', session.run_id)
    assert session.session_path == tmp_path / session.run_id


def test_sanitize_run_id_prefix_keeps_abbreviations_filesystem_safe():
    assert sanitize_run_id_prefix(' acc/SB pp 2d ') == 'acc_SB_pp_2d'
    assert sanitize_run_id_prefix('small_mid_stan_3d') == 'small_mid_stan_3d'
