from __future__ import annotations

import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_LAUNCH = REPO_ROOT / 'sim_car' / 'launch' / 'full_sim_launch.launch.py'


def test_sensor_pipeline_enables_vehicle_state_logging_without_logging_flag():
    content = FULL_LAUNCH.read_text(encoding='utf-8')

    expected = (
        "'enable_state_logging': PythonExpression([\n"
        "                \"'true' if ('\",\n"
        "                LaunchConfiguration('sensor_pipeline'),\n"
        "                \"'.lower() == 'true' or '\",\n"
        "                LaunchConfiguration('logging'),\n"
        "                \"'.lower() == 'true') else 'false'\"\n"
        "            ]),"
    )

    assert expected in content
