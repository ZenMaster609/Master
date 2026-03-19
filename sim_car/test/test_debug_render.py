from sim_car.perception.debug_render import _format_debug_label


def test_format_debug_label_abbreviates_blue_and_yellow():
    assert _format_debug_label('blue') == 'b'
    assert _format_debug_label('yellow') == 'y'


def test_format_debug_label_preserves_other_labels_and_whitespace():
    assert _format_debug_label(' orange ') == 'orange'
    assert _format_debug_label('') == ''
