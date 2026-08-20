from pathlib import Path

import pytest

from parastell.port_visualization import VIEW_NAMES
from parastell.port_visualization import _load


def test_render_configuration_has_required_variants():
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "equatorial_diagnostic_port.yaml"
    )
    config = _load(path)
    assert config["port"]["cross_section"] == {
        "shape": "rectangle",
        "width": 40.0,
        "height": 25.0,
    }
    assert config["port"]["layer_span"] == {
        "start": "first_wall",
        "count": 3,
        "direction": "outward",
    }
    assert set(config["variants"]) == {
        "one_layer_rectangle",
        "three_layer_rectangle",
        "two_layer_circle",
    }
    assert len(VIEW_NAMES) == 10
    assert len(set(VIEW_NAMES)) == 10


def test_bad_render_configuration_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        _load(path)
