import numpy as np
import pytest

import parastell.invessel_build as ivb
from parastell.ports import (
    PortGeometryNotImplementedError,
    parse_ports,
)


class _DummyReferenceSurface:
    def angles_to_xyz(self, toroidal_angles, poloidal_angles, s, scale):
        return np.zeros((1, 3))


def _base_port(name="equatorial_diagnostic_port"):
    return {
        "name": name,
        "placement": {
            "mode": "cartesian",
            "anchor": [600.0, 0.0, 0.0],
            "axis": [-1.0, 0.0, 0.0],
            "reference_direction": [0.0, 0.0, 1.0],
        },
        "cross_section": {
            "shape": "rectangle",
            "width": 40.0,
            "height": 25.0,
        },
        "layer_span": {
            "start": "first_wall",
            "count": 3,
            "direction": "outward",
        },
    }


@pytest.fixture
def user_layer_names():
    return ("first_wall", "breeder", "shield")


def test_parse_ports_noop(user_layer_names):
    assert parse_ports(None, user_layer_names) == ()
    assert parse_ports([], user_layer_names) == ()


def test_parse_port_contract_defaults(user_layer_names):
    ports = parse_ports([_base_port()], user_layer_names)
    assert len(ports) == 1
    spec = ports[0]

    assert spec.name == "equatorial_diagnostic_port"
    assert spec.placement.mode == "cartesian"
    assert np.allclose(spec.placement.anchor, [600.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(spec.placement.axis), 1.0)
    assert np.isclose(spec.placement.max_search_length, 1200.0)
    assert spec.cross_section.shape == "rectangle"
    assert spec.cross_section.width == 40.0
    assert spec.cross_section.height == 25.0
    assert spec.layer_span.start == "first_wall"
    assert spec.layer_span.count == 3
    assert spec.layer_span.direction == "outward"
    assert spec.resolution.layers == ("first_wall", "breeder", "shield")
    assert spec.fill.mat_tag == "Vacuum"
    assert spec.repetition.mode == "single"


def test_parse_port_reference_frame_geometry(user_layer_names):
    spec = parse_ports([_base_port()], user_layer_names)[0]

    axis = np.array(spec.placement.local_axis)
    reference = np.array(spec.placement.local_reference)
    normal = np.array(spec.placement.local_normal)

    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert np.isclose(np.linalg.norm(reference), 1.0)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.isclose(np.dot(axis, reference), 0.0)
    assert np.isclose(np.dot(axis, normal), 0.0)
    assert np.isclose(np.dot(reference, normal), 0.0)
    assert np.allclose(np.cross(reference, normal), axis)


def test_parse_port_layer_span(user_layer_names):
    data = _base_port()
    data["layer_span"] = {
        "start": "breeder",
        "count": 2,
        "direction": "inward",
    }
    spec = parse_ports([data], user_layer_names)[0]
    assert spec.resolution.layers == ("breeder", "first_wall")

    data = _base_port()
    data["layer_span"] = {
        "start": "breeder",
        "count": 2,
        "direction": "outward",
    }
    spec = parse_ports([data], user_layer_names)[0]
    assert spec.resolution.layers == ("breeder", "shield")


def test_parse_ports_rejects_oob_layers(user_layer_names):
    data = _base_port()
    data["layer_span"]["count"] = 4
    with pytest.raises(
        ValueError, match="extends beyond available radial layers"
    ):
        parse_ports([data], user_layer_names)

    data = _base_port("other")
    data["layer_span"] = {
        "start": "not_a_layer",
        "count": 1,
        "direction": "outward",
    }
    with pytest.raises(ValueError, match="not in radial build layers"):
        parse_ports([data], user_layer_names)


def test_parse_ports_rejects_duplicates(user_layer_names):
    p1 = _base_port("same")
    p2 = _base_port("same")
    with pytest.raises(ValueError, match="duplicate port name"):
        parse_ports([p1, p2], user_layer_names)


def test_parse_ports_rejects_unknown_keys(user_layer_names):
    with pytest.raises(ValueError, match="contains unknown keys"):
        parse_ports(
            [{"name": "bad", **_base_port(), "unexpected": 1}],
            user_layer_names,
        )

    invalid_placement = _base_port()
    invalid_placement["placement"]["unexpected"] = 1
    with pytest.raises(ValueError, match="contains unknown keys"):
        parse_ports([invalid_placement], user_layer_names)


def test_parse_ports_requires_cross_section(user_layer_names):
    bad_port = _base_port()
    bad_port.pop("cross_section")
    with pytest.raises(ValueError, match="cross_section is required"):
        parse_ports([bad_port], user_layer_names)


def test_parse_ports_layer_resolution_uses_user_layers():
    radial_build_dict = {
        "first_wall": {"thickness_matrix": np.ones((2, 4)) * 1},
        "breeder": {"thickness_matrix": np.ones((2, 4)) * 2},
        "shield": {"thickness_matrix": np.ones((2, 4)) * 3},
    }
    radial_build = ivb.RadialBuild(
        [0.0, 5.0],
        [0.0, 90.0, 180.0, 360.0],
        1.08,
        radial_build_dict,
        split_chamber=True,
    )
    assert radial_build.user_layer_names == ("first_wall", "breeder", "shield")

    ports = parse_ports([_base_port()], radial_build.user_layer_names)
    assert ports[0].resolution.layers == ("first_wall", "breeder", "shield")


def test_generate_components_rejects_port_geometry():
    radial_build = ivb.RadialBuild(
        [0.0, 5.0],
        [0.0, 90.0, 180.0, 360.0],
        1.08,
        {"component": {"thickness_matrix": np.ones((2, 4)) * 1}},
    )
    port = _base_port()
    port["layer_span"] = {
        "start": "component",
        "count": 1,
        "direction": "outward",
    }
    ivb_obj = ivb.InVesselBuild(
        _DummyReferenceSurface(),
        radial_build,
        ports=[port],
        use_pydagmc=True,
    )

    with pytest.raises(PortGeometryNotImplementedError):
        ivb_obj.generate_components()
