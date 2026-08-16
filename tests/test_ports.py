import numpy as np
import pytest
import cadquery as cq

import parastell.invessel_build as ivb
from parastell.ports import parse_ports


class _SyntheticReferenceSurface:
    def angles_to_xyz(self, toroidal_angles, poloidal_angles, s, scale):
        phi = np.deg2rad(np.asarray(toroidal_angles, dtype=float))
        theta = np.deg2rad(np.asarray(poloidal_angles, dtype=float))
        major_radius = 6.0 * scale
        minor_radius = 1.6 * (1.0 + 0.2 * (s - 1.0)) * scale
        phi = np.atleast_1d(phi)[:, None]
        theta = theta[None, :]
        x = (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
        y = (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
        z = minor_radius * np.sin(theta)
        points = np.stack([x, y, z], axis=-1)
        if points.ndim == 2:
            return points[0]
        return points


def _layer_box(
    x_min,
    x_max,
    y_span=200.0,
    z_span=200.0,
    x_offset=0.0,
    y_offset=0.0,
    z_offset=0.0,
):
    width = x_max - x_min
    body = (
        cq.Workplane("XY")
        .box(width, y_span, z_span)
        .translate(((x_min + x_max) / 2.0 + x_offset, y_offset, z_offset))
    )
    return body.val()


def _disconnected_layer_pair():
    upper = _layer_box(-30, -10, y_offset=110.0)
    lower = _layer_box(-30, -10, y_offset=-110.0)
    return upper.fuse(lower)


def _torus_layer():
    return cq.Solid.makeTorus(80.0, 20.0)


TOROIDAL_ANGLES = [0.0, 90.0, 180.0, 270.0]
POLOIDAL_ANGLES = [0.0, 120.0, 240.0, 360.0]
WALL_S = 1.08


def _base_port(name="equatorial_diagnostic_port"):
    return {
        "name": name,
        "placement": {
            "mode": "cartesian",
            "anchor": [600.0, 0.0, 0.0],
            "axis": [-1.0, 0.0, 0.0],
            "reference_direction": [0.0, 0.0, 1.0],
            "max_search_length": 500.0,
        },
        "cross_section": {
            "shape": "rectangle",
            "width": 40.0,
            "height": 25.0,
        },
        "layer_span": {
            "start": "first_wall",
            "count": 1,
            "direction": "outward",
        },
    }


def _base_box_port(
    name="equatorial_diagnostic_port",
    *,
    anchor=(10.0, 0.0, 0.0),
    axis=(-1.0, 0.0, 0.0),
    layer_span=None,
    cross_section=None,
):
    if layer_span is None:
        layer_span = {
            "start": "first_wall",
            "count": 1,
            "direction": "outward",
        }
    if cross_section is None:
        cross_section = {
            "shape": "rectangle",
            "width": 40.0,
            "height": 25.0,
        }
    return {
        "name": name,
        "placement": {
            "mode": "cartesian",
            "anchor": list(anchor),
            "axis": list(axis),
            "reference_direction": [0.0, 0.0, 1.0],
            "max_search_length": 500.0,
        },
        "cross_section": cross_section,
        "layer_span": layer_span,
        "fill": {"mat_tag": "Vacuum"},
        "repetition": {"mode": "single"},
    }


def _radial_build(layers):
    return {
        layer: {
            "thickness_matrix": np.ones(
                (len(TOROIDAL_ANGLES), len(POLOIDAL_ANGLES))
            )
            * (idx + 1)
            * 5.0,
        }
        for idx, layer in enumerate(layers)
    }


def _synthetic_ivb_from_ports(
    ports=None,
    *,
    layer_components=None,
):
    if layer_components is None:
        layer_components = {
            "first_wall": _layer_box(0.0, 20.0),
            "breeder": _layer_box(-20.0, 0.0),
            "shield": _layer_box(-40.0, -20.0),
        }

    radial_build = ivb.RadialBuild(
        TOROIDAL_ANGLES,
        POLOIDAL_ANGLES,
        WALL_S,
        _radial_build(layer_components.keys()),
    )
    if ports is None:
        kwargs = {}
    else:
        kwargs = {"ports": ports}

    obj = ivb.InVesselBuild(
        _SyntheticReferenceSurface(),
        radial_build,
        num_ribs=11,
        **kwargs,
    )
    obj.Components = dict(layer_components)
    obj._ported_layer_names = set()
    obj._port_fill_components = {}
    obj._port_fill_specs = {}

    if ports:
        obj._apply_ports_to_components()

    return obj


def _component_volumes(obj):
    return {
        name: float(solid.Volume()) for name, solid in obj.Components.items()
    }


def test_parse_ports_noop():
    user_layer_names = ("first_wall", "breeder", "shield")
    assert parse_ports(None, user_layer_names) == ()
    assert parse_ports([], user_layer_names) == ()


def test_parse_port_contract_defaults():
    ports = parse_ports([_base_port()], ("first_wall", "breeder", "shield"))
    spec = ports[0]

    assert spec.name == "equatorial_diagnostic_port"
    assert spec.placement.mode == "cartesian"
    assert np.allclose(spec.placement.anchor, [600.0, 0.0, 0.0])
    assert np.isclose(spec.placement.max_search_length, 500.0)
    assert spec.cross_section.shape == "rectangle"
    assert spec.cross_section.width == 40.0
    assert spec.cross_section.height == 25.0
    assert spec.layer_span.start == "first_wall"
    assert spec.layer_span.count == 1
    assert spec.layer_span.direction == "outward"
    assert spec.resolution.layers == ("first_wall",)
    assert spec.fill.mat_tag == "Vacuum"
    assert spec.repetition.mode == "single"


def test_parse_per_period_mode_is_accepted():
    port = _base_port()
    port["repetition"] = {"mode": "per_period"}
    spec = parse_ports([port], ("first_wall", "breeder", "shield"))[0]
    assert spec.repetition.mode == "per_period"


def test_parse_port_reference_frame_geometry():
    spec = parse_ports([_base_port()], ("first_wall", "breeder", "shield"))[0]

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


def test_parse_port_layer_span():
    data = _base_port()
    data["layer_span"] = {
        "start": "breeder",
        "count": 2,
        "direction": "inward",
    }
    spec = parse_ports([data], ("first_wall", "breeder", "shield"))[0]
    assert spec.resolution.layers == ("breeder", "first_wall")

    data = _base_port()
    data["layer_span"] = {
        "start": "breeder",
        "count": 2,
        "direction": "outward",
    }
    spec = parse_ports([data], ("first_wall", "breeder", "shield"))[0]
    assert spec.resolution.layers == ("breeder", "shield")


def test_parse_ports_rejects_oob_layers():
    data = _base_port()
    data["layer_span"]["count"] = 4
    with pytest.raises(
        ValueError, match="extends beyond available radial layers"
    ):
        parse_ports([data], ("first_wall", "breeder", "shield"))


def test_parse_ports_rejects_unknown_start_layer():
    data = _base_port("bad")
    data["layer_span"]["start"] = "not_a_layer"
    with pytest.raises(ValueError, match="not in radial build layers"):
        parse_ports([data], ("first_wall", "breeder", "shield"))


def test_parse_ports_rejects_duplicates():
    p1 = _base_port("same")
    p2 = _base_port("same")
    with pytest.raises(ValueError, match="duplicate port name"):
        parse_ports([p1, p2], ("first_wall", "breeder", "shield"))


def test_parse_ports_rejects_unknown_keys():
    with pytest.raises(ValueError, match="contains unknown keys"):
        parse_ports(
            [{"name": "bad", **_base_port(), "unexpected": 1}],
            ("first_wall", "breeder", "shield"),
        )

    invalid_placement = _base_port()
    invalid_placement["placement"]["unexpected"] = 1
    with pytest.raises(ValueError, match="contains unknown keys"):
        parse_ports([invalid_placement], ("first_wall", "breeder", "shield"))


def test_parse_ports_requires_cross_section():
    bad_port = _base_port()
    bad_port.pop("cross_section")
    with pytest.raises(ValueError, match="cross_section is required"):
        parse_ports([bad_port], ("first_wall", "breeder", "shield"))


def test_parse_ports_layer_resolution_uses_user_layers():
    radial_build = ivb.RadialBuild(
        [0.0, 90.0, 180.0, 270.0],
        [0.0, 120.0, 240.0, 360.0],
        1.08,
        {
            "first_wall": {"thickness_matrix": np.ones((4, 4)) * 1},
            "breeder": {"thickness_matrix": np.ones((4, 4)) * 2},
            "shield": {"thickness_matrix": np.ones((4, 4)) * 3},
        },
        split_chamber=True,
    )
    assert radial_build.user_layer_names == ("first_wall", "breeder", "shield")

    ports = parse_ports([_base_port()], radial_build.user_layer_names)
    assert ports[0].resolution.layers == ("first_wall",)


def test_generate_rectangular_port_single_layer():
    port = _base_box_port()
    ivb_with_port = _synthetic_ivb_from_ports([port])
    ivb_without_port = _synthetic_ivb_from_ports(None)

    remaining = ivb_with_port.Components["first_wall"].Volume()
    fill = ivb_with_port.Components[port["name"]].Volume()
    original = ivb_without_port.Components["first_wall"].Volume()
    assert np.isclose(original, remaining + fill, rtol=1e-10, atol=1e-6)


def test_generate_rectangular_port_multiple_outward_layers():
    port = _base_box_port(
        "multi",
        layer_span={
            "start": "breeder",
            "count": 2,
            "direction": "outward",
        },
        anchor=(-10.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
    )
    ivb_with_port = _synthetic_ivb_from_ports([port])
    ivb_without_port = _synthetic_ivb_from_ports(None)

    assert (
        ivb_with_port.Components["breeder"].Volume()
        < ivb_without_port.Components["breeder"].Volume()
    )
    assert (
        ivb_with_port.Components["shield"].Volume()
        < ivb_without_port.Components["shield"].Volume()
    )
    assert ivb_with_port.Components["first_wall"].Volume() == pytest.approx(
        ivb_without_port.Components["first_wall"].Volume()
    )


def test_generate_port_from_middle_layer_inward_and_outward_span():
    inward_port = _base_box_port(
        "mid_inward",
        layer_span={
            "start": "breeder",
            "count": 2,
            "direction": "inward",
        },
        anchor=(-10.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
    )
    ivb_with_port = _synthetic_ivb_from_ports([inward_port])
    ivb_without_port = _synthetic_ivb_from_ports(None)

    assert (
        ivb_with_port.Components["breeder"].Volume()
        < ivb_without_port.Components["breeder"].Volume()
    )
    assert (
        ivb_with_port.Components["first_wall"].Volume()
        < ivb_without_port.Components["first_wall"].Volume()
    )
    assert ivb_with_port.Components["shield"].Volume() == pytest.approx(
        ivb_without_port.Components["shield"].Volume()
    )


def test_generate_circular_port():
    port = _base_box_port(
        "circular",
        cross_section={"shape": "circle", "radius": 10.0},
        anchor=(10.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
    )
    ivb_obj = _synthetic_ivb_from_ports([port])
    assert "circular" in ivb_obj.Components
    assert ivb_obj.Components["circular"].Volume() > 0


def test_generate_non_axis_aligned_port():
    port = _base_box_port(
        "tilted",
        anchor=(10.0, 0.0, 0.0),
        axis=(0.2, 1.0, 0.4),
    )
    ivb_obj = _synthetic_ivb_from_ports([port])
    assert "tilted" in ivb_obj.Components
    assert ivb_obj.Components["tilted"].Volume() > 0


def test_selected_layers_reduced_and_unselected_layers_preserved():
    port = _base_box_port(
        "selector",
        layer_span={
            "start": "breeder",
            "count": 2,
            "direction": "outward",
        },
        anchor=(-10.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
    )
    ported = _synthetic_ivb_from_ports([port])
    plain = _synthetic_ivb_from_ports(None)

    assert (
        ported.Components["breeder"].Volume()
        < plain.Components["breeder"].Volume()
    )
    assert (
        ported.Components["shield"].Volume()
        < plain.Components["shield"].Volume()
    )
    assert ported.Components["first_wall"].Volume() == pytest.approx(
        plain.Components["first_wall"].Volume()
    )


def test_port_volume_closure_and_no_overlap():
    port = _base_box_port(
        "closure",
        layer_span={
            "start": "first_wall",
            "count": 3,
            "direction": "outward",
        },
        anchor=(10.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
    )
    plain = _synthetic_ivb_from_ports(None)
    ivb_obj = _synthetic_ivb_from_ports([port])

    selected = ["first_wall", "breeder", "shield"]
    selected_original = sum(
        plain.Components[name].Volume() for name in selected
    )
    selected_remaining = sum(
        ivb_obj.Components[name].Volume() for name in selected
    )
    port_fill = ivb_obj.Components["closure"].Volume()

    # expected closure only for the selected material span
    assert np.isclose(
        selected_original,
        selected_remaining + port_fill,
        rtol=1e-10,
        atol=1e-6,
    )

    for name in selected:
        overlap = (
            ivb_obj.Components[name]
            .intersect(ivb_obj.Components["closure"])
            .Volume()
        )
        assert overlap == pytest.approx(0.0, abs=1e-8)


def test_explicit_port_fill_metadata_preserved():
    port = _base_box_port("metadata")
    ivb_obj = _synthetic_ivb_from_ports([port])
    solids, mat_tags = ivb_obj.extract_solids_and_mat_tags()

    idx = list(ivb_obj.Components.keys()).index("metadata")
    assert mat_tags[idx] == "Vacuum"

    custom = _base_box_port(
        "custom",
        layer_span={"start": "breeder", "count": 1, "direction": "outward"},
        anchor=(-10.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
    )
    custom["fill"] = {"mat_tag": "void_tag"}
    custom_ivb = _synthetic_ivb_from_ports([custom])
    _, mat_tags = custom_ivb.extract_solids_and_mat_tags()
    idx = list(custom_ivb.Components.keys()).index("custom")
    assert mat_tags[idx] == "void_tag"


def test_generate_port_rejects_missed_layer_by_geometry():
    port = _base_box_port(
        "miss", anchor=(1000.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="does not intersect"):
        _synthetic_ivb_from_ports([port])


def test_generate_port_rejects_out_of_range_span():
    port = _base_box_port("oob")
    port["layer_span"]["count"] = 10
    with pytest.raises(
        ValueError, match="extends beyond available radial layers"
    ):
        parse_ports([port], ("first_wall", "breeder", "shield"))


def test_generate_port_rejects_multiple_intersections_per_layer():
    port = _base_box_port(
        "multi_frag",
        cross_section={"shape": "rectangle", "width": 120.0, "height": 120.0},
    )
    components = {
        "first_wall": _disconnected_layer_pair(),
        "breeder": _layer_box(-20.0, 0.0),
        "shield": _layer_box(-40.0, -20.0),
    }
    with pytest.raises(NotImplementedError, match="multiple disconnected"):
        _synthetic_ivb_from_ports([port], layer_components=components)


def test_generate_port_rejects_sector_seam_ambiguity():
    port = _base_box_port(
        "seam",
        anchor=(140.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
        cross_section={"shape": "rectangle", "width": 60.0, "height": 60.0},
    )
    components = {
        "first_wall": _torus_layer(),
        "breeder": _layer_box(-20.0, 0.0),
        "shield": _layer_box(-40.0, -20.0),
    }
    with pytest.raises(NotImplementedError, match="multiple disconnected"):
        _synthetic_ivb_from_ports([port], layer_components=components)


def test_generate_two_non_overlapping_ports():
    p1 = _base_box_port(
        "p1",
        layer_span={"start": "first_wall", "count": 1, "direction": "outward"},
        anchor=(10.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
    )
    p2 = _base_box_port(
        "p2",
        layer_span={"start": "shield", "count": 1, "direction": "outward"},
        anchor=(-30.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
    )
    ivb_obj = _synthetic_ivb_from_ports([p1, p2])

    assert "p1" in ivb_obj.Components
    assert "p2" in ivb_obj.Components
    assert ivb_obj.Components["p1"].intersect(
        ivb_obj.Components["p2"]
    ).Volume() == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_generate_rejects_overlapping_port_policy():
    p1 = _base_box_port(
        "overlap",
        layer_span={"start": "first_wall", "count": 1, "direction": "outward"},
        anchor=(10.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
    )
    p2 = _base_box_port(
        "overlap_2",
        layer_span={"start": "first_wall", "count": 1, "direction": "outward"},
        anchor=(10.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
    )

    with pytest.raises(
        NotImplementedError, match="overlaps an existing port fill"
    ):
        _synthetic_ivb_from_ports([p1, p2])


def test_generate_unported_geometry_unchanged_when_ports_is_none():
    no_ports = _synthetic_ivb_from_ports(None)
    none_ports = _synthetic_ivb_from_ports([])
    assert _component_volumes(no_ports) == _component_volumes(none_ports)


def test_generate_unported_geometry_unchanged_when_ports_empty_list():
    with_list = _synthetic_ivb_from_ports([])
    with_no = _synthetic_ivb_from_ports(None)
    assert _component_volumes(with_list) == _component_volumes(with_no)


def test_generate_with_use_pydagmc_and_ports_not_implemented():
    radial_build = ivb.RadialBuild(
        TOROIDAL_ANGLES,
        POLOIDAL_ANGLES,
        WALL_S,
        _radial_build(["first_wall", "breeder", "shield"]),
    )
    ivb_obj = ivb.InVesselBuild(
        _SyntheticReferenceSurface(),
        radial_build,
        ports=[_base_port()],
        use_pydagmc=True,
    )
    with pytest.raises(
        NotImplementedError, match="Port geometry is not supported"
    ):
        ivb_obj.generate_components()


def test_mesh_components_moab_rejects_affected_ported_components():
    port = _base_box_port("moab")
    ivb_obj = _synthetic_ivb_from_ports([port])

    with pytest.raises(
        NotImplementedError, match="MOAB mesh workflow is not supported"
    ):
        ivb_obj.mesh_components_moab(["first_wall"])


def test_per_period_not_implemented():
    port = _base_box_port(
        "per_period", anchor=(10.0, 0.0, 0.0), axis=(-1.0, 0.0, 0.0)
    )
    port["repetition"] = {"mode": "per_period"}
    with pytest.raises(NotImplementedError, match="per_period"):
        _synthetic_ivb_from_ports([port])
