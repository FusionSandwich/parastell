from pathlib import Path

import numpy as np
import pytest
import cadquery as cq

import parastell.invessel_build as ivb
import parastell.parastell as ps
import parastell.magnet_coils as magnet_coils
from parastell.magnet_coils import MagnetSolidRecord
from parastell.ports import parse_ports
from parastell.utils import ribs_from_kisslinger_format


class _SyntheticReferenceSurface(ivb.ReferenceSurface):
    def __init__(self):
        super().__init__()

    def angles_to_xyz(self, toroidal_angles, poloidal_angles, s, scale):
        phi = np.asarray(toroidal_angles, dtype=float)
        theta = np.asarray(poloidal_angles, dtype=float)
        scalar_phi = phi.ndim == 0
        major_radius = 6.0 * scale
        minor_radius = 1.6 * (1.0 + 0.2 * (s - 1.0)) * scale
        phi = np.atleast_1d(phi)[:, None]
        theta = theta[None, :]
        x = (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
        y = (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
        z = minor_radius * np.sin(theta)
        points = np.stack([x, y, z], axis=-1)
        if scalar_phi:
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


def _layer_box_z(z_min, z_max, x_span=200.0, y_span=200.0):
    depth = z_max - z_min
    return (
        cq.Workplane("XY")
        .box(x_span, y_span, depth)
        .translate((0.0, 0.0, (z_min + z_max) / 2.0))
        .val()
    )


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
    reference_direction=(0.0, 0.0, 1.0),
    max_search_length=500.0,
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
            "reference_direction": list(reference_direction),
            "max_search_length": max_search_length,
        },
        "cross_section": cross_section,
        "layer_span": layer_span,
        "fill": {"mat_tag": "Vacuum"},
        "repetition": {"mode": "single"},
    }


def _explicit_box_port(
    name="engineering_port",
    *,
    start=None,
    end=None,
    outer_extension=0.0,
    liner=None,
    collision=None,
    expected_layers=None,
    cross_section=None,
    anchor=(10.0, 0.0, 0.0),
    axis=(-1.0, 0.0, 0.0),
):
    if start is None:
        start = {"reference": "layer", "layer": "first_wall", "fraction": 0.0}
    if end is None:
        end = {"reference": "layer", "layer": "shield", "fraction": 1.0}
    if cross_section is None:
        cross_section = {
            "shape": "rectangle",
            "width": 40.0,
            "height": 25.0,
            "dimensions_are": "clear_aperture",
        }
    port = {
        "name": name,
        "placement": {
            "mode": "cartesian",
            "anchor": list(anchor),
            "axis": list(axis),
            "reference_direction": [0.0, 0.0, 1.0],
            "max_search_length": 500.0,
        },
        "cross_section": cross_section,
        "extent": {
            "start": start,
            "end": end,
            "outer_extension": outer_extension,
        },
        "liner": liner or {"enabled": False, "thickness": 0.0},
        "fill": {"mat_tag": "Vacuum"},
        "repetition": {"mode": "single"},
        "collision": collision
        or {"magnet_policy": "error", "minimum_magnet_clearance": 0.0},
    }
    if expected_layers is not None:
        port["expected_layers"] = expected_layers
    return port


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
    endpoint_reference_solids=None,
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
    obj.port_void_components = {}
    obj.port_liner_components = {}
    obj.port_outer_envelopes = {}
    obj.port_geometry_diagnostics = {}
    obj._port_fill_components = obj.port_void_components
    obj._port_fill_specs = obj.port_specs
    if endpoint_reference_solids:
        obj._endpoint_reference_solids.update(endpoint_reference_solids)

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

    nonorthogonal = _base_port("projected")
    nonorthogonal["placement"]["reference_direction"] = [1.0, 1.0, 0.0]
    nonorthogonal["placement"]["axis"] = [1.0, 0.0, 0.0]
    projected = parse_ports(
        [nonorthogonal], ("first_wall", "breeder", "shield")
    )[0]
    assert np.allclose(projected.placement.local_reference, [0.0, 1.0, 0.0])


def test_parse_port_rejects_nonpositive_search_length():
    port = _base_port("bad_length")
    port["placement"]["max_search_length"] = 0.0
    with pytest.raises(ValueError, match="max_search_length must be positive"):
        parse_ports([port], ("first_wall", "breeder", "shield"))


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


@pytest.mark.parametrize("reference", ["plasma_surface", "wall_surface"])
def test_parse_surface_endpoint_references(reference):
    port = _explicit_box_port(
        start={"reference": reference, "axial_offset": 1.5}
    )
    spec = parse_ports([port], ("first_wall", "breeder", "shield"))[0]
    assert spec.extent.start.reference == reference
    assert spec.extent.start.axial_offset == 1.5


@pytest.mark.parametrize("fraction", [0.0, 0.25, 1.0])
def test_parse_layer_endpoint_fractions(fraction):
    port = _explicit_box_port(
        start={
            "reference": "layer",
            "layer": "breeder",
            "fraction": fraction,
        }
    )
    spec = parse_ports([port], ("first_wall", "breeder", "shield"))[0]
    assert spec.extent.start.fraction == fraction


def test_parse_same_layer_extent_and_interior_end():
    port = _explicit_box_port(
        start={"reference": "layer", "layer": "breeder", "fraction": 0.1},
        end={"reference": "layer", "layer": "breeder", "fraction": 0.75},
    )
    spec = parse_ports([port], ("first_wall", "breeder", "shield"))[0]
    assert spec.extent.start.layer == spec.extent.end.layer == "breeder"
    assert spec.extent.end.fraction == 0.75


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_parse_rejects_invalid_endpoint_fraction(fraction):
    port = _explicit_box_port(
        start={
            "reference": "layer",
            "layer": "first_wall",
            "fraction": fraction,
        }
    )
    with pytest.raises(ValueError, match="fraction must be between"):
        parse_ports([port], ("first_wall", "breeder", "shield"))


def test_parse_rejects_unknown_endpoint_layer_and_negative_extension():
    unknown = _explicit_box_port(
        start={"reference": "layer", "layer": "unknown", "fraction": 0.0}
    )
    with pytest.raises(ValueError, match="not in radial build layers"):
        parse_ports([unknown], ("first_wall", "breeder", "shield"))

    negative = _explicit_box_port(outer_extension=-1.0)
    with pytest.raises(
        ValueError, match="outer_extension must be nonnegative"
    ):
        parse_ports([negative], ("first_wall", "breeder", "shield"))


def test_parse_liner_zero_positive_and_material_validation():
    zero = _explicit_box_port(
        liner={"enabled": True, "thickness": 0.0, "mat_tag": "steel"}
    )
    zero_spec = parse_ports([zero], ("first_wall", "breeder", "shield"))[0]
    assert zero_spec.liner.enabled is False

    lined = _explicit_box_port(
        liner={"enabled": True, "thickness": 2.0, "mat_tag": "SS316L"}
    )
    lined_spec = parse_ports([lined], ("first_wall", "breeder", "shield"))[0]
    assert lined_spec.liner.enabled
    assert lined_spec.liner.thickness == 2.0
    assert lined_spec.liner.mat_tag == "SS316L"

    invalid = _explicit_box_port(
        liner={"enabled": True, "thickness": 2.0, "mat_tag": ""}
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_ports([invalid], ("first_wall", "breeder", "shield"))


def test_legacy_layer_span_migrates_to_extent_with_warning():
    with pytest.warns(DeprecationWarning, match="layer_span is deprecated"):
        spec = parse_ports(
            [_base_port()], ("first_wall", "breeder", "shield")
        )[0]
    assert spec.extent.start.layer == "first_wall"
    assert spec.extent.start.fraction == 0.0
    assert spec.extent.end.layer == "first_wall"
    assert spec.extent.end.fraction == 1.0


def test_parse_rejects_extent_layer_span_conflict():
    port = _explicit_box_port()
    port["layer_span"] = {
        "start": "first_wall",
        "count": 3,
        "direction": "outward",
    }
    with pytest.raises(ValueError, match="cannot define both"):
        parse_ports([port], ("first_wall", "breeder", "shield"))


def test_generate_rectangular_port_single_layer():
    port = _base_box_port()
    ivb_with_port = _synthetic_ivb_from_ports([port])
    ivb_without_port = _synthetic_ivb_from_ports(None)

    remaining = ivb_with_port.Components["first_wall"].Volume()
    fill = ivb_with_port.Components[f"{port['name']}__void"].Volume()
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
    assert "circular__void" in ivb_obj.Components
    assert ivb_obj.Components["circular__void"].Volume() > 0


def test_generate_non_axis_aligned_port():
    port = _explicit_box_port(
        "tilted",
        anchor=(10.0, 0.0, 0.0),
        axis=(-1.0, 0.1, 0.05),
        start={
            "reference": "layer",
            "layer": "first_wall",
            "fraction": 0.1,
        },
        end={
            "reference": "layer",
            "layer": "first_wall",
            "fraction": 0.9,
        },
        expected_layers=["first_wall"],
        cross_section={"shape": "circle", "radius": 2.0},
    )
    ivb_obj = _synthetic_ivb_from_ports([port])
    assert "tilted__void" in ivb_obj.Components
    assert ivb_obj.Components["tilted__void"].Volume() > 0


def test_generate_port_with_axis_opposite_global_z():
    port = _base_box_port(
        "negative_z",
        anchor=(0.0, 0.0, 10.0),
        axis=(0.0, 0.0, -1.0),
        reference_direction=(1.0, 0.0, 0.0),
        max_search_length=100.0,
        layer_span={
            "start": "first_wall",
            "count": 3,
            "direction": "outward",
        },
    )
    components = {
        "first_wall": _layer_box_z(0.0, 20.0),
        "breeder": _layer_box_z(-20.0, 0.0),
        "shield": _layer_box_z(-40.0, -20.0),
    }
    ivb_obj = _synthetic_ivb_from_ports([port], layer_components=components)
    assert ivb_obj.Components["negative_z__void"].isValid()
    assert ivb_obj.Components["negative_z__void"].Volume() > 0.0


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
    port_fill = ivb_obj.Components["closure__void"].Volume()

    # expected closure only for the selected material span
    assert np.isclose(
        selected_original,
        selected_remaining + port_fill,
        rtol=1e-10,
        atol=1e-6,
    )

    for name in selected:
        assert ivb_obj.Components[name].isValid()
        overlap = (
            ivb_obj.Components[name]
            .intersect(ivb_obj.Components["closure__void"])
            .Volume()
        )
        assert overlap == pytest.approx(0.0, abs=1e-8)
    assert ivb_obj.Components["closure__void"].isValid()

    for inner, outer in zip(selected, selected[1:]):
        overlap = (
            ivb_obj.Components[inner]
            .intersect(ivb_obj.Components[outer])
            .Volume()
        )
        assert overlap == pytest.approx(0.0, abs=1e-8)


def test_ported_step_export(tmp_path):
    port = _base_box_port("step_port")
    ivb_obj = _synthetic_ivb_from_ports([port])
    ivb_obj.export_step(export_dir=tmp_path)

    assert (tmp_path / "first_wall.step").is_file()
    assert (tmp_path / "step_port__void.step").is_file()


def test_gmsh_consumes_boolean_modified_component():
    port = _base_box_port("gmsh_port")
    ivb_obj = _synthetic_ivb_from_ports([port])

    try:
        ivb_obj.mesh_components_gmsh(
            ["first_wall"], min_mesh_size=20.0, max_mesh_size=40.0
        )
        assert ivb.gmsh.model.getEntities(3)
    finally:
        if ivb.gmsh.isInitialized():
            ivb.gmsh.clear()
            ivb.gmsh.finalize()


def test_real_rib_based_ivb_fixture_rejects_unstable_boolean():
    toroidal_angles = [0.0, 5.0, 10.0, 15.0]
    poloidal_angles = [0.0, 120.0, 240.0, 360.0]
    thickness = np.ones((len(toroidal_angles), len(poloidal_angles)))
    radial_build = ivb.RadialBuild(
        toroidal_angles,
        poloidal_angles,
        1.08,
        {
            "component_1": {"thickness_matrix": thickness * 10.0},
            "component_2": {"thickness_matrix": thickness * 10.0},
            "component_3": {"thickness_matrix": thickness * 10.0},
        },
    )
    ribs_path = (
        Path(__file__).parent
        / "files_for_tests"
        / "kisslinger_file_example.txt"
    )
    (
        custom_toroidal_angles,
        _,
        num_poloidal_angles,
        _,
        custom_surface_rz_ribs,
    ) = ribs_from_kisslinger_format(
        ribs_path, delimiter=" ", scale=1.0, format=False
    )
    reference_surface = ivb.RibBasedSurface(
        custom_surface_rz_ribs,
        custom_toroidal_angles,
        np.linspace(0.0, 360.0, num_poloidal_angles),
    )
    model = ivb.InVesselBuild(
        reference_surface,
        radial_build,
        num_ribs=11,
        num_rib_pts=61,
    )
    model.populate_surfaces()
    model.calculate_loci()
    model.generate_components_cadquery()

    surface_loci = model.Surfaces["component_2"].get_loci()
    toroidal_index = len(surface_loci) // 2
    poloidal_index = len(surface_loci[toroidal_index]) // 4
    anchor = np.asarray(surface_loci[toroidal_index][poloidal_index])
    toroidal_tangent = np.asarray(
        surface_loci[toroidal_index + 1][poloidal_index]
    ) - np.asarray(surface_loci[toroidal_index - 1][poloidal_index])
    poloidal_tangent = np.asarray(
        surface_loci[toroidal_index][poloidal_index + 1]
    ) - np.asarray(surface_loci[toroidal_index][poloidal_index - 1])
    axis = np.cross(toroidal_tangent, poloidal_tangent)
    reference_direction = np.array([anchor[0], anchor[1], 0.0])
    port = _base_box_port(
        "rib_fixture_port",
        anchor=anchor,
        axis=axis,
        reference_direction=reference_direction,
        max_search_length=100.0,
        cross_section={"shape": "circle", "radius": 5.0},
        layer_span={
            "start": "component_2",
            "count": 1,
            "direction": "outward",
        },
    )
    model.ports = parse_ports([port], radial_build.user_layer_names)
    with pytest.raises(
        (ValueError, NotImplementedError),
        match="valid|multiple|finite geometry",
    ):
        model._apply_ports_to_components()


def test_explicit_port_fill_metadata_preserved():
    port = _base_box_port("metadata")
    ivb_obj = _synthetic_ivb_from_ports([port])
    solids, mat_tags = ivb_obj.extract_solids_and_mat_tags()

    idx = list(ivb_obj.Components.keys()).index("metadata__void")
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
    idx = list(custom_ivb.Components.keys()).index("custom__void")
    assert mat_tags[idx] == "void_tag"

    lined = _explicit_box_port(
        "tagged_liner",
        liner={"enabled": True, "thickness": 2.0, "mat_tag": "steel_tag"},
    )
    lined_ivb = _synthetic_ivb_from_ports([lined])
    _, mat_tags = lined_ivb.extract_solids_and_mat_tags()
    liner_idx = list(lined_ivb.Components).index("tagged_liner__liner")
    assert mat_tags[liner_idx] == "steel_tag"


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
    with pytest.raises(NotImplementedError, match="disconnected.*multiple"):
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

    assert "p1__void" in ivb_obj.Components
    assert "p2__void" in ivb_obj.Components
    assert ivb_obj.Components["p1__void"].intersect(
        ivb_obj.Components["p2__void"]
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

    with pytest.raises(ValueError, match="outer envelope overlaps port"):
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
    with pytest.raises(
        NotImplementedError, match="MOAB mesh workflow is not supported"
    ):
        ivb_obj.mesh_components_moab(["moab__void"])


def test_per_period_not_implemented():
    port = _base_box_port(
        "per_period", anchor=(10.0, 0.0, 0.0), axis=(-1.0, 0.0, 0.0)
    )
    port["repetition"] = {"mode": "per_period"}
    with pytest.raises(NotImplementedError, match="per_period"):
        _synthetic_ivb_from_ports([port])


@pytest.mark.parametrize(
    "cross_section",
    [
        {
            "shape": "circle",
            "radius": 8.0,
            "dimensions_are": "clear_aperture",
        },
        {
            "shape": "rectangle",
            "width": 30.0,
            "height": 18.0,
            "dimensions_are": "clear_aperture",
        },
    ],
)
def test_lined_port_has_connected_hollow_liner(cross_section):
    port = _explicit_box_port(
        "lined",
        cross_section=cross_section,
        liner={"enabled": True, "thickness": 2.0, "mat_tag": "SS316L"},
    )
    model = _synthetic_ivb_from_ports([port])
    void = model.Components["lined__void"]
    liner = model.Components["lined__liner"]
    outer = model.port_outer_envelopes["lined"]
    assert void.isValid() and liner.isValid() and outer.isValid()
    assert len(liner.Solids()) == 1
    assert outer.Volume() == pytest.approx(
        void.Volume() + liner.Volume(), rel=1e-10, abs=1e-6
    )
    assert void.intersect(liner).Volume() == pytest.approx(0.0, abs=1e-8)


def test_liner_dimensions_expand_outward_from_clear_aperture():
    port = _explicit_box_port(
        "dimensions",
        start={"reference": "layer", "layer": "breeder", "fraction": 0.1},
        end={"reference": "layer", "layer": "breeder", "fraction": 0.9},
        cross_section={
            "shape": "rectangle",
            "width": 30.0,
            "height": 18.0,
            "dimensions_are": "clear_aperture",
        },
        liner={"enabled": True, "thickness": 3.0, "mat_tag": "steel"},
    )
    model = _synthetic_ivb_from_ports([port])
    void_box = model.port_void_components["dimensions"].BoundingBox()
    outer_box = model.port_outer_envelopes["dimensions"].BoundingBox()
    assert sorted([void_box.ylen, void_box.zlen]) == pytest.approx(
        [18.0, 30.0]
    )
    assert sorted([outer_box.ylen, outer_box.zlen]) == pytest.approx(
        [24.0, 36.0]
    )


def test_same_layer_blind_penetration_and_reversed_axis_diagnostic():
    port = _explicit_box_port(
        "blind",
        start={"reference": "layer", "layer": "breeder", "fraction": 0.1},
        end={"reference": "layer", "layer": "breeder", "fraction": 0.75},
        expected_layers=["breeder"],
    )
    model = _synthetic_ivb_from_ports([port])
    result = model.port_geometry_diagnostics["blind"]
    assert result.resolved_end - result.resolved_start == pytest.approx(13.0)
    assert result.ordered_intersected_layers == ("breeder",)

    reversed_port = _explicit_box_port(
        "reversed",
        start={"reference": "layer", "layer": "breeder", "fraction": 0.75},
        end={"reference": "layer", "layer": "breeder", "fraction": 0.1},
    )
    with pytest.raises(ValueError, match="axis may need to be reversed"):
        _synthetic_ivb_from_ports([reversed_port])


def test_partial_layer_to_layer_extent_and_expected_layer_assertion():
    port = _explicit_box_port(
        "partial",
        start={"reference": "layer", "layer": "breeder", "fraction": 0.2},
        end={"reference": "layer", "layer": "shield", "fraction": 0.65},
        expected_layers=["breeder", "shield"],
    )
    model = _synthetic_ivb_from_ports([port])
    assert model.port_geometry_diagnostics[
        "partial"
    ].ordered_intersected_layers == ("breeder", "shield")

    mismatch = _explicit_box_port(
        "mismatch",
        expected_layers=["first_wall", "breeder"],
    )
    with pytest.raises(ValueError, match="expected layers"):
        _synthetic_ivb_from_ports([mismatch])


def test_plasma_flush_lined_port_with_external_protrusion_and_accounting():
    plasma_volume = _layer_box(20.0, 200.0)
    port = _explicit_box_port(
        "plasma_to_exterior",
        start={"reference": "plasma_surface", "axial_offset": 0.0},
        end={"reference": "layer", "layer": "shield", "fraction": 1.0},
        outer_extension=30.0,
        liner={"enabled": True, "thickness": 2.0, "mat_tag": "SS316L"},
        expected_layers=["first_wall", "breeder", "shield"],
    )
    model = _synthetic_ivb_from_ports(
        [port], endpoint_reference_solids={"plasma_surface": plasma_volume}
    )
    result = model.port_geometry_diagnostics["plasma_to_exterior"]
    liner = model.port_liner_components["plasma_to_exterior"]
    assert result.resolved_start == pytest.approx(-10.0)
    assert result.resolved_end == pytest.approx(50.0)
    assert result.outer_extension == 30.0
    assert result.void_volume_inside_blanket > 0.0
    assert result.liner_volume_inside_blanket > 0.0
    assert result.void_volume_outside_blanket > 0.0
    assert result.liner_volume_outside_blanket > 0.0
    assert result.closure_error < model._bool_tolerance(
        result.original_blanket_volume
    )
    assert result.maximum_liner_overlap_with_plasma == pytest.approx(
        0.0, abs=1e-8
    )
    assert liner.intersect(plasma_volume).Volume() == pytest.approx(
        0.0, abs=1e-8
    )


def test_wall_surface_start_is_conformally_trimmed():
    wall_volume = _layer_box(20.0, 200.0)
    port = _explicit_box_port(
        "wall_start",
        start={"reference": "wall_surface", "axial_offset": 0.0},
        end={"reference": "layer", "layer": "first_wall", "fraction": 1.0},
        liner={"enabled": True, "thickness": 1.0, "mat_tag": "steel"},
        expected_layers=["first_wall"],
    )
    model = _synthetic_ivb_from_ports(
        [port], endpoint_reference_solids={"wall_surface": wall_volume}
    )
    assert model.port_geometry_diagnostics[
        "wall_start"
    ].resolved_start == pytest.approx(-10.0)
    assert model.port_outer_envelopes["wall_start"].intersect(
        wall_volume
    ).Volume() == pytest.approx(0.0, abs=1e-8)


def test_lined_step_round_trip_and_gmsh_smoke(tmp_path):
    port = _explicit_box_port(
        "round_trip",
        liner={"enabled": True, "thickness": 2.0, "mat_tag": "SS316L"},
        outer_extension=10.0,
    )
    model = _synthetic_ivb_from_ports([port])
    model.export_step(export_dir=tmp_path)
    for suffix, expected in (
        ("void", model.port_void_components["round_trip"]),
        ("liner", model.port_liner_components["round_trip"]),
    ):
        path = tmp_path / f"round_trip__{suffix}.step"
        assert path.is_file()
        imported = cq.importers.importStep(str(path)).val()
        assert imported.isValid()
        assert len(imported.Solids()) == 1
        assert imported.Volume() == pytest.approx(
            expected.Volume(), rel=1e-9, abs=1e-6
        )

    try:
        model.mesh_components_gmsh(
            ["round_trip__void", "round_trip__liner"],
            min_mesh_size=10.0,
            max_mesh_size=25.0,
        )
        assert len(ivb.gmsh.model.getEntities(3)) == 2
    finally:
        if ivb.gmsh.isInitialized():
            ivb.gmsh.clear()
            ivb.gmsh.finalize()


class _SyntheticMagnetSet:
    def __init__(self, records):
        self.records = records

    def iter_coil_solids(self):
        return iter(self.records)


def _stellarator_with_port_and_magnets(port, magnet_records):
    model = _synthetic_ivb_from_ports([port])
    stellarator = ps.Stellarator.__new__(ps.Stellarator)
    stellarator.invessel_build = model
    stellarator.magnet_set = _SyntheticMagnetSet(magnet_records)
    stellarator.port_magnet_collision_report = ()
    stellarator._logger = model._logger
    return stellarator


def _magnet_record(coil_id, region_kind, x, y, z=0.0):
    solid = cq.Workplane("XY").box(10.0, 10.0, 10.0).translate((x, y, z)).val()
    return MagnetSolidRecord(coil_id, region_kind, "magnet", solid)


def test_magnet_collision_clearance_and_multiple_coil_records():
    port = _explicit_box_port(
        "magnet_check",
        start={"reference": "layer", "layer": "first_wall", "fraction": 0.0},
        end={"reference": "layer", "layer": "first_wall", "fraction": 1.0},
        collision={
            "magnet_policy": "report",
            "clearance_policy": "report",
            "minimum_magnet_clearance": 10.0,
        },
    )
    records = [
        _magnet_record(0, "inner_conductor", 10.0, 0.0),
        _magnet_record(1, "outer_casing", 10.0, 20.0),
        _magnet_record(2, "inner_conductor", 10.0, 60.0),
    ]
    stellarator = _stellarator_with_port_and_magnets(port, records)
    report = stellarator.check_port_magnet_clearance()
    assert [record.status for record in report] == [
        "collision",
        "clearance_violation",
        "clear",
    ]
    assert report[0].actual_overlap_volume > 0.0
    assert report[1].actual_overlap_volume == pytest.approx(0.0, abs=1e-8)
    assert report[1].clearance_envelope_overlap_volume > 0.0
    assert report[2].estimated_minimum_distance > 10.0


@pytest.mark.parametrize("policy", ["error", "warn", "report", "ignore"])
def test_all_hard_collision_policies(policy):
    port = _explicit_box_port(
        f"policy_{policy}",
        start={"reference": "layer", "layer": "first_wall", "fraction": 0.0},
        end={"reference": "layer", "layer": "first_wall", "fraction": 1.0},
        collision={
            "magnet_policy": policy,
            "minimum_magnet_clearance": 0.0,
        },
    )
    stellarator = _stellarator_with_port_and_magnets(
        port, [_magnet_record(0, "inner_conductor", 10.0, 0.0)]
    )
    if policy == "error":
        with pytest.raises(ValueError, match="collision with coil"):
            stellarator.check_port_magnet_clearance()
    elif policy == "warn":
        with pytest.warns(RuntimeWarning, match="collision with coil"):
            report = stellarator.check_port_magnet_clearance()
        assert report[0].status == "collision"
    else:
        assert (
            stellarator.check_port_magnet_clearance()[0].status == "collision"
        )


def test_clearance_exactly_at_tolerance_is_clear_and_construction_order_hook():
    port = _explicit_box_port(
        "touching",
        start={"reference": "layer", "layer": "first_wall", "fraction": 0.0},
        end={"reference": "layer", "layer": "first_wall", "fraction": 1.0},
        collision={
            "magnet_policy": "report",
            "clearance_policy": "report",
            "minimum_magnet_clearance": 10.0,
        },
    )
    # Clear-aperture half-height is 12.5; a 10-wide magnet centered at 27.5
    # touches the 10-unit clearance envelope without positive overlap.
    stellarator = _stellarator_with_port_and_magnets(
        port, [_magnet_record(0, "inner_conductor", 10.0, 27.5)]
    )
    assert (
        stellarator._validate_port_magnet_clearance_if_available()[0].status
        == "clear"
    )


def test_clearance_validation_is_order_independent():
    port = _explicit_box_port(
        "order",
        start={"reference": "layer", "layer": "first_wall", "fraction": 0.0},
        end={"reference": "layer", "layer": "first_wall", "fraction": 1.0},
        collision={"magnet_policy": "report", "minimum_magnet_clearance": 0.0},
    )
    model = _synthetic_ivb_from_ports([port])
    magnets = _SyntheticMagnetSet(
        [_magnet_record(0, "inner_conductor", 10.0, 60.0)]
    )

    magnets_first = ps.Stellarator.__new__(ps.Stellarator)
    magnets_first.invessel_build = None
    magnets_first.magnet_set = magnets
    magnets_first._logger = model._logger
    assert magnets_first._validate_port_magnet_clearance_if_available() == []
    magnets_first.invessel_build = model
    assert magnets_first._validate_port_magnet_clearance_if_available()

    ports_first = ps.Stellarator.__new__(ps.Stellarator)
    ports_first.invessel_build = model
    ports_first.magnet_set = None
    ports_first._logger = model._logger
    assert ports_first._validate_port_magnet_clearance_if_available() == []
    ports_first.magnet_set = magnets
    assert ports_first._validate_port_magnet_clearance_if_available()


@pytest.mark.parametrize("split_chamber", [False, True])
def test_real_sector_plasma_to_exterior_with_filament_magnets(
    tmp_path, split_chamber
):
    toroidal_angles = [0.0, 10.0, 20.0, 30.0]
    poloidal_angles = [0.0, 90.0, 180.0, 270.0, 360.0]
    thickness = np.ones((len(toroidal_angles), len(poloidal_angles))) * 5.0
    radial_build = ivb.RadialBuild(
        toroidal_angles,
        poloidal_angles,
        1.08,
        {
            "first_wall": {"thickness_matrix": thickness},
            "breeder": {"thickness_matrix": thickness},
            "shield": {"thickness_matrix": thickness},
            "vacuum_vessel": {"thickness_matrix": thickness},
        },
        split_chamber=split_chamber,
    )
    phi = np.deg2rad(15.0)
    outward = np.array([np.cos(phi), np.sin(phi), 0.0])
    anchor = outward * 760.0
    port = _explicit_box_port(
        "real_heating_port",
        anchor=anchor,
        axis=outward,
        start={"reference": "plasma_surface", "axial_offset": 0.0},
        end={
            "reference": "layer",
            "layer": "vacuum_vessel",
            "fraction": 1.0,
        },
        outer_extension=25.0,
        cross_section={
            "shape": "circle",
            "radius": 3.0,
            "dimensions_are": "clear_aperture",
        },
        liner={"enabled": True, "thickness": 1.0, "mat_tag": "SS316L"},
        expected_layers=["first_wall", "breeder", "shield", "vacuum_vessel"],
        collision={
            "magnet_policy": "report",
            "clearance_policy": "report",
            "minimum_magnet_clearance": 5.0,
        },
    )
    model = ivb.InVesselBuild(
        _SyntheticReferenceSurface(),
        radial_build,
        num_ribs=13,
        num_rib_pts=49,
        ports=[port],
    )
    model.populate_surfaces()
    model.calculate_loci()
    model.generate_components_cadquery()
    result = model.port_geometry_diagnostics["real_heating_port"]
    assert model.port_void_components["real_heating_port"].isValid()
    assert model.port_liner_components["real_heating_port"].isValid()
    assert result.void_volume_inside_blanket > 0.0
    assert result.liner_volume_inside_blanket > 0.0
    assert result.maximum_liner_overlap_with_plasma == pytest.approx(
        0.0, abs=1e-7
    )

    model.export_step(export_dir=tmp_path)
    for kind in ("void", "liner"):
        imported = cq.importers.importStep(
            str(tmp_path / f"real_heating_port__{kind}.step")
        ).val()
        assert imported.isValid()
        assert imported.Volume() > 0.0

    coils_path = Path(__file__).parent / "files_for_tests" / "coils.example"
    magnets = magnet_coils.MagnetSetFromFilaments(
        coils_path,
        width=40.0,
        thickness=50.0,
        toroidal_extent=30.0,
        sample_mod=10,
    )
    magnets.populate_magnet_coils()
    magnets.build_magnet_coils()
    stellarator = ps.Stellarator.__new__(ps.Stellarator)
    stellarator.invessel_build = model
    stellarator.magnet_set = magnets
    stellarator.port_magnet_collision_report = ()
    stellarator._logger = model._logger
    collision_report = stellarator.check_port_magnet_clearance()
    assert collision_report
    assert all(
        record.status in {"collision", "clearance_violation", "clear"}
        for record in collision_report
    )
