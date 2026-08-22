"""Native point-cloud port DAGMC and discrete-PLC regression tests."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pydagmc
import pytest
from pymoab import core, types

import parastell.parastell as ps
from parastell.dagmc_assembly import (
    audit_dagmc_model,
    close_with_graveyard,
    ensure_geometry_names,
)
from parastell.native_port_geometry import _box_triangles
from parastell.utils import combine_dagmc_models

sys.path.insert(0, str(Path(__file__).parent))
from test_ports import _surface_ivb, _surface_port  # noqa: E402


def _native_model():
    model = _surface_ivb(num_ribs=9, num_rib_pts=33)
    model.use_pydagmc = True
    model.generate_components()
    return model


def test_native_surface_complex_is_closed_and_unique():
    complex_ = _native_model().native_port_complex
    result = complex_.validate()

    assert complex_.topology_summary()["sha256"] == (
        "5bbb1d6f985e80951ed3179ff06ae883847453533e1309e6d41b74a379d0fc2b"
    )

    assert result.duplicate_facet_count == 0
    assert result.zero_area_triangle_count == 0
    assert result.unsealed_edge_count == 0
    assert result.centerline_region_sequence == (
        "plasma",
        "surface_port__void",
        "graveyard",
        "external",
    )
    assert result.liner_region_sequence == (
        "plasma",
        "surface_port__liner",
        "graveyard",
        "external",
    )
    assert result.blanket_region_sequence == (
        "plasma",
        "chamber",
        "first_wall",
        "breeder",
        "shield",
        "graveyard",
        "external",
    )
    assert all(
        surface.reverse_volume is not None for surface in complex_.surfaces
    )
    assert len(complex_.loops) == 6


def test_native_h5m_independent_round_trip(tmp_path):
    complex_ = _native_model().native_port_complex
    path = complex_.write_dagmc(tmp_path / "native.h5m")
    result = complex_.validate_dagmc_file(path)

    assert result["pymoab_load"]
    assert result["pydagmc_load"]
    assert result["duplicate_facet_count"] == 0
    assert result["orphan_triangle_count"] == 0
    assert result["unreferenced_vertex_count"] == 0

    openmc = pytest.importorskip("openmc")
    universe = openmc.DAGMCUniverse(str(path))
    assert set(universe.material_names) == {
        "Vacuum",
        "first_wall",
        "breeder",
        "shield",
        "steel",
        "Graveyard",
    }


def test_native_volume_mesh_is_conformal_and_tagged(tmp_path):
    complex_ = _native_model().native_port_complex
    mesh = complex_.tetrahedralize(20.0, 60.0)
    result = mesh.validate()

    assert result.tetrahedron_count > 0
    assert result.inverted_tetrahedron_count == 0
    assert result.zero_volume_tetrahedron_count == 0
    assert result.duplicate_tetrahedron_count == 0
    assert result.nonconformal_interface_face_count == 0
    assert result.disconnected_region_count == 0
    assert max(result.region_relative_volume_error.values()) < 1e-10
    assert result.region_tetrahedron_counts["surface_port__void"] > 0
    assert result.region_tetrahedron_counts["surface_port__liner"] > 0
    assert min(result.region_minimum_scaled_jacobian.values()) > 0.0
    assert min(result.region_minimum_mean_ratio.values()) > 0.0
    assert min(result.region_minimum_radius_ratio.values()) > 0.0
    assert min(result.region_minimum_dihedral_angle.values()) > 0.0
    assert max(result.region_maximum_dihedral_angle.values()) < 180.0
    assert all(
        sum(counts.values()) == 0
        for counts in result.region_quality_threshold_counts.values()
    )

    path = mesh.write(tmp_path / "native_volume_mesh.h5m")
    mb = core.Core()
    mb.load_file(str(path))
    assert len(mb.get_entities_by_type(mb.get_root_set(), types.MBTET)) == (
        result.tetrahedron_count
    )
    mb.tag_get_handle("MATERIAL")
    file_result = mesh.validate_file(path)
    assert file_result["region_count"] == sum(
        volume.kind != "graveyard" for volume in complex_.volumes
    )
    assert file_result["tetrahedron_count"] == result.tetrahedron_count


def test_native_path_does_not_generate_cadquery(monkeypatch):
    model = _surface_ivb(num_ribs=9, num_rib_pts=33)

    def reject_cad(*args, **kwargs):
        raise AssertionError("native port path called CadQuery")

    monkeypatch.setattr(model, "generate_components_cadquery", reject_cad)
    model.use_pydagmc = True
    model.generate_components()

    assert hasattr(model, "native_port_complex")
    assert not model.Components


def test_centerline_crosses_one_termination_facet_not_fan_vertex():
    complex_ = _native_model().native_port_complex
    anchor = np.asarray(complex_.port.placement.anchor)
    axis = np.asarray(complex_.port.placement.local_axis)
    origin = anchor - 2.0 * axis
    for surface in complex_.surfaces:
        if surface.kind not in {"plasma_connection", "external_termination"}:
            continue
        if not surface.name.endswith(":void"):
            continue
        hits = sum(
            complex_._ray_triangle(origin, axis, triangle) is not None
            for triangle in surface.triangles
        )
        assert hits == 1


def test_rectangular_rolled_tilted_native_port_is_watertight_and_conformal():
    port = _surface_port(
        shape="rectangle", poloidal_tilt=7.0, toroidal_tilt=-4.0, roll=23.0
    )
    model = _surface_ivb(num_ribs=9, num_rib_pts=33, port=port)
    model.use_pydagmc = True
    model.generate_components()
    complex_ = model.native_port_complex
    validation = complex_.validate()

    assert validation.duplicate_facet_count == 0
    assert validation.unsealed_edge_count == 0
    assert all(len(loop.inner_points) - 1 == 4 for loop in complex_.loops)
    placement = complex_.port.placement
    basis = np.asarray(
        (
            placement.local_reference,
            placement.local_normal,
            placement.local_axis,
        )
    )
    assert np.linalg.det(basis) == pytest.approx(1.0, abs=1e-12)
    inner = complex_.loops[len(complex_.loops) // 2].inner_points[:-1]
    relative = inner - np.asarray(placement.anchor)
    uv = np.column_stack(
        (
            relative @ np.asarray(placement.local_reference),
            relative @ np.asarray(placement.local_normal),
        )
    )
    assert np.ptp(uv[:, 0]) == pytest.approx(6.0, abs=1e-6)
    assert np.ptp(uv[:, 1]) == pytest.approx(4.0, abs=1e-6)

    mesh = complex_.tetrahedralize(20.0, 60.0)
    mesh_result = mesh.validate()
    assert mesh_result.nonconformal_interface_face_count == 0
    assert mesh_result.disconnected_region_count == 0


def _one_sided_box_model(lower=(900.0, 0.0, 0.0), upper=(920.0, 20.0, 20.0)):
    model = pydagmc.Model(core.Core())
    volume = model.create_volume(global_id=1)
    material = pydagmc.Group.create(model, name="mat:conductor")
    material.add_set(volume)
    triangles = _box_triangles(lower, upper)
    coordinate_map = {}
    coordinates = []
    connectivity = []
    for triangle in triangles:
        indices = []
        for point in triangle:
            key = tuple(point)
            if key not in coordinate_map:
                coordinate_map[key] = len(coordinates)
                coordinates.append(point)
            indices.append(coordinate_map[key])
        connectivity.append(indices)
    vertices = np.asarray(
        model.mb.create_vertices(np.asarray(coordinates).ravel()),
        dtype=np.uint64,
    )
    facets = model.mb.create_elements(
        types.MBTRI, vertices[np.asarray(connectivity)]
    )
    surface = model.create_surface(global_id=1)
    model.mb.add_entities(surface.handle, facets)
    surface.senses = [volume, None]
    return model


def test_post_assembly_graveyard_closes_all_physical_submodels():
    ivb = _surface_ivb(num_ribs=9, num_rib_pts=33)
    ivb.use_pydagmc = True
    ivb.generate_components_pydagmc(include_graveyard=False)
    assert all(
        volume.kind != "graveyard"
        for volume in ivb.native_port_complex.volumes
    )

    combined = combine_dagmc_models(
        [ivb.dag_model.mb, _one_sided_box_model().mb]
    )
    result = close_with_graveyard(combined, margin=25.0)
    ensure_geometry_names(combined)
    audit = audit_dagmc_model(combined)

    assert (
        len([name for name in combined.group_names if name == "mat:Graveyard"])
        == 1
    )
    graveyard = combined.volumes_by_id[result["graveyard_volume_id"]]
    one_sided = []
    for surface in combined.surfaces:
        senses = surface.senses
        if None in senses:
            one_sided.append(surface)
        else:
            assert graveyard in senses or all(
                sense is not None for sense in senses
            )
    assert [surface.id for surface in one_sided] == [
        result["outer_surface_id"]
    ]
    assert audit["duplicate_facet_count"] == 0
    assert audit["unsealed_edge_count"] == 0

    facet_counts = Counter()
    edge_counts = Counter()
    for surface in graveyard.surfaces:
        for triangle in np.asarray(surface.triangle_coords).reshape(
            (-1, 3, 3)
        ):
            keys = [tuple(np.round(point, 9)) for point in triangle]
            facet_counts[tuple(sorted(keys))] += 1
            for left, right in ((0, 1), (1, 2), (2, 0)):
                edge_counts[tuple(sorted((keys[left], keys[right])))] += 1
    assert max(facet_counts.values()) == 1
    assert set(edge_counts.values()) == {2}


def test_post_assembly_rejects_existing_graveyard():
    complex_ = _native_model().native_port_complex
    with pytest.raises(ValueError, match="already contains"):
        close_with_graveyard(complex_.dag_model)


def test_public_moab_api_selects_native_surface_complex(tmp_path):
    model = _surface_ivb(num_ribs=9, num_rib_pts=33)
    stellarator = ps.Stellarator.__new__(ps.Stellarator)
    stellarator.invessel_build = model

    path = stellarator.export_invessel_build_mesh_moab(
        ["first_wall"],
        "public_native_volume_mesh",
        export_dir=tmp_path,
        geometry_source="auto",
        min_mesh_size=20.0,
        max_mesh_size=60.0,
    )

    assert path.exists()
    assert path.with_suffix(".vtk").exists()
    assert (
        stellarator.invessel_build.native_volume_mesh_validation.tetrahedron_count
        > 0
    )
    with pytest.raises(ValueError, match="cannot represent a port"):
        stellarator.export_invessel_build_mesh_moab(
            ["first_wall"],
            "legacy",
            export_dir=tmp_path,
            geometry_source="legacy_point_cloud",
        )
