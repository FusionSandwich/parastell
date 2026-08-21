"""Native point-cloud port DAGMC and discrete-PLC regression tests."""

import sys
from pathlib import Path

import pytest
from pymoab import core, types

sys.path.insert(0, str(Path(__file__).parent))
from test_ports import _surface_ivb  # noqa: E402


def _native_model():
    model = _surface_ivb(num_ribs=9, num_rib_pts=33)
    model.use_pydagmc = True
    model.generate_components()
    return model


def test_native_surface_complex_is_closed_and_unique():
    complex_ = _native_model().native_port_complex
    result = complex_.validate()

    assert result.duplicate_facet_count == 0
    assert result.zero_area_triangle_count == 0
    assert result.unsealed_edge_count == 0
    assert result.centerline_region_sequence == (
        "plasma",
        "surface_port__void",
        "external",
    )
    assert result.liner_region_sequence == (
        "plasma",
        "surface_port__liner",
        "external",
    )
    assert result.blanket_region_sequence == (
        "plasma",
        "chamber",
        "first_wall",
        "breeder",
        "shield",
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

    path = mesh.write(tmp_path / "native_volume_mesh.h5m")
    mb = core.Core()
    mb.load_file(str(path))
    assert len(mb.get_entities_by_type(mb.get_root_set(), types.MBTET)) == (
        result.tetrahedron_count
    )
    mb.tag_get_handle("MATERIAL")
    file_result = mesh.validate_file(path)
    assert file_result["region_count"] == len(complex_.volumes)
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
