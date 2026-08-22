"""Generic ParaView multiblock export tests."""

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from parastell.paraview_export import (
    ParaViewBlock,
    audit_public_names,
    discover_paraview_executables,
    export_paraview_bundle,
    write_polydata,
    write_unstructured_grid,
    write_vtm,
)


def _block(name="port__void", component_class="port_void"):
    return ParaViewBlock(
        component_name=name,
        component_kind=component_class,
        component_class=component_class,
        material_tag="Vacuum",
        dagmc_volume_id=7,
        field_period_id=0,
        instance_id="period-0-volume-7",
        port_name="port",
        hidden_by_default=component_class == "graveyard",
    )


def test_paraview_executable_discovery(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("paraview", "pvpython", "pvbatch"):
        executable = bin_dir / name
        executable.write_text("#!/bin/sh\necho 'paraview version 9.8.7'\n")
        executable.chmod(0o755)

    result = discover_paraview_executables([tmp_path])

    assert Path(result.pvbatch).name == "pvbatch"
    assert result.version == "9.8.7"


def test_named_polydata_and_vtm_retain_metadata(tmp_path):
    triangles = np.asarray([((0, 0, 0), (1, 0, 0), (0, 1, 0))], dtype=float)
    block = write_polydata(tmp_path / "void.vtp", triangles, _block())
    vtm = write_vtm(tmp_path / "geometry.vtm", [block], "dagmc")

    leaf = ET.parse(block.file)
    fields = {
        node.attrib["Name"]: (node.text or "")
        for node in leaf.findall(".//FieldData/DataArray")
    }
    dataset = ET.parse(vtm).find(".//DataSet")
    point_arrays = leaf.findall(".//PointData/DataArray")

    assert "port__void" in fields["component_name"]
    assert "Vacuum" in fields["material_tag"]
    assert dataset.attrib["name"] == "port__void"
    assert point_arrays == []
    assert block.cell_count == 1
    assert block.bounds == (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)


def test_named_tetrahedra_retain_region_and_material(tmp_path):
    points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float
    )
    block = replace(
        _block("port__liner", "port_liner"),
        component_kind="port_liner",
        material_tag="SS316L",
        volumetric_geometry=True,
        transport_geometry=False,
    )
    written = write_unstructured_grid(
        tmp_path / "liner.vtu", points, [[0, 1, 2, 3]], block
    )

    fields = {
        node.attrib["Name"]: (node.text or "")
        for node in ET.parse(written.file).findall(".//FieldData/DataArray")
    }
    assert "port__liner" in fields["component_name"]
    assert "SS316L" in fields["material_tag"]
    assert written.cell_count == 1


def test_graveyard_is_hidden_by_default():
    graveyard = replace(
        _block("graveyard", "graveyard"),
        component_kind="graveyard",
        hidden_by_default=True,
    )
    assert graveyard.hidden_by_default


def test_clean_public_naming_audit():
    assert (
        audit_public_names(
            [
                "ports/full-reactor-validation",
                "Add named DAGMC multiblock export",
            ]
        )
        == ()
    )
    assert audit_public_names(["agent/old-name", "ordinary-name"]) == (
        "agent/old-name",
    )


def test_dagmc_bundle_names_blocks_and_materials(tmp_path):
    h5m = Path("files_for_tests") / "one_cube.h5m"
    manifest = export_paraview_bundle(
        tmp_path,
        dagmc_h5m=h5m,
        repository_sha="0123456789abcdef",
        scope="sector_transport_model",
    )

    assert (tmp_path / "full_reactor_dagmc.vtm").is_file()
    assert (tmp_path / "full_reactor_volume_mesh.vtm").is_file()
    assert (tmp_path / "full_reactor_render.py").is_file()
    assert manifest["dagmc_blocks"]
    assert all(block["component_name"] for block in manifest["dagmc_blocks"])
    assert all("material_tag" in block for block in manifest["dagmc_blocks"])
    assert manifest["graveyard_hidden_by_default"]
