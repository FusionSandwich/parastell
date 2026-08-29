from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from parastell import parametric_openmc16_model as builder


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _packet(tmp_path):
    dagmc = tmp_path / "dagmc.h5m"
    dagmc.write_bytes(b"dagmc")
    source = tmp_path / "source.h5m"
    source.write_bytes(b"source")
    materials = tmp_path / "materials.xml"
    materials.write_text("<materials/>", encoding="utf-8")
    cross_sections = tmp_path / "cross_sections.xml"
    cross_sections.write_text("<cross_sections/>", encoding="utf-8")
    geometry_gate = _write(
        tmp_path / "geometry.json",
        {
            "schema": "parastell.dagmc_native_qualification/v1.0.0",
            "native_dagmc_gate_pass": True,
            "raw_h5m_sha256_before": _sha(dagmc),
            "raw_h5m_sha256_after": _sha(dagmc),
            "h5m_unchanged": True,
            "check_watertight": {
                "pass": True,
                "unmatched_edge_count": 0,
                "unsealed_surface_count": 0,
                "unsealed_volume_count": 0,
            },
            "overlap_checks": [
                {
                    "points_per_edge": value,
                    "pass": True,
                    "terminal_overlap_location_count": 0,
                    "parsed_overlap_location_count": 0,
                }
                for value in (1, 2, 4)
            ],
        },
    )
    premesh = _write(
        tmp_path / "premesh.json",
        {
            "schema": "parastell.parametric_direct90_premesh/v1.0.0",
            "status": "PREMESH_TOPOLOGY_PASS_MESH_PENDING",
            "semantic_roles": list(builder.SEMANTIC_ROLES),
            "topology": {"status": "PREMESH_26_VOLUME_TOPOLOGY_PASS"},
        },
    )
    export = _write(
        tmp_path / "export.json",
        {
            "schema": "parastell.parametric_direct90_dagmc_export/v1.0.0",
            "status": "DAGMC_EXPORTED_NATIVE_GATES_PENDING",
            "h5m": {"sha256": _sha(dagmc)},
            "geometry": {
                "extent_degrees": 90.0,
                "combined_from_45_degree_models": False,
                "radial_volume_count": 8,
                "magnet_volume_count": 18,
                "physical_volume_count": 26,
                "ports": False,
                "casing_winding_split": False,
            },
            "premesh_topology": {
                "sha256": _sha(premesh),
                "status": "PREMESH_26_VOLUME_TOPOLOGY_PASS",
            },
        },
    )
    source_domain = _write(
        tmp_path / "domain.json",
        {
            "schema": "parastell.source_domain_audit/v1.0.0",
            "source_domain_gate_pass": True,
            "raw_h5m_sha256": _sha(dagmc),
            "source_mesh_sha256": _sha(source),
            "source_volume_id": 1,
            "source_component": "chamber",
            "source_material": "Vacuum",
            "input_immutability_pass": True,
        },
    )
    writeback = _write(
        tmp_path / "writeback.json",
        {
            "schema": "parastell.parametric_direct90_h5m_writeback/v1.0.0",
            "pass": True,
            "h5m_sha256": _sha(dagmc),
            "semantic_roles_by_global_volume_id": list(builder.SEMANTIC_ROLES),
            "magnet_identity_by_global_volume_id": [
                {
                    "magnet_id": f"magnet-{index:04d}",
                    "volume_global_id": index + 9,
                    "unique_match": True,
                    "pass": True,
                }
                for index in range(18)
            ],
        },
    )
    source_physics = _write(
        tmp_path / "source-physics.json",
        {
            "schema": "parastell.source_physics_manifest/v1.0.0",
            "particle": "neutron",
            "angle": "isotropic",
            "energy_law": "discrete",
            "energy_eV": [14_100_000.0],
            "probability": [1.0],
            "claim": "BOUNDED_SMOKE_ONLY",
        },
    )
    files = {
        "dagmc_h5m": dagmc,
        "source_mesh_h5m": source,
        "materials_xml": materials,
        "cross_sections_xml": cross_sections,
        "geometry_gate_receipt": geometry_gate,
        "dagmc_export_receipt": export,
        "premesh_topology": premesh,
        "source_domain_receipt": source_domain,
        "h5m_writeback": writeback,
        "source_physics_manifest": source_physics,
    }
    control = {
        "schema": builder.CONTROL_SCHEMA,
        "inputs": {
            name: {"path": str(path), "sha256": _sha(path)}
            for name, path in files.items()
        },
        "n_field_periods": 4,
        "modeled_extent_degrees": 90.0,
        "external_vacuum_radius_cm": 2500.0,
        "run": {"particles": 100, "batches": 2, "seed": 20260829},
    }
    control_path = _write(tmp_path / "control.json", control)
    return control_path, control


def test_accepts_hash_bound_direct90_evidence(tmp_path):
    control_path, _ = _packet(tmp_path)
    control = builder._load_control(control_path, _sha(control_path))
    paths = builder._validate_evidence(control)
    assert paths["dagmc_h5m"].name == "dagmc.h5m"


def test_committed_smoke_inputs_are_explicitly_nonproduction():
    root = Path(__file__).resolve().parents[1]
    source = json.loads(
        (
            root / "configs/wistell_d_parametric_openmc16_smoke_source.json"
        ).read_text(encoding="utf-8")
    )
    assert source["claim"] == "BOUNDED_SMOKE_ONLY"
    materials = ET.parse(
        root / "configs/wistell_d_parametric_openmc16_smoke_materials.xml"
    ).getroot()
    assert {row.get("name") for row in materials.findall("./material")} == (
        builder.MATERIAL_NAMES
    )
    assert not materials.findall("./material/element")
    assert not materials.findall("./material/sab")


def test_rejects_half_period_or_assembled_extent(tmp_path):
    control_path, control = _packet(tmp_path)
    control["modeled_extent_degrees"] = 45.0
    control_path.write_text(json.dumps(control), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 90"):
        builder._validate_evidence(
            builder._load_control(control_path, _sha(control_path))
        )


def test_rejects_noninteger_history_controls(tmp_path):
    control_path, control = _packet(tmp_path)
    control["run"]["particles"] = 100.5
    control_path.write_text(json.dumps(control), encoding="utf-8")
    with pytest.raises(ValueError, match="run controls"):
        builder._validate_evidence(
            builder._load_control(control_path, _sha(control_path))
        )


def test_rejects_semantic_magnet_permutation(tmp_path):
    control_path, control = _packet(tmp_path)
    writeback = control["inputs"]["h5m_writeback"]
    path = tmp_path / "writeback.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["magnet_identity_by_global_volume_id"][0]["volume_global_id"] = 10
    path.write_text(json.dumps(value), encoding="utf-8")
    writeback["sha256"] = _sha(path)
    control_path.write_text(json.dumps(control), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mapping"):
        builder._validate_evidence(
            builder._load_control(control_path, _sha(control_path))
        )


def test_requires_exact_unique_material_tag_names():
    materials = [SimpleNamespace(name=name) for name in builder.MATERIAL_NAMES]
    assert builder._material_names(materials) == builder.MATERIAL_NAMES
    materials[0].name = "wrong-name"
    with pytest.raises(ValueError, match="do not match"):
        builder._material_names(materials)


def test_nuclear_data_manifest_hashes_neutron_and_photon_files(tmp_path):
    neutron = tmp_path / "Fe56.h5"
    neutron.write_bytes(b"neutron")
    photon = tmp_path / "Fe.h5"
    photon.write_bytes(b"photon")
    materials = tmp_path / "materials.xml"
    materials.write_text(
        '<materials><material id="1" name="first_wall">'
        '<nuclide name="Fe56" ao="1.0"/></material></materials>',
        encoding="utf-8",
    )
    catalog = tmp_path / "cross_sections.xml"
    catalog.write_text(
        '<cross_sections><library type="neutron" materials="Fe56" '
        'path="Fe56.h5"/><library type="photon" materials="Fe" '
        'path="Fe.h5"/></cross_sections>',
        encoding="utf-8",
    )
    result = builder._nuclear_data_manifest(materials, catalog)
    assert {row["type"] for row in result["libraries"]} == {
        "neutron",
        "photon",
    }
    assert len(result["manifest_sha256"]) == 64


def test_build_model_preserves_mesh_source_and_photon_contract(
    tmp_path, monkeypatch
):
    paths = {}
    for name in (
        "dagmc_h5m",
        "source_mesh_h5m",
        "materials_xml",
        "cross_sections_xml",
        "geometry_gate_receipt",
        "dagmc_export_receipt",
        "premesh_topology",
        "source_domain_receipt",
        "h5m_writeback",
        "source_physics_manifest",
    ):
        path = tmp_path / name
        if name == "source_physics_manifest":
            path.write_text(
                json.dumps(
                    {
                        "energy_eV": [14_100_000.0],
                        "probability": [1.0],
                    }
                ),
                encoding="utf-8",
            )
        else:
            path.write_bytes(name.encode())
        paths[name] = path
    control = {
        "inputs": {
            name: {"sha256": _sha(path)} for name, path in paths.items()
        },
        "external_vacuum_radius_cm": 2500.0,
        "run": {"particles": 10, "batches": 2, "seed": 7},
    }
    monkeypatch.setattr(builder, "_load_control", lambda *_: control)
    monkeypatch.setattr(builder, "_validate_evidence", lambda *_: paths)
    monkeypatch.setattr(builder, "_maximum_vertex_radius", lambda *_: 100.0)
    monkeypatch.setattr(
        builder,
        "_source_arrays",
        lambda *_: (
            __import__("numpy").asarray(
                [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                dtype=float,
            ),
            __import__("numpy").asarray([1.0 / 6.0]),
            __import__("numpy").asarray([2.0]),
        ),
    )
    monkeypatch.setattr(
        builder,
        "audit_source_tetrahedra_arrays",
        lambda *_: {"tetrahedron_data_gate_pass": True},
    )
    monkeypatch.setattr(
        builder,
        "_nuclear_data_manifest",
        lambda *_: {"libraries": [], "manifest_sha256": "a" * 64},
    )

    class Reference:
        def openmc_one_period_geometry(self, **kwargs):
            assert kwargs["n_field_periods"] == 4
            return SimpleNamespace(dagmc_universe_count=1)

        def verify_unchanged(self):
            return True

    monkeypatch.setattr(
        builder.ReferenceGeometry,
        "open",
        classmethod(lambda cls, *args, **kwargs: Reference()),
    )

    captured = {}

    class Materials(list):
        cross_sections = None

        @classmethod
        def from_xml(cls, **kwargs):
            return cls(
                SimpleNamespace(name=name) for name in builder.MATERIAL_NAMES
            )

    class MeshSpatial:
        def __init__(self, mesh, *, strengths, volume_normalized):
            captured["mesh"] = mesh
            captured["strengths"] = strengths
            captured["volume_normalized"] = volume_normalized

    class Settings:
        pass

    class Model:
        def __init__(self, *, geometry, materials, settings):
            self.geometry = geometry
            self.materials = materials
            self.settings = settings

        def export_to_model_xml(self, path):
            path.write_text("<model/>", encoding="utf-8")

    fake_openmc = SimpleNamespace(
        __version__="0.16.0",
        Materials=Materials,
        UnstructuredMesh=lambda *args: args,
        IndependentSource=lambda **kwargs: SimpleNamespace(**kwargs),
        Settings=Settings,
        Model=Model,
        stats=SimpleNamespace(
            MeshSpatial=MeshSpatial,
            Isotropic=lambda: "isotropic",
            Discrete=lambda energies, probabilities: (energies, probabilities),
        ),
    )
    monkeypatch.setitem(sys.modules, "openmc", fake_openmc)
    output = tmp_path / "output"
    model, receipt = builder.build_model(
        tmp_path / "control.json", "b" * 64, output
    )
    assert captured["strengths"] == [2.0]
    assert captured["volume_normalized"] is False
    assert model.settings.photon_transport is True
    assert receipt["magnet_cell_ids"]["magnet-0017"] == 26
    assert (
        receipt["source"]["physical_rate_n_per_s_for_modeled_90_degrees"]
        == 2.0
    )
