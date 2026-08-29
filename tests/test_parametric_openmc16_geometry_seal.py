from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from parastell import parametric_openmc16_geometry_seal as seal


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in (
        "dagmc_h5m",
        "source_mesh_h5m",
        "geometry_gate_receipt",
        "dagmc_export_receipt",
        "premesh_topology",
        "source_domain_receipt",
        "h5m_writeback",
        "source_physics_manifest",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
    materials = tmp_path / "materials.xml"
    names = ["Vacuum", *sorted(seal.MATERIAL_NAMES)]
    materials.write_text(
        "<materials>"
        + "".join(
            f'<material id="{i + 1}" name="{name}"/>'
            for i, name in enumerate(names)
        )
        + "</materials>",
        encoding="utf-8",
    )
    paths["materials_xml"] = materials
    cross_sections = tmp_path / "cross_sections.xml"
    cross_sections.write_text("<cross_sections/>", encoding="utf-8")
    paths["cross_sections_xml"] = cross_sections
    control_path = tmp_path / "control.json"
    control_path.write_text("{}", encoding="utf-8")
    control = {
        "modeled_extent_degrees": 90.0,
        "n_field_periods": 4,
        "external_vacuum_radius_cm": 2500.0,
        "run": {"particles": 10, "batches": 2, "seed": 7},
    }
    monkeypatch.setattr(seal, "_load_control", lambda *_: control)
    monkeypatch.setattr(seal, "_validate_evidence", lambda *_: paths)
    monkeypatch.setattr(
        seal,
        "native_dagmc_id_inventory",
        lambda path: {
            "raw_h5m_sha256": _sha(path),
            "surface_ids": [1, 20, 40],
            "volume_ids": list(range(1, 27)),
            "native_id_gate_pass": True,
            "maximum_native_id": 40,
        },
    )
    monkeypatch.setattr(seal, "_maximum_vertex_radius", lambda *_: 1000.0)
    monkeypatch.setattr(
        seal,
        "_source_arrays",
        lambda *_: (
            np.asarray(
                [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                dtype=float,
            ),
            np.asarray([1.0 / 6.0]),
            np.asarray([2.0]),
        ),
    )
    monkeypatch.setattr(
        seal,
        "_nuclear_data_manifest",
        lambda *_: {"manifest_sha256": "a" * 64, "libraries": []},
    )
    monkeypatch.setattr(
        seal,
        "_runtime_identity",
        lambda: {
            "python_executable": "/qualified/python3.11",
            "python_executable_sha256": "b" * 64,
            "python_version": "3.11.15",
            "numpy_version": "1.26.4",
            "native_modules": {},
        },
    )
    return control_path, paths


def test_writes_create_only_geometry_seal_without_openmc_import(
    tmp_path, monkeypatch
):
    control, paths = _case(tmp_path, monkeypatch)
    output = tmp_path / "sealed"
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "openmc" or name.startswith("openmc."):
            raise AssertionError("geometry seal imported OpenMC")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = seal.create_geometry_transport_seal(
        control, _sha(control), output
    )
    stored = json.loads(
        (output / seal.SEAL_FILENAME).read_text(encoding="utf-8")
    )
    strengths = np.load(output / seal.STRENGTHS_FILENAME, allow_pickle=False)
    content_hash = stored.pop("seal_content_sha256")

    assert result["status"] == "GEOMETRY_TRANSPORT_SEAL_PASS"
    assert content_hash == seal._canonical_sha(stored)
    assert strengths.dtype.str == "<f8"
    assert np.array_equal(strengths, np.asarray([2.0], dtype="<f8"))
    assert result["geometry"]["first_available_wrapper_id"] == 41
    assert result["geometry"]["native_id_inventory"]["volume_ids"] == list(
        range(1, 27)
    )
    assert result["source"]["strengths_file_sha256"] == _sha(
        output / seal.STRENGTHS_FILENAME
    )
    assert result["source"]["physical_rate_n_per_s_for_modeled_extent"] == 2.0
    assert result["openmc_model_export_status"] == "NOT_RUN"
    assert result["production_run_authorized"] is False
    assert result["inputs"] == seal._input_hashes(paths)


def test_rejects_existing_output_and_bad_geometry_bounds(
    tmp_path, monkeypatch
):
    control, _ = _case(tmp_path, monkeypatch)
    output = tmp_path / "sealed"
    output.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        seal.create_geometry_transport_seal(control, _sha(control), output)

    output.rmdir()
    monkeypatch.setattr(seal, "_maximum_vertex_radius", lambda *_: 2500.0)
    with pytest.raises(ValueError, match="does not enclose"):
        seal.create_geometry_transport_seal(control, _sha(control), output)
    assert not output.exists()


def test_rejects_native_id_or_source_data_failure(tmp_path, monkeypatch):
    control, _ = _case(tmp_path, monkeypatch)
    output = tmp_path / "sealed"
    monkeypatch.setattr(
        seal,
        "native_dagmc_id_inventory",
        lambda *_: {"native_id_gate_pass": False},
    )
    with pytest.raises(ValueError, match="ID inventory"):
        seal.create_geometry_transport_seal(control, _sha(control), output)
    assert not output.exists()

    control, _ = _case(tmp_path / "second", monkeypatch)
    monkeypatch.setattr(
        seal,
        "_source_arrays",
        lambda *_: (
            np.asarray(
                [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                dtype=float,
            ),
            np.asarray([1.0 / 6.0]),
            np.asarray([-1.0]),
        ),
    )
    with pytest.raises(ValueError, match="no valid positive-strength"):
        seal.create_geometry_transport_seal(
            control, _sha(control), tmp_path / "second-sealed"
        )


@pytest.mark.parametrize(
    "target",
    ["dagmc_h5m", "source_mesh_h5m", "geometry_gate_receipt", "control"],
)
def test_rejects_input_mutation_before_writing_output(
    tmp_path, monkeypatch, target
):
    control, paths = _case(tmp_path, monkeypatch)
    output = tmp_path / "sealed"
    original = seal._source_arrays

    def mutate_during_audit(path):
        values = original(path)
        mutation_target = control if target == "control" else paths[target]
        mutation_target.write_bytes(b"mutated")
        return values

    monkeypatch.setattr(seal, "_source_arrays", mutate_during_audit)
    with pytest.raises(RuntimeError, match="changed during|control changed"):
        seal.create_geometry_transport_seal(control, _sha(control), output)
    assert not output.exists()
