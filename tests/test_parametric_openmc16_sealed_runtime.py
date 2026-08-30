from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from parastell import parametric_openmc16_sealed_runtime as runtime


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _binding(path):
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _install_fake_openmc(monkeypatch):
    captured = {}

    class Region:
        def __init__(self, parts):
            self.parts = list(parts)

        def __and__(self, other):
            return Region(self.parts + other.parts)

    class Surface:
        def __init__(self, *, surface_id, **kwargs):
            self.id = surface_id
            self.kwargs = kwargs
            self.periodic_surface = None

        def __neg__(self):
            return Region([("-", self)])

        def __pos__(self):
            return Region([("+", self)])

    class Materials(list):
        cross_sections = None

        @classmethod
        def from_xml(cls, **_kwargs):
            return cls(
                SimpleNamespace(name=name)
                for name in sorted(runtime.MATERIAL_NAMES)
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
            Path(path).write_text("<model/>", encoding="utf-8")

    fake = SimpleNamespace(
        __version__="0.16.0",
        YPlane=Surface,
        Plane=Surface,
        Sphere=Surface,
        DAGMCUniverse=lambda **kwargs: SimpleNamespace(**kwargs),
        Cell=lambda **kwargs: SimpleNamespace(**kwargs),
        Universe=lambda **kwargs: SimpleNamespace(**kwargs),
        Geometry=lambda root: SimpleNamespace(root_universe=root),
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
    monkeypatch.setitem(sys.modules, "openmc", fake)
    return captured


def _packet(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in runtime.EXPECTED_INPUTS:
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
    paths["materials_xml"].write_text("<materials/>", encoding="utf-8")
    paths["cross_sections_xml"].write_text(
        "<cross_sections/>", encoding="utf-8"
    )
    _write(
        paths["source_physics_manifest"],
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
    control = tmp_path / "control.json"
    control.write_text("{}", encoding="utf-8")
    neutron = tmp_path / "H1.h5"
    neutron.write_bytes(b"neutron")
    photon = tmp_path / "H.h5"
    photon.write_bytes(b"photon")
    libraries = [
        {"type": "neutron", "material": "H1", **_binding(neutron)},
        {"type": "photon", "material": "H", **_binding(photon)},
    ]
    nuclear = {
        "cross_sections_xml_sha256": _sha(paths["cross_sections_xml"]),
        "required_nuclides": ["H1"],
        "libraries": libraries,
    }
    nuclear["manifest_sha256"] = _canonical(nuclear)
    strengths = np.asarray([2.0, 3.0], dtype="<f8")
    strengths_path = tmp_path / runtime.STRENGTHS_FILENAME
    np.save(strengths_path, strengths, allow_pickle=False)
    dagmc_hash = _sha(paths["dagmc_h5m"])
    source_hash = _sha(paths["source_mesh_h5m"])
    seal = {
        "schema": runtime.SEAL_SCHEMA,
        "status": "GEOMETRY_TRANSPORT_SEAL_PASS",
        "claim": "BOUNDED_SMOKE_MODEL_INPUT_ONLY",
        "control": _binding(control),
        "inputs": {name: _binding(path) for name, path in paths.items()},
        "runtime": {},
        "geometry": {
            "dagmc_sha256": dagmc_hash,
            "modeled_extent_degrees": 90.0,
            "n_field_periods": 4,
            "native_id_inventory": {
                "native_id_gate_pass": True,
                "raw_h5m_sha256": dagmc_hash,
                "surface_ids": [1, 40],
                "volume_ids": list(range(1, 27)),
                "maximum_native_id": 40,
            },
            "first_available_wrapper_id": 41,
            "maximum_h5m_vertex_radius_cm": 1000.0,
            "external_vacuum_radius_cm": 2500.0,
            "containment_margin_cm": 1500.0,
        },
        "source": {
            "source_mesh_sha256": source_hash,
            "strengths_filename": runtime.STRENGTHS_FILENAME,
            "strengths_file_sha256": _sha(strengths_path),
            "strengths_raw_little_endian_f8_sha256": hashlib.sha256(
                strengths.tobytes()
            ).hexdigest(),
            "strength_dtype": "<f8",
            "strength_count": 2,
            "physical_rate_n_per_s_for_modeled_extent": 5.0,
            "tetrahedron_audit": {
                "tetrahedron_count": 2,
                "invalid_tetrahedron_count": 0,
                "negative_source_strength_count": 0,
                "total_source_strength_n_per_s": 5.0,
                "tetrahedron_data_gate_pass": True,
            },
            "volume_normalized": False,
        },
        "materials": {
            "names": sorted(runtime.MATERIAL_NAMES),
            "nuclear_data_manifest": nuclear,
        },
        "run": {"particles": 100, "batches": 2, "seed": 20260829},
        "magnet_cell_ids": {
            f"magnet-{index:04d}": index + 9 for index in range(18)
        },
        "inputs_immutable": True,
        "physical_h5m_mutation": False,
        "source_mesh_mutation": False,
        "openmc_model_export_status": "NOT_RUN",
        "openmc_geometry_debug_status": "NOT_RUN",
        "production_run_authorized": False,
    }
    seal["seal_content_sha256"] = _canonical(seal)
    seal_path = _write(tmp_path / runtime.SEAL_FILENAME, seal)
    return seal_path, seal


def _rewrite_seal(path, seal):
    value = copy.deepcopy(seal)
    value.pop("seal_content_sha256", None)
    value["seal_content_sha256"] = _canonical(value)
    _write(path, value)
    return value


def test_exports_periodic_model_from_seal_without_native_geometry_imports(
    tmp_path, monkeypatch
):
    seal_path, _ = _packet(tmp_path)
    captured = _install_fake_openmc(monkeypatch)
    attempted = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        attempted.append(name)
        if name.split(".")[0] in runtime.FORBIDDEN_RUNTIME_IMPORTS:
            raise AssertionError(f"forbidden runtime import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    output = tmp_path / "output"
    model, receipt = runtime.export_sealed_openmc16_model(
        seal_path, _sha(seal_path), output
    )
    assert not set(runtime.FORBIDDEN_RUNTIME_IMPORTS) & {
        name.split(".")[0] for name in attempted
    }
    assert captured["strengths"] == [2.0, 3.0]
    assert captured["volume_normalized"] is False
    assert model.settings.photon_transport is True
    assert receipt["transport_periodic_wrapper"] == {
        "start_surface_id": 41,
        "end_surface_id": 42,
        "vacuum_surface_id": 43,
        "wrapper_cell_id": 44,
        "root_universe_id": 45,
        "dagmc_universe_id": 46,
        "periodic_planes_paired": True,
        "period_degrees": 90.0,
        "auto_geom_ids": False,
    }
    assert receipt["source"]["strength_count"] == 2
    assert receipt["source"]["physical_rate_n_per_s_for_modeled_extent"] == 5.0
    assert receipt["geometry"]["volume_count"] == 26
    assert receipt["magnet_cell_ids"]["magnet-0017"] == 26
    assert receipt["transport_executed"] is False
    assert receipt["production_run_authorized"] is False
    assert (output / runtime.MODEL_FILENAME).is_file()
    stored = json.loads(
        (output / runtime.RECEIPT_FILENAME).read_text(encoding="utf-8")
    )
    claimed = stored.pop("receipt_content_sha256")
    assert claimed == _canonical(stored)


def test_runtime_module_has_no_static_native_geometry_imports():
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & set(runtime.FORBIDDEN_RUNTIME_IMPORTS)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda seal: seal["geometry"].update(
                first_available_wrapper_id=40
            ),
            "IDs/counts or wrapper",
        ),
        (
            lambda seal: seal["geometry"].update(containment_margin_cm=1499.0),
            "clearance",
        ),
        (
            lambda seal: seal["source"].update(strength_count=3),
            "count, raw hash, or sum",
        ),
        (
            lambda seal: seal["source"].update(
                physical_rate_n_per_s_for_modeled_extent=6.0
            ),
            "count, raw hash, or sum",
        ),
        (
            lambda seal: seal["magnet_cell_ids"].update({"magnet-0017": 25}),
            "magnet cell IDs",
        ),
    ],
)
def test_rejects_bad_ids_clearance_counts_and_sum(
    tmp_path, monkeypatch, change, match
):
    seal_path, seal = _packet(tmp_path)
    change(seal)
    _rewrite_seal(seal_path, seal)
    _install_fake_openmc(monkeypatch)
    with pytest.raises(ValueError, match=match):
        runtime.export_sealed_openmc16_model(
            seal_path, _sha(seal_path), tmp_path / "output"
        )
    assert not (tmp_path / "output").exists()


def test_rejects_strength_file_or_bound_input_hash_change(
    tmp_path, monkeypatch
):
    seal_path, _ = _packet(tmp_path)
    _install_fake_openmc(monkeypatch)
    (tmp_path / runtime.STRENGTHS_FILENAME).write_bytes(b"changed")
    with pytest.raises(ValueError, match="strength file hash"):
        runtime.export_sealed_openmc16_model(
            seal_path, _sha(seal_path), tmp_path / "output"
        )

    seal_path, seal = _packet(tmp_path / "second")
    Path(seal["inputs"]["dagmc_h5m"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="dagmc_h5m binding"):
        runtime.export_sealed_openmc16_model(
            seal_path, _sha(seal_path), tmp_path / "second-output"
        )


def test_requires_exact_openmc_version_and_create_only_output(
    tmp_path, monkeypatch
):
    seal_path, _ = _packet(tmp_path)
    _install_fake_openmc(monkeypatch)
    sys.modules["openmc"].__version__ = "0.15.2"
    with pytest.raises(RuntimeError, match="0.16.0 required"):
        runtime.export_sealed_openmc16_model(
            seal_path, _sha(seal_path), tmp_path / "output"
        )
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        runtime.export_sealed_openmc16_model(
            seal_path, _sha(seal_path), output
        )
