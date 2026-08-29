from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest

from parastell.local_magnet_case import resolve_local_magnet_case
from parastell.material_identity import read_fusion_material_db
from parastell.material_identity import validate_material_record
from parastell.transport_response_plan import build_response_plan


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _binding(path):
    return {"path": path.name, "sha256": _sha(path)}


VECTOR = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
SOURCE_DTYPE = np.dtype(
    [
        ("r", VECTOR),
        ("u", VECTOR),
        ("E", "<f8"),
        ("time", "<f8"),
        ("wgt", "<f8"),
        ("delayed_group", "<i4"),
        ("surf_id", "<i4"),
        ("particle", "<i4"),
    ]
)


def _write_bank(path, surface_ids=(101, 102)):
    records = np.zeros(2, dtype=SOURCE_DTYPE)
    records["u"]["x"] = [1.0, 0.0]
    records["u"]["y"] = [0.0, 1.0]
    records["E"] = [14.1e6, 1.0e6]
    records["wgt"] = [1.0, 0.5]
    records["surf_id"] = surface_ids
    records["particle"] = [2112, 22]
    with h5py.File(path, "x") as target:
        target.attrs["filetype"] = np.bytes_(b"source")
        target.attrs["version"] = np.asarray([18, 2], dtype=int)
        target.create_dataset("source_bank", data=records)


def _case(tmp_path):
    coordinates_sha = "a" * 64
    coils = tmp_path / "coils.wistell-d"
    coils.write_text("periods 2\nend\n", encoding="utf-8")
    h5m = tmp_path / "dagmc.h5m"
    h5m.write_bytes(b"global dagmc")
    source = tmp_path / "source.json"
    _write(
        source,
        {
            "schema": "parastell.parametric_source_cad/v1.0.0",
            "status": "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING",
            "plan": {
                "extent_degrees": 90.0,
                "magnet_representation": "swept_filaments",
                "inputs": {
                    "coils": {"path": "/inputs/coils", "sha256": _sha(coils)}
                },
            },
            "magnet_inventory": {
                "casing_winding_split": False,
                "magnets": [
                    {
                        "magnet_id": "coil-alpha",
                        "coordinates_sha256": coordinates_sha,
                        "cross_section_cm": {
                            "width": 30.0,
                            "thickness": 30.0,
                        },
                    }
                ],
            },
        },
    )
    surfaces = tmp_path / "surfaces.json"
    _write(
        surfaces,
        {
            "schema": ("parastell.parametric_magnet_surface_manifest/v1.0.0"),
            "status": "PASS",
            "dagmc_sha256": _sha(h5m),
            "all_envelopes_close": True,
            "magnets": [
                {
                    "magnet": "coil-alpha",
                    "dagmc_volume_id": 9,
                    "dagmc_surface_ids": [101, 102],
                    "closed": True,
                    "coupling_interface": (
                        "homogenized_magnet_outer_boundary"
                    ),
                }
            ],
        },
    )
    root = tmp_path / "root-acceptance.json"
    accepted_inventory = tmp_path / "accepted-inventory.json"
    _write(
        accepted_inventory,
        {
            "schema": "parastell.accepted_magnet_component_inventory/v1.0.0",
            "geometry_gate_status": "PASS",
            "dagmc_sha256": _sha(h5m),
            "canonical_geometry_fingerprint": "c" * 64,
            "magnet_material_tags": ["mat:homogenized_magnet"],
            "components": [
                {
                    "magnet_id": "coil-alpha",
                    "component_id": "coil-alpha-homogenized-magnet",
                    "dagmc_volume_id": 9,
                    "material_tag": "mat:homogenized_magnet",
                    "surface_ids": [101, 102],
                    "source_geometry": {
                        "kind": "parametric_reference_manifest",
                        "path": str(source),
                        "sha256": _sha(source),
                    },
                }
            ],
        },
    )
    _write(
        root,
        {
            "schema": "parastell.root_accepted_magnet_inventory/v1.0.0",
            "acceptance_authority": "ROOT_GEOMETRY_GATE_ACCEPTED",
            "accepted_magnet_inventory_sha256": _sha(accepted_inventory),
            "dagmc_sha256": _sha(h5m),
            "canonical_geometry_fingerprint": "c" * 64,
            "accepted_canonical_material_tags": ["homogenized_magnet"],
        },
    )
    bank = tmp_path / "surface_source.h5"
    _write_bank(bank)
    model = tmp_path / "model.xml"
    model.write_text(
        "<model><settings><particles>500</particles><batches>2</batches>"
        "<seed>17</seed></settings></model>",
        encoding="utf-8",
    )
    statepoint = tmp_path / "statepoint.h5"
    with h5py.File(statepoint, "x") as target:
        target.attrs["filetype"] = np.bytes_(b"statepoint")
        target.attrs["openmc_version"] = np.asarray([0, 16, 0])
        target.create_dataset("run_mode", data=np.bytes_(b"fixed source"))
        target.create_dataset("n_particles", data=500)
        target.create_dataset("n_batches", data=2)
        target.create_dataset("seed", data=17)
    history = tmp_path / "history.json"
    _write(
        history,
        {
            "kind": "fixed_source_run",
            "run_id": "local-case-test",
            "settings_payload_path": str(model),
            "settings_payload_sha256": _sha(model),
            "statepoint_path": str(statepoint),
            "statepoint_sha256": _sha(statepoint),
        },
    )
    strict = tmp_path / "strict.json"
    verified_root = {
        "path": str(root),
        "sha256": _sha(root),
        "expected_sha256_from_frozen_run_control": _sha(root),
        "schema": "parastell.root_accepted_magnet_inventory/v1.0.0",
        "acceptance_authority": "ROOT_GEOMETRY_GATE_ACCEPTED",
        "accepted_magnet_inventory_sha256": _sha(accepted_inventory),
        "dagmc_sha256": _sha(h5m),
        "canonical_geometry_fingerprint": "c" * 64,
        "accepted_canonical_material_tags": ["homogenized_magnet"],
    }
    verified_root["verified_acceptance_sha256"] = _canonical_sha(verified_root)
    verified_inventory = {
        "path": str(accepted_inventory),
        "sha256": _sha(accepted_inventory),
        "accepted_sha256_from_root_receipt": _sha(accepted_inventory),
        "root_acceptance_receipt_sha256": _sha(root),
        "verified_root_acceptance_sha256": verified_root[
            "verified_acceptance_sha256"
        ],
        "schema": "parastell.accepted_magnet_component_inventory/v1.0.0",
        "geometry_gate_status": "PASS",
        "dagmc_sha256": _sha(h5m),
        "canonical_geometry_fingerprint": "c" * 64,
        "magnet_material_tags": ["mat:homogenized_magnet"],
        "components": [
            {
                "magnet_id": "coil-alpha",
                "component_id": "coil-alpha-homogenized-magnet",
                "dagmc_volume_id": 9,
                "material_tag": "mat:homogenized_magnet",
                "surface_ids": [101, 102],
                "source_geometry": {
                    "kind": "parametric_reference_manifest",
                    "path": str(source),
                    "sha256": _sha(source),
                },
            }
        ],
        "all_material_group_volumes_selected": True,
    }
    verified_inventory["verified_inventory_sha256"] = _canonical_sha(
        verified_inventory
    )
    _write(
        strict,
        {
            "schema": "parastell.openmc16_surface_run_audit/v1.0.0",
            "classification": "COMPLETE_CROSSING_BANK",
            "complete_classification_source": "parsed_artifacts_only",
            "dagmc": {
                "sha256": _sha(h5m),
                "canonical_geometry_fingerprints": ["c" * 64],
            },
            "model": {
                "path": str(model),
                "sha256": _sha(model),
                "source_histories": 1000,
                "particles_per_batch": 500,
                "batches": 2,
                "seed": 17,
            },
            "statepoint": {
                "path": str(statepoint),
                "sha256": _sha(statepoint),
                "particles_per_batch": 500,
                "batches": 2,
                "seed": 17,
            },
            "root_acceptance_receipt": verified_root,
            "surface_source_files": [
                {"path": str(bank), "sha256": _sha(bank)}
            ],
            "capacity": {"capacity_proven_not_reached": True},
            "same_run_integrity": {
                "passes": True,
                "model_xml_sha256": _sha(model),
                "statepoint_sha256": _sha(statepoint),
                "accepted_magnet_inventory_sha256": _sha(accepted_inventory),
                "verified_magnet_inventory_sha256": verified_inventory[
                    "verified_inventory_sha256"
                ],
                "root_acceptance_receipt_sha256": _sha(root),
                "verified_root_acceptance_sha256": verified_root[
                    "verified_acceptance_sha256"
                ],
                "localization_topology_manifest_sha256": "f" * 64,
            },
            "localization_topology_binding": {"manifest_sha256": "f" * 64},
            "accepted_magnet_inventory": verified_inventory,
        },
    )
    response = tmp_path / "response.json"
    _write(
        response,
        build_response_plan(
            case_id="local-alpha",
            magnet_ids=["coil-alpha"],
            neutron_energy_edges_eV=[0.0, 1.0, 20.0e6],
            photon_energy_edges_eV=[0.0, 1.0e6, 20.0e6],
        ),
    )
    material_db = tmp_path / "materials.json"
    _write(
        material_db,
        {
            "case": {
                "density": 8.0,
                "comp": {"Fe56": 1.0},
                "metadata": {"name": "case", "citation": "test"},
            },
            "winding": {
                "density": 6.0,
                "comp": {"Cu63": 1.0},
                "metadata": {"name": "winding", "citation": "test"},
            },
        },
    )
    materials = read_fusion_material_db(material_db, temperature_K=20.0)
    case = {
        "schema": "parastell.local_magnet_case/v1.0.0",
        "case_id": "local-alpha",
        "global_geometry": {
            "source_cad_manifest": _binding(source),
            "magnet_surface_manifest": _binding(surfaces),
            "dagmc_h5m": _binding(h5m),
            "coil_input": _binding(coils),
            "root_acceptance_receipt": _binding(root),
        },
        "magnet_selection": {
            "magnet_id": "coil-alpha",
            "filament_index": 0,
            "filament_coordinates_sha256": coordinates_sha,
            "global_bank_interface": ("homogenized_magnet_outer_boundary"),
            "local_coupling_interface": "outer_casing_external",
        },
        "case_thickness_cm": 1.0,
        "components": [
            {
                "component_id": "alpha-case",
                "role": "outer_casing",
                "material_id": "case",
            },
            {
                "component_id": "alpha-winding",
                "role": "winding_pack",
                "material_id": "winding",
            },
        ],
        "materials": materials,
        "surface_source": {
            "bank": _binding(bank),
            "history_receipt": _binding(history),
            "strict_surface_run_audit": _binding(strict),
            "source_histories": 1000,
        },
        "response_plan": _binding(response),
    }
    case_path = tmp_path / "case.json"
    _write(case_path, case)
    return case_path, case


def test_arbitrary_magnet_id_inherits_outer_geometry_and_binds_source(
    tmp_path,
):
    case_path, _ = _case(tmp_path)
    plan = resolve_local_magnet_case(case_path)
    assert plan["magnet_id"] == "coil-alpha"
    assert plan["geometry_request"]["filament_indices"] == [0]
    assert plan["geometry_request"]["outer_width_cm"] == 30.0
    assert plan["geometry_request"]["inner_width_cm"] == 28.0
    assert plan["global_envelope"]["dagmc_surface_ids"] == [101, 102]
    assert plan["claims"]["tape_resolved"] is False
    assert plan["claims"]["production_authorized"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda case: case["magnet_selection"].update(filament_index=1),
            "filament index/hash",
        ),
        (
            lambda case: case.update(case_thickness_cm=15.0),
            "positive winding pack",
        ),
        (
            lambda case: case["magnet_selection"].update(
                local_coupling_interface="winding_pack"
            ),
            "coupling interface",
        ),
        (
            lambda case: case["surface_source"].update(source_histories=0),
            "positive integer",
        ),
    ],
)
def test_case_rejects_identity_geometry_interface_and_history_drift(
    tmp_path, mutation, match
):
    case_path, case = _case(tmp_path)
    mutation(case)
    _write(case_path, case)
    with pytest.raises(ValueError, match=match):
        resolve_local_magnet_case(case_path)


def test_case_rejects_mutated_bound_artifact(tmp_path):
    case_path, _ = _case(tmp_path)
    (tmp_path / "surface_source.h5").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="surface source bank hash mismatch"):
        resolve_local_magnet_case(case_path)


def test_case_rejects_foreign_surface_even_when_hash_is_updated(tmp_path):
    case_path, case = _case(tmp_path)
    bank = tmp_path / "surface_source.h5"
    bank.unlink()
    _write_bank(bank, surface_ids=(101, 999))
    case["surface_source"]["bank"] = _binding(bank)
    strict = tmp_path / "strict.json"
    strict_value = json.loads(strict.read_text(encoding="utf-8"))
    strict_value["surface_source_files"][0]["sha256"] = _sha(bank)
    _write(strict, strict_value)
    case["surface_source"]["strict_surface_run_audit"] = _binding(strict)
    _write(case_path, case)
    with pytest.raises(ValueError, match="outside the requested envelope"):
        resolve_local_magnet_case(case_path)


def test_case_rejects_history_and_strict_run_drift(tmp_path):
    case_path, case = _case(tmp_path)
    history = tmp_path / "history.json"
    _write(
        history,
        {
            "kind": "fixed_source_run",
            "run_id": "wrong-run",
            "settings_payload_path": str(tmp_path / "model.xml"),
            "settings_payload_sha256": _sha(tmp_path / "model.xml"),
            "statepoint_path": str(tmp_path / "statepoint.h5"),
            "statepoint_sha256": "0" * 64,
        },
    )
    case["surface_source"]["history_receipt"] = _binding(history)
    _write(case_path, case)
    with pytest.raises(ValueError, match="history binding hash"):
        resolve_local_magnet_case(case_path)


def test_case_rejects_serializer_fixture_history(tmp_path):
    case_path, case = _case(tmp_path)
    history = tmp_path / "history.json"
    _write(
        history,
        {
            "kind": "serializer_fixture",
            "fixture_id": "synthetic-only",
            "source_histories": 1000,
        },
    )
    case["surface_source"]["history_receipt"] = _binding(history)
    _write(case_path, case)
    with pytest.raises(ValueError, match="strict OpenMC run"):
        resolve_local_magnet_case(case_path)


def test_case_rejects_unaccepted_strict_run(tmp_path):
    case_path, case = _case(tmp_path)
    strict = tmp_path / "strict.json"
    value = json.loads(strict.read_text(encoding="utf-8"))
    value["classification"] = "TRUNCATED_INVALID_BANK"
    _write(strict, value)
    case["surface_source"]["strict_surface_run_audit"] = _binding(strict)
    _write(case_path, case)
    with pytest.raises(ValueError, match="accepted complete run"):
        resolve_local_magnet_case(case_path)


def test_case_rejects_copied_verification_hashes_and_topology_drift(tmp_path):
    case_path, case = _case(tmp_path)
    strict = tmp_path / "strict.json"
    value = json.loads(strict.read_text(encoding="utf-8"))
    value["root_acceptance_receipt"]["verified_acceptance_sha256"] = "1" * 64
    value["same_run_integrity"]["verified_root_acceptance_sha256"] = "1" * 64
    _write(strict, value)
    case["surface_source"]["strict_surface_run_audit"] = _binding(strict)
    _write(case_path, case)
    with pytest.raises(ValueError, match="accepted complete run"):
        resolve_local_magnet_case(case_path)


def test_case_rejects_rehashed_embedded_inventory_drift(tmp_path):
    case_path, case = _case(tmp_path)
    strict = tmp_path / "strict.json"
    value = json.loads(strict.read_text(encoding="utf-8"))
    inventory = value["accepted_magnet_inventory"]
    inventory["components"][0]["source_geometry"]["sha256"] = "9" * 64
    inventory.pop("verified_inventory_sha256")
    inventory["verified_inventory_sha256"] = _canonical_sha(inventory)
    value["same_run_integrity"]["verified_magnet_inventory_sha256"] = (
        inventory["verified_inventory_sha256"]
    )
    _write(strict, value)
    case["surface_source"]["strict_surface_run_audit"] = _binding(strict)
    _write(case_path, case)
    with pytest.raises(ValueError, match="accepted complete run"):
        resolve_local_magnet_case(case_path)

    topology = tmp_path / "topology"
    topology.mkdir()
    case_path, case = _case(topology)
    strict = topology / "strict.json"
    value = json.loads(strict.read_text(encoding="utf-8"))
    value["same_run_integrity"]["localization_topology_manifest_sha256"] = (
        "0" * 64
    )
    _write(strict, value)
    case["surface_source"]["strict_surface_run_audit"] = _binding(strict)
    _write(case_path, case)
    with pytest.raises(ValueError, match="accepted complete run"):
        resolve_local_magnet_case(case_path)


def test_material_validator_rejects_composition_drift(tmp_path):
    _, case = _case(tmp_path)
    material = dict(case["materials"][0])
    material["isotopes"] = {"Fe56": 0.5, "Fe57": 0.5}
    with pytest.raises(ValueError, match="composition hash"):
        validate_material_record(material)


def test_material_validator_rejects_source_and_record_drift(tmp_path):
    _, case = _case(tmp_path)
    material = dict(case["materials"][0])
    material["source"] = dict(material["source"])
    material["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source provenance hash"):
        validate_material_record(material)

    material = dict(case["materials"][0])
    material["citation"] = "changed"
    with pytest.raises(ValueError, match="material record hash"):
        validate_material_record(material)
