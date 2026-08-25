import hashlib
import json

import h5py
import numpy as np
import pytest

from parastell.pilot_fixture import read_pilot_fixture_manifest
from parastell.pilot_fixture import validate_pilot_fixture_manifest
from parastell.pilot_fixture import write_pilot_fixture_manifest


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_validated_bundle(
    tmp_path, monkeypatch, *, particles=("neutron", "photon")
):
    root = tmp_path / "external-pilot"
    boundary_path = root / "boundary" / "magnet-A" / "bank.h5"
    boundary_path.parent.mkdir(parents=True)
    manifest = {
        "time_semantics": "prompt particle flight time, not irradiation time",
        "field_availability": {"time_s": {"available": True}},
        "energy_axes": {
            "neutron_energy_edges_eV": [0.0, 20.0e6],
            "photon_energy_edges_eV": [0.0, 1.0e9],
        },
        "envelope": {
            "surfaces": [
                {
                    "surface_id": 1,
                    "centroid_global_cm": [1.0, 0.0, 0.0],
                    "outward_normal_global": [1.0, 0.0, 0.0],
                    "toroidal_direction_global": [0.0, 1.0, 0.0],
                    "poloidal_direction_global": [0.0, 0.0, 1.0],
                }
            ]
        },
    }
    strings = h5py.string_dtype("utf-8")
    count = len(particles)
    particle_pdg = [
        {"neutron": 2112, "photon": 22}[name] for name in particles
    ]
    energies = [14.1e6 if name == "neutron" else 1.0e6 for name in particles]
    with h5py.File(boundary_path, "w") as handle:
        handle.create_dataset(
            "manifest_json", data=json.dumps(manifest), dtype=strings
        )
        records = handle.create_group("records")
        records.create_dataset("record_id", data=np.arange(count))
        records.create_dataset(
            "position_global_cm", data=np.tile([[1.0, 0.0, 0.0]], (count, 1))
        )
        records.create_dataset("position_local_cm", data=np.zeros((count, 3)))
        records.create_dataset(
            "direction_global", data=np.tile([[-1.0, 0.0, 0.0]], (count, 1))
        )
        records.create_dataset(
            "direction_local", data=np.tile([[0.0, 0.0, -1.0]], (count, 1))
        )
        records.create_dataset(
            "outward_normal_global",
            data=np.tile([[1.0, 0.0, 0.0]], (count, 1)),
        )
        records.create_dataset("surface_id", data=np.ones(count, dtype=int))
        records.create_dataset("particle", data=list(particles), dtype=strings)
        records.create_dataset("particle_pdg", data=particle_pdg)
        records.create_dataset("energy_eV", data=energies)
        records.create_dataset("energy_group", data=np.zeros(count, dtype=int))
        records.create_dataset("time_s", data=np.full(count, 2.0e-8))
        records.create_dataset(
            "facet_id", data=["facet-1"] * count, dtype=strings
        )
        records.create_dataset(
            "barycentric_coordinates",
            data=np.tile([[0.2, 0.3, 0.5]], (count, 1)),
        )
        records.create_dataset(
            "distance_to_facet_residual_cm", data=np.zeros(count)
        )
        records.create_dataset(
            "facet_mapping_status", data=["EXACT"] * count, dtype=strings
        )

    scalar_path = root / "volume_flux" / "all-selected-magnets" / "scalar.h5"
    scalar_path.parent.mkdir(parents=True)
    scalar_path.write_bytes(b"compact scalar placeholder validated upstream")
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    products = [
        {
            "kind": "volume_scalar_flux",
            "path": scalar_path.relative_to(root).as_posix(),
            "sha256": _sha256(scalar_path),
            "size_bytes": scalar_path.stat().st_size,
        },
        {
            "kind": "boundary_phase_space",
            "path": boundary_path.relative_to(root).as_posix(),
            "sha256": _sha256(boundary_path),
            "size_bytes": boundary_path.stat().st_size,
        },
    ]
    bundle = {
        "provenance": {
            "parastell_commit": "a" * 40,
            "openmc_version": "0.16.0",
            "openmc_commit": "b" * 40,
            "statepoint_sha256": "c" * 64,
            "histories": 100,
            "batches": 10,
        },
        "geometry": {"canonical_geometry_fingerprint": "d" * 64},
        "source": {"physical_source_rate_per_s": 3.0},
        "magnet_inventory": [{"magnet_id": "magnet-A"}],
        "products": products,
        "bank_completeness": {
            "status": "COMPLETE",
            "record_count_total": count,
            "all_banks_valid": True,
        },
    }
    monkeypatch.setattr(
        "parastell.magnet_radiation_field_bundle.read_radiation_field_bundle",
        lambda path: bundle,
    )
    return root, boundary_path


def test_real_pilot_receipt_is_external_hash_bound_and_source_safe(
    tmp_path, monkeypatch
):
    bundle, _ = _fake_validated_bundle(tmp_path, monkeypatch)
    output = tmp_path / "committed" / "pilot-manifest.json"
    written = write_pilot_fixture_manifest(
        output, bundle, artifact_uri="artifact://qualified/pilot-A"
    )
    restored = read_pilot_fixture_manifest(output)

    assert restored["manifest_sha256"] == written["manifest_sha256"]
    assert (
        restored["external_artifact"]["uri"] == "artifact://qualified/pilot-A"
    )
    assert restored["pilot_summary"]["particle_record_counts"] == {
        "neutron": 1,
        "photon": 1,
    }
    assert restored["pilot_summary"]["prompt_time_record_count"] == 2
    assert {item["status"] for item in restored["acceptance"].values()} == {
        "PASS"
    }
    assert (
        restored["qualification_status"] == "STRUCTURAL_PILOT_POPULATION_PASS"
    )
    assert (
        restored["qualification_scope"]
        == "boundary_record_particle_population_only_not_physics_qualification"
    )
    assert restored["population_qualification"]["status"] == "PASS"
    assert restored["committed_payload"]["included_large_artifacts"] == []
    assert list(output.parent.iterdir()) == [output]


def test_pilot_receipt_rejects_failed_acceptance_gate():
    value = {
        "schema": "parastell.magnet_pilot_fixture_manifest/v1.0.0",
        "created_utc": "2026-08-25T00:00:00+00:00",
        "acceptance_scope": "reader_contract_and_normalization_plumbing_only",
        "qualification_scope": (
            "boundary_record_particle_population_only_not_physics_qualification"
        ),
        "qualification_status": "STRUCTURAL_PILOT_POPULATION_BLOCKED",
        "qualification_limitations": ["not physics qualified"],
        "external_artifact": {
            "uri": "artifact://pilot",
            "local_path_hint": "external",
            "bundle_manifest_sha256": "a" * 64,
            "canonical_tree_sha256": "b" * 64,
            "file_count": 1,
            "size_bytes": 1,
        },
        "producer_identity": {},
        "pilot_summary": {
            "physical_source_rate_per_s": 1.0,
            "bank_count": 0,
            "record_count": 0,
            "particle_record_counts": {"neutron": 0, "photon": 0},
        },
        "acceptance": {
            name: {"status": "PASS", "evidence": "checked"}
            for name in (
                "schema_round_trip",
                "current_conservation",
                "scalar_flux_normalization",
                "coordinate_transform",
                "particle_identity",
                "energy_preservation",
                "direction_preservation",
                "time_semantics",
            )
        },
        "population_qualification": {
            "status": "BLOCKED",
            "requirement": (
                "at least one validated boundary bank and positive neutron "
                "and photon boundary-record populations"
            ),
            "criteria": {
                name: {
                    "status": "BLOCKED",
                    "evidence": (
                        "not observed in the validated external boundary banks"
                    ),
                }
                for name in (
                    "boundary_bank_present",
                    "boundary_record_present",
                    "neutron_records_present",
                    "photon_records_present",
                )
            },
            "blocking_reasons": [
                "boundary_bank_present",
                "boundary_record_present",
                "neutron_records_present",
                "photon_records_present",
            ],
        },
        "committed_payload": {
            "included_large_artifacts": [],
            "boundary_downsample": {"status": "OMITTED", "reason": "unsafe"},
            "scalar_downsample": {"status": "OMITTED", "reason": "external"},
        },
    }
    value["acceptance"]["current_conservation"]["status"] = "FAIL"
    with pytest.raises(ValueError, match="did not pass"):
        validate_pilot_fixture_manifest(value)


def test_pilot_receipt_detects_direction_transform_corruption(
    tmp_path, monkeypatch
):
    bundle, boundary = _fake_validated_bundle(tmp_path, monkeypatch)
    with h5py.File(boundary, "r+") as handle:
        handle["records/direction_local"][...] = [[1.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match="direction transform"):
        write_pilot_fixture_manifest(tmp_path / "bad.json", bundle)


def test_structurally_valid_neutron_only_pilot_is_population_blocked(
    tmp_path, monkeypatch
):
    bundle, _ = _fake_validated_bundle(
        tmp_path, monkeypatch, particles=("neutron",)
    )
    output = tmp_path / "neutron-only.json"
    written = write_pilot_fixture_manifest(output, bundle)

    assert {item["status"] for item in written["acceptance"].values()} == {
        "PASS"
    }
    assert (
        written["qualification_status"]
        == "STRUCTURAL_PILOT_POPULATION_BLOCKED"
    )
    assert (
        written["population_qualification"]["criteria"][
            "photon_records_present"
        ]["status"]
        == "BLOCKED"
    )
    assert written["population_qualification"]["blocking_reasons"] == [
        "photon_records_present"
    ]
