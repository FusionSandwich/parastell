import json

import pytest

from parastell.parametric_openmc16_model import _canonical_sha
from scripts.audit_source_mesh_cfs_candidates import _candidate_passes
from scripts.audit_source_mesh_cfs_candidates import _verified_control
from scripts.audit_source_mesh_cfs_candidates import _verified_manifest


def _manifest():
    value = {
        "schema": "parastell.source_mesh_cfs_candidates/v1.0.0",
        "status": "CANDIDATES_BUILT_CONTAINMENT_PENDING",
        "geometry_mutated": False,
        "inner_cfs_planes_mutated": False,
    }
    value["receipt_content_sha256"] = _canonical_sha(value)
    return value


def test_verified_manifest_binds_file_and_canonical_content(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _verified_manifest(path, digest)["geometry_mutated"] is False


def test_verified_manifest_rejects_content_tampering(tmp_path):
    value = _manifest()
    value["status"] = "TAMPERED"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="content hash mismatch"):
        _verified_manifest(path, digest)


def test_verified_control_binds_scientific_acceptance_values(tmp_path):
    value = {
        "schema": "parastell.source_mesh_outer_cfs_audit_control/v1.0.0",
        "dagmc_sha256": "a" * 64,
        "candidate_manifest_sha256": "b" * 64,
        "source_volume_id": 1,
        "source_material": "Vacuum",
        "minimum_clearance_cm": 1.0e-4,
        "maximum_omitted_source_fraction": 1.0e-6,
    }
    value["receipt_content_sha256"] = _canonical_sha(value)
    path = tmp_path / "control.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _verified_control(path, digest)[
        "maximum_omitted_source_fraction"
    ] == pytest.approx(1.0e-6)


@pytest.mark.parametrize("source_volume_id", [True, 1.5, "1", 0])
def test_verified_control_rejects_coerced_volume_ids(
    tmp_path, source_volume_id
):
    value = {
        "schema": "parastell.source_mesh_outer_cfs_audit_control/v1.0.0",
        "dagmc_sha256": "a" * 64,
        "candidate_manifest_sha256": "b" * 64,
        "source_volume_id": source_volume_id,
        "source_material": "Vacuum",
        "minimum_clearance_cm": 1.0e-4,
        "maximum_omitted_source_fraction": 1.0e-6,
    }
    value["receipt_content_sha256"] = _canonical_sha(value)
    path = tmp_path / "control.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="volume ID"):
        _verified_control(path, digest)


def _passing_candidate(**changes):
    values = {
        "domain_pass": True,
        "all_samples_enclosed": True,
        "minimum_clearance_cm": 0.01,
        "required_clearance_cm": 0.001,
        "reference_rate": 1.0e20,
        "candidate_rate": 0.9999995e20,
        "omitted_fraction": 5.0e-7,
        "maximum_omitted_fraction": 1.0e-6,
        "manifest_identity_matches": True,
        "input_immutability_pass": True,
    }
    values.update(changes)
    return _candidate_passes(**values)


def test_candidate_acceptance_enforces_source_loss_and_immutability():
    assert _passing_candidate() is True
    assert _passing_candidate(omitted_fraction=2.0e-6) is False
    assert _passing_candidate(input_immutability_pass=False) is False
    assert _passing_candidate(manifest_identity_matches=False) is False
