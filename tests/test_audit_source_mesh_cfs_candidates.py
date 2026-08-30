import json

import pytest

from parastell.parametric_openmc16_model import _canonical_sha
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
