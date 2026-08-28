import hashlib
import json

import pytest

from scripts.rebind_regenerated_cad_comparison import rebind


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root, *, brep=b"same", exporter=False):
    artifacts = {}
    for index, role in enumerate(("casing", "winding_pack"), start=7):
        path = root / f"{role}.brep"
        path.write_bytes(brep + role.encode())
        artifacts[role] = {
            "dagmc_volume_id": index,
            "brep": {"path": path.name, "sha256": _hash(path)},
        }
    source = {
        "revision": "abc",
        "coils_sha256": "c" * 64,
        "parameters": {"width_cm": 40.0},
    }
    if exporter:
        source["exporter_implementation"] = {
            "local_identifier": "magnet_source_cad.py",
            "sha256": "e" * 64,
        }
    payload = {
        "magnet_id": "magnet-8",
        "built_coil_index": 0,
        "source_filament_index": 0,
        "coordinate_contract": {"cad_length_unit": "cm"},
        "target_dagmc": {
            "canonical_geometry_fingerprint": "f" * 64,
            "role_volume_ids": {"casing": 7, "winding_pack": 8},
        },
        "source": source,
        "artifacts": artifacts,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _case(tmp_path, *, new_brep=b"same"):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    h5m = tmp_path / "canonical.h5m"
    h5m.write_bytes(b"canonical")
    old = _manifest(old_root, exporter=False)
    new = _manifest(new_root, brep=new_brep, exporter=True)
    comparison = old_root / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "status": "PASS",
                "classification": "PARASTELL_SOURCE_CAD_REGENERATED_EXACT_INPUTS",
                "canonical_h5m": {"sha256": _hash(h5m)},
                "source_cad_manifest": {"sha256": _hash(old)},
                "roles": {"casing": {"status": "PASS"}},
                "limitations": [],
            }
        ),
        encoding="utf-8",
    )
    return old, comparison, new, h5m


def test_rebinds_only_byte_identical_source_solids(tmp_path):
    old, comparison, new, h5m = _case(tmp_path)
    output = tmp_path / "rebound.json"
    result = rebind(
        prior_manifest_path=old,
        prior_comparison_path=comparison,
        new_manifest_path=new,
        canonical_h5m_path=h5m,
        output_path=output,
    )
    assert result["status"] == "PASS"
    assert result["target_geometry_fingerprint"] == "f" * 64
    assert result["evidence_reuse"]["geometric_metrics_recomputed"] is False
    assert all(
        row["byte_identical_to_prior_compared_brep"]
        for row in result["evidence_reuse"]["roles"].values()
    )


def test_rebind_rejects_changed_brep(tmp_path):
    old, comparison, new, h5m = _case(tmp_path, new_brep=b"changed")
    with pytest.raises(ValueError, match="not byte-identical"):
        rebind(
            prior_manifest_path=old,
            prior_comparison_path=comparison,
            new_manifest_path=new,
            canonical_h5m_path=h5m,
            output_path=tmp_path / "rebound.json",
        )
