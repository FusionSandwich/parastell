import hashlib
import json

import pytest

from scripts.build_multirebco_outer_layer_request import build_request


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_cli_builder_binds_master_packet_and_two_family_contracts(tmp_path):
    manifest = _write(
        tmp_path / "master.json",
        {
            "schema": "parastell.source_conformal_geometry_manifest/v1.0.0",
            "master_interface": {"master_interface_sha256": "a" * 64},
            "master_interface_packet": {
                "path": "MASTER_SOURCE_INTERFACE.npz",
                "size_bytes": 123,
                "sha256": "b" * 64,
            },
        },
    )
    family_a = _write(
        tmp_path / "family-a.json",
        {"family_id": "family-a", "tape_ids": ["tape-a-1", "tape-a-2"]},
    )
    family_b = _write(
        tmp_path / "family-b.json",
        {"family_id": "family-b", "tape_ids": ["tape-b-1"]},
    )

    request = build_request(manifest, [family_a, family_b])

    assert request["status"] == "BLOCKED_PHYSICAL_CANDIDATE"
    assert request["master_interface_packet"]["sha256"] == "b" * 64
    assert [
        row["family_id"] for row in request["requested_rebco_families"]
    ] == ["family-a", "family-b"]
    assert request["p4_launch_authorized"] is False
    assert request["openmc_launch_authorized"] is False
    expected_hash = request.pop("request_sha256")
    assert (
        expected_hash
        == hashlib.sha256(
            json.dumps(
                request, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
    )


def test_cli_builder_rejects_single_family_request(tmp_path):
    manifest = _write(
        tmp_path / "master.json",
        {
            "master_interface": {"master_interface_sha256": "a" * 64},
            "master_interface_packet": {
                "path": "MASTER_SOURCE_INTERFACE.npz",
                "size_bytes": 123,
                "sha256": "b" * 64,
            },
        },
    )
    family = _write(
        tmp_path / "family.json",
        {"family_id": "family-a", "tape_ids": ["tape-a"]},
    )

    with pytest.raises(ValueError, match="at least two"):
        build_request(manifest, [family])
