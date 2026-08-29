from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import build_parametric_magnet_surface_manifest as builder


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path):
    dagmc = tmp_path / "dagmc.h5m"
    dagmc.write_bytes(b"immutable-h5m")
    magnets = []
    for index in range(18):
        magnets.append(
            {
                "magnet_id": f"magnet-{index:04d}",
                "volume_id": index + 9,
                "closed": True,
                "bounding_box_cm": {
                    "lower": [100.0 + index, 20.0, -5.0],
                    "upper": [110.0 + index, 30.0, 5.0],
                },
                "outer_envelope_surface_ids": [100 + index],
                "surface_count": 1,
                "material_tag": "magnets",
                "physical_role": "original_homogenized_magnet_solid",
                "coupling_interface": "homogenized_magnet_outer",
                "casing_winding_split": False,
            }
        )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "parastell.original_magnet_envelopes/v1.1.0",
                "raw_h5m_sha256": _sha(dagmc),
                "inventory_gate_pass": True,
                "all_envelopes_close": True,
                "magnet_count": 18,
                "magnets": magnets,
            }
        ),
        encoding="utf-8",
    )
    writeback = tmp_path / "writeback.json"
    writeback.write_text(
        json.dumps(
            {
                "schema": "parastell.parametric_direct90_h5m_writeback/v1.0.0",
                "pass": True,
                "h5m_sha256": _sha(dagmc),
                "semantic_roles_by_global_volume_id": list(
                    builder.EXPECTED_SEMANTIC_ROLES
                ),
                "magnet_identity_by_global_volume_id": [
                    {
                        "magnet_id": f"magnet-{index:04d}",
                        "volume_global_id": index + 9,
                        "unique_match": True,
                        "pass": True,
                    }
                    for index in range(18)
                ],
            }
        ),
        encoding="utf-8",
    )
    return dagmc, inventory, writeback


def _fake_envelopes(_dagmc, requests):
    output = []
    for request in requests:
        index = int(request["volume_id"]) - 9
        surface = SimpleNamespace(
            surface_id=100 + index,
            openmc_normal_sign=(-1 if index % 2 else 1),
            area_cm2=12.0 + index,
            role="plasma_side",
        )
        envelope = SimpleNamespace(
            dagmc_volume_id=request["volume_id"],
            magnet_component=request["magnet_id"],
            surface_ids=(100 + index,),
            surfaces=(surface,),
        )
        output.append(SimpleNamespace(envelope=envelope))
    return tuple(output)


def test_builds_complete_surface_and_normal_manifest(tmp_path, monkeypatch):
    dagmc, inventory, writeback = _inputs(tmp_path)
    monkeypatch.setattr(
        builder, "_extract_envelopes_from_h5m", _fake_envelopes
    )
    result = builder.build_manifest(
        dagmc,
        inventory,
        writeback,
        expected_dagmc_sha256=_sha(dagmc),
        expected_topology_inventory_sha256=_sha(inventory),
        expected_h5m_writeback_sha256=_sha(writeback),
    )
    assert result["status"] == "PASS"
    assert result["magnet_count"] == 18
    assert result["all_surface_ids"] == list(range(100, 118))
    assert (
        result["magnets"][1]["surfaces"][0]["magnet_outward_normal_multiplier"]
        == -1
    )


def test_rejects_unqualified_inventory(tmp_path, monkeypatch):
    dagmc, inventory, writeback = _inputs(tmp_path)
    value = json.loads(inventory.read_text(encoding="utf-8"))
    value["inventory_gate_pass"] = False
    inventory.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        builder, "_extract_envelopes_from_h5m", _fake_envelopes
    )
    with pytest.raises(ValueError, match="qualified all 18"):
        builder.build_manifest(
            dagmc,
            inventory,
            writeback,
            expected_dagmc_sha256=_sha(dagmc),
            expected_topology_inventory_sha256=_sha(inventory),
            expected_h5m_writeback_sha256=_sha(writeback),
        )


def test_rejects_surface_identity_disagreement(tmp_path, monkeypatch):
    dagmc, inventory, writeback = _inputs(tmp_path)

    def wrong(dagmc_path, requests):
        rows = list(_fake_envelopes(dagmc_path, requests))
        rows[0].envelope.surface_ids = (999,)
        return tuple(rows)

    monkeypatch.setattr(builder, "_extract_envelopes_from_h5m", wrong)
    with pytest.raises(ValueError, match="surface sets disagree"):
        builder.build_manifest(
            dagmc,
            inventory,
            writeback,
            expected_dagmc_sha256=_sha(dagmc),
            expected_topology_inventory_sha256=_sha(inventory),
            expected_h5m_writeback_sha256=_sha(writeback),
        )


def test_rejects_unproven_writeback_mapping(tmp_path, monkeypatch):
    dagmc, inventory, writeback = _inputs(tmp_path)
    value = json.loads(writeback.read_text(encoding="utf-8"))
    value["magnet_identity_by_global_volume_id"][0]["unique_match"] = False
    writeback.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        builder, "_extract_envelopes_from_h5m", _fake_envelopes
    )
    with pytest.raises(ValueError, match="mapping is invalid"):
        builder.build_manifest(
            dagmc,
            inventory,
            writeback,
            expected_dagmc_sha256=_sha(dagmc),
            expected_topology_inventory_sha256=_sha(inventory),
            expected_h5m_writeback_sha256=_sha(writeback),
        )
