from __future__ import annotations

import hashlib
import json

import pytest

from scripts import run_openmc16_dagmc_surface_qualification as runner


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(dagmc):
    magnets = []
    for index in range(18):
        magnets.append(
            {
                "magnet": f"magnet-{index:04d}",
                "dagmc_volume_id": index + 9,
                "closed": True,
                "coupling_interface": "homogenized_magnet_outer_boundary",
                "surfaces": [
                    {
                        "dagmc_surface_id": index + 100,
                        "magnet_outward_normal_multiplier": (
                            -1 if index % 2 else 1
                        ),
                    }
                ],
            }
        )
    return {
        "schema": "parastell.parametric_magnet_surface_manifest/v1.0.0",
        "status": "PASS",
        "dagmc_sha256": _sha(dagmc),
        "coupling_interface": "homogenized_magnet_outer_boundary",
        "magnet_count": 18,
        "all_envelopes_close": True,
        "physical_h5m_mutation": False,
        "all_surface_ids": list(range(100, 118)),
        "magnets": magnets,
    }


def test_surface_manifest_is_bound_to_exact_h5m(tmp_path):
    dagmc = tmp_path / "dagmc.h5m"
    dagmc.write_bytes(b"geometry-a")
    manifest = tmp_path / "surfaces.json"
    manifest.write_text(json.dumps(_manifest(dagmc)), encoding="utf-8")
    signs = runner._surface_signs(manifest, dagmc)
    assert signs[100] == 1
    assert signs[101] == -1


def test_surface_manifest_rejects_different_h5m(tmp_path):
    dagmc = tmp_path / "dagmc.h5m"
    dagmc.write_bytes(b"geometry-a")
    manifest = tmp_path / "surfaces.json"
    manifest.write_text(json.dumps(_manifest(dagmc)), encoding="utf-8")
    dagmc.write_bytes(b"geometry-b")
    with pytest.raises(ValueError, match="bound to qualified DAGMC"):
        runner._surface_signs(manifest, dagmc)


def test_surface_manifest_rejects_semantic_permutation(tmp_path):
    dagmc = tmp_path / "dagmc.h5m"
    dagmc.write_bytes(b"geometry-a")
    payload = _manifest(dagmc)
    payload["magnets"][0]["magnet"] = "magnet-0001"
    manifest = tmp_path / "surfaces.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity is invalid"):
        runner._surface_signs(manifest, dagmc)
