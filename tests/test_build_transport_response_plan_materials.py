from __future__ import annotations

import hashlib
import json

import pytest

from scripts.build_transport_response_plan import _material_derivation


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hash_bound_material_inventory_drives_mt_requests(tmp_path):
    materials = tmp_path / "materials.json"
    materials.write_text(
        json.dumps(
            {
                "materials": [
                    {
                        "material_id": "winding",
                        "composition_sha256": "a" * 64,
                        "isotopes": {"Cu63": 0.7, "Cu65": 0.3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "plan.json"
    control = {
        "material_mt_derivation": {
            "materials": {"path": materials.name, "sha256": _sha(materials)},
            "default_mts": [2, 102],
            "nuclide_overrides": {"Cu63": [16, 103]},
        }
    }
    requests, receipt = _material_derivation(control, config_path)
    assert requests == {"Cu63": [16, 103], "Cu65": [2, 102]}
    assert receipt["material_inventory_sha256"] == _sha(materials)

    control["material_mt_derivation"]["materials"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _material_derivation(control, config_path)
