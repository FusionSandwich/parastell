import hashlib
import json

import pytest

from parastell.material_manifest import deterministic_component_colors
from parastell.material_manifest import resolve_material_manifest


def write_configuration(tmp_path):
    generated = tmp_path / "generated.json"
    generated.write_text(
        json.dumps(
            {
                "Cu": {
                    "density": 8.96,
                    "comp": {"Cu63": 0.7, "Cu65": 0.3},
                    "metadata": {"citation": "fixture"},
                },
                "Steel": {
                    "density": 8.0,
                    "comp": {"Fe56": 1.0},
                    "metadata": {"citation": "fixture"},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(generated.read_bytes()).hexdigest()
    config = tmp_path / "materials.json"
    config.write_text(
        json.dumps(
            {
                "fusion_material_db": {
                    "repository": "owner/repo",
                    "commit": "a" * 40,
                },
                "generated_artifacts": [
                    {
                        "path": str(generated),
                        "sha256": digest,
                        "selected_materials": ["Cu", "Steel"],
                    }
                ],
                "winding_pack_variants": {
                    "PackReference": {
                        "fraction_basis": "volume",
                        "constituent_volume_fractions": {
                            "Cu": 0.5,
                            "Steel": 0.5,
                        },
                        "source": "explicit_project_assumption",
                    }
                },
                "material_tags": {
                    "winding_pack": "PackReference",
                    "casing": "Steel",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config, generated


def test_pinned_material_manifest_and_mixture_are_reproducible(tmp_path):
    config, generated = write_configuration(tmp_path)
    first = resolve_material_manifest(config)
    second = resolve_material_manifest(config)
    assert (
        first["resolved_manifest_sha256"] == second["resolved_manifest_sha256"]
    )
    pack = first["materials"]["PackReference"]
    assert pack["density_g_cm3"] == pytest.approx(8.48)
    assert sum(pack["nuclides"].values()) == pytest.approx(1.0)
    assert not pack["metadata"]["manufacturer_authoritative"]
    generated.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_material_manifest(config)


def test_deterministic_colors_do_not_use_random_assignment():
    first = deterministic_component_colors(["blanket", "winding_pack"])
    second = deterministic_component_colors(["winding_pack", "blanket"])
    assert first == second
    assert all(
        value.startswith("#") and len(value) == 7 for value in first.values()
    )
