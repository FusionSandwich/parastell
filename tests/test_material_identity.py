from __future__ import annotations

import json

import pytest

from parastell.material_identity import (
    bind_material_domains,
    read_fusion_material_db,
    read_openmc_materials_xml,
)


def test_fusion_material_db_adapter_preserves_mass_basis_and_citation(
    tmp_path,
):
    path = tmp_path / "materials.json"
    path.write_text(
        json.dumps(
            {
                "W": {
                    "comp": {"W182": 0.3, "W184": 0.7},
                    "density": 19.3,
                    "metadata": {"name": "Tungsten", "citation": "fixture"},
                }
            }
        ),
        encoding="utf-8",
    )
    row = read_fusion_material_db(path, temperature_K=293.6)[0]
    assert row["composition_basis"] == "mass_fraction"
    assert row["isotopes"] == {"W182": 0.3, "W184": 0.7}
    assert row["citation"] == "fixture"
    assert len(row["composition_sha256"]) == 64


def test_openmc_material_adapter_preserves_atom_basis(tmp_path):
    path = tmp_path / "materials.xml"
    path.write_text(
        '<materials><material id="7" name="REBCO" temperature="600">'
        '<density value="6.3" units="g/cm3"/>'
        '<nuclide name="B10" ao="0.2"/>'
        '<nuclide name="B11" ao="0.8"/>'
        "</material></materials>",
        encoding="utf-8",
    )
    row = read_openmc_materials_xml(path)[0]
    assert row["material_id"] == "7"
    assert row["composition_basis"] == "atom_fraction"
    assert row["temperature_K"] == 600.0


def test_material_domain_binding_rejects_unknown_material(tmp_path):
    path = tmp_path / "materials.json"
    path.write_text(
        json.dumps(
            {
                "W": {
                    "comp": {"W184": 1.0},
                    "density": 19.3,
                    "metadata": {},
                }
            }
        ),
        encoding="utf-8",
    )
    materials = read_fusion_material_db(path)
    with pytest.raises(ValueError, match="unknown material"):
        bind_material_domains(
            materials,
            {"magnet-0": {"material_id": "REBCO", "volume_cm3": 1.0}},
        )
