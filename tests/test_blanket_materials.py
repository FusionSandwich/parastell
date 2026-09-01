from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET

import pytest

from parastell.blanket_materials import (
    COMPONENT_ROLES,
    build_blanket_material_bundle,
    validate_blanket_material_bundle,
    write_blanket_material_bundle,
    write_openmc_materials_xml,
)


def _pure_database(path):
    names = {
        "MF82H",
        "HeNIST",
        "Pb157Li90",
        "HeT410P80",
        "SiC",
        "Be12Ti",
        "Li4SiO4Li60.0",
        "Li2TiO3Li60.0",
        "WC",
        "BMF82H",
        "SS316L",
        "SS316LN",
        "Cr3FS",
        "Water",
        "Cu",
        "Pb",
        "Sn",
        "JK2LBSteel",
        "TernaryNb3Sn",
        "Eins",
        "EUROFER97",
        "W",
    }
    payload = {}
    for index, name in enumerate(sorted(names), start=1):
        payload[name] = {
            "density": float(index),
            "comp": {f"H{index}": 0.25, f"He{index}": 0.75},
            "metadata": {"citation": f"citation-{name}"},
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dcll_bundle_resolves_all_transport_roles(tmp_path):
    source = _pure_database(tmp_path / "pure.json")
    bundle = build_blanket_material_bundle(source, preset="dcll")
    assert [row["material_id"] for row in bundle["materials"]] == list(
        COMPONENT_ROLES
    )
    assert bundle["global_magnet_representation"] == "homogenized"
    assert all(
        row["local_tape_geometry_created"] is False
        for row in bundle["materials"]
    )
    assert (
        bundle["component_recipe_bindings"]["breeder"]
        == "dcll_pb157li90_breeder"
    )


def test_controlled_breeder_and_shield_overrides_do_not_change_geometry(
    tmp_path,
):
    source = _pure_database(tmp_path / "pure.json")
    bundle = build_blanket_material_bundle(
        source,
        preset="dcll",
        role_recipe_overrides={
            "breeder": "hcpb_beryllide_ceramic_breeder",
            "high_temperature_shield": "hcpb_borated_rafs_helium_shield",
        },
    )
    assert bundle["physical_geometry_changed"] is False
    assert bundle["role_recipe_overrides"]["breeder"].startswith("hcpb_")
    assert bundle["transport_or_production_authorized"] is False


def test_bundle_and_openmc_xml_are_create_only_and_element_expanded(tmp_path):
    source = _pure_database(tmp_path / "pure.json")
    bundle = build_blanket_material_bundle(source, preset="hcpb")
    bundle_path = write_blanket_material_bundle(
        tmp_path / "bundle.json", bundle
    )
    xml_path = write_openmc_materials_xml(tmp_path / "materials.xml", bundle)
    root = ET.parse(xml_path).getroot()
    assert [node.get("name") for node in root.findall("material")] == list(
        COMPONENT_ROLES
    )
    assert all(node.findall("nuclide") for node in root.findall("material"))
    assert bundle_path.is_file()
    with pytest.raises(FileExistsError):
        write_openmc_materials_xml(xml_path, bundle)


def test_bundle_rejects_tampering_and_unknown_overrides(tmp_path):
    source = _pure_database(tmp_path / "pure.json")
    bundle = build_blanket_material_bundle(source)
    bad = copy.deepcopy(bundle)
    bad["global_magnet_representation"] = "explicit_tape"
    with pytest.raises(ValueError, match="homogenized"):
        validate_blanket_material_bundle(bad)
    with pytest.raises(ValueError, match="unknown component"):
        build_blanket_material_bundle(
            source, role_recipe_overrides={"port": "dcll_pb157li90_breeder"}
        )
