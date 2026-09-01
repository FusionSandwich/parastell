from __future__ import annotations

import copy
import json

import pytest

from parastell.poster_parametric_cases import (
    build_poster_parametric_case_plan,
    validate_poster_parametric_case_plan,
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
        "Inconel718",
        "FlibeLi60.0",
        "ZrH2",
    }
    payload = {}
    for index, name in enumerate(sorted(names), start=1):
        composition = {f"H{index}": 0.25, f"He{index}": 0.75}
        if name == "W":
            composition = {"W182": 0.265, "W184": 0.735}
        elif name == "BMF82H":
            composition = {"B10": 0.006, "B11": 0.024, "Fe56": 0.97}
        payload[name] = {
            "density": float(index),
            "comp": composition,
            "metadata": {"citation": f"citation-{name}"},
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_p00_p07_are_staged_and_mandatory_cases_resolve(tmp_path):
    plan = build_poster_parametric_case_plan(
        _pure_database(tmp_path / "pure.json")
    )
    assert [row["case_id"] for row in plan["cases"]] == [
        f"P{index:02d}" for index in range(8)
    ]
    assert plan["mandatory_cases_ready"] is True
    assert all(row["source_case_id"] == "P00" for row in plan["cases"][1:])
    assert all(
        row["implicit_cartesian_product"] is False for row in plan["cases"]
    )


def test_material_cases_cover_flibe_hcll_and_type_one(tmp_path):
    plan = build_poster_parametric_case_plan(
        _pure_database(tmp_path / "pure.json")
    )
    cases = {row["case_id"]: row for row in plan["cases"]}
    assert cases["P02"]["material_bundle"]["preset"] == "flibe_lib"
    assert cases["P03"]["material_bundle"]["preset"] == "hcll"
    assert (
        cases["P05"]["material_bundle"]["component_recipe_bindings"][
            "high_temperature_shield"
        ]
        == "natural_boron_w2b5"
    )
    assert (
        cases["P06"]["material_bundle"]["component_recipe_bindings"][
            "low_temperature_shield"
        ]
        == "type_one_rafs_borated_rafs_water_lts"
    )


def test_incomplete_sclv_and_tih2_fail_closed(tmp_path):
    plan = build_poster_parametric_case_plan(
        _pure_database(tmp_path / "pure.json")
    )
    cases = {row["case_id"]: row for row in plan["cases"]}
    for case_id in ("P04", "P07"):
        assert cases[case_id]["status"].startswith("BLOCKED_")
        assert cases[case_id]["blockers"]
        assert "material_bundle" not in cases[case_id]


def test_transport_controls_require_analog_checkpoints_and_authorization(
    tmp_path,
):
    plan = build_poster_parametric_case_plan(
        _pure_database(tmp_path / "pure.json")
    )
    controls = plan["transport_controls"]
    assert controls["weight_windows_enabled"] is False
    assert controls["regular_statepoints"] is True
    assert controls["target_segment_minutes"] == 60
    assert plan["execution_controls"]["large_campaign_authorized"] is False

    bad = copy.deepcopy(plan)
    bad["transport_controls"]["weight_windows_enabled"] = True
    with pytest.raises(ValueError, match="baseline gate"):
        validate_poster_parametric_case_plan(bad)
