from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from parastell.magnet_tape_stack import (
    DEFINITION_SCHEMA,
    PROVISIONAL_CLASSIFICATION,
    USER_CLASSIFICATION,
    build_tape_stack_manifest,
    validate_tape_stack_definition,
    validate_tape_stack_manifest,
    write_manifest_create_only,
)
from parastell.material_identity import read_fusion_material_db


def _definition(magnet_id="coil:alpha.17"):
    return {
        "schema": DEFINITION_SCHEMA,
        "stack_id": "user-stack-a",
        "magnet_id": magnet_id,
        "classification": USER_CLASSIFICATION,
        "geometry": {
            "tape_width_mm": "6.5",
            "longitudinal_mode": "finite_segment",
            "model_length_mm": "25",
            "stack_axis": "local_z",
            "reference_plane": "rebco_midplane",
            "layer_order": "below_to_above",
        },
        "material_bindings": {
            "bottom": "mat-bottom",
            "rebco": "mat-rebco",
            "top": "mat-top",
        },
        "layers": [
            {
                "layer_id": "below",
                "role": "substrate",
                "side": "below_rebco",
                "material_binding": "bottom",
                "thickness_um": "37.5",
            },
            {
                "layer_id": "active",
                "role": "rebco_active",
                "side": "rebco",
                "material_binding": "rebco",
                "thickness_um": "1.25",
            },
            {
                "layer_id": "above",
                "role": "stabilizer",
                "side": "above_rebco",
                "material_binding": "top",
                "thickness_um": "12.75",
            },
        ],
        "provenance": {
            "source": "user engineering input revision A",
            "citation": None,
            "provisional": False,
            "design_authority": True,
            "manufacturer_specification": False,
        },
    }


def _materials(tmp_path):
    source = tmp_path / "materials.json"
    source.write_text(
        json.dumps(
            {
                "mat-bottom": {
                    "comp": {"Fe56": 1.0},
                    "density": 8.0,
                    "metadata": {},
                },
                "mat-rebco": {
                    "comp": {"O16": 1.0},
                    "density": 6.3,
                    "metadata": {},
                },
                "mat-top": {
                    "comp": {"Cu63": 1.0},
                    "density": 8.9,
                    "metadata": {},
                },
            }
        ),
        encoding="utf-8",
    )
    return read_fusion_material_db(source)


def _write_definition(tmp_path, value=None):
    path = tmp_path / "stack.json"
    path.write_text(
        json.dumps(_definition() if value is None else value),
        encoding="utf-8",
    )
    return path


def test_arbitrary_magnet_stack_generates_exact_closed_layer_boxes(tmp_path):
    manifest = build_tape_stack_manifest(
        _write_definition(tmp_path), _materials(tmp_path)
    )
    assert manifest["magnet_id"] == "coil:alpha.17"
    assert manifest["geometry"]["total_thickness_um"] == "51.5"
    assert manifest["geometry"]["closure"] == {
        "contiguous": True,
        "gaps_cm": "0",
        "overlaps_cm": "0",
        "cumulative_decimal_arithmetic": True,
    }
    assert (
        manifest["layers"][0]["exact_z_bounds_cm"][1]
        == manifest["layers"][1]["exact_z_bounds_cm"][0]
    )
    assert manifest["layers"][1]["exact_z_bounds_cm"] == [
        "-0.0000625",
        "0.0000625",
    ]
    assert manifest["rebco_focus"]["recommended_distinct_tally_regions"] == [
        "below",
        "active",
        "above",
    ]
    assert manifest["local_coordinate_system"]["right_handed"] is True
    assert manifest["claims"]["global_geometry_mutated"] is False
    assert manifest["qualification"]["transport"] == "NOT_RUN"
    assert validate_tape_stack_manifest(manifest) == manifest


def test_material_bindings_are_exact_and_source_hash_bound(tmp_path):
    materials = _materials(tmp_path)
    path = _write_definition(tmp_path)
    with pytest.raises(ValueError, match="exactly match"):
        build_tape_stack_manifest(path, materials[:-1])
    materials[0]["density_g_cm3"] = 99.0
    with pytest.raises(ValueError, match="composition hash mismatch"):
        build_tape_stack_manifest(path, materials)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda value: value.update(magnet_id="bad/id"), "magnet_id"),
        (
            lambda value: value["layers"].__setitem__(
                1, {**value["layers"][1], "side": "above_rebco"}
            ),
            "exactly one REBCO",
        ),
        (
            lambda value: value["layers"].__setitem__(
                0, {**value["layers"][0], "side": "above_rebco"}
            ),
            "explicitly bracket",
        ),
        (
            lambda value: value["layers"][0].update(thickness_um="0"),
            "positive and finite",
        ),
        (
            lambda value: value["material_bindings"].update(unused="unused"),
            "used exactly",
        ),
    ],
)
def test_definition_rejects_ambiguous_or_nonphysical_inputs(change, match):
    value = _definition()
    change(value)
    with pytest.raises(ValueError, match=match):
        validate_tape_stack_definition(value)


def test_provisional_163p2um_example_is_explicit_and_not_design_truth():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "examples"
        / "provisional_rebco_tape_stack_163p2um.json"
    )
    value = validate_tape_stack_definition(
        json.loads(path.read_text(encoding="utf-8"))
    )
    assert value["classification"] == PROVISIONAL_CLASSIFICATION
    assert value["provenance"]["provisional"] is True
    assert value["provenance"]["design_authority"] is False
    assert value["provenance"]["manufacturer_specification"] is False
    assert sum(float(row["thickness_um"]) for row in value["layers"]) == 163.2
    assert value["geometry"]["tape_width_mm"] == "4"


def test_manifest_is_self_sealed_and_output_is_create_only(tmp_path):
    manifest = build_tape_stack_manifest(
        _write_definition(tmp_path), _materials(tmp_path)
    )
    output = tmp_path / "output" / "manifest.json"
    write_manifest_create_only(output, manifest)
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError):
        write_manifest_create_only(output, manifest)

    changed = copy.deepcopy(manifest)
    changed["layers"][0]["volume_cm3"] *= 2
    with pytest.raises(ValueError, match="seal or geometry"):
        validate_tape_stack_manifest(changed)


def test_manifest_rejects_definition_mutation_after_generation(tmp_path):
    definition_path = _write_definition(tmp_path)
    manifest = build_tape_stack_manifest(definition_path, _materials(tmp_path))
    definition_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="definition binding changed"):
        validate_tape_stack_manifest(manifest)
