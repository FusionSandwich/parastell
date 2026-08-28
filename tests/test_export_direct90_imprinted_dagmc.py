from __future__ import annotations

import argparse
import math
import hashlib

import pytest

from scripts.export_direct90_imprinted_dagmc import (
    COARSE_MAX_MESH_SIZE_CM,
    COARSE_MESH_ALGORITHM,
    COARSE_MIN_MESH_SIZE_CM,
    COARSE_THREADS,
    COMPONENT_ORDER,
    MATERIAL_TAGS,
    STEP_SHA256,
    _expected_volumes,
    _fragment_one_to_one,
    _validated_step_artifacts,
    _validate_frozen_mesh_controls,
    _unique_volume_assignment,
    export,
)


def test_volume_assignment_restores_component_order():
    expected = {
        name: float(index + 1) for index, name in enumerate(COMPONENT_ORDER)
    }
    imported = [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0]
    indices, evidence = _unique_volume_assignment(imported, expected)
    assert [imported[index] for index in indices] == list(expected.values())
    assert [row["component"] for row in evidence] == list(COMPONENT_ORDER)


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0])
def test_volume_assignment_rejects_invalid_import_mass(bad):
    expected = {
        name: float(index + 1) for index, name in enumerate(COMPONENT_ORDER)
    }
    imported = list(expected.values())
    imported[0] = bad
    with pytest.raises(RuntimeError, match="non-finite or non-positive"):
        _unique_volume_assignment(imported, expected)


def test_volume_assignment_rejects_ambiguous_match():
    expected = {
        name: float(index + 1) for index, name in enumerate(COMPONENT_ORDER)
    }
    imported = list(expected.values())
    imported[1] = imported[0]
    with pytest.raises(RuntimeError, match="volume matches"):
        _unique_volume_assignment(imported, expected)


def test_step_artifacts_reject_substitution(tmp_path, monkeypatch):
    steps = {name: tmp_path / f"{name}.step" for name in COMPONENT_ORDER}
    for index, path in enumerate(steps.values()):
        path.write_bytes(f"step-{index}".encode())
    monkeypatch.setattr(
        "scripts.export_direct90_imprinted_dagmc.STEP_SHA256",
        {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in steps.items()
        },
    )
    _validated_step_artifacts(steps)
    steps["breeder"].write_bytes(b"substituted")
    with pytest.raises(ValueError, match="STEP hash mismatch for breeder"):
        _validated_step_artifacts(steps)


@pytest.mark.parametrize(
    "controls",
    [
        (
            math.nan,
            COARSE_MAX_MESH_SIZE_CM,
            COARSE_MESH_ALGORITHM,
            COARSE_THREADS,
        ),
        (1.0, COARSE_MAX_MESH_SIZE_CM, COARSE_MESH_ALGORITHM, COARSE_THREADS),
        (COARSE_MIN_MESH_SIZE_CM, 1.0, COARSE_MESH_ALGORITHM, COARSE_THREADS),
        (COARSE_MIN_MESH_SIZE_CM, COARSE_MAX_MESH_SIZE_CM, 2, COARSE_THREADS),
        (
            COARSE_MIN_MESH_SIZE_CM,
            COARSE_MAX_MESH_SIZE_CM,
            COARSE_MESH_ALGORITHM,
            31,
        ),
    ],
)
def test_frozen_mesh_controls_reject_changes(controls):
    with pytest.raises(ValueError, match="mesh"):
        _validate_frozen_mesh_controls(*controls)


def test_fragment_mapping_requires_one_output_volume_per_input():
    class Occ:
        @staticmethod
        def fragment(*_args, **_kwargs):
            return [], [[(3, 10)], [(3, 11), (3, 12)]]

        @staticmethod
        def synchronize():
            return None

    class Model:
        occ = Occ()

    class Gmsh:
        model = Model()

    with pytest.raises(RuntimeError, match="maps to 2 fragmented volumes"):
        _fragment_one_to_one(Gmsh(), [(3, 1), (3, 2)])


def test_source_constants_cover_exact_component_order():
    assert tuple(STEP_SHA256) == COMPONENT_ORDER
    assert len(set(STEP_SHA256.values())) == len(COMPONENT_ORDER)
    assert all(len(value) == 64 for value in STEP_SHA256.values())
    assert MATERIAL_TAGS == (
        "Vacuum",
        "first_wall",
        "breeder",
        "back_wall",
        "high_temperature_shield",
        "vacuum_vessel",
        "low_temperature_shield",
        "Vacuum",
        "magnet_envelope",
    )


def test_reference_volume_inventory_rejects_wrong_order():
    rows = [
        {
            "name": name,
            "volume_cm3": index + 1.0,
            "valid": True,
            "solid_count": 1,
        }
        for index, name in enumerate(reversed(COMPONENT_ORDER))
    ]
    with pytest.raises(ValueError, match="component order"):
        _expected_volumes({"cad_validation": {"components": rows}})


def test_export_rejects_existing_output_root_before_reading_source(tmp_path):
    output = tmp_path / "already-exists"
    output.mkdir()
    args = argparse.Namespace(source_root=tmp_path, output_root=output)
    with pytest.raises(FileExistsError, match="create-only"):
        export(args)
