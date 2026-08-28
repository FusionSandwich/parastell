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
    REQUIRED_THREAD_ENVIRONMENT,
    STEP_SHA256,
    _audit_fragmented_surface_topology,
    _curve_manifold_evidence,
    _expected_volumes,
    _fragment_one_to_one,
    _surface_set_is_connected,
    _validated_step_artifacts,
    _validate_frozen_mesh_controls,
    _validate_thread_environment,
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


def test_fragmented_surface_topology_requires_complete_radial_interfaces():
    class Occ:
        @staticmethod
        def getMass(_dimension, _tag):
            return 1.0

    class Model:
        occ = Occ()

        @staticmethod
        def getEntities(dimension):
            assert dimension == 2
            return [
                *((2, tag) for tag in range(100, 108)),
                *((2, tag) for tag in range(200, 209)),
                *((2, tag) for tag in range(300, 309)),
                (2, 400),
            ]

        @staticmethod
        def getAdjacencies(_dimension, surface_tag):
            if 100 <= surface_tag < 108:
                index = surface_tag - 99
                return [index, index + 1], []
            if 200 <= surface_tag < 209:
                return [surface_tag - 199], []
            if 300 <= surface_tag < 309:
                return [surface_tag - 299], []
            return [9], []

        @staticmethod
        def getBoundingBox(_dimension, surface_tag):
            if 200 <= surface_tag < 209:
                return [1.0, 0.0, -1.0, 2.0, 0.0, 1.0]
            if 300 <= surface_tag < 309:
                return [0.0, 1.0, -1.0, 0.0, 2.0, 1.0]
            return [1.0, 1.0, -1.0, 2.0, 2.0, 1.0]

        @staticmethod
        def getBoundary(dim_tags, **_kwargs):
            dimension, tag = dim_tags[0]
            if dimension == 2:
                if 100 <= tag < 108:
                    radial_index = tag - 99
                    return [
                        (1, 1000 + radial_index),
                        (1, -(2000 + radial_index)),
                    ]
                if 200 <= tag < 209:
                    volume_index = tag - 199
                    curves = [(1, -(1000 + volume_index))]
                    if volume_index > 1:
                        curves.append((1, 999 + volume_index))
                    return curves
                if 300 <= tag < 309:
                    volume_index = tag - 299
                    curves = [(1, 2000 + volume_index)]
                    if volume_index > 1:
                        curves.append((1, -(1999 + volume_index)))
                    return curves
                return [(1, 1009), (1, -2009)]
            volume_index = tag
            result = [(2, 199 + volume_index), (2, 299 + volume_index)]
            if volume_index > 1:
                result.append((2, -(98 + volume_index)))
            if volume_index < 9:
                result.append((2, 99 + volume_index))
            else:
                result.append((2, 400))
            return result

    class Gmsh:
        model = Model()

    evidence = _audit_fragmented_surface_topology(
        Gmsh(), [(3, tag) for tag in range(1, 10)]
    )
    assert evidence["status"] == "PREMESH_OCC_INCIDENCE_AND_MANIFOLD_PASS"
    assert evidence["shared_surface_count"] == 8
    assert evidence["external_surface_count"] == 19
    assert set(evidence["interface_surface_counts"].values()) == {1}
    assert evidence["nonperiodic_external_surface_counts"]["magnets"] == 1
    assert len(evidence["volume_shells"]) == 9
    assert len(evidence["interfaces"]) == 8


def test_fragmented_surface_topology_rejects_nonadjacent_owners():
    class Occ:
        @staticmethod
        def getMass(_dimension, _tag):
            return 1.0

    class Model:
        occ = Occ()

        @staticmethod
        def getEntities(_dimension):
            return [(2, 100)]

        @staticmethod
        def getAdjacencies(_dimension, _surface_tag):
            return [1, 3], []

        @staticmethod
        def getBoundingBox(_dimension, _surface_tag):
            return [1.0, 1.0, -1.0, 2.0, 2.0, 1.0]

        @staticmethod
        def getBoundary(_dim_tags, **_kwargs):
            return [(1, 1)]

    class Gmsh:
        model = Model()

    with pytest.raises(RuntimeError, match="nonadjacent components"):
        _audit_fragmented_surface_topology(
            Gmsh(), [(3, tag) for tag in range(1, 10)]
        )


def test_fragmented_surface_topology_rejects_internal_external_face():
    class Occ:
        @staticmethod
        def getMass(_dimension, _tag):
            return 1.0

    class Model:
        occ = Occ()

        @staticmethod
        def getEntities(_dimension):
            return [(2, 100)]

        @staticmethod
        def getAdjacencies(_dimension, _surface_tag):
            return [2], []

        @staticmethod
        def getBoundingBox(_dimension, _surface_tag):
            return [1.0, 1.0, -1.0, 2.0, 2.0, 1.0]

        @staticmethod
        def getBoundary(_dim_tags, **_kwargs):
            return [(1, 1)]

    class Gmsh:
        model = Model()

    with pytest.raises(RuntimeError, match="nonperiodic external surface"):
        _audit_fragmented_surface_topology(
            Gmsh(), [(3, tag) for tag in range(1, 10)]
        )


def test_surface_connectivity_requires_shared_curve_chain():
    assert _surface_set_is_connected(
        [1, 2, 3], {1: [10, 11], 2: [11, 12], 3: [12, 13]}
    )
    assert not _surface_set_is_connected([1, 2], {1: [10, 11], 2: [12, 13]})


def test_curve_manifold_preserves_valid_repeated_seam_occurrences():
    counts, sums = _curve_manifold_evidence([1], {1: [10, -10]})
    assert counts == {10: 2}
    assert sums == {10: 0}


def test_curve_manifold_rejects_repeated_seam_plus_extra_use():
    with pytest.raises(RuntimeError, match="multiplicity"):
        _curve_manifold_evidence([1, 2], {1: [10, -10], 2: [10]})


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


def test_thread_environment_must_be_frozen(monkeypatch):
    for name, value in REQUIRED_THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    _validate_thread_environment()
    monkeypatch.setenv("OMP_NUM_THREADS", "128")
    with pytest.raises(ValueError, match="thread environment"):
        _validate_thread_environment()


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
