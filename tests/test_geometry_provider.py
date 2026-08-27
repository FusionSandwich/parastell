from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from parastell.geometry_provider import (
    GeometryProvenanceError,
    KNOWN_EXAMPLE_SOURCE_HASHES,
    WISTELL_D_ACCEPTANCE_SCHEMA,
    WISTELL_D_GEOMETRY_INPUT_MODE,
    WISTELL_D_SOURCE_HASHES,
    canonical_patch_instances,
    complete_pairwise_audit,
    derive_wistell_d_transforms,
    require_complete_pairwise_acceptance,
    validate_wistell_d_manifest,
)


def _accepted_manifest(tmp_path: Path) -> dict:
    pairwise = complete_pairwise_audit(
        {name: name for name in ("a", "b", "c")},
        lambda _left, _right: 0.0,
    )
    return {
        "schema": WISTELL_D_ACCEPTANCE_SCHEMA,
        "geometry_input_mode": WISTELL_D_GEOMETRY_INPUT_MODE,
        "source": {
            "lineage_commit": "398032b8c0b4e7c0459c602f2af1e73b3fca0b9a",
            "hashes": dict(WISTELL_D_SOURCE_HASHES),
        },
        "model": {
            "device": "WISTELL-D",
            "wall_s": 1.0,
            "canonical_control_count": 452,
            "source_cad_grid": [61, 121],
            "toroidal_extent_degrees": 45.0,
            "magnet_representation": "continuous_30_cm_magnet_envelope",
            "global_explicit_coils": False,
        },
        "complete_pairwise_audit": pairwise,
        "artifacts": {
            "source_step": {
                "path": str((tmp_path / "source.step").resolve()),
                "sha256": "a" * 64,
            }
        },
    }


def test_complete_pairwise_audit_rejects_nonadjacent_overlap():
    components = {name: name for name in ("chamber", "lts", "gap", "magnets")}

    def intersection(left, right):
        return 27.0153787948 if {left, right} == {"lts", "magnets"} else 0.0

    report = complete_pairwise_audit(components, intersection)
    assert report["expected_pair_count"] == 6
    assert report["evaluated_pair_count"] == 6
    assert report["nonadjacent_overlap_count"] == 1
    with pytest.raises(GeometryProvenanceError, match="lts-magnets"):
        require_complete_pairwise_acceptance(report)


def test_complete_pairwise_audit_fails_closed_on_boolean_error():
    def intersection(left, right):
        if (left, right) == (1, 3):
            raise RuntimeError("kernel failed")
        return 0.0

    report = complete_pairwise_audit({"a": 1, "b": 2, "c": 3}, intersection)
    assert report["boolean_failure_count"] == 1
    with pytest.raises(GeometryProvenanceError, match="Boolean audit"):
        require_complete_pairwise_acceptance(report)


def test_wistell_transform_is_exact_involution_and_period_generator():
    transforms = derive_wistell_d_transforms(nfp=4, lasym=True)
    mate = transforms["half_period_mate"]
    period = transforms["field_period"]
    expected_mate = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    assert np.allclose(mate.linear, expected_mate, atol=1e-15)
    assert np.isclose(mate.determinant, 1.0)
    assert np.allclose(mate.array @ mate.array, np.eye(4), atol=1e-15)
    assert np.allclose(np.linalg.matrix_power(period.array, 4), np.eye(4), atol=1e-15)

    points = np.array([[14.0, 2.0, 3.0], [8.0, 5.0, -4.0]])
    assert np.allclose(
        mate.inverse.transform_points(mate.transform_points(points)), points
    )


def test_transform_requires_live_stellarator_symmetry():
    with pytest.raises(GeometryProvenanceError, match="stellarator symmetry"):
        derive_wistell_d_transforms(nfp=4, lasym=False)


def test_manifest_rejects_31x61_and_generic_example_hashes(tmp_path):
    manifest = _accepted_manifest(tmp_path)
    manifest["model"]["source_cad_grid"] = [31, 61]
    with pytest.raises(GeometryProvenanceError, match="61x121"):
        validate_wistell_d_manifest(manifest)

    manifest = _accepted_manifest(tmp_path)
    manifest["source"]["hashes"]["wout_wistell-d.nc"] = next(
        iter(KNOWN_EXAMPLE_SOURCE_HASHES)
    )
    with pytest.raises(GeometryProvenanceError, match="generic-example"):
        validate_wistell_d_manifest(manifest)


def test_manifest_rejects_example_path_and_has_no_fallback(tmp_path):
    manifest = _accepted_manifest(tmp_path)
    manifest["artifacts"]["source_step"]["path"] = str(
        (tmp_path / "examples" / "model.step").resolve()
    )
    # A generic path segment is insufficient by itself; the forbidden marker
    # is specifically the known example input path or fingerprint.
    validate_wistell_d_manifest(manifest)

    manifest["source"]["vmec_path"] = "examples/wout_vmec.nc"
    with pytest.raises(GeometryProvenanceError, match="example-derived"):
        validate_wistell_d_manifest(manifest)


def test_canonical_controls_are_not_renumbered_for_symmetry_instances():
    rows = canonical_patch_instances(
        canonical_count=452, instance_ids=("canonical", "mate")
    )
    assert len(rows) == 904
    assert {row["canonical_control_id"] for row in rows} == set(range(452))
    assert {row["symmetry_instance_id"] for row in rows} == {
        "canonical",
        "mate",
    }


def test_manifest_round_trip_is_stable(tmp_path):
    manifest = _accepted_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    reread = json.loads(path.read_text(encoding="utf-8"))
    validate_wistell_d_manifest(reread, required_extent_degrees=45.0)
    mutated = copy.deepcopy(reread)
    mutated["model"]["global_explicit_coils"] = True
    with pytest.raises(GeometryProvenanceError, match="explicit coils"):
        validate_wistell_d_manifest(mutated)
