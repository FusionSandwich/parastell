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
from scripts.wistell_d_geometry_lane import (
    concatenate_facet_tables,
    control_grid_aliases,
    patch_cells,
    qualify_engineering_frame,
    transform_facet_table,
    triangulate_envelope_boundary,
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
        "local_frames": {
            "status": "PASS",
            "frame_count": 1,
            "frames": [
                {
                    "canonical_control_point_id": 0,
                    "canonical_45_patch_id": "M2:canonical:location:t00:p00",
                    "symmetry_instance_id": "canonical",
                    "transform_id": "identity",
                    "global_90_surface_ids": [1],
                    "global_90_facet_ids": [0, 1],
                    "local_engineering_frame_id": "LEF:canonical:t00:p00",
                    "forward_transform": np.eye(4).tolist(),
                    "inverse_transform": np.eye(4).tolist(),
                    "frame_kind": (
                        "ORTHONORMAL_MAGNET_ALIGNED_TOROIDAL_POLOIDAL_RADIAL"
                    ),
                    "tape_twist_resolved": False,
                }
            ],
        },
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
    assert np.allclose(
        np.linalg.matrix_power(period.array, 4), np.eye(4), atol=1e-15
    )

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


def test_manifest_rejects_empty_or_noninvertible_local_frames(tmp_path):
    manifest = _accepted_manifest(tmp_path)
    manifest["local_frames"] = {}
    with pytest.raises(GeometryProvenanceError, match="local-frame"):
        validate_wistell_d_manifest(manifest)

    manifest = _accepted_manifest(tmp_path)
    manifest["local_frames"]["frames"][0]["inverse_transform"][0][0] = 2.0
    with pytest.raises(GeometryProvenanceError, match="do not close"):
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


def test_canonical_aliases_cover_control_grid_and_patches_partition_cells():
    aliases = {
        alias
        for canonical_id in range(452)
        for alias in control_grid_aliases(canonical_id)
    }
    assert aliases == {
        (toroidal, poloidal)
        for toroidal in range(16)
        for poloidal in range(30)
    }
    claimed = [
        cell for alias in sorted(aliases) for cell in patch_cells(*alias)
    ]
    assert len(claimed) == 60 * 120
    assert len(set(claimed)) == len(claimed)


def test_m2_facet_transform_preserves_area_and_ancestry():
    toroidal = np.linspace(0.0, 1.0, 61)[:, None]
    poloidal = np.linspace(0.0, 2.0 * np.pi, 121)[None, :]
    loci = np.stack(
        (
            np.broadcast_to(toroidal, (61, 121)),
            np.broadcast_to(np.cos(poloidal), (61, 121)),
            np.broadcast_to(np.sin(poloidal), (61, 121)),
        ),
        axis=2,
    )
    expected = np.stack(
        (
            np.zeros((61, 121)),
            np.broadcast_to(np.cos(poloidal), (61, 121)),
            np.broadcast_to(np.sin(poloidal), (61, 121)),
        ),
        axis=2,
    )
    canonical = triangulate_envelope_boundary(
        loci, expected, boundary_role_code=1
    )
    canonical["facet_id"] = np.arange(
        len(canonical["areas_cm2"]), dtype=np.uint32
    )
    canonical["parent_45d_facet_id"] = canonical["facet_id"].copy()
    mate = transform_facet_table(
        canonical,
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    )
    expanded = concatenate_facet_tables((canonical, mate))
    assert np.allclose(mate["areas_cm2"], canonical["areas_cm2"])
    assert np.array_equal(mate["parent_45d_facet_id"], canonical["facet_id"])
    assert np.isclose(
        expanded["areas_cm2"].sum(), 2.0 * canonical["areas_cm2"].sum()
    )


def test_qualified_engineering_frame_is_finite_invertible_and_handed():
    source = {
        "origin_cm": [10.0, 20.0, 30.0],
        "toroidal_unit": [1.0, 0.0, 0.0],
        "poloidal_unit": [0.0, 1.0, 0.0],
        "radially_outward_unit": [0.0, 0.0, 1.0],
        "basis_columns_determinant": 1.0,
    }
    mate = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    frame = qualify_engineering_frame(
        source, mate, frame_id="LEF:mate:t00:p00"
    )
    forward = np.asarray(frame["forward_transform"])
    inverse = np.asarray(frame["inverse_transform"])
    assert np.isfinite(forward).all()
    assert np.isfinite(inverse).all()
    assert np.allclose(forward @ inverse, np.eye(4), atol=1.0e-12)
    assert np.isclose(frame["basis_columns_determinant"], 1.0)
    assert frame["frame_kind"].startswith("ORTHONORMAL_MAGNET_ALIGNED")
    assert frame["tape_twist_resolved"] is False
    assert len(frame["frame_fingerprint_sha256"]) == 64


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
