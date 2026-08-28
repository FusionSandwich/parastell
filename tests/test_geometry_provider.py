from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from parastell.geometry_provider import (
    GeometryProvenanceError,
    EXPECTED_LAYER_ORDER,
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
    load_geometry_configuration,
    patch_cells,
    qualify_engineering_frame,
    transform_facet_table,
    triangulate_envelope_boundary,
)


def test_streaming_step_pairwise_audit_matches_complete_contract(tmp_path):
    from scripts.wistell_d_geometry_lane import streaming_step_pairwise_audit

    names = ("a", "b", "c")
    values = {"a": 1.0, "b": 2.0, "c": 3.0}
    loaded = []

    class Shape:
        def __init__(self, name):
            self.name = name

        def intersect(self, other):
            class Intersection:
                def isNull(self):
                    return False

                def Volume(self):
                    return (
                        0.0 if {self_name, other_name} != {"a", "c"} else 0.25
                    )

            self_name = self.name
            other_name = other.name
            return Intersection()

    def loader(path):
        loaded.append(path.name)
        return Shape(path.stem)

    report = streaming_step_pairwise_audit(
        tmp_path, names, tolerance_cm3=1.0e-5, loader=loader
    )

    assert loaded == [
        "a.step",
        "b.step",
        "a.step",
        "c.step",
        "b.step",
        "c.step",
    ]
    assert report["expected_pair_count"] == 3
    assert report["evaluated_pair_count"] == 3
    assert report["boolean_failure_count"] == 0
    assert report["overlap_count"] == 1
    assert report["nonadjacent_overlap_count"] == 1
    assert report["execution"] == "streaming_one_step_pair_at_a_time"
    assert report["native_memory_isolation"] == "one_subprocess_per_pair"


def test_streaming_step_pairwise_audit_isolates_real_step_pairs(tmp_path):
    import cadquery as cq

    from scripts.wistell_d_geometry_lane import streaming_step_pairwise_audit

    cq.exporters.export(
        cq.Workplane("XY").box(1, 1, 1), str(tmp_path / "a.step")
    )
    cq.exporters.export(
        cq.Workplane("XY").box(1, 1, 1).translate((2, 0, 0)),
        str(tmp_path / "b.step"),
    )
    report = streaming_step_pairwise_audit(
        tmp_path, ("a", "b"), tolerance_cm3=1.0e-5
    )

    assert report["evaluated_pair_count"] == 1
    assert report["boolean_failure_count"] == 0
    assert report["overlap_count"] == 0
    assert report["pairs"][0]["intersection_volume_cm3"] == 0.0


def test_build_45_uses_process_isolated_pairwise_audit():
    import inspect

    from scripts.wistell_d_geometry_lane import build_45

    source = inspect.getsource(build_45)
    assert "streaming_step_pairwise_audit(" in source
    assert "pairwise = complete_pairwise_audit(" not in source
    assert "del combined, components, stellarator" in source


def _half_period_receipt() -> dict:
    from scripts.wistell_d_geometry_lane import WISTELL_D_SOURCE_HASHES

    matrix = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "classification": "WISTELL_D_EXACT_HALF_PERIOD_TRANSFORM_PASS",
        "evidence": {
            "nfp": 4,
            "stellarator_symmetric": True,
            "phase_origin_degrees": 0.0,
            "canonical_domain_degrees": [0.0, 45.0],
            "derived_domain_degrees": [45.0, 90.0],
            "source_hashes": dict(WISTELL_D_SOURCE_HASHES),
        },
        "transforms": {
            "half_period_mate": {
                "matrix": matrix,
                "inverse_matrix": matrix,
            }
        },
    }


def test_half_period_transform_rejects_accepted_label_with_wrong_matrix():
    from scripts.wistell_d_geometry_lane import (
        validate_half_period_transform_receipt,
    )

    receipt = _half_period_receipt()
    receipt["transforms"]["half_period_mate"]["matrix"][0][0] = 1.0
    with pytest.raises(ValueError, match="qualified map"):
        validate_half_period_transform_receipt(receipt)


def test_accepted_component_steps_are_rehashed_before_derivation(tmp_path):
    from parastell.geometry_provider import sha256_file
    from scripts.wistell_d_geometry_lane import verify_accepted_component_steps

    step = tmp_path / "chamber.step"
    step.write_bytes(b"accepted")
    manifest = {
        "artifacts": {
            "component_step:chamber": {
                "path": str(step.resolve()),
                "bytes": step.stat().st_size,
                "sha256": sha256_file(step),
            }
        }
    }
    verify_accepted_component_steps(tmp_path, manifest, ("chamber",))
    step.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        verify_accepted_component_steps(tmp_path, manifest, ("chamber",))


def test_derive_90_parent_does_not_load_or_retain_cad_shapes():
    import inspect

    from scripts.wistell_d_geometry_lane import derive_90

    source = inspect.getsource(derive_90)
    assert "derive-step-component" in source
    assert "streaming_step_pairwise_audit(" in source
    assert "load_step_components(" not in source
    assert "complete_pairwise_audit(" not in source
    assert '"transport_eligible": False' in source


def _geometry_build_config() -> dict:
    return {
        "schema": "parastell.geometry_build_config/v1.0.0",
        "device": "WISTELL-D",
        "geometry_input_mode": WISTELL_D_GEOMETRY_INPUT_MODE,
        "inputs": dict(WISTELL_D_SOURCE_HASHES),
        "construction": {
            "wall_s": 1.0,
            "canonical_control_count": 452,
            "control_grid": [16, 31],
            "source_cad_grid": [61, 121],
            "source_extent_degrees": [0.0, 45.0],
            "transport_extent_degrees": [0.0, 90.0],
            "magnet_representation": "continuous_30_cm_magnet_envelope",
            "global_explicit_coils": False,
        },
        "radial_build_cm": {
            "first_wall": 4.0,
            "breeder_base": 25.0,
            "breeder_nwl_span": 30.0,
            "back_wall": 5.0,
            "high_temperature_shield": 20.0,
            "vacuum_vessel": 10.0,
            "low_temperature_shield_minimum": 5.0,
            "magnet_envelope": 30.0,
        },
        "materials": {name: name for name in EXPECTED_LAYER_ORDER},
        "dagmc": {
            label: {
                "min_mesh_size_cm": minimum,
                "max_mesh_size_cm": maximum,
                "algorithm": 1,
            }
            for label, minimum, maximum in (
                ("selected", 5.0, 20.0),
                ("refined", 2.5, 10.0),
            )
        },
        "selected_patch_mapping": {"canonical_control_point_ids": []},
    }


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
        validate_wistell_d_manifest(manifest, require_local_frames=True)

    manifest = _accepted_manifest(tmp_path)
    manifest["local_frames"]["frames"][0]["inverse_transform"][0][0] = 2.0
    with pytest.raises(GeometryProvenanceError, match="do not close"):
        validate_wistell_d_manifest(manifest, require_local_frames=True)


def test_base_cad_acceptance_does_not_require_optional_patch_frames(tmp_path):
    manifest = _accepted_manifest(tmp_path)
    manifest["local_frames"] = {
        "status": "OPTIONAL_DEFERRED_UNTIL_SELECTED_PATCHES_ARE_DECLARED",
        "frame_count": 0,
        "frames": [],
    }
    validate_wistell_d_manifest(manifest)


def test_parametric_geometry_configuration_is_hash_bound_and_fail_closed(
    tmp_path,
):
    config_path = tmp_path / "geometry.json"
    config = _geometry_build_config()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    loaded, resolved = load_geometry_configuration(config_path)
    assert resolved == config_path.resolve()
    assert loaded["construction"]["source_cad_grid"] == [61, 121]
    assert (
        loaded["selected_patch_mapping"]["canonical_control_point_ids"] == []
    )

    config["inputs"]["wout_wistell-d.nc"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="authoritative-input hashes"):
        load_geometry_configuration(config_path)


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
