from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from parastell.geometry_provider import (
    DIRECT45_REFERENCE_MANIFEST_SHA256,
    DIRECT90_DOCKER_IMAGE_ID,
    DIRECT90_REFERENCE_MANIFEST_SHA256,
    DIRECT90_RUNTIME_RECEIPT_SHA256,
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


def test_streaming_pair_audit_persists_create_only_progress(tmp_path):
    from scripts.wistell_d_geometry_lane import streaming_step_pairwise_audit

    progress = tmp_path / "progress"
    report = streaming_step_pairwise_audit(
        tmp_path,
        ("a", "b", "c"),
        tolerance_cm3=1.0e-5,
        loader=lambda path: path.stem,
        progress_dir=progress,
    )
    assert len(list(progress.glob("pair_[0-9][0-9]_*.json"))) == 3
    assert (
        json.loads(
            (progress / "PAIR_AUDIT_SUMMARY.json").read_text(encoding="utf-8")
        )
        == report
    )
    with pytest.raises(FileExistsError, match="create-only pair progress"):
        streaming_step_pairwise_audit(
            tmp_path,
            ("a", "b"),
            tolerance_cm3=1.0e-5,
            loader=lambda path: path.stem,
            progress_dir=progress,
        )


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


def test_streaming_pair_timeout_is_fail_closed(tmp_path, monkeypatch):
    import subprocess

    from scripts.wistell_d_geometry_lane import streaming_step_pairwise_audit

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    report = streaming_step_pairwise_audit(
        tmp_path,
        ("a", "b"),
        tolerance_cm3=1.0e-5,
        pair_timeout_seconds=3.0,
    )
    assert report["boolean_failure_count"] == 1
    assert "TimeoutExpired" in report["pairs"][0]["boolean_error"]


def test_build_45_uses_process_isolated_pairwise_audit():
    import inspect

    from scripts.wistell_d_geometry_lane import build_45

    source = inspect.getsource(build_45)
    assert "streaming_step_pairwise_audit(" in source
    assert "pairwise = complete_pairwise_audit(" not in source
    assert "del combined, components, stellarator" in source
    assert source.index("launch_identity =") < source.index(
        "output_root.mkdir"
    )
    assert 'launch_identity["git_head"]' in source


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


def test_finished_45_degree_cad_derivation_is_disabled():
    from scripts.wistell_d_geometry_lane import derive_90

    with pytest.raises(
        RuntimeError, match="constructed directly by ParaStell"
    ):
        derive_90(None)


def test_direct_90_builder_contains_no_finished_cad_transform_or_fuse():
    import inspect

    from scripts.wistell_d_geometry_lane import build_90_direct

    source = inspect.getsource(build_90_direct)
    assert "np.linspace(0.0, 90.0, DIRECT90_GRID[0])" in source
    assert "streaming_step_pairwise_audit(" in source
    assert ".rotate(" not in source
    assert ".fuse(" not in source
    assert "if not args.omit_combined_step:" in source
    assert '"derived_from_finished_45_degree_CAD": False' in source
    assert '"transport_eligible": False' in source


def test_canonical_values_expand_to_closed_full_period_matrix():
    from scripts.wistell_d_geometry_lane import array_to_full_period_matrix

    values = np.arange(452, dtype=float)
    matrix = array_to_full_period_matrix(values)
    assert matrix.shape == (31, 31)
    np.testing.assert_array_equal(matrix[:, 0], matrix[:, -1])
    np.testing.assert_array_equal(
        matrix[16:],
        np.asarray(
            [
                np.concatenate((row[:1], row[-2:0:-1], row[:1]))
                for row in matrix[:15][::-1]
            ]
        ),
    )


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


def _direct90_manifest(tmp_path: Path) -> dict:
    manifest = _accepted_manifest(tmp_path)
    head_sha = "b" * 40
    manifest["lane"] = {"head_sha_at_build": head_sha}
    manifest["source"]["live_vmec_metadata"] = {
        "nfp": 4,
        "lasym_vmec_logical": False,
        "stellarator_symmetric": True,
    }
    manifest["model"].update(
        {
            "toroidal_extent_degrees": 90.0,
            "source_cad_grid": [80, 90],
            "resolved_control_grid": [80, 90],
            "resolved_matrix_locations_per_layer": 7200,
            "construction_method": "direct_ParaStell_0_to_90_degrees",
            "derived_from_finished_45_degree_CAD": False,
            "component_order": list(EXPECTED_LAYER_ORDER),
            "complete_geometry_representation": "nine_component_step_set",
            "combined_step_exported": False,
        }
    )
    manifest.pop("complete_pairwise_audit")
    manifest["components"] = {
        name: {"volume_cm3": 2.0} for name in EXPECTED_LAYER_ORDER
    }
    manifest["adjacent_pair_audit"] = {
        "audit_scope": "adjacent_radial_pairs_only",
        "component_count": 9,
        "expected_pair_count": 8,
        "evaluated_pair_count": 8,
        "boolean_failure_count": 0,
        "overlap_count": 0,
        "pairs": [
            {
                "left": left,
                "right": right,
                "adjacent_in_radial_stack": True,
                "intersection_volume_cm3": 0.0,
                "boolean_error": None,
                "overlap": False,
            }
            for left, right in zip(
                EXPECTED_LAYER_ORDER[:-1], EXPECTED_LAYER_ORDER[1:]
            )
        ],
    }
    manifest["thickness_validation"] = {
        "all_thicknesses_strictly_positive": True,
        "strict_radial_order_at_all_resolved_locations": True,
    }
    manifest["nonadjacent_separation"] = {
        "native_all_volume_overlap_gate": "REQUIRED_AFTER_IMPRINTED_DAGMC_EXPORT"
    }
    receipt = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "reports"
            / "geometry_recovery"
            / "DIRECT90_DOCKER_RUNTIME_RECEIPT.json"
        ).read_text(encoding="utf-8")
    )
    manifest["container_runtime"] = {
        "image_id": DIRECT90_DOCKER_IMAGE_ID,
        "runtime_receipt": {
            "path": str((tmp_path / "runtime.json").resolve()),
            "sha256": DIRECT90_RUNTIME_RECEIPT_SHA256,
            "schema": receipt["schema"],
        },
        "network": "none",
        "container_execution_proof": "/.dockerenv",
        "python": receipt["python"],
        "modules": receipt["modules"],
        "launch_attestation": {
            "PARASTELL_CONTAINER_IMAGE_ID": DIRECT90_DOCKER_IMAGE_ID,
            "PARASTELL_CONTAINER_NETWORK": "none",
            "PARASTELL_MOUNTED_SOURCE_REVISION": head_sha,
            "PARASTELL_DOCKER_ATTESTATION": "direct90-create-only-v1",
        },
    }
    manifest["cad_periodicity"] = {
        "generated_loci": {
            "pass": True,
            "maximum_residual_cm": 0.0,
            "tolerance_cm": 1.0e-6,
            "surfaces": [{"surface": name} for name in EXPECTED_LAYER_ORDER],
        },
        "boundary_cut_faces": {
            name: {
                "pass": True,
                "phi_0_faces": [{"face_index": 0}],
                "phi_90_faces": [{"face_index": 1}],
                "area_residual_cm2": 0.0,
                "area_tolerance_cm2": 1.0e-6,
            }
            for name in EXPECTED_LAYER_ORDER
        },
    }
    manifest["reference_volume_regression"] = {
        "pass": True,
        "reference_manifest": {
            "path": str((tmp_path / "reference.json").resolve()),
            "sha256": DIRECT90_REFERENCE_MANIFEST_SHA256,
        },
        "components": {
            name: {
                "pass": True,
                "reference_volume_cm3": 1.0,
                "candidate_volume_cm3": 1.0,
                "relative_difference": 0.0,
                "tolerance": 1.0e-7,
            }
            for name in EXPECTED_LAYER_ORDER
        },
    }
    manifest["full_period_physical_measure"] = {
        "pass": True,
        "expected_ratio": 2.0,
        "relative_tolerance": 0.025,
        "half_period_reference_manifest": {
            "path": str((tmp_path / "reference_45.json").resolve()),
            "sha256": DIRECT45_REFERENCE_MANIFEST_SHA256,
        },
        "components": {
            name: {
                "pass": True,
                "volume_45_cm3": 1.0,
                "volume_90_cm3": 2.0,
                "ratio": 2.0,
                "relative_tolerance": 0.025,
            }
            for name in EXPECTED_LAYER_ORDER
        },
    }
    manifest["artifacts"] = {
        f"component_step:{name}": {
            "path": str((tmp_path / f"{name}.step").resolve()),
            "sha256": f"{index + 1:064x}",
        }
        for index, name in enumerate(EXPECTED_LAYER_ORDER)
    }
    return manifest


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


def test_manifest_rejects_90_degree_geometry_without_direct_parastell_build(
    tmp_path,
):
    manifest = _accepted_manifest(tmp_path)
    manifest["model"]["toroidal_extent_degrees"] = 90.0
    manifest["model"]["source_cad_grid"] = [80, 90]
    with pytest.raises(GeometryProvenanceError, match="constructed directly"):
        validate_wistell_d_manifest(manifest)
    manifest = _direct90_manifest(tmp_path)
    validate_wistell_d_manifest(manifest)
    manifest["adjacent_pair_audit"]["pairs"][0]["left"] = "duplicated"
    with pytest.raises(GeometryProvenanceError, match="adjacent radial-pair"):
        validate_wistell_d_manifest(manifest)


@pytest.mark.parametrize(
    "missing_key",
    [
        "container_runtime",
        "cad_periodicity",
        "reference_volume_regression",
        "full_period_physical_measure",
    ],
)
def test_direct90_manifest_requires_all_geometry_proofs(tmp_path, missing_key):
    manifest = _direct90_manifest(tmp_path)
    manifest.pop(missing_key)
    with pytest.raises(GeometryProvenanceError):
        validate_wistell_d_manifest(manifest)


def test_direct90_manifest_requires_live_vmec_proof(tmp_path):
    manifest = _direct90_manifest(tmp_path)
    manifest["source"].pop("live_vmec_metadata")
    with pytest.raises(GeometryProvenanceError, match="live VMEC"):
        validate_wistell_d_manifest(manifest)


def test_direct90_manifest_rejects_half_period_physical_measure(tmp_path):
    manifest = _direct90_manifest(tmp_path)
    row = manifest["full_period_physical_measure"]["components"][
        "magnet_envelope"
    ]
    row.update({"volume_90_cm3": 1.0, "ratio": 1.0, "pass": True})
    with pytest.raises(GeometryProvenanceError, match="physical-measure"):
        validate_wistell_d_manifest(manifest)


def test_direct90_manifest_rejects_forged_stored_volume_ratio(tmp_path):
    manifest = _direct90_manifest(tmp_path)
    row = manifest["full_period_physical_measure"]["components"][
        "magnet_envelope"
    ]
    row.update({"volume_45_cm3": 1.0, "volume_90_cm3": 1.0, "ratio": 2.0})
    with pytest.raises(GeometryProvenanceError, match="physical-measure"):
        validate_wistell_d_manifest(manifest)


def test_direct90_manifest_rejects_nonfinite_volume_evidence(tmp_path):
    manifest = _direct90_manifest(tmp_path)
    manifest["full_period_physical_measure"]["components"]["breeder"][
        "ratio"
    ] = math.nan
    with pytest.raises(GeometryProvenanceError, match="physical-measure"):
        validate_wistell_d_manifest(manifest)


def test_direct90_manifest_binds_ratio_volume_to_component_volume(tmp_path):
    manifest = _direct90_manifest(tmp_path)
    manifest["components"]["vacuum_vessel"]["volume_cm3"] = 3.0
    with pytest.raises(GeometryProvenanceError, match="physical-measure"):
        validate_wistell_d_manifest(manifest)


def test_direct90_manifest_requires_all_nine_component_steps(tmp_path):
    manifest = _direct90_manifest(tmp_path)
    manifest["artifacts"].pop("component_step:magnet_envelope")
    with pytest.raises(GeometryProvenanceError, match="component STEP set"):
        validate_wistell_d_manifest(manifest)


def test_direct90_manifest_rejects_extra_physical_component(tmp_path):
    manifest = _direct90_manifest(tmp_path)
    manifest["artifacts"]["component_step:port"] = {
        "path": str((tmp_path / "port.step").resolve()),
        "sha256": "f" * 64,
    }
    with pytest.raises(GeometryProvenanceError, match="component STEP set"):
        validate_wistell_d_manifest(manifest)


def test_direct90_combined_step_declaration_is_biconditional(tmp_path):
    manifest = _direct90_manifest(tmp_path)
    manifest["artifacts"]["source_step"] = {
        "path": str((tmp_path / "combined.step").resolve()),
        "sha256": "e" * 64,
    }
    with pytest.raises(GeometryProvenanceError, match="contradicts"):
        validate_wistell_d_manifest(manifest)
    manifest["model"]["combined_step_exported"] = True
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


def test_direct90_configuration_is_distinct_from_half_period_configuration():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "wistell_d_direct90_80x90.json"
    )
    config, _ = load_geometry_configuration(path)
    construction = config["construction"]
    assert construction["canonical_control_grid"] == [16, 31]
    assert construction["resolved_control_grid"] == [80, 90]
    assert construction["cad_grid"] == [80, 90]
    assert construction["model_extent_degrees"] == [0.0, 90.0]
    assert construction["derived_from_finished_45_degree_CAD"] is False


def test_direct90_radial_build_reconstructs_7200_ordered_locations(tmp_path):
    from scripts.wistell_d_geometry_lane import (
        load_geometry_only_parastell,
        make_direct90_radial_build,
    )

    np.save(tmp_path / "nwl.npy", np.linspace(1.0, 2.0, 452))
    np.save(tmp_path / "blanket_boundary.npy", np.full(452, 100.0))
    np.save(tmp_path / "magnet_boundary.npy", np.full(452, 120.0))
    load_geometry_only_parastell()
    _, arrays, diagnostics = make_direct90_radial_build(
        tmp_path, _geometry_build_config()
    )
    assert all(values.shape == (80, 90) for values in arrays.values())
    assert diagnostics["resolved_matrix_locations_per_layer"] == 7200
    assert diagnostics["all_thicknesses_strictly_positive"] is True
    assert diagnostics["strict_radial_order_at_all_resolved_locations"] is True
    assert diagnostics[
        "minimum_blanket_to_magnet_clearance_cm"
    ] == pytest.approx(20.0, abs=1.0e-12)
    assert max(diagnostics["full_period_symmetry"].values()) == 0.0
    assert (
        max(
            abs(diagnostics["outer_blanket_residual_cm"]["min"]),
            abs(diagnostics["outer_blanket_residual_cm"]["max"]),
            abs(diagnostics["magnet_boundary_residual_cm"]["min"]),
            abs(diagnostics["magnet_boundary_residual_cm"]["max"]),
        )
        < 1.0e-10
    )


def test_direct90_runtime_receipt_and_environment_must_match(
    tmp_path, monkeypatch
):
    from scripts.wistell_d_geometry_lane import (
        validate_direct90_container_runtime,
    )

    image = DIRECT90_DOCKER_IMAGE_ID
    revision = "b" * 40
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "geometry_recovery"
        / "DIRECT90_DOCKER_RUNTIME_RECEIPT.json"
    )
    container_marker = tmp_path / ".dockerenv"
    container_marker.touch()
    environment = {
        "PARASTELL_CONTAINER_IMAGE_ID": image,
        "PARASTELL_CONTAINER_NETWORK": "none",
        "PARASTELL_MOUNTED_SOURCE_REVISION": revision,
        "PARASTELL_DOCKER_ATTESTATION": "direct90-create-only-v1",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    result = validate_direct90_container_runtime(
        image,
        receipt_path,
        revision,
        container_marker=container_marker,
    )
    assert result["launch_attestation"] == environment
    monkeypatch.setenv("PARASTELL_CONTAINER_NETWORK", "bridge")
    with pytest.raises(RuntimeError, match="in-container launch attestation"):
        validate_direct90_container_runtime(
            image,
            receipt_path,
            revision,
            container_marker=container_marker,
        )


def test_direct90_runtime_rejects_host_execution(monkeypatch, tmp_path):
    from scripts.wistell_d_geometry_lane import (
        validate_direct90_container_runtime,
    )

    revision = "b" * 40
    for name, value in {
        "PARASTELL_CONTAINER_IMAGE_ID": DIRECT90_DOCKER_IMAGE_ID,
        "PARASTELL_CONTAINER_NETWORK": "none",
        "PARASTELL_MOUNTED_SOURCE_REVISION": revision,
        "PARASTELL_DOCKER_ATTESTATION": "direct90-create-only-v1",
    }.items():
        monkeypatch.setenv(name, value)
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "geometry_recovery"
        / "DIRECT90_DOCKER_RUNTIME_RECEIPT.json"
    )
    with pytest.raises(RuntimeError, match="not executing in Docker"):
        validate_direct90_container_runtime(
            DIRECT90_DOCKER_IMAGE_ID,
            receipt_path,
            revision,
            container_marker=tmp_path / "absent-dockerenv",
        )


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
