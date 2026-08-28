import inspect
import json
from pathlib import Path

import cadquery as cq
import pytest

import scripts.audit_source_cad_candidate as audit_module
from scripts.audit_source_cad_candidate import _clearance_summary
from scripts.audit_source_cad_candidate import _distance
from scripts.audit_source_cad_candidate import _intersection
from scripts.audit_source_cad_candidate import _pair_identity_contract
from scripts.audit_source_cad_candidate import _remap_receipt_artifact_path
from scripts.audit_source_cad_candidate import _shape_row
from scripts.audit_source_cad_candidate import _verify_manifest_artifacts


def test_audit_exposes_explicit_cross_host_receipt_paths():
    parameters = inspect.signature(audit_module.audit).parameters

    assert "physical_receipt_candidate_dir" in parameters
    assert "physical_receipt_reference_dir" in parameters
    assert parameters["physical_receipt_candidate_dir"].default is None
    assert parameters["physical_receipt_reference_dir"].default is None


def test_receipt_artifact_path_relocation_is_prefix_bounded(tmp_path):
    actual = _remap_receipt_artifact_path(
        "/artifacts/run/measurement.json",
        receipt_artifact_root="/artifacts",
        actual_artifact_root=tmp_path,
    )

    assert actual == tmp_path / "run" / "measurement.json"
    with pytest.raises(ValueError, match="outside"):
        _remap_receipt_artifact_path(
            "/different/run/measurement.json",
            receipt_artifact_root="/artifacts",
            actual_artifact_root=tmp_path,
        )


def test_receipt_artifact_path_relocation_requires_both_roots(tmp_path):
    with pytest.raises(ValueError, match="provided together"):
        _remap_receipt_artifact_path(
            "/artifacts/run/measurement.json",
            receipt_artifact_root="/artifacts",
            actual_artifact_root=None,
        )


def test_receipt_artifact_path_relocation_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="parent traversal"):
        _remap_receipt_artifact_path(
            "/artifacts/../outside.json",
            receipt_artifact_root="/artifacts",
            actual_artifact_root=tmp_path,
        )


def test_receipt_artifact_path_relocation_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="escapes"):
        _remap_receipt_artifact_path(
            "/artifacts/escape/result.json",
            receipt_artifact_root="/artifacts",
            actual_artifact_root=tmp_path,
        )


def test_distance_records_global_witness_points():
    first = cq.Workplane("XY").box(1.0, 1.0, 1.0).val()
    second = (
        cq.Workplane("XY")
        .transformed(offset=(2.0, 0.0, 0.0))
        .box(1.0, 1.0, 1.0)
        .val()
    )
    result = _distance(first, second)
    assert result["status"] == "MEASURED"
    assert result["minimum_distance_cm"] == pytest.approx(1.0)
    assert result["solution_count"] >= 1
    for witness in result["witnesses"]:
        assert witness["second_point_cm"][0] - witness["first_point_cm"][
            0
        ] == pytest.approx(1.0)
        assert witness["recomputed_distance_cm"] == pytest.approx(1.0)
        assert witness["reported_distance_residual_cm"] <= 1.0e-8
        assert witness["distance_reconstruction_pass"] is True
        assert witness["first_point_shape_residual_cm"] <= 1.0e-8
        assert witness["second_point_shape_residual_cm"] <= 1.0e-8
        assert witness["point_membership_pass"] is True


def test_clearance_requires_all_eighteen_successful_distances():
    good = {
        "distance": {
            "status": "MEASURED",
            "minimum_distance_cm": 5.1,
            "solution_count": 1,
            "witnesses": [
                {
                    "first_point_cm": [0.0, 0.0, 0.0],
                    "second_point_cm": [5.1, 0.0, 0.0],
                    "recomputed_distance_cm": 5.1,
                    "reported_distance_residual_cm": 0.0,
                    "distance_reconstruction_pass": True,
                    "first_point_shape_residual_cm": 0.0,
                    "second_point_shape_residual_cm": 0.0,
                    "point_membership_pass": True,
                }
            ],
        }
    }
    rows = [
        {**good, "magnet_id": f"magnet-{index:04d}"} for index in range(18)
    ]
    assert _clearance_summary(rows, 5.0)["clearance_pass"]
    rows[-1] = {
        "magnet_id": "magnet-0017",
        "distance": {"status": "ERROR", "reason": "test"},
    }
    summary = _clearance_summary(rows, 5.0)
    assert not summary["clearance_pass"]
    assert summary["distance_measurement_count"] == 17
    assert summary["distance_error_count"] == 1


def test_clearance_rejects_measured_distance_without_witnesses():
    rows = [
        {
            "magnet_id": f"magnet-{index:04d}",
            "distance": {
                "status": "MEASURED",
                "minimum_distance_cm": 5.1,
                "solution_count": 0,
                "witnesses": [],
            },
        }
        for index in range(18)
    ]
    summary = _clearance_summary(rows, 5.0)
    assert not summary["clearance_pass"]
    assert summary["distance_witness_failure_count"] == 18


def test_clearance_rejects_inconsistent_witness_distance():
    rows = [
        {
            "magnet_id": f"magnet-{index:04d}",
            "distance": {
                "status": "MEASURED",
                "minimum_distance_cm": 5.1,
                "solution_count": 1,
                "witnesses": [
                    {
                        "first_point_cm": [0.0, 0.0, 0.0],
                        "second_point_cm": [5.1, 0.0, 0.0],
                        "recomputed_distance_cm": 5.0,
                        "reported_distance_residual_cm": 0.1,
                        "distance_reconstruction_pass": False,
                        "first_point_shape_residual_cm": 0.0,
                        "second_point_shape_residual_cm": 0.0,
                        "point_membership_pass": True,
                    }
                ],
            },
        }
        for index in range(18)
    ]
    summary = _clearance_summary(rows, 5.0)
    assert not summary["clearance_pass"]
    assert summary["distance_witness_failure_count"] == 18


def test_clearance_rejects_duplicate_magnet_identity():
    rows = [
        {
            "magnet_id": "magnet-0000",
            "distance": {
                "status": "MEASURED",
                "minimum_distance_cm": 5.1,
                "solution_count": 1,
                "witnesses": [
                    {
                        "first_point_cm": [0.0, 0.0, 0.0],
                        "second_point_cm": [5.1, 0.0, 0.0],
                        "recomputed_distance_cm": 5.1,
                        "reported_distance_residual_cm": 0.0,
                        "distance_reconstruction_pass": True,
                        "first_point_shape_residual_cm": 0.0,
                        "second_point_shape_residual_cm": 0.0,
                        "point_membership_pass": True,
                    }
                ],
            },
        }
        for _ in range(18)
    ]

    summary = _clearance_summary(rows, 5.0)

    assert summary["magnet_identity_set_pass"] is False
    assert summary["clearance_pass"] is False


def test_pair_identity_contract_requires_exact_unique_sets():
    components = [
        name.removesuffix(".step") for name in audit_module.COMPONENT_FILES
    ]
    magnets = [f"magnet-{index:04d}" for index in range(18)]
    import itertools

    magnet_component = [
        {"component": component, "magnet_id": magnet}
        for component in components
        for magnet in magnets
    ]
    component_pairs = [
        {"components": list(pair)}
        for pair in itertools.combinations(components, 2)
    ]
    magnet_pairs = [
        {"magnets": list(pair)} for pair in itertools.combinations(magnets, 2)
    ]

    passed = _pair_identity_contract(
        magnet_component, component_pairs, magnet_pairs
    )
    magnet_component[-1] = magnet_component[0]
    failed = _pair_identity_contract(
        magnet_component, component_pairs, magnet_pairs
    )

    assert passed["pass"] is True
    assert passed["magnet_component"]["actual_count"] == 108
    assert passed["component_pair"]["actual_count"] == 15
    assert passed["magnet_pair"]["actual_count"] == 153
    assert failed["pass"] is False


def test_intersection_distinguishes_empty_and_nonempty_results():
    first = cq.Workplane("XY").box(1.0, 1.0, 1.0).val()
    separate = (
        cq.Workplane("XY")
        .transformed(offset=(2.0, 0.0, 0.0))
        .box(1.0, 1.0, 1.0)
        .val()
    )
    overlap = (
        cq.Workplane("XY")
        .transformed(offset=(0.25, 0.0, 0.0))
        .box(1.0, 1.0, 1.0)
        .val()
    )
    empty = _intersection(first, separate)
    common = _intersection(first, overlap)
    assert empty["status"] == "EXACT_AABB_DISJOINT"
    assert empty["volume_cm3"] == 0.0
    assert common["status"] == "MEASURED"
    assert common["volume_cm3"] == pytest.approx(0.75)


def test_valid_solid_uses_closed_shells_not_topods_solid_advisory_flag():
    box = cq.Workplane("XY").box(1.0, 1.0, 1.0).val()
    row = _shape_row(box, identity="box")

    assert row["topological_type"] == "Solid"
    assert row["valid"] is True
    assert row["topods_solid_closed_flag_advisory"] is False
    assert row["shell_count"] == 1
    assert row["shell_closed_flags"] == [True]
    assert row["closed"] is True


def test_exact_positive_vessel_distance_skips_common_volume_boolean():
    first = cq.Workplane("XY").box(1.0, 1.0, 1.0).val()
    second = (
        cq.Workplane("XY")
        .transformed(offset=(2.0, 0.0, 0.0))
        .box(1.0, 1.0, 1.0)
        .val()
    )
    audit_module._WORKER_COMPONENTS = {"vacuum_vessel": first}
    audit_module._WORKER_MAGNETS = [second]
    audit_module._WORKER_INTERSECTION_TOLERANCE_CM3 = 1.0e-6
    audit_module._WORKER_DUPLICATE_DISTANCE_CM = 1.0e-6
    row = audit_module._worker_magnet_component(("vacuum_vessel", 0))
    assert row["status"] == "EXACT_BREP_DISTANCE_DISJOINT"
    assert row["volume_cm3"] == 0.0
    assert row["separation_distance"]["minimum_distance_cm"] == pytest.approx(
        1.0
    )


def test_nonvessel_positive_distance_skips_slow_common_boolean(monkeypatch):
    first = cq.Workplane("XY").box(1.0, 1.0, 1.0).val()
    second = (
        cq.Workplane("XY")
        .transformed(offset=(2.0, 0.0, 0.0))
        .box(1.0, 1.0, 1.0)
        .val()
    )
    audit_module._WORKER_COMPONENTS = {"chamber": first}
    audit_module._WORKER_MAGNETS = [second]
    audit_module._WORKER_INTERSECTION_TOLERANCE_CM3 = 1.0e-6
    audit_module._WORKER_DUPLICATE_DISTANCE_CM = 1.0e-6
    monkeypatch.setattr(
        audit_module,
        "_intersection",
        lambda *args, **kwargs: pytest.fail(
            "positive exact distance still requested common Boolean"
        ),
    )

    row = audit_module._worker_magnet_component(("chamber", 0))

    assert row["status"] == "EXACT_BREP_DISTANCE_DISJOINT"
    assert row["distance"]["status"] == "MEASURED"
    assert row["distance"]["minimum_distance_cm"] == pytest.approx(1.0)
    assert row["distance"]["witnesses_required"] is False


def test_manifest_verification_fails_on_artifact_substitution(tmp_path: Path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("expected", encoding="utf-8")
    manifest = {
        "artifacts": {
            "artifact.txt": {
                "size_bytes": artifact.stat().st_size,
                "sha256": "0" * 64,
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = _verify_manifest_artifacts(tmp_path, "manifest.json")
    assert not result["pass"]
    assert not result["artifacts"][0]["pass"]


def test_manifest_verification_requires_declared_membership(tmp_path: Path):
    from parastell.reference_geometry import sha256_file

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("expected", encoding="utf-8")
    manifest = {
        "artifacts": {
            "artifact.txt": {
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = _verify_manifest_artifacts(
        tmp_path,
        "manifest.json",
        required_artifacts=frozenset({"artifact.txt", "missing.txt"}),
    )
    assert not result["pass"]
    assert result["missing_required_artifacts"] == ["missing.txt"]
