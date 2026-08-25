import hashlib
import json

import pytest

from parastell.source_convergence import (
    CONVERGED_STATUS,
    DT_ENERGY_MOMENTS,
    INCOMPLETE_STATUS,
    MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS,
    NOT_CONVERGED_STATUS,
    REQUIRED_CANDIDATE_SHAPES,
    SOURCE_METRICS,
    SPATIAL_MOMENTS,
    _payload_sha256,
    evaluate_source_mesh_convergence,
    validate_source_mesh_convergence,
    write_source_mesh_convergence,
)


def _hash(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _manifest(path, shape, delta=0.0, *, missing_moment=None):
    scale = 1.0 + delta
    spatial = {
        "mean_x_cm": 100.0 * scale,
        "mean_y_cm": 200.0 * scale,
        "mean_z_cm": 50.0 * scale,
        "mean_x2_cm2": 10_000.0 * scale**2 + 1_000.0,
        "mean_y2_cm2": 40_000.0 * scale**2 + 1_000.0,
        "mean_z2_cm2": 2_500.0 * scale**2 + 500.0,
    }
    if missing_moment:
        spatial.pop(missing_moment)
    return _write_json(
        path,
        {
            "schema": "parastell.magnet_source_stage/v1.0.0",
            "source_mesh": {
                "path": f"source-{'x'.join(map(str, shape))}.h5m",
                "sha256": _hash(f"mesh:{shape}"),
                "size_bytes": 123,
                "shape": list(shape),
                "tetrahedra": 10,
                "physical_source_rate_per_s": 1.0e20 * (1.0 + delta),
            },
            "convergence_observables": {
                "spatial_moments": spatial,
                "dt_energy_moments": {
                    "mean_eV": 14.1e6 * scale,
                    "mean_squared_eV2": (14.1e6 * scale) ** 2 + 1.2e12,
                },
            },
        },
    )


def _response_report(path, delta=0.0, *, include=True):
    responses = {}
    if include:
        responses["magnet-01.neutron_flux"] = {
            "value": 3.0e10 * (1.0 + delta),
            "units": "particle-cm/source",
        }
    return _write_json(
        path,
        {
            "schema": "test.whole_magnet_responses/v1",
            "whole_magnet_responses": responses,
        },
    )


def _candidates(tmp_path, deltas=(0.20, 0.08, 0.005, 0.0), reports=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = []
    for index, (shape, delta) in enumerate(
        zip(REQUIRED_CANDIDATE_SHAPES, deltas)
    ):
        candidate = {
            "source_manifest_path": _manifest(
                tmp_path / f"source-{index}.json", shape, delta
            )
        }
        if reports:
            candidate["response_report_path"] = _response_report(
                tmp_path / f"responses-{index}.json", delta
            )
        result.append(candidate)
    return result


def _source_tolerances(limit=0.01):
    return {
        name: {"maximum_relative_change": limit} for name in SOURCE_METRICS
    }


def test_required_ladder_and_finest_pair_can_establish_source_convergence(
    tmp_path,
):
    result = evaluate_source_mesh_convergence(
        _candidates(tmp_path),
        source_metric_tolerances=_source_tolerances(),
    )

    assert result["overall_status"] == CONVERGED_STATUS
    assert result["default_selection"]["selected_shape"] == [11, 81, 61]
    assert result["default_selection"]["status"] == (
        "SELECTED_FINEST_WITHOUT_COST_METRIC"
    )
    assert result["missing_evidence"] == []
    assert [item["shape"] for item in result["candidates"]] == [
        list(shape) for shape in REQUIRED_CANDIDATE_SHAPES
    ]
    assert len(result["comparisons"]) == 3
    assert result["qualification_pair"] == [
        list(REQUIRED_CANDIDATE_SHAPES[-2]),
        list(REQUIRED_CANDIDATE_SHAPES[-1]),
    ]
    # The coarse trajectory is retained, but only the prescribed finest pair
    # makes the convergence decision.
    assert not result["comparisons"][0]["metrics"][
        "integrated_source_rate_per_s"
    ]["passes"]
    assert result["comparisons"][-1]["metrics"][
        "integrated_source_rate_per_s"
    ]["passes"]


def test_configured_responses_are_read_from_hash_bound_reports(tmp_path):
    result = evaluate_source_mesh_convergence(
        _candidates(tmp_path, reports=True),
        source_metric_tolerances=_source_tolerances(),
        response_metric_tolerances={
            "magnet-01.neutron_flux": {"maximum_relative_change": 0.01}
        },
    )

    assert result["overall_status"] == CONVERGED_STATUS
    assert result["qualification_metric_count"] == len(SOURCE_METRICS) + 1
    candidate = result["candidates"][0]
    assert len(candidate["response_report"]["sha256"]) == 64
    assert (
        candidate["whole_magnet_responses"]["magnet-01.neutron_flux"]["units"]
        == "particle-cm/source"
    )


def test_missing_shape_moment_threshold_or_response_is_incomplete(tmp_path):
    candidates = _candidates(tmp_path)
    assert (
        evaluate_source_mesh_convergence(
            candidates[:-1], source_metric_tolerances=_source_tolerances()
        )["overall_status"]
        == INCOMPLETE_STATUS
    )

    missing_moment = _candidates(tmp_path / "moments")
    path = missing_moment[-1]["source_manifest_path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    del value["convergence_observables"]["spatial_moments"]["mean_x_cm"]
    path.write_text(json.dumps(value), encoding="utf-8")
    moment_result = evaluate_source_mesh_convergence(
        missing_moment, source_metric_tolerances=_source_tolerances()
    )
    assert moment_result["overall_status"] == INCOMPLETE_STATUS
    assert any(
        "spatial.mean_x_cm" in item
        for item in moment_result["missing_evidence"]
    )

    tolerances = _source_tolerances()
    del tolerances["dt_energy.mean_eV"]
    assert (
        evaluate_source_mesh_convergence(
            candidates, source_metric_tolerances=tolerances
        )["overall_status"]
        == INCOMPLETE_STATUS
    )

    response_result = evaluate_source_mesh_convergence(
        candidates,
        source_metric_tolerances=_source_tolerances(),
        response_metric_tolerances={
            "magnet-01.neutron_flux": {"maximum_relative_change": 0.01}
        },
    )
    assert response_result["overall_status"] == INCOMPLETE_STATUS
    assert any(
        "response_report" in item
        for item in response_result["missing_evidence"]
    )


def test_complete_finest_pair_above_threshold_is_not_converged(tmp_path):
    result = evaluate_source_mesh_convergence(
        _candidates(tmp_path, deltas=(0.20, 0.08, 0.0, 0.10)),
        source_metric_tolerances=_source_tolerances(),
    )
    assert result["overall_status"] == NOT_CONVERGED_STATUS
    assert result["missing_evidence"] == []
    assert result["default_selection"]["selected_shape"] is None


def test_declared_hash_shape_and_response_units_fail_closed(tmp_path):
    candidates = _candidates(tmp_path)
    candidates[0]["source_manifest_sha256"] = _hash("wrong")
    with pytest.raises(ValueError, match="does not match"):
        evaluate_source_mesh_convergence(
            candidates, source_metric_tolerances=_source_tolerances()
        )

    bad_shape = _manifest(tmp_path / "bad-shape.json", (3, 9, 9))
    value = json.loads(bad_shape.read_text(encoding="utf-8"))
    value["source_mesh"]["shape"] = [4, 10, 10]
    bad_shape.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="required ladder"):
        evaluate_source_mesh_convergence(
            [{"source_manifest_path": bad_shape}],
            source_metric_tolerances=_source_tolerances(),
        )

    bad_moment = _manifest(tmp_path / "bad-moment.json", (3, 9, 9))
    value = json.loads(bad_moment.read_text(encoding="utf-8"))
    value["convergence_observables"]["dt_energy_moments"][
        "mean_squared_eV2"
    ] = 1.0
    bad_moment.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="second energy moment"):
        evaluate_source_mesh_convergence(
            [{"source_manifest_path": bad_moment}],
            source_metric_tolerances=_source_tolerances(),
        )

    response_candidates = _candidates(tmp_path / "units", reports=True)
    last_report = response_candidates[-1]["response_report_path"]
    value = json.loads(last_report.read_text(encoding="utf-8"))
    value["whole_magnet_responses"]["magnet-01.neutron_flux"]["units"] = "W"
    last_report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="changed units"):
        evaluate_source_mesh_convergence(
            response_candidates,
            source_metric_tolerances=_source_tolerances(),
            response_metric_tolerances={
                "magnet-01.neutron_flux": {"maximum_relative_change": 0.01}
            },
        )


def test_neutral_json_round_trip_rejects_tampering_and_verifies_bound_files(
    tmp_path,
):
    output = tmp_path / "source-convergence.json"
    written = write_source_mesh_convergence(
        output,
        _candidates(tmp_path),
        source_metric_tolerances=_source_tolerances(),
    )
    summary = validate_source_mesh_convergence(output, verify_bound_files=True)
    assert summary["evidence_sha256"] == written["evidence_sha256"]
    assert summary["candidate_count"] == 4
    assert summary["overall_status"] == CONVERGED_STATUS

    value = json.loads(output.read_text(encoding="utf-8"))
    value["comparisons"][-1]["metrics"]["dt_energy.mean_eV"][
        "fine_value"
    ] = 1.0
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence hash"):
        validate_source_mesh_convergence(output)

    value["evidence_sha256"] = _payload_sha256(value)
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="comparisons are inconsistent"):
        validate_source_mesh_convergence(output)

    # Restore the valid artifact, then demonstrate optional provenance checks.
    write_source_mesh_convergence(
        output,
        _candidates(tmp_path / "fresh"),
        source_metric_tolerances=_source_tolerances(),
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    bound_manifest = artifact["candidates"][0]["source_manifest"]["path"]
    with open(bound_manifest, "a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="bound source_manifest .* changed"):
        validate_source_mesh_convergence(output, verify_bound_files=True)


def test_required_metric_inventories_are_explicit():
    assert SOURCE_METRICS == (
        "integrated_source_rate_per_s",
        *(f"spatial.{name}" for name in SPATIAL_MOMENTS),
        *(f"dt_energy.{name}" for name in DT_ENERGY_MOMENTS),
    )
    assert MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS == (
        "whole_magnet.incoming_neutron_current",
        "whole_magnet.incoming_photon_current",
        "whole_magnet.neutron_flux",
        "whole_magnet.photon_flux",
        "whole_magnet.heating",
        "whole_magnet.hotspot_location",
        "computational_cost",
    )


def test_bound_file_verification_reconstructs_serialized_candidates(tmp_path):
    output = tmp_path / "source-convergence.json"
    write_source_mesh_convergence(
        output,
        _candidates(tmp_path),
        source_metric_tolerances=_source_tolerances(),
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    value["candidates"][0]["source_manifest"]["schema"] = "forged.schema"
    value["evidence_sha256"] = _payload_sha256(value)
    output.write_text(json.dumps(value), encoding="utf-8")

    validate_source_mesh_convergence(output)
    with pytest.raises(ValueError, match="disagrees with bound files"):
        validate_source_mesh_convergence(output, verify_bound_files=True)


def test_hotspot_location_rejects_scalar_or_unidentified_evidence(tmp_path):
    candidates = _candidates(tmp_path, reports=True)
    for candidate in candidates:
        path = candidate["response_report_path"]
        value = json.loads(path.read_text(encoding="utf-8"))
        value["whole_magnet_responses"]["whole_magnet.hotspot_location"] = {
            "value": 1.0,
            "units": "cm",
        }
        path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="requires value_xyz_cm"):
        evaluate_source_mesh_convergence(
            candidates,
            source_metric_tolerances=_source_tolerances(),
            response_metric_tolerances={
                "whole_magnet.hotspot_location": {
                    "maximum_absolute_change": 1.0
                }
            },
            required_response_metric_ids=["whole_magnet.hotspot_location"],
        )
