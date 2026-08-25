import hashlib
import json
import math

import pytest

from parastell.production_statistics import (
    EMPTY_STATUS,
    INCOMPLETE_STATUS,
    QUALIFIED_STATUS,
    UNASSESSED_STATUS,
    UNDER_RESOLVED_STATUS,
    UNSUPPORTED_STATUS,
    _payload_sha256,
    aggregate_production_response_statistics,
    validate_production_statistical_qualification,
    write_production_statistical_qualification,
)


def _hash(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _row(
    response_id,
    seed,
    estimate,
    within,
    *,
    ess=10.0,
    ess_status="AVAILABLE_EXACT_WEIGHTED",
    raw_count=10,
):
    return {
        "variant": "unbiased",
        "is_key_response": True,
        "response_id": response_id,
        "response_definition_sha256": _hash(f"definition:{response_id}"),
        "run_contract_sha256": _hash("one-run-contract"),
        "run_artifact_sha256": _hash(f"statepoint:{seed}"),
        "seed": seed,
        "particle": "neutron",
        "quantity": response_id,
        "units": "per source",
        "normalization": "per_openmc_source_history",
        "estimate": estimate,
        "within_run_std_dev": within,
        "effective_sample_size": ess,
        "effective_sample_size_status": ess_status,
        "raw_count": raw_count,
    }


def _rows(response_id="flux", *, estimates=(1.0, 2.0, 3.0), within=0.1):
    return [
        _row(response_id, seed, estimate, within)
        for seed, estimate in zip((101, 102, 103), estimates)
    ]


def test_three_seed_aggregation_uses_conservative_uncertainty_and_explicit_thresholds():
    rows = _rows()
    result = aggregate_production_response_statistics(
        rows,
        qualification_thresholds={
            "flux": {
                "maximum_relative_standard_error": 0.30,
                "minimum_effective_sample_size": 25.0,
                "minimum_aggregate_raw_count": 25,
            }
        },
    )
    response = result["responses"][0]
    expected_between = 1.0 / math.sqrt(3.0)
    expected_within = math.sqrt(3.0 * 0.1**2) / 3.0
    assert response["mean"] == 2.0
    assert response["observed_between_seed_standard_error"] == pytest.approx(
        expected_between
    )
    assert response["propagated_within_run_standard_error"] == pytest.approx(
        expected_within
    )
    assert response["conservative_standard_error"] == pytest.approx(
        expected_between
    )
    assert response["effective_sample_size"]["value"] == 30.0
    assert response["raw_count"]["value"] == 30
    assert response["resolution_status"] == QUALIFIED_STATUS
    assert result["overall_resolution_status"] == QUALIFIED_STATUS
    reordered = aggregate_production_response_statistics(
        list(reversed(rows)),
        qualification_thresholds=result["qualification_thresholds"],
    )
    assert reordered["evidence_sha256"] == result["evidence_sha256"]


def test_partial_ess_is_not_aggregated_and_required_ess_marks_under_resolved():
    rows = _rows(estimates=(1.0, 1.0, 1.0), within=0.05)
    rows[1]["effective_sample_size"] = None
    rows[1][
        "effective_sample_size_status"
    ] = "UNAVAILABLE_STATEPOINT_BATCH_MOMENTS_ONLY"
    result = aggregate_production_response_statistics(
        rows,
        qualification_thresholds={
            "flux": {
                "maximum_relative_standard_error": 0.1,
                "minimum_effective_sample_size": 20.0,
            }
        },
    )
    response = result["responses"][0]
    assert response["effective_sample_size"] == {
        "status": "PARTIALLY_AVAILABLE_NOT_AGGREGATED",
        "value": None,
        "available_seed_count": 2,
        "seed_count": 3,
        "seed_statuses": [
            "AVAILABLE_EXACT_WEIGHTED",
            "UNAVAILABLE_STATEPOINT_BATCH_MOMENTS_ONLY",
            "AVAILABLE_EXACT_WEIGHTED",
        ],
    }
    assert response["resolution_status"] == UNDER_RESOLVED_STATUS
    ess_check = next(
        check
        for check in response["threshold_checks"]
        if check["metric"] == "effective_sample_size"
    )
    assert ess_check["observed"] is None
    assert not ess_check["passes"]


def test_unavailable_ess_remains_explicit_without_blocking_uncertainty_only_policy():
    rows = _rows(estimates=(1.0, 1.0, 1.0), within=0.05)
    for row in rows:
        row["effective_sample_size"] = None
        row["effective_sample_size_status"] = (
            "UNAVAILABLE_STATEPOINT_BATCH_MOMENTS_ONLY"
        )
    result = aggregate_production_response_statistics(
        rows,
        qualification_thresholds={
            "flux": {"maximum_relative_standard_error": 0.1}
        },
    )
    response = result["responses"][0]
    assert response["effective_sample_size"]["status"] == (
        "UNAVAILABLE_ALL_SEEDS"
    )
    assert response["effective_sample_size"]["value"] is None
    assert response["resolution_status"] == QUALIFIED_STATUS


def test_declared_unsupported_metrics_keep_overall_status_unassessed():
    result = aggregate_production_response_statistics(
        _rows(estimates=(1.0, 1.0, 1.0), within=0.01),
        qualification_thresholds={
            "flux": {"maximum_relative_standard_error": 0.1}
        },
        declared_unassessed_metrics={
            "per_magnet_heating": (
                "The campaign does not yet expand cell-filter bins into "
                "stable per-magnet response IDs."
            )
        },
    )
    assert result["responses"][0]["resolution_status"] == QUALIFIED_STATUS
    assert result["overall_resolution_status"] == INCOMPLETE_STATUS
    assert result["unassessed_metrics"] == [
        {
            "metric_id": "per_magnet_heating",
            "resolution_status": UNSUPPORTED_STATUS,
            "reason": (
                "The campaign does not yet expand cell-filter bins into "
                "stable per-magnet response IDs."
            ),
        }
    ]


def test_empty_and_unassessed_responses_do_not_claim_a_physical_zero_or_pass():
    empty = _rows("empty", estimates=(0.0, 0.0, 0.0), within=0.0)
    for row in empty:
        row["effective_sample_size"] = None
        row["effective_sample_size_status"] = "UNAVAILABLE_NO_EVENTS"
        row["raw_count"] = 0
    unassessed = _rows("unassessed", estimates=(1.0, 1.0, 1.0))
    result = aggregate_production_response_statistics(empty + unassessed)
    by_id = {item["response_id"]: item for item in result["responses"]}
    assert by_id["empty"]["resolution_status"] == EMPTY_STATUS
    assert by_id["empty"]["physical_zero_claimed"] is False
    assert by_id["unassessed"]["resolution_status"] == UNASSESSED_STATUS
    assert result["overall_resolution_status"] == "UNDER_RESOLVED_OR_EMPTY"

    event_evidence = _rows(
        "event-evidence", estimates=(0.0, 0.0, 0.0), within=0.0
    )
    event_result = aggregate_production_response_statistics(event_evidence)
    assert event_result["responses"][0]["empty_estimate"] is False
    assert (
        event_result["responses"][0]["resolution_status"] == UNASSESSED_STATUS
    )


def test_high_relative_uncertainty_is_marked_under_resolved():
    result = aggregate_production_response_statistics(
        _rows(estimates=(0.5, 1.0, 1.5), within=0.5),
        qualification_thresholds={
            "flux": {"maximum_relative_standard_error": 0.1}
        },
    )
    response = result["responses"][0]
    assert response["relative_standard_error"] > 0.1
    assert response["resolution_status"] == UNDER_RESOLVED_STATUS


def test_seed_and_hash_identity_fail_closed():
    duplicate = _rows()
    duplicate[-1]["seed"] = duplicate[-2]["seed"]
    with pytest.raises(ValueError, match="distinct qualification seeds"):
        aggregate_production_response_statistics(duplicate)

    mismatched = _rows() + _rows("heating")
    mismatched[-1]["seed"] = 104
    with pytest.raises(ValueError, match="one seed inventory"):
        aggregate_production_response_statistics(mismatched)

    bad_hash = _rows()
    bad_hash[0]["run_contract_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="run contract"):
        aggregate_production_response_statistics(bad_hash)

    changed_contract = _rows()
    changed_contract[-1]["run_contract_sha256"] = _hash("other contract")
    with pytest.raises(ValueError, match="changed run contract"):
        aggregate_production_response_statistics(changed_contract)

    changed_artifact = _rows() + _rows("heating")
    changed_artifact[-1]["run_artifact_sha256"] = _hash("wrong statepoint")
    with pytest.raises(ValueError, match="run artifact hash"):
        aggregate_production_response_statistics(changed_artifact)


def test_neutral_json_round_trip_and_tamper_rejection(tmp_path):
    path = tmp_path / "production-statistics.json"
    written = write_production_statistical_qualification(
        path,
        _rows(),
        qualification_thresholds={
            "flux": {"maximum_relative_standard_error": 0.5}
        },
    )
    summary = validate_production_statistical_qualification(path)
    assert summary["evidence_sha256"] == written["evidence_sha256"]
    assert summary["response_count"] == 1
    assert summary["seed_count"] == 3

    value = json.loads(path.read_text(encoding="utf-8"))
    value["responses"][0]["mean"] = 99.0
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence hash"):
        validate_production_statistical_qualification(path)

    value["evidence_sha256"] = _payload_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregates are inconsistent"):
        validate_production_statistical_qualification(path)


def test_source_report_verification_rejects_self_consistent_rehashed_tamper(
    tmp_path,
):
    rows = _rows(estimates=(1.0, 1.1, 0.9), within=0.05)
    source_reports = []
    for seed in (101, 102, 103):
        report_rows = [row for row in rows if row["seed"] == seed]
        report_path = tmp_path / f"seed-{seed}.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema": "parastell.unbiased_response_report/v1.0.0",
                    "seed": seed,
                    "run_artifact_sha256": report_rows[0][
                        "run_artifact_sha256"
                    ],
                    "responses": report_rows,
                }
            ),
            encoding="utf-8",
        )
        source_reports.append(
            {
                "path": str(report_path.resolve()),
                "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "size_bytes": report_path.stat().st_size,
                "seed": seed,
                "row_count": 1,
                "run_artifact_sha256": report_rows[0]["run_artifact_sha256"],
            }
        )

    path = tmp_path / "qualification.json"
    write_production_statistical_qualification(
        path,
        rows,
        qualification_thresholds={
            "flux": {"maximum_relative_standard_error": 0.2}
        },
        source_reports=source_reports,
    )
    validate_production_statistical_qualification(
        path, verify_source_reports=True
    )

    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["estimate"] = 9.0
    tampered = aggregate_production_response_statistics(
        tampered_rows,
        qualification_thresholds={
            "flux": {"maximum_relative_standard_error": 0.2}
        },
        source_reports=source_reports,
    )
    path.write_text(json.dumps(tampered), encoding="utf-8")
    validate_production_statistical_qualification(path)
    with pytest.raises(ValueError, match="source rows disagree"):
        validate_production_statistical_qualification(
            path, verify_source_reports=True
        )
