import hashlib
import json
from pathlib import Path

import pytest

from parastell.magnet_field_cli import STAGE_COMMANDS
from parastell.magnet_field_workflow import STAGE_DEPENDENCIES, STAGES
from parastell.magnet_field_workflow import MagnetRadiationWorkflow
from parastell.magnet_stage_handlers import qualify_production_statistics
from parastell.production_statistics import (
    QUALIFIED_STATUS,
    validate_production_statistical_qualification,
)


def _hash(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _response(seed, estimate):
    return {
        "variant": "unbiased",
        "is_key_response": True,
        "response_id": "magnet_flux",
        "response_definition_sha256": _hash("magnet-flux-definition"),
        "run_contract_sha256": _hash("shared-unbiased-run-contract"),
        "run_artifact_sha256": _hash(f"statepoint-{seed}"),
        "seed": seed,
        "particle": "neutron",
        "quantity": "scalar_flux",
        "units": "particle-cm per source",
        "normalization": "per_openmc_source_history",
        "estimate": estimate,
        "within_run_std_dev": 0.02,
        "effective_sample_size": 20.0,
        "effective_sample_size_status": "AVAILABLE_EXACT_WEIGHTED",
        "raw_count": 50,
    }


def _reports(tmp_path):
    paths = []
    for seed, estimate in ((101, 1.0), (102, 1.05), (103, 0.95)):
        response = _response(seed, estimate)
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "parastell.unbiased_response_report/v1.0.0",
                    "seed": seed,
                    "run_artifact_sha256": response["run_artifact_sha256"],
                    "responses": [response],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _campaign(tmp_path, reports):
    seeds = [
        json.loads(path.read_text(encoding="utf-8"))["seed"]
        for path in reports
    ]
    value = {
        "schema": "parastell.unbiased_qualification_campaign/v1.0.0",
        "status": "PASS",
        "transport": "UNBIASED_ONLY",
        "run_contract": {},
        "run_contract_sha256": _hash("shared-unbiased-run-contract"),
        "seeds": seeds,
        "minimum_seed_count": 3,
        "histories_per_seed": 100,
        "result_reports": [
            {
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": _file_hash(path),
                "size_bytes": path.stat().st_size,
                "run_artifact_sha256": _hash(f"statepoint-{seed}"),
                "evidence_sha256": _hash(f"evidence-{seed}"),
            }
            for seed, path in zip(seeds, reports)
        ],
        "openmc_executable_version": "OpenMC version 0.16.0",
        "run_diagnostics": {
            "lost_particles": 0,
            "dagmc_navigation_failures": 0,
            "runaway_histories": 0,
            "surface_source_capacity_reached": False,
        },
    }
    value["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_gate_i_handler_binds_three_seed_reports_and_is_restart_neutral(
    tmp_path,
):
    reports = _reports(tmp_path)
    campaign = _campaign(tmp_path, reports)
    output = tmp_path / "gate-i.json"
    result = qualify_production_statistics(
        {
            "result_report_paths": [str(path) for path in reports],
            "campaign_report_path": str(campaign),
            "qualification_path": str(output),
            "minimum_seed_count": 3,
            "qualification_thresholds": {
                "magnet_flux": {
                    "maximum_relative_standard_error": 0.1,
                    "minimum_effective_sample_size": 50,
                    "minimum_aggregate_raw_count": 100,
                }
            },
        },
        tmp_path,
    )

    value = json.loads(output.read_text(encoding="utf-8"))
    assert result["gate"] == "I_STATISTICAL_QUALIFICATION"
    assert result["overall_resolution_status"] == QUALIFIED_STATUS
    assert result["source_report_count"] == 3
    assert value["seeds"] == [101, 102, 103]
    assert [item["seed"] for item in value["source_reports"]] == [
        101,
        102,
        103,
    ]
    assert all(
        item["sha256"] == _file_hash(item["path"])
        for item in value["source_reports"]
    )
    assert (
        validate_production_statistical_qualification(
            output, verify_source_reports=True
        )["evidence_sha256"]
        == result["evidence_sha256"]
    )

    reports[0].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source (size|hash) changed"):
        validate_production_statistical_qualification(
            output, verify_source_reports=True
        )


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_gate_i_handler_rejects_a_mixed_seed_report(tmp_path):
    reports = _reports(tmp_path)
    value = json.loads(reports[0].read_text(encoding="utf-8"))
    value["responses"].append(_response(999, 1.0))
    reports[0].write_text(json.dumps(value), encoding="utf-8")
    campaign = _campaign(tmp_path, reports)
    with pytest.raises(ValueError, match="exactly one seed"):
        qualify_production_statistics(
            {
                "result_report_paths": [str(path) for path in reports],
                "campaign_report_path": str(campaign),
                "qualification_path": str(tmp_path / "gate-i.json"),
            },
            tmp_path,
        )


def test_gate_i_stage_and_cli_are_registered_after_unbiased_transport():
    assert "qualify_production_statistics" in STAGES
    assert "run_unbiased_qualification_campaign" in STAGES
    assert STAGE_DEPENDENCIES["run_unbiased_qualification_campaign"] == (
        "prepare_unbiased_model",
    )
    assert STAGE_DEPENDENCIES["qualify_production_statistics"] == (
        "run_unbiased_qualification_campaign",
    )
    assert (
        STAGE_COMMANDS["run-unbiased-campaign"]
        == "run_unbiased_qualification_campaign"
    )
    assert (
        STAGE_COMMANDS["qualify-statistics"] == "qualify_production_statistics"
    )


def test_example_gate_i_policy_is_explicit_and_scope_honest(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("PARASTELL_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("OPENMC_CROSS_SECTIONS", str(tmp_path / "xs.xml"))
    monkeypatch.setenv("PARASTELL_COMMIT", "TEST-COMMIT")
    examples = Path(__file__).parents[1] / "examples"

    smoke = MagnetRadiationWorkflow.from_json(
        examples / "magnet_radiation_smoke.json"
    ).config["stages"]
    production_workflow = MagnetRadiationWorkflow.from_json(
        examples / "magnet_radiation_production.json"
    )
    production = production_workflow.config["stages"]
    production_workflow.validate_configuration()

    for stages in (smoke, production):
        campaign = stages["run_unbiased_qualification_campaign"]
        qualification = stages["qualify_production_statistics"]
        response_ids = {
            response["response_id"] for response in campaign["responses"]
        }
        assert len(response_ids) == 8
        assert set(qualification["qualification_thresholds"]) == response_ids
        assert (
            campaign["unassessed_metrics"]
            == qualification["unassessed_metrics"]
        )
        assert {
            "incoming_surface_current_by_magnet",
            "major_surface_patch_current",
            "per_magnet_flux_and_heating",
            "selected_magnet_incoming_effective_records",
        } == set(qualification["unassessed_metrics"])
        for response in campaign["responses"]:
            thresholds = qualification["qualification_thresholds"][
                response["response_id"]
            ]
            assert "maximum_relative_standard_error" in thresholds
            if response["kind"] == "tally":
                assert "minimum_effective_sample_size" not in thresholds
                assert "minimum_aggregate_raw_count" not in thresholds
            else:
                assert "minimum_effective_sample_size" in thresholds
                assert "minimum_aggregate_raw_count" in thresholds

    policy = production_workflow.config["statistics_thresholds"]
    production_campaign = production["run_unbiased_qualification_campaign"]
    production_thresholds = production["qualify_production_statistics"][
        "qualification_thresholds"
    ]
    for response in production_campaign["responses"]:
        threshold = production_thresholds[response["response_id"]]
        expected_relative = policy[
            f"{response['particle']}_global_maximum_relative_standard_error"
        ]
        assert (
            threshold["maximum_relative_standard_error"] == expected_relative
        )
        if response["kind"] == "surface_bank_weight":
            assert (
                threshold["minimum_effective_sample_size"]
                == policy["recorded_boundary_minimum_effective_sample_size"]
            )
            assert (
                threshold["minimum_aggregate_raw_count"]
                == policy["recorded_boundary_minimum_aggregate_raw_count"]
            )
    assert set(policy["unassessed_metrics"]) == set(
        production["qualify_production_statistics"]["unassessed_metrics"]
    )
