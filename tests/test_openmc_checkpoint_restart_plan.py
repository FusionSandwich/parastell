from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from parastell.openmc_checkpoint_restart_plan import (
    SCHEMA,
    build_openmc_checkpoint_restart_plan,
    validate_openmc_checkpoint_restart_plan,
    write_openmc_checkpoint_restart_plan,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(tmp_path: Path):
    model_xml = tmp_path / "model.xml"
    model_xml.write_text("<model/>", encoding="utf-8")
    return {
        "schema": "parastell.parametric_openmc16_full_response_smoke/v1.0.0",
        "status": "PASS",
        "openmc_version": "0.16.0",
        "run_mode": "fixed source",
        "claim": "BOUNDED_INFRASTRUCTURE_SMOKE_ONLY",
        "production_run_authorized": False,
        "all_bound_inputs_immutable": True,
        "instrumented_model_xml": {
            "path": str(model_xml),
            "sha256": _digest(model_xml),
            "tallies_instrumented": True,
            "surface_bank_instrumented": True,
            "response_plan_sha256": "c" * 64,
            "surface_manifest_sha256": "b" * 64,
        },
        "receipt_content_sha256": "a" * 64,
    }


def _build(tmp_path: Path):
    return build_openmc_checkpoint_restart_plan(
        campaign_id="wistell-d-poster-fixed-source-segmented",
        producer="ParaStell",
        openmc_model_receipt=_receipt(tmp_path),
        particles_per_batch=20_000,
        total_batches=11,
        segment_batches=4,
        base_seed=101,
        statepoint_interval_batches=3,
        artifact_root=tmp_path / "campaign",
        stop_signal="SIGTERM",
        requeue_signal="SIGUSR1",
        stop_grace_seconds=120,
        requeue_delay_seconds=240,
        surface_manifest_sha256="b" * 64,
    )


def test_segmented_checkpoint_plan_expresses_restart_contract(tmp_path):
    plan = _build(tmp_path)
    assert plan["schema"] == SCHEMA
    assert plan["run_mode"] == "fixed source"
    assert plan["production_run_authorized"] is False
    assert plan["claim"] == "USER_AUTHORIZATION_REQUIRED_POSTER_CAMPAIGN"
    assert (
        plan["statepoint_restart_strategy"][
            "native_statepoint_continuation_claimed"
        ]
        is False
    )
    assert (
        plan["statepoint_restart_strategy"][
            "cumulative_statepoints_are_checkpoint_evidence"
        ]
        is True
    )
    assert (
        plan["statepoint_restart_strategy"]["segment_restart_mode"]
        == "segmented_independent_seed_runs"
    )
    assert len(plan["segments"]) == 3
    assert plan["surface_bank_strategy"]["requested"] is True
    assert plan["signal_requeue_contract"]["walltime_limit_seconds"] == 28_800
    assert [row["requested_batches"] for row in plan["segments"]] == [4, 4, 3]
    assert [row["seed"] for row in plan["segments"]] == [101, 102, 103]
    assert [row["statepoint_batches"] for row in plan["segments"]] == [
        [3, 4],
        [3, 4],
        [3],
    ]

    source_histories = sum(row["source_histories"] for row in plan["segments"])
    assert source_histories == 11 * 20_000

    validate_openmc_checkpoint_restart_plan(plan)
    assert len(plan["plan_sha256"]) == 64


def test_checkpoint_plan_writer_is_create_only(tmp_path):
    plan = _build(tmp_path)
    destination = tmp_path / "checkpoint-plan.json"

    first = write_openmc_checkpoint_restart_plan(destination, plan)
    with pytest.raises(FileExistsError):
        write_openmc_checkpoint_restart_plan(destination, plan)
    assert first == destination
    assert (tmp_path / "campaign").exists()
    assert (tmp_path / "campaign/segment_001").exists()


def test_writer_rejects_precreated_artifact_root(tmp_path):
    plan = _build(tmp_path)
    destination = tmp_path / "checkpoint-plan.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "campaign").mkdir()
    with pytest.raises(FileExistsError):
        write_openmc_checkpoint_restart_plan(destination, plan)


def test_plan_rejects_run_mode_or_interval_mismatch(tmp_path):
    bad_receipt = _receipt(tmp_path)
    bad_receipt["claim"] = "WIRED_RESULTS"  # not a valid bound claim
    with pytest.raises(ValueError, match="OpenMC model claim"):
        build_openmc_checkpoint_restart_plan(
            campaign_id="bad-claim",
            producer="ParaStell",
            openmc_model_receipt=bad_receipt,
            particles_per_batch=1000,
            total_batches=10,
            segment_batches=2,
            base_seed=1,
        )

    plan = _build(tmp_path)
    with pytest.raises(ValueError, match="interval"):
        plan["run_intent"]["statepoint_interval_batches"] = 999
        validate_openmc_checkpoint_restart_plan(plan)


def test_plan_rejects_sibling_artifact_root_and_walltime_overrun(tmp_path):
    plan = _build(tmp_path)
    plan["segments"][0]["artifact_root"] = str(tmp_path / "campaign-other")
    plan["plan_sha256"] = hashlib.sha256(
        __import__("json")
        .dumps(
            {k: v for k, v in plan.items() if k != "plan_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        .encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="under plan artifact_root"):
        validate_openmc_checkpoint_restart_plan(plan)

    with pytest.raises(ValueError, match="exceeds walltime"):
        build_openmc_checkpoint_restart_plan(
            campaign_id="bad-walltime",
            producer="ParaStell",
            openmc_model_receipt=_receipt(tmp_path),
            particles_per_batch=1000,
            total_batches=2,
            segment_batches=1,
            base_seed=1,
            walltime_limit_seconds=100,
            planned_transport_seconds=100,
            walltime_slack_seconds=1,
        )
