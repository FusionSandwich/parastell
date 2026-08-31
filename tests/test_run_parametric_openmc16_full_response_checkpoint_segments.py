from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from parastell.openmc_checkpoint_restart_plan import (
    build_openmc_checkpoint_restart_plan,
    write_openmc_checkpoint_restart_plan,
)
from scripts import (
    run_parametric_openmc16_full_response_checkpoint_segments as lane,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_receipt(tmp_path: Path):
    model_xml = tmp_path / "instrumented_model.xml"
    model_xml.write_text("<model/>", encoding="utf-8")
    return {
        "schema": lane.SCHEMA,
        "status": "PASS",
        "openmc_version": "0.16.0",
        "run_mode": "fixed source",
        "claim": "BOUNDED_INFRASTRUCTURE_SMOKE_ONLY",
        "production_run_authorized": False,
        "all_bound_inputs_immutable": True,
        "instrumented_model_xml": {
            "path": str(model_xml),
            "sha256": _sha(model_xml),
            "tallies_instrumented": True,
            "surface_bank_instrumented": True,
            "response_plan_sha256": "c" * 64,
            "surface_manifest_sha256": "b" * 64,
        },
        "receipt_content_sha256": "a" * 64,
    }


def _plan(tmp_path: Path):
    return build_openmc_checkpoint_restart_plan(
        campaign_id="wistell-d-full-response-checkpoint",
        producer="ParaStell",
        openmc_model_receipt=_model_receipt(tmp_path),
        particles_per_batch=1_000,
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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _full_response_receipt(tmp_path: Path) -> tuple[Path, str]:
    model_xml = tmp_path / "instrumented_model.xml"
    model_xml.write_text("<model/>", encoding="utf-8")
    payload = {
        "schema": lane.SCHEMA,
        "status": "PASS",
        "claim": "BOUNDED_INFRASTRUCTURE_SMOKE_ONLY",
        "openmc_version": "0.16.0",
        "run_mode": "fixed source",
        "production_run_authorized": False,
        "all_bound_inputs_immutable": True,
        "instrumented_model_xml": {
            "path": str(model_xml),
            "sha256": _sha(model_xml),
            "tallies_instrumented": True,
            "surface_bank_instrumented": True,
            "response_plan_sha256": "c" * 64,
            "surface_manifest_sha256": "b" * 64,
        },
        "receipt_content_sha256": "a" * 64,
    }
    path = tmp_path / "full_response_receipt.json"
    _write_json(path, payload)
    return path, _sha(path)


def _launch_authorization(tmp_path: Path, plan_path: Path, plan_sha: str):
    payload = {
        "schema": lane.LAUNCH_AUTH_SCHEMA,
        "status": "AUTHORIZED",
        "producer": "ParaStell",
        "launch_authorized": True,
        "production_launch_authorized": False,
        "launch_id": "tester-2026",
        "checkpoint_plan": {
            "path": str(plan_path),
            "sha256": plan_sha,
        },
    }
    payload["receipt_content_sha256"] = lane._canonical_sha(payload)
    path = tmp_path / "launch_auth.json"
    _write_json(path, payload)
    return path, _sha(path)


class FakeSettings:
    def __init__(self) -> None:
        self.particles = 0
        self.batches = 0
        self.seed = 0
        self.statepoint = {"batches": []}
        self.source = "SOURCE"


class FakeModel:
    def __init__(self) -> None:
        self.settings = FakeSettings()
        self.tallies = {"fixed": "wire"}

    def export_to_model_xml(self, path: Path) -> None:
        path.write_text("<model/>", encoding="utf-8")


def _fake_openmc_factory(path_log: list[FakeModel]):
    def _factory(model_xml: Path) -> FakeModel:
        del model_xml
        model = FakeModel()
        path_log.append(model)
        return model

    return _factory


def _fake_run(
    output: Path, threads: int, timeout_seconds: int, statepoint_batches
):
    del threads, timeout_seconds
    for batch in statepoint_batches:
        (output / f"statepoint.{batch}.h5").write_text(
            "statepoint", encoding="utf-8"
        )
    (output / "surface_source.h5").write_text("banks", encoding="utf-8")
    return {"exit_code": 0, "timed_out": False, "output": "ok"}


def _fake_run_without_banks(
    output: Path, threads: int, timeout_seconds: int, statepoint_batches
):
    del threads, timeout_seconds
    for batch in statepoint_batches:
        (output / f"statepoint.{batch}.h5").write_text(
            "statepoint", encoding="utf-8"
        )
    return {"exit_code": 0, "timed_out": False, "output": "ok"}


def _build_inputs(tmp_path: Path):
    plan = _plan(tmp_path)
    plan_path = tmp_path / "checkpoint_plan.json"
    write_openmc_checkpoint_restart_plan(plan_path, plan)
    plan_sha = _sha(plan_path)
    auth_path, auth_sha = _launch_authorization(tmp_path, plan_path, plan_sha)
    smoke_path, smoke_sha = _full_response_receipt(tmp_path)
    return (
        plan,
        plan_path,
        plan_sha,
        auth_path,
        auth_sha,
        smoke_path,
        smoke_sha,
    )


def test_segment_lane_builds_manifest_and_preserves_seed_control(
    monkeypatch, tmp_path
):
    (
        plan,
        plan_path,
        plan_sha,
        auth_path,
        auth_sha,
        smoke_path,
        smoke_sha,
    ) = _build_inputs(tmp_path)
    model_log: list[FakeModel] = []
    monkeypatch.setattr(
        lane, "_load_openmc_model", _fake_openmc_factory(model_log)
    )

    inventory = lane.run_checkpoint_segments(
        full_response_receipt_path=smoke_path,
        expected_full_response_receipt_sha256=smoke_sha,
        checkpoint_plan_path=plan_path,
        expected_checkpoint_plan_sha256=plan_sha,
        launch_authorization_path=auth_path,
        expected_launch_authorization_sha256=auth_sha,
        threads=2,
        timeout_seconds=120,
        run_openmc=_fake_run,
    )

    assert inventory["status"] == "SEGMENTS_COMPLETE"
    assert (
        inventory["restart_strategy"]["segment_restart_mode"]
        == "segmented_independent_seed_runs"
    )
    assert (
        inventory["restart_strategy"]["native_statepoint_continuation_claimed"]
        is False
    )
    total_histories = 11_000
    assert inventory["total_source_histories"] == total_histories
    assert (
        sum(row["source_histories"] for row in inventory["segment_results"])
        == total_histories
    )
    assert sum(
        row["history_weight"] for row in inventory["segment_results"]
    ) == pytest.approx(1.0)

    assert len(model_log) == 3
    assert all(
        model_log[index].settings.seed == 101 + index
        for index, model in enumerate(model_log)
    )

    first = json.loads(
        (
            tmp_path
            / "campaign/segment_001/OPENMC_FULL_RESPONSE_SEGMENT_RECEIPT.json"
        ).read_text(encoding="utf-8")
    )
    assert first["preserved_tallies_and_surface_settings"] is True
    assert first["status"] == "PASS"
    assert (
        first["statepoint_batches"]["observed"]
        == first["statepoint_batches"]["expected"]
    )


def test_existing_segment_reuse_rehashes_bound_artifacts(
    monkeypatch, tmp_path
):
    (
        plan,
        plan_path,
        plan_sha,
        auth_path,
        auth_sha,
        smoke_path,
        smoke_sha,
    ) = _build_inputs(tmp_path)
    monkeypatch.setattr(lane, "_load_openmc_model", _fake_openmc_factory([]))
    lane.run_checkpoint_segments(
        full_response_receipt_path=smoke_path,
        expected_full_response_receipt_sha256=smoke_sha,
        checkpoint_plan_path=plan_path,
        expected_checkpoint_plan_sha256=plan_sha,
        launch_authorization_path=auth_path,
        expected_launch_authorization_sha256=auth_sha,
        run_openmc=_fake_run,
    )

    segment = plan["segments"][0]
    segment_root = Path(segment["artifact_root"])
    receipt_path = segment_root / "OPENMC_FULL_RESPONSE_SEGMENT_RECEIPT.json"
    model_path = tmp_path / "instrumented_model.xml"
    lane._validated_existing_segment_receipt(
        receipt_path,
        segment=segment,
        model_path=model_path,
        model_sha256=_sha(model_path),
        particles_per_batch=1_000,
    )

    (segment_root / "surface_source.h5").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="surface bank hash is invalid"):
        lane._validated_existing_segment_receipt(
            receipt_path,
            segment=segment,
            model_path=model_path,
            model_sha256=_sha(model_path),
            particles_per_batch=1_000,
        )


def test_existing_segment_reuse_revalidates_safety_fields(
    monkeypatch, tmp_path
):
    (
        plan,
        plan_path,
        plan_sha,
        auth_path,
        auth_sha,
        smoke_path,
        smoke_sha,
    ) = _build_inputs(tmp_path)
    monkeypatch.setattr(lane, "_load_openmc_model", _fake_openmc_factory([]))
    lane.run_checkpoint_segments(
        full_response_receipt_path=smoke_path,
        expected_full_response_receipt_sha256=smoke_sha,
        checkpoint_plan_path=plan_path,
        expected_checkpoint_plan_sha256=plan_sha,
        launch_authorization_path=auth_path,
        expected_launch_authorization_sha256=auth_sha,
        run_openmc=_fake_run,
    )

    segment = plan["segments"][0]
    receipt_path = (
        Path(segment["artifact_root"])
        / "OPENMC_FULL_RESPONSE_SEGMENT_RECEIPT.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["run_mode"] = "eigenvalue"
    receipt.pop("receipt_content_sha256")
    receipt["receipt_content_sha256"] = lane._canonical_sha(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    model_path = tmp_path / "instrumented_model.xml"
    with pytest.raises(ValueError, match="not reusable"):
        lane._validated_existing_segment_receipt(
            receipt_path,
            segment=segment,
            model_path=model_path,
            model_sha256=_sha(model_path),
            particles_per_batch=1_000,
        )


def test_segment_lane_requires_surface_bank(monkeypatch, tmp_path):
    (
        plan,
        plan_path,
        plan_sha,
        auth_path,
        auth_sha,
        smoke_path,
        smoke_sha,
    ) = _build_inputs(tmp_path)
    model_log: list[FakeModel] = []
    monkeypatch.setattr(
        lane, "_load_openmc_model", _fake_openmc_factory(model_log)
    )

    inventory = lane.run_checkpoint_segments(
        full_response_receipt_path=smoke_path,
        expected_full_response_receipt_sha256=smoke_sha,
        checkpoint_plan_path=plan_path,
        expected_checkpoint_plan_sha256=plan_sha,
        launch_authorization_path=auth_path,
        expected_launch_authorization_sha256=auth_sha,
        run_openmc=_fake_run_without_banks,
    )

    assert inventory["status"] == "SEGMENTS_INCOMPLETE"
    assert any(
        row["status"] == "BLOCKED_SEGMENT_RESTART_RUN"
        for row in inventory["segment_results"]
    )


def test_segment_lane_requires_launch_authorization_binding(
    monkeypatch, tmp_path
):
    (
        plan,
        plan_path,
        plan_sha,
        auth_path,
        auth_sha,
        smoke_path,
        smoke_sha,
    ) = _build_inputs(tmp_path)
    model_log: list[FakeModel] = []
    monkeypatch.setattr(
        lane, "_load_openmc_model", _fake_openmc_factory(model_log)
    )

    # wrong launch authorization path invalidates hash-bound tie
    wrong_auth = {
        "schema": lane.LAUNCH_AUTH_SCHEMA,
        "status": "AUTHORIZED",
        "producer": "ParaStell",
        "launch_authorized": True,
        "production_launch_authorized": False,
        "launch_id": "wrong",
        "checkpoint_plan": {
            "path": str(tmp_path / "wrong-plan.json"),
            "sha256": plan_sha,
        },
    }
    wrong_auth["receipt_content_sha256"] = lane._canonical_sha(wrong_auth)
    auth_path.write_text(json.dumps(wrong_auth), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="launch authorization is not tied to this checkpoint plan",
    ):
        lane.run_checkpoint_segments(
            full_response_receipt_path=smoke_path,
            expected_full_response_receipt_sha256=smoke_sha,
            checkpoint_plan_path=plan_path,
            expected_checkpoint_plan_sha256=plan_sha,
            launch_authorization_path=auth_path,
            expected_launch_authorization_sha256=_sha(auth_path),
            run_openmc=_fake_run,
        )


def test_segment_lane_runs_one_requested_segment_per_invocation(
    monkeypatch, tmp_path
):
    (
        _plan_payload,
        plan_path,
        plan_sha,
        auth_path,
        auth_sha,
        smoke_path,
        smoke_sha,
    ) = _build_inputs(tmp_path)
    model_log: list[FakeModel] = []
    monkeypatch.setattr(
        lane, "_load_openmc_model", _fake_openmc_factory(model_log)
    )

    first = lane.run_checkpoint_segments(
        full_response_receipt_path=smoke_path,
        expected_full_response_receipt_sha256=smoke_sha,
        checkpoint_plan_path=plan_path,
        expected_checkpoint_plan_sha256=plan_sha,
        launch_authorization_path=auth_path,
        expected_launch_authorization_sha256=auth_sha,
        segment_index=1,
        run_openmc=_fake_run,
    )
    assert first["status"] == "SEGMENTS_PENDING"
    assert first["completed_segment_indices"] == [1]
    assert first["pending_segment_indices"] == [2, 3]
    assert first["aggregate_inventory_written"] is False
    assert len(model_log) == 1

    second = lane.run_checkpoint_segments(
        full_response_receipt_path=smoke_path,
        expected_full_response_receipt_sha256=smoke_sha,
        checkpoint_plan_path=plan_path,
        expected_checkpoint_plan_sha256=plan_sha,
        launch_authorization_path=auth_path,
        expected_launch_authorization_sha256=auth_sha,
        segment_index=2,
        run_openmc=_fake_run,
    )
    assert second["completed_segment_indices"] == [1, 2]
    assert len(model_log) == 2

    final = lane.run_checkpoint_segments(
        full_response_receipt_path=smoke_path,
        expected_full_response_receipt_sha256=smoke_sha,
        checkpoint_plan_path=plan_path,
        expected_checkpoint_plan_sha256=plan_sha,
        launch_authorization_path=auth_path,
        expected_launch_authorization_sha256=auth_sha,
        segment_index=3,
        run_openmc=_fake_run,
    )
    assert final["status"] == "SEGMENTS_COMPLETE"
    assert len(model_log) == 3


def test_segment_lane_walltime_guard_does_not_start_transport(
    monkeypatch, tmp_path
):
    (
        _plan_payload,
        plan_path,
        plan_sha,
        auth_path,
        auth_sha,
        smoke_path,
        smoke_sha,
    ) = _build_inputs(tmp_path)
    model_log: list[FakeModel] = []
    monkeypatch.setattr(
        lane, "_load_openmc_model", _fake_openmc_factory(model_log)
    )

    progress = lane.run_checkpoint_segments(
        full_response_receipt_path=smoke_path,
        expected_full_response_receipt_sha256=smoke_sha,
        checkpoint_plan_path=plan_path,
        expected_checkpoint_plan_sha256=plan_sha,
        launch_authorization_path=auth_path,
        expected_launch_authorization_sha256=auth_sha,
        segment_index=1,
        timeout_seconds=120,
        remaining_walltime_seconds=239,
        run_openmc=_fake_run,
    )

    assert progress["status"] == "WALLTIME_GUARD_STOP"
    assert progress["completed_segment_indices"] == []
    assert progress["pending_segment_indices"] == [1, 2, 3]
    assert progress["required_start_margin_seconds"] == 240
    assert model_log == []
    assert not (
        tmp_path
        / "campaign/segment_001/OPENMC_FULL_RESPONSE_SEGMENT_RECEIPT.json"
    ).exists()


def test_segment_lane_rejects_segment_not_in_plan(tmp_path):
    (
        _plan_payload,
        plan_path,
        plan_sha,
        auth_path,
        auth_sha,
        smoke_path,
        smoke_sha,
    ) = _build_inputs(tmp_path)
    with pytest.raises(ValueError, match="not present"):
        lane.run_checkpoint_segments(
            full_response_receipt_path=smoke_path,
            expected_full_response_receipt_sha256=smoke_sha,
            checkpoint_plan_path=plan_path,
            expected_checkpoint_plan_sha256=plan_sha,
            launch_authorization_path=auth_path,
            expected_launch_authorization_sha256=auth_sha,
            segment_index=4,
            run_openmc=_fake_run,
        )
