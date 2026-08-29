from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from parastell.parametric_source_cad_audit_shards import (
    EXPECTED_COMPONENTS,
    EXPECTED_MAGNET_IDS,
    MERGE_CONTROL_SCHEMA,
    SHARD_RESULT_SCHEMA,
    SHARD_SEAL_SCHEMA,
    SHARD_TERMINAL_SCHEMA,
    assigned_tasks,
    canonical_json_sha256,
    canonical_tasks,
    plan_payload,
    sha256_file,
)
from scripts import merge_parametric_source_cad_audit_shards as merger
from scripts import run_parametric_source_cad_audit_shard as runner

MANIFEST_SHA = "a" * 64
AUDIT_SHA = "b" * 64
SHARD_PRODUCER_SHA = "c" * 64


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _integrity(suffix: str = "") -> list[dict]:
    digest = ("d" if not suffix else "e") * 64
    return [
        {
            "path": "chamber.step",
            "expected_bytes": 3,
            "actual_bytes": 3,
            "expected_sha256": digest,
            "actual_sha256": digest,
            "pass": True,
        }
    ]


def _shape(identity: str) -> dict:
    return {
        "identity": identity,
        "valid": True,
        "closed": True,
        "volume_cm3": 1.0,
    }


def _row(task: dict) -> dict:
    arguments = task["arguments"]
    row = {
        "status": "CONSERVATIVE_AABB_DISJOINT",
        "volume_cm3": 0.0,
        "separation_distance": {"status": "CONSERVATIVE_AABB_DISJOINT"},
        "pass": True,
    }
    if task["stage"] == "component_pairs":
        row["components"] = arguments
    elif task["stage"] == "magnet_component_pairs":
        row["magnet_id"] = arguments[0]
        row["component"] = arguments[1]
        if arguments[1] == "vacuum_gap":
            separation = {
                "status": "MEASURED",
                "minimum_distance_cm": 1.0,
                "witnesses_required": True,
                "solution_count": 1,
                "witnesses": [
                    {
                        "distance_reconstruction_pass": True,
                        "point_membership_pass": True,
                    }
                ],
            }
            row["separation_distance"] = separation
            row["minimum_clearance"] = separation
            row["clearance_witness_pass"] = True
    else:
        row["magnets"] = arguments
    return row


def _packet(tmp_path: Path, shard_count: int = 3):
    plan = plan_payload(
        manifest_sha256=MANIFEST_SHA,
        shard_count=shard_count,
        audit_implementation_sha256=AUDIT_SHA,
        shard_producer_sha256=SHARD_PRODUCER_SHA,
    )
    plan_sha = canonical_json_sha256(plan)
    entries = []
    for index in range(shard_count):
        root = tmp_path / f"shard-{index}"
        root.mkdir()
        tasks = assigned_tasks(plan, index)
        result = {
            "schema": SHARD_RESULT_SCHEMA,
            "status": "PASS",
            "source": f"source-{index}",
            "source_manifest_sha256": MANIFEST_SHA,
            "source_integrity_before": _integrity(),
            "source_integrity_after": _integrity(),
            "source_immutable": True,
            "plan": plan,
            "plan_sha256": plan_sha,
            "shard_index": index,
            "shard_count": shard_count,
            "workers": 2,
            "expected_tasks": tasks,
            "task_records": [
                {"task": task, "row": _row(task)} for task in tasks
            ],
            "component_inventory": [
                _shape(identity) for identity in EXPECTED_COMPONENTS
            ],
            "magnet_inventory": [
                _shape(identity) for identity in EXPECTED_MAGNET_IDS
            ],
            "transport_eligible": False,
        }
        result_path = root / runner.RESULT_NAME
        _write(result_path, result)
        terminal = {
            "schema": SHARD_TERMINAL_SCHEMA,
            "status": "PASS",
            "return_code": 0,
            "error": None,
            "source_immutable": True,
            "source_manifest_sha256": MANIFEST_SHA,
            "plan_sha256": plan_sha,
            "shard_index": index,
            "shard_count": shard_count,
            "automatic_successor_launched": False,
            "result": {
                "filename": runner.RESULT_NAME,
                "sha256": sha256_file(result_path),
            },
        }
        terminal_path = root / runner.TERMINAL_NAME
        _write(terminal_path, terminal)
        seal = {
            "schema": SHARD_SEAL_SCHEMA,
            "status": "PASS",
            "source_manifest_sha256": MANIFEST_SHA,
            "plan_sha256": plan_sha,
            "shard_index": index,
            "shard_count": shard_count,
            "producer_sha256": SHARD_PRODUCER_SHA,
            "audit_implementation_sha256": AUDIT_SHA,
            "terminal_filename": runner.TERMINAL_NAME,
            "terminal_sha256": sha256_file(terminal_path),
            "result_filename": runner.RESULT_NAME,
            "result_sha256": sha256_file(result_path),
            "source_immutable": True,
        }
        seal_path = root / runner.SEAL_NAME
        _write(seal_path, seal)
        entries.append(
            {
                "shard_index": index,
                "seal_path": str(seal_path),
                "seal_sha256": sha256_file(seal_path),
            }
        )
    control = {
        "schema": MERGE_CONTROL_SCHEMA,
        "source_manifest_sha256": MANIFEST_SHA,
        "shard_count": shard_count,
        "audit_implementation_sha256": AUDIT_SHA,
        "shard_producer_sha256": SHARD_PRODUCER_SHA,
        "merge_producer_sha256": sha256_file(Path(merger.__file__)),
        "plan_sha256": plan_sha,
        "shards": entries,
    }
    control_path = tmp_path / "merge-control.json"
    _write(control_path, control)
    return control_path, control


def _reseal(control_path: Path, control: dict, shard_index: int) -> None:
    entry = control["shards"][shard_index]
    seal_path = Path(entry["seal_path"])
    result_path = seal_path.parent / runner.RESULT_NAME
    terminal_path = seal_path.parent / runner.TERMINAL_NAME
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["result"]["sha256"] = sha256_file(result_path)
    _write_replacing(terminal_path, terminal)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["result_sha256"] = sha256_file(result_path)
    seal["terminal_sha256"] = sha256_file(terminal_path)
    _write_replacing(seal_path, seal)
    entry["seal_sha256"] = sha256_file(seal_path)
    _write_replacing(control_path, control)


def _write_replacing(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _merge(tmp_path: Path, control_path: Path) -> tuple[int, dict]:
    output = tmp_path / "merged"
    code = merger.run(
        control_path,
        output,
        expected_control_sha256=sha256_file(control_path),
    )
    terminal = json.loads(
        (output / merger.MERGE_TERMINAL_NAME).read_text(encoding="utf-8")
    )
    return code, terminal


def test_canonical_plan_is_exact_and_disjoint():
    tasks = canonical_tasks()
    assert len(tasks) == 325
    assert [task["canonical_index"] for task in tasks] == list(range(325))
    plan = plan_payload(
        manifest_sha256=MANIFEST_SHA,
        shard_count=7,
        audit_implementation_sha256=AUDIT_SHA,
        shard_producer_sha256=SHARD_PRODUCER_SHA,
    )
    shards = [assigned_tasks(plan, index) for index in range(7)]
    indices = [task["canonical_index"] for shard in shards for task in shard]
    assert sorted(indices) == list(range(325))
    assert len(indices) == len(set(indices))


def test_zero_cad_merge_emits_exporter_compatible_audit_and_seal(tmp_path):
    control_path, _ = _packet(tmp_path)
    code, terminal = _merge(tmp_path, control_path)
    assert code == 0
    assert terminal["cad_imported"] is False
    report = json.loads(
        (tmp_path / "merged" / merger.AUDIT_NAME).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (tmp_path / "merged" / merger.AUDIT_SEAL_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert report["schema"] == "parastell.parametric_source_cad_audit/v1.0.0"
    assert report["status"] == "PASS"
    assert report["summary"]["component_pair_count"] == 28
    assert report["summary"]["magnet_component_pair_count"] == 144
    assert report["summary"]["magnet_pair_count"] == 153
    assert report["summary"]["clearance_witness_pass_count"] == 18
    assert seal["report_sha256"] == sha256_file(
        tmp_path / "merged" / merger.AUDIT_NAME
    )


def test_missing_shard_fails_closed(tmp_path):
    control_path, control = _packet(tmp_path)
    control["shards"].pop()
    _write_replacing(control_path, control)
    code, terminal = _merge(tmp_path, control_path)
    assert code == 2
    assert "every shard" in terminal["error"]


@pytest.mark.parametrize(
    "mutation,error_text",
    [
        ("duplicate", "metadata is invalid"),
        ("unexpected", "identity mismatch"),
        ("malformed", "failed"),
        ("failing", "failed"),
        ("source_drift", "source drift"),
    ],
)
def test_tampered_shard_fails_closed(tmp_path, mutation, error_text):
    control_path, control = _packet(tmp_path)
    result_path = (
        Path(control["shards"][0]["seal_path"]).parent / runner.RESULT_NAME
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        result["task_records"].append(copy.deepcopy(result["task_records"][0]))
    elif mutation == "unexpected":
        result["task_records"][0]["task"]["canonical_index"] = 999
    elif mutation == "malformed":
        result["task_records"][0]["row"].pop("volume_cm3")
    elif mutation == "failing":
        result["task_records"][0]["row"]["pass"] = False
    else:
        result["source_integrity_after"] = _integrity("changed")
    _write_replacing(result_path, result)
    _reseal(control_path, control, 0)
    code, terminal = _merge(tmp_path, control_path)
    assert code == 2
    assert error_text in terminal["error"]


def test_seal_hash_drift_fails_closed(tmp_path):
    control_path, control = _packet(tmp_path)
    control["shards"][0]["seal_sha256"] = "0" * 64
    _write_replacing(control_path, control)
    code, terminal = _merge(tmp_path, control_path)
    assert code == 2
    assert "hash mismatch" in terminal["error"]


def _fake_record(task: dict) -> dict:
    return {"task": task, "row": _row(task)}


def test_shard_runner_seals_success_and_source_drift_failure(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    artifact_rows = _integrity()
    monkeypatch.setattr(
        runner.monolithic,
        "_validate_source",
        lambda *_args, **_kwargs: {
            "components": list(EXPECTED_COMPONENTS),
            "artifacts": copy.deepcopy(artifact_rows),
        },
    )

    def fake_run(tasks, **_kwargs):
        runner.monolithic._COMPONENTS = {
            name: object() for name in EXPECTED_COMPONENTS
        }
        runner.monolithic._MAGNETS = [object() for _ in EXPECTED_MAGNET_IDS]
        return [_fake_record(task) for task in tasks]

    monkeypatch.setattr(runner, "_run_assigned_tasks", fake_run)
    monkeypatch.setattr(
        runner.monolithic,
        "_shape_row",
        lambda _value, identity: _shape(identity),
    )
    producer_sha = sha256_file(Path(runner.__file__))
    audit_sha = sha256_file(Path(runner.monolithic.__file__))
    output = tmp_path / "shard-success"
    assert (
        runner.run(
            source,
            output,
            expected_manifest_sha256=MANIFEST_SHA,
            shard_count=325,
            shard_index=0,
            workers=1,
            expected_producer_sha256=producer_sha,
            expected_audit_implementation_sha256=audit_sha,
        )
        == 0
    )
    assert (
        json.loads((output / runner.SEAL_NAME).read_text())["status"] == "PASS"
    )

    calls = 0

    def drifting(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "components": list(EXPECTED_COMPONENTS),
            "artifacts": copy.deepcopy(
                artifact_rows if calls == 1 else _integrity("changed")
            ),
        }

    monkeypatch.setattr(runner.monolithic, "_validate_source", drifting)
    failed = tmp_path / "shard-drift"
    assert (
        runner.run(
            source,
            failed,
            expected_manifest_sha256=MANIFEST_SHA,
            shard_count=325,
            shard_index=0,
            workers=1,
            expected_producer_sha256=producer_sha,
            expected_audit_implementation_sha256=audit_sha,
        )
        == 2
    )
    terminal = json.loads((failed / runner.TERMINAL_NAME).read_text())
    assert terminal["status"] == "FAIL_CLOSED"
    assert terminal["source_immutable"] is False
    assert not (failed / runner.RESULT_NAME).exists()
