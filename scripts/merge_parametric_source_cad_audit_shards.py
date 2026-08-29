#!/usr/bin/env python3
"""Merge complete sealed source-CAD audit shards without loading CAD."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from parastell.parametric_source_cad_audit_shards import (
    EXPECTED_COMPONENTS,
    EXPECTED_COUNTS,
    EXPECTED_MAGNET_IDS,
    MERGE_CONTROL_SCHEMA,
    MERGE_TERMINAL_SCHEMA,
    SHARD_RESULT_SCHEMA,
    SHARD_SEAL_SCHEMA,
    SHARD_TERMINAL_SCHEMA,
    assigned_tasks,
    canonical_json_sha256,
    canonical_tasks,
    finite_number,
    plan_payload,
    sha256_file,
    validate_integrity_rows,
    validate_shape_inventory,
    validate_task_record,
)

AUDIT_SCHEMA = "parastell.parametric_source_cad_audit/v1.0.0"
AUDIT_SEAL_SCHEMA = "parastell.parametric_source_cad_audit_seal/v1.0.0"
AUDIT_NAME = "PARAMETRIC_SOURCE_CAD_AUDIT.json"
AUDIT_SEAL_NAME = "PARAMETRIC_SOURCE_CAD_AUDIT_SEAL.json"
MERGE_TERMINAL_NAME = "PARAMETRIC_SOURCE_CAD_AUDIT_MERGE_TERMINAL.json"
SHARD_RESULT_NAME = "PARAMETRIC_SOURCE_CAD_AUDIT_SHARD.json"
SHARD_TERMINAL_NAME = "PARAMETRIC_SOURCE_CAD_AUDIT_SHARD_TERMINAL.json"


def _write_json_create_only(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _load_json_bound(path: Path, expected_sha256: str) -> dict:
    path = path.resolve(strict=True)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"hash mismatch for {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _validated_shard(
    entry: dict,
    *,
    control_root: Path,
    plan: dict,
    plan_sha256: str,
    expected_index: int,
) -> dict:
    if (
        not isinstance(entry, dict)
        or entry.get("shard_index") != expected_index
        or not isinstance(entry.get("seal_path"), str)
        or not isinstance(entry.get("seal_sha256"), str)
    ):
        raise ValueError(
            f"merge control shard entry is invalid: {expected_index}"
        )
    seal_path = Path(entry["seal_path"])
    if not seal_path.is_absolute():
        seal_path = control_root / seal_path
    seal = _load_json_bound(seal_path, entry["seal_sha256"])
    if (
        seal.get("schema") != SHARD_SEAL_SCHEMA
        or seal.get("status") != "PASS"
        or seal.get("source_manifest_sha256") != plan["source_manifest_sha256"]
        or seal.get("plan_sha256") != plan_sha256
        or seal.get("shard_index") != expected_index
        or seal.get("shard_count") != plan["shard_count"]
        or seal.get("producer_sha256") != plan["shard_producer_sha256"]
        or seal.get("audit_implementation_sha256")
        != plan["audit_implementation_sha256"]
        or seal.get("source_immutable") is not True
        or seal.get("terminal_filename") != SHARD_TERMINAL_NAME
        or seal.get("result_filename") != SHARD_RESULT_NAME
    ):
        raise ValueError(f"shard seal does not pass: {expected_index}")
    terminal_path = seal_path.parent / SHARD_TERMINAL_NAME
    result_path = seal_path.parent / SHARD_RESULT_NAME
    terminal = _load_json_bound(terminal_path, seal.get("terminal_sha256", ""))
    result = _load_json_bound(result_path, seal.get("result_sha256", ""))
    if (
        terminal.get("schema") != SHARD_TERMINAL_SCHEMA
        or terminal.get("status") != "PASS"
        or terminal.get("return_code") != 0
        or terminal.get("error") is not None
        or terminal.get("source_immutable") is not True
        or terminal.get("source_manifest_sha256")
        != plan["source_manifest_sha256"]
        or terminal.get("plan_sha256") != plan_sha256
        or terminal.get("shard_index") != expected_index
        or terminal.get("shard_count") != plan["shard_count"]
        or terminal.get("automatic_successor_launched") is not False
        or terminal.get("result", {}).get("sha256")
        != seal.get("result_sha256")
    ):
        raise ValueError(
            f"shard terminal receipt does not pass: {expected_index}"
        )
    expected_tasks = assigned_tasks(plan, expected_index)
    if (
        result.get("schema") != SHARD_RESULT_SCHEMA
        or result.get("status") != "PASS"
        or result.get("source_manifest_sha256")
        != plan["source_manifest_sha256"]
        or result.get("source_immutable") is not True
        or result.get("transport_eligible") is not False
        or result.get("plan") != plan
        or result.get("plan_sha256") != plan_sha256
        or result.get("shard_index") != expected_index
        or result.get("shard_count") != plan["shard_count"]
        or result.get("expected_tasks") != expected_tasks
        or not isinstance(result.get("task_records"), list)
        or len(result["task_records"]) != len(expected_tasks)
    ):
        raise ValueError(f"shard result metadata is invalid: {expected_index}")
    validate_integrity_rows(result.get("source_integrity_before"))
    validate_integrity_rows(result.get("source_integrity_after"))
    if result["source_integrity_before"] != result["source_integrity_after"]:
        raise ValueError(f"shard source drift detected: {expected_index}")
    validate_shape_inventory(
        result.get("component_inventory"),
        EXPECTED_COMPONENTS,
        label="component",
    )
    validate_shape_inventory(
        result.get("magnet_inventory"),
        EXPECTED_MAGNET_IDS,
        label="magnet",
    )
    for record, expected in zip(
        result["task_records"], expected_tasks, strict=True
    ):
        validate_task_record(record, expected)
    return result


def _load_and_validate(
    control: dict, control_path: Path
) -> tuple[dict, list[dict]]:
    if control.get("schema") != MERGE_CONTROL_SCHEMA:
        raise ValueError("merge control schema is invalid")
    producer = Path(__file__).resolve()
    if control.get("merge_producer_sha256") != sha256_file(producer):
        raise ValueError("merge producer hash mismatch")
    shard_count = control.get("shard_count")
    if not isinstance(shard_count, int) or isinstance(shard_count, bool):
        raise TypeError("merge shard count is invalid")
    plan = plan_payload(
        manifest_sha256=control.get("source_manifest_sha256", ""),
        shard_count=shard_count,
        audit_implementation_sha256=control.get(
            "audit_implementation_sha256", ""
        ),
        shard_producer_sha256=control.get("shard_producer_sha256", ""),
    )
    plan_sha256 = canonical_json_sha256(plan)
    if control.get("plan_sha256") != plan_sha256:
        raise ValueError("merge control shard plan hash mismatch")
    entries = control.get("shards")
    if not isinstance(entries, list) or len(entries) != shard_count:
        raise ValueError("merge control does not enumerate every shard")
    if [
        entry.get("shard_index")
        for entry in entries
        if isinstance(entry, dict)
    ] != list(range(shard_count)):
        raise ValueError(
            "merge control shard indices are missing or reordered"
        )
    results = [
        _validated_shard(
            entry,
            control_root=control_path.parent,
            plan=plan,
            plan_sha256=plan_sha256,
            expected_index=index,
        )
        for index, entry in enumerate(entries)
    ]
    return plan, results


def _authoritative_report(
    control: dict,
    control_sha256: str,
    plan: dict,
    results: list[dict],
) -> dict:
    first = results[0]
    integrity_before = first["source_integrity_before"]
    integrity_after = first["source_integrity_after"]
    component_inventory = first["component_inventory"]
    magnet_inventory = first["magnet_inventory"]
    for result in results[1:]:
        if (
            result["source_integrity_before"] != integrity_before
            or result["source_integrity_after"] != integrity_after
            or result["component_inventory"] != component_inventory
            or result["magnet_inventory"] != magnet_inventory
        ):
            raise ValueError("shards disagree on source or shape inventories")
    record_by_index: dict[int, dict] = {}
    for result in results:
        for record in result["task_records"]:
            index = record["task"]["canonical_index"]
            if index in record_by_index:
                raise ValueError(f"duplicate canonical task index: {index}")
            record_by_index[index] = record
    tasks = canonical_tasks()
    if set(record_by_index) != set(range(len(tasks))):
        raise ValueError("merged task coverage is incomplete or unexpected")
    grouped = {name: [] for name in EXPECTED_COUNTS}
    for expected in tasks:
        record = record_by_index[expected["canonical_index"]]
        row = validate_task_record(record, expected)
        grouped[expected["stage"]].append(row)
    if any(
        len(grouped[name]) != count for name, count in EXPECTED_COUNTS.items()
    ):
        raise ValueError("merged task stage counts are invalid")
    clearance_rows = [
        row["minimum_clearance"]
        for row in grouped["magnet_component_pairs"]
        if row["component"] == "vacuum_gap"
    ]
    clearances = [row["minimum_distance_cm"] for row in clearance_rows]
    if len(clearances) != 18 or not all(
        finite_number(value) for value in clearances
    ):
        raise ValueError("merged clearance evidence is incomplete")
    summary = {
        "component_count": 8,
        "magnet_count": 18,
        "component_pair_count": 28,
        "magnet_component_pair_count": 144,
        "magnet_pair_count": 153,
        "invalid_component_count": 0,
        "invalid_magnet_count": 0,
        "component_overlap_failures": 0,
        "magnet_component_overlap_failures": 0,
        "magnet_pair_overlap_failures": 0,
        "clearance_measurement_count": 18,
        "clearance_witness_pass_count": 18,
        "minimum_vacuum_gap_to_magnet_clearance_cm": min(clearances),
    }
    return {
        "schema": AUDIT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "source": control.get(
            "source_description",
            f"sharded-source:{plan['source_manifest_sha256']}",
        ),
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "source_integrity_before": integrity_before,
        "source_integrity_after": integrity_after,
        "source_immutable": True,
        "intersection_tolerance_cm3": 1.0e-6,
        "workers": max(int(result["workers"]) for result in results),
        "component_inventory": component_inventory,
        "magnet_inventory": magnet_inventory,
        **grouped,
        "summary": summary,
        "transport_eligible": False,
        "sharded_provenance": {
            "merge_control_sha256": control_sha256,
            "plan_sha256": canonical_json_sha256(plan),
            "shard_count": plan["shard_count"],
            "assignment_policy": plan["assignment_policy"],
            "all_shards_terminal_pass": True,
        },
    }


def run(
    control_path: Path,
    output: Path,
    *,
    expected_control_sha256: str,
) -> int:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only merge output exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    output.mkdir()
    audit_path = output / AUDIT_NAME
    terminal_path = output / MERGE_TERMINAL_NAME
    seal_path = output / AUDIT_SEAL_NAME
    started_utc = datetime.now(UTC).isoformat()
    started = time.monotonic()
    error = None
    report = None
    control_sha256 = None
    plan = None
    try:
        control_path = control_path.resolve(strict=True)
        control_sha256 = sha256_file(control_path)
        if control_sha256 != expected_control_sha256:
            raise ValueError("merge control hash mismatch")
        control = json.loads(control_path.read_text(encoding="utf-8"))
        if not isinstance(control, dict):
            raise TypeError("merge control must be a JSON object")
        plan, results = _load_and_validate(control, control_path)
        report = _authoritative_report(control, control_sha256, plan, results)
        _write_json_create_only(audit_path, report)
    except Exception as exc:  # noqa: BLE001 - terminal evidence is mandatory
        error = f"{type(exc).__name__}: {exc}"
    passed = report is not None and error is None
    terminal = {
        "schema": MERGE_TERMINAL_SCHEMA,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "return_code": 0 if passed else 2,
        "error": error,
        "control_sha256": control_sha256,
        "plan_sha256": canonical_json_sha256(plan) if plan else None,
        "audit": (
            {"filename": AUDIT_NAME, "sha256": sha256_file(audit_path)}
            if audit_path.is_file()
            else None
        ),
        "cad_imported": False,
        "automatic_successor_launched": False,
    }
    _write_json_create_only(terminal_path, terminal)
    if passed:
        seal = {
            "schema": AUDIT_SEAL_SCHEMA,
            "status": "PASS",
            "report_sha256": sha256_file(audit_path),
            "source_manifest_sha256": report["source_manifest_sha256"],
            "source_immutable": True,
            "merge_terminal_sha256": sha256_file(terminal_path),
            "merge_control_sha256": control_sha256,
            "plan_sha256": canonical_json_sha256(plan),
        }
        _write_json_create_only(seal_path, seal)
    return int(terminal["return_code"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-control-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    return run(
        arguments.control,
        arguments.output,
        expected_control_sha256=arguments.expected_control_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
