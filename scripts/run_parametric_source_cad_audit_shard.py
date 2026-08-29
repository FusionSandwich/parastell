#!/usr/bin/env python3
"""Run one deterministic, create-only parametric source-CAD audit shard."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.audit_parametric_source_cad as monolithic
from parastell.parametric_source_cad_audit_shards import (
    EXPECTED_COMPONENTS,
    EXPECTED_MAGNET_IDS,
    SHARD_RESULT_SCHEMA,
    SHARD_SEAL_SCHEMA,
    SHARD_TERMINAL_SCHEMA,
    assigned_tasks,
    canonical_json_sha256,
    plan_payload,
    sha256_file,
    validate_integrity_rows,
    validate_shape_inventory,
    validate_task_record,
)

RESULT_NAME = "PARAMETRIC_SOURCE_CAD_AUDIT_SHARD.json"
TERMINAL_NAME = "PARAMETRIC_SOURCE_CAD_AUDIT_SHARD_TERMINAL.json"
SEAL_NAME = "PARAMETRIC_SOURCE_CAD_AUDIT_SHARD_SEAL.json"
PROGRESS_NAME = "progress.jsonl"


def _write_json_create_only(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _execute(task: Mapping[str, Any]) -> dict:
    stage = task["stage"]
    arguments = task["arguments"]
    if stage == "component_pairs":
        row = monolithic._component_pair(tuple(arguments))
    elif stage == "magnet_component_pairs":
        row = monolithic._magnet_component(
            (int(str(arguments[0]).removeprefix("magnet-")), arguments[1])
        )
    elif stage == "magnet_pairs":
        row = monolithic._magnet_pair(
            (
                int(str(arguments[0]).removeprefix("magnet-")),
                int(str(arguments[1]).removeprefix("magnet-")),
            )
        )
    else:  # pragma: no cover - plan validation prevents this
        raise ValueError(f"unknown shard task stage: {stage}")
    return {"task": dict(task), "row": row}


def _run_assigned_tasks(
    tasks: list[dict],
    *,
    source: Path,
    workers: int,
    progress: Path,
) -> list[dict]:
    initializer = (str(source), list(EXPECTED_COMPONENTS))
    completed: list[tuple[int, dict]] = []
    if workers == 1:
        monolithic._configure(*initializer)
        for index, task in enumerate(tasks):
            record = _execute(task)
            completed.append((index, record))
            with progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=monolithic._configure,
            initargs=initializer,
        ) as executor:
            futures = {
                executor.submit(_execute, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                record = future.result()
                completed.append((futures[future], record))
                with progress.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
        monolithic._configure(*initializer)
    return [record for _, record in sorted(completed)]


def run(
    source: Path,
    output: Path,
    *,
    expected_manifest_sha256: str,
    shard_count: int,
    shard_index: int,
    workers: int,
    expected_producer_sha256: str,
    expected_audit_implementation_sha256: str,
) -> int:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only shard output exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if not 1 <= int(workers) <= 16:
        raise ValueError("workers must be between one and sixteen")
    output.mkdir()
    progress = output / PROGRESS_NAME
    result_path = output / RESULT_NAME
    terminal_path = output / TERMINAL_NAME
    seal_path = output / SEAL_NAME
    producer = Path(__file__).resolve()
    audit_implementation = Path(monolithic.__file__).resolve()
    started_utc = datetime.now(UTC).isoformat()
    started = time.monotonic()
    error = None
    result: dict[str, Any] | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    tasks: list[dict] = []
    try:
        if sha256_file(producer) != expected_producer_sha256:
            raise ValueError("shard producer hash mismatch")
        if (
            sha256_file(audit_implementation)
            != expected_audit_implementation_sha256
        ):
            raise ValueError("audit implementation hash mismatch")
        source = source.resolve(strict=True)
        before = monolithic._validate_source(source, expected_manifest_sha256)
        if tuple(before["components"]) != EXPECTED_COMPONENTS:
            raise ValueError(
                "source component order is not the direct-90 contract"
            )
        plan = plan_payload(
            manifest_sha256=expected_manifest_sha256,
            shard_count=shard_count,
            audit_implementation_sha256=(expected_audit_implementation_sha256),
            shard_producer_sha256=expected_producer_sha256,
        )
        tasks = assigned_tasks(plan, shard_index)
        records = _run_assigned_tasks(
            tasks,
            source=source,
            workers=workers,
            progress=progress,
        )
        if len(records) != len(tasks):
            raise RuntimeError("shard task execution is incomplete")
        for record, expected in zip(records, tasks, strict=True):
            validate_task_record(record, expected)
        component_inventory = [
            monolithic._shape_row(monolithic._COMPONENTS[name], identity=name)
            for name in EXPECTED_COMPONENTS
        ]
        magnet_inventory = [
            monolithic._shape_row(shape, identity=EXPECTED_MAGNET_IDS[index])
            for index, shape in enumerate(monolithic._MAGNETS)
        ]
        validate_shape_inventory(
            component_inventory, EXPECTED_COMPONENTS, label="component"
        )
        validate_shape_inventory(
            magnet_inventory, EXPECTED_MAGNET_IDS, label="magnet"
        )
        after = monolithic._validate_source(source, expected_manifest_sha256)
        validate_integrity_rows(before["artifacts"])
        validate_integrity_rows(after["artifacts"])
        source_immutable = before["artifacts"] == after["artifacts"]
        if not source_immutable:
            raise RuntimeError("source CAD changed during shard execution")
        result = {
            "schema": SHARD_RESULT_SCHEMA,
            "status": "PASS",
            "created_utc": datetime.now(UTC).isoformat(),
            "source": str(source),
            "source_manifest_sha256": expected_manifest_sha256,
            "source_integrity_before": before["artifacts"],
            "source_integrity_after": after["artifacts"],
            "source_immutable": True,
            "plan": plan,
            "plan_sha256": canonical_json_sha256(plan),
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "workers": int(workers),
            "expected_tasks": tasks,
            "task_records": records,
            "component_inventory": component_inventory,
            "magnet_inventory": magnet_inventory,
            "transport_eligible": False,
        }
        _write_json_create_only(result_path, result)
    except Exception as exc:  # noqa: BLE001 - terminal evidence is mandatory
        error = f"{type(exc).__name__}: {exc}"
        if before is not None:
            try:
                after = monolithic._validate_source(
                    source, expected_manifest_sha256
                )
            except Exception:  # noqa: BLE001 - preserve original failure
                after = None

    passed = result is not None and error is None
    terminal = {
        "schema": SHARD_TERMINAL_SCHEMA,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "return_code": 0 if passed else 2,
        "error": error,
        "source_manifest_sha256": expected_manifest_sha256,
        "source_integrity_before": before["artifacts"] if before else None,
        "source_integrity_after": after["artifacts"] if after else None,
        "source_immutable": bool(
            before is not None
            and after is not None
            and before["artifacts"] == after["artifacts"]
        ),
        "plan_sha256": canonical_json_sha256(plan) if plan else None,
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "assigned_task_count": len(tasks),
        "result": (
            {"filename": RESULT_NAME, "sha256": sha256_file(result_path)}
            if result_path.is_file()
            else None
        ),
        "progress": (
            {
                "filename": PROGRESS_NAME,
                "sha256": sha256_file(progress),
                "bytes": progress.stat().st_size,
            }
            if progress.is_file()
            else None
        ),
        "automatic_successor_launched": False,
    }
    _write_json_create_only(terminal_path, terminal)
    seal = {
        "schema": SHARD_SEAL_SCHEMA,
        "status": terminal["status"],
        "source_manifest_sha256": expected_manifest_sha256,
        "plan_sha256": terminal["plan_sha256"],
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "producer_sha256": expected_producer_sha256,
        "audit_implementation_sha256": (expected_audit_implementation_sha256),
        "terminal_filename": TERMINAL_NAME,
        "terminal_sha256": sha256_file(terminal_path),
        "result_filename": RESULT_NAME if result_path.is_file() else None,
        "result_sha256": (
            sha256_file(result_path) if result_path.is_file() else None
        ),
        "source_immutable": terminal["source_immutable"],
    }
    _write_json_create_only(seal_path, seal)
    return int(terminal["return_code"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--expected-producer-sha256", required=True)
    parser.add_argument(
        "--expected-audit-implementation-sha256", required=True
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    return run(
        arguments.source,
        arguments.output,
        expected_manifest_sha256=arguments.expected_manifest_sha256,
        shard_count=arguments.shard_count,
        shard_index=arguments.shard_index,
        workers=arguments.workers,
        expected_producer_sha256=arguments.expected_producer_sha256,
        expected_audit_implementation_sha256=(
            arguments.expected_audit_implementation_sha256
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
