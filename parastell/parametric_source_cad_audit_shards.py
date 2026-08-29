"""Pure contracts for deterministic parametric source-CAD audit shards.

This module deliberately has no CadQuery, OpenCascade, DAGMC, or OpenMC
imports.  It defines the complete protected pair universe and validates shard
rows so that the authoritative merger can remain a zero-CAD operation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SHARD_RESULT_SCHEMA = "parastell.parametric_source_cad_audit_shard/v1.0.0"
SHARD_TERMINAL_SCHEMA = (
    "parastell.parametric_source_cad_audit_shard_terminal/v1.0.0"
)
SHARD_SEAL_SCHEMA = "parastell.parametric_source_cad_audit_shard_seal/v1.0.0"
MERGE_CONTROL_SCHEMA = (
    "parastell.parametric_source_cad_audit_merge_control/v1.0.0"
)
MERGE_TERMINAL_SCHEMA = (
    "parastell.parametric_source_cad_audit_merge_terminal/v1.0.0"
)
ASSIGNMENT_POLICY = "canonical_global_index_modulo_shard_count"
INTERSECTION_TOLERANCE_CM3 = 1.0e-6
POSITIVE_SEPARATION_TOLERANCE_CM = 1.0e-8
EXPECTED_COMPONENTS = (
    "chamber",
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "vacuum_gap",
)
EXPECTED_MAGNET_IDS = tuple(f"magnet-{index:04d}" for index in range(18))
EXPECTED_COUNTS = {
    "component_pairs": 28,
    "magnet_component_pairs": 144,
    "magnet_pairs": 153,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _descriptor(index: int, stage: str, arguments: Sequence[Any]) -> dict:
    if stage == "component_pairs":
        task_id = f"component:{arguments[0]}|{arguments[1]}"
    elif stage == "magnet_component_pairs":
        task_id = f"magnet-component:{arguments[0]}|{arguments[1]}"
    elif stage == "magnet_pairs":
        task_id = f"magnet:{arguments[0]}|{arguments[1]}"
    else:  # pragma: no cover - construction is internal
        raise ValueError(f"unknown task stage: {stage}")
    return {
        "canonical_index": index,
        "stage": stage,
        "task_id": task_id,
        "arguments": list(arguments),
    }


def canonical_tasks(
    component_names: Sequence[str] = EXPECTED_COMPONENTS,
) -> list[dict]:
    components = tuple(str(value) for value in component_names)
    if components != EXPECTED_COMPONENTS:
        raise ValueError(
            "source component order is not the direct-90 contract"
        )
    rows: list[dict] = []
    index = 0
    for pair in itertools.combinations(components, 2):
        rows.append(_descriptor(index, "component_pairs", pair))
        index += 1
    for magnet_index, component in itertools.product(range(18), components):
        rows.append(
            _descriptor(
                index,
                "magnet_component_pairs",
                (f"magnet-{magnet_index:04d}", component),
            )
        )
        index += 1
    for first, second in itertools.combinations(range(18), 2):
        rows.append(
            _descriptor(
                index,
                "magnet_pairs",
                (f"magnet-{first:04d}", f"magnet-{second:04d}"),
            )
        )
        index += 1
    if len(rows) != 325:  # pragma: no cover - protects future edits
        raise AssertionError("canonical pair universe is not 325 rows")
    return rows


def plan_payload(
    *,
    manifest_sha256: str,
    shard_count: int,
    audit_implementation_sha256: str,
    shard_producer_sha256: str,
) -> dict:
    if not 1 <= int(shard_count) <= 325:
        raise ValueError("shard count must be between one and 325")
    tasks = canonical_tasks()
    return {
        "schema": "parastell.parametric_source_cad_audit_shard_plan/v1.0.0",
        "assignment_policy": ASSIGNMENT_POLICY,
        "source_manifest_sha256": manifest_sha256,
        "audit_implementation_sha256": audit_implementation_sha256,
        "shard_producer_sha256": shard_producer_sha256,
        "shard_count": int(shard_count),
        "component_order": list(EXPECTED_COMPONENTS),
        "task_count": len(tasks),
        "task_inventory_sha256": canonical_json_sha256(tasks),
    }


def assigned_tasks(plan: Mapping[str, Any], shard_index: int) -> list[dict]:
    shard_count = int(plan["shard_count"])
    if not 0 <= int(shard_index) < shard_count:
        raise ValueError("shard index is outside the shard plan")
    return [
        row
        for row in canonical_tasks(plan["component_order"])
        if int(row["canonical_index"]) % shard_count == int(shard_index)
    ]


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_integrity_rows(rows: Any) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("source integrity inventory is missing")
    paths = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("source integrity row is malformed")
        path = row.get("path")
        if (
            not isinstance(path, str)
            or not path
            or row.get("pass") is not True
            or row.get("expected_bytes") != row.get("actual_bytes")
            or row.get("expected_sha256") != row.get("actual_sha256")
        ):
            raise ValueError("source integrity row does not pass")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("source integrity inventory contains duplicate paths")
    return rows


def validate_shape_inventory(
    rows: Any, expected_identities: Sequence[str], *, label: str
) -> list[dict]:
    if not isinstance(rows, list) or len(rows) != len(expected_identities):
        raise ValueError(f"{label} inventory is incomplete")
    for row, identity in zip(rows, expected_identities, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("identity") != identity
            or row.get("valid") is not True
            or row.get("closed") is not True
            or not finite_number(row.get("volume_cm3"))
            or float(row["volume_cm3"]) <= 0.0
        ):
            raise ValueError(f"{label} inventory row is invalid: {identity}")
    return rows


def _intersection_row_pass(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("pass") is True
        and row.get("status") != "ERROR"
        and finite_number(row.get("volume_cm3"))
        and 0.0 <= float(row["volume_cm3"]) <= INTERSECTION_TOLERANCE_CM3
        and isinstance(row.get("separation_distance"), dict)
    )


def _clearance_pass(row: Mapping[str, Any]) -> bool:
    separation = row.get("minimum_clearance")
    witnesses = (
        separation.get("witnesses") if isinstance(separation, dict) else None
    )
    return bool(
        row.get("clearance_witness_pass") is True
        and separation == row.get("separation_distance")
        and separation.get("status") == "MEASURED"
        and separation.get("witnesses_required") is True
        and finite_number(separation.get("minimum_distance_cm"))
        and float(separation["minimum_distance_cm"])
        > POSITIVE_SEPARATION_TOLERANCE_CM
        and isinstance(separation.get("solution_count"), int)
        and not isinstance(separation.get("solution_count"), bool)
        and isinstance(witnesses, list)
        and len(witnesses) == separation["solution_count"]
        and bool(witnesses)
        and all(
            isinstance(witness, dict)
            and witness.get("distance_reconstruction_pass") is True
            and witness.get("point_membership_pass") is True
            for witness in witnesses
        )
    )


def validate_task_record(record: Any, expected: Mapping[str, Any]) -> dict:
    if not isinstance(record, dict):
        raise TypeError("shard task record is malformed")
    descriptor = record.get("task")
    row = record.get("row")
    if descriptor != dict(expected) or not isinstance(row, dict):
        raise ValueError(
            f"shard task identity mismatch: {expected['task_id']}"
        )
    if not _intersection_row_pass(row):
        raise ValueError(f"shard task failed: {expected['task_id']}")
    arguments = list(expected["arguments"])
    stage = expected["stage"]
    if stage == "component_pairs":
        identity_pass = row.get("components") == arguments
    elif stage == "magnet_component_pairs":
        identity_pass = (
            row.get("magnet_id") == arguments[0]
            and row.get("component") == arguments[1]
        )
        if identity_pass and arguments[1] == "vacuum_gap":
            identity_pass = _clearance_pass(row)
    else:
        identity_pass = row.get("magnets") == arguments
    if not identity_pass:
        raise ValueError(f"shard row payload mismatch: {expected['task_id']}")
    return row
