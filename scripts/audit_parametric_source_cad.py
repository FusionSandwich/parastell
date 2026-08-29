#!/usr/bin/env python3
"""Fail-closed physical audit for generic parametric ParaStell source CAD."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cadquery as cq

from scripts.audit_source_cad_candidate import _distance
from scripts.audit_source_cad_candidate import _finite_number
from scripts.audit_source_cad_candidate import _intersection
from scripts.audit_source_cad_candidate import _one_solid
from scripts.audit_source_cad_candidate import _shape_row


SCHEMA = "parastell.parametric_source_cad_audit/v1.0.0"
INTERSECTION_TOLERANCE_CM3 = 1.0e-6
_COMPONENTS: dict[str, Any] | None = None
_MAGNETS: list[Any] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source(source: Path, expected_manifest_sha256: str) -> dict:
    manifest_path = source / "PARAMETRIC_GEOMETRY_MANIFEST.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("source-CAD manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "parastell.parametric_source_cad/v1.0.0"
        or manifest.get("status") != "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING"
        or manifest.get("transport_eligible") is not False
    ):
        raise ValueError("source-CAD manifest status is invalid")
    rows = []
    for name, expected in sorted(manifest.get("artifacts", {}).items()):
        path = source / name
        actual_size = path.stat().st_size if path.is_file() else None
        actual_hash = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "path": name,
                "expected_bytes": expected.get("bytes"),
                "actual_bytes": actual_size,
                "expected_sha256": expected.get("sha256"),
                "actual_sha256": actual_hash,
                "pass": actual_size == expected.get("bytes")
                and actual_hash == expected.get("sha256"),
            }
        )
    if not rows or not all(row["pass"] for row in rows):
        raise ValueError("source-CAD artifact integrity failed")
    names = list(manifest.get("actual_invessel_component_order", ()))
    if not names or names[0] != "chamber":
        raise ValueError("source-CAD component order is invalid")
    magnet_rows = manifest.get("magnet_inventory", {}).get("magnets", ())
    expected_ids = [f"magnet-{index:04d}" for index in range(18)]
    if [row.get("magnet_id") for row in magnet_rows] != expected_ids:
        raise ValueError("source-CAD magnet identity is incomplete")
    return {"manifest": manifest, "artifacts": rows, "components": names}


def _configure(source: str, component_names: list[str]) -> None:
    global _COMPONENTS, _MAGNETS
    root = Path(source)
    _COMPONENTS = {
        name: _one_solid(root / f"{name}.step") for name in component_names
    }
    _MAGNETS = (
        cq.importers.importStep(str(root / "magnet_set.step")).solids().vals()
    )
    if len(_MAGNETS) != 18:
        raise RuntimeError(f"expected 18 magnet solids, got {len(_MAGNETS)}")


def _intersection_pass(row: dict) -> bool:
    return bool(
        row.get("status") != "ERROR"
        and _finite_number(row.get("volume_cm3"))
        and float(row["volume_cm3"]) <= INTERSECTION_TOLERANCE_CM3
    )


def _component_pair(task: tuple[str, str]) -> dict:
    first, second = task
    result = _intersection(
        _COMPONENTS[first], _COMPONENTS[second], multithread=False
    )
    row = {"components": [first, second], **result}
    row["pass"] = _intersection_pass(row)
    return row


def _magnet_component(task: tuple[int, str]) -> dict:
    index, component = task
    result = _intersection(
        _MAGNETS[index], _COMPONENTS[component], multithread=False
    )
    row = {
        "magnet_id": f"magnet-{index:04d}",
        "component": component,
        **result,
    }
    row["pass"] = _intersection_pass(row)
    if component == "vacuum_gap":
        row["minimum_clearance"] = _distance(
            _MAGNETS[index],
            _COMPONENTS[component],
            multithread=False,
            witnesses_required=True,
        )
    return row


def _magnet_pair(task: tuple[int, int]) -> dict:
    first, second = task
    result = _intersection(
        _MAGNETS[first], _MAGNETS[second], multithread=False
    )
    row = {
        "magnets": [f"magnet-{first:04d}", f"magnet-{second:04d}"],
        **result,
    }
    row["pass"] = _intersection_pass(row)
    return row


def _write_progress(path: Path, stage: str, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": stage, "row": row}) + "\n")


def _run_tasks(
    *,
    work: tuple[tuple[str, Any, list], ...],
    rows: dict[str, list[dict]],
    progress: Path,
    workers: int,
    initializer: tuple[str, list[str]],
) -> None:
    if workers == 1:
        _configure(*initializer)
        for key, function, tasks in work:
            for row in map(function, tasks):
                rows[key].append(row)
                _write_progress(progress, key, row)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_configure,
        initargs=initializer,
    ) as executor:
        for key, function, tasks in work:
            for row in executor.map(function, tasks, chunksize=1):
                rows[key].append(row)
                _write_progress(progress, key, row)


def audit(
    source: Path,
    output: Path,
    *,
    expected_manifest_sha256: str,
    workers: int,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only output exists: {output}")
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between one and four")
    source = source.resolve()
    before = _validate_source(source, expected_manifest_sha256)
    output.mkdir(parents=False)
    progress = output / "progress.jsonl"
    component_names = before["components"]
    initializer = (str(source), component_names)
    component_tasks = list(itertools.combinations(component_names, 2))
    magnet_component_tasks = list(
        itertools.product(range(18), component_names)
    )
    magnet_tasks = list(itertools.combinations(range(18), 2))
    rows: dict[str, list[dict]] = {
        "component_pairs": [],
        "magnet_component_pairs": [],
        "magnet_pairs": [],
    }
    work = (
        ("component_pairs", _component_pair, component_tasks),
        ("magnet_component_pairs", _magnet_component, magnet_component_tasks),
        ("magnet_pairs", _magnet_pair, magnet_tasks),
    )
    _run_tasks(
        work=work,
        rows=rows,
        progress=progress,
        workers=workers,
        initializer=initializer,
    )
    if workers != 1:
        _configure(str(source), component_names)
    component_inventory = [
        _shape_row(_COMPONENTS[name], identity=name)
        for name in component_names
    ]
    magnet_inventory = [
        _shape_row(shape, identity=f"magnet-{index:04d}")
        for index, shape in enumerate(_MAGNETS)
    ]
    clearance_rows = [
        row["minimum_clearance"]
        for row in rows["magnet_component_pairs"]
        if row["component"] == "vacuum_gap"
    ]
    clearances = [
        row.get("minimum_distance_cm")
        for row in clearance_rows
        if row.get("status") == "MEASURED"
        and _finite_number(row.get("minimum_distance_cm"))
    ]
    summary = {
        "component_count": len(component_inventory),
        "magnet_count": len(magnet_inventory),
        "component_pair_count": len(rows["component_pairs"]),
        "magnet_component_pair_count": len(rows["magnet_component_pairs"]),
        "magnet_pair_count": len(rows["magnet_pairs"]),
        "invalid_component_count": sum(
            not row["valid"] or not row["closed"] or row["volume_cm3"] <= 0.0
            for row in component_inventory
        ),
        "invalid_magnet_count": sum(
            not row["valid"] or not row["closed"] or row["volume_cm3"] <= 0.0
            for row in magnet_inventory
        ),
        "component_overlap_failures": sum(
            not row["pass"] for row in rows["component_pairs"]
        ),
        "magnet_component_overlap_failures": sum(
            not row["pass"] for row in rows["magnet_component_pairs"]
        ),
        "magnet_pair_overlap_failures": sum(
            not row["pass"] for row in rows["magnet_pairs"]
        ),
        "clearance_measurement_count": len(clearances),
        "minimum_vacuum_gap_to_magnet_clearance_cm": (
            min(clearances) if clearances else None
        ),
    }
    integrity_after = _validate_source(source, expected_manifest_sha256)
    report = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS"
            if all(
                (
                    summary["invalid_component_count"] == 0,
                    summary["invalid_magnet_count"] == 0,
                    summary["component_overlap_failures"] == 0,
                    summary["magnet_component_overlap_failures"] == 0,
                    summary["magnet_pair_overlap_failures"] == 0,
                    summary["clearance_measurement_count"] == 18,
                )
            )
            else "BLOCKED_SOURCE_CAD_PHYSICAL_GATE"
        ),
        "source": str(source),
        "source_manifest_sha256": expected_manifest_sha256,
        "source_integrity_before": before["artifacts"],
        "source_integrity_after": integrity_after["artifacts"],
        "source_immutable": before["artifacts"]
        == integrity_after["artifacts"],
        "intersection_tolerance_cm3": INTERSECTION_TOLERANCE_CM3,
        "workers": workers,
        "component_inventory": component_inventory,
        "magnet_inventory": magnet_inventory,
        **rows,
        "summary": summary,
        "transport_eligible": False,
    }
    report_path = output / "PARAMETRIC_SOURCE_CAD_AUDIT.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    seal = {
        "schema": "parastell.parametric_source_cad_audit_seal/v1.0.0",
        "status": report["status"],
        "report_sha256": sha256_file(report_path),
        "source_manifest_sha256": expected_manifest_sha256,
        "source_immutable": report["source_immutable"],
    }
    (output / "PARAMETRIC_SOURCE_CAD_AUDIT_SEAL.json").write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    report = audit(
        arguments.source,
        arguments.output,
        expected_manifest_sha256=arguments.expected_manifest_sha256,
        workers=arguments.workers,
    )
    print(
        json.dumps(
            {"status": report["status"], "summary": report["summary"]},
            indent=2,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
