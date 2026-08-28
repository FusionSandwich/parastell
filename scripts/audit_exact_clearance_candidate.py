"""Measure the exact global vessel/magnet clearance of a source-CAD candidate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import cadquery as cq

from parastell.reference_geometry import sha256_file
from scripts.audit_source_cad_candidate import _distance
from scripts.audit_source_cad_candidate import _verify_manifest_artifacts


_VESSEL = None
_MAGNETS = None


def _configure_worker(candidate_dir: str) -> None:
    global _VESSEL
    global _MAGNETS
    directory = Path(candidate_dir)
    vessels = (
        cq.importers.importStep(str(directory / "vacuum_vessel.step"))
        .solids()
        .vals()
    )
    if len(vessels) != 1:
        raise RuntimeError(f"expected one vessel solid, got {len(vessels)}")
    _VESSEL = vessels[0]
    _MAGNETS = (
        cq.importers.importStep(str(directory / "magnet_set.step"))
        .solids()
        .vals()
    )
    if len(_MAGNETS) != 18:
        raise RuntimeError(f"expected 18 magnet solids, got {len(_MAGNETS)}")


def _measure(index: int) -> dict:
    result = _distance(_VESSEL, _MAGNETS[index], multithread=True)
    return {"magnet_id": f"magnet-{index:04d}", **result}


def audit(
    candidate_dir: Path,
    output: Path,
    *,
    expected_candidate_manifest_sha256: str,
    acceptance_criteria: Path,
    expected_acceptance_criteria_sha256: str,
    workers: int = 2,
) -> dict:
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    manifest_path = candidate_dir / "candidate_build_manifest.json"
    if sha256_file(manifest_path) != expected_candidate_manifest_sha256:
        raise ValueError("candidate manifest hash mismatch")
    if sha256_file(acceptance_criteria) != expected_acceptance_criteria_sha256:
        raise ValueError("acceptance-criteria hash mismatch")
    criteria = json.loads(acceptance_criteria.read_text(encoding="utf-8"))
    minimum_cm = float(
        criteria["separated_interface"]["minimum_accepted_clearance_cm"]
    )
    integrity = _verify_manifest_artifacts(
        candidate_dir,
        "candidate_build_manifest.json",
        required_artifacts=frozenset(
            {
                "back_wall.step",
                "breeder.step",
                "chamber.step",
                "first_wall.step",
                "magnet_set.step",
                "shield.step",
                "source_mesh.h5m",
                "vacuum_vessel.step",
            }
        ),
    )
    if not integrity["pass"]:
        raise ValueError("candidate manifest artifact integrity failed")

    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_configure_worker,
        initargs=(str(candidate_dir.resolve()),),
    ) as executor:
        rows = list(executor.map(_measure, range(18)))
    measured = [row for row in rows if row.get("status") == "MEASURED"]
    exact_expected_ids = {f"magnet-{index:04d}" for index in range(18)}
    actual_ids = [str(row.get("magnet_id", "")) for row in rows]
    unique_id_gate = (
        len(actual_ids) == len(set(actual_ids))
        and set(actual_ids) == exact_expected_ids
    )
    witness_failures = [
        row
        for row in measured
        if row.get("solution_count", 0) < 1
        or row.get("solution_count") != len(row.get("witnesses", []))
        or not row.get("witnesses")
        or not all(
            witness.get("distance_reconstruction_pass") is True
            and witness.get("point_membership_pass") is True
            for witness in row.get("witnesses", [])
        )
    ]
    errors = [row for row in rows if row.get("status") != "MEASURED"]
    minimum = min(
        (float(row["minimum_distance_cm"]) for row in measured),
        default=None,
    )
    gate = (
        len(rows) == 18
        and unique_id_gate
        and len(measured) == 18
        and not witness_failures
        and not errors
        and minimum is not None
        and minimum >= minimum_cm
    )
    result = {
        "schema": "parastell.exact_global_clearance_audit/v1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": str(candidate_dir),
        "candidate_manifest_sha256": expected_candidate_manifest_sha256,
        "acceptance_criteria_sha256": expected_acceptance_criteria_sha256,
        "minimum_required_clearance_cm": minimum_cm,
        "minimum_measured_clearance_cm": minimum,
        "measurement_count": len(measured),
        "error_count": len(errors),
        "witness_failure_count": len(witness_failures),
        "exact_unique_magnet_id_gate_pass": unique_id_gate,
        "workers": int(workers),
        "candidate_integrity": integrity,
        "audit_implementation": {
            "script_sha256": sha256_file(Path(__file__)),
            "source_cad_audit_sha256": sha256_file(
                Path(__file__)
                .resolve()
                .with_name("audit_source_cad_candidate.py")
            ),
        },
        "measurements": rows,
        "exact_global_clearance_gate_pass": gate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--acceptance-criteria", required=True, type=Path)
    parser.add_argument("--expected-acceptance-criteria-sha256", required=True)
    parser.add_argument("--workers", type=int, default=2)
    arguments = parser.parse_args()
    audit(
        arguments.candidate_dir.resolve(),
        arguments.output.resolve(),
        expected_candidate_manifest_sha256=(
            arguments.expected_candidate_manifest_sha256
        ),
        acceptance_criteria=arguments.acceptance_criteria.resolve(),
        expected_acceptance_criteria_sha256=(
            arguments.expected_acceptance_criteria_sha256
        ),
        workers=arguments.workers,
    )


if __name__ == "__main__":
    main()
