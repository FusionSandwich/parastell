#!/usr/bin/env python3
"""Audit the nine-volume continuous-magnet ParaStell source CAD.

Only the eight touching radial pairs receive exact OCC common-volume checks.
Every nonadjacent pair is covered by the hash-chained cumulative-boundary and
strictly-positive intervening-layer proof recorded by the source producer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from parastell.continuous_radial_contract import ADJACENT_ROLE_PAIRS
from parastell.continuous_radial_contract import COMPONENT_ORDER
from parastell.continuous_radial_contract import COUPLING_INTERFACE
from parastell.continuous_radial_contract import SOURCE_AUDIT_SCHEMA
from parastell.continuous_radial_contract import SOURCE_AUDIT_SEAL_SCHEMA
from parastell.continuous_radial_contract import radial_separation_proof
from parastell.continuous_radial_contract import sha256_file
from parastell.continuous_radial_contract import validate_source_artifacts
from parastell.continuous_radial_contract import validate_source_manifest
from scripts.audit_source_cad_candidate import _finite_number
from scripts.audit_source_cad_candidate import _intersection
from scripts.audit_source_cad_candidate import _one_solid
from scripts.audit_source_cad_candidate import _shape_row


INTERSECTION_TOLERANCE_CM3 = 1.0e-5


def _load_source(source: Path, expected_manifest_sha256: str) -> dict:
    manifest_path = source / "PARAMETRIC_GEOMETRY_MANIFEST.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("source manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_source_manifest(manifest)
    return {
        "manifest": manifest,
        "artifacts": validate_source_artifacts(source, manifest),
    }


def audit(
    source: Path, output: Path, *, expected_manifest_sha256: str
) -> dict:
    source = source.resolve()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only output exists: {output}")
    before = _load_source(source, expected_manifest_sha256)
    output.mkdir(parents=False)
    inventory = []
    for name in COMPONENT_ORDER:
        solid = _one_solid(source / f"{name}.step")
        inventory.append(_shape_row(solid, identity=name))
        del solid
        gc.collect()
    adjacent = []
    for left, right in ADJACENT_ROLE_PAIRS:
        left_solid = _one_solid(source / f"{left}.step")
        right_solid = _one_solid(source / f"{right}.step")
        row = {
            "components": [left, right],
            **_intersection(left_solid, right_solid, multithread=False),
            "shared_boundary_proof": (
                "consecutive cumulative outer/inner matrix SHA-256 identity"
            ),
        }
        row["pass"] = bool(
            row.get("status") != "ERROR"
            and _finite_number(row.get("volume_cm3"))
            and 0.0 <= float(row["volume_cm3"]) <= INTERSECTION_TOLERANCE_CM3
        )
        adjacent.append(row)
        del left_solid, right_solid
        gc.collect()
    nonadjacent = radial_separation_proof(before["manifest"])
    after = _load_source(source, expected_manifest_sha256)
    source_immutable = before["artifacts"] == after["artifacts"]
    invalid = sum(
        row["valid"] is not True
        or row["closed"] is not True
        or row["numeric_values_finite"] is not True
        or row["volume_cm3"] <= 0.0
        for row in inventory
    )
    passed = bool(
        source_immutable
        and invalid == 0
        and len(adjacent) == 8
        and all(row["pass"] for row in adjacent)
        and len(nonadjacent) == 28
        and all(row["pass"] for row in nonadjacent)
    )
    report = {
        "schema": SOURCE_AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "BLOCKED_CONTINUOUS_SOURCE_CAD_GATE",
        "transport_eligible": False,
        "source": str(source),
        "source_manifest_sha256": expected_manifest_sha256,
        "source_integrity_before": before["artifacts"],
        "source_integrity_after": after["artifacts"],
        "source_immutable": source_immutable,
        "geometry_identity": {
            "extent_degrees": 90.0,
            "volume_count": 9,
            "component_order": list(COMPONENT_ORDER),
            "magnet_representation": "continuous_30_cm_radial_envelope",
            "explicit_swept_coil_count": 0,
            "coupling_interface": COUPLING_INTERFACE,
        },
        "component_inventory": inventory,
        "adjacent_exact_common_volume_checks": adjacent,
        "nonadjacent_separation_by_construction": nonadjacent,
        "intersection_tolerance_cm3": INTERSECTION_TOLERANCE_CM3,
        "summary": {
            "component_count": len(inventory),
            "invalid_component_count": invalid,
            "adjacent_exact_check_count": len(adjacent),
            "adjacent_overlap_failure_count": sum(
                not row["pass"] for row in adjacent
            ),
            "nonadjacent_construction_proof_count": len(nonadjacent),
            "nonadjacent_proof_failure_count": sum(
                not row["pass"] for row in nonadjacent
            ),
        },
    }
    report_path = output / "CONTINUOUS_RADIAL_SOURCE_CAD_AUDIT.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    seal = {
        "schema": SOURCE_AUDIT_SEAL_SCHEMA,
        "status": report["status"],
        "report_sha256": sha256_file(report_path),
        "source_manifest_sha256": expected_manifest_sha256,
        "source_immutable": source_immutable,
        "automatic_successor_launched": False,
    }
    (output / "CONTINUOUS_RADIAL_SOURCE_CAD_AUDIT_SEAL.json").write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    report = audit(
        args.source,
        args.output,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(
        json.dumps({"status": report["status"], "summary": report["summary"]})
    )
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
