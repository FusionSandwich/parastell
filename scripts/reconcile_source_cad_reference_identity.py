"""Reconcile canonical reference identity without repeating CAD Booleans."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from parastell.reference_geometry import sha256_file
from parastell.source_geometry_identity import canonical_step_payload_sha256
from parastell.source_geometry_identity import source_mesh_fingerprint


def reconcile(
    raw_report_path: Path,
    output_dir: Path,
    *,
    candidate_dir: Path,
    reference_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(
            f"create-only output already exists: {output_dir}"
        )
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    candidate_dir = candidate_dir.resolve()
    reference_dir = reference_dir.resolve()
    if Path(raw_report["candidate_dir"]).resolve() != candidate_dir:
        raise ValueError("raw audit candidate path does not match")
    required_hashes = {
        "candidate_h5m_sha256": candidate_dir / "dagmc.h5m",
        "magnet_step_sha256": candidate_dir / "magnet_set.step",
        "source_mesh_sha256": candidate_dir / "source_mesh.h5m",
    }
    for key, path in required_hashes.items():
        if raw_report[key] != sha256_file(path):
            raise ValueError(f"raw audit {key} no longer matches {path}")

    candidate_source = source_mesh_fingerprint(
        candidate_dir / "source_mesh.h5m"
    )
    reference_source = source_mesh_fingerprint(
        reference_dir / "source_mesh.h5m"
    )
    candidate_step = canonical_step_payload_sha256(
        candidate_dir / "magnet_set.step"
    )
    reference_step = canonical_step_payload_sha256(
        reference_dir / "magnet_set.step"
    )
    comparison = {
        "magnet_step_sha256": sha256_file(reference_dir / "magnet_set.step"),
        "magnet_step_byte_identical": (
            sha256_file(candidate_dir / "magnet_set.step")
            == sha256_file(reference_dir / "magnet_set.step")
        ),
        "magnet_step_canonical_sha256": reference_step,
        "candidate_magnet_step_canonical_sha256": candidate_step,
        "magnet_step_canonical_identical": candidate_step == reference_step,
        "source_mesh_sha256": sha256_file(reference_dir / "source_mesh.h5m"),
        "source_mesh_byte_identical": (
            sha256_file(candidate_dir / "source_mesh.h5m")
            == sha256_file(reference_dir / "source_mesh.h5m")
        ),
        "candidate_source_mesh_identity": candidate_source,
        "reference_source_mesh_identity": reference_source,
        "source_mesh_canonical_identical": (
            candidate_source["canonical_fingerprint"]
            == reference_source["canonical_fingerprint"]
        ),
    }
    report = deepcopy(raw_report)
    report["schema"] = "parastell.source_cad_candidate_audit/v1.1.0"
    report["identity_reconciled_at"] = datetime.now(timezone.utc).isoformat()
    report["identity_reconciliation_source_report"] = {
        "path": str(raw_report_path.resolve()),
        "sha256": sha256_file(raw_report_path),
    }
    report["reference_comparison"] = comparison
    summary = report["summary"]
    report["source_cad_gate_pass"] = all(
        (
            summary["nonpositive_component_count"] == 0,
            summary["nonpositive_magnet_count"] == 0,
            summary["magnet_component_intersection_failures"] == 0,
            summary["component_pair_intersection_failures"] == 0,
            summary["clearance_pass"],
            comparison["magnet_step_canonical_identical"],
            comparison["source_mesh_canonical_identical"],
        )
    )
    output_dir.mkdir(parents=True)
    output_path = output_dir / "source_cad_audit.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)
    arguments = parser.parse_args()
    reconcile(
        arguments.raw_report,
        arguments.output_dir,
        candidate_dir=arguments.candidate_dir,
        reference_dir=arguments.reference_dir,
    )


if __name__ == "__main__":
    main()
