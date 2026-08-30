"""Audit and select from an immutable set of source-mesh CFS candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from parastell.parametric_openmc16_model import _canonical_sha
from parastell.reference_geometry import sha256_file
from parastell.source_domain import audit_source_domain
from parastell.source_geometry_identity import source_mesh_fingerprint
from scripts.select_source_mesh_outer_cfs_cap import (
    SCHEMA,
    _outer_boundary_clearance,
    _validated_caps,
)


def _verified_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("candidate-manifest file hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    content_digest = manifest.get("receipt_content_sha256")
    content = dict(manifest)
    content.pop("receipt_content_sha256", None)
    if content_digest != _canonical_sha(content):
        raise ValueError("candidate-manifest content hash mismatch")
    if (
        manifest.get("schema") != "parastell.source_mesh_cfs_candidates/v1.0.0"
        or manifest.get("status") != "CANDIDATES_BUILT_CONTAINMENT_PENDING"
        or manifest.get("geometry_mutated") is not False
        or manifest.get("inner_cfs_planes_mutated") is not False
    ):
        raise ValueError("candidate manifest is not eligible for audit")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc_h5m", type=Path)
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-dagmc-sha256", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--source-volume-id", type=int, default=1)
    parser.add_argument("--source-material", default="Vacuum")
    parser.add_argument("--minimum-clearance-cm", type=float, default=1.0e-4)
    args = parser.parse_args()

    dagmc = args.dagmc_h5m.resolve(strict=True)
    manifest_path = args.candidate_manifest.resolve(strict=True)
    if sha256_file(dagmc) != args.expected_dagmc_sha256:
        raise ValueError("DAGMC H5M hash mismatch")
    manifest = _verified_manifest(
        manifest_path, args.expected_candidate_manifest_sha256
    )
    if args.output_directory.exists():
        raise FileExistsError(
            f"create-only output exists: {args.output_directory}"
        )
    if args.minimum_clearance_cm <= 0.0:
        raise ValueError("minimum clearance must be positive")

    caps = _validated_caps(
        manifest["caps_descending"], manifest["radial_points"]
    )
    rows = manifest["candidates"]
    if len(rows) != len(caps):
        raise ValueError("candidate rows and preregistered caps disagree")
    vmec = Path(manifest["vmec"]["path"]).resolve(strict=True)
    reference = Path(manifest["reference_source_mesh"]["path"]).resolve(
        strict=True
    )
    if sha256_file(vmec) != manifest["vmec"]["sha256"]:
        raise ValueError("VMEC hash mismatch")
    if sha256_file(reference) != manifest["reference_source_mesh"]["sha256"]:
        raise ValueError("reference source-mesh hash mismatch")
    reference_identity = source_mesh_fingerprint(reference)
    reference_rate = float(reference_identity["total_source_strength_n_per_s"])

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audited_rows = []
    selected_index = None
    for rank, (cap, row) in enumerate(zip(caps, rows, strict=True)):
        if (
            int(row["rank_descending"]) != rank
            or float(row["outer_cfs_cap"]) != cap
        ):
            raise ValueError("candidate rank or cap differs from manifest")
        candidate = Path(row["source_mesh"]["path"]).resolve(strict=True)
        if sha256_file(candidate) != row["source_mesh"]["sha256"]:
            raise ValueError(f"candidate {rank} source-mesh hash mismatch")
        domain = audit_source_domain(
            dagmc,
            candidate,
            candidate,
            source_volume_id=args.source_volume_id,
            source_material=args.source_material,
        )
        clearance = _outer_boundary_clearance(
            dagmc,
            vmec,
            outer_cfs_cap=cap,
            poloidal_points=manifest["poloidal_points"],
            toroidal_points=manifest["toroidal_points"],
            extent_degrees=manifest["extent_degrees"],
            source_volume_id=args.source_volume_id,
            source_material=args.source_material,
        )
        candidate_rate = float(
            domain["source_mesh_identity"]["total_source_strength_n_per_s"]
        )
        omitted_fraction = (reference_rate - candidate_rate) / reference_rate
        candidate_pass = bool(
            domain["source_domain_gate_pass"]
            and clearance["all_samples_enclosed"]
            and clearance["minimum_clearance_cm"] >= args.minimum_clearance_cm
            and 0.0 <= omitted_fraction < 1.0
        )
        domain_path = output / f"candidate-{rank:02d}-SOURCE_DOMAIN_AUDIT.json"
        domain_path.write_text(
            json.dumps(domain, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        audited_rows.append(
            {
                "rank_descending": rank,
                "outer_cfs_cap": cap,
                "candidate_source_mesh": {
                    "path": str(candidate),
                    "sha256": sha256_file(candidate),
                },
                "source_domain_audit": {
                    "path": str(domain_path),
                    "sha256": sha256_file(domain_path),
                    "invalid_sample_point_count": domain[
                        "invalid_sample_point_count"
                    ],
                    "pass": domain["source_domain_gate_pass"],
                },
                "outer_boundary_clearance": clearance,
                "physical_source_rate_n_per_s": candidate_rate,
                "omitted_edge_source_fraction": omitted_fraction,
                "candidate_pass": candidate_pass,
            }
        )
        if candidate_pass:
            selected_index = rank
            break

    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "SOURCE_MESH_OUTER_CFS_CAP_SELECTED"
            if selected_index is not None
            else "BLOCKED_NO_SOURCE_MESH_CFS_CAP_PASSED"
        ),
        "selection_rule": (
            "first passing cap in the preregistered strictly descending list"
        ),
        "candidate_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "receipt_content_sha256": manifest["receipt_content_sha256"],
        },
        "geometry_mutated": False,
        "inner_source_cfs_planes_mutated": False,
        "dagmc_h5m": {"path": str(dagmc), "sha256": sha256_file(dagmc)},
        "vmec": {"path": str(vmec), "sha256": sha256_file(vmec)},
        "reference_source_mesh": {
            "path": str(reference),
            "sha256": sha256_file(reference),
            "canonical_fingerprint": reference_identity[
                "canonical_fingerprint"
            ],
            "physical_source_rate_n_per_s": reference_rate,
        },
        "minimum_clearance_cm": args.minimum_clearance_cm,
        "preregistered_caps": caps,
        "candidates": audited_rows,
        "selected_candidate_index": selected_index,
        "selected_candidate": (
            audited_rows[selected_index]
            if selected_index is not None
            else None
        ),
        "input_immutability_pass": bool(
            sha256_file(dagmc) == args.expected_dagmc_sha256
            and sha256_file(manifest_path)
            == args.expected_candidate_manifest_sha256
            and sha256_file(vmec) == manifest["vmec"]["sha256"]
            and sha256_file(reference)
            == manifest["reference_source_mesh"]["sha256"]
        ),
        "production_run_authorized": False,
    }
    result["receipt_content_sha256"] = _canonical_sha(result)
    result_path = output / "SOURCE_MESH_OUTER_CFS_SELECTION.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(result_path)
    if selected_index is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
