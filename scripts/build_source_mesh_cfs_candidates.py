"""Build hash-bound CFS-cap source candidates without requiring CAD/PyVista."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from parastell.parametric_openmc16_model import _canonical_sha
from parastell.reference_geometry import sha256_file
from parastell.source_geometry_identity import source_mesh_fingerprint
from scripts.select_source_mesh_outer_cfs_cap import (
    _build_source_mesh,
    _validated_caps,
)


SCHEMA = "parastell.source_mesh_cfs_candidates/v1.0.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vmec", type=Path)
    parser.add_argument("reference_source_mesh", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-vmec-sha256", required=True)
    parser.add_argument("--expected-reference-source-sha256", required=True)
    parser.add_argument(
        "--outer-cfs-cap", action="append", type=float, required=True
    )
    parser.add_argument("--radial-points", type=int, default=11)
    parser.add_argument("--poloidal-points", type=int, default=81)
    parser.add_argument("--toroidal-points", type=int, default=61)
    parser.add_argument("--extent-degrees", type=float, default=90.0)
    args = parser.parse_args()

    vmec = args.vmec.resolve(strict=True)
    reference = args.reference_source_mesh.resolve(strict=True)
    if sha256_file(vmec) != args.expected_vmec_sha256:
        raise ValueError("VMEC hash mismatch")
    if sha256_file(reference) != args.expected_reference_source_sha256:
        raise ValueError("reference source-mesh hash mismatch")
    if args.output_directory.exists():
        raise FileExistsError(
            f"create-only output exists: {args.output_directory}"
        )
    caps = _validated_caps(args.outer_cfs_cap, args.radial_points)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference_identity = source_mesh_fingerprint(reference)
    reference_rate = float(reference_identity["total_source_strength_n_per_s"])
    rows = []
    for rank, cap in enumerate(caps):
        candidate_root = output / f"candidate-{rank:02d}-cfs-{cap:.9f}"
        candidate_root.mkdir()
        candidate = candidate_root / "source_mesh.h5m"
        _build_source_mesh(
            vmec,
            candidate,
            outer_cfs_cap=cap,
            radial_points=args.radial_points,
            poloidal_points=args.poloidal_points,
            toroidal_points=args.toroidal_points,
            extent_degrees=args.extent_degrees,
        )
        identity = source_mesh_fingerprint(candidate)
        rate = float(identity["total_source_strength_n_per_s"])
        rows.append(
            {
                "rank_descending": rank,
                "outer_cfs_cap": cap,
                "source_mesh": {
                    "path": str(candidate),
                    "sha256": sha256_file(candidate),
                    "bytes": candidate.stat().st_size,
                    "canonical_fingerprint": identity["canonical_fingerprint"],
                },
                "physical_source_rate_n_per_s": rate,
                "omitted_edge_source_fraction": (reference_rate - rate)
                / reference_rate,
                "containment_status": "NOT_RUN",
            }
        )
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATES_BUILT_CONTAINMENT_PENDING",
        "vmec": {"path": str(vmec), "sha256": sha256_file(vmec)},
        "reference_source_mesh": {
            "path": str(reference),
            "sha256": sha256_file(reference),
            "canonical_fingerprint": reference_identity[
                "canonical_fingerprint"
            ],
            "physical_source_rate_n_per_s": reference_rate,
        },
        "radial_points": args.radial_points,
        "poloidal_points": args.poloidal_points,
        "toroidal_points": args.toroidal_points,
        "extent_degrees": args.extent_degrees,
        "caps_descending": caps,
        "inner_cfs_planes_mutated": False,
        "geometry_mutated": False,
        "candidates": rows,
        "transport_eligible": False,
        "production_run_authorized": False,
    }
    result["receipt_content_sha256"] = _canonical_sha(result)
    receipt = output / "SOURCE_MESH_CFS_CANDIDATES.json"
    receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(receipt)


if __name__ == "__main__":
    main()
