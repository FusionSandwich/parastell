"""Select the largest preregistered source-mesh CFS cap that fits a DAGMC chamber."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from parastell.parametric_openmc16_model import _canonical_sha
from parastell.reference_geometry import sha256_file
from parastell.source_domain import (
    _source_volume_surface,
    audit_source_domain,
)
from parastell.source_geometry_identity import source_mesh_fingerprint


SCHEMA = "parastell.source_mesh_outer_cfs_selection/v1.0.0"


def _validated_caps(
    values: Iterable[float], radial_points: int
) -> list[float]:
    caps = [float(value) for value in values]
    penultimate = (int(radial_points) - 2) / (int(radial_points) - 1)
    if (
        int(radial_points) < 2
        or not caps
        or len(caps) != len(set(caps))
        or any(not np.isfinite(value) for value in caps)
        or any(value <= penultimate or value > 1.0 for value in caps)
        or caps != sorted(caps, reverse=True)
    ):
        raise ValueError(
            "outer CFS caps must be unique, finite, descending, at most 1.0, "
            "and above the legacy penultimate radial plane"
        )
    return caps


def _build_source_mesh(
    vmec_path: Path,
    output_path: Path,
    *,
    outer_cfs_cap: float,
    radial_points: int,
    poloidal_points: int,
    toroidal_points: int,
    extent_degrees: float,
) -> None:
    from parastell.pystell import read_vmec
    from parastell.source_mesh import SourceMesh

    radial = np.linspace(0.0, 1.0, int(radial_points))
    radial[-1] = float(outer_cfs_cap)
    mesh = SourceMesh(
        read_vmec.VMECData(str(vmec_path)),
        radial,
        np.linspace(0.0, 360.0, int(poloidal_points)),
        np.linspace(0.0, float(extent_degrees), int(toroidal_points)),
    )
    mesh.create_vertices()
    mesh.create_mesh()
    mesh.export_mesh(
        filename=output_path.stem, export_dir=str(output_path.parent)
    )
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(
            "source-mesh export did not produce the requested H5M"
        )


def _outer_boundary_clearance(
    dagmc_path: Path,
    vmec_path: Path,
    *,
    outer_cfs_cap: float,
    poloidal_points: int,
    toroidal_points: int,
    extent_degrees: float,
    source_volume_id: int,
    source_material: str,
) -> dict[str, Any]:
    import pyvista as pv
    from parastell.pystell import read_vmec

    _, surface = _source_volume_surface(
        dagmc_path, int(source_volume_id), source_material
    )
    vmec = read_vmec.VMECData(str(vmec_path))
    poloidal = np.deg2rad(np.linspace(0.0, 360.0, int(poloidal_points))[:-1])
    toroidal = np.deg2rad(
        np.linspace(0.0, float(extent_degrees), int(toroidal_points))[1:-1]
    )
    points = np.asarray(
        [
            np.asarray(vmec.vmec2xyz(outer_cfs_cap, theta, phi), dtype=float)
            * 100.0
            for phi in toroidal
            for theta in poloidal
        ]
    )
    cloud = pv.PolyData(points)
    selected = np.asarray(
        cloud.select_enclosed_points(
            surface, tolerance=1.0e-9, check_surface=True
        )["SelectedPoints"],
        dtype=np.uint8,
    )
    distance = np.abs(
        np.asarray(
            cloud.compute_implicit_distance(surface, inplace=False)[
                "implicit_distance"
            ],
            dtype=float,
        )
    )
    return {
        "sample_count": int(len(points)),
        "all_samples_enclosed": bool(np.all(selected == 1)),
        "minimum_clearance_cm": float(distance.min()),
        "maximum_clearance_cm": float(distance.max()),
        "periodic_cut_planes_excluded": True,
    }


def _selection_index(rows: list[dict[str, Any]]) -> int | None:
    for index, row in enumerate(rows):
        if row.get("candidate_pass") is True:
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc_h5m", type=Path)
    parser.add_argument("vmec", type=Path)
    parser.add_argument("reference_source_mesh", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-dagmc-sha256", required=True)
    parser.add_argument("--expected-vmec-sha256", required=True)
    parser.add_argument("--expected-reference-source-sha256", required=True)
    parser.add_argument(
        "--outer-cfs-cap", action="append", type=float, required=True
    )
    parser.add_argument("--radial-points", type=int, default=11)
    parser.add_argument("--poloidal-points", type=int, default=81)
    parser.add_argument("--toroidal-points", type=int, default=61)
    parser.add_argument("--extent-degrees", type=float, default=90.0)
    parser.add_argument("--source-volume-id", type=int, default=1)
    parser.add_argument("--source-material", default="Vacuum")
    parser.add_argument("--minimum-clearance-cm", type=float, default=1.0e-4)
    args = parser.parse_args()

    dagmc = args.dagmc_h5m.resolve(strict=True)
    vmec = args.vmec.resolve(strict=True)
    reference = args.reference_source_mesh.resolve(strict=True)
    expected = {
        dagmc: args.expected_dagmc_sha256,
        vmec: args.expected_vmec_sha256,
        reference: args.expected_reference_source_sha256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"input hash mismatch: {path}")
    if args.output_directory.exists():
        raise FileExistsError(
            f"create-only output exists: {args.output_directory}"
        )
    if args.minimum_clearance_cm <= 0.0:
        raise ValueError("minimum clearance must be positive")
    caps = _validated_caps(args.outer_cfs_cap, args.radial_points)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference_identity = source_mesh_fingerprint(reference)
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
            poloidal_points=args.poloidal_points,
            toroidal_points=args.toroidal_points,
            extent_degrees=args.extent_degrees,
            source_volume_id=args.source_volume_id,
            source_material=args.source_material,
        )
        identity = source_mesh_fingerprint(candidate)
        reference_rate = float(
            reference_identity["total_source_strength_n_per_s"]
        )
        candidate_rate = float(identity["total_source_strength_n_per_s"])
        omitted_fraction = (reference_rate - candidate_rate) / reference_rate
        passed = bool(
            domain["source_domain_gate_pass"]
            and clearance["all_samples_enclosed"]
            and clearance["minimum_clearance_cm"] >= args.minimum_clearance_cm
            and 0.0 <= omitted_fraction < 1.0
        )
        domain_path = candidate_root / "SOURCE_DOMAIN_AUDIT.json"
        domain_path.write_text(
            json.dumps(domain, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        rows.append(
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
                "candidate_pass": passed,
            }
        )
        if passed:
            break

    selected_index = _selection_index(rows)
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
            "physical_source_rate_n_per_s": reference_identity[
                "total_source_strength_n_per_s"
            ],
        },
        "radial_points": args.radial_points,
        "poloidal_points": args.poloidal_points,
        "toroidal_points": args.toroidal_points,
        "extent_degrees": args.extent_degrees,
        "minimum_clearance_cm": args.minimum_clearance_cm,
        "preregistered_caps": caps,
        "candidates": rows,
        "selected_candidate_index": selected_index,
        "selected_candidate": (
            rows[selected_index] if selected_index is not None else None
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
