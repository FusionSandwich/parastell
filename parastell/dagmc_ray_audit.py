"""Deterministic seeded DAGMC ray-fire audit for every magnet surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

import numpy as np

from .dagmc_envelope import _openmc_to_outward_normal_sign
from .dagmc_envelope import _triangles
from .magnet_surface_audit import _load_associations
from .magnet_surface_audit import _pair_associations
from .magnet_surface_audit import adjacent_volume_id


_HIT = re.compile(r"hits surf_id\s+(\d+)\s+dist=([0-9eE+.-]+)")


def _ray_on_surface(
    surface: Any,
    volume_id: int,
    rng: np.random.Generator,
    *,
    offset_cm: float,
) -> tuple[list[float], list[float], int]:
    triangles = _triangles(surface.triangle_coords)
    facet_index = int(rng.integers(0, len(triangles)))
    triangle = triangles[facet_index]
    barycentric = rng.dirichlet(np.ones(3))
    crossing = barycentric @ triangle
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    normal *= _openmc_to_outward_normal_sign(surface, volume_id)
    start = crossing - float(offset_cm) * normal
    return start.tolist(), normal.tolist(), facet_index


def _run_volume_rays(
    executable: str,
    dagmc_path: Path,
    volume_id: int,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if not rows:
        return rows, ""
    command = [executable, "-i", str(volume_id), "-n", "0"]
    for row in rows:
        command.extend(["-f", *[f"{value:.17g}" for value in row["start_cm"]]])
        command.extend(f"{value:.17g}" for value in row["direction"])
    command.append(str(dagmc_path))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    hits = _HIT.findall(output)
    if completed.returncode != 0 or len(hits) != len(rows):
        for row in rows:
            row.update(
                {
                    "actual_surface_id": None,
                    "distance_cm": None,
                    "pass": False,
                    "failure": "RAY_FIRE_EXECUTION_OR_PARSE_FAILURE",
                }
            )
        return rows, output
    for row, (surface_id, distance) in zip(rows, hits):
        row["actual_surface_id"] = int(surface_id)
        row["distance_cm"] = float(distance)
        row["pass"] = int(surface_id) == int(row["expected_surface_id"])
        row["failure"] = None if row["pass"] else "UNEXPECTED_FIRST_SURFACE"
    return rows, output


def audit_all_magnet_rays(
    dagmc_path: str | Path,
    associations_path: str | Path,
    *,
    executable: str = "ray_fire_test",
    seed: int = 20260826,
    offset_cm: float = 1.0e-3,
) -> dict[str, Any]:
    """Fire a seeded interior ray through every casing and pack surface."""
    import pydagmc

    path = Path(dagmc_path).resolve()
    model = pydagmc.Model(str(path))
    pairs = _pair_associations(_load_associations(associations_path))
    rng = np.random.default_rng(int(seed))
    rows_by_volume: dict[int, list[dict[str, Any]]] = {}
    direct_external = []
    for magnet_id, coil_id, casing_id, winding_id in pairs:
        casing = model.volumes_by_id[casing_id]
        winding = model.volumes_by_id[winding_id]
        for component, volume, other_id in (
            ("winding_pack", winding, casing_id),
            ("outer_casing", casing, winding_id),
        ):
            for surface in sorted(
                volume.surfaces, key=lambda item: int(item.id)
            ):
                start, direction, facet_index = _ray_on_surface(
                    surface, int(volume.id), rng, offset_cm=offset_cm
                )
                adjacent = adjacent_volume_id(surface, int(volume.id))
                if component == "winding_pack" and adjacent == other_id:
                    category = "winding_pack_toward_casing"
                elif component == "winding_pack":
                    category = "winding_pack_directly_toward_external_region"
                    direct_external.append(
                        {
                            "magnet_id": magnet_id,
                            "surface_id": int(surface.id),
                            "adjacent_volume_id": adjacent,
                        }
                    )
                elif adjacent == other_id:
                    category = "casing_toward_winding_pack"
                else:
                    category = "casing_toward_external_region"
                rows_by_volume.setdefault(int(volume.id), []).append(
                    {
                        "magnet_id": magnet_id,
                        "coil_id": coil_id,
                        "origin_component": component,
                        "origin_volume_id": int(volume.id),
                        "adjacent_volume_id": adjacent,
                        "category": category,
                        "expected_surface_id": int(surface.id),
                        "seeded_facet_index": facet_index,
                        "start_cm": start,
                        "direction": direction,
                    }
                )
    all_rows = []
    logs = []
    for volume_id in sorted(rows_by_volume):
        rows, output = _run_volume_rays(
            executable, path, volume_id, rows_by_volume[volume_id]
        )
        all_rows.extend(rows)
        logs.append({"volume_id": volume_id, "output": output})
    return {
        "schema": "parastell.dagmc_seeded_ray_audit/v1",
        "dagmc_path": str(path),
        "seed": int(seed),
        "offset_cm": float(offset_cm),
        "ray_count": len(all_rows),
        "all_first_surface_ids_pass": all(row["pass"] for row in all_rows),
        "direct_winding_to_external_topologies": direct_external,
        "no_winding_to_external_bypass_pass": not direct_external,
        "coverage": (
            "one deterministic seeded barycentric point on every DAGMC surface "
            "bounding every casing and winding pack"
        ),
        "rays": all_rows,
        "raw_logs": logs,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc_path")
    parser.add_argument("associations_path")
    parser.add_argument("output_path")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--offset-cm", type=float, default=1.0e-3)
    parser.add_argument("--executable", default="ray_fire_test")
    options = parser.parse_args(argv)
    report = audit_all_magnet_rays(
        options.dagmc_path,
        options.associations_path,
        executable=options.executable,
        seed=options.seed,
        offset_cm=options.offset_cm,
    )
    target = Path(options.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if report["all_first_surface_ids_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
