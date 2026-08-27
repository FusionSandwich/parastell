"""Deterministic faceted-ray and point-in-volume checks for R1 magnets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from parastell.reference_geometry import _outward_sign
from parastell.reference_geometry import _triangles
from parastell.reference_geometry import sha256_file


def _hits(origin, direction, triangles, *, tolerance=1.0e-10):
    """Vectorized Moller-Trumbore intersections."""
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    valid = np.abs(determinant) > tolerance
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    tvec = origin - triangles[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse
    qvec = np.cross(tvec, edge1)
    v = np.einsum("j,ij->i", direction, qvec) * inverse
    distance = np.einsum("ij,ij->i", edge2, qvec) * inverse
    mask = (
        valid
        & (u >= -tolerance)
        & (v >= -tolerance)
        & (u + v <= 1.0 + tolerance)
        & (distance > tolerance)
    )
    return np.flatnonzero(mask), distance[mask]


def _first_surface(origin, direction, triangles, surface_ids):
    indices, distances = _hits(origin, direction, triangles)
    if not len(indices):
        return {"distance_cm": None, "surface_ids": []}
    minimum = float(distances.min())
    selected = np.abs(distances - minimum) <= max(1.0e-7, minimum * 1.0e-10)
    return {
        "distance_cm": minimum,
        "surface_ids": sorted(
            {int(item) for item in surface_ids[indices[selected]]}
        ),
    }


def _point_inside_volume(point, direction, volume):
    triangles = np.concatenate(
        [_triangles(surface.triangle_coords) for surface in volume.surfaces]
    )
    _, distances = _hits(point, direction, triangles)
    unique = []
    for value in sorted(float(item) for item in distances):
        if not unique or abs(value - unique[-1]) > 1.0e-7:
            unique.append(value)
    return len(unique) % 2 == 1, len(unique)


def audit(dagmc_path: Path, inventory_path: Path) -> dict:
    import pydagmc

    model = pydagmc.Model(str(dagmc_path))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    unique_surfaces = {
        int(surface.id): surface
        for volume in model.volumes_by_id.values()
        for surface in volume.surfaces
    }
    triangles = []
    surface_ids = []
    for surface_id, surface in sorted(unique_surfaces.items()):
        values = _triangles(surface.triangle_coords)
        triangles.append(values)
        surface_ids.extend([surface_id] * len(values))
    all_triangles = np.concatenate(triangles)
    all_surface_ids = np.asarray(surface_ids, dtype=int)
    parity_direction = np.asarray([0.371, 0.613, 0.697], dtype=float)
    parity_direction /= np.linalg.norm(parity_direction)
    rows = []
    for magnet in inventory["magnets"]:
        volume_id = int(magnet["volume_id"])
        volume = model.volumes_by_id[volume_id]
        candidates = []
        for surface in volume.surfaces:
            values = _triangles(surface.triangle_coords)
            cross = np.cross(
                values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]
            )
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            index = int(np.argmax(areas))
            candidates.append(
                (float(areas[index]), surface, values[index], cross[index])
            )
        _, surface, triangle, native_cross = max(
            candidates, key=lambda item: item[0]
        )
        normal = native_cross / np.linalg.norm(native_cross)
        normal *= _outward_sign(surface, volume_id)
        centroid = triangle.mean(axis=0)
        epsilon_cm = 0.1
        outside = centroid + epsilon_cm * normal
        inside = centroid - epsilon_cm * normal
        inbound = _first_surface(
            outside, -normal, all_triangles, all_surface_ids
        )
        outbound = _first_surface(
            inside, normal, all_triangles, all_surface_ids
        )
        outside_class, outside_crossings = _point_inside_volume(
            outside, parity_direction, volume
        )
        inside_class, inside_crossings = _point_inside_volume(
            inside, parity_direction, volume
        )
        expected_surface = int(surface.id)
        rows.append(
            {
                "magnet_id": magnet["magnet_id"],
                "volume_id": volume_id,
                "seed_surface_id": expected_surface,
                "seed_triangle_centroid_cm": [
                    float(item) for item in centroid
                ],
                "outward_normal": [float(item) for item in normal],
                "epsilon_cm": epsilon_cm,
                "outside_to_inside": {
                    "expected_volume_sequence": [
                        "implicit_complement",
                        volume_id,
                    ],
                    "expected_surface_id": expected_surface,
                    "actual_first_crossing": inbound,
                    "pass": expected_surface in inbound["surface_ids"],
                },
                "inside_to_outside": {
                    "expected_volume_sequence": [
                        volume_id,
                        "implicit_complement",
                    ],
                    "expected_surface_id": expected_surface,
                    "actual_first_crossing": outbound,
                    "pass": expected_surface in outbound["surface_ids"],
                },
                "point_in_volume": {
                    "outside_classified_inside": outside_class,
                    "inside_classified_inside": inside_class,
                    "outside_forward_crossing_count": outside_crossings,
                    "inside_forward_crossing_count": inside_crossings,
                    "pass": (not outside_class) and inside_class,
                },
            }
        )
    return {
        "schema": "parastell.reference_deterministic_ray_audit/v1.0.0",
        "raw_h5m_sha256": sha256_file(dagmc_path),
        "ray_algorithm": "vectorized Moller-Trumbore on actual DAGMC facets",
        "point_in_volume_algorithm": "odd-even forward ray parity",
        "deterministic_direction": [float(item) for item in parity_direction],
        "magnet_count": len(rows),
        "ray_count": 2 * len(rows),
        "all_first_surface_ids_pass": all(
            row["outside_to_inside"]["pass"]
            and row["inside_to_outside"]["pass"]
            for row in rows
        ),
        "all_point_in_volume_samples_pass": all(
            row["point_in_volume"]["pass"] for row in rows
        ),
        "rays": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.dagmc.resolve(), arguments.inventory.resolve())
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
