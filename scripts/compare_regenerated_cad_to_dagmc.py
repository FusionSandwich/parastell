"""Compare regenerated ParaStell solids with their canonical DAGMC volumes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cadquery as cq
import numpy as np

from parastell.dagmc_envelope import (
    _openmc_to_outward_normal_sign,
    _triangles,
    audit_closed_triangle_volume,
)

SCHEMA = "parastell.regenerated_cad_dagmc_comparison/v1.0.0"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def triangle_metrics(triangles):
    triangles = np.asarray(triangles, dtype=float)
    vectors = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(vectors, axis=1)
    signed_volumes = (
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        )
        / 6.0
    )
    signed_volume = float(np.sum(signed_volumes))
    if signed_volume < 0.0:
        triangles = triangles[:, [0, 2, 1]]
        signed_volumes = -signed_volumes
        signed_volume = -signed_volume
    if not np.isfinite(signed_volume) or signed_volume <= 0.0:
        raise ValueError("triangle shell has non-positive signed volume")
    volume_centroid = (
        np.sum(
            signed_volumes[:, None] * np.sum(triangles, axis=1) / 4.0,
            axis=0,
        )
        / signed_volume
    )
    surface_centroid = np.sum(
        areas[:, None] * np.mean(triangles, axis=1), axis=0
    ) / float(np.sum(areas))
    return {
        "volume_cm3": signed_volume,
        "surface_area_cm2": float(np.sum(areas)),
        "volume_centroid_cm": volume_centroid,
        "surface_centroid_cm": surface_centroid,
        "bounding_box_cm": np.asarray(
            [
                np.min(triangles.reshape((-1, 3)), axis=0),
                np.max(triangles.reshape((-1, 3)), axis=0),
            ]
        ),
    }


def tessellate_cad(solid, linear_tolerance_cm, angular_tolerance_rad):
    vertices, connectivity = solid.tessellate(
        linear_tolerance_cm, angularTolerance=angular_tolerance_rad
    )
    coordinates = np.asarray(
        [[vertex.x, vertex.y, vertex.z] for vertex in vertices], dtype=float
    )
    connectivity = np.asarray(connectivity, dtype=int)
    return coordinates, connectivity, coordinates[connectivity]


def dagmc_volume_triangles(path, volume_id):
    import pydagmc

    volume = pydagmc.Model(str(path)).volumes_by_id[int(volume_id)]
    rows = []
    for surface in volume.surfaces:
        triangles = _triangles(surface.triangle_coords).copy()
        if _openmc_to_outward_normal_sign(surface, int(volume_id)) == -1:
            triangles = triangles[:, [0, 2, 1]]
        rows.append(triangles)
    audit = audit_closed_triangle_volume(volume).to_dict()
    return np.concatenate(rows), audit


def polydata(points, connectivity):
    import pyvista as pv

    faces = np.column_stack(
        [np.full(len(connectivity), 3, dtype=int), connectivity]
    ).reshape(-1)
    return pv.PolyData(np.asarray(points, dtype=float), faces)


def triangle_soup_polydata(triangles):
    points = np.asarray(triangles, dtype=float).reshape((-1, 3))
    connectivity = np.arange(len(points), dtype=int).reshape((-1, 3))
    return polydata(points, connectivity)


def maximum_vertex_to_surface_distance(source, target):
    measured = source.compute_implicit_distance(target, inplace=False)
    values = np.abs(np.asarray(measured["implicit_distance"], dtype=float))
    return {
        "maximum_cm": float(np.max(values)),
        "mean_cm": float(np.mean(values)),
        "p95_cm": float(np.quantile(values, 0.95)),
        "sample_count": len(values),
    }


def compare_role(
    solid,
    h5m_path,
    volume_id,
    *,
    linear_tolerance_cm,
    angular_tolerance_rad,
    limits,
):
    cad_points, cad_connectivity, cad_triangles = tessellate_cad(
        solid, linear_tolerance_cm, angular_tolerance_rad
    )
    dagmc_triangles, dagmc_audit = dagmc_volume_triangles(h5m_path, volume_id)
    cad_metrics = triangle_metrics(cad_triangles)
    dagmc_metrics = triangle_metrics(dagmc_triangles)
    cad_surface = polydata(cad_points, cad_connectivity)
    dagmc_surface = triangle_soup_polydata(dagmc_triangles)
    cad_to_dagmc = maximum_vertex_to_surface_distance(
        cad_surface, dagmc_surface
    )
    dagmc_to_cad = maximum_vertex_to_surface_distance(
        dagmc_surface, cad_surface
    )
    differences = {
        "volume_relative": abs(
            cad_metrics["volume_cm3"] - dagmc_metrics["volume_cm3"]
        )
        / dagmc_metrics["volume_cm3"],
        "surface_area_relative": abs(
            cad_metrics["surface_area_cm2"] - dagmc_metrics["surface_area_cm2"]
        )
        / dagmc_metrics["surface_area_cm2"],
        "volume_centroid_distance_cm": float(
            np.linalg.norm(
                cad_metrics["volume_centroid_cm"]
                - dagmc_metrics["volume_centroid_cm"]
            )
        ),
        "bounding_box_maximum_difference_cm": float(
            np.max(
                np.abs(
                    cad_metrics["bounding_box_cm"]
                    - dagmc_metrics["bounding_box_cm"]
                )
            )
        ),
        "symmetric_vertex_surface_distance_maximum_cm": max(
            cad_to_dagmc["maximum_cm"], dagmc_to_cad["maximum_cm"]
        ),
    }
    checks = {
        "volume": differences["volume_relative"]
        <= limits["maximum_volume_relative_difference"],
        "surface_area": differences["surface_area_relative"]
        <= limits["maximum_surface_area_relative_difference"],
        "volume_centroid": differences["volume_centroid_distance_cm"]
        <= limits["maximum_volume_centroid_distance_cm"],
        "bounding_box": differences["bounding_box_maximum_difference_cm"]
        <= limits["maximum_bounding_box_difference_cm"],
        "surface_distance": differences[
            "symmetric_vertex_surface_distance_maximum_cm"
        ]
        <= limits["maximum_surface_distance_cm"],
    }

    def serialise(metrics):
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }

    return {
        "dagmc_volume_id": int(volume_id),
        "cad": {
            **serialise(cad_metrics),
            "occ_mass_volume_cm3": float(solid.Volume()),
            "triangle_count": len(cad_triangles),
        },
        "dagmc": {
            **serialise(dagmc_metrics),
            "triangle_count": len(dagmc_triangles),
            "closed_volume_audit": dagmc_audit,
        },
        "distances": {
            "cad_vertices_to_dagmc_surface": cad_to_dagmc,
            "dagmc_vertices_to_cad_surface": dagmc_to_cad,
            "method": "VTK implicit point-to-triangle distance, symmetric vertices",
        },
        "differences": differences,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5m", required=True)
    parser.add_argument("--source-cad-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--linear-tolerance-cm", type=float, default=0.5)
    parser.add_argument("--angular-tolerance-rad", type=float, default=0.1)
    parser.add_argument(
        "--maximum-volume-relative-difference", type=float, default=0.0025
    )
    parser.add_argument(
        "--maximum-surface-area-relative-difference", type=float, default=0.02
    )
    parser.add_argument(
        "--maximum-volume-centroid-distance-cm", type=float, default=1.0
    )
    parser.add_argument(
        "--maximum-bounding-box-difference-cm", type=float, default=5.0
    )
    parser.add_argument(
        "--maximum-surface-distance-cm", type=float, default=5.0
    )
    return parser.parse_args()


def main():
    args = parse_args()
    h5m = Path(args.h5m).resolve()
    manifest_path = Path(args.source_cad_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    limits = {
        name: float(getattr(args, name))
        for name in (
            "maximum_volume_relative_difference",
            "maximum_surface_area_relative_difference",
            "maximum_volume_centroid_distance_cm",
            "maximum_bounding_box_difference_cm",
            "maximum_surface_distance_cm",
        )
    }
    roles = {}
    for role in ("casing", "winding_pack"):
        record = manifest["artifacts"][role]
        brep = manifest_path.parent / record["brep"]["path"]
        solid = cq.importers.importBrep(str(brep))
        if hasattr(solid, "val"):
            solid = solid.val()
        roles[role] = compare_role(
            solid,
            h5m,
            record["dagmc_volume_id"],
            linear_tolerance_cm=args.linear_tolerance_cm,
            angular_tolerance_rad=args.angular_tolerance_rad,
            limits=limits,
        )
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in roles.values())
        else "FAIL"
    )
    result = {
        "schema": SCHEMA,
        "status": status,
        "classification": (
            "PARASTELL_SOURCE_CAD_REGENERATED_EXACT_INPUTS"
            if status == "PASS"
            else "PARASTELL_SOURCE_CAD_REGENERATION_MISMATCH"
        ),
        "canonical_h5m": {
            "local_identifier": h5m.name,
            "sha256": sha256(h5m),
        },
        "source_cad_manifest": {
            "local_identifier": manifest_path.name,
            "sha256": sha256(manifest_path),
        },
        "target_geometry_fingerprint": manifest["target_dagmc"][
            "canonical_geometry_fingerprint"
        ],
        "tessellation": {
            "linear_tolerance_cm": args.linear_tolerance_cm,
            "angular_tolerance_rad": args.angular_tolerance_rad,
        },
        "acceptance_limits": limits,
        "roles": roles,
        "limitations": [
            "distance is a symmetric vertex-to-opposite-triangle metric",
            "pass qualifies deterministic source-solid regeneration only; it does not qualify a direct edit",
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
