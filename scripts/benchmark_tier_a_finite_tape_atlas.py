"""Six-face Tier-A finite-tape localization and cache benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns
import tracemalloc

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from parastell.facet_patch_atlas import (  # noqa: E402
    build_facet_patch_atlas,
    localize_surface_crossings_cached,
)
from parastell.source_conformal_interface import (  # noqa: E402
    write_compact_json_create_only,
)
from parastell.surface_source_localization import (  # noqa: E402
    localize_surface_crossings,
)


PACKET_HASHES = {
    "YBCO": "78732861bcf898f84612701a64f8551e1ec41fcdb1a1692bed3a8def645b6b90",
    "GdBCO": "75973f9530a68559b080d7ce3e2ed692b616d26a320f41d480ca0ea28433e782",
    "EuBCO": "70e77dbe8f2ba0e7b818fe47ee60f32c95f0438db55de67ed33dc3f41e589e56",
    "SmBCO": "aee8fe606e37fdfa1f9f73b52485cf1fa6b6b3a0a638997cef4fffb1b6ddb9c7",
}
ANGLES_FROM_TAPE_PLANE_DEG = (90.0, 45.0, 10.0, 5.0, 1.0)


def _oriented_triangles(
    corners: np.ndarray, outward: np.ndarray
) -> list[np.ndarray]:
    rows = []
    for indices in ((0, 1, 2), (0, 2, 3)):
        triangle = corners[list(indices)].copy()
        geometric = np.cross(
            triangle[1] - triangle[0], triangle[2] - triangle[0]
        )
        if np.dot(geometric, outward) < 0.0:
            triangle[[1, 2]] = triangle[[2, 1]]
        rows.append(triangle)
    return rows


def finite_tape_fixture(
    *,
    length_cm: float = 100.0,
    width_cm: float = 0.4,
    thickness_um: float = 0.1,
) -> tuple[dict, dict, list[str], list[float]]:
    if min(length_cm, width_cm, thickness_um) <= 0.0:
        raise ValueError("finite-tape dimensions must be positive")
    x, y, z = length_cm / 2.0, width_cm / 2.0, thickness_um * 1.0e-4 / 2.0
    faces = [
        (
            "front",
            101,
            np.array([0.0, 0.0, 1.0]),
            np.array([[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]]),
        ),
        (
            "rear",
            102,
            np.array([0.0, 0.0, -1.0]),
            np.array([[-x, -y, -z], [-x, y, -z], [x, y, -z], [x, -y, -z]]),
        ),
        (
            "edge_positive",
            103,
            np.array([0.0, 1.0, 0.0]),
            np.array([[-x, y, -z], [-x, y, z], [x, y, z], [x, y, -z]]),
        ),
        (
            "edge_negative",
            104,
            np.array([0.0, -1.0, 0.0]),
            np.array([[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]]),
        ),
        (
            "end_positive",
            105,
            np.array([1.0, 0.0, 0.0]),
            np.array([[x, -y, -z], [x, y, -z], [x, y, z], [x, -y, z]]),
        ),
        (
            "end_negative",
            106,
            np.array([-1.0, 0.0, 0.0]),
            np.array([[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]]),
        ),
    ]
    triangles: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    surface_ids: list[int] = []
    facet_ids: list[str] = []
    face_by_facet: list[str] = []
    for face, surface_id, normal, corners in faces:
        for local_index, triangle in enumerate(
            _oriented_triangles(corners, normal)
        ):
            triangles.append(triangle)
            normals.append(normal)
            surface_ids.append(surface_id)
            facet_ids.append(f"tier-a-{face}-{local_index}")
            face_by_facet.append(face)
    coordinate_ids: dict[tuple[float, float, float], int] = {}
    vertex_ids = []
    for triangle in triangles:
        row = []
        for vertex in triangle:
            row.append(
                coordinate_ids.setdefault(tuple(vertex), len(coordinate_ids))
            )
        vertex_ids.append(row)
    catalog = {
        "facet_id": facet_ids,
        "surface_id": np.asarray(surface_ids),
        "vertex_id": np.asarray(vertex_ids),
        "vertices_global_cm": np.asarray(triangles),
        "outward_normal_global": np.asarray(normals),
        "normal_source": "dagmc_forward_reverse_topology",
    }
    positions = []
    directions = []
    query_surface_ids = []
    query_faces = []
    query_angles = []
    for face_index, (face, surface_id, normal, _) in enumerate(faces):
        triangle = triangles[2 * face_index]
        position = np.mean(triangle, axis=0)
        tangent = triangle[1] - triangle[0]
        tangent -= np.dot(tangent, normal) * normal
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        for angle in ANGLES_FROM_TAPE_PLANE_DEG:
            pairs = (
                ((1.0, 0.0),)
                if angle == 90.0
                else ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
            )
            for first, second in pairs:
                if angle == 90.0:
                    direction = -normal
                else:
                    in_plane = first * tangent + second * bitangent
                    in_plane /= np.linalg.norm(in_plane)
                    radians = np.deg2rad(angle)
                    direction = (
                        -np.sin(radians) * normal + np.cos(radians) * in_plane
                    )
                positions.append(position)
                directions.append(direction)
                query_surface_ids.append(surface_id)
                query_faces.append(face)
                query_angles.append(angle)
    phase = {
        "position_global_cm": np.asarray(positions),
        "direction_global": np.asarray(directions),
        "surface_id": np.asarray(query_surface_ids),
    }
    return catalog, phase, query_faces, query_angles


def _median_ns(operation, repeats: int) -> int:
    values = []
    for _ in range(repeats):
        start = perf_counter_ns()
        operation()
        values.append(perf_counter_ns() - start)
    return int(statistics.median(values))


def benchmark(repeats: int = 9) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    source_start = perf_counter_ns()
    catalog, phase, query_faces, query_angles = finite_tape_fixture()
    source_mapping_ns = perf_counter_ns() - source_start
    start = perf_counter_ns()
    atlas = build_facet_patch_atlas(catalog)
    preprocessing_ns = perf_counter_ns() - start
    cached, cached_audit = localize_surface_crossings_cached(phase, atlas)
    uncached, _ = localize_surface_crossings(phase, catalog)
    if not (
        np.array_equal(cached["facet_id"], uncached["facet_id"])
        and np.allclose(
            cached["barycentric_coordinates"],
            uncached["barycentric_coordinates"],
        )
        and np.allclose(cached["mu"], uncached["mu"])
        and np.array_equal(
            cached["crossing_sense"], uncached["crossing_sense"]
        )
    ):
        raise RuntimeError(
            "cached and uncached finite-tape localization disagree"
        )
    cached_ns = _median_ns(
        lambda: localize_surface_crossings_cached(phase, atlas), repeats
    )
    uncached_ns = _median_ns(
        lambda: localize_surface_crossings(phase, catalog), repeats
    )
    score_start = perf_counter_ns()
    protected_score = float(np.sum(np.abs(cached["mu"])))
    scoring_ns = perf_counter_ns() - score_start
    remap_start = perf_counter_ns()
    _ = protected_score * np.ones(atlas.facet_count) / atlas.facet_count
    remapping_ns = perf_counter_ns() - remap_start
    tracemalloc.start()
    localize_surface_crossings_cached(phase, atlas)
    _, cached_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    octants = sorted(
        {
            "".join("+" if value > 0.0 else "-" for value in direction)
            for direction in phase["direction_global"]
            if np.all(np.abs(direction) > 1.0e-14)
        }
    )
    return {
        "schema": "parastell.tier_a_finite_tape_atlas_benchmark/v1.0.0",
        "status": "TIER_A_REFERENCE_CANDIDATE_DIAGNOSTIC",
        "material_packet_sha256": PACKET_HASHES,
        "minimum_candidate_thickness_um": 0.1,
        "length_to_thickness_ratio": 1.0e7,
        "source_faces": sorted(set(query_faces)),
        "angles_from_tape_plane_deg": sorted(set(query_angles), reverse=True),
        "direction_octants_without_axis_zeros": octants,
        "all_six_faces_pass": len(set(query_faces)) == 6,
        "all_octants_pass": len(octants) == 8,
        "query_count": len(query_faces),
        "facet_count": atlas.facet_count,
        "preprocessing_nanoseconds": preprocessing_ns,
        "source_mapping_nanoseconds": source_mapping_ns,
        "cached_query_median_nanoseconds": cached_ns,
        "uncached_query_median_nanoseconds": uncached_ns,
        "scoring_nanoseconds": scoring_ns,
        "remapping_nanoseconds": remapping_ns,
        "total_cached_pipeline_nanoseconds": (
            preprocessing_ns
            + source_mapping_ns
            + cached_ns
            + scoring_ns
            + remapping_ns
        ),
        "cached_peak_python_tracemalloc_bytes": cached_peak_bytes,
        "query_speedup_ratio": uncached_ns / cached_ns,
        "cached_uncached_equivalence_pass": True,
        "cached_uncached_maximum_mu_difference": float(
            np.max(np.abs(cached["mu"] - uncached["mu"]))
        ),
        "maximum_reconstruction_residual_cm": cached_audit[
            "maximum_reconstruction_residual_cm"
        ],
        "incoming_record_count": cached_audit["incoming_records"],
        "grazing_record_count": cached_audit["grazing_records"],
        "source_conformal_production_geometry_used": False,
        "finite_tape_transport_solve_claimed": False,
        "physical_qualification_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    receipt = benchmark(arguments.repeats)
    if arguments.output is None:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        write_compact_json_create_only(arguments.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
