"""Source-mesh-master interface contracts for curved DAGMC construction.

The routines in this module never move, project, rescale, or otherwise alter
source coordinates.  They extract the exterior interface from tetrahedron
connectivity and provide fail-closed audits for a geometry producer that uses
that exact indexed triangle set as the single chamber/first-wall interface.

This module deliberately does not turn an independently faceted CAD surface
into a conformal one.  A producer must consume the returned master facets and
construct the remaining layer boundaries around them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .source_domain import (
    SOURCE_QUADRATURE_BARYCENTRICS,
    audit_source_tetrahedra_arrays,
)
from .source_geometry_identity import canonical_source_mesh_arrays

SOURCE_CONFORMAL_MANIFEST_SCHEMA = (
    "parastell.source_conformal_geometry_manifest/v1.0.0"
)
SOURCE_CONFORMAL_CONTAINMENT_SCHEMA = (
    "parastell.source_conformal_containment/v1.0.0"
)
SOURCE_CONFORMAL_V2_SCHEMA = (
    "parastell.source_conformal_physical_qualification/v2.0.0"
)
OUTER_LAYER_READBACK_SCHEMA = (
    "parastell.source_conformal_outer_layer_readback/v2.0.0"
)
MULTIREBCO_PRODUCER_REQUEST_SCHEMA = (
    "parastell.source_conformal_multirebco_producer_request/v2.1.0"
)
P4_PLAN_SCHEMA = "parastell.dagmc_targeted_p4_plan/v1.0.0"
P4_RECEIPT_SCHEMA = "parastell.dagmc_p4_receipt/v1.0.0"

P4_REGION_ORDER = (
    "SOURCE_CHAMBER_INTERFACE",
    "SECTOR_SEAMS",
    "HIGH_CURVATURE_LOCATIONS",
    "THIN_BLANKET_GAPS",
    "MATERIAL_INTERSECTIONS",
)


def load_indexed_source_mesh(path: str | Path) -> dict[str, Any]:
    """Load source H5M values while preserving its exact coordinate payload."""

    from pymoab import core, types

    source = Path(path).resolve()
    mesh = core.Core()
    mesh.load_file(str(source))
    root = mesh.get_root_set()
    vertex_handles = list(mesh.get_entities_by_type(root, types.MBVERTEX))
    tetrahedron_handles = list(mesh.get_entities_by_type(root, types.MBTET))
    if not vertex_handles or not tetrahedron_handles:
        raise ValueError("source H5M lacks vertices or tetrahedra")
    index_by_handle = {
        int(handle): index for index, handle in enumerate(vertex_handles)
    }
    connectivity_handles = np.asarray(
        mesh.get_connectivity(tetrahedron_handles), dtype=np.uint64
    ).reshape((-1, 4))
    try:
        connectivity = np.asarray(
            [
                [index_by_handle[int(handle)] for handle in row]
                for row in connectivity_handles
            ],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            "source tetrahedron references an unknown vertex"
        ) from exc
    coordinates = np.asarray(
        mesh.get_coords(vertex_handles), dtype=np.float64
    ).reshape((-1, 3))
    volume_tag = mesh.tag_get_handle("Volume")
    strength_tag = mesh.tag_get_handle("Source Strength")
    volumes = np.asarray(
        mesh.tag_get_data(volume_tag, tetrahedron_handles, flat=True),
        dtype=np.float64,
    )
    strengths = np.asarray(
        mesh.tag_get_data(strength_tag, tetrahedron_handles, flat=True),
        dtype=np.float64,
    )
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "vertices_cm": coordinates,
        "tetrahedra": connectivity,
        "tagged_volumes_cm3": volumes,
        "source_strengths_n_per_s": strengths,
    }


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_payload_sha256(label: str, value: Any, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(label.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validated_mesh_arrays(
    vertices_cm: Any, tetrahedra: Any
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices_cm, dtype=np.float64)
    cells = np.asarray(tetrahedra)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("vertices must have shape (n,3)")
    if not len(vertices) or not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must be nonempty and finite")
    if cells.ndim != 2 or cells.shape[1:] != (4,):
        raise ValueError("tetrahedra must have shape (n,4)")
    if not np.issubdtype(cells.dtype, np.integer):
        if not np.all(np.isfinite(cells)) or not np.all(
            cells == np.floor(cells)
        ):
            raise ValueError("tetrahedron connectivity must be integral")
    cells = np.asarray(cells, dtype=np.int64)
    if not len(cells) or np.any(cells < 0) or np.any(cells >= len(vertices)):
        raise ValueError("tetrahedron connectivity is empty or out of range")
    if any(len(set(int(value) for value in row)) != 4 for row in cells):
        raise ValueError("tetrahedron connectivity contains repeated vertices")
    determinant = np.linalg.det(
        np.stack(
            (
                vertices[cells[:, 1]] - vertices[cells[:, 0]],
                vertices[cells[:, 2]] - vertices[cells[:, 0]],
                vertices[cells[:, 3]] - vertices[cells[:, 0]],
            ),
            axis=1,
        )
    )
    if not np.all(np.isfinite(determinant)) or np.any(determinant == 0.0):
        raise ValueError("source mesh contains a degenerate tetrahedron")
    return vertices.copy(), cells.copy()


def _plane_distance(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle = math.radians(float(angle_degrees))
    return np.abs(
        -points[:, 0] * math.sin(angle) + points[:, 1] * math.cos(angle)
    )


def _face_payload_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    used = np.unique(faces)
    compact_index = {int(value): index for index, value in enumerate(used)}
    compact_faces = np.asarray(
        [[compact_index[int(value)] for value in row] for row in faces],
        dtype="<u8",
    )
    canonical_faces = np.sort(compact_faces, axis=1)
    canonical_faces = canonical_faces[
        np.lexsort(tuple(canonical_faces[:, i] for i in (2, 1, 0)))
    ]
    coordinates = np.asarray(vertices[used], dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"parastell.source_master_interface/v1\0")
    digest.update(np.asarray(coordinates.shape, dtype="<u8").tobytes())
    digest.update(coordinates.tobytes())
    digest.update(np.asarray(canonical_faces.shape, dtype="<u8").tobytes())
    digest.update(canonical_faces.tobytes())
    return digest.hexdigest()


def extract_source_master_interface(
    vertices_cm: Any,
    tetrahedra: Any,
    *,
    periodic_cut_plane_angles_degrees: Sequence[float] = (0.0, 90.0),
    geometric_tolerance_cm: float = 1.0e-9,
) -> dict[str, Any]:
    """Extract the source/chamber master facets from source connectivity.

    Boundary faces on the two declared periodic planes are sector seams.  All
    remaining source boundary faces form the master interface.  Each returned
    interface triangle is oriented out of the source volume.
    """

    vertices, cells = _validated_mesh_arrays(vertices_cm, tetrahedra)
    angles = np.asarray(
        periodic_cut_plane_angles_degrees, dtype=float
    ).reshape(-1)
    tolerance = float(geometric_tolerance_cm)
    if (
        angles.shape != (2,)
        or not np.all(np.isfinite(angles))
        or not angles[0] < angles[1]
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("source master-interface controls are invalid")

    local_faces = ((1, 2, 3, 0), (0, 3, 2, 1), (0, 1, 3, 2), (0, 2, 1, 3))
    incidence: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for cell in cells:
        for first, second, third, opposite in local_faces:
            face = [int(cell[first]), int(cell[second]), int(cell[third])]
            xyz = vertices[face]
            normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
            if float(normal @ (vertices[int(cell[opposite])] - xyz[0])) > 0.0:
                face[1], face[2] = face[2], face[1]
            incidence.setdefault(tuple(sorted(face)), []).append(tuple(face))

    nonmanifold = [key for key, owners in incidence.items() if len(owners) > 2]
    if nonmanifold:
        raise ValueError(
            f"source tetrahedra have {len(nonmanifold)} nonmanifold faces"
        )
    boundary = np.asarray(
        [owners[0] for owners in incidence.values() if len(owners) == 1],
        dtype=np.int64,
    )
    if not len(boundary):
        raise ValueError("source tetrahedra have no exterior boundary")
    seam_masks = []
    for angle in angles:
        distances = _plane_distance(vertices[boundary].reshape((-1, 3)), angle)
        seam_masks.append(
            np.all(distances.reshape((-1, 3)) <= tolerance, axis=1)
        )
    seam_mask = np.logical_or.reduce(seam_masks)
    interface = boundary[~seam_mask]
    seams = boundary[seam_mask]
    if not len(interface) or not len(seams):
        raise ValueError(
            "source boundary lacks interface or periodic seam facets"
        )

    xyz = vertices[interface]
    area_vectors = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    twice_area = np.linalg.norm(area_vectors, axis=1)
    if np.any(~np.isfinite(twice_area)) or np.any(twice_area <= 0.0):
        raise ValueError("source master interface contains degenerate facets")

    edge_counts: dict[tuple[int, int], int] = {}
    for face in interface:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(face[first]), int(face[second]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    invalid_edge_multiplicity = sorted(
        edge for edge, count in edge_counts.items() if count not in (1, 2)
    )
    open_edges = sorted(
        edge for edge, count in edge_counts.items() if count == 1
    )
    orphan_open_edges = []
    seam_edge_counts = [0, 0]
    for edge in open_edges:
        edge_points = vertices[list(edge)]
        matches = [
            bool(np.all(_plane_distance(edge_points, angle) <= tolerance))
            for angle in angles
        ]
        if not any(matches):
            orphan_open_edges.append(edge)
        for index, matched in enumerate(matches):
            seam_edge_counts[index] += int(matched)
    open_vertex_degree: dict[int, int] = {}
    for edge in open_edges:
        for vertex in edge:
            open_vertex_degree[vertex] = open_vertex_degree.get(vertex, 0) + 1
    invalid_seam_vertex_degree = []
    for vertex, degree in open_vertex_degree.items():
        plane_count = sum(
            float(_plane_distance(vertices[[vertex]], angle)[0]) <= tolerance
            for angle in angles
        )
        # A normal periodic-boundary vertex has degree two.  A vertex on the
        # line where both declared cut planes intersect contributes one pair
        # of boundary edges to each plane and therefore has degree four.
        if plane_count == 0 or degree != 2 * plane_count:
            invalid_seam_vertex_degree.append(vertex)
    invalid_seam_vertex_degree.sort()
    seam_continuity_pass = bool(
        not invalid_edge_multiplicity
        and not orphan_open_edges
        and not invalid_seam_vertex_degree
        and all(count > 0 for count in seam_edge_counts)
    )
    return {
        "schema": "parastell.source_master_interface/v1.0.0",
        "construction": "EXACT_IMMUTABLE_SOURCE_TETRAHEDRON_BOUNDARY",
        "coordinate_units": "cm",
        "periodic_cut_plane_angles_degrees": [
            float(value) for value in angles
        ],
        "geometric_tolerance_cm": tolerance,
        "source_vertex_count": int(len(vertices)),
        "source_tetrahedron_count": int(len(cells)),
        "source_boundary_triangle_count": int(len(boundary)),
        "interface_triangle_count": int(len(interface)),
        "periodic_seam_triangle_count": int(len(seams)),
        "interface_vertex_ids": [int(value) for value in np.unique(interface)],
        "interface_triangles": interface.tolist(),
        "periodic_seam_triangles": seams.tolist(),
        "interface_normal_unit_vectors": (
            area_vectors / twice_area[:, None]
        ).tolist(),
        "interface_open_edge_count": int(len(open_edges)),
        "periodic_seam_open_edge_counts": seam_edge_counts,
        "orphan_open_edges": [
            list(value) for value in orphan_open_edges[:100]
        ],
        "invalid_interface_edge_multiplicity": [
            list(value) for value in invalid_edge_multiplicity[:100]
        ],
        "invalid_seam_vertex_degree": invalid_seam_vertex_degree[:100],
        "sector_seam_continuity_pass": seam_continuity_pass,
        "master_interface_sha256": _face_payload_hash(vertices, interface),
        "coordinates_modified": False,
        "source_master_interface_pass": seam_continuity_pass,
    }


def audit_shared_interface_topology(
    master: Mapping[str, Any],
    *,
    shared_triangles: Any,
    surface_owners: Sequence[Sequence[str]],
    chamber_sense: int = 1,
    first_wall_sense: int = -1,
) -> dict[str, Any]:
    """Require one indexed facet set with exactly two opposite-sense owners."""

    expected = np.asarray(master.get("interface_triangles"))
    actual = np.asarray(shared_triangles)
    if expected.ndim != 2 or expected.shape[1:] != (3,):
        raise ValueError("master interface triangle inventory is malformed")
    if actual.ndim != 2 or actual.shape[1:] != (3,):
        raise ValueError("shared interface triangles must have shape (n,3)")
    for label, values in (("master", expected), ("shared", actual)):
        if not np.issubdtype(values.dtype, np.integer) and (
            not np.all(np.isfinite(values))
            or not np.all(values == np.floor(values))
        ):
            raise ValueError(
                f"{label} interface connectivity must be integral"
            )
    expected = np.asarray(expected, dtype=np.int64)
    actual = np.asarray(actual, dtype=np.int64)
    if np.any(expected < 0) or np.any(actual < 0):
        raise ValueError("interface connectivity cannot contain negative IDs")
    expected_keys = sorted(tuple(sorted(map(int, row))) for row in expected)
    actual_keys = sorted(tuple(sorted(map(int, row))) for row in actual)
    expected_orientation = {
        tuple(sorted(map(int, row))): tuple(map(int, row)) for row in expected
    }

    def same_cyclic_orientation(
        first: tuple[int, int, int], second: tuple[int, int, int]
    ) -> bool:
        return second in (
            first,
            (first[1], first[2], first[0]),
            (first[2], first[0], first[1]),
        )

    orientation_pass = len(actual) == len(expected) and all(
        key in expected_orientation
        and same_cyclic_orientation(
            expected_orientation[key], tuple(map(int, row))
        )
        for row in actual
        for key in (tuple(sorted(map(int, row))),)
    )
    ownership_rows = [
        tuple(str(value) for value in row) for row in surface_owners
    ]
    expected_owners = {"chamber", "first_wall"}
    owner_pass = bool(
        len(ownership_rows) == len(actual)
        and all(
            len(row) == 2 and set(row) == expected_owners
            for row in ownership_rows
        )
    )
    gates = {
        "master_interface_declared_pass": master.get(
            "source_master_interface_pass"
        )
        is True,
        "shared_connectivity_exact_pass": expected_keys == actual_keys,
        "outward_normal_orientation_pass": orientation_pass,
        "no_duplicate_interface_facets_pass": len(actual_keys)
        == len(set(actual_keys)),
        "surface_ownership_pass": owner_pass,
        "opposite_surface_senses_pass": int(chamber_sense) == 1
        and int(first_wall_sense) == -1,
    }
    return {
        "schema": "parastell.shared_source_chamber_interface/v1.0.0",
        "master_interface_sha256": master.get("master_interface_sha256"),
        "triangle_count": int(len(actual)),
        "owners": ["chamber", "first_wall"],
        "chamber_sense": int(chamber_sense),
        "first_wall_sense": int(first_wall_sense),
        "classification_rule": (
            "ON_SHARED_INTERFACE requires membership in this exact shared "
            "connectivity inventory; distance alone is never sufficient"
        ),
        "gates": gates,
        "shared_interface_topology_pass": all(gates.values()),
    }


def audit_sector_seam_continuity(
    vertices_cm: Any,
    interface_triangles: Any,
    *,
    periodic_cut_plane_angles_degrees: Sequence[float] = (0.0, 90.0),
    geometric_tolerance_cm: float = 1.0e-9,
) -> dict[str, Any]:
    """Require the two open sector boundaries to be rotation-periodic.

    This is a coordinate-and-connectivity check, rather than a declaration
    that both cut planes merely contain some facets.  The open interface edge
    set on the first plane is rigidly rotated about the global z axis and must
    match the complete edge set on the second plane one-to-one.
    """

    vertices = np.asarray(vertices_cm, dtype=np.float64)
    faces = np.asarray(interface_triangles)
    angles = np.asarray(
        periodic_cut_plane_angles_degrees, dtype=float
    ).reshape(-1)
    tolerance = float(geometric_tolerance_cm)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("vertices must have shape (n,3)")
    if not len(vertices) or not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must be nonempty and finite")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("interface triangles must have shape (n,3)")
    if not np.issubdtype(faces.dtype, np.integer):
        if not np.all(np.isfinite(faces)) or not np.all(
            faces == np.floor(faces)
        ):
            raise ValueError("interface connectivity must be integral")
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("interface connectivity is empty or out of range")
    if (
        angles.shape != (2,)
        or not np.all(np.isfinite(angles))
        or not angles[0] < angles[1]
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("sector-seam controls are invalid")

    edge_counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(face[first]), int(face[second]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    invalid_multiplicity = sorted(
        edge for edge, count in edge_counts.items() if count not in (1, 2)
    )
    open_edges = sorted(
        edge for edge, count in edge_counts.items() if count == 1
    )
    plane_edges: list[list[tuple[int, int]]] = [[], []]
    orphan_edges = []
    multiply_classified_edges = []
    for edge in open_edges:
        points = vertices[list(edge)]
        matches = [
            bool(np.all(_plane_distance(points, angle) <= tolerance))
            for angle in angles
        ]
        if sum(matches) == 0:
            orphan_edges.append(edge)
        elif sum(matches) > 1:
            multiply_classified_edges.append(edge)
        else:
            plane_edges[matches.index(True)].append(edge)

    def canonical_edge_coordinates(
        edges: Sequence[tuple[int, int]],
    ) -> np.ndarray:
        rows = []
        for edge in edges:
            points = np.asarray(vertices[list(edge)], dtype=np.float64)
            if tuple(points[1]) < tuple(points[0]):
                points = points[::-1]
            rows.append(points.reshape(-1))
        if not rows:
            return np.empty((0, 6), dtype=np.float64)
        values = np.asarray(rows)
        return values[
            np.lexsort(tuple(values[:, index] for index in range(5, -1, -1)))
        ]

    first = canonical_edge_coordinates(plane_edges[0]).reshape((-1, 2, 3))
    second = canonical_edge_coordinates(plane_edges[1]).reshape((-1, 2, 3))
    delta = math.radians(float(angles[1] - angles[0]))
    rotation = np.asarray(
        [
            [math.cos(delta), -math.sin(delta), 0.0],
            [math.sin(delta), math.cos(delta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_rows = []
    for edge_points in first:
        points = edge_points @ rotation.T
        if tuple(points[1]) < tuple(points[0]):
            points = points[::-1]
        rotated_rows.append(points.reshape(-1))
    rotated = np.asarray(rotated_rows, dtype=np.float64).reshape((-1, 6))
    if len(rotated):
        rotated = rotated[
            np.lexsort(tuple(rotated[:, index] for index in range(5, -1, -1)))
        ]
    second_rows = second.reshape((-1, 6))
    coordinate_match = bool(
        len(rotated) == len(second_rows)
        and len(rotated) > 0
        and np.allclose(rotated, second_rows, rtol=0.0, atol=tolerance)
    )
    gates = {
        "valid_open_edge_multiplicity_pass": not invalid_multiplicity,
        "all_open_edges_owned_by_one_seam_pass": not orphan_edges
        and not multiply_classified_edges,
        "seam_edge_count_match_pass": len(plane_edges[0])
        == len(plane_edges[1])
        and len(plane_edges[0]) > 0,
        "rigid_rotation_coordinate_match_pass": coordinate_match,
    }
    return {
        "schema": "parastell.periodic_sector_seam_audit/v1.0.0",
        "periodic_cut_plane_angles_degrees": [
            float(value) for value in angles
        ],
        "geometric_tolerance_cm": tolerance,
        "open_edge_count": len(open_edges),
        "plane_open_edge_counts": [len(value) for value in plane_edges],
        "orphan_open_edges": [list(value) for value in orphan_edges[:100]],
        "multiply_classified_open_edges": [
            list(value) for value in multiply_classified_edges[:100]
        ],
        "invalid_open_edge_multiplicity": [
            list(value) for value in invalid_multiplicity[:100]
        ],
        "gates": gates,
        "sector_seam_continuity_pass": all(gates.values()),
    }


def audit_source_conformal_containment_arrays(
    vertices_cm: Any,
    tetrahedra: Any,
    tagged_volumes_cm3: Any,
    source_strengths_n_per_s: Any,
    *,
    reference_vertices_cm: Any,
    reference_tetrahedra: Any,
    reference_tagged_volumes_cm3: Any,
    reference_source_strengths_n_per_s: Any,
    master_interface: Mapping[str, Any],
    shared_interface: Mapping[str, Any],
    source_strength_relative_tolerance: float = 1.0e-12,
    physical_candidate_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Constructional containment proof for an immutable source mesh.

    Quadrature points use strictly positive barycentric coordinates and are
    therefore inside their source tetrahedron.  A vertex is classified as
    ``ON_SHARED_INTERFACE`` only when its source vertex ID participates in the
    exact shared connectivity.  No distance-based forgiveness is used.
    """

    vertices, cells = _validated_mesh_arrays(vertices_cm, tetrahedra)
    reference_vertices, reference_cells = _validated_mesh_arrays(
        reference_vertices_cm, reference_tetrahedra
    )
    volumes = np.asarray(tagged_volumes_cm3, dtype=float).reshape(-1)
    strengths = np.asarray(source_strengths_n_per_s, dtype=float).reshape(-1)
    reference_volumes = np.asarray(
        reference_tagged_volumes_cm3, dtype=float
    ).reshape(-1)
    reference_strengths = np.asarray(
        reference_source_strengths_n_per_s, dtype=float
    ).reshape(-1)
    arrays = audit_source_tetrahedra_arrays(
        vertices[cells], volumes, strengths
    )
    identity = canonical_source_mesh_arrays(
        vertices, vertices[cells], strengths, volumes
    )
    reference_identity = canonical_source_mesh_arrays(
        reference_vertices,
        reference_vertices[reference_cells],
        reference_strengths,
        reference_volumes,
    )
    denominator = max(
        abs(reference_identity["total_source_strength_n_per_s"]), 1.0
    )
    relative_difference = (
        abs(
            identity["total_source_strength_n_per_s"]
            - reference_identity["total_source_strength_n_per_s"]
        )
        / denominator
    )
    semantic_identity_pass = (
        identity["canonical_fingerprint"]
        == reference_identity["canonical_fingerprint"]
    )
    interface_ids = set(
        int(value)
        for value in master_interface.get("interface_vertex_ids", [])
    )
    used_ids = set(int(value) for value in np.unique(cells))
    if not interface_ids or not interface_ids <= used_ids:
        raise ValueError("master interface vertex inventory is invalid")
    shared_pass = (
        shared_interface.get("shared_interface_topology_pass") is True
    )
    on_shared_count = sum(
        int(value in interface_ids) for value in cells.reshape(-1)
    )
    inside_vertex_count = int(cells.size - on_shared_count)
    quadrature_inside = bool(
        np.all(SOURCE_QUADRATURE_BARYCENTRICS > 0.0)
        and np.allclose(SOURCE_QUADRATURE_BARYCENTRICS.sum(axis=1), 1.0)
    )
    gates = {
        "tetrahedron_data_pass": arrays["tetrahedron_data_gate_pass"],
        "source_semantic_identity_pass": semantic_identity_pass,
        "source_strength_invariance_pass": relative_difference
        <= float(source_strength_relative_tolerance),
        "shared_interface_topology_pass": shared_pass,
        "sector_seam_continuity_pass": master_interface.get(
            "sector_seam_continuity_pass"
        )
        is True,
        "zero_outside_vertices_pass": True,
        "zero_outside_quadrature_pass": quadrature_inside,
        "zero_ambiguous_samples_pass": shared_pass,
    }
    constructional_pass = all(gates.values())
    evidence = dict(physical_candidate_evidence or {})
    physical_evidence_gates = {
        "post_export_signed_distance_audit_required": False,
        "post_export_normal_audit_required": False,
        "post_export_sector_seam_audit_required": False,
    }
    # This function proves properties of the immutable source packet only.  A
    # caller-provided mapping of hashes and booleans is not physical evidence
    # and must never promote that construction proof to transport eligibility.
    physical_candidate = False
    transport_eligible = False
    return {
        "schema": SOURCE_CONFORMAL_CONTAINMENT_SCHEMA,
        "status": (
            "PASS"
            if transport_eligible
            else "DIAGNOSTIC_ONLY_FULL_PHYSICAL_GEOMETRY_PENDING"
        ),
        "classification_basis": "SOURCE_TETRAHEDRON_AND_EXACT_SHARED_CONNECTIVITY",
        "distance_only_interface_classification_allowed": False,
        "vertex_occurrence_count": int(cells.size),
        "vertex_occurrence_classification": {
            "ON_SHARED_INTERFACE": int(on_shared_count),
            "INSIDE_BY_SOURCE_CONNECTIVITY": inside_vertex_count,
            "OUTSIDE": 0,
            "AMBIGUOUS": 0 if shared_pass else int(on_shared_count),
        },
        "quadrature_point_count": int(
            len(cells) * len(SOURCE_QUADRATURE_BARYCENTRICS)
        ),
        "quadrature_classification": {
            "INSIDE_BY_POSITIVE_BARYCENTRIC_COORDINATES": int(
                len(cells) * len(SOURCE_QUADRATURE_BARYCENTRICS)
            ),
            "OUTSIDE": 0,
            "AMBIGUOUS": 0,
        },
        "source_mesh_identity": identity,
        "reference_source_mesh_identity": reference_identity,
        "source_semantic_identity_pass": semantic_identity_pass,
        "source_strength_relative_difference": float(relative_difference),
        "source_strength_relative_tolerance": float(
            source_strength_relative_tolerance
        ),
        "coordinates_modified": False,
        "source_strengths_modified": False,
        "source_clipping_used": False,
        "source_projection_used": False,
        "source_rescaling_used": False,
        "provided_physical_candidate_evidence_ignored": bool(evidence),
        "physical_candidate_evidence_gates": physical_evidence_gates,
        "physical_candidate": physical_candidate,
        "full_geometry_topology_pass": False,
        "gates": gates,
        "constructional_containment_pass": constructional_pass,
        "transport_eligible": transport_eligible,
        **arrays,
    }


def qualify_source_conformal_physical_candidate_arrays(
    vertices_cm: Any,
    tetrahedra: Any,
    tagged_volumes_cm3: Any,
    source_strengths_n_per_s: Any,
    *,
    reference_vertices_cm: Any,
    reference_tetrahedra: Any,
    reference_tagged_volumes_cm3: Any,
    reference_source_strengths_n_per_s: Any,
    master_interface: Mapping[str, Any],
    actual_shared_vertex_coordinates_cm: Any,
    actual_shared_triangles: Any,
    surface_owners: Sequence[Sequence[str]],
    shared_normal_unit_vectors: Any,
    source_vertex_signed_distances_cm: Any,
    source_quadrature_signed_distances_cm: Any,
    observed_source_mesh_sha256: str,
    immutable_source_mesh_sha256: str,
    dagmc_h5m_sha256: str,
    outer_layer_mapping_sha256: str,
    material_identity_sha256: str,
    periodic_cut_plane_angles_degrees: Sequence[float] = (0.0, 90.0),
    geometric_tolerance_cm: float = 1.0e-9,
    source_strength_relative_tolerance: float = 1.0e-12,
    chamber_sense: int = 1,
    first_wall_sense: int = -1,
    input_class: str = "PHYSICAL",
    fixture_only: bool = False,
    prohibited_operations: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Qualify a post-export physical source-conformal candidate.

    Signed distance is preregistered as negative inside the intended chamber
    domain, zero on its boundary, and positive outside.  Interface vertices
    are accepted as ``ON_SHARED_INTERFACE`` only when they are present in the
    exact shared connectivity *and* their absolute signed distance is within
    the declared tolerance.  Distance alone can never create interface
    membership.
    """

    vertices, cells = _validated_mesh_arrays(vertices_cm, tetrahedra)
    reference_vertices, reference_cells = _validated_mesh_arrays(
        reference_vertices_cm, reference_tetrahedra
    )
    tolerance = float(geometric_tolerance_cm)
    strength_tolerance = float(source_strength_relative_tolerance)
    if (
        not math.isfinite(tolerance)
        or tolerance <= 0.0
        or not math.isfinite(strength_tolerance)
        or strength_tolerance < 0.0
    ):
        raise ValueError("containment tolerances are invalid")

    volumes = np.asarray(tagged_volumes_cm3, dtype=float).reshape(-1)
    strengths = np.asarray(source_strengths_n_per_s, dtype=float).reshape(-1)
    reference_volumes = np.asarray(
        reference_tagged_volumes_cm3, dtype=float
    ).reshape(-1)
    reference_strengths = np.asarray(
        reference_source_strengths_n_per_s, dtype=float
    ).reshape(-1)
    arrays = audit_source_tetrahedra_arrays(
        vertices[cells], volumes, strengths
    )
    identity = canonical_source_mesh_arrays(
        vertices, vertices[cells], strengths, volumes
    )
    reference_identity = canonical_source_mesh_arrays(
        reference_vertices,
        reference_vertices[reference_cells],
        reference_strengths,
        reference_volumes,
    )
    semantic_identity_pass = (
        identity["canonical_fingerprint"]
        == reference_identity["canonical_fingerprint"]
    )
    denominator = max(
        abs(reference_identity["total_source_strength_n_per_s"]), 1.0
    )
    strength_difference = (
        abs(
            identity["total_source_strength_n_per_s"]
            - reference_identity["total_source_strength_n_per_s"]
        )
        / denominator
    )

    shared = audit_shared_interface_topology(
        master_interface,
        shared_triangles=actual_shared_triangles,
        surface_owners=surface_owners,
        chamber_sense=chamber_sense,
        first_wall_sense=first_wall_sense,
    )
    actual_vertices = np.asarray(
        actual_shared_vertex_coordinates_cm, dtype=float
    )
    if actual_vertices.shape != vertices.shape:
        raise ValueError(
            "actual shared vertex coordinates must match source vertex shape"
        )
    if not np.all(np.isfinite(actual_vertices)):
        raise ValueError("actual shared vertex coordinates must be finite")
    actual_triangles = np.asarray(actual_shared_triangles, dtype=np.int64)
    if np.any(actual_triangles < 0) or np.any(
        actual_triangles >= len(vertices)
    ):
        raise ValueError("actual shared connectivity is out of source range")
    actual_normals = np.asarray(shared_normal_unit_vectors, dtype=float)
    if actual_normals.shape != (len(actual_triangles), 3):
        raise ValueError("shared normal vectors must have shape (n_facets,3)")
    if not np.all(np.isfinite(actual_normals)):
        raise ValueError("shared normal vectors must be finite")
    normal_norms = np.linalg.norm(actual_normals, axis=1)
    normals_normalized = bool(
        len(normal_norms)
        and np.allclose(normal_norms, 1.0, rtol=0.0, atol=1.0e-12)
    )
    expected_triangles = np.asarray(
        master_interface.get("interface_triangles"), dtype=np.int64
    )
    expected_normals = np.asarray(
        master_interface.get("interface_normal_unit_vectors"), dtype=float
    )
    if expected_normals.shape != (len(expected_triangles), 3):
        raise ValueError("master interface normal inventory is malformed")
    interface_ids_array = np.asarray(
        master_interface.get("interface_vertex_ids"), dtype=np.int64
    ).reshape(-1)
    if (
        not len(interface_ids_array)
        or np.any(interface_ids_array < 0)
        or np.any(interface_ids_array >= len(vertices))
    ):
        raise ValueError("master interface vertex inventory is invalid")
    shared_coordinates_exact = bool(
        len(interface_ids_array)
        and np.array_equal(
            actual_vertices[interface_ids_array], vertices[interface_ids_array]
        )
    )
    actual_xyz = actual_vertices[actual_triangles]
    actual_area_vectors = np.cross(
        actual_xyz[:, 1] - actual_xyz[:, 0],
        actual_xyz[:, 2] - actual_xyz[:, 0],
    )
    actual_twice_areas = np.linalg.norm(actual_area_vectors, axis=1)
    derived_normals_valid = bool(
        len(actual_twice_areas)
        and np.all(np.isfinite(actual_twice_areas))
        and np.all(actual_twice_areas > 0.0)
    )
    derived_normals = (
        actual_area_vectors / actual_twice_areas[:, None]
        if derived_normals_valid
        else np.full_like(actual_area_vectors, np.nan)
    )
    readback_normals_match_geometry = bool(
        derived_normals_valid
        and np.allclose(
            actual_normals, derived_normals, rtol=0.0, atol=1.0e-12
        )
    )
    expected_by_facet = {
        tuple(sorted(map(int, face))): expected_normals[index]
        for index, face in enumerate(expected_triangles)
    }
    normal_alignment = []
    for face, normal in zip(actual_triangles, actual_normals, strict=True):
        expected = expected_by_facet.get(tuple(sorted(map(int, face))))
        normal_alignment.append(
            float(expected @ normal) if expected is not None else -math.inf
        )
    normal_orientation_pass = bool(
        normals_normalized
        and readback_normals_match_geometry
        and normal_alignment
        and min(normal_alignment) >= 1.0 - 1.0e-12
    )

    seam = audit_sector_seam_continuity(
        actual_vertices,
        actual_triangles,
        periodic_cut_plane_angles_degrees=periodic_cut_plane_angles_degrees,
        geometric_tolerance_cm=tolerance,
    )
    vertex_distances = np.asarray(
        source_vertex_signed_distances_cm, dtype=float
    ).reshape(-1)
    quadrature_distances = np.asarray(
        source_quadrature_signed_distances_cm, dtype=float
    )
    expected_quadrature_shape = (
        len(cells),
        len(SOURCE_QUADRATURE_BARYCENTRICS),
    )
    if vertex_distances.shape != (len(vertices),):
        raise ValueError("source vertex signed distances have wrong shape")
    if quadrature_distances.shape != expected_quadrature_shape:
        raise ValueError("source quadrature signed distances have wrong shape")
    if not (
        np.all(np.isfinite(vertex_distances))
        and np.all(np.isfinite(quadrature_distances))
    ):
        raise ValueError("signed distances must be finite")

    interface_ids = set(
        int(value)
        for value in master_interface.get("interface_vertex_ids", [])
    )
    if not interface_ids or not interface_ids <= set(range(len(vertices))):
        raise ValueError("master interface vertex inventory is invalid")
    vertex_counts = {
        "ON_SHARED_INTERFACE": 0,
        "INSIDE": 0,
        "OUTSIDE": 0,
        "AMBIGUOUS": 0,
    }
    for vertex_id, distance in enumerate(vertex_distances):
        if vertex_id in interface_ids and abs(distance) <= tolerance:
            vertex_counts["ON_SHARED_INTERFACE"] += 1
        elif distance < -tolerance and vertex_id not in interface_ids:
            vertex_counts["INSIDE"] += 1
        elif distance > tolerance:
            vertex_counts["OUTSIDE"] += 1
        else:
            vertex_counts["AMBIGUOUS"] += 1
    quadrature_counts = {
        "INSIDE": int(np.count_nonzero(quadrature_distances < -tolerance)),
        "OUTSIDE": int(np.count_nonzero(quadrature_distances > tolerance)),
        "AMBIGUOUS": int(
            np.count_nonzero(np.abs(quadrature_distances) <= tolerance)
        ),
    }

    immutable_hash_pass = bool(
        _valid_sha256(observed_source_mesh_sha256)
        and _valid_sha256(immutable_source_mesh_sha256)
        and observed_source_mesh_sha256 == immutable_source_mesh_sha256
    )
    artifact_hashes_pass = all(
        _valid_sha256(value)
        for value in (
            dagmc_h5m_sha256,
            outer_layer_mapping_sha256,
            material_identity_sha256,
        )
    )
    operations = dict(prohibited_operations or {})
    prohibited_names = (
        "source_coordinate_projection",
        "source_mesh_rescaling",
        "source_strength_rescaling",
        "source_clipping",
        "blind_global_refinement",
    )
    no_prohibited_operations = bool(
        set(operations) == set(prohibited_names)
        and all(operations[name] is False for name in prohibited_names)
    )
    gates = {
        "physical_input_class_pass": input_class == "PHYSICAL"
        and fixture_only is False,
        "tetrahedron_data_pass": arrays["tetrahedron_data_gate_pass"],
        "immutable_input_hash_pass": immutable_hash_pass,
        "source_semantic_identity_pass": semantic_identity_pass,
        "source_strength_invariance_pass": strength_difference
        <= strength_tolerance,
        "artifact_hash_binding_pass": artifact_hashes_pass,
        "shared_interface_topology_pass": shared[
            "shared_interface_topology_pass"
        ],
        "shared_vertex_coordinates_exact_pass": shared_coordinates_exact,
        "normal_orientation_pass": normal_orientation_pass,
        "sector_seam_continuity_pass": seam["sector_seam_continuity_pass"],
        "zero_outside_vertices_pass": vertex_counts["OUTSIDE"] == 0,
        "zero_outside_quadrature_pass": quadrature_counts["OUTSIDE"] == 0,
        "zero_ambiguous_samples_pass": vertex_counts["AMBIGUOUS"] == 0
        and quadrature_counts["AMBIGUOUS"] == 0,
        "no_prohibited_operations_pass": no_prohibited_operations,
    }
    software_gates = {
        name: passed
        for name, passed in gates.items()
        if name != "physical_input_class_pass"
    }
    software_qualification_pass = all(software_gates.values())
    transport_eligible = all(gates.values())
    return {
        "schema": "parastell.source_conformal_physical_qualification/v1.0.0",
        "status": (
            "PASS"
            if transport_eligible
            else ("DIAGNOSTIC_ONLY" if software_qualification_pass else "FAIL")
        ),
        "input_class": input_class,
        "fixture_only": fixture_only,
        "signed_distance_convention": (
            "negative=inside, zero=boundary, positive=outside"
        ),
        "geometric_tolerance_cm": tolerance,
        "distance_only_interface_classification_allowed": False,
        "source_mesh_sha256": observed_source_mesh_sha256,
        "immutable_source_mesh_sha256": immutable_source_mesh_sha256,
        "dagmc_h5m_sha256": dagmc_h5m_sha256,
        "outer_layer_mapping_sha256": outer_layer_mapping_sha256,
        "material_identity_sha256": material_identity_sha256,
        "master_interface_sha256": master_interface.get(
            "master_interface_sha256"
        ),
        "source_mesh_canonical_fingerprint": identity["canonical_fingerprint"],
        "reference_source_mesh_canonical_fingerprint": reference_identity[
            "canonical_fingerprint"
        ],
        "source_strength_relative_difference": float(strength_difference),
        "source_strength_relative_tolerance": strength_tolerance,
        "vertex_count": len(vertices),
        "vertex_classification": vertex_counts,
        "quadrature_point_count": int(quadrature_distances.size),
        "quadrature_classification": quadrature_counts,
        "vertex_signed_distances_sha256": _array_payload_sha256(
            "parastell.source_conformal_vertex_signed_distance/v1",
            vertex_distances,
            "<f8",
        ),
        "quadrature_signed_distances_sha256": _array_payload_sha256(
            "parastell.source_conformal_quadrature_signed_distance/v1",
            quadrature_distances,
            "<f8",
        ),
        "shared_connectivity_sha256": _array_payload_sha256(
            "parastell.source_conformal_shared_connectivity/v1",
            actual_triangles,
            "<u8",
        ),
        "shared_vertex_coordinates_sha256": _array_payload_sha256(
            "parastell.source_conformal_shared_vertex_coordinates/v1",
            actual_vertices[interface_ids_array],
            "<f8",
        ),
        "shared_normals_sha256": _array_payload_sha256(
            "parastell.source_conformal_shared_normals/v1",
            actual_normals,
            "<f8",
        ),
        "minimum_normal_alignment": (
            float(min(normal_alignment)) if normal_alignment else None
        ),
        "shared_interface_audit": shared,
        "sector_seam_audit": seam,
        "prohibited_operations": operations,
        "software_qualification_pass": software_qualification_pass,
        "gates": gates,
        "transport_eligible": transport_eligible,
        **arrays,
    }


def _audit_exact_immutable_interface_readback(
    *,
    master_interface_sha256: str,
    expected_vertices_cm: Any | None,
    expected_triangles: Any | None,
    readback_vertices_cm: Any | None,
    readback_triangles: Any | None,
) -> dict[str, Any]:
    """Require bitwise-identical indexed interface arrays after H5M readback."""

    supplied = (
        expected_vertices_cm,
        expected_triangles,
        readback_vertices_cm,
        readback_triangles,
    )
    if all(value is None for value in supplied):
        return {
            "schema": "parastell.immutable_interface_readback/v1.0.0",
            "status": "BLOCKED_INPUT",
            "exact_immutable_interface_pass": False,
            "missing_inputs": [
                "master packet vertices_cm",
                "master packet interface_triangles",
                "post-export readback vertices_cm",
                "post-export readback interface_triangles",
            ],
        }
    if any(value is None for value in supplied):
        raise ValueError(
            "immutable-interface readback arrays must be supplied together"
        )
    expected_vertices = np.asarray(expected_vertices_cm, dtype="<f8")
    actual_vertices = np.asarray(readback_vertices_cm, dtype="<f8")
    expected_faces = np.asarray(expected_triangles)
    actual_faces = np.asarray(readback_triangles)
    for name, vertices in (
        ("master packet", expected_vertices),
        ("post-export readback", actual_vertices),
    ):
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or not np.all(np.isfinite(vertices))
        ):
            raise ValueError(
                f"{name} vertices must be finite with shape (n,3)"
            )
    normalized_faces = []
    for name, faces, vertex_count in (
        ("master packet", expected_faces, len(expected_vertices)),
        ("post-export readback", actual_faces, len(actual_vertices)),
    ):
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
            raise ValueError(
                f"{name} interface triangles must have shape (n,3)"
            )
        if not np.issubdtype(faces.dtype, np.integer) and (
            not np.all(np.isfinite(faces))
            or not np.all(faces == np.floor(faces))
        ):
            raise ValueError(f"{name} interface triangles must be integral")
        faces = np.asarray(faces, dtype="<i8")
        if np.any(faces < 0) or np.any(faces >= vertex_count):
            raise ValueError(
                f"{name} interface triangles reference invalid vertices"
            )
        if any(len({int(value) for value in row}) != 3 for row in faces):
            raise ValueError(
                f"{name} interface contains a degenerate triangle"
            )
        normalized_faces.append(faces)
    expected_faces, actual_faces = normalized_faces
    expected_hash = _face_payload_hash(expected_vertices, expected_faces)
    actual_hash = _face_payload_hash(actual_vertices, actual_faces)
    coordinates_exact = bool(
        expected_vertices.shape == actual_vertices.shape
        and np.array_equal(expected_vertices, actual_vertices)
    )
    connectivity_exact = bool(
        expected_faces.shape == actual_faces.shape
        and np.array_equal(expected_faces, actual_faces)
    )
    expected_hash_bound = expected_hash == master_interface_sha256
    readback_hash_bound = actual_hash == master_interface_sha256
    passed = bool(
        coordinates_exact
        and connectivity_exact
        and expected_hash_bound
        and readback_hash_bound
    )
    return {
        "schema": "parastell.immutable_interface_readback/v1.0.0",
        "status": "PASS" if passed else "FAIL",
        "coordinate_dtype": "float64-little-endian",
        "connectivity_dtype": "int64-little-endian",
        "vertex_count": len(actual_vertices),
        "triangle_count": len(actual_faces),
        "coordinates_bitwise_exact_pass": coordinates_exact,
        "connectivity_bitwise_exact_pass": connectivity_exact,
        "master_packet_recomputed_sha256": expected_hash,
        "readback_recomputed_sha256": actual_hash,
        "master_packet_hash_binding_pass": expected_hash_bound,
        "readback_hash_binding_pass": readback_hash_bound,
        "exact_immutable_interface_pass": passed,
    }


def _audit_multirebco_patch_tape_identity(
    *,
    family_rows: Sequence[Mapping[str, Any]] | None,
    patch_rows: Sequence[Mapping[str, Any]] | None,
    normalized_volumes: Sequence[Mapping[str, Any]],
    declared_surface_ids: set[int],
) -> dict[str, Any]:
    """Bind every REBCO facet patch to one family, tape, volume, and surface."""

    if family_rows is None and patch_rows is None:
        return {
            "schema": "parastell.multirebco_patch_tape_identity/v1.0.0",
            "status": "BLOCKED_INPUT",
            "multi_rebco_identity_pass": False,
            "missing_inputs": ["rebco_family_rows", "rebco_patch_rows"],
        }
    if family_rows is None or patch_rows is None:
        raise ValueError(
            "REBCO family and patch identity rows must be supplied together"
        )
    families = []
    family_ids = []
    tape_owner: dict[str, str] = {}
    for raw in family_rows:
        family_id = str(raw.get("family_id", ""))
        tape_stack_id = str(raw.get("tape_stack_id", ""))
        tape_stack_sha256 = str(raw.get("tape_stack_sha256", ""))
        material_id = str(raw.get("material_id", ""))
        tape_ids = [str(value) for value in raw.get("tape_ids", [])]
        if (
            not family_id
            or not tape_stack_id
            or not _valid_sha256(tape_stack_sha256)
            or not material_id
            or not tape_ids
            or any(not value for value in tape_ids)
            or len(tape_ids) != len(set(tape_ids))
        ):
            raise ValueError("REBCO family identity row is malformed")
        for tape_id in tape_ids:
            if tape_id in tape_owner:
                raise ValueError("REBCO tape IDs must be globally unique")
            tape_owner[tape_id] = family_id
        family_ids.append(family_id)
        families.append(
            {
                "family_id": family_id,
                "tape_stack_id": tape_stack_id,
                "tape_stack_sha256": tape_stack_sha256.lower(),
                "material_id": material_id,
                "tape_ids": sorted(tape_ids),
            }
        )
    tape_stack_ids = [row["tape_stack_id"] for row in families]
    if (
        len(families) < 2
        or len(family_ids) != len(set(family_ids))
        or len(tape_stack_ids) != len(set(tape_stack_ids))
    ):
        raise ValueError(
            "multi-REBCO qualification requires unique families and tape stacks"
        )
    family_by_id = {row["family_id"]: row for row in families}
    volume_by_id = {
        int(row["volume_global_id"]): row for row in normalized_volumes
    }
    normalized_patches = []
    patch_ids = []
    assigned_facets: set[str] = set()
    covered_tapes: set[str] = set()
    for raw in patch_rows:
        patch_id = str(raw.get("patch_id", ""))
        family_id = str(raw.get("family_id", ""))
        tape_id = str(raw.get("tape_id", ""))
        volume_id = int(raw.get("volume_global_id", 0))
        surface_ids = [
            int(value) for value in raw.get("surface_global_ids", [])
        ]
        facet_ids = [str(value) for value in raw.get("facet_ids", [])]
        facet_catalog_sha256 = str(raw.get("facet_catalog_sha256", ""))
        atlas_sha256 = str(raw.get("atlas_sha256", ""))
        mapping_sha256 = str(raw.get("mapping_sha256", ""))
        family = family_by_id.get(family_id)
        volume = volume_by_id.get(volume_id)
        valid = bool(
            patch_id
            and family is not None
            and tape_id
            and tape_owner.get(tape_id) == family_id
            and volume is not None
            and volume["material_id"] == family["material_id"]
            and surface_ids
            and len(surface_ids) == len(set(surface_ids))
            and set(surface_ids) <= declared_surface_ids
            and set(surface_ids) <= set(volume["surface_global_ids"])
            and facet_ids
            and all(facet_ids)
            and len(facet_ids) == len(set(facet_ids))
            and _valid_sha256(facet_catalog_sha256)
            and _valid_sha256(atlas_sha256)
            and _valid_sha256(mapping_sha256)
        )
        if not valid:
            raise ValueError(
                "REBCO patch identity row is malformed or inconsistent"
            )
        if assigned_facets.intersection(facet_ids):
            raise ValueError(
                "a facet cannot be assigned to multiple REBCO patches"
            )
        assigned_facets.update(facet_ids)
        covered_tapes.add(tape_id)
        patch_ids.append(patch_id)
        normalized_patches.append(
            {
                "patch_id": patch_id,
                "family_id": family_id,
                "tape_id": tape_id,
                "volume_global_id": volume_id,
                "surface_global_ids": sorted(surface_ids),
                "facet_ids": sorted(facet_ids),
                "facet_catalog_sha256": facet_catalog_sha256.lower(),
                "atlas_sha256": atlas_sha256.lower(),
                "mapping_sha256": mapping_sha256.lower(),
            }
        )
    if not normalized_patches or len(patch_ids) != len(set(patch_ids)):
        raise ValueError("REBCO patch IDs must be nonempty and unique")
    if len({row["facet_catalog_sha256"] for row in normalized_patches}) != 1:
        raise ValueError("all REBCO patches must bind one facet catalog")
    if len({row["atlas_sha256"] for row in normalized_patches}) != 1:
        raise ValueError("all REBCO patches must bind one cached facet atlas")
    if len({row["mapping_sha256"] for row in normalized_patches}) != 1:
        raise ValueError(
            "all REBCO patches must bind one conservative mapping"
        )
    missing_tapes = sorted(set(tape_owner) - covered_tapes)
    normalized_patches.sort(key=lambda row: row["patch_id"])
    families.sort(key=lambda row: row["family_id"])
    payload = {
        "schema": "parastell.multirebco_patch_tape_identity/v1.0.0",
        "families": families,
        "patches": normalized_patches,
        "family_count": len(families),
        "tape_count": len(tape_owner),
        "patch_count": len(normalized_patches),
        "facet_count": len(assigned_facets),
        "missing_tape_ids": missing_tapes,
        "unique_patch_ids_pass": len(patch_ids) == len(set(patch_ids)),
        "unique_facet_assignment_pass": True,
        "complete_tape_coverage_pass": not missing_tapes,
    }
    passed = bool(not missing_tapes)
    payload["multi_rebco_identity_pass"] = passed
    payload["status"] = "PASS" if passed else "FAIL"
    payload["identity_sha256"] = _canonical_json_sha256(payload)
    return payload


def build_source_conformal_multirebco_producer_request_v2(
    *,
    master_interface_sha256: str,
    master_interface_packet: Mapping[str, Any],
    requested_family_ids: Sequence[str],
    requested_tape_ids_by_family: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Create a compact fail-closed request when no physical candidate exists."""

    packet_sha256 = master_interface_packet.get("sha256")
    packet_path = str(master_interface_packet.get("path", ""))
    packet_size = int(master_interface_packet.get("size_bytes", 0))
    if (
        not _valid_sha256(master_interface_sha256)
        or not _valid_sha256(packet_sha256)
        or not packet_path
        or packet_size <= 0
    ):
        raise ValueError(
            "producer request requires immutable packet SHA-256 identities"
        )
    family_ids = [str(value) for value in requested_family_ids]
    if (
        len(family_ids) < 2
        or any(not value for value in family_ids)
        or len(family_ids) != len(set(family_ids))
    ):
        raise ValueError(
            "producer request requires at least two unique REBCO families"
        )
    unknown = set(requested_tape_ids_by_family) - set(family_ids)
    if unknown:
        raise ValueError(
            f"tape inventory contains unknown families: {sorted(unknown)}"
        )
    families = []
    global_tapes: set[str] = set()
    for family_id in family_ids:
        tapes = [
            str(value)
            for value in requested_tape_ids_by_family.get(family_id, [])
        ]
        if (
            not tapes
            or any(not value for value in tapes)
            or len(tapes) != len(set(tapes))
        ):
            raise ValueError(
                "every requested REBCO family needs unique nonempty tape IDs"
            )
        if global_tapes.intersection(tapes):
            raise ValueError("requested tape IDs must be globally unique")
        global_tapes.update(tapes)
        families.append({"family_id": family_id, "tape_ids": tapes})
    payload = {
        "schema": MULTIREBCO_PRODUCER_REQUEST_SCHEMA,
        "status": "BLOCKED_PHYSICAL_CANDIDATE",
        "transport_eligible": False,
        "p4_launch_authorized": False,
        "openmc_launch_authorized": False,
        "master_interface_sha256": master_interface_sha256.lower(),
        "master_interface_packet": dict(master_interface_packet),
        "requested_rebco_families": families,
        "required_candidate_readback": [
            "post-export DAGMC H5M path, size, and SHA-256",
            "bitwise float64 vertices_cm read back in master packet index order",
            "bitwise int64 interface_triangles read back in master packet order",
            "complete volume-to-surface and surface-to-volume/sense incidence",
            "one immutable family/tape/patch/facet/volume/surface identity row per patch",
            "family-tagged cached facet-atlas SHA-256 and facet-catalog SHA-256",
            "physical tape-stack SHA-256 and material identity for every family",
        ],
        "prohibited_operations": [
            "projecting, rescaling, clipping, reindexing, reordering, or remeshing the master interface",
            "merging distinct REBCO families, tapes, patches, or materials",
            "launching P4 or OpenMC before all physical gates pass",
        ],
        "missing_physical_inputs": [
            "physical post-export DAGMC candidate",
            "physical multi-REBCO family and tape-stack identities",
            "exact immutable-interface post-export readback arrays",
            "family-tagged patch/facet atlas mapping",
        ],
        "physical_qualification_claimed": False,
    }
    payload["request_sha256"] = _canonical_json_sha256(payload)
    return payload


def qualify_source_conformal_outer_layer_readback_v2(
    source_qualification: Mapping[str, Any],
    *,
    dagmc_h5m_sha256: str,
    master_interface_sha256: str,
    layer_order: Sequence[str],
    volume_rows: Sequence[Mapping[str, Any]],
    surface_rows: Sequence[Mapping[str, Any]],
    master_interface_surface_ids: Sequence[int],
    master_interface_owner_layer_ids: Sequence[str],
    expected_interface_vertices_cm: Any | None = None,
    expected_interface_triangles: Any | None = None,
    readback_interface_vertices_cm: Any | None = None,
    readback_interface_triangles: Any | None = None,
    rebco_family_rows: Sequence[Mapping[str, Any]] | None = None,
    rebco_patch_rows: Sequence[Mapping[str, Any]] | None = None,
    input_class: str = "FIXTURE",
    fixture_only: bool = True,
) -> dict[str, Any]:
    """Validate exact normalized post-export outer-layer topology readback.

    The V1 array qualifier remains the authority for immutable source
    coordinates, signed distances, and normals.  This additive V2 interface
    binds that result to the complete volume/surface/material incidence table
    read back from the written DAGMC candidate.
    """

    if not _valid_sha256(dagmc_h5m_sha256) or not _valid_sha256(
        master_interface_sha256
    ):
        raise ValueError("V2 readback requires geometry and interface SHA-256")
    layers = [str(value) for value in layer_order]
    if (
        not layers
        or any(not value for value in layers)
        or len(layers) != len(set(layers))
    ):
        raise ValueError("layer order must contain unique nonempty identities")

    normalized_volumes = []
    volume_ids = []
    for raw in volume_rows:
        volume_id = int(raw.get("volume_global_id", 0))
        layer_id = str(raw.get("layer_id", ""))
        material_id = str(raw.get("material_id", ""))
        surface_ids = [
            int(value) for value in raw.get("surface_global_ids", [])
        ]
        if (
            volume_id <= 0
            or layer_id not in layers
            or not material_id
            or not surface_ids
            or len(surface_ids) != len(set(surface_ids))
            or any(value <= 0 for value in surface_ids)
        ):
            raise ValueError("outer-layer volume readback row is malformed")
        volume_ids.append(volume_id)
        normalized_volumes.append(
            {
                "volume_global_id": volume_id,
                "layer_id": layer_id,
                "material_id": material_id,
                "surface_global_ids": sorted(surface_ids),
            }
        )
    if not normalized_volumes or len(volume_ids) != len(set(volume_ids)):
        raise ValueError("outer-layer volume IDs must be nonempty and unique")
    volume_set = set(volume_ids)
    layer_by_volume = {
        row["volume_global_id"]: row["layer_id"] for row in normalized_volumes
    }

    normalized_surfaces = []
    surface_ids = []
    owners_from_surfaces: dict[int, set[int]] = {}
    invalid_owner_rows = []
    for raw in surface_rows:
        surface_id = int(raw.get("surface_global_id", 0))
        owners = [int(value) for value in raw.get("owner_volume_ids", [])]
        senses = [int(value) for value in raw.get("owner_senses", [])]
        surface_ids.append(surface_id)
        valid = bool(
            surface_id > 0
            and len(owners) in (1, 2)
            and len(owners) == len(set(owners)) == len(senses)
            and set(owners) <= volume_set
            and all(value in (-1, 1) for value in senses)
            and (len(owners) == 1 or set(senses) == {-1, 1})
        )
        if not valid:
            invalid_owner_rows.append(surface_id)
        owners_from_surfaces[surface_id] = set(owners)
        normalized_surfaces.append(
            {
                "surface_global_id": surface_id,
                "owner_volume_ids": owners,
                "owner_senses": senses,
            }
        )
    duplicate_surface_ids = sorted(
        {value for value in surface_ids if surface_ids.count(value) > 1}
    )
    declared_surface_set = set(surface_ids)
    owners_from_volumes = {value: set() for value in declared_surface_set}
    unknown_volume_surface_ids = []
    for row in normalized_volumes:
        for surface_id in row["surface_global_ids"]:
            if surface_id not in owners_from_volumes:
                unknown_volume_surface_ids.append(surface_id)
            else:
                owners_from_volumes[surface_id].add(row["volume_global_id"])
    incidence_mismatches = sorted(
        surface_id
        for surface_id in declared_surface_set
        if owners_from_volumes[surface_id]
        != owners_from_surfaces.get(surface_id, set())
    )
    orphan_surfaces = sorted(
        surface_id
        for surface_id, owners in owners_from_surfaces.items()
        if not owners
    )
    layer_materials: dict[str, set[str]] = {value: set() for value in layers}
    for row in normalized_volumes:
        layer_materials[row["layer_id"]].add(row["material_id"])
    missing_layers = sorted(
        layer for layer, materials in layer_materials.items() if not materials
    )
    inconsistent_material_layers = sorted(
        layer
        for layer, materials in layer_materials.items()
        if len(materials) > 1
    )
    master_surfaces = [int(value) for value in master_interface_surface_ids]
    expected_master_owner_layers = [
        str(value) for value in master_interface_owner_layer_ids
    ]
    if (
        len(expected_master_owner_layers) != 2
        or len(set(expected_master_owner_layers)) != 2
        or not set(expected_master_owner_layers) <= set(layers)
    ):
        raise ValueError(
            "master interface requires two distinct declared owner layers"
        )
    valid_master_surface_inventory = bool(
        master_surfaces
        and len(master_surfaces) == len(set(master_surfaces))
        and set(master_surfaces) <= declared_surface_set
        and all(
            len(owners_from_surfaces.get(surface_id, set())) == 2
            for surface_id in master_surfaces
        )
    )
    master_owner_layers_pass = bool(
        valid_master_surface_inventory
        and all(
            {
                layer_by_volume[volume_id]
                for volume_id in owners_from_surfaces[surface_id]
            }
            == set(expected_master_owner_layers)
            for surface_id in master_surfaces
        )
    )
    source_schema_pass = (
        source_qualification.get("schema")
        == "parastell.source_conformal_physical_qualification/v1.0.0"
    )
    source_qualification_sha256 = _canonical_json_sha256(
        dict(source_qualification)
    )
    source_hash_binding_pass = bool(
        source_qualification.get("dagmc_h5m_sha256") == dagmc_h5m_sha256
        and source_qualification.get("master_interface_sha256")
        == master_interface_sha256
    )
    immutable_interface = _audit_exact_immutable_interface_readback(
        master_interface_sha256=master_interface_sha256,
        expected_vertices_cm=expected_interface_vertices_cm,
        expected_triangles=expected_interface_triangles,
        readback_vertices_cm=readback_interface_vertices_cm,
        readback_triangles=readback_interface_triangles,
    )
    multirebco_identity = _audit_multirebco_patch_tape_identity(
        family_rows=rebco_family_rows,
        patch_rows=rebco_patch_rows,
        normalized_volumes=normalized_volumes,
        declared_surface_ids=declared_surface_set,
    )
    topology_gates = {
        "unique_surface_ids_pass": bool(surface_ids)
        and not duplicate_surface_ids,
        "surface_owner_and_sense_pass": not invalid_owner_rows,
        "bidirectional_incidence_pass": not unknown_volume_surface_ids
        and not incidence_mismatches,
        "no_orphan_surfaces_pass": not orphan_surfaces,
        "layer_coverage_pass": not missing_layers,
        "one_material_per_layer_pass": not inconsistent_material_layers,
        "master_interface_surface_inventory_pass": valid_master_surface_inventory,
        "master_interface_owner_layers_pass": master_owner_layers_pass,
    }
    outer_layer_readback_pass = all(topology_gates.values())
    physical_input_class_pass = (
        input_class == "PHYSICAL" and fixture_only is False
    )
    source_transport_eligible_pass = (
        source_schema_pass
        and source_hash_binding_pass
        and source_qualification.get("transport_eligible") is True
    )
    gates = {
        "physical_input_class_pass": physical_input_class_pass,
        "source_v1_transport_eligible_pass": source_transport_eligible_pass,
        "outer_layer_readback_pass": outer_layer_readback_pass,
        "exact_immutable_interface_readback_pass": immutable_interface[
            "exact_immutable_interface_pass"
        ],
        "multi_rebco_patch_tape_identity_pass": multirebco_identity[
            "multi_rebco_identity_pass"
        ],
    }
    normalized_volumes.sort(key=lambda row: row["volume_global_id"])
    normalized_surfaces.sort(key=lambda row: row["surface_global_id"])
    readback_payload = {
        "schema": OUTER_LAYER_READBACK_SCHEMA,
        "dagmc_h5m_sha256": dagmc_h5m_sha256,
        "master_interface_sha256": master_interface_sha256,
        "layer_order": layers,
        "volumes": normalized_volumes,
        "surfaces": normalized_surfaces,
        "master_interface_surface_ids": sorted(master_surfaces),
        "master_interface_owner_layer_ids": expected_master_owner_layers,
        "immutable_interface_readback": immutable_interface,
        "multirebco_patch_tape_identity": multirebco_identity,
    }
    readback_sha256 = _canonical_json_sha256(readback_payload)
    transport_eligible = all(gates.values())
    software_qualification_pass = outer_layer_readback_pass
    return {
        "schema": SOURCE_CONFORMAL_V2_SCHEMA,
        "status": (
            "PASS"
            if transport_eligible
            else ("DIAGNOSTIC_ONLY" if software_qualification_pass else "FAIL")
        ),
        "input_class": input_class,
        "fixture_only": fixture_only,
        "dagmc_h5m_sha256": dagmc_h5m_sha256,
        "master_interface_sha256": master_interface_sha256,
        "source_v1_schema": source_qualification.get("schema"),
        "source_v1_qualification_sha256": source_qualification_sha256,
        "source_v1_transport_eligible": source_qualification.get(
            "transport_eligible"
        )
        is True,
        "outer_layer_readback": readback_payload,
        "outer_layer_readback_sha256": readback_sha256,
        "topology_gates": topology_gates,
        "duplicate_surface_ids": duplicate_surface_ids,
        "invalid_owner_surface_ids": sorted(invalid_owner_rows),
        "unknown_volume_surface_ids": sorted(set(unknown_volume_surface_ids)),
        "incidence_mismatch_surface_ids": incidence_mismatches,
        "orphan_surface_ids": orphan_surfaces,
        "missing_layer_ids": missing_layers,
        "inconsistent_material_layer_ids": inconsistent_material_layers,
        "immutable_interface_readback": immutable_interface,
        "multirebco_patch_tape_identity": multirebco_identity,
        "software_qualification_pass": software_qualification_pass,
        "gates": gates,
        "transport_eligible": transport_eligible,
        "physical_qualification_claimed": transport_eligible,
    }


def write_source_conformal_v2_receipts_create_only(
    output_dir: str | Path,
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish compact V2 manifest/readback receipts without overwriting."""

    if qualification.get("schema") != SOURCE_CONFORMAL_V2_SCHEMA:
        raise ValueError("unsupported source-conformal V2 qualification")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"create-only V2 output exists: {destination}")
    destination.mkdir(parents=True)
    manifest = {
        "schema": "parastell.source_conformal_geometry_manifest/v2.0.0",
        "status": qualification.get("status"),
        "input_class": qualification.get("input_class"),
        "fixture_only": qualification.get("fixture_only"),
        "dagmc_h5m_sha256": qualification.get("dagmc_h5m_sha256"),
        "master_interface_sha256": qualification.get(
            "master_interface_sha256"
        ),
        "outer_layer_readback_sha256": qualification.get(
            "outer_layer_readback_sha256"
        ),
        "source_v1_qualification_sha256": qualification.get(
            "source_v1_qualification_sha256"
        ),
        "exact_immutable_interface_readback_pass": qualification.get(
            "gates", {}
        ).get("exact_immutable_interface_readback_pass")
        is True,
        "multi_rebco_patch_tape_identity_pass": qualification.get(
            "gates", {}
        ).get("multi_rebco_patch_tape_identity_pass")
        is True,
        "multirebco_identity_sha256": qualification.get(
            "multirebco_patch_tape_identity", {}
        ).get("identity_sha256"),
        "transport_eligible": qualification.get("transport_eligible") is True,
        "physical_qualification_claimed": qualification.get(
            "physical_qualification_claimed"
        )
        is True,
    }
    manifest_identity = write_compact_json_create_only(
        destination / "SOURCE_CONFORMAL_GEOMETRY_MANIFEST_V2.json", manifest
    )
    qualification_identity = write_compact_json_create_only(
        destination / "SOURCE_CONFORMAL_OUTER_LAYER_READBACK_RECEIPT_V2.json",
        qualification,
    )
    return {
        "output_dir": str(destination.resolve()),
        "manifest": manifest_identity,
        "qualification": qualification_identity,
    }


def build_targeted_p4_plan(
    *,
    geometry_sha256: str,
    interface_sha256: str,
    timeout_seconds_per_region: int,
    region_bounds_cm: Mapping[str, Sequence[float] | None] | None = None,
) -> dict[str, Any]:
    """Create a deterministic, resumable precision-4 diagnostic plan."""

    if not (
        _valid_sha256(geometry_sha256) and _valid_sha256(interface_sha256)
    ):
        raise ValueError("P4 inputs require SHA-256 identities")
    timeout = int(timeout_seconds_per_region)
    if timeout <= 0:
        raise ValueError("P4 timeout must be positive")
    bounds = dict(region_bounds_cm or {})
    unknown = set(bounds) - set(P4_REGION_ORDER)
    if unknown:
        raise ValueError(f"unknown P4 regions: {sorted(unknown)}")
    regions = []
    for index, region_id in enumerate(P4_REGION_ORDER):
        value = bounds.get(region_id)
        if value is not None:
            value = [float(item) for item in value]
            if len(value) != 6 or not all(
                math.isfinite(item) for item in value
            ):
                raise ValueError(f"invalid bounds for P4 region {region_id}")
        regions.append(
            {
                "sequence": index,
                "region_id": region_id,
                "bounds_cm": value,
                "precision": 4,
                "timeout_seconds": timeout,
                "status": "PENDING",
            }
        )
    plan = {
        "schema": P4_PLAN_SCHEMA,
        "geometry_sha256": geometry_sha256,
        "master_interface_sha256": interface_sha256,
        "precision": 4,
        "resume_key_fields": [
            "geometry_sha256",
            "master_interface_sha256",
            "region_id",
            "precision",
        ],
        "regions": regions,
        "global_p4_allowed_only_after_all_targeted_pass": True,
    }
    plan["canonical_plan_sha256"] = _canonical_json_sha256(plan)
    return plan


def summarize_p4_results(
    plan: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Merge resumable region results without treating timeout as failure."""

    if plan.get("schema") != P4_PLAN_SCHEMA:
        raise ValueError("unsupported P4 plan schema")
    expected = [str(row["region_id"]) for row in plan.get("regions", [])]
    by_region: dict[str, Mapping[str, Any]] = {}
    allowed_statuses = {"PASS", "FAIL", "TIMEOUT", "ERROR", "UNEXECUTED"}
    for row in results:
        region = str(row.get("region_id"))
        if region not in expected or region in by_region:
            raise ValueError(
                "P4 results contain an unknown or duplicate region"
            )
        if row.get("status") not in allowed_statuses:
            raise ValueError("P4 result status is unsupported")
        if int(row.get("precision", -1)) != 4:
            raise ValueError("P4 result precision is not 4")
        if (
            row.get("geometry_sha256") != plan.get("geometry_sha256")
            or row.get("master_interface_sha256")
            != plan.get("master_interface_sha256")
            or row.get("plan_sha256") != plan.get("canonical_plan_sha256")
        ):
            raise ValueError("P4 result is not hash-bound to the plan")
        by_region[region] = dict(row)
    rows = []
    for region in expected:
        row = dict(
            by_region.get(
                region,
                {"region_id": region, "precision": 4, "status": "UNEXECUTED"},
            )
        )
        rows.append(row)
    all_pass = bool(rows and all(row["status"] == "PASS" for row in rows))
    status = (
        "PASS"
        if all_pass
        else (
            "BLOCKED_RUNTIME"
            if any(row["status"] == "TIMEOUT" for row in rows)
            else (
                "FAIL"
                if any(row["status"] == "FAIL" for row in rows)
                else "UNEXECUTED_DEPENDENCY"
            )
        )
    )
    return {
        "schema": P4_RECEIPT_SCHEMA,
        "scope": "TARGETED",
        "geometry_sha256": plan["geometry_sha256"],
        "master_interface_sha256": plan["master_interface_sha256"],
        "plan_sha256": plan["canonical_plan_sha256"],
        "precision": 4,
        "status": status,
        "regions": rows,
        "completed_region_count": sum(
            row["status"] not in {"UNEXECUTED"} for row in rows
        ),
        "targeted_p4_pass": all_pass,
        "global_p4_allowed": all_pass,
        "timeout_is_physics_failure": False,
    }


def write_compact_json_create_only(
    path: str | Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically create a compact receipt and return its identity."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"create-only receipt exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary receipt exists: {temporary}")
    payload = dict(value)
    payload.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard-link publication is atomic and, unlike Path.replace(), fails
        # if another writer creates the destination after the checks above.
        os.link(temporary, destination)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return {
        "path": str(destination.resolve()),
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }
