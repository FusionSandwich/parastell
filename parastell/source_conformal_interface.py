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

    expected = np.asarray(master.get("interface_triangles"), dtype=np.int64)
    actual = np.asarray(shared_triangles, dtype=np.int64)
    if expected.ndim != 2 or expected.shape[1:] != (3,):
        raise ValueError("master interface triangle inventory is malformed")
    if actual.ndim != 2 or actual.shape[1:] != (3,):
        raise ValueError("shared interface triangles must have shape (n,3)")
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
        "schema_pass": evidence.get("schema")
        == "parastell.source_conformal_physical_candidate/v1.0.0",
        "input_class_pass": evidence.get("input_class") == "PHYSICAL",
        "master_interface_bound_pass": evidence.get("master_interface_sha256")
        == master_interface.get("master_interface_sha256"),
        "dagmc_hash_pass": _valid_sha256(evidence.get("dagmc_h5m_sha256")),
        "outer_layer_mapping_hash_pass": _valid_sha256(
            evidence.get("outer_layer_mapping_sha256")
        ),
        "material_identity_hash_pass": _valid_sha256(
            evidence.get("material_identity_sha256")
        ),
        "full_geometry_topology_pass": evidence.get(
            "full_geometry_topology_pass"
        )
        is True,
        "fixture_evidence_rejected_pass": evidence.get("fixture_only")
        is False,
    }
    physical_candidate = all(physical_evidence_gates.values())
    transport_eligible = bool(constructional_pass and physical_candidate)
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
        "physical_candidate_evidence_gates": physical_evidence_gates,
        "physical_candidate": physical_candidate,
        "full_geometry_topology_pass": physical_evidence_gates[
            "full_geometry_topology_pass"
        ],
        "gates": gates,
        "constructional_containment_pass": constructional_pass,
        "transport_eligible": transport_eligible,
        **arrays,
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
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(destination)
    return {
        "path": str(destination.resolve()),
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }
