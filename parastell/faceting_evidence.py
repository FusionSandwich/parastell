"""Hash-bound evidence collection for matched DAGMC faceting levels.

The functions in this module measure a completed DAGMC mesh.  They never
classify faceting convergence and never modify the H5M or source CAD.  The
consumer remains responsible for comparing coarse and refined certificates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .reference_geometry import sha256_file
from .reference_geometry import audit_volume_envelope


EVIDENCE_SCHEMA = "parastell.faceting_evidence/v1.0.0"
SNAPSHOT_SCHEMA = "parastell.faceting_snapshot/v1.0.0"
DEVIATION_SCHEMA = "parastell.facet_deviation_certificate/v1.0.0"
SOURCE_MAP_SCHEMA = "parastell.faceting_source_map/v1.0.0"
SEAL_SCHEMA = "parastell.faceting_evidence_seal/v1.0.0"
EXPECTED_PROTOCOL_SHA256 = (
    "d1f4216a279ffe6608e3b334d85828bb126036833e1d17aa241880c73ec8cacf"
)
EXPECTED_PUBLIC_OVERLAP_REPORT_SHA256 = (
    "f30fd91d4ed6f797e498acc600138b5006f335ac698bfdaad8da8af66ccb7f62"
)
MAXIMUM_COVER_RADIUS_CM = 0.25
SOLVER_MARGIN_CM = 1.0e-7
FORMER_ZONE_RADIUS_CM = 40.0
HARD_MAXIMUM_PROJECTED_SUBTRIANGLE_CENTROIDS = 5_000_000
REQUIRED_FORMER_ZONE_IDS = (
    "magnet-0005",
    "magnet-0006",
    "magnet-0011",
    "magnet-0012",
)
EXPECTED_SEMANTIC_MATERIALS = {
    "Vacuum": "Vacuum",
    "first_wall": "first_wall",
    "breeder": "breeder",
    "back_wall": "back_wall",
    "shield": "shield",
    "vac_vessel": "vac_vessel",
    **{f"magnet-{index:04d}": "magnets" for index in range(18)},
}
EXPECTED_SOURCE_TARGETS = {
    "Vacuum": ("chamber.step", 0),
    "first_wall": ("first_wall.step", 0),
    "breeder": ("breeder.step", 0),
    "back_wall": ("back_wall.step", 0),
    "shield": ("shield.step", 0),
    "vac_vessel": ("vacuum_vessel.step", 0),
    **{
        f"magnet-{index:04d}": ("magnet_set.step", index)
        for index in range(18)
    },
}
_EXPECTED_GEOMETRY_DIMENSION = {
    "Vertex": 0,
    "Curve": 1,
    "Surface": 2,
    "Volume": 3,
    # DAGMC uses -1 for semantic Group meshsets.  MBENTITYSET's MOAB
    # entity-type value (4) is not a GEOM_DIMENSION value.
    "Group": -1,
}


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict:
    if not _valid_sha256(expected_sha256):
        raise ValueError(f"{label} expected SHA-256 is invalid")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} hash mismatch: expected {expected_sha256}, got {actual}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def _triangle(value: Any) -> np.ndarray:
    triangle = np.asarray(value, dtype=float)
    if triangle.shape != (3, 3) or not np.all(np.isfinite(triangle)):
        raise ValueError(
            "triangle must have shape (3, 3) and finite coordinates"
        )
    twice_area = np.linalg.norm(
        np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    )
    if not math.isfinite(float(twice_area)) or float(twice_area) <= 0.0:
        raise ValueError("triangle must be nondegenerate")
    return triangle


def _maximum_edge_cm(triangle: np.ndarray) -> float:
    return max(
        float(np.linalg.norm(triangle[first] - triangle[second]))
        for first, second in ((0, 1), (1, 2), (2, 0))
    )


def _centroid_cover_radius_cm(triangle: np.ndarray) -> float:
    centroid = triangle.mean(axis=0)
    return max(float(np.linalg.norm(vertex - centroid)) for vertex in triangle)


def _bisect_longest_edge(
    triangle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edges = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    first, second, opposite = max(
        edges,
        key=lambda row: float(
            np.linalg.norm(triangle[row[0]] - triangle[row[1]])
        ),
    )
    midpoint = 0.5 * (triangle[first] + triangle[second])
    return (
        np.asarray(
            [triangle[first], midpoint, triangle[opposite]], dtype=float
        ),
        np.asarray(
            [midpoint, triangle[second], triangle[opposite]], dtype=float
        ),
    )


def _leaf_triangles(
    triangle: Any,
    *,
    maximum_cover_radius_cm: float = MAXIMUM_COVER_RADIUS_CM,
    maximum_depth: int = 64,
) -> Iterable[tuple[np.ndarray, float]]:
    """Yield a centroid-covering subdivision of one original triangle."""
    source = _triangle(triangle)
    radius_limit = float(maximum_cover_radius_cm)
    if not math.isfinite(radius_limit) or radius_limit <= 0.0:
        raise ValueError("maximum cover radius must be finite and positive")
    if isinstance(maximum_depth, bool) or int(maximum_depth) < 1:
        raise ValueError("maximum depth must be a positive integer")
    stack: list[tuple[np.ndarray, int]] = [(source, 0)]
    while stack:
        candidate, depth = stack.pop()
        radius = _centroid_cover_radius_cm(candidate)
        if radius <= radius_limit:
            yield candidate, radius
            continue
        if depth >= int(maximum_depth):
            raise RuntimeError("triangle subdivision exceeded maximum depth")
        first, second = _bisect_longest_edge(candidate)
        stack.append((second, depth + 1))
        stack.append((first, depth + 1))


@dataclass(frozen=True)
class TriangleIncidence:
    """One DAGMC triangle as seen from one semantic volume boundary."""

    triangle_handle: int
    surface_global_id: int
    volume_global_id: int
    semantic_id: str
    material: str
    coordinates_cm: np.ndarray


def _canonical_triangle_vertices(triangle: np.ndarray) -> list[list[float]]:
    rows = [tuple(float(value) for value in vertex) for vertex in triangle]
    return [list(row) for row in sorted(rows)]


def _triangle_incidence_digest(rows: Sequence[TriangleIncidence]) -> str:
    payload = [
        {
            "triangle_handle": int(row.triangle_handle),
            "surface_global_id": int(row.surface_global_id),
            "volume_global_id": int(row.volume_global_id),
            "semantic_id": row.semantic_id,
            "material": row.material,
            "vertices_cm": _canonical_triangle_vertices(row.coordinates_cm),
        }
        for row in sorted(
            rows,
            key=lambda item: (
                item.volume_global_id,
                item.surface_global_id,
                item.triangle_handle,
                item.semantic_id,
            ),
        )
    ]
    return _json_sha256(payload)


def _physical_triangle_digest(rows: Sequence[TriangleIncidence]) -> str:
    by_handle: dict[int, TriangleIncidence] = {}
    for row in rows:
        prior = by_handle.get(row.triangle_handle)
        if prior is not None and (
            prior.surface_global_id != row.surface_global_id
            or not np.array_equal(prior.coordinates_cm, row.coordinates_cm)
        ):
            raise ValueError("one triangle handle has inconsistent geometry")
        by_handle[row.triangle_handle] = row
    payload = [
        {
            "triangle_handle": handle,
            "surface_global_id": row.surface_global_id,
            "vertices_cm": _canonical_triangle_vertices(row.coordinates_cm),
        }
        for handle, row in sorted(by_handle.items())
    ]
    return _json_sha256(payload)


def _measure_triangle_deviation(
    triangle: Any,
    projector: Callable[[Sequence[float]], float],
    *,
    maximum_cover_radius_cm: float = MAXIMUM_COVER_RADIUS_CM,
) -> dict[str, Any]:
    """Measure a certified one-sided bound for one original triangle."""
    maximum_distance = 0.0
    actual_cover_radius = 0.0
    leaf_count = 0
    for leaf, cover_radius in _leaf_triangles(
        triangle, maximum_cover_radius_cm=maximum_cover_radius_cm
    ):
        distance = float(projector(leaf.mean(axis=0)))
        if not math.isfinite(distance) or distance < 0.0:
            raise RuntimeError("source-boundary projection was not finite")
        maximum_distance = max(maximum_distance, distance)
        actual_cover_radius = max(actual_cover_radius, cover_radius)
        leaf_count += 1
    if leaf_count == 0:
        raise RuntimeError("triangle subdivision produced no covering samples")
    return {
        "leaf_sample_count": leaf_count,
        "actual_maximum_cover_radius_cm": actual_cover_radius,
        "maximum_projected_distance_cm": maximum_distance,
    }


def _estimate_leaf_sample_count(
    rows: Sequence[TriangleIncidence],
    *,
    maximum_cover_radius_cm: float = MAXIMUM_COVER_RADIUS_CM,
    maximum_count: int = HARD_MAXIMUM_PROJECTED_SUBTRIANGLE_CENTROIDS,
) -> int:
    if isinstance(maximum_count, bool) or int(maximum_count) < 1:
        raise ValueError(
            "maximum leaf-sample count must be a positive integer"
        )
    count = 0
    for row in rows:
        for _leaf, _radius in _leaf_triangles(
            row.coordinates_cm,
            maximum_cover_radius_cm=maximum_cover_radius_cm,
        ):
            count += 1
            if count > int(maximum_count):
                raise RuntimeError(
                    "projection-centroid preflight exceeds the declared "
                    f"bounded limit: > {int(maximum_count)}"
                )
    return count


def _certificate_arithmetic(
    *,
    maximum_projected_distance_cm: float,
    actual_maximum_cover_radius_cm: float,
    maximum_brep_entity_tolerance_cm: float,
    solver_margin_cm: float = SOLVER_MARGIN_CM,
) -> dict[str, Any]:
    values = (
        maximum_projected_distance_cm,
        actual_maximum_cover_radius_cm,
        maximum_brep_entity_tolerance_cm,
        solver_margin_cm,
    )
    if not all(_finite_nonnegative(value) for value in values):
        raise ValueError(
            "deviation-bound terms must be finite and nonnegative"
        )
    if float(solver_margin_cm) != SOLVER_MARGIN_CM:
        raise ValueError("solver margin must be exactly 1e-7 cm")
    terms = [float(value) for value in values]
    upper_bound = math.fsum(terms)
    return {
        "formula": (
            "maximum_projected_distance_cm + "
            "actual_maximum_cover_radius_cm + solver_margin_cm + "
            "maximum_brep_entity_tolerance_cm"
        ),
        "summation": "math.fsum",
        "maximum_projected_distance_cm": float(maximum_projected_distance_cm),
        "actual_maximum_cover_radius_cm": float(
            actual_maximum_cover_radius_cm
        ),
        "solver_margin_cm": float(solver_margin_cm),
        "maximum_brep_entity_tolerance_cm": float(
            maximum_brep_entity_tolerance_cm
        ),
        "upper_bound_cm": upper_bound,
        "arithmetic_reconstruction_pass": upper_bound == math.fsum(terms),
    }


@dataclass(frozen=True)
class SourceTarget:
    """A validated semantic source-CAD boundary projection target."""

    volume_global_id: int
    semantic_id: str
    material: str
    source_step: str
    source_step_sha256: str
    source_solid_index: int
    boundary_shape: Any
    maximum_entity_tolerance_cm: float
    brep_entity_count: int


def _safe_relative_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            "source STEP path escapes the declared root"
        ) from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _brep_tolerance(shape: Any) -> float:
    from OCP.BRep import BRep_Tool

    tolerance_method = getattr(BRep_Tool, "Tolerance_s", None)
    if tolerance_method is None:
        tolerance_method = getattr(BRep_Tool, "Tolerance")
    tolerance = float(tolerance_method(shape))
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise RuntimeError("OpenCascade reported an invalid entity tolerance")
    return tolerance


def _source_boundary(solid: Any) -> tuple[Any, float, int]:
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    faces = list(solid.Faces())
    edges = list(solid.Edges())
    vertices = list(solid.Vertices())
    if not faces:
        raise ValueError("source solid has no BRep faces")
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for face in faces:
        builder.Add(compound, face.wrapped)
    entities = [*faces, *edges, *vertices]
    tolerances = [_brep_tolerance(entity.wrapped) for entity in entities]
    return compound, max(tolerances), len(entities)


def _cadquery_solid_signature(solid: Any) -> dict[str, Any]:
    import cadquery as cq

    bounds = solid.BoundingBox()
    return {
        "mass_cm3": float(solid.Volume()),
        "center_cm": [float(value) for value in solid.Center().toTuple()],
        "bounding_box_cm": [
            float(bounds.xmin),
            float(bounds.ymin),
            float(bounds.zmin),
            float(bounds.xmax),
            float(bounds.ymax),
            float(bounds.zmax),
        ],
        "inertia_cm5": [
            float(value)
            for row in cq.Shape.matrixOfInertia(solid)
            for value in row
        ],
    }


def _source_signature_matches_sealed(
    actual: Mapping[str, Any], sealed: Mapping[str, Any]
) -> bool:
    lengths = {
        "center_cm": 3,
        "bounding_box_cm": 6,
        "inertia_cm5": 9,
    }
    try:
        actual_mass = float(actual["mass_cm3"])
        sealed_mass = float(sealed["mass_cm3"])
        if (
            not math.isfinite(actual_mass)
            or not math.isfinite(sealed_mass)
            or actual_mass <= 0.0
            or sealed_mass <= 0.0
        ):
            return False
        if not math.isclose(
            actual_mass,
            sealed_mass,
            rel_tol=1.0e-12,
            abs_tol=1.0e-8,
        ):
            return False
        for name, length in lengths.items():
            first = list(actual[name])
            second = list(sealed[name])
            if len(first) != length or len(second) != length:
                return False
            if not all(
                math.isfinite(float(value)) for value in (*first, *second)
            ):
                return False
            if not all(
                math.isclose(
                    float(a),
                    float(b),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-8,
                )
                for a, b in zip(first, second)
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _load_source_targets(
    *,
    source_root: Path,
    source_map: Mapping[str, Any],
    volume_materials: Mapping[int, str],
    refaceting_manifest: Mapping[str, Any] | None = None,
) -> tuple[
    dict[int, SourceTarget], list[dict[str, Any]], list[dict[str, Any]]
]:
    if source_map.get("schema") != SOURCE_MAP_SCHEMA:
        raise ValueError("unsupported faceting source-map schema")
    if source_map.get("coordinate_units") != "cm":
        raise ValueError("faceting source map must declare centimetres")
    rows = source_map.get("volumes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("faceting source map has no volume rows")
    global_ids = [int(row.get("volume_global_id", 0)) for row in rows]
    semantic_ids = [str(row.get("semantic_id", "")).strip() for row in rows]
    if any(value <= 0 for value in global_ids):
        raise ValueError("source-map volume IDs must be positive")
    if len(global_ids) != len(set(global_ids)):
        raise ValueError("source-map volume IDs must be unique")
    if any(not value for value in semantic_ids):
        raise ValueError("source-map semantic IDs must be nonempty")
    if len(semantic_ids) != len(set(semantic_ids)):
        raise ValueError("source-map semantic IDs must be unique")
    if set(global_ids) != set(volume_materials):
        raise ValueError("source-map volume inventory differs from DAGMC")
    if set(semantic_ids) != set(EXPECTED_SEMANTIC_MATERIALS):
        raise ValueError(
            "source-map semantic inventory differs from ParaStell"
        )
    sealed_signature_by_semantic: dict[str, Mapping[str, Any]] = {}
    if refaceting_manifest is not None:
        mapping_evidence = sorted(
            refaceting_manifest.get("volume_mapping_evidence", []),
            key=lambda row: int(row.get("index", -1)),
        )
        written_order = refaceting_manifest.get(
            "written_h5m_material_order_evidence", {}
        )
        ordered_global_ids = written_order.get("volume_global_ids", [])
        if (
            len(mapping_evidence) != len(EXPECTED_SEMANTIC_MATERIALS)
            or len(ordered_global_ids) != len(mapping_evidence)
            or written_order.get("pass") is not True
            or any(
                int(row.get("index", -1)) != index
                or row.get("semantic_role")
                != list(EXPECTED_SEMANTIC_MATERIALS)[index]
                or row.get("source_to_import_signature_pass") is not True
                or row.get("signature_pass") is not True
                for index, row in enumerate(mapping_evidence)
            )
        ):
            raise ValueError("sealed refaceting semantic mapping is invalid")
        expected_global_by_semantic = {
            row["semantic_role"]: int(ordered_global_ids[index])
            for index, row in enumerate(mapping_evidence)
        }
        actual_global_by_semantic = {
            str(row["semantic_id"]): int(row["volume_global_id"])
            for row in rows
        }
        if actual_global_by_semantic != expected_global_by_semantic:
            raise ValueError(
                "source map differs from sealed refaceting semantic mapping"
            )
        sealed_signature_by_semantic = {
            str(row["semantic_role"]): row.get("source_solid_signature", {})
            for row in mapping_evidence
        }
        if not all(
            _source_signature_matches_sealed(signature, signature)
            for signature in sealed_signature_by_semantic.values()
        ):
            raise ValueError("sealed source-solid signature is incomplete")

    import cadquery as cq

    loaded_by_path: dict[Path, list[Any]] = {}
    inventory_by_path: dict[Path, dict[str, Any]] = {}
    targets = {}
    target_rows = []
    for row in rows:
        volume_id = int(row["volume_global_id"])
        semantic_id = str(row["semantic_id"]).strip()
        material = str(row.get("material", "")).strip()
        if material != volume_materials[volume_id]:
            raise ValueError(
                f"source-map material mismatch for volume {volume_id}"
            )
        if material != EXPECTED_SEMANTIC_MATERIALS[semantic_id]:
            raise ValueError(
                f"semantic material contract mismatch for {semantic_id}"
            )
        relative = str(row.get("source_step", "")).strip()
        expected_hash = str(row.get("source_step_sha256", ""))
        if not relative or not _valid_sha256(expected_hash):
            raise ValueError("source-map STEP identity is incomplete")
        path = _safe_relative_path(source_root, relative)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"source STEP hash mismatch for {relative}")
        if path not in loaded_by_path:
            solids = cq.importers.importStep(str(path)).solids().vals()
            if not solids:
                raise ValueError(f"source STEP has no solids: {relative}")
            loaded_by_path[path] = list(solids)
            inventory_by_path[path] = {
                "source_step": relative,
                "sha256": actual_hash,
                "size_bytes": path.stat().st_size,
                "solid_count": len(solids),
            }
        solid_index = int(row.get("source_solid_index", -1))
        if (relative, solid_index) != EXPECTED_SOURCE_TARGETS[semantic_id]:
            raise ValueError(
                f"semantic source target mismatch for {semantic_id}"
            )
        solids = loaded_by_path[path]
        if solid_index < 0 or solid_index >= len(solids):
            raise ValueError(
                f"source solid index is invalid for {semantic_id}"
            )
        actual_source_signature = _cadquery_solid_signature(
            solids[solid_index]
        )
        sealed_source_signature = sealed_signature_by_semantic.get(semantic_id)
        if sealed_source_signature is not None and not (
            _source_signature_matches_sealed(
                actual_source_signature, sealed_source_signature
            )
        ):
            raise ValueError(
                f"reloaded source solid differs from sealed identity for {semantic_id}"
            )
        boundary, tolerance, entity_count = _source_boundary(
            solids[solid_index]
        )
        target = SourceTarget(
            volume_global_id=volume_id,
            semantic_id=semantic_id,
            material=material,
            source_step=relative,
            source_step_sha256=actual_hash,
            source_solid_index=solid_index,
            boundary_shape=boundary,
            maximum_entity_tolerance_cm=tolerance,
            brep_entity_count=entity_count,
        )
        targets[volume_id] = target
        target_rows.append(
            {
                "volume_global_id": volume_id,
                "semantic_id": semantic_id,
                "material": material,
                "source_step": relative,
                "source_step_sha256": actual_hash,
                "source_solid_index": solid_index,
                "maximum_brep_entity_tolerance_cm": tolerance,
                "brep_entity_count": entity_count,
                "source_solid_signature": actual_source_signature,
                "sealed_source_solid_signature_match": (
                    sealed_source_signature is None
                    or _source_signature_matches_sealed(
                        actual_source_signature, sealed_source_signature
                    )
                ),
            }
        )
    source_inventory = [
        inventory_by_path[path] for path in sorted(inventory_by_path)
    ]
    target_rows.sort(key=lambda row: row["volume_global_id"])
    return targets, source_inventory, target_rows


def _project_to_boundary(
    boundary_shape: Any, point_cm: Sequence[float]
) -> float:
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt

    point = [float(value) for value in point_cm]
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        raise ValueError("projection point must be a finite three-vector")
    vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
    operation = BRepExtrema_DistShapeShape(vertex, boundary_shape)
    operation.Perform()
    if not operation.IsDone() or int(operation.NbSolution()) < 1:
        raise RuntimeError("OpenCascade boundary projection did not complete")
    distance = float(operation.Value())
    if not math.isfinite(distance) or distance < 0.0:
        raise RuntimeError("OpenCascade boundary distance is invalid")
    return distance


def _decode_opaque(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("expected one opaque tag value")
        value = value.reshape(-1)[0]
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8").strip()
    return str(value).split("\x00", 1)[0].strip()


def _optional_tag_value(mesh: Any, tag: Any, handle: int) -> Any | None:
    try:
        values = mesh.tag_get_data(
            tag, _entity_handle_array([handle]), flat=True
        )
    except RuntimeError as error:
        message = str(error).lower()
        if "mb_tag_not_found" in message or "tag not found" in message:
            return None
        raise
    if len(values) != 1:
        return None
    return values[0]


@dataclass
class NativeMesh:
    mesh: Any
    root_triangle_handles: set[int]
    surface_triangles: dict[int, list[int]]
    surface_global_ids: dict[int, int]
    surface_senses: dict[int, tuple[int, int]]
    volume_global_ids: dict[int, int]
    volume_materials: dict[int, str]
    volume_surfaces: dict[int, list[int]]


def _entity_handle_array(values: Any) -> np.ndarray:
    """Return the PyMOAB-compatible representation of entity handles."""

    return np.asarray(values, dtype=np.uint64).reshape(-1)


def _category_dimension_pass(category: str, dimension: Any) -> bool:
    expected = _EXPECTED_GEOMETRY_DIMENSION.get(str(category))
    if expected is None or dimension is None:
        return False
    try:
        return int(dimension) == expected
    except (TypeError, ValueError):
        return False


def _required_native_tag(
    mesh: Any, name: str, size: int, data_type: Any, storage: Any
) -> Any:
    return mesh.tag_get_handle(
        name,
        size,
        data_type,
        storage,
        create_if_missing=False,
    )


def _load_native_mesh(dagmc_path: Path) -> NativeMesh:
    from pymoab import core, types

    mesh = core.Core()
    mesh.load_file(str(dagmc_path))
    root = mesh.get_root_set()
    category_tag = _required_native_tag(
        mesh,
        types.CATEGORY_TAG_NAME,
        types.CATEGORY_TAG_SIZE,
        types.MB_TYPE_OPAQUE,
        types.MB_TAG_SPARSE,
    )
    global_id_tag = _required_native_tag(
        mesh,
        getattr(types, "GLOBAL_ID_TAG_NAME", "GLOBAL_ID"),
        1,
        types.MB_TYPE_INTEGER,
        types.MB_TAG_DENSE,
    )
    dimension_tag = _required_native_tag(
        mesh,
        getattr(types, "GEOM_DIMENSION_TAG_NAME", "GEOM_DIMENSION"),
        1,
        types.MB_TYPE_INTEGER,
        types.MB_TAG_DENSE,
    )
    sense_tag = _required_native_tag(
        mesh,
        "GEOM_SENSE_2",
        2,
        types.MB_TYPE_HANDLE,
        types.MB_TAG_SPARSE,
    )
    name_tag = _required_native_tag(
        mesh,
        getattr(types, "NAME_TAG_NAME", "NAME"),
        getattr(types, "NAME_TAG_SIZE", 32),
        types.MB_TYPE_OPAQUE,
        types.MB_TAG_SPARSE,
    )
    entity_sets = [
        int(value)
        for value in mesh.get_entities_by_type(root, types.MBENTITYSET)
    ]
    categories: dict[int, str] = {}
    global_ids: dict[int, int] = {}
    for handle in entity_sets:
        category_value = _optional_tag_value(mesh, category_tag, handle)
        if category_value is None:
            continue
        category = _decode_opaque(category_value)
        dimension = _optional_tag_value(mesh, dimension_tag, handle)
        if not _category_dimension_pass(category, dimension):
            raise ValueError(
                f"DAGMC {category} meshset has an invalid geometry dimension"
            )
        categories[handle] = category
        global_value = _optional_tag_value(mesh, global_id_tag, handle)
        if global_value is not None:
            global_ids[handle] = int(global_value)

    surface_handles = sorted(
        handle for handle, value in categories.items() if value == "Surface"
    )
    volume_handles = sorted(
        handle for handle, value in categories.items() if value == "Volume"
    )
    group_handles = sorted(
        handle for handle, value in categories.items() if value == "Group"
    )
    if not surface_handles or not volume_handles:
        raise ValueError("DAGMC native surface/volume inventory is empty")
    if any(
        handle not in global_ids for handle in surface_handles + volume_handles
    ):
        raise ValueError("DAGMC geometry entity lacks a global ID")
    surface_ids = [global_ids[handle] for handle in surface_handles]
    volume_ids = [global_ids[handle] for handle in volume_handles]
    if len(surface_ids) != len(set(surface_ids)) or any(
        value <= 0 for value in surface_ids
    ):
        raise ValueError("DAGMC surface global IDs are invalid")
    if len(volume_ids) != len(set(volume_ids)) or any(
        value <= 0 for value in volume_ids
    ):
        raise ValueError("DAGMC volume global IDs are invalid")

    volume_set = set(volume_handles)
    surface_senses = {}
    volume_surfaces: dict[int, list[int]] = defaultdict(list)
    for surface_handle in surface_handles:
        sense = tuple(
            int(value)
            for value in mesh.tag_get_data(
                sense_tag,
                _entity_handle_array([surface_handle]),
                flat=True,
            )
        )
        nonzero = [value for value in sense if value]
        if (
            len(sense) != 2
            or not nonzero
            or len(nonzero) != len(set(nonzero))
            or any(value not in volume_set for value in nonzero)
        ):
            raise ValueError("DAGMC surface has an invalid native sense")
        surface_senses[surface_handle] = (sense[0], sense[1])
        for volume_handle in nonzero:
            volume_surfaces[volume_handle].append(surface_handle)

    volume_material_names: dict[int, list[str]] = defaultdict(list)
    for group_handle in group_handles:
        name_value = _optional_tag_value(mesh, name_tag, group_handle)
        name = _decode_opaque(name_value) if name_value is not None else ""
        if not name.startswith("mat:"):
            continue
        material = name.removeprefix("mat:").strip()
        members = {
            int(value) for value in mesh.get_entities_by_handle(group_handle)
        }
        invalid_members = members - volume_set
        if not material or invalid_members:
            raise ValueError("DAGMC material group has invalid membership")
        for volume_handle in members:
            volume_material_names[volume_handle].append(material)
    volume_materials = {}
    for volume_handle in volume_handles:
        names = volume_material_names.get(volume_handle, [])
        if len(names) != 1:
            raise ValueError(
                "DAGMC volume must have exactly one material group"
            )
        volume_materials[global_ids[volume_handle]] = names[0]

    surface_triangles = {
        surface_handle: [
            int(value)
            for value in mesh.get_entities_by_type(surface_handle, types.MBTRI)
        ]
        for surface_handle in surface_handles
    }
    if any(not values for values in surface_triangles.values()):
        raise ValueError("DAGMC surface has no triangles")
    owners: dict[int, list[int]] = defaultdict(list)
    for surface_handle, triangles in surface_triangles.items():
        for triangle_handle in triangles:
            owners[triangle_handle].append(surface_handle)
    root_triangles = {
        int(value) for value in mesh.get_entities_by_type(root, types.MBTRI)
    }
    if set(owners) != root_triangles or any(
        len(values) != 1 for values in owners.values()
    ):
        raise ValueError("DAGMC triangle ownership is not exact")
    return NativeMesh(
        mesh=mesh,
        root_triangle_handles=root_triangles,
        surface_triangles=surface_triangles,
        surface_global_ids={
            handle: global_ids[handle] for handle in surface_handles
        },
        surface_senses=surface_senses,
        volume_global_ids={
            handle: global_ids[handle] for handle in volume_handles
        },
        volume_materials=volume_materials,
        volume_surfaces=dict(volume_surfaces),
    )


def _triangle_coordinates(mesh: Any, triangle_handle: int) -> np.ndarray:
    connectivity = [
        int(value)
        for value in mesh.get_connectivity(
            _entity_handle_array([triangle_handle])
        )
    ]
    if len(connectivity) != 3 or len(set(connectivity)) != 3:
        raise ValueError("DAGMC triangle connectivity is invalid")
    return _triangle(
        np.asarray(
            mesh.get_coords(_entity_handle_array(connectivity)), dtype=float
        ).reshape((3, 3))
    )


def _native_volume_snapshot(
    native: NativeMesh,
    volume_handle: int,
    target: SourceTarget,
) -> tuple[dict[str, Any], list[TriangleIncidence]]:
    surface_handles = sorted(native.volume_surfaces.get(volume_handle, []))
    if not surface_handles:
        raise ValueError("DAGMC volume has no boundary surfaces")
    edge_counts: Counter[tuple[int, int]] = Counter()
    directed_balance: Counter[tuple[int, int]] = Counter()
    vector_area = np.zeros(3)
    total_area = 0.0
    signed_six_volume = 0.0
    incidences = []
    for surface_handle in surface_handles:
        forward, reverse = native.surface_senses[surface_handle]
        if volume_handle == forward:
            sign = 1
        elif volume_handle == reverse:
            sign = -1
        else:
            raise ValueError("DAGMC surface sense omits its owning volume")
        for triangle_handle in native.surface_triangles[surface_handle]:
            coordinates = _triangle_coordinates(native.mesh, triangle_handle)
            connectivity = [
                int(value)
                for value in native.mesh.get_connectivity(
                    _entity_handle_array([triangle_handle])
                )
            ]
            oriented = (
                connectivity
                if sign > 0
                else [connectivity[0], connectivity[2], connectivity[1]]
            )
            for first, second in ((0, 1), (1, 2), (2, 0)):
                directed = (oriented[first], oriented[second])
                edge = tuple(sorted(directed))
                edge_counts[edge] += 1
                directed_balance[edge] += 1 if directed == edge else -1
            area_vector = 0.5 * np.cross(
                coordinates[1] - coordinates[0],
                coordinates[2] - coordinates[0],
            )
            total_area += float(np.linalg.norm(area_vector))
            vector_area += sign * area_vector
            signed_six_volume += sign * float(
                np.dot(
                    coordinates[0],
                    np.cross(coordinates[1], coordinates[2]),
                )
            )
            incidences.append(
                TriangleIncidence(
                    triangle_handle=triangle_handle,
                    surface_global_id=native.surface_global_ids[
                        surface_handle
                    ],
                    volume_global_id=target.volume_global_id,
                    semantic_id=target.semantic_id,
                    material=target.material,
                    coordinates_cm=coordinates,
                )
            )
    edge_errors = sum(value != 2 for value in edge_counts.values())
    directed_errors = sum(value != 0 for value in directed_balance.values())
    relative_vector_area = (
        float(np.linalg.norm(vector_area) / total_area)
        if total_area > 0.0
        else None
    )
    signed_volume = signed_six_volume / 6.0
    envelope_pass = bool(
        incidences
        and edge_errors == 0
        and directed_errors == 0
        and relative_vector_area is not None
        and relative_vector_area <= 1.0e-8
        and signed_volume > 0.0
    )
    return (
        {
            "volume_global_id": target.volume_global_id,
            "semantic_id": target.semantic_id,
            "material": target.material,
            "boundary_surface_global_ids": sorted(
                native.surface_global_ids[value] for value in surface_handles
            ),
            "boundary_surface_count": len(surface_handles),
            "triangle_incidence_count": len(incidences),
            "total_boundary_area_cm2": total_area,
            "signed_volume_cm3": signed_volume,
            "volume_cm3": signed_volume,
            "native_edge_multiplicity_error_count": edge_errors,
            "native_directed_edge_error_count": directed_errors,
            "vector_area_closure_relative": relative_vector_area,
            "envelope_pass": envelope_pass,
        },
        incidences,
    )


def _semantic_topology_signature(
    semantic_volumes: Sequence[Mapping[str, Any]],
    adjacency: Sequence[Mapping[str, Any]],
    exterior_surface_count: int,
) -> str:
    payload = {
        "semantic_volumes": sorted(
            [
                {
                    key: row[key]
                    for key in (
                        "semantic_id",
                        "material",
                        "boundary_surface_count",
                    )
                }
                for row in semantic_volumes
            ],
            key=lambda row: row["semantic_id"],
        ),
        "semantic_adjacency_multigraph": sorted(
            sorted(str(value) for value in row["semantic_pair"])
            for row in adjacency
        ),
        "exterior_surface_count": int(exterior_surface_count),
    }
    return _json_sha256(payload)


def _collect_snapshot(
    native: NativeMesh,
    targets: Mapping[int, SourceTarget],
    *,
    level: str,
    protocol_sha256: str,
    h5m_sha256: str,
    criteria_sha256: str,
    physical_input_identity: Mapping[str, Any],
    physical_input_identity_sha256: str,
    source_inventory: Sequence[Mapping[str, Any]],
    projection_targets: Sequence[Mapping[str, Any]],
    faceting_settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[TriangleIncidence]]:
    handles_by_global_id = {
        global_id: handle
        for handle, global_id in native.volume_global_ids.items()
    }
    semantic_volumes = []
    incidences = []
    for volume_id in sorted(handles_by_global_id):
        row, volume_incidences = _native_volume_snapshot(
            native, handles_by_global_id[volume_id], targets[volume_id]
        )
        semantic_volumes.append(row)
        incidences.extend(volume_incidences)

    semantic_by_handle = {
        handle: targets[global_id].semantic_id
        for handle, global_id in native.volume_global_ids.items()
    }
    adjacency = []
    exterior_surface_count = 0
    for surface_handle, sense in sorted(
        native.surface_senses.items(),
        key=lambda item: native.surface_global_ids[item[0]],
    ):
        owners = [value for value in sense if value]
        if len(owners) == 1:
            exterior_surface_count += 1
        elif len(owners) == 2:
            adjacency.append(
                {
                    "surface_global_id": native.surface_global_ids[
                        surface_handle
                    ],
                    "semantic_pair": sorted(
                        semantic_by_handle[value] for value in owners
                    ),
                }
            )
        else:
            raise ValueError("DAGMC surface sense cardinality is invalid")
    unique_covered = {row.triangle_handle for row in incidences}
    magnet_incidences = [
        row for row in incidences if row.material == "magnets"
    ]
    native_gate = bool(
        unique_covered == native.root_triangle_handles
        and all(row["envelope_pass"] for row in semantic_volumes)
    )
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "level": level,
        "criteria_sha256": criteria_sha256,
        "faceting_protocol_sha256": protocol_sha256,
        "construct_only": False,
        "build_status": "BUILD_COMPLETE_PENDING_QUALIFICATION",
        "raw_h5m_sha256_before": h5m_sha256,
        "raw_h5m_sha256_after": h5m_sha256,
        "h5m_unchanged": True,
        "physical_input_identity": dict(physical_input_identity),
        "physical_input_identity_sha256": physical_input_identity_sha256,
        "source_mesh_canonical_fingerprint": physical_input_identity.get(
            "source_mesh_canonical_fingerprint"
        ),
        "source_solid_hash_inventory": [dict(row) for row in source_inventory],
        "semantic_projection_targets": [
            dict(row) for row in projection_targets
        ],
        "faceting_settings": dict(faceting_settings),
        "semantic_volumes": semantic_volumes,
        "semantic_adjacency_multigraph": adjacency,
        "semantic_topology_signature": _semantic_topology_signature(
            semantic_volumes, adjacency, exterior_surface_count
        ),
        "physical_triangle_count": len(native.root_triangle_handles),
        "physical_triangle_incidence_count": len(incidences),
        "covered_physical_triangle_count": len(unique_covered),
        "magnet_triangle_incidence_count": len(magnet_incidences),
        "triangle_incidence_digest_sha256": _triangle_incidence_digest(
            incidences
        ),
        "physical_triangle_digest_sha256": _physical_triangle_digest(
            incidences
        ),
        "all_physical_triangles_covered": unique_covered
        == native.root_triangle_handles,
        "all_magnet_triangles_covered": bool(magnet_incidences),
        "semantic_projection_target_pass": set(targets)
        == set(native.volume_materials),
        "native_topology_gate_pass": native_gate,
    }
    return snapshot, incidences


def _pydagmc_reconciliation(
    dagmc_path: Path,
    semantic_volumes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import pydagmc

    model = pydagmc.Model(str(dagmc_path))
    rows_by_id = {
        int(row["volume_global_id"]): row for row in semantic_volumes
    }
    model_ids = {int(value) for value in model.volumes_by_id}
    rows = []
    for volume_id in sorted(rows_by_id):
        volume = model.volumes_by_id.get(volume_id)
        if volume is None:
            rows.append({"volume_global_id": volume_id, "pass": False})
            continue
        expected = rows_by_id[volume_id]
        envelope = audit_volume_envelope(volume)
        actual_material = str(getattr(volume, "material", "")).strip()
        actual_surfaces = sorted(int(item.id) for item in volume.surfaces)
        passed = bool(
            actual_material == expected["material"]
            and actual_surfaces == expected["boundary_surface_global_ids"]
            and envelope["closed"] is True
        )
        rows.append(
            {
                "volume_global_id": volume_id,
                "material": actual_material,
                "surface_global_ids": actual_surfaces,
                "envelope_pass": envelope["closed"],
                "pass": passed,
            }
        )
    passed = bool(
        model_ids == set(rows_by_id)
        and len(rows) == len(rows_by_id)
        and all(row["pass"] for row in rows)
    )
    return {
        "schema": "parastell.faceting_pydagmc_reconciliation/v1.0.0",
        "volume_ids_match": model_ids == set(rows_by_id),
        "volumes": rows,
        "pass": passed,
    }


def _collect_deviation_certificate(
    incidences: Sequence[TriangleIncidence],
    targets: Mapping[int, SourceTarget],
    snapshot: Mapping[str, Any],
    *,
    level: str,
    protocol_sha256: str,
    h5m_sha256: str,
    physical_input_identity_sha256: str,
    source_inventory: Sequence[Mapping[str, Any]],
    projection_targets: Sequence[Mapping[str, Any]],
    maximum_cover_radius_cm: float = MAXIMUM_COVER_RADIUS_CM,
    progress_path: Path | None = None,
    progress_every_triangle_incidences: int = 100,
) -> dict[str, Any]:
    if float(maximum_cover_radius_cm) > MAXIMUM_COVER_RADIUS_CM:
        raise ValueError("cover radius exceeds the preregistered maximum")
    if (
        isinstance(progress_every_triangle_incidences, bool)
        or int(progress_every_triangle_incidences) < 1
    ):
        raise ValueError("progress interval must be a positive integer")
    maximum_distance = 0.0
    actual_cover_radius = 0.0
    leaf_sample_count = 0
    projection_errors = []
    processed = []
    for index, row in enumerate(incidences, start=1):
        target = targets[row.volume_global_id]
        try:
            measurement = _measure_triangle_deviation(
                row.coordinates_cm,
                lambda point, boundary=target.boundary_shape: (
                    _project_to_boundary(boundary, point)
                ),
                maximum_cover_radius_cm=maximum_cover_radius_cm,
            )
        except Exception as error:
            projection_errors.append(
                {
                    "triangle_handle": row.triangle_handle,
                    "surface_global_id": row.surface_global_id,
                    "volume_global_id": row.volume_global_id,
                    "semantic_id": row.semantic_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if progress_path is not None and (
                index % int(progress_every_triangle_incidences) == 0
                or index == len(incidences)
            ):
                _write_json_atomic(
                    progress_path,
                    {
                        "schema": (
                            "parastell.faceting_projection_progress/v1.0.0"
                        ),
                        "status": (
                            "TERMINAL"
                            if index == len(incidences)
                            else "RUNNING"
                        ),
                        "level": level,
                        "faceting_protocol_sha256": protocol_sha256,
                        "raw_h5m_sha256": h5m_sha256,
                        "physical_input_identity_sha256": (
                            physical_input_identity_sha256
                        ),
                        "completed_triangle_incidence_count": index,
                        "total_triangle_incidence_count": len(incidences),
                        "successful_triangle_incidence_count": len(processed),
                        "projection_error_count": len(projection_errors),
                        "projected_subtriangle_centroid_count": (
                            leaf_sample_count
                        ),
                        "maximum_projected_distance_cm": maximum_distance,
                        "actual_maximum_cover_radius_cm": actual_cover_radius,
                    },
                )
            continue
        maximum_distance = max(
            maximum_distance,
            measurement["maximum_projected_distance_cm"],
        )
        actual_cover_radius = max(
            actual_cover_radius,
            measurement["actual_maximum_cover_radius_cm"],
        )
        leaf_sample_count += measurement["leaf_sample_count"]
        processed.append(row)
        if progress_path is not None and (
            index % int(progress_every_triangle_incidences) == 0
            or index == len(incidences)
        ):
            _write_json_atomic(
                progress_path,
                {
                    "schema": "parastell.faceting_projection_progress/v1.0.0",
                    "status": (
                        "TERMINAL" if index == len(incidences) else "RUNNING"
                    ),
                    "level": level,
                    "faceting_protocol_sha256": protocol_sha256,
                    "raw_h5m_sha256": h5m_sha256,
                    "physical_input_identity_sha256": (
                        physical_input_identity_sha256
                    ),
                    "completed_triangle_incidence_count": index,
                    "total_triangle_incidence_count": len(incidences),
                    "successful_triangle_incidence_count": len(processed),
                    "projection_error_count": len(projection_errors),
                    "projected_subtriangle_centroid_count": leaf_sample_count,
                    "maximum_projected_distance_cm": maximum_distance,
                    "actual_maximum_cover_radius_cm": actual_cover_radius,
                },
            )
    maximum_tolerance = max(
        target.maximum_entity_tolerance_cm for target in targets.values()
    )
    arithmetic = _certificate_arithmetic(
        maximum_projected_distance_cm=maximum_distance,
        actual_maximum_cover_radius_cm=actual_cover_radius,
        maximum_brep_entity_tolerance_cm=maximum_tolerance,
    )
    expected_incidence_count = int(
        snapshot["physical_triangle_incidence_count"]
    )
    magnet_expected = int(snapshot["magnet_triangle_incidence_count"])
    magnet_processed = sum(row.material == "magnets" for row in processed)
    processed_digest = _triangle_incidence_digest(processed)
    processed_physical_digest = _physical_triangle_digest(processed)
    expected_digest = str(snapshot["triangle_incidence_digest_sha256"])
    expected_physical_digest = str(snapshot["physical_triangle_digest_sha256"])
    processed_physical_count = len({row.triangle_handle for row in processed})
    reconciliation = bool(
        len(processed) == expected_incidence_count
        and processed_digest == expected_digest
        and processed_physical_count == snapshot["physical_triangle_count"]
        and processed_physical_digest == expected_physical_digest
        and not projection_errors
    )
    certified = bool(
        reconciliation
        and snapshot.get("all_physical_triangles_covered") is True
        and snapshot.get("all_magnet_triangles_covered") is True
        and snapshot.get("semantic_projection_target_pass") is True
        and actual_cover_radius <= MAXIMUM_COVER_RADIUS_CM
        and arithmetic["arithmetic_reconstruction_pass"]
    )
    source_hashes = {
        str(row["source_step"]): str(row["sha256"]) for row in source_inventory
    }
    return {
        "schema": DEVIATION_SCHEMA,
        "level": level,
        "direction": "dagmc_facets_to_source_cad_boundary",
        "faceting_protocol_sha256": protocol_sha256,
        "raw_h5m_sha256_before": h5m_sha256,
        "raw_h5m_sha256_after": h5m_sha256,
        "h5m_unchanged": True,
        "physical_input_identity_sha256": physical_input_identity_sha256,
        "source_solid_hash_inventory": [dict(row) for row in source_inventory],
        "source_solid_sha256": source_hashes,
        "semantic_projection_targets": [
            dict(row) for row in projection_targets
        ],
        "source_boundary": "compound_of_all_brep_faces",
        "maximum_allowed_cover_radius_cm": MAXIMUM_COVER_RADIUS_CM,
        "requested_cover_radius_cm": float(maximum_cover_radius_cm),
        "actual_maximum_cover_radius_cm": actual_cover_radius,
        "solver_margin_cm": SOLVER_MARGIN_CM,
        "maximum_brep_entity_tolerance_cm": maximum_tolerance,
        "maximum_projected_distance_cm": maximum_distance,
        "upper_bound_arithmetic": arithmetic,
        "upper_bound_cm": arithmetic["upper_bound_cm"],
        "physical_triangle_count": snapshot["physical_triangle_count"],
        "unique_physical_triangle_count": snapshot["physical_triangle_count"],
        "physical_triangle_incidence_count": expected_incidence_count,
        "covered_triangle_incidence_count": len(processed),
        "covered_triangle_count": processed_physical_count,
        "covered_physical_triangle_count": processed_physical_count,
        "magnet_triangle_incidence_count": magnet_expected,
        "covered_magnet_triangle_incidence_count": magnet_processed,
        "leaf_sample_count": leaf_sample_count,
        "projected_subtriangle_centroid_count": leaf_sample_count,
        "expected_triangle_incidence_digest_sha256": expected_digest,
        "covered_triangle_incidence_digest_sha256": processed_digest,
        "expected_physical_triangle_digest_sha256": expected_physical_digest,
        "covered_physical_triangle_digest_sha256": processed_physical_digest,
        "triangle_id_digest_sha256": expected_digest,
        "projection_error_count": len(projection_errors),
        "projection_errors": projection_errors,
        "all_triangles_covered": reconciliation,
        "all_physical_volume_triangles": reconciliation,
        "all_magnet_triangles": magnet_processed == magnet_expected
        and magnet_expected > 0,
        "all_physical_triangles_covered": reconciliation
        and snapshot["all_physical_triangles_covered"],
        "all_magnet_triangles_covered": magnet_processed == magnet_expected
        and magnet_expected > 0,
        "semantic_projection_target_pass": snapshot[
            "semantic_projection_target_pass"
        ],
        "projection_target_semantic_match_pass": snapshot[
            "semantic_projection_target_pass"
        ],
        "exact_triangle_reconciliation_pass": reconciliation,
        "coverage_reconciliation_pass": reconciliation,
        "certified_upper_bound": certified,
    }


def _point_triangle_distance_cm(point: Any, triangle: Any) -> float:
    """Return the Euclidean point-to-triangle distance."""
    p = np.asarray(point, dtype=float)
    t = _triangle(triangle)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError("zone witness must be a finite three-vector")
    a, b, c = t
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        fraction = d1 / (d1 - d3)
        return float(np.linalg.norm(p - (a + fraction * ab)))
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        fraction = d2 / (d2 - d6)
        return float(np.linalg.norm(p - (a + fraction * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(p - (b + fraction * (c - b))))
    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    projection = a + ab * v + ac * w
    return float(np.linalg.norm(p - projection))


def _validate_public_overlap_report(
    report: Mapping[str, Any], report_sha256: str
) -> dict[str, list[float]]:
    if report_sha256 != EXPECTED_PUBLIC_OVERLAP_REPORT_SHA256:
        raise ValueError(
            "public overlap witness report hash is not preregistered"
        )
    if (
        report.get("schema")
        != "parastell.public_reference_overlap_quantification/v1.0.0"
    ):
        raise ValueError("unsupported public overlap witness schema")
    rows = report.get("intersections")
    if not isinstance(rows, list):
        raise ValueError("public overlap report has no intersection rows")
    by_id = {}
    for row in rows:
        magnet_id = str(row.get("magnet_id", ""))
        witness = row.get("dagmc_witness_cm")
        try:
            witness_values = [float(value) for value in witness]
        except (TypeError, ValueError):
            witness_values = []
        if (
            magnet_id in by_id
            or magnet_id not in REQUIRED_FORMER_ZONE_IDS
            or row.get("classification")
            != "TRUE_SOURCE_CAD_PHYSICAL_INTERSECTION"
            or not isinstance(witness, list)
            or len(witness_values) != 3
            or not all(math.isfinite(value) for value in witness_values)
        ):
            raise ValueError("public overlap witness row is invalid")
        by_id[magnet_id] = witness_values
    if set(by_id) != set(REQUIRED_FORMER_ZONE_IDS):
        raise ValueError(
            "public overlap witness inventory differs from protocol"
        )
    return by_id


def _zone_role_statistics(
    rows: Sequence[TriangleIncidence],
    *,
    witness_cm: Sequence[float],
    radius_cm: float,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if _point_triangle_distance_cm(witness_cm, row.coordinates_cm)
        <= radius_cm
    ]
    maximum_edge = (
        max(_maximum_edge_cm(row.coordinates_cm) for row in selected)
        if selected
        else None
    )
    return {
        "triangle_count": len(selected),
        "maximum_edge_cm": maximum_edge,
        "covered": bool(selected),
        "triangle_digest_sha256": _triangle_incidence_digest(selected),
    }


def _collect_former_overlap_zones(
    incidences: Sequence[TriangleIncidence],
    *,
    level: str,
    public_report: Mapping[str, Any],
    public_report_sha256: str,
    radius_cm: float = FORMER_ZONE_RADIUS_CM,
) -> list[dict[str, Any]]:
    if float(radius_cm) != FORMER_ZONE_RADIUS_CM:
        raise ValueError("former-overlap-zone radius must be exactly 40 cm")
    witnesses = _validate_public_overlap_report(
        public_report, public_report_sha256
    )
    rows = []
    for magnet_id in REQUIRED_FORMER_ZONE_IDS:
        witness = witnesses[magnet_id]
        magnet_triangles = [
            row for row in incidences if row.semantic_id == magnet_id
        ]
        vessel_triangles = [
            row for row in incidences if row.semantic_id == "vac_vessel"
        ]
        magnet_statistics = _zone_role_statistics(
            magnet_triangles, witness_cm=witness, radius_cm=radius_cm
        )
        vessel_statistics = _zone_role_statistics(
            vessel_triangles, witness_cm=witness, radius_cm=radius_cm
        )
        rows.append(
            {
                "zone_id": magnet_id,
                "level": level,
                "classification": "GLOBAL_REFINEMENT_COVERS_FORMER_ZONES",
                "refinement_classification": (
                    "GLOBAL_REFINEMENT_COVERS_FORMER_ZONES"
                ),
                "public_overlap_report_sha256": public_report_sha256,
                "public_reference_overlap_report_sha256": (
                    public_report_sha256
                ),
                "preregistered_witness_cm": witness,
                "witness_cm": witness,
                "preregistered_witness_sha256": _json_sha256(
                    {
                        "zone_id": magnet_id,
                        "dagmc_witness_cm": witness,
                    }
                ),
                "sphere_radius_cm": float(radius_cm),
                "neighborhood_radius_cm": float(radius_cm),
                "magnet": magnet_statistics,
                "vac_vessel": vessel_statistics,
                "zone_coverage_pass": magnet_statistics["covered"]
                and vessel_statistics["covered"],
            }
        )
    return rows


def combine_former_overlap_zone_evidence(
    coarse_rows: Sequence[Mapping[str, Any]],
    refined_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Package two raw level inventories for the convergence reducer."""
    coarse = {str(row.get("zone_id")): row for row in coarse_rows}
    refined = {str(row.get("zone_id")): row for row in refined_rows}
    if set(coarse) != set(REQUIRED_FORMER_ZONE_IDS) or set(refined) != set(
        REQUIRED_FORMER_ZONE_IDS
    ):
        raise ValueError("former-overlap-zone level inventory is incomplete")
    combined = []
    for zone_id in REQUIRED_FORMER_ZONE_IDS:
        first = coarse[zone_id]
        second = refined[zone_id]
        identity_keys = (
            "public_reference_overlap_report_sha256",
            "witness_cm",
            "neighborhood_radius_cm",
            "refinement_classification",
        )
        if any(first.get(key) != second.get(key) for key in identity_keys):
            raise ValueError("former-overlap-zone identities differ by level")
        combined.append(
            {
                "zone_id": zone_id,
                **{key: first[key] for key in identity_keys},
                "coarse": {
                    role: dict(first[role])
                    for role in ("magnet", "vac_vessel")
                },
                "refined": {
                    role: dict(second[role])
                    for role in ("magnet", "vac_vessel")
                },
            }
        )
    return combined


def _protocol_level(protocol: Mapping[str, Any], level: str) -> dict[str, Any]:
    if level not in {"coarse", "refined"}:
        raise ValueError("faceting level must be coarse or refined")
    if (
        protocol.get("schema")
        != "parastell.faceting_comparison_protocol/v1.0.0"
    ):
        raise ValueError("unsupported faceting protocol schema")
    row = protocol.get("levels", {}).get(level, {})
    settings = {
        "minimum_size_cm": float(row.get("minimum_mesh_size_cm", -1.0)),
        "maximum_size_cm": float(row.get("maximum_mesh_size_cm", -1.0)),
        "algorithm": int(row.get("algorithm", -1)),
    }
    if (
        settings["minimum_size_cm"] <= 0.0
        or settings["maximum_size_cm"] < settings["minimum_size_cm"]
        or settings["algorithm"] != 1
    ):
        raise ValueError("faceting protocol level settings are invalid")
    certificate = protocol.get("facet_deviation_certificate", {})
    if (
        certificate.get("direction") != "dagmc_facets_to_source_cad_boundary"
        or certificate.get("source_boundary") != "compound_of_all_brep_faces"
        or float(certificate.get("maximum_cover_radius_cm", -1.0))
        != MAXIMUM_COVER_RADIUS_CM
        or float(certificate.get("solver_margin_cm", -1.0)) != SOLVER_MARGIN_CM
        or certificate.get("include_maximum_brep_entity_tolerance") is not True
        or certificate.get("require_all_physical_volume_triangles") is not True
        or certificate.get("require_all_magnet_triangles") is not True
        or certificate.get("projection_errors_allowed") != 0
    ):
        raise ValueError("faceting deviation protocol is not supported")
    return settings


def _validate_refaceting_manifest(
    manifest: Mapping[str, Any],
    *,
    level: str,
    settings: Mapping[str, Any],
    protocol_sha256: str,
    h5m_sha256: str,
    expected_physical_identity_sha256: str,
) -> None:
    if (
        manifest.get("schema")
        != "parastell.refaceted_source_cad_candidate/v1.0.0"
    ):
        raise ValueError("unsupported refaceting manifest schema")
    if manifest.get("refaceting_only") is not True:
        raise ValueError("manifest is not a source-CAD-only refaceting build")
    if manifest.get("status") != "BUILD_COMPLETE_PENDING_QUALIFICATION":
        raise ValueError("refaceting manifest is not qualification-ready")
    if manifest.get("construct_only") is not False:
        raise ValueError("construct-only geometry cannot provide evidence")
    if manifest.get("inherited_source_cad_physical_gate_pass") is not True:
        raise ValueError("source-CAD physical gate was not inherited")
    if manifest.get("faceting_protocol", {}).get("sha256") != protocol_sha256:
        raise ValueError("refaceting manifest protocol hash mismatch")
    actual_settings = manifest.get("faceting", {})
    if actual_settings.get("level") != level:
        raise ValueError("refaceting manifest level differs from request")
    normalized = {
        "minimum_size_cm": float(
            actual_settings.get(
                "minimum_size_cm",
                actual_settings.get("minimum_mesh_size_cm", -1.0),
            )
        ),
        "maximum_size_cm": float(
            actual_settings.get(
                "maximum_size_cm",
                actual_settings.get("maximum_mesh_size_cm", -1.0),
            )
        ),
        "algorithm": int(actual_settings.get("algorithm", -1)),
    }
    if normalized != dict(settings):
        raise ValueError(f"{level} refaceting settings differ from protocol")
    physical_hash = manifest.get("physical_input_identity_sha256")
    if physical_hash != expected_physical_identity_sha256:
        raise ValueError("physical-input-identity hash mismatch")
    if _json_sha256(manifest.get("physical_input_identity")) != physical_hash:
        raise ValueError("physical-input-identity content hash mismatch")
    dagmc_row = manifest.get("artifacts", {}).get("dagmc.h5m", {})
    if dagmc_row.get("sha256") != h5m_sha256:
        raise ValueError("manifest DAGMC hash mismatch")
    if manifest.get("input_immutability_pass") is not True:
        raise ValueError("refaceting input immutability gate failed")
    identity = manifest.get("physical_input_identity", {})
    criteria_sha256 = manifest.get("acceptance_criteria", {}).get("sha256")
    if (
        not _valid_sha256(criteria_sha256)
        or identity.get("acceptance_criteria_sha256") != criteria_sha256
    ):
        raise ValueError("acceptance-criteria binding failed")
    source_qualification = manifest.get("source_cad_qualification", {})
    required_source_hashes = (
        "source_manifest_sha256",
        "source_audit_sha256",
        "source_audit_seal_sha256",
        "physical_change_report_sha256",
        "reference_manifest_sha256",
    )
    if not all(
        _valid_sha256(source_qualification.get(name))
        for name in required_source_hashes
    ):
        raise ValueError("source-CAD qualification hash binding is incomplete")
    topology_rows = (
        manifest.get("source_topology_contract", {}),
        manifest.get("gmsh_pre_mesh_topology_contract", {}),
        manifest.get("written_h5m_topology_contract", {}),
        manifest.get("written_h5m_material_order_evidence", {}),
    )
    if not all(row.get("pass") is True for row in topology_rows):
        raise ValueError("refaceting topology or material gate failed")


def _validate_refaceting_seal(
    seal: Mapping[str, Any],
    *,
    manifest_sha256: str,
    h5m_sha256: str,
    physical_input_identity_sha256: str,
) -> None:
    if (
        seal.get("schema")
        != "parastell.refaceted_source_cad_candidate_seal/v1.0.0"
    ):
        raise ValueError("unsupported refaceting build-seal schema")
    if seal.get("candidate_build_manifest_sha256") != manifest_sha256:
        raise ValueError("refaceting seal manifest binding failed")
    if seal.get("dagmc_h5m_sha256") != h5m_sha256:
        raise ValueError("refaceting seal H5M binding failed")
    if (
        seal.get("physical_input_identity_sha256")
        != physical_input_identity_sha256
    ):
        raise ValueError("refaceting seal physical identity binding failed")
    required_gates = (
        "inherited_source_cad_physical_gate_pass",
        "input_immutability_pass",
        "source_topology_contract_pass",
        "gmsh_pre_mesh_topology_contract_pass",
        "written_h5m_topology_contract_pass",
    )
    failed = [name for name in required_gates if seal.get(name) is not True]
    if failed:
        raise ValueError(f"refaceting seal failed gates: {failed}")


def _validate_native_qualification(
    report: Mapping[str, Any], h5m_sha256: str
) -> None:
    if report.get("schema") != "parastell.dagmc_native_qualification/v1.0.0":
        raise ValueError("unsupported native DAGMC qualification schema")
    watertight = report.get("check_watertight", {})
    overlaps = report.get("overlap_checks", [])
    precisions = sorted(row.get("points_per_edge") for row in overlaps)
    passed = bool(
        report.get("raw_h5m_sha256_before") == h5m_sha256
        and report.get("raw_h5m_sha256_after") == h5m_sha256
        and report.get("h5m_unchanged") is True
        and report.get("native_dagmc_gate_pass") is True
        and watertight.get("pass") is True
        and watertight.get("unmatched_edge_count") == 0
        and watertight.get("unsealed_surface_count") == 0
        and watertight.get("unsealed_volume_count") == 0
        and precisions == [1, 2, 4]
        and all(
            row.get("pass") is True
            and row.get("terminal_overlap_location_count") == 0
            and row.get("parsed_overlap_location_count") == 0
            and row.get("row_count_matches_terminal") is True
            and row.get("precision_header_pass") is True
            for row in overlaps
        )
    )
    if not passed:
        raise ValueError("native DAGMC qualification report failed")


def _source_clearance_cm(report: Mapping[str, Any]) -> float:
    value = report.get("summary", {}).get(
        "minimum_vessel_to_magnet_clearance_cm"
    )
    if (
        report.get("source_cad_gate_pass") is not True
        or report.get("source_cad_physical_gate_pass") is not True
        or not _finite_nonnegative(value)
        or float(value) <= 0.0
    ):
        raise ValueError("source-CAD clearance evidence failed")
    return float(value)


def _runtime_identity() -> dict[str, Any]:
    import cadquery
    import OCP
    import pymoab

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "cadquery": getattr(cadquery, "__version__", None),
        "OCP": getattr(OCP, "__version__", None),
        "pymoab": getattr(pymoab, "__version__", None),
        "numpy": np.__version__,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    if json.loads(temporary.read_text(encoding="utf-8")) != value:
        raise RuntimeError("JSON readback differs before publication")
    temporary.replace(path)
    if json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError("published JSON readback differs")


def _assert_input_immutability(
    controls_before: Mapping[str, Any],
    controls_after: Mapping[str, Any],
    source_inventory_before: Sequence[Mapping[str, Any]],
    source_inventory_after: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> None:
    if dict(controls_after) != dict(controls_before):
        raise RuntimeError(f"faceting controls changed {phase}")
    if list(source_inventory_after) != list(source_inventory_before):
        raise RuntimeError(f"source STEP inventory changed {phase}")


def collect_faceting_evidence(
    *,
    dagmc_path: str | Path,
    level: str,
    refaceting_manifest_path: str | Path,
    expected_refaceting_manifest_sha256: str,
    refaceting_seal_path: str | Path,
    expected_refaceting_seal_sha256: str,
    faceting_protocol_path: str | Path,
    expected_faceting_protocol_sha256: str,
    expected_h5m_sha256: str,
    expected_physical_input_identity_sha256: str,
    native_qualification_path: str | Path,
    expected_native_qualification_sha256: str,
    source_cad_audit_path: str | Path,
    expected_source_cad_audit_sha256: str,
    source_map_path: str | Path,
    expected_source_map_sha256: str,
    source_root: str | Path,
    public_overlap_report_path: str | Path,
    expected_public_overlap_report_sha256: str,
    maximum_projected_subtriangle_centroids: int,
    progress_every_triangle_incidences: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Collect one faceting level's sealed measurement evidence.

    The output is intentionally not a scientific pass/fail decision.  A
    separate reducer must compare two independently collected levels.
    """
    dagmc = Path(dagmc_path).resolve()
    manifest_path = Path(refaceting_manifest_path).resolve()
    build_seal_path = Path(refaceting_seal_path).resolve()
    native_report_path = Path(native_qualification_path).resolve()
    source_audit_path = Path(source_cad_audit_path).resolve()
    protocol_path = Path(faceting_protocol_path).resolve()
    mapping_path = Path(source_map_path).resolve()
    source_directory = Path(source_root).resolve()
    overlap_path = Path(public_overlap_report_path).resolve()
    destination = Path(output_dir).resolve()
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "collect_faceting_evidence.py"
    )
    if not script_path.is_file():
        raise FileNotFoundError(script_path)
    if destination.exists():
        raise FileExistsError(
            f"faceting evidence output must be create-only: {destination}"
        )
    if expected_faceting_protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            "faceting protocol hash is not the preregistered hash"
        )
    if (
        expected_public_overlap_report_sha256
        != EXPECTED_PUBLIC_OVERLAP_REPORT_SHA256
    ):
        raise ValueError("public overlap report hash is not preregistered")
    if not _valid_sha256(expected_h5m_sha256):
        raise ValueError("expected H5M SHA-256 is invalid")
    if not _valid_sha256(expected_physical_input_identity_sha256):
        raise ValueError("expected physical-input-identity SHA-256 is invalid")
    if (
        isinstance(maximum_projected_subtriangle_centroids, bool)
        or int(maximum_projected_subtriangle_centroids) < 1
        or int(maximum_projected_subtriangle_centroids)
        > HARD_MAXIMUM_PROJECTED_SUBTRIANGLE_CENTROIDS
    ):
        raise ValueError(
            "projection-centroid limit must be between 1 and the hard "
            f"cap of {HARD_MAXIMUM_PROJECTED_SUBTRIANGLE_CENTROIDS}"
        )
    if (
        isinstance(progress_every_triangle_incidences, bool)
        or int(progress_every_triangle_incidences) < 1
    ):
        raise ValueError("progress interval must be a positive integer")

    controls_before = {
        "refaceting_manifest_sha256": sha256_file(manifest_path),
        "refaceting_seal_sha256": sha256_file(build_seal_path),
        "native_qualification_sha256": sha256_file(native_report_path),
        "source_cad_audit_sha256": sha256_file(source_audit_path),
        "faceting_protocol_sha256": sha256_file(protocol_path),
        "source_map_sha256": sha256_file(mapping_path),
        "public_overlap_report_sha256": sha256_file(overlap_path),
        "h5m_sha256": sha256_file(dagmc),
        "implementation_sha256": sha256_file(Path(__file__)),
        "script_sha256": sha256_file(script_path),
    }
    expected_controls = {
        "refaceting_manifest_sha256": expected_refaceting_manifest_sha256,
        "refaceting_seal_sha256": expected_refaceting_seal_sha256,
        "native_qualification_sha256": expected_native_qualification_sha256,
        "source_cad_audit_sha256": expected_source_cad_audit_sha256,
        "faceting_protocol_sha256": expected_faceting_protocol_sha256,
        "source_map_sha256": expected_source_map_sha256,
        "public_overlap_report_sha256": (
            expected_public_overlap_report_sha256
        ),
        "h5m_sha256": expected_h5m_sha256,
        "implementation_sha256": controls_before["implementation_sha256"],
        "script_sha256": controls_before["script_sha256"],
    }
    if controls_before != expected_controls:
        raise ValueError("faceting evidence control hash mismatch")
    protocol = _load_bound_json(
        protocol_path,
        expected_faceting_protocol_sha256,
        "faceting protocol",
    )
    manifest = _load_bound_json(
        manifest_path,
        expected_refaceting_manifest_sha256,
        "refaceting manifest",
    )
    build_seal = _load_bound_json(
        build_seal_path,
        expected_refaceting_seal_sha256,
        "refaceting build seal",
    )
    native_qualification = _load_bound_json(
        native_report_path,
        expected_native_qualification_sha256,
        "native qualification",
    )
    source_cad_audit = _load_bound_json(
        source_audit_path,
        expected_source_cad_audit_sha256,
        "source-CAD audit",
    )
    source_map = _load_bound_json(
        mapping_path, expected_source_map_sha256, "source map"
    )
    public_report = _load_bound_json(
        overlap_path,
        expected_public_overlap_report_sha256,
        "public overlap report",
    )
    settings = _protocol_level(protocol, level)
    _validate_refaceting_manifest(
        manifest,
        level=level,
        settings=settings,
        protocol_sha256=expected_faceting_protocol_sha256,
        h5m_sha256=expected_h5m_sha256,
        expected_physical_identity_sha256=(
            expected_physical_input_identity_sha256
        ),
    )
    if (
        manifest["source_cad_qualification"]["source_audit_sha256"]
        != expected_source_cad_audit_sha256
    ):
        raise ValueError(
            "source-CAD audit differs from sealed refaceting input"
        )
    _validate_refaceting_seal(
        build_seal,
        manifest_sha256=expected_refaceting_manifest_sha256,
        h5m_sha256=expected_h5m_sha256,
        physical_input_identity_sha256=(
            expected_physical_input_identity_sha256
        ),
    )
    _validate_native_qualification(native_qualification, expected_h5m_sha256)
    exact_source_clearance_cm = _source_clearance_cm(source_cad_audit)
    native = _load_native_mesh(dagmc)
    targets, source_inventory, projection_targets = _load_source_targets(
        source_root=source_directory,
        source_map=source_map,
        volume_materials=native.volume_materials,
        refaceting_manifest=manifest,
    )
    snapshot, incidences = _collect_snapshot(
        native,
        targets,
        level=level,
        protocol_sha256=expected_faceting_protocol_sha256,
        h5m_sha256=expected_h5m_sha256,
        criteria_sha256=manifest["acceptance_criteria"]["sha256"],
        physical_input_identity=manifest["physical_input_identity"],
        physical_input_identity_sha256=(
            expected_physical_input_identity_sha256
        ),
        source_inventory=source_inventory,
        projection_targets=projection_targets,
        faceting_settings=settings,
    )
    estimated_leaf_samples = _estimate_leaf_sample_count(
        incidences,
        maximum_count=int(maximum_projected_subtriangle_centroids),
    )
    snapshot["estimated_projected_subtriangle_centroid_count"] = (
        estimated_leaf_samples
    )
    snapshot["maximum_projected_subtriangle_centroids"] = int(
        maximum_projected_subtriangle_centroids
    )
    destination.mkdir(parents=True, exist_ok=False)
    pydagmc_reconciliation = _pydagmc_reconciliation(
        dagmc, snapshot["semantic_volumes"]
    )
    snapshot["pydagmc_topology"] = pydagmc_reconciliation
    snapshot["pydagmc_topology_gate_pass"] = pydagmc_reconciliation["pass"]
    snapshot["native_qualification"] = native_qualification
    deviation = _collect_deviation_certificate(
        incidences,
        targets,
        snapshot,
        level=level,
        protocol_sha256=expected_faceting_protocol_sha256,
        h5m_sha256=expected_h5m_sha256,
        physical_input_identity_sha256=(
            expected_physical_input_identity_sha256
        ),
        source_inventory=source_inventory,
        projection_targets=projection_targets,
        progress_path=destination / "projection_progress.json",
        progress_every_triangle_incidences=(
            progress_every_triangle_incidences
        ),
    )
    snapshot["faceted_clearance_cm"] = (
        exact_source_clearance_cm - 2.0 * deviation["upper_bound_cm"]
    )
    zones = _collect_former_overlap_zones(
        incidences,
        level=level,
        public_report=public_report,
        public_report_sha256=expected_public_overlap_report_sha256,
    )

    controls_after = {
        "refaceting_manifest_sha256": sha256_file(manifest_path),
        "refaceting_seal_sha256": sha256_file(build_seal_path),
        "native_qualification_sha256": sha256_file(native_report_path),
        "source_cad_audit_sha256": sha256_file(source_audit_path),
        "faceting_protocol_sha256": sha256_file(protocol_path),
        "source_map_sha256": sha256_file(mapping_path),
        "public_overlap_report_sha256": sha256_file(overlap_path),
        "h5m_sha256": sha256_file(dagmc),
        "implementation_sha256": sha256_file(Path(__file__)),
        "script_sha256": sha256_file(script_path),
    }
    source_inventory_after = []
    for row in source_inventory:
        path = _safe_relative_path(source_directory, str(row["source_step"]))
        source_inventory_after.append(
            {
                **dict(row),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    _assert_input_immutability(
        controls_before,
        controls_after,
        source_inventory,
        source_inventory_after,
        phase="during collection",
    )
    snapshot["raw_h5m_sha256_after"] = controls_after["h5m_sha256"]
    snapshot["h5m_unchanged"] = controls_after["h5m_sha256"] == (
        expected_h5m_sha256
    )
    deviation["raw_h5m_sha256_after"] = controls_after["h5m_sha256"]
    deviation["h5m_unchanged"] = snapshot["h5m_unchanged"]
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "scientific_decision": "NOT_PERFORMED_EVIDENCE_ONLY",
        "level": level,
        "coordinate_units": "cm",
        "controls_before": controls_before,
        "controls_after": controls_after,
        "control_immutability_pass": controls_before == controls_after,
        "runtime_identity": _runtime_identity(),
        "snapshot": snapshot,
        "deviation_certificate": deviation,
        "former_overlap_zones": zones,
    }
    evidence_path = destination / "faceting_evidence.json"
    _write_json_atomic(evidence_path, evidence)
    controls_final = {
        "refaceting_manifest_sha256": sha256_file(manifest_path),
        "refaceting_seal_sha256": sha256_file(build_seal_path),
        "native_qualification_sha256": sha256_file(native_report_path),
        "source_cad_audit_sha256": sha256_file(source_audit_path),
        "faceting_protocol_sha256": sha256_file(protocol_path),
        "source_map_sha256": sha256_file(mapping_path),
        "public_overlap_report_sha256": sha256_file(overlap_path),
        "h5m_sha256": sha256_file(dagmc),
        "implementation_sha256": sha256_file(Path(__file__)),
        "script_sha256": sha256_file(script_path),
    }
    source_inventory_final = []
    for row in source_inventory:
        path = _safe_relative_path(source_directory, str(row["source_step"]))
        source_inventory_final.append(
            {
                **dict(row),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    _assert_input_immutability(
        controls_before,
        controls_final,
        source_inventory,
        source_inventory_final,
        phase="before terminal sealing",
    )
    seal = {
        "schema": SEAL_SCHEMA,
        "scientific_decision": "NOT_PERFORMED_EVIDENCE_ONLY",
        "level": level,
        "faceting_evidence_sha256": sha256_file(evidence_path),
        "faceting_protocol_sha256": expected_faceting_protocol_sha256,
        "raw_h5m_sha256": expected_h5m_sha256,
        "physical_input_identity_sha256": (
            expected_physical_input_identity_sha256
        ),
        "refaceting_build_seal_sha256": expected_refaceting_seal_sha256,
        "source_map_sha256": expected_source_map_sha256,
        "public_overlap_report_sha256": (
            expected_public_overlap_report_sha256
        ),
        "control_immutability_pass": evidence["control_immutability_pass"],
        "h5m_unchanged": snapshot["h5m_unchanged"],
        "exact_triangle_reconciliation_pass": deviation[
            "exact_triangle_reconciliation_pass"
        ],
        "certified_upper_bound": deviation["certified_upper_bound"],
        "projection_progress_sha256": sha256_file(
            destination / "projection_progress.json"
        ),
        "controls_final": controls_final,
        "controls_final_match": controls_final == controls_before,
    }
    seal_path = destination / "faceting_evidence_seal.json"
    pending_seal_path = destination / ".faceting_evidence_seal.json.pending"
    _write_json_atomic(pending_seal_path, seal)
    controls_terminal = {
        "refaceting_manifest_sha256": sha256_file(manifest_path),
        "refaceting_seal_sha256": sha256_file(build_seal_path),
        "native_qualification_sha256": sha256_file(native_report_path),
        "source_cad_audit_sha256": sha256_file(source_audit_path),
        "faceting_protocol_sha256": sha256_file(protocol_path),
        "source_map_sha256": sha256_file(mapping_path),
        "public_overlap_report_sha256": sha256_file(overlap_path),
        "h5m_sha256": sha256_file(dagmc),
        "implementation_sha256": sha256_file(Path(__file__)),
        "script_sha256": sha256_file(script_path),
    }
    source_inventory_terminal = []
    for row in source_inventory:
        path = _safe_relative_path(source_directory, str(row["source_step"]))
        source_inventory_terminal.append(
            {
                **dict(row),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    _assert_input_immutability(
        controls_before,
        controls_terminal,
        source_inventory,
        source_inventory_terminal,
        phase="after seal serialization and before terminal publication",
    )
    pending_seal_path.replace(seal_path)
    if json.loads(seal_path.read_text(encoding="utf-8")) != seal:
        raise RuntimeError("terminal faceting seal readback differs")
    if sha256_file(evidence_path) != seal["faceting_evidence_sha256"]:
        raise RuntimeError("faceting evidence changed after sealing")
    if (
        sha256_file(destination / "projection_progress.json")
        != seal["projection_progress_sha256"]
    ):
        raise RuntimeError("projection progress changed after sealing")
    return {
        "evidence_path": str(evidence_path),
        "evidence_sha256": seal["faceting_evidence_sha256"],
        "seal_path": str(seal_path),
        "seal_sha256": sha256_file(seal_path),
        "scientific_decision": "NOT_PERFORMED_EVIDENCE_ONLY",
    }


__all__ = [
    "DEVIATION_SCHEMA",
    "EVIDENCE_SCHEMA",
    "EXPECTED_PROTOCOL_SHA256",
    "EXPECTED_PUBLIC_OVERLAP_REPORT_SHA256",
    "FORMER_ZONE_RADIUS_CM",
    "HARD_MAXIMUM_PROJECTED_SUBTRIANGLE_CENTROIDS",
    "MAXIMUM_COVER_RADIUS_CM",
    "SNAPSHOT_SCHEMA",
    "SOLVER_MARGIN_CM",
    "SOURCE_MAP_SCHEMA",
    "combine_former_overlap_zone_evidence",
    "collect_faceting_evidence",
]
