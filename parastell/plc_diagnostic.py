"""Deterministic ownership-aware diagnostics for faceted PLCs.

The routines in this module inspect a surface ledger produced by ParaStell's
authoritative native geometry builder.  They do not construct or repair
geometry.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import traceback

import numpy as np

from .native_port_geometry import _coordinate_key, _rectangle_boundary_indices


INTERSECTION_TOLERANCE_CM = 1.0e-8
SPATIAL_CELL_SIZE_CM = 20.0


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    port: bool
    num_ribs: int
    num_rib_pts: int

    def to_dict(self):
        return {
            "name": self.name,
            "port": self.port,
            "num_ribs": self.num_ribs,
            "num_rib_pts": self.num_rib_pts,
            "resolution": f"{self.num_ribs}x{self.num_rib_pts}",
        }


CASE_MATRIX = (
    DiagnosticCase("A0", False, 13, 49),
    DiagnosticCase("A1", True, 13, 49),
    DiagnosticCase("B0", False, 17, 65),
    DiagnosticCase("B1", True, 17, 65),
    DiagnosticCase("C0", False, 21, 81),
    DiagnosticCase("C1", True, 21, 81),
)


def case_by_name(name):
    for case in CASE_MATRIX:
        if case.name == name:
            return case
    raise ValueError(f"Unknown PLC diagnostic case {name!r}")


def fresh_directory(path):
    """Create *path* and fail closed if anything already occupies it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def file_record(path, *, logical_name=None, external_path=None):
    path = Path(path)
    record = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }
    if logical_name is not None:
        record["logical_name"] = logical_name
    if external_path is not None:
        record["external_path"] = str(external_path)
    return record


def canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path


def exception_record(error):
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }


def _unit_normal(triangle):
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    length = np.linalg.norm(normal)
    return normal / length if length else normal


def _point_segment_distance(point, left, right):
    point = np.asarray(point, dtype=float)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    vector = right - left
    denominator = float(np.dot(vector, vector))
    if denominator == 0.0:
        return float(np.linalg.norm(point - left))
    parameter = np.clip(np.dot(point - left, vector) / denominator, 0.0, 1.0)
    return float(np.linalg.norm(point - (left + parameter * vector)))


def _polyline_distance(point, points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return None
    return min(
        _point_segment_distance(point, left, right)
        for left, right in zip(points, points[1:])
    )


def _segment_triangle(left, right, triangle, tolerance):
    """Return an interior segment/facet hit or ``None``.

    Shared-topology contacts are filtered by the caller.  Barycentric boundary
    hits are retained because Gmsh also rejects a segment crossing a facet edge.
    """
    direction = right - left
    edge_1 = triangle[1] - triangle[0]
    edge_2 = triangle[2] - triangle[0]
    cross = np.cross(direction, edge_2)
    determinant = float(np.dot(edge_1, cross))
    scale = max(
        np.linalg.norm(direction)
        * np.linalg.norm(edge_1)
        * np.linalg.norm(edge_2),
        1.0,
    )
    if abs(determinant) <= tolerance * scale:
        return None
    inverse = 1.0 / determinant
    displacement = left - triangle[0]
    u = float(inverse * np.dot(displacement, cross))
    q = np.cross(displacement, edge_1)
    v = float(inverse * np.dot(direction, q))
    parameter = float(inverse * np.dot(edge_2, q))
    edge_length = float(np.linalg.norm(direction))
    parameter_tolerance = tolerance / max(edge_length, tolerance)
    if (
        u < -parameter_tolerance
        or v < -parameter_tolerance
        or u + v > 1.0 + parameter_tolerance
        or parameter <= parameter_tolerance
        or parameter >= 1.0 - parameter_tolerance
    ):
        return None
    return {
        "point": left + parameter * direction,
        "segment_parameter": parameter,
        "barycentric": (1.0 - u - v, u, v),
    }


def entity_ledger(complex_):
    """Assign deterministic vertex, edge and facet IDs with full ownership."""
    vertex_ids = {}
    vertices = {}
    facets = []
    edges = {}
    tolerance = complex_.vertex_merge_tolerance
    for surface_id, surface in enumerate(complex_.surfaces, start=1):
        for local_facet_id, triangle in enumerate(surface.triangles):
            ids = []
            for point in triangle:
                key = _coordinate_key(point, tolerance)
                if key not in vertex_ids:
                    vertex_id = len(vertex_ids) + 1
                    vertex_ids[key] = vertex_id
                    vertices[vertex_id] = np.asarray(point, dtype=float)
                ids.append(vertex_ids[key])
            facet_id = len(facets) + 1
            facet = {
                "id": facet_id,
                "surface_id": surface_id,
                "surface_local_id": local_facet_id,
                "surface_name": surface.name,
                "surface_kind": surface.kind,
                "reverse_volume": surface.reverse_volume,
                "forward_volume": surface.forward_volume,
                "vertex_ids": tuple(ids),
                "triangle": np.asarray(triangle, dtype=float),
            }
            facets.append(facet)
            for left_index, right_index in ((0, 1), (1, 2), (2, 0)):
                key = tuple(sorted((ids[left_index], ids[right_index])))
                if key not in edges:
                    edges[key] = {
                        "id": len(edges) + 1,
                        "vertex_ids": key,
                        "endpoints": np.asarray(
                            (vertices[key[0]], vertices[key[1]]), dtype=float
                        ),
                        "owners": [],
                    }
                edges[key]["owners"].append(
                    {
                        "facet_id": facet_id,
                        "surface_id": surface_id,
                        "surface_name": surface.name,
                        "surface_kind": surface.kind,
                        "reverse_volume": surface.reverse_volume,
                        "forward_volume": surface.forward_volume,
                    }
                )
    return {
        "vertices": vertices,
        "facets": facets,
        "edges": sorted(edges.values(), key=lambda item: item["id"]),
    }


def _grid_cells(lower, upper, cell_size):
    first = np.floor(np.asarray(lower) / cell_size).astype(int)
    last = np.floor(np.asarray(upper) / cell_size).astype(int)
    count = int(np.prod(last - first + 1))
    if count > 4096:
        return None
    return (
        (x, y, z)
        for x in range(first[0], last[0] + 1)
        for y in range(first[1], last[1] + 1)
        for z in range(first[2], last[2] + 1)
    )


def _spatial_index(facets, tolerance, cell_size):
    cells = defaultdict(list)
    global_facets = []
    for facet in facets:
        triangle = facet["triangle"]
        keys = _grid_cells(
            triangle.min(axis=0) - tolerance,
            triangle.max(axis=0) + tolerance,
            cell_size,
        )
        if keys is None:
            global_facets.append(facet["id"])
        else:
            for key in keys:
                cells[key].append(facet["id"])
    return cells, global_facets


def _surface_identity(item):
    return {
        "surface_id": item["surface_id"],
        "surface_name": item["surface_name"],
        "component_or_layer": item["surface_name"].split(":")[-1],
        "surface_kind": item["surface_kind"],
        "reverse_volume": item["reverse_volume"],
        "forward_volume": item["forward_volume"],
        "owning_volumes": [
            value
            for value in (item["reverse_volume"], item["forward_volume"])
            if value is not None
        ],
    }


def _nearest_grid_location(complex_, point):
    if not getattr(complex_, "radial_data", {}).get("grids"):
        return {}
    best = None
    for radial_index, grid in enumerate(complex_.radial_data["grids"]):
        distances = np.linalg.norm(grid - point, axis=2)
        flat_index = int(np.argmin(distances))
        phi_index, theta_index = (
            int(value)
            for value in np.unravel_index(flat_index, distances.shape)
        )
        candidate = float(distances[phi_index, theta_index])
        if best is None or candidate < best[0]:
            best = (candidate, int(radial_index), phi_index, theta_index)
    _, radial_index, phi_index, theta_index = best
    phi_values = complex_.radial_data["phi_values"]
    theta_values = complex_.radial_data["theta_values"]
    return {
        "nearest_radial_surface": complex_.radial_data["names"][radial_index],
        "radial_surface_index": radial_index,
        "toroidal_cell_index": min(phi_index, len(phi_values) - 2),
        "poloidal_cell_index": theta_index % len(theta_values),
        "toroidal_parameter_degrees": float(np.rad2deg(phi_values[phi_index])),
        "poloidal_parameter_degrees": float(
            np.rad2deg(theta_values[theta_index]) % 360.0
        ),
        "distance_to_nearest_radial_grid_vertex_cm": best[0],
    }


def _port_distances(complex_, point):
    loops = getattr(complex_, "loops", ())
    if not loops:
        return {
            "aperture_boundary_cm": None,
            "liner_boundary_cm": None,
            "patch_boundary_cm": None,
        }
    aperture = min(
        _polyline_distance(point, loop.inner_points) for loop in loops
    )
    liner_values = [
        _polyline_distance(point, loop.outer_points)
        for loop in loops
        if loop.outer_points is not None
    ]
    bounds = None
    if loops and complex_.radial_data.get("radial_loop_indices"):
        loop_index = next(
            (
                index
                for index in complex_.radial_data["radial_loop_indices"]
                if index is not None
            ),
            None,
        )
        if loop_index is not None:
            phi_values = complex_.radial_data["phi_values"]
            theta_values = complex_.radial_data["theta_values"]
            port = complex_.port
            anchor_spec = port.placement.surface_anchor
            phi = np.deg2rad(anchor_spec.toroidal_angle)
            theta = np.deg2rad(anchor_spec.poloidal_angle)
            surface = complex_.source_model.native_radial_stack()[0][1]
            delta = 1.0e-5
            phi_speed = np.linalg.norm(
                surface.evaluate(phi + delta, theta)
                - surface.evaluate(phi - delta, theta)
            ) / (2.0 * delta)
            theta_speed = np.linalg.norm(
                surface.evaluate(phi, theta + delta)
                - surface.evaluate(phi, theta - delta)
            ) / (2.0 * delta)
            half_width = (
                complex_.source_model._port_aperture_half_width(port) * 1.8
            )
            angular_bounds = (
                phi - half_width / phi_speed,
                phi + half_width / phi_speed,
                theta - half_width / theta_speed,
                theta + half_width / theta_speed,
            )
            indices = _rectangle_boundary_indices(
                phi_values, theta_values, angular_bounds
            )
            boundary_distances = []
            for grid in complex_.radial_data["grids"]:
                boundary = np.asarray([grid[index] for index in indices])
                boundary = np.vstack((boundary, boundary[0]))
                boundary_distances.append(_polyline_distance(point, boundary))
            bounds = min(boundary_distances)
    return {
        "aperture_boundary_cm": aperture,
        "liner_boundary_cm": min(liner_values) if liner_values else None,
        "patch_boundary_cm": bounds,
    }


def _local_coordinates(complex_, point):
    base = {
        "cartesian_cm": point.tolist(),
        "cylindrical_toroidal_degrees": float(
            np.rad2deg(np.arctan2(point[1], point[0])) % 360.0
        ),
    }
    if not hasattr(complex_, "port"):
        return {**base, **_nearest_grid_location(complex_, point)}
    port = complex_.port
    anchor = np.asarray(port.placement.anchor)
    displacement = point - anchor
    axis = np.asarray(port.placement.local_axis)
    reference = np.asarray(port.placement.local_reference)
    normal = np.asarray(port.placement.local_normal)
    return {
        **base,
        "port_local_radial_or_axial_w_cm": float(np.dot(displacement, axis)),
        "port_local_poloidal_u_cm": float(np.dot(displacement, reference)),
        "port_local_toroidal_v_cm": float(np.dot(displacement, normal)),
        **_nearest_grid_location(complex_, point),
    }


def inspect_edge_facet_intersections(
    complex_,
    *,
    tolerance_cm=INTERSECTION_TOLERANCE_CM,
    cell_size_cm=SPATIAL_CELL_SIZE_CM,
    max_intersections=10,
):
    """Locate non-topological segment/facet crossings in a native PLC."""
    ledger = entity_ledger(complex_)
    facets = ledger["facets"]
    facet_by_id = {facet["id"]: facet for facet in facets}
    facet_lowers = np.asarray(
        [facet["triangle"].min(axis=0) for facet in facets]
    )
    facet_uppers = np.asarray(
        [facet["triangle"].max(axis=0) for facet in facets]
    )
    cells, global_facets = _spatial_index(facets, tolerance_cm, cell_size_cm)
    intersections = []
    tested_pairs = set()
    for edge in ledger["edges"]:
        endpoints = edge["endpoints"]
        keys = _grid_cells(
            endpoints.min(axis=0) - tolerance_cm,
            endpoints.max(axis=0) + tolerance_cm,
            cell_size_cm,
        )
        candidates = set(global_facets)
        if keys is None:
            candidates.update(facet_by_id)
        else:
            for key in keys:
                candidates.update(cells.get(key, ()))
        candidate_indices = np.fromiter(
            (facet_id - 1 for facet_id in candidates), dtype=int
        )
        if len(candidate_indices):
            edge_lower = endpoints.min(axis=0) - tolerance_cm
            edge_upper = endpoints.max(axis=0) + tolerance_cm
            overlaps = np.all(
                facet_lowers[candidate_indices] <= edge_upper, axis=1
            ) & np.all(facet_uppers[candidate_indices] >= edge_lower, axis=1)
            candidates = set((candidate_indices[overlaps] + 1).tolist())
        owner_facets = {owner["facet_id"] for owner in edge["owners"]}
        edge_vertices = set(edge["vertex_ids"])
        for facet_id in sorted(candidates):
            pair = (edge["id"], facet_id)
            if pair in tested_pairs:
                continue
            tested_pairs.add(pair)
            facet = facet_by_id[facet_id]
            if facet_id in owner_facets or edge_vertices.intersection(
                facet["vertex_ids"]
            ):
                continue
            hit = _segment_triangle(
                endpoints[0], endpoints[1], facet["triangle"], tolerance_cm
            )
            if hit is None:
                continue
            point = hit["point"]
            edge_owner_identities = []
            for owner in edge["owners"]:
                owner_facet = facet_by_id[owner["facet_id"]]
                edge_owner_identities.append(
                    {
                        "facet_id": owner_facet["id"],
                        "surface_local_id": owner_facet["surface_local_id"],
                        "vertex_ids": list(owner_facet["vertex_ids"]),
                        "vertices_cm": owner_facet["triangle"].tolist(),
                        "normal": _unit_normal(
                            owner_facet["triangle"]
                        ).tolist(),
                        "orientation": {
                            "normal_points_from": owner_facet[
                                "reverse_volume"
                            ],
                            "normal_points_to": owner_facet["forward_volume"],
                        },
                        **_surface_identity(owner),
                    }
                )
            triangle_identity = _surface_identity(facet)
            edge_surfaces = {item["surface_name"] for item in edge["owners"]}
            triangle_surface = facet["surface_name"]
            radial_names = getattr(complex_, "radial_data", {}).get(
                "names", ()
            )
            involved_names = edge_surfaces | {triangle_surface}
            radial_indices = sorted(
                radial_names.index(name.removeprefix("radial:"))
                for name in involved_names
                if name.startswith("radial:")
                and name.removeprefix("radial:") in radial_names
            )
            adjacent = (
                len(radial_indices) >= 2
                and max(radial_indices) - min(radial_indices) == 1
            )
            triangle_edge_distances = [
                _point_segment_distance(point, left, right)
                for left, right in zip(
                    facet["triangle"], np.roll(facet["triangle"], -1, axis=0)
                )
            ]
            intersection = {
                "intersection_id": len(intersections) + 1,
                "edge": {
                    "id": edge["id"],
                    "vertex_ids": list(edge["vertex_ids"]),
                    "endpoints_cm": endpoints.tolist(),
                    "owners": edge_owner_identities,
                },
                "facet": {
                    "id": facet_id,
                    "surface_local_id": facet["surface_local_id"],
                    "vertex_ids": list(facet["vertex_ids"]),
                    "vertices_cm": facet["triangle"].tolist(),
                    "normal": _unit_normal(facet["triangle"]).tolist(),
                    "orientation": {
                        "normal_points_from": facet["reverse_volume"],
                        "normal_points_to": facet["forward_volume"],
                    },
                    **triangle_identity,
                },
                "intersection_coordinates": _local_coordinates(
                    complex_, point
                ),
                "barycentric_coordinates": list(hit["barycentric"]),
                "segment_parameter": hit["segment_parameter"],
                "relationship": (
                    "self_surface"
                    if triangle_surface in edge_surfaces
                    else (
                        "adjacent_radial_surfaces"
                        if adjacent
                        else "inter_surface"
                    )
                ),
                "distance_from_port": _port_distances(complex_, point),
                "geometric_violation_length_scale_cm": min(
                    np.linalg.norm(point - endpoints[0]),
                    np.linalg.norm(point - endpoints[1]),
                    *triangle_edge_distances,
                ),
                "intersection_tolerance_cm": tolerance_cm,
            }
            intersections.append(intersection)
            if len(intersections) >= max_intersections:
                break
        if len(intersections) >= max_intersections:
            break
    return {
        "schema_version": "1.0",
        "scope": "PLC/topology diagnostic; not qualified transport geometry",
        "intersection_test": "non-coplanar segment-triangle Moller-Trumbore",
        "intersection_tolerance_cm": tolerance_cm,
        "spatial_cell_size_cm": cell_size_cm,
        "max_intersections": max_intersections,
        "truncated": len(intersections) >= max_intersections,
        "entity_counts": {
            "volumes": len(complex_.volumes),
            "surfaces": len(complex_.surfaces),
            "facets": len(facets),
            "vertices": len(ledger["vertices"]),
            "edges": len(ledger["edges"]),
            "surface_kinds": dict(
                sorted(
                    Counter(item.kind for item in complex_.surfaces).items()
                )
            ),
        },
        "intersection_count": len(intersections),
        "intersections": intersections,
        "_ledger": ledger,
    }


def serializable_intersection_report(report):
    return {key: value for key, value in report.items() if key != "_ledger"}


def write_minimal_vtk(path, report):
    """Write only the first implicated facet and facets owning its edge."""
    if not report["intersections"]:
        raise ValueError(
            "Cannot export implicated facets without an intersection"
        )
    intersection = report["intersections"][0]
    facet_ids = {intersection["facet"]["id"]}
    facet_ids.update(
        owner["facet_id"]
        for owner in report["_ledger"]["edges"][
            intersection["edge"]["id"] - 1
        ]["owners"]
    )
    facets = [
        report["_ledger"]["facets"][facet_id - 1]
        for facet_id in sorted(facet_ids)
    ]
    points = np.concatenate([facet["triangle"] for facet in facets], axis=0)
    lines = [
        "# vtk DataFile Version 3.0",
        "ParaStell minimal implicated PLC facets; not transport geometry",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} double",
    ]
    lines.extend(
        " ".join(f"{value:.17g}" for value in point) for point in points
    )
    lines.append(f"POLYGONS {len(facets)} {len(facets) * 4}")
    lines.extend(
        f"3 {index * 3} {index * 3 + 1} {index * 3 + 2}"
        for index in range(len(facets))
    )
    lines.extend(
        (
            f"CELL_DATA {len(facets)}",
            "SCALARS facet_id int 1",
            "LOOKUP_TABLE default",
        )
    )
    lines.extend(str(facet["id"]) for facet in facets)
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def classify_case(port_stitched, intersection_count, downstream_error):
    if intersection_count:
        return (
            "BLOCKED_PORT_STITCH_PLC_INTERSECTION"
            if port_stitched
            else "BLOCKED_BASELINE_RADIAL_PLC_INTERSECTION"
        )
    if downstream_error is not None:
        return "BLOCKED_OTHER_REPRODUCIBLE_PLC_FAILURE"
    return "PASS_NO_REPRODUCIBLE_PLC_FAILURE"


def terminal_classification(case_results):
    baseline = any(
        item["intersection_count"] > 0 and not item["case"]["port"]
        for item in case_results
    )
    ported = any(
        item["intersection_count"] > 0 and item["case"]["port"]
        for item in case_results
    )
    if baseline and ported:
        return "BLOCKED_BASELINE_AND_PORTED_PLC_INTERSECTION"
    if baseline:
        return "BLOCKED_BASELINE_RADIAL_PLC_INTERSECTION"
    if ported:
        return "BLOCKED_PORT_STITCH_PLC_INTERSECTION"
    if any(item.get("downstream_error") for item in case_results):
        return "BLOCKED_OTHER_REPRODUCIBLE_PLC_FAILURE"
    if len(case_results) != len(CASE_MATRIX):
        return "BLOCKED_ENVIRONMENT_OR_INPUT_IDENTITY"
    return "PASS_NO_REPRODUCIBLE_PLC_FAILURE"
