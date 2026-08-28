"""Closed magnet-envelope extraction from a production DAGMC volume."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .magnet_boundary_envelope import EnvelopeSurface
from .magnet_boundary_envelope import MagnetBoundaryEnvelope


DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM = 1.0e-6
DEFAULT_FACET_BARYCENTRIC_TOLERANCE = 1.0e-7
DEFAULT_FACET_SOURCE_TOLERANCE_CM = 1.0e-5


def canonical_geometry_policy(
    coordinate_quantum_cm=DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM,
    faceting_tolerances=None,
) -> dict[str, Any]:
    """Normalize the one policy that defines canonical DAGMC identity."""
    quantum = float(coordinate_quantum_cm)
    if not np.isfinite(quantum) or quantum <= 0.0:
        raise ValueError("coordinate_quantum_cm must be positive and finite")
    tolerances = dict(faceting_tolerances or {})
    try:
        json.dumps(tolerances, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "faceting_tolerances must be a finite JSON mapping"
        ) from exc
    return {
        "coordinate_quantum_cm": quantum,
        "faceting_tolerances": tolerances,
    }


def _unit(value, name):
    array = np.asarray(value, dtype=float)
    norm = np.linalg.norm(array)
    if array.shape != (3,) or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must be a finite nonzero three-vector")
    return array / norm


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantize_coordinates(coordinates, quantum: float) -> np.ndarray:
    """Map finite coordinates to the declared canonical integer grid."""
    values = np.asarray(coordinates, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("DAGMC coordinates must be finite")
    scaled = values / quantum
    limit = np.nextafter(float(np.iinfo(np.int64).max), 0.0)
    if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) > limit):
        raise ValueError("DAGMC coordinates exceed the canonical integer grid")
    return np.rint(scaled).astype(np.int64)


def _canonical_oriented_connectivity(triangle) -> tuple[int, int, int]:
    """Canonicalize a triangle start vertex while preserving its winding."""
    indices = tuple(int(value) for value in triangle)
    return min(indices[index:] + indices[:index] for index in range(3))


def _canonical_oriented_vertices(triangle):
    """Canonicalize quantized vertices without discarding triangle winding."""
    vertices = tuple(
        tuple(int(value) for value in vertex) for vertex in triangle
    )
    return min(vertices[index:] + vertices[:index] for index in range(3))


def canonical_dagmc_fingerprint(
    dagmc_path,
    *,
    coordinate_quantum_cm=DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM,
    faceting_tolerances=None,
):
    """Return a stable geometry/topology identity independent of H5M ordering."""
    import pydagmc

    policy = canonical_geometry_policy(
        coordinate_quantum_cm, faceting_tolerances
    )
    quantum = policy["coordinate_quantum_cm"]
    model = pydagmc.Model(str(Path(dagmc_path).resolve()))
    volume_objects = []
    surface_objects = {}
    surface_senses = {}
    quantized_surface_triangles = {}
    quantized_vertex_set = set()
    for volume_id, volume in sorted(model.volumes_by_id.items()):
        surface_ids = []
        for surface in sorted(volume.surfaces, key=lambda item: int(item.id)):
            surface_id = int(surface.id)
            surface_ids.append(surface_id)
            quantized = _quantize_coordinates(
                _triangles(surface.triangle_coords), quantum
            )
            candidate = tuple(
                sorted(
                    _canonical_oriented_vertices(triangle)
                    for triangle in quantized
                )
            )
            forward = getattr(surface, "forward_volume", None)
            reverse = getattr(surface, "reverse_volume", None)
            sense = (
                int(forward.id) if forward is not None else None,
                int(reverse.id) if reverse is not None else None,
            )
            if surface_id in quantized_surface_triangles:
                previous = quantized_surface_triangles[surface_id]
                if previous != candidate:
                    raise ValueError(
                        f"DAGMC surface {surface_id} has inconsistent facets"
                    )
                if surface_senses[surface_id] != sense:
                    raise ValueError(
                        f"DAGMC surface {surface_id} has inconsistent senses"
                    )
            else:
                surface_objects[surface_id] = surface
                surface_senses[surface_id] = sense
                quantized_surface_triangles[surface_id] = candidate
                quantized_vertex_set.update(
                    vertex for triangle in candidate for vertex in triangle
                )
        volume_objects.append(
            {
                "volume_id": int(volume_id),
                "component_name": str(getattr(volume, "name", "")),
                "component_id": str(
                    getattr(volume, "component_id", "")
                    or getattr(volume, "name", "")
                    or f"dagmc-volume-{int(volume_id)}"
                ),
                "material_tag": str(getattr(volume, "material", "")),
                "surface_ids": surface_ids,
            }
        )
    quantized_vertices = tuple(sorted(quantized_vertex_set))
    vertex_indices = {
        vertex: index for index, vertex in enumerate(quantized_vertices)
    }
    surfaces = {}
    for surface_id in sorted(surface_objects):
        surface = surface_objects[surface_id]
        connectivity = tuple(
            sorted(
                _canonical_oriented_connectivity(
                    tuple(vertex_indices[vertex] for vertex in triangle)
                )
                for triangle in quantized_surface_triangles[surface_id]
            )
        )
        coordinates = (
            np.asarray(
                [
                    [quantized_vertices[index] for index in triangle]
                    for triangle in connectivity
                ],
                dtype=float,
            )
            * quantum
        )
        area_vectors = 0.5 * np.cross(
            coordinates[:, 1] - coordinates[:, 0],
            coordinates[:, 2] - coordinates[:, 0],
        )
        areas = np.linalg.norm(area_vectors, axis=1)
        total_area = float(areas.sum())
        if not np.isfinite(total_area) or total_area <= 0.0:
            raise ValueError(
                f"DAGMC surface {surface_id} has zero faceted area"
            )
        centroid = (
            np.sum(coordinates.mean(axis=1) * areas[:, None], axis=0)
            / total_area
        )
        forward_id, reverse_id = surface_senses[surface_id]
        surfaces[surface_id] = {
            "surface_id": surface_id,
            "forward_volume_id": forward_id,
            "reverse_volume_id": reverse_id,
            "area_cm2": total_area,
            "centroid_cm": [float(value) for value in centroid],
            "canonical_triangle_connectivity": connectivity,
        }
    volumes = []
    for volume_object, volume in zip(
        volume_objects,
        [model.volumes_by_id[item["volume_id"]] for item in volume_objects],
    ):
        signed_six_volume = 0.0
        for surface in sorted(volume.surfaces, key=lambda item: int(item.id)):
            connectivity = surfaces[int(surface.id)][
                "canonical_triangle_connectivity"
            ]
            coordinates = (
                np.asarray(
                    [
                        [quantized_vertices[index] for index in triangle]
                        for triangle in connectivity
                    ],
                    dtype=float,
                )
                * quantum
            )
            cross = np.cross(coordinates[:, 1], coordinates[:, 2])
            native_six_volume = float(np.sum(coordinates[:, 0] * cross))
            signed_six_volume += (
                _openmc_to_outward_normal_sign(surface, int(volume.id))
                * native_six_volume
            )
        volume_object["volume_cm3"] = abs(signed_six_volume) / 6.0
        volumes.append(volume_object)
    canonical = {
        "canonicalization_version": 2,
        "coordinate_units": "cm",
        "coordinate_quantum_cm": quantum,
        "faceting_tolerances": policy["faceting_tolerances"],
        "quantized_vertices": quantized_vertices,
        "volumes": volumes,
        "surfaces": [surfaces[key] for key in sorted(surfaces)],
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "algorithm": "sha256",
        "canonical_fingerprint": hashlib.sha256(encoded).hexdigest(),
        "raw_h5m_sha256": _hash(dagmc_path),
        "coordinate_quantum_cm": quantum,
        "volume_count": len(volumes),
        "surface_count": len(surfaces),
        "canonical_geometry": canonical,
    }


@dataclass(frozen=True)
class FacetedSurface:
    surface_id: int
    role: str
    triangles_cm: np.ndarray
    triangle_normals_outward: np.ndarray
    triangle_areas_cm2: np.ndarray
    centroid_global_cm: np.ndarray
    canonical_facet_ids: tuple[str, ...] = ()
    dagmc_volume_id: int | None = None

    @property
    def area_cm2(self) -> float:
        return float(self.triangle_areas_cm2.sum())

    @staticmethod
    def _closest_point(
        point: np.ndarray, triangle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the closest triangle point and its constrained barycentrics."""
        a, b, c = np.asarray(triangle, dtype=float)
        ab = b - a
        ac = c - a
        ap = point - a
        d1 = float(np.dot(ab, ap))
        d2 = float(np.dot(ac, ap))
        if d1 <= 0.0 and d2 <= 0.0:
            return a, np.asarray([1.0, 0.0, 0.0])
        bp = point - b
        d3 = float(np.dot(ab, bp))
        d4 = float(np.dot(ac, bp))
        if d3 >= 0.0 and d4 <= d3:
            return b, np.asarray([0.0, 1.0, 0.0])
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            fraction = d1 / (d1 - d3)
            return a + fraction * ab, np.asarray(
                [1.0 - fraction, fraction, 0.0]
            )
        cp = point - c
        d5 = float(np.dot(ab, cp))
        d6 = float(np.dot(ac, cp))
        if d6 >= 0.0 and d5 <= d6:
            return c, np.asarray([0.0, 0.0, 1.0])
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            fraction = d2 / (d2 - d6)
            return a + fraction * ac, np.asarray(
                [1.0 - fraction, 0.0, fraction]
            )
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b + fraction * (c - b), np.asarray(
                [0.0, 1.0 - fraction, fraction]
            )
        denominator = 1.0 / (va + vb + vc)
        bary_u_value = vb * denominator
        bary_v_value = vc * denominator
        barycentric = np.asarray(
            [1.0 - bary_u_value - bary_v_value, bary_u_value, bary_v_value]
        )
        return np.einsum("i,ij->j", barycentric, triangle), barycentric

    def facet_metadata(
        self, centreline_frame: Any | None = None
    ) -> dict[str, Any]:
        """Return a canonical per-facet geometry and centreline catalog.

        Facet rows are sorted by their stable IDs rather than storage order in
        the H5M file.  A centreline is optional because an imported DAGMC model
        may not carry filament provenance.  In that case the catalog records
        the missing linkage explicitly and does not emit fabricated frame
        coordinates.
        """
        count = len(self.triangles_cm)
        facet_ids = np.asarray(
            self.canonical_facet_ids
            or tuple(
                f"surface-{self.surface_id}-facet-{index}"
                for index in range(count)
            ),
            dtype=object,
        )
        if facet_ids.shape != (count,):
            raise ValueError(
                f"surface {self.surface_id} facet IDs do not align with facets"
            )
        order = np.argsort(facet_ids.astype(str), kind="stable")
        centroids = np.asarray(self.triangles_cm, dtype=float).mean(axis=1)
        metadata: dict[str, Any] = {
            "facet_id": facet_ids[order],
            "canonical_facet_id": facet_ids[order],
            "facet_index": np.arange(count, dtype=int)[order],
            "surface_id": np.full(count, int(self.surface_id), dtype=int),
            "dagmc_volume_id": np.full(
                count,
                -1 if self.dagmc_volume_id is None else self.dagmc_volume_id,
                dtype=int,
            ),
            "surface_role": np.full(count, self.role, dtype=object),
            "triangle_vertices_global_cm": np.asarray(
                self.triangles_cm, dtype=float
            )[order],
            "facet_centroid_global_cm": centroids[order],
            "outward_normal_global": np.asarray(
                self.triangle_normals_outward, dtype=float
            )[order],
            "triangle_outward_normal_global": np.asarray(
                self.triangle_normals_outward, dtype=float
            )[order],
            "facet_area_cm2": np.asarray(self.triangle_areas_cm2, dtype=float)[
                order
            ],
            "triangle_area_cm2": np.asarray(
                self.triangle_areas_cm2, dtype=float
            )[order],
            "centreline_linkage_available": centreline_frame is not None,
            "centreline_linkage_status": np.full(
                count,
                (
                    "LINKED_NEAREST_CENTRELINE_SEGMENT"
                    if centreline_frame is not None
                    else "UNAVAILABLE_CENTRELINE_FRAME_NOT_SUPPLIED"
                ),
                dtype=object,
            ),
        }
        if centreline_frame is None:
            return metadata

        sampled = centreline_frame.sample(centroids)
        for name in (
            "nearest_centreline_global_cm",
            "centreline_tangent",
            "centreline_radial",
            "centreline_transverse",
            "local_centreline_coordinates_cm",
        ):
            values = np.asarray(sampled[name], dtype=float)
            if values.shape != (count, 3):
                raise ValueError(
                    f"centreline frame returned invalid {name} shape"
                )
            metadata[name] = values[order]
        for name in (
            "centreline_arclength_cm",
            "normalized_arclength",
            "distance_to_centreline_cm",
        ):
            values = np.asarray(sampled[name], dtype=float)
            if values.shape != (count,):
                raise ValueError(
                    f"centreline frame returned invalid {name} shape"
                )
            metadata[name] = values[order]
        metadata["frame_type"] = np.full(
            count, str(sampled["frame_type"]), dtype=object
        )
        metadata["frame_quality_status"] = np.full(
            count,
            str(sampled["frame_quality_status"]),
            dtype=object,
        )
        return metadata

    def locate(
        self,
        position_global_cm: Sequence[float],
        *,
        barycentric_tolerance: float = DEFAULT_FACET_BARYCENTRIC_TOLERANCE,
        source_tolerance_cm: float = DEFAULT_FACET_SOURCE_TOLERANCE_CM,
    ) -> dict[str, Any]:
        """Locate a crossing on a facet without fabricating nearest identity.

        A match is accepted only when the projected point is inside a facet up
        to the declared barycentric tolerance and the point-to-facet residual
        is no larger than ``source_tolerance_cm``.  Failed rows deliberately
        carry no canonical ID and are fatal to the production exporter.
        """
        point = np.asarray(position_global_cm, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("facet query position must be a finite vector")
        if barycentric_tolerance < 0.0 or source_tolerance_cm <= 0.0:
            raise ValueError("facet mapping tolerances must be positive")

        def select_unambiguous(candidates, residuals) -> int:
            candidates = np.asarray(candidates, dtype=int)
            if len(candidates) > 1:
                reference = int(candidates[0])
                reference_normal = self.triangle_normals_outward[reference]
                reference_origin = self.triangles_cm[reference, 0]
                for candidate in candidates[1:]:
                    candidate = int(candidate)
                    same_normal = (
                        np.dot(
                            reference_normal,
                            self.triangle_normals_outward[candidate],
                        )
                        >= 1.0 - 1.0e-10
                    )
                    same_plane = (
                        abs(
                            np.dot(
                                self.triangles_cm[candidate, 0]
                                - reference_origin,
                                reference_normal,
                            )
                        )
                        <= source_tolerance_cm
                    )
                    if not (same_normal and same_plane):
                        raise ValueError(
                            "crossing lies on an ambiguous noncoplanar facet "
                            "edge; OpenMC 0.16 does not store native hit-facet "
                            "identity"
                        )
            facet_ids = self.canonical_facet_ids or tuple(
                f"surface-{self.surface_id}-facet-{index}"
                for index in range(len(self.triangles_cm))
            )
            return int(
                min(
                    candidates,
                    key=lambda index: (
                        float(residuals[int(index)]),
                        str(facet_ids[int(index)]),
                    ),
                )
            )

        first = self.triangles_cm[:, 0]
        edge_u = self.triangles_cm[:, 1] - first
        edge_v = self.triangles_cm[:, 2] - first
        offset = point - first
        plane_distance = np.einsum(
            "ij,ij->i", offset, self.triangle_normals_outward
        )
        projected = (
            offset - plane_distance[:, None] * self.triangle_normals_outward
        )
        uu = np.einsum("ij,ij->i", edge_u, edge_u)
        uv = np.einsum("ij,ij->i", edge_u, edge_v)
        vv = np.einsum("ij,ij->i", edge_v, edge_v)
        pu = np.einsum("ij,ij->i", projected, edge_u)
        pv = np.einsum("ij,ij->i", projected, edge_v)
        denominator = uu * vv - uv * uv
        if np.any(denominator <= 0.0):
            raise ValueError(
                f"surface {self.surface_id} has a degenerate facet"
            )
        bary_u = (vv * pu - uv * pv) / denominator
        bary_v = (uu * pv - uv * pu) / denominator
        tolerance = float(barycentric_tolerance)
        contains = (
            (bary_u >= -tolerance)
            & (bary_v >= -tolerance)
            & (bary_u + bary_v <= 1.0 + tolerance)
            & (np.abs(plane_distance) <= source_tolerance_cm)
        )
        if np.any(contains):
            candidates = np.flatnonzero(contains)
            index = select_unambiguous(candidates, np.abs(plane_distance))
            barycentric = np.asarray(
                [
                    1.0 - bary_u[index] - bary_v[index],
                    bary_u[index],
                    bary_v[index],
                ]
            )
            near_boundary = barycentric <= tolerance
            boundary_count = int(np.count_nonzero(near_boundary))
            if boundary_count >= 2:
                mapping_status = "VERTEX_TOLERANCE_MATCH"
            elif boundary_count == 1:
                mapping_status = "EDGE_TOLERANCE_MATCH"
            else:
                mapping_status = "EXACT_FACET_MATCH"
        else:
            closest = [
                self._closest_point(point, triangle)
                for triangle in self.triangles_cm
            ]
            distances = np.asarray(
                [np.linalg.norm(point - row[0]) for row in closest]
            )
            close_candidates = np.flatnonzero(distances <= source_tolerance_cm)
            index = (
                select_unambiguous(close_candidates, distances)
                if len(close_candidates)
                else int(np.argmin(distances))
            )
            reconstructed, barycentric = closest[index]
            nearest_residual = float(distances[index])
            if nearest_residual <= source_tolerance_cm:
                boundary_count = int(
                    np.count_nonzero(barycentric <= tolerance)
                )
                mapping_status = (
                    "VERTEX_TOLERANCE_MATCH"
                    if boundary_count >= 2
                    else "EDGE_TOLERANCE_MATCH"
                )
            else:
                return {
                    "facet_id": "",
                    "canonical_facet_id": "",
                    "facet_index": -1,
                    "barycentric_coordinates": barycentric,
                    "reconstructed_position_global_cm": reconstructed,
                    "signed_plane_residual_cm": float(plane_distance[index]),
                    "nearest_point_residual_cm": nearest_residual,
                    "distance_to_facet_residual_cm": abs(
                        float(plane_distance[index])
                    ),
                    "inside_facet": False,
                    "outward_normal_global": np.full(3, np.nan),
                    "mapping_status": "NO_VALID_FACET_MATCH",
                }
        facet_id = (
            self.canonical_facet_ids[index]
            if self.canonical_facet_ids
            else f"surface-{self.surface_id}-facet-{index}"
        )
        reconstructed = np.einsum(
            "i,ij->j", barycentric, self.triangles_cm[index]
        )
        signed_residual = float(plane_distance[index])
        nearest_residual = float(np.linalg.norm(point - reconstructed))
        return {
            "facet_id": facet_id,
            "canonical_facet_id": facet_id,
            "facet_index": index,
            "barycentric_coordinates": barycentric,
            "reconstructed_position_global_cm": reconstructed,
            "signed_plane_residual_cm": signed_residual,
            "nearest_point_residual_cm": nearest_residual,
            # Backward-compatible v2.1 alias retained with its absolute meaning.
            "distance_to_facet_residual_cm": abs(signed_residual),
            "inside_facet": bool(np.all(barycentric >= -tolerance)),
            "outward_normal_global": self.triangle_normals_outward[
                index
            ].copy(),
            "mapping_status": mapping_status,
        }

    def normal_at(self, position_global_cm: Sequence[float]) -> np.ndarray:
        result = self.locate(position_global_cm)
        if result["mapping_status"] == "NO_VALID_FACET_MATCH":
            raise ValueError(
                f"position has no valid facet match on surface {self.surface_id}"
            )
        return result["outward_normal_global"]


@dataclass(frozen=True)
class DagmcEnvelope:
    envelope: MagnetBoundaryEnvelope
    faceted_surfaces: tuple[FacetedSurface, ...]
    triangle_edges: int
    maximum_edge_multiplicity_error: int
    vector_area_closure_relative: float
    centreline_frame: Any | None = None

    def surface(self, surface_id: int) -> FacetedSurface:
        for surface in self.faceted_surfaces:
            if surface.surface_id == int(surface_id):
                return surface
        raise KeyError(surface_id)

    def normals_at(self, surface_ids, positions_global_cm) -> np.ndarray:
        ids = np.asarray(surface_ids, dtype=int)
        positions = np.asarray(positions_global_cm, dtype=float)
        return np.asarray(
            [
                self.surface(sid).normal_at(point)
                for sid, point in zip(ids, positions)
            ]
        )

    def facet_mappings(
        self,
        surface_ids,
        positions_global_cm,
        *,
        require_valid: bool = False,
        barycentric_tolerance: float = DEFAULT_FACET_BARYCENTRIC_TOLERANCE,
        source_tolerance_cm: float = DEFAULT_FACET_SOURCE_TOLERANCE_CM,
    ) -> dict[str, np.ndarray]:
        """Map crossing points back to canonical DAGMC facets."""
        ids = np.asarray(surface_ids, dtype=int)
        positions = np.asarray(positions_global_cm, dtype=float)
        if positions.shape != (len(ids), 3):
            raise ValueError("surface IDs and positions must align")
        rows = [
            self.surface(surface_id).locate(
                point,
                barycentric_tolerance=barycentric_tolerance,
                source_tolerance_cm=source_tolerance_cm,
            )
            for surface_id, point in zip(ids, positions)
        ]
        invalid = [
            index
            for index, row in enumerate(rows)
            if row["mapping_status"] == "NO_VALID_FACET_MATCH"
        ]
        if require_valid and invalid:
            raise ValueError(
                "facet-complete export rejected NO_VALID_FACET_MATCH rows; "
                f"count={len(invalid)}, sample_record_indices={invalid[:10]}"
            )
        if not rows:
            return {
                "facet_id": np.empty(0, dtype=object),
                "canonical_facet_id": np.empty(0, dtype=object),
                "facet_index": np.empty(0, dtype=int),
                "barycentric_coordinates": np.empty((0, 3), dtype=float),
                "reconstructed_position_global_cm": np.empty(
                    (0, 3), dtype=float
                ),
                "signed_plane_residual_cm": np.empty(0, dtype=float),
                "nearest_point_residual_cm": np.empty(0, dtype=float),
                "distance_to_facet_residual_cm": np.empty(0, dtype=float),
                "inside_facet": np.empty(0, dtype=bool),
                "facet_mapping_status": np.empty(0, dtype=object),
                "outward_normal_global": np.empty((0, 3), dtype=float),
            }
        return {
            "facet_id": np.asarray(
                [row["facet_id"] for row in rows], dtype=object
            ),
            "canonical_facet_id": np.asarray(
                [row["canonical_facet_id"] for row in rows], dtype=object
            ),
            "facet_index": np.asarray(
                [row["facet_index"] for row in rows], dtype=int
            ),
            "barycentric_coordinates": np.asarray(
                [row["barycentric_coordinates"] for row in rows], dtype=float
            ),
            "reconstructed_position_global_cm": np.asarray(
                [row["reconstructed_position_global_cm"] for row in rows],
                dtype=float,
            ),
            "signed_plane_residual_cm": np.asarray(
                [row["signed_plane_residual_cm"] for row in rows],
                dtype=float,
            ),
            "nearest_point_residual_cm": np.asarray(
                [row["nearest_point_residual_cm"] for row in rows],
                dtype=float,
            ),
            "distance_to_facet_residual_cm": np.asarray(
                [row["distance_to_facet_residual_cm"] for row in rows],
                dtype=float,
            ),
            "facet_mapping_status": np.asarray(
                [row["mapping_status"] for row in rows], dtype=object
            ),
            "inside_facet": np.asarray(
                [row["inside_facet"] for row in rows], dtype=bool
            ),
            "outward_normal_global": np.asarray(
                [row["outward_normal_global"] for row in rows], dtype=float
            ),
        }

    def facet_metadata(self) -> dict[str, Any]:
        """Return every envelope facet in deterministic canonical order."""
        catalogs = [
            surface.facet_metadata(self.centreline_frame)
            for surface in sorted(
                self.faceted_surfaces, key=lambda item: int(item.surface_id)
            )
        ]
        if not catalogs:
            return {
                "facet_id": np.empty(0, dtype=object),
                "canonical_facet_id": np.empty(0, dtype=object),
                "facet_index": np.empty(0, dtype=int),
                "dagmc_volume_id": np.empty(0, dtype=int),
                "surface_id": np.empty(0, dtype=int),
                "surface_role": np.empty(0, dtype=object),
                "triangle_vertices_global_cm": np.empty(
                    (0, 3, 3), dtype=float
                ),
                "facet_centroid_global_cm": np.empty((0, 3), dtype=float),
                "outward_normal_global": np.empty((0, 3), dtype=float),
                "triangle_outward_normal_global": np.empty(
                    (0, 3), dtype=float
                ),
                "facet_area_cm2": np.empty(0, dtype=float),
                "triangle_area_cm2": np.empty(0, dtype=float),
                "centreline_linkage_available": self.centreline_frame
                is not None,
                "centreline_linkage_status": np.empty(0, dtype=object),
            }
        fields = catalogs[0]
        metadata = {
            name: np.concatenate([catalog[name] for catalog in catalogs])
            for name in fields
            if name != "centreline_linkage_available"
        }
        metadata["centreline_linkage_available"] = (
            self.centreline_frame is not None
        )
        return metadata


def _triangles(coordinates) -> np.ndarray:
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) % 3:
        raise ValueError("DAGMC triangle coordinates must have shape (3*N, 3)")
    return values.reshape((-1, 3, 3))


@dataclass(frozen=True)
class ClosedTriangleVolumeAudit:
    """Independent volume and orientation audit for one DAGMC volume."""

    volume_id: int
    volume_cm3: float
    signed_volume_cm3: float
    backend_volume_cm3: float | None
    backend_relative_error: float | None
    triangle_count: int
    edge_count: int
    vector_area_closure_relative: float
    coordinate_quantum_cm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume_id": self.volume_id,
            "volume_cm3": self.volume_cm3,
            "signed_volume_cm3": self.signed_volume_cm3,
            "backend_volume_cm3": self.backend_volume_cm3,
            "backend_relative_error": self.backend_relative_error,
            "triangle_count": self.triangle_count,
            "edge_count": self.edge_count,
            "vector_area_closure_relative": (
                self.vector_area_closure_relative
            ),
            "coordinate_quantum_cm": self.coordinate_quantum_cm,
        }


def audit_closed_triangle_volume(
    volume,
    *,
    coordinate_quantum_cm: float = DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM,
    vector_area_closure_tolerance: float = 1.0e-7,
    backend_relative_tolerance: float = 1.0e-9,
) -> ClosedTriangleVolumeAudit:
    """Compute volume from a consistently oriented, closed triangle shell.

    This does not trust a geometry backend's reported volume. Facets are
    oriented outward from the DAGMC surface senses, their topology is checked
    edge by edge on the canonical quantization grid, and their original
    coordinates are integrated about a local origin for numerical stability.
    A backend volume, when present, is only accepted after it agrees with this
    independent result.
    """
    quantum = float(coordinate_quantum_cm)
    closure_tolerance = float(vector_area_closure_tolerance)
    backend_tolerance = float(backend_relative_tolerance)
    if not np.isfinite(quantum) or quantum <= 0.0:
        raise ValueError("coordinate_quantum_cm must be positive and finite")
    if not np.isfinite(closure_tolerance) or closure_tolerance <= 0.0:
        raise ValueError(
            "vector_area_closure_tolerance must be positive and finite"
        )
    if not np.isfinite(backend_tolerance) or backend_tolerance <= 0.0:
        raise ValueError(
            "backend_relative_tolerance must be positive and finite"
        )

    volume_id = int(volume.id)
    oriented_triangles = []
    for surface in getattr(volume, "surfaces", ()):
        triangles = _triangles(surface.triangle_coords).copy()
        _quantize_coordinates(triangles, quantum)
        if _openmc_to_outward_normal_sign(surface, volume_id) == -1:
            triangles = triangles[:, [0, 2, 1], :]
        oriented_triangles.append(triangles)
    if not oriented_triangles:
        raise ValueError(f"DAGMC volume {volume_id} has no boundary facets")
    triangles = np.concatenate(oriented_triangles)
    area_vectors = 0.5 * np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = np.linalg.norm(area_vectors, axis=1)
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError(
            f"DAGMC volume {volume_id} contains degenerate facets"
        )

    quantized_vertices = _quantize_coordinates(triangles, quantum)
    edge_balance: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]], list[int]
    ] = {}
    for triangle in quantized_vertices:
        vertices = [
            tuple(int(value) for value in vertex) for vertex in triangle
        ]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            start = vertices[first]
            end = vertices[second]
            key = (start, end) if start < end else (end, start)
            direction = 1 if start < end else -1
            counts = edge_balance.setdefault(key, [0, 0])
            counts[0] += 1
            counts[1] += direction
    bad_edges = {
        edge: tuple(counts)
        for edge, counts in edge_balance.items()
        if counts[0] != 2 or counts[1] != 0
    }
    if bad_edges:
        sample = list(bad_edges.items())[:5]
        raise ValueError(
            f"DAGMC volume {volume_id} is not a consistently oriented "
            f"closed triangle shell; edge counts {sample}"
        )

    total_area = float(areas.sum())
    closure = float(np.linalg.norm(area_vectors.sum(axis=0)) / total_area)
    if not np.isfinite(closure) or closure > closure_tolerance:
        raise ValueError(
            f"DAGMC volume {volume_id} vector-area closure failed: {closure}"
        )

    origin = np.mean(triangles.reshape((-1, 3)), axis=0)
    local = triangles - origin
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            local[:, 0],
            np.cross(local[:, 1], local[:, 2]),
        ).sum()
        / 6.0
    )
    if not np.isfinite(signed_volume) or signed_volume <= 0.0:
        raise ValueError(
            f"DAGMC volume {volume_id} has non-positive outward signed "
            f"volume {signed_volume} cm3"
        )

    backend_value = getattr(volume, "volume", None)
    backend_volume = None
    backend_error = None
    if backend_value is not None:
        backend_volume = float(backend_value)
        if not np.isfinite(backend_volume) or backend_volume <= 0.0:
            raise ValueError(
                f"DAGMC volume {volume_id} backend volume must be positive "
                "and finite"
            )
        backend_error = abs(backend_volume - signed_volume) / signed_volume
        if backend_error > backend_tolerance:
            raise ValueError(
                f"DAGMC volume {volume_id} backend volume disagrees with "
                f"the closed triangle shell: relative error {backend_error}"
            )
    return ClosedTriangleVolumeAudit(
        volume_id=volume_id,
        volume_cm3=signed_volume,
        signed_volume_cm3=signed_volume,
        backend_volume_cm3=backend_volume,
        backend_relative_error=backend_error,
        triangle_count=len(triangles),
        edge_count=len(edge_balance),
        vector_area_closure_relative=closure,
        coordinate_quantum_cm=quantum,
    )


def _role(centroid, volume_centroid, plasma, toroidal, poloidal):
    direction = _unit(centroid - volume_centroid, "surface displacement")
    scores = {
        "plasma_facing": float(np.dot(direction, plasma)),
        "rear": float(np.dot(direction, -plasma)),
        "upper_edge": float(np.dot(direction, poloidal)),
        "lower_edge": float(np.dot(direction, -poloidal)),
        "toroidal_leading_edge": float(np.dot(direction, toroidal)),
        "toroidal_trailing_edge": float(np.dot(direction, -toroidal)),
    }
    name, score = max(scores.items(), key=lambda item: item[1])
    return name if score >= 0.55 else "other"


def _openmc_to_outward_normal_sign(surface, volume_id: int) -> int:
    """Return the sign mapping an OpenMC/DAGMC facet normal to outward."""
    reverse_id = getattr(getattr(surface, "reverse_volume", None), "id", None)
    forward_id = getattr(getattr(surface, "forward_volume", None), "id", None)
    if forward_id == int(volume_id):
        return 1
    if reverse_id == int(volume_id):
        return -1
    raise ValueError(
        f"surface {surface.id} has no sense for DAGMC volume {volume_id}"
    )


def _canonical_facet_id(
    geometry_fingerprint: str,
    volume_id: int,
    surface_id: int,
    triangle_cm: np.ndarray,
    outward_sign: int,
    coordinate_quantum_cm: float,
) -> str:
    """Hash an orientation-aware triangle independently of its start vertex."""
    triangle = np.asarray(triangle_cm, dtype=float)
    if outward_sign == -1:
        triangle = triangle[[0, 2, 1]]
    quantized = np.rint(triangle / coordinate_quantum_cm).astype(np.int64)
    cycles = [np.roll(quantized, -index, axis=0) for index in range(3)]
    oriented = min(
        (
            tuple(tuple(int(value) for value in vertex) for vertex in cycle)
            for cycle in cycles
        )
    )
    payload = json.dumps(
        {
            "canonical_geometry_fingerprint": geometry_fingerprint,
            "dagmc_volume_id": int(volume_id),
            "surface_id": int(surface_id),
            "openmc_to_outward_normal_sign": int(outward_sign),
            "oriented_quantized_vertices": oriented,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "facet-" + hashlib.sha256(payload).hexdigest()


def extract_closed_envelope(
    dagmc_path: str | Path,
    volume_id: int,
    *,
    envelope_id: str,
    magnet_id: str,
    plasma_direction_global: Sequence[float],
    toroidal_direction_global: Sequence[float],
    poloidal_direction_global: Sequence[float],
    spatial_bins: tuple[int, int] = (4, 4),
    edge_round_decimals: int = 8,
    coordinate_quantum_cm: float = DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM,
    faceting_tolerances: Mapping[str, Any] | None = None,
    centreline_frame: Any | None = None,
) -> DagmcEnvelope:
    """Extract every surface of one DAGMC volume and prove faceted closure."""
    import pydagmc

    fingerprint = canonical_dagmc_fingerprint(
        dagmc_path,
        coordinate_quantum_cm=coordinate_quantum_cm,
        faceting_tolerances=faceting_tolerances,
    )
    model = pydagmc.Model(str(dagmc_path))
    if int(volume_id) not in model.volumes_by_id:
        raise ValueError(f"DAGMC volume {volume_id} does not exist")
    volume = model.volumes_by_id[int(volume_id)]
    plasma = _unit(plasma_direction_global, "plasma direction")
    toroidal = _unit(toroidal_direction_global, "toroidal direction")
    poloidal = _unit(poloidal_direction_global, "poloidal direction")
    all_triangles = np.concatenate(
        [_triangles(surface.triangle_coords) for surface in volume.surfaces]
    )
    volume_centroid = np.mean(all_triangles.reshape((-1, 3)), axis=0)
    edge_counts = {}
    faceted = []
    schema_surfaces = []
    vector_area = np.zeros(3)
    total_area = 0.0
    for surface in volume.surfaces:
        triangles = _triangles(surface.triangle_coords)
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        lengths = np.linalg.norm(cross, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError(
                f"surface {surface.id} contains degenerate triangles"
            )
        normals = cross / lengths[:, None]
        areas = 0.5 * lengths
        centroid = np.average(triangles.mean(axis=1), axis=0, weights=areas)
        # PyDAGMC's native facet winding is outward for the forward-sense
        # volume and inward for the reverse-sense volume. Use that topological
        # sense directly: centroid tests are not valid for concave or toroidal
        # magnet volumes.
        openmc_normal_sign = _openmc_to_outward_normal_sign(surface, volume.id)
        normals *= openmc_normal_sign
        facet_ids = tuple(
            _canonical_facet_id(
                fingerprint["canonical_fingerprint"],
                int(volume.id),
                int(surface.id),
                triangle,
                openmc_normal_sign,
                coordinate_quantum_cm,
            )
            for triangle in triangles
        )
        role = _role(centroid, volume_centroid, plasma, toroidal, poloidal)
        faceted.append(
            FacetedSurface(
                surface.id,
                role,
                triangles,
                normals,
                areas,
                centroid,
                facet_ids,
                int(volume.id),
            )
        )
        area_vector = np.sum(normals * areas[:, None], axis=0)
        vector_area += area_vector
        total_area += float(areas.sum())
        average_normal = _unit(
            area_vector, f"surface {surface.id} average normal"
        )
        tangent = toroidal - np.dot(toroidal, average_normal) * average_normal
        if np.linalg.norm(tangent) <= 1.0e-10:
            tangent = (
                poloidal - np.dot(poloidal, average_normal) * average_normal
            )
        tangent = _unit(tangent, "surface tangent")
        width = np.cross(average_normal, tangent)
        local = (triangles.reshape((-1, 3)) - centroid) @ np.vstack(
            (tangent, width, average_normal)
        ).T
        u_edges = np.linspace(
            local[:, 0].min(), local[:, 0].max(), spatial_bins[0] + 1
        )
        v_edges = np.linspace(
            local[:, 1].min(), local[:, 1].max(), spatial_bins[1] + 1
        )
        schema_surfaces.append(
            EnvelopeSurface(
                surface_id=int(surface.id),
                role=role,
                area_cm2=float(areas.sum()),
                centroid_global_cm=tuple(centroid),
                outward_normal_global=tuple(average_normal),
                toroidal_direction_global=tuple(tangent),
                poloidal_direction_global=tuple(width),
                u_edges_cm=tuple(u_edges),
                v_edges_cm=tuple(v_edges),
                openmc_normal_sign=openmc_normal_sign,
                vector_area_global_cm2=tuple(area_vector),
            )
        )
        for triangle in triangles:
            rounded = np.round(triangle, edge_round_decimals)
            for first, second in ((0, 1), (1, 2), (2, 0)):
                points = sorted(
                    (tuple(rounded[first]), tuple(rounded[second]))
                )
                key = tuple(points)
                edge_counts[key] = edge_counts.get(key, 0) + 1
    bad = {edge: count for edge, count in edge_counts.items() if count != 2}
    if bad:
        sample = list(bad.items())[:5]
        raise ValueError(
            f"DAGMC envelope is not watertight; edge counts {sample}"
        )
    closure = float(np.linalg.norm(vector_area) / total_area)
    if closure > 1.0e-7:
        raise ValueError(
            f"DAGMC envelope vector-area closure failed: {closure}"
        )
    volume_audit = audit_closed_triangle_volume(
        volume, coordinate_quantum_cm=coordinate_quantum_cm
    )
    envelope = MagnetBoundaryEnvelope(
        envelope_id=envelope_id,
        magnet_component=magnet_id,
        dagmc_volume_id=int(volume_id),
        surfaces=tuple(schema_surfaces),
        dagmc_geometry_sha256=_hash(dagmc_path),
        metadata={
            "material": volume.material,
            "volume_cm3": volume_audit.volume_cm3,
            "faceted_volume_audit": volume_audit.to_dict(),
            "watertight_proof": "every rounded faceted edge occurs exactly twice",
            "faceted_vector_area_closure_relative": closure,
            "canonical_geometry_fingerprint": fingerprint[
                "canonical_fingerprint"
            ],
            "canonical_coordinate_quantum_cm": coordinate_quantum_cm,
            "canonical_faceting_tolerances": dict(faceting_tolerances or {}),
            "canonical_facet_count": int(
                sum(len(item.canonical_facet_ids) for item in faceted)
            ),
        },
    )
    return DagmcEnvelope(
        envelope,
        tuple(faceted),
        len(edge_counts),
        0,
        closure,
        centreline_frame,
    )


@dataclass(frozen=True)
class MagnetVolumeRecord:
    """Stable identity for one magnet-related DAGMC volume."""

    volume_id: int
    material: str
    component_role: str
    surface_ids: tuple[int, ...]
    component_name: str = ""
    component_id: str = ""
    magnet_id: str = ""
    coil_id: str = ""
    centroid_global_cm: tuple[float, float, float] | None = None
    bounding_box_cm: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None
    volume_cm3: float | None = None
    source_coil_provenance: Mapping[str, Any] | None = None
    faceted_volume_audit: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume_id": self.volume_id,
            "material": self.material,
            "component_role": self.component_role,
            "surface_ids": list(self.surface_ids),
            "component_name": self.component_name,
            "component_id": self.component_id,
            "magnet_id": self.magnet_id,
            "coil_id": self.coil_id,
            "centroid_global_cm": (
                list(self.centroid_global_cm)
                if self.centroid_global_cm is not None
                else None
            ),
            "bounding_box_cm": (
                [list(value) for value in self.bounding_box_cm]
                if self.bounding_box_cm is not None
                else None
            ),
            "volume_cm3": self.volume_cm3,
            "source_coil_provenance": (
                dict(self.source_coil_provenance)
                if self.source_coil_provenance is not None
                else None
            ),
            "faceted_volume_audit": (
                dict(self.faceted_volume_audit)
                if self.faceted_volume_audit is not None
                else None
            ),
        }


@dataclass(frozen=True)
class MagnetPairRecord:
    """One deterministic casing/winding-pack association."""

    magnet_id: str
    coil_id: str
    winding_pack: MagnetVolumeRecord
    casing: MagnetVolumeRecord | None
    pairing_basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "magnet_id": self.magnet_id,
            "coil_id": self.coil_id,
            "winding_pack_volume_id": self.winding_pack.volume_id,
            "casing_volume_id": (
                self.casing.volume_id if self.casing is not None else None
            ),
            "pairing_basis": self.pairing_basis,
            "winding_pack": self.winding_pack.to_dict(),
            "casing": (
                self.casing.to_dict() if self.casing is not None else None
            ),
        }


@dataclass(frozen=True)
class MagnetVolumeInventory:
    """Deterministic inventory used before any envelope is selected."""

    dagmc_path: str
    dagmc_geometry_sha256: str
    volumes: tuple[MagnetVolumeRecord, ...]
    canonical_geometry_fingerprint: str = ""
    pairs: tuple[MagnetPairRecord, ...] = ()
    coordinate_quantum_cm: float = DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM
    faceting_tolerances: Mapping[str, Any] | None = None

    @property
    def winding_packs(self) -> tuple[MagnetVolumeRecord, ...]:
        return tuple(
            item
            for item in self.volumes
            if item.component_role == "winding_pack"
        )

    @property
    def casings(self) -> tuple[MagnetVolumeRecord, ...]:
        return tuple(
            item
            for item in self.volumes
            if item.component_role == "magnet_casing"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dagmc_path": self.dagmc_path,
            "dagmc_geometry_sha256": self.dagmc_geometry_sha256,
            "canonical_geometry_fingerprint": self.canonical_geometry_fingerprint,
            "canonical_geometry_policy": canonical_geometry_policy(
                self.coordinate_quantum_cm, self.faceting_tolerances
            ),
            "volumes": [item.to_dict() for item in self.volumes],
            "winding_pack_volume_ids": [
                item.volume_id for item in self.winding_packs
            ],
            "magnet_casing_volume_ids": [
                item.volume_id for item in self.casings
            ],
            "magnet_pairs": [item.to_dict() for item in self.pairs],
        }


def _material_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _safe_identifier(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
    return result.strip("-")


def _record_spatial_key(record: MagnetVolumeRecord):
    centroid = record.centroid_global_cm or (0.0, 0.0, 0.0)
    return tuple(round(float(value), 8) for value in centroid) + (
        round(float(record.volume_cm3 or 0.0), 8),
        record.volume_id,
    )


def _pair_magnet_records(
    winding_packs: Sequence[MagnetVolumeRecord],
    casings: Sequence[MagnetVolumeRecord],
) -> tuple[MagnetPairRecord, ...]:
    winding = tuple(sorted(winding_packs, key=_record_spatial_key))
    casing = tuple(sorted(casings, key=_record_spatial_key))
    if casing and len(casing) != len(winding):
        raise ValueError(
            "winding-pack/casing counts differ; an explicit association manifest is required"
        )
    remaining = set(range(len(casing)))
    rows = []
    for index, pack in enumerate(winding):
        match = None
        basis = "winding_pack_only"
        if casing:
            pack_center = np.asarray(pack.centroid_global_cm, dtype=float)
            containing = []
            for candidate in sorted(remaining):
                bounds = casing[candidate].bounding_box_cm
                if bounds is None:
                    continue
                lower, upper = (
                    np.asarray(value, dtype=float) for value in bounds
                )
                if np.all(pack_center >= lower - 1.0e-6) and np.all(
                    pack_center <= upper + 1.0e-6
                ):
                    containing.append(candidate)
            candidates = containing or sorted(remaining)
            distances = [
                float(
                    np.linalg.norm(
                        pack_center
                        - np.asarray(
                            casing[candidate].centroid_global_cm, dtype=float
                        )
                    )
                )
                for candidate in candidates
            ]
            order = np.argsort(distances)
            if len(order) > 1 and np.isclose(
                distances[int(order[0])],
                distances[int(order[1])],
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise ValueError(
                    f"ambiguous casing pairing for winding-pack volume {pack.volume_id}"
                )
            match_index = candidates[int(order[0])]
            match = casing[match_index]
            remaining.remove(match_index)
            basis = (
                "unique_bounding_box_containment_then_centroid"
                if containing
                else "unique_nearest_centroid"
            )
        identity = pack.coil_id or _safe_identifier(pack.component_name)
        if not identity:
            identity = f"coil-{index:04d}"
        magnet_id = pack.magnet_id or f"magnet-{identity}"
        rows.append(MagnetPairRecord(magnet_id, identity, pack, match, basis))
    if remaining:
        raise ValueError("unpaired magnet casing volumes remain")
    return tuple(rows)


def discover_magnet_volumes(
    dagmc_path: str | Path,
    *,
    winding_pack_materials: Sequence[str] = ("winding_pack",),
    casing_materials: Sequence[str] = ("magnet_casing",),
    associations: Mapping[int, Mapping[str, Any]] | None = None,
    coordinate_quantum_cm: float = DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM,
    faceting_tolerances: Mapping[str, Any] | None = None,
) -> MagnetVolumeInventory:
    """Inventory magnet volumes by stable DAGMC IDs and exact material tags."""
    import pydagmc

    path = Path(dagmc_path).resolve()
    model = pydagmc.Model(str(path))
    winding_keys = {_material_key(item) for item in winding_pack_materials}
    casing_keys = {_material_key(item) for item in casing_materials}
    overlap = winding_keys & casing_keys
    if overlap:
        raise ValueError(f"magnet material roles overlap: {sorted(overlap)}")
    records = []
    association_map = {
        int(key): dict(value) for key, value in (associations or {}).items()
    }
    for volume_id, volume in sorted(model.volumes_by_id.items()):
        material = str(getattr(volume, "material", "") or "")
        key = _material_key(material)
        if key in winding_keys:
            role = "winding_pack"
        elif key in casing_keys:
            role = "magnet_casing"
        else:
            continue
        surface_ids = tuple(
            sorted(
                int(surface.id) for surface in getattr(volume, "surfaces", ())
            )
        )
        if not surface_ids:
            raise ValueError(
                f"magnet DAGMC volume {volume_id} has no boundary surfaces"
            )
        triangles = np.concatenate(
            [
                _triangles(surface.triangle_coords)
                for surface in volume.surfaces
            ]
        ).reshape((-1, 3))
        association = association_map.get(int(volume_id), {})
        component_name = str(
            association.get(
                "component_name", getattr(volume, "name", "") or ""
            )
        )
        component_id = str(
            association.get(
                "component_id",
                component_name or f"dagmc-volume-{int(volume_id)}",
            )
        )
        volume_audit = audit_closed_triangle_volume(
            volume, coordinate_quantum_cm=coordinate_quantum_cm
        )
        records.append(
            MagnetVolumeRecord(
                volume_id=int(volume_id),
                material=material,
                component_role=role,
                surface_ids=surface_ids,
                component_name=component_name,
                component_id=component_id,
                magnet_id=str(association.get("magnet_id", "")),
                coil_id=str(association.get("coil_id", "")),
                centroid_global_cm=tuple(np.mean(triangles, axis=0)),
                bounding_box_cm=(
                    tuple(np.min(triangles, axis=0)),
                    tuple(np.max(triangles, axis=0)),
                ),
                volume_cm3=volume_audit.volume_cm3,
                source_coil_provenance=association.get(
                    "source_coil_provenance"
                ),
                faceted_volume_audit=volume_audit.to_dict(),
            )
        )
    policy = canonical_geometry_policy(
        coordinate_quantum_cm, faceting_tolerances
    )
    fingerprint = canonical_dagmc_fingerprint(
        path,
        coordinate_quantum_cm=policy["coordinate_quantum_cm"],
        faceting_tolerances=policy["faceting_tolerances"],
    )
    pairs = _pair_magnet_records(
        [item for item in records if item.component_role == "winding_pack"],
        [item for item in records if item.component_role == "magnet_casing"],
    )
    inventory = MagnetVolumeInventory(
        str(path),
        _hash(path),
        tuple(records),
        fingerprint["canonical_fingerprint"],
        pairs,
        policy["coordinate_quantum_cm"],
        policy["faceting_tolerances"],
    )
    if not inventory.winding_packs:
        raise ValueError(
            "DAGMC geometry has no winding-pack volume with an accepted "
            "material tag"
        )
    return inventory


def select_magnet_pairs(
    inventory: MagnetVolumeInventory,
    selection: str | int | Sequence[str | int] = "all",
) -> tuple[MagnetPairRecord, ...]:
    """Select by magnet ID, coil ID, or winding-pack DAGMC volume ID."""
    if selection == "all":
        return inventory.pairs
    requested = (
        (selection,) if isinstance(selection, (str, int)) else tuple(selection)
    )
    if not requested:
        raise ValueError("magnet selection cannot be empty")
    matches = []
    for value in requested:
        candidates = [
            pair
            for pair in inventory.pairs
            if str(value)
            in {
                pair.magnet_id,
                pair.coil_id,
                str(pair.winding_pack.volume_id),
            }
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"magnet selection {value!r} resolved to {len(candidates)} records"
            )
        matches.append(candidates[0])
    if len({item.magnet_id for item in matches}) != len(matches):
        raise ValueError("magnet selection contains duplicates")
    return tuple(matches)


def select_winding_pack_volumes(
    inventory: MagnetVolumeInventory,
    volume_ids: Sequence[int] | None = None,
) -> tuple[MagnetVolumeRecord, ...]:
    """Select winding packs without silently choosing among multiple magnets."""
    candidates = {item.volume_id: item for item in inventory.winding_packs}
    if volume_ids is None:
        if len(candidates) != 1:
            raise ValueError(
                "multiple winding-pack volumes are present; explicit volume "
                "IDs are required"
            )
        return tuple(candidates.values())
    requested = tuple(int(item) for item in volume_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("winding-pack volume IDs must be nonempty and unique")
    missing = sorted(set(requested) - set(candidates))
    if missing:
        raise ValueError(
            f"requested IDs are not winding-pack volumes: {missing}; "
            f"available IDs: {sorted(candidates)}"
        )
    return tuple(candidates[item] for item in requested)


def extract_closed_envelopes(
    dagmc_path: str | Path,
    volume_ids: Sequence[int],
    *,
    frames_by_volume: Mapping[int, Mapping[str, Sequence[float]]],
    spatial_bins: tuple[int, int] = (4, 4),
    coordinate_quantum_cm: float = DEFAULT_CANONICAL_COORDINATE_QUANTUM_CM,
    faceting_tolerances: Mapping[str, Any] | None = None,
) -> tuple[DagmcEnvelope, ...]:
    """Extract multiple envelopes using an explicit frame for every magnet."""
    selected = tuple(int(item) for item in volume_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("envelope volume IDs must be nonempty and unique")
    frame_ids = {int(key) for key in frames_by_volume}
    missing_frames = sorted(set(selected) - frame_ids)
    if missing_frames:
        raise ValueError(
            "local orientation frames are missing for magnet volumes "
            f"{missing_frames}"
        )
    results = []
    for volume_id in selected:
        frame = frames_by_volume[volume_id]
        required = {
            "plasma_direction_global",
            "toroidal_direction_global",
            "poloidal_direction_global",
        }
        missing = required - set(frame)
        if missing:
            raise ValueError(
                f"magnet volume {volume_id} frame is missing {sorted(missing)}"
            )
        results.append(
            extract_closed_envelope(
                dagmc_path,
                volume_id,
                envelope_id=f"winding-pack-{volume_id}",
                magnet_id=f"magnet-{volume_id}",
                plasma_direction_global=frame["plasma_direction_global"],
                toroidal_direction_global=frame["toroidal_direction_global"],
                poloidal_direction_global=frame["poloidal_direction_global"],
                spatial_bins=spatial_bins,
                coordinate_quantum_cm=coordinate_quantum_cm,
                faceting_tolerances=faceting_tolerances,
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class DagmcWatertightnessReport:
    dagmc_geometry_sha256: str
    volume_count: int
    surface_count: int
    leaky_volume_ids: tuple[int, ...]
    unmatched_edge_count: int
    edge_round_decimals: int

    @property
    def passes(self) -> bool:
        return not self.leaky_volume_ids and self.unmatched_edge_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dagmc_geometry_sha256": self.dagmc_geometry_sha256,
            "volume_count": self.volume_count,
            "surface_count": self.surface_count,
            "leaky_volume_ids": list(self.leaky_volume_ids),
            "unmatched_edge_count": self.unmatched_edge_count,
            "edge_round_decimals": self.edge_round_decimals,
            "passes": self.passes,
        }


def validate_dagmc_watertightness(
    dagmc_path: str | Path,
    *,
    edge_round_decimals: int = 8,
) -> DagmcWatertightnessReport:
    """Prove that every physical DAGMC volume has a closed faceted shell."""
    import pydagmc

    path = Path(dagmc_path).resolve()
    model = pydagmc.Model(str(path))
    leaky = []
    unmatched_total = 0
    surface_ids = set()
    for volume_id, volume in sorted(model.volumes_by_id.items()):
        edge_counts: dict[tuple[Any, ...], int] = {}
        for surface in volume.surfaces:
            surface_ids.add(int(surface.id))
            triangles = _triangles(surface.triangle_coords)
            for triangle in np.round(triangles, edge_round_decimals):
                for first, second in ((0, 1), (1, 2), (2, 0)):
                    edge = tuple(
                        sorted(
                            (tuple(triangle[first]), tuple(triangle[second]))
                        )
                    )
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
        unmatched = sum(count != 2 for count in edge_counts.values())
        if unmatched:
            leaky.append(int(volume_id))
            unmatched_total += int(unmatched)
    return DagmcWatertightnessReport(
        dagmc_geometry_sha256=_hash(path),
        volume_count=len(model.volumes_by_id),
        surface_count=len(surface_ids),
        leaky_volume_ids=tuple(leaky),
        unmatched_edge_count=unmatched_total,
        edge_round_decimals=int(edge_round_decimals),
    )


def require_watertight_dagmc(
    dagmc_path: str | Path,
    *,
    edge_round_decimals: int = 8,
) -> DagmcWatertightnessReport:
    report = validate_dagmc_watertightness(
        dagmc_path, edge_round_decimals=edge_round_decimals
    )
    if not report.passes:
        raise RuntimeError(
            "combined DAGMC geometry is not watertight: "
            f"leaky volumes={list(report.leaky_volume_ids)}, "
            f"unmatched edges={report.unmatched_edge_count}"
        )
    return report
