"""Reusable facet-local patch atlas for arbitrary triangulated CAD surfaces.

The atlas caches geometry-only quantities used repeatedly by boundary-source
localization and response remapping.  It is not a transport sweep and does not
promote fixture or unqualified geometry to a physical candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .surface_source_localization import _barycentric, _unit


ATLAS_SCHEMA = "parastell.facet_patch_atlas/v1.0.0"
MAPPING_SCHEMA = "parastell.conservative_facet_patch_mapping/v1.0.0"


def _array_hash(label: str, value: Any, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(label.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


@dataclass(frozen=True)
class FacetPatchAtlas:
    """In-memory cache; arrays use catalog facet order."""

    facet_ids: np.ndarray
    patch_ids: np.ndarray
    surface_ids: np.ndarray
    triangles_cm: np.ndarray
    normals_global: np.ndarray
    origins_cm: np.ndarray
    frames_global_to_local: np.ndarray
    centroids_cm: np.ndarray
    areas_cm2: np.ndarray
    bounds_min_cm: np.ndarray
    bounds_max_cm: np.ndarray
    neighbor_offsets: np.ndarray
    neighbor_indices: np.ndarray
    surface_to_indices: Mapping[int, np.ndarray]
    neighbor_basis: str
    catalog_sha256: str
    atlas_sha256: str

    @property
    def facet_count(self) -> int:
        return len(self.facet_ids)

    def neighbors(self, facet_index: int) -> np.ndarray:
        index = int(facet_index)
        if index < 0 or index >= self.facet_count:
            raise IndexError("facet index is out of range")
        return self.neighbor_indices[
            self.neighbor_offsets[index] : self.neighbor_offsets[index + 1]
        ].copy()

    def compact_metadata(self) -> dict[str, Any]:
        return {
            "schema": ATLAS_SCHEMA,
            "status": "SOFTWARE_CACHE_READY",
            "facet_count": self.facet_count,
            "patch_count": len(set(self.patch_ids.tolist())),
            "surface_count": len(self.surface_to_indices),
            "neighbor_basis": self.neighbor_basis,
            "directed_neighbor_count": int(len(self.neighbor_indices)),
            "maximum_neighbors_per_facet": int(
                np.max(np.diff(self.neighbor_offsets))
            ),
            "total_area_cm2": float(np.sum(self.areas_cm2)),
            "catalog_sha256": self.catalog_sha256,
            "atlas_sha256": self.atlas_sha256,
            "frame_convention": (
                "rows are tangent, bitangent, outward-normal; multiply a "
                "global vector to obtain local coordinates"
            ),
            "physical_qualification_claimed": False,
            "arbitrary_polyhedral_transport_sweep_claimed": False,
        }


def _validated_catalog(
    facet_catalog: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    required = {
        "facet_id",
        "surface_id",
        "vertices_global_cm",
        "outward_normal_global",
        "normal_source",
    }
    missing = required - set(facet_catalog)
    if missing:
        raise ValueError(f"facet catalog omits {sorted(missing)}")
    if facet_catalog["normal_source"] != "dagmc_forward_reverse_topology":
        raise ValueError(
            "outward normals must come from DAGMC forward/reverse topology"
        )
    facet_ids = np.asarray(facet_catalog["facet_id"]).astype(str)
    surface_ids = np.asarray(
        facet_catalog["surface_id"], dtype=np.int64
    ).reshape(-1)
    triangles = np.asarray(facet_catalog["vertices_global_cm"], dtype=float)
    normals = _unit(
        np.asarray(facet_catalog["outward_normal_global"], dtype=float),
        "outward_normal_global",
    )
    count = len(facet_ids)
    if not count or len(set(facet_ids.tolist())) != count:
        raise ValueError("facet IDs must be nonempty and unique")
    if surface_ids.shape != (count,) or triangles.shape != (count, 3, 3):
        raise ValueError("facet catalog arrays are misaligned")
    if normals.shape != (count, 3) or not np.all(np.isfinite(triangles)):
        raise ValueError("facet coordinates or normals are malformed")
    geometric = _unit(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        "triangle geometric normal",
    )
    if np.any(np.sum(geometric * normals, axis=1) < 1.0 - 1.0e-10):
        raise ValueError(
            "catalog normals disagree with oriented facet geometry"
        )
    vertex_ids = None
    if "vertex_id" in facet_catalog:
        raw = np.asarray(facet_catalog["vertex_id"])
        if raw.shape != (count, 3):
            raise ValueError("vertex IDs must have shape (facets,3)")
        if not np.issubdtype(raw.dtype, np.integer) and (
            not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw))
        ):
            raise ValueError("vertex IDs must be integral")
        vertex_ids = np.asarray(raw, dtype=np.int64)
        if np.any(vertex_ids < 0):
            raise ValueError("vertex IDs cannot be negative")
    return facet_ids, surface_ids, triangles, normals, vertex_ids


def build_facet_patch_atlas(
    facet_catalog: Mapping[str, object],
) -> FacetPatchAtlas:
    """Precompute facet frames, bounds, areas, and edge neighbors once."""

    facet_ids, surface_ids, triangles, normals, vertex_ids = (
        _validated_catalog(facet_catalog)
    )
    count = len(facet_ids)
    tangent = np.empty((count, 3), dtype=float)
    for facet_index, triangle in enumerate(triangles):
        candidates = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            start = triangle[first]
            end = triangle[second]
            if tuple(end) < tuple(start):
                start, end = end, start
            vector = end - start
            candidates.append(
                (
                    -float(np.dot(vector, vector)),
                    tuple(start),
                    tuple(end),
                    vector,
                )
            )
        # Longest edge first, with endpoint coordinates as a deterministic
        # tie-break. This is invariant to cyclic facet ordering.
        candidates.sort(key=lambda row: row[:3])
        tangent[facet_index] = candidates[0][3]
    tangent -= np.sum(tangent * normals, axis=1)[:, None] * normals
    tangent = _unit(tangent, "facet tangent")
    bitangent = _unit(np.cross(normals, tangent), "facet bitangent")
    frames = np.stack((tangent, bitangent, normals), axis=1)
    handedness = np.sum(
        np.cross(frames[:, 0], frames[:, 1]) * frames[:, 2], axis=1
    )
    if np.any(handedness < 1.0 - 1.0e-12):
        raise RuntimeError("facet atlas contains a non-right-handed frame")
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(area_vectors, axis=1)

    if vertex_ids is not None:
        neighbor_basis = "shared_vertex_id"
        coordinates_by_id: dict[int, np.ndarray] = {}
        for facet_index in range(count):
            for local_index in range(3):
                vertex_id = int(vertex_ids[facet_index, local_index])
                coordinate = triangles[facet_index, local_index]
                previous = coordinates_by_id.setdefault(vertex_id, coordinate)
                if not np.array_equal(previous, coordinate):
                    raise ValueError(
                        "a shared vertex ID has inconsistent coordinates"
                    )

        def vertex_key(facet: int, local: int) -> tuple[str, int]:
            return ("id", int(vertex_ids[facet, local]))

    else:
        neighbor_basis = "exact_float64_coordinate"

        def vertex_key(facet: int, local: int) -> tuple[str, bytes]:
            coordinate = np.asarray(triangles[facet, local], dtype="<f8")
            return ("xyz", coordinate.tobytes())

    edge_owners: dict[tuple[object, object], list[int]] = {}
    for facet in range(count):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            endpoints = sorted(
                (vertex_key(facet, first), vertex_key(facet, second)),
                key=repr,
            )
            edge = (endpoints[0], endpoints[1])
            edge_owners.setdefault(edge, []).append(facet)
    nonmanifold = [
        edge for edge, owners in edge_owners.items() if len(owners) > 2
    ]
    if nonmanifold:
        raise ValueError(
            f"facet catalog contains {len(nonmanifold)} nonmanifold edges"
        )
    neighbor_sets = [set() for _ in range(count)]
    for owners in edge_owners.values():
        if len(owners) == 2:
            first, second = owners
            neighbor_sets[first].add(second)
            neighbor_sets[second].add(first)
    neighbor_offsets = [0]
    neighbor_indices = []
    for values in neighbor_sets:
        neighbor_indices.extend(sorted(values))
        neighbor_offsets.append(len(neighbor_indices))
    surface_to_indices = {
        int(surface): np.flatnonzero(surface_ids == surface)
        for surface in sorted(set(surface_ids.tolist()))
    }
    patch_ids = np.asarray([f"patch:{value}" for value in facet_ids])
    canonical_triangles = np.empty_like(triangles)
    for index, triangle in enumerate(triangles):
        order = sorted(range(3), key=lambda local: tuple(triangle[local]))
        canonical_triangles[index] = triangle[order]
    catalog_identity = {
        "facet_ids_sha256": _canonical_hash({"facet_ids": facet_ids.tolist()}),
        "surface_ids_sha256": _array_hash("surface_ids", surface_ids, "<i8"),
        "triangles_sha256": _array_hash(
            "triangles_cm", canonical_triangles, "<f8"
        ),
        "normals_sha256": _array_hash("normals_global", normals, "<f8"),
        "normal_source": facet_catalog["normal_source"],
    }
    if vertex_ids is not None:
        catalog_identity["vertex_ids_sha256"] = _array_hash(
            "vertex_ids", np.sort(vertex_ids, axis=1), "<i8"
        )
    catalog_sha256 = _canonical_hash(catalog_identity)
    atlas_identity = {
        "schema": ATLAS_SCHEMA,
        "catalog_sha256": catalog_sha256,
        "frames_sha256": _array_hash("frames", frames, "<f8"),
        "areas_sha256": _array_hash("areas", areas, "<f8"),
        "neighbor_offsets_sha256": _array_hash(
            "neighbor_offsets", neighbor_offsets, "<i8"
        ),
        "neighbor_indices_sha256": _array_hash(
            "neighbor_indices", neighbor_indices, "<i8"
        ),
        "neighbor_basis": neighbor_basis,
    }
    return FacetPatchAtlas(
        facet_ids=facet_ids,
        patch_ids=patch_ids,
        surface_ids=surface_ids,
        triangles_cm=triangles,
        normals_global=normals,
        origins_cm=triangles[:, 0].copy(),
        frames_global_to_local=frames,
        centroids_cm=np.mean(triangles, axis=1),
        areas_cm2=areas,
        bounds_min_cm=np.min(triangles, axis=1),
        bounds_max_cm=np.max(triangles, axis=1),
        neighbor_offsets=np.asarray(neighbor_offsets, dtype=np.int64),
        neighbor_indices=np.asarray(neighbor_indices, dtype=np.int64),
        surface_to_indices=surface_to_indices,
        neighbor_basis=neighbor_basis,
        catalog_sha256=catalog_sha256,
        atlas_sha256=_canonical_hash(atlas_identity),
    )


def localize_surface_crossings_cached(
    phase_space: Mapping[str, np.ndarray],
    atlas: FacetPatchAtlas,
    *,
    barycentric_tolerance: float = 1.0e-10,
    residual_tolerance_cm: float = 1.0e-8,
    grazing_tolerance: float = 1.0e-12,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Localize crossings using cached frames and AABB candidate pruning."""

    required = {"position_global_cm", "direction_global", "surface_id"}
    missing = required - set(phase_space)
    if missing:
        raise ValueError(f"phase space omits {sorted(missing)}")
    positions = np.asarray(phase_space["position_global_cm"], dtype=float)
    directions = _unit(
        np.asarray(phase_space["direction_global"], dtype=float),
        "direction_global",
    )
    record_surfaces = np.asarray(
        phase_space["surface_id"], dtype=np.int64
    ).reshape(-1)
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or directions.shape != positions.shape
    ):
        raise ValueError(
            "positions and directions must have shape (records,3)"
        )
    if len(record_surfaces) != len(positions):
        raise ValueError("surface IDs do not align with records")
    for name, value in (
        ("barycentric_tolerance", barycentric_tolerance),
        ("residual_tolerance_cm", residual_tolerance_cm),
        ("grazing_tolerance", grazing_tolerance),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    selected = np.empty(len(positions), dtype=np.int64)
    barycentrics = np.empty((len(positions), 3))
    residuals = np.empty(len(positions))
    reconstructions = np.empty_like(positions)
    candidates_considered = 0
    for record_index, (point, surface_id) in enumerate(
        zip(positions, record_surfaces, strict=True)
    ):
        surface_candidates = atlas.surface_to_indices.get(int(surface_id))
        if surface_candidates is None:
            raise ValueError(f"surface {surface_id} has no atlas facets")
        lower = atlas.bounds_min_cm[surface_candidates] - residual_tolerance_cm
        upper = atlas.bounds_max_cm[surface_candidates] + residual_tolerance_cm
        candidates = surface_candidates[
            np.all((point >= lower) & (point <= upper), axis=1)
        ]
        candidates_considered += len(candidates)
        matches = []
        for facet_index in candidates:
            normal = atlas.normals_global[facet_index]
            signed_distance = float(
                np.dot(point - atlas.origins_cm[facet_index], normal)
            )
            projected = point - signed_distance * normal
            barycentric = _barycentric(
                projected, atlas.triangles_cm[facet_index]
            )
            if (
                np.min(barycentric) >= -barycentric_tolerance
                and np.max(barycentric) <= 1.0 + barycentric_tolerance
                and abs(signed_distance) <= residual_tolerance_cm
            ):
                matches.append(
                    (
                        abs(signed_distance),
                        int(facet_index),
                        barycentric,
                        projected,
                    )
                )
        if not matches:
            raise ValueError(
                f"record {record_index} is not contained by a facet on surface {surface_id}"
            )
        if len(matches) > 1:
            reference = matches[0][1]
            for _, candidate, _, _ in matches[1:]:
                same_normal = (
                    np.dot(
                        atlas.normals_global[reference],
                        atlas.normals_global[candidate],
                    )
                    >= 1.0 - 1.0e-10
                )
                same_plane = (
                    abs(
                        np.dot(
                            atlas.origins_cm[candidate]
                            - atlas.origins_cm[reference],
                            atlas.normals_global[reference],
                        )
                    )
                    <= residual_tolerance_cm
                )
                if not (same_normal and same_plane):
                    raise ValueError(
                        f"record {record_index} lies on an ambiguous noncoplanar facet edge"
                    )
        matches.sort(key=lambda item: (item[0], str(atlas.facet_ids[item[1]])))
        residual, facet_index, barycentric, reconstructed = matches[0]
        selected[record_index] = facet_index
        barycentrics[record_index] = barycentric
        residuals[record_index] = residual
        reconstructions[record_index] = reconstructed

    frames = atlas.frames_global_to_local[selected]
    local_positions = np.einsum(
        "nij,nj->ni", frames, positions - atlas.origins_cm[selected]
    )
    local_directions = np.einsum("nij,nj->ni", frames, directions)
    normals = atlas.normals_global[selected]
    mu = np.sum(directions * normals, axis=1)
    sense = np.where(
        mu < -grazing_tolerance,
        "incoming",
        np.where(mu > grazing_tolerance, "outgoing", "grazing"),
    )
    output = {
        **{name: np.asarray(value) for name, value in phase_space.items()},
        "facet_id": atlas.facet_ids[selected],
        "patch_id": atlas.patch_ids[selected],
        "facet_index": selected,
        "barycentric_coordinates": barycentrics,
        "reconstructed_position_global_cm": reconstructions,
        "distance_to_facet_residual_cm": residuals,
        "outward_normal_global": normals,
        "mu": mu,
        "crossing_sense": sense,
        "position_local_cm": local_positions,
        "direction_local": local_directions,
        "local_frame_global": frames,
    }
    audit = {
        "schema": "parastell.cached_surface_crossing_localization/v1.0.0",
        "status": "PASS",
        "atlas_sha256": atlas.atlas_sha256,
        "record_count": len(positions),
        "candidate_facets_considered": candidates_considered,
        "full_surface_facets_available": int(
            sum(
                len(atlas.surface_to_indices[int(value)])
                for value in record_surfaces
            )
        ),
        "aabb_pruning_used": True,
        "cached_frames_used": True,
        "maximum_reconstruction_residual_cm": (
            float(np.max(residuals)) if len(residuals) else 0.0
        ),
        "incoming_records": int(np.count_nonzero(sense == "incoming")),
        "outgoing_records": int(np.count_nonzero(sense == "outgoing")),
        "grazing_records": int(np.count_nonzero(sense == "grazing")),
        "physical_qualification_claimed": False,
    }
    return output, audit


def build_conservative_mapping_metadata(
    atlas: FacetPatchAtlas,
    patch_id_by_facet: Sequence[str] | None = None,
    *,
    family_id_by_facet: Sequence[str] | None = None,
    tape_id_by_facet: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe conservative integral/density maps without moving geometry."""

    patch_ids = np.asarray(
        atlas.patch_ids if patch_id_by_facet is None else patch_id_by_facet
    ).astype(str)
    if patch_ids.shape != (atlas.facet_count,) or np.any(patch_ids == ""):
        raise ValueError("patch IDs must align one-to-one with facets")
    if (family_id_by_facet is None) != (tape_id_by_facet is None):
        raise ValueError(
            "family and tape facet identities must be supplied together"
        )
    family_ids = None
    tape_ids = None
    if family_id_by_facet is not None:
        family_ids = np.asarray(family_id_by_facet).astype(str)
        tape_ids = np.asarray(tape_id_by_facet).astype(str)
        if (
            family_ids.shape != (atlas.facet_count,)
            or tape_ids.shape != (atlas.facet_count,)
            or np.any(family_ids == "")
            or np.any(tape_ids == "")
        ):
            raise ValueError(
                "family and tape IDs must align one-to-one with facets"
            )
    unique_patches = sorted(set(patch_ids.tolist()))
    patch_index = {value: index for index, value in enumerate(unique_patches)}
    patch_areas = np.zeros(len(unique_patches))
    rows = np.asarray(
        [patch_index[value] for value in patch_ids], dtype=np.int64
    )
    columns = np.arange(atlas.facet_count, dtype=np.int64)
    for row, area in zip(rows, atlas.areas_cm2, strict=True):
        patch_areas[row] += area
    if np.any(~np.isfinite(patch_areas)) or np.any(patch_areas <= 0.0):
        raise ValueError("patch areas must be finite and positive")
    area_fractions = atlas.areas_cm2 / patch_areas[rows]
    closure = np.bincount(
        rows, weights=area_fractions, minlength=len(unique_patches)
    )
    metadata = {
        "schema": MAPPING_SCHEMA,
        "status": "SOFTWARE_CONTRACT_READY",
        "atlas_sha256": atlas.atlas_sha256,
        "facet_count": atlas.facet_count,
        "patch_count": len(unique_patches),
        "patch_ids": unique_patches,
        "patch_area_cm2": [float(value) for value in patch_areas],
        "coo_row_patch_index": rows.tolist(),
        "coo_column_facet_index": columns.tolist(),
        "facet_integral_to_patch_integral_weights": np.ones(
            atlas.facet_count
        ).tolist(),
        "facet_density_to_patch_density_weights": area_fractions.tolist(),
        "patch_integral_to_facet_integral_weights": area_fractions.tolist(),
        "source_mapping_rule": (
            "sum facet-integrated current/rate; use area weights only for densities"
        ),
        "response_mapping_rule": (
            "sum facet-integrated responses to patches; redistribution of a patch "
            "integral uses declared area fractions and is an explicit uniform-density approximation"
        ),
        "direction_mapping_rule": (
            "global/local directions use the hash-bound right-handed facet frame; "
            "direction vectors are never scalar-averaged"
        ),
        "area_fraction_closure_max_abs": float(np.max(np.abs(closure - 1.0))),
        "conservative_integral_closure_pass": bool(
            np.allclose(closure, 1.0, rtol=0.0, atol=1.0e-12)
        ),
        "hidden_renormalization_used": False,
        "physical_qualification_claimed": False,
    }
    if family_ids is not None and tape_ids is not None:
        patch_bindings = []
        for patch_index_value, patch_id in enumerate(unique_patches):
            facet_indices = np.flatnonzero(rows == patch_index_value)
            patch_families = sorted(set(family_ids[facet_indices].tolist()))
            patch_tapes = sorted(set(tape_ids[facet_indices].tolist()))
            if len(patch_families) != 1 or len(patch_tapes) != 1:
                raise ValueError(
                    "every cached patch must bind exactly one REBCO family and tape"
                )
            patch_bindings.append(
                {
                    "patch_id": patch_id,
                    "family_id": patch_families[0],
                    "tape_id": patch_tapes[0],
                    "facet_count": len(facet_indices),
                    "facet_ids": sorted(
                        atlas.facet_ids[facet_indices].tolist()
                    ),
                    "patch_area_cm2": float(patch_areas[patch_index_value]),
                }
            )
        metadata["identity_binding"] = {
            "schema": "parastell.family_tagged_facet_patch_identity/v1.0.0",
            "family_ids": sorted(set(family_ids.tolist())),
            "tape_ids": sorted(set(tape_ids.tolist())),
            "family_ids_by_facet_sha256": _canonical_hash(
                {"family_ids": family_ids.tolist()}
            ),
            "tape_ids_by_facet_sha256": _canonical_hash(
                {"tape_ids": tape_ids.tolist()}
            ),
            "patch_bindings": patch_bindings,
            "complete_facet_identity_pass": sum(
                row["facet_count"] for row in patch_bindings
            )
            == atlas.facet_count,
        }
    metadata["mapping_sha256"] = _canonical_hash(metadata)
    return metadata


def write_facet_patch_atlas_create_only(
    output_dir: str | Path,
    atlas: FacetPatchAtlas,
    mapping: Mapping[str, Any],
    *,
    parent_geometry_sha256: str,
    input_class: str = "FIXTURE",
) -> dict[str, Any]:
    """Publish an atlas cache and compact manifest in a new directory."""

    from .source_conformal_interface import write_compact_json_create_only

    if not _valid_sha256(parent_geometry_sha256):
        raise ValueError("parent geometry requires a SHA-256 identity")
    if mapping.get("atlas_sha256") != atlas.atlas_sha256:
        raise ValueError("mapping is not bound to this atlas")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"create-only atlas output exists: {destination}"
        )
    destination.mkdir(parents=True)
    cache = destination / "FACET_PATCH_ATLAS.npz"
    np.savez_compressed(
        cache,
        facet_ids=atlas.facet_ids.astype("U"),
        patch_ids=atlas.patch_ids.astype("U"),
        surface_ids=atlas.surface_ids,
        triangles_cm=atlas.triangles_cm,
        normals_global=atlas.normals_global,
        frames_global_to_local=atlas.frames_global_to_local,
        areas_cm2=atlas.areas_cm2,
        bounds_min_cm=atlas.bounds_min_cm,
        bounds_max_cm=atlas.bounds_max_cm,
        neighbor_offsets=atlas.neighbor_offsets,
        neighbor_indices=atlas.neighbor_indices,
    )
    cache_sha256 = hashlib.sha256(cache.read_bytes()).hexdigest()
    mapping_identity = write_compact_json_create_only(
        destination / "CONSERVATIVE_FACET_PATCH_MAPPING.json", mapping
    )
    manifest = {
        **atlas.compact_metadata(),
        "schema": "parastell.facet_patch_atlas_manifest/v1.0.0",
        "input_class": input_class,
        "parent_geometry_sha256": parent_geometry_sha256,
        "mapping_sha256": mapping.get("mapping_sha256"),
        "mapping_artifact": {
            "name": "CONSERVATIVE_FACET_PATCH_MAPPING.json",
            "size_bytes": mapping_identity["size_bytes"],
            "sha256": mapping_identity["sha256"],
        },
        "cache": {
            "name": cache.name,
            "size_bytes": cache.stat().st_size,
            "sha256": cache_sha256,
        },
        "transport_eligible": False,
    }
    identity = write_compact_json_create_only(
        destination / "FACET_PATCH_ATLAS_MANIFEST.json", manifest
    )
    return {
        "output_dir": str(destination.resolve()),
        "cache_sha256": cache_sha256,
        "mapping": mapping_identity,
        "manifest": identity,
    }
