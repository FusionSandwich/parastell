"""Closed magnet-envelope extraction from a production DAGMC volume."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .magnet_boundary_envelope import EnvelopeSurface
from .magnet_boundary_envelope import MagnetBoundaryEnvelope


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


@dataclass(frozen=True)
class FacetedSurface:
    surface_id: int
    role: str
    triangles_cm: np.ndarray
    triangle_normals_outward: np.ndarray
    triangle_areas_cm2: np.ndarray
    centroid_global_cm: np.ndarray

    @property
    def area_cm2(self) -> float:
        return float(self.triangle_areas_cm2.sum())

    def normal_at(self, position_global_cm: Sequence[float]) -> np.ndarray:
        point = np.asarray(position_global_cm, dtype=float)
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
        bary_u = (vv * pu - uv * pv) / denominator
        bary_v = (uu * pv - uv * pu) / denominator
        tolerance = 1.0e-7
        contains = (
            (bary_u >= -tolerance)
            & (bary_v >= -tolerance)
            & (bary_u + bary_v <= 1.0 + tolerance)
        )
        if np.any(contains):
            candidates = np.flatnonzero(contains)
            index = int(
                candidates[np.argmin(np.abs(plane_distance[candidates]))]
            )
        else:
            centroids = self.triangles_cm.mean(axis=1)
            index = int(np.argmin(np.sum((centroids - point) ** 2, axis=1)))
        return self.triangle_normals_outward[index].copy()


@dataclass(frozen=True)
class DagmcEnvelope:
    envelope: MagnetBoundaryEnvelope
    faceted_surfaces: tuple[FacetedSurface, ...]
    triangle_edges: int
    maximum_edge_multiplicity_error: int
    vector_area_closure_relative: float

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


def _triangles(coordinates) -> np.ndarray:
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) % 3:
        raise ValueError("DAGMC triangle coordinates must have shape (3*N, 3)")
    return values.reshape((-1, 3, 3))


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
) -> DagmcEnvelope:
    """Extract every surface of one DAGMC volume and prove faceted closure."""
    import pydagmc

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
        # DAGMC facet normals point from the reverse volume toward the forward
        # volume.  Use that topological sense directly: centroid tests are not
        # valid for concave or toroidal magnet volumes.
        reverse_id = getattr(surface.reverse_volume, "id", None)
        forward_id = getattr(surface.forward_volume, "id", None)
        if reverse_id == volume.id:
            pass
        elif forward_id == volume.id:
            normals *= -1.0
        else:
            raise ValueError(
                f"surface {surface.id} has no sense for DAGMC volume {volume.id}"
            )
        role = _role(centroid, volume_centroid, plasma, toroidal, poloidal)
        faceted.append(
            FacetedSurface(
                surface.id, role, triangles, normals, areas, centroid
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
    envelope = MagnetBoundaryEnvelope(
        envelope_id=envelope_id,
        magnet_component=magnet_id,
        dagmc_volume_id=int(volume_id),
        surfaces=tuple(schema_surfaces),
        dagmc_geometry_sha256=_hash(dagmc_path),
        metadata={
            "material": volume.material,
            "volume_cm3": float(volume.volume),
            "watertight_proof": "every rounded faceted edge occurs exactly twice",
            "faceted_vector_area_closure_relative": closure,
        },
    )
    return DagmcEnvelope(
        envelope,
        tuple(faceted),
        len(edge_counts),
        0,
        closure,
    )


@dataclass(frozen=True)
class MagnetVolumeRecord:
    """Stable identity for one magnet-related DAGMC volume."""

    volume_id: int
    material: str
    component_role: str
    surface_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume_id": self.volume_id,
            "material": self.material,
            "component_role": self.component_role,
            "surface_ids": list(self.surface_ids),
        }


@dataclass(frozen=True)
class MagnetVolumeInventory:
    """Deterministic inventory used before any envelope is selected."""

    dagmc_path: str
    dagmc_geometry_sha256: str
    volumes: tuple[MagnetVolumeRecord, ...]

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
            "volumes": [item.to_dict() for item in self.volumes],
            "winding_pack_volume_ids": [
                item.volume_id for item in self.winding_packs
            ],
            "magnet_casing_volume_ids": [
                item.volume_id for item in self.casings
            ],
        }


def _material_key(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def discover_magnet_volumes(
    dagmc_path: str | Path,
    *,
    winding_pack_materials: Sequence[str] = ("winding_pack",),
    casing_materials: Sequence[str] = ("magnet_casing",),
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
                int(surface.id)
                for surface in getattr(volume, "surfaces", ())
            )
        )
        if not surface_ids:
            raise ValueError(
                f"magnet DAGMC volume {volume_id} has no boundary surfaces"
            )
        records.append(
            MagnetVolumeRecord(
                volume_id=int(volume_id),
                material=material,
                component_role=role,
                surface_ids=surface_ids,
            )
        )
    inventory = MagnetVolumeInventory(str(path), _hash(path), tuple(records))
    if not inventory.winding_packs:
        raise ValueError(
            "DAGMC geometry has no winding-pack volume with an accepted "
            "material tag"
        )
    return inventory


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
                poloidal_direction_global=frame[
                    "poloidal_direction_global"
                ],
                spatial_bins=spatial_bins,
            )
        )
    return tuple(results)
