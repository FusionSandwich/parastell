"""Optional, qualification-gated magnet-targeted weight-window utilities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_SCHEMA = "parastell.magnet_weight_window_contract/v1.0.0"
QUALIFICATION_SCHEMA = "parastell.magnet_weight_window_qualification/v1.0.0"
TERMINAL_CLASSIFICATIONS = {
    "QUALIFIED_AND_ENABLED",
    "QUALIFIED_FOR_NEUTRONS_ONLY",
    "QUALIFIED_FOR_SELECTED_MAGNETS_ONLY",
    "NO_MATERIAL_BENEFIT_DISABLE",
    "REJECTED_BIAS",
    "REJECTED_INSTABILITY",
    "INSUFFICIENT_PILOT_STATISTICS",
}


def build_weight_window_mesh_from_step(
    component_step_paths: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    target_magnet_ids: Sequence[str],
    geometry_fingerprint: str,
    characteristic_length_cm: Mapping[str, float],
    mode: str = "all_magnet_coarse",
    volume_meshing_algorithm: int = 1,
    fuse_multi_volume_components: bool = True,
) -> dict[str, Any]:
    """Build one conformal Gmsh/MOAB auxiliary mesh without requiring Cubit."""
    if mode not in {"all_magnet_coarse", "selected_magnet_targeted"}:
        raise ValueError("unsupported weight-window mesh mode")
    if not component_step_paths:
        raise ValueError("WW mesh requires physical component STEP inputs")
    missing_sizes = set(component_step_paths) - set(characteristic_length_cm)
    if missing_sizes:
        raise ValueError(
            f"WW mesh sizes are missing for {sorted(missing_sizes)}"
        )
    if any(float(value) <= 0.0 for value in characteristic_length_cm.values()):
        raise ValueError("WW characteristic lengths must be positive")
    volume_meshing_algorithm = int(volume_meshing_algorithm)
    if volume_meshing_algorithm not in {1, 4, 7, 9, 10}:
        raise ValueError(
            "unsupported Gmsh 3-D meshing algorithm; expected one of "
            "1, 4, 7, 9, or 10"
        )
    try:
        import gmsh
        from pymoab import core
    except ImportError as exc:
        raise RuntimeError(
            "Gmsh and PyMOAB are required to build a WW mesh"
        ) from exc
    destination = Path(output_path).resolve().with_suffix(".h5m")
    destination.parent.mkdir(parents=True, exist_ok=True)
    intermediate = destination.with_suffix(".gmsh.msh")
    gmsh.initialize()
    try:
        gmsh.logger.start()
        gmsh.model.add("parastell_magnet_weight_window")
        source_entities: dict[str, list[tuple[int, int]]] = {}
        all_volumes = []
        for component, source_path in sorted(component_step_paths.items()):
            path = Path(source_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            volumes = [
                tuple(value)
                for value in gmsh.model.occ.importShapes(str(path))
                if int(value[0]) == 3
            ]
            if not volumes:
                raise ValueError(
                    f"component {component!r} has no STEP volumes"
                )
            if fuse_multi_volume_components and len(volumes) > 1:
                fused, _ = gmsh.model.occ.fuse(
                    volumes[:1],
                    volumes[1:],
                    removeObject=True,
                    removeTool=True,
                )
                volumes = [
                    tuple(value) for value in fused if int(value[0]) == 3
                ]
                if not volumes:
                    raise ValueError(
                        f"component {component!r} could not be fused into a "
                        "spatial WW domain"
                    )
            source_entities[component] = volumes
            all_volumes.extend((component, value) for value in volumes)
        objects = [value for _, value in all_volumes[:1]]
        tools = [value for _, value in all_volumes[1:]]
        if tools:
            _, entity_maps = gmsh.model.occ.fragment(objects, tools)
            mapped: dict[str, set[tuple[int, int]]] = {
                name: set() for name in source_entities
            }
            for (component, _), mapping in zip(all_volumes, entity_maps):
                mapped[component].update(
                    tuple(value) for value in mapping if int(value[0]) == 3
                )
        else:
            mapped = {
                name: {tuple(value) for value in values}
                for name, values in source_entities.items()
            }
        gmsh.model.occ.synchronize()
        owners: dict[tuple[int, int], str] = {}
        for component, volumes in mapped.items():
            for volume in volumes:
                prior = owners.setdefault(volume, component)
                if prior != component:
                    raise ValueError(
                        f"fragmented WW volume belongs to both {prior!r} and {component!r}"
                    )
        physical_groups = {}
        for group_id, (component, volumes) in enumerate(
            sorted(mapped.items()), start=1
        ):
            tags = sorted(tag for dim, tag in volumes if dim == 3)
            physical = gmsh.model.addPhysicalGroup(3, tags, group_id)
            gmsh.model.setPhysicalName(3, physical, component)
            physical_groups[component] = {
                "physical_group_id": physical,
                "volume_tags": tags,
            }
            boundary = gmsh.model.getBoundary(
                [(3, tag) for tag in tags], combined=True, recursive=True
            )
            points = [(dim, tag) for dim, tag in boundary if dim == 0]
            if points:
                gmsh.model.mesh.setSize(
                    points, float(characteristic_length_cm[component])
                )
        gmsh.option.setNumber("Mesh.Algorithm3D", volume_meshing_algorithm)
        try:
            gmsh.model.mesh.generate(3)
        except Exception as exc:
            diagnostics = [
                message
                for message in gmsh.logger.get()
                if "PLC Error" in message
                or "failed to recover" in message
                or "3D mesh failed" in message
            ]
            detail = " | ".join(diagnostics[-5:]) or str(exc)
            raise RuntimeError(
                "Gmsh 3-D WW meshing failed with algorithm "
                f"{volume_meshing_algorithm}: {detail}"
            ) from exc
        for component, group in physical_groups.items():
            count = 0
            for volume_tag in group["volume_tags"]:
                _, element_tags, _ = gmsh.model.mesh.getElements(3, volume_tag)
                count += sum(len(value) for value in element_tags)
            group["element_count"] = int(count)
            if count <= 0:
                raise ValueError(
                    f"WW component {component!r} has no tetrahedra"
                )
        # The PyMOAB build used by the pinned OpenMC 0.16 environment reads
        # Gmsh 2.2, not the Gmsh 4.1 default.
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(intermediate))
    finally:
        with suppress(Exception):
            gmsh.logger.stop()
        gmsh.finalize()
    mesh = core.Core()
    mesh.load_file(str(intermediate))
    mesh.write_file(str(destination))
    intermediate.unlink(missing_ok=True)
    manifest = {
        "schema": "parastell.magnet_weight_window_mesh/v1.0.0",
        "mode": mode,
        "geometry_fingerprint": geometry_fingerprint,
        "target_magnet_ids": list(target_magnet_ids),
        "component_inputs": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(path),
                "characteristic_length_cm": float(
                    characteristic_length_cm[name]
                ),
            }
            for name, path in sorted(component_step_paths.items())
        },
        "physical_groups": physical_groups,
        "h5m": {
            "path": str(destination),
            "sha256": _sha256(destination),
            "size_bytes": destination.stat().st_size,
        },
        "backend": {
            "surface_volume_mesh": "Gmsh",
            "storage": "MOAB H5M",
            "cubit": False,
            "volume_meshing_algorithm": volume_meshing_algorithm,
            "fuse_multi_volume_components": bool(fuse_multi_volume_components),
            "gmsh_intermediate_format_version": "2.2",
        },
    }
    sidecar = destination.with_suffix(".mesh-manifest.json")
    sidecar.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validation = validate_moab_weight_window_mesh(
        destination,
        mesh_manifest_path=sidecar,
        required_components=tuple(component_step_paths),
        target_magnet_ids=target_magnet_ids,
        geometry_fingerprint=geometry_fingerprint,
    )
    return {
        **manifest,
        "validation": validation,
        "manifest_path": str(sidecar),
    }


def validate_moab_weight_window_mesh(
    path: str | Path,
    *,
    mesh_manifest_path: str | Path,
    required_components: Sequence[str],
    target_magnet_ids: Sequence[str],
    geometry_fingerprint: str,
) -> dict[str, Any]:
    """Validate a written MOAB tetrahedral mesh and its physical mapping."""
    try:
        from pymoab import core, types
    except ImportError as exc:
        raise RuntimeError(
            "PyMOAB is required to validate a WW H5M mesh"
        ) from exc
    source = Path(path).resolve()
    manifest = json.loads(Path(mesh_manifest_path).read_text(encoding="utf-8"))
    if manifest["geometry_fingerprint"] != geometry_fingerprint:
        raise ValueError("WW mesh geometry-fingerprint association is stale")
    if manifest["h5m"]["sha256"] != _sha256(source):
        raise ValueError("WW mesh H5M hash does not match its manifest")
    groups = manifest["physical_groups"]
    missing = set(required_components) - set(groups)
    missing_targets = set(target_magnet_ids) - set(groups)
    if missing or missing_targets:
        raise ValueError(
            f"WW mesh physical mapping is incomplete: components={sorted(missing)}, magnets={sorted(missing_targets)}"
        )
    mb = core.Core()
    mb.load_file(str(source))
    tetrahedra = list(mb.get_entities_by_type(0, types.MBTET))
    if not tetrahedra:
        raise ValueError("WW H5M contains no tetrahedra")
    connectivity = np.asarray(
        [list(mb.get_connectivity(element)) for element in tetrahedra],
        dtype=np.uint64,
    )
    if connectivity.shape[1] != 4:
        raise ValueError("WW H5M contains non-tetrahedral 3D elements")
    canonical = np.sort(connectivity, axis=1)
    if len(np.unique(canonical, axis=0)) != len(connectivity):
        raise ValueError("WW H5M contains duplicate tetrahedra")
    material_set_tag = mb.tag_get_handle("MATERIAL_SET")
    material_sets: dict[int, list[int]] = {}
    for entity_set in mb.get_entities_by_type(0, types.MBENTITYSET):
        tag_names = {
            tag.get_name() for tag in mb.tag_get_tags_on_entity(entity_set)
        }
        if "MATERIAL_SET" not in tag_names:
            continue
        group_id = int(
            np.asarray(mb.tag_get_data(material_set_tag, entity_set)).ravel()[
                0
            ]
        )
        group_tetrahedra = list(
            mb.get_entities_by_type(entity_set, types.MBTET)
        )
        if group_id in material_sets:
            raise ValueError(
                f"WW H5M repeats MATERIAL_SET physical group {group_id}"
            )
        material_sets[group_id] = [int(value) for value in group_tetrahedra]
    expected_group_ids = {
        int(value["physical_group_id"]) for value in groups.values()
    }
    if set(material_sets) != expected_group_ids:
        raise ValueError(
            "WW H5M MATERIAL_SET groups do not match the mesh manifest"
        )
    actual_group_counts = {}
    group_connectivity = {}
    tetrahedron_index = {
        int(handle): index for index, handle in enumerate(tetrahedra)
    }
    for component, group in groups.items():
        group_id = int(group["physical_group_id"])
        actual_count = len(material_sets[group_id])
        if actual_count != int(group["element_count"]):
            raise ValueError(
                f"WW H5M element count for {component!r} does not match "
                "the mesh manifest"
            )
        actual_group_counts[component] = actual_count
        indices_for_group = np.asarray(
            [tetrahedron_index[handle] for handle in material_sets[group_id]],
            dtype=int,
        )
        component_count = _tetrahedral_connected_component_count(
            canonical[indices_for_group]
        )
        if component_count != 1:
            raise ValueError(
                f"WW component {component!r} has {component_count} "
                "disconnected tetrahedral regions"
            )
        group_connectivity[component] = component_count
    global_component_count = _tetrahedral_connected_component_count(canonical)
    if global_component_count != 1:
        raise ValueError(
            f"WW H5M has {global_component_count} disconnected selected regions"
        )
    handles = np.unique(connectivity)
    coordinates = np.asarray(mb.get_coords(handles)).reshape((-1, 3))
    lookup = {int(handle): index for index, handle in enumerate(handles)}
    indices = np.asarray(
        [[lookup[int(handle)] for handle in row] for row in connectivity],
        dtype=int,
    )
    points = coordinates[indices]
    volumes = (
        np.abs(
            np.einsum(
                "ij,ij->i",
                points[:, 1] - points[:, 0],
                np.cross(
                    points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]
                ),
            )
        )
        / 6.0
    )
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("WW H5M contains nonpositive element volumes")
    return {
        "status": "PASS",
        "tetrahedron_count": len(tetrahedra),
        "vertex_count": len(handles),
        "total_volume_cm3": float(volumes.sum()),
        "minimum_volume_cm3": float(volumes.min()),
        "maximum_volume_cm3": float(volumes.max()),
        "positive_element_volumes": True,
        "duplicate_tetrahedra": False,
        "physical_component_mapping": actual_group_counts,
        "physical_component_connected_region_counts": group_connectivity,
        "global_connected_region_count": global_component_count,
        "material_sets_verified_from_h5m": True,
        "geometry_fingerprint": geometry_fingerprint,
        "target_magnet_ids": list(target_magnet_ids),
    }


def _tetrahedral_connected_component_count(
    canonical_connectivity: np.ndarray,
) -> int:
    """Count face-connected tetrahedral components and reject non-manifolds."""
    connectivity = np.asarray(canonical_connectivity, dtype=np.uint64)
    if connectivity.ndim != 2 or connectivity.shape[1] != 4:
        raise ValueError("tetrahedral connectivity must have shape (N, 4)")
    if len(connectivity) == 0:
        return 0
    parents = np.arange(len(connectivity), dtype=int)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parents[right] = left

    faces: dict[tuple[int, int, int], int] = {}
    duplicate_faces = set()
    for element_index, element in enumerate(connectivity):
        for omitted in range(4):
            face = tuple(int(value) for value in np.delete(element, omitted))
            if face in duplicate_faces:
                raise ValueError(
                    "WW H5M contains a non-manifold tetrahedral face"
                )
            prior = faces.get(face)
            if prior is None:
                faces[face] = element_index
            else:
                union(prior, element_index)
                duplicate_faces.add(face)
    return len({find(index) for index in range(len(connectivity))})


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_weight_window_energy_grid(
    edges_eV: Sequence[float],
    *,
    transport_min_eV: float,
    transport_max_eV: float,
) -> tuple[float, ...]:
    """Validate a VR control grid without conflating it with tally groups."""
    edges = np.asarray(edges_eV, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError(
            "weight-window energy edges must be finite and increasing"
        )
    if transport_min_eV < 0.0 or transport_max_eV <= transport_min_eV:
        raise ValueError("transport energy range is invalid")
    if edges[0] > transport_min_eV or edges[-1] < transport_max_eV:
        raise ValueError(
            "weight-window grid does not cover the transport range"
        )
    if edges[-1] < 15.0e6:
        raise ValueError(
            "weight-window grid does not cover the 13-15 MeV D-T region"
        )
    return tuple(float(value) for value in edges)


def candidate_neutron_weight_window_grids(
    transport_max_eV: float = 20.0e6,
) -> dict[str, tuple[float, ...]]:
    """Return documented 4/8/16-group candidates covering fusion energies."""
    maximum = float(transport_max_eV)
    if maximum < 15.0e6:
        raise ValueError("neutron transport maximum must include D-T energies")
    grids = {
        "ww_4_group": (0.0, 0.625, 1.0e5, 13.0e6, maximum),
        "ww_8_group": (
            0.0,
            0.625,
            100.0,
            1.0e4,
            1.0e5,
            1.0e6,
            2.5e6,
            13.0e6,
            maximum,
        ),
        "ww_16_group": (
            0.0,
            0.0253,
            0.625,
            10.0,
            100.0,
            1.0e3,
            1.0e4,
            1.0e5,
            5.0e5,
            1.0e6,
            2.5e6,
            5.0e6,
            10.0e6,
            13.0e6,
            14.0e6,
            15.0e6,
            maximum,
        ),
    }
    return {
        name: validate_weight_window_energy_grid(
            edges, transport_min_eV=0.0, transport_max_eV=maximum
        )
        for name, edges in grids.items()
    }


def validate_split_controls(
    *,
    max_history_splits: int,
    max_split: int,
    survival_ratio: float = 3.0,
    weight_cutoff: float = 1.0e-38,
) -> dict[str, Any]:
    """Reject unbounded or internally inconsistent splitting controls."""
    if (
        not isinstance(max_history_splits, int)
        or not 1 <= max_history_splits <= 20_000
    ):
        raise ValueError(
            "max_history_splits must be an integer from 1 to 20000"
        )
    if not isinstance(max_split, int) or not 1 <= max_split <= 1_000:
        raise ValueError("max_split must be an integer from 1 to 1000")
    if not np.isfinite(survival_ratio) or not 1.0 < survival_ratio <= 20.0:
        raise ValueError("survival_ratio must be in (1, 20]")
    if not np.isfinite(weight_cutoff) or not 0.0 < weight_cutoff < 1.0:
        raise ValueError("weight_cutoff must be in (0, 1)")
    return {
        "max_history_splits": max_history_splits,
        "max_split": max_split,
        "survival_ratio": float(survival_ratio),
        "weight_cutoff": float(weight_cutoff),
        "historical_unqualified_100000_setting_used": False,
    }


def validate_weight_window_mesh(
    vertices_cm: Sequence[Sequence[float]],
    tetrahedra: Sequence[Sequence[int]],
    *,
    component_by_element: Sequence[str],
    required_components: Sequence[str],
    target_magnet_ids: Sequence[str],
    geometry_fingerprint: str,
) -> dict[str, Any]:
    """Validate an auxiliary unstructured mesh independently of transport CAD."""
    vertices = np.asarray(vertices_cm, dtype=float)
    elements = np.asarray(tetrahedra, dtype=int)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError("WW vertices must have shape (N, 3) and be finite")
    if elements.ndim != 2 or elements.shape[1] != 4 or len(elements) == 0:
        raise ValueError("WW elements must be nonempty tetrahedra")
    if np.any(elements < 0) or np.any(elements >= len(vertices)):
        raise ValueError("WW tetrahedra reference invalid vertices")
    canonical = np.sort(elements, axis=1)
    if len(np.unique(canonical, axis=0)) != len(elements):
        raise ValueError("WW mesh contains duplicate tetrahedra")
    points = vertices[elements]
    six_volume = np.einsum(
        "ij,ij->i",
        points[:, 1] - points[:, 0],
        np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]),
    )
    volumes = np.abs(six_volume) / 6.0
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("WW mesh contains nonpositive-volume tetrahedra")
    components = np.asarray(component_by_element, dtype=object).astype(str)
    if components.shape != (len(elements),):
        raise ValueError("component mapping must align with WW tetrahedra")
    missing_components = sorted(set(required_components) - set(components))
    missing_magnets = sorted(set(target_magnet_ids) - set(components))
    if missing_components or missing_magnets:
        raise ValueError(
            f"WW mesh coverage missing components={missing_components}, magnets={missing_magnets}"
        )
    faces: dict[tuple[int, int, int], list[int]] = {}
    for element_index, element in enumerate(elements):
        for omitted in range(4):
            face = tuple(sorted(np.delete(element, omitted).tolist()))
            faces.setdefault(face, []).append(element_index)
    adjacency = [set() for _ in elements]
    for owners in faces.values():
        if len(owners) == 2:
            first, second = owners
            adjacency[first].add(second)
            adjacency[second].add(first)
        elif len(owners) > 2:
            raise ValueError(
                "WW mesh contains a non-manifold tetrahedral face"
            )
    visited = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    if len(visited) != len(elements):
        raise ValueError("WW mesh contains disconnected selected regions")
    return {
        "status": "PASS",
        "geometry_fingerprint": geometry_fingerprint,
        "element_count": len(elements),
        "vertex_count": len(vertices),
        "volume_cm3": float(volumes.sum()),
        "minimum_element_volume_cm3": float(volumes.min()),
        "maximum_element_volume_cm3": float(volumes.max()),
        "component_element_counts": {
            name: int(np.count_nonzero(components == name))
            for name in sorted(set(components))
        },
        "target_magnet_ids": list(target_magnet_ids),
        "positive_volumes": True,
        "duplicate_tetrahedra": False,
        "connected": True,
    }


@dataclass(frozen=True)
class WeightWindowArtifactContract:
    canonical_geometry_fingerprint: str
    raw_h5m_sha256: str
    source_definition_sha256: str
    source_mesh_sha256: str
    physical_source_rate_per_s: float
    material_manifest_sha256: str
    nuclear_data_manifest_sha256: str
    weight_window_mesh_sha256: str
    particle_type: str
    energy_bounds_eV: tuple[float, ...]
    generator_method: str
    generator_settings: Mapping[str, Any]
    openmc_version: str
    openmc_source_sha: str
    generation_histories: int
    generation_batches: int
    generation_seed: int
    selected_magnet_ids: tuple[str, ...]
    schema: str = CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        hashes = (
            self.canonical_geometry_fingerprint,
            self.raw_h5m_sha256,
            self.source_definition_sha256,
            self.source_mesh_sha256,
            self.material_manifest_sha256,
            self.nuclear_data_manifest_sha256,
            self.weight_window_mesh_sha256,
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError(
                "artifact contract hashes must be SHA-256 hex digests"
            )
        if (
            not np.isfinite(self.physical_source_rate_per_s)
            or self.physical_source_rate_per_s <= 0.0
        ):
            raise ValueError(
                "physical source rate must be finite and positive"
            )
        validate_weight_window_energy_grid(
            self.energy_bounds_eV,
            transport_min_eV=self.energy_bounds_eV[0],
            transport_max_eV=self.energy_bounds_eV[-1],
        )
        if self.particle_type not in {"neutron", "photon"}:
            raise ValueError(
                "weight-window particle type must be neutron or photon"
            )
        if self.generator_method != "magic":
            raise ValueError("the pinned OpenMC 0.16 workflow supports MAGIC")
        if (
            min(
                self.generation_histories,
                self.generation_batches,
                self.generation_seed,
            )
            <= 0
        ):
            raise ValueError(
                "generation histories, batches, and seed must be positive"
            )
        if not self.selected_magnet_ids:
            raise ValueError(
                "weight-window contract requires selected magnets"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def write_weight_window_contract(
    path: str | Path,
    contract: WeightWindowArtifactContract,
    *,
    weight_window_path: str | Path,
) -> dict[str, Any]:
    artifact = Path(weight_window_path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    value = {
        **contract.to_dict(),
        "contract_sha256": contract.sha256,
        "weight_window_artifact": {
            "path": str(artifact),
            "sha256": _sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        },
    }
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        stored_artifact_path = artifact.relative_to(output.parent).as_posix()
        artifact_path_basis = "CONTRACT_DIRECTORY_RELATIVE"
    except ValueError:
        stored_artifact_path = str(artifact)
        artifact_path_basis = "ABSOLUTE_EXTERNAL"
    value["weight_window_artifact"]["path"] = stored_artifact_path
    value["weight_window_artifact"]["path_basis"] = artifact_path_basis
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def require_compatible_weight_window(
    contract_path: str | Path,
    expected: WeightWindowArtifactContract,
) -> dict[str, Any]:
    contract_source = Path(contract_path).resolve()
    value = json.loads(contract_source.read_text(encoding="utf-8"))
    actual = {
        key: value[key]
        for key in WeightWindowArtifactContract.__dataclass_fields__
    }
    actual["energy_bounds_eV"] = tuple(actual["energy_bounds_eV"])
    actual["selected_magnet_ids"] = tuple(actual["selected_magnet_ids"])
    restored = WeightWindowArtifactContract(**actual)
    if value.get("contract_sha256") != restored.sha256:
        raise ValueError("stored weight-window contract hash is invalid")
    if restored.sha256 != expected.sha256:
        differing = [
            name
            for name in WeightWindowArtifactContract.__dataclass_fields__
            if restored.to_dict()[name] != expected.to_dict()[name]
        ]
        raise ValueError(f"weight-window contract mismatch: {differing}")
    artifact = value["weight_window_artifact"]
    artifact_path = Path(artifact["path"])
    path_basis = artifact.get("path_basis")
    if path_basis == "CONTRACT_DIRECTORY_RELATIVE":
        if artifact_path.is_absolute():
            raise ValueError(
                "relative weight-window artifact path basis contains an "
                "absolute path"
            )
        artifact_path = (contract_source.parent / artifact_path).resolve()
        try:
            artifact_path.relative_to(contract_source.parent)
        except ValueError as exc:
            raise ValueError(
                "relative weight-window artifact path escapes the contract "
                "directory"
            ) from exc
    elif path_basis == "ABSOLUTE_EXTERNAL":
        if not artifact_path.is_absolute():
            raise ValueError(
                "external weight-window artifact path basis requires an "
                "absolute path"
            )
        artifact_path = artifact_path.resolve()
    else:
        raise ValueError("weight-window artifact path basis is invalid")
    if (
        not artifact_path.is_file()
        or artifact_path.stat().st_size != int(artifact["size_bytes"])
        or _sha256(artifact_path) != artifact["sha256"]
    ):
        raise ValueError("weight-window artifact hash mismatch")
    return {
        **value,
        "resolved_weight_window_artifact_path": str(artifact_path.resolve()),
    }


def weight_window_contract_from_mapping(
    value: Mapping[str, Any],
) -> WeightWindowArtifactContract:
    """Reconstruct a typed contract from a prepared or finalized manifest."""
    fields = {
        name: value[name]
        for name in WeightWindowArtifactContract.__dataclass_fields__
    }
    fields["energy_bounds_eV"] = tuple(fields["energy_bounds_eV"])
    fields["selected_magnet_ids"] = tuple(fields["selected_magnet_ids"])
    return WeightWindowArtifactContract(**fields)


def configure_magic_generator(
    settings,
    *,
    mesh_path: str | Path,
    energy_bounds_eV: Sequence[float],
    particle_type: str,
    batches: int,
    update_interval: int = 1,
    max_history_splits: int,
):
    """Configure the exact OpenMC 0.16 API after pure contract validation."""
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC 0.16 is required to configure MAGIC"
        ) from exc
    if tuple(
        int(value) for value in openmc.__version__.split("+")[0].split(".")[:2]
    ) < (0, 16):
        raise RuntimeError("OpenMC 0.16 or newer is required")
    edges = validate_weight_window_energy_grid(
        energy_bounds_eV,
        transport_min_eV=float(energy_bounds_eV[0]),
        transport_max_eV=float(energy_bounds_eV[-1]),
    )
    if batches <= 0 or update_interval <= 0:
        raise ValueError("MAGIC batch controls must be positive")
    validate_split_controls(
        max_history_splits=max_history_splits, max_split=10
    )
    mesh = openmc.UnstructuredMesh(str(Path(mesh_path).resolve()), "moab")
    generator = openmc.WeightWindowGenerator(
        mesh=mesh,
        energy_bounds=edges,
        particle_type=particle_type,
        method="magic",
        max_realizations=int(batches),
        update_interval=int(update_interval),
        on_the_fly=True,
    )
    settings.weight_windows_on = True
    settings.weight_window_generators = [generator]
    settings.weight_window_checkpoints = {"collision": True, "surface": True}
    settings.max_history_splits = int(max_history_splits)
    return generator


def validate_weight_window_hdf5(
    path: str | Path,
    *,
    expected_contract: WeightWindowArtifactContract | None = None,
) -> dict[str, Any]:
    """Validate semantic bounds and mesh coverage in an OpenMC WW artifact."""
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC 0.16 is required to validate a weight-window artifact"
        ) from exc
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    windows = openmc.WeightWindowsList.from_hdf5(source)
    if len(windows) != 1:
        raise ValueError("expected exactly one weight-window definition")
    window = windows[0]
    particle = getattr(window.particle_type, "name", str(window.particle_type))
    particle = str(particle).lower()
    if particle not in {"neutron", "photon"}:
        raise ValueError("weight-window artifact particle type is unsupported")
    edges = np.asarray(window.energy_bounds, dtype=float)
    lower = np.asarray(window.lower_ww_bounds, dtype=float)
    upper = np.asarray(window.upper_ww_bounds, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("weight-window artifact has invalid energy bounds")
    if lower.shape != upper.shape or lower.size == 0:
        raise ValueError("weight-window lower/upper bounds are misaligned")
    if (
        np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(lower < 0.0)
        or np.any(upper < lower)
    ):
        raise ValueError("weight-window artifact contains invalid bounds")
    nonzero = int(np.count_nonzero(lower > 0.0))
    if nonzero == 0:
        raise ValueError(
            "weight-window artifact has zero populated lower bounds"
        )
    mesh_elements = int(window.mesh.n_elements)
    if lower.size != mesh_elements * (len(edges) - 1):
        raise ValueError(
            "weight-window artifact bounds do not cover every mesh/energy bin"
        )
    max_split = int(window.max_split)
    survival_ratio = float(window.survival_ratio)
    max_lower_bound_ratio = (
        None
        if window.max_lower_bound_ratio is None
        else float(window.max_lower_bound_ratio)
    )
    weight_cutoff = float(window.weight_cutoff)
    if max_split <= 1:
        raise ValueError("weight-window artifact has invalid max_split")
    if not np.isfinite(survival_ratio) or survival_ratio <= 1.0:
        raise ValueError("weight-window artifact has invalid survival_ratio")
    if max_lower_bound_ratio is not None and (
        not np.isfinite(max_lower_bound_ratio) or max_lower_bound_ratio < 1.0
    ):
        raise ValueError(
            "weight-window artifact has invalid max_lower_bound_ratio"
        )
    if not np.isfinite(weight_cutoff) or weight_cutoff <= 0.0:
        raise ValueError("weight-window artifact has invalid weight_cutoff")
    if expected_contract is not None:
        if particle != expected_contract.particle_type:
            raise ValueError("weight-window artifact particle type is stale")
        if not np.array_equal(
            edges, np.asarray(expected_contract.energy_bounds_eV, dtype=float)
        ):
            raise ValueError("weight-window artifact energy grid is stale")
        actual_controls = {
            "max_split": max_split,
            "survival_ratio": survival_ratio,
            "max_lower_bound_ratio": max_lower_bound_ratio,
            "weight_cutoff": weight_cutoff,
        }
        for name, actual in actual_controls.items():
            if name not in expected_contract.generator_settings:
                continue
            configured = expected_contract.generator_settings[name]
            if configured is None or actual is None:
                matches = configured is None and actual is None
            elif name == "max_split":
                matches = int(configured) == actual
            else:
                matches = bool(
                    np.isclose(
                        float(configured), float(actual), rtol=0.0, atol=0.0
                    )
                )
            if not matches:
                raise ValueError(
                    f"weight-window artifact {name} control is stale"
                )
    return {
        "status": "PASS",
        "particle_type": particle,
        "energy_bounds_eV": edges.tolist(),
        "energy_bin_count": len(edges) - 1,
        "mesh_element_count": mesh_elements,
        "weight_window_bin_count": int(lower.size),
        "nonzero_lower_bound_count": nonzero,
        "finite_nonnegative_bounds": True,
        "max_split": max_split,
        "survival_ratio": survival_ratio,
        "max_lower_bound_ratio": max_lower_bound_ratio,
        "weight_cutoff": weight_cutoff,
    }


def figure_of_merit(
    mean: float, std_dev: float, wall_time_s: float
) -> float | None:
    if wall_time_s <= 0.0 or std_dev < 0.0:
        raise ValueError(
            "FOM requires positive time and nonnegative uncertainty"
        )
    if mean <= 0.0 or std_dev <= 0.0:
        return None
    relative = std_dev / mean
    return float(1.0 / (relative * relative * wall_time_s))


def aggregate_weight_window_campaign_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_seed_count: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate independent seed estimates without hiding between-run noise."""
    if minimum_seed_count < 3:
        raise ValueError("WW qualification requires at least three seeds")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        variant = str(row.get("variant", ""))
        response_id = str(row.get("response_id", ""))
        if variant not in {"unbiased", "weight_window"} or not response_id:
            raise ValueError("campaign row has invalid variant or response ID")
        grouped.setdefault((variant, response_id), []).append(row)
    variants = {
        variant: {
            response_id
            for group_variant, response_id in grouped
            if group_variant == variant
        }
        for variant in ("unbiased", "weight_window")
    }
    if (
        not variants["unbiased"]
        or variants["unbiased"] != variants["weight_window"]
    ):
        raise ValueError(
            "campaign variants have mismatched response inventories"
        )
    seed_inventory: dict[tuple[str, str], tuple[int, ...]] = {}
    for (variant, response_id), group in grouped.items():
        seeds = tuple(sorted(int(row["seed"]) for row in group))
        if (
            len(seeds) < minimum_seed_count
            or len(set(seeds)) != len(seeds)
            or min(seeds) <= 0
        ):
            raise ValueError(
                f"{variant} response {response_id!r} lacks distinct positive seeds"
            )
        seed_inventory[(variant, response_id)] = seeds
    for variant, response_ids in variants.items():
        seed_sets = {
            seed_inventory[(variant, response_id)]
            for response_id in response_ids
        }
        if len(seed_sets) != 1:
            raise ValueError(
                f"{variant} responses do not share one seed inventory"
            )
    for response_id in variants["unbiased"]:
        if (
            seed_inventory[("unbiased", response_id)]
            != seed_inventory[("weight_window", response_id)]
        ):
            raise ValueError(
                f"response {response_id!r} does not use paired seed inventories"
            )
    result = {"unbiased": [], "weight_window": []}
    for variant, aggregated_rows in result.items():
        for response_id in sorted(variants[variant]):
            group = grouped[(variant, response_id)]
            seeds = [int(row["seed"]) for row in group]
            if len(seeds) < minimum_seed_count or len(set(seeds)) != len(
                seeds
            ):
                raise ValueError(
                    f"{variant} response {response_id!r} lacks distinct seeds"
                )
            definition_hashes = {str(row["definition_hash"]) for row in group}
            run_contracts = {str(row["run_contract_sha256"]) for row in group}
            if len(definition_hashes) != 1 or len(run_contracts) != 1:
                raise ValueError(
                    f"{variant} response {response_id!r} changed definition"
                )
            definition_hash = next(iter(definition_hashes))
            run_contract = next(iter(run_contracts))
            if any(
                len(value) != 64
                or any(
                    character not in "0123456789abcdef" for character in value
                )
                for value in (definition_hash, run_contract)
            ):
                raise ValueError("campaign definition hashes must be SHA-256")
            identities = {
                (
                    str(row["particle"]),
                    bool(row.get("primary", False)),
                    bool(row.get("critical", False)),
                )
                for row in group
            }
            if len(identities) != 1 or next(iter(identities))[0] not in {
                "neutron",
                "photon",
            }:
                raise ValueError(
                    f"{variant} response {response_id!r} changed identity"
                )
            estimates = np.asarray(
                [row["estimate"] for row in group], dtype=float
            )
            within = np.asarray(
                [row["within_run_std_dev"] for row in group], dtype=float
            )
            wall = np.asarray(
                [row["wall_time_s"] for row in group], dtype=float
            )
            if (
                np.any(~np.isfinite(estimates))
                or np.any(~np.isfinite(within))
                or np.any(within < 0.0)
                or np.any(~np.isfinite(wall))
                or np.any(wall <= 0.0)
            ):
                raise ValueError(
                    "campaign estimates, uncertainties, and times are invalid"
                )
            observed_sem = float(
                np.std(estimates, ddof=1) / np.sqrt(len(group))
            )
            propagated_sem = float(
                np.sqrt(np.sum(within * within)) / len(group)
            )
            conservative_sem = max(observed_sem, propagated_sem)
            template = group[0]
            aggregate = {
                "response_id": response_id,
                "definition_hash": definition_hash,
                "run_contract_sha256": run_contract,
                "particle": str(template["particle"]),
                "primary": bool(template.get("primary", False)),
                "critical": bool(template.get("critical", False)),
                "mean": float(np.mean(estimates)),
                "std_dev": conservative_sem,
                "observed_between_seed_standard_error": observed_sem,
                "propagated_within_run_standard_error": propagated_sem,
                "uncertainty_combination": "maximum_of_between_seed_and_propagated_within_run_sem",
                "wall_time_s": float(np.sum(wall)),
                "seed_count": len(group),
                "seeds": sorted(seeds),
                "individual_seed_rows": group,
            }
            ess_values = [row.get("effective_sample_size") for row in group]
            if all(value is not None for value in ess_values):
                effective_sample_sizes = np.asarray(ess_values, dtype=float)
                if np.any(~np.isfinite(effective_sample_sizes)) or np.any(
                    effective_sample_sizes < 0.0
                ):
                    raise ValueError(
                        "campaign effective sample sizes are invalid"
                    )
                aggregate["effective_sample_size"] = float(
                    np.sum(effective_sample_sizes)
                )
            aggregated_rows.append(aggregate)
    return result


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = count - reverse_rank + 1
        running = min(running, values[index] * count / rank)
        adjusted[index] = running
    return np.clip(adjusted, 0.0, 1.0)


def qualify_weight_windows(
    unbiased: Sequence[Mapping[str, Any]],
    weight_window: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.05,
    minimum_geometric_mean_fom_ratio: float = 2.0,
    minimum_improved_fraction: float = 0.75,
    minimum_critical_fom_ratio: float = 0.8,
    minimum_seed_count: int = 3,
    run_diagnostics: Mapping[str, Any] | None = None,
    selected_magnets_only: bool = False,
    artifact_particle_type: str = "neutron",
) -> dict[str, Any]:
    """Apply a predeclared bias/FOM policy with BH multiple-test correction."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("qualification alpha must be in (0, 1)")
    if artifact_particle_type not in {"neutron", "photon"}:
        raise ValueError("WW artifact particle type must be neutron or photon")
    unbiased_rows = [dict(row) for row in unbiased]
    weighted_rows = [dict(row) for row in weight_window]
    left_ids = [str(row.get("response_id", "")) for row in unbiased_rows]
    right_ids = [str(row.get("response_id", "")) for row in weighted_rows]
    if (
        any(not value for value in left_ids + right_ids)
        or len(set(left_ids)) != len(left_ids)
        or len(set(right_ids)) != len(right_ids)
    ):
        raise ValueError(
            "qualification response IDs must be nonempty and unique"
        )
    left = dict(zip(left_ids, unbiased_rows))
    right = dict(zip(right_ids, weighted_rows))
    if not left or set(left) != set(right):
        raise ValueError(
            "unbiased and WW response inventories must match and be nonempty"
        )
    comparisons = []
    p_values = []
    run_contracts = set()
    for response_id in sorted(left):
        baseline, candidate = left[response_id], right[response_id]
        definition_hash = str(baseline.get("definition_hash", ""))
        if definition_hash != str(candidate.get("definition_hash", "")):
            raise ValueError(f"tally definition changed for {response_id}")
        if len(definition_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in definition_hash
        ):
            raise ValueError(
                f"response {response_id!r} has an invalid definition hash"
            )
        particle = str(baseline.get("particle", ""))
        if particle not in {"neutron", "photon"} or particle != str(
            candidate.get("particle", "")
        ):
            raise ValueError(
                f"response {response_id!r} changed particle identity"
            )
        for flag in ("primary", "critical"):
            if bool(baseline.get(flag, False)) != bool(
                candidate.get(flag, False)
            ):
                raise ValueError(
                    f"response {response_id!r} changed {flag} status"
                )
        mean_u, mean_w = float(baseline["mean"]), float(candidate["mean"])
        std_u, std_w = float(baseline["std_dev"]), float(candidate["std_dev"])
        if (
            not all(
                np.isfinite(value) for value in (mean_u, mean_w, std_u, std_w)
            )
            or std_u < 0.0
            or std_w < 0.0
        ):
            raise ValueError(
                f"response {response_id!r} has invalid statistics"
            )
        run_contract = str(baseline.get("run_contract_sha256", ""))
        if run_contract != str(candidate.get("run_contract_sha256", "")):
            raise ValueError(f"run contract changed for {response_id}")
        if len(run_contract) != 64 or any(
            character not in "0123456789abcdef" for character in run_contract
        ):
            raise ValueError(
                f"response {response_id!r} has an invalid run contract"
            )
        run_contracts.add(run_contract)
        combined = math.hypot(std_u, std_w)
        difference = mean_w - mean_u
        z_score = difference / combined if combined > 0.0 else 0.0
        p_value = (
            math.erfc(abs(z_score) / math.sqrt(2.0)) if combined > 0.0 else 1.0
        )
        fom_u = figure_of_merit(mean_u, std_u, float(baseline["wall_time_s"]))
        fom_w = figure_of_merit(mean_w, std_w, float(candidate["wall_time_s"]))
        fom_ratio = (
            fom_w / fom_u
            if fom_u is not None and fom_w is not None and fom_u > 0.0
            else None
        )
        comparisons.append(
            {
                "response_id": response_id,
                "particle": particle,
                "primary": bool(baseline.get("primary", False)),
                "critical": bool(baseline.get("critical", False)),
                "unbiased_mean": mean_u,
                "unbiased_std_dev": std_u,
                "weight_window_mean": mean_w,
                "weight_window_std_dev": std_w,
                "difference": difference,
                "combined_uncertainty": combined,
                "z_score": z_score,
                "p_value": p_value,
                "fom_unbiased": fom_u,
                "fom_weight_window": fom_w,
                "fom_ratio": fom_ratio,
                "ess_ratio": (
                    float(candidate.get("effective_sample_size", 0.0))
                    / float(baseline.get("effective_sample_size", 0.0))
                    if float(baseline.get("effective_sample_size", 0.0)) > 0.0
                    else None
                ),
                "unbiased_seed_count": int(baseline.get("seed_count", 1)),
                "weight_window_seed_count": int(
                    candidate.get("seed_count", 1)
                ),
            }
        )
        p_values.append(p_value)
    if len(run_contracts) != 1:
        raise ValueError("qualification responses changed run contract")
    adjusted = _benjamini_hochberg(p_values)
    for row, p_adjusted in zip(comparisons, adjusted):
        row["p_value_bh_adjusted"] = float(p_adjusted)
        row["statistically_significant_bias"] = bool(p_adjusted < alpha)
    diagnostics = dict(run_diagnostics or {})
    instability = any(
        float(diagnostics.get(name, 0.0)) > 0.0
        for name in (
            "lost_particles",
            "dagmc_navigation_failures",
            "runaway_histories",
        )
    ) or bool(diagnostics.get("pathological_split_behavior", False))
    particle_decisions = {}
    for particle in sorted({row["particle"] for row in comparisons}):
        primary = [
            row
            for row in comparisons
            if row["particle"] == particle and row["primary"]
        ]
        if not primary:
            particle_decisions[particle] = {
                "classification": "INSUFFICIENT_PILOT_STATISTICS"
            }
            continue
        insufficient = any(
            min(row["unbiased_seed_count"], row["weight_window_seed_count"])
            < minimum_seed_count
            for row in primary
        )
        ratios = [
            row["fom_ratio"] for row in primary if row["fom_ratio"] is not None
        ]
        geometric = (
            float(np.exp(np.mean(np.log(ratios))))
            if len(ratios) == len(primary)
            else None
        )
        improved = (
            sum(value > 1.0 for value in ratios) / len(primary)
            if primary
            else 0.0
        )
        critical_bad = any(
            row["critical"]
            and row["fom_ratio"] is not None
            and row["fom_ratio"] < minimum_critical_fom_ratio
            for row in primary
        )
        bias = any(row["statistically_significant_bias"] for row in primary)
        if instability:
            classification = "REJECTED_INSTABILITY"
        elif insufficient:
            classification = "INSUFFICIENT_PILOT_STATISTICS"
        elif bias:
            classification = "REJECTED_BIAS"
        elif (
            geometric is not None
            and geometric >= minimum_geometric_mean_fom_ratio
            and improved >= minimum_improved_fraction
            and not critical_bad
        ):
            classification = "QUALIFIED_AND_ENABLED"
        else:
            classification = "NO_MATERIAL_BENEFIT_DISABLE"
        particle_decisions[particle] = {
            "classification": classification,
            "geometric_mean_primary_fom_ratio": geometric,
            "primary_improved_fraction": improved,
            "critical_response_below_floor": critical_bad,
        }
    qualified = [
        particle
        for particle, value in particle_decisions.items()
        if value["classification"] == "QUALIFIED_AND_ENABLED"
    ]
    classifications = {
        value["classification"] for value in particle_decisions.values()
    }
    if instability:
        overall = "REJECTED_INSTABILITY"
    elif "REJECTED_BIAS" in classifications:
        overall = "REJECTED_BIAS"
    elif "INSUFFICIENT_PILOT_STATISTICS" in classifications:
        overall = "INSUFFICIENT_PILOT_STATISTICS"
    elif (
        artifact_particle_type == "neutron"
        and "neutron" in qualified
        and "photon" in particle_decisions
        and "photon" not in qualified
    ):
        overall = "QUALIFIED_FOR_NEUTRONS_ONLY"
    elif artifact_particle_type in qualified:
        overall = (
            "QUALIFIED_FOR_SELECTED_MAGNETS_ONLY"
            if selected_magnets_only
            else "QUALIFIED_AND_ENABLED"
        )
    else:
        overall = "NO_MATERIAL_BENEFIT_DISABLE"
    return {
        "schema": QUALIFICATION_SCHEMA,
        "classification": overall,
        "weight_windows_enabled": overall.startswith("QUALIFIED"),
        "particle_decisions": particle_decisions,
        "comparisons": comparisons,
        "multiple_comparison_method": "Benjamini-Hochberg",
        "alpha": alpha,
        "artifact_particle_type": artifact_particle_type,
        "run_contract_sha256": run_contracts.pop(),
        "policy": {
            "minimum_geometric_mean_fom_ratio": minimum_geometric_mean_fom_ratio,
            "minimum_improved_fraction": minimum_improved_fraction,
            "minimum_critical_fom_ratio": minimum_critical_fom_ratio,
            "minimum_seed_count": minimum_seed_count,
            "zero_lost_particles_required": True,
        },
        "run_diagnostics": diagnostics,
    }


def weight_window_disabled_fallback(
    reason: str,
    *,
    classification: str = "NO_MATERIAL_BENEFIT_DISABLE",
    evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if not reason:
        raise ValueError("disabled weight-window fallback requires a reason")
    if classification not in {
        "NO_MATERIAL_BENEFIT_DISABLE",
        "REJECTED_BIAS",
        "REJECTED_INSTABILITY",
        "INSUFFICIENT_PILOT_STATISTICS",
    }:
        raise ValueError("unsupported disabled weight-window classification")
    return {
        "schema": QUALIFICATION_SCHEMA,
        "classification": classification,
        "weight_windows_enabled": False,
        "production_transport": "UNBIASED",
        "reason": reason,
        "evidence": [dict(value) for value in evidence],
    }
