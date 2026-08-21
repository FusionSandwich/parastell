"""Native DAGMC and discrete-PLC geometry for surface-anchored ports.

This module deliberately consumes only continuous in-vessel surfaces and the
verified point-cloud aperture loops. CadQuery solids are neither inspected nor
used as fallback geometry.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha1
import json
from pathlib import Path

import numpy as np
import gmsh
import pydagmc
from pymoab import core, types

from .port_aperture import build_aperture_loops


COORDINATE_DIGITS = 9


def _coordinate_key(point):
    return tuple(np.round(np.asarray(point, dtype=float), COORDINATE_DIGITS))


def _facet_key(triangle):
    return tuple(sorted(_coordinate_key(point) for point in triangle))


def _triangle_area(triangle):
    triangle = np.asarray(triangle, dtype=float)
    return float(
        np.linalg.norm(
            np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        )
        / 2.0
    )


def _orient_triangle(triangle, desired):
    triangle = np.asarray(triangle, dtype=float)
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    if np.dot(normal, desired) < 0.0:
        return triangle[[0, 2, 1]]
    return triangle


def _disk_triangles(points, desired):
    points = np.asarray(points, dtype=float)[:-1]
    center = points.mean(axis=0)
    triangles = []
    for index in range(len(points)):
        triangle = np.asarray(
            (center, points[index], points[(index + 1) % len(points)])
        )
        triangles.append(_orient_triangle(triangle, desired))
    return np.asarray(triangles)


def _annulus_triangles(inner_points, outer_points, desired):
    inner = np.asarray(inner_points, dtype=float)[:-1]
    outer = np.asarray(outer_points, dtype=float)[:-1]
    if len(inner) != len(outer):
        raise ValueError("Aperture annulus loops have different point counts")
    triangles = []
    for index in range(len(inner)):
        following = (index + 1) % len(inner)
        triangles.extend(
            (
                _orient_triangle(
                    (inner[index], outer[index], outer[following]), desired
                ),
                _orient_triangle(
                    (inner[index], outer[following], inner[following]), desired
                ),
            )
        )
    return np.asarray(triangles)


def _sidewall_triangles(left_points, right_points, anchor, reference, normal):
    left = np.asarray(left_points, dtype=float)[:-1]
    right = np.asarray(right_points, dtype=float)[:-1]
    if len(left) != len(right):
        raise ValueError("Corresponding aperture loops have different counts")
    triangles = []
    for index in range(len(left)):
        following = (index + 1) % len(left)
        candidates = (
            np.asarray((left[index], left[following], right[following])),
            np.asarray((left[index], right[following], right[index])),
        )
        for triangle in candidates:
            center = triangle.mean(axis=0) - anchor
            desired = (
                np.dot(center, reference) * reference
                + np.dot(center, normal) * normal
            )
            triangles.append(_orient_triangle(triangle, desired))
    return np.asarray(triangles)


@dataclass(frozen=True)
class NativeSurfaceRecord:
    name: str
    kind: str
    triangles: np.ndarray
    reverse_volume: str | None
    forward_volume: str | None

    @property
    def triangle_count(self):
        return len(self.triangles)


@dataclass(frozen=True)
class NativeVolumeRecord:
    name: str
    kind: str
    material_tag: str


@dataclass(frozen=True)
class NativeValidationResult:
    volume_count: int
    surface_count: int
    triangle_count: int
    duplicate_facet_count: int
    zero_area_triangle_count: int
    orphan_triangle_count: int
    unreferenced_vertex_count: int
    unsealed_edge_count: int
    internal_surface_count: int
    boundary_surface_count: int
    centerline_region_sequence: tuple[str, ...]
    liner_region_sequence: tuple[str, ...]
    blanket_region_sequence: tuple[str, ...]
    volume_ids: dict[str, int]
    surface_ids: dict[str, int]

    def to_dict(self):
        result = dict(self.__dict__)
        for key in (
            "centerline_region_sequence",
            "liner_region_sequence",
            "blanket_region_sequence",
        ):
            result[key] = list(result[key])
        return result


@dataclass(frozen=True)
class NativeVolumeMeshResult:
    region_tetrahedron_counts: dict[str, int]
    region_minimum_tetrahedron_volume: dict[str, float]
    region_maximum_tetrahedron_volume: dict[str, float]
    region_total_tetrahedron_volume: dict[str, float]
    tetrahedron_count: int
    inverted_tetrahedron_count: int
    zero_volume_tetrahedron_count: int
    duplicate_tetrahedron_count: int
    nonconformal_interface_face_count: int
    disconnected_region_count: int
    region_reference_volume: dict[str, float]
    region_relative_volume_error: dict[str, float]

    def to_dict(self):
        return dict(self.__dict__)


class NativePortSurfaceComplex:
    """One conformal surface ledger with no coincident duplicate facets."""

    def __init__(self, model, port, loops, surfaces, volumes, radial_data):
        self.source_model = model
        self.port = port
        self.loops = tuple(loops)
        self.surfaces = tuple(surfaces)
        self.volumes = tuple(volumes)
        self.radial_data = radial_data
        self.dag_model = None
        self.volume_ids = {}
        self.surface_ids = {}

    @property
    def triangle_count(self):
        return sum(surface.triangle_count for surface in self.surfaces)

    def _facet_counts(self):
        return Counter(
            _facet_key(triangle)
            for surface in self.surfaces
            for triangle in surface.triangles
        )

    def duplicate_facets(self):
        return {
            key: count
            for key, count in self._facet_counts().items()
            if count > 1
        }

    def zero_area_triangles(self, tolerance=1e-12):
        return [
            (surface.name, index)
            for surface in self.surfaces
            for index, triangle in enumerate(surface.triangles)
            if _triangle_area(triangle) <= tolerance
        ]

    def _volume_edge_counts(self, volume_name):
        counts = Counter()
        for surface in self.surfaces:
            if volume_name not in {
                surface.reverse_volume,
                surface.forward_volume,
            }:
                continue
            for triangle in surface.triangles:
                keys = [_coordinate_key(point) for point in triangle]
                for left, right in ((0, 1), (1, 2), (2, 0)):
                    counts[tuple(sorted((keys[left], keys[right])))] += 1
        return counts

    def unsealed_edges(self):
        failures = {}
        for volume in self.volumes:
            bad = {
                edge: count
                for edge, count in self._volume_edge_counts(
                    volume.name
                ).items()
                if count != 2
            }
            if bad:
                failures[volume.name] = bad
        return failures

    def reference_volumes(self):
        """Return closed-PLC volumes using each surface's DAGMC sense."""
        result = {}
        for volume in self.volumes:
            signed = 0.0
            for surface in self.surfaces:
                if volume.name == surface.reverse_volume:
                    triangles = surface.triangles
                elif volume.name == surface.forward_volume:
                    triangles = surface.triangles[:, [0, 2, 1]]
                else:
                    continue
                signed += float(
                    np.sum(
                        np.einsum(
                            "ij,ij->i",
                            triangles[:, 0],
                            np.cross(triangles[:, 1], triangles[:, 2]),
                        )
                    )
                    / 6.0
                )
            if signed <= 0.0:
                raise ValueError(
                    f"Volume {volume.name!r} has reversed surface senses"
                )
            result[volume.name] = signed
        return result

    @staticmethod
    def _set_component_name_tag(mb, entity, name):
        if len(name.encode("utf-8")) >= types.NAME_TAG_SIZE:
            digest = sha1(name.encode("utf-8")).hexdigest()[:8]
            name = f"{name[:22]}_{digest}"
        tag = mb.tag_get_handle(
            "PARASTELL_NAME",
            types.NAME_TAG_SIZE,
            types.MB_TYPE_OPAQUE,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )
        value = np.asarray(
            [name.encode("utf-8")], dtype=f"S{types.NAME_TAG_SIZE}"
        )
        mb.tag_set_data(tag, [entity], value)

    def to_pydagmc(self):
        duplicates = self.duplicate_facets()
        if duplicates:
            raise ValueError(
                f"Native surface complex has {len(duplicates)} duplicate facets"
            )
        zero_area = self.zero_area_triangles()
        if zero_area:
            raise ValueError(
                f"Native surface complex has {len(zero_area)} zero-area facets"
            )
        unsealed = self.unsealed_edges()
        if unsealed:
            summary = {name: len(edges) for name, edges in unsealed.items()}
            raise ValueError(f"Native volume shells are not closed: {summary}")

        mb = core.Core()
        dag_model = pydagmc.Model(mb)
        volumes = {}
        for global_id, record in enumerate(self.volumes, start=1):
            volume = dag_model.create_volume(global_id=global_id)
            volumes[record.name] = volume
            self._set_component_name_tag(mb, volume.handle, record.name)
            material = pydagmc.Group.create(
                dag_model, name=f"mat:{record.material_tag}"
            )
            material.add_set(volume)
            component = pydagmc.Group.create(
                dag_model, name=f"component:{record.name}"
            )
            component.add_set(volume)

        coordinate_map = {}
        coordinates = []
        surface_connectivity = []
        for surface in self.surfaces:
            connectivity = []
            for triangle in surface.triangles:
                indices = []
                for point in triangle:
                    key = _coordinate_key(point)
                    if key not in coordinate_map:
                        coordinate_map[key] = len(coordinates)
                        coordinates.append(np.asarray(point, dtype=float))
                    indices.append(coordinate_map[key])
                connectivity.append(indices)
            surface_connectivity.append(np.asarray(connectivity, dtype=int))
        vertex_handles = mb.create_vertices(np.asarray(coordinates).ravel())
        vertex_handles = np.asarray(list(vertex_handles), dtype=np.uint64)

        surfaces = {}
        for global_id, (record, connectivity) in enumerate(
            zip(self.surfaces, surface_connectivity), start=1
        ):
            dag_surface = dag_model.create_surface(global_id=global_id)
            element_connectivity = vertex_handles[connectivity]
            triangle_handles = mb.create_elements(
                types.MBTRI, element_connectivity
            )
            mb.add_entities(dag_surface.handle, triangle_handles)
            dag_surface.senses = [
                volumes.get(record.reverse_volume),
                volumes.get(record.forward_volume),
            ]
            self._set_component_name_tag(mb, dag_surface.handle, record.name)
            surfaces[record.name] = dag_surface

        self.dag_model = dag_model
        self.volume_ids = {
            name: int(volume.id) for name, volume in volumes.items()
        }
        self.surface_ids = {
            name: int(surface.id) for name, surface in surfaces.items()
        }
        return dag_model

    def write_dagmc(self, filename):
        if self.dag_model is None:
            self.to_pydagmc()
        filename = Path(filename).with_suffix(".h5m")
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.dag_model.write_file(str(filename))
        return filename

    @staticmethod
    def _ray_triangle(origin, direction, triangle, tolerance=1e-10):
        edge_1 = triangle[1] - triangle[0]
        edge_2 = triangle[2] - triangle[0]
        h = np.cross(direction, edge_2)
        determinant = np.dot(edge_1, h)
        if abs(determinant) <= tolerance:
            return None
        inverse = 1.0 / determinant
        displacement = origin - triangle[0]
        barycentric_u = inverse * np.dot(displacement, h)
        if barycentric_u < -tolerance or barycentric_u > 1.0 + tolerance:
            return None
        q = np.cross(displacement, edge_1)
        barycentric_v = inverse * np.dot(direction, q)
        if (
            barycentric_v < -tolerance
            or barycentric_u + barycentric_v > 1.0 + tolerance
        ):
            return None
        distance = inverse * np.dot(edge_2, q)
        return float(distance) if distance > tolerance else None

    def ray_region_sequence(self, origin, direction, initial_region):
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        hits = []
        for surface in self.surfaces:
            distances = []
            for triangle in surface.triangles:
                distance = self._ray_triangle(origin, direction, triangle)
                if distance is not None:
                    distances.append(distance)
            for distance in sorted(distances):
                if not hits or abs(distance - hits[-1][0]) > 1e-7:
                    hits.append((distance, surface))
        hits.sort(key=lambda item: item[0])
        unique_hits = []
        for distance, surface in hits:
            if unique_hits and abs(distance - unique_hits[-1][0]) <= 1e-6:
                if surface.name != unique_hits[-1][1].name:
                    raise ValueError(
                        "Ray encounters multiple physical surfaces at one point: "
                        f"{unique_hits[-1][1].name!r}, {surface.name!r}"
                    )
                continue
            unique_hits.append((distance, surface))
        sequence = [initial_region]
        current = initial_region
        for _, surface in unique_hits:
            if current == surface.reverse_volume:
                current = surface.forward_volume or "external"
            elif current == surface.forward_volume:
                current = surface.reverse_volume or "external"
            else:
                continue
            if sequence[-1] != current:
                sequence.append(current)
        return tuple(sequence)

    def validate(self):
        duplicates = self.duplicate_facets()
        zero_area = self.zero_area_triangles()
        unsealed = self.unsealed_edges()
        if duplicates or zero_area or unsealed:
            raise ValueError(
                "Native validation failed before export: "
                f"duplicates={len(duplicates)}, zero_area={len(zero_area)}, "
                f"unsealed={sum(len(value) for value in unsealed.values())}"
            )
        port = self.port
        anchor = np.asarray(port.placement.anchor)
        axis = np.asarray(port.placement.local_axis)
        reference = np.asarray(port.placement.local_reference)
        result = self.radial_data["port_result"]
        origin = anchor - axis * 2.0
        centerline = self.ray_region_sequence(origin, axis, "plasma")
        if port.liner.enabled:
            liner_offset = (
                port.cross_section.radius + port.liner.thickness / 2.0
                if port.cross_section.shape == "circle"
                else port.cross_section.width / 2.0
                + port.liner.thickness / 2.0
            )
            liner = self.ray_region_sequence(
                origin + reference * liner_offset, axis, "plasma"
            )
        else:
            liner = ()
        blanket_offset = (
            self.source_model._port_aperture_half_width(port) + 2.0
        )
        blanket = self.ray_region_sequence(
            origin + reference * blanket_offset, axis, "plasma"
        )
        internal = sum(
            surface.reverse_volume is not None
            and surface.forward_volume is not None
            for surface in self.surfaces
        )
        boundary = len(self.surfaces) - internal
        return NativeValidationResult(
            len(self.volumes),
            len(self.surfaces),
            self.triangle_count,
            len(duplicates),
            len(zero_area),
            0,
            0,
            0,
            internal,
            boundary,
            centerline,
            liner,
            blanket,
            dict(self.volume_ids),
            dict(self.surface_ids),
        )

    def write_validation(self, filename, result=None):
        result = self.validate() if result is None else result
        path = Path(filename)
        path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        return path

    def validate_dagmc_file(self, filename):
        """Reload and independently audit a native DAGMC H5M file."""
        filename = str(filename)
        mb = core.Core()
        mb.load_file(filename)
        loaded = pydagmc.Model(filename)
        root = mb.get_root_set()
        dimension = mb.tag_get_handle("GEOM_DIMENSION")
        global_id = mb.tag_get_handle("GLOBAL_ID")
        category = mb.tag_get_handle("CATEGORY")
        parastell_name = mb.tag_get_handle("PARASTELL_NAME")
        senses = mb.tag_get_handle("GEOM_SENSE_2")
        meshsets = mb.get_entities_by_type(root, types.MBENTITYSET)
        geometry = []
        for meshset in meshsets:
            try:
                dim = int(mb.tag_get_data(dimension, [meshset], flat=True)[0])
            except RuntimeError:
                continue
            if dim in (2, 3):
                geometry.append((meshset, dim))
                # These accesses are intentionally strict: missing mandatory
                # DAGMC metadata is a validation failure.
                mb.tag_get_data(global_id, [meshset], flat=True)
                mb.tag_get_data(category, [meshset], flat=True)
                mb.tag_get_data(parastell_name, [meshset], flat=True)
        volumes = [entity for entity, dim in geometry if dim == 3]
        surfaces = [entity for entity, dim in geometry if dim == 2]
        if len(volumes) != len(self.volumes):
            raise ValueError(
                f"H5M volume count changed: {len(volumes)} != {len(self.volumes)}"
            )
        if len(surfaces) != len(self.surfaces):
            raise ValueError(
                f"H5M surface count changed: {len(surfaces)} != {len(self.surfaces)}"
            )
        volume_set = set(volumes)
        surface_triangles = []
        for surface in surfaces:
            sense_handles = mb.tag_get_data(senses, [surface], flat=True)
            nonzero = [int(value) for value in sense_handles if int(value)]
            if not nonzero or any(
                value not in volume_set for value in nonzero
            ):
                raise ValueError("H5M surface has invalid volume senses")
            parents = set(mb.get_parent_meshsets(surface))
            if not set(nonzero).issubset(parents):
                raise ValueError("H5M sense volume is not a parent meshset")
            triangles = list(mb.get_entities_by_type(surface, types.MBTRI))
            if not triangles:
                raise ValueError("H5M physical surface has no triangles")
            surface_triangles.extend(triangles)
        if len(surface_triangles) != len(set(surface_triangles)):
            raise ValueError(
                "H5M contains orphaned or multiply-owned triangles"
            )
        all_triangles = set(mb.get_entities_by_type(root, types.MBTRI))
        if all_triangles != set(surface_triangles):
            raise ValueError("H5M contains orphan triangles")
        referenced = set()
        facet_keys = Counter()
        zero_area = 0
        for triangle in all_triangles:
            connectivity = list(mb.get_connectivity(triangle))
            referenced.update(connectivity)
            coordinates = mb.get_coords(connectivity).reshape((-1, 3))
            facet_keys[_facet_key(coordinates)] += 1
            zero_area += _triangle_area(coordinates) <= 1e-12
        all_vertices = set(mb.get_entities_by_type(root, types.MBVERTEX))
        duplicates = sum(
            count - 1 for count in facet_keys.values() if count > 1
        )
        if zero_area or duplicates or all_vertices != referenced:
            raise ValueError(
                "H5M facet audit failed: "
                f"zero_area={zero_area}, duplicates={duplicates}, "
                f"unreferenced_vertices={len(all_vertices - referenced)}"
            )
        return {
            "pymoab_load": True,
            "pydagmc_load": bool(loaded),
            "volume_count": len(volumes),
            "surface_count": len(surfaces),
            "triangle_count": len(all_triangles),
            "duplicate_facet_count": duplicates,
            "zero_area_triangle_count": zero_area,
            "orphan_triangle_count": len(
                all_triangles - set(surface_triangles)
            ),
            "unreferenced_vertex_count": len(all_vertices - referenced),
        }

    def tetrahedralize(self, min_mesh_size=5.0, max_mesh_size=25.0):
        """Tetrahedralize this exact discrete PLC without a CAD/OCC import."""
        if self.duplicate_facets() or self.zero_area_triangles():
            raise ValueError("Discrete PLC failed its facet audit")
        if self.unsealed_edges():
            raise ValueError("Discrete PLC has open volume shells")
        if gmsh.isInitialized():
            gmsh.finalize()
        gmsh.initialize()
        try:
            gmsh.model.add("parastell_native_port_plc")
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", min_mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh_size)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            coordinate_map = {}
            coordinates = []
            connectivities = []
            mesh_volumes = tuple(
                volume for volume in self.volumes if volume.kind != "graveyard"
            )
            mesh_volume_names = {volume.name for volume in mesh_volumes}
            plc_surfaces = tuple(
                surface
                for surface in self.surfaces
                if mesh_volume_names.intersection(
                    {surface.reverse_volume, surface.forward_volume}
                )
            )
            for surface in plc_surfaces:
                connectivity = []
                for triangle in surface.triangles:
                    indices = []
                    for point in triangle:
                        key = _coordinate_key(point)
                        if key not in coordinate_map:
                            coordinate_map[key] = len(coordinates) + 1
                            coordinates.append(np.asarray(point, dtype=float))
                        indices.append(coordinate_map[key])
                    connectivity.append(indices)
                connectivities.append(np.asarray(connectivity, dtype=np.int64))

            for surface_id in range(1, len(plc_surfaces) + 1):
                gmsh.model.addDiscreteEntity(2, surface_id)
            # Gmsh node tags are global. Classifying the shared PLC nodes on
            # one discrete surface keeps every interface vertex physically
            # identical; all other surface elements reference those same tags.
            node_tags = np.arange(1, len(coordinates) + 1, dtype=np.int64)
            gmsh.model.mesh.addNodes(
                2,
                1,
                node_tags,
                np.asarray(coordinates, dtype=float).ravel(),
            )
            next_element = 1
            for surface_id, connectivity in enumerate(connectivities, start=1):
                element_tags = np.arange(
                    next_element,
                    next_element + len(connectivity),
                    dtype=np.int64,
                )
                next_element += len(connectivity)
                gmsh.model.mesh.addElementsByType(
                    surface_id,
                    2,
                    element_tags,
                    connectivity.ravel(),
                )

            gmsh.model.mesh.reclassifyNodes()
            volume_entities = []
            for volume_id, volume in enumerate(mesh_volumes, start=1):
                boundary = [
                    surface_id
                    for surface_id, surface in enumerate(plc_surfaces, start=1)
                    if volume.name
                    in {surface.reverse_volume, surface.forward_volume}
                ]
                gmsh.model.addDiscreteEntity(3, volume_id, boundary)
                volume_entities.append((3, volume_id))
                physical_id = gmsh.model.addPhysicalGroup(3, [volume_id])
                gmsh.model.setPhysicalName(3, physical_id, volume.name)
            gmsh.model.mesh.createGeometry(volume_entities)
            gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
            gmsh.model.mesh.generate(3)
            nodes, node_coordinates, _ = gmsh.model.mesh.getNodes()
            coordinate_by_tag = dict(
                zip(
                    (int(tag) for tag in nodes),
                    np.asarray(node_coordinates).reshape((-1, 3)),
                )
            )
            regions = {}
            for volume_id, volume in enumerate(mesh_volumes, start=1):
                element_types, _, element_nodes = gmsh.model.mesh.getElements(
                    3, volume_id
                )
                tetrahedra = []
                for element_type, connectivity in zip(
                    element_types, element_nodes
                ):
                    _, _, _, node_count, _, _ = (
                        gmsh.model.mesh.getElementProperties(element_type)
                    )
                    if node_count != 4:
                        continue
                    tetrahedra.extend(
                        np.asarray(connectivity, dtype=np.int64).reshape(
                            (-1, 4)
                        )
                    )
                if not tetrahedra:
                    raise ValueError(
                        f"Discrete tetrahedralizer produced no elements for {volume.name!r}"
                    )
                regions[volume.name] = np.asarray(tetrahedra, dtype=np.int64)
            interface_faces = {
                surface.name: connectivity
                for surface, connectivity in zip(plc_surfaces, connectivities)
            }
            interface_senses = {
                surface.name: (
                    (
                        surface.reverse_volume
                        if surface.reverse_volume in mesh_volume_names
                        else None
                    ),
                    (
                        surface.forward_volume
                        if surface.forward_volume in mesh_volume_names
                        else None
                    ),
                )
                for surface in plc_surfaces
            }
            return NativeVolumeMesh(
                mesh_volumes,
                coordinate_by_tag,
                regions,
                interface_faces,
                interface_senses,
                {
                    name: volume
                    for name, volume in self.reference_volumes().items()
                    if name in mesh_volume_names
                },
            )
        finally:
            gmsh.clear()
            gmsh.finalize()


class NativeVolumeMesh:
    """Conformal tetrahedra generated from a NativePortSurfaceComplex PLC."""

    def __init__(
        self,
        volumes,
        coordinates,
        regions,
        interface_faces,
        interface_senses,
        reference_volumes,
    ):
        self.volumes = volumes
        self.coordinates = coordinates
        self.regions = regions
        self.interface_faces = interface_faces
        self.interface_senses = interface_senses
        self.reference_volumes = reference_volumes

    def _tetrahedron_points(self, connectivity):
        return np.asarray(
            [
                [self.coordinates[int(tag)] for tag in tet]
                for tet in connectivity
            ]
        )

    @staticmethod
    def _signed_volumes(points):
        return (
            np.einsum(
                "ij,ij->i",
                points[:, 1] - points[:, 0],
                np.cross(
                    points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]
                ),
            )
            / 6.0
        )

    def validate(self, tolerance=1e-12):
        counts = {}
        minima = {}
        maxima = {}
        totals = {}
        inverted = 0
        zero = 0
        duplicate_keys = Counter()
        face_regions = {}
        disconnected_regions = 0
        for record in self.volumes:
            connectivity = self.regions[record.name]
            points = self._tetrahedron_points(connectivity)
            signed = self._signed_volumes(points)
            inverted += int(np.sum(signed < -tolerance))
            zero += int(np.sum(np.abs(signed) <= tolerance))
            volumes = np.abs(signed)
            counts[record.name] = len(connectivity)
            minima[record.name] = float(np.min(volumes))
            maxima[record.name] = float(np.max(volumes))
            totals[record.name] = float(np.sum(volumes))
            for tetrahedron in connectivity:
                duplicate_keys[
                    tuple(sorted(int(tag) for tag in tetrahedron))
                ] += 1
            region_faces = {}
            neighbors = [set() for _ in connectivity]
            for tet_index, tetrahedron in enumerate(connectivity):
                for indices in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                    key = tuple(
                        sorted(int(tetrahedron[index]) for index in indices)
                    )
                    face_regions.setdefault(key, []).append(record.name)
                    if key in region_faces:
                        other = region_faces[key]
                        neighbors[tet_index].add(other)
                        neighbors[other].add(tet_index)
                    else:
                        region_faces[key] = tet_index
            reached = {0}
            pending = [0]
            while pending:
                current = pending.pop()
                unseen = neighbors[current] - reached
                reached.update(unseen)
                pending.extend(unseen)
            disconnected_regions += len(reached) != len(connectivity)
        duplicates = sum(
            count - 1 for count in duplicate_keys.values() if count > 1
        )
        nonconformal = 0
        for surface_name, faces in self.interface_faces.items():
            reverse, forward = self.interface_senses[surface_name]
            expected_count = (
                2 if reverse is not None and forward is not None else 1
            )
            expected_regions = {
                value for value in (reverse, forward) if value is not None
            }
            for face in faces:
                actual_regions = face_regions.get(
                    tuple(sorted(int(tag) for tag in face)), []
                )
                if (
                    len(actual_regions) != expected_count
                    or set(actual_regions) != expected_regions
                ):
                    nonconformal += 1
        if (
            inverted
            or zero
            or duplicates
            or nonconformal
            or disconnected_regions
        ):
            raise ValueError(
                "Volumetric mesh audit failed: "
                f"inverted={inverted}, zero={zero}, duplicates={duplicates}, "
                f"nonconformal={nonconformal}, "
                f"disconnected_regions={disconnected_regions}"
            )
        relative_errors = {
            name: abs(totals[name] - reference) / reference
            for name, reference in self.reference_volumes.items()
        }
        return NativeVolumeMeshResult(
            counts,
            minima,
            maxima,
            totals,
            sum(counts.values()),
            inverted,
            zero,
            duplicates,
            nonconformal,
            disconnected_regions,
            dict(self.reference_volumes),
            relative_errors,
        )

    def write(self, filename):
        path = Path(filename).with_suffix(".h5m")
        path.parent.mkdir(parents=True, exist_ok=True)
        mb = core.Core()
        used_tags = sorted(
            {
                int(tag)
                for connectivity in self.regions.values()
                for tetrahedron in connectivity
                for tag in tetrahedron
            }
        )
        handles = mb.create_vertices(
            np.asarray([self.coordinates[tag] for tag in used_tags]).ravel()
        )
        handle_by_tag = dict(zip(used_tags, handles))
        global_id = mb.tag_get_handle(
            "GLOBAL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True
        )
        category = mb.tag_get_handle(
            "CATEGORY", 32, types.MB_TYPE_OPAQUE, types.MB_TAG_SPARSE, True
        )
        name_tag = mb.tag_get_handle(
            "NAME", 32, types.MB_TYPE_OPAQUE, types.MB_TAG_SPARSE, True
        )
        material = mb.tag_get_handle(
            "MATERIAL", 32, types.MB_TYPE_OPAQUE, types.MB_TAG_SPARSE, True
        )
        for volume_id, record in enumerate(self.volumes, start=1):
            connectivity = self.regions[record.name].copy()
            points = self._tetrahedron_points(connectivity)
            inverted = self._signed_volumes(points) < 0.0
            connectivity[inverted, 2:4] = connectivity[inverted, 3:1:-1]
            element_handles = mb.create_elements(
                types.MBTET,
                np.asarray(
                    [
                        [handle_by_tag[int(tag)] for tag in tetrahedron]
                        for tetrahedron in connectivity
                    ],
                    dtype=np.uint64,
                ),
            )
            meshset = mb.create_meshset()
            mb.add_entities(meshset, element_handles)
            mb.tag_set_data(global_id, [meshset], [volume_id])
            for tag, value in (
                (category, "Volume"),
                (name_tag, record.name),
                (material, record.material_tag),
            ):
                mb.tag_set_data(
                    tag,
                    [meshset],
                    np.asarray([value.encode()], dtype="S32"),
                )
        mb.write_file(str(path))
        mb.write_file(str(path.with_suffix(".vtk")))
        return path

    @staticmethod
    def validate_file(filename):
        """Independently reload a volumetric H5M and audit region tags."""

        def opaque_string(value):
            if isinstance(value, str):
                return value.strip("\x00")
            return bytes(value).decode(errors="ignore").strip("\x00")

        mb = core.Core()
        mb.load_file(str(filename))
        root = mb.get_root_set()
        tetrahedra = set(mb.get_entities_by_type(root, types.MBTET))
        global_id = mb.tag_get_handle("GLOBAL_ID")
        category = mb.tag_get_handle("CATEGORY")
        name = mb.tag_get_handle("NAME")
        material = mb.tag_get_handle("MATERIAL")
        owned = []
        regions = []
        for meshset in mb.get_entities_by_type(root, types.MBENTITYSET):
            try:
                category_value = mb.tag_get_data(
                    category, [meshset], flat=True
                )[0]
            except RuntimeError:
                continue
            category_name = opaque_string(category_value)
            if category_name != "Volume":
                continue
            mb.tag_get_data(global_id, [meshset], flat=True)
            region_name = opaque_string(
                mb.tag_get_data(name, [meshset], flat=True)[0]
            )
            material_name = opaque_string(
                mb.tag_get_data(material, [meshset], flat=True)[0]
            )
            entities = list(mb.get_entities_by_type(meshset, types.MBTET))
            if not entities or not region_name or not material_name:
                raise ValueError(
                    "Volumetric H5M has an empty or untagged region"
                )
            owned.extend(entities)
            regions.append(
                {
                    "name": region_name,
                    "material": material_name,
                    "tetrahedron_count": len(entities),
                }
            )
        if set(owned) != tetrahedra or len(owned) != len(set(owned)):
            raise ValueError(
                "Volumetric H5M has orphaned or multiply-owned tetrahedra"
            )
        referenced = {
            vertex
            for tetrahedron in tetrahedra
            for vertex in mb.get_connectivity(tetrahedron)
        }
        vertices = set(mb.get_entities_by_type(root, types.MBVERTEX))
        if vertices != referenced:
            raise ValueError("Volumetric H5M has unreferenced vertices")
        return {
            "region_count": len(regions),
            "tetrahedron_count": len(tetrahedra),
            "orphan_tetrahedron_count": len(tetrahedra - set(owned)),
            "unreferenced_vertex_count": len(vertices - referenced),
            "regions": regions,
        }


def _merge_parameter_values(base, local, lower, upper):
    outside = [value for value in base if not lower < value < upper]
    return np.asarray(sorted(set((*outside, *local))), dtype=float)


def _rectangle_boundary_indices(phi_values, theta_values, bounds):
    phi_lower, phi_upper, theta_lower, theta_upper = bounds
    phi_indices = [
        index
        for index, value in enumerate(phi_values)
        if phi_lower - 1e-12 <= value <= phi_upper + 1e-12
    ]
    theta_indices = [
        index
        for index, value in enumerate(theta_values)
        if theta_lower - 1e-12 <= value <= theta_upper + 1e-12
    ]
    lower_phi, upper_phi = phi_indices[0], phi_indices[-1]
    lower_theta, upper_theta = theta_indices[0], theta_indices[-1]
    cycle = []
    cycle.extend((index, lower_theta) for index in phi_indices[:-1])
    cycle.extend((upper_phi, index) for index in theta_indices[:-1])
    cycle.extend((index, upper_theta) for index in reversed(phi_indices[1:]))
    cycle.extend((lower_phi, index) for index in reversed(theta_indices[1:]))
    return cycle


def _align_boundary_cycle(points, aperture_points, anchor, reference, normal):
    points = np.asarray(points, dtype=float)
    aperture = np.asarray(aperture_points, dtype=float)[:-1]
    if len(points) != len(aperture):
        raise ValueError(
            "Local patch boundary does not match aperture sampling: "
            f"{len(points)} != {len(aperture)}"
        )

    def projected_area(values):
        relative = values - anchor
        uv = np.column_stack((relative @ reference, relative @ normal))
        return float(
            np.sum(
                uv[:, 0] * np.roll(uv[:, 1], -1)
                - uv[:, 1] * np.roll(uv[:, 0], -1)
            )
            / 2.0
        )

    if projected_area(points) < 0.0:
        points = points[::-1]
    start = int(np.argmin(np.linalg.norm(points - aperture[0], axis=1)))
    return np.roll(points, -start, axis=0)


def _surface_grid(surface, phi_values, theta_values):
    return np.asarray(
        [
            [surface.evaluate(phi, theta) for theta in theta_values]
            for phi in phi_values
        ]
    )


def _radial_surface_triangles(
    grid,
    phi_values,
    theta_values,
    bounds=None,
    aperture_loop=None,
    anchor=None,
    reference=None,
    normal=None,
):
    triangles = []
    phi_lower = phi_upper = theta_lower = theta_upper = None
    if bounds is not None:
        phi_lower, phi_upper, theta_lower, theta_upper = bounds
    for phi_index in range(len(phi_values) - 1):
        for theta_index in range(len(theta_values)):
            next_theta = (theta_index + 1) % len(theta_values)
            theta_left = theta_values[theta_index]
            theta_right = (
                theta_values[next_theta]
                if next_theta > theta_index
                else theta_values[next_theta] + 2.0 * np.pi
            )
            center_phi = (
                phi_values[phi_index] + phi_values[phi_index + 1]
            ) / 2
            center_theta = (theta_left + theta_right) / 2
            inside_patch = (
                bounds is not None
                and phi_lower < center_phi < phi_upper
                and theta_lower < center_theta < theta_upper
            )
            if inside_patch:
                continue
            a = grid[phi_index, theta_index]
            b = grid[phi_index + 1, theta_index]
            c = grid[phi_index + 1, next_theta]
            d = grid[phi_index, next_theta]
            triangles.extend((np.asarray((a, b, c)), np.asarray((a, c, d))))
    if aperture_loop is not None:
        indices = _rectangle_boundary_indices(phi_values, theta_values, bounds)
        square = np.asarray([grid[index] for index in indices])
        square = _align_boundary_cycle(
            square, aperture_loop.outer_points, anchor, reference, normal
        )
        aperture = np.asarray(aperture_loop.outer_points)[:-1]
        for index in range(len(aperture)):
            following = (index + 1) % len(aperture)
            triangles.extend(
                (
                    np.asarray(
                        (aperture[index], square[index], square[following])
                    ),
                    np.asarray(
                        (
                            aperture[index],
                            square[following],
                            aperture[following],
                        )
                    ),
                )
            )
    return np.asarray(triangles)


def _cap_ring(inner, outer, desired):
    triangles = []
    for index in range(len(inner)):
        following = (index + 1) % len(inner)
        triangles.extend(
            (
                _orient_triangle(
                    (inner[index], outer[index], outer[following]), desired
                ),
                _orient_triangle(
                    (inner[index], outer[following], inner[following]), desired
                ),
            )
        )
    return np.asarray(triangles)


def _box_triangles(lower, upper):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    corners = np.asarray(
        (
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        )
    )
    triangles = []
    for indices, desired in (
        ((0, 3, 7, 4), (-1.0, 0.0, 0.0)),
        ((1, 5, 6, 2), (1.0, 0.0, 0.0)),
        ((0, 4, 5, 1), (0.0, -1.0, 0.0)),
        ((3, 2, 6, 7), (0.0, 1.0, 0.0)),
        ((0, 1, 2, 3), (0.0, 0.0, -1.0)),
        ((4, 7, 6, 5), (0.0, 0.0, 1.0)),
    ):
        a, b, c, d = corners[list(indices)]
        triangles.extend(
            (
                _orient_triangle((a, b, c), desired),
                _orient_triangle((a, c, d), desired),
            )
        )
    return np.asarray(triangles)


def build_native_port_surface_complex(model):
    """Build one native faceted complex from continuous surfaces and loops."""
    if len(model.ports) != 1:
        raise NotImplementedError(
            "Native conformal port export currently requires exactly one port"
        )
    port = model.ports[0]
    if port.placement.mode != "surface":
        raise NotImplementedError(
            "Native conformal port export requires placement.mode='surface'"
        )
    user_layers = list(model.radial_build.user_layer_names)
    resolved_start = model._resolve_port_endpoint(port, port.extent.start, {})
    resolved_end = model._resolve_port_endpoint(port, port.extent.end, {})
    target_layers = []
    for layer in user_layers:
        low, high = model._port_layer_interval(port, layer, None)
        if high > resolved_start and low < resolved_end:
            target_layers.append(layer)
    boundaries = model._aperture_boundaries(
        port,
        {name: None for name in user_layers},
        target_layers,
        resolved_start,
        resolved_end,
    )
    loops = build_aperture_loops(port, boundaries)
    # A layer-referenced end can intentionally coincide with that layer's
    # radial boundary. It is one physical aperture loop, not a microscopic
    # sidewall segment between two independently sampled copies.
    unique_loops = []
    for loop in loops:
        if (
            unique_loops
            and np.max(
                np.linalg.norm(
                    loop.inner_points - unique_loops[-1].inner_points, axis=1
                )
            )
            <= 0.05
        ):
            if loop.boundary_name.startswith("end:"):
                unique_loops[-1] = loop
            continue
        unique_loops.append(loop)
    loops = tuple(unique_loops)

    anchor = np.asarray(port.placement.anchor)
    axis = np.asarray(port.placement.local_axis)
    reference = np.asarray(port.placement.local_reference)
    normal = np.asarray(port.placement.local_normal)
    plasma_surface = model._anchor_reference_surface("plasma_surface")
    radial_names = ["plasma", *list(model.Surfaces)]
    radial_surfaces = [plasma_surface, *list(model.Surfaces.values())]
    volume_names = list(radial_names)
    volume_records = []
    for name in volume_names:
        if name == "plasma":
            material = "Vacuum"
            kind = "plasma_or_chamber"
        else:
            data = model.radial_build.radial_build[name]
            material = data.get("mat_tag", name)
            kind = (
                "plasma_or_chamber" if name == "chamber" else "blanket_layer"
            )
        volume_records.append(NativeVolumeRecord(name, kind, material))
    volume_records.append(
        NativeVolumeRecord(
            port.name + "__void", "port_void", port.fill.mat_tag
        )
    )
    if port.liner.enabled:
        volume_records.append(
            NativeVolumeRecord(
                port.name + "__liner", "port_liner", port.liner.mat_tag
            )
        )

    anchor_spec = port.placement.surface_anchor
    phi = np.deg2rad(anchor_spec.toroidal_angle)
    theta = np.deg2rad(anchor_spec.poloidal_angle)
    delta = 1e-5
    phi_speed = np.linalg.norm(
        plasma_surface.evaluate(phi + delta, theta)
        - plasma_surface.evaluate(phi - delta, theta)
    ) / (2.0 * delta)
    theta_speed = np.linalg.norm(
        plasma_surface.evaluate(phi, theta + delta)
        - plasma_surface.evaluate(phi, theta - delta)
    ) / (2.0 * delta)
    patch_half_width = model._port_aperture_half_width(port) * 1.8
    phi_half = patch_half_width / phi_speed
    theta_half = patch_half_width / theta_speed
    bounds = (
        phi - phi_half,
        phi + phi_half,
        theta - theta_half,
        theta + theta_half,
    )
    if (
        bounds[0] <= plasma_surface.phi_list[0]
        or bounds[1] >= plasma_surface.phi_list[-1]
    ):
        raise ValueError(
            "Port aperture patch intersects a toroidal sector seam"
        )
    sampled_points = (
        loops[0].outer_points
        if loops[0].outer_points is not None
        else loops[0].inner_points
    )
    sample_count = len(sampled_points) - 1
    if sample_count % 4 != 0:
        raise ValueError("Aperture sampling must be divisible by four")
    side_count = sample_count // 4 + 1
    local_phi = np.linspace(bounds[0], bounds[1], side_count)
    local_theta = np.linspace(bounds[2], bounds[3], side_count)
    base_phi = np.asarray(plasma_surface.phi_list, dtype=float)
    base_theta = np.linspace(
        theta - np.pi,
        theta + np.pi,
        len(plasma_surface.theta_list) - 1,
        endpoint=False,
    )
    phi_values = _merge_parameter_values(
        base_phi, local_phi, bounds[0], bounds[1]
    )
    theta_values = _merge_parameter_values(
        base_theta, local_theta, bounds[2], bounds[3]
    )

    loop_means = np.asarray(
        [np.mean((loop.inner_points[:-1] - anchor) @ axis) for loop in loops]
    )
    radial_w = []
    grids = []
    radial_meshes = []
    radial_loop_indices = []
    for name, surface in zip(radial_names, radial_surfaces):
        triangles = model._port_surface_triangles(port, surface)
        expected = surface.evaluate(phi, theta)
        coordinate = model._point_cloud_surface_coordinate(
            port, triangles, expected, f"native radial surface {name!r}"
        )
        radial_w.append(coordinate)
        loop_index = int(np.argmin(np.abs(loop_means - coordinate)))
        aperture_loop = None
        if (
            resolved_start - 0.1 <= coordinate <= resolved_end + 0.1
            and abs(loop_means[loop_index] - coordinate) <= 0.1
        ):
            aperture_loop = loops[loop_index]
            radial_loop_indices.append(loop_index)
        else:
            radial_loop_indices.append(None)
        grid = _surface_grid(surface, phi_values, theta_values)
        grids.append(grid)
        radial_meshes.append(
            _radial_surface_triangles(
                grid,
                phi_values,
                theta_values,
                bounds if aperture_loop is not None else None,
                aperture_loop,
                anchor,
                reference,
                normal,
            )
        )

    records = []

    def add_surface(name, kind, triangles, reverse, forward):
        triangles = np.asarray(triangles, dtype=float)
        if len(triangles) == 0:
            raise ValueError(f"Native surface {name!r} has no facets")
        records.append(
            NativeSurfaceRecord(name, kind, triangles, reverse, forward)
        )

    for index, (name, triangles) in enumerate(
        zip(radial_names, radial_meshes)
    ):
        reverse = volume_names[index]
        forward = (
            volume_names[index + 1] if index + 1 < len(volume_names) else None
        )
        add_surface(
            f"radial:{name}", "radial_surface", triangles, reverse, forward
        )

    phi_min = float(phi_values[0])
    phi_max = float(phi_values[-1])
    for volume_index, volume_name in enumerate(volume_names):
        for phi_index, phi_value, suffix, sign in (
            (0, phi_min, "toroidal_min", -1.0),
            (-1, phi_max, "toroidal_max", 1.0),
        ):
            desired = sign * np.asarray(
                (-np.sin(phi_value), np.cos(phi_value), 0.0)
            )
            outer = grids[volume_index][phi_index]
            if volume_index == 0:
                triangles = _disk_triangles(
                    np.vstack((outer, outer[0])), desired
                )
            else:
                inner = grids[volume_index - 1][phi_index]
                triangles = _cap_ring(inner, outer, desired)
            add_surface(
                f"sector_cap:{volume_name}:{suffix}",
                "sector_cap",
                triangles,
                volume_name,
                None,
            )

    void_name = port.name + "__void"
    liner_name = port.name + "__liner" if port.liner.enabled else None

    def region_at(coordinate):
        index = int(np.searchsorted(radial_w, coordinate, side="right"))
        return volume_names[index] if index < len(volume_names) else None

    if port.liner.enabled:
        inner_sidewall = []
        for left, right in zip(loops, loops[1:]):
            inner_sidewall.extend(
                _sidewall_triangles(
                    left.inner_points,
                    right.inner_points,
                    anchor,
                    reference,
                    normal,
                )
            )
        add_surface(
            f"port:{port.name}:void_liner",
            "void_liner_interface",
            inner_sidewall,
            void_name,
            liner_name,
        )

    for index, (left, right) in enumerate(zip(loops, loops[1:])):
        midpoint = (
            loop_means[min(index, len(loop_means) - 1)]
            + loop_means[min(index + 1, len(loop_means) - 1)]
        ) / 2.0
        surrounding = region_at(midpoint)
        left_points = (
            left.outer_points if port.liner.enabled else left.inner_points
        )
        right_points = (
            right.outer_points if port.liner.enabled else right.inner_points
        )
        triangles = _sidewall_triangles(
            left_points, right_points, anchor, reference, normal
        )
        add_surface(
            f"port:{port.name}:outer_sidewall:{index}:{surrounding or 'external'}",
            (
                "liner_blanket_interface"
                if port.liner.enabled
                else "void_blanket_interface"
            ),
            triangles,
            liner_name if port.liner.enabled else void_name,
            surrounding,
        )

    start_loop = loops[0]
    start_coordinate = float(
        np.mean((start_loop.inner_points[:-1] - anchor) @ axis)
    )
    start_region = region_at(start_coordinate - 0.2)
    add_surface(
        f"port:{port.name}:start:void",
        (
            "plasma_connection"
            if port.extent.start.reference == "plasma_surface"
            else "blind_termination"
        ),
        _disk_triangles(start_loop.inner_points, axis),
        start_region,
        void_name,
    )
    if port.liner.enabled:
        add_surface(
            f"port:{port.name}:start:liner",
            "liner_termination",
            _annulus_triangles(
                start_loop.inner_points, start_loop.outer_points, axis
            ),
            start_region,
            liner_name,
        )

    end_loop = loops[-1]
    end_coordinate = float(
        np.mean((end_loop.inner_points[:-1] - anchor) @ axis)
    )
    end_region = region_at(end_coordinate + 0.2)
    add_surface(
        f"port:{port.name}:end:void",
        "external_termination" if end_region is None else "blind_termination",
        _disk_triangles(end_loop.inner_points, axis),
        void_name,
        end_region,
    )
    if port.liner.enabled:
        add_surface(
            f"port:{port.name}:end:liner",
            (
                "external_termination"
                if end_region is None
                else "liner_termination"
            ),
            _annulus_triangles(
                end_loop.inner_points, end_loop.outer_points, axis
            ),
            liner_name,
            end_region,
        )

    graveyard_name = "graveyard"
    volume_records.append(
        NativeVolumeRecord(graveyard_name, "graveyard", "Graveyard")
    )
    records = [
        (
            replace(surface, forward_volume=graveyard_name)
            if surface.forward_volume is None
            else surface
        )
        for surface in records
    ]
    physical_points = np.concatenate(
        [surface.triangles.reshape((-1, 3)) for surface in records], axis=0
    )
    graveyard_margin = max(50.0, float(port.extent.outer_extension) + 10.0)
    lower = physical_points.min(axis=0) - graveyard_margin
    upper = physical_points.max(axis=0) + graveyard_margin
    add_surface(
        "graveyard:outer_boundary",
        "graveyard_boundary",
        _box_triangles(lower, upper),
        graveyard_name,
        None,
    )

    port_result = type(
        "NativePortResult",
        (),
        {
            "resolved_start": resolved_start,
            "resolved_end": resolved_end,
            "outer_extension": port.extent.outer_extension,
            "ordered_intersected_layers": tuple(target_layers),
        },
    )()
    return NativePortSurfaceComplex(
        model,
        port,
        loops,
        records,
        volume_records,
        {
            "names": radial_names,
            "coordinates": radial_w,
            "grids": grids,
            "phi_values": phi_values,
            "theta_values": theta_values,
            "port_result": port_result,
            "radial_loop_indices": radial_loop_indices,
        },
    )
