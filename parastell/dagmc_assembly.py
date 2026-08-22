"""Post-assembly operations for independently faceted DAGMC submodels."""

from __future__ import annotations

from collections import Counter
from hashlib import sha1

import numpy as np
import pydagmc
from pymoab import types

from .native_port_geometry import _box_triangles


def _opaque_name(value):
    if isinstance(value, str):
        return value.strip("\x00")
    return bytes(value).decode(errors="ignore").strip("\x00")


def _set_parastell_name(mb, entity, name):
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
    mb.tag_set_data(
        tag,
        [entity],
        np.asarray([name.encode()], dtype=f"S{types.NAME_TAG_SIZE}"),
    )


def graveyard_groups(model):
    """Return groups that claim the reserved graveyard material name."""
    return tuple(
        group
        for group in model.groups
        if group.name.casefold() == "mat:graveyard".casefold()
    )


def assert_no_graveyard(model, *, label="DAGMC submodel"):
    """Reject submodels that would conflict with post-assembly closure."""
    groups = graveyard_groups(model)
    if groups:
        raise ValueError(
            f"{label} already contains {len(groups)} mat:Graveyard group(s); "
            "global graveyard closure requires physical-only inputs"
        )


def tag_volume_component(model, volume, name):
    """Attach a stable ParaStell name and component group to a volume."""
    _set_parastell_name(model.mb, volume.handle, name)
    group_name = f"component:{name}"
    if group_name in model.groups_by_name:
        group = model.groups_by_name[group_name]
    else:
        group = pydagmc.Group.create(model, name=group_name)
    group.add_set(volume)


def ensure_geometry_names(model):
    """Fill missing ParaStell names without replacing existing source names."""
    mb = model.mb
    try:
        tag = mb.tag_get_handle("PARASTELL_NAME")
    except RuntimeError:
        tag = None
    for kind, entities in (
        ("volume", model.volumes),
        ("surface", model.surfaces),
    ):
        for entity in entities:
            present = False
            if tag is not None:
                try:
                    value = mb.tag_get_data(tag, [entity.handle], flat=True)[0]
                    present = bool(_opaque_name(value))
                except RuntimeError:
                    pass
            if not present:
                _set_parastell_name(
                    mb, entity.handle, f"{kind}:{int(entity.id)}"
                )


def close_with_graveyard(model, *, margin=50.0):
    """Close a combined physical DAGMC model with exactly one graveyard.

    Every existing one-sided physical surface receives the new graveyard as its
    missing sense.  A single enclosing box is then added as the only remaining
    one-sided boundary.  Existing physical triangles are not copied or changed.
    """
    assert_no_graveyard(model, label="Combined DAGMC model")
    margin = float(margin)
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("Graveyard margin must be a positive finite distance")

    exterior_surfaces = []
    for surface in model.surfaces:
        reverse, forward = surface.senses
        if reverse is None and forward is None:
            raise ValueError(f"Surface {surface.id} has no volume sense")
        if reverse is None or forward is None:
            exterior_surfaces.append(surface)
    if not exterior_surfaces:
        raise ValueError(
            "Combined DAGMC model has no one-sided exterior surface"
        )

    volume_id = max((volume.id for volume in model.volumes), default=0) + 1
    surface_id = max((surface.id for surface in model.surfaces), default=0) + 1
    graveyard = model.create_volume(global_id=volume_id)
    _set_parastell_name(model.mb, graveyard.handle, "graveyard")
    material = pydagmc.Group.create(model, name="mat:Graveyard")
    material.add_set(graveyard)
    component = pydagmc.Group.create(model, name="component:graveyard")
    component.add_set(graveyard)

    for surface in exterior_surfaces:
        reverse, forward = surface.senses
        surface.senses = [
            graveyard if reverse is None else reverse,
            graveyard if forward is None else forward,
        ]

    root = model.mb.get_root_set()
    vertices = list(model.mb.get_entities_by_type(root, types.MBVERTEX))
    if not vertices:
        raise ValueError("Combined DAGMC model has no vertices")
    coordinates = model.mb.get_coords(vertices).reshape((-1, 3))
    lower = coordinates.min(axis=0) - margin
    upper = coordinates.max(axis=0) + margin
    triangles = _box_triangles(lower, upper)
    unique_coordinates = []
    coordinate_indices = {}
    connectivity_indices = []
    for triangle in triangles:
        indices = []
        for point in triangle:
            key = tuple(float(value) for value in point)
            if key not in coordinate_indices:
                coordinate_indices[key] = len(unique_coordinates)
                unique_coordinates.append(point)
            indices.append(coordinate_indices[key])
        connectivity_indices.append(indices)
    box_vertices = np.asarray(
        model.mb.create_vertices(np.asarray(unique_coordinates).ravel()),
        dtype=np.uint64,
    )
    connectivity = box_vertices[np.asarray(connectivity_indices, dtype=int)]
    triangle_handles = model.mb.create_elements(types.MBTRI, connectivity)
    boundary = model.create_surface(global_id=surface_id)
    model.mb.add_entities(boundary.handle, triangle_handles)
    boundary.senses = [graveyard, None]
    _set_parastell_name(model.mb, boundary.handle, "graveyard:outer_boundary")
    return {
        "graveyard_volume_id": int(graveyard.id),
        "outer_surface_id": int(boundary.id),
        "closed_exterior_surface_ids": tuple(
            int(surface.id) for surface in exterior_surfaces
        ),
        "bounding_box": (lower.tolist(), upper.tolist()),
    }


def _coordinate_key(point, tolerance):
    return tuple(np.rint(np.asarray(point) / tolerance).astype(np.int64))


def audit_dagmc_model(model, *, vertex_tolerance=1.0e-9):
    """Perform a strict, PyMOAB-backed structural audit of an assembled model."""
    mb = model.mb
    root = mb.get_root_set()
    all_triangles = set(mb.get_entities_by_type(root, types.MBTRI))
    all_vertices = set(mb.get_entities_by_type(root, types.MBVERTEX))
    global_id = mb.tag_get_handle("GLOBAL_ID")
    category = mb.tag_get_handle("CATEGORY")
    geom_dimension = mb.tag_get_handle("GEOM_DIMENSION")
    senses_tag = mb.tag_get_handle("GEOM_SENSE_2")
    component_name = mb.tag_get_handle("PARASTELL_NAME")

    volume_handles = {volume.handle for volume in model.volumes}
    surface_triangles = []
    facet_counts = Counter()
    zero_area = 0
    one_sided = []
    surface_summary = []
    for surface in model.surfaces:
        for tag in (global_id, category, geom_dimension, component_name):
            mb.tag_get_data(tag, [surface.handle], flat=True)
        sense_handles = mb.tag_get_data(
            senses_tag, [surface.handle], flat=True
        )
        nonzero_senses = [int(value) for value in sense_handles if int(value)]
        if not nonzero_senses or any(
            value not in volume_handles for value in nonzero_senses
        ):
            raise ValueError(f"Surface {surface.id} has invalid volume senses")
        if not set(nonzero_senses).issubset(
            set(mb.get_parent_meshsets(surface.handle))
        ):
            raise ValueError(
                f"Surface {surface.id} sense volumes are not parent meshsets"
            )
        if len(nonzero_senses) == 1:
            one_sided.append(int(surface.id))
        triangles = list(mb.get_entities_by_type(surface.handle, types.MBTRI))
        if not triangles:
            raise ValueError(f"Surface {surface.id} has no facets")
        surface_triangles.extend(triangles)
        for triangle in triangles:
            coordinates = mb.get_coords(mb.get_connectivity(triangle)).reshape(
                (3, 3)
            )
            key = tuple(
                sorted(
                    _coordinate_key(point, vertex_tolerance)
                    for point in coordinates
                )
            )
            facet_counts[key] += 1
            zero_area += (
                np.linalg.norm(
                    np.cross(
                        coordinates[1] - coordinates[0],
                        coordinates[2] - coordinates[0],
                    )
                )
                / 2.0
                <= 1.0e-12
            )
        surface_summary.append(
            {
                "id": int(surface.id),
                "triangle_count": len(triangles),
                "sense_volume_ids": [
                    int(volume.id) if volume is not None else None
                    for volume in surface.senses
                ],
            }
        )

    duplicates = sum(count - 1 for count in facet_counts.values() if count > 1)
    owned = set(surface_triangles)
    orphan = all_triangles - owned
    multiply_owned = len(surface_triangles) - len(owned)
    referenced = {
        vertex
        for triangle in all_triangles
        for vertex in mb.get_connectivity(triangle)
    }

    unsealed = {}
    volume_summary = []
    for volume in model.volumes:
        for tag in (global_id, category, geom_dimension, component_name):
            mb.tag_get_data(tag, [volume.handle], flat=True)
        edge_counts = Counter()
        for surface in volume.surfaces:
            for triangle in mb.get_entities_by_type(
                surface.handle, types.MBTRI
            ):
                coordinates = mb.get_coords(
                    mb.get_connectivity(triangle)
                ).reshape((3, 3))
                keys = [
                    _coordinate_key(point, vertex_tolerance)
                    for point in coordinates
                ]
                for left, right in ((0, 1), (1, 2), (2, 0)):
                    edge_counts[tuple(sorted((keys[left], keys[right])))] += 1
        bad = sum(count != 2 for count in edge_counts.values())
        if bad:
            unsealed[int(volume.id)] = bad
        raw_name = mb.tag_get_data(component_name, [volume.handle], flat=True)[
            0
        ]
        volume_summary.append(
            {
                "id": int(volume.id),
                "name": _opaque_name(raw_name),
                "material": volume.material,
                "surface_count": len(volume.surfaces),
            }
        )

    failures = {
        "duplicate_facet_count": duplicates,
        "zero_area_triangle_count": int(zero_area),
        "orphan_triangle_count": len(orphan),
        "multiply_owned_triangle_count": multiply_owned,
        "unreferenced_vertex_count": len(all_vertices - referenced),
        "unsealed_edge_count": sum(unsealed.values()),
    }
    if any(failures.values()):
        raise ValueError(f"Assembled DAGMC audit failed: {failures}")
    if len(one_sided) != 1:
        raise ValueError(
            "Assembled DAGMC model must have exactly one one-sided outer "
            f"boundary, found {one_sided}"
        )
    return {
        "volume_count": len(model.volumes),
        "surface_count": len(model.surfaces),
        "triangle_count": len(all_triangles),
        **failures,
        "one_sided_surface_ids": one_sided,
        "volume_shell_unsealed_edges": unsealed,
        "volumes": volume_summary,
        "surfaces": surface_summary,
        "material_groups": sorted(
            name for name in model.group_names if name.startswith("mat:")
        ),
    }


def dagmc_component_names(model):
    """Map volume IDs to stable ParaStell component names."""
    tag = model.mb.tag_get_handle("PARASTELL_NAME")
    return {
        int(volume.id): _opaque_name(
            model.mb.tag_get_data(tag, [volume.handle], flat=True)[0]
        )
        for volume in model.volumes
    }


def _ray_triangle(origin, direction, triangle, tolerance=1.0e-10):
    edge_1 = triangle[1] - triangle[0]
    edge_2 = triangle[2] - triangle[0]
    cross = np.cross(direction, edge_2)
    determinant = float(np.dot(edge_1, cross))
    if abs(determinant) <= tolerance:
        return None
    inverse = 1.0 / determinant
    displacement = origin - triangle[0]
    u = inverse * float(np.dot(displacement, cross))
    if u < -tolerance or u > 1.0 + tolerance:
        return None
    q = np.cross(displacement, edge_1)
    v = inverse * float(np.dot(direction, q))
    if v < -tolerance or u + v > 1.0 + tolerance:
        return None
    distance = inverse * float(np.dot(edge_2, q))
    return distance if distance > tolerance else None


def ray_region_sequence(model, origin, direction, initial_volume_id):
    """Trace an independent directed facet ray through assembled volume senses."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    hits = []
    for surface in model.surfaces:
        distances = []
        triangles = np.asarray(surface.triangle_coords).reshape((-1, 3, 3))
        for triangle in triangles:
            distance = _ray_triangle(origin, direction, triangle)
            if distance is not None:
                distances.append(distance)
        for distance in sorted(distances):
            if not hits or not (
                surface.id == hits[-1][1].id
                and abs(distance - hits[-1][0]) <= 1.0e-7
            ):
                hits.append((distance, surface))
    hits.sort(key=lambda item: item[0])
    names = dagmc_component_names(model)
    current = int(initial_volume_id)
    sequence = [names[current]]
    for _, surface in hits:
        sense_ids = [
            int(volume.id) if volume is not None else None
            for volume in surface.senses
        ]
        if current == sense_ids[0]:
            current = sense_ids[1]
        elif current == sense_ids[1]:
            current = sense_ids[0]
        else:
            continue
        name = "external" if current is None else names[current]
        if sequence[-1] != name:
            sequence.append(name)
        if current is None:
            break
    return tuple(sequence)
