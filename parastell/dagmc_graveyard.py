"""Post-assembly closure for physical-only DAGMC reactor models."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path

import numpy as np


def _orient_triangle(triangle, desired):
    triangle = np.asarray(triangle, dtype=float)
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    return triangle[[0, 2, 1]] if np.dot(normal, desired) < 0.0 else triangle


def _box_triangles(lower, upper):
    x0, y0, z0 = np.asarray(lower, dtype=float)
    x1, y1, z1 = np.asarray(upper, dtype=float)
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


def _set_name(model, entity, name):
    from pymoab import types

    if len(name.encode("utf-8")) >= types.NAME_TAG_SIZE:
        digest = sha1(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:22]}_{digest}"
    tag = model.mb.tag_get_handle(
        "PARASTELL_NAME",
        types.NAME_TAG_SIZE,
        types.MB_TYPE_OPAQUE,
        types.MB_TAG_SPARSE,
        create_if_missing=True,
    )
    model.mb.tag_set_data(
        tag,
        [entity.handle],
        np.asarray([name.encode()], dtype=f"S{types.NAME_TAG_SIZE}"),
    )


def _add_box_surface(model, lower, upper, surface_id, senses, name):
    from pymoab import types

    triangles = _box_triangles(lower, upper)
    unique = []
    indices = {}
    connectivity = []
    for triangle in triangles:
        row = []
        for point in triangle:
            key = tuple(float(value) for value in point)
            if key not in indices:
                indices[key] = len(unique)
                unique.append(point)
            row.append(indices[key])
        connectivity.append(row)
    box_vertices = np.asarray(
        model.mb.create_vertices(np.asarray(unique).ravel()), dtype=np.uint64
    )
    elements = model.mb.create_elements(
        types.MBTRI, box_vertices[np.asarray(connectivity, dtype=int)]
    )
    surface = model.create_surface(global_id=surface_id)
    model.mb.add_entities(surface.handle, elements)
    surface.senses = senses
    _set_name(model, surface, name)
    return surface


def close_with_graveyard(model, *, margin_cm=50.0):
    """Create an interstitial vacuum and outer graveyard after assembly.

    ParaStell's one-sided reactor surfaces bound both the plasma chamber and
    exterior gaps. Those disconnected regions must remain transport vacuum;
    assigning them directly to ``mat:Graveyard`` would terminate every source
    history. The graveyard is therefore a separate shell outside an inner box.
    """
    import pydagmc
    from pymoab import types

    existing = [
        group
        for group in model.groups
        if group.name.casefold() == "mat:graveyard"
    ]
    if existing:
        raise ValueError("DAGMC model already contains a mat:Graveyard group")
    margin_cm = float(margin_cm)
    if not np.isfinite(margin_cm) or margin_cm <= 0.0:
        raise ValueError("graveyard margin must be positive and finite")

    exterior = []
    for surface in model.surfaces:
        reverse, forward = surface.senses
        if reverse is None and forward is None:
            raise ValueError(f"surface {surface.id} has no volume sense")
        if reverse is None or forward is None:
            exterior.append(surface)
    if not exterior:
        raise ValueError("DAGMC model has no one-sided physical surfaces")

    next_volume_id = max(volume.id for volume in model.volumes) + 1
    interstitial = model.create_volume(global_id=next_volume_id)
    _set_name(model, interstitial, "interstitial_vacuum")
    vacuum_groups = [
        group
        for group in model.groups
        if group.name.casefold() == "mat:vacuum"
    ]
    if len(vacuum_groups) > 1:
        raise ValueError("DAGMC model contains multiple mat:Vacuum groups")
    if vacuum_groups:
        vacuum = vacuum_groups[0]
    else:
        vacuum = pydagmc.Group.create(model, name="mat:Vacuum")
    vacuum.add_set(interstitial)
    component = pydagmc.Group.create(
        model, name="component:interstitial_vacuum"
    )
    component.add_set(interstitial)

    graveyard = model.create_volume(global_id=next_volume_id + 1)
    _set_name(model, graveyard, "graveyard")
    material = pydagmc.Group.create(model, name="mat:Graveyard")
    material.add_set(graveyard)
    component = pydagmc.Group.create(model, name="component:graveyard")
    component.add_set(graveyard)
    for surface in exterior:
        reverse, forward = surface.senses
        surface.senses = [
            interstitial if reverse is None else reverse,
            interstitial if forward is None else forward,
        ]

    root = model.mb.get_root_set()
    vertices = list(model.mb.get_entities_by_type(root, types.MBVERTEX))
    if not vertices:
        raise ValueError("DAGMC model has no vertices")
    coordinates = model.mb.get_coords(vertices).reshape((-1, 3))
    physical_lower = coordinates.min(axis=0)
    physical_upper = coordinates.max(axis=0)
    inner_lower = physical_lower - margin_cm
    inner_upper = physical_upper + margin_cm
    next_surface_id = max(surface.id for surface in model.surfaces) + 1
    inner_boundary = _add_box_surface(
        model,
        inner_lower,
        inner_upper,
        next_surface_id,
        [interstitial, graveyard],
        "graveyard:inner_boundary",
    )
    outer_lower = physical_lower - 2.0 * margin_cm
    outer_upper = physical_upper + 2.0 * margin_cm
    outer_boundary = _add_box_surface(
        model,
        outer_lower,
        outer_upper,
        next_surface_id + 1,
        [graveyard, None],
        "graveyard:outer_boundary",
    )
    one_sided = [
        surface.id
        for surface in model.surfaces
        if surface.senses[0] is None or surface.senses[1] is None
    ]
    if one_sided != [outer_boundary.id]:
        raise ValueError(
            f"post-assembly closure left unexpected one-sided surfaces {one_sided}"
        )
    return {
        "interstitial_vacuum_volume_id": int(interstitial.id),
        "graveyard_volume_id": int(graveyard.id),
        "inner_surface_id": int(inner_boundary.id),
        "outer_surface_id": int(outer_boundary.id),
        "closed_exterior_surface_ids": [int(item.id) for item in exterior],
        "interstitial_bounding_box_cm": [
            inner_lower.tolist(),
            inner_upper.tolist(),
        ],
        "graveyard_outer_bounding_box_cm": [
            outer_lower.tolist(),
            outer_upper.tolist(),
        ],
        "margin_cm": margin_cm,
    }


def close_dagmc_file(input_path, output_path, *, margin_cm=50.0):
    """Write a closed copy of a physical-only DAGMC H5M file."""
    import pydagmc

    input_path = Path(input_path)
    output_path = Path(output_path)
    model = pydagmc.Model(str(input_path))
    report = close_with_graveyard(model, margin_cm=margin_cm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.write_file(str(output_path))
    report["input_path"] = str(input_path.resolve())
    report["output_path"] = str(output_path.resolve())
    return report
