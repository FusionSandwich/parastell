"""Independent PyMOAB audit of native DAGMC topology and material ownership."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .reference_geometry import sha256_file


_EXPECTED_DIMENSION = {
    "Vertex": 0,
    "Curve": 1,
    "Surface": 2,
    "Volume": 3,
    # DAGMC material/group meshsets are non-geometric and carry -1 in the
    # GEOM_DIMENSION tag.  The value 4 is the MOAB entity-set category, not a
    # valid geometry dimension for these groups.
    "Group": -1,
}


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
            tag, np.asarray([handle], dtype=np.uint64), flat=True
        )
    except RuntimeError as error:
        message = str(error).lower()
        if "mb_tag_not_found" in message or "tag not found" in message:
            return None
        raise
    if len(values) != 1:
        return None
    return values[0]


def _entity_handle_array(values: Any) -> np.ndarray:
    """Return the uint64 EntityHandle array required by real PyMOAB."""
    return np.asarray(values, dtype=np.uint64).reshape(-1)


def _required_tag(
    mesh: Any, name: str, size: int, data_type: Any, storage: Any
):
    return mesh.tag_get_handle(
        name,
        size,
        data_type,
        storage,
        create_if_missing=False,
    )


def _native_volume_closure(
    mesh: Any,
    volume_handle: int,
    surface_handles: list[int],
    senses: dict[int, tuple[int, int]],
    *,
    vector_area_relative_tolerance: float,
) -> dict[str, Any]:
    from pymoab import types

    edge_counts: Counter[tuple[int, int]] = Counter()
    directed_balance: Counter[tuple[int, int]] = Counter()
    vector_area = np.zeros(3)
    total_area = 0.0
    signed_six_volume = 0.0
    triangle_count = 0
    degenerate_triangle_count = 0
    repeated_vertex_triangle_count = 0
    for surface_handle in surface_handles:
        forward, reverse = senses[surface_handle]
        if volume_handle == forward:
            sign = 1
        elif volume_handle == reverse:
            sign = -1
        else:
            raise ValueError("volume/surface sense incidence is inconsistent")
        triangles = mesh.get_entities_by_type(surface_handle, types.MBTRI)
        for triangle_handle in triangles:
            connectivity = [
                int(value)
                for value in mesh.get_connectivity(
                    _entity_handle_array([triangle_handle])
                )
            ]
            if len(connectivity) != 3:
                raise ValueError("DAGMC facet is not a triangle")
            triangle_count += 1
            if len(set(connectivity)) != 3:
                repeated_vertex_triangle_count += 1
            oriented = (
                connectivity
                if sign > 0
                else [
                    connectivity[0],
                    connectivity[2],
                    connectivity[1],
                ]
            )
            for first, second in ((0, 1), (1, 2), (2, 0)):
                directed = (oriented[first], oriented[second])
                edge = tuple(sorted(directed))
                edge_counts[edge] += 1
                directed_balance[edge] += 1 if directed == edge else -1
            coordinates = np.asarray(
                mesh.get_coords(_entity_handle_array(connectivity)),
                dtype=float,
            ).reshape((3, 3))
            area_vector = 0.5 * np.cross(
                coordinates[1] - coordinates[0],
                coordinates[2] - coordinates[0],
            )
            area = float(np.linalg.norm(area_vector))
            if (
                not np.all(np.isfinite(coordinates))
                or not np.all(np.isfinite(area_vector))
                or not np.isfinite(area)
                or area <= 0.0
            ):
                degenerate_triangle_count += 1
                # This facet has already failed the topology gate. Skipping
                # its arithmetic keeps the evidence JSON finite and writable.
                continue
            total_area += area
            vector_area += sign * area_vector
            signed_six_volume += sign * float(
                np.dot(
                    coordinates[0],
                    np.cross(coordinates[1], coordinates[2]),
                )
            )
    edge_errors = sum(value != 2 for value in edge_counts.values())
    direction_errors = sum(value != 0 for value in directed_balance.values())
    relative_vector_area = (
        float(np.linalg.norm(vector_area) / total_area)
        if total_area > 0.0
        else None
    )
    signed_volume = signed_six_volume / 6.0
    passed = (
        bool(surface_handles)
        and triangle_count > 0
        and edge_errors == 0
        and direction_errors == 0
        and degenerate_triangle_count == 0
        and repeated_vertex_triangle_count == 0
        and relative_vector_area is not None
        and relative_vector_area <= vector_area_relative_tolerance
        and signed_volume > 0.0
    )
    return {
        "volume_handle": int(volume_handle),
        "surface_handles": [int(value) for value in surface_handles],
        "triangle_count": triangle_count,
        "native_edge_count": len(edge_counts),
        "native_edge_multiplicity_error_count": edge_errors,
        "native_directed_edge_error_count": direction_errors,
        "repeated_vertex_triangle_count": repeated_vertex_triangle_count,
        "degenerate_triangle_count": degenerate_triangle_count,
        "total_area_cm2": total_area,
        "vector_area_cm2": [float(value) for value in vector_area],
        "vector_area_closure_relative": relative_vector_area,
        "vector_area_relative_tolerance": vector_area_relative_tolerance,
        "signed_volume_cm3": signed_volume,
        "pass": passed,
    }


def audit_native_moab_topology(
    dagmc_path: str | Path,
    *,
    expected_material_counts: dict[str, int],
    vector_area_relative_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Audit the unmodified file before any PyDAGMC model is constructed."""
    path = Path(dagmc_path).resolve()
    before = sha256_file(path)
    if (
        isinstance(vector_area_relative_tolerance, bool)
        or not isinstance(vector_area_relative_tolerance, Real)
        or not math.isfinite(float(vector_area_relative_tolerance))
        or float(vector_area_relative_tolerance) <= 0.0
    ):
        raise ValueError(
            "vector-area relative tolerance must be finite and positive"
        )
    from pymoab import core, types

    mesh = core.Core()
    mesh.load_file(str(path))
    root = mesh.get_root_set()
    tag_error = None
    try:
        category_tag = _required_tag(
            mesh,
            types.CATEGORY_TAG_NAME,
            types.CATEGORY_TAG_SIZE,
            types.MB_TYPE_OPAQUE,
            types.MB_TAG_SPARSE,
        )
        dimension_tag = _required_tag(
            mesh,
            getattr(types, "GEOM_DIMENSION_TAG_NAME", "GEOM_DIMENSION"),
            1,
            types.MB_TYPE_INTEGER,
            types.MB_TAG_DENSE,
        )
        global_id_tag = _required_tag(
            mesh,
            getattr(types, "GLOBAL_ID_TAG_NAME", "GLOBAL_ID"),
            1,
            types.MB_TYPE_INTEGER,
            types.MB_TAG_DENSE,
        )
        sense_tag = _required_tag(
            mesh,
            "GEOM_SENSE_2",
            2,
            types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE,
        )
        name_tag = _required_tag(
            mesh,
            getattr(types, "NAME_TAG_NAME", "NAME"),
            getattr(types, "NAME_TAG_SIZE", 32),
            types.MB_TYPE_OPAQUE,
            types.MB_TAG_SPARSE,
        )
    except Exception as error:
        tag_error = f"{type(error).__name__}: {error}"
    if tag_error is not None:
        after = sha256_file(path)
        return {
            "schema": "parastell.native_moab_topology/v1.0.0",
            "raw_h5m_sha256_before": before,
            "raw_h5m_sha256_after": after,
            "h5m_unchanged": before == after,
            "required_tag_error": tag_error,
            "native_topology_gate_pass": False,
        }

    entity_sets = [
        int(value)
        for value in mesh.get_entities_by_type(root, types.MBENTITYSET)
    ]
    rows = []
    classified: dict[int, dict[str, Any]] = {}
    schema_errors = []
    for handle in entity_sets:
        category_value = _optional_tag_value(mesh, category_tag, handle)
        dimension_value = _optional_tag_value(mesh, dimension_tag, handle)
        category = (
            _decode_opaque(category_value)
            if category_value is not None
            else None
        )
        dimension = (
            int(dimension_value) if dimension_value is not None else None
        )
        if category is None:
            continue
        global_value = _optional_tag_value(mesh, global_id_tag, handle)
        global_id = int(global_value) if global_value is not None else None
        expected_dimension = _EXPECTED_DIMENSION.get(category)
        valid = (
            expected_dimension is not None
            and dimension == expected_dimension
            and global_id is not None
        )
        row = {
            "handle": handle,
            "category": category,
            "geometry_dimension": dimension,
            "expected_geometry_dimension": expected_dimension,
            "global_id": global_id,
            "schema_pass": valid,
        }
        rows.append(row)
        classified[handle] = row
        if not valid:
            schema_errors.append(row)

    ids_by_category: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row["global_id"] is not None and row["category"] is not None:
            ids_by_category[row["category"]].append(row["global_id"])
    id_errors = {}
    for category in ("Surface", "Volume"):
        values = ids_by_category.get(category, [])
        duplicates = sorted(
            {value for value in values if values.count(value) > 1}
        )
        nonpositive = sorted({value for value in values if value <= 0})
        if duplicates or nonpositive:
            id_errors[category] = {
                "duplicate_ids": duplicates,
                "nonpositive_ids": nonpositive,
            }

    surface_handles = sorted(
        handle
        for handle, row in classified.items()
        if row["category"] == "Surface" and row["schema_pass"]
    )
    volume_handles = sorted(
        handle
        for handle, row in classified.items()
        if row["category"] == "Volume" and row["schema_pass"]
    )
    volume_set = set(volume_handles)
    surface_set = set(surface_handles)
    sense_rows = []
    senses: dict[int, tuple[int, int]] = {}
    incidence_by_volume: dict[int, list[int]] = defaultdict(list)
    for surface_handle in surface_handles:
        try:
            raw_senses = mesh.tag_get_data(
                sense_tag,
                np.asarray([surface_handle], dtype=np.uint64),
                flat=True,
            )
            sense = tuple(int(value) for value in raw_senses)
        except Exception as error:
            sense = ()
            sense_error = f"{type(error).__name__}: {error}"
        else:
            sense_error = None
        nonzero = [value for value in sense if value != 0]
        valid_handles = all(value in volume_set for value in nonzero)
        distinct = len(nonzero) == len(set(nonzero))
        sense_shape_pass = (
            len(sense) == 2 and bool(nonzero) and valid_handles and distinct
        )
        parents = {
            int(value)
            for value in mesh.get_parent_meshsets(surface_handle)
            if int(value) in classified
        }
        expected_parents = set(nonzero)
        parent_pass = parents == expected_parents
        if sense_shape_pass:
            senses[surface_handle] = (sense[0], sense[1])
            for volume_handle in nonzero:
                incidence_by_volume[volume_handle].append(surface_handle)
        sense_rows.append(
            {
                "surface_handle": surface_handle,
                "surface_global_id": classified[surface_handle]["global_id"],
                "sense_handles": list(sense),
                "sense_volume_global_ids": [
                    (
                        classified.get(value, {}).get("global_id")
                        if value
                        else None
                    )
                    for value in sense
                ],
                "nonzero_sense_handles": nonzero,
                "classified_parent_handles": sorted(parents),
                "sense_error": sense_error,
                "sense_shape_pass": sense_shape_pass,
                "parent_incidence_pass": parent_pass,
                "pass": sense_shape_pass and parent_pass,
            }
        )

    volume_incidence_rows = []
    for volume_handle in volume_handles:
        children = {
            int(value)
            for value in mesh.get_child_meshsets(volume_handle)
            if int(value) in classified
        }
        expected_children = set(incidence_by_volume.get(volume_handle, []))
        volume_incidence_rows.append(
            {
                "volume_handle": volume_handle,
                "volume_global_id": classified[volume_handle]["global_id"],
                "classified_child_handles": sorted(children),
                "classified_child_global_ids": sorted(
                    classified[value]["global_id"] for value in children
                ),
                "sense_surface_handles": sorted(expected_children),
                "sense_surface_global_ids": sorted(
                    classified[value]["global_id"]
                    for value in expected_children
                ),
                "pass": children == expected_children
                and children.issubset(surface_set),
            }
        )

    root_triangles = {
        int(value) for value in mesh.get_entities_by_type(root, types.MBTRI)
    }
    triangle_owners: dict[int, list[int]] = defaultdict(list)
    empty_surface_handles = []
    invalid_triangle_handles = []
    for surface_handle in surface_handles:
        triangles = [
            int(value)
            for value in mesh.get_entities_by_type(surface_handle, types.MBTRI)
        ]
        if not triangles:
            empty_surface_handles.append(surface_handle)
        for triangle_handle in triangles:
            triangle_owners[triangle_handle].append(surface_handle)
            connectivity = [
                int(value)
                for value in mesh.get_connectivity(
                    _entity_handle_array([triangle_handle])
                )
            ]
            if len(connectivity) != 3 or len(set(connectivity)) != 3:
                invalid_triangle_handles.append(triangle_handle)
                continue
            coordinates = np.asarray(
                mesh.get_coords(_entity_handle_array(connectivity)),
                dtype=float,
            ).reshape((3, 3))
            area = 0.5 * np.linalg.norm(
                np.cross(
                    coordinates[1] - coordinates[0],
                    coordinates[2] - coordinates[0],
                )
            )
            if not np.all(np.isfinite(coordinates)) or area <= 0.0:
                invalid_triangle_handles.append(triangle_handle)
    unowned_triangles = sorted(root_triangles - set(triangle_owners))
    multiply_owned_triangles = sorted(
        triangle
        for triangle, owners in triangle_owners.items()
        if len(owners) != 1
    )
    foreign_owned_triangles = sorted(set(triangle_owners) - root_triangles)
    triangle_ownership_pass = not (
        empty_surface_handles
        or invalid_triangle_handles
        or unowned_triangles
        or multiply_owned_triangles
        or foreign_owned_triangles
    )

    volume_closures = []
    if all(row["pass"] for row in sense_rows) and all(
        row["pass"] for row in volume_incidence_rows
    ):
        for volume_handle in volume_handles:
            closure = _native_volume_closure(
                mesh,
                volume_handle,
                sorted(incidence_by_volume.get(volume_handle, [])),
                senses,
                vector_area_relative_tolerance=(
                    vector_area_relative_tolerance
                ),
            )
            closure["volume_global_id"] = classified[volume_handle][
                "global_id"
            ]
            closure["surface_global_ids"] = sorted(
                classified[value]["global_id"]
                for value in incidence_by_volume.get(volume_handle, [])
            )
            volume_closures.append(closure)

    group_handles = sorted(
        handle
        for handle, row in classified.items()
        if row["category"] == "Group" and row["schema_pass"]
    )
    material_groups = []
    volume_materials: dict[int, list[str]] = defaultdict(list)
    for group_handle in group_handles:
        name_value = _optional_tag_value(mesh, name_tag, group_handle)
        name = _decode_opaque(name_value) if name_value is not None else ""
        if not name.startswith("mat:"):
            continue
        material = name.removeprefix("mat:").strip()
        members = {
            int(value) for value in mesh.get_entities_by_handle(group_handle)
        }
        volume_members = sorted(members & volume_set)
        invalid_members = sorted(members - volume_set)
        for volume_handle in volume_members:
            volume_materials[volume_handle].append(material)
        material_groups.append(
            {
                "group_handle": group_handle,
                "group_global_id": classified[group_handle]["global_id"],
                "name": name,
                "material": material,
                "volume_handles": volume_members,
                "volume_global_ids": sorted(
                    classified[value]["global_id"] for value in volume_members
                ),
                "invalid_member_handles": invalid_members,
                "pass": bool(material)
                and bool(volume_members)
                and not invalid_members,
            }
        )
    volume_material_rows = []
    actual_material_counts: Counter[str] = Counter()
    for volume_handle in volume_handles:
        materials = volume_materials.get(volume_handle, [])
        if len(materials) == 1:
            actual_material_counts[materials[0]] += 1
        volume_material_rows.append(
            {
                "volume_handle": volume_handle,
                "volume_global_id": classified[volume_handle]["global_id"],
                "material_groups": materials,
                "pass": len(materials) == 1,
            }
        )
    material_gate = (
        bool(material_groups)
        and all(row["pass"] for row in material_groups)
        and all(row["pass"] for row in volume_material_rows)
        and dict(sorted(actual_material_counts.items()))
        == dict(sorted(expected_material_counts.items()))
    )

    after = sha256_file(path)
    gate = (
        before == after
        and bool(rows)
        and not schema_errors
        and not id_errors
        and bool(surface_handles)
        and bool(volume_handles)
        and all(row["pass"] for row in sense_rows)
        and all(row["pass"] for row in volume_incidence_rows)
        and triangle_ownership_pass
        and len(volume_closures) == len(volume_handles)
        and all(row["pass"] for row in volume_closures)
        and material_gate
    )
    return {
        "schema": "parastell.native_moab_topology/v1.0.0",
        "raw_h5m_sha256_before": before,
        "raw_h5m_sha256_after": after,
        "h5m_unchanged": before == after,
        "required_tag_error": None,
        "classified_entity_sets": rows,
        "schema_errors": schema_errors,
        "id_errors": id_errors,
        "surface_handles": surface_handles,
        "surface_global_ids": sorted(
            classified[value]["global_id"] for value in surface_handles
        ),
        "volume_handles": volume_handles,
        "volume_global_ids": sorted(
            classified[value]["global_id"] for value in volume_handles
        ),
        "surface_senses": sense_rows,
        "volume_incidence": volume_incidence_rows,
        "triangle_ownership": {
            "root_triangle_count": len(root_triangles),
            "owned_triangle_count": len(triangle_owners),
            "empty_surface_handles": empty_surface_handles,
            "invalid_triangle_handles": sorted(set(invalid_triangle_handles)),
            "unowned_triangle_handles": unowned_triangles,
            "multiply_owned_triangle_handles": multiply_owned_triangles,
            "foreign_owned_triangle_handles": foreign_owned_triangles,
            "pass": triangle_ownership_pass,
        },
        "volume_closures": volume_closures,
        "material_groups": material_groups,
        "volume_materials": volume_material_rows,
        "expected_material_counts": dict(
            sorted(expected_material_counts.items())
        ),
        "actual_material_counts": dict(sorted(actual_material_counts.items())),
        "material_group_gate_pass": material_gate,
        "native_topology_gate_pass": gate,
    }
