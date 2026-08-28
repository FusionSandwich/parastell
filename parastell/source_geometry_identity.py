"""Canonical identity helpers for regenerated STEP and source-mesh files."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np


SOURCE_MESH_IDENTITY_SCHEMA = "parastell.source_mesh_identity/v1.1.0"


def canonical_step_payload_sha256(path: str | Path) -> str:
    """Hash STEP payload while ignoring only its export timestamp.

    Open CASCADE writes the wall-clock export time into ``FILE_NAME``.  That
    field is provenance, not shape.  Every other byte of the normalized text
    remains covered by the digest.
    """
    text = Path(path).read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = r"(FILE_NAME\(\s*'[^']*'\s*,\s*)'[^']*'"
    normalized, count = re.subn(
        pattern,
        r"\1'<normalized-export-timestamp>'",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"STEP FILE_NAME timestamp not found in {path}")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _finite_float64(values: Any, *, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(
            f"expected shape (n,{','.join(map(str, shape_tail))}), "
            f"got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("canonical source-mesh values must be finite")
    result = np.asarray(array, dtype="<f8").copy()
    result[result == 0.0] = 0.0
    return result


def _lexicographic_order(rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.empty(0, dtype=np.int64)
    return np.lexsort(
        tuple(rows[:, index] for index in reversed(range(rows.shape[1])))
    )


def canonical_source_mesh_arrays(
    vertex_coordinates_cm: Any,
    tetrahedron_vertex_coordinates_cm: Any,
    source_strengths_n_per_s: Any,
    volumes_cm3: Any,
) -> dict[str, Any]:
    """Fingerprint source semantics independent of MOAB handles and ordering."""
    vertices = _finite_float64(vertex_coordinates_cm, shape_tail=(3,))
    tetrahedra = _finite_float64(
        tetrahedron_vertex_coordinates_cm, shape_tail=(4, 3)
    )
    strengths = np.asarray(source_strengths_n_per_s, dtype="<f8").reshape(-1)
    volumes = np.asarray(volumes_cm3, dtype="<f8").reshape(-1)
    if len(tetrahedra) != len(strengths) or len(tetrahedra) != len(volumes):
        raise ValueError("tetrahedron, strength, and volume counts disagree")
    if not np.all(np.isfinite(strengths)) or not np.all(np.isfinite(volumes)):
        raise ValueError("source strengths and volumes must be finite")
    strengths = strengths.copy()
    volumes = volumes.copy()
    strengths[strengths == 0.0] = 0.0
    volumes[volumes == 0.0] = 0.0

    vertex_rows = vertices[_lexicographic_order(vertices)]
    vertex_order = np.lexsort(
        (
            tetrahedra[:, :, 2],
            tetrahedra[:, :, 1],
            tetrahedra[:, :, 0],
        ),
        axis=1,
    )
    sorted_vertices = np.take_along_axis(
        tetrahedra, vertex_order[:, :, None], axis=1
    )
    tet_rows = np.column_stack(
        (sorted_vertices.reshape((-1, 12)), strengths, volumes)
    )
    tet_rows = tet_rows[_lexicographic_order(tet_rows)]

    digest = hashlib.sha256()

    def update_section(label: str, values: np.ndarray) -> None:
        """Add an explicitly framed array section to the digest."""

        label_bytes = label.encode("ascii")
        array = np.asarray(values, dtype="<f8")
        digest.update(struct.pack("<Q", len(label_bytes)))
        digest.update(label_bytes)
        digest.update(struct.pack("<Q", array.ndim))
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(struct.pack("<Q", array.nbytes))
        digest.update(array.tobytes(order="C"))

    schema_bytes = SOURCE_MESH_IDENTITY_SCHEMA.encode("ascii")
    digest.update(struct.pack("<Q", len(schema_bytes)))
    digest.update(schema_bytes)
    update_section("vertices_cm", vertex_rows)
    update_section("tetrahedra_strength_volume", tet_rows)
    return {
        "schema": SOURCE_MESH_IDENTITY_SCHEMA,
        "canonical_fingerprint": digest.hexdigest(),
        "coordinate_units": "cm",
        "source_strength_units": "neutrons/s",
        "volume_units": "cm3",
        "vertex_count": int(len(vertices)),
        "tetrahedron_count": int(len(tetrahedra)),
        "total_source_strength_n_per_s": float(np.sum(strengths)),
        "total_tagged_volume_cm3": float(np.sum(volumes)),
        "bounding_box_cm": {
            "minimum": [float(value) for value in vertices.min(axis=0)],
            "maximum": [float(value) for value in vertices.max(axis=0)],
        },
    }


def source_mesh_fingerprint(path: str | Path) -> dict[str, Any]:
    """Load a ParaStell source H5M and return its semantic fingerprint."""
    from pymoab import core, types

    mesh = core.Core()
    mesh.load_file(str(Path(path).resolve()))
    root = mesh.get_root_set()
    vertices = mesh.get_entities_by_type(root, types.MBVERTEX)
    tetrahedra = mesh.get_entities_by_type(root, types.MBTET)
    vertex_coordinates = np.asarray(mesh.get_coords(vertices)).reshape((-1, 3))
    connectivity = mesh.get_connectivity(tetrahedra)
    tet_coordinates = np.asarray(mesh.get_coords(connectivity)).reshape(
        (-1, 4, 3)
    )
    source_tag = mesh.tag_get_handle("Source Strength")
    volume_tag = mesh.tag_get_handle("Volume")
    strengths = mesh.tag_get_data(source_tag, tetrahedra, flat=True)
    volumes = mesh.tag_get_data(volume_tag, tetrahedra, flat=True)
    return canonical_source_mesh_arrays(
        vertex_coordinates, tet_coordinates, strengths, volumes
    )
