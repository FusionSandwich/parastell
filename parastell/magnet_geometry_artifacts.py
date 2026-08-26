"""Deterministic external artifacts for magnet geometry interchange records."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

import numpy as np


_BINARY_STL_HEADER = (
    b"ParaStell deterministic magnet envelope; coordinates in centimetres"
).ljust(80, b" ")


def _canonical_oriented_triangles(triangles_cm: Any) -> np.ndarray:
    """Canonicalize triangle starts/order without reversing their winding."""
    triangles = np.asarray(triangles_cm, dtype=np.float64)
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or len(triangles) == 0
        or not np.all(np.isfinite(triangles))
    ):
        raise ValueError("triangles_cm must be a nonempty finite (N, 3, 3) array")
    canonical = []
    for triangle in triangles:
        starts = [tuple(triangle[index]) for index in range(3)]
        start = min(range(3), key=starts.__getitem__)
        canonical.append(np.roll(triangle, -start, axis=0))
    canonical.sort(key=lambda value: tuple(value.ravel()))
    result = np.asarray(canonical, dtype=np.float64)
    cross = np.cross(result[:, 1] - result[:, 0], result[:, 2] - result[:, 0])
    magnitude = np.linalg.norm(cross, axis=1)
    if np.any(~np.isfinite(magnitude)) or np.any(magnitude <= 0.0):
        raise ValueError("magnet envelope contains a degenerate triangle")
    return result


def write_binary_stl(path: str | Path, triangles_cm: Any) -> dict[str, Any]:
    """Write an oriented envelope as deterministic binary STL.

    STL does not encode units.  The owning geometry interchange records that
    coordinates are centimetres; this function deliberately performs no unit
    conversion.
    """
    triangles = _canonical_oriented_triangles(triangles_cm)
    if len(triangles) > (2**32 - 1):
        raise ValueError("binary STL triangle count exceeds uint32 capacity")
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = cross / np.linalg.norm(cross, axis=1)[:, None]
    output = Path(path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("xb") as stream:
        header = _BINARY_STL_HEADER + struct.pack("<I", len(triangles))
        stream.write(header)
        digest.update(header)
        for normal, triangle in zip(normals, triangles):
            record = struct.pack(
                "<12fH",
                *normal.astype(np.float32),
                *triangle.astype(np.float32).ravel(),
                0,
            )
            stream.write(record)
            digest.update(record)
    return {
        "status": "available",
        "sha256": digest.hexdigest(),
        "size_bytes": output.stat().st_size,
        "triangle_count": int(len(triangles)),
        "coordinate_units": "cm",
        "format": "binary_stl",
        "path": str(output.resolve()),
    }


def export_dagmc_envelope_stl(path: str | Path, envelope: Any) -> dict[str, Any]:
    """Export all outward-oriented faceted surfaces in one DAGMC envelope."""
    surfaces = sorted(
        tuple(envelope.faceted_surfaces), key=lambda value: value.surface_id
    )
    if not surfaces:
        raise ValueError("DAGMC envelope has no faceted surfaces")
    result = write_binary_stl(
        path,
        np.concatenate(
            [np.asarray(value.triangles_cm, dtype=float) for value in surfaces]
        ),
    )
    result["envelope_id"] = str(envelope.envelope.envelope_id)
    result["surface_ids"] = [int(value.surface_id) for value in surfaces]
    return result
