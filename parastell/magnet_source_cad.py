"""Immutable, provenance-rich export of one ParaStell filament magnet.

This module deliberately exports the CadQuery solids retained by
``MagnetSetFromFilaments``.  It does not reconstruct a centreline host and it
does not infer transport volume identities from list order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np

from .magnet_radiation_field import cad_solid_boundary_signature

SCHEMA = "parastell.selected_magnet_source_cad/v1.0.0"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_transform(
    transform: Sequence[Sequence[float]],
) -> list[list[float]]:
    array = np.asarray(transform, dtype=float)
    if array.shape != (4, 4) or np.any(~np.isfinite(array)):
        raise ValueError("global_transform must be a finite 4x4 matrix")
    if not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12):
        raise ValueError("global_transform must be affine homogeneous")
    linear = array[:3, :3]
    if abs(float(np.linalg.det(linear))) <= 1.0e-12:
        raise ValueError("global_transform must be invertible")
    return [[float(value) for value in row] for row in array]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_selected_magnet_source_cad(
    magnet_set,
    *,
    coil_index: int,
    source_filament_index: int,
    output_dir: str | Path,
    magnet_id: str,
    casing_volume_id: int,
    winding_pack_volume_id: int,
    coils_path: str | Path,
    source_revision: str,
    target_geometry_fingerprint: str,
    global_transform: Sequence[Sequence[float]],
    source_parameters: Mapping[str, Any],
    filename_prefix: str | None = None,
    overwrite: bool = False,
    boundary_signature_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export one authoritative casing/winding-pack CAD pair.

    The caller must provide the transport identities explicitly.  The result
    is an *unqualified source-CAD candidate* until a separate comparison binds
    its boundary to the target DAGMC volumes.  This distinction prevents a
    successful file export from being mistaken for geometry equivalence.
    """
    if not isinstance(coil_index, int) or coil_index < 0:
        raise ValueError("coil_index must be a non-negative integer")
    if not isinstance(source_filament_index, int) or source_filament_index < 0:
        raise ValueError(
            "source_filament_index must be a non-negative integer"
        )
    if not str(magnet_id).strip():
        raise ValueError("magnet_id is required")
    if not str(source_revision).strip():
        raise ValueError("source_revision is required")
    if not str(target_geometry_fingerprint).strip():
        raise ValueError("target_geometry_fingerprint is required")
    volume_ids = (int(casing_volume_id), int(winding_pack_volume_id))
    if any(volume_id <= 0 for volume_id in volume_ids):
        raise ValueError("DAGMC volume IDs must be positive")
    if volume_ids[0] == volume_ids[1]:
        raise ValueError("casing and winding-pack volume IDs must be distinct")

    coils_file = Path(coils_path).resolve()
    if not coils_file.is_file():
        raise FileNotFoundError(coils_file)
    if not getattr(magnet_set, "has_casing", False):
        raise ValueError("selected magnet export requires an explicit casing")
    groups = getattr(magnet_set, "coil_solids", None)
    coils = getattr(magnet_set, "magnet_coils", None)
    if groups is None or coils is None or len(groups) != len(coils):
        raise ValueError("built coil solids and source filaments do not align")
    if coil_index >= len(groups):
        raise IndexError("coil_index is outside the built magnet set")
    solids = tuple(groups[coil_index])
    if len(solids) != 2:
        raise ValueError(
            "selected filament magnet must contain casing and winding pack"
        )

    transform = _normalise_transform(global_transform)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or str(magnet_id)
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("filename_prefix must be one path-safe filename stem")
    manifest_path = destination / f"{prefix}.source-cad.json"
    artifact_paths = {
        "casing_step": destination / f"{prefix}.casing.step",
        "casing_brep": destination / f"{prefix}.casing.brep",
        "winding_pack_step": destination / f"{prefix}.winding-pack.step",
        "winding_pack_brep": destination / f"{prefix}.winding-pack.brep",
    }
    targets = (manifest_path, *artifact_paths.values())
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "selected magnet source-CAD export would overwrite: "
            + ", ".join(existing)
        )

    signature_options = dict(boundary_signature_options or {})
    role_records = {}
    created = []
    try:
        for role, solid, volume_id in (
            ("casing", solids[0], volume_ids[0]),
            ("winding_pack", solids[1], volume_ids[1]),
        ):
            step_path = artifact_paths[f"{role}_step"]
            brep_path = artifact_paths[f"{role}_brep"]
            cq.exporters.export(solid, str(step_path))
            created.append(step_path)
            solid.exportBrep(str(brep_path))
            created.append(brep_path)
            role_records[role] = {
                "dagmc_volume_id": volume_id,
                "step": {
                    "path": step_path.name,
                    "sha256": _sha256(step_path),
                    "size_bytes": step_path.stat().st_size,
                },
                "brep": {
                    "path": brep_path.name,
                    "sha256": _sha256(brep_path),
                    "size_bytes": brep_path.stat().st_size,
                },
                "cad_boundary_signature": cad_solid_boundary_signature(
                    solid, **signature_options
                ),
            }

        coil = coils[coil_index]
        coordinates = np.asarray(coil.coords, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                "selected source filament coordinates are invalid"
            )
        manifest = {
            "schema": SCHEMA,
            "qualification": {
                "status": "UNQUALIFIED_UNTIL_DAGMC_BOUNDARY_COMPARISON",
                "meaning": (
                    "authoritative pre-export ParaStell solids; target DAGMC "
                    "identity is asserted by the caller and not yet proven"
                ),
            },
            "magnet_id": str(magnet_id),
            "built_coil_index": coil_index,
            "source_filament_index": source_filament_index,
            "source": {
                "kind": "ParaStell MagnetSetFromFilaments retained CAD",
                "revision": str(source_revision),
                "coils_path_local_identifier": coils_file.name,
                "coils_sha256": _sha256(coils_file),
                "centreline_point_count": len(coordinates),
                "parameters": dict(source_parameters),
            },
            "coordinate_contract": {
                "cad_length_unit": "cm",
                "global_transform": transform,
                "global_transform_inverse": np.linalg.inv(
                    np.asarray(transform, dtype=float)
                ).tolist(),
            },
            "target_dagmc": {
                "canonical_geometry_fingerprint": str(
                    target_geometry_fingerprint
                ),
                "role_volume_ids": {
                    "casing": volume_ids[0],
                    "winding_pack": volume_ids[1],
                },
            },
            "artifacts": role_records,
        }
        _write_json(manifest_path, manifest)
        created.append(manifest_path)
        return {
            **manifest,
            "manifest": {
                "path": manifest_path.name,
                "sha256": _sha256(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
            },
        }
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
