"""Strict, geometry-neutral ingestion of OpenMC 0.16 surface-source banks.

OpenMC stores the transport phase-space variables at a surface crossing.  A
later DAGMC localization step adds facet identity, barycentric coordinates,
and the topology-oriented outward normal.  Keeping those two operations
separate makes this reader reusable for different plasma equilibria and coil
sets without weakening either contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


OPENMC16_PHASE_FIELDS = (
    "r",
    "u",
    "E",
    "time",
    "wgt",
    "delayed_group",
    "surf_id",
    "particle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(records: np.ndarray, field: str) -> np.ndarray:
    values = records[field]
    names = values.dtype.names
    if names is None or not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"OpenMC field {field!r} is not an x/y/z vector")
    return np.column_stack([values[axis] for axis in "xyz"]).astype(
        float, copy=False
    )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def read_openmc16_surface_sources(
    paths: Sequence[str | Path],
    *,
    source_histories: int,
    requested_surface_ids: Iterable[int] | None = None,
    direction_tolerance: float = 1.0e-12,
) -> tuple[dict[str, np.ndarray], dict]:
    """Read and validate lossless OpenMC 0.16 crossing phase space.

    The canonical statistical weight is the OpenMC record weight divided only
    by the exact number of source histories.  No tally conditioning or
    resampling is performed here.
    """

    histories = _positive_integer(source_histories, "source_histories")
    source_paths = tuple(Path(path).resolve() for path in paths)
    if not source_paths:
        raise ValueError("at least one surface-source bank is required")
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("surface-source bank paths must be unique")

    requested = None
    if requested_surface_ids is not None:
        requested = tuple(int(value) for value in requested_surface_ids)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError(
                "requested_surface_ids must be nonempty and unique"
            )

    arrays = []
    file_indices = []
    record_indices = []
    file_rows = []
    for file_index, path in enumerate(source_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as source:
            filetype = source.attrs.get("filetype")
            if isinstance(filetype, bytes):
                filetype = filetype.decode()
            elif isinstance(filetype, np.bytes_):
                filetype = filetype.tobytes().decode()
            if filetype != "source":
                raise ValueError(f"{path} is not an OpenMC source file")
            file_format_version = np.asarray(
                source.attrs.get("version", ()), dtype=int
            ).reshape(-1)
            if file_format_version.shape != (2,):
                raise ValueError(
                    f"{path} omits its OpenMC source format version"
                )
            if "source_bank" not in source:
                raise ValueError(f"{path} has no source_bank dataset")
            dataset = source["source_bank"]
            fields = dataset.dtype.names
            if fields is None:
                raise ValueError(f"{path} source_bank is not structured")
            missing = set(OPENMC16_PHASE_FIELDS) - set(fields)
            if missing:
                raise ValueError(f"{path} source_bank omits {sorted(missing)}")
            records = dataset[:]
        arrays.append(records)
        file_indices.append(np.full(len(records), file_index, dtype=np.int64))
        record_indices.append(np.arange(len(records), dtype=np.int64))
        file_rows.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "record_count": int(len(records)),
                "dtype_fields": list(fields),
                "openmc_source_format_version": file_format_version.tolist(),
            }
        )

    records = np.concatenate(arrays)
    source_file_index = np.concatenate(file_indices)
    source_record_index = np.concatenate(record_indices)
    position = _vector(records, "r")
    direction = _vector(records, "u")
    energy = np.asarray(records["E"], dtype=float).reshape(-1)
    time = np.asarray(records["time"], dtype=float).reshape(-1)
    openmc_weight = np.asarray(records["wgt"], dtype=float).reshape(-1)
    delayed_group = np.asarray(
        records["delayed_group"], dtype=np.int64
    ).reshape(-1)
    surface_id = np.asarray(records["surf_id"], dtype=np.int64).reshape(-1)
    particle_pdg = np.asarray(records["particle"], dtype=np.int64).reshape(-1)

    finite_arrays = {
        "position_global_cm": position,
        "direction_global": direction,
        "energy_eV": energy,
        "time_s": time,
        "openmc_weight": openmc_weight,
    }
    for name, values in finite_arrays.items():
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} contains nonfinite values")
    norms = np.linalg.norm(direction, axis=1)
    if np.any(np.abs(norms - 1.0) > float(direction_tolerance)):
        raise ValueError("OpenMC directions are not unit vectors")
    if np.any(energy <= 0.0):
        raise ValueError("OpenMC energies must be positive")
    if np.any(time < 0.0):
        raise ValueError("OpenMC flight times cannot be negative")
    if np.any(openmc_weight < 0.0):
        raise ValueError("OpenMC weights cannot be negative")
    if np.any(surface_id <= 0):
        raise ValueError("surface IDs must be positive")
    if np.any(particle_pdg == 0):
        raise ValueError("particle PDG identifiers cannot be zero")

    if requested is not None:
        foreign = set(surface_id.tolist()) - set(requested)
        if foreign:
            raise ValueError(
                "surface bank contains crossings outside the requested "
                f"envelope: {sorted(foreign)}"
            )

    columns = {
        "position_global_cm": position,
        "direction_global": direction,
        "energy_eV": energy,
        "time_s": time,
        "openmc_weight": openmc_weight,
        "weight_per_source_history": openmc_weight / histories,
        "delayed_group": delayed_group,
        "surface_id": surface_id,
        "particle_pdg": particle_pdg,
        "source_file_index": source_file_index,
        "source_record_index": source_record_index,
    }
    manifest = {
        "schema": "parastell.openmc16_surface_phase_space/v1.0.0",
        "openmc_surface_bank_fields_required": list(OPENMC16_PHASE_FIELDS),
        "source_histories": histories,
        "requested_surface_ids": (
            None if requested is None else sorted(requested)
        ),
        "record_count": int(len(records)),
        "source_files": file_rows,
        "normalization": {
            "canonical_weight": "openmc_weight / source_histories",
            "tally_conditioning": False,
            "sampling_or_resampling": False,
        },
        "raw_phase_space_pass": True,
        "derived_topology_fields_pending": [
            "facet_id",
            "barycentric_coordinates",
            "outward_normal_global",
            "mu",
            "crossing_sense",
            "position_local_cm",
            "direction_local",
        ],
    }
    return columns, manifest
