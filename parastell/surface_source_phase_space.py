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
from typing import Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

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


def _verify_history_binding(
    history_binding: Mapping[str, object], histories: int
) -> dict:
    kind = history_binding.get("kind")
    if kind == "serializer_fixture":
        declared = _positive_integer(
            history_binding.get("source_histories"), "bound source_histories"
        )
        fixture_id = str(history_binding.get("fixture_id", "")).strip()
        if declared != histories or not fixture_id:
            raise ValueError(
                "serializer fixture history binding is inconsistent"
            )
        return {
            "kind": kind,
            "fixture_id": fixture_id,
            "source_histories": declared,
        }
    if kind != "fixed_source_run":
        raise ValueError("unknown history_binding kind")

    run_id = str(history_binding.get("run_id", "")).strip()
    settings_path = Path(
        str(history_binding.get("settings_payload_path", ""))
    ).resolve()
    statepoint_path = Path(
        str(history_binding.get("statepoint_path", ""))
    ).resolve()
    if (
        not run_id
        or not settings_path.is_file()
        or not statepoint_path.is_file()
    ):
        raise ValueError(
            "fixed-source history binding paths/run_id are invalid"
        )
    settings_hash = _sha256(settings_path)
    statepoint_hash = _sha256(statepoint_path)
    if (
        settings_hash
        != str(history_binding.get("settings_payload_sha256", "")).lower()
        or statepoint_hash
        != str(history_binding.get("statepoint_sha256", "")).lower()
    ):
        raise ValueError("fixed-source history binding hash mismatch")

    root = ET.parse(settings_path).getroot()
    settings = root if root.tag == "settings" else root.find("settings")
    if settings is None:
        raise ValueError("settings payload has no settings element")
    particles = _positive_integer(
        int(settings.findtext("particles", "0")), "settings particles"
    )
    batches = _positive_integer(
        int(settings.findtext("batches", "0")), "settings batches"
    )
    seed = _positive_integer(
        int(settings.findtext("seed", "0")), "settings seed"
    )
    if histories != particles * batches:
        raise ValueError(
            "source_histories disagrees with fixed-source particles times batches"
        )
    with h5py.File(statepoint_path, "r") as statepoint:
        filetype = statepoint.attrs.get("filetype")
        if isinstance(filetype, (bytes, np.bytes_)):
            filetype = bytes(filetype).decode()
        version = tuple(
            np.asarray(
                statepoint.attrs.get("openmc_version", ()), dtype=int
            ).tolist()
        )
        run_mode = statepoint["run_mode"][()]
        if isinstance(run_mode, (bytes, np.bytes_)):
            run_mode = bytes(run_mode).decode()
        state_values = {
            "particles": int(statepoint["n_particles"][()]),
            "batches": int(statepoint["n_batches"][()]),
            "seed": int(statepoint["seed"][()]),
        }
    if (
        filetype != "statepoint"
        or version != (0, 16, 0)
        or run_mode != "fixed source"
        or state_values
        != {"particles": particles, "batches": batches, "seed": seed}
    ):
        raise ValueError(
            "statepoint disagrees with exact OpenMC 0.16 settings"
        )
    return {
        "kind": kind,
        "run_id": run_id,
        "source_histories": histories,
        "particles_per_batch": particles,
        "batches": batches,
        "seed": seed,
        "settings_payload_path": str(settings_path),
        "settings_payload_sha256": settings_hash,
        "statepoint_path": str(statepoint_path),
        "statepoint_sha256": statepoint_hash,
        "openmc_version": "0.16.0",
    }


def read_openmc16_surface_sources(
    paths: Sequence[str | Path],
    *,
    source_histories: int,
    history_binding: Mapping[str, object],
    requested_surface_ids: Iterable[int] | None = None,
    direction_tolerance: float = 1.0e-12,
) -> tuple[dict[str, np.ndarray], dict]:
    """Read and validate lossless OpenMC 0.16 crossing phase space.

    The canonical statistical weight is the OpenMC record weight divided only
    by the exact number of source histories.  No tally conditioning or
    resampling is performed here.
    """

    histories = _positive_integer(source_histories, "source_histories")
    if not isinstance(history_binding, Mapping):
        raise ValueError("history_binding is required")
    verified_history_binding = _verify_history_binding(
        history_binding, histories
    )
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
            if tuple(file_format_version.tolist()) != (18, 2):
                raise ValueError(
                    f"{path} has unsupported OpenMC source format "
                    f"{file_format_version.tolist()}"
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
        "history_binding": verified_history_binding,
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
        "native_field_limitations": {
            "parent_history_id": (
                "not stored by the OpenMC 0.16 native surface-source format; "
                "source_file_index plus source_record_index provide stable "
                "record identity but not cross-record ancestry"
            ),
            "polarization": "not represented by OpenMC 0.16 SourceParticle",
        },
    }
    return columns, manifest
