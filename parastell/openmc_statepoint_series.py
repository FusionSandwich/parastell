"""Hash-bind and qualify periodic OpenMC 0.16 statepoint checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

import h5py
import numpy as np


SCHEMA = "parastell.openmc16_statepoint_series/v1.0.0"
_STATEPOINT_NAME = re.compile(r"^statepoint\.(\d+)\.h5$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def qualify_statepoint_series(
    statepoint_paths: Iterable[str | Path],
    *,
    expected_batches: Iterable[int],
    particles_per_batch: int,
    total_batches: int,
    seed: int,
) -> dict[str, Any]:
    """Validate every requested intermediate checkpoint and the final file."""
    expected = [int(value) for value in expected_batches]
    if (
        not expected
        or expected != sorted(set(expected))
        or min(expected) <= 0
        or expected[-1] != int(total_batches)
        or min(int(particles_per_batch), int(total_batches), int(seed)) <= 0
    ):
        raise ValueError("expected statepoint schedule is invalid")
    paths = [Path(value).resolve(strict=True) for value in statepoint_paths]
    if len(paths) != len(set(paths)):
        raise ValueError("statepoint paths must be unique")
    by_batch = {}
    for path in paths:
        match = _STATEPOINT_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(
                f"invalid OpenMC statepoint filename: {path.name}"
            )
        batch = int(match.group(1))
        if batch in by_batch:
            raise ValueError(f"duplicate statepoint batch: {batch}")
        by_batch[batch] = path
    if sorted(by_batch) != expected:
        raise ValueError(
            "observed statepoint batches do not match the schedule"
        )

    rows = []
    for batch in expected:
        path = by_batch[batch]
        with h5py.File(path, "r") as statepoint:
            version = tuple(
                int(value)
                for value in np.asarray(
                    statepoint.attrs.get("openmc_version", ()), dtype=int
                ).reshape(-1)
            )
            metadata = {
                "filetype": _text(statepoint.attrs.get("filetype", "")),
                "openmc_version": list(version),
                "run_mode": _text(statepoint["run_mode"][()]),
                "particles_per_batch": int(statepoint["n_particles"][()]),
                "total_batches": int(statepoint["n_batches"][()]),
                "current_batch": int(statepoint["current_batch"][()]),
                "seed": int(statepoint["seed"][()]),
            }
        valid = bool(
            metadata["filetype"] == "statepoint"
            and tuple(metadata["openmc_version"][:3]) == (0, 16, 0)
            and metadata["run_mode"] == "fixed source"
            and metadata["particles_per_batch"] == particles_per_batch
            and metadata["total_batches"] == total_batches
            and metadata["current_batch"] == batch
            and metadata["seed"] == seed
        )
        if not valid:
            raise ValueError(f"statepoint metadata mismatch at batch {batch}")
        rows.append(
            {
                "batch": batch,
                "cumulative_source_histories": batch * particles_per_batch,
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "metadata": metadata,
                "final_statepoint": batch == total_batches,
            }
        )
    return {
        "schema": SCHEMA,
        "status": "STATEPOINT_SERIES_COMPLETE",
        "expected_batches": expected,
        "observed_batches": [row["batch"] for row in rows],
        "particles_per_batch": particles_per_batch,
        "total_batches": total_batches,
        "seed": seed,
        "final_batch_included": rows[-1]["final_statepoint"],
        "statepoints": rows,
        "poster_checkpoint_files_available": True,
        "production_run_authorized": False,
    }
