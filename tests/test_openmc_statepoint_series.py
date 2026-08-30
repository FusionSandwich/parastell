from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from parastell.openmc_statepoint_series import qualify_statepoint_series


def _statepoint(path: Path, *, current_batch: int, total_batches: int = 10):
    with h5py.File(path, "x") as target:
        target.attrs["filetype"] = np.bytes_("statepoint")
        target.attrs["openmc_version"] = np.asarray([0, 16, 0])
        target["run_mode"] = np.bytes_("fixed source")
        target["n_particles"] = 100
        target["n_batches"] = total_batches
        target["current_batch"] = current_batch
        target["seed"] = 17


def test_periodic_statepoint_series_preserves_intermediate_and_final_files(
    tmp_path,
):
    paths = []
    for batch in (3, 6, 9, 10):
        path = tmp_path / f"statepoint.{batch}.h5"
        _statepoint(path, current_batch=batch)
        paths.append(path)
    result = qualify_statepoint_series(
        paths,
        expected_batches=[3, 6, 9, 10],
        particles_per_batch=100,
        total_batches=10,
        seed=17,
    )
    assert result["status"] == "STATEPOINT_SERIES_COMPLETE"
    assert [
        row["cumulative_source_histories"] for row in result["statepoints"]
    ] == [
        300,
        600,
        900,
        1000,
    ]
    assert result["statepoints"][-1]["final_statepoint"] is True
    assert all(len(row["sha256"]) == 64 for row in result["statepoints"])


def test_statepoint_series_fails_on_missing_or_misbound_checkpoint(tmp_path):
    first = tmp_path / "statepoint.5.h5"
    final = tmp_path / "statepoint.10.h5"
    _statepoint(first, current_batch=4)
    _statepoint(final, current_batch=10)
    with pytest.raises(ValueError, match="metadata mismatch"):
        qualify_statepoint_series(
            [first, final],
            expected_batches=[5, 10],
            particles_per_batch=100,
            total_batches=10,
            seed=17,
        )
    with pytest.raises(ValueError, match="do not match"):
        qualify_statepoint_series(
            [final],
            expected_batches=[5, 10],
            particles_per_batch=100,
            total_batches=10,
            seed=17,
        )
