from pathlib import Path

import h5py
import numpy as np
import pytest

from parastell.surface_source_phase_space import (
    OPENMC16_PHASE_FIELDS,
    read_openmc16_surface_sources,
)


VECTOR = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
SOURCE_DTYPE = np.dtype(
    [
        ("r", VECTOR),
        ("u", VECTOR),
        ("E", "<f8"),
        ("time", "<f8"),
        ("wgt", "<f8"),
        ("delayed_group", "<i4"),
        ("surf_id", "<i4"),
        ("particle", "<i4"),
    ]
)


def _write(path: Path, surface_ids=(17, 18)):
    records = np.zeros(2, dtype=SOURCE_DTYPE)
    records["r"]["x"] = [1.0, 2.0]
    records["r"]["y"] = [3.0, 4.0]
    records["r"]["z"] = [5.0, 6.0]
    records["u"]["x"] = [1.0, 0.0]
    records["u"]["y"] = [0.0, 1.0]
    records["E"] = [14.1e6, 1.0e6]
    records["time"] = [1.0e-9, 2.0e-9]
    records["wgt"] = [1.0, 0.25]
    records["delayed_group"] = [0, 2]
    records["surf_id"] = surface_ids
    records["particle"] = [2112, 22]
    with h5py.File(path, "x") as target:
        target.attrs["filetype"] = np.bytes_(b"source")
        target.attrs["version"] = np.asarray([18, 2], dtype=int)
        target.create_dataset("source_bank", data=records)


def test_all_openmc_phase_fields_roundtrip_without_conditioning(tmp_path):
    path = tmp_path / "bank.h5"
    _write(path)
    columns, manifest = read_openmc16_surface_sources(
        [path], source_histories=100, requested_surface_ids=[17, 18]
    )
    assert manifest["openmc_surface_bank_fields_required"] == list(
        OPENMC16_PHASE_FIELDS
    )
    assert np.array_equal(columns["position_global_cm"][0], [1, 3, 5])
    assert np.array_equal(columns["direction_global"], [[1, 0, 0], [0, 1, 0]])
    assert np.array_equal(columns["energy_eV"], [14.1e6, 1.0e6])
    assert np.array_equal(columns["time_s"], [1.0e-9, 2.0e-9])
    assert np.array_equal(columns["particle_pdg"], [2112, 22])
    assert np.array_equal(columns["delayed_group"], [0, 2])
    assert np.array_equal(columns["surface_id"], [17, 18])
    assert np.array_equal(columns["openmc_weight"], [1.0, 0.25])
    assert np.array_equal(columns["weight_per_source_history"], [0.01, 0.0025])
    assert manifest["normalization"]["tally_conditioning"] is False


def test_foreign_surface_and_nonunit_direction_fail_closed(tmp_path):
    foreign = tmp_path / "foreign.h5"
    _write(foreign, surface_ids=(17, 999))
    with pytest.raises(ValueError, match="outside the requested envelope"):
        read_openmc16_surface_sources(
            [foreign], source_histories=100, requested_surface_ids=[17, 18]
        )

    bad = tmp_path / "bad-direction.h5"
    _write(bad)
    with h5py.File(bad, "r+") as source:
        records = source["source_bank"][:]
        records["u"]["x"][0] = 2.0
        source["source_bank"][:] = records
    with pytest.raises(ValueError, match="not unit vectors"):
        read_openmc16_surface_sources([bad], source_histories=100)


def test_missing_field_duplicate_path_and_bad_histories_fail_closed(tmp_path):
    path = tmp_path / "bank.h5"
    _write(path)
    with pytest.raises(ValueError, match="positive integer"):
        read_openmc16_surface_sources([path], source_histories=0)
    with pytest.raises(ValueError, match="paths must be unique"):
        read_openmc16_surface_sources([path, path], source_histories=1)

    incomplete = tmp_path / "incomplete.h5"
    with h5py.File(incomplete, "x") as target:
        target.attrs["filetype"] = np.bytes_(b"source")
        target.attrs["version"] = np.asarray([18, 2], dtype=int)
        target.create_dataset(
            "source_bank", data=np.zeros(1, dtype=[("E", "f8")])
        )
    with pytest.raises(ValueError, match="omits"):
        read_openmc16_surface_sources([incomplete], source_histories=1)
