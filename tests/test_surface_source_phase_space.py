from pathlib import Path
import hashlib

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


def _fixture_binding(histories=100):
    return {
        "kind": "serializer_fixture",
        "fixture_id": "unit-test-fixture",
        "source_histories": histories,
    }


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
        [path],
        source_histories=100,
        history_binding=_fixture_binding(),
        requested_surface_ids=[17, 18],
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
            [foreign],
            source_histories=100,
            history_binding=_fixture_binding(),
            requested_surface_ids=[17, 18],
        )

    bad = tmp_path / "bad-direction.h5"
    _write(bad)
    with h5py.File(bad, "r+") as source:
        records = source["source_bank"][:]
        records["u"]["x"][0] = 2.0
        source["source_bank"][:] = records
    with pytest.raises(ValueError, match="not unit vectors"):
        read_openmc16_surface_sources(
            [bad], source_histories=100, history_binding=_fixture_binding()
        )


def test_missing_field_duplicate_path_and_bad_histories_fail_closed(tmp_path):
    path = tmp_path / "bank.h5"
    _write(path)
    with pytest.raises(ValueError, match="positive integer"):
        read_openmc16_surface_sources(
            [path], source_histories=0, history_binding=_fixture_binding(0)
        )
    with pytest.raises(ValueError, match="paths must be unique"):
        read_openmc16_surface_sources(
            [path, path],
            source_histories=1,
            history_binding=_fixture_binding(1),
        )

    incomplete = tmp_path / "incomplete.h5"
    with h5py.File(incomplete, "x") as target:
        target.attrs["filetype"] = np.bytes_(b"source")
        target.attrs["version"] = np.asarray([18, 2], dtype=int)
        target.create_dataset(
            "source_bank", data=np.zeros(1, dtype=[("E", "f8")])
        )
    with pytest.raises(ValueError, match="omits"):
        read_openmc16_surface_sources(
            [incomplete],
            source_histories=1,
            history_binding=_fixture_binding(1),
        )


def test_history_binding_and_exact_file_format_are_required(tmp_path):
    path = tmp_path / "bank.h5"
    _write(path)
    with pytest.raises(ValueError, match="history_binding"):
        read_openmc16_surface_sources(
            [path], source_histories=100, history_binding={}
        )
    model = tmp_path / "model.xml"
    model.write_text(
        "<model><settings><particles>20</particles><batches>4</batches>"
        "<seed>17</seed></settings></model>"
    )
    statepoint = tmp_path / "statepoint.h5"
    with h5py.File(statepoint, "x") as target:
        target.attrs["filetype"] = np.bytes_(b"statepoint")
        target.attrs["openmc_version"] = np.asarray([0, 16, 0])
        target.create_dataset("n_particles", data=20)
        target.create_dataset("n_batches", data=4)
        target.create_dataset("seed", data=17)
        target.create_dataset("run_mode", data=np.bytes_(b"fixed source"))
    digest = lambda value: hashlib.sha256(value.read_bytes()).hexdigest()
    fixed = {
        "kind": "fixed_source_run",
        "run_id": "run-a",
        "settings_payload_path": str(model),
        "settings_payload_sha256": digest(model),
        "statepoint_path": str(statepoint),
        "statepoint_sha256": digest(statepoint),
    }
    with pytest.raises(ValueError, match="particles times batches"):
        read_openmc16_surface_sources(
            [path], source_histories=100, history_binding=fixed
        )
    _, manifest = read_openmc16_surface_sources(
        [path], source_histories=80, history_binding=fixed
    )
    assert manifest["history_binding"]["statepoint_sha256"] == digest(
        statepoint
    )
    with h5py.File(path, "r+") as source:
        source.attrs["version"] = np.asarray([19, 0], dtype=int)
    with pytest.raises(ValueError, match="unsupported"):
        read_openmc16_surface_sources(
            [path], source_histories=100, history_binding=_fixture_binding()
        )
