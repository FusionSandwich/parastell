from __future__ import annotations

import h5py
import numpy as np
import pytest

from parastell.openmc16_response_results import (
    read_openmc16_response_set,
    read_openmc16_tally,
    tally_result_to_domain_estimators,
    validate_response_coverage,
)


def _statepoint(path, *, names=("damage",), realizations=3):
    with h5py.File(path, "w") as handle:
        handle.attrs["filetype"] = np.bytes_("statepoint")
        handle.attrs["version"] = [18, 2]
        handle.attrs["openmc_version"] = [0, 16, 0]
        handle["run_mode"] = np.bytes_("fixed source")
        handle["n_particles"] = 10
        handle["n_batches"] = realizations
        handle["current_batch"] = realizations
        handle["seed"] = 19
        tallies = handle.create_group("tallies")
        filters = tallies.create_group("filters")
        cell = filters.create_group("filter 1")
        cell["type"] = np.bytes_("cell")
        cell["n_bins"] = 1
        cell["bins"] = [9008]
        energy = filters.create_group("filter 2")
        energy["type"] = np.bytes_("energy")
        energy["n_bins"] = 2
        energy["bins"] = [0.0, 1.0e6, 2.0e7]
        for index, name in enumerate(names, start=20):
            tally = tallies.create_group(f"tally {index}")
            tally["name"] = np.bytes_(name)
            tally["filters"] = [1, 2]
            tally["score_bins"] = [np.bytes_("damage-energy")]
            tally["nuclides"] = [np.bytes_("W184")]
            tally["estimator"] = np.bytes_("tracklength")
            tally["n_realizations"] = realizations
            means = np.asarray([2.0, 4.0])
            deviations = np.asarray([0.2, 0.4])
            sums = realizations * means
            squares = realizations * (
                means**2 + deviations**2 * (realizations - 1)
            )
            tally["results"] = np.stack([sums, squares], axis=-1)[
                :, np.newaxis, :
            ]


def test_generic_reader_preserves_filters_nuclides_scores_and_uncertainty(
    tmp_path,
):
    path = tmp_path / "statepoint.h5"
    _statepoint(path)
    result = read_openmc16_tally(path, "damage")
    assert result["scores"] == ["damage-energy"]
    assert result["nuclides"] == ["W184"]
    assert [item["type"] for item in result["filters"]] == ["cell", "energy"]
    assert result["mean_per_source"] == [[[[2.0]], [[4.0]]]]
    assert result["covariance"]["status"] == "UNAVAILABLE"
    assert result["source_histories"] == 30
    rows = tally_result_to_domain_estimators(
        result,
        domains_by_cell_id={
            9008: {"domain_id": "magnet-alpha", "volume_cm3": 12.0}
        },
    )
    assert len(rows) == 1
    assert rows[0]["observable"] == "damage_energy"
    assert rows[0]["mean_per_source"] == [2.0, 4.0]
    assert rows[0]["unit"] == "eV/source_history/bin"


def test_response_set_coverage_reports_missing_as_missing_not_zero(tmp_path):
    path = tmp_path / "statepoint.h5"
    _statepoint(path)
    response_set = read_openmc16_response_set(path, ["damage"])
    coverage = validate_response_coverage(response_set, ["damage", "gas"])
    assert coverage["status"] == "INCOMPLETE"
    assert coverage["missing_tallies"] == ["gas"]
    assert coverage["missing_response_semantics"] == "MISSING_IS_NOT_ZERO"


def test_generic_reader_needs_multiple_realizations(tmp_path):
    path = tmp_path / "statepoint.h5"
    _statepoint(path, realizations=1)
    with pytest.raises(ValueError, match="at least two"):
        read_openmc16_tally(path, "damage")
