from __future__ import annotations

import h5py
import numpy as np
import pytest

from parastell.openmc16_volume_results import (
    read_cell_scalar_flux_spectrum,
    scalar_flux_estimators,
)


def _statepoint(path, *, version=(0, 16, 0), score="flux", realizations=2):
    means = np.asarray([[2.0, 4.0], [6.0, 8.0]])
    deviations = np.asarray([[0.5, 1.0], [1.5, 2.0]])
    sums = realizations * means
    squares = realizations * (means**2 + deviations**2 * (realizations - 1))
    results = np.stack([sums.reshape(-1), squares.reshape(-1)], axis=-1)
    results = results[:, np.newaxis, :]
    with h5py.File(path, "w") as handle:
        handle.attrs["filetype"] = np.bytes_("statepoint")
        handle.attrs["version"] = [18, 2]
        handle.attrs["openmc_version"] = version
        handle["run_mode"] = np.bytes_("fixed source")
        handle["n_particles"] = 100
        handle["n_batches"] = 2
        handle["current_batch"] = 2
        handle["seed"] = 17
        tallies = handle.create_group("tallies")
        filters = tallies.create_group("filters")
        cell = filters.create_group("filter 1")
        cell["type"] = np.bytes_("cell")
        cell["n_bins"] = 2
        cell["bins"] = [9, 10]
        particle = filters.create_group("filter 2")
        particle["type"] = np.bytes_("particle")
        particle["n_bins"] = 1
        particle["bins"] = [np.bytes_("neutron")]
        energy = filters.create_group("filter 3")
        energy["type"] = np.bytes_("energy")
        energy["n_bins"] = 2
        energy["bins"] = [0.0, 1.0e6, 2.0e7]
        tally = tallies.create_group("tally 17")
        tally["name"] = np.bytes_("pstl_magnet_neutron_volume_flux")
        tally["filters"] = [1, 2, 3]
        tally["score_bins"] = [np.bytes_(score)]
        tally["nuclides"] = [np.bytes_("total")]
        tally["estimator"] = np.bytes_("tracklength")
        tally["n_realizations"] = realizations
        tally["results"] = results


def test_statepoint_scalar_flux_export_is_bin_integrated_and_volume_normalized(
    tmp_path,
):
    path = tmp_path / "statepoint.h5"
    _statepoint(path)
    spectrum = read_cell_scalar_flux_spectrum(
        path, "pstl_magnet_neutron_volume_flux"
    )
    assert spectrum["cell_ids"] == [9, 10]
    assert spectrum["particles_per_batch"] == 100
    assert spectrum["batches"] == 2
    assert spectrum["source_histories"] == 200
    assert spectrum["energy_edges_eV"] == [0.0, 1.0e6, 2.0e7]
    assert spectrum["track_length_mean_cm_per_source"] == [
        [2.0, 4.0],
        [6.0, 8.0],
    ]
    estimators = scalar_flux_estimators(
        spectrum,
        domains_by_cell_id={
            9: {
                "domain_id": "magnet-0",
                "volume_cm3": 2.0,
                "volume_receipt_sha256": "1" * 64,
                "raw_h5m_sha256": "a" * 64,
                "canonical_geometry_fingerprint": "b" * 64,
            },
            10: {
                "domain_id": "magnet-1",
                "volume_cm3": 4.0,
                "volume_receipt_sha256": "2" * 64,
                "raw_h5m_sha256": "a" * 64,
                "canonical_geometry_fingerprint": "b" * 64,
            },
        },
        geometry_binding={
            "raw_h5m_sha256": "a" * 64,
            "canonical_geometry_fingerprint": "b" * 64,
        },
        tally_definition_sha256="3" * 64,
        run_binding={
            "statepoint_sha256": spectrum["statepoint_sha256"],
            "model_xml_sha256": "4" * 64,
            "settings_xml_sha256": "5" * 64,
            "strict_run_audit_sha256": "6" * 64,
            "root_acceptance_receipt_sha256": "7" * 64,
        },
    )
    assert estimators[0]["mean_per_source"] == [1.0, 2.0]
    assert estimators[1]["mean_per_source"] == [1.5, 2.0]
    assert estimators[0]["unit"] == "1/cm2/source_history/bin"


def test_statepoint_reader_rejects_wrong_version_or_observable(tmp_path):
    wrong_version = tmp_path / "wrong-version.h5"
    _statepoint(wrong_version, version=(0, 15, 0))
    with pytest.raises(ValueError, match="not from OpenMC 0.16.0"):
        read_cell_scalar_flux_spectrum(
            wrong_version, "pstl_magnet_neutron_volume_flux"
        )
    wrong_score = tmp_path / "wrong-score.h5"
    _statepoint(wrong_score, score="current")
    with pytest.raises(ValueError, match="does not score only total flux"):
        read_cell_scalar_flux_spectrum(
            wrong_score, "pstl_magnet_neutron_volume_flux"
        )


def test_statepoint_reader_does_not_report_zero_uncertainty_from_one_batch(
    tmp_path,
):
    path = tmp_path / "one-realization.h5"
    _statepoint(path, realizations=1)
    with pytest.raises(ValueError, match="at least two realizations"):
        read_cell_scalar_flux_spectrum(path, "pstl_magnet_neutron_volume_flux")


def test_statepoint_scalar_flux_requires_independently_audited_cell_volume(
    tmp_path,
):
    path = tmp_path / "statepoint.h5"
    _statepoint(path)
    spectrum = read_cell_scalar_flux_spectrum(
        path, "pstl_magnet_neutron_volume_flux"
    )
    with pytest.raises(ValueError, match="no audited domain"):
        scalar_flux_estimators(
            spectrum,
            domains_by_cell_id={
                9: {
                    "domain_id": "magnet-0",
                    "volume_cm3": 2.0,
                    "volume_receipt_sha256": "1" * 64,
                    "raw_h5m_sha256": "a" * 64,
                    "canonical_geometry_fingerprint": "b" * 64,
                }
            },
            geometry_binding={
                "raw_h5m_sha256": "a" * 64,
                "canonical_geometry_fingerprint": "b" * 64,
            },
            tally_definition_sha256="3" * 64,
            run_binding={
                "statepoint_sha256": spectrum["statepoint_sha256"],
                "model_xml_sha256": "4" * 64,
                "settings_xml_sha256": "5" * 64,
                "strict_run_audit_sha256": "6" * 64,
                "root_acceptance_receipt_sha256": "7" * 64,
            },
        )
