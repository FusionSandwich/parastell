import json

import h5py
import numpy as np
import pytest

from parastell.magnet_volume_flux import export_volume_scalar_flux


def _statepoint(path):
    with h5py.File(path, "w") as statepoint:
        tallies = statepoint.create_group("tallies")
        filters = tallies.create_group("filters")
        definitions = (
            (1, "cell", [16, 17], 2),
            (2, "particle", [b"neutron"], 1),
            (3, "energy", [0.0, 1.0, 2.0], 2),
        )
        for filter_id, kind, bins, count in definitions:
            group = filters.create_group(f"filter {filter_id}")
            group["type"] = np.bytes_(kind)
            group["bins"] = bins
            group["n_bins"] = count
        tally = tallies.create_group("tally 1")
        tally["name"] = np.bytes_("pstl_magnet_neutron_ccfe_709_volume_flux")
        tally["filters"] = [1, 2, 3]
        tally["n_realizations"] = 10
        means = np.asarray([10.0, 20.0, 30.0, 40.0])
        results = np.zeros((4, 1, 2))
        results[:, 0, 0] = means * 10.0
        results[:, 0, 1] = means**2 * 10.0
        tally["results"] = results


def test_volume_flux_divides_track_length_by_volume_and_source_scales(
    tmp_path,
):
    statepoint = tmp_path / "statepoint.h5"
    output = tmp_path / "flux.h5"
    _statepoint(statepoint)
    manifest = export_volume_scalar_flux(
        output,
        statepoint_path=statepoint,
        tally_names=["pstl_magnet_neutron_ccfe_709_volume_flux"],
        cell_volumes_cm3={16: 2.0, 17: 10.0},
        physical_source_rate_per_s=100.0,
        magnet_ids_by_cell={16: "coil-a", 17: "coil-b"},
    )
    assert manifest["quantity"] == "volume_scalar_flux"
    with h5py.File(output) as result:
        group = result["volume_scalar_flux/neutron_ccfe_709"]
        np.testing.assert_allclose(
            group["mean_per_source"][:], [[5.0, 10.0], [3.0, 4.0]]
        )
        np.testing.assert_allclose(
            group["mean_physical"][:], [[500.0, 1000.0], [300.0, 400.0]]
        )
        stored = json.loads(result["manifest_json"].asstr()[()])
        assert stored["normalization"]["physical_units"] == "particles/cm2/s"


def test_volume_flux_requires_every_cell_volume(tmp_path):
    statepoint = tmp_path / "statepoint.h5"
    _statepoint(statepoint)
    with pytest.raises(ValueError, match="missing DAGMC volumes"):
        export_volume_scalar_flux(
            tmp_path / "flux.h5",
            statepoint_path=statepoint,
            tally_names=["pstl_magnet_neutron_ccfe_709_volume_flux"],
            cell_volumes_cm3={16: 2.0},
            physical_source_rate_per_s=None,
        )
