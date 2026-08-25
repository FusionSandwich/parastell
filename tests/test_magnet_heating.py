import json
import sys
import types

import h5py
import numpy as np
import pytest

from parastell.magnet_heating import EV_TO_JOULE, SCHEMA, export_magnet_heating


class CellFilter:
    bins = np.asarray([8])


class ParticleFilter:
    bins = ("neutron",)


class EnergyFilter:
    values = np.asarray([0.0, 1.0, 2.0])


class _Tally:
    scores = ["heating"]
    filters = [CellFilter(), ParticleFilter(), EnergyFilter()]
    mean = np.asarray([[[2.0]], [[3.0]]])
    std_dev = np.asarray([[[0.2]], [[0.3]]])


class _StatePoint:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get_tally(self, name):
        assert name == "neutron-heating"
        return _Tally()


def test_export_magnet_heating(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, "openmc", types.SimpleNamespace(StatePoint=_StatePoint)
    )
    output = tmp_path / "heating.h5"
    manifest = export_magnet_heating(
        output,
        statepoint_path=tmp_path / "statepoint.h5",
        tally_names=["neutron-heating"],
        cell_volumes_cm3={8: 10.0},
        cell_magnet_ids={8: "magnet-8"},
        physical_source_rate_per_s=4.0,
    )
    assert manifest["schema"] == SCHEMA
    with h5py.File(output, "r") as handle:
        stored = json.loads(handle["manifest_json"][()].decode())
        assert stored["physical_source_rate_per_s"] == 4.0
        group = handle["heating/neutron"]
        assert group["magnet_ids"].asstr()[0] == "magnet-8"
        assert group["mean_eV_per_source"][:].tolist() == [[2.0, 3.0]]
        np.testing.assert_allclose(
            group["mean_W"][:], [[8.0 * EV_TO_JOULE, 12.0 * EV_TO_JOULE]]
        )
        np.testing.assert_allclose(
            group["mean_W_cm3"][:],
            [[0.8 * EV_TO_JOULE, 1.2 * EV_TO_JOULE]],
        )


def test_export_magnet_heating_requires_physical_source(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, "openmc", types.SimpleNamespace(StatePoint=_StatePoint)
    )
    with pytest.raises(ValueError, match="must be positive"):
        export_magnet_heating(
            tmp_path / "heating.h5",
            statepoint_path=tmp_path / "statepoint.h5",
            tally_names=["neutron-heating"],
            cell_volumes_cm3={8: 10.0},
            cell_magnet_ids={8: "magnet-8"},
            physical_source_rate_per_s=0.0,
        )
