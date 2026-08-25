from types import SimpleNamespace

import h5py
import numpy as np

from parastell import openmc16
from parastell.magnet_volume_flux import _read_mesh_flux_tally


class _Filter:
    def __init__(self, bins, **kwargs):
        self.bins = bins
        self.options = kwargs


class _Tally:
    def __init__(self, name):
        self.name = name
        self.filters = []
        self.scores = []


class _Tallies(list):
    pass


def _fake_openmc():
    filter_names = (
        "SurfaceFilter",
        "MuSurfaceFilter",
        "CellFilter",
        "ParticleFilter",
        "EnergyFilter",
        "ReactionFilter",
        "ParticleProductionFilter",
    )
    values = {name: type(name, (_Filter,), {}) for name in filter_names}
    values.update({"Tally": _Tally, "Tallies": _Tallies})
    return SimpleNamespace(**values)


def test_local_casing_and_winding_flux_is_component_filtered(monkeypatch):
    fake = _fake_openmc()
    monkeypatch.setattr(openmc16, "require_capabilities", lambda: {})
    monkeypatch.setattr(openmc16, "_openmc", lambda: fake)
    local_mesh_filter = object()
    model = SimpleNamespace(tallies=None)

    inventory = openmc16.add_envelope_tallies(
        model,
        surface_ids=[101, 102],
        cell_ids=[17, 18],
        neutron_edges_eV=[0.0, 1.0, 20.0e6],
        photon_edges_eV=[0.0, 1.0e6, 20.0e6],
        tally_profile="core",
        local_mesh_filters_by_cell={18: local_mesh_filter},
    )

    local_names = set(inventory.local_mesh_flux) | set(
        inventory.local_mesh_heating
    )
    local_tallies = [
        tally for tally in model.tallies if tally.name in local_names
    ]
    assert len(local_tallies) == 4
    for tally in local_tallies:
        assert type(tally.filters[0]).__name__ == "CellFilter"
        assert tally.filters[0].bins == [18]
        assert tally.filters[1] is local_mesh_filter


def test_local_mesh_for_an_unselected_component_fails_closed(monkeypatch):
    fake = _fake_openmc()
    monkeypatch.setattr(openmc16, "require_capabilities", lambda: {})
    monkeypatch.setattr(openmc16, "_openmc", lambda: fake)

    try:
        openmc16.add_envelope_tallies(
            SimpleNamespace(tallies=None),
            surface_ids=[101],
            cell_ids=[18],
            neutron_edges_eV=[0.0, 20.0e6],
            photon_edges_eV=[0.0, 20.0e6],
            tally_profile="core",
            local_mesh_filters_by_cell={99: object()},
        )
    except ValueError as error:
        assert "unselected cell 99" in str(error)
    else:
        raise AssertionError("an unselected local component was accepted")


def test_component_filtered_local_flux_statepoint_round_trip(tmp_path):
    statepoint = tmp_path / "statepoint.h5"
    with h5py.File(statepoint, "w") as stream:
        tallies = stream.create_group("tallies")
        filters = tallies.create_group("filters")
        definitions = (
            ("cell", 1, np.asarray([18], dtype=np.int64)),
            ("mesh", 2, np.asarray(88, dtype=np.int64)),
            ("particle", 1, np.asarray([b"neutron"])),
            ("energy", 2, np.asarray([0.0, 1.0, 20.0e6])),
        )
        for identifier, (kind, bins, values) in enumerate(
            definitions, start=1
        ):
            group = filters.create_group(f"filter {identifier}")
            group.create_dataset("type", data=np.bytes_(kind))
            group.create_dataset("n_bins", data=bins)
            group.create_dataset("bins", data=values)
        tally = tallies.create_group("tally 1")
        tally.create_dataset("name", data=np.bytes_("local"))
        tally.create_dataset("filters", data=[1, 2, 3, 4])
        tally.create_dataset("n_realizations", data=2)
        means = np.asarray([1.0, 2.0, 3.0, 4.0])
        results = np.zeros((4, 1, 2))
        results[:, 0, 0] = 2.0 * means
        results[:, 0, 1] = 2.0 * means**2
        tally.create_dataset("results", data=results)

    result = _read_mesh_flux_tally(statepoint, "local")

    assert result["cell_id"] == 18
    assert result["mesh_id"] == 88
    assert result["particle"] == "neutron"
    assert result["track_length_mean_cm_per_source"].shape == (2, 2)
    assert np.all(result["track_length_std_dev_cm_per_source"] == 0.0)
