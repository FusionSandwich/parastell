from __future__ import annotations

from types import SimpleNamespace

import parastell.openmc16 as openmc16


class _Filter:
    def __init__(self, bins, **kwargs):
        self.bins = bins
        self.kwargs = kwargs


class _Tally:
    def __init__(self, name):
        self.name = name
        self.filters = []
        self.scores = []


class _Tallies(list):
    pass


def _fake_openmc():
    return SimpleNamespace(
        Tallies=_Tallies,
        Tally=_Tally,
        SurfaceFilter=_Filter,
        MuSurfaceFilter=_Filter,
        ParticleFilter=_Filter,
        EnergyFilter=_Filter,
        CellFilter=_Filter,
        ReactionFilter=_Filter,
        ParticleProductionFilter=_Filter,
    )


def test_activation_ready_profile_configures_every_downstream_observable(
    monkeypatch,
):
    fake = _fake_openmc()
    monkeypatch.setattr(openmc16, "_openmc", lambda: fake)
    monkeypatch.setattr(
        openmc16, "require_capabilities", lambda: {"passes": True}
    )
    model = SimpleNamespace(tallies=None)
    inventory = openmc16.add_envelope_tallies(
        model,
        surface_ids=[101, 102, 103, 104],
        cell_ids=[9],
        neutron_edges_eV=[0.0, 1.0e6, 2.0e7],
        photon_edges_eV=[0.0, 1.0e5, 2.0e7],
        tally_profile="activation_ready",
        supported_responses=[
            "damage-energy",
            "reaction_families",
            "H1-production",
            "H2-production",
            "H3-production",
            "He3-production",
            "He4-production",
            "produce:photon",
            "produce:neutron",
            "produce:electron",
            "produce:positron",
        ],
    )
    by_name = {tally.name: tally for tally in model.tallies}
    assert set(inventory.volume_flux) == {
        "pstl_magnet_neutron_configured_volume_flux",
        "pstl_magnet_photon_configured_volume_flux",
    }
    assert inventory.damage_energy == "pstl_magnet_neutron_damage_energy"
    assert inventory.gas_production == "pstl_magnet_gas_production"
    assert by_name[inventory.gas_production].scores == [
        "H1-production",
        "H2-production",
        "H3-production",
        "He3-production",
        "He4-production",
    ]
    assert inventory.reactions == "pstl_magnet_neutron_reactions"
    assert len(inventory.heating) == 2
    assert inventory.total_heating == "pstl_magnet_total_heating"
    assert {
        "pstl_magnet_production_photon",
        "pstl_magnet_production_neutron",
        "pstl_magnet_production_electron",
        "pstl_magnet_production_positron",
    } == set(inventory.production)
    for name in inventory.volume_flux:
        assert by_name[name].scores == ["flux"]
    assert by_name[inventory.damage_energy].scores == ["damage-energy"]


def test_unavailable_nuclear_data_response_is_missing_not_zero(monkeypatch):
    fake = _fake_openmc()
    monkeypatch.setattr(openmc16, "_openmc", lambda: fake)
    monkeypatch.setattr(
        openmc16, "require_capabilities", lambda: {"passes": True}
    )
    model = SimpleNamespace(tallies=None)
    inventory = openmc16.add_envelope_tallies(
        model,
        surface_ids=[101],
        cell_ids=[9],
        neutron_edges_eV=[0.0, 2.0e7],
        photon_edges_eV=[0.0, 2.0e7],
        tally_profile="activation_ready",
        supported_responses=["reaction_families"],
    )
    assert inventory.damage_energy is None
    assert inventory.gas_production is None
    assert inventory.production == ()
    assert inventory.response_availability["damage-energy"] == {
        "status": "UNAVAILABLE_IN_CONFIGURED_NUCLEAR_DATA",
        "available": False,
    }
