from __future__ import annotations

from types import SimpleNamespace

import pytest

import parastell.selected_case_instrumentation as selected
from parastell.openmc16 import TallyInventory
from parastell.transport_response_plan import build_response_plan


def _plan():
    return build_response_plan(
        case_id="arbitrary-stellarator-case",
        magnet_ids=["magnet-alpha"],
        neutron_energy_edges_eV=[0.0, 1.0e6, 2.0e7],
        photon_energy_edges_eV=[0.0, 1.0e5, 2.0e7],
        nuclide_mt_requests={"W184": [2, 102]},
    )


def _surface_spec():
    return {
        "coupling_interface": "homogenized_magnet_outer_boundary",
        "surface_ids": [41, 42],
        "energy_edges_eV": {
            "neutron": [0.0, 1.0e6, 2.0e7],
            "photon": [0.0, 1.0e5, 2.0e7],
        },
        "configured_capacity": 10_000,
    }


def test_selected_case_wires_all_tallies_for_arbitrary_magnet_id(monkeypatch):
    configured = []
    captured = {}
    monkeypatch.setattr(
        selected,
        "configure_openmc16_surface_bank",
        lambda model, spec: configured.append(spec),
    )

    def fake_add(model, **kwargs):
        captured.update(kwargs)
        return TallyInventory(
            current=("n-current", "p-current"),
            directional_current=("n-mu", "p-mu"),
            surface_flux=("n-surface", "p-surface"),
            volume_flux=("n-flux", "p-flux"),
            reactions="reactions",
            nuclide_reactions=("W184-mt",),
            production=("photon-production",),
            heating=("n-heat", "p-heat"),
            total_heating="total-heat",
            damage_energy="damage",
            gas_production="gas",
        )

    monkeypatch.setattr(selected, "add_envelope_tallies", fake_add)
    manifest = selected.instrument_selected_case(
        SimpleNamespace(),
        response_plan=_plan(),
        surface_spec=_surface_spec(),
        magnet_cell_ids={"magnet-alpha": 9008},
        supported_responses=["nuclide_mt:W184"],
    )
    assert len(configured) == 1
    assert captured["cell_ids"] == [9008]
    assert captured["nuclide_mt_requests"] == {"W184": [2, 102]}
    assert manifest["status"] == "WIRED_NOT_EXECUTED"
    assert manifest["geometry_mutation"] is False
    assert manifest["response_plan"]["proof_level"] == "WIRED"


def test_selected_case_rejects_plan_cell_mapping_disagreement(monkeypatch):
    monkeypatch.setattr(
        selected,
        "configure_openmc16_surface_bank",
        lambda model, spec: None,
    )
    with pytest.raises(ValueError, match="disagree"):
        selected.instrument_selected_case(
            SimpleNamespace(),
            response_plan=_plan(),
            surface_spec=_surface_spec(),
            magnet_cell_ids={"different-magnet": 9008},
        )
