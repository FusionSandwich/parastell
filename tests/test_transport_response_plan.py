from __future__ import annotations

import copy

import pytest

from parastell.reaction_identity import derive_material_nuclide_mt_requests
from parastell.transport_response_plan import (
    bind_response_plan,
    build_response_plan,
    estimate_response_cardinality,
    validate_response_plan,
)


def _plan():
    return build_response_plan(
        case_id="wistell-d-direct-90",
        magnet_ids=["magnet-0008"],
        neutron_energy_edges_eV=[0.0, 1.0e6, 2.0e7],
        photon_energy_edges_eV=[0.0, 1.0e5, 2.0e7],
        nuclide_mt_requests={"W184": [2, 102], "B10": [107]},
    )


def test_default_plan_covers_transport_activation_and_pka_responses():
    plan = _plan()
    capabilities = {row["capability_id"] for row in plan["responses"]}
    assert {
        "TAL-NFLX",
        "TAL-PFLX",
        "TAL-CURR",
        "TAL-HEAT",
        "TAL-RXN",
        "TAL-DMG",
        "TAL-GAS",
        "ACT-01",
        "PKA-01",
    } <= capabilities
    assert plan["proof_level"] == "DECLARED"
    assert len(plan["plan_sha256"]) == 64


def test_binding_is_wired_not_executed_or_research_qualified():
    bound = bind_response_plan(
        _plan(),
        tally_names=["flux", "damage", "W184-mt"],
        surface_bank_configured=True,
    )
    assert bound["proof_level"] == "WIRED"
    assert "SMOKE_EXECUTED" not in bound.values()


def test_plan_rejects_duplicate_magnets_and_nonmonotonic_bins():
    duplicate = copy.deepcopy(_plan())
    duplicate["magnet_ids"].append("magnet-0008")
    with pytest.raises(ValueError, match="unique"):
        validate_response_plan(duplicate)
    bins = copy.deepcopy(_plan())
    bins["energy_axes_eV"]["neutron"] = [0.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="strictly increase"):
        validate_response_plan(bins)


def test_plan_refuses_to_claim_wired_without_phase_space_bank():
    with pytest.raises(ValueError, match="phase-space bank"):
        bind_response_plan(
            _plan(), tally_names=["flux"], surface_bank_configured=False
        )


def test_cardinality_estimate_scales_with_local_mesh_and_surface_count():
    coarse = estimate_response_cardinality(
        _plan(), surface_count=12, local_mesh_bins_per_magnet=10
    )
    fine = estimate_response_cardinality(
        _plan(), surface_count=24, local_mesh_bins_per_magnet=100
    )
    assert fine["total_response_bins"] > coarse["total_response_bins"]
    assert fine["stored_moment_values"] == 2 * fine["total_response_bins"]


def test_response_plan_binds_material_derived_mt_receipt():
    derivation = derive_material_nuclide_mt_requests(
        [
            {
                "material_id": "winding",
                "composition_sha256": "a" * 64,
                "isotopes": {"Cu63": 0.7, "Cu65": 0.3},
            }
        ],
        default_mts=[2, 102],
    )
    plan = build_response_plan(
        case_id="derived",
        magnet_ids=["magnet-0000"],
        neutron_energy_edges_eV=[0.0, 2.0e7],
        photon_energy_edges_eV=[0.0, 2.0e7],
        nuclide_mt_requests=derivation["nuclide_mt_requests"],
        nuclide_mt_derivation=derivation,
    )
    assert plan["nuclide_mt_request_origin"]["derivation_sha256"] == (
        derivation["derivation_sha256"]
    )


def test_legacy_v1_explicit_plan_without_origin_remains_readable():
    plan = _plan()
    plan.pop("nuclide_mt_request_origin")
    validate_response_plan(plan)
