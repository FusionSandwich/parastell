"""Selected-case wiring for arbitrary magnet IDs without geometry mutation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .openmc16 import add_envelope_tallies
from .surface_source_instrumentation import (
    configure_openmc16_surface_bank,
)
from .transport_response_plan import bind_response_plan, validate_response_plan


SCHEMA = "parastell.selected_case_instrumentation/v1.0.0"


def _inventory_names(inventory) -> list[str]:
    payload = asdict(inventory)
    names = []
    for key, value in payload.items():
        if key in {"profile", "response_availability"} or value is None:
            continue
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, (list, tuple)):
            names.extend(str(item) for item in value)
    if len(names) != len(set(names)):
        raise ValueError("selected-case tally names collide")
    return names


def instrument_selected_case(
    model,
    *,
    response_plan: Mapping[str, Any],
    surface_spec: Mapping[str, Any],
    magnet_cell_ids: Mapping[str, int],
    tally_profile: str = "magnet_damage_and_handoff",
    reaction_bins: Sequence[str | int] = (2, 4, 16, 102, 103, 107),
    supported_responses: Sequence[str] | None = None,
    local_mesh_filters_by_cell: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    """Attach all declared responses and the complete correlated surface bank."""
    validate_response_plan(response_plan)
    planned_magnets = list(response_plan["magnet_ids"])
    if set(planned_magnets) != set(magnet_cell_ids):
        raise ValueError("response-plan and selected magnet IDs disagree")
    cell_ids = [int(magnet_cell_ids[item]) for item in planned_magnets]
    if min(cell_ids) <= 0 or len(cell_ids) != len(set(cell_ids)):
        raise ValueError(
            "selected OpenMC cell IDs must be unique and positive"
        )
    if surface_spec.get("coupling_interface") not in {
        "homogenized_magnet_outer_boundary",
        "outer_casing_external",
        "winding_pack",
    }:
        raise ValueError("surface source coupling interface is ambiguous")
    axes = response_plan["energy_axes_eV"]
    if surface_spec.get("energy_edges_eV") != axes:
        raise ValueError("surface-bank and response-plan energy axes disagree")
    configure_openmc16_surface_bank(model, surface_spec)
    inventory = add_envelope_tallies(
        model,
        surface_ids=surface_spec["surface_ids"],
        cell_ids=cell_ids,
        neutron_edges_eV=axes["neutron"],
        photon_edges_eV=axes["photon"],
        reaction_bins=reaction_bins,
        tally_profile=tally_profile,
        local_mesh_filters_by_cell=local_mesh_filters_by_cell,
        supported_responses=supported_responses,
        nuclide_mt_requests=response_plan.get("nuclide_mt_requests"),
    )
    names = _inventory_names(inventory)
    bound_plan = bind_response_plan(
        response_plan,
        tally_names=names,
        surface_bank_configured=True,
    )
    return {
        "schema": SCHEMA,
        "status": "WIRED_NOT_EXECUTED",
        "case_id": response_plan["case_id"],
        "magnet_cell_ids": {
            key: int(magnet_cell_ids[key]) for key in sorted(magnet_cell_ids)
        },
        "coupling_interface": surface_spec["coupling_interface"],
        "surface_ids": list(surface_spec["surface_ids"]),
        "surface_bank_capacity": int(surface_spec["configured_capacity"]),
        "tally_inventory": inventory.to_dict(),
        "response_plan": bound_plan,
        "geometry_mutation": False,
        "production_run_authorized": False,
    }
