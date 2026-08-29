"""Declarative transport-response contract for magnet radiation handoffs.

The plan is intentionally independent of OpenMC so it can be validated before
an accepted geometry, nuclear-data library, or transport runtime is available.
It is the single inventory used to distinguish a declared response from one
that has been wired, executed, imported, or scientifically qualified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .reaction_identity import canonicalize_nuclide_mt_requests
from .reaction_identity import MATERIAL_MT_SCHEMA


SCHEMA = "parastell.transport_response_plan/v1.0.0"
PROOF_LEVELS = (
    "DECLARED",
    "WIRED",
    "SMOKE_EXECUTED",
    "IMPORT_VALIDATED",
    "RESEARCH_QUALIFIED",
)


@dataclass(frozen=True)
class ResponseRequirement:
    response_id: str
    capability_id: str
    observable: str
    scope: str
    particle: str
    unit: str
    estimator: str
    mandatory: bool = True
    energy_axis: str | None = None
    scores: tuple[str, ...] = ()
    nuclides: tuple[str, ...] = ()
    reactions: tuple[str, ...] = ()
    covariance_required: bool = False


def default_magnet_response_requirements() -> tuple[ResponseRequirement, ...]:
    """Return the geometry-neutral minimum response set for downstream work."""
    requirements = [
        ResponseRequirement(
            "magnet-neutron-volume-flux",
            "TAL-NFLX",
            "scalar_flux_spectrum",
            "magnet_volume",
            "neutron",
            "1/cm2/source_history/bin",
            "tracklength",
            energy_axis="neutron",
            scores=("flux",),
        ),
        ResponseRequirement(
            "magnet-photon-volume-flux",
            "TAL-PFLX",
            "scalar_flux_spectrum",
            "magnet_volume",
            "photon",
            "1/cm2/source_history/bin",
            "tracklength",
            energy_axis="photon",
            scores=("flux",),
        ),
        ResponseRequirement(
            "magnet-boundary-current",
            "TAL-CURR",
            "surface_current_phase_space",
            "magnet_outer_boundary",
            "neutron+photon",
            "particles/source_history",
            "analog",
            energy_axis="particle_specific",
            scores=("current",),
        ),
        ResponseRequirement(
            "magnet-heating",
            "TAL-HEAT",
            "heating",
            "magnet_volume_and_local_mesh",
            "neutron+photon",
            "eV/source_history",
            "heating",
            scores=("heating",),
        ),
        ResponseRequirement(
            "magnet-reaction-rates",
            "TAL-RXN",
            "reaction_rate",
            "magnet_volume",
            "neutron",
            "reactions/source_history",
            "analog",
            energy_axis="neutron",
            scores=("events",),
        ),
        ResponseRequirement(
            "magnet-nuclide-mt-rates",
            "PKA-01",
            "nuclide_mt_reaction_rate",
            "magnet_volume",
            "neutron",
            "reactions/source_history/bin",
            "analog",
            energy_axis="neutron",
            scores=("events",),
            covariance_required=True,
        ),
        ResponseRequirement(
            "magnet-damage-energy",
            "TAL-DMG",
            "damage_energy",
            "magnet_volume_and_local_mesh",
            "neutron",
            "eV/source_history",
            "tracklength",
            energy_axis="neutron",
            scores=("damage-energy",),
        ),
        ResponseRequirement(
            "magnet-gas-production",
            "TAL-GAS",
            "gas_production",
            "magnet_volume",
            "neutron",
            "atoms/source_history",
            "analog",
            energy_axis="neutron",
            scores=(
                "H1-production",
                "H2-production",
                "H3-production",
                "He3-production",
                "He4-production",
            ),
        ),
        ResponseRequirement(
            "magnet-secondary-production",
            "ACT-01",
            "secondary_particle_production",
            "magnet_volume",
            "neutron",
            "particles/source_history/bin",
            "analog",
            energy_axis="incident_and_outgoing",
            scores=("events",),
        ),
    ]
    return tuple(requirements)


def build_response_plan(
    *,
    case_id: str,
    magnet_ids: Sequence[str],
    neutron_energy_edges_eV: Sequence[float],
    photon_energy_edges_eV: Sequence[float],
    proof_level: str = "DECLARED",
    nuclide_mt_requests: Mapping[str, Sequence[str | int]] | None = None,
    nuclide_mt_derivation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a stable, hashable response plan."""
    response_rows = [
        asdict(item) for item in default_magnet_response_requirements()
    ]
    requests = canonicalize_nuclide_mt_requests(nuclide_mt_requests or {})
    if nuclide_mt_derivation is not None:
        derivation = dict(nuclide_mt_derivation)
        if (
            derivation.get("schema") != MATERIAL_MT_SCHEMA
            or derivation.get("status")
            != "MATERIAL_ISOTOPE_MT_REQUESTS_DERIVED"
            or derivation.get("nuclide_mt_requests") != requests
        ):
            raise ValueError(
                "material-derived nuclide/MT requests are not bound"
            )
        derivation_hash = str(derivation.get("derivation_sha256", ""))
        unsigned = dict(derivation)
        unsigned.pop("derivation_sha256", None)
        canonical_derivation = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":")
        )
        if (
            derivation_hash
            != hashlib.sha256(canonical_derivation.encode()).hexdigest()
        ):
            raise ValueError(
                "material-derived nuclide/MT receipt hash is invalid"
            )
        request_origin = derivation
    else:
        request_origin = {
            "kind": "explicit_or_empty",
            "missing_semantics": "MISSING_IS_NOT_ZERO",
        }
    payload = {
        "schema": SCHEMA,
        "case_id": str(case_id).strip(),
        "magnet_ids": [str(value).strip() for value in magnet_ids],
        "proof_level": proof_level,
        "normalization": "per_source_history",
        "energy_axes_eV": {
            "neutron": [float(value) for value in neutron_energy_edges_eV],
            "photon": [float(value) for value in photon_energy_edges_eV],
        },
        "nuclide_mt_requests": requests,
        "nuclide_mt_request_origin": request_origin,
        "responses": response_rows,
        "missing_response_semantics": "MISSING_IS_NOT_ZERO",
        "covariance_policy": (
            "UNAVAILABLE_UNLESS_EXPLICITLY_COMPUTED_AND_HASH_BOUND"
        ),
    }
    validate_response_plan(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def validate_response_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported transport response plan schema")
    if not str(plan.get("case_id", "")).strip():
        raise ValueError("response plan case_id is required")
    magnets = plan.get("magnet_ids")
    if (
        not isinstance(magnets, list)
        or not magnets
        or any(not item for item in magnets)
    ):
        raise ValueError("response plan needs at least one magnet ID")
    if len(magnets) != len(set(magnets)):
        raise ValueError("response plan magnet IDs must be unique")
    if plan.get("proof_level") not in PROOF_LEVELS:
        raise ValueError("response plan proof level is invalid")
    if plan.get("normalization") != "per_source_history":
        raise ValueError("response plan normalization is unsafe")
    axes = plan.get("energy_axes_eV")
    if not isinstance(axes, Mapping) or set(axes) != {"neutron", "photon"}:
        raise ValueError("response plan energy axes are incomplete")
    for particle, values in axes.items():
        try:
            edges = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{particle} energy edges are invalid") from exc
        if len(edges) < 2 or edges != sorted(set(edges)) or edges[0] < 0.0:
            raise ValueError(f"{particle} energy edges must strictly increase")
    responses = plan.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("response inventory is empty")
    identifiers = [str(row.get("response_id", "")) for row in responses]
    if any(not value for value in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("response IDs must be nonempty and unique")
    required = {
        "TAL-NFLX",
        "TAL-PFLX",
        "TAL-CURR",
        "TAL-HEAT",
        "TAL-RXN",
        "TAL-DMG",
        "TAL-GAS",
        "ACT-01",
        "PKA-01",
    }
    capabilities = {str(row.get("capability_id")) for row in responses}
    missing = sorted(required - capabilities)
    if missing:
        raise ValueError(
            "response plan is missing capabilities: " + ", ".join(missing)
        )
    if plan.get("missing_response_semantics") != "MISSING_IS_NOT_ZERO":
        raise ValueError("missing response semantics must fail closed")
    canonical = canonicalize_nuclide_mt_requests(
        plan.get("nuclide_mt_requests", {})
    )
    if canonical != plan.get("nuclide_mt_requests"):
        raise ValueError("nuclide/MT requests are not canonical")
    origin = plan.get("nuclide_mt_request_origin")
    if origin is None:
        # Existing v1 plans predate the explicit origin field.  They remain
        # valid only as legacy explicit requests, never as material-derived.
        return
    if not isinstance(origin, Mapping):
        raise ValueError("nuclide/MT request origin is missing")
    if origin.get("schema") == MATERIAL_MT_SCHEMA:
        if (
            origin.get("status") != "MATERIAL_ISOTOPE_MT_REQUESTS_DERIVED"
            or origin.get("nuclide_mt_requests") != canonical
        ):
            raise ValueError(
                "material-derived nuclide/MT origin is inconsistent"
            )
        digest = str(origin.get("derivation_sha256", ""))
        unsigned = dict(origin)
        unsigned.pop("derivation_sha256", None)
        encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
        if digest != hashlib.sha256(encoded.encode()).hexdigest():
            raise ValueError(
                "material-derived nuclide/MT origin hash is invalid"
            )
    elif origin != {
        "kind": "explicit_or_empty",
        "missing_semantics": "MISSING_IS_NOT_ZERO",
    }:
        raise ValueError("nuclide/MT request origin is unsupported")


def bind_response_plan(
    plan: Mapping[str, Any],
    *,
    tally_names: Sequence[str],
    surface_bank_configured: bool,
) -> dict[str, Any]:
    """Record actual OpenMC wiring without claiming that transport ran."""
    validate_response_plan(plan)
    names = [str(name).strip() for name in tally_names]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("wired tally names must be nonempty and unique")
    if not surface_bank_configured:
        raise ValueError("magnet phase-space bank was not configured")
    result = dict(plan)
    result["proof_level"] = "WIRED"
    result["wired_tally_names"] = names
    result["surface_bank_configured"] = True
    result.pop("plan_sha256", None)
    validate_response_plan(result)
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return result


def estimate_response_cardinality(
    plan: Mapping[str, Any],
    *,
    surface_count: int,
    local_mesh_bins_per_magnet: int = 0,
    reaction_family_count: int = 6,
) -> dict[str, int]:
    """Estimate response-bin cardinality before creating an OpenMC model."""
    validate_response_plan(plan)
    magnets = len(plan["magnet_ids"])
    surfaces = int(surface_count)
    local_bins = int(local_mesh_bins_per_magnet)
    reactions = int(reaction_family_count)
    if min(surfaces, reactions) <= 0 or local_bins < 0:
        raise ValueError("response cardinality inputs are invalid")
    neutron_bins = len(plan["energy_axes_eV"]["neutron"]) - 1
    photon_bins = len(plan["energy_axes_eV"]["photon"]) - 1
    nuclide_mt_count = sum(
        len(values) for values in plan["nuclide_mt_requests"].values()
    )
    rows = {
        "volume_flux": magnets * (neutron_bins + photon_bins),
        "surface_current_and_flux": 4
        * surfaces
        * (neutron_bins + photon_bins),
        "volume_heating": magnets
        * (neutron_bins + photon_bins + neutron_bins + photon_bins),
        "local_mesh_flux_heating": magnets
        * local_bins
        * 2
        * (neutron_bins + photon_bins),
        "reaction_families": magnets * neutron_bins * reactions,
        "nuclide_mt_reactions": magnets * neutron_bins * nuclide_mt_count,
        "damage_energy": magnets * neutron_bins * (1 + local_bins),
        "gas_production": magnets * neutron_bins * 5 * (1 + local_bins),
        "secondary_production_lower_bound": magnets
        * neutron_bins
        * (neutron_bins + 3 * photon_bins),
    }
    total = sum(rows.values())
    return {
        **rows,
        "total_response_bins": total,
        "stored_moment_values": total * 2,
    }
