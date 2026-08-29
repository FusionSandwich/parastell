"""Fail-closed OpenMC 0.16 response-to-consumer handoff adapter.

This adapter deliberately retains bin-integrated statepoint results.  It does
not sum energy bins because OpenMC statepoints do not contain the cross-bin
covariance needed to assign an uncertainty to such a sum.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np

from .openmc16_response_results import SCHEMA as RESULT_SCHEMA
from .radiation_consumer_handoff import (
    build_radiation_consumer_handoff,
    validate_radiation_consumer_handoff,
)
from .reaction_identity import canonical_mt, canonical_nuclide, mt_label

SCHEMA = "parastell.openmc16_response_handoff_adapter/v1.0.0"
RESPONSE_SET_SCHEMA = "parastell.openmc16_response_set/v1.0.0"
_ALLOWED_FILTERS = {
    "cell",
    "particle",
    "energy",
    "reaction",
    "particleproduction",
}
_GAS_IDENTITIES = {
    "H1-production": ("hydrogen_production", "H1", "hydrogen"),
    "H2-production": ("hydrogen_production", "H2", "hydrogen"),
    "H3-production": ("hydrogen_production", "H3", "hydrogen"),
    "He3-production": ("helium_production", "He3", "helium"),
    "He4-production": ("helium_production", "He4", "helium"),
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: Any, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return text


def _strict_binding(
    result: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    tally_id = str(result.get("tally_id", ""))
    _digest(binding.get("tally_definition_sha256"), "tally definition")
    actual_filters = [
        str(row.get("type", "")) for row in result.get("filters", ())
    ]
    declared_filters = [
        str(value) for value in binding.get("filter_types", ())
    ]
    if actual_filters != declared_filters:
        raise ValueError(f"{tally_id} filter axes do not match their binding")
    actual_scores = [str(value) for value in result.get("scores", ())]
    declared_scores = [str(value) for value in binding.get("scores", ())]
    if actual_scores != declared_scores:
        raise ValueError(f"{tally_id} score axes do not match their binding")
    actual_nuclides = [str(value) for value in result.get("nuclides", ())]
    declared_nuclides = [str(value) for value in binding.get("nuclides", ())]
    if actual_nuclides != declared_nuclides:
        raise ValueError(f"{tally_id} nuclide axes do not match their binding")
    return dict(binding)


def _validate_response_set(
    response_set: Mapping[str, Any], provenance: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if response_set.get("schema") != RESPONSE_SET_SCHEMA:
        raise ValueError("unsupported OpenMC response-set schema")
    if response_set.get("normalization") != "per_source_history":
        raise ValueError("response-set normalization is ambiguous")
    if response_set.get("status") != "SMOKE_RESULT":
        raise ValueError("response set is not a completed extracted result")
    if response_set.get("openmc_version") != "0.16.0":
        raise ValueError("response set is not from OpenMC 0.16.0")
    for key in ("statepoint_sha256", "source_histories"):
        if response_set.get(key) != provenance.get(key):
            raise ValueError(f"response-set {key} is not provenance-bound")
    responses = response_set.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("response set contains no responses")
    tally_ids = [str(row.get("tally_id", "")) for row in responses]
    if any(not value for value in tally_ids) or len(tally_ids) != len(
        set(tally_ids)
    ):
        raise ValueError("response tally identities are empty or duplicated")
    for result in responses:
        if result.get("schema") != RESULT_SCHEMA:
            raise ValueError("unsupported OpenMC response-result schema")
        for key in ("statepoint_sha256", "source_histories", "openmc_version"):
            if result.get(key) != response_set.get(key):
                raise ValueError(f"response result {key} is not set-bound")
        if result.get("covariance", {}).get("status") != "UNAVAILABLE":
            raise ValueError("unexpected statepoint covariance claim")
    return responses


def _filter_map(
    result: Mapping[str, Any],
) -> dict[str, tuple[int, Mapping[str, Any]]]:
    filters = result.get("filters")
    if not isinstance(filters, list):
        raise TypeError("response filters are missing")
    output: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, row in enumerate(filters):
        kind = str(row.get("type", ""))
        if kind not in _ALLOWED_FILTERS:
            raise ValueError(
                f"unsupported or ambiguous response axis {kind!r}"
            )
        if kind in output:
            raise ValueError(f"response repeats the {kind} axis")
        output[kind] = (index, row)
    if "cell" not in output or "particle" not in output:
        raise ValueError(
            "response requires explicit cell and incident-particle axes"
        )
    if "reaction" in output and "particleproduction" in output:
        raise ValueError(
            "reaction and particle-production axes cannot be combined"
        )
    return output


def _common_estimator(
    *,
    result: Mapping[str, Any],
    binding: Mapping[str, Any],
    material: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "domain_id": material["domain_id"],
        "openmc_cell_id": material["openmc_cell_id"],
        "dagmc_volume_id": material["dagmc_volume_id"],
        "volume_cm3": material["volume_cm3"],
        "volume_receipt_sha256": material["volume_receipt_sha256"],
        "tally_id": result["tally_id"],
        "tally_definition_sha256": binding["tally_definition_sha256"],
        "tally_binding_sha256": _canonical_sha256(binding),
        "estimator": result["estimator"],
        "normalization": "per_source_history",
        "statepoint_sha256": provenance["statepoint_sha256"],
        "raw_h5m_sha256": provenance["raw_h5m_sha256"],
        "canonical_geometry_fingerprint": provenance[
            "canonical_geometry_fingerprint"
        ],
        "model_xml_sha256": provenance["model_xml_sha256"],
        "settings_xml_sha256": provenance["settings_xml_sha256"],
        "strict_run_audit_sha256": provenance["strict_run_audit_sha256"],
        "root_acceptance_receipt_sha256": provenance[
            "root_acceptance_receipt_sha256"
        ],
        "source_histories": provenance["source_histories"],
        "covariance": dict(result["covariance"]),
    }


def _score_semantics(
    score: str,
    *,
    has_energy_axis: bool,
    reaction: bool,
    production: bool,
) -> dict[str, Any]:
    suffix = "/bin" if has_energy_axis else ""
    if score == "flux" and not reaction and not production:
        return {
            "observable": "scalar_flux_spectrum",
            "unit": "1/cm2/source_history/bin",
            "integration_semantics": (
                "tracklength_scalar_flux_divided_by_audited_domain_volume; "
                "each value is integrated over its incident-energy bin"
            ),
            "divide_by_volume": True,
        }
    if (
        score in {"heating", "damage-energy"}
        and not reaction
        and not production
    ):
        return {
            "observable": "heating" if score == "heating" else "damage_energy",
            "unit": f"eV/source_history{suffix}",
            "integration_semantics": (
                "response integrated over the audited domain"
                + (
                    " and separately over each incident-energy bin; bins are not summed"
                    if has_energy_axis
                    else " and all incident energies"
                )
            ),
            "divide_by_volume": False,
        }
    if score in _GAS_IDENTITIES and not reaction and not production:
        observable, isotope, element = _GAS_IDENTITIES[score]
        return {
            "observable": observable,
            "unit": f"atoms/source_history{suffix}",
            "integration_semantics": (
                "produced atoms integrated over the audited domain"
                + (
                    " and separately over each incident-energy bin; bins are not summed"
                    if has_energy_axis
                    else " and all incident energies"
                )
            ),
            "divide_by_volume": False,
            "gas_isotope": isotope,
            "gas_element": element,
        }
    if score == "events" and reaction and not production:
        return {
            "observable": "reaction_rate",
            "unit": f"reactions/source_history{suffix}",
            "integration_semantics": (
                "reaction events integrated over the audited domain"
                + (
                    " and separately over each incident-energy bin; bins are not summed"
                    if has_energy_axis
                    else " and all incident energies"
                )
            ),
            "divide_by_volume": False,
        }
    if score == "events" and production and not reaction:
        return {
            "observable": "particle_production",
            "unit": f"particles/source_history{suffix}",
            "integration_semantics": (
                "secondary-particle weight integrated over the audited domain, "
                "the declared outgoing-energy bin, and"
                + (
                    " separately over each incident-energy bin; bins are not summed"
                    if has_energy_axis
                    else " all incident energies"
                )
            ),
            "divide_by_volume": False,
        }
    raise ValueError(f"score {score!r} has ambiguous response semantics")


def response_result_to_consumer_estimators(
    result: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    materials_by_cell_id: Mapping[int, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Explode identity axes while retaining incident-energy response bins."""
    binding = _strict_binding(result, binding)
    axes = _filter_map(result)
    filters = list(result["filters"])
    dimensions = [int(row["n_bins"]) for row in filters]
    nuclides = [str(value) for value in result["nuclides"]]
    scores = [str(value) for value in result["scores"]]
    shape = tuple(dimensions) + (len(nuclides), len(scores))
    means = np.asarray(result["mean_per_source"], dtype=float)
    deviations = np.asarray(result["std_dev_per_source"], dtype=float)
    if means.shape != shape or deviations.shape != shape:
        raise ValueError("response arrays do not match their declared axes")

    cell_axis, cell_filter = axes["cell"]
    particle_axis, particle_filter = axes["particle"]
    cell_ids = [int(value) for value in cell_filter["bins"]]
    particles = [
        str(value).strip().lower() for value in particle_filter["bins"]
    ]
    if len(cell_ids) != int(cell_filter["n_bins"]) or any(
        value not in {"neutron", "photon"} for value in particles
    ):
        raise ValueError("cell or incident-particle identities are invalid")
    energy_axis = axes.get("energy")
    reaction_axis = axes.get("reaction")
    production_axis = axes.get("particleproduction")
    if energy_axis is not None:
        energy_edges = [float(value) for value in energy_axis[1]["bins"]]
    else:
        energy_edges = None

    reaction_bins: Sequence[Any] = (
        reaction_axis[1]["bins"] if reaction_axis is not None else (None,)
    )
    production_bins: Sequence[Any] = (
        production_axis[1]["bins"] if production_axis is not None else (None,)
    )
    mt_map = binding.get("reaction_mt_by_bin", {})
    if reaction_axis is not None:
        if not isinstance(mt_map, Mapping) or set(mt_map) != {
            str(value) for value in reaction_bins
        }:
            raise ValueError("ReactionFilter MT identities are incomplete")
    elif mt_map:
        raise ValueError(
            "reaction MT identities were bound without a reaction axis"
        )

    outputs: list[dict[str, Any]] = []
    for (
        cell_index,
        particle_index,
        nuclide_index,
        score_index,
        reaction_index,
        production_index,
    ) in product(
        range(len(cell_ids)),
        range(len(particles)),
        range(len(nuclides)),
        range(len(scores)),
        range(len(reaction_bins)),
        range(len(production_bins)),
    ):
        cell_id = cell_ids[cell_index]
        if cell_id not in materials_by_cell_id:
            raise ValueError(
                f"no audited material domain for OpenMC cell {cell_id}"
            )
        material = materials_by_cell_id[cell_id]
        score = scores[score_index]
        semantics = _score_semantics(
            score,
            has_energy_axis=energy_axis is not None,
            reaction=reaction_axis is not None,
            production=production_axis is not None,
        )
        selection: list[Any] = [slice(None)] * len(dimensions)
        selection[cell_axis] = cell_index
        selection[particle_axis] = particle_index
        if reaction_axis is not None:
            selection[reaction_axis[0]] = reaction_index
        if production_axis is not None:
            selection[production_axis[0]] = production_index
        selected_mean = means[tuple(selection) + (nuclide_index, score_index)]
        selected_std = deviations[
            tuple(selection) + (nuclide_index, score_index)
        ]
        selected_mean = np.asarray(selected_mean, dtype=float).reshape(-1)
        selected_std = np.asarray(selected_std, dtype=float).reshape(-1)
        expected_values = 1 if energy_edges is None else len(energy_edges) - 1
        if (
            len(selected_mean) != expected_values
            or len(selected_std) != expected_values
        ):
            raise ValueError("unresolved or ambiguous response axes remain")
        if semantics.pop("divide_by_volume"):
            volume = float(material["volume_cm3"])
            selected_mean = selected_mean / volume
            selected_std = selected_std / volume
        row = {
            **_common_estimator(
                result=result,
                binding=binding,
                material=material,
                provenance=provenance,
            ),
            **semantics,
            "score": score,
            "particle": particles[particle_index],
            "nuclide": nuclides[nuclide_index],
            "mean_per_source": selected_mean.tolist(),
            "std_dev_per_source": selected_std.tolist(),
            "result_state": (
                "EMPTY_OR_UNDER_SAMPLED"
                if np.all(selected_mean == 0.0)
                else "SCORED"
            ),
        }
        if energy_edges is not None:
            if row["observable"] == "scalar_flux_spectrum":
                row["energy_edges_eV"] = energy_edges
            else:
                row["response_axes"] = [
                    {
                        "axis": "incident_energy",
                        "unit": "eV",
                        "bin_edges": energy_edges,
                        "bin_semantics": "integrated_over_bin",
                    }
                ]
        nuclide = nuclides[nuclide_index]
        if reaction_axis is not None:
            raw_reaction = str(reaction_bins[reaction_index])
            mt = canonical_mt(mt_map[raw_reaction])
            row.update(
                {
                    "reaction_filter_bin": raw_reaction,
                    "mt": mt,
                    "reaction": mt_label(mt),
                }
            )
            if nuclide == "total":
                row["observable"] = "reaction_family_rate"
            else:
                row["nuclide"] = canonical_nuclide(nuclide)
        elif nuclide != "total":
            row["nuclide"] = canonical_nuclide(nuclide)
        if production_axis is not None:
            production_bin = production_bins[production_index]
            if not isinstance(production_bin, Mapping):
                raise ValueError("particle-production identity is malformed")
            produced = str(production_bin.get("produced_particle", "")).lower()
            if not produced:
                raise ValueError("produced-particle identity is missing")
            row["produced_particle"] = produced
            if "outgoing_energy_low_eV" in production_bin:
                low = float(production_bin["outgoing_energy_low_eV"])
                high = float(production_bin["outgoing_energy_high_eV"])
                if not (0.0 <= low < high):
                    raise ValueError("outgoing-energy identity is invalid")
                row["outgoing_energy_bin_eV"] = [low, high]
            else:
                row["outgoing_energy_integration"] = "all_outgoing_energies"
        outputs.append(row)
    return outputs


def build_openmc16_radiation_consumer_handoff(
    *,
    response_set: Mapping[str, Any],
    tally_bindings: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    materials: Sequence[Mapping[str, Any]],
    boundary_phase_space: Mapping[str, Any],
    activation_schedule_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a hash-bound consumer handoff directly from extracted results."""
    responses = _validate_response_set(response_set, provenance)
    response_ids = {str(row["tally_id"]) for row in responses}
    if set(tally_bindings) != response_ids:
        raise ValueError("tally bindings must exactly cover the response set")
    materials_by_cell: dict[int, Mapping[str, Any]] = {}
    for row in materials:
        cell_id = int(row.get("openmc_cell_id", -1))
        if cell_id <= 0 or cell_id in materials_by_cell:
            raise ValueError("material OpenMC cell identities are invalid")
        materials_by_cell[cell_id] = row
    estimators: list[dict[str, Any]] = []
    for result in responses:
        tally_id = str(result["tally_id"])
        estimators.extend(
            response_result_to_consumer_estimators(
                result,
                binding=tally_bindings[tally_id],
                materials_by_cell_id=materials_by_cell,
                provenance=provenance,
            )
        )
    handoff = build_radiation_consumer_handoff(
        provenance=provenance,
        materials=materials,
        volume_estimators=estimators,
        boundary_phase_space=boundary_phase_space,
        activation_schedule_reference=activation_schedule_reference,
    )
    handoff["producer_adapter"] = {
        "schema": SCHEMA,
        "status": "STATEPOINT_RESULTS_BOUND",
        "response_set_sha256": _canonical_sha256(response_set),
        "tally_bindings_sha256": _canonical_sha256(tally_bindings),
        "volume_estimators_sha256": _canonical_sha256(estimators),
        "statepoint_sha256": provenance["statepoint_sha256"],
        "covariance_status": "UNAVAILABLE_NOT_FABRICATED",
        "integrated_totals_derived": False,
    }
    validate_radiation_consumer_handoff(handoff)
    handoff["handoff_content_sha256"] = _canonical_sha256(handoff)
    validate_radiation_consumer_handoff(handoff)
    return handoff
