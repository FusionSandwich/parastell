"""Strict converters from ParaStell radiation handoffs to downstream inputs.

These functions prepare transport/activation/PKA inputs; they do not execute
activation, SPECTRA-PKA, Beyond-DPA, or a local magnet solver.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .radiation_consumer_handoff import (
    PHASE_SPACE_FIELDS,
    spectra_pka_inputs,
    validate_radiation_consumer_handoff,
)
from .reaction_identity import reaction_matrix_identity


SCHEMA = "parastell.downstream_response_exports/v1.0.0"


def _material_payload(material: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "material_id",
        "composition_sha256",
        "volume_cm3",
        "density_g_cm3",
        "temperature_K",
        "composition_basis",
        "isotopes",
    )
    missing = [key for key in required if material.get(key) is None]
    if missing:
        raise ValueError(
            "material identity is incomplete: " + ", ".join(missing)
        )
    density = float(material["density_g_cm3"])
    volume = float(material["volume_cm3"])
    temperature = float(material["temperature_K"])
    isotopes = material["isotopes"]
    if (
        not math.isfinite(volume)
        or volume <= 0.0
        or not math.isfinite(density)
        or density <= 0.0
        or not math.isfinite(temperature)
        or temperature <= 0.0
        or not isinstance(isotopes, Mapping)
        or not isotopes
    ):
        raise ValueError(
            "material density, temperature, or isotopes are invalid"
        )
    payload = {key: material[key] for key in required}
    if "alara_constituents" in material:
        constituents = material["alara_constituents"]
        if not isinstance(constituents, list) or not constituents:
            raise ValueError("ALARA constituents must be a nonempty list")
        payload["alara_constituents"] = [dict(row) for row in constituents]
    return payload


def build_downstream_exports(
    handoff: Mapping[str, Any],
    *,
    ownership_contribution_id: str,
    delayed_photon_source_id: str | None = None,
) -> dict[str, Any]:
    """Create portable inputs while retaining one authoritative normalization."""
    validate_radiation_consumer_handoff(handoff)
    contribution_id = str(ownership_contribution_id).strip()
    if not contribution_id:
        raise ValueError("ownership contribution ID is required")
    provenance = handoff["provenance"]
    materials = {
        str(item["domain_id"]): _material_payload(item)
        for item in handoff["materials"]
    }
    pka = spectra_pka_inputs(handoff)
    for row in pka:
        row["material"] = materials[str(row["layer_id"])]
        row["ownership_contribution_id"] = contribution_id
        row["covariance"] = {
            "status": "UNAVAILABLE",
            "reason": "OpenMC statepoint exports per-bin moments, not covariance",
        }

    activation = []
    for estimator in handoff["volume_estimators"]:
        if (
            estimator.get("observable") == "scalar_flux_spectrum"
            and estimator.get("particle") == "neutron"
        ):
            domain_id = str(estimator["domain_id"])
            activation.append(
                {
                    "domain_id": domain_id,
                    "material": materials[domain_id],
                    "volume_cm3": materials[domain_id]["volume_cm3"],
                    "energy_edges_eV": list(estimator["energy_edges_eV"]),
                    "bin_integrated_flux_per_source": list(
                        estimator["mean_per_source"]
                    ),
                    "bin_integrated_flux_std_dev_per_source": list(
                        estimator["std_dev_per_source"]
                    ),
                    "physical_source_rate_per_s": provenance[
                        "physical_source_rate_per_s"
                    ],
                    "source_rate_scope": provenance["source_rate_scope"],
                    "normalization_operation": (
                        "multiply_per_source_flux_once_by_physical_source_rate"
                    ),
                    "surface_bank_used_as_flux": False,
                    "ownership_contribution_id": contribution_id,
                }
            )

    reaction_matrix = []
    for estimator in handoff["volume_estimators"]:
        if estimator.get("observable") != "reaction_rate":
            continue
        identity = reaction_matrix_identity(
            nuclide=estimator["nuclide"],
            mt=estimator["mt"],
            nuclear_data_sha256=provenance["nuclear_data_sha256"],
        )
        reaction_matrix.append(
            {
                "domain_id": estimator["domain_id"],
                **identity,
                "energy_edges_eV": estimator.get("energy_edges_eV"),
                "mean_per_source": list(estimator["mean_per_source"]),
                "std_dev_per_source": list(estimator["std_dev_per_source"]),
                "covariance": estimator.get(
                    "covariance",
                    {
                        "status": "UNAVAILABLE",
                        "reason": "not scored by the producer",
                    },
                ),
                "ownership_contribution_id": contribution_id,
            }
        )

    bank = handoff["boundary_phase_space"]
    fields = tuple(bank["fields"])
    if not set(PHASE_SPACE_FIELDS).issubset(fields):
        raise ValueError(
            "replay export is missing correlated phase-space fields"
        )
    prompt_source_id = f"{contribution_id}:prompt-boundary"
    if delayed_photon_source_id is not None and (
        str(delayed_photon_source_id).strip() == prompt_source_id
    ):
        raise ValueError("prompt and delayed photon source IDs collide")
    replay = {
        "prompt_source_id": prompt_source_id,
        "delayed_photon_source_id": (
            None
            if delayed_photon_source_id is None
            else str(delayed_photon_source_id).strip()
        ),
        "artifact_sha256": bank["artifact_sha256"],
        "fields": list(fields),
        "canonical_weight_semantics": bank["canonical_weight_semantics"],
        "joint_records_preserved": True,
        "normalization": "per_source_history",
        "physical_source_rate_per_s": provenance["physical_source_rate_per_s"],
        "source_rate_may_be_applied_more_than_once": False,
        "solver_routes": {
            "openmc": "correlated_particle_list_replay",
            "mcnp6.3": "correlated_ssr_or_generated_source_replay",
            "geant4": "correlated_primary_generator_replay",
            "opensn": "conservative_boundary_moment_projection",
            "radiant": "conservative_boundary_moment_projection",
        },
    }
    return {
        "schema": SCHEMA,
        "status": "IMPORT_INPUTS_VALIDATED",
        "claim": "WORKFLOW_SMOKE_ONLY",
        "provenance": dict(provenance),
        "spectra_pka": pka,
        "activation": activation,
        "nuclide_mt_reaction_matrix": reaction_matrix,
        "magnet_boundary_replay": replay,
        "activation_schedule_reference": dict(
            handoff["activation_schedule_reference"]
        ),
        "production_run_authorized": False,
    }


def write_downstream_exports(
    directory: str | Path, exports: Mapping[str, Any]
) -> list[Path]:
    """Write deterministic, solver-neutral JSON files for consumer import."""
    if exports.get("schema") != SCHEMA or exports.get("status") != (
        "IMPORT_INPUTS_VALIDATED"
    ):
        raise ValueError("downstream exports were not validated")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    payloads = {
        "spectra_pka_inputs.json": exports["spectra_pka"],
        "activation_inputs.json": exports["activation"],
        "nuclide_mt_reaction_matrix.json": exports[
            "nuclide_mt_reaction_matrix"
        ],
        "magnet_boundary_replay.json": exports["magnet_boundary_replay"],
        "downstream_export_manifest.json": exports,
    }
    paths = []
    for name, payload in payloads.items():
        path = root / name
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths
