"""Portable OpenMC producer contract for downstream radiation workflows.

The contract deliberately separates two different Monte Carlo products:

* volume scalar-flux and reaction estimators for activation, SPECTRA-PKA,
  and Beyond-DPA; and
* a correlated surface-crossing bank for replay in local transport models.

The surface bank is the authoritative sampled phase-space measure.  A
deterministic angular/space/energy projection may be derived from it, but the
projection is never allowed to replace or renormalize the canonical bank.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "parastell.radiation_consumer_handoff/v1.0.0"
SCHEDULE_REFERENCE_SCHEMA = (
    "parastell.activation_campaign_schedule_reference/v1.0.0"
)

PHASE_SPACE_FIELDS = (
    "record_id",
    "position_global_cm",
    "position_local_cm",
    "direction_global",
    "direction_local",
    "energy_eV",
    "particle",
    "particle_pdg",
    "openmc_weight",
    "weight",
    "time_s",
    "delayed_group",
    "surface_id",
    "envelope_id",
    "crossing_sense",
    "surface_role",
    "outward_normal_global",
    "mu",
    "azimuth_rad",
    "grazing",
    "patch_id",
    "energy_group",
    "angle_bin_id",
    "facet_id",
    "canonical_facet_id",
    "barycentric_coordinates",
)

VOLUME_OBSERVABLES = {
    "scalar_flux_spectrum",
    "heating",
    "damage_energy",
    "hydrogen_production",
    "helium_production",
    "photon_production",
    "reaction_rate",
}


class RadiationHandoffError(ValueError):
    """Raised when a radiation-consumer bundle is ambiguous or unsafe."""


def _finite_positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RadiationHandoffError(f"{name} must be finite and positive")
    return number


def _sha256(value: Any, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise RadiationHandoffError(f"{name} must be a SHA-256 digest")
    return text


def build_activation_schedule_reference(
    *,
    schedule_id: str,
    schedule_sha256: str,
    verification_receipt_path: str | Path,
) -> dict[str, Any]:
    """Bind the producer handoff to a schedule owned by DPA_workflow.

    ParaStell intentionally does not duplicate executable depletion segments.
    The reference binds multiplier 1.0 to the accepted producer rate so a
    consumer cannot silently substitute a conflicting absolute source rate.
    """
    receipt_path = Path(verification_receipt_path).resolve()
    if not receipt_path.is_file():
        raise RadiationHandoffError(
            "activation schedule verification receipt does not exist"
        )
    reference = {
        "schema": SCHEDULE_REFERENCE_SCHEMA,
        "owner": "DPA_workflow",
        "schedule_schema": "dpa_workflow.activation_campaign_schedule/v1.0.0",
        "schedule_id": str(schedule_id),
        "schedule_sha256": str(schedule_sha256).lower(),
        "verification_status": "PASS",
        "verification_receipt_path": str(receipt_path),
        "verification_receipt_sha256": _file_sha256(receipt_path),
        "full_power_rate_binding": {
            "json_pointer": "/provenance/physical_source_rate_per_s",
            "multiplier": 1.0,
            "modeled_domain_scope_pointer": "/provenance/source_rate_scope",
        },
        "inline_executable_segments": False,
        "production_activation_authorized": False,
    }
    validate_activation_schedule_reference(reference)
    return reference


def validate_activation_schedule_reference(
    reference: Mapping[str, Any],
) -> None:
    """Reject copied rates, inline schedules, and ambiguous ownership."""
    if reference.get("schema") != SCHEDULE_REFERENCE_SCHEMA:
        raise RadiationHandoffError("unsupported schedule-reference schema")
    if reference.get("owner") != "DPA_workflow":
        raise RadiationHandoffError(
            "activation schedule owner must be DPA_workflow"
        )
    if reference.get("schedule_schema") != (
        "dpa_workflow.activation_campaign_schedule/v1.0.0"
    ):
        raise RadiationHandoffError(
            "unsupported DPA activation schedule schema"
        )
    if not str(reference.get("schedule_id", "")).strip():
        raise RadiationHandoffError("activation schedule ID is missing")
    _sha256(reference.get("schedule_sha256"), "activation schedule")
    if reference.get("verification_status") != "PASS":
        raise RadiationHandoffError("activation schedule is not verified")
    _sha256(
        reference.get("verification_receipt_sha256"),
        "activation schedule verification receipt",
    )
    receipt_path = Path(
        str(reference.get("verification_receipt_path", ""))
    ).resolve()
    if not receipt_path.is_file() or _file_sha256(
        receipt_path
    ) != reference.get("verification_receipt_sha256"):
        raise RadiationHandoffError(
            "activation schedule verification receipt hash mismatch"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RadiationHandoffError(
            "activation schedule verification receipt is invalid"
        ) from exc
    if (
        receipt.get("schema")
        != "dpa_workflow.activation_campaign_schedule_verification/v1.0.0"
        or receipt.get("status") != "PASS"
        or receipt.get("owner") != "DPA_workflow"
        or receipt.get("schedule_id") != reference.get("schedule_id")
        or receipt.get("schedule_sha256") != reference.get("schedule_sha256")
    ):
        raise RadiationHandoffError(
            "activation schedule verification receipt is not bound"
        )
    binding = reference.get("full_power_rate_binding")
    if not isinstance(binding, Mapping) or binding != {
        "json_pointer": "/provenance/physical_source_rate_per_s",
        "multiplier": 1.0,
        "modeled_domain_scope_pointer": "/provenance/source_rate_scope",
    }:
        raise RadiationHandoffError("activation source-rate binding is unsafe")
    if reference.get("inline_executable_segments") is not False:
        raise RadiationHandoffError("ParaStell cannot own depletion segments")
    if reference.get("production_activation_authorized") is not False:
        raise RadiationHandoffError("production activation is not authorized")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_radiation_consumer_handoff(
    *,
    provenance: Mapping[str, Any],
    materials: Sequence[Mapping[str, Any]],
    volume_estimators: Sequence[Mapping[str, Any]],
    boundary_phase_space: Mapping[str, Any],
    activation_schedule_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a geometry-neutral, workflow-completeness handoff.

    Low-history or synthetic fixtures are permitted, but their statistical
    class remains visible and cannot be promoted to a qualified physical
    field by this function.
    """
    bundle = {
        "schema": SCHEMA,
        "status": "WORKFLOW_CONTRACT_VALID",
        "provenance": dict(provenance),
        "materials": [dict(row) for row in materials],
        "volume_estimators": [dict(row) for row in volume_estimators],
        "boundary_phase_space": dict(boundary_phase_space),
        "activation_schedule_reference": dict(activation_schedule_reference),
        "consumer_routes": {
            "activation": {
                "input": "volume scalar-flux spectrum and reaction rates",
                "surface_bank_allowed_as_flux_substitute": False,
                "execution_owner": "DPA_workflow",
            },
            "spectra_pka": {
                "input": "bin-integrated neutron scalar-flux spectrum",
                "output": "recoil spectra computed by SPECTRA-PKA",
                "openmc_claims_to_directly_produce_pka": False,
            },
            "beyond_dpa": {
                "input": "SPECTRA-PKA recoil spectra plus material and response provenance",
                "openmc_flux_is_not_a_defect_yield": True,
            },
            "openmc_replay": _replay_route("continuous_correlated_bank"),
            "mcnp_replay": _replay_route("ssw_or_sdef_adapter"),
            "geant4_replay": _replay_route("primary_generator_adapter"),
            "deterministic_transport": {
                **_replay_route("binned_space_angle_energy_projection"),
                "projection_may_replace_canonical_bank": False,
            },
        },
    }
    validate_radiation_consumer_handoff(bundle)
    return bundle


def _replay_route(adapter: str) -> dict[str, Any]:
    return {
        "input": "canonical correlated boundary phase-space bank",
        "adapter": adapter,
        "preserve_joint_distribution": True,
        "renormalize_canonical_weights": False,
    }


def validate_radiation_consumer_handoff(bundle: Mapping[str, Any]) -> None:
    """Fail closed on missing provenance, wrong units, or consumer mixing."""
    if bundle.get("schema") != SCHEMA:
        raise RadiationHandoffError("unsupported radiation handoff schema")
    if bundle.get("status") != "WORKFLOW_CONTRACT_VALID":
        raise RadiationHandoffError("radiation workflow contract is invalid")
    provenance = bundle.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RadiationHandoffError("missing run provenance")
    for key in (
        "raw_h5m_sha256",
        "canonical_geometry_fingerprint",
        "source_definition_sha256",
        "statepoint_sha256",
        "nuclear_data_sha256",
        "model_xml_sha256",
        "settings_xml_sha256",
        "strict_run_audit_sha256",
        "root_acceptance_receipt_sha256",
    ):
        _sha256(provenance.get(key), key)
    if provenance.get("openmc_version") != "0.16.0":
        raise RadiationHandoffError("producer must identify OpenMC 0.16.0")
    histories = provenance.get("source_histories")
    if isinstance(histories, bool) or int(histories) <= 0:
        raise RadiationHandoffError("source histories must be positive")
    _finite_positive(
        provenance.get("physical_source_rate_per_s"), "source rate"
    )
    if not str(provenance.get("source_rate_scope", "")).strip():
        raise RadiationHandoffError("modeled source-rate scope is missing")
    if provenance.get("normalization") != "per_source_history":
        raise RadiationHandoffError(
            "canonical normalization must be per history"
        )
    if provenance.get("statistics_classification") not in {
        "WORKFLOW_SMOKE_ONLY",
        "INSUFFICIENT_STATISTICS",
        "QUALIFIED",
    }:
        raise RadiationHandoffError("statistics classification is missing")

    materials = bundle.get("materials")
    if not isinstance(materials, list) or not materials:
        raise RadiationHandoffError("material inventory is empty")
    domain_ids = set()
    openmc_cell_ids = set()
    dagmc_volume_ids = set()
    for row in materials:
        if not isinstance(row, Mapping):
            raise RadiationHandoffError("material row must be a mapping")
        domain_id = str(row.get("domain_id", "")).strip()
        if not domain_id or domain_id in domain_ids:
            raise RadiationHandoffError("material domain IDs must be unique")
        domain_ids.add(domain_id)
        if not str(row.get("material_id", "")).strip():
            raise RadiationHandoffError("material ID is missing")
        _sha256(row.get("composition_sha256"), "composition_sha256")
        _finite_positive(row.get("volume_cm3"), "material volume")
        openmc_cell_id = int(row.get("openmc_cell_id", -1))
        dagmc_volume_id = int(row.get("dagmc_volume_id", -1))
        if (
            openmc_cell_id <= 0
            or dagmc_volume_id <= 0
            or openmc_cell_id in openmc_cell_ids
            or dagmc_volume_id in dagmc_volume_ids
        ):
            raise RadiationHandoffError(
                "material cell/volume identities must be unique and positive"
            )
        openmc_cell_ids.add(openmc_cell_id)
        dagmc_volume_ids.add(dagmc_volume_id)
        _sha256(row.get("volume_receipt_sha256"), "volume receipt")
        if row.get("raw_h5m_sha256") != provenance.get(
            "raw_h5m_sha256"
        ) or row.get("canonical_geometry_fingerprint") != provenance.get(
            "canonical_geometry_fingerprint"
        ):
            raise RadiationHandoffError(
                "material volume used another geometry"
            )

    estimators = bundle.get("volume_estimators")
    if not isinstance(estimators, list) or not estimators:
        raise RadiationHandoffError("volume estimators are empty")
    neutron_spectrum_domains: list[str] = []
    materials_by_domain = {str(row["domain_id"]): row for row in materials}
    for row in estimators:
        _validate_estimator(row, domain_ids, provenance, materials_by_domain)
        if (
            row.get("observable") == "scalar_flux_spectrum"
            and row.get("particle") == "neutron"
        ):
            neutron_spectrum_domains.append(str(row["domain_id"]))
    if set(neutron_spectrum_domains) != domain_ids or len(
        neutron_spectrum_domains
    ) != len(domain_ids):
        raise RadiationHandoffError(
            "every material domain needs a neutron scalar-flux spectrum"
        )

    bank = bundle.get("boundary_phase_space")
    if not isinstance(bank, Mapping):
        raise RadiationHandoffError(
            "boundary phase-space descriptor is missing"
        )
    if bank.get("bank_classification") != "COMPLETE_CROSSING_BANK":
        raise RadiationHandoffError(
            "canonical replay requires a complete crossing bank"
        )
    _sha256(bank.get("artifact_sha256"), "boundary bank artifact")
    _sha256(bank.get("envelope_manifest_sha256"), "boundary envelope")
    for key in (
        "statepoint_sha256",
        "raw_h5m_sha256",
        "canonical_geometry_fingerprint",
        "strict_run_audit_sha256",
        "root_acceptance_receipt_sha256",
    ):
        if bank.get(key) != provenance.get(key):
            raise RadiationHandoffError(
                f"boundary bank {key} is not run-bound"
            )
    if bank.get("canonical_weight_semantics") != (
        "raw_openmc_weight_divided_only_by_exact_source_histories"
    ):
        raise RadiationHandoffError(
            "canonical bank weights were reconditioned"
        )
    fields = tuple(bank.get("fields", ()))
    missing = sorted(set(PHASE_SPACE_FIELDS) - set(fields))
    if missing:
        raise RadiationHandoffError(
            "boundary bank is missing phase fields: " + ", ".join(missing)
        )
    if bank.get("joint_records_preserved") is not True:
        raise RadiationHandoffError(
            "phase-space correlations were not preserved"
        )
    if bank.get("parent_history_available") is not False:
        raise RadiationHandoffError(
            "OpenMC 0.16 parent-history limitation must be explicit"
        )
    if bank.get("polarization_available") is not False:
        raise RadiationHandoffError(
            "OpenMC 0.16 polarization limitation must be explicit"
        )

    validate_activation_schedule_reference(
        bundle.get("activation_schedule_reference", {})
    )
    routes = bundle.get("consumer_routes")
    if not isinstance(routes, Mapping):
        raise RadiationHandoffError("consumer routes are missing")
    if (
        routes.get("activation", {}).get(
            "surface_bank_allowed_as_flux_substitute"
        )
        is not False
    ):
        raise RadiationHandoffError(
            "surface current cannot substitute for scalar flux"
        )
    if (
        routes.get("spectra_pka", {}).get(
            "openmc_claims_to_directly_produce_pka"
        )
        is not False
    ):
        raise RadiationHandoffError("OpenMC cannot be labeled a PKA solver")
    for name in ("openmc_replay", "mcnp_replay", "geant4_replay"):
        route = routes.get(name)
        if (
            not isinstance(route, Mapping)
            or route.get("preserve_joint_distribution") is not True
            or route.get("renormalize_canonical_weights") is not False
        ):
            raise RadiationHandoffError(f"unsafe {name} route")
    deterministic = routes.get("deterministic_transport")
    if (
        not isinstance(deterministic, Mapping)
        or deterministic.get("projection_may_replace_canonical_bank")
        is not False
    ):
        raise RadiationHandoffError(
            "deterministic projection replaced canonical bank"
        )


def _validate_estimator(
    row: Any,
    domain_ids: set[str],
    provenance: Mapping[str, Any],
    materials_by_domain: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(row, Mapping):
        raise RadiationHandoffError("volume estimator row must be a mapping")
    domain_id = str(row.get("domain_id", ""))
    if domain_id not in domain_ids:
        raise RadiationHandoffError(
            "estimator domain is not in material inventory"
        )
    observable = row.get("observable")
    if observable not in VOLUME_OBSERVABLES:
        raise RadiationHandoffError("unsupported volume observable")
    if row.get("estimator") not in {"tracklength", "collision", "analog"}:
        raise RadiationHandoffError("estimator type is missing")
    if row.get("normalization") != "per_source_history":
        raise RadiationHandoffError("volume estimator normalization is unsafe")
    material = materials_by_domain[domain_id]
    for key in (
        "openmc_cell_id",
        "dagmc_volume_id",
        "volume_cm3",
        "volume_receipt_sha256",
    ):
        if row.get(key) != material.get(key):
            raise RadiationHandoffError(
                f"estimator {key} does not match its material domain"
            )
    for key in (
        "statepoint_sha256",
        "raw_h5m_sha256",
        "canonical_geometry_fingerprint",
        "model_xml_sha256",
        "settings_xml_sha256",
        "strict_run_audit_sha256",
        "root_acceptance_receipt_sha256",
    ):
        _sha256(row.get(key), f"estimator {key}")
        if row.get(key) != provenance.get(key):
            raise RadiationHandoffError(f"estimator {key} is not run-bound")
    _sha256(row.get("tally_definition_sha256"), "tally definition")
    if int(row.get("source_histories", -1)) != int(
        provenance.get("source_histories", -2)
    ):
        raise RadiationHandoffError(
            "estimator source histories are not run-bound"
        )
    values = row.get("mean_per_source")
    deviations = row.get("std_dev_per_source")
    if (
        not isinstance(values, list)
        or not values
        or not isinstance(deviations, list)
    ):
        raise RadiationHandoffError(
            "estimator mean/std-dev arrays are missing"
        )
    if len(values) != len(deviations):
        raise RadiationHandoffError("estimator mean/std-dev lengths differ")
    for value in [*values, *deviations]:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise RadiationHandoffError("estimator contains invalid values")
    if observable == "scalar_flux_spectrum":
        if row.get("particle") not in {"neutron", "photon"}:
            raise RadiationHandoffError("scalar spectrum particle is invalid")
        if row.get("unit") != "1/cm2/source_history/bin":
            raise RadiationHandoffError(
                "scalar spectrum must be bin integrated"
            )
        edges = row.get("energy_edges_eV")
        if not isinstance(edges, list) or len(edges) != len(values) + 1:
            raise RadiationHandoffError("spectrum energy edges are malformed")
        numeric_edges = [float(value) for value in edges]
        if any(
            not math.isfinite(value) or value < 0.0 for value in numeric_edges
        ):
            raise RadiationHandoffError("spectrum energy edge is invalid")
        if numeric_edges != sorted(set(numeric_edges)):
            raise RadiationHandoffError("spectrum energy edges must increase")
        if row.get("estimator") != "tracklength":
            raise RadiationHandoffError(
                "scalar spectrum requires track-length estimator"
            )
    elif "energy_edges_eV" in row:
        raise RadiationHandoffError("non-spectrum estimator has energy edges")
    else:
        expected_units = {
            "heating": "eV/source_history",
            "damage_energy": "eV/source_history",
            "hydrogen_production": "atoms/source_history",
            "helium_production": "atoms/source_history",
            "photon_production": "particles/source_history",
            "reaction_rate": "reactions/source_history",
        }
        if row.get("unit") != expected_units[observable]:
            raise RadiationHandoffError(
                f"{observable} estimator unit is invalid"
            )
    if observable == "reaction_rate":
        if (
            not str(row.get("nuclide", "")).strip()
            or not str(row.get("reaction", "")).strip()
        ):
            raise RadiationHandoffError("reaction-rate identity is incomplete")


def spectra_pka_inputs(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project validated neutron spectra into the DPA_workflow input fields."""
    validate_radiation_consumer_handoff(bundle)
    provenance = bundle["provenance"]
    material_by_domain = {
        str(row["domain_id"]): row for row in bundle["materials"]
    }
    outputs = []
    for row in bundle["volume_estimators"]:
        if (
            row.get("observable") != "scalar_flux_spectrum"
            or row.get("particle") != "neutron"
        ):
            continue
        material = material_by_domain[str(row["domain_id"])]
        outputs.append(
            {
                "layer_id": row["domain_id"],
                "material_id": material["material_id"],
                "energy_bin_eV": list(row["energy_edges_eV"]),
                "bin_integrated_flux": list(row["mean_per_source"]),
                "bin_integrated_flux_std_dev": list(row["std_dev_per_source"]),
                "normalization": "per_source_history",
                "physical_source_rate_per_s": provenance[
                    "physical_source_rate_per_s"
                ],
                "openmc_version": provenance["openmc_version"],
                "nuclear_data_library": provenance["nuclear_data_library"],
                "cross_sections_xml_sha256": provenance["nuclear_data_sha256"],
                "tally_id": row["tally_id"],
                "geometry_hash": provenance["raw_h5m_sha256"],
            }
        )
    return outputs
