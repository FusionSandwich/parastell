"""Hash-bound ALARA delayed-photon spectra for downstream source adapters.

The bridge consumes ALARA 2.9.x tab-separated gamma-source output.  It uses
only ``TOTAL`` rows, preserves each activation zone and cooling snapshot, and
never combines prompt and delayed photons.  Monte Carlo routes receive a
probability distribution plus its absolute photons/s multiplier; deterministic
routes receive the corresponding zone-average photons/cm3/s multigroup source.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .alara_activation import (
    qualified_alara_runtime,
    validate_alara_handoff,
)


SCHEMA = "parastell.delayed_photon_solver_handoff/v1.0.0"
PACKAGE_SCHEMA = "parastell.delayed_photon_solver_package/v1.0.0"
SOLVERS = ("openmc", "mcnp", "geant4", "opensn", "radiant")
NEAR_ZERO_ALIAS_MAX_S = 1.0e-12
NEAR_ZERO_ALIAS_SPECTRUM_REL_TOL = 1.0e-12
NEAR_ZERO_ALIAS_SPECTRUM_ABS_TOL = 0.0


class DelayedPhotonSourceError(ValueError):
    """Raised when a delayed-photon source is ambiguous or unbound."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: Any, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise DelayedPhotonSourceError(f"{name} must be a SHA-256 digest")
    return text


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DelayedPhotonSourceError(f"{name} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise DelayedPhotonSourceError(f"{name} must contain a JSON object")
    return value


_ALARA_TIME = re.compile(
    r"^([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([smhdwy])$"
)


def _time_offset_s(label: str) -> float:
    normalized = " ".join(label.split()).lower()
    if normalized == "shutdown":
        return 0.0
    match = _ALARA_TIME.fullmatch(normalized)
    if match is None:
        raise DelayedPhotonSourceError(f"invalid ALARA cooling label: {label}")
    multipliers = {
        "s": 1.0,
        "m": 60.0,
        "h": 3_600.0,
        "d": 86_400.0,
        "w": 604_800.0,
        "y": 31_557_600.0,
    }
    value = float(match.group(1)) * multipliers[match.group(2)]
    if not math.isfinite(value) or value < 0.0:
        raise DelayedPhotonSourceError(f"invalid ALARA cooling label: {label}")
    return value


def _parse_rows(
    text: str,
    *,
    zone_ids: Sequence[str],
    cooling_offset_s: Sequence[int],
    group_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse qualified ALARA 2.9.x tab-separated gamma-source rows."""
    expected_offsets = [int(value) for value in cooling_offset_s]
    if expected_offsets != sorted(set(expected_offsets)):
        raise DelayedPhotonSourceError(
            "cooling offsets are not unique and sorted"
        )
    blocks: list[list[tuple[str, str, list[float]]]] = [[]]
    previous_nuclide: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        if len(fields) != group_count + 2:
            raise DelayedPhotonSourceError(
                f"ALARA photon row {line_number} has the wrong field count"
            )
        nuclide = fields[0].strip()
        time_label = " ".join(fields[1].split())
        if not nuclide or not time_label:
            raise DelayedPhotonSourceError(
                f"ALARA photon row {line_number} has an empty identity"
            )
        try:
            values = [float(value) for value in fields[2:]]
        except ValueError as exc:
            raise DelayedPhotonSourceError(
                f"ALARA photon row {line_number} is not numeric"
            ) from exc
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise DelayedPhotonSourceError(
                f"ALARA photon row {line_number} has an invalid rate"
            )
        if previous_nuclide == "TOTAL" and nuclide != "TOTAL":
            blocks.append([])
        blocks[-1].append((nuclide, time_label, values))
        previous_nuclide = nuclide

    if len(blocks) != len(zone_ids):
        raise DelayedPhotonSourceError(
            "ALARA photon zone count does not match the execution binding"
        )
    parsed = []
    accepted_aliases = []
    for zone_id, block in zip(zone_ids, blocks):
        totals: dict[int, tuple[str, list[float]]] = {}
        for nuclide, time_label, values in block:
            if nuclide != "TOTAL":
                continue
            parsed_offset = _time_offset_s(time_label)
            matches = [
                offset
                for offset in expected_offsets
                if math.isclose(
                    parsed_offset,
                    float(offset),
                    rel_tol=0.0,
                    abs_tol=max(1.0e-12, abs(float(offset)) * 1.0e-14),
                )
            ]
            if len(matches) != 1:
                raise DelayedPhotonSourceError(
                    f"ALARA photon cooling time is not scheduled: {time_label}"
                )
            offset = matches[0]
            if offset in totals:
                previous_label, previous_values = totals[offset]
                labels = {previous_label.lower(), time_label.lower()}
                offsets = {
                    _time_offset_s(previous_label),
                    _time_offset_s(time_label),
                }
                is_shutdown_alias = (
                    offset == 0
                    and "shutdown" in labels
                    and len(labels) == 2
                    and 0.0 < max(offsets)
                    and max(offsets) <= NEAR_ZERO_ALIAS_MAX_S
                )
                spectra_match = is_shutdown_alias and all(
                    math.isclose(
                        old,
                        new,
                        rel_tol=NEAR_ZERO_ALIAS_SPECTRUM_REL_TOL,
                        abs_tol=NEAR_ZERO_ALIAS_SPECTRUM_ABS_TOL,
                    )
                    for old, new in zip(previous_values, values)
                )
                if not spectra_match:
                    raise DelayedPhotonSourceError(
                        "ALARA photon source has conflicting duplicate TOTAL rows"
                    )
                canonical_values = (
                    previous_values
                    if previous_label.lower() == "shutdown"
                    else values
                )
                absolute_differences = [
                    abs(old - new) for old, new in zip(previous_values, values)
                ]
                relative_differences = [
                    (
                        difference / max(abs(old), abs(new))
                        if max(abs(old), abs(new)) > 0.0
                        else 0.0
                    )
                    for old, new, difference in zip(
                        previous_values, values, absolute_differences
                    )
                ]
                totals[offset] = ("shutdown", canonical_values)
                accepted_aliases.append(
                    {
                        "zone_id": zone_id,
                        "canonical_cooling_offset_s": 0,
                        "canonical_label": "shutdown",
                        "alias_label": next(
                            label for label in labels if label != "shutdown"
                        ),
                        "alias_offset_s": max(offsets),
                        "spectra_match_status": "PASS",
                        "maximum_absolute_difference": max(
                            absolute_differences, default=0.0
                        ),
                        "maximum_relative_difference": max(
                            relative_differences, default=0.0
                        ),
                    }
                )
                continue
            totals[offset] = (time_label, values)
        if set(totals) != set(expected_offsets):
            raise DelayedPhotonSourceError(
                f"ALARA photon TOTAL rows are incomplete for zone {zone_id}"
            )
        for offset in expected_offsets:
            label, values = totals[offset]
            parsed.append(
                {
                    "zone_id": zone_id,
                    "cooling_time_label": label,
                    "cooling_offset_s": offset,
                    "group_emission_rate_density_photons_per_cm3_s": values,
                }
            )
    return parsed, accepted_aliases


def _monte_carlo_route(solver: str) -> dict[str, Any]:
    adapters = {
        "openmc": "IndependentSource_or_compiled_source_adapter",
        "mcnp": "SDEF_SI_SP_cell_distribution_adapter",
        "geant4": "G4GeneralParticleSource_histogram_adapter",
    }
    return {
        "solver": solver,
        "adapter": adapters[solver],
        "particle": "photon",
        "snapshot_selection_required": True,
        "zone_sampling": "probability_proportional_to_zone_emission_rate",
        "energy_sampling": "probability_proportional_to_bin_emission_rate",
        "within_bin_sampling": "uniform_in_energy",
        "space_sampling": "uniform_within_bound_activation_zone",
        "angle_sampling": "isotropic_4pi",
        "transport_normalization": "per_sampled_delayed_photon",
        "absolute_multiplier_pointer": (
            "/snapshots/*/total_emission_rate_photons_per_s"
        ),
        "prompt_photon_source_may_be_merged": False,
    }


def _deterministic_route(solver: str) -> dict[str, Any]:
    return {
        "solver": solver,
        "adapter": "isotropic_multigroup_volume_source",
        "particle": "photon",
        "snapshot_selection_required": True,
        "source_pointer": (
            "/snapshots/*/zones/*/group_emission_rate_density_photons_per_cm3_s"
        ),
        "energy_group_order": "ascending_energy",
        "angular_representation": "isotropic_scalar_source_over_4pi",
        "spatial_representation": "zone_average_volume_source",
        "prompt_photon_source_may_be_merged": False,
    }


def build_delayed_photon_solver_handoff(
    *,
    delayed_photon_path: str | Path,
    alara_result_receipt_path: str | Path,
    alara_input_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build one hash-bound, solver-neutral delayed-photon source handoff."""
    photon_path = Path(delayed_photon_path).resolve()
    result_path = Path(alara_result_receipt_path).resolve()
    manifest_path = Path(alara_input_manifest_path).resolve()
    if not photon_path.is_file():
        raise FileNotFoundError(photon_path)
    receipt = _read_json(result_path, "ALARA result receipt")
    package = _read_json(manifest_path, "ALARA input manifest")
    package_sha256 = _file_sha256(manifest_path)
    photon_sha256 = _file_sha256(photon_path)

    if package.get("schema") != "parastell.alara_input_package/v1.0.0":
        raise DelayedPhotonSourceError("unsupported ALARA input manifest")
    handoff = package.get("handoff")
    if not isinstance(handoff, Mapping):
        raise DelayedPhotonSourceError("ALARA input handoff is missing")
    try:
        validate_alara_handoff(handoff)
    except ValueError as exc:
        raise DelayedPhotonSourceError(
            "ALARA input handoff is invalid"
        ) from exc

    if (
        receipt.get("schema") != "parastell.alara_bounded_result/v1.0.0"
        or receipt.get("status") != "PASSED"
        or receipt.get("claim") != "WORKFLOW_SMOKE_ONLY"
        or receipt.get("provenance_origin")
        not in {"SYNTHETIC_WORKFLOW_FIXTURE", "PHYSICAL_OPENMC_STATEPOINT"}
        or receipt.get("production_activation_authorized") is not False
    ):
        raise DelayedPhotonSourceError("ALARA result receipt did not pass")
    if receipt.get("runtime") != qualified_alara_runtime():
        raise DelayedPhotonSourceError("ALARA result used another runtime")
    if receipt.get("input_manifest_sha256") != package_sha256:
        raise DelayedPhotonSourceError(
            "ALARA result used another input manifest"
        )
    if receipt.get("provenance_origin") != handoff["openmc_provenance"].get(
        "provenance_origin"
    ):
        raise DelayedPhotonSourceError(
            "ALARA result and OpenMC handoff provenance origins differ"
        )
    photon_file = receipt.get("files", {}).get("delayed_photon_source", {})
    if (
        photon_file.get("sha256") != photon_sha256
        or int(photon_file.get("size_bytes", -1)) != photon_path.stat().st_size
    ):
        raise DelayedPhotonSourceError("delayed-photon artifact hash mismatch")

    binding = receipt.get("execution_binding")
    if not isinstance(binding, Mapping):
        raise DelayedPhotonSourceError(
            "ALARA result lacks checkpoint and zone execution binding"
        )
    irradiation_s = int(binding.get("irradiation_checkpoint_s", -1))
    if irradiation_s not in handoff["campaign"]["irradiation_checkpoint_s"]:
        raise DelayedPhotonSourceError(
            "irradiation checkpoint is not scheduled"
        )
    expected_zone_ids = [str(zone["domain_id"]) for zone in handoff["zones"]]
    if list(binding.get("activation_zone_ids", ())) != expected_zone_ids:
        raise DelayedPhotonSourceError("ALARA result zone order is not bound")
    cooling_offsets = [
        int(value) for value in handoff["campaign"]["cooling_offset_s"]
    ]
    if list(binding.get("cooling_offset_s", ())) != cooling_offsets:
        raise DelayedPhotonSourceError(
            "ALARA result cooling schedule is not bound"
        )
    input_relative = f"irradiation_{irradiation_s:012d}s/alara.inp"
    input_rows = {
        str(row.get("path")): row for row in package.get("files", ())
    }
    input_row = input_rows.get(input_relative)
    if not isinstance(input_row, Mapping) or binding.get(
        "alara_input_sha256"
    ) != input_row.get("sha256"):
        raise DelayedPhotonSourceError(
            "executed ALARA checkpoint deck is not bound"
        )

    photon_output = handoff["delayed_photon_output"]
    edges = [float(value) for value in photon_output["energy_edges_eV"]]
    records, accepted_aliases = _parse_rows(
        photon_path.read_text(encoding="utf-8"),
        zone_ids=expected_zone_ids,
        cooling_offset_s=cooling_offsets,
        group_count=len(edges) - 1,
    )
    volumes = {
        str(zone["domain_id"]): float(zone["volume_cm3"])
        for zone in handoff["zones"]
    }
    snapshots = []
    for cooling_s in cooling_offsets:
        zone_rows = []
        for row in records:
            if row["cooling_offset_s"] != cooling_s:
                continue
            volume = volumes[row["zone_id"]]
            rate_density = row["group_emission_rate_density_photons_per_cm3_s"]
            rates = [value * volume for value in rate_density]
            zone_rate = sum(rates)
            zone_rows.append(
                {
                    **row,
                    "volume_cm3": volume,
                    "group_emission_rate_photons_per_s": rates,
                    "total_emission_rate_photons_per_s": zone_rate,
                    "group_probability": (
                        [value / zone_rate for value in rates]
                        if zone_rate > 0.0
                        else [0.0 for value in rates]
                    ),
                    "uncertainty": {
                        "status": "UNAVAILABLE_NOT_PROPAGATED",
                        "group_std_dev": None,
                    },
                    "covariance": {
                        "status": "UNAVAILABLE_NOT_FABRICATED",
                        "matrix": None,
                    },
                }
            )
        total_rate = sum(
            row["total_emission_rate_photons_per_s"] for row in zone_rows
        )
        for row in zone_rows:
            row["zone_probability"] = (
                row["total_emission_rate_photons_per_s"] / total_rate
                if total_rate > 0.0
                else 0.0
            )
        snapshots.append(
            {
                "irradiation_checkpoint_s": irradiation_s,
                "cooling_offset_s": cooling_s,
                "cooling_time_label": next(
                    row["cooling_time_label"] for row in zone_rows
                ),
                "total_emission_rate_photons_per_s": total_rate,
                "zones": zone_rows,
            }
        )

    openmc_provenance = handoff["openmc_provenance"]
    result = {
        "schema": SCHEMA,
        "status": "SOLVER_SOURCE_CONTRACTS_READY",
        "claim": receipt.get("claim"),
        "provenance_origin": receipt.get("provenance_origin"),
        "particle": "photon",
        "source_class": "delayed_activation_photon",
        "energy": {
            "energy_edges_eV": edges,
            "group_count": len(edges) - 1,
            "group_order": "ascending_energy",
            "value_semantics": "bin_integrated_emission_rate_density",
        },
        "absolute_normalization": {
            "alara_source_value_unit": "photons/cm3/s/bin",
            "zone_integrated_value_unit": "photons/s/bin/zone",
            "zone_integrated_rate_operation": (
                "multiply_once_by_bound_zone_volume_cm3"
            ),
            "openmc_neutron_input_normalization": handoff["normalization"],
            "physical_neutron_source_rate_per_s": openmc_provenance[
                "physical_source_rate_per_s"
            ],
            "physical_neutron_source_rate_scope": openmc_provenance[
                "source_rate_scope"
            ],
            "alara_output_density_is_absolute": True,
            "bound_zone_volume_applied_exactly_once": True,
            "additional_rate_multiplier_allowed": False,
        },
        "activation_binding": {
            "campaign_id": handoff["campaign"]["campaign_id"],
            "irradiation_checkpoint_s": irradiation_s,
            "cooling_offset_s": cooling_offsets,
            "zone_ids": expected_zone_ids,
            "zone_order_semantics": "alara_input_handoff_order",
        },
        "near_zero_time_alias_policy": {
            "maximum_alias_offset_s": NEAR_ZERO_ALIAS_MAX_S,
            "spectrum_relative_tolerance": (NEAR_ZERO_ALIAS_SPECTRUM_REL_TOL),
            "spectrum_absolute_tolerance": (NEAR_ZERO_ALIAS_SPECTRUM_ABS_TOL),
            "canonical_label": "shutdown",
            "accepted_aliases": accepted_aliases,
        },
        "artifact_provenance": {
            "alara_input_manifest_sha256": package_sha256,
            "alara_checkpoint_input_sha256": input_row["sha256"],
            "alara_result_receipt_sha256": _file_sha256(result_path),
            "delayed_photon_source_sha256": photon_sha256,
            "alara_runtime": receipt["runtime"],
            "openmc_provenance": openmc_provenance,
        },
        "source_assumptions": {
            "space": "uniform_within_each_bound_activation_zone",
            "angle": "isotropic_4pi",
            "within_energy_bin": "uniform_in_energy",
            "time": "steady_snapshot_at_exact_cooling_offset",
            "polarization": "unavailable_unpolarized_approximation",
            "space_energy_angle_correlations": "unavailable_not_fabricated",
        },
        "uncertainty": {
            "status": "UNAVAILABLE_NOT_PROPAGATED",
            "zero_uncertainty_may_be_claimed": False,
        },
        "covariance": {
            "status": "UNAVAILABLE_NOT_FABRICATED",
            "zero_covariance_may_be_claimed": False,
        },
        "snapshots": snapshots,
        "solver_routes": {
            "openmc": _monte_carlo_route("openmc"),
            "mcnp": _monte_carlo_route("mcnp"),
            "geant4": _monte_carlo_route("geant4"),
            "opensn": _deterministic_route("opensn"),
            "radiant": _deterministic_route("radiant"),
        },
        "prompt_delayed_separation": {
            "prompt_photons_included": False,
            "delayed_photons_included": True,
            "combination_requires_separate_normalization_ledger": True,
        },
        "production_transport_authorized": False,
    }
    validate_delayed_photon_solver_handoff(result)
    result["handoff_content_sha256"] = _canonical_sha256(result)
    return result


def validate_delayed_photon_solver_handoff(bundle: Mapping[str, Any]) -> None:
    """Fail closed on altered provenance, units, schedules, or solver routes."""
    if bundle.get("schema") != SCHEMA or bundle.get("status") != (
        "SOLVER_SOURCE_CONTRACTS_READY"
    ):
        raise DelayedPhotonSourceError("unsupported delayed-photon handoff")
    if bundle.get("particle") != "photon" or bundle.get("source_class") != (
        "delayed_activation_photon"
    ):
        raise DelayedPhotonSourceError("delayed photon identity is invalid")
    energy = bundle.get("energy")
    if not isinstance(energy, Mapping):
        raise DelayedPhotonSourceError(
            "delayed photon energy contract is invalid"
        )
    edges = [float(value) for value in energy.get("energy_edges_eV", ())]
    if (
        len(edges) < 2
        or edges[0] != 0.0
        or edges != sorted(set(edges))
        or any(not math.isfinite(value) or value < 0.0 for value in edges)
        or energy
        != {
            "energy_edges_eV": edges,
            "group_count": len(edges) - 1,
            "group_order": "ascending_energy",
            "value_semantics": "bin_integrated_emission_rate_density",
        }
    ):
        raise DelayedPhotonSourceError(
            "delayed photon energy contract is invalid"
        )
    normalization = bundle.get("absolute_normalization")
    if not isinstance(normalization, Mapping) or (
        normalization.get("alara_source_value_unit") != "photons/cm3/s/bin"
        or normalization.get("zone_integrated_value_unit")
        != "photons/s/bin/zone"
        or normalization.get("zone_integrated_rate_operation")
        != "multiply_once_by_bound_zone_volume_cm3"
        or normalization.get("alara_output_density_is_absolute") is not True
        or normalization.get("bound_zone_volume_applied_exactly_once")
        is not True
        or normalization.get("additional_rate_multiplier_allowed") is not False
    ):
        raise DelayedPhotonSourceError(
            "absolute source normalization is invalid"
        )
    if bundle.get("source_assumptions") != {
        "space": "uniform_within_each_bound_activation_zone",
        "angle": "isotropic_4pi",
        "within_energy_bin": "uniform_in_energy",
        "time": "steady_snapshot_at_exact_cooling_offset",
        "polarization": "unavailable_unpolarized_approximation",
        "space_energy_angle_correlations": "unavailable_not_fabricated",
    }:
        raise DelayedPhotonSourceError(
            "delayed source assumptions are ambiguous"
        )
    provenance = bundle.get("artifact_provenance")
    if not isinstance(provenance, Mapping):
        raise DelayedPhotonSourceError("artifact provenance is missing")
    for key in (
        "alara_input_manifest_sha256",
        "alara_checkpoint_input_sha256",
        "alara_result_receipt_sha256",
        "delayed_photon_source_sha256",
    ):
        _digest(provenance.get(key), key)
    if provenance.get("alara_runtime") != qualified_alara_runtime():
        raise DelayedPhotonSourceError("ALARA runtime provenance is invalid")
    origin = bundle.get("provenance_origin")
    if (
        origin
        not in {
            "SYNTHETIC_WORKFLOW_FIXTURE",
            "PHYSICAL_OPENMC_STATEPOINT",
        }
        or provenance.get("openmc_provenance", {}).get("provenance_origin")
        != origin
    ):
        raise DelayedPhotonSourceError("delayed source provenance is invalid")
    if bundle.get("uncertainty") != {
        "status": "UNAVAILABLE_NOT_PROPAGATED",
        "zero_uncertainty_may_be_claimed": False,
    } or bundle.get("covariance") != {
        "status": "UNAVAILABLE_NOT_FABRICATED",
        "zero_covariance_may_be_claimed": False,
    }:
        raise DelayedPhotonSourceError("uncertainty status is unsafe")

    binding = bundle.get("activation_binding")
    snapshots = bundle.get("snapshots")
    if not isinstance(binding, Mapping) or not isinstance(snapshots, list):
        raise DelayedPhotonSourceError("activation snapshots are missing")
    zone_ids = list(binding.get("zone_ids", ()))
    offsets = list(binding.get("cooling_offset_s", ()))
    alias_policy = bundle.get("near_zero_time_alias_policy")
    if not isinstance(alias_policy, Mapping) or {
        key: alias_policy.get(key)
        for key in (
            "maximum_alias_offset_s",
            "spectrum_relative_tolerance",
            "spectrum_absolute_tolerance",
            "canonical_label",
        )
    } != {
        "maximum_alias_offset_s": NEAR_ZERO_ALIAS_MAX_S,
        "spectrum_relative_tolerance": NEAR_ZERO_ALIAS_SPECTRUM_REL_TOL,
        "spectrum_absolute_tolerance": NEAR_ZERO_ALIAS_SPECTRUM_ABS_TOL,
        "canonical_label": "shutdown",
    }:
        raise DelayedPhotonSourceError(
            "near-zero time alias policy is invalid"
        )
    aliases = alias_policy.get("accepted_aliases")
    if not isinstance(aliases, list):
        raise DelayedPhotonSourceError("near-zero time aliases are malformed")
    alias_zone_ids = []
    for alias in aliases:
        if not isinstance(alias, Mapping):
            raise DelayedPhotonSourceError("near-zero time alias is malformed")
        alias_zone_ids.append(alias.get("zone_id"))
        if (
            alias.get("zone_id") not in zone_ids
            or alias.get("canonical_cooling_offset_s") != 0
            or alias.get("canonical_label") != "shutdown"
            or not str(alias.get("alias_label", "")).strip()
            or not (
                0.0
                < float(alias.get("alias_offset_s", math.nan))
                <= NEAR_ZERO_ALIAS_MAX_S
            )
            or alias.get("spectra_match_status") != "PASS"
            or not math.isfinite(
                float(alias.get("maximum_absolute_difference", math.nan))
            )
            or float(alias.get("maximum_absolute_difference", math.nan)) < 0.0
            or not (
                0.0
                <= float(alias.get("maximum_relative_difference", math.nan))
                <= NEAR_ZERO_ALIAS_SPECTRUM_REL_TOL
            )
        ):
            raise DelayedPhotonSourceError("near-zero time alias is unsafe")
    if len(alias_zone_ids) != len(set(alias_zone_ids)):
        raise DelayedPhotonSourceError(
            "near-zero time alias zone is duplicated"
        )
    if len(snapshots) != len(offsets):
        raise DelayedPhotonSourceError(
            "activation snapshot count is incomplete"
        )
    seen_offsets = []
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            raise DelayedPhotonSourceError("activation snapshot is malformed")
        seen_offsets.append(int(snapshot.get("cooling_offset_s", -1)))
        zones = snapshot.get("zones")
        if (
            not isinstance(zones, list)
            or [row.get("zone_id") for row in zones] != zone_ids
        ):
            raise DelayedPhotonSourceError("snapshot zone order is invalid")
        recomputed_total = 0.0
        zone_probabilities = []
        for row in zones:
            rates = row.get("group_emission_rate_photons_per_s")
            density = row.get("group_emission_rate_density_photons_per_cm3_s")
            if (
                not isinstance(rates, list)
                or len(rates) != len(edges) - 1
                or not isinstance(density, list)
                or len(density) != len(rates)
            ):
                raise DelayedPhotonSourceError("zone spectrum is malformed")
            volume = float(row.get("volume_cm3", math.nan))
            if not math.isfinite(volume) or volume <= 0.0:
                raise DelayedPhotonSourceError("zone volume is invalid")
            if any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in [*rates, *density]
            ) or any(
                not math.isclose(
                    float(per_volume),
                    float(rate) / volume,
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                )
                for rate, per_volume in zip(rates, density)
            ):
                raise DelayedPhotonSourceError(
                    "zone source density is invalid"
                )
            zone_total = sum(float(value) for value in rates)
            if not math.isclose(
                zone_total,
                float(row.get("total_emission_rate_photons_per_s", math.nan)),
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ):
                raise DelayedPhotonSourceError(
                    "zone source total does not close"
                )
            probabilities = row.get("group_probability")
            if not isinstance(probabilities, list) or len(
                probabilities
            ) != len(rates):
                raise DelayedPhotonSourceError(
                    "zone group probabilities are malformed"
                )
            expected_probabilities = (
                [float(value) / zone_total for value in rates]
                if zone_total > 0.0
                else [0.0 for value in rates]
            )
            if any(
                not math.isclose(
                    float(actual), expected, rel_tol=1.0e-12, abs_tol=0.0
                )
                for actual, expected in zip(
                    probabilities, expected_probabilities
                )
            ):
                raise DelayedPhotonSourceError(
                    "zone group probabilities are invalid"
                )
            if row.get("uncertainty", {}).get("status") != (
                "UNAVAILABLE_NOT_PROPAGATED"
            ) or row.get("covariance", {}).get("status") != (
                "UNAVAILABLE_NOT_FABRICATED"
            ):
                raise DelayedPhotonSourceError("zone uncertainty is unsafe")
            recomputed_total += zone_total
            zone_probabilities.append(
                float(row.get("zone_probability", math.nan))
            )
        if not math.isclose(
            recomputed_total,
            float(snapshot.get("total_emission_rate_photons_per_s", math.nan)),
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise DelayedPhotonSourceError(
                "snapshot source total does not close"
            )
        expected_zone_probabilities = (
            [
                float(row["total_emission_rate_photons_per_s"])
                / recomputed_total
                for row in zones
            ]
            if recomputed_total > 0.0
            else [0.0 for row in zones]
        )
        if any(
            not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=0.0)
            for actual, expected in zip(
                zone_probabilities, expected_zone_probabilities
            )
        ):
            raise DelayedPhotonSourceError(
                "snapshot zone probabilities are invalid"
            )
    if seen_offsets != offsets:
        raise DelayedPhotonSourceError("snapshot cooling order is invalid")

    routes = bundle.get("solver_routes")
    if not isinstance(routes, Mapping) or set(routes) != set(SOLVERS):
        raise DelayedPhotonSourceError("solver routes are incomplete")
    for solver in ("openmc", "mcnp", "geant4"):
        if routes.get(solver) != _monte_carlo_route(solver):
            raise DelayedPhotonSourceError(f"unsafe {solver} source route")
    for solver in ("opensn", "radiant"):
        if routes.get(solver) != _deterministic_route(solver):
            raise DelayedPhotonSourceError(f"unsafe {solver} source route")
    if bundle.get("prompt_delayed_separation") != {
        "prompt_photons_included": False,
        "delayed_photons_included": True,
        "combination_requires_separate_normalization_ledger": True,
    }:
        raise DelayedPhotonSourceError("prompt and delayed photons were mixed")
    if bundle.get("production_transport_authorized") is not False:
        raise DelayedPhotonSourceError(
            "production transport is not authorized"
        )
    content_sha256 = bundle.get("handoff_content_sha256")
    if content_sha256 is not None:
        _digest(content_sha256, "delayed photon handoff content")
        unhashed = dict(bundle)
        del unhashed["handoff_content_sha256"]
        if content_sha256 != _canonical_sha256(unhashed):
            raise DelayedPhotonSourceError(
                "delayed photon handoff was modified"
            )


def write_delayed_photon_solver_package(
    directory: str | Path, bundle: Mapping[str, Any]
) -> list[Path]:
    """Write create-only canonical and per-solver JSON source contracts."""
    validate_delayed_photon_solver_handoff(bundle)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    written = []
    source_path = root / "delayed_photon_source.json"
    source_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written.append(source_path)
    for solver in SOLVERS:
        route_path = root / f"{solver}_delayed_photon_source_contract.json"
        route_payload = {
            "schema": "parastell.delayed_photon_solver_route/v1.0.0",
            "status": "READY_FOR_ADAPTER_EXECUTION",
            "solver": solver,
            "source_handoff_sha256": _file_sha256(source_path),
            "source_handoff_content_sha256": bundle["handoff_content_sha256"],
            "route": bundle["solver_routes"][solver],
            "production_transport_authorized": False,
        }
        route_path.write_text(
            json.dumps(
                route_payload, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(route_path)
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "status": "SOLVER_SOURCE_CONTRACTS_READY",
        "files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in written
        ],
        "production_transport_authorized": False,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written.append(manifest_path)
    return written
