"""Staged P00--P07 material cases for WISTELL-D poster transport.

This module prepares case artifacts only.  It does not run OpenMC, authorize a
large campaign, alter the accepted ParaStell geometry, or create an implicit
Cartesian product.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .blanket_materials import build_blanket_material_bundle


SCHEMA = "parastell.poster_parametric_case_plan/v1.0.0"

CASE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "P00",
        "family": "baseline",
        "concept": "LiPb/DCLL-like breeder with WC baseline shield",
        "priority": "MANDATORY",
        "preset": "dcll",
        "overrides": {},
        "status": "READY_FOR_BOUNDED_SMOKE",
    },
    {
        "case_id": "P01",
        "family": "breeder",
        "concept": "HCPB",
        "priority": "MANDATORY",
        "preset": "hcpb",
        "overrides": {},
        "status": "READY_FOR_BOUNDED_SMOKE",
    },
    {
        "case_id": "P02",
        "family": "breeder",
        "concept": "FLiBe-LIB with 60-percent Li-6 FLiBe",
        "priority": "MANDATORY",
        "preset": "flibe_lib",
        "overrides": {},
        "status": "READY_FOR_BOUNDED_SMOKE",
    },
    {
        "case_id": "P03",
        "family": "breeder",
        "concept": "HCLL",
        "priority": "CONDITIONAL",
        "preset": "hcll",
        "overrides": {},
        "status": "CONDITIONAL_READY_FOR_BOUNDED_SMOKE",
    },
    {
        "case_id": "P04",
        "family": "breeder",
        "concept": "SCLV",
        "priority": "CONDITIONAL_ON_COMPLETE_DATA_COVERAGE",
        "preset": None,
        "overrides": {},
        "status": "BLOCKED_INCOMPLETE_PORTABLE_MATERIAL_DATA",
        "blockers": [
            "approved CaO constituent is absent from the bound fusion-material-db",
            "the historical 6.5-percent Li-6 liquid-lithium primitive is not a bound database material",
        ],
    },
    {
        "case_id": "P05",
        "family": "shield",
        "concept": "natural-boron W2B5",
        "priority": "POSTER_SHIELD_CONTRAST",
        "preset": "dcll",
        "overrides": {"high_temperature_shield": "natural_boron_w2b5"},
        "status": "READY_FOR_BOUNDED_SMOKE",
    },
    {
        "case_id": "P06",
        "family": "shield",
        "concept": "Type One optimized HTS/LTS",
        "priority": "POSTER_REFERENCE_CONTINUITY",
        "preset": "dcll",
        "overrides": {
            "high_temperature_shield": "type_one_w2b5_rafs_helium_hts",
            "low_temperature_shield": "type_one_rafs_borated_rafs_water_lts",
        },
        "status": "READY_FOR_BOUNDED_SMOKE",
    },
    {
        "case_id": "P07",
        "family": "shield",
        "concept": "TiH2 then W2B5 layered shield",
        "priority": "POSTER_LAYER_ORDER_CONTRAST",
        "preset": None,
        "overrides": {},
        "status": "BLOCKED_INCOMPLETE_PORTABLE_MATERIAL_DATA",
        "blockers": [
            "the reviewed source provides only a TiH2 alias, not an approved density and isotope recipe",
            "the intended mapping onto the separated HTS and LTS ParaStell volumes must be declared",
        ],
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _optional_binding(path: str | Path | None, role: str) -> dict[str, Any]:
    if path is None:
        return {"role": role, "status": "NOT_BOUND"}
    resolved = Path(path).resolve(strict=True)
    return {
        "role": role,
        "status": "HASH_BOUND",
        "path": str(resolved),
        "sha256": _sha256(resolved),
    }


def build_poster_parametric_case_plan(
    pure_materials_path: str | Path,
    *,
    geometry_path: str | Path | None = None,
    source_path: str | Path | None = None,
    temperature_K: float = 293.6,
) -> dict[str, Any]:
    """Resolve every runnable P00--P07 case against one material database."""
    cases: list[dict[str, Any]] = []
    for definition in CASE_DEFINITIONS:
        case = dict(definition)
        case["source_case_id"] = None if case["case_id"] == "P00" else "P00"
        case["geometry_changed"] = False
        case["implicit_cartesian_product"] = False
        if case["preset"] is not None:
            bundle = build_blanket_material_bundle(
                pure_materials_path,
                preset=case["preset"],
                role_recipe_overrides=case["overrides"],
                temperature_K=temperature_K,
            )
            case["material_bundle"] = bundle
            case["material_bundle_sha256"] = bundle["bundle_sha256"]
        cases.append(case)

    bindings = {
        "geometry": _optional_binding(geometry_path, "accepted_90_degree_H5M"),
        "source": _optional_binding(source_path, "accepted_physical_source"),
    }
    required_ready = all(
        row["status"] == "READY_FOR_BOUNDED_SMOKE"
        for row in cases
        if row["priority"] == "MANDATORY"
    )
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": "wistell-d-poster-parametric-p00-p07-v1",
        "producer": "ParaStell",
        "device": "WISTELL-D",
        "modeled_extent_degrees": 90.0,
        "case_strategy": "staged_without_cartesian_product",
        "baseline_case_id": "P00",
        "cases": cases,
        "case_count": len(cases),
        "mandatory_cases_ready": required_ready,
        "bindings": bindings,
        "transport_controls": {
            "openmc_version": "0.16.0",
            "transport_mode": "analog_baseline",
            "weight_windows_enabled": False,
            "photon_transport": True,
            "global_magnets": "homogenized",
            "surface_phase_space_export": "required",
            "regular_statepoints": True,
            "checkpoint_strategy": "restartable_bounded_segments",
            "target_segment_minutes": 60,
            "maximum_segment_minutes": 120,
        },
        "execution_controls": {
            "large_campaign_authorized": False,
            "production_statistics_authorized": False,
            "run_command_created": False,
            "scheduler_submission_created": False,
        },
        "downstream_contracts": {
            "activation_schedule": [
                "1 day",
                "1 week",
                "1 full-power year",
                "5 full-power years",
                "10 full-power years",
            ],
            "required_outputs": [
                "statepoints",
                "surface_phase_space_banks",
                "neutron_and_photon_spectra",
                "heating",
                "damage_energy",
                "gas_production",
                "photon_production",
                "isotope_MT_reaction_rates",
                "activation_spectra",
            ],
        },
    }
    if all(row["status"] == "HASH_BOUND" for row in bindings.values()):
        plan["status"] = "READY_FOR_BOUNDED_SMOKE_TRANSPORT_NOT_AUTHORIZED"
    else:
        plan["status"] = "READY_FOR_CASE_ARTIFACT_GENERATION_INPUTS_NOT_BOUND"
    plan["plan_sha256"] = _canonical_sha(plan)
    validate_poster_parametric_case_plan(plan)
    return plan


def validate_poster_parametric_case_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != SCHEMA or plan.get("producer") != "ParaStell":
        raise ValueError("unsupported poster parametric plan")
    if plan.get("modeled_extent_degrees") != 90.0:
        raise ValueError("poster campaign must use the direct 90-degree model")
    if plan.get("case_strategy") != "staged_without_cartesian_product":
        raise ValueError("implicit Cartesian campaigns are forbidden")
    cases = plan.get("cases")
    if not isinstance(cases, list) or [
        row.get("case_id") for row in cases
    ] != [f"P{index:02d}" for index in range(8)]:
        raise ValueError("P00--P07 case inventory is incomplete or unordered")
    for case in cases:
        if (
            case.get("case_id") != "P00"
            and case.get("source_case_id") != "P00"
        ):
            raise ValueError("every variation must derive directly from P00")
        if case.get("geometry_changed") is not False:
            raise ValueError("material cases cannot change accepted geometry")
        if case.get("preset") is None and "material_bundle" in case:
            raise ValueError("blocked cases cannot contain a material bundle")
    mandatory = [row for row in cases if row.get("priority") == "MANDATORY"]
    if len(mandatory) != 3 or not all(
        row.get("material_bundle_sha256") for row in mandatory
    ):
        raise ValueError("mandatory material cases are unresolved")
    controls = plan.get("transport_controls", {})
    if (
        controls.get("transport_mode") != "analog_baseline"
        or controls.get("weight_windows_enabled") is not False
        or controls.get("regular_statepoints") is not True
    ):
        raise ValueError("poster transport controls violate the baseline gate")
    execution = plan.get("execution_controls", {})
    if execution.get("large_campaign_authorized") is not False:
        raise ValueError("case preparation cannot authorize a large run")
    expected = _canonical_sha(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan.get("plan_sha256") != expected:
        raise ValueError("poster parametric plan hash is invalid")
