"""Non-executable activation schedule contract for downstream consumers.

ParaStell records the requested full-power checkpoints and independent cooling
branches, but it does not execute depletion.  The physical source rate remains
a provenance pointer so it cannot be copied, rounded, or applied twice.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "parastell.activation_campaign_schedule/v1.0.0"
JULIAN_YEAR_S = 31_557_600
IRRADIATION_CHECKPOINTS_S = (
    86_400,
    604_800,
    JULIAN_YEAR_S,
    5 * JULIAN_YEAR_S,
    10 * JULIAN_YEAR_S,
)
COOLING_OFFSETS_S = (
    0,
    1,
    60,
    3_600,
    86_400,
    604_800,
    2_592_000,
    JULIAN_YEAR_S,
    5 * JULIAN_YEAR_S,
    10 * JULIAN_YEAR_S,
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_activation_campaign(
    *, campaign_id: str = "wistell-d-full-power-checkpoints-v1"
) -> dict[str, Any]:
    """Build the requested full-power/cooling branch schedule.

    Each irradiation checkpoint is a separate full-power exposure from the
    unirradiated material.  Cooling offsets are independent zero-source
    branches from that checkpoint; they are never serialized into one long
    cooling history.
    """
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": str(campaign_id).strip(),
        "owner": "DPA_workflow",
        "producer": "ParaStell",
        "year_definition": "Julian_year_365.25_days",
        "julian_year_s": JULIAN_YEAR_S,
        "irradiation_checkpoint_s": list(IRRADIATION_CHECKPOINTS_S),
        "cooling_offset_s": list(COOLING_OFFSETS_S),
        "irradiation_semantics": (
            "independent_full_power_exposure_from_unirradiated_material"
        ),
        "cooling_semantics": (
            "independent_zero_source_branch_from_each_end_of_irradiation"
        ),
        "full_power_source_rate_binding": {
            "json_pointer": "/provenance/physical_source_rate_per_s",
            "scope_pointer": "/provenance/source_rate_scope",
            "multiplier": 1.0,
            "copied_absolute_rate": False,
        },
        "prompt_source_during_cooling": False,
        "delayed_photon_source_separate_from_prompt": True,
        "production_activation_authorized": False,
    }
    validate_activation_campaign(payload)
    payload["schedule_sha256"] = _canonical_sha256(payload)
    return payload


def validate_activation_campaign(campaign: Mapping[str, Any]) -> None:
    if campaign.get("schema") != SCHEMA:
        raise ValueError("unsupported activation campaign schema")
    if not str(campaign.get("campaign_id", "")).strip():
        raise ValueError("activation campaign ID is required")
    if campaign.get("owner") != "DPA_workflow":
        raise ValueError("activation execution owner must be DPA_workflow")
    if campaign.get("producer") != "ParaStell":
        raise ValueError("activation producer identity is invalid")
    if (
        campaign.get("year_definition") != "Julian_year_365.25_days"
        or int(campaign.get("julian_year_s", -1)) != JULIAN_YEAR_S
    ):
        raise ValueError("activation year must be the Julian year")
    if tuple(campaign.get("irradiation_checkpoint_s", ())) != (
        IRRADIATION_CHECKPOINTS_S
    ):
        raise ValueError("irradiation checkpoints differ from the contract")
    if tuple(campaign.get("cooling_offset_s", ())) != COOLING_OFFSETS_S:
        raise ValueError("cooling offsets differ from the contract")
    if campaign.get("irradiation_semantics") != (
        "independent_full_power_exposure_from_unirradiated_material"
    ):
        raise ValueError("irradiation checkpoints must be independent")
    if campaign.get("cooling_semantics") != (
        "independent_zero_source_branch_from_each_end_of_irradiation"
    ):
        raise ValueError("cooling branches must be independent and source-off")
    binding = campaign.get("full_power_source_rate_binding")
    if binding != {
        "json_pointer": "/provenance/physical_source_rate_per_s",
        "scope_pointer": "/provenance/source_rate_scope",
        "multiplier": 1.0,
        "copied_absolute_rate": False,
    }:
        raise ValueError("activation source-rate binding is unsafe")
    if campaign.get("prompt_source_during_cooling") is not False:
        raise ValueError("prompt source must be off during cooling")
    if campaign.get("delayed_photon_source_separate_from_prompt") is not True:
        raise ValueError("prompt and delayed photon ledgers must be separate")
    if campaign.get("production_activation_authorized") is not False:
        raise ValueError("production activation is not authorized")
    for sequence_name in (
        "irradiation_checkpoint_s",
        "cooling_offset_s",
    ):
        values = campaign[sequence_name]
        if any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in values
        ):
            raise ValueError(f"{sequence_name} contains an invalid duration")


def write_activation_campaign(
    path: str | Path, campaign: Mapping[str, Any]
) -> Path:
    """Write one create-only, deterministic schedule contract."""
    validate_activation_campaign(campaign)
    expected_hash = campaign.get("schedule_sha256")
    payload = dict(campaign)
    payload.pop("schedule_sha256", None)
    if expected_hash != _canonical_sha256(payload):
        raise ValueError("activation schedule hash is invalid")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(campaign, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
