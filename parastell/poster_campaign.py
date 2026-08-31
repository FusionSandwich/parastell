"""Geometry-neutral poster campaign manifest contract.

The plan is intentionally declarative and non-executing.  It defines a staged
matrix for poster geometry variants (non-Cartesian): baseline, controlled breeder,
shield, spatial/thickness, and selected-location cases.  Variants in each stage
derive directly from baseline so no accidental Cartesian product can be introduced.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "parastell.poster_campaign_plan/v1.0.0"
STAGE_NAMES = (
    "baseline",
    "controlled_breeder",
    "shield",
    "spatial_thickness",
    "selected_location",
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _coerce_text(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _coerce_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    payload = dict(value)
    if not payload:
        raise ValueError(f"{label} cannot be empty")
    return payload


def _coerce_case_rows(
    values: Sequence[Mapping[str, Any]] | None, stage: str
) -> list[dict[str, Any]]:
    rows = list(values or ())
    for value in rows:
        if not isinstance(value, Mapping):
            raise ValueError(f"{stage} case entries must be mappings")
    return [dict(value) for value in rows]


def _nonnegative_int(value: Any, label: str) -> int:
    number = int(value)
    if isinstance(value, bool) or number != value or number < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _merge_case(
    base: Mapping[str, Any], overlay: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _merge_case(result[key], value)
        else:
            result[key] = value
    return result


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value[key], child))
        return out
    if isinstance(value, (list, tuple)):
        out = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            out.update(_flatten(item, child))
        return out
    return {prefix: value}


def _field_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    baseline_flat = _flatten(baseline)
    candidate_flat = _flatten(candidate)
    all_fields = set(baseline_flat) | set(candidate_flat)
    changed: list[str] = []
    held: list[str] = []
    sentinel = object()
    for field in sorted(all_fields):
        if baseline_flat.get(field, sentinel) == candidate_flat.get(
            field, sentinel
        ):
            held.append(field)
        else:
            changed.append(field)
    return changed, held


def _build_case(
    *,
    campaign_id: str,
    stage: str,
    case_suffix: str,
    source_case_id: str | None,
    case_index: int,
    payload: Mapping[str, Any],
    baseline_payload: Mapping[str, Any],
) -> dict[str, Any]:
    changed, held = _field_deltas(baseline_payload, payload)
    return {
        "case_id": f"{campaign_id}-{case_suffix}",
        "stage": stage,
        "case_index": case_index,
        "source_case_id": source_case_id,
        "payload": dict(payload),
        "changed_fields": sorted(changed),
        "held_fixed_fields": sorted(held),
    }


def build_poster_campaign_manifest(
    *,
    campaign_id: str = "wistell-d-poster-campaign-v1",
    producer: str = "ParaStell",
    device: str = "WISTELL-D",
    baseline_case: Mapping[str, Any],
    controlled_breeder_cases: Sequence[Mapping[str, Any]] | None = None,
    shield_cases: Sequence[Mapping[str, Any]] | None = None,
    spatial_thickness_cases: Sequence[Mapping[str, Any]] | None = None,
    selected_location_cases: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and hash a staged non-Cartesian poster manifest."""
    campaign_id = _coerce_text(campaign_id, "campaign_id")
    if _coerce_text(producer, "producer") != "ParaStell":
        raise ValueError("poster campaign producer must be ParaStell")
    device = _coerce_text(device, "device")
    if "private" in device.lower():
        raise ValueError(
            "device identifier is not allowed in an example contract"
        )

    baseline_payload = _coerce_mapping(baseline_case, "baseline_case")
    stage_rows = {
        "controlled_breeder": _coerce_case_rows(
            controlled_breeder_cases, "controlled_breeder"
        ),
        "shield": _coerce_case_rows(shield_cases, "shield"),
        "spatial_thickness": _coerce_case_rows(
            spatial_thickness_cases, "spatial_thickness"
        ),
        "selected_location": _coerce_case_rows(
            selected_location_cases, "selected_location"
        ),
    }

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "producer": "ParaStell",
        "device": device,
        "non_cartesian": True,
        "case_strategy": "staged_without_cartesian_product",
        "baseline_case_id": f"{campaign_id}-baseline",
        "producer_controls": {
            "run_mode": "poster",
            "claim": "geometry_neutral",
        },
        "stages": [],
        "cases": [],
    }

    cases = manifest["cases"]
    baseline_case_id = manifest["baseline_case_id"]
    cases.append(
        _build_case(
            campaign_id=campaign_id,
            stage="baseline",
            case_suffix="baseline",
            source_case_id=None,
            case_index=0,
            payload=baseline_payload,
            baseline_payload=baseline_payload,
        )
    )

    for stage_name in STAGE_NAMES[1:]:
        rows = stage_rows[stage_name]
        case_ids: list[str] = []
        for row_index, row in enumerate(rows):
            case_suffix = f"{stage_name}-{row_index:03d}"
            merged = _merge_case(baseline_payload, row)
            case = _build_case(
                campaign_id=campaign_id,
                stage=stage_name,
                case_suffix=case_suffix,
                source_case_id=baseline_case_id,
                case_index=len(cases),
                payload=merged,
                baseline_payload=baseline_payload,
            )
            cases.append(case)
            case_ids.append(case["case_id"])
        manifest["stages"].append(
            {
                "stage": stage_name,
                "case_count": len(rows),
                "case_ids": case_ids,
                "source_case_id": baseline_case_id,
            }
        )

    if not manifest["stages"]:
        manifest["stages"] = [
            {
                "stage": stage,
                "case_count": 0,
                "case_ids": [],
                "source_case_id": baseline_case_id,
            }
            for stage in STAGE_NAMES[1:]
        ]

    manifest["case_count"] = len(cases)
    manifest["manifest_sha256"] = _canonical_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    validate_poster_campaign_manifest(manifest)
    return manifest


def validate_poster_campaign_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported poster campaign schema")
    campaign_id = _coerce_text(manifest.get("campaign_id"), "campaign_id")
    if manifest.get("producer") != "ParaStell":
        raise ValueError("poster campaign producer identity is invalid")
    if manifest.get("non_cartesian") is not True:
        raise ValueError("poster campaign must be non-Cartesian")
    if manifest.get("case_strategy") != "staged_without_cartesian_product":
        raise ValueError(
            "poster campaign strategy is not staged non-Cartesian"
        )
    producer_controls = _coerce_mapping(
        manifest.get("producer_controls"), "producer_controls"
    )
    if (
        producer_controls.get("run_mode") != "poster"
        or producer_controls.get("claim") != "geometry_neutral"
    ):
        raise ValueError("poster campaign producer controls are invalid")

    stages = manifest.get("stages")
    if not isinstance(stages, list):
        raise ValueError("poster campaign stages are missing")
    if len(stages) != 4:
        raise ValueError("poster campaign must define all four stages")
    if {str(stage.get("stage")) for stage in stages} != set(STAGE_NAMES[1:]):
        raise ValueError("poster campaign must define all four request stages")
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ValueError("poster campaign stage entry is invalid")
        if stage.get("stage") not in STAGE_NAMES[1:]:
            raise ValueError("poster campaign contains unknown stage")
        stage_case_count = _nonnegative_int(
            stage.get("case_count"), "poster campaign stage case_count"
        )
        if len(stage.get("case_ids", ())) != stage_case_count:
            raise ValueError("poster campaign stage_count is invalid")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) < 1:
        raise ValueError("poster campaign must contain at least one case")

    baseline_case_id = _coerce_text(
        manifest.get("baseline_case_id"), "baseline_case_id"
    )
    if baseline_case_id != f"{campaign_id}-baseline":
        raise ValueError("poster campaign baseline_case_id is invalid")

    case_ids: set[str] = set()
    case_indices: set[int] = set()
    baseline_payload: Mapping[str, Any] | None = None
    baseline_position = None
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("poster campaign case is invalid")
        case_id = str(case.get("case_id", "")).strip()
        if not case_id or case_id in case_ids:
            raise ValueError(
                "poster campaign case IDs must be unique and nonempty"
            )
        case_ids.add(case_id)
        stage = str(case.get("stage", "")).strip()
        if stage not in STAGE_NAMES:
            raise ValueError("poster campaign contains unknown stage")

        payload = _coerce_mapping(
            case.get("payload"), f"case {case_id} payload"
        )
        case_index = _nonnegative_int(
            case.get("case_index"), "poster campaign case_index"
        )
        if case_index < 0:
            raise ValueError("poster campaign case_index is invalid")
        if case_index in case_indices:
            raise ValueError("poster campaign case indices must be unique")
        case_indices.add(case_index)
        changed = sorted(
            str(value) for value in case.get("changed_fields", ())
        )
        held = sorted(
            str(value) for value in case.get("held_fixed_fields", ())
        )
        if baseline_payload is None:
            if stage != "baseline":
                raise ValueError("poster campaign is missing a baseline case")
            observed_changed, observed_held = _field_deltas(payload, payload)
            baseline_payload = payload
            baseline_position = index
        else:
            observed_changed, observed_held = _field_deltas(
                baseline_payload, payload
            )
        if stage == "baseline":
            if case.get("source_case_id") is not None:
                raise ValueError(
                    "baseline case cannot inherit from another case"
                )
            if changed:
                raise ValueError("baseline case cannot have changed fields")
            if case.get("case_index") != 0:
                raise ValueError("baseline case index must be zero")
        else:
            if case.get("source_case_id") != baseline_case_id:
                raise ValueError("non-baseline cases must point to baseline")
        if set(changed) != set(observed_changed) or set(held) != set(
            observed_held
        ):
            raise ValueError(
                "poster campaign case diff metadata is inconsistent"
            )

    if len(cases) != _nonnegative_int(
        manifest.get("case_count"), "poster campaign case_count"
    ):
        raise ValueError("poster campaign case_count is incorrect")
    if baseline_payload is None:
        raise ValueError("poster campaign missing baseline case")
    if baseline_position != 0:
        raise ValueError("baseline case must be first entry")
    if baseline_case_id != cases[0]["case_id"]:
        raise ValueError("baseline case id must align with campaign id")
    if case_indices != set(range(len(cases))):
        raise ValueError(
            "poster campaign case indices must be contiguous from zero"
        )

    declared = {
        str(item) for stage in stages for item in stage.get("case_ids", ())
    }
    if any(case_id not in case_ids for case_id in declared):
        raise ValueError("poster campaign stage references unknown case id")

    provided_hash = str(manifest.get("manifest_sha256", ""))
    unsigned = {
        k: v for k, v in dict(manifest).items() if k != "manifest_sha256"
    }
    if provided_hash != _canonical_sha256(unsigned):
        raise ValueError("poster campaign manifest hash is invalid")


def write_poster_campaign_manifest(
    path: str | Path, manifest: Mapping[str, Any]
) -> Path:
    """Write one create-only poster campaign manifest."""
    validate_poster_campaign_manifest(manifest)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
