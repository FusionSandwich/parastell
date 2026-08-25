"""Neutral multi-seed statistical evidence for key unbiased responses."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

SCHEMA = "parastell.production_statistical_qualification/v1.0.0"
ALLOWED_THRESHOLDS = {
    "maximum_relative_standard_error",
    "minimum_effective_sample_size",
    "minimum_aggregate_raw_count",
}
EMPTY_STATUS = "EMPTY_NOT_A_PHYSICAL_ZERO"
UNDER_RESOLVED_STATUS = "UNDER_RESOLVED"
UNASSESSED_STATUS = "UNASSESSED_NO_EXPLICIT_THRESHOLDS"
UNSUPPORTED_STATUS = "UNASSESSED_UNSUPPORTED_METRIC"
INCOMPLETE_STATUS = "UNASSESSED_OR_INCOMPLETE"
QUALIFIED_STATUS = "QUALIFIED_EXPLICIT_THRESHOLDS"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("campaign seed must be an integer")
    seed = int(value)
    if seed <= 0:
        raise ValueError("campaign seeds must be positive")
    return seed


def _definition_hash(row: Mapping[str, Any]) -> str:
    explicit = row.get("response_definition_sha256")
    legacy = row.get("definition_hash")
    if explicit is not None and legacy is not None and explicit != legacy:
        raise ValueError("response definition hash aliases disagree")
    value = explicit if explicit is not None else legacy
    if not _is_sha256(value):
        raise ValueError("response definition hash must be a SHA-256 digest")
    return str(value)


def _optional_sha256(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return str(value)


def _normalize_ess(row: Mapping[str, Any]) -> tuple[float | None, str]:
    value = row.get("effective_sample_size")
    source_status = row.get("effective_sample_size_status")
    status = "" if source_status is None else str(source_status)
    if value is None:
        if not status:
            status = "UNAVAILABLE_NOT_REPORTED"
        if status.startswith("AVAILABLE"):
            raise ValueError("ESS status claims availability without an ESS")
        return None, status
    result = _finite_float(value, "effective sample size")
    if result < 0.0:
        raise ValueError("effective sample size cannot be negative")
    if status and not status.startswith("AVAILABLE"):
        raise ValueError("reported ESS has an unavailable status")
    return result, status or "AVAILABLE_REPORTED"


def _normalize_raw_count(row: Mapping[str, Any]) -> int | None:
    value = row.get("raw_count")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("raw count must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("raw count cannot be negative")
    return result


def _normalize_source_reports(
    reports: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized = []
    for source in reports or ():
        if not isinstance(source, Mapping):
            raise TypeError("source report metadata must be a mapping")
        path = str(source.get("path", ""))
        if not path:
            raise ValueError("source report path must be nonempty")
        sha256 = source.get("sha256")
        if not _is_sha256(sha256):
            raise ValueError("source report hash must be a SHA-256 digest")
        size_bytes = source.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(
            size_bytes, Integral
        ):
            raise TypeError("source report size must be an integer")
        if int(size_bytes) < 0:
            raise ValueError("source report size cannot be negative")
        row_count = source.get("row_count")
        if isinstance(row_count, bool) or not isinstance(row_count, Integral):
            raise TypeError("source report row count must be an integer")
        if int(row_count) <= 0:
            raise ValueError("source report row count must be positive")
        normalized.append(
            {
                "path": path,
                "sha256": str(sha256),
                "size_bytes": int(size_bytes),
                "seed": _positive_seed(source.get("seed")),
                "row_count": int(row_count),
                "run_artifact_sha256": _optional_sha256(
                    source.get("run_artifact_sha256"),
                    "source report run artifact hash",
                ),
            }
        )
    normalized.sort(key=lambda item: (item["seed"], item["path"]))
    if len({item["seed"] for item in normalized}) != len(normalized):
        raise ValueError("source reports must have distinct seeds")
    if len({item["sha256"] for item in normalized}) != len(normalized):
        raise ValueError("source reports must have distinct file hashes")
    return normalized


def _normalize_row(source: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(source)
    if row.get("variant") != "unbiased":
        raise ValueError("production statistics accept only unbiased rows")
    if row.get("is_key_response") is not True:
        raise ValueError(
            "production statistic rows must be marked as key responses"
        )
    response_id = str(row.get("response_id", ""))
    if not response_id:
        raise ValueError("response ID must be nonempty")
    run_contract = row.get("run_contract_sha256")
    if not _is_sha256(run_contract):
        raise ValueError("run contract must be a SHA-256 digest")
    metadata = {}
    for name in ("particle", "quantity", "units", "normalization"):
        value = str(row.get(name, ""))
        if not value:
            raise ValueError(f"response {name} must be nonempty")
        metadata[name] = value
    estimate = _finite_float(row.get("estimate"), "response estimate")
    within = _finite_float(
        row.get("within_run_std_dev"), "within-run standard deviation"
    )
    if within < 0.0:
        raise ValueError("within-run standard deviation cannot be negative")
    ess, ess_status = _normalize_ess(row)
    return {
        "variant": "unbiased",
        "is_key_response": True,
        "response_id": response_id,
        "response_definition_sha256": _definition_hash(row),
        "run_contract_sha256": str(run_contract),
        "seed": _positive_seed(row.get("seed")),
        **metadata,
        "estimate": estimate,
        "within_run_std_dev": within,
        "effective_sample_size": ess,
        "effective_sample_size_status": ess_status,
        "raw_count": _normalize_raw_count(row),
        "run_artifact_sha256": _optional_sha256(
            row.get("run_artifact_sha256"), "run artifact hash"
        ),
    }


def _normalize_thresholds(
    thresholds: Mapping[str, Mapping[str, Any]] | None,
    response_ids: set[str],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for response_id, source in dict(thresholds or {}).items():
        if response_id not in response_ids:
            raise ValueError(
                f"qualification threshold targets unknown response {response_id!r}"
            )
        if not isinstance(source, Mapping) or not source:
            raise ValueError(
                f"qualification thresholds for {response_id!r} must be nonempty"
            )
        if unknown := set(source) - ALLOWED_THRESHOLDS:
            raise ValueError(
                f"unknown qualification thresholds for {response_id!r}: "
                f"{sorted(unknown)}"
            )
        normalized: dict[str, float | int] = {}
        if "maximum_relative_standard_error" in source:
            value = _finite_float(
                source["maximum_relative_standard_error"],
                "maximum relative standard error",
            )
            if value <= 0.0:
                raise ValueError(
                    "maximum relative standard error must be positive"
                )
            normalized["maximum_relative_standard_error"] = value
        if "minimum_effective_sample_size" in source:
            value = _finite_float(
                source["minimum_effective_sample_size"],
                "minimum effective sample size",
            )
            if value < 0.0:
                raise ValueError(
                    "minimum effective sample size cannot be negative"
                )
            normalized["minimum_effective_sample_size"] = value
        if "minimum_aggregate_raw_count" in source:
            value = source["minimum_aggregate_raw_count"]
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    "minimum aggregate raw count must be an integer"
                )
            if int(value) < 0:
                raise ValueError(
                    "minimum aggregate raw count cannot be negative"
                )
            normalized["minimum_aggregate_raw_count"] = int(value)
        result[str(response_id)] = normalized
    return result


def _normalize_unassessed_metrics(
    values: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    if values is None:
        return []
    if not isinstance(values, Mapping):
        raise TypeError("declared unassessed metrics must be a mapping")
    result = []
    for metric_id, raw_reason in values.items():
        metric_id = str(metric_id)
        reason = str(raw_reason)
        if not metric_id or not reason:
            raise ValueError(
                "declared unassessed metric IDs and reasons must be nonempty"
            )
        result.append(
            {
                "metric_id": metric_id,
                "resolution_status": UNSUPPORTED_STATUS,
                "reason": reason,
            }
        )
    return sorted(result, key=lambda item: item["metric_id"])


def _availability_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_name: str,
    status_name: str | None = None,
) -> dict[str, Any]:
    values = [row[value_name] for row in rows]
    available = [value is not None for value in values]
    statuses = (
        [str(row[status_name]) for row in rows]
        if status_name is not None
        else [
            (
                "AVAILABLE_REPORTED"
                if value is not None
                else "UNAVAILABLE_NOT_REPORTED"
            )
            for value in values
        ]
    )
    available_count = sum(available)
    if available_count == len(rows):
        status = "AVAILABLE_ALL_SEEDS"
        aggregate = sum(values)
    elif available_count == 0:
        status = "UNAVAILABLE_ALL_SEEDS"
        aggregate = None
    else:
        status = "PARTIALLY_AVAILABLE_NOT_AGGREGATED"
        aggregate = None
    return {
        "status": status,
        "value": aggregate,
        "available_seed_count": available_count,
        "seed_count": len(rows),
        "seed_statuses": statuses,
    }


def _threshold_evaluation(
    response: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    if response["empty_estimate"]:
        return EMPTY_STATUS, []
    if not thresholds:
        return UNASSESSED_STATUS, []
    checks = []
    if "maximum_relative_standard_error" in thresholds:
        observed = response["relative_standard_error"]
        threshold = thresholds["maximum_relative_standard_error"]
        passed = observed is not None and observed <= threshold
        checks.append(
            {
                "metric": "relative_standard_error",
                "observed": observed,
                "operator": "<=",
                "threshold": threshold,
                "passes": passed,
            }
        )
    if "minimum_effective_sample_size" in thresholds:
        observed = response["effective_sample_size"]["value"]
        threshold = thresholds["minimum_effective_sample_size"]
        passed = observed is not None and observed >= threshold
        checks.append(
            {
                "metric": "effective_sample_size",
                "observed": observed,
                "availability_status": response["effective_sample_size"][
                    "status"
                ],
                "operator": ">=",
                "threshold": threshold,
                "passes": passed,
            }
        )
    if "minimum_aggregate_raw_count" in thresholds:
        observed = response["raw_count"]["value"]
        threshold = thresholds["minimum_aggregate_raw_count"]
        passed = observed is not None and observed >= threshold
        checks.append(
            {
                "metric": "aggregate_raw_count",
                "observed": observed,
                "availability_status": response["raw_count"]["status"],
                "operator": ">=",
                "threshold": threshold,
                "passes": passed,
            }
        )
    return (
        (
            QUALIFIED_STATUS
            if all(check["passes"] for check in checks)
            else UNDER_RESOLVED_STATUS
        ),
        checks,
    )


def _aggregate_response(
    response_id: str,
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda item: int(item["seed"]))
    definitions = {row["response_definition_sha256"] for row in rows}
    identities = {
        (
            row["particle"],
            row["quantity"],
            row["units"],
            row["normalization"],
        )
        for row in rows
    }
    if len(definitions) != 1 or len(identities) != 1:
        raise ValueError(
            f"response {response_id!r} changed definition or identity"
        )
    estimates = [float(row["estimate"]) for row in rows]
    within = [float(row["within_run_std_dev"]) for row in rows]
    seed_count = len(rows)
    mean = sum(estimates) / seed_count
    squared_deviations = sum((value - mean) ** 2 for value in estimates)
    observed_sem = math.sqrt(
        squared_deviations / (seed_count - 1)
    ) / math.sqrt(seed_count)
    propagated_sem = (
        math.sqrt(sum(value * value for value in within)) / seed_count
    )
    conservative_sem = max(observed_sem, propagated_sem)
    relative = conservative_sem / abs(mean) if mean != 0.0 else None
    ess = _availability_summary(
        rows,
        value_name="effective_sample_size",
        status_name="effective_sample_size_status",
    )
    raw_count = _availability_summary(rows, value_name="raw_count")
    empty = bool(
        all(value == 0.0 for value in estimates)
        and all(value == 0.0 for value in within)
        and not any(
            row["raw_count"] is not None and row["raw_count"] > 0
            for row in rows
        )
        and not any(
            row["effective_sample_size"] is not None
            and row["effective_sample_size"] > 0.0
            for row in rows
        )
    )
    particle, quantity, units, normalization = identities.pop()
    response = {
        "response_id": response_id,
        "response_definition_sha256": definitions.pop(),
        "particle": particle,
        "quantity": quantity,
        "units": units,
        "normalization": normalization,
        "seed_count": seed_count,
        "seeds": sorted(int(row["seed"]) for row in rows),
        "mean": mean,
        "conservative_standard_error": conservative_sem,
        "observed_between_seed_standard_error": observed_sem,
        "propagated_within_run_standard_error": propagated_sem,
        "uncertainty_combination": (
            "maximum_of_between_seed_and_propagated_within_run_standard_error"
        ),
        "relative_standard_error": relative,
        "effective_sample_size": ess,
        "raw_count": raw_count,
        "empty_estimate": empty,
        "physical_zero_claimed": False,
        "seed_evidence": [
            {
                "seed": int(row["seed"]),
                "estimate": float(row["estimate"]),
                "within_run_std_dev": float(row["within_run_std_dev"]),
                "effective_sample_size": row["effective_sample_size"],
                "effective_sample_size_status": row[
                    "effective_sample_size_status"
                ],
                "raw_count": row["raw_count"],
                "run_artifact_sha256": row["run_artifact_sha256"],
            }
            for row in sorted(rows, key=lambda item: int(item["seed"]))
        ],
    }
    status, checks = _threshold_evaluation(response, thresholds)
    response["resolution_status"] = status
    response["explicit_thresholds"] = dict(thresholds)
    response["threshold_checks"] = checks
    return response


def _stable_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": value["schema"],
        "minimum_seed_count": value["minimum_seed_count"],
        "run_contract_sha256": value["run_contract_sha256"],
        "seeds": value["seeds"],
        "source_reports": value["source_reports"],
        "qualification_thresholds": value["qualification_thresholds"],
        "responses": value["responses"],
        "unassessed_metrics": value.get("unassessed_metrics", []),
        "overall_resolution_status": value["overall_resolution_status"],
        "status_counts": value["status_counts"],
        "scientific_claim_scope": value["scientific_claim_scope"],
        "effective_sample_size_semantics": value[
            "effective_sample_size_semantics"
        ],
    }


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _stable_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def aggregate_production_response_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    qualification_thresholds: Mapping[str, Mapping[str, Any]] | None = None,
    minimum_seed_count: int = 3,
    source_reports: Sequence[Mapping[str, Any]] | None = None,
    declared_unassessed_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate key unbiased responses while retaining per-seed evidence."""

    if isinstance(minimum_seed_count, bool) or not isinstance(
        minimum_seed_count, Integral
    ):
        raise TypeError("minimum seed count must be an integer")
    minimum_seed_count = int(minimum_seed_count)
    if minimum_seed_count < 3:
        raise ValueError(
            "statistical qualification requires at least three seeds"
        )
    normalized = [_normalize_row(row) for row in rows]
    if not normalized:
        raise ValueError("statistical qualification requires response rows")
    run_contracts = {row["run_contract_sha256"] for row in normalized}
    if len(run_contracts) != 1:
        raise ValueError("production response rows changed run contract")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        grouped.setdefault(row["response_id"], []).append(row)
    seed_inventories = {}
    for response_id, group in grouped.items():
        seeds = tuple(sorted(row["seed"] for row in group))
        if len(seeds) < minimum_seed_count or len(set(seeds)) != len(seeds):
            raise ValueError(
                f"response {response_id!r} lacks distinct qualification seeds"
            )
        seed_inventories[response_id] = seeds
    if len(set(seed_inventories.values())) != 1:
        raise ValueError("key responses do not share one seed inventory")
    campaign_seeds = tuple(next(iter(seed_inventories.values())))
    normalized_reports = _normalize_source_reports(source_reports)
    if normalized_reports:
        report_seeds = tuple(item["seed"] for item in normalized_reports)
        if report_seeds != campaign_seeds:
            raise ValueError(
                "source report seeds do not match the response seed inventory"
            )
    run_artifacts_by_seed: dict[int, set[str | None]] = {}
    for row in normalized:
        run_artifacts_by_seed.setdefault(row["seed"], set()).add(
            row["run_artifact_sha256"]
        )
    if any(len(values) != 1 for values in run_artifacts_by_seed.values()):
        raise ValueError("responses from one seed changed run artifact hash")
    artifacts = [
        next(iter(values)) for values in run_artifacts_by_seed.values()
    ]
    reported_artifacts = [value for value in artifacts if value is not None]
    if len(reported_artifacts) != len(set(reported_artifacts)):
        raise ValueError("distinct seeds reuse a run artifact hash")
    definitions = []
    for response_id, group in grouped.items():
        values = {row["response_definition_sha256"] for row in group}
        if len(values) != 1:
            raise ValueError(
                f"response {response_id!r} changed definition hash"
            )
        definitions.append(values.pop())
    if len(set(definitions)) != len(definitions):
        raise ValueError("key responses reuse a response definition hash")
    thresholds = _normalize_thresholds(qualification_thresholds, set(grouped))
    responses = [
        _aggregate_response(
            response_id, grouped[response_id], thresholds.get(response_id, {})
        )
        for response_id in sorted(grouped)
    ]
    unassessed_metrics = _normalize_unassessed_metrics(
        declared_unassessed_metrics
    )
    statuses = Counter(response["resolution_status"] for response in responses)
    if unassessed_metrics:
        statuses[UNSUPPORTED_STATUS] += len(unassessed_metrics)
    if any(
        response["resolution_status"] in {EMPTY_STATUS, UNDER_RESOLVED_STATUS}
        for response in responses
    ):
        overall = "UNDER_RESOLVED_OR_EMPTY"
    elif unassessed_metrics:
        overall = INCOMPLETE_STATUS
    elif any(
        response["resolution_status"] == UNASSESSED_STATUS
        for response in responses
    ):
        overall = UNASSESSED_STATUS
    else:
        overall = QUALIFIED_STATUS
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "minimum_seed_count": minimum_seed_count,
        "run_contract_sha256": run_contracts.pop(),
        "seeds": list(campaign_seeds),
        "source_reports": normalized_reports,
        "qualification_thresholds": thresholds,
        "responses": responses,
        "unassessed_metrics": unassessed_metrics,
        "overall_resolution_status": overall,
        "status_counts": dict(sorted(statuses.items())),
        "scientific_claim_scope": (
            "Statuses evaluate only caller-supplied thresholds; this artifact "
            "does not independently establish transport convergence or a "
            "physical zero. Declared unsupported metrics remain explicitly "
            "unassessed and prevent an overall qualified status."
        ),
        "effective_sample_size_semantics": (
            "ESS is summed only when every seed reports it; partial or absent "
            "ESS remains unavailable."
        ),
    }
    result["evidence_sha256"] = _payload_sha256(result)
    return result


def write_production_statistical_qualification(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    qualification_thresholds: Mapping[str, Mapping[str, Any]] | None = None,
    minimum_seed_count: int = 3,
    source_reports: Sequence[Mapping[str, Any]] | None = None,
    declared_unassessed_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a dependency-light JSON qualification artifact."""

    result = aggregate_production_response_statistics(
        rows,
        qualification_thresholds=qualification_thresholds,
        minimum_seed_count=minimum_seed_count,
        source_reports=source_reports,
        declared_unassessed_metrics=declared_unassessed_metrics,
    )
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result["path"] = str(output)
    return result


def _rows_from_artifact(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for response in value["responses"]:
        for seed in response["seed_evidence"]:
            rows.append(
                {
                    "variant": "unbiased",
                    "is_key_response": True,
                    "response_id": response["response_id"],
                    "response_definition_sha256": response[
                        "response_definition_sha256"
                    ],
                    "run_contract_sha256": value["run_contract_sha256"],
                    "seed": seed["seed"],
                    "particle": response["particle"],
                    "quantity": response["quantity"],
                    "units": response["units"],
                    "normalization": response["normalization"],
                    "estimate": seed["estimate"],
                    "within_run_std_dev": seed["within_run_std_dev"],
                    "effective_sample_size": seed["effective_sample_size"],
                    "effective_sample_size_status": seed[
                        "effective_sample_size_status"
                    ],
                    "raw_count": seed["raw_count"],
                    "run_artifact_sha256": seed["run_artifact_sha256"],
                }
            )
    return rows


def validate_production_statistical_qualification(
    path: str | Path,
    *,
    verify_source_reports: bool = False,
) -> dict[str, Any]:
    """Validate hashes and recompute aggregates without transport dependencies."""

    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported production statistical schema")
    if value.get("evidence_sha256") != _payload_sha256(value):
        raise ValueError("production statistical evidence hash is invalid")
    recalculated = aggregate_production_response_statistics(
        _rows_from_artifact(value),
        qualification_thresholds=value["qualification_thresholds"],
        minimum_seed_count=value["minimum_seed_count"],
        source_reports=value["source_reports"],
        declared_unassessed_metrics={
            metric["metric_id"]: metric["reason"]
            for metric in value.get("unassessed_metrics", [])
        },
    )
    if _stable_payload(recalculated) != _stable_payload(value):
        raise ValueError("production statistical aggregates are inconsistent")
    if verify_source_reports:
        source_rows = []
        for report in value["source_reports"]:
            report_path = Path(report["path"]).resolve()
            if not report_path.is_file():
                raise FileNotFoundError(report_path)
            if report_path.stat().st_size != report["size_bytes"]:
                raise ValueError("production statistical source size changed")
            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            if digest != report["sha256"]:
                raise ValueError("production statistical source hash changed")
            report_value = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(report_value, Mapping):
                report_rows = report_value.get(
                    "responses", report_value.get("rows")
                )
            elif isinstance(report_value, list):
                report_rows = report_value
            else:
                raise TypeError(
                    "production statistical source report must be a mapping or list"
                )
            if not isinstance(report_rows, list) or not report_rows:
                raise ValueError(
                    "production statistical source report has no response rows"
                )
            if len(report_rows) != report["row_count"]:
                raise ValueError(
                    "production statistical source row count changed"
                )
            if any(not isinstance(row, Mapping) for row in report_rows):
                raise TypeError(
                    "production statistical source rows must be mappings"
                )
            if isinstance(report_value, Mapping) and report_value.get(
                "seed"
            ) not in {None, report["seed"]}:
                raise ValueError(
                    "production statistical source seed binding changed"
                )
            if any(row.get("seed") != report["seed"] for row in report_rows):
                raise ValueError(
                    "production statistical source row seed binding changed"
                )
            if isinstance(report_value, Mapping) and report_value.get(
                "run_artifact_sha256"
            ) not in {None, report["run_artifact_sha256"]}:
                raise ValueError(
                    "production statistical source run artifact changed"
                )
            if any(
                row.get("run_artifact_sha256") != report["run_artifact_sha256"]
                for row in report_rows
            ):
                raise ValueError(
                    "production statistical source row artifact binding changed"
                )
            source_rows.extend(dict(row) for row in report_rows)
        source_recalculated = aggregate_production_response_statistics(
            source_rows,
            qualification_thresholds=value["qualification_thresholds"],
            minimum_seed_count=value["minimum_seed_count"],
            source_reports=value["source_reports"],
            declared_unassessed_metrics={
                metric["metric_id"]: metric["reason"]
                for metric in value.get("unassessed_metrics", [])
            },
        )
        if _stable_payload(source_recalculated) != _stable_payload(value):
            raise ValueError(
                "production statistical source rows disagree with the qualification"
            )
    return {
        "schema": SCHEMA,
        "evidence_sha256": value["evidence_sha256"],
        "response_count": len(value["responses"]),
        "seed_count": len(value["seeds"]),
        "overall_resolution_status": value["overall_resolution_status"],
        "status_counts": value["status_counts"],
        "source_report_count": len(value["source_reports"]),
        "unassessed_metric_count": len(value.get("unassessed_metrics", [])),
    }
