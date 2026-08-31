"""Checkpoint/restart contract for long fixed-source OpenMC campaigns."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "parastell.openmc_checkpoint_restart_plan/v1.0.0"
OPENMC_SCHEMA_PREFIX = "parastell.parametric_openmc16_model/"
OPENMC_VERSION = "0.16.0"
SUPPORTED_SIGNALS = {"SIGTERM", "SIGINT", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGHUP"}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _positive_int(value: Any, label: str, minimum: int = 1) -> int:
    number = int(value)
    if isinstance(value, bool) or number < minimum:
        raise ValueError(f"{label} must be integer >= {minimum}")
    return number


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _segment_lengths(total_batches: int, segment_batches: int) -> list[int]:
    if segment_batches > total_batches:
        raise ValueError("segment_batches may not exceed total_batches")
    remaining = total_batches
    rows: list[int] = []
    while remaining > 0:
        row = min(segment_batches, remaining)
        rows.append(row)
        remaining -= row
    return rows


def _statepoint_schedule(batches: int, interval: int | None) -> list[int]:
    if interval is None:
        return [batches]
    if interval <= 0:
        raise ValueError("statepoint interval must be positive")
    if interval > batches:
        raise ValueError("statepoint interval exceeds segment length")
    batch_rows = list(range(interval, batches + 1, interval))
    if batch_rows[-1] != batches:
        batch_rows.append(batches)
    return batch_rows


def _validate_signal(value: Any) -> str:
    signal = _text(value, "signal").upper()
    if signal not in SUPPORTED_SIGNALS:
        raise ValueError("unsupported stop/requeue signal")
    return signal


def _validate_openmc_receipt(receipt: Mapping[str, Any]) -> None:
    if not str(receipt.get("schema", "")).startswith(OPENMC_SCHEMA_PREFIX):
        raise ValueError("unsupported OpenMC model receipt schema")
    if receipt.get("openmc_version") != OPENMC_VERSION:
        raise ValueError("OpenMC model receipt must be fixed-source 0.16.0")
    if receipt.get("production_run_authorized") is not False:
        raise ValueError("production execution is not authorized")
    model_xml = _mapping(receipt.get("model_xml"), "openmc_model.model_xml")
    if str(model_xml.get("path", "")).strip() == "":
        raise ValueError("model_xml path must be present")
    model_sha = _text(model_xml.get("sha256"), "model_xml.sha256")
    if len(model_sha) != 64 or any(ch not in "0123456789abcdef" for ch in model_sha):
        raise ValueError("model_xml hash is invalid")
    if receipt.get("claim") != "BOUNDED_SMOKE_ONLY":
        raise ValueError("OpenMC model claim must be bounded-smoke only")
    if receipt.get("all_bound_inputs_immutable") is not True:
        raise ValueError("OpenMC model inputs must be immutable")


def build_openmc_checkpoint_restart_plan(
    *,
    campaign_id: str,
    producer: str,
    openmc_model_receipt: Mapping[str, Any],
    particles_per_batch: int,
    total_batches: int,
    segment_batches: int,
    base_seed: int,
    statepoint_interval_batches: int | None = None,
    artifact_root: str | Path = "openmc_checkpoint_restart_plan",
    stop_signal: str = "SIGTERM",
    requeue_signal: str = "SIGTERM",
    stop_grace_seconds: int = 60,
    requeue_delay_seconds: int = 300,
    walltime_slack_seconds: int = 300,
) -> dict[str, Any]:
    """Build and hash a deterministic checkpoint/restart campaign plan."""
    campaign_id = _text(campaign_id, "campaign_id")
    if _text(producer, "producer") != "ParaStell":
        raise ValueError("checkpoint plan producer must be ParaStell")
    if statepoint_interval_batches is not None:
        statepoint_interval_batches = _positive_int(
            statepoint_interval_batches, "statepoint_interval_batches"
        )

    receipt = _mapping(openmc_model_receipt, "openmc_model_receipt")
    _validate_openmc_receipt(receipt)
    particles_per_batch = _positive_int(particles_per_batch, "particles_per_batch")
    total_batches = _positive_int(total_batches, "total_batches")
    segment_batches = _positive_int(segment_batches, "segment_batches")
    base_seed = _positive_int(base_seed, "base_seed")
    stop_grace_seconds = _positive_int(stop_grace_seconds, "stop_grace_seconds")
    requeue_delay_seconds = _positive_int(requeue_delay_seconds, "requeue_delay_seconds")
    walltime_slack_seconds = _positive_int(
        walltime_slack_seconds, "walltime_slack_seconds", minimum=0
    )
    if requeue_delay_seconds <= stop_grace_seconds:
        raise ValueError("requeue delay must exceed stop-grace delay")

    root = Path(artifact_root)
    stop = _validate_signal(stop_signal)
    requeue = _validate_signal(requeue_signal)

    segment_sizes = _segment_lengths(total_batches, segment_batches)
    if statepoint_interval_batches is None:
        segment_schedules = [_statepoint_schedule(size, None) for size in segment_sizes]
    else:
        segment_schedules = [
            _statepoint_schedule(size, statepoint_interval_batches)
            for size in segment_sizes
        ]

    segments: list[dict[str, Any]] = []
    total_source_histories = 0
    for index, (size, schedule) in enumerate(
        zip(segment_sizes, segment_schedules), start=1
    ):
        segment_histories = size * particles_per_batch
        total_source_histories += segment_histories
        segments.append(
            {
                "segment_index": index,
                "seed": base_seed + index - 1,
                "requested_batches": size,
                "source_histories": segment_histories,
                "statepoint_batches": schedule,
                "artifact_root": str(root / f"segment_{index:03d}"),
            }
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "producer": "ParaStell",
        "claim": "BOUNDED_SMOKE_ONLY",
        "run_mode": "fixed source",
        "artifact_root": str(root),
        "statepoint_restart_strategy": {
            "native_statepoint_continuation_claimed": False,
            "cumulative_statepoints_are_checkpoint_evidence": True,
            "segment_restart_mode": "segmented_independent_seed_runs",
            "statepoints_are_recoverable_plots_and_summaries_only": True,
        },
        "openmc_model_binding": {
            "schema": str(receipt["schema"]),
            "openmc_version": receipt["openmc_version"],
            "receipt_content_sha256": str(receipt.get("receipt_content_sha256", "")),
            "model_xml_sha256": str(receipt["model_xml"]["sha256"]),
        },
        "run_intent": {
            "total_batches": total_batches,
            "segment_batches": segment_batches,
            "particles_per_batch": particles_per_batch,
            "statepoint_interval_batches": statepoint_interval_batches,
            "total_source_histories": total_source_histories,
        },
        "signal_requeue_contract": {
            "stop_signal": stop,
            "requeue_signal": requeue,
            "stop_grace_seconds": stop_grace_seconds,
            "requeue_delay_seconds": requeue_delay_seconds,
            "walltime_slack_seconds": walltime_slack_seconds,
            "requeue_required": True,
        },
        "segments": segments,
        "production_run_authorized": False,
        "resource_controls": {
            "requested_cores": 1,
            "memory_max_bytes": 64 * 1024**3,
            "memory_swap_max_bytes": 0,
        },
    }
    payload["plan_sha256"] = _canonical_sha256(
        {k: v for k, v in payload.items() if k != "plan_sha256"}
    )
    validate_openmc_checkpoint_restart_plan(payload)
    return payload


def validate_openmc_checkpoint_restart_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported checkpoint plan schema")
    if plan.get("producer") != "ParaStell":
        raise ValueError("checkpoint plan producer identity is invalid")
    if plan.get("claim") != "BOUNDED_SMOKE_ONLY":
        raise ValueError("checkpoint plan claim is invalid")
    if plan.get("run_mode") != "fixed source":
        raise ValueError("checkpoint plan must use fixed source mode")

    strategy = _mapping(plan.get("statepoint_restart_strategy"), "strategy")
    if strategy.get("native_statepoint_continuation_claimed") is not False:
        raise ValueError("native statepoint continuation must not be claimed")
    if strategy.get("cumulative_statepoints_are_checkpoint_evidence") is not True:
        raise ValueError("cumulative statepoints must be evidence only")
    if strategy.get("segment_restart_mode") != "segmented_independent_seed_runs":
        raise ValueError("segment restart mode is invalid")

    run_intent = _mapping(plan.get("run_intent"), "run_intent")
    total_batches = _positive_int(run_intent.get("total_batches"), "total_batches")
    segment_batches = _positive_int(run_intent.get("segment_batches"), "segment_batches")
    particles_per_batch = _positive_int(
        run_intent.get("particles_per_batch"), "particles_per_batch"
    )
    if segment_batches > total_batches:
        raise ValueError("segment_batches cannot exceed total_batches")

    interval = run_intent.get("statepoint_interval_batches")
    if interval is not None:
        interval = _positive_int(interval, "statepoint_interval_batches")

    binding = _mapping(plan.get("openmc_model_binding"), "openmc_model_binding")
    if not str(binding.get("schema", "")).startswith(OPENMC_SCHEMA_PREFIX):
        raise ValueError("openmc binding schema is invalid")
    if str(binding.get("openmc_version", "")) != OPENMC_VERSION:
        raise ValueError("openmc version is invalid")
    receipt_hash = _text(binding.get("receipt_content_sha256"), "receipt hash")
    if len(receipt_hash) != 64 or any(
        ch not in "0123456789abcdef" for ch in receipt_hash
    ):
        raise ValueError("openmc receipt hash is invalid")
    model_xml_sha = _text(binding.get("model_xml_sha256"), "model_xml hash")
    if len(model_xml_sha) != 64 or any(
        ch not in "0123456789abcdef" for ch in model_xml_sha
    ):
        raise ValueError("model_xml hash is invalid")

    segments = plan.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("checkpoint plan must define segments")
    expected_segments = _segment_lengths(total_batches, segment_batches)
    if len(segments) != len(expected_segments):
        raise ValueError("segment count is invalid")

    expected_seed = None
    total_sources = 0
    observed_case_roots: set[str] = set()
    for index, (segment, expected_size) in enumerate(
        zip(segments, expected_segments), start=1
    ):
        if not isinstance(segment, Mapping):
            raise ValueError("segment definition must be a mapping")
        if _positive_int(segment.get("segment_index"), "segment_index") != index:
            raise ValueError("segment indices must be 1,2,...")
        requested = _positive_int(segment.get("requested_batches"), "requested_batches")
        if requested != expected_size:
            raise ValueError("segment requested_batches is inconsistent")
        seed = _positive_int(segment.get("seed"), "seed")
        if expected_seed is None:
            expected_seed = seed
        elif seed != expected_seed + index - 1:
            raise ValueError("segment seeds must be strictly sequential")
        schedule = segment.get("statepoint_batches")
        if not isinstance(schedule, list) or not schedule:
            raise ValueError("segment statepoint schedule must be a non-empty list")
        observed_schedule = [int(value) for value in schedule]
        if sorted(observed_schedule) != observed_schedule or any(
            value <= 0 for value in observed_schedule
        ):
            raise ValueError("segment statepoint schedule is invalid")
        if observed_schedule[-1] != requested:
            raise ValueError("segment statepoint schedule must include final batch")
        expected_schedule = _statepoint_schedule(requested, interval)
        if observed_schedule != expected_schedule:
            raise ValueError("segment statepoint schedule is not deterministic")
        source_histories = _positive_int(
            segment.get("source_histories"), "segment source_histories"
        )
        if source_histories != requested * particles_per_batch:
            raise ValueError("segment source histories is inconsistent")
        case_root = _text(segment.get("artifact_root"), "segment artifact_root")
        if not case_root.startswith(str(plan.get("artifact_root", ""))):
            raise ValueError("segment artifact_root must be under plan artifact_root")
        if not case_root:
            raise ValueError("segment artifact_root is missing")
        if case_root in observed_case_roots:
            raise ValueError("segment artifact_root values must be unique")
        observed_case_roots.add(case_root)
        total_sources += source_histories

    if total_sources != total_batches * particles_per_batch:
        raise ValueError("segment source histories do not match total intent")

    signal_plan = _mapping(plan.get("signal_requeue_contract"), "signal_requeue_contract")
    _validate_signal(signal_plan.get("stop_signal"))
    _validate_signal(signal_plan.get("requeue_signal"))
    stop_grace = _positive_int(
        signal_plan.get("stop_grace_seconds"), "stop_grace_seconds"
    )
    requeue_delay = _positive_int(
        signal_plan.get("requeue_delay_seconds"), "requeue_delay_seconds"
    )
    if requeue_delay <= stop_grace:
        raise ValueError("requeue delay must exceed stop grace")
    _positive_int(signal_plan.get("walltime_slack_seconds"), "walltime_slack_seconds")
    if signal_plan.get("requeue_required") is not True:
        raise ValueError("requeue requirement must be true")

    if plan.get("production_run_authorized") is not False:
        raise ValueError("production execution is not authorized")

    if _text(plan.get("artifact_root"), "artifact_root") == "":
        raise ValueError("artifact_root is required")

    provided = _text(plan.get("plan_sha256"), "plan_sha256")
    unsigned = {k: v for k, v in dict(plan).items() if k != "plan_sha256"}
    if provided != _canonical_sha256(unsigned):
        raise ValueError("checkpoint plan hash is invalid")


def write_openmc_checkpoint_restart_plan(
    path: str | Path, plan: Mapping[str, Any]
) -> Path:
    """Write a create-only checkpoint plan and reserve segment artifact roots."""
    validate_openmc_checkpoint_restart_plan(plan)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    artifact_root = Path(str(plan.get("artifact_root", "")))
    if artifact_root.exists():
        raise FileExistsError(artifact_root)
    artifact_root.mkdir(parents=True)
    for segment in plan.get("segments", ()):  # strict contract: validated above
        segment_root = Path(str(segment["artifact_root"]))
        if segment_root.exists():
            raise FileExistsError(segment_root)
        segment_root.parent.mkdir(parents=True, exist_ok=True)
        segment_root.mkdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
