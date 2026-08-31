"""Run one or more explicit fixed-source checkpoint segments.

The lane reads a full-response smoke receipt, a checkpoint plan, and a
user launch authorization receipt.  It then reruns the same instrumented model
with segment-local ``particles``/``batches``/``seed``/``statepoint`` overrides,
leaving tallies and surface-source settings untouched.

The executor is transport-only and local; no Slurm or other batch launcher is
used.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

from parastell.openmc_checkpoint_restart_plan import (
    validate_openmc_checkpoint_restart_plan,
)

SCHEMA = "parastell.parametric_openmc16_full_response_smoke/v1.0.0"
SEGMENT_SCHEMA = "parastell.parametric_openmc16_full_response_segment/v1.0.0"
AGGREGATE_SCHEMA = (
    "parastell.parametric_openmc16_full_response_segment_aggregate/v1.0.0"
)
LAUNCH_AUTH_SCHEMA = (
    "parastell.parametric_openmc16_full_response_launch_authorization/v1.0.0"
)
RUN_TIMEOUT_SECONDS = 1_800
OPENMC_VERSION = "0.16.0"


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_and_validate_binding(
    path: Path, expected_sha256: str, *, label: str
) -> Path:
    resolved = path.resolve(strict=True)
    actual = _sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(f"{label} hash mismatch")
    return resolved


def _load_json(
    path: Path, expected_sha256: str, *, label: str
) -> dict[str, Any]:
    source = _read_and_validate_binding(path, expected_sha256, label=label)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON is invalid")
    return value


def _is_text(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _validate_launch_authorization_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_sha256: str,
    checkpoint_plan_path: Path,
    checkpoint_plan_sha256: str,
) -> dict[str, Any]:
    if str(receipt.get("schema", "")) != LAUNCH_AUTH_SCHEMA:
        raise ValueError("launch authorization schema is invalid")
    if receipt.get("status") != "AUTHORIZED":
        raise ValueError("launch authorization is not active")
    if receipt.get("producer") != "ParaStell":
        raise ValueError("launch authorization producer is invalid")
    if receipt.get("launch_authorized") is not True:
        raise ValueError("launch authorization was not granted")
    if receipt.get("production_launch_authorized") is not False:
        raise ValueError("production launch is not authorized")
    unsigned = dict(receipt)
    content_hash = unsigned.pop("receipt_content_sha256", None)
    if content_hash != _canonical_sha(unsigned):
        raise ValueError("launch authorization content hash is invalid")
    binding = receipt.get("checkpoint_plan")
    if not isinstance(binding, Mapping):
        raise ValueError(
            "checkpoint authorization must bind a checkpoint plan"
        )
    bound_path = Path(str(binding.get("path", "")))
    if str(bound_path.resolve()) != str(checkpoint_plan_path.resolve()):
        raise ValueError(
            "launch authorization is not tied to this checkpoint plan"
        )
    if str(binding.get("sha256", "")) != checkpoint_plan_sha256:
        raise ValueError(
            "checkpoint plan hash is not accepted by launch receipt"
        )
    _is_text(binding.get("sha256"), "checkpoint plan hash")
    launch_id = _is_text(receipt.get("launch_id"), "launch id")
    return {
        "schema": LAUNCH_AUTH_SCHEMA,
        "path": str(checkpoint_plan_path.resolve()),
        "sha256": expected_sha256,
        "launch_id": launch_id,
    }


def _load_full_response_receipt(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, str]]:
    receipt = _load_json(
        path,
        expected_sha256,
        label="full-response smoke receipt",
    )
    if receipt.get("schema") != SCHEMA:
        raise ValueError("full-response smoke receipt schema is invalid")
    if receipt.get("status") != "PASS":
        raise ValueError("full-response smoke receipt is not PASS")
    if receipt.get("claim") != "BOUNDED_INFRASTRUCTURE_SMOKE_ONLY":
        raise ValueError("full-response smoke claim is unexpected")
    if receipt.get("openmc_version") != OPENMC_VERSION:
        raise ValueError("full-response openmc version is not 0.16.0")
    if receipt.get("run_mode") != "fixed source":
        raise ValueError("full-response model is not fixed source")
    if receipt.get("all_bound_inputs_immutable") is not True:
        raise ValueError("full-response model inputs are not immutable")
    if receipt.get("production_run_authorized") is not False:
        raise ValueError("full-response run is not non-production")
    instrumented = receipt.get("instrumented_model_xml")
    if not isinstance(instrumented, Mapping):
        raise ValueError(
            "full-response receipt must include instrumented_model_xml"
        )
    if (
        instrumented.get("tallies_instrumented") is not True
        or instrumented.get("surface_bank_instrumented") is not True
    ):
        raise ValueError("full-response model is not fully instrumented")
    for key in ("response_plan_sha256", "surface_manifest_sha256"):
        digest = str(instrumented.get(key, ""))
        if len(digest) != 64 or any(
            ch not in "0123456789abcdef" for ch in digest
        ):
            raise ValueError(f"instrumented {key} is invalid")
    instrumented_path = _read_and_validate_binding(
        Path(str(instrumented.get("path", ""))),
        str(instrumented.get("sha256", "")),
        label="instrumented model.xml",
    )
    source_path = Path(str(instrumented_path))
    return receipt, {
        "path": str(source_path),
        "sha256": _sha256(source_path),
    }


def _run_openmc(
    output: Path,
    *,
    threads: int,
    timeout_seconds: int,
    statepoint_batches: list[int],
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["openmc", "-s", str(threads)],
            cwd=output,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "exit_code": int(completed.returncode),
            "timed_out": False,
            "output": completed.stdout,
            "statepoint_batches": list(statepoint_batches),
        }
    except subprocess.TimeoutExpired as error:
        value = error.stdout or ""
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return {
            "exit_code": -1,
            "timed_out": True,
            "output": value,
            "statepoint_batches": list(statepoint_batches),
        }


def _load_openmc_model(model_xml: Path):
    import openmc

    return openmc.Model.from_model_xml(model_xml)


def _statepoint_file_rows(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output.glob("statepoint.*.h5")):
        try:
            batch = int(path.name.split(".")[1])
        except (IndexError, ValueError):
            batch = -1
        rows.append(
            {
                "batch": batch,
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    return rows


def _surface_banks(output: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "sha256": _sha256(path)}
        for path in sorted(output.glob("surface_source*.h5"))
    ]


def _segment_receipt_path(segment_root: Path) -> Path:
    return segment_root / "OPENMC_FULL_RESPONSE_SEGMENT_RECEIPT.json"


def _validated_existing_segment_receipt(
    path: Path, *, segment: Mapping[str, Any], model_sha256: str
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(receipt)
    content_hash = unsigned.pop("receipt_content_sha256", None)
    if (
        receipt.get("schema") != SEGMENT_SCHEMA
        or receipt.get("status") != "PASS"
        or content_hash != _canonical_sha(unsigned)
        or receipt.get("segment_index") != segment.get("segment_index")
        or receipt.get("settings", {}).get("seed") != segment.get("seed")
        or receipt.get("settings", {}).get("requested_batches")
        != segment.get("requested_batches")
        or receipt.get("instrumented_model_xml", {}).get("sha256")
        != model_sha256
        or not receipt.get("statepoints")
        or not receipt.get("surface_source_files")
    ):
        raise ValueError("existing segment receipt is not reusable")
    return receipt


def _required_schedule(segment: Mapping[str, Any]) -> list[int]:
    rows = list(segment.get("statepoint_batches", ()))
    if not rows:
        raise ValueError("segment has no statepoint schedule")
    values = [int(value) for value in rows]
    if sorted(values) != values or values[-1] <= 0:
        raise ValueError("segment schedule is invalid")
    return values


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _run_segment(
    *,
    segment: Mapping[str, Any],
    openmc_model_xml: Path,
    particles_per_batch: int,
    threads: int,
    timeout_seconds: int,
    run_openmc: Callable[
        [Path, int, int, list[int]],
        dict[str, Any],
    ],
    segment_root: Path,
) -> tuple[dict[str, Any], Path]:
    model_xml = openmc_model_xml
    segment_root = segment_root.resolve()
    if segment_root.exists() is False:
        raise ValueError("segment artifact root must exist")
    segment_index = int(segment.get("segment_index", -1))
    requested = int(segment.get("requested_batches"))
    seed = int(segment.get("seed"))
    statepoint_batches = _required_schedule(segment)
    model = _load_openmc_model(model_xml)

    original_tallies = model.tallies
    original_source_settings = vars(model.settings).get("source", None)
    original_settings = dict(vars(model.settings))

    model.settings.particles = int(particles_per_batch)
    model.settings.batches = int(requested)
    model.settings.seed = int(seed)
    model.settings.statepoint = {"batches": statepoint_batches}
    # Preserve all tally and surface-source settings by only mutating explicit
    # transport controls above.
    segment_model_xml = segment_root / "model.xml"
    model.export_to_model_xml(segment_model_xml)

    runtime = run_openmc(
        segment_root,
        threads,
        timeout_seconds,
        statepoint_batches,
    )

    statepoints = _statepoint_file_rows(segment_root)
    observed = [
        row["batch"]
        for row in sorted(statepoints, key=lambda row: row["batch"])
    ]
    completed = bool(
        not runtime.get("timed_out")
        and int(runtime.get("exit_code", -1)) == 0
        and observed == statepoint_batches
    )
    banks = _surface_banks(segment_root)
    current_settings = vars(model.settings)
    changed_setting_keys = [
        key
        for key, value in current_settings.items()
        if original_settings.get(key) != value
    ]
    preserve_ok = (
        model.tallies == original_tallies
        and current_settings.get("source") == original_source_settings
        and set(changed_setting_keys)
        <= {"particles", "batches", "seed", "statepoint"}
    )

    passed = bool(completed and banks and preserve_ok)
    receipt = {
        "schema": SEGMENT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "segment_index": segment_index,
        "segment_root": str(segment_root),
        "claim": "SEGMENT_RESTART_EVIDENCE_ONLY",
        "status": "PASS" if passed else "BLOCKED_SEGMENT_RESTART_RUN",
        "run_mode": "fixed source",
        "preserved_tallies_and_surface_settings": bool(preserve_ok),
        "instrumented_model_xml": {
            "path": str(openmc_model_xml),
            "sha256": _sha256(openmc_model_xml),
        },
        "settings": {
            "particles_per_batch": particles_per_batch,
            "requested_batches": requested,
            "seed": seed,
            "statepoint_batches": statepoint_batches,
            "threads": threads,
            "timeout_seconds": timeout_seconds,
        },
        "statepoint_batches": {
            "expected": statepoint_batches,
            "observed": observed,
            "complete": observed == statepoint_batches,
        },
        "statepoints": statepoints,
        "surface_source_files": banks,
        "segment_source_histories": int(requested * particles_per_batch),
        "preserved_setting_keys": sorted(changed_setting_keys),
        "run": {
            "exit_code": int(runtime.get("exit_code", -1)),
            "timed_out": bool(runtime.get("timed_out")),
            "output": runtime.get("output", ""),
        },
        "production_run_authorized": False,
    }
    receipt["receipt_content_sha256"] = _canonical_sha(receipt)
    receipt_path = _segment_receipt_path(segment_root)
    _write_create_only(receipt_path, receipt)
    return receipt, receipt_path


def _build_inventory(
    *,
    campaign_id: str,
    full_response_receipt: Mapping[str, Any],
    checkpoint_plan: Mapping[str, Any],
    launch_authorization: Mapping[str, Any],
    segment_rows: list[dict[str, Any]],
    instrumented_model_xml: Mapping[str, Any],
) -> dict[str, Any]:
    total_histories = sum(
        row["segment_source_histories"] for row in segment_rows
    )
    if total_histories <= 0:
        raise ValueError("segment history total is invalid")
    entries = []
    for row in segment_rows:
        weight = row["segment_source_histories"] / float(total_histories)
        entries.append(
            {
                "segment_index": row["segment_index"],
                "segment_root": row["segment_root"],
                "segment_seed": row["segment_seed"],
                "requested_batches": row["requested_batches"],
                "source_histories": row["segment_source_histories"],
                "statepoint_schedule": row["statepoint_batches"],
                "history_weight": weight,
                "segment_receipt": {
                    "path": row["segment_receipt_path"],
                    "sha256": row["segment_receipt_sha256"],
                },
                "statepoints": row["statepoints"],
                "surface_source_files": row["surface_source_files"],
                "status": row["status"],
            }
        )
    return {
        "schema": AGGREGATE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "claim": "SEGMENTED_FIXED_SOURCE_RESTART_ONLY",
        "status": (
            "SEGMENTS_COMPLETE"
            if all(row["status"] == "PASS" for row in segment_rows)
            else "SEGMENTS_INCOMPLETE"
        ),
        "run_mode": "fixed source",
        "production_run_authorized": False,
        "restart_strategy": {
            "native_statepoint_continuation_claimed": False,
            "cumulative_statepoints_are_checkpoint_evidence": True,
            "segment_restart_mode": checkpoint_plan[
                "statepoint_restart_strategy"
            ]["segment_restart_mode"],
        },
        "full_response_smoke_receipt": {
            "schema": full_response_receipt.get("schema", SCHEMA),
            "path": str(full_response_receipt["__source_path"]),
            "sha256": full_response_receipt["__source_sha256"],
        },
        "checkpoint_plan": {
            "path": str(checkpoint_plan["__source_path"]),
            "sha256": checkpoint_plan["__source_sha256"],
        },
        "launch_authorization": launch_authorization,
        "instrumented_model_xml": dict(instrumented_model_xml),
        "statepoint_policy": {
            "cumulative_statepoints_are_evidence_only": True,
            "segments": [
                {
                    "segment_index": row["segment_index"],
                    "statepoint_batches": row["statepoint_batches"],
                }
                for row in segment_rows
            ],
        },
        "segment_results": entries,
        "total_source_histories": total_histories,
        "signal_requeue_contract": checkpoint_plan["signal_requeue_contract"],
        "segments": segment_rows,
    }


def run_checkpoint_segments(
    *,
    full_response_receipt_path: Path,
    expected_full_response_receipt_sha256: str,
    checkpoint_plan_path: Path,
    expected_checkpoint_plan_sha256: str,
    launch_authorization_path: Path,
    expected_launch_authorization_sha256: str,
    threads: int = 1,
    timeout_seconds: int = RUN_TIMEOUT_SECONDS,
    run_openmc: Callable[..., dict[str, Any]] = _run_openmc,
) -> dict[str, Any]:
    receipt_path = _read_and_validate_binding(
        full_response_receipt_path,
        expected_full_response_receipt_sha256,
        label="full-response smoke receipt",
    )
    if not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    full_response_receipt, instrumented_binding = _load_full_response_receipt(
        receipt_path, expected_full_response_receipt_sha256
    )
    full_response_receipt["__source_path"] = str(receipt_path)
    full_response_receipt["__source_sha256"] = (
        expected_full_response_receipt_sha256
    )

    checkpoint_plan = _load_json(
        checkpoint_plan_path,
        expected_checkpoint_plan_sha256,
        label="checkpoint plan",
    )
    if not isinstance(checkpoint_plan, dict):
        raise ValueError("checkpoint plan must be an object")
    validate_openmc_checkpoint_restart_plan(checkpoint_plan)
    checkpoint_plan["__source_path"] = str(
        checkpoint_plan_path.resolve(strict=True)
    )
    checkpoint_plan["__source_sha256"] = expected_checkpoint_plan_sha256
    if timeout_seconds > int(
        checkpoint_plan["signal_requeue_contract"]["walltime_limit_seconds"]
    ):
        raise ValueError("timeout exceeds checkpoint-plan walltime limit")

    launch_receipt = _load_json(
        launch_authorization_path,
        expected_launch_authorization_sha256,
        label="launch authorization",
    )
    auth_binding = _validate_launch_authorization_receipt(
        launch_receipt,
        expected_sha256=expected_launch_authorization_sha256,
        checkpoint_plan_path=checkpoint_plan_path.resolve(strict=True),
        checkpoint_plan_sha256=expected_checkpoint_plan_sha256,
    )

    campaign_root = Path(str(checkpoint_plan["artifact_root"])).resolve()
    if not campaign_root.exists():
        raise ValueError("checkpoint campaign root does not exist")

    instrumented_model_xml = Path(str(instrumented_binding["path"]))
    if not instrumented_model_xml.exists():
        raise ValueError("instrumented model.xml is missing")
    plan_binding = checkpoint_plan["openmc_model_binding"]
    instrumented_receipt = full_response_receipt["instrumented_model_xml"]
    if (
        plan_binding.get("receipt_content_sha256")
        != full_response_receipt.get("receipt_content_sha256")
        or plan_binding.get("model_xml_sha256")
        != instrumented_binding["sha256"]
        or plan_binding.get("response_plan_sha256")
        != instrumented_receipt["response_plan_sha256"]
        or plan_binding.get("surface_manifest_sha256")
        != instrumented_receipt["surface_manifest_sha256"]
    ):
        raise ValueError("checkpoint plan and instrumented model disagree")

    campaign_rows = []
    for segment in checkpoint_plan["segments"]:
        segment_root = Path(str(segment["artifact_root"]))
        expected_batches = _required_schedule(segment)
        segment_root = segment_root.resolve()
        if segment_root.parent != campaign_root:
            raise ValueError("segment root is outside campaign root")
        segment_receipt_path = _segment_receipt_path(segment_root)
        if segment_receipt_path.exists():
            segment_receipt = _validated_existing_segment_receipt(
                segment_receipt_path,
                segment=segment,
                model_sha256=instrumented_binding["sha256"],
            )
        else:
            if any(segment_root.iterdir()):
                raise ValueError(
                    "partial segment without a sealed receipt requires a new plan"
                )
            segment_receipt, segment_receipt_path = _run_segment(
                segment=segment,
                openmc_model_xml=instrumented_model_xml,
                particles_per_batch=int(
                    checkpoint_plan["run_intent"]["particles_per_batch"]
                ),
                threads=threads,
                timeout_seconds=timeout_seconds,
                run_openmc=run_openmc,
                segment_root=segment_root,
            )

        segment_rows = json.loads(
            segment_receipt_path.read_text(encoding="utf-8")
        )
        if segment_rows != segment_receipt:
            raise RuntimeError("segment receipt hash computation changed")
        campaign_rows.append(
            {
                "segment_index": segment["segment_index"],
                "segment_seed": segment["seed"],
                "segment_root": str(segment_root),
                "segment_source_histories": segment_receipt[
                    "segment_source_histories"
                ],
                "requested_batches": segment_receipt["settings"][
                    "requested_batches"
                ],
                "statepoint_batches": expected_batches,
                "segment_receipt_path": str(segment_receipt_path),
                "segment_receipt_sha256": _sha256(segment_receipt_path),
                "statepoints": segment_receipt["statepoints"],
                "surface_source_files": segment_receipt[
                    "surface_source_files"
                ],
                "status": segment_receipt["status"],
            }
        )

    inventory = _build_inventory(
        campaign_id=checkpoint_plan["campaign_id"],
        full_response_receipt=full_response_receipt,
        checkpoint_plan=checkpoint_plan,
        launch_authorization=auth_binding,
        segment_rows=campaign_rows,
        instrumented_model_xml=instrumented_binding,
    )
    inventory["aggregate_receipt_content_sha256"] = _canonical_sha(inventory)

    inventory_path = (
        campaign_root / "OPENMC_FULL_RESPONSE_SEGMENT_INVENTORY.json"
    )
    _write_create_only(inventory_path, inventory)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_plan", type=Path)
    parser.add_argument("full_response_receipt", type=Path)
    parser.add_argument("launch_authorization_receipt", type=Path)
    parser.add_argument("--expected-checkpoint-plan-sha256", required=True)
    parser.add_argument(
        "--expected-full-response-receipt-sha256", required=True
    )
    parser.add_argument(
        "--expected-launch-authorization-sha256", required=True
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--timeout-seconds", type=int, default=RUN_TIMEOUT_SECONDS
    )
    args = parser.parse_args()

    run_checkpoint_segments(
        full_response_receipt_path=args.full_response_receipt,
        expected_full_response_receipt_sha256=args.expected_full_response_receipt_sha256,
        checkpoint_plan_path=args.checkpoint_plan,
        expected_checkpoint_plan_sha256=args.expected_checkpoint_plan_sha256,
        launch_authorization_path=args.launch_authorization_receipt,
        expected_launch_authorization_sha256=args.expected_launch_authorization_sha256,
        threads=args.threads,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
