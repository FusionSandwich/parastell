"""Run sealed OpenMC 0.16 geometry debug in a stdlib-only process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping

from parastell.parametric_openmc16_geometry_debug_contract import (
    LOG_FILENAME,
    RUN_RECEIPT_FILENAME,
    RUN_RECEIPT_SCHEMA,
    bind_openmc_executable,
    canonical_sha256,
    load_runtime_environment,
    load_sealed_model_receipt,
    parse_openmc_geometry_debug_log,
    sha256_file,
)


SCHEMA = RUN_RECEIPT_SCHEMA


def _load_bound_receipt(
    path: Path,
    expected_sha256: str,
    *,
    model_xml: Path,
    dagmc_h5m: Path,
) -> dict[str, Any]:
    """Compatibility adapter for the bounded full-response smoke script."""
    receipt, _, bound_paths = load_sealed_model_receipt(
        path, expected_sha256, model_xml
    )
    sealed_dagmc = bound_paths.get("sealed_input:dagmc_h5m")
    if sealed_dagmc != dagmc_h5m.resolve(strict=True):
        raise ValueError("sealed receipt references another DAGMC H5M")
    return receipt


def _nuclear_data_hashes(receipt: Mapping[str, Any]) -> dict[str, str]:
    bindings = receipt.get("validated_input_bindings", {})
    result = {}
    for label, row in bindings.items():
        if not str(label).startswith("nuclear_data:"):
            continue
        path = Path(str(row["path"])).resolve(strict=True)
        if sha256_file(path) != row.get("sha256"):
            raise ValueError(f"selected nuclear-data hash mismatch: {path}")
        result[str(path)] = str(row["sha256"])
    if not result:
        raise ValueError("sealed receipt has no nuclear-data bindings")
    return result


def _runtime_inputs(
    model_xml: Path, dagmc_h5m: Path, receipt: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """Return stage-2 external bindings without importing OpenMC or MOAB."""
    bindings = receipt.get("validated_input_bindings", {})
    names = {
        "dagmc_h5m": "dagmc_h5m",
        "source_mesh": "source_mesh_h5m",
        "cross_sections_xml": "cross_sections_xml",
    }
    result = {}
    for output_name, binding_name in names.items():
        row = bindings.get(binding_name)
        if not isinstance(row, Mapping):
            raise ValueError(f"sealed {binding_name} binding is missing")
        path = Path(str(row["path"])).resolve(strict=True)
        if sha256_file(path) != row.get("sha256"):
            raise ValueError("sealed runtime input hash mismatch")
        result[output_name] = {
            "path": str(path),
            "sha256": str(row["sha256"]),
        }
    if Path(result["dagmc_h5m"]["path"]) != dagmc_h5m.resolve(strict=True):
        raise ValueError("sealed runtime references another DAGMC H5M")
    if not model_xml.resolve(strict=True).is_file():
        raise ValueError("runtime model XML is missing")
    return result


def _observed_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=False)
    result: dict[str, Any] = {
        "path": str(resolved),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": None,
        "sha256": None,
    }
    if path.is_file():
        result["size_bytes"] = path.stat().st_size
        result["sha256"] = sha256_file(path)
    return result


def _snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        label: _observed_file(path) for label, path in sorted(paths.items())
    }


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment),
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
            "wall_time_seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {
            "exit_code": -1,
            "timed_out": True,
            "output": output,
            "wall_time_seconds": time.monotonic() - started,
        }


def _terminal_classification(
    *,
    failure_stage: str | None,
    run: Mapping[str, Any],
    qualification: Mapping[str, Any] | None,
    immutable: bool,
) -> str:
    if not immutable:
        return "BLOCKED_INPUT_MUTATION"
    if failure_stage == "PREFLIGHT":
        return "BLOCKED_PREFLIGHT"
    if failure_stage == "EXECUTION":
        return "BLOCKED_EXECUTION_ERROR"
    if failure_stage == "LOG_PARSER":
        return "BLOCKED_LOG_PARSER"
    if bool(run.get("timed_out")):
        return "BLOCKED_TIMEOUT"
    if run.get("exit_code") is not None and int(run["exit_code"]) != 0:
        return "BLOCKED_NONZERO_EXIT"
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("pass") is not True
    ):
        return "BLOCKED_LOG_QUALIFICATION"
    return "PASS"


def execute_geometry_debug(
    *,
    model_xml: Path,
    model_receipt: Path,
    expected_model_receipt_sha256: str,
    output_directory: Path,
    openmc_executable: Path,
    expected_openmc_executable_sha256: str,
    runtime_env: Path,
    expected_runtime_env_sha256: str,
    threads: int,
    timeout_seconds: int,
) -> tuple[Path, bool]:
    """Execute once and always seal a terminal receipt after output creation."""
    output = output_directory.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / RUN_RECEIPT_FILENAME
    tracked = {
        "model_xml": model_xml,
        "model_receipt": model_receipt,
        "openmc_executable": openmc_executable,
        "runtime_env": runtime_env,
    }
    before = _snapshot(tracked)
    command: list[str] | None = None
    runtime_binding: dict[str, Any] | None = None
    executable_binding: dict[str, Any] | None = None
    environment_names: list[str] = []
    required_volume_ids: list[int] = []
    qualification: Mapping[str, Any] | None = None
    failure: dict[str, str] | None = None
    failure_stage: str | None = None
    run: dict[str, Any] = {
        "exit_code": None,
        "timed_out": False,
        "wall_time_seconds": 0.0,
    }
    log_binding: dict[str, Any] | None = None
    run_model: Path | None = None
    expected_run_model_sha256: str | None = None
    try:
        failure_stage = "PREFLIGHT"
        if isinstance(threads, bool) or threads < 1:
            raise ValueError("threads must be a positive integer")
        if isinstance(timeout_seconds, bool) or timeout_seconds < 1:
            raise ValueError("timeout must be a positive integer")
        _, required_volume_ids, sealed_paths = load_sealed_model_receipt(
            model_receipt,
            expected_model_receipt_sha256,
            model_xml,
        )
        executable, executable_binding = bind_openmc_executable(
            openmc_executable, expected_openmc_executable_sha256
        )
        environment, runtime_binding = load_runtime_environment(
            runtime_env, expected_runtime_env_sha256
        )
        environment_names = sorted(environment)
        tracked.update(sealed_paths)
        tracked["openmc_executable"] = executable
        tracked["runtime_env"] = Path(runtime_binding["path"])
        run_model = output / "model.xml"
        with run_model.open("xb") as destination:
            with Path(model_xml).resolve(strict=True).open("rb") as source:
                shutil.copyfileobj(source, destination)
        tracked["run_model_xml"] = run_model
        expected_run_model_sha256 = sha256_file(
            Path(model_xml).resolve(strict=True)
        )
        before = _snapshot(tracked)
        command = [str(executable), "-g", "-s", str(threads)]
        failure_stage = "EXECUTION"
        run = _run(
            command,
            cwd=output,
            timeout_seconds=timeout_seconds,
            environment=environment,
        )
        log = output / LOG_FILENAME
        with log.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(str(run.get("output", "")))
        log_binding = {
            "path": str(log),
            "sha256": sha256_file(log),
            "size_bytes": log.stat().st_size,
        }
        failure_stage = "LOG_PARSER"
        qualification = parse_openmc_geometry_debug_log(
            str(run.get("output", "")),
            exit_code=int(run["exit_code"]),
            expected_threads=threads,
            required_cell_ids=required_volume_ids,
        )
        failure_stage = None
    except Exception as error:  # terminal receipt is the fail-closed boundary
        failure = {
            "stage": failure_stage or "INTERNAL",
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    finally:
        after = _snapshot(tracked)
        immutable = before == after
        if run_model is not None and run_model.is_file():
            immutable = bool(
                immutable
                and sha256_file(run_model) == expected_run_model_sha256
            )
        classification = _terminal_classification(
            failure_stage=(failure or {}).get("stage"),
            run=run,
            qualification=qualification,
            immutable=immutable,
        )
        passed = classification == "PASS"
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if passed else "BLOCKED",
            "terminal_classification": classification,
            "terminal": True,
            "claim": "BOUNDED_NONPRODUCTION",
            "command": command,
            "threads": threads,
            "timeout_seconds": timeout_seconds,
            "timed_out": bool(run.get("timed_out")),
            "exit_code": run.get("exit_code"),
            "wall_time_seconds": float(run.get("wall_time_seconds", 0.0)),
            "input_model_receipt_expected_sha256": (
                expected_model_receipt_sha256
            ),
            "openmc_executable": executable_binding,
            "runtime_environment": {
                "file": runtime_binding,
                "variable_names": environment_names,
                "inherit_parent_environment": False,
                "values_recorded_in_receipt": False,
            },
            "required_native_volume_ids": required_volume_ids,
            "input_hashes_before": before,
            "input_hashes_after": after,
            "inputs_immutable": immutable,
            "openmc_log": log_binding,
            "log_qualification": qualification,
            "failure": failure,
            "production_run_authorized": False,
        }
        result["receipt_content_sha256"] = canonical_sha256(result)
        with result_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                result, stream, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
    return result_path, passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_xml", type=Path)
    parser.add_argument("model_receipt", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-model-receipt-sha256", required=True)
    parser.add_argument("--openmc-executable", type=Path, required=True)
    parser.add_argument("--expected-openmc-executable-sha256", required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--expected-runtime-env-sha256", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    result_path, passed = execute_geometry_debug(
        model_xml=args.model_xml,
        model_receipt=args.model_receipt,
        expected_model_receipt_sha256=args.expected_model_receipt_sha256,
        output_directory=args.output_directory,
        openmc_executable=args.openmc_executable,
        expected_openmc_executable_sha256=(
            args.expected_openmc_executable_sha256
        ),
        runtime_env=args.runtime_env,
        expected_runtime_env_sha256=args.expected_runtime_env_sha256,
        threads=args.threads,
        timeout_seconds=args.timeout_seconds,
    )
    print(result_path)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
