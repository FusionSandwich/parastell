"""Run one hash-bound parametric direct-90 DAGMC export.

This wrapper is intentionally separate from the historical direct-90 runner.
It preserves the physical source-audit bindings required by either the
alternate swept-coil exporter or the authoritative continuous-radial exporter
and never launches a native-DAGMC or OpenMC successor.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - Linux execution path
    resource = None


SCHEMA = "parastell.parametric_direct90_export_terminal/v1.0.0"
WALLTIME_SECONDS = 21_600
TERMINATION_GRACE_SECONDS = 30
REQUESTED_THREADS = 32
POLICY_TOTAL_PHYSICAL_CORES = 256
POLICY_LIMIT_CORES = 64
EXPECTED_LEASE_ROOT = Path("/home/apollon/josma/.codex/ssh-poly-core-budget")
OUTPUT_NAMES = (
    "PREMESH_TOPOLOGY.json",
    "dagmc.h5m",
    "H5M_WRITEBACK_AUDIT.json",
    "DAGMC_EXPORT_RECEIPT.json",
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "32",
    "OMP_THREAD_LIMIT": "32",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
ALLOWED_EXPORTERS = {
    "export_parametric_direct90_dagmc.py": (
        "scripts.export_parametric_direct90_dagmc"
    ),
    "export_continuous_parametric_direct90_dagmc.py": (
        "scripts.export_continuous_parametric_direct90_dagmc"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _proc_start_time(proc_root: Path, pid: int) -> int | None:
    try:
        text = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        fields = text[text.rindex(")") + 2 :].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _discover_lease(
    root: Path,
    *,
    guard_pid: int | None = None,
    session_id: int | None = None,
    proc_root: Path = Path("/proc"),
    timeout_seconds: float = 5.0,
) -> dict[str, int | str]:
    guard = os.getppid() if guard_pid is None else int(guard_pid)
    sid = os.getsid(guard) if session_id is None else int(session_id)
    deadline = time.monotonic() + timeout_seconds
    while True:
        matches = []
        for path in root.glob("*.lease"):
            fields = path.read_text(encoding="utf-8").split()
            if len(fields) != 4 or not all(
                value.isdigit() for value in fields
            ):
                continue
            cores, pid, start_time, lease_sid = map(int, fields)
            if (
                cores == REQUESTED_THREADS
                and pid == guard
                and lease_sid == sid
                and _proc_start_time(proc_root, pid) == start_time
            ):
                matches.append(
                    {
                        "lease_id": path.stem,
                        "cores": cores,
                        "pid": pid,
                        "start_time": start_time,
                        "session_id": lease_sid,
                    }
                )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError("multiple matching live ssh-poly leases")
        if time.monotonic() >= deadline:
            raise RuntimeError("live 32-core ssh-poly lease was not found")
        time.sleep(0.05)


def _terminate(process: subprocess.Popen[Any]) -> dict[str, Any]:
    actions: list[str] = []
    errors: list[str] = []
    for name, sig in (
        ("SIGTERM", signal.SIGTERM),
        ("SIGKILL", signal.SIGKILL),
    ):
        try:
            os.killpg(process.pid, sig)
            actions.append(name)
        except OSError as exc:
            errors.append(f"{name} {type(exc).__name__}: {exc}")
        try:
            process.wait(timeout=TERMINATION_GRACE_SECONDS)
            break
        except subprocess.TimeoutExpired:
            continue
        except OSError as exc:
            errors.append(f"wait {type(exc).__name__}: {exc}")
            break
    return {
        "actions": actions,
        "errors": errors,
        "child_reaped": process.poll() is not None,
    }


def _file_binding(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(f"hash mismatch for {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual,
    }


def _output_evidence(output_root: Path) -> dict[str, Any]:
    return {
        name: (
            {
                "present": True,
                "bytes": (output_root / name).stat().st_size,
                "sha256": sha256_file(output_root / name),
            }
            if (output_root / name).is_file()
            else {"present": False}
        )
        for name in OUTPUT_NAMES
    }


def _command(args: argparse.Namespace) -> list[str]:
    module = ALLOWED_EXPORTERS.get(args.exporter.name)
    if module is None:
        raise ValueError("exporter is not an allowed direct-90 lane")
    return [
        sys.executable,
        "-m",
        module,
        "--source-root",
        str(args.source_root.resolve()),
        "--manifest-sha256",
        args.expected_manifest_sha256,
        "--audit-path",
        str(args.audit_path.resolve()),
        "--audit-sha256",
        args.expected_audit_sha256,
        "--audit-seal-path",
        str(args.audit_seal_path.resolve()),
        "--audit-seal-sha256",
        args.expected_audit_seal_sha256,
        "--audit-producer-path",
        str(args.audit_producer_path.resolve()),
        "--audit-producer-sha256",
        args.expected_audit_producer_sha256,
        "--output-root",
        str(args.output_root.resolve()),
        "--min-mesh-size-cm",
        "5.0",
        "--max-mesh-size-cm",
        "20.0",
        "--algorithm",
        "1",
        "--threads",
        "32",
    ]


def run(args: argparse.Namespace) -> int:
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    receipt_path = args.terminal_receipt.resolve()
    log_path = args.export_log.resolve()
    output_root = args.output_root.resolve()
    if receipt_path.exists() or log_path.exists() or output_root.exists():
        raise FileExistsError("receipt, log, and output root are create-only")
    if not receipt_path.parent.is_dir() or not log_path.parent.is_dir():
        raise FileNotFoundError("receipt and log parents must already exist")

    process: subprocess.Popen[Any] | None = None
    return_code = 125
    status = "WRAPPER_PRELAUNCH_FAILURE"
    error = None
    reaping = None
    bindings: dict[str, Any] = {}
    command: list[str] = []
    try:
        wrapper = Path(__file__).resolve()
        repository_root = wrapper.parents[1]
        if sha256_file(wrapper) != args.expected_wrapper_sha256:
            raise ValueError("wrapper hash mismatch")
        if args.requested_threads != REQUESTED_THREADS:
            raise ValueError("requested thread count is not 32")
        if args.exporter.name not in ALLOWED_EXPORTERS:
            raise ValueError("exporter is not an allowed direct-90 lane")
        expected_exporter = (
            repository_root / "scripts" / args.exporter.name
        ).resolve()
        if args.exporter.resolve() != expected_exporter:
            raise ValueError("exporter path is not the repository module")
        if (
            args.policy_total_physical_cores != POLICY_TOTAL_PHYSICAL_CORES
            or args.policy_limit_cores != POLICY_LIMIT_CORES
        ):
            raise ValueError("ssh-poly physical-core policy mismatch")
        lease_root = args.lease_root.resolve()
        if lease_root != EXPECTED_LEASE_ROOT:
            raise ValueError("unexpected ssh-poly lease root")
        environment = {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        }
        if environment != THREAD_ENVIRONMENT:
            raise ValueError(f"thread environment mismatch: {environment}")
        if not args.attempt_id or not args.nonce:
            raise ValueError("attempt and nonce are required")
        lease = _discover_lease(lease_root)
        source_manifest = (
            args.source_root.resolve() / "PARAMETRIC_GEOMETRY_MANIFEST.json"
        )
        file_bindings = {
            "exporter": _file_binding(
                args.exporter, args.expected_exporter_sha256
            ),
            "runtime_receipt": _file_binding(
                args.runtime_receipt, args.expected_runtime_receipt_sha256
            ),
            "source_manifest": _file_binding(
                source_manifest, args.expected_manifest_sha256
            ),
            "audit": _file_binding(
                args.audit_path, args.expected_audit_sha256
            ),
            "audit_seal": _file_binding(
                args.audit_seal_path, args.expected_audit_seal_sha256
            ),
            "audit_producer": _file_binding(
                args.audit_producer_path,
                args.expected_audit_producer_sha256,
            ),
        }
        bindings = {
            "attempt_id": args.attempt_id,
            "nonce": args.nonce,
            "lease": lease,
            "requested_threads": REQUESTED_THREADS,
            "policy_total_physical_cores": POLICY_TOTAL_PHYSICAL_CORES,
            "policy_limit_cores": POLICY_LIMIT_CORES,
            "python_executable": sys.executable,
            "python_sha256": sha256_file(Path(sys.executable)),
            "python_version": platform.python_version(),
            "thread_environment": environment,
            "files": file_bindings,
        }
        command = _command(args)
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=repository_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = int(process.wait(timeout=WALLTIME_SECONDS))
                status = (
                    "EXPORT_COMPLETED_NATIVE_GATES_PENDING"
                    if return_code == 0
                    else "EXPORT_FAILED_NO_AUTOMATIC_SUCCESSOR"
                )
            except subprocess.TimeoutExpired:
                reaping = _terminate(process)
                return_code = 124
                status = (
                    "EXPORT_TIMEOUT_REAPED_NO_AUTOMATIC_SUCCESSOR"
                    if reaping["child_reaped"]
                    else "EXPORT_TIMEOUT_REAP_INCOMPLETE"
                )
    except Exception as exc:  # preserve terminal evidence for every failure
        if process is not None and process.poll() is None:
            reaping = _terminate(process)
        error = f"{type(exc).__name__}: {exc}"

    maximum_rss_kib = (
        int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if resource is not None
        else None
    )
    elapsed = time.monotonic() - started
    outputs = _output_evidence(output_root)
    if return_code == 0 and not all(
        row["present"] for row in outputs.values()
    ):
        return_code = 126
        status = "EXPORT_OUTPUT_INCOMPLETE_NO_AUTOMATIC_SUCCESSOR"
    receipt = {
        "schema": SCHEMA,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed if math.isfinite(elapsed) else None,
        "return_code": return_code,
        "terminal_status": status,
        "error": error,
        "bindings": bindings,
        "command": command,
        "export_log": (
            {
                "path": str(log_path),
                "bytes": log_path.stat().st_size,
                "sha256": sha256_file(log_path),
            }
            if log_path.is_file()
            else {"path": str(log_path), "present": False}
        ),
        "maximum_child_resident_set_kib": maximum_rss_kib,
        "child_pid": process.pid if process is not None else None,
        "child_reaped": process is None or process.poll() is not None,
        "reaping": reaping,
        "outputs": outputs,
        "automatic_successor_launched": False,
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--expected-exporter-sha256", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--expected-runtime-receipt-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--audit-path", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--audit-seal-path", type=Path, required=True)
    parser.add_argument("--expected-audit-seal-sha256", required=True)
    parser.add_argument("--audit-producer-path", type=Path, required=True)
    parser.add_argument("--expected-audit-producer-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--export-log", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--requested-threads", type=int, required=True)
    parser.add_argument(
        "--policy-total-physical-cores", type=int, required=True
    )
    parser.add_argument("--policy-limit-cores", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
