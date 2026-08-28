"""Run one bounded direct-90 exporter and always persist terminal evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - available on the Bateman Linux host
    resource = None


SCHEMA = "wistell_d.direct90_export_terminal/v1.1.0"
WALLTIME_SECONDS = 21_600
TOTAL_ATTEMPT_SECONDS = 22_000
TERMINATION_GRACE_SECONDS = 30
REQUESTED_THREADS = 32
POLICY_TOTAL_THREADS = 256
POLICY_LIMIT_THREADS = 64
EXPECTED_LEASE_ROOT = Path("/home/apollon/josma/.codex/ssh-poly-core-budget")
SIGTERM = getattr(signal, "SIGTERM", 15)
SIGKILL = getattr(signal, "SIGKILL", 9)
SIGALRM = getattr(signal, "SIGALRM", None)


def _total_attempt_timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError("total direct-90 attempt walltime exceeded")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _proc_start_time(proc_root: Path, pid: int) -> int | None:
    try:
        text = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        remainder = text[text.rindex(")") + 2 :].split()
        return int(remainder[19])
    except (OSError, ValueError, IndexError):
        return None


def _discover_lease(
    root: Path | None = None,
    *,
    guard_pid: int | None = None,
    session_id: int | None = None,
    proc_root: Path = Path("/proc"),
    timeout_seconds: float = 5.0,
    poll_seconds: float = 0.05,
    maximum_polls: int = 101,
) -> dict[str, Any]:
    lease_root = root or (Path.home() / ".codex" / "ssh-poly-core-budget")
    guard = os.getppid() if guard_pid is None else int(guard_pid)
    sid = os.getsid(guard) if session_id is None else int(session_id)
    deadline = time.monotonic() + timeout_seconds
    for poll_index in range(maximum_polls):
        matches = []
        for path in lease_root.glob("*.lease"):
            fields = path.read_text(encoding="utf-8").split()
            if len(fields) != 4 or not all(
                field.isdigit() for field in fields
            ):
                continue
            cores, lease_pid, start_time, lease_sid = map(int, fields)
            if (
                cores == REQUESTED_THREADS
                and lease_pid == guard
                and lease_sid == sid
                and _proc_start_time(proc_root, lease_pid) == start_time
            ):
                matches.append(
                    {
                        "lease_id": path.stem,
                        "cores": cores,
                        "pid": lease_pid,
                        "start_time": start_time,
                        "session_id": lease_sid,
                    }
                )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"multiple live ssh-poly leases match session {sid}: {matches}"
            )
        if time.monotonic() >= deadline or poll_index + 1 >= maximum_polls:
            raise RuntimeError(
                f"timed out waiting for one live {REQUESTED_THREADS}-core "
                f"ssh-poly lease for guard PID {guard}, session {sid}"
            )
        time.sleep(poll_seconds)
    raise AssertionError("unreachable lease-discovery loop exit")


def _load_exporter(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "frozen_direct90_exporter", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _terminate(process: subprocess.Popen[Any]) -> dict[str, Any]:
    errors = []
    actions = []
    try:
        os.killpg(process.pid, SIGTERM)
        actions.append("SIGTERM")
    except OSError as exc:
        errors.append(f"SIGTERM {type(exc).__name__}: {exc}")
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, SIGKILL)
            actions.append("SIGKILL")
        except OSError as exc:
            errors.append(f"SIGKILL {type(exc).__name__}: {exc}")
        try:
            process.wait(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            errors.append(f"final wait TimeoutExpired: {exc}")
        except OSError as exc:
            errors.append(f"final wait {type(exc).__name__}: {exc}")
    except OSError as exc:
        errors.append(f"wait {type(exc).__name__}: {exc}")
    try:
        child_reaped = process.poll() is not None
    except OSError as exc:
        errors.append(f"poll {type(exc).__name__}: {exc}")
        child_reaped = False
    return {
        "actions": actions,
        "errors": errors,
        "child_reaped": child_reaped,
    }


def _child_reaped(process: subprocess.Popen[Any] | None) -> bool:
    if process is None:
        return True
    try:
        return process.poll() is not None
    except OSError:
        return False


def _output_evidence(output_root: Path) -> dict[str, Any]:
    outputs = {}
    for name in ("dagmc.h5m", "DAGMC_EXPORT_RECEIPT.json"):
        path = output_root / name
        outputs[name] = (
            {
                "present": True,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.is_file()
            else {"present": False}
        )
    return outputs


def run(args: argparse.Namespace) -> int:
    exporter = args.exporter.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    receipt_path = args.terminal_receipt.resolve()
    runtime_receipt = args.runtime_receipt.resolve()
    if receipt_path.exists():
        raise FileExistsError(
            f"terminal receipt is create-only: {receipt_path}"
        )
    if not receipt_path.parent.is_dir():
        raise FileNotFoundError(receipt_path.parent)

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process = None
    return_code = 125
    terminal_status = "WRAPPER_PRELAUNCH_FAILURE"
    error = None
    reaping = None
    bindings: dict[str, Any] = {}
    command: list[str] = []
    previous_alarm_handler = None
    if SIGALRM is not None:
        previous_alarm_handler = signal.signal(SIGALRM, _total_attempt_timeout)
        signal.alarm(TOTAL_ATTEMPT_SECONDS)
    try:
        wrapper = Path(__file__).resolve()
        if sha256_file(wrapper) != args.expected_wrapper_sha256:
            raise ValueError("wrapper hash mismatch")
        if sha256_file(exporter) != args.expected_exporter_sha256:
            raise ValueError("exporter hash mismatch")
        if (
            sha256_file(runtime_receipt)
            != args.expected_runtime_receipt_sha256
        ):
            raise ValueError("runtime receipt hash mismatch")
        if (
            args.requested_threads != REQUESTED_THREADS
            or args.policy_total_threads != POLICY_TOTAL_THREADS
            or args.policy_limit_threads != POLICY_LIMIT_THREADS
        ):
            raise ValueError("ssh-poly policy binding mismatch")
        if args.lease_id != "AUTO_DISCOVER":
            raise ValueError("explicit ssh-poly lease IDs are forbidden")
        lease_root = args.lease_root.resolve()
        if lease_root != EXPECTED_LEASE_ROOT:
            raise ValueError("unexpected ssh-poly lease root")
        lease = _discover_lease(lease_root)
        module = _load_exporter(exporter)
        manifest = source_root / "manifest.json"
        if sha256_file(manifest) != module.REFERENCE_MANIFEST_SHA256:
            raise ValueError("source manifest hash mismatch")
        source_steps = {
            name: source_root / f"{name}.step"
            for name in module.COMPONENT_ORDER
        }
        step_evidence = module._validated_step_artifacts(source_steps)
        environment = {
            name: os.environ.get(name)
            for name in module.REQUIRED_THREAD_ENVIRONMENT
        }
        if environment != module.REQUIRED_THREAD_ENVIRONMENT:
            raise ValueError("frozen thread environment mismatch")
        if not args.attempt_id or not args.nonce:
            raise ValueError(
                "attempt, nonce, and lease identities are required"
            )
        bindings = {
            "attempt_id": args.attempt_id,
            "nonce": args.nonce,
            "host": socket.gethostname(),
            "lease": lease,
            "lease_root": str(lease_root),
            "lease_discovery": "matched_live_current_session_lease",
            "policy_total_threads": args.policy_total_threads,
            "policy_limit_threads": args.policy_limit_threads,
            "requested_threads": args.requested_threads,
            "wrapper_sha256": args.expected_wrapper_sha256,
            "exporter_sha256": args.expected_exporter_sha256,
            "runtime_receipt": str(runtime_receipt),
            "runtime_receipt_sha256": args.expected_runtime_receipt_sha256,
            "python_executable": sys.executable,
            "python_executable_sha256": sha256_file(Path(sys.executable)),
            "python_version": platform.python_version(),
            "source_manifest_sha256": module.REFERENCE_MANIFEST_SHA256,
            "source_steps": step_evidence,
            "thread_environment": environment,
            "walltime_seconds": WALLTIME_SECONDS,
            "total_attempt_walltime_seconds": TOTAL_ATTEMPT_SECONDS,
            "termination_grace_seconds": TERMINATION_GRACE_SECONDS,
        }
        command = [
            sys.executable,
            str(exporter),
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--min-mesh-size-cm",
            "5.0",
            "--max-mesh-size-cm",
            "20.0",
            "--algorithm",
            "1",
            "--threads",
            "32",
        ]
        process = subprocess.Popen(command, start_new_session=True)
        try:
            return_code = int(process.wait(timeout=WALLTIME_SECONDS))
            terminal_status = (
                "EXPORT_COMPLETED_NATIVE_GATES_PENDING"
                if return_code == 0
                else "EXPORT_FAILED_NO_AUTOMATIC_SUCCESSOR"
            )
        except subprocess.TimeoutExpired:
            reaping = _terminate(process)
            return_code = 124
            terminal_status = (
                "EXPORT_TIMEOUT_REAPED_NO_AUTOMATIC_SUCCESSOR"
                if reaping["child_reaped"]
                else "EXPORT_TIMEOUT_REAP_INCOMPLETE_EXTERNAL_CLEANUP_REQUIRED"
            )
    except Exception as exc:  # preserve an honest wrapper/launch failure
        if not _child_reaped(process):
            reaping = _terminate(process)
        if isinstance(exc, TimeoutError):
            return_code = 124
            terminal_status = (
                "TOTAL_ATTEMPT_TIMEOUT_REAPED_NO_AUTOMATIC_SUCCESSOR"
                if _child_reaped(process)
                else "TOTAL_ATTEMPT_TIMEOUT_REAP_INCOMPLETE"
            )
        error = f"{type(exc).__name__}: {exc}"

    if SIGALRM is not None:
        signal.alarm(0)
        if previous_alarm_handler is not None:
            signal.signal(SIGALRM, previous_alarm_handler)

    maximum_rss_kib = (
        int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if resource is not None
        else None
    )
    elapsed = time.monotonic() - started
    receipt = {
        "schema": SCHEMA,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed if math.isfinite(elapsed) else None,
        "return_code": return_code,
        "terminal_status": terminal_status,
        "error": error,
        "bindings": bindings,
        "command": command,
        "maximum_child_resident_set_kib": maximum_rss_kib,
        "child_pid": process.pid if process is not None else None,
        "child_reaped": _child_reaped(process),
        "reaping": reaping,
        "outputs": _output_evidence(output_root),
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
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--expected-exporter-sha256", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--expected-runtime-receipt-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--requested-threads", type=int, required=True)
    parser.add_argument("--policy-total-threads", type=int, required=True)
    parser.add_argument("--policy-limit-threads", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
