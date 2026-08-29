"""Run bounded native DAGMC checks with explicit exit-code receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

from parastell.dagmc_qualification import parse_check_watertight_log
from parastell.dagmc_qualification import parse_overlap_check_log
from parastell.reference_geometry import sha256_file
from parastell.reference_geometry import native_dagmc_id_inventory


def _run(command: list[str], *, timeout_seconds: int) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=int(timeout_seconds),
        )
        return {
            "command": command,
            "started_at": started,
            "wall_time_seconds": time.monotonic() - clock,
            "exit_code": int(result.returncode),
            "timed_out": False,
            "output": result.stdout,
        }
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {
            "command": command,
            "started_at": started,
            "wall_time_seconds": time.monotonic() - clock,
            "exit_code": None,
            "timed_out": True,
            "output": output,
        }


def _supports_short_option(executable: str, option: str) -> bool:
    """Return whether the installed executable advertises a short option."""

    result = subprocess.run(
        [executable, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=30,
    )
    for line in result.stdout.splitlines():
        tokens = line.replace(",", " ").split()
        if option in tokens:
            return True
    return False


def _overlap_command(
    executable: str,
    *,
    precision: int,
    threads: int,
    dagmc_path: Path,
    thread_option_supported: bool,
) -> list[str]:
    command = [executable, "-p", str(int(precision))]
    if thread_option_supported:
        command.extend(["-t", str(int(threads))])
    command.append(str(dagmc_path))
    return command


def run(
    dagmc_path: Path,
    output_dir: Path,
    *,
    expected_dagmc_sha256: str,
    acceptance_criteria: Path,
    expected_acceptance_criteria_sha256: str,
    threads: int = 4,
    timeout_seconds: int = 1800,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"create-only output exists: {output_dir}")
    actual_h5m = sha256_file(dagmc_path)
    if actual_h5m != expected_dagmc_sha256:
        raise ValueError("DAGMC H5M hash mismatch")
    native_ids = native_dagmc_id_inventory(dagmc_path)
    if not native_ids["native_id_gate_pass"]:
        raise ValueError("native DAGMC entity-ID inventory is invalid")
    actual_criteria = sha256_file(acceptance_criteria)
    if actual_criteria != expected_acceptance_criteria_sha256:
        raise ValueError("acceptance-criteria hash mismatch")
    criteria = json.loads(acceptance_criteria.read_text(encoding="utf-8"))
    precision_levels = [
        int(value) for value in criteria["dagmc"]["overlap_checks"]
    ]
    if precision_levels != [1, 2, 4]:
        raise ValueError(
            "native overlap precisions are not preregistered [1,2,4]"
        )
    executables = {}
    for executable in ("check_watertight", "overlap_check"):
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(
                f"required native executable is unavailable: {executable}"
            )
        executables[executable] = resolved
    overlap_thread_option_supported = _supports_short_option(
        executables["overlap_check"], "-t"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    derived = output_dir / "watertight_checked.NONAUTHORITATIVE.h5m"
    watertight_run = _run(
        [
            executables["check_watertight"],
            "-v",
            "-o",
            str(derived),
            str(dagmc_path),
        ],
        timeout_seconds=timeout_seconds,
    )
    (output_dir / "check_watertight.log").write_text(
        watertight_run["output"], encoding="utf-8"
    )
    watertight = parse_check_watertight_log(
        watertight_run["output"],
        exit_code=(
            watertight_run["exit_code"]
            if watertight_run["exit_code"] is not None
            else -1
        ),
        expected_surface_count=native_ids["surface_entity_count"],
        expected_volume_count=native_ids["volume_entity_count"],
    )
    watertight["timed_out"] = watertight_run["timed_out"]
    watertight["command"] = watertight_run["command"]
    watertight["wall_time_seconds"] = watertight_run["wall_time_seconds"]

    overlap_results = []
    for precision in precision_levels:
        native = _run(
            _overlap_command(
                executables["overlap_check"],
                precision=precision,
                threads=threads,
                dagmc_path=dagmc_path,
                thread_option_supported=overlap_thread_option_supported,
            ),
            timeout_seconds=timeout_seconds,
        )
        (output_dir / f"overlap_check_p{precision}_t{threads}.log").write_text(
            native["output"], encoding="utf-8"
        )
        parsed = parse_overlap_check_log(
            native["output"],
            exit_code=(
                native["exit_code"] if native["exit_code"] is not None else -1
            ),
            points_per_edge=precision,
            threads=threads,
        )
        parsed["timed_out"] = native["timed_out"]
        parsed["command"] = native["command"]
        parsed["wall_time_seconds"] = native["wall_time_seconds"]
        parsed["native_thread_option_supported"] = (
            overlap_thread_option_supported
        )
        parsed["native_threads_argument_used"] = (
            overlap_thread_option_supported
        )
        parsed["requested_threads"] = int(threads)
        overlap_results.append(parsed)

    after = sha256_file(dagmc_path)
    report = {
        "schema": "parastell.dagmc_native_qualification/v1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dagmc_path": str(dagmc_path),
        "raw_h5m_sha256_before": actual_h5m,
        "raw_h5m_sha256_after": after,
        "h5m_unchanged": after == actual_h5m,
        "acceptance_criteria_path": str(acceptance_criteria),
        "acceptance_criteria_sha256": actual_criteria,
        "native_id_inventory": native_ids,
        "threads": int(threads),
        "native_executables": executables,
        "overlap_thread_option_supported": overlap_thread_option_supported,
        "per_command_timeout_seconds": int(timeout_seconds),
        "check_watertight": watertight,
        "overlap_checks": overlap_results,
        "derived_watertight_h5m": {
            "path": str(derived),
            "authoritative": False,
            "exists": derived.is_file(),
            "sha256": sha256_file(derived) if derived.is_file() else None,
        },
    }
    report["native_dagmc_gate_pass"] = bool(
        report["h5m_unchanged"]
        and watertight["pass"]
        and all(row["pass"] for row in overlap_results)
    )
    (output_dir / "native_dagmc_qualification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-dagmc-sha256", required=True)
    parser.add_argument("--acceptance-criteria", type=Path, required=True)
    parser.add_argument("--expected-acceptance-criteria-sha256", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    arguments = parser.parse_args()
    run(
        arguments.dagmc.resolve(),
        arguments.output.resolve(),
        expected_dagmc_sha256=arguments.expected_dagmc_sha256,
        acceptance_criteria=arguments.acceptance_criteria.resolve(),
        expected_acceptance_criteria_sha256=(
            arguments.expected_acceptance_criteria_sha256
        ),
        threads=arguments.threads,
        timeout_seconds=arguments.timeout_seconds,
    )


if __name__ == "__main__":
    main()
