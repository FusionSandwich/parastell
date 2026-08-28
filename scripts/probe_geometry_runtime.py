"""Create a hash-bound geometry-runtime receipt without meshing or transport."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import parastell.source_cad_refaceting as refaceting_module
from parastell.reference_geometry import sha256_file
from parastell.source_cad_refaceting import REQUIRED_RUNTIME_MODULES
from parastell.source_cad_refaceting import RUNTIME_RECEIPT_SCHEMA
from parastell.source_cad_refaceting import runtime_module_receipt


SCHEMA = RUNTIME_RECEIPT_SCHEMA
REQUIRED_MODULES = REQUIRED_RUNTIME_MODULES


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _file_receipt(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _snapshot_controls(
    controls: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    rows = {}
    for name in sorted(controls):
        row = controls[name]
        path = Path(row["path"]).resolve()
        expected = row["expected_sha256"]
        actual = _file_receipt(path)
        if not _valid_sha256(expected) or actual["sha256"] != expected:
            raise ValueError(f"runtime control hash mismatch: {name}")
        rows[name] = {**actual, "expected_sha256": expected}
    return rows


def _parse_control(value: str) -> tuple[str, dict[str, str]]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not parts[0]:
        raise argparse.ArgumentTypeError(
            "control must be NAME=ABSOLUTE_PATH=EXPECTED_SHA256"
        )
    name, path, expected = parts
    if not Path(path).is_absolute() or not _valid_sha256(expected):
        raise argparse.ArgumentTypeError(
            "control path must be absolute and SHA-256 must be valid"
        )
    return name, {"path": path, "expected_sha256": expected}


def _publish_json_create_only(
    destination: Path, payload: Mapping[str, Any]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.pending"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if json.loads(temporary.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("runtime receipt readback failed")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if json.loads(destination.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("published runtime receipt readback failed")


def probe_geometry_runtime(
    *,
    h5m_path: str | Path,
    expected_h5m_sha256: str,
    output_path: str | Path,
    attempt_id: str,
    launch_nonce: str,
    host_alias: str,
    expected_hostname: str,
    expected_machine: str,
    expected_python_executable: str | Path,
    expected_python_sha256: str,
    attempt_root: str | Path,
    refacet_output_dir: str | Path,
    faceting_level: str,
    threads: int,
    refacet_module_path: str | Path,
    refacet_cli_path: str | Path,
    controls: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    from pymoab import core, types

    source = Path(h5m_path).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if (
        not attempt_id
        or any(character in attempt_id for character in "/\\")
        or len(launch_nonce) < 32
        or any(
            character not in "0123456789abcdef" for character in launch_nonce
        )
    ):
        raise ValueError("attempt ID or high-entropy launch nonce is invalid")
    if faceting_level not in {"coarse", "refined"}:
        raise ValueError("faceting level must be coarse or refined")
    if threads < 1 or threads > 64:
        raise ValueError("threads must be between 1 and 64")
    intended_root = Path(attempt_root).resolve()
    intended_output = Path(refacet_output_dir).resolve()
    if intended_output.parent != intended_root:
        raise ValueError("refaceting output must be a child of attempt root")
    if intended_root.exists() or intended_output.exists():
        raise FileExistsError("intended refaceting root/output must be absent")
    executable = Path(sys.executable).resolve()
    expected_executable = Path(expected_python_executable).resolve()
    python_sha256 = sha256_file(executable)
    if (
        executable != expected_executable
        or python_sha256 != expected_python_sha256
    ):
        raise ValueError(
            "active Python does not match the expected executable"
        )
    actual_hostname = platform.node().split(".", 1)[0]
    actual_machine = platform.machine()
    if (
        actual_hostname != expected_hostname
        or actual_machine != expected_machine
    ):
        raise ValueError(
            "active host does not match the expected Bateman host"
        )
    before = sha256_file(source)
    if before != expected_h5m_sha256:
        raise ValueError("runtime-probe H5M hash mismatch")
    implementation_before = {
        "probe_script": _file_receipt(Path(__file__)),
        "refacet_module": _file_receipt(refacet_module_path),
        "refacet_cli": _file_receipt(refacet_cli_path),
    }
    if (
        Path(refacet_module_path).resolve()
        != Path(refaceting_module.__file__).resolve()
    ):
        raise ValueError("refacet-module path differs from imported module")
    controls_before = _snapshot_controls(controls)
    modules = [runtime_module_receipt(name) for name in REQUIRED_MODULES]
    mesh = core.Core()
    mesh.load_file(str(source))
    root = mesh.get_root_set()
    entity_set_count = len(mesh.get_entities_by_type(root, types.MBENTITYSET))
    after = sha256_file(source)
    implementation_after = {
        "probe_script": _file_receipt(Path(__file__)),
        "refacet_module": _file_receipt(refacet_module_path),
        "refacet_cli": _file_receipt(refacet_cli_path),
    }
    controls_after = _snapshot_controls(controls)
    module_names = [row["module"] for row in modules]
    modules_complete = bool(
        module_names == list(REQUIRED_MODULES)
        and len(module_names) == len(set(module_names))
        and all(row["import_pass"] is True for row in modules)
        and all(Path(row["path"]).is_absolute() for row in modules)
        and all(_valid_sha256(row["file_sha256"]) for row in modules)
        and all(
            row["module_version"] or row["distribution_version"]
            for row in modules
        )
        and all(
            Path(artifact["path"]).is_absolute()
            and _valid_sha256(artifact["file_sha256"])
            for row in modules
            for artifact in row["native_artifacts"]
        )
    )
    receipt = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "scientific_decision": "NOT_PERFORMED_RUNTIME_PREFLIGHT_ONLY",
        "run_binding": {
            "attempt_id": attempt_id,
            "launch_nonce": launch_nonce,
            "host_alias": host_alias,
            "hostname": actual_hostname,
            "machine": actual_machine,
            "attempt_root": str(intended_root),
            "refacet_output_dir": str(intended_output),
            "faceting_level": faceting_level,
            "threads": threads,
        },
        "python": {
            "requested_executable": str(Path(expected_python_executable)),
            "resolved_executable": str(executable),
            "executable_sha256": python_sha256,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "modules": modules,
        "required_modules": list(REQUIRED_MODULES),
        "all_required_imports_pass": modules_complete,
        "source_h5m": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256_before": before,
            "sha256_after": after,
            "unchanged": before == after == expected_h5m_sha256,
            "native_entity_set_count": entity_set_count,
            "native_reload_pass": entity_set_count > 0,
        },
        "implementation_before": implementation_before,
        "implementation_after": implementation_after,
        "implementation_unchanged": implementation_before
        == implementation_after,
        "control_inputs_before": controls_before,
        "control_inputs_after": controls_after,
        "control_inputs_unchanged": controls_before == controls_after,
    }
    receipt["runtime_gate_pass"] = bool(
        receipt["all_required_imports_pass"]
        and receipt["source_h5m"]["unchanged"]
        and receipt["source_h5m"]["native_reload_pass"]
        and receipt["implementation_unchanged"]
        and receipt["control_inputs_unchanged"]
    )
    if not receipt["runtime_gate_pass"]:
        receipt["status"] = "BLOCKED"
    _publish_json_create_only(destination, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5m", required=True)
    parser.add_argument("--expected-h5m-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--expected-machine", required=True)
    parser.add_argument("--expected-python-executable", required=True)
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--refacet-output-dir", required=True)
    parser.add_argument(
        "--faceting-level", choices=("coarse", "refined"), required=True
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--refacet-module", required=True)
    parser.add_argument("--refacet-cli", required=True)
    parser.add_argument(
        "--control", action="append", default=[], type=_parse_control
    )
    arguments = parser.parse_args()
    controls = dict(arguments.control)
    if len(controls) != len(arguments.control):
        raise SystemExit("duplicate runtime control name")
    receipt = probe_geometry_runtime(
        h5m_path=arguments.h5m,
        expected_h5m_sha256=arguments.expected_h5m_sha256,
        output_path=arguments.output,
        attempt_id=arguments.attempt_id,
        launch_nonce=arguments.launch_nonce,
        host_alias=arguments.host_alias,
        expected_hostname=arguments.expected_hostname,
        expected_machine=arguments.expected_machine,
        expected_python_executable=arguments.expected_python_executable,
        expected_python_sha256=arguments.expected_python_sha256,
        attempt_root=arguments.attempt_root,
        refacet_output_dir=arguments.refacet_output_dir,
        faceting_level=arguments.faceting_level,
        threads=arguments.threads,
        refacet_module_path=arguments.refacet_module,
        refacet_cli_path=arguments.refacet_cli,
        controls=controls,
    )
    if not receipt["runtime_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
