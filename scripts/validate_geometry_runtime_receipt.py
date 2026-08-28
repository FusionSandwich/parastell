"""Revalidate a geometry-runtime receipt without meshing or transport."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from parastell.reference_geometry import sha256_file
from parastell.source_cad_refaceting import _validate_runtime_receipt


SCHEMA = "parastell.geometry_runtime_validation/v1.0.0"


def _valid_sha256(value: str) -> bool:
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _parse_control(value: str) -> tuple[str, str]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0] or not _valid_sha256(parts[1]):
        raise argparse.ArgumentTypeError("control must be NAME=SHA256")
    return parts[0], parts[1]


def _publish_create_only(destination: Path, payload: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.pending"
    )
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if json.loads(destination.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("runtime-validation publication mismatch")


def validate_runtime(
    *,
    receipt_path: Path,
    expected_receipt_sha256: str,
    expected_attempt_id: str,
    expected_launch_nonce: str,
    expected_host_alias: str,
    expected_output_dir: Path,
    expected_level: str,
    expected_threads: int,
    source_h5m: Path,
    expected_source_h5m_sha256: str,
    controls: dict[str, str],
    output_path: Path,
) -> dict:
    result = _validate_runtime_receipt(
        receipt_path,
        expected_sha256=expected_receipt_sha256,
        expected_attempt_id=expected_attempt_id,
        expected_launch_nonce=expected_launch_nonce,
        expected_host_alias=expected_host_alias,
        expected_output_dir=expected_output_dir,
        expected_level=expected_level,
        expected_threads=expected_threads,
        expected_source_h5m_path=source_h5m,
        expected_source_h5m_sha256=expected_source_h5m_sha256,
        expected_controls=controls,
    )
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_decision": "NOT_PERFORMED_RUNTIME_VALIDATION_ONLY",
        "runtime_receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": expected_receipt_sha256,
        },
        "validator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "gates": result["gates"],
        "active_python": result["active_python"],
        "active_modules": result["active_modules"],
        "canonical_runtime_sha256": result["canonical_runtime_sha256"],
        "runtime_validation_pass": result["pass"] is True,
    }
    _publish_create_only(output_path.resolve(), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-attempt-id", required=True)
    parser.add_argument("--expected-launch-nonce", required=True)
    parser.add_argument("--expected-host-alias", required=True)
    parser.add_argument("--expected-output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-level", choices=("coarse", "refined"), required=True
    )
    parser.add_argument("--expected-threads", type=int, required=True)
    parser.add_argument("--source-h5m", type=Path, required=True)
    parser.add_argument("--expected-source-h5m-sha256", required=True)
    parser.add_argument(
        "--control", action="append", default=[], type=_parse_control
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    controls = dict(arguments.control)
    if len(controls) != len(arguments.control):
        raise SystemExit("duplicate runtime control name")
    validate_runtime(
        receipt_path=arguments.receipt,
        expected_receipt_sha256=arguments.expected_receipt_sha256,
        expected_attempt_id=arguments.expected_attempt_id,
        expected_launch_nonce=arguments.expected_launch_nonce,
        expected_host_alias=arguments.expected_host_alias,
        expected_output_dir=arguments.expected_output_dir,
        expected_level=arguments.expected_level,
        expected_threads=arguments.expected_threads,
        source_h5m=arguments.source_h5m,
        expected_source_h5m_sha256=arguments.expected_source_h5m_sha256,
        controls=controls,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
