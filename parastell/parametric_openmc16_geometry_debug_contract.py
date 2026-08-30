"""Standard-library contract for split-runtime OpenMC geometry debug.

This module is safe to import in the native OpenMC runtime: it imports no
OpenMC, NumPy, MOAB, DAGMC, CAD, or meshing package.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import shlex
from typing import Any

from .openmc_geometry_debug import parse_openmc_geometry_debug_log


MODEL_RECEIPT_SCHEMA = "parastell.parametric_openmc16_sealed_model/v1.0.0"
RUN_RECEIPT_SCHEMA = "parastell.parametric_openmc16_geometry_debug_run/v2.0.0"
RUN_RECEIPT_FILENAME = "PARAMETRIC_OPENMC16_GEOMETRY_DEBUG.json"
LOG_FILENAME = "openmc_geometry_debug.log"
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def bound_file(
    row: Mapping[str, Any], label: str, *, require_absolute: bool = True
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(row, Mapping) or set(row) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError(f"{label} binding is malformed")
    declared = Path(str(row["path"]))
    if require_absolute and not declared.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = declared.resolve(strict=True)
    expected_hash = str(row["sha256"])
    if (
        not path.is_file()
        or not valid_sha256(expected_hash)
        or sha256_file(path) != expected_hash
        or path.stat().st_size != int(row["size_bytes"])
    ):
        raise ValueError(f"{label} binding does not match the file")
    evidence = {
        "path": str(path),
        "sha256": expected_hash,
        "size_bytes": path.stat().st_size,
    }
    return path, evidence


def load_sealed_model_receipt(
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    model_xml: str | Path,
) -> tuple[dict[str, Any], list[int], dict[str, Path]]:
    """Validate the stage-2 receipt and derive native coverage IDs."""
    path = Path(receipt_path).resolve(strict=True)
    if (
        not valid_sha256(expected_receipt_sha256)
        or sha256_file(path) != expected_receipt_sha256
    ):
        raise ValueError("sealed model receipt hash mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise ValueError("sealed model receipt is not a JSON object")
    candidate = dict(receipt)
    claimed = candidate.pop("receipt_content_sha256", None)
    if (
        receipt.get("schema") != MODEL_RECEIPT_SCHEMA
        or receipt.get("status") != "MODEL_EXPORTED_TRANSPORT_PENDING"
        or receipt.get("claim") != "BOUNDED_SMOKE_ONLY"
        or not valid_sha256(claimed)
        or canonical_sha256(candidate) != claimed
        or receipt.get("openmc_runtime", {}).get("version") != "0.16.0"
        or receipt.get("photon_transport") is not True
        or receipt.get("all_bound_inputs_immutable") is not True
        or receipt.get("physical_h5m_mutation") is not False
        or receipt.get("source_mesh_mutation") is not False
        or receipt.get("transport_executed") is not False
        or receipt.get("production_run_authorized") is not False
    ):
        raise ValueError("sealed model receipt is not accepted")

    declared_model, model_binding = bound_file(
        receipt.get("model_xml", {}), "model XML"
    )
    supplied_model = Path(model_xml).resolve(strict=True)
    if declared_model != supplied_model:
        raise ValueError("sealed receipt references another model XML")

    bindings = receipt.get("validated_input_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("sealed model input bindings are missing")
    paths: dict[str, Path] = {"model_xml": declared_model}
    for label, row in sorted(bindings.items()):
        bound_path, _ = bound_file(row, f"sealed input {label}")
        paths[f"sealed_input:{label}"] = bound_path
    dagmc_row = bindings.get("dagmc_h5m")
    if not isinstance(dagmc_row, Mapping):
        raise ValueError("sealed DAGMC binding is missing")

    geometry = receipt.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("sealed native DAGMC inventory is missing")
    try:
        volume_ids = [int(value) for value in geometry.get("volume_ids", ())]
    except (TypeError, ValueError) as error:
        raise ValueError("sealed native volume IDs are invalid") from error
    if (
        not volume_ids
        or volume_ids != sorted(set(volume_ids))
        or any(value <= 0 for value in volume_ids)
        or int(geometry.get("volume_count", -1)) != len(volume_ids)
        or geometry.get("dagmc_sha256") != dagmc_row.get("sha256")
    ):
        raise ValueError("sealed native volume inventory is invalid")
    magnets = receipt.get("magnet_cell_ids")
    if (
        not isinstance(magnets, Mapping)
        or not magnets
        or len({int(value) for value in magnets.values()}) != len(magnets)
        or not {int(value) for value in magnets.values()}.issubset(
            set(volume_ids)
        )
    ):
        raise ValueError("sealed magnet IDs are not native volumes")
    paths["model_receipt"] = path
    if model_binding["sha256"] != sha256_file(supplied_model):
        raise ValueError("sealed model XML changed during preflight")
    return dict(receipt), volume_ids, paths


def bind_openmc_executable(
    executable: str | Path, expected_sha256: str
) -> tuple[Path, dict[str, Any]]:
    declared = Path(executable)
    if not declared.is_absolute():
        raise ValueError("OpenMC executable path must be absolute")
    resolved = declared.resolve(strict=True)
    return bound_file(
        {
            "path": str(resolved),
            "sha256": expected_sha256,
            "size_bytes": resolved.stat().st_size,
        },
        "OpenMC executable",
    )


def load_runtime_environment(
    runtime_env: str | Path, expected_sha256: str
) -> tuple[dict[str, str], dict[str, Any]]:
    declared = Path(runtime_env)
    if not declared.is_absolute():
        raise ValueError("runtime.env path must be absolute")
    path = declared.resolve(strict=True)
    _, binding = bound_file(
        {
            "path": str(path),
            "sha256": expected_sha256,
            "size_bytes": path.stat().st_size,
        },
        "runtime.env",
    )
    environment: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        raw_value = raw_value.strip()
        if (
            separator != "="
            or not _ENVIRONMENT_NAME.fullmatch(name)
            or name in environment
            or "\x00" in raw_value
            or "$" in raw_value
            or "`" in raw_value
        ):
            raise ValueError(
                f"runtime.env line {line_number} is not materialized safely"
            )
        if raw_value.startswith(("'", '"')):
            try:
                values = shlex.split(raw_value, posix=True)
            except ValueError as error:
                raise ValueError(
                    f"runtime.env line {line_number} has invalid quoting"
                ) from error
            if len(values) != 1:
                raise ValueError(
                    f"runtime.env line {line_number} is ambiguous"
                )
            value = values[0]
        else:
            if any(character.isspace() for character in raw_value):
                raise ValueError(
                    f"runtime.env line {line_number} has unquoted whitespace"
                )
            value = raw_value
        environment[name] = value
    if not environment:
        raise ValueError("runtime.env contains no materialized variables")
    return environment, binding


__all__ = [
    "LOG_FILENAME",
    "MODEL_RECEIPT_SCHEMA",
    "RUN_RECEIPT_FILENAME",
    "RUN_RECEIPT_SCHEMA",
    "bind_openmc_executable",
    "canonical_sha256",
    "load_runtime_environment",
    "load_sealed_model_receipt",
    "parse_openmc_geometry_debug_log",
    "sha256_file",
    "valid_sha256",
]
