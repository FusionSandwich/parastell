"""Run geometry-debug on an already exported parametric OpenMC 0.16 model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
from xml.etree import ElementTree

from parastell.openmc_geometry_debug import parse_openmc_geometry_debug_log
from parastell.parametric_openmc16_model import RECEIPT_SCHEMA
from parastell.parametric_openmc16_model import _canonical_sha


SCHEMA = "parastell.parametric_openmc16_geometry_debug_run/v1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound_receipt(
    path: Path,
    expected_sha256: str,
    *,
    model_xml: Path,
    dagmc_h5m: Path,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("parametric model receipt hash mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    content_hash = receipt.pop("receipt_content_sha256", None)
    content_pass = content_hash == _canonical_sha(receipt)
    receipt["receipt_content_sha256"] = content_hash
    expected_magnets = {
        f"magnet-{index:04d}": index + 9 for index in range(18)
    }
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "MODEL_EXPORTED_TRANSPORT_PENDING"
        or receipt.get("claim") != "BOUNDED_SMOKE_ONLY"
        or receipt.get("openmc_version") != "0.16.0"
        or receipt.get("model_xml", {}).get("sha256") != sha256_file(model_xml)
        or receipt.get("dagmc_sha256") != sha256_file(dagmc_h5m)
        or receipt.get("modeled_extent_degrees") != 90.0
        or receipt.get("n_field_periods") != 4
        or receipt.get("magnet_cell_ids") != expected_magnets
        or receipt.get("photon_transport") is not True
        or receipt.get("physical_h5m_mutation") is not False
        or receipt.get("all_bound_inputs_immutable") is not True
        or receipt.get("selected_nuclear_data_immutable") is not True
        or not content_pass
    ):
        raise ValueError("parametric model receipt is not accepted")
    return receipt


def _nuclear_data_hashes(receipt: dict[str, Any]) -> dict[str, str]:
    rows = receipt.get("nuclear_data_manifest", {}).get("libraries")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "model receipt has no selected nuclear-data inventory"
        )
    result = {}
    for row in rows:
        path = Path(row["path"]).resolve(strict=True)
        actual = sha256_file(path)
        if actual != row.get("sha256"):
            raise ValueError(f"selected nuclear-data hash mismatch: {path}")
        result[str(path)] = actual
    return result


def _runtime_inputs(
    model_xml: Path, dagmc_h5m: Path, receipt: dict[str, Any]
) -> dict[str, dict[str, str]]:
    root = ElementTree.parse(model_xml).getroot()
    dagmc_nodes = root.findall("./geometry/dagmc_universe")
    mesh_nodes = root.findall("./settings/mesh[@type='unstructured']/filename")
    cross_nodes = root.findall("./materials/cross_sections")
    if len(dagmc_nodes) != 1 or len(mesh_nodes) != 1 or len(cross_nodes) != 1:
        raise ValueError("model XML external-input inventory is ambiguous")

    def resolve_declared(value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = model_xml.parent / path
        return path.resolve(strict=True)

    declared_dagmc = resolve_declared(dagmc_nodes[0].attrib["filename"])
    source_mesh = resolve_declared(mesh_nodes[0].text or "")
    cross_sections = resolve_declared(cross_nodes[0].text or "")
    if declared_dagmc != dagmc_h5m.resolve(strict=True):
        raise ValueError("model XML does not reference the supplied H5M")
    result = {
        "dagmc_h5m": {
            "path": str(declared_dagmc),
            "sha256": sha256_file(declared_dagmc),
        },
        "source_mesh": {
            "path": str(source_mesh),
            "sha256": sha256_file(source_mesh),
        },
        "cross_sections_xml": {
            "path": str(cross_sections),
            "sha256": sha256_file(cross_sections),
        },
    }
    if (
        result["dagmc_h5m"]["sha256"] != receipt.get("dagmc_sha256")
        or result["source_mesh"]["sha256"] != receipt.get("source_mesh_sha256")
        or result["cross_sections_xml"]["sha256"]
        != receipt.get("cross_sections_xml_sha256")
    ):
        raise ValueError("model XML external-input hash binding failed")
    return result


def _run(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_xml", type=Path)
    parser.add_argument("model_receipt", type=Path)
    parser.add_argument("dagmc_h5m", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-model-receipt-sha256", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    if args.threads < 1 or args.timeout_seconds < 1:
        raise ValueError("threads and timeout must be positive")
    model_xml = args.model_xml.resolve(strict=True)
    model_receipt = args.model_receipt.resolve(strict=True)
    dagmc_h5m = args.dagmc_h5m.resolve(strict=True)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)

    receipt = _load_bound_receipt(
        model_receipt,
        args.expected_model_receipt_sha256,
        model_xml=model_xml,
        dagmc_h5m=dagmc_h5m,
    )
    before = {
        "model_xml": sha256_file(model_xml),
        "model_receipt": sha256_file(model_receipt),
        "runtime_inputs": _runtime_inputs(model_xml, dagmc_h5m, receipt),
        "nuclear_data": _nuclear_data_hashes(receipt),
    }
    run_model = output / "model.xml"
    shutil.copy2(model_xml, run_model)
    command = ["openmc", "-g", "-s", str(args.threads)]
    run = _run(command, cwd=output, timeout_seconds=args.timeout_seconds)
    log = output / "openmc_geometry_debug.log"
    log.write_text(run["output"], encoding="utf-8")
    qualification = parse_openmc_geometry_debug_log(
        run["output"],
        exit_code=run["exit_code"],
        expected_threads=args.threads,
        required_cell_ids=list(range(1, 27)),
    )
    after = {
        "model_xml": sha256_file(model_xml),
        "model_receipt": sha256_file(model_receipt),
        "runtime_inputs": _runtime_inputs(model_xml, dagmc_h5m, receipt),
        "nuclear_data": _nuclear_data_hashes(receipt),
    }
    immutable = (
        before == after and sha256_file(run_model) == before["model_xml"]
    )
    passed = bool(
        not run["timed_out"] and qualification["pass"] is True and immutable
    )
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "BLOCKED_GEOMETRY_DEBUG",
        "claim": "BOUNDED_NONPRODUCTION",
        "command": command,
        "threads": args.threads,
        "timeout_seconds": args.timeout_seconds,
        "timed_out": run["timed_out"],
        "exit_code": run["exit_code"],
        "wall_time_seconds": run["wall_time_seconds"],
        "input_model_receipt_sha256": before["model_receipt"],
        "model_xml_sha256": before["model_xml"],
        "runtime_inputs": before["runtime_inputs"],
        "selected_nuclear_data": before["nuclear_data"],
        "inputs_immutable": immutable,
        "openmc_log_sha256": sha256_file(log),
        "log_qualification": qualification,
        "production_run_authorized": False,
    }
    result_path = output / "PARAMETRIC_OPENMC16_GEOMETRY_DEBUG.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(result_path)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
