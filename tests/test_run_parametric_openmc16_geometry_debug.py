from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from parastell import parametric_openmc16_geometry_debug_contract as contract
from scripts import run_parametric_openmc16_geometry_debug as runner


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path):
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _packet(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model.xml"
    dagmc = tmp_path / "dagmc.h5m"
    source = tmp_path / "source_mesh.h5m"
    cross_sections = tmp_path / "cross_sections.xml"
    data = tmp_path / "Fe56.h5"
    executable = tmp_path / "openmc-0.16"
    runtime_env = tmp_path / "runtime.env"
    model.write_text("<model/>", encoding="utf-8")
    dagmc.write_bytes(b"dagmc")
    source.write_bytes(b"source")
    cross_sections.write_text("<cross_sections/>", encoding="utf-8")
    data.write_bytes(b"data")
    executable.write_bytes(b"openmc executable")
    runtime_env.write_text(
        "LD_LIBRARY_PATH=/qualified/lib\nOMP_PROC_BIND=true\n",
        encoding="utf-8",
    )
    bindings = {
        "dagmc_h5m": _binding(dagmc),
        "source_mesh_h5m": _binding(source),
        "cross_sections_xml": _binding(cross_sections),
        f"nuclear_data:{data.resolve()}": _binding(data),
    }
    receipt = {
        "schema": contract.MODEL_RECEIPT_SCHEMA,
        "status": "MODEL_EXPORTED_TRANSPORT_PENDING",
        "claim": "BOUNDED_SMOKE_ONLY",
        "openmc_runtime": {"version": "0.16.0"},
        "geometry": {
            "dagmc_sha256": _sha(dagmc),
            "volume_ids": [2, 5, 9],
            "volume_count": 3,
        },
        "magnet_cell_ids": {"magnet-alpha": 9},
        "validated_input_bindings": bindings,
        "model_xml": _binding(model),
        "photon_transport": True,
        "all_bound_inputs_immutable": True,
        "physical_h5m_mutation": False,
        "source_mesh_mutation": False,
        "transport_executed": False,
        "production_run_authorized": False,
    }
    receipt["receipt_content_sha256"] = contract.canonical_sha256(receipt)
    receipt_path = tmp_path / "model-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return {
        "model": model,
        "dagmc": dagmc,
        "source": source,
        "cross_sections": cross_sections,
        "data": data,
        "executable": executable,
        "runtime_env": runtime_env,
        "receipt": receipt_path,
    }


def _execute(packet, output, monkeypatch, *, run=None, parser=None, **changes):
    seen = {}

    def default_run(command, *, cwd, timeout_seconds, environment):
        seen.update(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            environment=dict(environment),
        )
        return {
            "exit_code": 0,
            "timed_out": False,
            "output": "geometry debug log",
            "wall_time_seconds": 1.25,
        }

    def default_parser(
        text, *, exit_code, expected_threads, required_cell_ids
    ):
        seen.update(
            parser_text=text,
            parser_exit_code=exit_code,
            expected_threads=expected_threads,
            required_cell_ids=required_cell_ids,
        )
        return {"pass": True}

    monkeypatch.setattr(runner, "_run", run or default_run)
    monkeypatch.setattr(
        runner,
        "parse_openmc_geometry_debug_log",
        parser or default_parser,
    )
    arguments = {
        "model_xml": packet["model"],
        "model_receipt": packet["receipt"],
        "expected_model_receipt_sha256": _sha(packet["receipt"]),
        "output_directory": output,
        "openmc_executable": packet["executable"].resolve(),
        "expected_openmc_executable_sha256": _sha(packet["executable"]),
        "runtime_env": packet["runtime_env"].resolve(),
        "expected_runtime_env_sha256": _sha(packet["runtime_env"]),
        "threads": 3,
        "timeout_seconds": 77,
    }
    arguments.update(changes)
    result_path, passed = runner.execute_geometry_debug(**arguments)
    return (
        json.loads(result_path.read_text(encoding="utf-8")),
        passed,
        seen,
    )


def _assert_sealed(receipt, classification):
    assert receipt["terminal"] is True
    assert receipt["terminal_classification"] == classification
    assert receipt["status"] == (
        "PASS" if classification == "PASS" else "BLOCKED"
    )
    claimed = receipt.pop("receipt_content_sha256")
    assert claimed == contract.canonical_sha256(receipt)


def test_pass_uses_hash_bound_runtime_and_sealed_native_volume_ids(
    tmp_path, monkeypatch
):
    packet = _packet(tmp_path / "packet")
    before = {key: _sha(packet[key]) for key in ("model", "dagmc", "source")}
    receipt, passed, seen = _execute(packet, tmp_path / "run", monkeypatch)
    assert passed is True
    assert seen["command"] == [
        str(packet["executable"].resolve()),
        "-g",
        "-s",
        "3",
    ]
    assert seen["environment"] == {
        "LD_LIBRARY_PATH": "/qualified/lib",
        "OMP_PROC_BIND": "true",
    }
    assert seen["required_cell_ids"] == [2, 5, 9]
    assert receipt["required_native_volume_ids"] == [2, 5, 9]
    assert receipt["inputs_immutable"] is True
    assert (
        receipt["runtime_environment"]["inherit_parent_environment"] is False
    )
    assert (
        receipt["runtime_environment"]["values_recorded_in_receipt"] is False
    )
    assert before == {
        key: _sha(packet[key]) for key in ("model", "dagmc", "source")
    }
    _assert_sealed(receipt, "PASS")


def test_preflight_failure_still_writes_terminal_receipt(
    tmp_path, monkeypatch
):
    packet = _packet(tmp_path / "packet")
    receipt, passed, seen = _execute(
        packet,
        tmp_path / "run",
        monkeypatch,
        expected_openmc_executable_sha256="0" * 64,
    )
    assert passed is False
    assert seen == {}
    assert receipt["failure"]["stage"] == "PREFLIGHT"
    assert receipt["command"] is None
    _assert_sealed(receipt, "BLOCKED_PREFLIGHT")


@pytest.mark.parametrize(
    ("failure", "classification"),
    [
        ("timeout", "BLOCKED_TIMEOUT"),
        ("nonzero", "BLOCKED_NONZERO_EXIT"),
        ("parser", "BLOCKED_LOG_PARSER"),
        ("mutation", "BLOCKED_INPUT_MUTATION"),
    ],
)
def test_runtime_failures_always_write_terminal_receipt(
    tmp_path, monkeypatch, failure, classification
):
    packet = _packet(tmp_path / "packet")

    def fake_run(command, *, cwd, timeout_seconds, environment):
        if failure == "mutation":
            packet["source"].write_bytes(b"mutated during run")
        return {
            "exit_code": (
                9
                if failure == "nonzero"
                else (-1 if failure == "timeout" else 0)
            ),
            "timed_out": failure == "timeout",
            "output": "terminal log",
            "wall_time_seconds": 1.0,
        }

    def fake_parser(*args, **kwargs):
        if failure == "parser":
            raise ValueError("unparseable log")
        return {"pass": failure != "nonzero"}

    receipt, passed, _ = _execute(
        packet,
        tmp_path / "run",
        monkeypatch,
        run=fake_run,
        parser=fake_parser,
    )
    assert passed is False
    assert receipt["openmc_log"]["sha256"]
    if failure == "mutation":
        assert receipt["inputs_immutable"] is False
        assert receipt["input_hashes_before"] != receipt["input_hashes_after"]
    if failure == "parser":
        assert receipt["failure"]["stage"] == "LOG_PARSER"
    _assert_sealed(receipt, classification)


def test_runtime_environment_rejects_shell_expansion_fail_closed(
    tmp_path, monkeypatch
):
    packet = _packet(tmp_path / "packet")
    packet["runtime_env"].write_text(
        "PATH=$PATH:/qualified/bin\n", encoding="utf-8"
    )
    receipt, passed, _ = _execute(packet, tmp_path / "run", monkeypatch)
    assert passed is False
    _assert_sealed(receipt, "BLOCKED_PREFLIGHT")


def test_runner_and_contract_have_no_heavy_static_imports():
    forbidden = {"openmc", "numpy", "pymoab", "pydagmc", "cadquery", "gmsh"}
    for module in (runner, contract):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & forbidden


def test_output_directory_is_create_only(tmp_path, monkeypatch):
    packet = _packet(tmp_path / "packet")
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(FileExistsError):
        _execute(packet, output, monkeypatch)
