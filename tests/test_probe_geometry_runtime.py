import json
from pathlib import Path
import platform
import sys

import pytest

from parastell.reference_geometry import sha256_file
from parastell.source_cad_refaceting import REQUIRED_RUNTIME_MODULES
from parastell.source_cad_refaceting import RUNTIME_NATIVE_MODULES
from parastell.source_cad_refaceting import RUNTIME_RECEIPT_SCHEMA
from parastell.source_cad_refaceting import _validate_runtime_receipt
from parastell.source_cad_refaceting import runtime_module_receipt
from scripts.probe_geometry_runtime import REQUIRED_MODULES
from scripts.probe_geometry_runtime import SCHEMA
from scripts.probe_geometry_runtime import _publish_json_create_only


def test_geometry_runtime_receipt_contract_is_complete():
    assert SCHEMA == "parastell.geometry_runtime_receipt/v1.1.0"
    assert REQUIRED_MODULES == (
        "cadquery",
        "cad_to_dagmc",
        "gmsh",
        "pymoab",
        "h5py",
        "OCP",
        "numpy",
    )
    assert REQUIRED_MODULES == REQUIRED_RUNTIME_MODULES
    assert SCHEMA == RUNTIME_RECEIPT_SCHEMA
    assert "OCP" not in RUNTIME_NATIVE_MODULES
    assert RUNTIME_NATIVE_MODULES["pymoab"] == (
        "pymoab.core",
        "pymoab.types",
    )


def test_runtime_receipt_is_bound_to_active_interpreter_modules_and_h5m(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.h5m"
    source.write_bytes(b"source")
    source_sha256 = sha256_file(source)
    module_file = tmp_path / "module.py"
    module_file.write_text("# module\n", encoding="utf-8")
    module_row = {
        "import_pass": True,
        "module_version": "1.0",
        "distribution_version": None,
        "path": str(module_file.resolve()),
        "file_sha256": sha256_file(module_file),
        "native_artifacts": [],
    }
    module_rows = [
        {"module": name, **module_row} for name in REQUIRED_RUNTIME_MODULES
    ]
    monkeypatch.setattr(
        "parastell.source_cad_refaceting.runtime_module_receipt",
        lambda name: {"module": name, **module_row},
    )
    executable = Path(sys.executable).resolve()
    output_dir = tmp_path / "attempt" / "coarse"
    implementation = {
        "probe_script": {"sha256": "a" * 64},
        "refacet_module": {
            "sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "parastell"
                / "source_cad_refaceting.py"
            )
        },
        "refacet_cli": {
            "sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "refacet_source_cad_candidate.py"
            )
        },
    }
    controls = {"source_manifest_sha256": "b" * 64}
    control_rows = {
        name: {"sha256": value} for name, value in controls.items()
    }
    receipt = {
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "scientific_decision": "NOT_PERFORMED_RUNTIME_PREFLIGHT_ONLY",
        "run_binding": {
            "attempt_id": "attempt-v5",
            "launch_nonce": "c" * 32,
            "host_alias": "poly-bateman",
            "hostname": platform.node().split(".", 1)[0],
            "machine": platform.machine(),
            "attempt_root": str(output_dir.parent.resolve()),
            "refacet_output_dir": str(output_dir.resolve()),
            "faceting_level": "coarse",
            "threads": 4,
        },
        "python": {
            "resolved_executable": str(executable),
            "executable_sha256": sha256_file(executable),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "modules": module_rows,
        "required_modules": list(REQUIRED_RUNTIME_MODULES),
        "all_required_imports_pass": True,
        "source_h5m": {
            "path": str(source.resolve()),
            "size_bytes": source.stat().st_size,
            "sha256_before": source_sha256,
            "sha256_after": source_sha256,
            "unchanged": True,
            "native_entity_set_count": 1,
            "native_reload_pass": True,
        },
        "implementation_before": implementation,
        "implementation_after": implementation,
        "implementation_unchanged": True,
        "control_inputs_before": control_rows,
        "control_inputs_after": control_rows,
        "control_inputs_unchanged": True,
        "runtime_gate_pass": True,
    }
    receipt_path = tmp_path / "runtime.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_sha256 = sha256_file(receipt_path)

    result = _validate_runtime_receipt(
        receipt_path,
        expected_sha256=receipt_sha256,
        expected_attempt_id="attempt-v5",
        expected_launch_nonce="c" * 32,
        expected_host_alias="poly-bateman",
        expected_output_dir=output_dir,
        expected_level="coarse",
        expected_threads=4,
        expected_source_h5m_path=source,
        expected_source_h5m_sha256=source_sha256,
        expected_controls=controls,
    )
    assert result["pass"] is True

    drifted = dict(receipt)
    drifted["modules"] = [dict(row) for row in module_rows]
    drifted["modules"][0]["file_sha256"] = "0" * 64
    drifted_path = tmp_path / "runtime-drifted.json"
    drifted_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="module_identity"):
        _validate_runtime_receipt(
            drifted_path,
            expected_sha256=sha256_file(drifted_path),
            expected_attempt_id="attempt-v5",
            expected_launch_nonce="c" * 32,
            expected_host_alias="poly-bateman",
            expected_output_dir=output_dir,
            expected_level="coarse",
            expected_threads=4,
            expected_source_h5m_path=source,
            expected_source_h5m_sha256=source_sha256,
            expected_controls=controls,
        )


def test_runtime_receipt_publication_is_create_only(tmp_path):
    destination = tmp_path / "receipt.json"
    destination.write_text('{"owner":"other"}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        _publish_json_create_only(destination, {"owner": "probe"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "owner": "other"
    }
    assert list(tmp_path.glob("*.pending")) == []


def test_ocp_single_extension_is_hashed_without_child_import(
    tmp_path, monkeypatch
):
    extension = tmp_path / "OCP.so"
    extension.write_bytes(b"extension")
    module = type(
        "OcpExtension",
        (),
        {"__file__": str(extension), "__version__": "7.8.1"},
    )()
    imported = []

    def fake_import(name):
        imported.append(name)
        if name == "OCP":
            return module
        raise AssertionError(f"unexpected child import: {name}")

    monkeypatch.setattr(
        "parastell.source_cad_refaceting.importlib.import_module", fake_import
    )
    monkeypatch.setattr(
        "parastell.source_cad_refaceting._distribution_version",
        lambda name: None,
    )

    receipt = runtime_module_receipt("OCP")

    assert imported == ["OCP"]
    assert receipt["path"] == str(extension.resolve())
    assert receipt["file_sha256"] == sha256_file(extension)
    assert receipt["native_artifacts"] == []
