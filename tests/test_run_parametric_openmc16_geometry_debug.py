import hashlib
import json
import sys

import pytest

from parastell.parametric_openmc16_model import _canonical_sha
from scripts import run_parametric_openmc16_geometry_debug as runner


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(tmp_path):
    model = tmp_path / "model.xml"
    h5m = tmp_path / "dagmc.h5m"
    source = tmp_path / "source_mesh.h5m"
    cross_sections = tmp_path / "cross_sections.xml"
    data = tmp_path / "Fe56.h5"
    h5m.write_bytes(b"dagmc")
    source.write_bytes(b"source")
    cross_sections.write_text("<cross_sections/>", encoding="utf-8")
    data.write_bytes(b"data")
    model.write_text(
        "<model>"
        "<materials><cross_sections>cross_sections.xml</cross_sections>"
        "</materials>"
        '<geometry><dagmc_universe id="27" filename="dagmc.h5m"/>'
        "</geometry>"
        '<settings><mesh id="1" type="unstructured">'
        "<filename>source_mesh.h5m</filename></mesh></settings>"
        "</model>",
        encoding="utf-8",
    )
    receipt = {
        "schema": "parastell.parametric_openmc16_model/v1.0.0",
        "status": "MODEL_EXPORTED_TRANSPORT_PENDING",
        "claim": "BOUNDED_SMOKE_ONLY",
        "openmc_version": "0.16.0",
        "model_xml": {"path": str(model), "sha256": _sha(model)},
        "dagmc_sha256": _sha(h5m),
        "source_mesh_sha256": _sha(source),
        "cross_sections_xml_sha256": _sha(cross_sections),
        "modeled_extent_degrees": 90.0,
        "n_field_periods": 4,
        "magnet_cell_ids": {
            f"magnet-{index:04d}": index + 9 for index in range(18)
        },
        "photon_transport": True,
        "physical_h5m_mutation": False,
        "all_bound_inputs_immutable": True,
        "selected_nuclear_data_immutable": True,
        "nuclear_data_manifest": {
            "libraries": [{"path": str(data), "sha256": _sha(data)}]
        },
    }
    receipt["receipt_content_sha256"] = _canonical_sha(receipt)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return model, h5m, source, cross_sections, data, receipt_path


def test_accepts_exact_hash_bound_parametric_model_packet(tmp_path):
    model, h5m, source, cross_sections, data, receipt_path = _packet(tmp_path)
    receipt = runner._load_bound_receipt(
        receipt_path,
        _sha(receipt_path),
        model_xml=model,
        dagmc_h5m=h5m,
    )
    assert receipt["magnet_cell_ids"]["magnet-0017"] == 26
    assert runner._nuclear_data_hashes(receipt) == {
        str(data.resolve()): _sha(data)
    }
    assert runner._runtime_inputs(model, h5m, receipt) == {
        "dagmc_h5m": {"path": str(h5m.resolve()), "sha256": _sha(h5m)},
        "source_mesh": {
            "path": str(source.resolve()),
            "sha256": _sha(source),
        },
        "cross_sections_xml": {
            "path": str(cross_sections.resolve()),
            "sha256": _sha(cross_sections),
        },
    }


def test_rejects_receipt_content_or_h5m_mutation(tmp_path):
    model, h5m, _, _, _, receipt_path = _packet(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["photon_transport"] = False
    receipt_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="not accepted"):
        runner._load_bound_receipt(
            receipt_path,
            _sha(receipt_path),
            model_xml=model,
            dagmc_h5m=h5m,
        )

    model, h5m, _, _, _, receipt_path = _packet(tmp_path)
    h5m.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="not accepted"):
        runner._load_bound_receipt(
            receipt_path,
            _sha(receipt_path),
            model_xml=model,
            dagmc_h5m=h5m,
        )


def test_rejects_selected_nuclear_data_mutation(tmp_path):
    model, h5m, _, _, data, receipt_path = _packet(tmp_path)
    receipt = runner._load_bound_receipt(
        receipt_path,
        _sha(receipt_path),
        model_xml=model,
        dagmc_h5m=h5m,
    )
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="nuclear-data hash mismatch"):
        runner._nuclear_data_hashes(receipt)


@pytest.mark.parametrize("target", ["dagmc", "source", "cross_sections"])
def test_rejects_xml_external_input_mismatch_or_mutation(tmp_path, target):
    model, h5m, source, cross_sections, _, receipt_path = _packet(tmp_path)
    receipt = runner._load_bound_receipt(
        receipt_path,
        _sha(receipt_path),
        model_xml=model,
        dagmc_h5m=h5m,
    )
    if target == "dagmc":
        other = tmp_path / "other.h5m"
        other.write_bytes(h5m.read_bytes())
        model.write_text(
            model.read_text(encoding="utf-8").replace(
                'filename="dagmc.h5m"', 'filename="other.h5m"'
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="supplied H5M"):
            runner._runtime_inputs(model, h5m, receipt)
    else:
        path = source if target == "source" else cross_sections
        path.write_bytes(b"mutated")
        with pytest.raises(ValueError, match="hash binding failed"):
            runner._runtime_inputs(model, h5m, receipt)


def test_main_writes_pass_receipt_for_bound_geometry_debug(
    tmp_path, monkeypatch
):
    model, h5m, source, cross_sections, data, receipt_path = _packet(tmp_path)
    output = tmp_path / "run"
    seen = {}

    def fake_run(command, *, cwd, timeout_seconds):
        seen.update(command=command, cwd=cwd, timeout=timeout_seconds)
        return {
            "exit_code": 0,
            "timed_out": False,
            "output": "geometry debug passed",
            "wall_time_seconds": 1.25,
        }

    def fake_parse(log, *, exit_code, expected_threads, required_cell_ids):
        seen.update(
            log=log,
            exit_code=exit_code,
            expected_threads=expected_threads,
            required_cell_ids=required_cell_ids,
        )
        return {"pass": True}

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "parse_openmc_geometry_debug_log", fake_parse)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parametric_openmc16_geometry_debug",
            str(model),
            str(receipt_path),
            str(h5m),
            str(output),
            "--expected-model-receipt-sha256",
            _sha(receipt_path),
            "--threads",
            "3",
            "--timeout-seconds",
            "77",
        ],
    )
    before = {
        path: _sha(path) for path in (model, h5m, source, cross_sections, data)
    }
    runner.main()
    result = json.loads(
        (output / "PARAMETRIC_OPENMC16_GEOMETRY_DEBUG.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "PASS"
    assert result["inputs_immutable"] is True
    assert result["production_run_authorized"] is False
    assert seen["command"] == ["openmc", "-g", "-s", "3"]
    assert seen["required_cell_ids"] == list(range(1, 27))
    assert seen["timeout"] == 77
    assert before == {
        path: _sha(path) for path in (model, h5m, source, cross_sections, data)
    }


@pytest.mark.parametrize("failure", ["timeout", "parser", "mutation"])
def test_main_fails_closed_on_runtime_failure(tmp_path, monkeypatch, failure):
    model, h5m, source, _, _, receipt_path = _packet(tmp_path)
    output = tmp_path / "run"

    def fake_run(command, *, cwd, timeout_seconds):
        if failure == "mutation":
            source.write_bytes(b"mutated-during-run")
        return {
            "exit_code": -1 if failure == "timeout" else 1,
            "timed_out": failure == "timeout",
            "output": "blocked geometry debug",
            "wall_time_seconds": 1.0,
        }

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(
        runner,
        "parse_openmc_geometry_debug_log",
        lambda *args, **kwargs: {"pass": failure != "parser"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parametric_openmc16_geometry_debug",
            str(model),
            str(receipt_path),
            str(h5m),
            str(output),
            "--expected-model-receipt-sha256",
            _sha(receipt_path),
        ],
    )
    expected_error = ValueError if failure == "mutation" else SystemExit
    with pytest.raises(expected_error):
        runner.main()
    result_path = output / "PARAMETRIC_OPENMC16_GEOMETRY_DEBUG.json"
    if result_path.exists():
        assert (
            json.loads(result_path.read_text(encoding="utf-8"))["status"]
            != "PASS"
        )
