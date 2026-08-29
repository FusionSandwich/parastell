from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from parastell.transport_response_plan import build_response_plan
from scripts import run_parametric_openmc16_full_response_smoke as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path):
    dagmc = tmp_path / "dagmc.h5m"
    dagmc.write_bytes(b"dagmc")
    magnets = []
    for index in range(18):
        surface_id = 100 + index
        magnets.append(
            {
                "magnet": f"magnet-{index:04d}",
                "dagmc_volume_id": index + 9,
                "centroid_cm": [100.0 + index, 0.0, 0.0],
                "dagmc_surface_ids": [surface_id],
                "closed": True,
                "coupling_interface": "homogenized_magnet_outer_boundary",
                "surfaces": [
                    {
                        "dagmc_surface_id": surface_id,
                        "magnet_outward_normal_multiplier": (
                            -1 if index % 2 else 1
                        ),
                    }
                ],
            }
        )
    payload = {
        "schema": "parastell.parametric_magnet_surface_manifest/v1.0.0",
        "status": "PASS",
        "dagmc_sha256": _sha(dagmc),
        "coupling_interface": "homogenized_magnet_outer_boundary",
        "magnet_count": 18,
        "all_envelopes_close": True,
        "physical_h5m_mutation": False,
        "magnets": magnets,
    }
    path = tmp_path / "surfaces.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return dagmc, path


def _plan(magnets=("magnet-0000",)):
    return build_response_plan(
        case_id="direct90-smoke",
        magnet_ids=list(magnets),
        neutron_energy_edges_eV=[0.0, 1.0e6, 2.0e7],
        photon_energy_edges_eV=[0.0, 1.0e5, 2.0e7],
        nuclide_mt_requests={"W184": [2, 102]},
    )


def test_selects_arbitrary_magnets_and_exact_surface_union(tmp_path):
    dagmc, manifest = _manifest(tmp_path)
    signs, cells = runner._selected_surface_inputs(
        manifest,
        dagmc,
        ["magnet-0000", "magnet-0017"],
        _sha(manifest),
    )
    assert signs == {100: 1, 117: -1}
    assert cells == {"magnet-0000": 9, "magnet-0017": 26}


@pytest.mark.parametrize(
    "selected", [["unknown"], ["magnet-0000", "magnet-0000"]]
)
def test_rejects_unknown_or_duplicate_magnet_selection(tmp_path, selected):
    dagmc, manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="unknown|nonempty and unique"):
        runner._selected_surface_inputs(
            manifest, dagmc, selected, _sha(manifest)
        )


def test_rejects_shared_selected_surface(tmp_path):
    dagmc, manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["magnets"][1]["dagmc_surface_ids"] = [100]
    payload["magnets"][1]["surfaces"][0]["dagmc_surface_id"] = 100
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping is invalid"):
        runner._selected_surface_inputs(
            manifest,
            dagmc,
            ["magnet-0000", "magnet-0001"],
            _sha(manifest),
        )


def test_loads_hash_bound_wrapped_response_plan(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"response_plan": _plan()}), encoding="utf-8")
    assert (
        runner._load_response_plan(path, _sha(path))["case_id"]
        == "direct90-smoke"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        runner._load_response_plan(path, "0" * 64)


@pytest.mark.parametrize(
    "changes",
    [
        {"particles": 50_001, "batches": 2},
        {"threads": 5},
        {"timeout_seconds": 0},
        {"timeout_seconds": 3_601},
        {"max_particles": 1_000_001},
    ],
)
def test_smoke_controls_are_hard_capped(changes):
    values = {
        "particles": 500,
        "batches": 2,
        "seed": 1,
        "max_particles": 100_000,
        "threads": 1,
        "timeout_seconds": 60,
    }
    values.update(changes)
    with pytest.raises(ValueError, match="capped|invalid|100,000"):
        runner._validate_smoke_controls(**values)


@pytest.mark.parametrize("make_bank", [True, False])
def test_main_wires_and_runs_or_fails_closed(tmp_path, monkeypatch, make_bank):
    model_xml = tmp_path / "model.xml"
    receipt_path = tmp_path / "receipt.json"
    dagmc = tmp_path / "dagmc.h5m"
    surface = tmp_path / "surface.json"
    plan_path = tmp_path / "plan.json"
    for path in (model_xml, receipt_path, dagmc, surface, plan_path):
        path.write_bytes(path.name.encode())
    output = tmp_path / "run"
    model = SimpleNamespace(
        settings=SimpleNamespace(),
        tallies=None,
        export_to_model_xml=lambda path: path.write_text(
            "<model/>", encoding="utf-8"
        ),
    )
    fake_openmc = SimpleNamespace(
        __version__="0.16.0",
        Model=SimpleNamespace(from_model_xml=lambda path: model),
        Tallies=list,
    )
    monkeypatch.setitem(sys.modules, "openmc", fake_openmc)
    monkeypatch.setattr(
        runner,
        "_load_bound_receipt",
        lambda *a, **k: {
            "source": {
                "physical_rate_n_per_s_for_modeled_90_degrees": 1.0,
                "normalization_application": "apply exactly once downstream",
            }
        },
    )
    monkeypatch.setattr(runner, "_load_response_plan", lambda *a, **k: _plan())
    monkeypatch.setattr(
        runner,
        "_selected_surface_inputs",
        lambda *a, **k: ({100: 1}, {"magnet-0000": 9}),
    )
    monkeypatch.setattr(
        runner,
        "_runtime_inputs",
        lambda *a, **k: {"dagmc_h5m": {"sha256": _sha(dagmc)}},
    )
    monkeypatch.setattr(runner, "_nuclear_data_hashes", lambda receipt: {})
    monkeypatch.setattr(runner, "_replace_dagmc_filename", lambda *a: 27)
    monkeypatch.setattr(
        runner,
        "build_surface_instrumentation_spec",
        lambda **kwargs: {
            "coupling_interface": "homogenized_magnet_outer_boundary",
            "surface_ids": kwargs["surface_ids"],
        },
    )
    monkeypatch.setattr(
        runner,
        "instrument_selected_case",
        lambda *a, **k: {"status": "WIRED_NOT_EXECUTED"},
    )

    def fake_run(path, *, threads, timeout_seconds):
        (path / "statepoint.2.h5").write_bytes(b"statepoint")
        if make_bank:
            (path / "surface_source.h5").write_bytes(b"bank")
        return {"exit_code": 0, "timed_out": False, "output": "ok"}

    monkeypatch.setattr(runner, "_run_openmc", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parametric_openmc16_full_response_smoke",
            str(model_xml),
            str(receipt_path),
            str(dagmc),
            str(surface),
            str(plan_path),
            str(output),
            "--expected-model-receipt-sha256",
            _sha(receipt_path),
            "--expected-surface-manifest-sha256",
            _sha(surface),
            "--expected-response-plan-sha256",
            _sha(plan_path),
        ],
    )
    if make_bank:
        runner.main()
    else:
        with pytest.raises(SystemExit):
            runner.main()
    result = json.loads(
        (output / "FULL_RESPONSE_SMOKE_RECEIPT.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == (
        "PASS" if make_bank else "BLOCKED_FULL_RESPONSE_SMOKE"
    )
    assert result["statistics_qualified"] is False
    assert result["production_run_authorized"] is False
    sealed = dict(result)
    content_hash = sealed.pop("receipt_content_sha256")
    assert content_hash == runner._canonical_sha(sealed)
