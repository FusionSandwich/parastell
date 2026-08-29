import hashlib
import json

import pytest

from parastell.parametric_openmc16_model import _canonical_sha
from parastell.transport_response_plan import bind_response_plan
from parastell.transport_response_plan import build_response_plan
from scripts import (
    extract_parametric_openmc16_full_response_results as extractor,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(tmp_path):
    statepoint = tmp_path / "statepoint.2.h5"
    statepoint.write_bytes(b"statepoint")
    plan = build_response_plan(
        case_id="case",
        magnet_ids=["magnet-0003"],
        neutron_energy_edges_eV=[0.0, 2.0e7],
        photon_energy_edges_eV=[0.0, 2.0e7],
    )
    plan = bind_response_plan(
        plan,
        tally_names=["neutron-flux", "photon-flux", "damage"],
        surface_bank_configured=True,
    )
    receipt = {
        "schema": "parastell.parametric_openmc16_full_response_smoke/v1.0.0",
        "status": "PASS",
        "claim": "BOUNDED_INFRASTRUCTURE_SMOKE_ONLY",
        "inputs_immutable": True,
        "statistics_qualified": False,
        "production_run_authorized": False,
        "selected_magnet_ids": ["magnet-0003"],
        "selected_magnet_cell_ids": {"magnet-0003": 12},
        "histories": 1000,
        "particles_per_batch": 500,
        "batches": 2,
        "seed": 19,
        "source_normalization": {
            "physical_rate_n_per_s_for_modeled_90_degrees": 1.0,
            "normalization_application": "apply exactly once downstream",
        },
        "full_response_wiring": {"response_plan": plan},
        "statepoints": [{"path": str(statepoint), "sha256": _sha(statepoint)}],
    }
    receipt["receipt_content_sha256"] = _canonical_sha(receipt)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path, statepoint


def test_extracts_every_wired_tally_and_self_seals(tmp_path, monkeypatch):
    receipt, statepoint = _packet(tmp_path)
    names = ["neutron-flux", "photon-flux", "damage"]
    monkeypatch.setattr(
        extractor,
        "read_openmc16_response_set",
        lambda path, tally_names: {
            "schema": "parastell.openmc16_response_set/v1.0.0",
            "status": "SMOKE_RESULT",
            "responses": [{"tally_id": name} for name in tally_names],
            "statepoint_sha256": _sha(path),
            "source_histories": 1000,
            "particles_per_batch": 500,
            "batches": 2,
            "seed": 19,
        },
    )
    result = extractor.extract(
        receipt,
        statepoint,
        expected_smoke_receipt_sha256=_sha(receipt),
        expected_statepoint_sha256=_sha(statepoint),
    )
    assert result["coverage"]["status"] == "COMPLETE"
    assert result["response_plan"]["wired_tally_names"] == names
    assert result["statistics_qualified"] is False
    sealed = dict(result)
    digest = sealed.pop("bundle_content_sha256")
    assert digest == _canonical_sha(sealed)


def test_rejects_missing_tally_or_statepoint_mismatch(tmp_path, monkeypatch):
    receipt, statepoint = _packet(tmp_path)
    monkeypatch.setattr(
        extractor,
        "read_openmc16_response_set",
        lambda path, tally_names: {
            "responses": [{"tally_id": tally_names[0]}],
            "statepoint_sha256": _sha(path),
            "source_histories": 1000,
            "particles_per_batch": 500,
            "batches": 2,
            "seed": 19,
        },
    )
    with pytest.raises(ValueError, match="every wired tally"):
        extractor.extract(
            receipt,
            statepoint,
            expected_smoke_receipt_sha256=_sha(receipt),
            expected_statepoint_sha256=_sha(statepoint),
        )
    with pytest.raises(ValueError, match="statepoint hash mismatch"):
        extractor.extract(
            receipt,
            statepoint,
            expected_smoke_receipt_sha256=_sha(receipt),
            expected_statepoint_sha256="0" * 64,
        )


def test_rejects_mutated_or_unpassed_smoke_receipt(tmp_path):
    receipt, statepoint = _packet(tmp_path)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["status"] = "BLOCKED_FULL_RESPONSE_SMOKE"
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="not accepted"):
        extractor.extract(
            receipt,
            statepoint,
            expected_smoke_receipt_sha256=_sha(receipt),
            expected_statepoint_sha256=_sha(statepoint),
        )


def test_runnable_consumer_handoff_requires_exact_run_bound_inputs(
    tmp_path, monkeypatch
):
    extraction = {
        "bundle_content_sha256": "f" * 64,
        "statepoint": {"sha256": "a" * 64},
        "response_plan": {"wired_tally_names": ["flux"]},
        "response_set": {"source_histories": 30},
    }
    tally_bindings = tmp_path / "bindings.json"
    binding_payload = {
        "schema": "parastell.openmc16_exact_tally_bindings/v1.0.0",
        "status": "PASS",
        "statepoint_sha256": "a" * 64,
        "model_xml_sha256": "b" * 64,
        "tallies": {"flux": {"definition": "bound"}},
    }
    binding_payload["bindings_sha256"] = _canonical_sha(binding_payload)
    tally_bindings.write_text(json.dumps(binding_payload), encoding="utf-8")
    consumer = tmp_path / "consumer.json"
    consumer.write_text(
        json.dumps(
            {
                "schema": "parastell.openmc16_consumer_handoff_input/v1.0.0",
                "status": "PASS",
                "provenance": {
                    "statepoint_sha256": "a" * 64,
                    "model_xml_sha256": "b" * 64,
                    "source_histories": 30,
                },
                "materials": [],
                "boundary_phase_space": {},
                "activation_schedule_reference": {},
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return {"schema": "test-handoff", "status": "PASS"}

    monkeypatch.setattr(
        extractor, "build_openmc16_radiation_consumer_handoff", fake_builder
    )
    monkeypatch.setattr(
        extractor, "validate_radiation_consumer_handoff", lambda value: None
    )
    handoff = extractor.build_consumer_handoff(
        extraction,
        tally_bindings_path=tally_bindings,
        expected_tally_bindings_sha256=_sha(tally_bindings),
        consumer_input_path=consumer,
        expected_consumer_input_sha256=_sha(consumer),
    )
    assert captured["tally_bindings"] == {"flux": {"definition": "bound"}}
    assert handoff["extraction_binding"]["result_bundle_sha256"] == "f" * 64
    assert len(handoff["handoff_content_sha256"]) == 64

    changed = json.loads(tally_bindings.read_text(encoding="utf-8"))
    changed["statepoint_sha256"] = "0" * 64
    unsigned = dict(changed)
    unsigned.pop("bindings_sha256")
    changed["bindings_sha256"] = _canonical_sha(unsigned)
    tally_bindings.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="run-mismatched"):
        extractor.build_consumer_handoff(
            extraction,
            tally_bindings_path=tally_bindings,
            expected_tally_bindings_sha256=_sha(tally_bindings),
            consumer_input_path=consumer,
            expected_consumer_input_sha256=_sha(consumer),
        )
