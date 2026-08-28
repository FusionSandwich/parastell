import hashlib
import json
import sys

from scripts import audit_dagmc_topology


def test_cli_exception_writes_failed_receipts_and_returns_nonzero(
    tmp_path, monkeypatch
):
    candidate = tmp_path / "candidate.h5m"
    candidate.write_bytes(b"candidate")
    candidate_sha256 = hashlib.sha256(b"candidate").hexdigest()
    criteria = tmp_path / "criteria.json"
    criteria.write_text(
        json.dumps(
            {"magnet_envelopes": {"vector_area_relative_tolerance": 1.0e-8}}
        ),
        encoding="utf-8",
    )
    criteria_sha256 = hashlib.sha256(criteria.read_bytes()).hexdigest()
    output = tmp_path / "receipt"
    monkeypatch.setattr(
        audit_dagmc_topology,
        "audit_dagmc_topology",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("deliberate topology failure")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_dagmc_topology.py",
            str(candidate),
            str(output),
            "--expected-dagmc-sha256",
            candidate_sha256,
            "--acceptance-criteria",
            str(criteria),
            "--expected-acceptance-criteria-sha256",
            criteria_sha256,
        ],
    )

    assert audit_dagmc_topology.main() == 1
    topology = json.loads(
        (output / "dagmc_topology_audit.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (output / "original_magnet_envelope_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    assert topology["topology_gate_pass"] is False
    assert topology["pydagmc_load_attempted"] is None
    assert topology["audit_stage"] == "native_and_pydagmc_topology_audit"
    assert inventory["inventory_gate_pass"] is False


def test_cli_false_gate_writes_receipts_and_returns_nonzero(
    tmp_path, monkeypatch
):
    candidate = tmp_path / "candidate.h5m"
    candidate.write_bytes(b"candidate")
    candidate_sha256 = hashlib.sha256(b"candidate").hexdigest()
    criteria = tmp_path / "criteria.json"
    criteria.write_text(
        json.dumps(
            {"magnet_envelopes": {"vector_area_relative_tolerance": 1.0e-8}}
        ),
        encoding="utf-8",
    )
    criteria_sha256 = hashlib.sha256(criteria.read_bytes()).hexdigest()
    output = tmp_path / "receipt"
    monkeypatch.setattr(
        audit_dagmc_topology,
        "audit_dagmc_topology",
        lambda *args, **kwargs: {
            "topology_gate_pass": False,
            "volume_envelopes": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_dagmc_topology.py",
            str(candidate),
            str(output),
            "--expected-dagmc-sha256",
            candidate_sha256,
            "--acceptance-criteria",
            str(criteria),
            "--expected-acceptance-criteria-sha256",
            criteria_sha256,
        ],
    )

    assert audit_dagmc_topology.main() == 1
    assert (output / "dagmc_topology_audit.json").is_file()
    assert (output / "original_magnet_envelope_inventory.json").is_file()
