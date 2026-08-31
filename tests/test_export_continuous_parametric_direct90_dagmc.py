from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_continuous_parametric_direct90_dagmc import (
    FROZEN_MESHING,
    REFACET_PROTOCOL_SCHEMA,
    _validate_meshing_contract,
)


SOURCE_PACKET = {
    "manifest_sha256": "1" * 64,
    "audit_sha256": "2" * 64,
    "audit_seal_sha256": "3" * 64,
    "audit_producer_sha256": "4" * 64,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        output_root=tmp_path / "candidate",
        min_mesh_size_cm=1.0,
        max_mesh_size_cm=5.0,
        algorithm=1,
        threads=16,
        manifest_sha256=SOURCE_PACKET["manifest_sha256"],
        audit_sha256=SOURCE_PACKET["audit_sha256"],
        audit_seal_sha256=SOURCE_PACKET["audit_seal_sha256"],
        audit_producer_sha256=SOURCE_PACKET["audit_producer_sha256"],
        refacet_protocol_path=None,
        refacet_protocol_sha256=None,
        reference_h5m_path=None,
        reference_h5m_sha256=None,
    )


def _bind_protocol(tmp_path: Path, args: Namespace) -> dict:
    reference = tmp_path / "coarse.h5m"
    reference.write_bytes(b"bound coarse DAGMC reference")
    args.reference_h5m_path = reference
    args.reference_h5m_sha256 = _sha256(reference)
    protocol = {
        "schema": REFACET_PROTOCOL_SCHEMA,
        "status": "PREREGISTERED",
        "change_class": "REFACET_ONLY_SAME_CAD_SOURCE_AND_MATERIALS",
        "source_packet": SOURCE_PACKET,
        "reference": {
            "h5m_sha256": args.reference_h5m_sha256,
            "meshing": FROZEN_MESHING,
        },
        "candidate": {
            "output_root": str(args.output_root.resolve()),
            "meshing": {
                "minimum_cm": 1.0,
                "maximum_cm": 5.0,
                "algorithm": 1,
                "threads": 16,
            },
        },
        "constraints": {
            "source_and_cad_immutable": True,
            "materials_immutable": True,
            "physical_h5m_mutation": False,
            "output_create_only": True,
            "transport_eligible_before_successor_gates": False,
        },
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    args.refacet_protocol_path = path
    args.refacet_protocol_sha256 = _sha256(path)
    return protocol


def test_frozen_mode_remains_exact(tmp_path):
    args = _args(tmp_path)
    args.min_mesh_size_cm = 5.0
    args.max_mesh_size_cm = 20.0
    args.threads = 32
    assert _validate_meshing_contract(args) is None


def test_unregistered_mesh_change_remains_rejected(tmp_path):
    with pytest.raises(ValueError, match="frozen coarse attempt"):
        _validate_meshing_contract(_args(tmp_path))


def test_hash_bound_refacet_protocol_passes(tmp_path):
    args = _args(tmp_path)
    protocol = _bind_protocol(tmp_path, args)
    assert _validate_meshing_contract(args) == protocol


@pytest.mark.parametrize(
    "mutation",
    [
        lambda args, protocol: setattr(args, "max_mesh_size_cm", 6.0),
        lambda args, protocol: protocol["constraints"].update(
            {"materials_immutable": False}
        ),
        lambda args, protocol: protocol["reference"].update(
            {"h5m_sha256": "f" * 64}
        ),
        lambda args, protocol: protocol["source_packet"].update(
            {"manifest_sha256": "0" * 64}
        ),
    ],
)
def test_refacet_protocol_fails_closed_on_mismatch(tmp_path, mutation):
    args = _args(tmp_path)
    protocol = _bind_protocol(tmp_path, args)
    mutation(args, protocol)
    args.refacet_protocol_path.write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    args.refacet_protocol_sha256 = _sha256(args.refacet_protocol_path)
    with pytest.raises(ValueError, match="does not match|bounded refinement"):
        _validate_meshing_contract(args)


def test_reference_artifact_hash_is_rechecked(tmp_path):
    args = _args(tmp_path)
    _bind_protocol(tmp_path, args)
    args.reference_h5m_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="reference H5M hash mismatch"):
        _validate_meshing_contract(args)
