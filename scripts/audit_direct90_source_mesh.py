"""Qualify the immutable WISTELL-D 0-45 degree source input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pymoab import core, types

from parastell.activation_handoff import (
    HALF45_MODELED_SOURCE_RATE_PER_S,
    HALF45_SOURCE_MESH_SHA256,
    HALF45_STRENGTHS_SHA256,
)
from parastell.reference_geometry import sha256_file
from parastell.source_domain import (
    _source_arrays,
    audit_source_tetrahedra_arrays,
)
from parastell.source_geometry_identity import source_mesh_fingerprint
from parastell.source_mesh_order_audit import audit_source_order_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_mesh", type=Path)
    parser.add_argument("strengths", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source_mesh = args.source_mesh.resolve()
    strengths_path = args.strengths.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    if sha256_file(source_mesh) != HALF45_SOURCE_MESH_SHA256:
        raise ValueError("half-period source-mesh hash mismatch")
    if sha256_file(strengths_path) != HALF45_STRENGTHS_SHA256:
        raise ValueError("half-period strengths hash mismatch")

    tetrahedra, volumes, tagged_strengths = _source_arrays(source_mesh)
    external_strengths = np.load(strengths_path, allow_pickle=False)
    order = audit_source_order_arrays(
        tagged_strengths,
        external_strengths,
        expected_rate_per_s=HALF45_MODELED_SOURCE_RATE_PER_S,
    )
    tetrahedron_audit = audit_source_tetrahedra_arrays(
        tetrahedra, volumes, tagged_strengths
    )
    identity = source_mesh_fingerprint(source_mesh)

    mesh = core.Core()
    mesh.load_file(str(source_mesh))
    vertices = mesh.get_entities_by_type(mesh.get_root_set(), types.MBVERTEX)
    coordinates = np.asarray(mesh.get_coords(vertices)).reshape((-1, 3))
    phi = np.mod(
        np.degrees(np.arctan2(coordinates[:, 1], coordinates[:, 0])), 360.0
    )
    phi[np.isclose(phi, 360.0, atol=1.0e-9)] = 0.0
    angular_pass = bool(
        np.all(phi >= -1.0e-9)
        and np.all(phi <= 45.0 + 1.0e-9)
        and np.isclose(phi.min(), 0.0, atol=1.0e-9)
        and np.isclose(phi.max(), 45.0, atol=1.0e-9)
    )
    partial_pass = bool(
        order["order_and_rate_gate_pass"]
        and tetrahedron_audit["tetrahedron_data_gate_pass"]
        and angular_pass
    )
    result = {
        "schema": "wistell_d.source_mesh_order_audit/v1.0.0",
        "status": (
            "QUALIFIED_HALF_PERIOD_SOURCE_INPUT"
            if partial_pass
            else "SOURCE_MESH_ORDER_RATE_OR_ANGULAR_DOMAIN_FAIL"
        ),
        "source_mesh_sha256": sha256_file(source_mesh),
        "strengths_sha256": sha256_file(strengths_path),
        "source_mesh_identity": identity,
        "order_and_rate": order,
        "tetrahedron_audit": tetrahedron_audit,
        "angular_domain_degrees": {
            "minimum": float(phi.min()),
            "maximum": float(phi.max()),
            "expected": [0.0, 45.0],
            "pass": angular_pass,
        },
        "domain_element_order_audit_status": "PENDING_ACCEPTED_DAGMC_CONTAINMENT",
        "accepted_geometry_containment_run": False,
        "partial_gate_pass": partial_pass,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
