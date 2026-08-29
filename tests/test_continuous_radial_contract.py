from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from parastell.continuous_radial_contract import (
    COMPONENT_ORDER,
    COUPLING_INTERFACE,
    radial_separation_proof,
    validate_source_manifest,
)
from scripts.build_continuous_magnet_surface_manifest import (
    build_manifest,
    classify_surface_role,
)


def _manifest():
    stages = []
    previous = "0" * 64
    for index, component in enumerate(COMPONENT_ORDER[1:]):
        outer = f"{index + 1:x}" * 64
        stages.append(
            {
                "stage_index": index,
                "component": component,
                "cumulative_inner_matrix_sha256": previous,
                "layer_matrix_sha256": "a" * 64,
                "cumulative_outer_matrix_sha256": outer,
            }
        )
        previous = outer
    return {
        "schema": "parastell.parametric_source_cad/v1.0.0",
        "status": "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING",
        "transport_eligible": False,
        "plan": {
            "device": "WISTELL-D",
            "extent_degrees": 90.0,
            "field_period_degrees": 90.0,
            "grid_shape": [80, 90],
            "magnet_representation": "radial_envelope",
            "radial_build_mode": "isolated_cumulative_shells",
            "component_order": list(COMPONENT_ORDER),
            "safety": {
                "ports": False,
                "post_build_transforms": False,
                "imported_physical_solids": False,
            },
        },
        "live_vmec_metadata": {
            "pass": True,
            "n_field_periods": 4,
            "stellarator_symmetric": True,
            "lasym_vmec_logical": False,
        },
        "actual_invessel_component_order": list(COMPONENT_ORDER),
        "radial_build_mode": "isolated_cumulative_shells",
        "magnet_inventory": {
            "representation": "radial_envelope",
            "material": "homogenized_magnet",
        },
        "component_evidence": {
            role: {"brep_valid": True, "solid_count": 1, "volume_cm3": 1.0}
            for role in COMPONENT_ORDER
        },
        "radial_build_construction_stages": stages,
        "resolved_layers": {
            role: {
                "shape": [80, 90],
                "minimum_cm": 1.0,
                "finite": True,
                "strictly_positive": True,
                "poloidally_closed": True,
            }
            for role in COMPONENT_ORDER[1:]
        },
    }


def test_authoritative_manifest_accepts_only_continuous_nine_volume_model():
    manifest = _manifest()
    validate_source_manifest(manifest)
    proofs = radial_separation_proof(manifest)
    assert len(proofs) == 28
    assert all(row["pass"] for row in proofs)
    assert all(row["exact_boolean_required"] is False for row in proofs)


def test_authoritative_manifest_survives_canonical_json_key_order():
    manifest = json.loads(json.dumps(_manifest(), sort_keys=True))
    validate_source_manifest(manifest)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("plan", "magnet_representation"), "swept_filaments"),
        (("plan", "component_order"), [*COMPONENT_ORDER, "magnet-0000"]),
        (("magnet_inventory", "coil_count"), 18),
    ],
)
def test_swept_or_relabelled_global_geometry_is_rejected(path, value):
    manifest = copy.deepcopy(_manifest())
    manifest[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_source_manifest(manifest)


def test_broken_cumulative_boundary_hash_chain_is_rejected():
    manifest = _manifest()
    manifest["radial_build_construction_stages"][2][
        "cumulative_inner_matrix_sha256"
    ] = ("f" * 64)
    with pytest.raises(ValueError, match="construction proof"):
        validate_source_manifest(manifest)


def test_surface_role_uses_adjacency_for_inner_interface():
    vertices = np.asarray(
        [[[1.0, 2.0, 3.0], [1.0, 3.0, 3.0], [2.0, 2.0, 3.0]]]
    )
    assert (
        classify_surface_role(adjacent_volume_ids=[9, 8], vertices_cm=vertices)
        == COUPLING_INTERFACE
    )


def test_surface_role_separates_periodic_ends_and_outer_boundary():
    phi_zero = np.asarray(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 1.0]]]
    )
    phi_ninety = np.asarray(
        [[[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 1.0]]]
    )
    outer = np.asarray([[[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 1.0]]])
    assert (
        classify_surface_role(adjacent_volume_ids=[9], vertices_cm=phi_zero)
        == "periodic_end_phi_0"
    )
    assert (
        classify_surface_role(adjacent_volume_ids=[9], vertices_cm=phi_ninety)
        == "periodic_end_phi_90"
    )
    assert (
        classify_surface_role(adjacent_volume_ids=[9], vertices_cm=outer)
        == "continuous_magnet_layer_outer_boundary"
    )


def test_non_magnet_surface_adjacency_is_rejected():
    with pytest.raises(ValueError, match="adjacency"):
        classify_surface_role(
            adjacent_volume_ids=[7, 8],
            vertices_cm=np.zeros((1, 3, 3)),
        )


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_surface_manifest_captures_closed_boundary_and_selects_inner_subset(
    tmp_path, monkeypatch
):
    dagmc = tmp_path / "dagmc.h5m"
    dagmc.write_bytes(b"immutable-h5m")
    dagmc_hash = hashlib.sha256(dagmc.read_bytes()).hexdigest()
    topology = {
        "schema": "parastell.dagmc_topology_audit/v1.1.0",
        "topology_gate_pass": True,
        "h5m_unchanged": True,
        "surface_senses": [
            {
                "surface_id": 10,
                "adjacent_volume_ids": [8, 9],
                "forward_volume_id": 8,
                "reverse_volume_id": 9,
                "sense_pass": True,
            },
            {
                "surface_id": 11,
                "adjacent_volume_ids": [9],
                "forward_volume_id": 9,
                "reverse_volume_id": None,
                "sense_pass": True,
            },
            {
                "surface_id": 12,
                "adjacent_volume_ids": [9],
                "forward_volume_id": 9,
                "reverse_volume_id": None,
                "sense_pass": True,
            },
        ],
    }
    inventory = {
        "schema": "parastell.continuous_magnet_volume_inventory/v1.0.0",
        "status": "PASS",
        "continuous_magnet": {
            "volume_id": 9,
            "complete_boundary_closed": True,
            "complete_boundary_surface_ids": [10, 11, 12],
            "bounding_box_cm": {
                "lower": [1.0, 1.0, -1.0],
                "upper": [3.0, 3.0, 1.0],
            },
        },
    }
    writeback = {
        "schema": "parastell.continuous_radial_direct90_h5m_writeback/v1.0.0",
        "pass": True,
        "semantic_roles_by_global_volume_id": list(COMPONENT_ORDER),
    }
    topology_path = tmp_path / "topology.json"
    inventory_path = tmp_path / "inventory.json"
    writeback_path = tmp_path / "writeback.json"
    topology_hash = _write_json(topology_path, topology)
    inventory_hash = _write_json(inventory_path, inventory)
    writeback_hash = _write_json(writeback_path, writeback)

    schema_surfaces = tuple(
        SimpleNamespace(
            surface_id=surface_id,
            openmc_normal_sign=1,
            area_cm2=2.0,
        )
        for surface_id in (10, 11, 12)
    )
    inner = np.asarray([[[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 1.0]]])
    phi_zero = np.asarray(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 1.0]]]
    )
    outer = np.asarray([[[2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [2.0, 3.0, 1.0]]])
    fake = SimpleNamespace(
        envelope=SimpleNamespace(
            surfaces=schema_surfaces,
            surface_ids=(10, 11, 12),
        ),
        faceted_surfaces=tuple(
            SimpleNamespace(surface_id=surface_id, triangles_cm=triangles)
            for surface_id, triangles in zip(
                (10, 11, 12), (inner, phi_zero, outer)
            )
        ),
    )
    monkeypatch.setattr(
        "scripts.build_continuous_magnet_surface_manifest._extract_envelopes_from_h5m",
        lambda *_: (fake,),
    )
    result = build_manifest(
        dagmc,
        topology_path,
        inventory_path,
        writeback_path,
        expected_dagmc_sha256=dagmc_hash,
        expected_topology_sha256=topology_hash,
        expected_inventory_sha256=inventory_hash,
        expected_writeback_sha256=writeback_hash,
    )
    assert result["capture_bank"]["surface_ids"] == [10, 11, 12]
    assert result["capture_bank"]["closed"] is True
    assert result["incoming_replay_interface"]["surface_ids"] == [10]
    assert (
        result["incoming_replay_interface"]["closed_manifold_claim"] is False
    )
    assert result["physical_h5m_mutation"] is False
