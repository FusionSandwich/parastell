#!/usr/bin/env python3
"""Build the complete continuous-magnet boundary and replay-interface manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from parastell.continuous_radial_contract import COMPONENT_ORDER
from parastell.continuous_radial_contract import COUPLING_INTERFACE
from parastell.continuous_radial_contract import H5M_WRITEBACK_SCHEMA
from parastell.continuous_radial_contract import sha256_file
from parastell.openmc16_export import _extract_envelopes_from_h5m


SCHEMA = "parastell.continuous_magnet_surface_manifest/v1.0.0"
PERIODIC_PLANE_TOLERANCE_CM = 1.0e-5


def classify_surface_role(
    *, adjacent_volume_ids: list[int], vertices_cm: np.ndarray
) -> str:
    adjacent = set(int(value) for value in adjacent_volume_ids)
    if adjacent == {8, 9}:
        return COUPLING_INTERFACE
    vertices = np.asarray(vertices_cm, dtype=float).reshape((-1, 3))
    if adjacent != {9} or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("continuous magnet surface adjacency is invalid")
    if np.max(np.abs(vertices[:, 1])) <= PERIODIC_PLANE_TOLERANCE_CM:
        return "periodic_end_phi_0"
    if np.max(np.abs(vertices[:, 0])) <= PERIODIC_PLANE_TOLERANCE_CM:
        return "periodic_end_phi_90"
    return "continuous_magnet_layer_outer_boundary"


def _frame(bounds: dict[str, Any]) -> tuple[list[float], ...]:
    lower = np.asarray(bounds["lower"], dtype=float)
    upper = np.asarray(bounds["upper"], dtype=float)
    center = 0.5 * (lower + upper)
    radial = center.copy()
    radial[2] = 0.0
    radial /= np.linalg.norm(radial)
    plasma = -radial
    toroidal = np.asarray([-radial[1], radial[0], 0.0])
    poloidal = np.cross(plasma, toroidal)
    return (
        center.tolist(),
        plasma.tolist(),
        toroidal.tolist(),
        poloidal.tolist(),
    )


def build_manifest(
    dagmc: Path,
    topology_path: Path,
    inventory_path: Path,
    writeback_path: Path,
    *,
    expected_dagmc_sha256: str,
    expected_topology_sha256: str,
    expected_inventory_sha256: str,
    expected_writeback_sha256: str,
) -> dict[str, Any]:
    dagmc = dagmc.resolve(strict=True)
    before = sha256_file(dagmc)
    bindings = (
        (dagmc, expected_dagmc_sha256),
        (topology_path, expected_topology_sha256),
        (inventory_path, expected_inventory_sha256),
        (writeback_path, expected_writeback_sha256),
    )
    if any(sha256_file(path) != digest for path, digest in bindings):
        raise ValueError("surface-manifest input hash mismatch")
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    writeback = json.loads(writeback_path.read_text(encoding="utf-8"))
    magnet = inventory.get("continuous_magnet")
    if (
        topology.get("schema") != "parastell.dagmc_topology_audit/v1.1.0"
        or topology.get("topology_gate_pass") is not True
        or topology.get("h5m_unchanged") is not True
        or inventory.get("schema")
        != "parastell.continuous_magnet_volume_inventory/v1.0.0"
        or inventory.get("status") != "PASS"
        or not isinstance(magnet, dict)
        or magnet.get("volume_id") != 9
        or magnet.get("complete_boundary_closed") is not True
        or writeback.get("schema") != H5M_WRITEBACK_SCHEMA
        or writeback.get("pass") is not True
        or writeback.get("semantic_roles_by_global_volume_id")
        != list(COMPONENT_ORDER)
    ):
        raise ValueError("continuous magnet topology packet is invalid")
    center, plasma, toroidal, poloidal = _frame(magnet["bounding_box_cm"])
    extracted = _extract_envelopes_from_h5m(
        dagmc,
        [
            {
                "volume_id": 9,
                "envelope_id": "continuous-magnet-layer-complete-boundary",
                "magnet_id": "continuous-magnet-layer",
                "plasma_direction_global": plasma,
                "toroidal_direction_global": toroidal,
                "poloidal_direction_global": poloidal,
            }
        ],
    )[0]
    envelope = extracted.envelope
    declared = sorted(
        int(value) for value in magnet["complete_boundary_surface_ids"]
    )
    observed = sorted(int(value) for value in envelope.surface_ids)
    if observed != declared:
        raise ValueError(
            "complete continuous-magnet boundary surface set changed"
        )
    sense_by_surface = {
        int(row["surface_id"]): row
        for row in topology.get("surface_senses", ())
    }
    rows = []
    selected_inner = []
    role_counts: dict[str, int] = {}
    for schema_surface, faceted in zip(
        sorted(envelope.surfaces, key=lambda item: item.surface_id),
        sorted(extracted.faceted_surfaces, key=lambda item: item.surface_id),
        strict=True,
    ):
        sense = sense_by_surface.get(int(schema_surface.surface_id))
        if not sense or sense.get("sense_pass") is not True:
            raise ValueError(
                "surface lacks valid native forward/reverse topology"
            )
        adjacent = [
            int(value) for value in sense.get("adjacent_volume_ids", ())
        ]
        role = classify_surface_role(
            adjacent_volume_ids=adjacent,
            vertices_cm=np.asarray(faceted.triangles_cm),
        )
        role_counts[role] = role_counts.get(role, 0) + 1
        if role == COUPLING_INTERFACE:
            selected_inner.append(int(schema_surface.surface_id))
        rows.append(
            {
                "dagmc_surface_id": int(schema_surface.surface_id),
                "role": role,
                "adjacent_volume_ids": adjacent,
                "forward_volume_id": sense.get("forward_volume_id"),
                "reverse_volume_id": sense.get("reverse_volume_id"),
                "magnet_outward_normal_multiplier": int(
                    schema_surface.openmc_normal_sign
                ),
                "area_cm2": float(schema_surface.area_cm2),
                "facet_count": len(faceted.triangles_cm),
                "complete_boundary_capture": True,
                "selected_for_incoming_replay": role == COUPLING_INTERFACE,
            }
        )
    if not selected_inner or len(rows) != len(declared):
        raise ValueError(
            "continuous magnet inner-interface selection is empty/incomplete"
        )
    after = sha256_file(dagmc)
    if after != before:
        raise RuntimeError("DAGMC H5M changed during surface extraction")
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "dagmc_sha256": before,
        "topology_sha256": expected_topology_sha256,
        "inventory_sha256": expected_inventory_sha256,
        "writeback_sha256": expected_writeback_sha256,
        "geometry": {
            "volume_count": 9,
            "continuous_magnet_volume_id": 9,
            "continuous_magnet_centroid_cm": center,
            "explicit_swept_coils": False,
        },
        "capture_bank": {
            "surface_ids": declared,
            "surface_count": len(declared),
            "scope": "complete_closed_continuous_magnet_volume_boundary",
            "records_all_entering_and_leaving_crossings": True,
            "closed": True,
        },
        "incoming_replay_interface": {
            "coupling_interface": COUPLING_INTERFACE,
            "surface_ids": sorted(selected_inner),
            "surface_count": len(selected_inner),
            "adjacency": {"from_volume_id": 8, "to_volume_id": 9},
            "selection": "mu < 0 using the magnet-volume outward normal",
            "closed_manifold_claim": False,
            "closure_status": "NOT_APPLICABLE_SUBSET_OF_CLOSED_VOLUME_BOUNDARY",
        },
        "surface_role_counts": dict(sorted(role_counts.items())),
        "surfaces": rows,
        "normal_source": "DAGMC forward/reverse topology",
        "physical_h5m_mutation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc", type=Path)
    parser.add_argument("topology", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("writeback", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-dagmc-sha256", required=True)
    parser.add_argument("--expected-topology-sha256", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--expected-writeback-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"create-only output exists: {args.output}")
    payload = build_manifest(
        args.dagmc,
        args.topology,
        args.inventory,
        args.writeback,
        expected_dagmc_sha256=args.expected_dagmc_sha256,
        expected_topology_sha256=args.expected_topology_sha256,
        expected_inventory_sha256=args.expected_inventory_sha256,
        expected_writeback_sha256=args.expected_writeback_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
