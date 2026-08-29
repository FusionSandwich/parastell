"""Derive an OpenMC-ready 18-magnet surface manifest from qualified DAGMC.

The H5M is read only. Surface membership and outward-normal signs come from
PyDAGMC forward/reverse topology through ``extract_closed_envelope``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from parastell.openmc16_export import _extract_envelopes_from_h5m
from parastell.reference_geometry import sha256_file


SCHEMA = "parastell.parametric_magnet_surface_manifest/v1.0.0"
EXPECTED_SEMANTIC_ROLES = (
    "chamber",
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "vacuum_gap",
    *(f"magnet-{index:04d}" for index in range(18)),
)


def _frame(
    bounding_box: dict[str, Any],
) -> tuple[list[float], list[float], list[float], list[float]]:
    lower = np.asarray(bounding_box["lower"], dtype=float)
    upper = np.asarray(bounding_box["upper"], dtype=float)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("magnet bounding box must contain two 3-vectors")
    center = 0.5 * (lower + upper)
    radial = center.copy()
    radial[2] = 0.0
    norm = np.linalg.norm(radial)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("magnet bounding box cannot define a local frame")
    radial /= norm
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
    topology_inventory: Path,
    h5m_writeback: Path,
    *,
    expected_dagmc_sha256: str,
    expected_topology_inventory_sha256: str,
    expected_h5m_writeback_sha256: str,
) -> dict[str, Any]:
    dagmc = dagmc.resolve(strict=True)
    topology_inventory = topology_inventory.resolve(strict=True)
    if sha256_file(dagmc) != expected_dagmc_sha256:
        raise ValueError("DAGMC H5M hash mismatch")
    if sha256_file(topology_inventory) != expected_topology_inventory_sha256:
        raise ValueError("topology-inventory hash mismatch")
    h5m_writeback = h5m_writeback.resolve(strict=True)
    if sha256_file(h5m_writeback) != expected_h5m_writeback_sha256:
        raise ValueError("H5M-writeback hash mismatch")
    writeback = json.loads(h5m_writeback.read_text(encoding="utf-8"))
    identities = writeback.get("magnet_identity_by_global_volume_id")
    if (
        writeback.get("schema")
        != "parastell.parametric_direct90_h5m_writeback/v1.0.0"
        or writeback.get("pass") is not True
        or writeback.get("h5m_sha256") != expected_dagmc_sha256
        or tuple(writeback.get("semantic_roles_by_global_volume_id", ()))
        != EXPECTED_SEMANTIC_ROLES
        or not isinstance(identities, list)
        or len(identities) != 18
    ):
        raise ValueError("H5M writeback lacks the exact semantic mapping")
    for index, identity in enumerate(identities):
        if (
            identity.get("magnet_id") != f"magnet-{index:04d}"
            or identity.get("volume_global_id") != index + 9
            or identity.get("unique_match") is not True
            or identity.get("pass") is not True
        ):
            raise ValueError("H5M writeback magnet mapping is invalid")
    inventory = json.loads(topology_inventory.read_text(encoding="utf-8"))
    magnets = inventory.get("magnets")
    if (
        inventory.get("schema") != "parastell.original_magnet_envelopes/v1.1.0"
        or inventory.get("raw_h5m_sha256") != expected_dagmc_sha256
        or inventory.get("inventory_gate_pass") is not True
        or inventory.get("all_envelopes_close") is not True
        or inventory.get("magnet_count") != 18
        or not isinstance(magnets, list)
        or len(magnets) != 18
    ):
        raise ValueError("topology inventory has not qualified all 18 magnets")

    requests = []
    for expected_index, row in enumerate(magnets):
        magnet_id = f"magnet-{expected_index:04d}"
        surface_ids = row.get("outer_envelope_surface_ids")
        if (
            row.get("magnet_id") != magnet_id
            or row.get("volume_id") != expected_index + 9
            or row.get("material_tag") != "magnets"
            or row.get("physical_role") != "original_homogenized_magnet_solid"
            or row.get("coupling_interface") != "homogenized_magnet_outer"
            or row.get("casing_winding_split") is not False
            or row.get("closed") is not True
            or not isinstance(surface_ids, list)
            or row.get("surface_count") != len(surface_ids)
        ):
            raise ValueError("magnet identity/order/closure is invalid")
        centroid, plasma, toroidal, poloidal = _frame(row["bounding_box_cm"])
        requests.append(
            {
                "volume_id": int(row["volume_id"]),
                "envelope_id": f"{magnet_id}-outer-envelope",
                "magnet_id": magnet_id,
                "centroid_cm": centroid,
                "plasma_direction_global": plasma,
                "toroidal_direction_global": toroidal,
                "poloidal_direction_global": poloidal,
            }
        )

    envelopes = _extract_envelopes_from_h5m(dagmc, requests)
    if len(envelopes) != 18:
        raise ValueError("envelope extractor did not return 18 magnets")
    rows = []
    all_surface_ids: list[int] = []
    for source, extracted, request in zip(
        magnets, envelopes, requests, strict=True
    ):
        envelope = extracted.envelope
        surface_ids = sorted(int(value) for value in envelope.surface_ids)
        declared = sorted(
            int(value) for value in source["outer_envelope_surface_ids"]
        )
        if int(envelope.dagmc_volume_id) != int(source["volume_id"]):
            raise ValueError("extracted envelope volume identity mismatch")
        if envelope.magnet_component != source["magnet_id"]:
            raise ValueError("extracted envelope magnet identity mismatch")
        if surface_ids != declared:
            raise ValueError("extracted and qualified surface sets disagree")
        surfaces = [
            {
                "dagmc_surface_id": int(surface.surface_id),
                "magnet_outward_normal_multiplier": int(
                    surface.openmc_normal_sign
                ),
                "area_cm2": float(surface.area_cm2),
                "role": str(surface.role),
            }
            for surface in sorted(
                envelope.surfaces, key=lambda item: int(item.surface_id)
            )
        ]
        if any(
            row["magnet_outward_normal_multiplier"] not in {-1, 1}
            for row in surfaces
        ):
            raise ValueError("invalid DAGMC outward-normal multiplier")
        if any(
            not np.isfinite(row["area_cm2"]) or row["area_cm2"] <= 0.0
            for row in surfaces
        ):
            raise ValueError("invalid extracted surface area")
        rows.append(
            {
                "magnet": source["magnet_id"],
                "dagmc_volume_id": int(source["volume_id"]),
                "centroid_cm": list(request["centroid_cm"]),
                "dagmc_surface_ids": surface_ids,
                "surfaces": surfaces,
                "closed": True,
                "coupling_interface": "homogenized_magnet_outer_boundary",
            }
        )
        all_surface_ids.extend(surface_ids)
    if len(all_surface_ids) != len(set(all_surface_ids)):
        raise ValueError("magnet envelopes share one or more DAGMC surfaces")
    if sha256_file(dagmc) != expected_dagmc_sha256:
        raise RuntimeError("DAGMC H5M changed during surface extraction")
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "dagmc_sha256": expected_dagmc_sha256,
        "topology_inventory_sha256": expected_topology_inventory_sha256,
        "h5m_writeback_sha256": expected_h5m_writeback_sha256,
        "selected_material_group": "mat:magnets",
        "coupling_interface": "homogenized_magnet_outer_boundary",
        "magnet_count": 18,
        "all_envelopes_close": True,
        "all_surface_ids": sorted(all_surface_ids),
        "magnets": rows,
        "physical_h5m_mutation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc_h5m", type=Path)
    parser.add_argument("topology_inventory", type=Path)
    parser.add_argument("h5m_writeback", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-dagmc-sha256", required=True)
    parser.add_argument("--expected-topology-inventory-sha256", required=True)
    parser.add_argument("--expected-h5m-writeback-sha256", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    payload = build_manifest(
        args.dagmc_h5m,
        args.topology_inventory,
        args.h5m_writeback,
        expected_dagmc_sha256=args.expected_dagmc_sha256,
        expected_topology_inventory_sha256=(
            args.expected_topology_inventory_sha256
        ),
        expected_h5m_writeback_sha256=args.expected_h5m_writeback_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
