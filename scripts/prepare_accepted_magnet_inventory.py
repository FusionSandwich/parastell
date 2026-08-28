"""Prepare a geometry-derived magnet inventory candidate for root acceptance.

This command never emits the separate root-acceptance receipt.  The root
operator must review and independently hash-pin the candidate before a strict
surface-run audit can use it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from parastell.openmc16_export import _extract_envelopes_from_h5m


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_create_only(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _frame(centroid) -> tuple[list[float], list[float], list[float]]:
    point = np.asarray(centroid, dtype=float)
    radial = point.copy()
    radial[2] = 0.0
    length = np.linalg.norm(radial)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("magnet centroid cannot define a toroidal frame")
    radial /= length
    plasma = -radial
    toroidal = np.asarray([-radial[1], radial[0], 0.0])
    poloidal = np.cross(plasma, toroidal)
    return plasma.tolist(), toroidal.tolist(), poloidal.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc_h5m", type=Path)
    parser.add_argument("surface_manifest", type=Path)
    parser.add_argument("source_cad", type=Path)
    parser.add_argument("geometry_gate", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    dagmc = args.dagmc_h5m.resolve(strict=True)
    surface_manifest_path = args.surface_manifest.resolve(strict=True)
    source_cad = args.source_cad.resolve(strict=True)
    gate_path = args.geometry_gate.resolve(strict=True)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    dagmc_hash = _sha256(dagmc)
    if (
        gate.get("geometry_gate_pass") is not True
        or gate.get("dagmc", {}).get("unchanged") is not True
        or gate.get("dagmc", {}).get("sha256_before") != dagmc_hash
        or gate.get("dagmc", {}).get("sha256_after") != dagmc_hash
    ):
        raise ValueError("native geometry gate does not accept the exact H5M")

    surface_manifest = json.loads(
        surface_manifest_path.read_text(encoding="utf-8")
    )
    if surface_manifest.get("dagmc_sha256") != dagmc_hash:
        raise ValueError("surface manifest H5M hash mismatch")
    magnets = surface_manifest.get("magnets")
    if not isinstance(magnets, list) or not magnets:
        raise ValueError("surface manifest has no magnets")

    requests = []
    components = []
    for row in sorted(magnets, key=lambda item: int(item["dagmc_volume_id"])):
        volume_id = int(row["dagmc_volume_id"])
        magnet_id = str(row.get("magnet", "")).strip()
        if volume_id <= 0 or not magnet_id:
            raise ValueError("surface manifest has an invalid magnet identity")
        plasma, toroidal, poloidal = _frame(row["centroid_cm"])
        request = {
            "volume_id": volume_id,
            "envelope_id": f"{magnet_id}-outer-envelope",
            "magnet_id": magnet_id,
            "plasma_direction_global": plasma,
            "toroidal_direction_global": toroidal,
            "poloidal_direction_global": poloidal,
        }
        requests.append(request)
        components.append(
            {
                "magnet_id": magnet_id,
                "component_id": f"{magnet_id}-homogenized-winding-pack",
                "dagmc_volume_id": volume_id,
                "material_tag": "mat:winding_pack",
                "surface_ids": sorted(
                    int(value) for value in row["dagmc_surface_ids"]
                ),
                "source_cad": {
                    "path": str(source_cad),
                    "sha256": _sha256(source_cad),
                },
            }
        )

    envelopes = _extract_envelopes_from_h5m(dagmc, requests)
    fingerprints = {
        envelope.envelope.metadata["canonical_geometry_fingerprint"]
        for envelope in envelopes
    }
    if len(fingerprints) != 1:
        raise ValueError(
            "envelopes disagree on canonical geometry fingerprint"
        )
    envelope_by_volume = {
        int(envelope.envelope.dagmc_volume_id): sorted(
            int(value) for value in envelope.envelope.surface_ids
        )
        for envelope in envelopes
    }
    for component in components:
        if (
            envelope_by_volume.get(component["dagmc_volume_id"])
            != component["surface_ids"]
        ):
            raise ValueError("surface manifest disagrees with H5M topology")

    requests_path = output / "MAGNET_ENVELOPE_REQUESTS.json"
    _write_json_create_only(
        requests_path,
        {
            "schema": "parastell.magnet_envelope_requests/v1.0.0",
            "frame_semantics": (
                "magnet-centroid cylindrical basis; outward normals remain "
                "DAGMC forward/reverse topology derived"
            ),
            "envelope_requests": requests,
        },
    )
    inventory_path = output / "ACCEPTED_MAGNET_INVENTORY_CANDIDATE.json"
    _write_json_create_only(
        inventory_path,
        {
            "schema": "parastell.accepted_magnet_component_inventory/v1.0.0",
            "geometry_gate_status": "PASS",
            "dagmc_sha256": dagmc_hash,
            "canonical_geometry_fingerprint": next(iter(fingerprints)),
            "magnet_material_tags": ["mat:winding_pack"],
            "components": components,
            "evidence": {
                "native_geometry_gate_path": str(gate_path),
                "native_geometry_gate_sha256": _sha256(gate_path),
                "surface_manifest_path": str(surface_manifest_path),
                "surface_manifest_sha256": _sha256(surface_manifest_path),
            },
        },
    )
    receipt = {
        "candidate_inventory": {
            "path": str(inventory_path),
            "sha256": _sha256(inventory_path),
        },
        "envelope_requests": {
            "path": str(requests_path),
            "sha256": _sha256(requests_path),
        },
        "root_acceptance_emitted": False,
    }
    receipt_path = output / "INVENTORY_PREPARATION_RECEIPT.json"
    _write_json_create_only(receipt_path, receipt)
    print(receipt_path)


if __name__ == "__main__":
    main()
