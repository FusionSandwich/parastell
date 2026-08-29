"""Export the authoritative nine-volume direct-90 radial model to DAGMC.

The source and physical-audit packets are SHA-256 bound. The H5M contains no
instrumentation solids and is only read after export to write a topology
receipt. Native DAGMC and OpenMC gates remain mandatory successors.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from parastell.continuous_radial_contract import COMPONENT_ORDER
from parastell.continuous_radial_contract import H5M_WRITEBACK_SCHEMA
from parastell.continuous_radial_contract import MATERIAL_TAGS
from parastell.continuous_radial_contract import SOURCE_AUDIT_SCHEMA
from parastell.continuous_radial_contract import SOURCE_AUDIT_SEAL_SCHEMA
from parastell.continuous_radial_contract import sha256_file
from parastell.continuous_radial_contract import validate_source_artifacts
from parastell.continuous_radial_contract import validate_source_manifest
from scripts import export_direct90_imprinted_dagmc as topology_backend


LEGACY_TO_CANONICAL = {
    "hts": "high_temperature_shield",
    "lts": "low_temperature_shield",
    "gap": "vacuum_gap",
}
BACKEND_ORDER = (
    "chamber",
    "first_wall",
    "breeder",
    "back_wall",
    "hts",
    "vacuum_vessel",
    "lts",
    "gap",
    "magnets",
)


def _load_bound_json(path: Path, digest: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"hash mismatch for {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON control is not an object: {path}")
    return value


def _validate_source_packet(args: argparse.Namespace) -> tuple[dict, dict]:
    source = args.source_root.resolve()
    manifest_path = source / "PARAMETRIC_GEOMETRY_MANIFEST.json"
    manifest = _load_bound_json(manifest_path, args.manifest_sha256)
    validate_source_manifest(manifest)
    validate_source_artifacts(source, manifest)
    audit = _load_bound_json(args.audit_path.resolve(), args.audit_sha256)
    seal = _load_bound_json(
        args.audit_seal_path.resolve(), args.audit_seal_sha256
    )
    if (
        sha256_file(args.audit_producer_path.resolve())
        != args.audit_producer_sha256
    ):
        raise ValueError("source-audit producer hash mismatch")
    if (
        audit.get("schema") != SOURCE_AUDIT_SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("source_manifest_sha256") != args.manifest_sha256
        or audit.get("source_immutable") is not True
        or audit.get("summary", {}).get("component_count") != 9
        or audit.get("summary", {}).get("adjacent_exact_check_count") != 8
        or audit.get("summary", {}).get("nonadjacent_construction_proof_count")
        != 28
        or audit.get("summary", {}).get("invalid_component_count") != 0
        or audit.get("summary", {}).get("adjacent_overlap_failure_count") != 0
        or audit.get("summary", {}).get("nonadjacent_proof_failure_count") != 0
    ):
        raise ValueError("continuous source-CAD audit did not pass")
    if (
        seal.get("schema") != SOURCE_AUDIT_SEAL_SCHEMA
        or seal.get("status") != "PASS"
        or seal.get("report_sha256") != args.audit_sha256
        or seal.get("source_manifest_sha256") != args.manifest_sha256
        or seal.get("source_immutable") is not True
    ):
        raise ValueError("continuous source-CAD audit seal is invalid")
    return manifest, audit


def _canonicalize_roles(value: Any) -> Any:
    if isinstance(value, str):
        return LEGACY_TO_CANONICAL.get(value, value)
    if isinstance(value, list):
        return [_canonicalize_roles(item) for item in value]
    if isinstance(value, dict):
        return {
            "__".join(
                LEGACY_TO_CANONICAL.get(part, part) for part in key.split("__")
            ): _canonicalize_roles(item)
            for key, item in value.items()
        }
    return value


def _writeback(h5m_path: Path) -> dict[str, Any]:
    import pydagmc

    before = sha256_file(h5m_path)
    model = pydagmc.Model(str(h5m_path))
    volumes = [model.volumes_by_id[key] for key in sorted(model.volumes_by_id)]
    rows = []
    for index, (role, expected_material, volume) in enumerate(
        zip(COMPONENT_ORDER, MATERIAL_TAGS, volumes, strict=True), start=1
    ):
        surface_ids = sorted(int(surface.id) for surface in volume.surfaces)
        actual_material = str(getattr(volume, "material", "") or "")
        material_match = (
            actual_material.removeprefix("mat:") == expected_material
        )
        rows.append(
            {
                "volume_global_id": int(volume.id),
                "semantic_role": role,
                "expected_material": expected_material,
                "actual_material": actual_material,
                "surface_ids": surface_ids,
                "pass": int(volume.id) == index
                and material_match
                and bool(surface_ids),
            }
        )
    after = sha256_file(h5m_path)
    passed = (
        len(volumes) == 9
        and before == after
        and all(row["pass"] for row in rows)
    )
    if not passed:
        raise RuntimeError("H5M semantic writeback audit failed")
    return {
        "schema": H5M_WRITEBACK_SCHEMA,
        "status": "H5M_WRITEBACK_PASS_NATIVE_GATES_PENDING",
        "pass": True,
        "h5m_sha256": before,
        "h5m_unchanged": True,
        "volume_count": 9,
        "semantic_roles_by_global_volume_id": list(COMPONENT_ORDER),
        "continuous_magnet_volume_global_id": 9,
        "volumes": rows,
    }


def export(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    manifest, _ = _validate_source_packet(args)
    topology_backend._validate_frozen_mesh_controls(
        args.min_mesh_size_cm,
        args.max_mesh_size_cm,
        args.algorithm,
        args.threads,
    )
    topology_backend._validate_thread_environment()
    import cad_to_dagmc
    import cadquery as cq

    source = args.source_root.resolve()
    solids = []
    for role in COMPONENT_ORDER:
        values = (
            cq.importers.importStep(str(source / f"{role}.step"))
            .solids()
            .vals()
        )
        if len(values) != 1:
            raise RuntimeError(f"{role} STEP contains {len(values)} solids")
        solids.append(values[0])
    expected_canonical = {
        role: float(manifest["component_evidence"][role]["volume_cm3"])
        for role in COMPONENT_ORDER
    }
    expected_backend = {
        backend: expected_canonical[LEGACY_TO_CANONICAL.get(backend, backend)]
        for backend in BACKEND_ORDER
    }
    geometry = cq.Compound.makeCompound(solids)
    output.mkdir(parents=True)
    h5m_path = output / "dagmc.h5m"
    started = time.time()
    gmsh = cad_to_dagmc.init_gmsh()
    try:
        for option in (
            "General.NumThreads",
            "Mesh.MaxNumThreads1D",
            "Mesh.MaxNumThreads2D",
            "Mesh.MaxNumThreads3D",
        ):
            gmsh.option.setNumber(option, args.threads)
        _, imported = cad_to_dagmc.get_volumes(
            gmsh, geometry, method="in memory"
        )
        imported = [(int(dim), int(tag)) for dim, tag in imported]
        masses = [float(gmsh.model.occ.getMass(*item)) for item in imported]
        indices, assignment = topology_backend._unique_volume_assignment(
            masses, expected_backend
        )
        assignment = [
            {
                **row,
                "component": COMPONENT_ORDER[index],
                "material": MATERIAL_TAGS[index],
            }
            for index, row in enumerate(assignment)
        ]
        ordered = [imported[index] for index in indices]
        fragmented = topology_backend._fragment_one_to_one(gmsh, ordered)
        topology = _canonicalize_roles(
            topology_backend._audit_fragmented_surface_topology(
                gmsh, fragmented
            )
        )
        if topology.get("volume_count") != 9:
            raise RuntimeError("pre-mesh topology is not nine volumes")
        premesh = {
            "schema": "parastell.continuous_radial_premesh_topology/v1.0.0",
            "status": "PREMESH_9_VOLUME_TOPOLOGY_PASS",
            "transport_eligible": False,
            "source_manifest_sha256": args.manifest_sha256,
            "component_order": list(COMPONENT_ORDER),
            "magnet_representation": "continuous_30_cm_radial_envelope",
            "topology": topology,
        }
        (output / "PREMESH_TOPOLOGY.json").write_text(
            json.dumps(premesh, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cad_to_dagmc.set_sizes_for_mesh(
            gmsh,
            min_mesh_size=args.min_mesh_size_cm,
            max_mesh_size=args.max_mesh_size_cm,
            mesh_algorithm=args.algorithm,
        )
        gmsh.model.mesh.generate(dim=2)
        vertices, triangles = cad_to_dagmc.mesh_to_vertices_and_triangles(
            fragmented
        )
    finally:
        gmsh.finalize()
    cad_to_dagmc.vertices_to_h5m(
        vertices, triangles, list(MATERIAL_TAGS), h5m_filename=h5m_path
    )
    writeback = _writeback(h5m_path)
    writeback_path = output / "H5M_WRITEBACK_AUDIT.json"
    writeback_path.write_text(
        json.dumps(writeback, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "parastell.continuous_radial_direct90_dagmc/v1.0.0",
        "status": "EXPORTED_NATIVE_DAGMC_AND_OPENMC_GATES_PENDING",
        "transport_eligible": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": args.manifest_sha256,
        "source_audit_sha256": args.audit_sha256,
        "producer_sha256": sha256_file(Path(__file__).resolve()),
        "geometry": {
            "extent_degrees": 90.0,
            "direct_parastell_full_period": True,
            "component_order": list(COMPONENT_ORDER),
            "volume_count": 9,
            "continuous_magnet_volume_global_id": 9,
            "explicit_swept_coils": False,
            "physical_h5m_mutation": False,
        },
        "meshing": {
            "imprint": True,
            "minimum_cm": args.min_mesh_size_cm,
            "maximum_cm": args.max_mesh_size_cm,
            "algorithm": args.algorithm,
            "threads": args.threads,
        },
        "import_assignment": assignment,
        "h5m": {
            "path": str(h5m_path),
            "bytes": h5m_path.stat().st_size,
            "sha256": sha256_file(h5m_path),
        },
        "premesh_topology_sha256": sha256_file(
            output / "PREMESH_TOPOLOGY.json"
        ),
        "h5m_writeback_sha256": sha256_file(writeback_path),
        "elapsed_seconds": time.time() - started,
        "automatic_successor_launched": False,
    }
    (output / "DAGMC_EXPORT_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--audit-path", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--audit-seal-path", type=Path, required=True)
    parser.add_argument("--audit-seal-sha256", required=True)
    parser.add_argument("--audit-producer-path", type=Path, required=True)
    parser.add_argument("--audit-producer-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-mesh-size-cm", type=float, default=5.0)
    parser.add_argument("--max-mesh-size-cm", type=float, default=20.0)
    parser.add_argument("--algorithm", type=int, default=1)
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
