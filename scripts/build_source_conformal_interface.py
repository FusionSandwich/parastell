"""Extract an immutable source-master interface and create fail-closed receipts.

This bounded producer creates the interface constraint packet only.  It does
not claim a physical DAGMC candidate: the remaining chamber/first-wall and
radial-layer boundaries must be constructed around the exact packet by a
separate create-only geometry campaign.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import parastell.source_conformal_interface as source_conformal  # noqa: E402
from parastell.source_conformal_interface import (  # noqa: E402
    P4_RECEIPT_SCHEMA,
    SOURCE_CONFORMAL_MANIFEST_SCHEMA,
    audit_shared_interface_topology,
    audit_source_conformal_containment_arrays,
    extract_source_master_interface,
    load_indexed_source_mesh,
    write_compact_json_create_only,
)
from parastell.source_geometry_identity import (  # noqa: E402
    source_mesh_fingerprint,
)


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only output exists: {output}")
    source_path = args.source_mesh.resolve()
    reference_path = args.reference_source_mesh.resolve()
    if _hash_file(source_path) != args.source_mesh_sha256:
        raise ValueError("source mesh hash mismatch")
    if _hash_file(reference_path) != args.reference_source_mesh_sha256:
        raise ValueError("reference source mesh hash mismatch")
    source_before = _hash_file(source_path)
    reference_before = _hash_file(reference_path)
    source = load_indexed_source_mesh(source_path)
    reference = load_indexed_source_mesh(reference_path)
    master = extract_source_master_interface(
        source["vertices_cm"],
        source["tetrahedra"],
        periodic_cut_plane_angles_degrees=args.periodic_cut_plane_angles_degrees,
        geometric_tolerance_cm=args.shared_interface_tolerance_cm,
    )
    shared = audit_shared_interface_topology(
        master,
        shared_triangles=master["interface_triangles"],
        surface_owners=[
            ("chamber", "first_wall") for _ in master["interface_triangles"]
        ],
    )
    containment = audit_source_conformal_containment_arrays(
        source["vertices_cm"],
        source["tetrahedra"],
        source["tagged_volumes_cm3"],
        source["source_strengths_n_per_s"],
        reference_vertices_cm=reference["vertices_cm"],
        reference_tetrahedra=reference["tetrahedra"],
        reference_tagged_volumes_cm3=reference["tagged_volumes_cm3"],
        reference_source_strengths_n_per_s=reference[
            "source_strengths_n_per_s"
        ],
        master_interface=master,
        shared_interface=shared,
    )
    if (
        _hash_file(source_path) != source_before
        or _hash_file(reference_path) != reference_before
    ):
        raise RuntimeError(
            "source input changed during master-interface extraction"
        )

    output.mkdir(parents=True)
    packet_path = output / "MASTER_SOURCE_INTERFACE.npz"
    np.savez_compressed(
        packet_path,
        vertices_cm=np.asarray(source["vertices_cm"], dtype="<f8"),
        interface_triangles=np.asarray(
            master["interface_triangles"], dtype="<u8"
        ),
        interface_normal_unit_vectors=np.asarray(
            master["interface_normal_unit_vectors"], dtype="<f8"
        ),
        periodic_seam_triangles=np.asarray(
            master["periodic_seam_triangles"], dtype="<u8"
        ),
    )
    reported_output = (
        Path(args.reported_output_root)
        if args.reported_output_root is not None
        else output
    )
    packet_identity = {
        "path": str(reported_output / packet_path.name),
        "size_bytes": packet_path.stat().st_size,
        "sha256": _hash_file(packet_path),
    }
    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": SOURCE_CONFORMAL_MANIFEST_SCHEMA,
        "created_utc": created,
        "status": "MASTER_INTERFACE_READY_OUTER_LAYER_PRODUCER_REQUIRED",
        "transport_eligible": False,
        "construction_mode": "IMMUTABLE_SOURCE_MESH_MASTER_INTERFACE",
        "source_mesh": {
            "path": str(source_path),
            "sha256": source_before,
            "semantic_identity": source_mesh_fingerprint(source_path),
        },
        "reference_source_mesh": {
            "path": str(reference_path),
            "sha256": reference_before,
            "semantic_identity": source_mesh_fingerprint(reference_path),
        },
        "master_interface_packet": packet_identity,
        "implementation": {
            "producer_sha256": _hash_file(Path(__file__).resolve()),
            "module_sha256": _hash_file(
                Path(source_conformal.__file__).resolve()
            ),
        },
        "master_interface": {
            key: master[key]
            for key in (
                "master_interface_sha256",
                "interface_triangle_count",
                "periodic_seam_triangle_count",
                "interface_open_edge_count",
                "periodic_seam_open_edge_counts",
                "sector_seam_continuity_pass",
                "coordinates_modified",
                "source_master_interface_pass",
            )
        },
        "shared_interface_topology": shared,
        "prohibited_operations": {
            "source_coordinate_projection": False,
            "source_mesh_rescaling": False,
            "source_strength_rescaling": False,
            "source_clipping": False,
            "blind_global_refinement": False,
        },
        "required_successors": [
            "construct full chamber/first-wall and radial stack around the "
            "exact packet",
            "audit written H5M shared surface ownership and senses",
            "run targeted native precision-4 plan",
            "run global native precision-4 only after targeted pass",
            "launch OpenMC geometry debug only after physical containment and P4 pass",
        ],
    }
    compact_containment = {
        key: value
        for key, value in containment.items()
        if key
        not in {"source_mesh_identity", "reference_source_mesh_identity"}
    }
    compact_containment["source_mesh_canonical_fingerprint"] = containment[
        "source_mesh_identity"
    ]["canonical_fingerprint"]
    compact_containment["reference_source_mesh_canonical_fingerprint"] = (
        containment["reference_source_mesh_identity"]["canonical_fingerprint"]
    )
    compact_containment.update(
        {
            "source_mesh_sha256": source_before,
            "reference_source_mesh_sha256": reference_before,
            "master_interface_sha256": master["master_interface_sha256"],
            "master_interface_packet_sha256": packet_identity["sha256"],
        }
    )
    targeted = {
        "schema": P4_RECEIPT_SCHEMA,
        "scope": "TARGETED",
        "status": "UNEXECUTED_DEPENDENCY",
        "precision": 4,
        "geometry_sha256": None,
        "master_interface_sha256": master["master_interface_sha256"],
        "targeted_p4_pass": False,
        "blocked_reason": "full physical DAGMC candidate has not been constructed",
        "timeout_is_physics_failure": False,
    }
    global_p4 = {
        "schema": P4_RECEIPT_SCHEMA,
        "scope": "GLOBAL",
        "status": "UNEXECUTED_DEPENDENCY",
        "precision": 4,
        "geometry_sha256": None,
        "global_p4_pass": False,
        "blocked_reason": "targeted precision-4 and full geometry are pending",
        "timeout_is_physics_failure": False,
    }
    openmc = {
        "schema": "parastell.openmc_geometry_debug_receipt/v1.0.0",
        "status": "UNEXECUTED_DEPENDENCY",
        "geometry_sha256": None,
        "openmc_geometry_debug_pass": False,
        "blocked_reason": "physical containment and native precision-4 have not passed",
        "launch_attempted": False,
    }
    blocker = {
        "schema": "parastell.source_conformal_outer_layer_request/v1.0.0",
        "status": "BLOCKED_INPUT",
        "master_interface_sha256": master["master_interface_sha256"],
        "master_interface_packet": packet_identity,
        "request": (
            "Construct the chamber/first-wall and remaining radial layers using "
            "MASTER_SOURCE_INTERFACE.npz as the exact inner master facet set. "
            "Reuse its vertex coordinates and triangle connectivity without "
            "projection, rescaling, clipping, or remeshing; bind every outer "
            "parametric surface, material ID, sector map, and seam mapping."
        ),
        "acceptance": [
            "one shared source/chamber interface surface with chamber and "
            "first_wall owners",
            "opposite valid senses and outward-normal closure",
            "no duplicate or orphan interface facets",
            "sector seam continuity",
            "exact source semantic identity and relative strength difference <= 1e-12",
            "zero outside or ambiguous source vertices and quadrature points",
        ],
    }
    write_compact_json_create_only(
        output / "SOURCE_CONFORMAL_GEOMETRY_MANIFEST.json", manifest
    )
    write_compact_json_create_only(
        output / "SOURCE_CONFORMAL_CONTAINMENT_RECEIPT.json",
        compact_containment,
    )
    write_compact_json_create_only(
        output / "DAGMC_TARGETED_P4_RECEIPT.json", targeted
    )
    write_compact_json_create_only(
        output / "DAGMC_GLOBAL_P4_RECEIPT.json", global_p4
    )
    write_compact_json_create_only(
        output / "OPENMC_GEOMETRY_DEBUG_RECEIPT.json", openmc
    )
    write_compact_json_create_only(
        output / "SOURCE_CONFORMAL_OUTER_LAYER_PRODUCER_REQUEST.json", blocker
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-mesh", type=Path, required=True)
    parser.add_argument("--source-mesh-sha256", required=True)
    parser.add_argument("--reference-source-mesh", type=Path, required=True)
    parser.add_argument("--reference-source-mesh-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reported-output-root",
        help=(
            "optional host-visible output path recorded in receipts when the "
            "producer runs through a container mount"
        ),
    )
    parser.add_argument(
        "--periodic-cut-plane-angles-degrees",
        type=float,
        nargs=2,
        default=(0.0, 90.0),
    )
    parser.add_argument(
        "--shared-interface-tolerance-cm", type=float, default=1.0e-9
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
