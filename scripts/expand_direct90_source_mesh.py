"""Expand the immutable 0-45 degree WISTELL-D source to one 90-degree period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
from pymoab import core, types

from parastell.geometry_provider import derive_wistell_d_transforms
from parastell.reference_geometry import sha256_file
from parastell.source_domain import (
    _source_arrays,
    audit_source_tetrahedra_arrays,
)
from parastell.source_geometry_identity import source_mesh_fingerprint
from parastell.source_mesh_order_audit import (
    audit_source_order_arrays,
    audit_tetrahedral_seam_topology,
    deduplicate_half_period_seam,
)


HALF_SOURCE_MESH_SHA256 = (
    "65264e15669d09c43f107c3b43c2af24ffbd15173e3bbd0e990b527bfa0b5322"
)
HALF_STRENGTHS_SHA256 = (
    "0ed18ab58bcc1e9884bf1b5c8bf19a7b7558ce7afe1869f1a2b01710148af6df"
)
EXPECTED_HALF_VERTEX_COUNT = 18_631
EXPECTED_HALF_TETRAHEDRON_COUNT = 86_400
EXPECTED_SEAM_VERTEX_COUNT = 601


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("half_source_mesh", type=Path)
    parser.add_argument("half_strengths", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    source = args.half_source_mesh.resolve()
    strengths_path = args.half_strengths.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    if sha256_file(source) != HALF_SOURCE_MESH_SHA256:
        raise ValueError("half-period source mesh hash mismatch")
    if sha256_file(strengths_path) != HALF_STRENGTHS_SHA256:
        raise ValueError("half-period strengths hash mismatch")

    source_core = core.Core()
    source_core.load_file(str(source))
    root = source_core.get_root_set()
    vertex_handles = source_core.get_entities_by_type(root, types.MBVERTEX)
    tet_handles = source_core.get_entities_by_type(root, types.MBTET)
    coordinates = np.asarray(source_core.get_coords(vertex_handles)).reshape(
        (-1, 3)
    )
    if (
        len(coordinates) != EXPECTED_HALF_VERTEX_COUNT
        or len(tet_handles) != EXPECTED_HALF_TETRAHEDRON_COUNT
    ):
        raise RuntimeError("frozen half-period source topology count mismatch")
    handle_to_index = {
        int(handle): index for index, handle in enumerate(vertex_handles)
    }
    connectivity = np.asarray(
        source_core.get_connectivity(tet_handles)
    ).reshape((-1, 4))
    connectivity_index = np.vectorize(
        lambda value: handle_to_index[int(value)]
    )(connectivity)
    source_tag = source_core.tag_get_handle("Source Strength")
    volume_tag = source_core.tag_get_handle("Volume")
    strengths = np.asarray(
        source_core.tag_get_data(source_tag, tet_handles, flat=True),
        dtype=float,
    )
    volumes = np.asarray(
        source_core.tag_get_data(volume_tag, tet_handles, flat=True),
        dtype=float,
    )
    external = np.load(strengths_path, allow_pickle=False)
    order = audit_source_order_arrays(
        strengths, external, expected_rate_per_s=float(strengths.sum())
    )
    if not order["order_and_rate_gate_pass"]:
        raise RuntimeError("half-period source order is not qualified")

    transform = derive_wistell_d_transforms(nfp=4, lasym=True)[
        "half_period_mate"
    ]
    transform_matrix = np.asarray(transform.matrix, dtype=float)
    expected_linear = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    if not (
        np.allclose(transform_matrix[:3, :3], expected_linear, atol=1.0e-15)
        and np.isclose(np.linalg.det(transform_matrix[:3, :3]), 1.0)
        and np.allclose(transform_matrix @ transform_matrix, np.eye(4))
    ):
        raise RuntimeError("unexpected WISTELL-D half-period transform")
    mate_coordinates = transform.transform_points(coordinates)
    combined_coordinates, combined_index, seam_audit = (
        deduplicate_half_period_seam(coordinates, mate_coordinates)
    )
    if seam_audit["merged_vertex_count"] != EXPECTED_SEAM_VERTEX_COUNT:
        raise RuntimeError("frozen half-period seam population mismatch")
    original_map = combined_index[: len(coordinates)]
    mate_map = combined_index[len(coordinates) :]
    combined_connectivity = np.vstack(
        (original_map[connectivity_index], mate_map[connectivity_index])
    )
    canonical_tets = np.sort(combined_connectivity, axis=1)
    if len(np.unique(canonical_tets, axis=0)) != len(canonical_tets):
        raise RuntimeError(
            "duplicate tetrahedron connectivity after expansion"
        )
    seam_topology = audit_tetrahedral_seam_topology(
        combined_coordinates, combined_connectivity
    )
    if not seam_topology["conformal_internal_seam_pass"]:
        raise RuntimeError(
            "expanded source mesh has unmatched internal seam faces"
        )
    parent_points = coordinates[connectivity_index]
    mate_points = mate_coordinates[connectivity_index]
    parent_signed = (
        np.linalg.det(parent_points[:, 1:] - parent_points[:, :1]) / 6.0
    )
    mate_signed = np.linalg.det(mate_points[:, 1:] - mate_points[:, :1]) / 6.0
    orientation_pass = bool(
        np.all(parent_signed != 0.0)
        and np.allclose(parent_signed, mate_signed, rtol=1.0e-12, atol=1.0e-12)
    )
    if not orientation_pass:
        raise RuntimeError(
            "half-period transform did not preserve tetrahedron orientation"
        )
    combined_strengths = np.concatenate((strengths, strengths))
    combined_volumes = np.concatenate((volumes, volumes))

    output.mkdir(parents=True)
    output_mesh = output / "source_mesh_90deg.h5m"
    output_strengths = output / "strengths_90deg.npy"
    output_core = core.Core()
    new_vertices = np.asarray(
        output_core.create_vertices(combined_coordinates.reshape(-1)),
        dtype=np.uint64,
    )
    new_tets = []
    for row in combined_connectivity:
        new_tets.append(
            output_core.create_element(types.MBTET, new_vertices[row])
        )
    source_out = output_core.tag_get_handle(
        "Source Strength", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_DENSE, True
    )
    volume_out = output_core.tag_get_handle(
        "Volume", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_DENSE, True
    )
    output_core.tag_set_data(source_out, new_tets, combined_strengths)
    output_core.tag_set_data(volume_out, new_tets, combined_volumes)
    output_core.write_file(str(output_mesh))
    np.save(output_strengths, combined_strengths, allow_pickle=False)

    tetrahedra, tagged_volumes, tagged_strengths = _source_arrays(output_mesh)
    reload_order = audit_source_order_arrays(
        tagged_strengths,
        np.load(output_strengths, allow_pickle=False),
        expected_rate_per_s=float(2.0 * strengths.sum()),
    )
    tetrahedron_audit = audit_source_tetrahedra_arrays(
        tetrahedra, tagged_volumes, tagged_strengths
    )
    order_construction_pass = bool(
        np.array_equal(tagged_strengths, combined_strengths)
        and np.array_equal(tagged_volumes, combined_volumes)
    )
    tagged_volume_closure_pass = bool(
        np.isclose(
            tagged_volumes.sum(),
            2.0 * volumes.sum(),
            rtol=1.0e-13,
            atol=0.0,
        )
    )
    phi = np.mod(
        np.degrees(
            np.arctan2(combined_coordinates[:, 1], combined_coordinates[:, 0])
        ),
        360.0,
    )
    phi[np.isclose(phi, 360.0, atol=1.0e-9)] = 0.0
    angular_pass = bool(
        np.isclose(phi.min(), 0.0, atol=1.0e-9)
        and np.isclose(phi.max(), 90.0, atol=1.0e-9)
        and np.all(phi <= 90.0 + 1.0e-9)
    )
    gate = bool(
        reload_order["order_and_rate_gate_pass"]
        and tetrahedron_audit["tetrahedron_data_gate_pass"]
        and angular_pass
        and len(tagged_strengths) == 2 * len(strengths)
        and order_construction_pass
        and tagged_volume_closure_pass
        and seam_topology["conformal_internal_seam_pass"]
        and orientation_pass
    )
    manifest = {
        "schema": "wistell_d.expanded_90_degree_source_mesh/v1.0.0",
        "status": (
            "EXPANDED_90_SOURCE_ORDER_RATE_ANGULAR_PASS_"
            "ACCEPTED_GEOMETRY_CONTAINMENT_PENDING"
            if gate
            else "EXPANDED_90_SOURCE_GATE_FAIL"
        ),
        "source": {
            "half_source_mesh_sha256": HALF_SOURCE_MESH_SHA256,
            "half_strengths_sha256": HALF_STRENGTHS_SHA256,
            "half_extent_degrees": 45.0,
            "half_tetrahedron_count": int(len(strengths)),
            "half_source_rate_per_s": float(strengths.sum()),
        },
        "provenance": {
            "implementation_path": str(Path(__file__).resolve()),
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "expansion": {
            "transform": transform.receipt(),
            "seam_audit": seam_audit,
            "merged_vertex_count": int(
                2 * len(coordinates) - len(combined_coordinates)
            ),
            "combined_vertex_count": int(len(combined_coordinates)),
            "combined_tetrahedron_count": int(len(combined_strengths)),
            "duplicate_tetrahedron_count": 0,
            "output_order_exact_parent_then_mate": order_construction_pass,
            "tagged_volume_doubling_pass": tagged_volume_closure_pass,
            "tetrahedron_orientation_preserved": orientation_pass,
            "seam_topology": seam_topology,
        },
        "output": {
            "source_mesh_path": str(output_mesh),
            "source_mesh_sha256": sha256_file(output_mesh),
            "strengths_path": str(output_strengths),
            "strengths_sha256": sha256_file(output_strengths),
            "source_rate_scope": "modeled_90_degree_period",
            "source_rate_per_s": float(tagged_strengths.sum()),
            "source_mesh_identity": source_mesh_fingerprint(output_mesh),
        },
        "reload_order_and_rate": reload_order,
        "tetrahedron_audit": tetrahedron_audit,
        "angular_domain_degrees": {
            "minimum": float(phi.min()),
            "maximum": float(phi.max()),
            "pass": angular_pass,
        },
        "accepted_geometry_containment_run": False,
        "partial_gate_pass": gate,
    }
    with (output / "EXPANDED_90_SOURCE_MANIFEST.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
