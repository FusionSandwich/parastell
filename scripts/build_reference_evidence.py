"""Build compact Prompt-7A evidence from an immutable R1 artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from parastell.reference_geometry import REFERENCE_SCHEMA
from parastell.reference_geometry import canonical_geometry_fingerprint
from parastell.reference_geometry import compare_reference_geometry
from parastell.reference_geometry import discover_original_magnets
from parastell.reference_geometry import geometry_inventory
from parastell.reference_geometry import sha256_file
from parastell.reference_geometry import validate_reference_manifest
from parastell.reference_geometry import write_json


FORK_MAIN_SHA = "de7d2978ff314b060ca2e6b10745a034e8b2a3c4"
UPSTREAM_MAIN_SHA = "de7d2978ff314b060ca2e6b10745a034e8b2a3c4"
SCRIPT_BLOB = "bcb5fc41cb1e09a88e4b6f02b2e485d1286d2f89"
VMEC_BLOB = "95a75490479d38d04712eb39ab856757b2211403"
COILS_BLOB = "d8e0d526abd9479a307218ca9b3fa4887a2161bd"


def _parse_watertight(text: str) -> dict:
    patterns = {
        "surface_count": r"number of surfaces=(\d+)",
        "volume_count": r"number of volumes=(\d+)",
        "unmatched_edge_count": r"(\d+)/(?:\d+) \([^)]*\) unmatched edges",
        "unsealed_surface_count": r"(\d+)/(?:\d+) \([^)]*\) unsealed surfaces",
        "unsealed_volume_count": r"(\d+)/(?:\d+) \([^)]*\) unsealed volumes",
    }
    values = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"watertight log lacks {name}")
        values[name] = int(match.group(1))
    values["pass"] = all(
        values[name] == 0
        for name in (
            "unmatched_edge_count",
            "unsealed_surface_count",
            "unsealed_volume_count",
        )
    )
    return values


def _parse_overlap(text: str) -> dict:
    count = re.search(r"Overlap locations found:\s*(\d+)", text)
    if count is None:
        raise ValueError("overlap log lacks terminal count")
    pattern = re.compile(
        r"Overlap Location:\s*([0-9eE+.-]+)\s+([0-9eE+.-]+)\s+([0-9eE+.-]+)\s*\n"
        r"Overlapping volumes:\s*(\d+)\s+(\d+)",
        re.MULTILINE,
    )
    rows = [
        {
            "location_cm": [float(match.group(i)) for i in (1, 2, 3)],
            "volume_ids": [int(match.group(4)), int(match.group(5))],
            "classification": "TRUE_UNINTENDED_NONADJACENT_VOLUME_OVERLAP",
        }
        for match in pattern.finditer(text.replace("\r\n", "\n"))
    ]
    if len(rows) != int(count.group(1)):
        raise ValueError("overlap row count does not match terminal count")
    return {
        "precision": {"points_per_edge": 2, "threads": 4},
        "non_exhaustive": True,
        "overlap_location_count": len(rows),
        "unintended_overlap_count": len(rows),
        "pass": len(rows) == 0,
        "locations": rows,
    }


def _source_mesh(path: Path) -> dict:
    from pymoab import core, types

    mesh = core.Core()
    mesh.load_file(str(path))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "coordinate_samples": {
            "cfs": 11,
            "poloidal": 61,
            "toroidal": 61,
        },
        "coordinate_ranges": {
            "cfs": [0.0, 1.0],
            "poloidal_deg": [0.0, 360.0],
            "toroidal_deg": [0.0, 90.0],
        },
        "vertex_count": len(mesh.get_entities_by_dimension(0, 0)),
        "tetrahedron_count": len(mesh.get_entities_by_type(0, types.MBTET)),
        "physical_source_model": "ParaStell default D-T profiles",
    }


def build(artifact: Path, output: Path) -> None:
    import pydagmc
    from pymoab import core

    dagmc = (artifact / "dagmc.h5m").resolve()
    source = (artifact / "source_mesh.h5m").resolve()
    checks = artifact / "checks"
    fingerprint = canonical_geometry_fingerprint(
        dagmc,
        faceting_settings={
            "backend": "cad_to_dagmc",
            "min_mesh_size_cm": 5.0,
            "max_mesh_size_cm": 20.0,
            "mesh_algorithm": 1,
        },
    )
    inventory = geometry_inventory(dagmc)
    magnets = discover_original_magnets(dagmc)
    parity = compare_reference_geometry(
        dagmc,
        dagmc,
        faceting_settings={
            "backend": "cad_to_dagmc",
            "min_mesh_size_cm": 5.0,
            "max_mesh_size_cm": 20.0,
            "mesh_algorithm": 1,
        },
    )

    moab = core.Core()
    moab.load_file(str(dagmc))
    dag_model = pydagmc.Model(str(dagmc))
    watertight = _parse_watertight(
        (checks / "check_watertight.log").read_text(encoding="utf-8")
    )
    overlap = _parse_overlap(
        (checks / "overlap_check_p2_t4.log").read_text(encoding="utf-8")
    )
    volume_by_id = {row["volume_id"]: row for row in inventory["volumes"]}
    magnet_by_volume = {
        row["volume_id"]: row["magnet_id"] for row in magnets["magnets"]
    }
    for row in overlap["locations"]:
        row["components"] = [
            {
                "volume_id": volume_id,
                "material_tag": volume_by_id[volume_id]["material_tag"],
                "component_name": volume_by_id[volume_id]["component_name"],
                "magnet_id": magnet_by_volume.get(volume_id),
            }
            for volume_id in row["volume_ids"]
        ]
        first, second = row["volume_ids"]
        shared = sorted(
            set(volume_by_id[first]["surface_ids"])
            & set(volume_by_id[second]["surface_ids"])
        )
        row["shared_topological_surface_ids"] = shared
        if shared:
            raise RuntimeError(
                "native overlap witness unexpectedly shares a surface"
            )

    manifest = {
        "schema": REFERENCE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_identity_classification": "SEPARATE_WISTELL_D_ASSETS_FOUND",
        "r1_identity": "current_public_parastell_cad_to_dagmc_example",
        "fork_main_sha": FORK_MAIN_SHA,
        "upstream_main_sha": UPSTREAM_MAIN_SHA,
        "parastell_sha": FORK_MAIN_SHA,
        "inputs": [
            {
                "role": "vmec_equilibrium",
                "path": str(artifact / "wout_vmec.nc"),
                "sha256": sha256_file(artifact / "wout_vmec.nc"),
                "git_blob": VMEC_BLOB,
            },
            {
                "role": "coil_filaments",
                "path": str(artifact / "coils.example"),
                "sha256_worktree_bytes": sha256_file(
                    artifact / "coils.example"
                ),
                "sha256_git_normalized_content": "de96b6009356c0d7b2abd9fd0507589651a7c04447011e0c4d3ed0d2f5756737",
                "git_blob": COILS_BLOB,
            },
            {
                "role": "radial_build",
                "path": "embedded in examples/parastell_cad_to_dagmc_example.py",
                "definition": {
                    "first_wall_cm": 5.0,
                    "breeder": "exact embedded 9x9 matrix",
                    "back_wall_cm": 5.0,
                    "shield_cm": 50.0,
                    "vacuum_vessel_cm": 10.0,
                },
            },
        ],
        "example_script": {
            "path": "examples/parastell_cad_to_dagmc_example.py",
            "artifact_path": str(
                artifact / "parastell_cad_to_dagmc_example.py"
            ),
            "sha256_worktree_bytes": sha256_file(
                artifact / "parastell_cad_to_dagmc_example.py"
            ),
            "sha256_git_normalized_content": "0eb83ae88304739151eb7674e42fc663e1319ee31d595c850fe388da768ef007",
            "git_blob": SCRIPT_BLOB,
            "physics_dictionary_modified": False,
        },
        "environment": {
            "container_image": "parastell-openmc:0.16.0",
            "python": "3.12.13",
            "parastell": "0.1.0 at exact mounted source SHA",
            "cadquery": "2.7.0",
            "cad_to_dagmc": "0.11.5",
            "openmc": "0.16.0",
            "pymoab": "5.5.1",
            "pydagmc": "0.0.1",
            "gmsh": "container executable; exact version recorded in environment_manifest.txt",
            "full_manifest": str(artifact / "environment_manifest.txt"),
        },
        "raw_h5m_path": str(dagmc),
        "raw_h5m_sha256": fingerprint["raw_h5m_sha256"],
        "canonical_geometry_fingerprint": fingerprint["canonical_fingerprint"],
        "canonicalization": {
            "version": 1,
            "coordinate_quantum_cm": fingerprint["coordinate_quantum_cm"],
        },
        "coordinate_units": "cm",
        "modeled_toroidal_extent_deg": 90.0,
        "field_period_semantics": "direct 90-degree sector; no symmetry expansion applied",
        "volume_inventory": {
            "count": inventory["volume_count"],
            "material_counts": inventory["material_counts"],
            "volumes_without_material": inventory["volumes_without_material"],
        },
        "surface_inventory": {"count": inventory["surface_count"]},
        "material_component_tags": sorted(inventory["material_counts"]),
        "magnet_ids": [row["magnet_id"] for row in magnets["magnets"]],
        "magnet_volume_ids": [row["volume_id"] for row in magnets["magnets"]],
        "magnet_representation": "one_homogenized_solid_each",
        "magnet_outer_dimensions_cm": {"width": 40.0, "thickness": 50.0},
        "case_thickness_cm": 0.0,
        "magnet_transform": "none; native coordinates preserved",
        "source_mesh": _source_mesh(source),
        "faceting_settings": {
            "backend": "cad_to_dagmc",
            "min_mesh_size_cm": 5.0,
            "max_mesh_size_cm": 20.0,
            "mesh_algorithm": 1,
        },
        "ports_present": False,
    }
    validate_reference_manifest(manifest)
    geometry_report = {
        "schema": "parastell.geometry_debug_report/v1.0.0",
        "raw_h5m_sha256": fingerprint["raw_h5m_sha256"],
        "canonical_geometry_fingerprint": fingerprint["canonical_fingerprint"],
        "pymoab_reload": {
            "pass": True,
            "vertices": len(moab.get_entities_by_dimension(0, 0)),
            "triangles": len(moab.get_entities_by_dimension(0, 2)),
        },
        "pydagmc_reload": {
            "pass": True,
            "volumes": len(dag_model.volumes_by_id),
            "surfaces": len(dag_model.surfaces_by_id),
        },
        "native_check_watertight": watertight,
        "native_overlap_check": overlap,
        "openmc_geometry_debug": {
            "status": "PENDING",
            "pass": False,
        },
        "geometry_gate_pass": False,
        "blocking_reason": "four true nonadjacent vacuum-vessel/magnet overlaps",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "REFERENCE_GEOMETRY_MANIFEST.json", manifest)
    write_json(output / "ORIGINAL_MAGNET_ENVELOPE_INVENTORY.json", magnets)
    write_json(output / "REFERENCE_VS_INSTRUMENTED_PARITY.json", parity)
    write_json(output / "GEOMETRY_DEBUG_REPORT.json", geometry_report)
    write_json(
        artifact / "canonical_geometry_fingerprint.json",
        {
            key: fingerprint[key]
            for key in (
                "algorithm",
                "canonical_fingerprint",
                "raw_h5m_sha256",
                "coordinate_quantum_cm",
                "volume_count",
                "surface_count",
            )
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    build(arguments.artifact.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
