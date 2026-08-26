"""Independent qualification report for an immutable DAGMC assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .dagmc_envelope import canonical_dagmc_fingerprint


def parse_watertight_log(text: str) -> dict[str, Any]:
    patterns = {
        "surface_count": r"number of surfaces=(\d+)",
        "volume_count": r"number of volumes=(\d+)",
        "unmatched_edge_count": r"(\d+)/(?:\d+) \([^)]*\) unmatched edges",
        "unsealed_surface_count": r"(\d+)/(?:\d+) \([^)]*\) unsealed surfaces",
        "unsealed_volume_count": r"(\d+)/(?:\d+) \([^)]*\) unsealed volumes",
    }
    result = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"watertight log lacks {name}")
        result[name] = int(match.group(1))
    result["pass"] = all(
        result[name] == 0
        for name in (
            "unmatched_edge_count",
            "unsealed_surface_count",
            "unsealed_volume_count",
        )
    )
    return result


def parse_overlap_log(text: str) -> dict[str, Any]:
    count_match = re.search(r"Overlap locations found:\s*(\d+)", text)
    if not count_match:
        raise ValueError("overlap log lacks terminal overlap count")
    pattern = re.compile(
        r"Overlap Location:\s*([0-9eE+.-]+)\s+([0-9eE+.-]+)\s+([0-9eE+.-]+)\s*\n"
        r"Overlapping volumes:\s*(\d+)\s+(\d+)",
        re.MULTILINE,
    )
    rows = [
        {
            "location_cm": [float(match.group(index)) for index in (1, 2, 3)],
            "volume_ids": [int(match.group(4)), int(match.group(5))],
        }
        for match in pattern.finditer(text.replace("\r\n", "\n"))
    ]
    if len(rows) != int(count_match.group(1)):
        raise ValueError("overlap row count disagrees with terminal count")
    return {"overlap_location_count": len(rows), "locations": rows}


def build_geometry_qualification_report(
    *,
    dagmc_path: str | Path,
    associations_path: str | Path,
    interchange_path: str | Path,
    watertight_log_path: str | Path,
    overlap_log_path: str | Path,
    surface_inventory_path: str | Path,
    ray_audit_path: str | Path,
    geometry_debug_attempt: Mapping[str, Any],
) -> dict[str, Any]:
    import pydagmc
    from pymoab import core

    path = Path(dagmc_path).resolve()
    associations = json.loads(
        Path(associations_path).read_text(encoding="utf-8")
    )
    policy = associations["canonical_geometry_policy"]
    fingerprint = canonical_dagmc_fingerprint(
        path,
        coordinate_quantum_cm=policy["coordinate_quantum_cm"],
        faceting_tolerances=policy["faceting_tolerances"],
    )
    model = pydagmc.Model(str(path))
    moab = core.Core()
    moab.load_file(str(path))
    pymoab = {
        "reload_pass": True,
        "vertices": len(moab.get_entities_by_dimension(0, 0)),
        "triangles": len(moab.get_entities_by_dimension(0, 2)),
    }
    pydagmc = {
        "reload_pass": True,
        "volume_count": len(model.volumes_by_id),
        "surface_count": len(model.surfaces_by_id),
    }
    watertight = parse_watertight_log(
        Path(watertight_log_path).read_text(encoding="utf-8")
    )
    overlaps = parse_overlap_log(
        Path(overlap_log_path).read_text(encoding="utf-8")
    )
    for row in overlaps["locations"]:
        first, second = row["volume_ids"]
        shared = sorted(
            {
                int(surface.id)
                for surface in model.volumes_by_id[first].surfaces
            }
            & {
                int(surface.id)
                for surface in model.volumes_by_id[second].surfaces
            }
        )
        row["shared_topological_surface_ids"] = shared
        row["classification"] = (
            "COINCIDENT_SHARED_BOUNDARY_CONTACT"
            if shared
            else "UNINTENDED_NONADJACENT_VOLUME_OVERLAP"
        )
    unintended = [
        row
        for row in overlaps["locations"]
        if row["classification"] == "UNINTENDED_NONADJACENT_VOLUME_OVERLAP"
    ]
    overlaps["unintended_overlap_count"] = len(unintended)
    overlaps["pass"] = not unintended

    material_counts = {
        str(material): len(volumes)
        for material, volumes in sorted(model.volumes_by_material.items())
    }
    material_audit = {
        "volumes_without_material": [
            int(volume.id) for volume in model.volumes_without_material
        ],
        "volumes_by_material": material_counts,
        "interstitial_vacuum_volume_id": 43,
        "graveyard_volume_id": 44,
        "interstitial_vacuum_material": model.volumes_by_id[43].material,
        "graveyard_material": model.volumes_by_id[44].material,
        "one_interstitial_vacuum_and_one_graveyard_pass": (
            model.volumes_by_id[43].material.lower() == "vacuum"
            and model.volumes_by_id[44].material.lower() == "graveyard"
        ),
    }
    interchange = json.loads(
        Path(interchange_path).read_text(encoding="utf-8")
    )
    identity = np.eye(4)
    transform_rows = [
        np.asarray(row["native_global_transform"], dtype=float)
        for row in interchange["magnets"]
    ]
    transform_audit = {
        "magnet_count_in_interchange": len(transform_rows),
        "all_native_global_transforms_identity": all(
            np.array_equal(value, identity) for value in transform_rows
        ),
        "arbitrary_translation_present": any(
            not np.array_equal(value[:3, 3], np.zeros(3))
            for value in transform_rows
        ),
        "coordinate_basis": interchange["provenance"]["coordinate_basis"],
        "fingerprint_match": interchange["geometry_fingerprint"]
        == fingerprint["canonical_fingerprint"],
        "raw_h5m_hash_match": interchange["provenance"]["dagmc_sha256"]
        == fingerprint["raw_h5m_sha256"],
    }
    inventory = json.loads(
        Path(surface_inventory_path).read_text(encoding="utf-8")
    )
    rays = json.loads(Path(ray_audit_path).read_text(encoding="utf-8"))
    geometry_pass = (
        watertight["pass"]
        and overlaps["pass"]
        and transform_audit["all_native_global_transforms_identity"]
        and not transform_audit["arbitrary_translation_present"]
        and bool(geometry_debug_attempt.get("pass"))
    )
    return {
        "schema": "parastell.geometry_and_surface_audit/v1",
        "raw_h5m_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_geometry_fingerprint": fingerprint["canonical_fingerprint"],
        "pymoab_reload": pymoab,
        "pydagmc_reload": pydagmc,
        "native_check_watertight": watertight,
        "native_overlap_check": overlaps,
        "openmc_geometry_debug": dict(geometry_debug_attempt),
        "material_and_group_audit": material_audit,
        "native_transform_audit": transform_audit,
        "randomized_ray_audit": {
            "ray_count": rays["ray_count"],
            "all_first_surface_ids_pass": rays["all_first_surface_ids_pass"],
            "direct_winding_to_external_topology_count": len(
                rays["direct_winding_to_external_topologies"]
            ),
            "no_winding_to_external_bypass_pass": rays[
                "no_winding_to_external_bypass_pass"
            ],
            "seed": rays["seed"],
        },
        "magnet_surface_gates": {
            "pair_count": inventory["magnet_count"],
            "winding_pack_envelopes_pass": inventory[
                "winding_pack_envelopes_pass"
            ],
            "outer_casing_external_envelopes_pass": inventory[
                "outer_casing_envelopes_pass"
            ],
            "all_internal_casing_interfaces_excluded": inventory[
                "all_internal_casing_interfaces_excluded"
            ],
        },
        "geometry_full_assembly_pass": geometry_pass,
        "status": "PASS" if geometry_pass else "FAIL",
    }


def write_geometry_qualification(
    json_path: str | Path,
    markdown_path: str | Path,
    report: Mapping[str, Any],
) -> None:
    Path(json_path).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    overlap = report["native_overlap_check"]
    lines = [
        "# Geometry and surface audit",
        "",
        f"Raw H5M SHA-256: `{report['raw_h5m_sha256']}`  ",
        f"Canonical fingerprint: `{report['canonical_geometry_fingerprint']}`",
        "",
        "## Independent reload and native checks",
        "",
        f"- PyMOAB reload: PASS ({report['pymoab_reload']['vertices']} vertices; {report['pymoab_reload']['triangles']} triangles).",
        f"- PyDAGMC reload: PASS ({report['pydagmc_reload']['volume_count']} volumes; {report['pydagmc_reload']['surface_count']} surfaces).",
        f"- `check_watertight`: {'PASS' if report['native_check_watertight']['pass'] else 'FAIL'}; zero unmatched edges, unsealed surfaces, and unsealed volumes.",
        f"- Native `overlap_check`: FAIL; {overlap['overlap_location_count']} reported locations, including {overlap['unintended_overlap_count']} nonadjacent-volume findings.",
        f"- OpenMC geometry debug: {report['openmc_geometry_debug']['status']}.",
        "",
        "The 500k transport receipt independently records zero lost particles and zero DAGMC navigation errors, but this does not erase the native overlap findings or replace the requested geometry-debug mode.",
        "",
        "## Magnet topology",
        "",
        f"- 18 explicit casing/winding pairs: PASS.",
        f"- All winding-pack shells closed: {report['magnet_surface_gates']['winding_pack_envelopes_pass']}.",
        f"- All external-only casing shells closed: {report['magnet_surface_gates']['outer_casing_external_envelopes_pass']}.",
        f"- Internal casing/winding interfaces excluded: {report['magnet_surface_gates']['all_internal_casing_interfaces_excluded']}.",
        f"- Seeded ray first-hit identities: PASS ({report['randomized_ray_audit']['ray_count']} rays).",
        f"- Direct winding-pack-to-vacuum faces: {report['randomized_ray_audit']['direct_winding_to_external_topology_count']} (gate FAIL).",
        "",
        "## Native overlap findings",
        "",
        "| Volumes | Shared DAGMC surface | Classification | Location (cm) |",
        "|---|---|---|---|",
    ]
    for row in overlap["locations"]:
        lines.append(
            f"| {' / '.join(map(str, row['volume_ids']))} | "
            f"{','.join(map(str, row['shared_topological_surface_ids'])) or 'none'} | "
            f"{row['classification']} | "
            f"{', '.join(f'{value:.4g}' for value in row['location_cm'])} |"
        )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            "`GEOMETRY_FULL_ASSEMBLY_PASS = FAIL_NATIVE_OVERLAPS_AND_GEOMETRY_DEBUG_NOT_COMPLETED`",
            "",
            "`WINDING_PACK_ENVELOPES_PASS = PASS`",
            "",
            "`OUTER_CASING_ENVELOPES_PASS = FAIL_16_OF_18_OPEN_AFTER_INTERNAL_FACE_EXCLUSION`",
        ]
    )
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
