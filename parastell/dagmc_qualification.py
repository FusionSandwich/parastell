"""Fail-closed native DAGMC topology and command-result qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .reference_geometry import _quantize
from .reference_geometry import _triangles
from .reference_geometry import _volume_id
from .reference_geometry import audit_volume_envelope
from .reference_geometry import native_dagmc_id_inventory
from .reference_geometry import sha256_file


def parse_check_watertight_log(
    text: str,
    *,
    exit_code: int,
    expected_surface_count: int | None = None,
    expected_volume_count: int | None = None,
) -> dict[str, Any]:
    patterns = {
        "surface_count": (r"number of surfaces=(\d+)", 1),
        "volume_count": (r"number of volumes=(\d+)", 1),
        "unmatched_edge_fraction": (
            r"(\d+)/(\d+) \([^)]*\) unmatched edges",
            2,
        ),
        "unsealed_surface_fraction": (
            r"(\d+)/(\d+) \([^)]*\) unsealed surfaces",
            2,
        ),
        "unsealed_volume_fraction": (
            r"(\d+)/(\d+) \([^)]*\) unsealed volumes",
            2,
        ),
    }
    values: dict[str, Any] = {"exit_code": int(exit_code)}
    missing = []
    match_counts = {}
    for name, (pattern, width) in patterns.items():
        matches = re.findall(pattern, text)
        match_counts[name] = len(matches)
        if len(matches) != 1:
            missing.append(name)
        else:
            if width == 1:
                values[name] = int(matches[0])
            else:
                numerator, denominator = matches[0]
                prefix = name.removesuffix("_fraction")
                values[f"{prefix}_count"] = int(numerator)
                values[f"{prefix}_denominator"] = int(denominator)
    values["missing_terminal_fields"] = missing
    values["terminal_field_match_counts"] = match_counts
    values["error_lines"] = [
        line.strip()
        for line in re.findall(
            r"^\s*(?:ERROR|FATAL)\b.*$",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
    ]
    leaky_surface_matches = re.findall(
        r"^[ \t]*leaky surface ids[ \t]*=[ \t]*(.*)$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    leaky_volume_matches = re.findall(
        r"^[ \t]*leaky volume ids[ \t]*=[ \t]*(.*)$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    values["leaky_surface_id_payloads"] = [
        value.strip() for value in leaky_surface_matches
    ]
    values["leaky_volume_id_payloads"] = [
        value.strip() for value in leaky_volume_matches
    ]
    values["leaky_id_lists_pass"] = leaky_surface_matches == [
        ""
    ] and leaky_volume_matches == [""]
    expected_count_gate = (
        expected_surface_count is None
        or values.get("surface_count") == int(expected_surface_count)
    ) and (
        expected_volume_count is None
        or values.get("volume_count") == int(expected_volume_count)
    )
    values["expected_surface_count"] = expected_surface_count
    values["expected_volume_count"] = expected_volume_count
    values["native_inventory_count_reconciliation_pass"] = bool(
        expected_count_gate
    )
    denominator_gate = (
        not missing
        and values["surface_count"] > 0
        and values["volume_count"] > 0
        and values["unmatched_edge_denominator"] > 0
        and values["unsealed_surface_denominator"] == values["surface_count"]
        and values["unsealed_volume_denominator"] == values["volume_count"]
        and values["leaky_id_lists_pass"]
        and expected_count_gate
    )
    values["denominator_reconciliation_pass"] = bool(denominator_gate)
    values["pass"] = (
        int(exit_code) == 0
        and not missing
        and not values["error_lines"]
        and denominator_gate
        and all(
            values[name] == 0
            for name in (
                "unmatched_edge_count",
                "unsealed_surface_count",
                "unsealed_volume_count",
            )
        )
    )
    return values


def parse_overlap_check_log(
    text: str, *, exit_code: int, points_per_edge: int, threads: int
) -> dict[str, Any]:
    count_matches = re.findall(r"Overlap locations found:\s*(\d+)", text)
    no_overlap_matches = re.findall(
        r"^\s*No overlaps were found\.\s*$", text, re.MULTILINE
    )
    precision_matches = re.findall(
        r"Checking\s+(\d+)\s+points along each triangle edge", text
    )
    reported_precision = (
        int(precision_matches[0]) if len(precision_matches) == 1 else None
    )
    precision_pass = len(precision_matches) == 1 and reported_precision == int(
        points_per_edge
    )
    pattern = re.compile(
        r"Overlap Location:\s*([0-9eE+.-]+)\s+([0-9eE+.-]+)\s+"
        r"([0-9eE+.-]+)\s*\nOverlapping volumes:\s*(\d+)\s+(\d+)",
        re.MULTILINE,
    )
    rows = [
        {
            "location_cm": [float(match.group(index)) for index in (1, 2, 3)],
            "volume_ids": [int(match.group(4)), int(match.group(5))],
        }
        for match in pattern.finditer(text.replace("\r\n", "\n"))
    ]
    terminal_summary_match_count = len(count_matches) + len(no_overlap_matches)
    if len(count_matches) == 1 and not no_overlap_matches:
        terminal_count = int(count_matches[0])
        terminal_summary_kind = "explicit_count"
    elif len(no_overlap_matches) == 1 and not count_matches:
        terminal_count = 0
        terminal_summary_kind = "no_overlaps_sentence"
    else:
        terminal_count = None
        terminal_summary_kind = None
    row_count_matches = (
        terminal_count is not None and len(rows) == terminal_count
    )
    location_token_count = len(
        re.findall(r"Overlap Location:\s*", text, re.IGNORECASE)
    )
    volume_token_count = len(
        re.findall(r"Overlapping volumes:\s*", text, re.IGNORECASE)
    )
    unparsed_overlap_token_count = abs(location_token_count - len(rows)) + abs(
        volume_token_count - len(rows)
    )
    error_lines = [
        line.strip()
        for line in re.findall(r"^\s*ERROR:\s*.*$", text, re.MULTILINE)
    ]
    return {
        "exit_code": int(exit_code),
        "points_per_edge": int(points_per_edge),
        "reported_points_per_edge": reported_precision,
        "precision_header_match_count": len(precision_matches),
        "precision_header_pass": precision_pass,
        "threads": int(threads),
        "terminal_overlap_location_count": terminal_count,
        "parsed_overlap_location_count": len(rows),
        "terminal_count_match_count": len(count_matches),
        "no_overlap_sentence_match_count": len(no_overlap_matches),
        "terminal_summary_match_count": terminal_summary_match_count,
        "terminal_summary_kind": terminal_summary_kind,
        "terminal_count_present": terminal_summary_match_count == 1,
        "row_count_matches_terminal": row_count_matches,
        "overlap_location_token_count": location_token_count,
        "overlapping_volumes_token_count": volume_token_count,
        "unparsed_overlap_token_count": unparsed_overlap_token_count,
        "error_lines": error_lines,
        "locations": rows,
        "pass": bool(
            int(exit_code) == 0
            and row_count_matches
            and terminal_count == 0
            and precision_pass
            and unparsed_overlap_token_count == 0
            and not error_lines
        ),
    }


def _surface_shape_signature(surface: Any, quantum_cm: float) -> str:
    """Return a winding-independent geometric signature for duplicates."""
    quantized = _quantize(_triangles(surface.triangle_coords), quantum_cm)
    triangles = [
        sorted(tuple(int(value) for value in vertex) for vertex in triangle)
        for triangle in quantized
    ]
    encoded = json.dumps(
        sorted(triangles), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _volume_shape_signature(
    volume: Any, envelope: dict, quantum_cm: float
) -> str:
    row = {
        "surface_shapes": sorted(
            _surface_shape_signature(surface, quantum_cm)
            for surface in volume.surfaces
        ),
        "bounding_box": _quantize(
            [
                envelope["bounding_box_cm"]["lower"],
                envelope["bounding_box_cm"]["upper"],
            ],
            quantum_cm,
        ).tolist(),
        "volume_quantized_cm3": int(
            round(envelope["volume_cm3"] / quantum_cm**3)
        ),
    }
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def audit_dagmc_topology(
    dagmc_path: str | Path,
    *,
    coordinate_quantum_cm: float = 1.0e-6,
    magnet_material: str = "magnets",
    expected_magnet_count: int = 18,
    vector_area_relative_tolerance: float = 1.0e-8,
    expected_material_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Audit topology, native senses, duplicates, and magnet envelopes."""
    path = Path(dagmc_path).resolve()
    before = sha256_file(path)
    expected_counts = (
        expected_material_counts
        if expected_material_counts is not None
        else {
            "Vacuum": 1,
            "first_wall": 1,
            "breeder": 1,
            "back_wall": 1,
            "shield": 1,
            "vac_vessel": 1,
            "magnets": 18,
        }
    )
    from .native_dagmc_topology import audit_native_moab_topology

    native_topology = audit_native_moab_topology(
        path,
        expected_material_counts=expected_counts,
        vector_area_relative_tolerance=vector_area_relative_tolerance,
    )
    if native_topology.get("native_topology_gate_pass") is not True:
        after = sha256_file(path)
        return {
            "schema": "parastell.dagmc_topology_audit/v1.1.0",
            "raw_h5m_sha256_before": before,
            "raw_h5m_sha256_after": after,
            "h5m_unchanged": before == after,
            "native_topology": native_topology,
            "pydagmc_load_attempted": False,
            "volume_envelopes": [],
            "magnet_count": 0,
            "magnet_volume_ids": [],
            "all_magnet_envelopes_close": False,
            "topology_gate_pass": False,
        }

    from pymoab import core, types
    import pydagmc

    mesh = core.Core()
    mesh.load_file(str(path))
    moab_vertices = mesh.get_entities_by_dimension(mesh.get_root_set(), 0)
    moab_triangles = mesh.get_entities_by_type(
        mesh.get_root_set(), types.MBTRI
    )
    model = pydagmc.Model(str(path))
    native_ids = native_dagmc_id_inventory(path)
    pydagmc_surface_ids = sorted(int(value) for value in model.surfaces_by_id)
    pydagmc_volume_ids = sorted(int(value) for value in model.volumes_by_id)
    surface_id_reconciliation_pass = (
        native_ids["surface_ids"] == pydagmc_surface_ids
    )
    volume_id_reconciliation_pass = (
        native_ids["volume_ids"] == pydagmc_volume_ids
    )

    incidence: dict[int, set[int]] = {
        int(surface_id): set() for surface_id in model.surfaces_by_id
    }
    for volume_id, volume in model.volumes_by_id.items():
        for surface in volume.surfaces:
            incidence.setdefault(int(surface.id), set()).add(int(volume_id))

    sense_rows = []
    for surface_id, surface in sorted(model.surfaces_by_id.items()):
        forward = _volume_id(getattr(surface, "forward_volume", None))
        reverse = _volume_id(getattr(surface, "reverse_volume", None))
        adjacent = sorted(incidence.get(int(surface_id), set()))
        if len(adjacent) == 1:
            sense_pass = (forward == adjacent[0] and reverse is None) or (
                reverse == adjacent[0] and forward is None
            )
            role = "external"
        elif len(adjacent) == 2:
            sense_pass = (
                forward is not None
                and reverse is not None
                and forward != reverse
                and {forward, reverse} == set(adjacent)
            )
            role = "internal"
        else:
            sense_pass = False
            role = "orphan" if not adjacent else "nonmanifold"
        sense_rows.append(
            {
                "surface_id": int(surface_id),
                "adjacent_volume_ids": adjacent,
                "forward_volume_id": forward,
                "reverse_volume_id": reverse,
                "topological_role": role,
                "sense_pass": bool(sense_pass),
            }
        )

    envelopes = []
    signatures: dict[str, list[int]] = {}
    for volume_id, volume in sorted(model.volumes_by_id.items()):
        envelope = audit_volume_envelope(
            volume,
            coordinate_quantum_cm=coordinate_quantum_cm,
            vector_area_relative_tolerance=vector_area_relative_tolerance,
        )
        signature = _volume_shape_signature(
            volume, envelope, coordinate_quantum_cm
        )
        signatures.setdefault(signature, []).append(int(volume_id))
        envelopes.append(
            {
                **envelope,
                "shape_only_fingerprint": signature,
            }
        )
    duplicate_sets = [
        volume_ids for volume_ids in signatures.values() if len(volume_ids) > 1
    ]
    potential_coincident_pairs = []
    for index, first in enumerate(envelopes):
        first_bounds = np.asarray(
            [
                first["bounding_box_cm"]["lower"],
                first["bounding_box_cm"]["upper"],
            ]
        )
        for second in envelopes[index + 1 :]:
            second_bounds = np.asarray(
                [
                    second["bounding_box_cm"]["lower"],
                    second["bounding_box_cm"]["upper"],
                ]
            )
            bounds_match = bool(
                np.allclose(
                    first_bounds,
                    second_bounds,
                    rtol=0.0,
                    atol=coordinate_quantum_cm,
                )
            )
            volume_scale = max(first["volume_cm3"], second["volume_cm3"], 1.0)
            volume_match = (
                abs(first["volume_cm3"] - second["volume_cm3"])
                <= 1.0e-9 * volume_scale
            )
            if bounds_match and volume_match:
                potential_coincident_pairs.append(
                    {
                        "volume_ids": [
                            first["volume_id"],
                            second["volume_id"],
                        ],
                        "reason": (
                            "bounding boxes and volumes coincide within the "
                            "fail-closed duplicate threshold"
                        ),
                    }
                )
    magnets = [
        row
        for row in envelopes
        if row["material_tag"].strip().lower()
        == str(magnet_material).strip().lower()
    ]
    material_counts: dict[str, int] = {}
    volumes_without_material = []
    for row in envelopes:
        material = row["material_tag"]
        if not material:
            volumes_without_material.append(row["volume_id"])
        material_counts[material] = material_counts.get(material, 0) + 1
    material_group_gate_pass = (
        material_counts == expected_counts and not volumes_without_material
    )
    invalid_senses = [row for row in sense_rows if not row["sense_pass"]]
    orphan_surfaces = [
        row["surface_id"]
        for row in sense_rows
        if row["topological_role"] == "orphan"
    ]
    nonmanifold_surfaces = [
        row["surface_id"]
        for row in sense_rows
        if row["topological_role"] == "nonmanifold"
    ]
    after = sha256_file(path)
    gate = (
        before == after
        and native_topology["native_topology_gate_pass"]
        and native_ids["native_id_gate_pass"]
        and surface_id_reconciliation_pass
        and volume_id_reconciliation_pass
        and not invalid_senses
        and not orphan_surfaces
        and not nonmanifold_surfaces
        and not duplicate_sets
        and not potential_coincident_pairs
        and material_group_gate_pass
        and len(magnets) == int(expected_magnet_count)
        and all(row["closed"] for row in envelopes)
        and all(row["closed"] for row in magnets)
    )
    return {
        "schema": "parastell.dagmc_topology_audit/v1.1.0",
        "raw_h5m_sha256_before": before,
        "raw_h5m_sha256_after": after,
        "h5m_unchanged": before == after,
        "native_topology": native_topology,
        "pydagmc_load_attempted": True,
        "native_ids": native_ids,
        "pydagmc_native_id_reconciliation": {
            "pydagmc_surface_ids": pydagmc_surface_ids,
            "pydagmc_volume_ids": pydagmc_volume_ids,
            "surface_ids_match": surface_id_reconciliation_pass,
            "volume_ids_match": volume_id_reconciliation_pass,
            "pass": surface_id_reconciliation_pass
            and volume_id_reconciliation_pass,
        },
        "pymoab_reload": {
            "pass": True,
            "vertex_count": len(moab_vertices),
            "triangle_count": len(moab_triangles),
        },
        "pydagmc_reload": {
            "pass": True,
            "volume_count": len(model.volumes_by_id),
            "surface_count": len(model.surfaces_by_id),
        },
        "orphan_surface_ids": orphan_surfaces,
        "nonmanifold_surface_ids": nonmanifold_surfaces,
        "invalid_surface_sense_count": len(invalid_senses),
        "surface_senses": sense_rows,
        "duplicate_shape_only_volume_sets": duplicate_sets,
        "potential_coincident_volume_pairs": potential_coincident_pairs,
        "duplicate_physical_volume_count": sum(
            len(volume_ids) - 1 for volume_ids in duplicate_sets
        ),
        "material_group_audit": {
            "expected_material_counts": expected_counts,
            "actual_material_counts": dict(sorted(material_counts.items())),
            "volumes_without_material": volumes_without_material,
            "pass": material_group_gate_pass,
        },
        "volume_envelopes": envelopes,
        "magnet_material": str(magnet_material),
        "expected_magnet_count": int(expected_magnet_count),
        "magnet_count": len(magnets),
        "magnet_volume_ids": [row["volume_id"] for row in magnets],
        "all_magnet_envelopes_close": bool(magnets)
        and all(row["closed"] for row in magnets),
        "topology_gate_pass": gate,
    }
