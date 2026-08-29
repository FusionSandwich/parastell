"""Export the audited 90-degree parametric WISTELL-D CAD to DAGMC.

This producer is intentionally separate from the historical continuous-magnet
example exporter.  It preserves the eight radial solids and the eighteen
source-CAD swept-coil solids as 26 independently identified volumes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from parastell.source_cad_refaceting import _cadquery_solid_signature
from parastell.source_cad_refaceting import _gmsh_volume_signature
from parastell.source_cad_refaceting import _source_import_signature_match
from parastell.source_cad_refaceting import _volume_signature_match


SOURCE_MANIFEST_SCHEMA = "parastell.parametric_source_cad/v1.0.0"
SOURCE_AUDIT_SCHEMA = "parastell.parametric_source_cad_audit/v1.0.0"
SOURCE_AUDIT_SEAL_SCHEMA = "parastell.parametric_source_cad_audit_seal/v1.0.0"
RADIAL_COMPONENTS = (
    "chamber",
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "vacuum_gap",
)
RADIAL_MATERIALS = (
    "Vacuum",
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "Vacuum",
)
MAGNET_IDS = tuple(f"magnet-{index:04d}" for index in range(18))
SEMANTIC_ROLES = (*RADIAL_COMPONENTS, *MAGNET_IDS)
MATERIAL_TAGS = (
    *RADIAL_MATERIALS,
    *("magnets" for _ in MAGNET_IDS),
)
ALLOWED_SHARED_ROLE_PAIRS = frozenset(
    frozenset((RADIAL_COMPONENTS[index], RADIAL_COMPONENTS[index + 1]))
    for index in range(len(RADIAL_COMPONENTS) - 1)
)
EXPECTED_PAIR_COUNTS = {
    "component_pair_count": 28,
    "magnet_component_pair_count": 144,
    "magnet_pair_count": 153,
}
MESH_MIN_CM = 5.0
MESH_MAX_CM = 20.0
MESH_ALGORITHM = 1
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
POSITION_TOLERANCE_CM = 5.0e-7
FACETED_MAGNET_IDENTITY_TOLERANCE_CM = 25.0
VOLUME_RELATIVE_TOLERANCE = 1.0e-9
VOLUME_ABSOLUTE_TOLERANCE_CM3 = 1.0e-7
PERIODIC_PLANE_TOLERANCE_CM = 1.0e-5
PERIODIC_AREA_RELATIVE_TOLERANCE = 1.0e-6
PERIODIC_AREA_ABSOLUTE_TOLERANCE_CM2 = 1.0e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_identity() -> dict[str, Any]:
    packages = {}
    for name in ("cadquery", "cad_to_dagmc", "gmsh", "numpy", "pymoab"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "NOT_DISTRIBUTION_DISCOVERABLE"
    executable = Path(sys.executable).resolve()
    return {
        "python_executable": str(executable),
        "python_sha256": sha256_file(executable),
        "python_version": sys.version,
        "packages": packages,
    }


def _load_bound_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"hash mismatch for {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON control is not an object: {path}")
    return value


def _validate_source_packet(
    source_root: Path,
    *,
    manifest_sha256: str,
    audit_path: Path,
    audit_sha256: str,
    audit_seal_path: Path,
    audit_seal_sha256: str,
    audit_producer_path: Path,
    audit_producer_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = source_root / "PARAMETRIC_GEOMETRY_MANIFEST.json"
    manifest = _load_bound_json(manifest_path, manifest_sha256)
    audit = _load_bound_json(audit_path, audit_sha256)
    seal = _load_bound_json(audit_seal_path, audit_seal_sha256)
    if sha256_file(audit_producer_path) != audit_producer_sha256:
        raise ValueError("source physical-audit producer hash mismatch")
    if (
        manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("status") != "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING"
        or manifest.get("transport_eligible") is not False
    ):
        raise ValueError("source manifest is not the gated parametric CAD")
    plan = manifest.get("plan", {})
    if (
        plan.get("device") != "WISTELL-D"
        or float(plan.get("extent_degrees", math.nan)) != 90.0
        or plan.get("field_period_degrees") != 90.0
        or plan.get("grid_shape") != [80, 90]
        or plan.get("magnet_representation") != "swept_filaments"
        or plan.get("radial_build_mode") != "isolated_cumulative_shells"
        or plan.get("component_order") != [*RADIAL_COMPONENTS, "magnets"]
        or plan.get("safety", {}).get("ports") is not False
        or plan.get("safety", {}).get("post_build_transforms") is not False
        or plan.get("safety", {}).get("imported_physical_solids") is not False
    ):
        raise ValueError("source manifest changes the direct-90 contract")
    vmec = manifest.get("live_vmec_metadata", {})
    if (
        vmec.get("pass") is not True
        or vmec.get("n_field_periods") != 4
        or vmec.get("stellarator_symmetric") is not True
        or vmec.get("lasym_vmec_logical") is not False
        or manifest.get("radial_build_mode") != "isolated_cumulative_shells"
    ):
        raise ValueError("live VMEC/field-period evidence is incomplete")
    if (
        tuple(manifest.get("actual_invessel_component_order", ()))
        != RADIAL_COMPONENTS
    ):
        raise ValueError("radial component order differs from the contract")
    magnet_rows = manifest.get("magnet_inventory", {}).get("magnets", ())
    if (
        manifest.get("magnet_inventory", {}).get("solid_count") != 18
        or [row.get("magnet_id") for row in magnet_rows] != list(MAGNET_IDS)
        or not all(
            row.get("material") == "homogenized_magnet" for row in magnet_rows
        )
        or not all(
            row.get("brep_valid") is True
            and row.get("solid_count") == 1
            and row.get("casing_winding_split") is False
            and row.get("cross_section_cm")
            == {"thickness": 30.0, "width": 30.0}
            for row in magnet_rows
        )
        or manifest.get("magnet_inventory", {}).get(
            "one_homogenized_solid_per_coil"
        )
        is not True
        or manifest.get("magnet_inventory", {}).get("casing_winding_split")
        is not False
    ):
        raise ValueError("magnet inventory is incomplete or split")
    if (
        audit.get("schema") != SOURCE_AUDIT_SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("source_manifest_sha256") != manifest_sha256
        or audit.get("source_immutable") is not True
    ):
        raise ValueError("source physical audit did not pass")
    summary = audit.get("summary", {})
    required = {
        "component_count": 8,
        "magnet_count": 18,
        **EXPECTED_PAIR_COUNTS,
        "invalid_component_count": 0,
        "invalid_magnet_count": 0,
        "component_overlap_failures": 0,
        "magnet_component_overlap_failures": 0,
        "magnet_pair_overlap_failures": 0,
        "clearance_measurement_count": 18,
    }
    if any(summary.get(key) != expected for key, expected in required.items()):
        raise ValueError("source physical-audit summary is incomplete")
    component_pairs = {
        tuple(sorted(row.get("components", ())))
        for row in audit.get("component_pairs", ())
        if row.get("pass") is True
    }
    magnet_component_pairs = {
        (row.get("magnet_id"), row.get("component"))
        for row in audit.get("magnet_component_pairs", ())
        if row.get("pass") is True
    }
    magnet_pairs = {
        tuple(sorted(row.get("magnets", ())))
        for row in audit.get("magnet_pairs", ())
        if row.get("pass") is True
    }
    expected_component_pairs = {
        tuple(sorted((first, second)))
        for index, first in enumerate(RADIAL_COMPONENTS)
        for second in RADIAL_COMPONENTS[index + 1 :]
    }
    expected_magnet_component_pairs = {
        (magnet, component)
        for magnet in MAGNET_IDS
        for component in RADIAL_COMPONENTS
    }
    expected_magnet_pairs = {
        tuple(sorted((first, second)))
        for index, first in enumerate(MAGNET_IDS)
        for second in MAGNET_IDS[index + 1 :]
    }
    if (
        len(audit.get("component_pairs", ())) != 28
        or component_pairs != expected_component_pairs
        or len(audit.get("magnet_component_pairs", ())) != 144
        or magnet_component_pairs != expected_magnet_component_pairs
        or len(audit.get("magnet_pairs", ())) != 153
        or magnet_pairs != expected_magnet_pairs
    ):
        raise ValueError("source physical-audit pair evidence is incomplete")
    clearance_rows = [
        row
        for row in audit["magnet_component_pairs"]
        if row.get("component") == "vacuum_gap"
    ]
    if not all(
        row.get("minimum_clearance", {}).get("status") == "MEASURED"
        and math.isfinite(
            float(row.get("minimum_clearance", {}).get("minimum_distance_cm"))
        )
        and float(row["minimum_clearance"]["minimum_distance_cm"]) > 0.0
        for row in clearance_rows
    ):
        raise ValueError("source physical-audit clearance evidence is invalid")
    if (
        seal.get("schema") != SOURCE_AUDIT_SEAL_SCHEMA
        or seal.get("status") != "PASS"
        or seal.get("report_sha256") != audit_sha256
        or seal.get("source_manifest_sha256") != manifest_sha256
        or seal.get("source_immutable") is not True
    ):
        raise ValueError("source physical-audit seal is invalid")
    artifacts = manifest.get("artifacts", {})
    required_files = (*RADIAL_COMPONENTS, "magnet_set")
    for name in required_files:
        filename = f"{name}.step"
        row = artifacts.get(filename, {})
        path = source_root / filename
        if (
            not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            raise ValueError(f"source artifact integrity failed: {filename}")
    for inventory_name in (
        "source_integrity_before",
        "source_integrity_after",
    ):
        rows = audit.get(inventory_name, ())
        by_path = {row.get("path"): row for row in rows}
        if set(by_path) != set(artifacts) or not all(
            row.get("pass") is True
            and row.get("actual_bytes") == artifacts[name].get("bytes")
            and row.get("actual_sha256") == artifacts[name].get("sha256")
            for name, row in by_path.items()
        ):
            raise ValueError(
                "source physical-audit integrity evidence is invalid"
            )
    return manifest, audit


def _match_manifest_magnets(
    solids: list[Any], rows: list[Mapping[str, Any]]
) -> list[Any]:
    if len(solids) != 18 or len(rows) != 18:
        raise RuntimeError("magnet solid count differs from manifest")
    signatures = [_cadquery_solid_signature(solid) for solid in solids]
    available = set(range(len(solids)))
    ordered = []
    for row in rows:
        expected_volume = float(row["volume_cm3"])
        raw_center = row.get("centroid_cm")
        if (
            not math.isfinite(expected_volume)
            or expected_volume <= 0.0
            or not isinstance(raw_center, (list, tuple))
            or len(raw_center) != 3
        ):
            raise RuntimeError(f"{row.get('magnet_id')} identity is invalid")
        expected_center = [float(value) for value in raw_center]
        if not all(math.isfinite(value) for value in expected_center):
            raise RuntimeError(f"{row.get('magnet_id')} centroid is invalid")
        matches = [
            index
            for index in available
            if math.isclose(
                float(signatures[index]["mass_cm3"]),
                expected_volume,
                rel_tol=VOLUME_RELATIVE_TOLERANCE,
                abs_tol=VOLUME_ABSOLUTE_TOLERANCE_CM3,
            )
            and max(
                abs(float(first) - second)
                for first, second in zip(
                    signatures[index]["center_cm"], expected_center
                )
            )
            <= POSITION_TOLERANCE_CM
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{row['magnet_id']} has {len(matches)} source-solid matches"
            )
        index = matches[0]
        available.remove(index)
        ordered.append(solids[index])
    if available:
        raise RuntimeError("magnet assignment is not bijective")
    return ordered


def _load_ordered_solids(
    source_root: Path, manifest: Mapping[str, Any]
) -> list[Any]:
    import cadquery as cq

    radial = []
    for name in RADIAL_COMPONENTS:
        solids = (
            cq.importers.importStep(str(source_root / f"{name}.step"))
            .val()
            .Solids()
        )
        if len(solids) != 1:
            raise RuntimeError(f"{name} STEP contains {len(solids)} solids")
        radial.append(solids[0])
    magnet_solids = (
        cq.importers.importStep(str(source_root / "magnet_set.step"))
        .val()
        .Solids()
    )
    ordered_magnets = _match_manifest_magnets(
        list(magnet_solids), list(manifest["magnet_inventory"]["magnets"])
    )
    return [*radial, *ordered_magnets]


def _match_imported_volumes(
    gmsh: Any, imported: list[tuple[int, int]], solids: list[Any]
):
    source_signatures = [_cadquery_solid_signature(solid) for solid in solids]
    imported_signatures = [
        _gmsh_volume_signature(gmsh, volume) for volume in imported
    ]
    available = set(range(len(imported)))
    ordered = []
    evidence = []
    for index, (role, source_signature) in enumerate(
        zip(SEMANTIC_ROLES, source_signatures)
    ):
        matches = [
            candidate
            for candidate in available
            if _source_import_signature_match(
                source_signature, imported_signatures[candidate]
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{role} has {len(matches)} Gmsh import matches"
            )
        candidate = matches[0]
        available.remove(candidate)
        ordered.append(imported[candidate])
        evidence.append(
            {
                "source_index": index,
                "semantic_role": role,
                "material": MATERIAL_TAGS[index],
                "imported_dimtag": list(imported[candidate]),
                "source_signature": source_signature,
                "imported_signature": imported_signatures[candidate],
                "pass": True,
            }
        )
    if available or len(set(ordered)) != len(SEMANTIC_ROLES):
        raise RuntimeError("Gmsh import assignment is not bijective")
    return ordered, evidence, source_signatures


def _fragment_one_to_one(gmsh: Any, ordered: list[tuple[int, int]]):
    _, mapping = gmsh.model.occ.fragment(
        ordered[:1], ordered[1:], removeObject=True, removeTool=True
    )
    gmsh.model.occ.synchronize()
    if len(mapping) != len(ordered):
        raise RuntimeError("fragment mapping changed the input count")
    fragmented = []
    for role, row in zip(SEMANTIC_ROLES, mapping):
        volumes = [(int(dim), int(tag)) for dim, tag in row if int(dim) == 3]
        if len(volumes) != 1:
            raise RuntimeError(
                f"{role} maps to {len(volumes)} fragmented volumes"
            )
        fragmented.append(volumes[0])
    entities = {(int(dim), int(tag)) for dim, tag in gmsh.model.getEntities(3)}
    if len(set(fragmented)) != 26 or set(fragmented) != entities:
        raise RuntimeError("fragmented 26-volume inventory is not bijective")
    return fragmented


def _curve_shell_gate(gmsh: Any, volume: tuple[int, int]) -> dict[str, Any]:
    oriented_boundary = gmsh.model.getBoundary(
        [volume], combined=False, oriented=True, recursive=False
    )
    if any(int(dim) != 2 for dim, _ in oriented_boundary):
        raise RuntimeError("volume boundary contains a non-surface entity")
    oriented_surfaces = [int(tag) for _, tag in oriented_boundary]
    counts = Counter()
    orientation = Counter()
    surface_curves: dict[int, set[int]] = {}
    for signed_surface in oriented_surfaces:
        surface = abs(signed_surface)
        surface_sign = 1 if signed_surface > 0 else -1
        curve_boundary = gmsh.model.getBoundary(
            [(2, surface)], combined=False, oriented=True, recursive=False
        )
        if any(int(dim) != 1 for dim, _ in curve_boundary):
            raise RuntimeError("surface boundary contains a non-curve entity")
        signed_curves = [int(tag) for _, tag in curve_boundary]
        surface_curves[surface] = {abs(value) for value in signed_curves}
        for signed_curve in signed_curves:
            curve = abs(signed_curve)
            counts[curve] += 1
            orientation[curve] += surface_sign * (
                1 if signed_curve > 0 else -1
            )
    remaining = {abs(value) for value in oriented_surfaces}
    visited = {remaining.pop()} if remaining else set()
    while remaining:
        known_curves = {
            curve for surface in visited for curve in surface_curves[surface]
        }
        neighbors = {
            surface
            for surface in remaining
            if known_curves & surface_curves[surface]
        }
        if not neighbors:
            break
        visited.update(neighbors)
        remaining.difference_update(neighbors)
    gate = {
        "surface_count": len(oriented_surfaces),
        "surface_ids": sorted(abs(value) for value in oriented_surfaces),
        "curve_count": len(counts),
        "edge_multiplicity_two": bool(counts)
        and all(value == 2 for value in counts.values()),
        "oriented_edge_cancellation": bool(counts)
        and all(value == 0 for value in orientation.values()),
        "connected": bool(visited) and not remaining,
    }
    gate["pass"] = all(
        gate[key]
        for key in (
            "edge_multiplicity_two",
            "oriented_edge_cancellation",
            "connected",
        )
    )
    return gate


def _surface_set_connected(gmsh: Any, surface_ids: list[int]) -> bool:
    if not surface_ids:
        return False
    curves = {}
    for surface in surface_ids:
        boundary = gmsh.model.getBoundary(
            [(2, int(surface))],
            combined=False,
            oriented=False,
            recursive=False,
        )
        if any(int(dim) != 1 for dim, _ in boundary):
            return False
        curves[int(surface)] = {abs(int(tag)) for _, tag in boundary}
    remaining = set(surface_ids)
    visited = {remaining.pop()}
    while remaining:
        known = {curve for surface in visited for curve in curves[surface]}
        neighbors = {
            surface for surface in remaining if known & curves[surface]
        }
        if not neighbors:
            return False
        visited.update(neighbors)
        remaining.difference_update(neighbors)
    return True


def _surface_role_contract(
    surface_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    observed_pairs: dict[frozenset[str], list[int]] = {}
    for row in surface_rows:
        owners = list(row.get("owner_roles", ()))
        area = float(row.get("area_cm2", math.nan))
        if (
            len(owners) not in (1, 2)
            or len(set(owners)) != len(owners)
            or not math.isfinite(area)
            or area <= 0.0
        ):
            raise RuntimeError("surface role/area evidence is invalid")
        if len(owners) == 2:
            pair = frozenset(owners)
            if pair not in ALLOWED_SHARED_ROLE_PAIRS:
                raise RuntimeError(f"surface joins forbidden roles {owners}")
            observed_pairs.setdefault(pair, []).append(int(row["surface_tag"]))
    missing = ALLOWED_SHARED_ROLE_PAIRS - set(observed_pairs)
    if missing:
        raise RuntimeError(
            "missing radial interfaces: "
            + str(sorted(sorted(pair) for pair in missing))
        )
    external = {}
    for role in RADIAL_COMPONENTS:
        surface_ids = [
            int(row["surface_tag"])
            for row in surface_rows
            if row.get("owner_roles") == [role]
            and row.get("periodic_plane") is None
        ]
        permitted = role == RADIAL_COMPONENTS[-1]
        if surface_ids and not permitted:
            raise RuntimeError(
                f"{role} has exposed internal-interface surfaces {surface_ids}"
            )
        if permitted and not surface_ids:
            raise RuntimeError(f"{role} physical exterior is missing")
        external[role] = sorted(surface_ids)
    return {
        "interface_surface_ids": {
            "__".join(sorted(pair)): sorted(surface_ids)
            for pair, surface_ids in observed_pairs.items()
        },
        "radial_nonperiodic_external_surface_ids": external,
        "pass": True,
    }


def _audit_topology(
    gmsh: Any, fragmented: list[tuple[int, int]]
) -> dict[str, Any]:
    volume_index = {
        int(tag): index for index, (_, tag) in enumerate(fragmented)
    }
    model_surfaces = sorted(int(tag) for dim, tag in gmsh.model.getEntities(2))
    owners: dict[int, list[int]] = {surface: [] for surface in model_surfaces}
    per_volume: list[dict[str, Any]] = []
    for index, volume in enumerate(fragmented):
        boundary = gmsh.model.getBoundary(
            [volume], combined=False, oriented=False, recursive=False
        )
        if any(int(dim) != 2 for dim, _ in boundary):
            raise RuntimeError(
                f"{SEMANTIC_ROLES[index]} boundary contains a non-surface"
            )
        surfaces = sorted(int(tag) for _, tag in boundary)
        if len(surfaces) != len(set(surfaces)):
            raise RuntimeError(f"{SEMANTIC_ROLES[index]} repeats a surface")
        for surface in surfaces:
            if surface not in owners:
                raise RuntimeError(
                    "volume boundary references an unknown surface"
                )
            owners[surface].append(index)
        shell = _curve_shell_gate(gmsh, volume)
        if not shell["pass"]:
            raise RuntimeError(
                f"{SEMANTIC_ROLES[index]} OCC shell is not closed"
            )
        if shell["surface_ids"] != surfaces:
            raise RuntimeError(
                f"{SEMANTIC_ROLES[index]} oriented boundary disagrees"
            )
        per_volume.append(
            {
                "index": index,
                "role": SEMANTIC_ROLES[index],
                **shell,
            }
        )
    if any(not value for value in owners.values()):
        raise RuntimeError("Gmsh model has orphan surfaces")
    surface_rows = []
    observed_shared_pairs = set()
    for surface, indices in owners.items():
        unique = sorted(set(indices))
        if len(unique) not in (1, 2) or len(unique) != len(indices):
            raise RuntimeError(
                f"surface {surface} has invalid ownership {indices}"
            )
        upward, _ = gmsh.model.getAdjacencies(2, surface)
        upward_indices = sorted(volume_index[int(tag)] for tag in upward)
        if upward_indices != unique:
            raise RuntimeError(
                f"surface {surface} adjacency disagrees with volume boundaries"
            )
        roles = [SEMANTIC_ROLES[index] for index in unique]
        bounds = [
            float(value) for value in gmsh.model.getBoundingBox(2, surface)
        ]
        periodic_plane = None
        if max(abs(bounds[1]), abs(bounds[4])) <= PERIODIC_PLANE_TOLERANCE_CM:
            periodic_plane = "phi_0"
        elif (
            max(abs(bounds[0]), abs(bounds[3])) <= PERIODIC_PLANE_TOLERANCE_CM
        ):
            periodic_plane = "phi_90"
        area = float(gmsh.model.occ.getMass(2, surface))
        if not math.isfinite(area) or area <= 0.0:
            raise RuntimeError(f"surface {surface} has invalid area {area}")
        if len(roles) == 2:
            pair = frozenset(roles)
            if pair not in ALLOWED_SHARED_ROLE_PAIRS:
                raise RuntimeError(
                    f"surface {surface} joins forbidden roles {roles}"
                )
            observed_shared_pairs.add(pair)
        surface_rows.append(
            {
                "surface_tag": surface,
                "owner_indices": unique,
                "owner_roles": roles,
                "owner_multiplicity": len(unique),
                "area_cm2": area,
                "bounding_box_cm": bounds,
                "periodic_plane": periodic_plane,
            }
        )
    role_contract = _surface_role_contract(surface_rows)
    interface_evidence = []
    for pair in sorted(
        ALLOWED_SHARED_ROLE_PAIRS, key=lambda item: sorted(item)
    ):
        surface_ids = [
            row["surface_tag"]
            for row in surface_rows
            if set(row["owner_roles"]) == set(pair)
        ]
        if not _surface_set_connected(gmsh, surface_ids):
            raise RuntimeError(
                f"radial interface {sorted(pair)} is disconnected"
            )
        interface_evidence.append(
            {
                "roles": sorted(pair),
                "surface_ids": sorted(surface_ids),
                "area_cm2": sum(
                    row["area_cm2"]
                    for row in surface_rows
                    if row["surface_tag"] in surface_ids
                ),
                "connected": True,
                "pass": True,
            }
        )
    radial_external_evidence = {}
    for role in RADIAL_COMPONENTS:
        nonperiodic = [
            row["surface_tag"]
            for row in surface_rows
            if row["owner_roles"] == [role] and row["periodic_plane"] is None
        ]
        permitted = role == RADIAL_COMPONENTS[-1]
        if not permitted and nonperiodic:
            raise RuntimeError(
                f"{role} has exposed internal-interface surfaces {nonperiodic}"
            )
        if permitted and not _surface_set_connected(gmsh, nonperiodic):
            raise RuntimeError(f"{role} physical exterior is disconnected")
        radial_external_evidence[role] = {
            "nonperiodic_external_surface_ids": sorted(nonperiodic),
            "expected_physical_exterior": permitted,
            "connected": permitted,
            "pass": (permitted and bool(nonperiodic)) or not nonperiodic,
        }
    magnet_rows = [
        row for row in per_volume if row["role"].startswith("magnet-")
    ]
    if len(magnet_rows) != 18:
        raise RuntimeError("magnet topology inventory is incomplete")
    periodic_end_areas = {}
    for role in RADIAL_COMPONENTS:
        areas = {
            plane: sum(
                row["area_cm2"]
                for row in surface_rows
                if row["owner_roles"] == [role]
                and row["periodic_plane"] == plane
            )
            for plane in ("phi_0", "phi_90")
        }
        if not (
            areas["phi_0"] > 0.0
            and areas["phi_90"] > 0.0
            and math.isclose(
                areas["phi_0"],
                areas["phi_90"],
                rel_tol=PERIODIC_AREA_RELATIVE_TOLERANCE,
                abs_tol=PERIODIC_AREA_ABSOLUTE_TOLERANCE_CM2,
            )
        ):
            raise RuntimeError(
                f"{role} periodic end-area gate failed: {areas}"
            )
        periodic_end_areas[role] = {**areas, "pass": True}
    return {
        "status": "PREMESH_26_VOLUME_TOPOLOGY_PASS",
        "volume_count": len(fragmented),
        "surface_count": len(surface_rows),
        "shared_surface_count": sum(
            row["owner_multiplicity"] == 2 for row in surface_rows
        ),
        "external_surface_count": sum(
            row["owner_multiplicity"] == 1 for row in surface_rows
        ),
        "allowed_shared_role_pairs": [
            sorted(pair)
            for pair in sorted(
                ALLOWED_SHARED_ROLE_PAIRS, key=lambda item: sorted(item)
            )
        ],
        "per_volume": per_volume,
        "surfaces": surface_rows,
        "surface_role_contract": role_contract,
        "radial_interfaces": interface_evidence,
        "radial_external_surfaces": radial_external_evidence,
        "periodic_end_areas": periodic_end_areas,
        "all_magnet_shells_close": all(row["pass"] for row in magnet_rows),
        "magnet_surface_owner_multiplicity_one": all(
            row["owner_multiplicity"] == 1
            for row in surface_rows
            if any(role.startswith("magnet-") for role in row["owner_roles"])
        ),
        "pass": True,
    }


def _validate_controls(args: argparse.Namespace) -> None:
    if (
        args.min_mesh_size_cm != MESH_MIN_CM
        or args.max_mesh_size_cm != MESH_MAX_CM
        or args.algorithm != MESH_ALGORITHM
        or not 1 <= args.threads <= 64
    ):
        raise ValueError(
            "meshing controls are outside the qualified coarse contract"
        )
    expected = {
        "OMP_NUM_THREADS": str(args.threads),
        "OMP_THREAD_LIMIT": str(args.threads),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    actual = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if actual != expected:
        raise ValueError(
            f"thread environment differs from the contract: {actual}"
        )


def _validate_surface_mesh(vertices: Any, triangles: Any) -> dict[str, Any]:
    import numpy as np

    if len(vertices) != 26 or len(triangles) != 26:
        raise RuntimeError("surface-mesh volume count is not 26")
    rows = []
    for index, (volume_vertices, volume_triangles) in enumerate(
        zip(vertices, triangles)
    ):
        vertex_array = np.asarray(volume_vertices)
        triangle_array = np.asarray(volume_triangles)
        passed = bool(
            vertex_array.size > 0
            and triangle_array.size > 0
            and vertex_array.ndim == 2
            and vertex_array.shape[1] == 3
            and triangle_array.ndim == 2
            and triangle_array.shape[1] == 3
            and np.isfinite(vertex_array).all()
            and np.isfinite(triangle_array).all()
        )
        if not passed:
            raise RuntimeError(
                f"{SEMANTIC_ROLES[index]} surface mesh is empty or malformed"
            )
        rows.append(
            {
                "semantic_role": SEMANTIC_ROLES[index],
                "vertex_count": int(vertex_array.shape[0]),
                "triangle_count": int(triangle_array.shape[0]),
                "pass": True,
            }
        )
    return {"volume_count": len(rows), "per_volume": rows, "pass": True}


def _faceted_volume_identity(volume: Any) -> dict[str, Any]:
    import numpy as np

    from parastell.reference_geometry import _outward_sign
    from parastell.reference_geometry import _triangles
    from parastell.reference_geometry import audit_volume_envelope

    six_volume = 0.0
    centroid_numerator = np.zeros(3)
    for surface in volume.surfaces:
        triangles = _triangles(surface.triangle_coords)
        sign = _outward_sign(surface, int(volume.id))
        signed_six = sign * np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        )
        six_volume += float(signed_six.sum())
        centroid_numerator += np.sum(
            triangles.sum(axis=1) * signed_six[:, None], axis=0
        )
    if not math.isfinite(six_volume) or six_volume <= 0.0:
        raise RuntimeError(f"volume {volume.id} has invalid signed volume")
    center = centroid_numerator / (4.0 * six_volume)
    if not np.isfinite(center).all():
        raise RuntimeError(f"volume {volume.id} has invalid faceted centroid")
    envelope = audit_volume_envelope(volume)
    if envelope.get("closed") is not True:
        raise RuntimeError(
            f"volume {volume.id} faceted envelope is not closed"
        )
    return {
        "volume_global_id": int(volume.id),
        "centroid_cm": [float(value) for value in center],
        "bounding_box_cm": envelope["bounding_box_cm"],
        "volume_cm3": envelope["volume_cm3"],
    }


def _match_faceted_magnet_centers(
    faceted_centers: list[list[float]],
    source_centers: list[list[float]],
    *,
    tolerance_cm: float = FACETED_MAGNET_IDENTITY_TOLERANCE_CM,
) -> list[dict[str, Any]]:
    if len(faceted_centers) != 18 or len(source_centers) != 18:
        raise RuntimeError("magnet centroid inventory is incomplete")
    rows = []
    for index, (magnet_id, center) in enumerate(
        zip(MAGNET_IDS, faceted_centers)
    ):
        matches = [
            candidate
            for candidate, expected in enumerate(source_centers)
            if max(
                abs(float(first) - float(second))
                for first, second in zip(center, expected)
            )
            <= tolerance_cm
        ]
        if matches != [index]:
            raise RuntimeError(
                f"global volume {index + 9} does not uniquely map to {magnet_id}: "
                f"matches={matches}"
            )
        rows.append(
            {
                "magnet_id": magnet_id,
                "volume_global_id": index + 9,
                "source_centroid_cm": source_centers[index],
                "maximum_centroid_delta_cm": max(
                    abs(float(first) - float(second))
                    for first, second in zip(center, source_centers[index])
                ),
                "tolerance_cm": tolerance_cm,
                "unique_match": True,
                "pass": True,
            }
        )
    return rows


def _h5m_writeback_audit(
    h5m_path: Path, source_signatures: list[Mapping[str, Any]]
) -> dict[str, Any]:
    from parastell.native_dagmc_topology import audit_native_moab_topology

    expected_counts = {
        "Vacuum": 2,
        "first_wall": 1,
        "breeder": 1,
        "back_wall": 1,
        "high_temperature_shield": 1,
        "vacuum_vessel": 1,
        "low_temperature_shield": 1,
        "magnets": 18,
    }
    native = audit_native_moab_topology(
        h5m_path, expected_material_counts=expected_counts
    )
    rows = sorted(
        native.get("volume_materials", ()),
        key=lambda row: int(row.get("volume_global_id", -1)),
    )
    ids = [int(row.get("volume_global_id", -1)) for row in rows]
    ordered_materials = [
        (
            row.get("material_groups", [None])[0]
            if len(row.get("material_groups", ())) == 1
            else None
        )
        for row in rows
    ]
    order_pass = ids == list(range(1, 27)) and ordered_materials == list(
        MATERIAL_TAGS
    )
    passed = bool(
        native.get("native_topology_gate_pass") is True
        and native.get("h5m_unchanged") is True
        and order_pass
    )
    if not passed:
        raise RuntimeError("written H5M topology/material-order audit failed")
    import pydagmc

    model = pydagmc.Model(str(h5m_path))
    source_magnet_centers = [
        [float(value) for value in signature["center_cm"]]
        for signature in source_signatures[8:]
    ]
    if len(source_magnet_centers) != 18:
        raise RuntimeError("source magnet signature inventory is incomplete")
    faceted_identities = []
    for index, magnet_id in enumerate(MAGNET_IDS):
        global_id = index + 9
        faceted = _faceted_volume_identity(model.volumes_by_id[global_id])
        faceted_identities.append(faceted)
    magnet_identity = _match_faceted_magnet_centers(
        [row["centroid_cm"] for row in faceted_identities],
        source_magnet_centers,
    )
    for row, faceted in zip(magnet_identity, faceted_identities):
        row["faceted_identity"] = faceted
    return {
        "schema": "parastell.parametric_direct90_h5m_writeback/v1.0.0",
        "status": "H5M_WRITEBACK_PASS_NATIVE_OVERLAP_GATE_PENDING",
        "h5m_sha256": sha256_file(h5m_path),
        "semantic_roles_by_global_volume_id": list(SEMANTIC_ROLES),
        "material_tags_by_global_volume_id": ordered_materials,
        "magnet_identity_by_global_volume_id": magnet_identity,
        "native_topology": native,
        "pass": True,
    }


def export(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output root exists: {output_root}")
    _validate_controls(args)
    manifest, audit = _validate_source_packet(
        source_root,
        manifest_sha256=args.manifest_sha256,
        audit_path=Path(args.audit_path).resolve(),
        audit_sha256=args.audit_sha256,
        audit_seal_path=Path(args.audit_seal_path).resolve(),
        audit_seal_sha256=args.audit_seal_sha256,
        audit_producer_path=Path(args.audit_producer_path).resolve(),
        audit_producer_sha256=args.audit_producer_sha256,
    )
    source_hashes_before = {
        name: sha256_file(source_root / name)
        for name in (
            *[f"{value}.step" for value in RADIAL_COMPONENTS],
            "magnet_set.step",
        )
    }
    solids = _load_ordered_solids(source_root, manifest)
    if len(solids) != 26:
        raise RuntimeError("source CAD does not contain exactly 26 solids")

    import cad_to_dagmc
    import cadquery as cq

    geometry = cq.Compound.makeCompound(solids)
    output_root.mkdir(parents=True)
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
            if int(gmsh.option.getNumber(option)) != args.threads:
                raise RuntimeError(f"Gmsh did not retain {option}")
        _, imported_raw = cad_to_dagmc.get_volumes(
            gmsh, geometry, method="in memory"
        )
        imported = [(int(dim), int(tag)) for dim, tag in imported_raw]
        ordered, import_evidence, source_signatures = _match_imported_volumes(
            gmsh, imported, solids
        )
        fragmented = _fragment_one_to_one(gmsh, ordered)
        fragment_evidence = []
        for index, volume in enumerate(fragmented):
            signature = _gmsh_volume_signature(gmsh, volume)
            passed = _volume_signature_match(
                import_evidence[index]["imported_signature"], signature
            )
            fragment_evidence.append(
                {
                    "semantic_role": SEMANTIC_ROLES[index],
                    "material": MATERIAL_TAGS[index],
                    "fragmented_dimtag": list(volume),
                    "source_signature": source_signatures[index],
                    "fragmented_signature": signature,
                    "pass": passed,
                }
            )
        if not all(row["pass"] for row in fragment_evidence):
            raise RuntimeError(
                "fragmentation changed one or more physical volumes"
            )
        topology = _audit_topology(gmsh, fragmented)
        premesh_path = output_root / "PREMESH_TOPOLOGY.json"
        premesh_path.write_text(
            json.dumps(
                {
                    "schema": "parastell.parametric_direct90_premesh/v1.0.0",
                    "status": "PREMESH_TOPOLOGY_PASS_MESH_PENDING",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_manifest_sha256": args.manifest_sha256,
                    "source_audit_sha256": args.audit_sha256,
                    "source_audit_producer_sha256": (
                        args.audit_producer_sha256
                    ),
                    "producer_sha256": sha256_file(Path(__file__).resolve()),
                    "runtime": _runtime_identity(),
                    "semantic_roles": list(SEMANTIC_ROLES),
                    "material_tags": list(MATERIAL_TAGS),
                    "import_assignment": import_evidence,
                    "fragment_assignment": fragment_evidence,
                    "topology": topology,
                    "transport_eligible": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
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
        mesh_evidence = _validate_surface_mesh(vertices, triangles)
    finally:
        gmsh.finalize()
    h5m_path = output_root / "dagmc.h5m"
    cad_to_dagmc.vertices_to_h5m(
        vertices, triangles, list(MATERIAL_TAGS), h5m_filename=h5m_path
    )
    writeback = _h5m_writeback_audit(h5m_path, source_signatures)
    writeback_path = output_root / "H5M_WRITEBACK_AUDIT.json"
    writeback_path.write_text(
        json.dumps(writeback, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_hashes_after = {
        name: sha256_file(source_root / name) for name in source_hashes_before
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("source CAD changed during export")
    receipt = {
        "schema": "parastell.parametric_direct90_dagmc_export/v1.0.0",
        "status": "DAGMC_EXPORTED_NATIVE_GATES_PENDING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "transport_eligible": False,
        "producer_sha256": sha256_file(Path(__file__).resolve()),
        "runtime": _runtime_identity(),
        "source_manifest_sha256": args.manifest_sha256,
        "source_audit_sha256": args.audit_sha256,
        "source_audit_seal_sha256": args.audit_seal_sha256,
        "source_audit_producer_sha256": args.audit_producer_sha256,
        "source_immutable": True,
        "semantic_roles": list(SEMANTIC_ROLES),
        "material_tags": list(MATERIAL_TAGS),
        "geometry": {
            "device": "WISTELL-D",
            "extent_degrees": 90.0,
            "combined_from_45_degree_models": False,
            "radial_volume_count": 8,
            "magnet_volume_count": 18,
            "physical_volume_count": 26,
            "ports": False,
            "casing_winding_split": False,
        },
        "meshing": {
            "backend": "cad_to_dagmc",
            "imprint": True,
            "minimum_size_cm": args.min_mesh_size_cm,
            "maximum_size_cm": args.max_mesh_size_cm,
            "algorithm": args.algorithm,
            "threads": args.threads,
        },
        "premesh_topology": {
            "path": str(premesh_path),
            "sha256": sha256_file(premesh_path),
            "status": topology["status"],
        },
        "surface_mesh": mesh_evidence,
        "h5m_writeback": {
            "path": str(writeback_path),
            "sha256": sha256_file(writeback_path),
            "status": writeback["status"],
        },
        "h5m": {
            "path": str(h5m_path),
            "bytes": h5m_path.stat().st_size,
            "sha256": sha256_file(h5m_path),
        },
        "elapsed_seconds": time.time() - started,
        "required_successor_gate": "NATIVE_WATERTIGHT_OVERLAP_TOPOLOGY_AND_OPENMC",
        "source_audit_summary": audit["summary"],
    }
    receipt_path = output_root / "DAGMC_EXPORT_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--audit-path", required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--audit-seal-path", required=True)
    parser.add_argument("--audit-seal-sha256", required=True)
    parser.add_argument("--audit-producer-path", required=True)
    parser.add_argument("--audit-producer-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--min-mesh-size-cm", type=float, default=MESH_MIN_CM)
    parser.add_argument("--max-mesh-size-cm", type=float, default=MESH_MAX_CM)
    parser.add_argument("--algorithm", type=int, default=MESH_ALGORITHM)
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
