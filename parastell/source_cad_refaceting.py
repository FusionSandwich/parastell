"""Byte-preserving source-CAD reuse for controlled DAGMC refaceting."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping

from .reference_geometry import sha256_file


COMPONENT_STEP_FILES = (
    "chamber.step",
    "first_wall.step",
    "breeder.step",
    "back_wall.step",
    "shield.step",
    "vacuum_vessel.step",
)
SOURCE_CAD_FILES = (
    *COMPONENT_STEP_FILES,
    "magnet_set.step",
    "source_mesh.h5m",
)
MATERIAL_TAGS = (
    "Vacuum",
    "first_wall",
    "breeder",
    "back_wall",
    "shield",
    "vac_vessel",
    *("magnets" for _ in range(18)),
)
EXPECTED_MATERIAL_COUNTS = {
    "Vacuum": 1,
    "first_wall": 1,
    "breeder": 1,
    "back_wall": 1,
    "shield": 1,
    "vac_vessel": 1,
    "magnets": 18,
}
EXPECTED_INTERNAL_ROLE_PAIRS = (
    ("Vacuum", "first_wall"),
    ("first_wall", "breeder"),
    ("breeder", "back_wall"),
    ("back_wall", "shield"),
    ("shield", "vac_vessel"),
)
MANIFEST_SCHEMA = "parastell.refaceted_source_cad_candidate/v1.0.0"
SEAL_SCHEMA = "parastell.refaceted_source_cad_candidate_seal/v1.0.0"
RUNTIME_RECEIPT_SCHEMA = "parastell.geometry_runtime_receipt/v1.1.0"
REQUIRED_RUNTIME_MODULES = (
    "cadquery",
    "cad_to_dagmc",
    "gmsh",
    "pymoab",
    "h5py",
    "OCP",
    "numpy",
)
RUNTIME_NATIVE_MODULES = {
    "pymoab": ("pymoab.core", "pymoab.types"),
    "h5py": ("h5py.h5f",),
    "numpy": ("numpy._core._multiarray_umath",),
}
VOLUME_SIGNATURE_RELATIVE_TOLERANCE = 1.0e-9
VOLUME_SIGNATURE_ABSOLUTE_VOLUME_TOLERANCE_CM3 = 1.0e-7
VOLUME_SIGNATURE_POSITION_TOLERANCE_CM = 5.0e-7
VOLUME_SIGNATURE_ABSOLUTE_INERTIA_TOLERANCE_CM5 = 1.0e-5


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _load_hash_bound_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"hash mismatch for JSON control: {path}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON control must contain an object: {path}")
    return value


def _distribution_version(module_name: str) -> str | None:
    candidates = {
        "OCP": ("cadquery-ocp", "OCP"),
        "pymoab": ("pymoab", "moab"),
        "cad_to_dagmc": ("cad-to-dagmc", "cad_to_dagmc"),
    }.get(module_name, (module_name,))
    for candidate in candidates:
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def runtime_module_receipt(module_name: str) -> dict[str, Any]:
    """Return the import identity used by the probe and the refaceter."""
    module = importlib.import_module(module_name)
    raw_path = getattr(module, "__file__", None)
    path = Path(raw_path).resolve() if raw_path else None
    native_module_names = RUNTIME_NATIVE_MODULES.get(module_name, ())
    native_artifacts = []
    for native_name in native_module_names:
        native_module = importlib.import_module(native_name)
        native_raw_path = getattr(native_module, "__file__", None)
        native_path = (
            Path(native_raw_path).resolve() if native_raw_path else None
        )
        native_artifacts.append(
            {
                "module": native_name,
                "path": str(native_path) if native_path else None,
                "file_sha256": (
                    sha256_file(native_path)
                    if native_path and native_path.is_file()
                    else None
                ),
            }
        )
    if module_name == "gmsh":
        library_name = getattr(getattr(module, "lib", None), "_name", None)
        library_path = None
        if library_name:
            candidate = Path(library_name)
            if not candidate.is_absolute() and path:
                sibling = path.parent / candidate
                candidate = sibling if sibling.is_file() else candidate
            library_path = candidate.resolve()
        native_artifacts.append(
            {
                "module": "gmsh.lib",
                "path": str(library_path) if library_path else None,
                "file_sha256": (
                    sha256_file(library_path)
                    if library_path and library_path.is_file()
                    else None
                ),
            }
        )
    return {
        "module": module_name,
        "import_pass": True,
        "module_version": getattr(module, "__version__", None),
        "distribution_version": _distribution_version(module_name),
        "path": str(path) if path else None,
        "file_sha256": sha256_file(path) if path and path.is_file() else None,
        "native_artifacts": native_artifacts,
    }


def _validate_runtime_receipt(
    path: Path,
    *,
    expected_sha256: str,
    expected_attempt_id: str,
    expected_launch_nonce: str,
    expected_host_alias: str,
    expected_output_dir: Path,
    expected_level: str,
    expected_threads: int,
    expected_source_h5m_path: Path,
    expected_source_h5m_sha256: str,
    expected_controls: Mapping[str, str],
) -> dict[str, Any]:
    """Validate a sealed preflight against the active Python runtime."""
    receipt = _load_hash_bound_json(path, expected_sha256)
    run_binding = receipt.get("run_binding", {})
    gates = {
        "schema": receipt.get("schema") == RUNTIME_RECEIPT_SCHEMA,
        "status": receipt.get("status") == "COMPLETE",
        "scientific_scope": receipt.get("scientific_decision")
        == "NOT_PERFORMED_RUNTIME_PREFLIGHT_ONLY",
        "runtime_gate": receipt.get("runtime_gate_pass") is True,
        "all_imports": receipt.get("all_required_imports_pass") is True,
        "required_modules": receipt.get("required_modules")
        == list(REQUIRED_RUNTIME_MODULES),
        "attempt_id": run_binding.get("attempt_id") == expected_attempt_id,
        "launch_nonce": run_binding.get("launch_nonce")
        == expected_launch_nonce,
        "host_alias": run_binding.get("host_alias") == expected_host_alias,
        "hostname": run_binding.get("hostname")
        == platform.node().split(".", 1)[0],
        "machine": run_binding.get("machine") == platform.machine(),
        "attempt_root": run_binding.get("attempt_root")
        == str(expected_output_dir.resolve().parent),
        "output_dir": run_binding.get("refacet_output_dir")
        == str(expected_output_dir.resolve()),
        "faceting_level": run_binding.get("faceting_level") == expected_level,
        "threads": run_binding.get("threads") == expected_threads,
    }
    python_receipt = receipt.get("python", {})
    executable = Path(sys.executable).resolve()
    active_python = {
        "resolved_executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    gates.update(
        {
            "python_path": python_receipt.get("resolved_executable")
            == active_python["resolved_executable"],
            "python_hash": python_receipt.get("executable_sha256")
            == active_python["executable_sha256"],
            "python_version": python_receipt.get("version")
            == active_python["version"],
            "python_implementation": python_receipt.get("implementation")
            == active_python["implementation"],
        }
    )
    rows = receipt.get("modules", [])
    rows_by_name = {
        row.get("module"): row for row in rows if isinstance(row, Mapping)
    }
    gates["unique_module_rows"] = (
        len(rows) == len(rows_by_name) == len(REQUIRED_RUNTIME_MODULES)
    )
    gates["module_names"] = set(rows_by_name) == set(REQUIRED_RUNTIME_MODULES)
    active_modules = []
    if gates["unique_module_rows"] and gates["module_names"]:
        active_modules = [
            runtime_module_receipt(name) for name in REQUIRED_RUNTIME_MODULES
        ]
        gates["module_identity"] = active_modules == [
            rows_by_name[name] for name in REQUIRED_RUNTIME_MODULES
        ]
        gates["module_versions_recorded"] = all(
            row.get("module_version") or row.get("distribution_version")
            for row in active_modules
        )
        gates["module_files_bound"] = all(
            row.get("path") and _valid_sha256(row.get("file_sha256"))
            for row in active_modules
        )
        gates["native_module_files_bound"] = all(
            artifact.get("path") and _valid_sha256(artifact.get("file_sha256"))
            for row in active_modules
            for artifact in row.get("native_artifacts", [])
        )
    else:
        gates["module_identity"] = False
        gates["module_versions_recorded"] = False
        gates["module_files_bound"] = False
        gates["native_module_files_bound"] = False
    source = receipt.get("source_h5m", {})
    gates.update(
        {
            "source_h5m_path": source.get("path")
            == str(expected_source_h5m_path.resolve()),
            "source_h5m_size": source.get("size_bytes")
            == expected_source_h5m_path.stat().st_size,
            "source_h5m_hash": source.get("sha256_before")
            == source.get("sha256_after")
            == expected_source_h5m_sha256,
            "source_h5m_unchanged": source.get("unchanged") is True,
            "source_h5m_reload": source.get("native_reload_pass") is True
            and isinstance(source.get("native_entity_set_count"), int)
            and source.get("native_entity_set_count") > 0,
        }
    )
    implementation_before = receipt.get("implementation_before", {})
    implementation_after = receipt.get("implementation_after", {})
    current_implementation = {
        "refacet_module": {
            "sha256": sha256_file(Path(__file__)),
        },
        "refacet_cli": {
            "sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "refacet_source_cad_candidate.py"
            ),
        },
    }
    gates["probe_implementation_unchanged"] = (
        receipt.get("implementation_unchanged") is True
        and implementation_before == implementation_after
    )
    gates["refacet_implementation_identity"] = all(
        implementation_before.get(name, {}).get("sha256") == row["sha256"]
        for name, row in current_implementation.items()
    )
    controls_before = receipt.get("control_inputs_before", {})
    controls_after = receipt.get("control_inputs_after", {})
    normalized_receipt_controls = {
        name: row.get("sha256")
        for name, row in controls_before.items()
        if isinstance(row, Mapping)
    }
    gates["control_inputs_unchanged"] = (
        receipt.get("control_inputs_unchanged") is True
        and controls_before == controls_after
    )
    gates["control_input_identity"] = normalized_receipt_controls == dict(
        expected_controls
    )
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise ValueError(f"geometry-runtime receipt failed: {failed}")
    canonical_runtime = {
        "run_binding": run_binding,
        "python": active_python,
        "modules": active_modules,
        "source_h5m_sha256": expected_source_h5m_sha256,
        "implementation": current_implementation,
        "controls": dict(expected_controls),
    }
    return {
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "receipt": receipt,
        "active_python": active_python,
        "active_modules": active_modules,
        "canonical_runtime_sha256": _json_sha256(canonical_runtime),
        "gates": gates,
        "pass": True,
    }


def _artifact_snapshot(directory: Path, names: tuple[str, ...]) -> dict:
    rows = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"required source artifact missing: {path}"
            )
        rows[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return rows


def _fragment_ordered_volumes(
    gmsh: Any, imported: list[tuple[int, int]]
) -> list:
    """Imprint all source solids and retain one material slot per input solid."""
    if len(imported) != len(MATERIAL_TAGS):
        raise RuntimeError(
            f"expected {len(MATERIAL_TAGS)} imported volumes, got {len(imported)}"
        )
    if any(int(dimension) != 3 for dimension, _ in imported):
        raise RuntimeError("CAD import returned a non-volume entity")
    _, mapping = gmsh.model.occ.fragment(
        imported[:1], imported[1:], removeObject=True, removeTool=True
    )
    gmsh.model.occ.synchronize()
    if len(mapping) != len(imported):
        raise RuntimeError(
            "Gmsh fragment did not return one mapping row per input"
        )
    ordered = []
    for index, mapped in enumerate(mapping):
        volumes = [tuple(value) for value in mapped if int(value[0]) == 3]
        if len(volumes) != 1:
            raise RuntimeError(
                f"source solid {index} maps to {len(volumes)} fragmented volumes"
            )
        ordered.append(volumes[0])
    if len(set(ordered)) != len(ordered):
        raise RuntimeError(
            "multiple source solids map to one fragmented volume"
        )
    output_volumes = {tuple(value) for value in gmsh.model.getEntities(3)}
    if set(ordered) != output_volumes:
        raise RuntimeError(
            "fragmented volume inventory differs from material order"
        )
    return ordered


def _gmsh_volume_signature(gmsh: Any, volume: tuple[int, int]) -> dict:
    dimension, tag = volume
    return {
        "mass_cm3": float(gmsh.model.occ.getMass(dimension, tag)),
        "center_cm": [
            float(value)
            for value in gmsh.model.occ.getCenterOfMass(dimension, tag)
        ],
        "bounding_box_cm": [
            float(value) for value in gmsh.model.getBoundingBox(dimension, tag)
        ],
        "inertia_cm5": [
            float(value)
            for value in gmsh.model.occ.getMatrixOfInertia(dimension, tag)
        ],
    }


def _cadquery_solid_signature(solid: Any) -> dict:
    import cadquery as cq

    bounds = solid.BoundingBox()
    return {
        "mass_cm3": float(solid.Volume()),
        "center_cm": [float(value) for value in solid.Center().toTuple()],
        "bounding_box_cm": [
            float(bounds.xmin),
            float(bounds.ymin),
            float(bounds.zmin),
            float(bounds.xmax),
            float(bounds.ymax),
            float(bounds.zmax),
        ],
        "inertia_cm5": [
            float(value)
            for row in cq.Shape.matrixOfInertia(solid)
            for value in row
        ],
    }


def _valid_volume_signature(value: Mapping[str, Any]) -> bool:
    center = value.get("center_cm")
    bounds = value.get("bounding_box_cm")
    inertia = value.get("inertia_cm5")
    mass = value.get("mass_cm3")
    if not isinstance(center, (list, tuple)) or len(center) != 3:
        return False
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 6:
        return False
    if not isinstance(inertia, (list, tuple)) or len(inertia) != 9:
        return False
    values = [mass, *center, *bounds, *inertia]
    try:
        finite = all(math.isfinite(float(item)) for item in values)
    except (TypeError, ValueError):
        return False
    return bool(finite and float(mass) > 0.0)


def _volume_signature_match(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    if not _valid_volume_signature(first) or not _valid_volume_signature(
        second
    ):
        return False
    return bool(
        math.isclose(
            float(first["mass_cm3"]),
            float(second["mass_cm3"]),
            rel_tol=VOLUME_SIGNATURE_RELATIVE_TOLERANCE,
            abs_tol=VOLUME_SIGNATURE_ABSOLUTE_VOLUME_TOLERANCE_CM3,
        )
        and max(
            abs(float(a) - float(b))
            for a, b in zip(first["center_cm"], second["center_cm"])
        )
        <= VOLUME_SIGNATURE_POSITION_TOLERANCE_CM
        and max(
            abs(float(a) - float(b))
            for a, b in zip(
                first["bounding_box_cm"], second["bounding_box_cm"]
            )
        )
        <= VOLUME_SIGNATURE_POSITION_TOLERANCE_CM
        and all(
            math.isclose(
                float(a),
                float(b),
                rel_tol=VOLUME_SIGNATURE_RELATIVE_TOLERANCE,
                abs_tol=VOLUME_SIGNATURE_ABSOLUTE_INERTIA_TOLERANCE_CM5,
            )
            for a, b in zip(first["inertia_cm5"], second["inertia_cm5"])
        )
    )


def _source_import_signature_match(
    source: Mapping[str, Any], imported: Mapping[str, Any]
) -> bool:
    """Match a source solid to its Gmsh OCC import by volume integrals.

    CadQuery 2.7's ``BoundingBox`` is not location-normalized for some nested
    STEP magnet solids, whereas Gmsh's OCC bounds are. Mass, center of mass,
    and the full inertia tensor are computed by OpenCascade volume integrals
    and are exact for the accepted inputs. Bounds remain recorded as
    diagnostic evidence and are required for the subsequent
    Gmsh-import-to-fragment comparison.
    """

    if not _valid_volume_signature(source) or not _valid_volume_signature(
        imported
    ):
        return False
    return bool(
        math.isclose(
            float(source["mass_cm3"]),
            float(imported["mass_cm3"]),
            rel_tol=VOLUME_SIGNATURE_RELATIVE_TOLERANCE,
            abs_tol=VOLUME_SIGNATURE_ABSOLUTE_VOLUME_TOLERANCE_CM3,
        )
        and max(
            abs(float(a) - float(b))
            for a, b in zip(source["center_cm"], imported["center_cm"])
        )
        <= VOLUME_SIGNATURE_POSITION_TOLERANCE_CM
        and all(
            math.isclose(
                float(a),
                float(b),
                rel_tol=VOLUME_SIGNATURE_RELATIVE_TOLERANCE,
                abs_tol=VOLUME_SIGNATURE_ABSOLUTE_INERTIA_TOLERANCE_CM5,
            )
            for a, b in zip(source["inertia_cm5"], imported["inertia_cm5"])
        )
    )


def _match_imported_volumes_to_source_solids(
    gmsh: Any,
    imported_volumes: list[tuple[int, int]],
    source_signatures: list[Mapping[str, Any]],
) -> tuple[list[tuple[int, int]], list[dict]]:
    if len(imported_volumes) != len(source_signatures):
        raise RuntimeError("Gmsh import changed the source-solid count")
    imported_signatures = [
        _gmsh_volume_signature(gmsh, volume) for volume in imported_volumes
    ]
    available = set(range(len(imported_volumes)))
    ordered = []
    evidence = []
    roles = _semantic_roles(list(MATERIAL_TAGS))
    for source_index, source_signature in enumerate(source_signatures):
        matches = sorted(
            imported_index
            for imported_index in available
            if _source_import_signature_match(
                source_signature, imported_signatures[imported_index]
            )
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"source solid {source_index} has {len(matches)} Gmsh import matches"
            )
        imported_index = matches[0]
        available.remove(imported_index)
        ordered.append(imported_volumes[imported_index])
        evidence.append(
            {
                "source_index": source_index,
                "semantic_role": roles[source_index],
                "original_import_index": imported_index,
                "imported_dimtag": list(imported_volumes[imported_index]),
                "source_signature": dict(source_signature),
                "imported_signature": imported_signatures[imported_index],
                "signature_pass": True,
            }
        )
    if available or len(set(ordered)) != len(ordered):
        raise RuntimeError("Gmsh import assignment is not bijective")
    return ordered, evidence


def _semantic_roles(material_tags: tuple[str, ...] | list[str]) -> list[str]:
    magnet_index = 0
    roles = []
    for material in material_tags:
        if material == "magnets":
            roles.append(f"magnet-{magnet_index:04d}")
            magnet_index += 1
        else:
            roles.append(str(material))
    if magnet_index != 18:
        raise ValueError(f"expected 18 magnets, got {magnet_index}")
    if len(roles) != len(MATERIAL_TAGS):
        raise ValueError("semantic role count differs from material contract")
    return roles


def _canonical_role_pair(first: str, second: str) -> list[str]:
    return sorted((str(first), str(second)))


def _topology_summary(
    *,
    roles: list[str],
    surface_owners: Mapping[int, list[int]],
    per_volume_surface_ids: Mapping[int, list[int]],
) -> dict:
    if len(set(roles)) != len(roles):
        raise ValueError("semantic volume roles are not unique")
    invalid_surfaces = []
    internal_pairs = []
    exterior_count = 0
    duplicate_ownership_rows = []
    for surface_id, owner_indices in sorted(surface_owners.items()):
        raw_owners = [int(value) for value in owner_indices]
        owners = sorted(set(raw_owners))
        if len(raw_owners) != len(owners):
            duplicate_ownership_rows.append(
                {"surface_id": int(surface_id), "owner_indices": raw_owners}
            )
        if len(raw_owners) == 1 and len(owners) == 1:
            exterior_count += 1
        elif len(raw_owners) == 2 and len(owners) == 2:
            internal_pairs.append(
                {
                    "surface_id": int(surface_id),
                    "roles": _canonical_role_pair(
                        roles[owners[0]], roles[owners[1]]
                    ),
                }
            )
        else:
            invalid_surfaces.append(
                {"surface_id": int(surface_id), "owner_indices": owners}
            )
    actual_pairs = sorted(row["roles"] for row in internal_pairs)
    expected_pairs = sorted(
        _canonical_role_pair(first, second)
        for first, second in EXPECTED_INTERNAL_ROLE_PAIRS
    )
    per_volume = []
    for index, role in enumerate(roles):
        surface_ids = sorted(
            set(int(value) for value in per_volume_surface_ids.get(index, []))
        )
        per_volume.append(
            {
                "index": index,
                "role": role,
                "surface_count": len(surface_ids),
                "surface_ids": surface_ids,
            }
        )
    total_incidence_count = sum(
        int(row["surface_count"]) for row in per_volume
    )
    surface_count = len(surface_owners)
    gates = {
        "volume_count": len(roles) == 24,
        "surface_count": surface_count == 142,
        "internal_surface_count": len(internal_pairs) == 5,
        "expected_internal_role_pairs": actual_pairs == expected_pairs,
        "surface_owner_multiplicity": not invalid_surfaces,
        "no_duplicate_surface_ownership": not duplicate_ownership_rows,
        "every_volume_has_surfaces": all(
            row["surface_count"] > 0 for row in per_volume
        ),
        "total_incidence_count": total_incidence_count == 147,
    }
    return {
        "volume_count": len(roles),
        "surface_count": surface_count,
        "internal_surface_count": len(internal_pairs),
        "exterior_surface_count": exterior_count,
        "total_incidence_count": total_incidence_count,
        "internal_interfaces": internal_pairs,
        "invalid_surface_owners": invalid_surfaces,
        "duplicate_surface_ownership": duplicate_ownership_rows,
        "per_volume": per_volume,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _topology_contract_from_native_report(report: Mapping[str, Any]) -> dict:
    if report.get("schema") != "parastell.native_moab_topology/v1.0.0":
        raise ValueError("source native report schema is unsupported")
    if report.get("h5m_unchanged") is not True:
        raise ValueError("source native report did not preserve the H5M")
    material_rows = sorted(
        report.get("volume_materials", []),
        key=lambda row: int(row.get("volume_global_id", -1)),
    )
    global_ids = [int(row["volume_global_id"]) for row in material_rows]
    if global_ids != list(range(1, len(MATERIAL_TAGS) + 1)):
        raise ValueError("source native report has unexpected volume IDs")
    material_tags = []
    for row in material_rows:
        groups = row.get("material_groups", [])
        if len(groups) != 1:
            raise ValueError("source volume lacks one material group")
        material_tags.append(str(groups[0]))
    if tuple(material_tags) != MATERIAL_TAGS:
        raise ValueError("source native material order differs from contract")
    roles = _semantic_roles(material_tags)
    index_by_global_id = {
        global_id: index for index, global_id in enumerate(global_ids)
    }
    declared_surface_ids = [
        int(value) for value in report.get("surface_global_ids", [])
    ]
    if len(declared_surface_ids) != len(set(declared_surface_ids)):
        raise ValueError(
            "source native report has duplicate declared surfaces"
        )
    sense_rows = report.get("surface_senses", [])
    sense_surface_ids = [int(row["surface_global_id"]) for row in sense_rows]
    if (
        len(sense_surface_ids) != len(set(sense_surface_ids))
        or sorted(sense_surface_ids) != sorted(declared_surface_ids)
        or not all(row.get("pass") is True for row in sense_rows)
    ):
        raise ValueError("source surface-sense inventory is inconsistent")
    surface_owners: dict[int, list[int]] = {}
    for row in sense_rows:
        surface_id = int(row["surface_global_id"])
        owners = [
            index_by_global_id[int(value)]
            for value in row.get("sense_volume_global_ids", [])
            if value is not None
        ]
        surface_owners[surface_id] = owners
    incidence_rows = report.get("volume_incidence", [])
    closure_rows = report.get("volume_closures", [])
    incidence_ids = [int(row["volume_global_id"]) for row in incidence_rows]
    closure_ids = [int(row["volume_global_id"]) for row in closure_rows]
    if (
        len(incidence_ids) != len(set(incidence_ids))
        or sorted(incidence_ids) != global_ids
        or not all(row.get("pass") is True for row in incidence_rows)
    ):
        raise ValueError("source volume-incidence inventory is inconsistent")
    if (
        len(closure_ids) != len(set(closure_ids))
        or sorted(closure_ids) != global_ids
        or not all(row.get("pass") is True for row in closure_rows)
    ):
        raise ValueError("source volume-closure inventory is inconsistent")
    closure_by_id = {int(row["volume_global_id"]): row for row in closure_rows}
    per_volume_surface_ids = {}
    owners_from_incidence = {surface_id: [] for surface_id in surface_owners}
    for row in incidence_rows:
        global_id = int(row["volume_global_id"])
        sense_ids = [
            int(value) for value in row.get("sense_surface_global_ids", [])
        ]
        child_ids = [
            int(value) for value in row.get("classified_child_global_ids", [])
        ]
        closure_ids_for_volume = [
            int(value)
            for value in closure_by_id[global_id].get("surface_global_ids", [])
        ]
        if (
            len(sense_ids) != len(set(sense_ids))
            or sorted(sense_ids) != sorted(child_ids)
            or sorted(sense_ids) != sorted(closure_ids_for_volume)
            or not set(sense_ids).issubset(surface_owners)
        ):
            raise ValueError(
                f"source topology representations disagree for volume {global_id}"
            )
        index = index_by_global_id[global_id]
        per_volume_surface_ids[index] = sense_ids
        for surface_id in sense_ids:
            owners_from_incidence[surface_id].append(index)
    if {key: sorted(value) for key, value in surface_owners.items()} != {
        key: sorted(value) for key, value in owners_from_incidence.items()
    }:
        raise ValueError("source sense and incidence ownership disagree")
    summary = _topology_summary(
        roles=roles,
        surface_owners=surface_owners,
        per_volume_surface_ids=per_volume_surface_ids,
    )
    summary["native_topology_gate_pass"] = (
        report.get("native_topology_gate_pass") is True
    )
    summary["source_h5m_sha256"] = report.get("raw_h5m_sha256_before")
    summary["pass"] = bool(
        summary["pass"] and summary["native_topology_gate_pass"]
    )
    canonical_payload = {
        key: summary[key]
        for key in (
            "volume_count",
            "surface_count",
            "internal_surface_count",
            "exterior_surface_count",
            "total_incidence_count",
            "internal_interfaces",
            "per_volume",
            "source_h5m_sha256",
        )
    }
    summary["canonical_contract_sha256"] = _json_sha256(canonical_payload)
    return summary


def _topology_contract_from_gmsh(
    gmsh: Any, ordered_volumes: list[tuple[int, int]]
) -> dict:
    roles = _semantic_roles(list(MATERIAL_TAGS))
    surface_owners: dict[int, list[int]] = {}
    per_volume_surface_ids: dict[int, list[int]] = {}
    for index, volume in enumerate(ordered_volumes):
        boundary = gmsh.model.getBoundary(
            [volume], combined=False, oriented=False, recursive=False
        )
        if any(int(dimension) != 2 for dimension, _ in boundary):
            raise RuntimeError("Gmsh volume boundary contains a non-surface")
        raw_surface_ids = [int(tag) for _, tag in boundary]
        if len(raw_surface_ids) != len(set(raw_surface_ids)):
            raise RuntimeError("Gmsh volume boundary repeats a surface")
        surface_ids = sorted(raw_surface_ids)
        per_volume_surface_ids[index] = surface_ids
        for surface_id in surface_ids:
            surface_owners.setdefault(surface_id, []).append(index)
    model_surface_ids = [
        int(tag) for dimension, tag in gmsh.model.getEntities(2)
    ]
    if len(model_surface_ids) != len(set(model_surface_ids)):
        raise RuntimeError("Gmsh model surface inventory contains duplicates")
    if set(model_surface_ids) != set(surface_owners):
        raise RuntimeError("Gmsh model contains orphan or missing surfaces")
    volume_index_by_tag = {
        int(tag): index for index, (_, tag) in enumerate(ordered_volumes)
    }
    upward_adjacency_pass = True
    upward_rows = []
    for surface_id in sorted(model_surface_ids):
        upward, _ = gmsh.model.getAdjacencies(2, surface_id)
        upward_indices = sorted(
            volume_index_by_tag[int(tag)]
            for tag in upward
            if int(tag) in volume_index_by_tag
        )
        expected_indices = sorted(surface_owners[surface_id])
        row_pass = upward_indices == expected_indices
        upward_rows.append(
            {
                "surface_id": surface_id,
                "upward_volume_indices": upward_indices,
                "boundary_owner_indices": expected_indices,
                "pass": row_pass,
            }
        )
        upward_adjacency_pass = upward_adjacency_pass and row_pass
    summary = _topology_summary(
        roles=roles,
        surface_owners=surface_owners,
        per_volume_surface_ids=per_volume_surface_ids,
    )
    summary["model_surface_inventory_pass"] = True
    summary["upward_adjacency"] = upward_rows
    summary["upward_adjacency_pass"] = upward_adjacency_pass
    summary["pass"] = bool(summary["pass"] and upward_adjacency_pass)
    return summary


def _topology_contract_equivalent(
    source: Mapping[str, Any], candidate: Mapping[str, Any]
) -> bool:
    keys = (
        "volume_count",
        "surface_count",
        "internal_surface_count",
        "exterior_surface_count",
        "total_incidence_count",
    )
    if any(source.get(key) != candidate.get(key) for key in keys):
        return False
    source_pairs = sorted(
        row["roles"] for row in source.get("internal_interfaces", [])
    )
    candidate_pairs = sorted(
        row["roles"] for row in candidate.get("internal_interfaces", [])
    )
    if source_pairs != candidate_pairs:
        return False
    source_counts = {
        row["role"]: int(row["surface_count"])
        for row in source.get("per_volume", [])
    }
    candidate_counts = {
        row["role"]: int(row["surface_count"])
        for row in candidate.get("per_volume", [])
    }
    return source_counts == candidate_counts


def _material_order_from_native_report(report: Mapping[str, Any]) -> dict:
    rows = sorted(
        report.get("volume_materials", []),
        key=lambda row: int(row.get("volume_global_id", -1)),
    )
    global_ids = [int(row.get("volume_global_id", -1)) for row in rows]
    tags = [
        (
            row.get("material_groups", [None])[0]
            if len(row.get("material_groups", [])) == 1
            else None
        )
        for row in rows
    ]
    passed = bool(
        report.get("h5m_unchanged") is True
        and report.get("native_topology_gate_pass") is True
        and report.get("material_group_gate_pass") is True
        and len(rows) == len(MATERIAL_TAGS)
        and global_ids == list(range(1, len(MATERIAL_TAGS) + 1))
        and tuple(tags) == MATERIAL_TAGS
    )
    return {
        "source_h5m_sha256": report.get("raw_h5m_sha256_before"),
        "volume_global_ids": global_ids,
        "material_tags_by_global_volume_id": tags,
        "expected_material_counts": dict(EXPECTED_MATERIAL_COUNTS),
        "pass": passed,
    }


def validate_source_cad_packet(
    source_dir: Path,
    source_audit_path: Path,
    source_audit_seal_path: Path,
    physical_change_report_path: Path,
    *,
    expected_source_manifest_sha256: str,
    expected_source_audit_sha256: str,
    expected_source_audit_seal_sha256: str,
    expected_physical_change_report_sha256: str,
    expected_reference_manifest_sha256: str,
    expected_acceptance_criteria_sha256: str,
) -> dict:
    """Validate the terminal source-CAD packet used by every remesh level."""
    manifest_path = source_dir / "candidate_build_manifest.json"
    manifest = _load_hash_bound_json(
        manifest_path, expected_source_manifest_sha256
    )
    audit = _load_hash_bound_json(
        source_audit_path, expected_source_audit_sha256
    )
    seal = _load_hash_bound_json(
        source_audit_seal_path, expected_source_audit_seal_sha256
    )
    physical_change = _load_hash_bound_json(
        physical_change_report_path,
        expected_physical_change_report_sha256,
    )
    manifest_sha256 = expected_source_manifest_sha256
    audit_sha256 = expected_source_audit_sha256
    seal_sha256 = expected_source_audit_seal_sha256
    physical_change_sha256 = expected_physical_change_report_sha256
    snapshot = _artifact_snapshot(source_dir, (*SOURCE_CAD_FILES, "dagmc.h5m"))
    declared = manifest.get("artifacts", {})
    artifact_rows = {
        name: {
            **actual,
            "declared": declared.get(name),
            "pass": declared.get(name) == actual,
        }
        for name, actual in snapshot.items()
    }
    gates = {
        "source_manifest_hash": manifest_sha256
        == expected_source_manifest_sha256,
        "source_audit_hash": audit_sha256 == expected_source_audit_sha256,
        "source_audit_seal_hash": seal_sha256
        == expected_source_audit_seal_sha256,
        "physical_change_report_hash": physical_change_sha256
        == expected_physical_change_report_sha256,
        "physical_change_report_pass": physical_change.get(
            "physical_change_gate_pass"
        )
        is True,
        "source_artifacts_match_manifest": all(
            row["pass"] for row in artifact_rows.values()
        ),
        "source_build_complete": manifest.get("construct_only") is False
        and manifest.get("status") == "BUILD_COMPLETE_PENDING_QUALIFICATION",
        "source_audit_schema": audit.get("schema")
        in {
            "parastell.source_cad_candidate_audit/v1.3.0",
            "parastell.source_cad_candidate_audit/v1.4.0",
        },
        "source_audit_pass": audit.get("source_cad_gate_pass") is True,
        "source_audit_physical_pass": audit.get(
            "source_cad_physical_gate_pass"
        )
        is True,
        "source_audit_reference_integrity": audit.get(
            "reference_integrity", {}
        ).get("pass")
        is True,
        "source_audit_candidate_integrity": audit.get(
            "candidate_integrity", {}
        ).get("pass")
        is True,
        "source_audit_support_integrity": audit.get(
            "physical_change_support_evidence", {}
        ).get("pass")
        is True,
        "source_audit_inputs_immutable": audit.get("postflight", {}).get(
            "input_immutability_pass"
        )
        is True,
        "source_audit_pair_identity": audit.get(
            "pair_identity_contract", {}
        ).get("pass")
        is True,
        "source_audit_physical_binding": audit.get(
            "physical_change_binding", {}
        ).get("pass")
        is True,
        "source_audit_physical_hash_binding": audit.get(
            "physical_change_binding", {}
        ).get("sha256")
        == physical_change_sha256,
        "source_audit_manifest_binding": audit.get(
            "candidate_integrity", {}
        ).get("manifest_sha256")
        == manifest_sha256,
        "source_audit_h5m_binding": audit.get("candidate_h5m_sha256")
        == snapshot["dagmc.h5m"]["sha256"],
        "reference_manifest_binding": audit.get("reference_integrity", {}).get(
            "manifest_sha256"
        )
        == expected_reference_manifest_sha256,
        "acceptance_criteria_binding": audit.get("criteria", {}).get(
            "acceptance_criteria_sha256"
        )
        == expected_acceptance_criteria_sha256,
        "seal_binds_audit": seal.get("source_cad_audit_sha256")
        == audit_sha256,
        "seal_schema": seal.get("schema")
        == "parastell.source_cad_candidate_audit_seal/v1.0.0",
        "seal_pass": seal.get("source_cad_gate_pass") is True
        and seal.get("input_immutability_pass") is True,
    }
    return {
        "source_dir": str(source_dir.resolve()),
        "source_manifest_sha256": manifest_sha256,
        "source_audit_sha256": audit_sha256,
        "source_audit_seal_sha256": seal_sha256,
        "physical_change_report_sha256": physical_change_sha256,
        "reference_manifest_sha256": expected_reference_manifest_sha256,
        "acceptance_criteria_sha256": expected_acceptance_criteria_sha256,
        "artifact_snapshot": snapshot,
        "artifact_rows": artifact_rows,
        "source_manifest": manifest,
        "physical_change_report": physical_change,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _faceting_level(
    criteria: Mapping[str, Any],
    protocol: Mapping[str, Any],
    level: str,
) -> dict:
    if level not in {"coarse", "refined"}:
        raise ValueError("faceting level must be coarse or refined")
    protocol_level = protocol["levels"][level]
    criteria_level = criteria["faceting"][f"{level}_mesh_size_cm"]
    actual = [
        float(protocol_level["minimum_mesh_size_cm"]),
        float(protocol_level["maximum_mesh_size_cm"]),
    ]
    if actual != [float(value) for value in criteria_level]:
        raise ValueError("faceting protocol and acceptance criteria disagree")
    if int(protocol_level["algorithm"]) != 1:
        raise ValueError("only preregistered faceting algorithm 1 is allowed")
    return {
        "level": level,
        "minimum_mesh_size_cm": actual[0],
        "maximum_mesh_size_cm": actual[1],
        "algorithm": 1,
    }


GMSH_PRE_IMPORT_THREAD_OPTIONS = (
    "General.NumThreads",
    "Geometry.OCCParallel",
)
GMSH_PRE_MESH_THREAD_OPTIONS = (
    "General.NumThreads",
    "Geometry.OCCParallel",
    "Mesh.MaxNumThreads1D",
    "Mesh.MaxNumThreads2D",
    "Mesh.MaxNumThreads3D",
)


def _set_and_read_gmsh_thread_contract(
    gmsh: Any,
    threads: int,
    *,
    phase: str,
) -> dict[str, int]:
    """Set and read the frozen Gmsh thread contract for one phase."""
    if threads < 1 or threads > 64:
        raise ValueError("threads must be between 1 and 64")
    if phase == "PRE_IMPORT_FRAGMENT":
        names = GMSH_PRE_IMPORT_THREAD_OPTIONS
    elif phase in {"POST_SIZING", "PRE_MESH"}:
        names = GMSH_PRE_MESH_THREAD_OPTIONS
    else:
        raise ValueError(f"unsupported Gmsh thread phase: {phase}")
    expected = {
        name: 1 if name == "Geometry.OCCParallel" else threads
        for name in names
    }
    for name, value in expected.items():
        gmsh.option.setNumber(name, value)
    actual = {name: int(gmsh.option.getNumber(name)) for name in names}
    if actual != expected:
        raise RuntimeError(
            f"Gmsh thread contract readback failed at {phase}: {actual}"
        )
    return actual


def _emit_stage(
    callback: Any | None,
    stage: str,
    state: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback(stage, state, details=details)


def _load_ordered_source_solids(
    cq: Any,
    input_dir: Path,
    *,
    stage_callback: Any | None = None,
) -> list[Any]:
    """Load the six source STEP files into exact material order."""
    solids: list[Any] = []
    for name in COMPONENT_STEP_FILES:
        _emit_stage(stage_callback, "STEP_IMPORT", "STARTED", filename=name)
        loaded = cq.importers.importStep(str(input_dir / name)).solids().vals()
        if len(loaded) != 1:
            raise RuntimeError(
                f"expected one solid in {name}, got {len(loaded)}"
            )
        solids.append(loaded[0])
        _emit_stage(
            stage_callback,
            "STEP_IMPORT",
            "COMPLETED",
            filename=name,
            solid_count=1,
        )
    _emit_stage(
        stage_callback,
        "STEP_IMPORT",
        "STARTED",
        filename="magnet_set.step",
    )
    magnets = (
        cq.importers.importStep(str(input_dir / "magnet_set.step"))
        .solids()
        .vals()
    )
    if len(magnets) != 18:
        raise RuntimeError(f"expected 18 magnet solids, got {len(magnets)}")
    solids.extend(magnets)
    _emit_stage(
        stage_callback,
        "STEP_IMPORT",
        "COMPLETED",
        filename="magnet_set.step",
        solid_count=len(magnets),
    )
    if len(solids) != len(MATERIAL_TAGS):
        raise RuntimeError("solid/material ordering mismatch")
    return solids


def _run_import_fragment_topology(
    gmsh: Any,
    cad_to_dagmc: Any,
    geometry: Any,
    source_solid_signatures: list[dict[str, Any]],
    source_topology_contract: Mapping[str, Any],
    *,
    threads: int,
    stage_callback: Any | None = None,
) -> dict[str, Any]:
    """Run the shared production import, mapping, fragment, topology path."""
    pre_import_thread_options = _set_and_read_gmsh_thread_contract(
        gmsh, threads, phase="PRE_IMPORT_FRAGMENT"
    )
    _emit_stage(stage_callback, "GMSH_IMPORT", "STARTED")
    _, imported_volumes = cad_to_dagmc.get_volumes(
        gmsh, geometry, method="in memory"
    )
    imported_volumes = list(imported_volumes)
    _emit_stage(
        stage_callback,
        "GMSH_IMPORT",
        "COMPLETED",
        imported_volume_count=len(imported_volumes),
    )
    _emit_stage(stage_callback, "SOURCE_IMPORT_MAPPING", "STARTED")
    imported_volumes, source_import_mapping_evidence = (
        _match_imported_volumes_to_source_solids(
            gmsh, imported_volumes, source_solid_signatures
        )
    )
    imported_signatures = [
        _gmsh_volume_signature(gmsh, volume) for volume in imported_volumes
    ]
    import_mapping_pass = [
        _source_import_signature_match(source_signature, imported_signature)
        for source_signature, imported_signature in zip(
            source_solid_signatures, imported_signatures
        )
    ]
    if not all(import_mapping_pass):
        raise RuntimeError(
            "Gmsh imported volume order differs from source-solid order"
        )
    _emit_stage(
        stage_callback,
        "SOURCE_IMPORT_MAPPING",
        "COMPLETED",
        mapping_count=len(imported_volumes),
    )
    _emit_stage(stage_callback, "BOOLEAN_FRAGMENTS", "STARTED")
    volumes = _fragment_ordered_volumes(gmsh, imported_volumes)
    _emit_stage(
        stage_callback,
        "BOOLEAN_FRAGMENTS",
        "COMPLETED",
        fragmented_volume_count=len(volumes),
    )
    _emit_stage(stage_callback, "FRAGMENT_MAPPING", "STARTED")
    mapped_signatures = [
        _gmsh_volume_signature(gmsh, volume) for volume in volumes
    ]
    semantic_roles = _semantic_roles(list(MATERIAL_TAGS))
    volume_mapping_evidence = []
    for index, (original, mapped) in enumerate(
        zip(imported_signatures, mapped_signatures)
    ):
        signature_pass = _volume_signature_match(original, mapped)
        volume_mapping_evidence.append(
            {
                "index": index,
                "semantic_role": semantic_roles[index],
                "imported_dimtag": list(imported_volumes[index]),
                "fragmented_dimtag": list(volumes[index]),
                "source_solid_signature": source_solid_signatures[index],
                "imported_signature": original,
                "fragmented_signature": mapped,
                "source_to_import_signature_pass": import_mapping_pass[index],
                "signature_pass": signature_pass,
            }
        )
    if not all(row["signature_pass"] for row in volume_mapping_evidence):
        raise RuntimeError(
            "Gmsh fragment changed a source-solid physical signature"
        )
    _emit_stage(
        stage_callback,
        "FRAGMENT_MAPPING",
        "COMPLETED",
        mapping_count=len(volume_mapping_evidence),
    )
    _emit_stage(stage_callback, "TOPOLOGY_CONTRACT", "STARTED")
    gmsh_topology_contract = _topology_contract_from_gmsh(gmsh, volumes)
    gmsh_topology_contract["equivalent_to_source_h5m"] = (
        _topology_contract_equivalent(
            source_topology_contract, gmsh_topology_contract
        )
    )
    gmsh_topology_contract["pass"] = bool(
        gmsh_topology_contract["pass"]
        and gmsh_topology_contract["equivalent_to_source_h5m"]
    )
    if not gmsh_topology_contract["pass"]:
        raise RuntimeError(
            "Gmsh pre-mesh topology differs from source H5M contract"
        )
    _emit_stage(
        stage_callback,
        "TOPOLOGY_CONTRACT",
        "COMPLETED",
        canonical_contract_sha256=gmsh_topology_contract.get(
            "canonical_contract_sha256"
        ),
    )
    return {
        "volumes": volumes,
        "source_import_mapping_evidence": source_import_mapping_evidence,
        "volume_mapping_evidence": volume_mapping_evidence,
        "gmsh_topology_contract": gmsh_topology_contract,
        "pre_import_thread_options": pre_import_thread_options,
    }


@contextmanager
def source_cad_import_fragment_context(
    input_dir: Path,
    source_topology_contract: Mapping[str, Any],
    *,
    threads: int,
    cq_module: Any | None = None,
    cad_to_dagmc_module: Any | None = None,
    stage_callback: Any | None = None,
):
    """Yield the one production import/fragment/topology context.

    Diagnostic and coarse callers must use this same routine.  It owns Gmsh
    initialization/finalization and applies the thread contract before the
    first STEP import and again before any Gmsh import/fragment operation.
    """
    if cq_module is None:
        import cadquery as cq_module
    if cad_to_dagmc_module is None:
        import cad_to_dagmc as cad_to_dagmc_module
    _emit_stage(stage_callback, "GMSH_INITIALIZE", "STARTED")
    gmsh = cad_to_dagmc_module.init_gmsh()
    try:
        initial_thread_options = _set_and_read_gmsh_thread_contract(
            gmsh, threads, phase="PRE_IMPORT_FRAGMENT"
        )
        _emit_stage(
            stage_callback,
            "GMSH_INITIALIZE",
            "COMPLETED",
            thread_options=initial_thread_options,
        )
        solids = _load_ordered_source_solids(
            cq_module,
            Path(input_dir),
            stage_callback=stage_callback,
        )
        source_solid_signatures = [
            _cadquery_solid_signature(solid) for solid in solids
        ]
        if not all(
            _valid_volume_signature(signature)
            for signature in source_solid_signatures
        ):
            raise RuntimeError(
                "source CadQuery solid has an invalid signature"
            )
        geometry = cq_module.Compound.makeCompound(solids)
        result = _run_import_fragment_topology(
            gmsh,
            cad_to_dagmc_module,
            geometry,
            source_solid_signatures,
            source_topology_contract,
            threads=threads,
            stage_callback=stage_callback,
        )
        result.update(
            {
                "gmsh": gmsh,
                "cad_to_dagmc": cad_to_dagmc_module,
                "source_solid_signatures": source_solid_signatures,
                "initial_thread_options": initial_thread_options,
            }
        )
        yield result
    finally:
        gmsh.finalize()


def refacet_source_cad_candidate(
    source_dir: Path,
    source_audit_path: Path,
    source_audit_seal_path: Path,
    physical_change_report_path: Path,
    output_dir: Path,
    *,
    acceptance_criteria_path: Path,
    faceting_protocol_path: Path,
    runtime_receipt_path: Path,
    level: str,
    threads: int,
    expected_source_manifest_sha256: str,
    expected_source_audit_sha256: str,
    expected_source_audit_seal_sha256: str,
    expected_physical_change_report_sha256: str,
    expected_reference_manifest_sha256: str,
    expected_acceptance_criteria_sha256: str,
    expected_faceting_protocol_sha256: str,
    expected_runtime_receipt_sha256: str,
    expected_attempt_id: str,
    expected_launch_nonce: str,
    expected_host_alias: str,
) -> None:
    """Remesh immutable, qualified STEP solids without reconstructing CAD."""
    if output_dir.exists():
        raise FileExistsError(
            f"create-only output already exists: {output_dir}"
        )
    if threads < 1 or threads > 64:
        raise ValueError("threads must be between 1 and 64")
    criteria = _load_hash_bound_json(
        acceptance_criteria_path, expected_acceptance_criteria_sha256
    )
    protocol = _load_hash_bound_json(
        faceting_protocol_path, expected_faceting_protocol_sha256
    )
    packet = validate_source_cad_packet(
        source_dir,
        source_audit_path,
        source_audit_seal_path,
        physical_change_report_path,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_source_audit_sha256=expected_source_audit_sha256,
        expected_source_audit_seal_sha256=(expected_source_audit_seal_sha256),
        expected_physical_change_report_sha256=(
            expected_physical_change_report_sha256
        ),
        expected_reference_manifest_sha256=(
            expected_reference_manifest_sha256
        ),
        expected_acceptance_criteria_sha256=(
            expected_acceptance_criteria_sha256
        ),
    )
    if not packet["pass"]:
        failed = sorted(
            name for name, value in packet["gates"].items() if not value
        )
        raise ValueError(f"source-CAD packet failed: {failed}")
    source_before = _artifact_snapshot(
        source_dir, (*SOURCE_CAD_FILES, "dagmc.h5m")
    )
    if source_before != packet["artifact_snapshot"]:
        raise RuntimeError("source-CAD packet changed after validation")
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "refacet_source_cad_candidate.py"
    )
    implementation_before = {
        "module_sha256": sha256_file(Path(__file__)),
        "script_sha256": sha256_file(script_path),
    }
    control_before = {
        "source_manifest_sha256": sha256_file(
            source_dir / "candidate_build_manifest.json"
        ),
        "source_audit_sha256": sha256_file(source_audit_path),
        "source_audit_seal_sha256": sha256_file(source_audit_seal_path),
        "physical_change_report_sha256": sha256_file(
            physical_change_report_path
        ),
        "acceptance_criteria_sha256": sha256_file(acceptance_criteria_path),
        "faceting_protocol_sha256": sha256_file(faceting_protocol_path),
        "runtime_receipt_sha256": sha256_file(runtime_receipt_path),
    }
    expected_control = {
        "source_manifest_sha256": expected_source_manifest_sha256,
        "source_audit_sha256": expected_source_audit_sha256,
        "source_audit_seal_sha256": expected_source_audit_seal_sha256,
        "physical_change_report_sha256": (
            expected_physical_change_report_sha256
        ),
        "acceptance_criteria_sha256": expected_acceptance_criteria_sha256,
        "faceting_protocol_sha256": expected_faceting_protocol_sha256,
        "runtime_receipt_sha256": expected_runtime_receipt_sha256,
    }
    if control_before != expected_control:
        raise RuntimeError("refaceting control snapshot differs from expected")
    runtime_control_hashes = {
        name: value
        for name, value in expected_control.items()
        if name != "runtime_receipt_sha256"
    }

    def revalidate_runtime() -> dict[str, Any]:
        return _validate_runtime_receipt(
            runtime_receipt_path,
            expected_sha256=expected_runtime_receipt_sha256,
            expected_attempt_id=expected_attempt_id,
            expected_launch_nonce=expected_launch_nonce,
            expected_host_alias=expected_host_alias,
            expected_output_dir=output_dir,
            expected_level=level,
            expected_threads=threads,
            expected_source_h5m_path=source_dir / "dagmc.h5m",
            expected_source_h5m_sha256=source_before["dagmc.h5m"]["sha256"],
            expected_controls=runtime_control_hashes,
        )

    runtime_qualification = revalidate_runtime()
    from .native_dagmc_topology import audit_native_moab_topology

    source_material_report = audit_native_moab_topology(
        source_dir / "dagmc.h5m",
        expected_material_counts=EXPECTED_MATERIAL_COUNTS,
    )
    source_material_order = _material_order_from_native_report(
        source_material_report
    )
    if (
        source_material_report.get("raw_h5m_sha256_before")
        != source_before["dagmc.h5m"]["sha256"]
        or source_material_report.get("raw_h5m_sha256_after")
        != source_before["dagmc.h5m"]["sha256"]
    ):
        raise ValueError("source native report is not bound to source H5M")
    if not source_material_order["pass"]:
        raise ValueError("source H5M material/volume ordering is unqualified")
    source_topology_contract = _topology_contract_from_native_report(
        source_material_report
    )
    if not source_topology_contract["pass"]:
        raise ValueError("source H5M topology contract is unqualified")
    faceting = _faceting_level(criteria, protocol, level)

    output_dir.mkdir(parents=True)
    for name in SOURCE_CAD_FILES:
        shutil.copyfile(source_dir / name, output_dir / name)
    copied = _artifact_snapshot(output_dir, SOURCE_CAD_FILES)
    if copied != {name: source_before[name] for name in SOURCE_CAD_FILES}:
        raise RuntimeError("copied source-CAD artifact mismatch")

    runtime_identity = {
        "qualification_receipt": runtime_qualification,
        "python_runtime": sys.version,
        "h5m_write_method": "h5py",
    }
    with source_cad_import_fragment_context(
        output_dir,
        source_topology_contract,
        threads=threads,
    ) as shared:
        gmsh = shared["gmsh"]
        cad_to_dagmc = shared["cad_to_dagmc"]
        volumes = shared["volumes"]
        source_solid_signatures = shared["source_solid_signatures"]
        source_import_mapping_evidence = shared[
            "source_import_mapping_evidence"
        ]
        volume_mapping_evidence = shared["volume_mapping_evidence"]
        gmsh_topology_contract = shared["gmsh_topology_contract"]
        cad_to_dagmc.set_sizes_for_mesh(
            gmsh,
            min_mesh_size=faceting["minimum_mesh_size_cm"],
            max_mesh_size=faceting["maximum_mesh_size_cm"],
            mesh_algorithm=faceting["algorithm"],
        )
        post_sizing_thread_options = _set_and_read_gmsh_thread_contract(
            gmsh, threads, phase="POST_SIZING"
        )
        pre_mesh_thread_options = _set_and_read_gmsh_thread_contract(
            gmsh, threads, phase="PRE_MESH"
        )
        gmsh_thread_options = {
            "initial": shared["initial_thread_options"],
            "pre_import_fragment": shared["pre_import_thread_options"],
            "post_sizing": post_sizing_thread_options,
            "pre_mesh": pre_mesh_thread_options,
        }
        gmsh.model.mesh.generate(dim=2)
        vertices, triangles = cad_to_dagmc.mesh_to_vertices_and_triangles(
            volumes
        )
    cad_to_dagmc.vertices_to_h5m(
        vertices,
        triangles,
        list(MATERIAL_TAGS),
        h5m_filename=output_dir / "dagmc.h5m",
        method="h5py",
    )
    output_native_report = audit_native_moab_topology(
        output_dir / "dagmc.h5m",
        expected_material_counts=EXPECTED_MATERIAL_COUNTS,
    )
    output_material_order = _material_order_from_native_report(
        output_native_report
    )
    output_topology_contract = _topology_contract_from_native_report(
        output_native_report
    )
    output_topology_contract["equivalent_to_source_h5m"] = (
        _topology_contract_equivalent(
            source_topology_contract, output_topology_contract
        )
    )
    output_topology_contract["pass"] = bool(
        output_topology_contract["pass"]
        and output_topology_contract["equivalent_to_source_h5m"]
        and output_material_order["pass"]
    )
    if not output_topology_contract["pass"]:
        raise RuntimeError("written H5M topology/material readback failed")

    source_after = _artifact_snapshot(
        source_dir, (*SOURCE_CAD_FILES, "dagmc.h5m")
    )
    implementation_after = {
        "module_sha256": sha256_file(Path(__file__)),
        "script_sha256": sha256_file(script_path),
    }
    control_after = {
        "source_manifest_sha256": sha256_file(
            source_dir / "candidate_build_manifest.json"
        ),
        "source_audit_sha256": sha256_file(source_audit_path),
        "source_audit_seal_sha256": sha256_file(source_audit_seal_path),
        "physical_change_report_sha256": sha256_file(
            physical_change_report_path
        ),
        "acceptance_criteria_sha256": sha256_file(acceptance_criteria_path),
        "faceting_protocol_sha256": sha256_file(faceting_protocol_path),
        "runtime_receipt_sha256": sha256_file(runtime_receipt_path),
    }
    if source_after != source_before:
        raise RuntimeError(
            "qualified source artifacts changed during refaceting"
        )
    if implementation_after != implementation_before:
        raise RuntimeError(
            "refaceting implementation changed during execution"
        )
    if control_after != control_before:
        raise RuntimeError("refaceting control input changed during execution")
    copied_after = _artifact_snapshot(output_dir, SOURCE_CAD_FILES)
    if copied_after != copied:
        raise RuntimeError("copied source-CAD input changed during refaceting")
    runtime_after = revalidate_runtime()
    if (
        runtime_after["canonical_runtime_sha256"]
        != runtime_qualification["canonical_runtime_sha256"]
    ):
        raise RuntimeError("geometry runtime changed during refaceting")
    artifacts = _artifact_snapshot(
        output_dir, (*SOURCE_CAD_FILES, "dagmc.h5m")
    )
    source_manifest = packet["source_manifest"]
    physical_change = packet["physical_change_report"]
    component_canonical = {
        row["component"]: row["candidate"]["canonical_step_sha256"]
        for row in physical_change.get("components", [])
    }
    if set(component_canonical) != {
        name.removesuffix(".step") for name in COMPONENT_STEP_FILES
    }:
        raise RuntimeError(
            "physical receipt lacks canonical component identities"
        )
    physical_identity = {
        "source_cad_artifacts": {
            name: source_before[name] for name in SOURCE_CAD_FILES
        },
        "measurement_sha256": source_manifest.get("measurement_sha256"),
        "vmec_sha256": source_manifest.get("vmec_sha256"),
        "coils_sha256": source_manifest.get("coils_sha256"),
        "acceptance_criteria_sha256": expected_acceptance_criteria_sha256,
        "faceting_protocol_sha256": expected_faceting_protocol_sha256,
        "source_manifest_sha256": packet["source_manifest_sha256"],
        "source_audit_sha256": packet["source_audit_sha256"],
        "source_audit_seal_sha256": packet["source_audit_seal_sha256"],
        "physical_change_report_sha256": packet[
            "physical_change_report_sha256"
        ],
        "source_h5m_sha256": source_before["dagmc.h5m"]["sha256"],
        "source_topology_contract_sha256": source_topology_contract[
            "canonical_contract_sha256"
        ],
        "runtime_receipt_sha256": expected_runtime_receipt_sha256,
        "corrected_thickness_matrices_cm": source_manifest.get(
            "corrected_thickness_matrices_cm"
        ),
        "geometry_invariants": source_manifest.get("geometry_invariants"),
        "material_tags_by_solid_order": list(MATERIAL_TAGS),
        "component_canonical_step_sha256": component_canonical,
        "magnet_canonical_step_sha256": physical_change.get(
            "magnet_identity", {}
        ).get("candidate_canonical_step_sha256"),
        "source_mesh_canonical_fingerprint": physical_change.get(
            "source_mesh_identity", {}
        )
        .get("candidate", {})
        .get("canonical_fingerprint"),
        "source_cad_topology_operation": (
            "One ordered CadQuery compound containing 24 source solids is "
            "imported into Gmsh, each returned volume is matched back to its "
            "source solid by OpenCascade volume, center of mass, and full "
            "inertia tensor, then Gmsh OCC BooleanFragments imprints shared "
            "topology; exactly one unique output volume is required per "
            "mapped input. Gmsh-import-to-fragment preservation additionally "
            "requires bounding-box equality"
        ),
    }
    required_identity_hashes = {
        key: physical_identity[key]
        for key in (
            "measurement_sha256",
            "vmec_sha256",
            "coils_sha256",
            "acceptance_criteria_sha256",
            "faceting_protocol_sha256",
            "source_manifest_sha256",
            "source_audit_sha256",
            "source_audit_seal_sha256",
            "physical_change_report_sha256",
            "source_h5m_sha256",
            "source_topology_contract_sha256",
            "runtime_receipt_sha256",
            "magnet_canonical_step_sha256",
            "source_mesh_canonical_fingerprint",
        )
    }
    required_identity_hashes.update(
        {
            f"component:{name}": value
            for name, value in component_canonical.items()
        }
    )
    required_identity_hashes.update(
        {
            f"artifact:{name}": row["sha256"]
            for name, row in physical_identity["source_cad_artifacts"].items()
        }
    )
    invalid_identity_hashes = sorted(
        name
        for name, value in required_identity_hashes.items()
        if not _valid_sha256(value)
    )
    if invalid_identity_hashes:
        raise RuntimeError(
            f"physical identity contains invalid hashes: {invalid_identity_hashes}"
        )
    if not physical_identity["corrected_thickness_matrices_cm"]:
        raise RuntimeError("physical identity lacks corrected radial build")
    if not physical_identity["geometry_invariants"]:
        raise RuntimeError("physical identity lacks geometry invariants")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "BUILD_COMPLETE_PENDING_QUALIFICATION",
        "construct_only": False,
        "refaceting_only": True,
        "faceting": faceting,
        "threads": threads,
        "acceptance_criteria": {
            "path": str(acceptance_criteria_path.resolve()),
            "sha256": expected_acceptance_criteria_sha256,
        },
        "faceting_protocol": {
            "path": str(faceting_protocol_path.resolve()),
            "sha256": expected_faceting_protocol_sha256,
        },
        "source_cad_qualification": {
            key: packet[key]
            for key in (
                "source_manifest_sha256",
                "source_audit_sha256",
                "source_audit_seal_sha256",
                "physical_change_report_sha256",
                "reference_manifest_sha256",
            )
        },
        "inherited_source_cad_physical_gate_pass": packet["pass"],
        "physical_input_identity": physical_identity,
        "physical_input_identity_sha256": _json_sha256(physical_identity),
        "material_tags_by_solid_order": list(MATERIAL_TAGS),
        "source_material_order_evidence": source_material_order,
        "source_topology_contract": source_topology_contract,
        "source_to_gmsh_import_mapping_evidence": (
            source_import_mapping_evidence
        ),
        "volume_signature_tolerances": {
            "relative_volume": VOLUME_SIGNATURE_RELATIVE_TOLERANCE,
            "absolute_volume_cm3": (
                VOLUME_SIGNATURE_ABSOLUTE_VOLUME_TOLERANCE_CM3
            ),
            "position_cm": VOLUME_SIGNATURE_POSITION_TOLERANCE_CM,
            "absolute_inertia_cm5": (
                VOLUME_SIGNATURE_ABSOLUTE_INERTIA_TOLERANCE_CM5
            ),
            "source_to_import_required_invariants": [
                "volume",
                "center_of_mass",
                "full_inertia_tensor",
                "unique_bijection",
            ],
            "import_to_fragment_required_invariants": [
                "volume",
                "center_of_mass",
                "full_inertia_tensor",
                "bounding_box",
                "unique_bijection",
            ],
            "cadquery_source_bounding_box_status": (
                "DIAGNOSTIC_ONLY_NOT_LOCATION_NORMALIZED_FOR_NESTED_MAGNETS"
            ),
        },
        "volume_mapping_evidence": volume_mapping_evidence,
        "gmsh_pre_mesh_topology_contract": gmsh_topology_contract,
        "written_h5m_material_order_evidence": output_material_order,
        "written_h5m_topology_contract": output_topology_contract,
        "gmsh_thread_options": gmsh_thread_options,
        "implementation": implementation_before,
        "runtime_identity": runtime_identity,
        "runtime_immutability_pass": True,
        "canonical_runtime_sha256": runtime_qualification[
            "canonical_runtime_sha256"
        ],
        "control_inputs_before": control_before,
        "control_inputs_after": control_after,
        "input_immutability_pass": source_before == source_after
        and control_before == control_after
        and copied_after == copied,
        "artifacts": artifacts,
    }
    serialized = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary = output_dir / ".candidate_build_manifest.json.tmp"
    temporary.write_text(serialized, encoding="utf-8")
    if json.loads(temporary.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("refaceting manifest readback mismatch")
    temporary.replace(output_dir / "candidate_build_manifest.json")
    manifest_path = output_dir / "candidate_build_manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    source_final = _artifact_snapshot(
        source_dir, (*SOURCE_CAD_FILES, "dagmc.h5m")
    )
    copied_final = _artifact_snapshot(output_dir, SOURCE_CAD_FILES)
    implementation_final = {
        "module_sha256": sha256_file(Path(__file__)),
        "script_sha256": sha256_file(script_path),
    }
    control_final = {
        "source_manifest_sha256": sha256_file(
            source_dir / "candidate_build_manifest.json"
        ),
        "source_audit_sha256": sha256_file(source_audit_path),
        "source_audit_seal_sha256": sha256_file(source_audit_seal_path),
        "physical_change_report_sha256": sha256_file(
            physical_change_report_path
        ),
        "acceptance_criteria_sha256": sha256_file(acceptance_criteria_path),
        "faceting_protocol_sha256": sha256_file(faceting_protocol_path),
        "runtime_receipt_sha256": sha256_file(runtime_receipt_path),
    }
    if source_final != source_before:
        raise RuntimeError("qualified source artifacts changed before sealing")
    if copied_final != copied:
        raise RuntimeError("copied source-CAD input changed before sealing")
    if control_final != control_before:
        raise RuntimeError("refaceting controls changed before sealing")
    if implementation_final != implementation_before:
        raise RuntimeError("refaceting implementation changed before sealing")
    if sha256_file(manifest_path) != manifest_sha256:
        raise RuntimeError("refaceting manifest changed before sealing")
    if (
        sha256_file(output_dir / "dagmc.h5m")
        != artifacts["dagmc.h5m"]["sha256"]
    ):
        raise RuntimeError("refaceted H5M changed before sealing")
    runtime_final = revalidate_runtime()
    if (
        runtime_final["canonical_runtime_sha256"]
        != runtime_qualification["canonical_runtime_sha256"]
    ):
        raise RuntimeError("geometry runtime changed before sealing")
    seal = {
        "schema": SEAL_SCHEMA,
        "candidate_build_manifest_sha256": manifest_sha256,
        "dagmc_h5m_sha256": artifacts["dagmc.h5m"]["sha256"],
        "physical_input_identity_sha256": manifest[
            "physical_input_identity_sha256"
        ],
        "inherited_source_cad_physical_gate_pass": packet["pass"],
        "input_immutability_pass": manifest["input_immutability_pass"],
        "source_topology_contract_pass": source_topology_contract["pass"],
        "gmsh_pre_mesh_topology_contract_pass": gmsh_topology_contract["pass"],
        "written_h5m_topology_contract_pass": output_topology_contract["pass"],
        "runtime_qualification_pass": runtime_qualification["pass"],
        "runtime_receipt_sha256": expected_runtime_receipt_sha256,
        "runtime_immutability_pass": True,
        "canonical_runtime_sha256": runtime_qualification[
            "canonical_runtime_sha256"
        ],
    }
    seal_serialized = (
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    seal_temporary = output_dir / ".refaceting_build_seal.json.tmp"
    seal_temporary.write_text(seal_serialized, encoding="utf-8")
    if json.loads(seal_temporary.read_text(encoding="utf-8")) != seal:
        raise RuntimeError("refaceting seal readback mismatch")
    seal_temporary.replace(output_dir / "refaceting_build_seal.json")
    seal_path = output_dir / "refaceting_build_seal.json"
    if json.loads(seal_path.read_text(encoding="utf-8")) != seal:
        raise RuntimeError("published refaceting seal readback mismatch")
