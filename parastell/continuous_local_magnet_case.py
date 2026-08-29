"""Fail-closed binding of a local swept magnet to a continuous global model."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .coil_filament_inventory import SCHEMA as COIL_INVENTORY_SCHEMA
from .coil_filament_inventory import (
    sha256_file,
    validate_coil_filament_inventory,
)
from .continuous_radial_contract import (
    COUPLING_INTERFACE,
    validate_source_manifest,
)
from .material_identity import validate_material_record

CASE_SCHEMA = "parastell.continuous_local_magnet_case/v1.0.0"
PLAN_SCHEMA = "parastell.continuous_local_magnet_case_plan/v1.0.0"
SURFACE_SCHEMA = "parastell.continuous_magnet_surface_manifest/v1.0.0"
ROOT_SCHEMA = "parastell.root_accepted_continuous_geometry/v1.0.0"
MAP_SCHEMA = "parastell.local_magnet_spatial_coupling_map/v1.0.0"
PHASE_SCHEMA = "parastell.openmc16_surface_phase_space/v1.0.0"
LOCAL_INTERFACE = "outer_casing_external"
CAPTURE_SCOPE = "complete_closed_continuous_magnet_volume_boundary"


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _binding(base: Path, value: Mapping[str, Any], label: str):
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} binding is invalid")
    path = Path(str(value["path"]))
    if not path.is_absolute():
        path = base / path
    path = path.resolve(strict=True)
    digest = sha256_file(path)
    if digest != value["sha256"]:
        raise ValueError(f"{label} hash mismatch")
    return path, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def _bound_json(base: Path, value: Mapping[str, Any], label: str):
    path, evidence = _binding(base, value, label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} is not a JSON object")
    return payload, evidence


def _source_coil_hash(source: Mapping[str, Any]) -> str:
    value = (
        source.get("plan", {}).get("inputs", {}).get("coils", {}).get("sha256")
    )
    if not _valid_sha256(value):
        raise ValueError(
            "global source lacks its upstream coil provenance hash"
        )
    return str(value)


def _proper_rotation(
    transform: Mapping[str, Any],
) -> tuple[list[list[float]], list[float]]:
    rotation = np.asarray(
        transform.get("rotation_global_to_local"), dtype=float
    )
    translation = np.asarray(transform.get("translation_cm"), dtype=float)
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-10
        )
        or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-10
        )
    ):
        raise ValueError("coupling transform is not finite and right-handed")
    return rotation.tolist(), translation.tolist()


def _validate_map(
    mapping: Mapping[str, Any],
    *,
    h5m_sha256: str,
    surface_manifest_sha256: str,
    inner_surface_ids: list[int],
    bank_sha256: str,
    coil_inventory_sha256: str,
    filament_id: str,
    coordinates_sha256: str,
) -> dict[str, Any]:
    global_side = mapping.get("global_source", {})
    local_side = mapping.get("local_target", {})
    station = mapping.get("station_selection", {})
    phase = mapping.get("phase_space_contract", {})
    rotation, translation = _proper_rotation(mapping.get("transform", {}))
    metrics = {"scalar_flux", "heating", "damage_energy"}
    global_position = np.asarray(
        station.get("global_position_cm"), dtype=float
    )
    if (
        mapping.get("schema") != MAP_SCHEMA
        or mapping.get("status") != "QUALIFIED_FOR_BOUNDED_REPLAY"
        or global_side.get("dagmc_sha256") != h5m_sha256
        or global_side.get("surface_manifest_sha256")
        != surface_manifest_sha256
        or global_side.get("bank_sha256") != bank_sha256
        or global_side.get("coupling_interface") != COUPLING_INTERFACE
        or sorted(global_side.get("surface_ids", ())) != inner_surface_ids
        or local_side.get("coil_inventory_sha256") != coil_inventory_sha256
        or local_side.get("filament_id") != filament_id
        or local_side.get("coordinates_sha256") != coordinates_sha256
        or local_side.get("receiving_interface") != LOCAL_INTERFACE
        or station.get("metric") not in metrics
        or global_position.shape != (3,)
        or not np.isfinite(global_position).all()
        or not math.isfinite(float(station.get("arc_length_cm", math.nan)))
        or station.get("periodic_tiling_applied") is not False
        or phase.get("position_mapping") != "declared_spatial_map"
        or phase.get("direction_mapping") != "proper_rotation"
        or phase.get("preserved_fields")
        != [
            "particle_pdg",
            "energy_eV",
            "time_s",
            "openmc_weight",
            "weight_per_source_history",
            "delayed_group",
            "source_file_index",
            "source_record_index",
        ]
        or phase.get("parent_history_id_available") is not False
        or phase.get("canonical_weights_reconditioned") is not False
    ):
        raise ValueError("spatial coupling map is incomplete or cross-bound")
    return {
        "rotation_global_to_local": rotation,
        "translation_cm": translation,
        "determinant": 1.0,
        "right_handed": True,
    }


def _local_components(
    case: Mapping[str, Any],
) -> tuple[list[dict], list[dict]]:
    components = case.get("components")
    materials = case.get("materials")
    if (
        not isinstance(components, list)
        or len(components) != 2
        or not isinstance(materials, list)
        or len(materials) != 2
    ):
        raise ValueError(
            "exactly two local components and materials are required"
        )
    by_material = {}
    for row in materials:
        validate_material_record(row)
        material_id = str(row.get("material_id", "")).strip()
        if not material_id or material_id in by_material:
            raise ValueError("local material IDs are invalid or duplicated")
        by_material[material_id] = dict(row)
    normalized = []
    for row in components:
        if not isinstance(row, Mapping) or set(row) != {
            "component_id",
            "role",
            "material_id",
        }:
            raise ValueError("local component record is invalid")
        normalized.append({key: str(row[key]).strip() for key in row})
    if (
        {row["role"] for row in normalized} != {"outer_casing", "winding_pack"}
        or len({row["component_id"] for row in normalized}) != 2
        or {row["material_id"] for row in normalized} != set(by_material)
        or any(not row["component_id"] for row in normalized)
    ):
        raise ValueError("local component/material mapping is ambiguous")
    return sorted(normalized, key=lambda row: row["role"]), [
        by_material[key] for key in sorted(by_material)
    ]


def resolve_continuous_local_magnet_case(path: str | Path) -> dict[str, Any]:
    """Resolve a case without treating a local filament as a global volume."""
    case_path = Path(path).resolve(strict=True)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    required_case = {
        "schema",
        "case_id",
        "global_transport_geometry",
        "local_magnet_geometry",
        "components",
        "materials",
        "surface_source",
    }
    if (
        not isinstance(case, Mapping)
        or set(case) != required_case
        or case.get("schema") != CASE_SCHEMA
        or not str(case.get("case_id", "")).strip()
    ):
        raise ValueError("unknown continuous local magnet case schema")
    base = case_path.parent
    global_geometry = case.get("global_transport_geometry")
    local_geometry = case.get("local_magnet_geometry")
    source = case.get("surface_source")
    if not all(
        isinstance(value, Mapping)
        for value in (global_geometry, local_geometry, source)
    ):
        raise ValueError("global, local, or source contract is missing")
    if set(global_geometry) != {
        "source_manifest",
        "surface_manifest",
        "dagmc_h5m",
        "root_acceptance_receipt",
    }:
        raise ValueError("global transport geometry fields are ambiguous")
    if set(local_geometry) != {
        "coil_filament_inventory",
        "filament_id",
        "filament_coordinates_sha256",
        "outer_cross_section_cm",
        "case_thickness_cm",
        "sample_mod",
        "dimension_authority",
    }:
        raise ValueError("local magnet geometry fields are ambiguous")
    if set(source) != {
        "bank",
        "phase_space_manifest",
        "source_histories",
        "spatial_coupling_map",
    }:
        raise ValueError("surface source fields are ambiguous")

    source_manifest, source_evidence = _bound_json(
        base, global_geometry["source_manifest"], "global source manifest"
    )
    validate_source_manifest(source_manifest)
    surface_manifest, surface_evidence = _bound_json(
        base,
        global_geometry["surface_manifest"],
        "continuous surface manifest",
    )
    _, h5m_evidence = _binding(
        base, global_geometry["dagmc_h5m"], "global DAGMC H5M"
    )
    root, root_evidence = _bound_json(
        base,
        global_geometry["root_acceptance_receipt"],
        "root acceptance receipt",
    )
    if (
        surface_manifest.get("schema") != SURFACE_SCHEMA
        or surface_manifest.get("status") != "PASS"
        or surface_manifest.get("dagmc_sha256") != h5m_evidence["sha256"]
        or surface_manifest.get("geometry", {}).get("explicit_swept_coils")
        is not False
        or surface_manifest.get("capture_bank", {}).get("closed") is not True
        or surface_manifest.get("capture_bank", {}).get("scope")
        != CAPTURE_SCOPE
        or surface_manifest.get("incoming_replay_interface", {}).get(
            "coupling_interface"
        )
        != COUPLING_INTERFACE
        or surface_manifest.get("incoming_replay_interface", {}).get(
            "closed_manifold_claim"
        )
        is not False
        or surface_manifest.get("physical_h5m_mutation") is not False
    ):
        raise ValueError(
            "global surface manifest is not the continuous-layer contract"
        )
    inner_ids = sorted(
        int(value)
        for value in surface_manifest["incoming_replay_interface"][
            "surface_ids"
        ]
    )
    capture_ids = sorted(
        int(value) for value in surface_manifest["capture_bank"]["surface_ids"]
    )
    if not inner_ids or not set(inner_ids).issubset(capture_ids):
        raise ValueError(
            "continuous inner replay interface is empty or foreign"
        )
    fingerprint = root.get("canonical_geometry_fingerprint")
    if (
        root.get("schema") != ROOT_SCHEMA
        or root.get("acceptance_authority") != "ROOT_GEOMETRY_GATE_ACCEPTED"
        or root.get("dagmc_sha256") != h5m_evidence["sha256"]
        or root.get("source_manifest_sha256") != source_evidence["sha256"]
        or root.get("surface_manifest_sha256") != surface_evidence["sha256"]
        or not _valid_sha256(fingerprint)
        or root.get("global_magnet_representation")
        != "continuous_radial_envelope"
    ):
        raise ValueError("continuous global geometry lacks root acceptance")

    inventory, inventory_evidence = _bound_json(
        base,
        local_geometry["coil_filament_inventory"],
        "coil filament inventory",
    )
    if inventory.get("schema") != COIL_INVENTORY_SCHEMA:
        raise ValueError("local coil inventory schema is invalid")
    validate_coil_filament_inventory(
        inventory, expected_coil_sha256=_source_coil_hash(source_manifest)
    )
    filament_id = str(local_geometry.get("filament_id", ""))
    coordinates_sha256 = str(
        local_geometry.get("filament_coordinates_sha256", "")
    )
    matches = [
        row
        for row in inventory["filaments"]
        if row.get("filament_id") == filament_id
    ]
    if (
        len(matches) != 1
        or matches[0].get("coordinates_sha256") != coordinates_sha256
    ):
        raise ValueError(
            "selected local filament identity is missing or ambiguous"
        )

    dimensions = local_geometry.get("outer_cross_section_cm", {})
    try:
        width = float(dimensions["width"])
        thickness = float(dimensions["thickness"])
        case_thickness = float(local_geometry["case_thickness_cm"])
        sample_mod = int(local_geometry["sample_mod"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "user-supplied local dimensions are incomplete"
        ) from exc
    if (
        any(
            not math.isfinite(value) or value <= 0.0
            for value in (width, thickness, case_thickness)
        )
        or 2.0 * case_thickness >= min(width, thickness)
        or isinstance(local_geometry.get("sample_mod"), bool)
        or sample_mod <= 0
        or local_geometry.get("dimension_authority")
        != "user_supplied_local_model"
    ):
        raise ValueError("user-supplied local dimensions are invalid")
    components, materials = _local_components(case)

    _, bank_evidence = _binding(base, source["bank"], "surface source bank")
    histories = source.get("source_histories")
    if (
        isinstance(histories, bool)
        or not isinstance(histories, int)
        or histories <= 0
    ):
        raise ValueError("source_histories must be a positive integer")
    phase_manifest, phase_evidence = _bound_json(
        base,
        source["phase_space_manifest"],
        "surface phase-space manifest",
    )
    phase_files = phase_manifest.get("source_files")
    if (
        phase_manifest.get("schema") != PHASE_SCHEMA
        or phase_manifest.get("raw_phase_space_pass") is not True
        or phase_manifest.get("source_histories") != histories
        or phase_manifest.get("requested_surface_ids") != capture_ids
        or phase_manifest.get("openmc_surface_bank_fields_required")
        != [
            "r",
            "u",
            "E",
            "time",
            "wgt",
            "delayed_group",
            "surf_id",
            "particle",
        ]
        or not isinstance(phase_files, list)
        or len(phase_files) != 1
        or phase_files[0].get("sha256") != bank_evidence["sha256"]
        or phase_manifest.get("normalization", {}).get("canonical_weight")
        != "openmc_weight / source_histories"
        or phase_manifest.get("normalization", {}).get("tally_conditioning")
        is not False
        or phase_manifest.get("normalization", {}).get(
            "sampling_or_resampling"
        )
        is not False
        or "not stored"
        not in phase_manifest.get("native_field_limitations", {}).get(
            "parent_history_id", ""
        )
    ):
        raise ValueError(
            "surface phase-space manifest is incomplete or cross-bound"
        )
    mapping, map_evidence = _bound_json(
        base, source["spatial_coupling_map"], "spatial coupling map"
    )
    transform = _validate_map(
        mapping,
        h5m_sha256=h5m_evidence["sha256"],
        surface_manifest_sha256=surface_evidence["sha256"],
        inner_surface_ids=inner_ids,
        bank_sha256=bank_evidence["sha256"],
        coil_inventory_sha256=inventory_evidence["sha256"],
        filament_id=filament_id,
        coordinates_sha256=coordinates_sha256,
    )
    return {
        "schema": PLAN_SCHEMA,
        "status": "CONTINUOUS_GLOBAL_AND_LOCAL_COIL_COUPLING_BOUND_NOT_EXECUTED",
        "case_id": str(case.get("case_id", "")).strip(),
        "global_transport_geometry": {
            "representation": "continuous_radial_envelope",
            "dagmc_h5m": h5m_evidence,
            "canonical_geometry_fingerprint": fingerprint,
            "capture_surface_ids": capture_ids,
            "incoming_replay_surface_ids": inner_ids,
            "coupling_interface": COUPLING_INTERFACE,
            "physical_h5m_mutation": False,
        },
        "local_geometry_request": {
            "filament_id": filament_id,
            "filament_source_ordinal": matches[0]["source_ordinal"],
            "filament_coordinates_sha256": coordinates_sha256,
            "coil_input": inventory["coil_input"],
            "parser": inventory["parser"],
            "sector": inventory["sector"],
            "outer_width_cm": width,
            "outer_thickness_cm": thickness,
            "case_thickness_cm": case_thickness,
            "inner_width_cm": width - 2.0 * case_thickness,
            "inner_thickness_cm": thickness - 2.0 * case_thickness,
            "sample_mod": sample_mod,
            "dimension_authority": "user_supplied_local_model",
            "global_volume_parity_claim": False,
        },
        "components": components,
        "materials": materials,
        "surface_source": {
            "bank": bank_evidence,
            "phase_space_manifest": phase_evidence,
            "source_histories": histories,
            "spatial_coupling_map": map_evidence,
            "transform": transform,
            "incoming_only": True,
            "canonical_records_reconditioned": False,
        },
        "bindings": {
            "case": {
                "path": str(case_path),
                "bytes": case_path.stat().st_size,
                "sha256": sha256_file(case_path),
            },
            "source_manifest": source_evidence,
            "surface_manifest": surface_evidence,
            "global_dagmc_h5m": h5m_evidence,
            "root_acceptance_receipt": root_evidence,
            "coil_filament_inventory": inventory_evidence,
            "local_coil_input": inventory["coil_input"],
            "surface_source_bank": bank_evidence,
            "spatial_coupling_map": map_evidence,
            "phase_space_manifest": phase_evidence,
        },
        "claims": {
            "global_swept_coils": False,
            "local_split_is_global_geometry": False,
            "cad_constructed": False,
            "transport_executed": False,
            "production_authorized": False,
        },
    }
