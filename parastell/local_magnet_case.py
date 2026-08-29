"""Hash-bound planning for one casing-plus-winding local magnet model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .material_identity import validate_material_record
from .surface_source_phase_space import read_openmc16_surface_sources
from .transport_response_plan import validate_response_plan


SCHEMA = "parastell.local_magnet_case/v1.0.0"
PLAN_SCHEMA = "parastell.local_magnet_case_plan/v1.0.0"
SOURCE_MANIFEST_SCHEMA = "parastell.parametric_source_cad/v1.0.0"
SURFACE_MANIFEST_SCHEMA = "parastell.parametric_magnet_surface_manifest/v1.0.0"
GLOBAL_INTERFACE = "homogenized_magnet_outer_boundary"
LOCAL_INTERFACE = "outer_casing_external"
ROOT_ACCEPTANCE_SCHEMA = "parastell.root_accepted_magnet_inventory/v1.0.0"
STRICT_AUDIT_SCHEMA = "parastell.openmc16_surface_run_audit/v1.0.0"
PHASE_SPACE_SCHEMA = "parastell.openmc16_surface_phase_space/v1.0.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str):
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _canonical_material_tag(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("mat:"):
        text = text[4:]
    return text.replace("-", "_").replace(" ", "_")


def _bound_path(
    base: Path, binding: Mapping[str, Any], *, label: str
) -> tuple[Path, dict[str, Any]]:
    _require_keys(binding, {"path", "sha256"}, label)
    expected = str(binding["sha256"])
    if len(expected) != 64 or any(
        c not in "0123456789abcdef" for c in expected
    ):
        raise ValueError(f"{label} SHA-256 is invalid")
    path = Path(str(binding["path"]))
    if not path.is_absolute():
        path = base / path
    path = path.resolve(strict=True)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch")
    return path, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def _bound_json(
    base: Path, binding: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path, evidence = _bound_path(base, binding, label=label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value, evidence


def _unique_row(
    rows: Any, key: str, value: str, label: str
) -> tuple[int, dict]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} is not a list")
    matches = [
        (index, row)
        for index, row in enumerate(rows)
        if isinstance(row, Mapping) and row.get(key) == value
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} does not contain one unique {value!r}")
    index, row = matches[0]
    return index, dict(row)


def _inventory_semantics(rows: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} components are invalid")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} component is invalid")
        source = row.get("source_geometry")
        if not isinstance(source, Mapping):
            raise ValueError(f"{label} source geometry is invalid")
        volume_id = _positive_integer(
            row.get("dagmc_volume_id"), f"{label} DAGMC volume ID"
        )
        surfaces = row.get("surface_ids")
        if (
            not isinstance(surfaces, list)
            or not surfaces
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in surfaces
            )
            or len(surfaces) != len(set(surfaces))
        ):
            raise ValueError(f"{label} surface identities are invalid")
        source_hash = str(source.get("sha256", ""))
        item = {
            "magnet_id": str(row.get("magnet_id", "")).strip(),
            "component_id": str(row.get("component_id", "")).strip(),
            "dagmc_volume_id": volume_id,
            "material_tag": _canonical_material_tag(row.get("material_tag")),
            "surface_ids": sorted(surfaces),
            "source_geometry_kind": source.get("kind"),
            "source_geometry_sha256": source_hash,
        }
        if (
            not item["magnet_id"]
            or not item["component_id"]
            or not item["material_tag"]
            or item["source_geometry_kind"] != "parametric_reference_manifest"
            or not _valid_sha256(source_hash)
        ):
            raise ValueError(f"{label} component semantics are invalid")
        normalized.append(item)
    uniqueness = (
        {row["magnet_id"] for row in normalized},
        {row["component_id"] for row in normalized},
        {row["dagmc_volume_id"] for row in normalized},
    )
    if any(len(values) != len(normalized) for values in uniqueness):
        raise ValueError(f"{label} component identities are duplicated")
    return sorted(normalized, key=lambda row: row["magnet_id"])


def resolve_local_magnet_case(path: str | Path) -> dict[str, Any]:
    """Resolve one local model without inventing dimensions or materials."""
    case_path = Path(path).resolve(strict=True)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(case, Mapping):
        raise ValueError("local magnet case is not a JSON object")
    _require_keys(
        case,
        {
            "schema",
            "case_id",
            "global_geometry",
            "magnet_selection",
            "case_thickness_cm",
            "components",
            "materials",
            "surface_source",
            "response_plan",
        },
        "local magnet case",
    )
    if (
        case.get("schema") != SCHEMA
        or not str(case.get("case_id", "")).strip()
    ):
        raise ValueError("local magnet case identity is invalid")
    base = case_path.parent

    global_geometry = case["global_geometry"]
    if not isinstance(global_geometry, Mapping):
        raise ValueError("global_geometry must be an object")
    _require_keys(
        global_geometry,
        {
            "source_cad_manifest",
            "magnet_surface_manifest",
            "dagmc_h5m",
            "coil_input",
            "root_acceptance_receipt",
        },
        "global_geometry",
    )
    source_manifest, source_evidence = _bound_json(
        base,
        global_geometry["source_cad_manifest"],
        label="source CAD manifest",
    )
    surface_manifest, surface_evidence = _bound_json(
        base,
        global_geometry["magnet_surface_manifest"],
        label="magnet surface manifest",
    )
    _, h5m_evidence = _bound_path(
        base, global_geometry["dagmc_h5m"], label="global DAGMC H5M"
    )
    coil_path, coil_evidence = _bound_path(
        base, global_geometry["coil_input"], label="global coil input"
    )
    root_acceptance, root_evidence = _bound_json(
        base,
        global_geometry["root_acceptance_receipt"],
        label="root acceptance receipt",
    )

    selection = case["magnet_selection"]
    if not isinstance(selection, Mapping):
        raise ValueError("magnet_selection must be an object")
    _require_keys(
        selection,
        {
            "magnet_id",
            "filament_index",
            "filament_coordinates_sha256",
            "global_bank_interface",
            "local_coupling_interface",
        },
        "magnet_selection",
    )
    magnet_id = str(selection["magnet_id"]).strip()
    filament_index = selection["filament_index"]
    if (
        not magnet_id
        or isinstance(filament_index, bool)
        or not isinstance(filament_index, int)
        or filament_index < 0
        or selection["global_bank_interface"] != GLOBAL_INTERFACE
        or selection["local_coupling_interface"] != LOCAL_INTERFACE
    ):
        raise ValueError("magnet selection or coupling interface is invalid")

    source_plan = source_manifest.get("plan", {})
    source_coils = source_plan.get("inputs", {}).get("coils", {})
    if (
        source_manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
        or source_manifest.get("status")
        != "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING"
        or source_manifest.get("plan", {}).get("extent_degrees") != 90.0
        or source_manifest.get("plan", {}).get("magnet_representation")
        != "swept_filaments"
        or source_manifest.get("magnet_inventory", {}).get(
            "casing_winding_split"
        )
        is not False
        or source_coils.get("sha256") != coil_evidence["sha256"]
    ):
        raise ValueError(
            "source CAD manifest is not the direct-90 global model"
        )
    source_index, source_magnet = _unique_row(
        source_manifest.get("magnet_inventory", {}).get("magnets"),
        "magnet_id",
        magnet_id,
        "source magnet inventory",
    )
    coordinates_sha256 = str(selection["filament_coordinates_sha256"])
    if (
        filament_index != source_index
        or source_magnet.get("coordinates_sha256") != coordinates_sha256
        or not _valid_sha256(coordinates_sha256)
    ):
        raise ValueError(
            "filament index/hash is not bound to the selected magnet"
        )
    cross_section = source_magnet.get("cross_section_cm", {})
    try:
        width = float(cross_section["width"])
        thickness = float(cross_section["thickness"])
        case_thickness = float(case["case_thickness_cm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("local magnet dimensions are invalid") from exc
    if (
        not all(
            math.isfinite(value)
            for value in (width, thickness, case_thickness)
        )
        or min(width, thickness, case_thickness) <= 0.0
        or 2.0 * case_thickness >= min(width, thickness)
    ):
        raise ValueError(
            "case thickness does not leave a positive winding pack"
        )

    if (
        surface_manifest.get("schema") != SURFACE_MANIFEST_SCHEMA
        or surface_manifest.get("status") != "PASS"
        or surface_manifest.get("dagmc_sha256") != h5m_evidence["sha256"]
        or surface_manifest.get("all_envelopes_close") is not True
    ):
        raise ValueError("global magnet surface manifest is not accepted")
    _, surface_magnet = _unique_row(
        surface_manifest.get("magnets"),
        "magnet",
        magnet_id,
        "surface magnet inventory",
    )
    surface_ids = surface_magnet.get("dagmc_surface_ids")
    volume_id = surface_magnet.get("dagmc_volume_id")
    if (
        surface_magnet.get("closed") is not True
        or surface_magnet.get("coupling_interface") != GLOBAL_INTERFACE
        or not isinstance(surface_ids, list)
        or not surface_ids
        or len(surface_ids) != len(set(surface_ids))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in surface_ids
        )
        or _positive_integer(volume_id, "DAGMC volume ID") != volume_id
    ):
        raise ValueError("selected global magnet envelope is invalid")

    if (
        root_acceptance.get("schema") != ROOT_ACCEPTANCE_SCHEMA
        or root_acceptance.get("acceptance_authority")
        != "ROOT_GEOMETRY_GATE_ACCEPTED"
        or root_acceptance.get("dagmc_sha256") != h5m_evidence["sha256"]
        or not _valid_sha256(
            root_acceptance.get("canonical_geometry_fingerprint")
        )
    ):
        raise ValueError("global geometry lacks root acceptance")

    components = case["components"]
    if not isinstance(components, list) or len(components) != 2:
        raise ValueError(
            "exactly casing and winding-pack components are required"
        )
    normalized_components = []
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("local component must be an object")
        _require_keys(
            component, {"component_id", "role", "material_id"}, "component"
        )
        normalized_components.append(
            {
                "component_id": str(component["component_id"]).strip(),
                "role": str(component["role"]),
                "material_id": str(component["material_id"]).strip(),
            }
        )
    roles = {row["role"] for row in normalized_components}
    component_ids = {row["component_id"] for row in normalized_components}
    material_ids = {row["material_id"] for row in normalized_components}
    if (
        roles != {"outer_casing", "winding_pack"}
        or len(component_ids) != 2
        or "" in component_ids
        or len(material_ids) != 2
        or "" in material_ids
    ):
        raise ValueError(
            "local component identities/roles/materials are ambiguous"
        )

    if not isinstance(case["materials"], list):
        raise ValueError("materials must be a list")
    materials = [validate_material_record(row) for row in case["materials"]]
    by_material = {row["material_id"]: row for row in materials}
    if len(by_material) != len(materials) or set(by_material) != material_ids:
        raise ValueError(
            "material records do not exactly cover local components"
        )

    surface_source = case["surface_source"]
    if not isinstance(surface_source, Mapping):
        raise ValueError("surface_source must be an object")
    _require_keys(
        surface_source,
        {
            "bank",
            "history_receipt",
            "strict_surface_run_audit",
            "source_histories",
        },
        "surface_source",
    )
    bank_path, bank_evidence = _bound_path(
        base, surface_source["bank"], label="surface source bank"
    )
    history_binding, history_evidence = _bound_json(
        base, surface_source["history_receipt"], label="history receipt"
    )
    strict_audit, strict_evidence = _bound_json(
        base,
        surface_source["strict_surface_run_audit"],
        label="strict surface-run audit",
    )
    histories = _positive_integer(
        surface_source["source_histories"], "source_histories"
    )
    _, verified_phase = read_openmc16_surface_sources(
        [bank_path],
        source_histories=histories,
        history_binding=history_binding,
        requested_surface_ids=surface_ids,
    )
    verified_history = verified_phase["history_binding"]
    strict_files = strict_audit.get("surface_source_files")
    strict_inventory = strict_audit.get("accepted_magnet_inventory", {})
    strict_components = strict_inventory.get("components", [])
    strict_component = [
        row
        for row in strict_components
        if isinstance(row, Mapping) and row.get("magnet_id") == magnet_id
    ]
    strict_model = strict_audit.get("model", {})
    strict_statepoint = strict_audit.get("statepoint", {})
    same_run = strict_audit.get("same_run_integrity", {})
    strict_root = strict_audit.get("root_acceptance_receipt", {})
    strict_dagmc = strict_audit.get("dagmc", {})
    strict_model_sha = strict_model.get("sha256")
    strict_statepoint_sha = strict_statepoint.get("sha256")
    accepted_inventory_sha = strict_inventory.get("sha256")
    verified_inventory_sha = strict_inventory.get("verified_inventory_sha256")
    verified_root_sha = strict_root.get("verified_acceptance_sha256")
    root_without_verification = dict(strict_root)
    root_without_verification.pop("verified_acceptance_sha256", None)
    inventory_without_verification = dict(strict_inventory)
    inventory_without_verification.pop("verified_inventory_sha256", None)
    strict_inventory_path = Path(str(strict_inventory.get("path", "")))
    if not strict_inventory_path.is_absolute():
        strict_inventory_path = base / strict_inventory_path
    strict_inventory_path = strict_inventory_path.resolve(strict=True)
    accepted_inventory = json.loads(
        strict_inventory_path.read_text(encoding="utf-8")
    )
    if not isinstance(accepted_inventory, Mapping):
        raise ValueError("accepted inventory artifact is invalid")
    artifact_components = _inventory_semantics(
        accepted_inventory.get("components"), "accepted inventory"
    )
    strict_component_semantics = _inventory_semantics(
        strict_components, "strict inventory"
    )
    if (
        strict_audit.get("schema") != STRICT_AUDIT_SCHEMA
        or strict_audit.get("classification") != "COMPLETE_CROSSING_BANK"
        or strict_audit.get("complete_classification_source")
        != "parsed_artifacts_only"
        or strict_dagmc.get("sha256") != h5m_evidence["sha256"]
        or strict_dagmc.get("canonical_geometry_fingerprints")
        != [root_acceptance["canonical_geometry_fingerprint"]]
        or strict_model.get("source_histories") != histories
        or strict_root.get("sha256") != root_evidence["sha256"]
        or strict_root.get("path") != root_evidence["path"]
        or strict_root.get("accepted_magnet_inventory_sha256")
        != root_acceptance["accepted_magnet_inventory_sha256"]
        or strict_root.get("canonical_geometry_fingerprint")
        != root_acceptance["canonical_geometry_fingerprint"]
        or strict_root.get("accepted_canonical_material_tags")
        != root_acceptance["accepted_canonical_material_tags"]
        or accepted_inventory_sha
        != root_acceptance["accepted_magnet_inventory_sha256"]
        or _sha256(strict_inventory_path) != accepted_inventory_sha
        or accepted_inventory.get("schema")
        != "parastell.accepted_magnet_component_inventory/v1.0.0"
        or accepted_inventory.get("geometry_gate_status") != "PASS"
        or accepted_inventory.get("dagmc_sha256") != h5m_evidence["sha256"]
        or accepted_inventory.get("canonical_geometry_fingerprint")
        != root_acceptance["canonical_geometry_fingerprint"]
        or sorted(
            _canonical_material_tag(value)
            for value in accepted_inventory.get("magnet_material_tags", [])
        )
        != root_acceptance["accepted_canonical_material_tags"]
        or artifact_components != strict_component_semantics
        or strict_inventory.get("schema")
        != "parastell.accepted_magnet_component_inventory/v1.0.0"
        or strict_inventory.get("geometry_gate_status") != "PASS"
        or strict_inventory.get("dagmc_sha256") != h5m_evidence["sha256"]
        or strict_inventory.get("accepted_sha256_from_root_receipt")
        != accepted_inventory_sha
        or strict_inventory.get("root_acceptance_receipt_sha256")
        != root_evidence["sha256"]
        or strict_inventory.get("verified_root_acceptance_sha256")
        != verified_root_sha
        or strict_inventory.get("all_material_group_volumes_selected")
        is not True
        or strict_inventory.get("canonical_geometry_fingerprint")
        != root_acceptance["canonical_geometry_fingerprint"]
        or sorted(
            _canonical_material_tag(value)
            for value in strict_inventory.get("magnet_material_tags", [])
        )
        != root_acceptance["accepted_canonical_material_tags"]
        or not all(
            _valid_sha256(value)
            for value in (
                strict_model_sha,
                strict_statepoint_sha,
                accepted_inventory_sha,
                verified_inventory_sha,
                verified_root_sha,
                strict_audit.get("localization_topology_binding", {}).get(
                    "manifest_sha256"
                ),
            )
        )
        or same_run.get("model_xml_sha256") != strict_model_sha
        or same_run.get("statepoint_sha256") != strict_statepoint_sha
        or same_run.get("accepted_magnet_inventory_sha256")
        != accepted_inventory_sha
        or same_run.get("verified_magnet_inventory_sha256")
        != verified_inventory_sha
        or same_run.get("root_acceptance_receipt_sha256")
        != root_evidence["sha256"]
        or same_run.get("verified_root_acceptance_sha256") != verified_root_sha
        or same_run.get("localization_topology_manifest_sha256")
        != strict_audit.get("localization_topology_binding", {}).get(
            "manifest_sha256"
        )
        or _canonical_sha256(root_without_verification) != verified_root_sha
        or _canonical_sha256(inventory_without_verification)
        != verified_inventory_sha
        or strict_audit.get("capacity", {}).get("capacity_proven_not_reached")
        is not True
        or same_run.get("passes") is not True
        or not isinstance(strict_files, list)
        or len(strict_files) != 1
        or strict_files[0].get("sha256") != bank_evidence["sha256"]
        or len(strict_component) != 1
        or strict_component[0].get("dagmc_volume_id") != volume_id
        or sorted(strict_component[0].get("surface_ids", []))
        != sorted(surface_ids)
    ):
        raise ValueError("surface source is not an accepted complete run")
    if (
        verified_history.get("kind") != "fixed_source_run"
        or verified_history.get("statepoint_sha256") != strict_statepoint_sha
        or verified_history.get("settings_payload_sha256") != strict_model_sha
        or verified_history.get("particles_per_batch")
        != strict_model.get("particles_per_batch")
        or verified_history.get("batches") != strict_model.get("batches")
        or verified_history.get("seed") != strict_model.get("seed")
        or strict_statepoint.get("particles_per_batch")
        != strict_model.get("particles_per_batch")
        or strict_statepoint.get("batches") != strict_model.get("batches")
        or strict_statepoint.get("seed") != strict_model.get("seed")
    ):
        raise ValueError("surface source history is not the strict OpenMC run")
    source_geometry = strict_component[0].get("source_geometry", {})
    if (
        source_geometry.get("kind") != "parametric_reference_manifest"
        or source_geometry.get("sha256") != source_evidence["sha256"]
    ):
        raise ValueError("accepted magnet inventory is not source-CAD bound")

    response_plan, response_evidence = _bound_json(
        base, case["response_plan"], label="response plan"
    )
    validate_response_plan(response_plan)
    if (
        response_plan.get("magnet_ids") != [magnet_id]
        or response_plan.get("case_id") != case["case_id"]
    ):
        raise ValueError("response plan must select exactly the local magnet")

    return {
        "schema": PLAN_SCHEMA,
        "status": "GEOMETRY_AND_COMPLETE_SOURCE_BOUND_NOT_EXECUTED",
        "case_id": str(case["case_id"]),
        "magnet_id": magnet_id,
        "geometry_request": {
            "filament_indices": [filament_index],
            "filament_coordinates_sha256": coordinates_sha256,
            "coil_input": coil_evidence,
            "outer_width_cm": width,
            "outer_thickness_cm": thickness,
            "case_thickness_cm": case_thickness,
            "inner_width_cm": width - 2.0 * case_thickness,
            "inner_thickness_cm": thickness - 2.0 * case_thickness,
            "global_coordinates_preserved": True,
            "arbitrary_transform": False,
        },
        "components": normalized_components,
        "materials": [by_material[key] for key in sorted(by_material)],
        "global_envelope": {
            "coupling_interface": GLOBAL_INTERFACE,
            "local_interface": LOCAL_INTERFACE,
            "dagmc_volume_id": volume_id,
            "dagmc_surface_ids": sorted(surface_ids),
            "closed": True,
        },
        "surface_source": {
            "bank": bank_evidence,
            "history_receipt": history_evidence,
            "strict_surface_run_audit": strict_evidence,
            "source_histories": histories,
            "verified_phase_space_schema": PHASE_SPACE_SCHEMA,
            "record_count": verified_phase["record_count"],
            "canonical_records_reconditioned": False,
            "incoming_only_replay_pending": True,
        },
        "response_plan": response_plan,
        "bindings": {
            "case": {
                "path": str(case_path),
                "bytes": case_path.stat().st_size,
                "sha256": _sha256(case_path),
            },
            "source_cad_manifest": source_evidence,
            "magnet_surface_manifest": surface_evidence,
            "global_dagmc_h5m": h5m_evidence,
            "global_coil_input": coil_evidence,
            "root_acceptance_receipt": root_evidence,
            "response_plan": response_evidence,
        },
        "claims": {
            "casing_and_homogenized_winding_only": True,
            "tape_resolved": False,
            "transport_executed": False,
            "statistics_qualified": False,
            "production_authorized": False,
        },
    }
