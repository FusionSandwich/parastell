"""Neutral, hash-bound metadata needed by downstream activation adapters.

This module deliberately contains no depletion schedule, inventory solver, or
OpenMC execution.  It records the identity and normalization needed for those
operations while keeping DAGMC volume IDs distinct from the actual OpenMC cell
IDs selected by the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .magnet_local_mesh import LocalMeshDefinition

SCHEMA = "parastell.activation_ready_metadata/v1.0.0"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMPONENT_ROLES = ("winding_pack", "casing")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("activation metadata must be finite JSON") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _artifact_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    return _canonical_sha256(payload)


def _require_keys(
    value: Mapping[str, Any], required: set[str], name: str
) -> None:
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise ValueError(
            f"{name} keys are invalid; missing={missing}, extra={extra}"
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} is forbidden")

    return json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )


def isotopic_composition_sha256(material: Mapping[str, Any]) -> str:
    """Hash only the basis and normalized isotope fractions of a material."""

    record = _mapping(material, "material")
    basis = record.get("composition_basis")
    if not isinstance(basis, str) or not basis:
        raise ValueError("material composition_basis must be nonempty")
    nuclides = _mapping(record.get("nuclides"), "material nuclides")
    if not nuclides:
        raise ValueError("material nuclides cannot be empty")
    normalized: dict[str, float] = {}
    for nuclide, fraction in nuclides.items():
        if not isinstance(nuclide, str) or not nuclide:
            raise ValueError("material nuclide names must be nonempty")
        if isinstance(fraction, bool):
            raise TypeError("material nuclide fractions must be numeric")
        value = float(fraction)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("material nuclide fractions must be finite")
        normalized[nuclide] = value
    if not math.isclose(
        sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise ValueError("material nuclide fractions must sum to one")
    return _canonical_sha256(
        {"composition_basis": basis, "nuclides": normalized}
    )


def _material_record(
    *,
    material_tag: str,
    material_manifest: Mapping[str, Any],
    openmc_material_ids_by_tag: Mapping[str, int],
) -> dict[str, Any]:
    tags = _mapping(
        material_manifest.get("material_tags"), "material manifest tags"
    )
    materials = _mapping(
        material_manifest.get("materials"), "material manifest materials"
    )
    if material_tag not in tags:
        raise ValueError(f"unknown component material tag {material_tag!r}")
    material_name = tags[material_tag]
    if not isinstance(material_name, str) or material_name not in materials:
        raise ValueError(
            f"material tag {material_tag!r} has no resolved material"
        )
    if material_tag not in openmc_material_ids_by_tag:
        raise ValueError(
            f"actual OpenMC material ID is missing for {material_tag!r}"
        )
    material = _mapping(materials[material_name], str(material_name))
    density = _positive_float(
        material.get("density_g_cm3"), f"{material_name} density"
    )
    temperature = _positive_float(
        material.get("temperature_K"), f"{material_name} temperature"
    )
    nuclides = _json_copy(
        _mapping(material.get("nuclides"), f"{material_name} nuclides")
    )
    composition = {
        "composition_basis": material.get("composition_basis"),
        "nuclides": nuclides,
    }
    digest = isotopic_composition_sha256(composition)
    return {
        "tag": material_tag,
        "name": material_name,
        "openmc_material_id": _positive_int(
            openmc_material_ids_by_tag[material_tag],
            f"OpenMC material ID for {material_tag}",
        ),
        "density_g_cm3": density,
        "temperature_K": temperature,
        "composition_basis": composition["composition_basis"],
        "nuclides": nuclides,
        "isotopic_composition_sha256": digest,
    }


def _component(
    *,
    value: Mapping[str, Any],
    role: str,
    material_manifest: Mapping[str, Any],
    openmc_cell_ids_by_dagmc_volume: Mapping[int, int],
    openmc_material_ids_by_tag: Mapping[str, int],
) -> dict[str, Any]:
    dagmc_volume_id = _positive_int(
        value.get("volume_id"), f"{role} DAGMC volume ID"
    )
    if dagmc_volume_id not in openmc_cell_ids_by_dagmc_volume:
        raise ValueError(
            "actual OpenMC cell ID is missing for DAGMC volume "
            f"{dagmc_volume_id}"
        )
    openmc_cell_id = _positive_int(
        openmc_cell_ids_by_dagmc_volume[dagmc_volume_id],
        f"OpenMC cell ID for DAGMC volume {dagmc_volume_id}",
    )
    material_tag = value.get("material")
    if not isinstance(material_tag, str) or not material_tag:
        raise ValueError(f"{role} material tag must be nonempty")
    material = _material_record(
        material_tag=material_tag,
        material_manifest=material_manifest,
        openmc_material_ids_by_tag=openmc_material_ids_by_tag,
    )
    volume = _positive_float(value.get("volume_cm3"), f"{role} volume")
    surfaces = value.get("surface_ids")
    if (
        not isinstance(surfaces, Sequence)
        or isinstance(surfaces, (str, bytes))
        or not surfaces
    ):
        raise ValueError(f"{role} surface IDs must be a nonempty sequence")
    surface_ids = [
        _positive_int(item, f"{role} surface ID") for item in surfaces
    ]
    if len(set(surface_ids)) != len(surface_ids):
        raise ValueError(f"{role} surface IDs must be unique")
    magnet_id = value.get("magnet_id")
    component_id = value.get("component_id")
    if not isinstance(magnet_id, str) or not magnet_id:
        raise ValueError(f"{role} magnet ID must be nonempty")
    if not isinstance(component_id, str) or not component_id:
        raise ValueError(f"{role} component ID must be nonempty")
    return {
        "component_id": component_id,
        "magnet_id": magnet_id,
        "component_role": role,
        "dagmc_volume_id": dagmc_volume_id,
        "openmc_cell_id": openmc_cell_id,
        "surface_ids": sorted(surface_ids),
        "material": material,
        "volume_cm3": volume,
        "mass_g": volume * material["density_g_cm3"],
        "activation_readiness": {
            "direct_r2s": {
                "metadata_ready": True,
                "execution_ready": False,
                "domain_kind": "openmc_cell",
                "status": (
                    "METADATA_READY_REQUIRES_DOWNSTREAM_CHAIN_SCHEDULE_"
                    "AND_TRANSPORT_MODEL"
                ),
            },
            "post_transport": {
                "metadata_ready": True,
                "execution_ready": False,
                "domain_kind": "material_homogeneous_component",
                "status": (
                    "METADATA_READY_REQUIRES_BOUND_SCALAR_FLUX_MICROXS_"
                    "CHAIN_AND_SCHEDULE"
                ),
            },
        },
    }


def _mesh_definition(
    value: Mapping[str, Any], magnet_id: str
) -> dict[str, Any]:
    fields = LocalMeshDefinition.__dataclass_fields__
    try:
        definition = LocalMeshDefinition(
            **{name: value[name] for name in fields if name in value}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid local mesh for {magnet_id!r}") from exc
    if definition.magnet_id != magnet_id:
        raise ValueError("local mesh key and magnet_id disagree")
    expected_count = definition.bin_count
    if int(value.get("bin_count", -1)) != expected_count:
        raise ValueError(f"local mesh {magnet_id!r} bin_count is inconsistent")
    supplied_volume = _positive_float(
        value.get("bin_volume_cm3"), f"{magnet_id} mesh bin volume"
    )
    if not math.isclose(
        supplied_volume,
        definition.bin_volume_cm3,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"local mesh {magnet_id!r} bin volume is inconsistent"
        )
    supplied_widths = value.get("actual_bin_widths_cm")
    if not isinstance(supplied_widths, Sequence) or len(supplied_widths) != 3:
        raise ValueError(f"local mesh {magnet_id!r} widths are invalid")
    for supplied, expected in zip(supplied_widths, definition.widths_cm):
        if not math.isclose(
            float(supplied), float(expected), rel_tol=1.0e-12, abs_tol=0.0
        ):
            raise ValueError(
                f"local mesh {magnet_id!r} widths are inconsistent"
            )
    return definition.to_dict()


def _material_fractions(
    value: Any,
    *,
    bin_count: int,
    known_tags: set[str],
    mesh_id: str,
) -> list[dict[str, float]] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != bin_count
    ):
        raise ValueError(
            f"material fractions for {mesh_id!r} must have one row per bin"
        )
    result = []
    for index, raw in enumerate(value):
        row = _mapping(raw, f"{mesh_id} material fractions bin {index}")
        if not row or set(row) - known_tags:
            raise ValueError(
                f"{mesh_id!r} bin {index} has unknown or empty material fractions"
            )
        normalized = {}
        for tag, fraction in row.items():
            if isinstance(fraction, bool):
                raise TypeError("mesh material fractions must be numeric")
            number = float(fraction)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError("mesh material fractions must be nonnegative")
            if number > 0.0:
                normalized[str(tag)] = number
        if not normalized or not math.isclose(
            sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError(
                f"{mesh_id!r} bin {index} material fractions must sum to one"
            )
        result.append(dict(sorted(normalized.items())))
    return result


def _mesh(
    *,
    magnet_id: str,
    value: Mapping[str, Any],
    fractions: Any,
    material_tags: set[str],
    cell_filter_applied: bool,
    nonoverlap_qualified: bool,
    intersection: Mapping[str, Any] | None,
    intersection_artifact_sha256: str | None,
) -> dict[str, Any]:
    definition = _mesh_definition(value, magnet_id)
    bin_count = int(definition["bin_count"])
    rows = _material_fractions(
        fractions,
        bin_count=bin_count,
        known_tags=material_tags,
        mesh_id=magnet_id,
    )
    fractions_available = rows is not None
    # A cell-filtered mesh still needs exact material/intersection fractions;
    # the filter alone cannot prove that a full geometric voxel is homogeneous.
    mixed = any(len(row) > 1 for row in rows) if rows is not None else True
    intersection_metadata = None
    if intersection is not None:
        if intersection_artifact_sha256 is None:
            raise ValueError(
                "mesh intersection volumes require their artifact SHA-256"
            )
        intersection_artifact_sha256 = _sha256(
            intersection_artifact_sha256,
            "material-intersection volume artifact SHA-256",
        )
        role = intersection.get("component_role")
        material_tag = intersection.get("material_tag")
        if role != "winding_pack" or material_tag not in material_tags:
            raise ValueError(
                f"{magnet_id!r} intersection component identity is invalid"
            )
        volumes = [
            float(value)
            for value in intersection.get("intersection_volume_cm3", [])
        ]
        deviations = [
            float(value)
            for value in intersection.get("standard_deviation_cm3", [])
        ]
        statuses = list(intersection.get("bin_status", []))
        if (
            len(volumes) != bin_count
            or len(deviations) != bin_count
            or len(statuses) != bin_count
            or any(
                not math.isfinite(value) or value < 0.0 for value in volumes
            )
            or any(
                not math.isfinite(value) or value < 0.0 for value in deviations
            )
            or not any(value > 0.0 for value in volumes)
            or intersection.get("status") != "QUALIFIED"
        ):
            raise ValueError(
                f"{magnet_id!r} material-intersection volumes are not qualified"
            )
        intersection_metadata = {
            "artifact_sha256": intersection_artifact_sha256,
            "component_role": role,
            "dagmc_volume_id": _positive_int(
                intersection.get("dagmc_volume_id"),
                "intersection DAGMC volume ID",
            ),
            "openmc_cell_id": _positive_int(
                intersection.get("openmc_cell_id"),
                "intersection OpenMC cell ID",
            ),
            "material_tag": material_tag,
            "openmc_material_id": _positive_int(
                intersection.get("openmc_material_id"),
                "intersection OpenMC material ID",
            ),
            "per_bin_volume_cm3": volumes,
            "per_bin_standard_deviation_cm3": deviations,
            "bin_status": statuses,
            "positive_intersection_bin_count": sum(
                value > 0.0 for value in volumes
            ),
            "total_intersection_volume_cm3": float(sum(volumes)),
            "qualified_volume_fraction": float(
                intersection.get("qualified_volume_fraction", 1.0)
            ),
            "excluded_insufficient_bin_count": int(
                intersection.get("excluded_insufficient_bin_count", 0)
            ),
            "volume_basis": "qualified_component_intersection_volume",
            "status": "QUALIFIED",
        }
    elif intersection_artifact_sha256 is not None:
        raise ValueError(
            "material-intersection artifact SHA-256 was supplied without volumes"
        )
    post_ready = fractions_available or intersection_metadata is not None
    direct_ready = nonoverlap_qualified
    return {
        "mesh_id": f"{magnet_id}:local_mesh",
        "magnet_id": magnet_id,
        "definition": definition,
        "bin_count": bin_count,
        "geometric_bin_volumes_cm3": [float(definition["bin_volume_cm3"])]
        * bin_count,
        "volume_basis": "full_geometric_voxel_volume",
        "material_resolution": {
            "cell_filter_applied": cell_filter_applied,
            "materially_mixed_bins_possible": mixed,
            "fractions_available": fractions_available,
            "per_bin_material_fractions": rows,
            "status": (
                "MATERIAL_FRACTIONS_EXPLICIT"
                if fractions_available
                else "MATERIAL_FRACTIONS_UNAVAILABLE"
            ),
        },
        "component_intersection_volumes": intersection_metadata,
        "activation_readiness": {
            "direct_r2s": {
                "metadata_ready": direct_ready,
                "execution_ready": False,
                "nonoverlap_qualified": nonoverlap_qualified,
                "domain_kind": "openmc_mesh",
                "status": (
                    "METADATA_READY_REQUIRES_DOWNSTREAM_CHAIN_SCHEDULE_"
                    "AND_TRANSPORT_MODEL"
                    if direct_ready
                    else "NOT_READY_NONOVERLAP_QUALIFICATION_REQUIRED"
                ),
            },
            "post_transport": {
                "metadata_ready": post_ready,
                "execution_ready": False,
                "post_transport_depletable": post_ready,
                "domain_kind": "material_resolved_mesh_bin",
                "status": (
                    "METADATA_READY_REQUIRES_BOUND_SCALAR_FLUX_MICROXS_"
                    "CHAIN_AND_SCHEDULE"
                    if post_ready
                    else "NOT_DEPLETABLE_COMPONENT_INTERSECTION_VOLUMES_REQUIRED"
                ),
            },
        },
    }


def build_activation_ready_metadata(
    *,
    associations: Mapping[str, Any],
    material_manifest: Mapping[str, Any],
    nuclear_data_manifest: Mapping[str, Any],
    local_mesh_manifest: Mapping[str, Any],
    openmc_cell_ids_by_dagmc_volume: Mapping[int, int],
    openmc_material_ids_by_tag: Mapping[str, int],
    physical_source_rate_per_s: float,
    mesh_material_fractions_by_magnet: (
        Mapping[str, Sequence[Mapping[str, float]]] | None
    ) = None,
    mesh_material_intersection_volumes_by_magnet: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
    mesh_material_intersection_artifact_sha256: str | None = None,
    flux_quantity: str = "volume_scalar_flux",
    flux_estimator: str = "track_length",
) -> dict[str, Any]:
    """Build activation inputs without performing activation or depletion.

    ``openmc_cell_ids_by_dagmc_volume`` is mandatory even when an OpenMC DAGMC
    build happens to use equal numeric IDs.  This prevents downstream code from
    silently treating a DAGMC volume ID as an OpenMC cell identity.
    """

    associations = _mapping(associations, "associations")
    material_manifest = _mapping(material_manifest, "material manifest")
    nuclear_data_manifest = _mapping(
        nuclear_data_manifest, "nuclear-data manifest"
    )
    local_mesh_manifest = _mapping(local_mesh_manifest, "local mesh manifest")
    if flux_quantity != "volume_scalar_flux":
        raise ValueError(
            "activation requires volume scalar flux; surface current is not "
            "an activation or SPECTRA-PKA flux"
        )
    if flux_estimator != "track_length":
        raise ValueError(
            "activation scalar flux must use a track-length estimator"
        )
    source_rate = _positive_float(
        physical_source_rate_per_s, "physical source rate"
    )
    geometry_fingerprint = _sha256(
        associations.get("canonical_geometry_fingerprint"),
        "canonical geometry fingerprint",
    )
    if (
        local_mesh_manifest.get("canonical_geometry_fingerprint")
        != geometry_fingerprint
    ):
        raise ValueError(
            "local meshes and associations use different geometry fingerprints"
        )
    inventory = _mapping(associations.get("inventory"), "magnet inventory")
    dagmc_sha = inventory.get("dagmc_geometry_sha256") or associations.get(
        "dagmc_sha256"
    )
    dagmc_sha = _sha256(dagmc_sha, "DAGMC geometry SHA-256")
    pairs = inventory.get("magnet_pairs")
    if (
        not isinstance(pairs, Sequence)
        or isinstance(pairs, (str, bytes))
        or not pairs
    ):
        raise ValueError("magnet inventory must contain at least one pair")
    components = []
    magnet_ids = []
    for raw_pair in pairs:
        pair = _mapping(raw_pair, "magnet pair")
        pair_magnet_id = pair.get("magnet_id")
        if not isinstance(pair_magnet_id, str) or not pair_magnet_id:
            raise ValueError("magnet pair ID must be nonempty")
        magnet_ids.append(pair_magnet_id)
        for source_name, role in (
            ("winding_pack", "winding_pack"),
            ("casing", "casing"),
        ):
            source = pair.get(source_name)
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"magnet {pair_magnet_id!r} lacks required {role} metadata"
                )
            item = _component(
                value=source,
                role=role,
                material_manifest=material_manifest,
                openmc_cell_ids_by_dagmc_volume=(
                    openmc_cell_ids_by_dagmc_volume
                ),
                openmc_material_ids_by_tag=openmc_material_ids_by_tag,
            )
            if item["magnet_id"] != pair_magnet_id:
                raise ValueError("component and pair magnet IDs disagree")
            components.append(item)
    if len(set(magnet_ids)) != len(magnet_ids):
        raise ValueError("magnet pair IDs must be unique")
    if len({item["dagmc_volume_id"] for item in components}) != len(
        components
    ):
        raise ValueError("DAGMC component volume IDs must be unique")
    if len({item["openmc_cell_id"] for item in components}) != len(components):
        raise ValueError("OpenMC component cell IDs must be unique")

    meshes = _mapping(local_mesh_manifest.get("meshes"), "local meshes")
    if set(meshes) != set(magnet_ids):
        raise ValueError(
            "local meshes must identify every selected magnet exactly once"
        )
    cell_filter_applied = local_mesh_manifest.get("cell_filter_applied")
    if type(cell_filter_applied) is not bool:
        raise ValueError("local mesh cell_filter_applied must be boolean")
    nonoverlap = local_mesh_manifest.get("nonoverlap_qualified", False)
    if type(nonoverlap) is not bool:
        raise ValueError("local mesh nonoverlap_qualified must be boolean")
    fractions_by_magnet = dict(mesh_material_fractions_by_magnet or {})
    if set(fractions_by_magnet) - set(magnet_ids):
        raise ValueError("material fractions reference unknown local meshes")
    intersections_by_magnet = dict(
        mesh_material_intersection_volumes_by_magnet or {}
    )
    if set(intersections_by_magnet) - set(magnet_ids):
        raise ValueError(
            "material-intersection volumes reference unknown local meshes"
        )
    if (
        intersections_by_magnet
        and mesh_material_intersection_artifact_sha256 is None
    ):
        raise ValueError(
            "material-intersection volumes require their artifact SHA-256"
        )
    if (
        not intersections_by_magnet
        and mesh_material_intersection_artifact_sha256 is not None
    ):
        raise ValueError(
            "material-intersection artifact SHA-256 was supplied without volumes"
        )
    material_tags = set(
        _mapping(
            material_manifest.get("material_tags"), "material manifest tags"
        )
    )
    local_meshes = [
        _mesh(
            magnet_id=magnet_id,
            value=_mapping(meshes[magnet_id], f"local mesh {magnet_id}"),
            fractions=fractions_by_magnet.get(magnet_id),
            material_tags=material_tags,
            cell_filter_applied=cell_filter_applied,
            nonoverlap_qualified=nonoverlap,
            intersection=intersections_by_magnet.get(magnet_id),
            intersection_artifact_sha256=(
                mesh_material_intersection_artifact_sha256
                if magnet_id in intersections_by_magnet
                else None
            ),
        )
        for magnet_id in sorted(magnet_ids)
    ]

    material_manifest_sha = material_manifest.get("resolved_manifest_sha256")
    material_manifest_sha = _sha256(
        material_manifest_sha, "resolved material manifest SHA-256"
    )
    nuclear_manifest = _json_copy(nuclear_data_manifest)
    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "geometry": {
            "canonical_geometry_fingerprint": geometry_fingerprint,
            "dagmc_geometry_sha256": dagmc_sha,
            "length_units": "cm",
            "volume_units": "cm3",
            "mass_units": "g",
        },
        "source_and_flux_normalization": {
            "quantity": "volume_scalar_flux",
            "estimator": "track_length",
            "physical_source_rate_per_s": source_rate,
            "track_length_tally_units": "particle-cm/source",
            "scalar_flux_per_source_units": "particle/source/cm2",
            "physical_scalar_flux_units": "particle/cm2/s",
            "conversion": (
                "track_length_mean_cm_per_source / volume_cm3 * "
                "physical_source_rate_per_s"
            ),
            "surface_current_is_scalar_flux": False,
        },
        "material_manifest": {
            "schema": material_manifest.get("schema"),
            "resolved_manifest_sha256": material_manifest_sha,
        },
        "nuclear_data": {
            "canonical_manifest_sha256": _canonical_sha256(nuclear_manifest),
            "manifest": nuclear_manifest,
        },
        "components": sorted(
            components,
            key=lambda item: (item["magnet_id"], item["component_role"]),
        ),
        "local_meshes": local_meshes,
        "activation_readiness": {
            "direct_r2s_cell_metadata_ready": True,
            "direct_r2s_mesh_metadata_ready": all(
                item["activation_readiness"]["direct_r2s"]["metadata_ready"]
                for item in local_meshes
            ),
            "post_transport_component_metadata_ready": True,
            "post_transport_mesh_metadata_ready": all(
                item["activation_readiness"]["post_transport"][
                    "metadata_ready"
                ]
                for item in local_meshes
            ),
            "activation_schedules_present": False,
            "depletion_results_present": False,
            "execution_owner": "downstream_consumer",
        },
    }
    payload["artifact_sha256"] = _artifact_hash(payload)
    validate_activation_ready_metadata(payload)
    return payload


def _validate_material(value: Mapping[str, Any], name: str) -> None:
    _require_keys(
        value,
        {
            "tag",
            "name",
            "openmc_material_id",
            "density_g_cm3",
            "temperature_K",
            "composition_basis",
            "nuclides",
            "isotopic_composition_sha256",
        },
        name,
    )
    if not isinstance(value["tag"], str) or not value["tag"]:
        raise ValueError(f"{name} tag is invalid")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ValueError(f"{name} name is invalid")
    _positive_int(value["openmc_material_id"], f"{name} OpenMC material ID")
    _positive_float(value["density_g_cm3"], f"{name} density")
    _positive_float(value["temperature_K"], f"{name} temperature")
    expected = isotopic_composition_sha256(value)
    if value["isotopic_composition_sha256"] != expected:
        raise ValueError(f"{name} isotopic composition hash mismatch")


def _validate_readiness(value: Mapping[str, Any], *, mesh: bool) -> None:
    required = {"direct_r2s", "post_transport"}
    _require_keys(value, required, "activation readiness")
    direct = _mapping(value["direct_r2s"], "direct R2S readiness")
    post = _mapping(value["post_transport"], "post-transport readiness")
    direct_keys = {
        "metadata_ready",
        "execution_ready",
        "domain_kind",
        "status",
    }
    post_keys = set(direct_keys)
    if mesh:
        direct_keys.add("nonoverlap_qualified")
        post_keys.add("post_transport_depletable")
    _require_keys(direct, direct_keys, "direct R2S readiness")
    _require_keys(post, post_keys, "post-transport readiness")
    for item, label in ((direct, "direct R2S"), (post, "post transport")):
        for key in ("metadata_ready", "execution_ready"):
            if type(item[key]) is not bool:
                raise ValueError(f"{label} {key} must be boolean")
        if item["execution_ready"]:
            raise ValueError(
                "ParaStell neutral metadata cannot claim activation execution readiness"
            )
        if not isinstance(item["domain_kind"], str) or not item["domain_kind"]:
            raise ValueError(f"{label} domain kind is invalid")
        if not isinstance(item["status"], str) or not item["status"]:
            raise ValueError(f"{label} status is invalid")
    if mesh:
        if type(direct["nonoverlap_qualified"]) is not bool:
            raise ValueError("mesh nonoverlap qualification must be boolean")
        if direct["metadata_ready"] != direct["nonoverlap_qualified"]:
            raise ValueError("mesh direct R2S readiness is inconsistent")
        if type(post["post_transport_depletable"]) is not bool:
            raise ValueError("mesh depletion readiness must be boolean")
        if post["metadata_ready"] != post["post_transport_depletable"]:
            raise ValueError("mesh post-transport readiness is inconsistent")


def _validate_component(value: Mapping[str, Any]) -> None:
    _require_keys(
        value,
        {
            "component_id",
            "magnet_id",
            "component_role",
            "dagmc_volume_id",
            "openmc_cell_id",
            "surface_ids",
            "material",
            "volume_cm3",
            "mass_g",
            "activation_readiness",
        },
        "component",
    )
    for key in ("component_id", "magnet_id"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"component {key} is invalid")
    if value["component_role"] not in COMPONENT_ROLES:
        raise ValueError("component role is invalid")
    _positive_int(value["dagmc_volume_id"], "DAGMC volume ID")
    _positive_int(value["openmc_cell_id"], "OpenMC cell ID")
    surfaces = value["surface_ids"]
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("component surface IDs are invalid")
    for surface in surfaces:
        _positive_int(surface, "component surface ID")
    if surfaces != sorted(set(surfaces)):
        raise ValueError("component surface IDs are not unique and sorted")
    material = _mapping(value["material"], "component material")
    _validate_material(material, "component material")
    volume = _positive_float(value["volume_cm3"], "component volume")
    mass = _positive_float(value["mass_g"], "component mass")
    expected_mass = volume * float(material["density_g_cm3"])
    if not math.isclose(mass, expected_mass, rel_tol=1.0e-12, abs_tol=0.0):
        raise ValueError(
            "component mass is inconsistent with density and volume"
        )
    _validate_readiness(
        _mapping(value["activation_readiness"], "component readiness"),
        mesh=False,
    )


def _validate_definition(value: Mapping[str, Any], magnet_id: str) -> None:
    expected_keys = {
        "magnet_id",
        "lower_left_local_cm",
        "upper_right_local_cm",
        "dimension",
        "rotation_local_to_global",
        "translation_global_cm",
        "requested_resolution_cm",
        "actual_bin_widths_cm",
        "bin_count",
        "bin_volume_cm3",
        "frame_type",
    }
    _require_keys(value, expected_keys, "local mesh definition")
    normalized = _mesh_definition(value, magnet_id)
    if value != normalized:
        raise ValueError("local mesh definition is not canonically normalized")


def _validate_mesh(value: Mapping[str, Any], material_tags: set[str]) -> None:
    base_keys = {
        "mesh_id",
        "magnet_id",
        "definition",
        "bin_count",
        "geometric_bin_volumes_cm3",
        "volume_basis",
        "material_resolution",
        "activation_readiness",
    }
    if set(value) not in (
        base_keys,
        base_keys | {"component_intersection_volumes"},
    ):
        raise ValueError("local mesh keys are invalid")
    magnet_id = value["magnet_id"]
    if not isinstance(magnet_id, str) or not magnet_id:
        raise ValueError("local mesh magnet ID is invalid")
    if value["mesh_id"] != f"{magnet_id}:local_mesh":
        raise ValueError("local mesh ID is invalid")
    definition = _mapping(value["definition"], "local mesh definition")
    _validate_definition(definition, magnet_id)
    count = _positive_int(value["bin_count"], "local mesh bin count")
    if count != definition["bin_count"]:
        raise ValueError("local mesh bin counts disagree")
    volumes = value["geometric_bin_volumes_cm3"]
    if not isinstance(volumes, list) or len(volumes) != count:
        raise ValueError("local mesh geometric bin volumes are incomplete")
    for volume in volumes:
        if not math.isclose(
            _positive_float(volume, "geometric bin volume"),
            float(definition["bin_volume_cm3"]),
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError("local mesh geometric bin volume is inconsistent")
    if value["volume_basis"] != "full_geometric_voxel_volume":
        raise ValueError("unsupported local mesh volume basis")
    resolution = _mapping(value["material_resolution"], "material resolution")
    _require_keys(
        resolution,
        {
            "cell_filter_applied",
            "materially_mixed_bins_possible",
            "fractions_available",
            "per_bin_material_fractions",
            "status",
        },
        "material resolution",
    )
    for key in (
        "cell_filter_applied",
        "materially_mixed_bins_possible",
        "fractions_available",
    ):
        if type(resolution[key]) is not bool:
            raise ValueError(f"material resolution {key} must be boolean")
    rows = _material_fractions(
        resolution["per_bin_material_fractions"],
        bin_count=count,
        known_tags=material_tags,
        mesh_id=magnet_id,
    )
    if resolution["fractions_available"] != (rows is not None):
        raise ValueError("mesh material fraction availability is inconsistent")
    expected_status = (
        "MATERIAL_FRACTIONS_EXPLICIT"
        if rows is not None
        else "MATERIAL_FRACTIONS_UNAVAILABLE"
    )
    if resolution["status"] != expected_status:
        raise ValueError("mesh material resolution status is inconsistent")
    intersection = value.get("component_intersection_volumes")
    intersection_ready = False
    if intersection is not None:
        intersection = _mapping(intersection, "component-intersection volumes")
        _require_keys(
            intersection,
            {
                "artifact_sha256",
                "component_role",
                "dagmc_volume_id",
                "openmc_cell_id",
                "material_tag",
                "openmc_material_id",
                "per_bin_volume_cm3",
                "per_bin_standard_deviation_cm3",
                "bin_status",
                "positive_intersection_bin_count",
                "total_intersection_volume_cm3",
                "qualified_volume_fraction",
                "excluded_insufficient_bin_count",
                "volume_basis",
                "status",
            },
            "component-intersection volumes",
        )
        _sha256(
            intersection["artifact_sha256"],
            "material-intersection volume artifact SHA-256",
        )
        if (
            intersection["component_role"] != "winding_pack"
            or intersection["material_tag"] not in material_tags
            or intersection["volume_basis"]
            != "qualified_component_intersection_volume"
            or intersection["status"] != "QUALIFIED"
        ):
            raise ValueError(
                "component-intersection volume identity is invalid"
            )
        for key in (
            "dagmc_volume_id",
            "openmc_cell_id",
            "openmc_material_id",
        ):
            _positive_int(intersection[key], key)
        intersection_volumes = intersection["per_bin_volume_cm3"]
        intersection_deviations = intersection[
            "per_bin_standard_deviation_cm3"
        ]
        intersection_statuses = intersection["bin_status"]
        if not all(
            isinstance(values, list) and len(values) == count
            for values in (
                intersection_volumes,
                intersection_deviations,
                intersection_statuses,
            )
        ):
            raise ValueError(
                "component-intersection bin metadata is incomplete"
            )
        normalized_volumes = []
        normalized_deviations = []
        for raw_volume, raw_deviation, raw_geometric in zip(
            intersection_volumes,
            intersection_deviations,
            volumes,
        ):
            volume = float(raw_volume)
            deviation = float(raw_deviation)
            if (
                not math.isfinite(volume)
                or not math.isfinite(deviation)
                or volume < 0.0
                or deviation < 0.0
                or volume > float(raw_geometric) * (1.0 + 1.0e-10)
            ):
                raise ValueError("component-intersection volume is invalid")
            normalized_volumes.append(volume)
            normalized_deviations.append(deviation)
        if not any(value > 0.0 for value in normalized_volumes):
            raise ValueError("component-intersection volumes are all empty")
        allowed_statuses = {
            "QUALIFIED",
            "EMPTY_INTERSECTION",
            "INSUFFICIENT_GEOMETRY_STATISTICS",
        }
        if any(
            value not in allowed_statuses for value in intersection_statuses
        ):
            raise ValueError("component-intersection bin status is invalid")
        positive_count = sum(value > 0.0 for value in normalized_volumes)
        if intersection["positive_intersection_bin_count"] != positive_count:
            raise ValueError(
                "component-intersection positive bin count disagrees"
            )
        if not math.isclose(
            float(intersection["total_intersection_volume_cm3"]),
            sum(normalized_volumes),
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError("component-intersection total volume disagrees")
        total_volume = sum(normalized_volumes)
        qualified_volume = sum(
            volume
            for volume, status in zip(
                normalized_volumes, intersection_statuses
            )
            if status == "QUALIFIED"
        )
        qualified_fraction = qualified_volume / total_volume
        if not math.isclose(
            float(intersection["qualified_volume_fraction"]),
            qualified_fraction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "component-intersection qualified volume fraction disagrees"
            )
        excluded_count = sum(
            status == "INSUFFICIENT_GEOMETRY_STATISTICS"
            for status in intersection_statuses
        )
        if intersection["excluded_insufficient_bin_count"] != excluded_count:
            raise ValueError(
                "component-intersection excluded bin count disagrees"
            )
        intersection_ready = True
    readiness = _mapping(value["activation_readiness"], "mesh readiness")
    _validate_readiness(readiness, mesh=True)
    post_ready = readiness["post_transport"]["metadata_ready"]
    if post_ready != (resolution["fractions_available"] or intersection_ready):
        raise ValueError(
            "mesh without material fractions or component-intersection volumes "
            "cannot be post-transport-depletable"
        )


def validate_activation_ready_metadata(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate a neutral activation metadata artifact."""

    value = _mapping(value, "activation metadata")
    _require_keys(
        value,
        {
            "schema",
            "created_utc",
            "geometry",
            "source_and_flux_normalization",
            "material_manifest",
            "nuclear_data",
            "components",
            "local_meshes",
            "activation_readiness",
            "artifact_sha256",
        },
        "activation metadata",
    )
    if value["schema"] != SCHEMA:
        raise ValueError("unsupported activation metadata schema")
    if not isinstance(value["created_utc"], str) or not value["created_utc"]:
        raise ValueError("activation metadata timestamp is invalid")
    geometry = _mapping(value["geometry"], "geometry")
    _require_keys(
        geometry,
        {
            "canonical_geometry_fingerprint",
            "dagmc_geometry_sha256",
            "length_units",
            "volume_units",
            "mass_units",
        },
        "geometry",
    )
    _sha256(
        geometry["canonical_geometry_fingerprint"],
        "canonical geometry fingerprint",
    )
    _sha256(geometry["dagmc_geometry_sha256"], "DAGMC geometry SHA-256")
    if (
        geometry["length_units"],
        geometry["volume_units"],
        geometry["mass_units"],
    ) != ("cm", "cm3", "g"):
        raise ValueError("geometry units are unsupported")
    normalization = _mapping(
        value["source_and_flux_normalization"], "flux normalization"
    )
    _require_keys(
        normalization,
        {
            "quantity",
            "estimator",
            "physical_source_rate_per_s",
            "track_length_tally_units",
            "scalar_flux_per_source_units",
            "physical_scalar_flux_units",
            "conversion",
            "surface_current_is_scalar_flux",
        },
        "flux normalization",
    )
    if (
        normalization["quantity"] != "volume_scalar_flux"
        or normalization["estimator"] != "track_length"
        or normalization["surface_current_is_scalar_flux"] is not False
    ):
        raise ValueError(
            "surface current cannot stand in for volume scalar flux"
        )
    _positive_float(
        normalization["physical_source_rate_per_s"], "physical source rate"
    )
    expected_units = (
        "particle-cm/source",
        "particle/source/cm2",
        "particle/cm2/s",
    )
    if (
        normalization["track_length_tally_units"],
        normalization["scalar_flux_per_source_units"],
        normalization["physical_scalar_flux_units"],
    ) != expected_units:
        raise ValueError("flux normalization units are unsupported")
    if normalization["conversion"] != (
        "track_length_mean_cm_per_source / volume_cm3 * "
        "physical_source_rate_per_s"
    ):
        raise ValueError("flux normalization conversion is invalid")
    material_manifest = _mapping(
        value["material_manifest"], "material manifest reference"
    )
    _require_keys(
        material_manifest,
        {"schema", "resolved_manifest_sha256"},
        "material manifest reference",
    )
    if not isinstance(material_manifest["schema"], str):
        raise ValueError("material manifest schema is invalid")
    _sha256(
        material_manifest["resolved_manifest_sha256"],
        "resolved material manifest SHA-256",
    )
    nuclear = _mapping(value["nuclear_data"], "nuclear data")
    _require_keys(
        nuclear,
        {"canonical_manifest_sha256", "manifest"},
        "nuclear data",
    )
    nuclear_manifest = _mapping(nuclear["manifest"], "nuclear-data manifest")
    if nuclear["canonical_manifest_sha256"] != _canonical_sha256(
        nuclear_manifest
    ):
        raise ValueError("nuclear-data manifest hash mismatch")
    components = value["components"]
    if not isinstance(components, list) or not components:
        raise ValueError("activation components cannot be empty")
    for item in components:
        _validate_component(_mapping(item, "component"))
    if len({item["component_id"] for item in components}) != len(components):
        raise ValueError("component IDs must be unique")
    if len({item["dagmc_volume_id"] for item in components}) != len(
        components
    ):
        raise ValueError("DAGMC volume IDs must be unique")
    if len({item["openmc_cell_id"] for item in components}) != len(components):
        raise ValueError("OpenMC cell IDs must be unique")
    material_tags = {item["material"]["tag"] for item in components}
    meshes = value["local_meshes"]
    if not isinstance(meshes, list) or not meshes:
        raise ValueError("activation local meshes cannot be empty")
    for item in meshes:
        _validate_mesh(_mapping(item, "local mesh"), material_tags)
    magnet_ids = {item["magnet_id"] for item in components}
    if {item["magnet_id"] for item in meshes} != magnet_ids:
        raise ValueError("component and local-mesh magnet identities disagree")
    for magnet_id in magnet_ids:
        roles = {
            item["component_role"]
            for item in components
            if item["magnet_id"] == magnet_id
        }
        if roles != set(COMPONENT_ROLES):
            raise ValueError(
                f"magnet {magnet_id!r} lacks winding-pack or casing metadata"
            )
    readiness = _mapping(value["activation_readiness"], "summary readiness")
    _require_keys(
        readiness,
        {
            "direct_r2s_cell_metadata_ready",
            "direct_r2s_mesh_metadata_ready",
            "post_transport_component_metadata_ready",
            "post_transport_mesh_metadata_ready",
            "activation_schedules_present",
            "depletion_results_present",
            "execution_owner",
        },
        "summary readiness",
    )
    for key in (
        "direct_r2s_cell_metadata_ready",
        "direct_r2s_mesh_metadata_ready",
        "post_transport_component_metadata_ready",
        "post_transport_mesh_metadata_ready",
        "activation_schedules_present",
        "depletion_results_present",
    ):
        if type(readiness[key]) is not bool:
            raise ValueError(f"summary readiness {key} must be boolean")
    if not readiness["direct_r2s_cell_metadata_ready"]:
        raise ValueError("cell R2S metadata must be complete")
    if not readiness["post_transport_component_metadata_ready"]:
        raise ValueError("component post-transport metadata must be complete")
    if readiness["direct_r2s_mesh_metadata_ready"] != all(
        item["activation_readiness"]["direct_r2s"]["metadata_ready"]
        for item in meshes
    ):
        raise ValueError("summary direct R2S mesh readiness is inconsistent")
    if readiness["post_transport_mesh_metadata_ready"] != all(
        item["activation_readiness"]["post_transport"]["metadata_ready"]
        for item in meshes
    ):
        raise ValueError(
            "summary post-transport mesh readiness is inconsistent"
        )
    if (
        readiness["activation_schedules_present"]
        or readiness["depletion_results_present"]
        or readiness["execution_owner"] != "downstream_consumer"
    ):
        raise ValueError(
            "ParaStell metadata must not own activation execution"
        )
    _sha256(value["artifact_sha256"], "activation artifact SHA-256")
    expected_hash = _artifact_hash(value)
    if value["artifact_sha256"] != expected_hash:
        raise ValueError("activation metadata artifact hash mismatch")
    return {
        "schema": SCHEMA,
        "components": len(components),
        "local_meshes": len(meshes),
        "artifact_sha256": expected_hash,
        "direct_r2s_mesh_metadata_ready": readiness[
            "direct_r2s_mesh_metadata_ready"
        ],
        "post_transport_mesh_metadata_ready": readiness[
            "post_transport_mesh_metadata_ready"
        ],
    }


def write_activation_ready_metadata(
    path: str | Path, value: Mapping[str, Any]
) -> Path:
    """Validate and write canonical human-readable JSON."""

    validate_activation_ready_metadata(value)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def read_activation_ready_metadata(path: str | Path) -> dict[str, Any]:
    """Read an activation artifact and reject schema or hash tampering."""

    source = Path(path)
    value = _strict_json_load(source)
    if not isinstance(value, dict):
        raise TypeError("activation metadata root must be an object")
    validate_activation_ready_metadata(value)
    return value
