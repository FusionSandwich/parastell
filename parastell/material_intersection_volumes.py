"""Qualified component-intersection volumes for magnet-local flux meshes.

OpenMC mesh tallies score track length in the intersection of every active
filter.  A rotated local mesh combined with a winding-pack ``CellFilter`` must
therefore be normalized by the winding-pack volume inside each mesh element,
not by the element's full geometric volume.  This module records and validates
those volumes without pretending that ParaStell has performed activation.
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

SCHEMA = "parastell.material_intersection_volumes/v1.0.0"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "intersection-volume data must be finite JSON"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: Any, label: str) -> str:
    text = str(value)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _artifact_hash(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("artifact_sha256", None)
    return _canonical_sha256(unhashed)


def mesh_definition_sha256(definition: Mapping[str, Any]) -> str:
    """Return the canonical identity of one local mesh definition."""
    return _canonical_sha256(_mapping(definition, "local mesh definition"))


def _component_by_magnet(associations: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _mapping(associations.get("inventory"), "magnet inventory")
    pairs = _sequence(inventory.get("magnet_pairs"), "magnet pairs")
    result = {}
    for raw_pair in pairs:
        pair = _mapping(raw_pair, "magnet pair")
        magnet_id = str(pair.get("magnet_id", ""))
        winding_pack = _mapping(
            pair.get("winding_pack"), f"{magnet_id} winding pack"
        )
        if not magnet_id or magnet_id in result:
            raise ValueError("magnet pair IDs must be nonempty and unique")
        result[magnet_id] = winding_pack
    if not result:
        raise ValueError("at least one magnet pair is required")
    return result


def _float_array(
    value: Any,
    *,
    count: int,
    label: str,
    nonnegative: bool = True,
) -> list[float]:
    values = [float(item) for item in _sequence(value, label)]
    if len(values) != count or any(not math.isfinite(item) for item in values):
        raise ValueError(f"{label} must contain {count} finite values")
    if nonnegative and any(item < 0.0 for item in values):
        raise ValueError(f"{label} must be nonnegative")
    return values


def build_material_intersection_volume_manifest(
    *,
    local_mesh_manifest: Mapping[str, Any],
    associations: Mapping[str, Any],
    openmc_cell_ids_by_dagmc_volume: Mapping[int, int],
    openmc_material_ids_by_tag: Mapping[str, int],
    results_by_magnet: Mapping[str, Mapping[str, Any]],
    method: str,
    estimator_provenance_sha256: str,
    coordinate_transform_qualification: Mapping[str, Any],
    maximum_relative_standard_deviation: float = 0.05,
    selected_magnet_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a hash-bound, fail-closed material-intersection volume manifest.

    ``results_by_magnet`` supplies ``intersection_volume_cm3`` and
    ``standard_deviation_cm3`` arrays in OpenMC mesh-filter bin order.  The
    producing calculation remains external to this neutral contract because a
    stock ``RegularMesh.material_volumes`` call does not automatically inherit
    a separate ``MeshFilter`` rotation and translation.
    """

    local_mesh_manifest = _mapping(local_mesh_manifest, "local mesh manifest")
    associations = _mapping(associations, "associations")
    if local_mesh_manifest.get("cell_filter_applied") is not True:
        raise ValueError(
            "material-intersection normalization requires component-filtered "
            "local mesh tallies"
        )
    all_meshes = _mapping(local_mesh_manifest.get("meshes"), "local meshes")
    all_components = _component_by_magnet(associations)
    if set(all_meshes) != set(all_components):
        raise ValueError(
            "local meshes and winding packs must identify the same magnets"
        )
    if selected_magnet_ids is None:
        selected = tuple(sorted(all_meshes))
    else:
        selected = tuple(str(value) for value in selected_magnet_ids)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(not value for value in selected)
            or not set(selected).issubset(all_meshes)
        ):
            raise ValueError(
                "selected magnet IDs must be a unique nonempty subset"
            )
        selected = tuple(sorted(selected))
    if set(results_by_magnet) != set(selected):
        raise ValueError(
            "intersection results must identify exactly the selected magnets"
        )
    fingerprint = _sha256(
        associations.get("canonical_geometry_fingerprint"),
        "canonical geometry fingerprint",
    )
    if (
        local_mesh_manifest.get("canonical_geometry_fingerprint")
        != fingerprint
    ):
        raise ValueError(
            "local meshes and associations use different geometry"
        )
    inventory = _mapping(associations.get("inventory"), "magnet inventory")
    dagmc_sha = _sha256(
        inventory.get("dagmc_geometry_sha256")
        or associations.get("dagmc_sha256"),
        "DAGMC geometry SHA-256",
    )
    if not isinstance(method, str) or not method.strip():
        raise ValueError("intersection-volume method must be nonempty")
    estimator_sha = _sha256(
        estimator_provenance_sha256, "estimator provenance SHA-256"
    )
    tolerance = float(maximum_relative_standard_deviation)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "intersection-volume relative tolerance must be positive"
        )
    transform = dict(
        _mapping(
            coordinate_transform_qualification,
            "coordinate-transform qualification",
        )
    )
    if (
        transform.get("status") != "PASS"
        or transform.get("rotation_applied") is not True
        or transform.get("translation_applied") is not True
    ):
        raise ValueError(
            "rotated-mesh material volumes require a passing rotation and "
            "translation qualification"
        )
    rows = {}
    cell_map = {
        int(key): int(value)
        for key, value in openmc_cell_ids_by_dagmc_volume.items()
    }
    material_map = {
        str(key): int(value)
        for key, value in openmc_material_ids_by_tag.items()
    }
    for magnet_id in selected:
        definition = _mapping(all_meshes[magnet_id], f"{magnet_id} local mesh")
        component = all_components[magnet_id]
        result = _mapping(results_by_magnet[magnet_id], f"{magnet_id} result")
        count = int(definition.get("bin_count", 0))
        geometric = float(definition.get("bin_volume_cm3", 0.0))
        if count <= 0 or not math.isfinite(geometric) or geometric <= 0.0:
            raise ValueError(
                f"{magnet_id} local mesh has invalid bin geometry"
            )
        volumes = _float_array(
            result.get("intersection_volume_cm3"),
            count=count,
            label=f"{magnet_id} intersection volumes",
        )
        deviations = _float_array(
            result.get("standard_deviation_cm3"),
            count=count,
            label=f"{magnet_id} intersection-volume standard deviations",
        )
        if any(value > geometric * (1.0 + 1.0e-10) for value in volumes):
            raise ValueError(
                f"{magnet_id} component intersection exceeds geometric bin volume"
            )
        if not any(value > 0.0 for value in volumes):
            raise ValueError(
                f"{magnet_id} local mesh does not intersect its winding pack"
            )
        relative = [
            deviation / volume if volume > 0.0 else None
            for volume, deviation in zip(volumes, deviations)
        ]
        if any(
            deviation > 0.0 and volume == 0.0
            for volume, deviation in zip(volumes, deviations)
        ):
            raise ValueError(
                "empty intersections cannot have positive uncertainty"
            )
        statuses = [
            (
                "EMPTY_INTERSECTION"
                if value is None
                else (
                    "QUALIFIED"
                    if value <= tolerance
                    else "INSUFFICIENT_GEOMETRY_STATISTICS"
                )
            )
            for value in relative
        ]
        dagmc_volume_id = int(component.get("volume_id", 0))
        material_tag = str(component.get("material", ""))
        if dagmc_volume_id not in cell_map or material_tag not in material_map:
            raise ValueError(
                f"{magnet_id} lacks an explicit cell or material identity mapping"
            )
        rows[magnet_id] = {
            "magnet_id": magnet_id,
            "component_role": "winding_pack",
            "dagmc_volume_id": dagmc_volume_id,
            "openmc_cell_id": cell_map[dagmc_volume_id],
            "material_tag": material_tag,
            "openmc_material_id": material_map[material_tag],
            "mesh_definition_sha256": mesh_definition_sha256(definition),
            "bin_count": count,
            "geometric_bin_volume_cm3": geometric,
            "intersection_volume_cm3": volumes,
            "standard_deviation_cm3": deviations,
            "relative_standard_deviation": relative,
            "bin_status": statuses,
            "positive_intersection_bin_count": sum(
                value > 0.0 for value in volumes
            ),
            "total_intersection_volume_cm3": float(sum(volumes)),
            "status": (
                "QUALIFIED"
                if all(
                    status in {"QUALIFIED", "EMPTY_INTERSECTION"}
                    for status in statuses
                )
                else "INSUFFICIENT_GEOMETRY_STATISTICS"
            ),
        }
    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "canonical_geometry_fingerprint": fingerprint,
        "dagmc_geometry_sha256": dagmc_sha,
        "local_mesh_manifest_sha256": _canonical_sha256(local_mesh_manifest),
        "method": method.strip(),
        "estimator_provenance_sha256": estimator_sha,
        "maximum_relative_standard_deviation": tolerance,
        "coordinate_transform_qualification": transform,
        "selected_magnet_ids": list(selected),
        "selection_scope": (
            "all_local_meshes"
            if set(selected) == set(all_meshes)
            else "explicit_selected_magnets"
        ),
        "meshes": rows,
        "status": (
            "QUALIFIED"
            if all(row["status"] == "QUALIFIED" for row in rows.values())
            else "INSUFFICIENT_GEOMETRY_STATISTICS"
        ),
    }
    payload["artifact_sha256"] = _artifact_hash(payload)
    validate_material_intersection_volume_manifest(
        payload,
        local_mesh_manifest=local_mesh_manifest,
        associations=associations,
    )
    return payload


def validate_material_intersection_volume_manifest(
    payload: Mapping[str, Any],
    *,
    local_mesh_manifest: Mapping[str, Any] | None = None,
    associations: Mapping[str, Any] | None = None,
) -> None:
    """Validate identities, dimensions, bounds, uncertainty, and artifact hash."""

    payload = _mapping(payload, "intersection-volume manifest")
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported material-intersection volume schema")
    _sha256(
        payload.get("canonical_geometry_fingerprint"), "geometry fingerprint"
    )
    _sha256(payload.get("dagmc_geometry_sha256"), "DAGMC geometry SHA-256")
    _sha256(
        payload.get("local_mesh_manifest_sha256"), "local mesh manifest hash"
    )
    _sha256(
        payload.get("estimator_provenance_sha256"), "estimator provenance hash"
    )
    if payload.get("artifact_sha256") != _artifact_hash(payload):
        raise ValueError("material-intersection volume artifact hash mismatch")
    transform = _mapping(
        payload.get("coordinate_transform_qualification"),
        "coordinate-transform qualification",
    )
    if (
        transform.get("status") != "PASS"
        or transform.get("rotation_applied") is not True
        or transform.get("translation_applied") is not True
    ):
        raise ValueError("coordinate transform is not qualified")
    tolerance = float(payload.get("maximum_relative_standard_deviation", 0.0))
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("invalid material-intersection volume tolerance")
    meshes = _mapping(payload.get("meshes"), "intersection meshes")
    if not meshes:
        raise ValueError("intersection-volume manifest contains no meshes")
    selected_raw = payload.get("selected_magnet_ids")
    if selected_raw is None:
        selected = tuple(sorted(meshes))
    else:
        selected = tuple(
            str(value)
            for value in _sequence(selected_raw, "selected magnet IDs")
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or tuple(sorted(selected)) != selected
            or set(selected) != set(meshes)
        ):
            raise ValueError(
                "selected magnet IDs must exactly identify the intersection meshes"
            )
    scope = payload.get("selection_scope")
    if scope is not None and scope not in {
        "all_local_meshes",
        "explicit_selected_magnets",
    }:
        raise ValueError("invalid material-intersection selection scope")
    expected_meshes = None
    if local_mesh_manifest is not None:
        local_mesh_manifest = _mapping(
            local_mesh_manifest, "local mesh manifest"
        )
        if payload["local_mesh_manifest_sha256"] != _canonical_sha256(
            local_mesh_manifest
        ):
            raise ValueError(
                "intersection volumes do not bind the local mesh manifest"
            )
        if local_mesh_manifest.get("cell_filter_applied") is not True:
            raise ValueError(
                "bound local mesh tallies are not component filtered"
            )
        expected_meshes = _mapping(
            local_mesh_manifest.get("meshes"), "local meshes"
        )
        if not set(meshes).issubset(expected_meshes):
            raise ValueError(
                "intersection volumes identify an unknown local mesh"
            )
        expected_scope = (
            "all_local_meshes"
            if set(meshes) == set(expected_meshes)
            else "explicit_selected_magnets"
        )
        if scope is not None and scope != expected_scope:
            raise ValueError(
                "material-intersection selection scope is inconsistent"
            )
    if associations is not None:
        associations = _mapping(associations, "associations")
        if payload["canonical_geometry_fingerprint"] != associations.get(
            "canonical_geometry_fingerprint"
        ):
            raise ValueError(
                "intersection volumes use a different geometry fingerprint"
            )
        components = _component_by_magnet(associations)
        if not set(meshes).issubset(components):
            raise ValueError(
                "intersection meshes identify an unknown winding pack"
            )
    overall = "QUALIFIED"
    for magnet_id, raw in meshes.items():
        row = _mapping(raw, f"{magnet_id} intersection mesh")
        if (
            row.get("magnet_id") != magnet_id
            or row.get("component_role") != "winding_pack"
        ):
            raise ValueError("intersection mesh component identity is invalid")
        count = int(row.get("bin_count", 0))
        geometric = float(row.get("geometric_bin_volume_cm3", 0.0))
        if count <= 0 or not math.isfinite(geometric) or geometric <= 0.0:
            raise ValueError("intersection mesh geometry is invalid")
        volumes = _float_array(
            row.get("intersection_volume_cm3"),
            count=count,
            label=f"{magnet_id} intersection volumes",
        )
        deviations = _float_array(
            row.get("standard_deviation_cm3"),
            count=count,
            label=f"{magnet_id} volume deviations",
        )
        if any(value > geometric * (1.0 + 1.0e-10) for value in volumes):
            raise ValueError(
                "intersection volume exceeds geometric bin volume"
            )
        if expected_meshes is not None:
            definition = _mapping(expected_meshes[magnet_id], "local mesh")
            if count != int(definition.get("bin_count", 0)):
                raise ValueError(
                    "intersection and local mesh bin counts disagree"
                )
            if row.get("mesh_definition_sha256") != mesh_definition_sha256(
                definition
            ):
                raise ValueError("intersection mesh definition hash mismatch")
        relative = row.get("relative_standard_deviation")
        statuses = row.get("bin_status")
        if not isinstance(relative, list) or len(relative) != count:
            raise ValueError("relative volume uncertainties are incomplete")
        if not isinstance(statuses, list) or len(statuses) != count:
            raise ValueError("intersection bin statuses are incomplete")
        expected_statuses = []
        for volume, deviation, stored in zip(volumes, deviations, relative):
            expected = deviation / volume if volume > 0.0 else None
            if expected is None:
                if stored is not None or deviation != 0.0:
                    raise ValueError(
                        "empty intersection uncertainty is invalid"
                    )
                expected_statuses.append("EMPTY_INTERSECTION")
            else:
                if stored is None or not math.isclose(
                    float(stored), expected, rel_tol=1.0e-12, abs_tol=0.0
                ):
                    raise ValueError(
                        "relative intersection uncertainty is inconsistent"
                    )
                expected_statuses.append(
                    "QUALIFIED"
                    if expected <= tolerance
                    else "INSUFFICIENT_GEOMETRY_STATISTICS"
                )
        if statuses != expected_statuses:
            raise ValueError(
                "intersection bin qualification statuses are inconsistent"
            )
        expected_row_status = (
            "QUALIFIED"
            if all(
                value in {"QUALIFIED", "EMPTY_INTERSECTION"}
                for value in statuses
            )
            else "INSUFFICIENT_GEOMETRY_STATISTICS"
        )
        if row.get("status") != expected_row_status:
            raise ValueError("intersection mesh qualification is inconsistent")
        if expected_row_status != "QUALIFIED":
            overall = "INSUFFICIENT_GEOMETRY_STATISTICS"
    if payload.get("status") != overall:
        raise ValueError(
            "intersection-volume qualification status is inconsistent"
        )


def write_material_intersection_volume_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    validate_material_intersection_volume_manifest(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_material_intersection_volume_manifest(
    path: str | Path,
    *,
    local_mesh_manifest: Mapping[str, Any] | None = None,
    associations: Mapping[str, Any] | None = None,
    require_qualified: bool = True,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_material_intersection_volume_manifest(
        payload,
        local_mesh_manifest=local_mesh_manifest,
        associations=associations,
    )
    if require_qualified and payload["status"] != "QUALIFIED":
        raise ValueError("material-intersection volumes are not qualified")
    return payload
