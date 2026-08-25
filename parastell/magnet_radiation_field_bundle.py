"""Consumer-neutral magnet radiation-field producer bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


SCHEMA = "parastell.magnet_radiation_field_bundle/v1.0.0"
PRODUCT_KINDS = {
    "activation_ready_metadata",
    "boundary_phase_space",
    "damage_and_gas_production",
    "magnet_geometry_interchange",
    "volume_scalar_flux",
    "heating",
    "reaction_production",
}
PRODUCT_DIRECTORIES = {
    "activation_ready_metadata": "activation_metadata",
    "boundary_phase_space": "boundary",
    "damage_and_gas_production": "damage_and_gas_production",
    "magnet_geometry_interchange": "geometry_interchange",
    "volume_scalar_flux": "volume_flux",
    "heating": "heating",
    "reaction_production": "reaction_production",
}
REQUIRED_GEOMETRY = {"raw_h5m_sha256", "canonical_geometry_fingerprint"}
REQUIRED_SOURCE = {
    "physical_source_rate_per_s",
    "source_definition_sha256",
    "source_mesh_sha256",
}
REQUIRED_PROVENANCE = {
    "parastell_commit",
    "openmc_version",
    "openmc_commit",
    "statepoint_sha256",
    "vmec_sha256",
    "coils_sha256",
    "source_mesh_sha256",
    "histories",
    "batches",
    "seeds",
}
SCALAR_FLUX_SCHEMA = "parastell.magnet_scalar_flux_fields/v1.0.0"
BOUNDARY_SCHEMA_NAME = "parastell.magnet_boundary_source"
SUPPORTED_BOUNDARY_SCHEMA_VERSIONS = {"2.0.0", "2.1.0"}
SUPPORTED_BANK_CLASSIFICATIONS = {
    "COMPLETE_CROSSING_BANK",
    "SAMPLED_CROSSING_BANK",
    "TRUNCATED_INVALID_BANK",
}
SUPPORTED_BOUNDARY_ROLES = {"outer_magnet", "winding_pack"}
REQUIRED_BOUNDARY_RECORD_FIELDS = {
    "record_id",
    "position_global_cm",
    "position_local_cm",
    "direction_global",
    "direction_local",
    "outward_normal_global",
    "energy_eV",
    "weight",
    "particle",
    "particle_pdg",
    "surface_id",
    "envelope_id",
    "crossing_sense",
    "surface_role",
    "mu",
    "azimuth_rad",
    "grazing",
    "patch_id",
    "energy_group",
    "angle_bin_id",
}
REQUIRED_SCALAR_DATASETS = {
    "energy_edges_eV",
    "volume_cm3",
    "track_length_mean_cm_per_source",
    "track_length_std_dev_cm_per_source",
    "mean_per_source",
    "std_dev_per_source",
    "mean_physical",
    "std_dev_physical",
    "relative_uncertainty",
    "global_centroid_cm",
    "local_centreline_coordinates_cm",
    "nearest_centreline_global_cm",
    "centreline_arclength_cm",
    "normalized_arclength",
    "centreline_tangent",
    "centreline_radial",
    "centreline_transverse",
    "distance_to_centreline_cm",
    "centreline_linkage_status",
    "frame_type",
    "frame_quality_status",
    "region_ids",
    "magnet_ids",
    "material_ids",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_fields(
    value: Mapping[str, Any], required: set[str], label: str
) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{label} is missing {sorted(missing)}")


def _require_sha256(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return text


def _require_git_sha(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a full Git commit SHA")
    return text


def _normalise_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical bundle provenance while accepting legacy aliases."""
    if not isinstance(provenance, Mapping):
        raise ValueError("bundle provenance must be an object")
    result = dict(provenance)
    aliases = {
        "openmc_statepoint_sha256": "statepoint_sha256",
        "coil_file_sha256": "coils_sha256",
        "vmec_file_sha256": "vmec_sha256",
        "openmc_git_sha": "openmc_commit",
        "openmc_sha": "openmc_commit",
        "parastell_sha": "parastell_commit",
    }
    for legacy, canonical in aliases.items():
        if canonical not in result and legacy in result:
            result[canonical] = result[legacy]
    if "seeds" not in result and "seed" in result:
        result["seeds"] = [result["seed"]]
    _require_fields(result, REQUIRED_PROVENANCE, "bundle provenance")
    result["parastell_commit"] = _require_git_sha(
        result["parastell_commit"], "parastell_commit"
    )
    result["openmc_commit"] = _require_git_sha(
        result["openmc_commit"], "openmc_commit"
    )
    for name in (
        "statepoint_sha256",
        "vmec_sha256",
        "coils_sha256",
        "source_mesh_sha256",
    ):
        result[name] = _require_sha256(result[name], name)
    if (
        not isinstance(result["openmc_version"], str)
        or not result["openmc_version"].strip()
    ):
        raise ValueError("openmc_version cannot be empty")
    for name in ("histories", "batches"):
        value = int(result[name])
        if (
            isinstance(result[name], bool)
            or value <= 0
            or value != result[name]
        ):
            raise ValueError(f"{name} must be a positive integer")
        result[name] = value
    raw_seeds = result["seeds"]
    if isinstance(raw_seeds, (str, bytes)) or not isinstance(
        raw_seeds, Sequence
    ):
        raise ValueError("seeds must be a sequence of positive integers")
    seeds = [int(value) for value in raw_seeds]
    if (
        not seeds
        or any(isinstance(raw, bool) or int(raw) != raw for raw in raw_seeds)
        or any(value <= 0 for value in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("seeds must contain distinct positive integers")
    result["seeds"] = seeds
    if result["histories"] < result["batches"] or (
        result["histories"] % result["batches"]
    ):
        raise ValueError("histories must be a whole positive number per batch")
    return result


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json_dataset(dataset: Any, label: str) -> dict[str, Any]:
    try:
        value = (
            dataset.asstr()[()] if hasattr(dataset, "asstr") else dataset[()]
        )
        document = json.loads(_decode(value))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return document


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1.0e-12)
    return left == right


def _require_matching_provenance(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    names: Sequence[tuple[str, str]],
    label: str,
) -> None:
    for actual_name, expected_name in names:
        if actual_name not in actual:
            raise ValueError(
                f"{label} is missing provenance field {actual_name!r}"
            )
        if not _same_value(actual[actual_name], expected[expected_name]):
            raise ValueError(
                f"{label} provenance {actual_name!r} does not match the bundle"
            )


def _validate_scalar_flux_hdf5(
    path: Path,
    *,
    product: Mapping[str, Any],
    inventory_magnet_ids: set[str],
    source: Mapping[str, Any],
    provenance: Mapping[str, Any],
    geometry: Mapping[str, Any],
    nuclear_data: Mapping[str, Any],
    materials: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate scalar-flux datasets, formulas, and embedded provenance."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "h5py and numpy are required to validate a bundle"
        ) from exc

    rate = float(source["physical_source_rate_per_s"])
    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise ValueError("scalar-flux product is not valid HDF5") from exc
    with handle:
        if _decode(handle.attrs.get("schema", "")) != SCALAR_FLUX_SCHEMA:
            raise ValueError("unsupported scalar-flux HDF5 schema")
        if _decode(handle.attrs.get("quantity", "")) != "volume_scalar_flux":
            raise ValueError("scalar-flux product contains surface current")
        if not {"manifest_json", "scalar_flux_fields"}.issubset(handle):
            raise ValueError("scalar-flux HDF5 is missing required objects")
        manifest = _json_dataset(
            handle["manifest_json"], "scalar-flux manifest"
        )
        if manifest.get("schema") != SCALAR_FLUX_SCHEMA:
            raise ValueError("scalar-flux manifest schema disagrees with HDF5")
        if manifest.get("quantity") != "volume_scalar_flux":
            raise ValueError("scalar-flux manifest has an invalid quantity")
        normalization = manifest.get("normalization")
        if not isinstance(normalization, Mapping):
            raise ValueError("scalar-flux normalization is absent")
        _require_fields(
            normalization,
            {
                "per_source_units",
                "physical_units",
                "physical_source_rate_per_s",
            },
            "scalar-flux normalization",
        )
        if (
            normalization["per_source_units"] != "1/cm2/source"
            or normalization["physical_units"] != "particles/cm2/s"
        ):
            raise ValueError("scalar-flux normalization units are invalid")
        if not _same_value(normalization["physical_source_rate_per_s"], rate):
            raise ValueError(
                "scalar-flux source rate does not match the bundle"
            )
        scalar_provenance = manifest.get("provenance")
        if not isinstance(scalar_provenance, Mapping):
            raise ValueError("scalar-flux provenance is absent")
        _require_matching_provenance(
            scalar_provenance,
            {
                **provenance,
                **geometry,
                **source,
                "nuclear_data_manifest_sha256": nuclear_data[
                    "manifest_sha256"
                ],
            },
            (
                ("parastell_commit", "parastell_commit"),
                ("openmc_version", "openmc_version"),
                ("openmc_commit", "openmc_commit"),
                ("statepoint_sha256", "statepoint_sha256"),
                ("vmec_sha256", "vmec_sha256"),
                ("coils_sha256", "coils_sha256"),
                ("histories", "histories"),
                ("batches", "batches"),
                ("seeds", "seeds"),
                ("raw_h5m_sha256", "raw_h5m_sha256"),
                (
                    "canonical_geometry_fingerprint",
                    "canonical_geometry_fingerprint",
                ),
                ("source_definition_sha256", "source_definition_sha256"),
                ("source_mesh_sha256", "source_mesh_sha256"),
                (
                    "nuclear_data_manifest_sha256",
                    "nuclear_data_manifest_sha256",
                ),
            ),
            "scalar-flux",
        )
        material_hash = manifest.get("material_manifest_sha256")
        expected_material_hashes = {
            materials.get("manifest_sha256"),
            materials.get("file_sha256"),
            materials.get("resolved_manifest_sha256"),
        } - {None}
        if material_hash not in expected_material_hashes:
            raise ValueError(
                "scalar-flux material provenance does not match the bundle"
            )

        root = handle["scalar_flux_fields"]
        if not root.keys():
            raise ValueError("scalar-flux HDF5 contains no fields")
        field_entries = manifest.get("fields")
        if not isinstance(field_entries, list):
            raise ValueError("scalar-flux field manifest is absent")
        entry_by_name = {
            str(item.get("name")): item
            for item in field_entries
            if isinstance(item, Mapping) and "name" in item
        }
        if len(field_entries) != len(entry_by_name) or set(
            entry_by_name
        ) != set(root.keys()):
            raise ValueError(
                "scalar-flux field inventory does not match HDF5 groups"
            )

        ccfe_709_ready = False
        observed_magnet_ids: set[str] = set()
        for field_name, group in root.items():
            missing = REQUIRED_SCALAR_DATASETS - set(group)
            if missing:
                raise ValueError(
                    f"scalar-flux field {field_name!r} is missing {sorted(missing)}"
                )
            if (
                _decode(group.attrs.get("quantity", ""))
                != "volume_scalar_flux"
            ):
                raise ValueError(
                    f"scalar-flux field {field_name!r} has bad quantity"
                )
            particle = _decode(group.attrs.get("particle", ""))
            field_kind = _decode(group.attrs.get("field_kind", ""))
            if particle not in {"neutron", "photon"} or field_kind not in {
                "whole_volume",
                "local_mesh",
            }:
                raise ValueError(
                    f"scalar-flux field {field_name!r} has bad metadata"
                )
            if (
                _decode(group.attrs.get("per_source_units", ""))
                != "1/cm2/source"
                or _decode(group.attrs.get("physical_units", ""))
                != "particles/cm2/s"
            ):
                raise ValueError(
                    f"scalar-flux field {field_name!r} has bad units"
                )

            edges = np.asarray(group["energy_edges_eV"][...], dtype=float)
            volumes = np.asarray(group["volume_cm3"][...], dtype=float)
            if (
                edges.ndim != 1
                or len(edges) < 2
                or np.any(~np.isfinite(edges))
                or np.any(np.diff(edges) <= 0.0)
            ):
                raise ValueError(
                    f"scalar-flux field {field_name!r} has bad energy edges"
                )
            if (
                volumes.ndim != 1
                or not len(volumes)
                or np.any(~np.isfinite(volumes) | (volumes <= 0.0))
            ):
                raise ValueError(
                    f"scalar-flux field {field_name!r} has bad volumes"
                )
            expected_shape = (len(volumes), len(edges) - 1)
            arrays = {
                name: np.asarray(group[name][...], dtype=float)
                for name in (
                    "track_length_mean_cm_per_source",
                    "track_length_std_dev_cm_per_source",
                    "mean_per_source",
                    "std_dev_per_source",
                    "mean_physical",
                    "std_dev_physical",
                    "relative_uncertainty",
                )
            }
            if any(array.shape != expected_shape for array in arrays.values()):
                raise ValueError(
                    f"scalar-flux field {field_name!r} arrays misalign"
                )
            for name in (
                "track_length_mean_cm_per_source",
                "track_length_std_dev_cm_per_source",
                "mean_per_source",
                "std_dev_per_source",
                "mean_physical",
                "std_dev_physical",
            ):
                if np.any(~np.isfinite(arrays[name]) | (arrays[name] < 0.0)):
                    raise ValueError(
                        f"scalar-flux field {field_name!r} has invalid {name}"
                    )
            field_magnet_ids = set(
                np.asarray(group["magnet_ids"].asstr()[()]).astype(str)
            )
            if not field_magnet_ids:
                raise ValueError(
                    f"scalar-flux field {field_name!r} has no magnet IDs"
                )
            observed_magnet_ids.update(field_magnet_ids)
            expected_per_source = (
                arrays["track_length_mean_cm_per_source"] / volumes[:, None]
            )
            expected_std = (
                arrays["track_length_std_dev_cm_per_source"] / volumes[:, None]
            )
            expected_relative = np.divide(
                expected_std,
                expected_per_source,
                out=np.full_like(expected_std, np.inf),
                where=expected_per_source > 0.0,
            )
            comparisons = (
                (arrays["mean_per_source"], expected_per_source),
                (arrays["std_dev_per_source"], expected_std),
                (arrays["mean_physical"], expected_per_source * rate),
                (arrays["std_dev_physical"], expected_std * rate),
                (arrays["relative_uncertainty"], expected_relative),
            )
            if any(
                not np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)
                for actual, expected in comparisons
            ):
                raise ValueError(
                    f"scalar-flux field {field_name!r} normalization is inconsistent"
                )
            for name in (
                "global_centroid_cm",
                "local_centreline_coordinates_cm",
                "nearest_centreline_global_cm",
                "centreline_tangent",
                "centreline_radial",
                "centreline_transverse",
            ):
                coordinates = np.asarray(group[name][...], dtype=float)
                if coordinates.shape != (len(volumes), 3) or np.any(
                    ~np.isfinite(coordinates)
                ):
                    raise ValueError(
                        f"scalar-flux field {field_name!r} has invalid coordinates"
                    )
            for name in (
                "centreline_arclength_cm",
                "normalized_arclength",
                "distance_to_centreline_cm",
            ):
                values = np.asarray(group[name][...], dtype=float)
                if values.shape != (len(volumes),) or np.any(
                    ~np.isfinite(values)
                ):
                    raise ValueError(
                        f"scalar-flux field {field_name!r} has invalid {name}"
                    )
            normalized_arclength = np.asarray(
                group["normalized_arclength"][...]
            )
            if np.any(
                (normalized_arclength < 0.0) | (normalized_arclength > 1.0)
            ):
                raise ValueError(
                    f"scalar-flux field {field_name!r} has invalid normalized arclength"
                )
            for name in (
                "region_ids",
                "magnet_ids",
                "material_ids",
                "centreline_linkage_status",
                "frame_type",
                "frame_quality_status",
            ):
                if group[name].shape != (len(volumes),):
                    raise ValueError(
                        f"scalar-flux field {field_name!r} has invalid {name}"
                    )
            linkage = np.asarray(
                group["centreline_linkage_status"].asstr()[()]
            ).astype(str)
            if np.any(linkage != "LINKED_NEAREST_CENTRELINE_SEGMENT"):
                raise ValueError(
                    f"scalar-flux field {field_name!r} lacks centreline linkage"
                )
            if "resolution_status" in group:
                status = np.asarray(group["resolution_status"][...])
                if status.shape != expected_shape or np.any(status > 2):
                    raise ValueError(
                        f"scalar-flux field {field_name!r} has bad resolution status"
                    )
            entry = entry_by_name[field_name]
            if (
                entry.get("particle") != particle
                or entry.get("field_kind") != field_kind
                or int(entry.get("regions", -1)) != len(volumes)
                or int(entry.get("energy_groups", -1)) != len(edges) - 1
            ):
                raise ValueError(
                    f"scalar-flux field {field_name!r} manifest metadata disagrees"
                )
            edge_hash = hashlib.sha256(
                json.dumps(
                    [float(value) for value in edges], separators=(",", ":")
                ).encode("ascii")
            ).hexdigest()
            if entry.get("energy_edges_sha256") != edge_hash:
                raise ValueError(
                    f"scalar-flux field {field_name!r} energy hash disagrees"
                )
            if (
                particle == "neutron"
                and entry.get("energy_structure") == "CCFE-709"
            ):
                from .energy_groups import get_structure

                expected_edges = np.asarray(
                    get_structure("CCFE-709", particle="neutron").edges_eV,
                    dtype=float,
                )
                if not np.array_equal(edges, expected_edges):
                    raise ValueError(
                        "CCFE-709 scalar flux does not use exact boundaries"
                    )
                ccfe_709_ready = True
        if not ccfe_709_ready:
            raise ValueError(
                "bundle scalar flux lacks exact CCFE-709 neutron output"
            )
        _validate_embedded_magnet_routing(
            product=product,
            observed_magnet_ids=observed_magnet_ids,
            inventory_magnet_ids=inventory_magnet_ids,
            label="scalar-flux",
        )
    return {"field_count": len(entry_by_name), "ccfe_709_ready": True}


def _validate_embedded_magnet_routing(
    *,
    product: Mapping[str, Any],
    observed_magnet_ids: set[str],
    inventory_magnet_ids: set[str],
    label: str,
) -> None:
    """Bind product-internal magnet identities to its bundle route."""
    if not observed_magnet_ids:
        raise ValueError(f"{label} product contains no magnet identities")
    unknown = observed_magnet_ids - inventory_magnet_ids
    if unknown:
        raise ValueError(
            f"{label} product contains magnets absent from the inventory: "
            f"{sorted(unknown)}"
        )
    route = str(product["magnet_id"])
    if (
        route == "all-selected-magnets"
        and observed_magnet_ids != inventory_magnet_ids
    ):
        raise ValueError(
            f"{label} all-selected route does not cover the full inventory"
        )
    if route != "all-selected-magnets" and observed_magnet_ids != {route}:
        raise ValueError(
            f"{label} product magnet identities disagree with its bundle route"
        )


def _validate_damage_gas_hdf5(
    path: Path,
    *,
    product: Mapping[str, Any],
    inventory_magnet_ids: set[str],
    source: Mapping[str, Any],
    provenance: Mapping[str, Any],
    geometry: Mapping[str, Any],
    nuclear_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and bind a damage/gas product to bundle provenance."""
    from .magnet_damage_gas import validate_magnet_damage_gas

    summary = validate_magnet_damage_gas(path)
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("h5py is required to validate a bundle") from exc

    with h5py.File(path, "r") as handle:
        manifest = _json_dataset(
            handle["manifest_json"], "damage/gas manifest"
        )
        if not _same_value(
            manifest.get("physical_source_rate_per_s"),
            source["physical_source_rate_per_s"],
        ):
            raise ValueError(
                "damage/gas source rate does not match the bundle"
            )
        product_provenance = manifest.get("provenance")
        if not isinstance(product_provenance, Mapping):
            raise ValueError("damage/gas provenance is absent")
        _require_matching_provenance(
            product_provenance,
            {
                **provenance,
                **geometry,
                **source,
                "nuclear_data_manifest_sha256": nuclear_data[
                    "manifest_sha256"
                ],
            },
            (
                ("parastell_commit", "parastell_commit"),
                ("openmc_version", "openmc_version"),
                ("openmc_commit", "openmc_commit"),
                ("statepoint_sha256", "statepoint_sha256"),
                ("vmec_sha256", "vmec_sha256"),
                ("coils_sha256", "coils_sha256"),
                ("histories", "histories"),
                ("batches", "batches"),
                ("seeds", "seeds"),
                ("raw_h5m_sha256", "raw_h5m_sha256"),
                (
                    "canonical_geometry_fingerprint",
                    "canonical_geometry_fingerprint",
                ),
                ("source_definition_sha256", "source_definition_sha256"),
                ("source_mesh_sha256", "source_mesh_sha256"),
                (
                    "nuclear_data_manifest_sha256",
                    "nuclear_data_manifest_sha256",
                ),
            ),
            "damage/gas",
        )
        observed_magnet_ids: set[str] = set()
        for group_name in (
            "damage_energy/neutron",
            "gas_production/neutron",
        ):
            if group_name in handle:
                observed_magnet_ids.update(
                    str(value)
                    for value in handle[group_name]["magnet_ids"].asstr()[()]
                )
        _validate_embedded_magnet_routing(
            product=product,
            observed_magnet_ids=observed_magnet_ids,
            inventory_magnet_ids=inventory_magnet_ids,
            label="damage/gas",
        )
    return summary


def _validate_bank_completeness(
    value: Mapping[str, Any], *, record_count: int
) -> dict[str, Any]:
    required = {
        "classification",
        "stored_record_count",
        "selected_record_count",
        "source_file_count",
        "max_particles_per_file",
        "max_source_files",
        "configured_capacity",
        "cap_reached",
        "sampling_applied",
        "mpi_ranks",
    }
    _require_fields(value, required, "boundary-bank completeness")
    classification = str(value["classification"])
    if classification not in SUPPORTED_BANK_CLASSIFICATIONS:
        raise ValueError(
            f"unknown boundary-bank classification {classification!r}"
        )
    stored = int(value["stored_record_count"])
    selected = int(value["selected_record_count"])
    source_files = int(value["source_file_count"])
    if stored < 0 or selected < 0 or selected > stored or source_files <= 0:
        raise ValueError("boundary-bank completeness counts are inconsistent")
    if selected != record_count:
        raise ValueError(
            "boundary-bank selected count does not match HDF5 records"
        )
    maximum = value["max_particles_per_file"]
    file_limit = value["max_source_files"]
    if (maximum is None) != (file_limit is None):
        raise ValueError(
            "boundary-bank capacity controls are only partially recorded"
        )
    if maximum is not None:
        maximum = int(maximum)
        file_limit = int(file_limit)
        if maximum <= 0 or file_limit <= 0:
            raise ValueError(
                "boundary-bank capacity controls must be positive"
            )
        capacity = maximum * file_limit
        if int(value["configured_capacity"]) != capacity:
            raise ValueError(
                "boundary-bank configured capacity is inconsistent"
            )
    else:
        capacity = None
        if value["configured_capacity"] is not None:
            raise ValueError(
                "boundary-bank capacity must be null when controls are absent"
            )
    cap_reached = bool(value["cap_reached"])
    sampling = bool(value["sampling_applied"])
    if cap_reached != (capacity is not None and stored >= capacity):
        raise ValueError("boundary-bank cap status is inconsistent")
    expected = (
        "TRUNCATED_INVALID_BANK"
        if cap_reached
        else (
            "SAMPLED_CROSSING_BANK"
            if sampling or capacity is None
            else "COMPLETE_CROSSING_BANK"
        )
    )
    if classification != expected:
        raise ValueError(
            "boundary-bank classification is inconsistent with capture controls"
        )
    mpi_ranks = value["mpi_ranks"]
    if mpi_ranks is not None and int(mpi_ranks) <= 0:
        raise ValueError("boundary-bank MPI rank count must be positive")
    return {
        "classification": classification,
        "record_count": record_count,
        "stored_record_count": stored,
        "sampling_applied": sampling,
        "cap_reached": cap_reached,
    }


def _validate_boundary_hdf5(
    path: Path,
    *,
    product: Mapping[str, Any],
    source: Mapping[str, Any],
    provenance: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a canonical correlated bank without importing OpenMC/DAGMC."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "h5py and numpy are required to validate a bundle"
        ) from exc

    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise ValueError(
            "boundary phase-space product is not valid HDF5"
        ) from exc
    with handle:
        schema = _decode(handle.attrs.get("schema", ""))
        version = _decode(handle.attrs.get("schema_version", ""))
        if (
            version not in SUPPORTED_BOUNDARY_SCHEMA_VERSIONS
            or schema != f"{BOUNDARY_SCHEMA_NAME}/v{version}"
        ):
            raise ValueError("unsupported boundary phase-space HDF5 schema")
        if not {"manifest_json", "records", "projection"}.issubset(handle):
            raise ValueError("boundary HDF5 is missing required objects")
        manifest = _json_dataset(handle["manifest_json"], "boundary manifest")
        if (
            manifest.get("schema") != schema
            or manifest.get("schema_version") != version
        ):
            raise ValueError("boundary manifest schema disagrees with HDF5")
        if manifest.get("quantity") != "partial current":
            raise ValueError("boundary manifest has an invalid quantity")
        if manifest.get("canonical_bank") is not True:
            raise ValueError(
                "bundle boundary product must be a canonical raw bank"
            )
        records = handle["records"]
        missing = REQUIRED_BOUNDARY_RECORD_FIELDS - set(records)
        if missing:
            raise ValueError(f"boundary records are missing {sorted(missing)}")
        lengths = {len(dataset) for dataset in records.values()}
        if len(lengths) != 1:
            raise ValueError(
                "boundary record columns have inconsistent lengths"
            )
        record_count = lengths.pop()
        if int(manifest.get("record_count", -1)) != record_count:
            raise ValueError(
                "boundary manifest record count disagrees with HDF5"
            )
        if (
            "record_count" in product
            and int(product["record_count"]) != record_count
        ):
            raise ValueError("bundle product record count disagrees with HDF5")
        for name in (
            "position_global_cm",
            "position_local_cm",
            "direction_global",
            "direction_local",
            "outward_normal_global",
        ):
            if records[name].shape != (record_count, 3):
                raise ValueError(
                    f"boundary record {name!r} has an invalid shape"
                )
        identifiers = np.asarray(records["record_id"][...])
        if (
            identifiers.shape != (record_count,)
            or len(np.unique(identifiers)) != record_count
        ):
            raise ValueError("boundary record IDs are not unique")
        finite_names = (
            "position_global_cm",
            "position_local_cm",
            "direction_global",
            "direction_local",
            "outward_normal_global",
            "energy_eV",
            "weight",
            "mu",
            "azimuth_rad",
        )
        if any(
            np.any(~np.isfinite(records[name][...])) for name in finite_names
        ):
            raise ValueError("boundary records contain nonfinite values")
        weights = np.asarray(records["weight"][...], dtype=float)
        energies = np.asarray(records["energy_eV"][...], dtype=float)
        if np.any(weights < 0.0) or np.any(energies < 0.0):
            raise ValueError("boundary energy and weight must be nonnegative")
        directions = np.asarray(records["direction_global"][...], dtype=float)
        normals = np.asarray(
            records["outward_normal_global"][...], dtype=float
        )
        if record_count and (
            not np.allclose(
                np.linalg.norm(directions, axis=1), 1.0, atol=1.0e-12
            )
            or not np.allclose(
                np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-12
            )
        ):
            raise ValueError(
                "boundary directions or normals are not normalized"
            )
        mu = np.asarray(records["mu"][...], dtype=float)
        if not np.allclose(
            mu, np.sum(directions * normals, axis=1), atol=1.0e-12
        ):
            raise ValueError(
                "boundary mu is inconsistent with direction and normal"
            )
        particles = np.asarray(records["particle"].asstr()[()]).astype(str)
        particle_pdg = np.asarray(records["particle_pdg"][...], dtype=int)
        expected_pdg = np.asarray(
            [
                {"neutron": 2112, "photon": 22}.get(value, -1)
                for value in particles
            ]
        )
        if not np.array_equal(particle_pdg, expected_pdg):
            raise ValueError("boundary particle names and PDG values disagree")
        senses = np.asarray(records["crossing_sense"].asstr()[()]).astype(str)
        expected_senses = np.where(
            mu > 1.0e-12,
            "outgoing",
            np.where(mu < -1.0e-12, "incoming", "grazing"),
        )
        if not np.array_equal(senses, expected_senses):
            raise ValueError("boundary crossing senses disagree with mu")
        local_directions = np.asarray(
            records["direction_local"][...], dtype=float
        )
        if record_count and not np.allclose(
            np.linalg.norm(local_directions, axis=1), 1.0, atol=1.0e-12
        ):
            raise ValueError("boundary local directions are not normalized")
        if not np.allclose(local_directions[:, 2], mu, atol=1.0e-12):
            raise ValueError("boundary local directions disagree with mu")
        if not math.isclose(
            float(np.sum(weights)),
            float(manifest.get("integrated_current", math.nan)),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "boundary integrated current does not match record weights"
            )
        projection = handle["projection"]
        if not {"mean", "std_dev"}.issubset(projection):
            raise ValueError(
                "boundary projection is missing mean or uncertainty"
            )
        projection_mean = np.asarray(projection["mean"][...], dtype=float)
        projection_std = np.asarray(projection["std_dev"][...], dtype=float)
        if (
            projection_mean.shape != projection_std.shape
            or np.any(~np.isfinite(projection_mean))
            or np.any(~np.isfinite(projection_std))
            or np.any(projection_mean < 0.0)
            or np.any(projection_std < 0.0)
            or not math.isclose(
                float(np.sum(projection_mean)),
                float(np.sum(weights)),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "boundary projection is inconsistent with canonical records"
            )

        normalization = manifest.get("normalization")
        if not isinstance(normalization, Mapping):
            raise ValueError("boundary normalization is absent")
        _require_fields(
            normalization,
            {
                "basis",
                "particles_per_source_history",
                "area_basis",
                "energy_bin_width",
                "solid_angle_measure",
                "quantity",
                "time_basis",
            },
            "boundary normalization",
        )
        if not _same_value(
            normalization.get("physical_source_rate_per_s"),
            source["physical_source_rate_per_s"],
        ):
            raise ValueError("boundary source rate does not match the bundle")
        bank_provenance = manifest.get("provenance")
        if not isinstance(bank_provenance, Mapping):
            raise ValueError("boundary provenance is absent")
        _require_matching_provenance(
            bank_provenance,
            {**provenance, **geometry, **source},
            (
                ("parastell_commit", "parastell_commit"),
                ("openmc_version", "openmc_version"),
                ("openmc_statepoint_sha256", "statepoint_sha256"),
                ("dagmc_geometry_sha256", "raw_h5m_sha256"),
                ("source_definition_sha256", "source_definition_sha256"),
                ("histories", "histories"),
            ),
            "boundary",
        )
        envelope = manifest.get("envelope")
        if not isinstance(envelope, Mapping):
            raise ValueError("boundary envelope description is absent")
        if envelope.get("magnet_component") != product["magnet_id"]:
            raise ValueError(
                "boundary envelope magnet does not match bundle product"
            )
        if "boundary_role" in product:
            actual_role = envelope.get("metadata", {}).get(
                "boundary_role", "winding_pack"
            )
            if actual_role != product["boundary_role"]:
                raise ValueError(
                    "boundary envelope role does not match bundle product"
                )
        if envelope.get("dagmc_geometry_sha256") != geometry["raw_h5m_sha256"]:
            raise ValueError(
                "boundary envelope geometry does not match bundle"
            )
        surfaces = envelope.get("surfaces")
        if (
            not isinstance(surfaces, list)
            or not surfaces
            or envelope.get("watertight") is not True
        ):
            raise ValueError("boundary envelope is not explicitly closed")
        envelope_surface_ids = [int(item["surface_id"]) for item in surfaces]
        if len(envelope_surface_ids) != len(set(envelope_surface_ids)):
            raise ValueError("boundary envelope surface IDs are not unique")
        if not set(np.asarray(records["surface_id"][...], dtype=int)).issubset(
            envelope_surface_ids
        ):
            raise ValueError(
                "boundary records reference a surface outside the envelope"
            )
        bank_metadata = manifest.get("bank_metadata")
        if not isinstance(bank_metadata, Mapping):
            raise ValueError("boundary bank metadata is absent")
        if (
            bank_metadata.get("canonical_bank", manifest.get("canonical_bank"))
            is not True
        ):
            raise ValueError(
                "boundary metadata does not identify a canonical bank"
            )
        availability = manifest.get("field_availability")
        if not isinstance(availability, Mapping):
            raise ValueError("boundary field-availability manifest is absent")
        for name, status in availability.items():
            if not isinstance(status, Mapping) or bool(
                status.get("available")
            ) != (name in records):
                raise ValueError(
                    f"boundary field availability disagrees for {name!r}"
                )
        for name in REQUIRED_BOUNDARY_RECORD_FIELDS:
            if not availability.get(name, {}).get("available"):
                raise ValueError(
                    f"required boundary field {name!r} is unavailable"
                )
        completeness = bank_metadata.get("surface_bank_completeness")
        if not isinstance(completeness, Mapping):
            raise ValueError("boundary bank completeness is absent")
        result = _validate_bank_completeness(
            completeness, record_count=record_count
        )
        zero = bank_metadata.get("zero_record_interpretation")
        if (
            not isinstance(zero, Mapping)
            or zero.get("physical_zero_claimed") is not False
        ):
            raise ValueError(
                "boundary bank must explicitly avoid a physical-zero claim"
            )
        if record_count == 0:
            if zero.get("status") != "EMPTY":
                raise ValueError(
                    "empty boundary bank lacks EMPTY interpretation"
                )
            upper = zero.get(
                "poisson_upper_bound_expected_crossings_per_source"
            )
            if result["classification"] == "COMPLETE_CROSSING_BANK" and (
                upper is None
                or not math.isfinite(float(upper))
                or float(upper) <= 0.0
            ):
                raise ValueError(
                    "empty complete bank lacks a positive confidence bound"
                )
        result["empty"] = record_count == 0
        return result


def _bank_completeness_summary(
    banks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = {name: 0 for name in sorted(SUPPORTED_BANK_CLASSIFICATIONS)}
    for bank in banks:
        counts[str(bank["classification"])] += 1
    status = (
        "NO_BOUNDARY_BANKS"
        if not banks
        else (
            "INVALID"
            if counts["TRUNCATED_INVALID_BANK"]
            else ("SAMPLED" if counts["SAMPLED_CROSSING_BANK"] else "COMPLETE")
        )
    )
    return {
        "status": status,
        "bank_count": len(banks),
        "classification_counts": counts,
        "record_count_total": sum(int(item["record_count"]) for item in banks),
        "empty_bank_count": sum(bool(item["empty"]) for item in banks),
        "all_banks_valid": counts["TRUNCATED_INVALID_BANK"] == 0,
        "all_banks_complete": bool(banks)
        and counts["COMPLETE_CROSSING_BANK"] == len(banks),
        "empty_banks_are_physical_zero": False,
    }


def _require_boundary_coverage(
    products: Sequence[Mapping[str, Any]], magnet_ids: Sequence[str]
) -> None:
    boundaries = [
        item for item in products if item.get("kind") == "boundary_phase_space"
    ]
    explicit_roles = ["boundary_role" in item for item in boundaries]
    if any(explicit_roles):
        if not all(explicit_roles):
            raise ValueError(
                "boundary products cannot mix legacy and role-qualified routes"
            )
        routes = [
            (str(item.get("magnet_id", "")), str(item["boundary_role"]))
            for item in boundaries
        ]
        expected = {
            (str(magnet_id), role)
            for magnet_id in magnet_ids
            for role in SUPPORTED_BOUNDARY_ROLES
        }
        if len(routes) != len(set(routes)) or set(routes) != expected:
            raise ValueError(
                "role-qualified boundary products must cover outer_magnet and "
                "winding_pack for every inventory magnet exactly once"
            )
        return
    boundary_routes = [str(item.get("magnet_id", "")) for item in boundaries]
    if len(boundary_routes) != len(set(boundary_routes)) or set(
        boundary_routes
    ) != set(magnet_ids):
        raise ValueError(
            "boundary products must cover every inventory magnet exactly once"
        )


def _validate_product(product: Mapping[str, Any]) -> None:
    if not isinstance(product, Mapping):
        raise ValueError("radiation product must be an object")
    _require_fields(
        product,
        {"kind", "magnet_id", "path", "units", "normalization"},
        "radiation product",
    )
    if product["kind"] not in PRODUCT_KINDS:
        raise ValueError(f"unknown radiation product kind {product['kind']!r}")
    magnet_id = str(product["magnet_id"])
    if not magnet_id or Path(magnet_id).name != magnet_id:
        raise ValueError("radiation product magnet_id must be a path-safe ID")
    if product["kind"] == "volume_scalar_flux":
        if product["units"] not in {"1/cm2/source", "particles/cm2/s"}:
            raise ValueError("volume scalar flux has an unknown unit")
        if product.get("quantity") != "volume_scalar_flux":
            raise ValueError(
                "surface current cannot be supplied as volume scalar flux"
            )
        if product["normalization"] != "physical_source_rate":
            raise ValueError(
                "volume scalar flux must use physical source normalization"
            )
    if product["kind"] == "boundary_phase_space":
        if "boundary_role" in product and product["boundary_role"] not in (
            SUPPORTED_BOUNDARY_ROLES
        ):
            raise ValueError(
                "boundary phase space has an unknown boundary role"
            )
        if product["units"] != "crossings/source":
            raise ValueError("boundary phase space has an unknown unit")
        if product.get("quantity") != "partial_crossing_current":
            raise ValueError("boundary phase space has an unknown quantity")
        if product["normalization"] != "per_source_history":
            raise ValueError(
                "boundary phase space must be normalized per source"
            )
    if product["kind"] == "heating":
        if product["units"] not in {"W", "W/cm3"}:
            raise ValueError("heating has an unknown unit")
        if product.get("quantity") != "heating":
            raise ValueError(
                "a non-heating quantity cannot be supplied as heating"
            )
    if product["kind"] == "reaction_production":
        if product["units"] not in {"events/source", "events/s"}:
            raise ValueError("reaction/production product has an unknown unit")
        if product.get("quantity") != "reaction_and_particle_production":
            raise ValueError(
                "reaction/production product has an unknown quantity"
            )
    if product["kind"] == "damage_and_gas_production":
        if product["units"] != "mixed_explicit_in_product":
            raise ValueError(
                "damage/gas product must retain explicit per-dataset units"
            )
        if product.get("quantity") != "damage_energy_and_gas_production":
            raise ValueError("damage/gas product has an unknown quantity")
        if product["normalization"] != "physical_source_rate":
            raise ValueError(
                "damage/gas product must use physical source normalization"
            )
    if product["kind"] == "activation_ready_metadata":
        if (
            product["units"] != "metadata"
            or product.get("quantity") != "activation_ready_metadata"
            or product["normalization"] != "not_applicable"
        ):
            raise ValueError(
                "activation-ready metadata product semantics are invalid"
            )
    if product["kind"] == "magnet_geometry_interchange":
        if (
            product["units"] != "metadata"
            or product.get("quantity") != "magnet_geometry_interchange"
            or product["normalization"] != "not_applicable"
        ):
            raise ValueError(
                "geometry-interchange product semantics are invalid"
            )


def _validate_bundle_metadata(
    *,
    provenance: Mapping[str, Any],
    geometry: Mapping[str, Any],
    source: Mapping[str, Any],
    nuclear_data: Mapping[str, Any],
    materials: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_provenance = _normalise_provenance(provenance)
    _require_fields(geometry, REQUIRED_GEOMETRY, "geometry manifest")
    for name in REQUIRED_GEOMETRY:
        _require_sha256(geometry[name], f"geometry.{name}")
    _require_fields(source, REQUIRED_SOURCE, "source manifest")
    rate = float(source["physical_source_rate_per_s"])
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(
            "physical_source_rate_per_s must be positive and finite"
        )
    for name in ("source_definition_sha256", "source_mesh_sha256"):
        _require_sha256(source[name], f"source.{name}")
    if (
        source["source_mesh_sha256"]
        != canonical_provenance["source_mesh_sha256"]
    ):
        raise ValueError("source mesh hash disagrees with bundle provenance")
    _require_fields(nuclear_data, {"manifest_sha256"}, "nuclear-data manifest")
    _require_sha256(
        nuclear_data["manifest_sha256"], "nuclear_data.manifest_sha256"
    )
    material_hashes = {
        key: value
        for key, value in materials.items()
        if key
        in {"manifest_sha256", "file_sha256", "resolved_manifest_sha256"}
    }
    if not material_hashes:
        raise ValueError("materials manifest has no hash-bound identity")
    for name, value in material_hashes.items():
        _require_sha256(value, f"materials.{name}")
    return canonical_provenance


def write_radiation_field_bundle(
    output_directory: str | Path,
    *,
    provenance: Mapping[str, Any],
    geometry: Mapping[str, Any],
    source: Mapping[str, Any],
    nuclear_data: Mapping[str, Any],
    materials: Mapping[str, Any],
    magnet_inventory: Sequence[Mapping[str, Any]],
    products: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy neutral products into a hash-bound, dependency-free bundle."""
    canonical_provenance = _validate_bundle_metadata(
        provenance=provenance,
        geometry=geometry,
        source=source,
        nuclear_data=nuclear_data,
        materials=materials,
    )
    if not magnet_inventory:
        raise ValueError("magnet inventory cannot be empty")
    magnet_ids = [str(item.get("magnet_id", "")) for item in magnet_inventory]
    if any(
        not value or Path(value).name != value for value in magnet_ids
    ) or len(magnet_ids) != len(set(magnet_ids)):
        raise ValueError("magnet inventory IDs must be present and unique")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    product_entries = []
    boundary_banks = []
    seen_relative_paths: set[str] = set()
    if not isinstance(products, Sequence) or isinstance(
        products, (str, bytes)
    ):
        raise ValueError("bundle products must be a sequence")
    if any(not isinstance(item, Mapping) for item in products):
        raise ValueError("radiation products must be objects")
    kinds = {str(item.get("kind", "")) for item in products}
    required_products = {"volume_scalar_flux", "boundary_phase_space"}
    missing_products = required_products - kinds
    if missing_products:
        raise ValueError(
            f"bundle is missing required products {sorted(missing_products)}"
        )
    _require_boundary_coverage(products, magnet_ids)
    for item in products:
        product = dict(item)
        _validate_product(product)
        source_path = Path(product.pop("path")).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_sha256 = _sha256(source_path)
        role_directory = PRODUCT_DIRECTORIES[product["kind"]]
        if product["kind"] == "volume_scalar_flux":
            _validate_scalar_flux_hdf5(
                source_path,
                product=product,
                inventory_magnet_ids=set(magnet_ids),
                source=source,
                provenance=canonical_provenance,
                geometry=geometry,
                nuclear_data=nuclear_data,
                materials=materials,
            )
        if product["kind"] == "boundary_phase_space":
            if product["magnet_id"] not in magnet_ids:
                raise ValueError(
                    "boundary product magnet is absent from inventory"
                )
            boundary_banks.append(
                _validate_boundary_hdf5(
                    source_path,
                    product=product,
                    source=source,
                    provenance=canonical_provenance,
                    geometry=geometry,
                )
            )
        if product["kind"] == "reaction_production":
            from .magnet_reaction_production import (
                validate_magnet_reaction_production,
            )

            validate_magnet_reaction_production(source_path)
        if product["kind"] == "damage_and_gas_production":
            _validate_damage_gas_hdf5(
                source_path,
                product=product,
                inventory_magnet_ids=set(magnet_ids),
                source=source,
                provenance=canonical_provenance,
                geometry=geometry,
                nuclear_data=nuclear_data,
            )
        if product["kind"] == "activation_ready_metadata":
            from .activation_ready_metadata import (
                read_activation_ready_metadata,
            )

            read_activation_ready_metadata(source_path)
        if product["kind"] == "magnet_geometry_interchange":
            from .magnet_geometry_interchange import (
                read_magnet_geometry_interchange,
            )

            read_magnet_geometry_interchange(source_path)
        relative = (
            Path(role_directory) / str(product["magnet_id"]) / source_path.name
        )
        if relative.as_posix() in seen_relative_paths:
            raise ValueError(
                f"duplicate bundle product path {relative.as_posix()!r}"
            )
        seen_relative_paths.add(relative.as_posix())
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        target_sha256 = _sha256(target)
        if target_sha256 != source_sha256:
            raise IOError(
                f"bundle copy changed radiation product {source_path}"
            )
        product.update(
            {
                "path": relative.as_posix(),
                "sha256": target_sha256,
                "size_bytes": target.stat().st_size,
            }
        )
        product_entries.append(product)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": canonical_provenance,
        "geometry": dict(geometry),
        "source": dict(source),
        "nuclear_data": dict(nuclear_data),
        "materials": dict(materials),
        "magnet_inventory": [dict(item) for item in magnet_inventory],
        "products": product_entries,
        "bank_completeness": _bank_completeness_summary(boundary_banks),
        "verification": dict(verification),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "bundle_manifest_sha256": _sha256(manifest_path)}


def read_radiation_field_bundle(path: str | Path) -> dict[str, Any]:
    """Validate a bundle without importing OpenMC, DAGMC, MOAB, or CadQuery."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("incompatible magnet radiation-field bundle schema")
    _require_fields(
        manifest,
        {
            "provenance",
            "geometry",
            "source",
            "nuclear_data",
            "materials",
            "magnet_inventory",
            "products",
            "bank_completeness",
            "verification",
        },
        "bundle manifest",
    )
    canonical_provenance = _validate_bundle_metadata(
        provenance=manifest["provenance"],
        geometry=manifest["geometry"],
        source=manifest["source"],
        nuclear_data=manifest["nuclear_data"],
        materials=manifest["materials"],
    )
    magnet_inventory = manifest["magnet_inventory"]
    if not isinstance(magnet_inventory, list) or not magnet_inventory:
        raise ValueError("magnet inventory cannot be empty")
    magnet_ids = [str(item.get("magnet_id", "")) for item in magnet_inventory]
    if any(
        not value or Path(value).name != value for value in magnet_ids
    ) or len(magnet_ids) != len(set(magnet_ids)):
        raise ValueError("magnet inventory IDs must be present and unique")
    products = manifest["products"]
    if not isinstance(products, list):
        raise ValueError("bundle products must be a list")
    kinds = {str(item.get("kind", "")) for item in products}
    missing_products = {"volume_scalar_flux", "boundary_phase_space"} - kinds
    if missing_products:
        raise ValueError(
            f"bundle is missing required products {sorted(missing_products)}"
        )
    _require_boundary_coverage(products, magnet_ids)
    boundary_banks = []
    seen_paths = set()
    for product in products:
        _validate_product(product)
        _require_fields(product, {"sha256", "size_bytes"}, "bundle product")
        _require_sha256(product["sha256"], "bundle product sha256")
        if (
            isinstance(product["size_bytes"], bool)
            or int(product["size_bytes"]) < 0
        ):
            raise ValueError("bundle product size_bytes must be nonnegative")
        relative = Path(str(product["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "bundle product path must remain inside the bundle"
            )
        expected_parent = Path(PRODUCT_DIRECTORIES[product["kind"]]) / str(
            product["magnet_id"]
        )
        if relative.parent != expected_parent:
            raise ValueError(
                "bundle product path disagrees with its kind and magnet route"
            )
        if relative.as_posix() in seen_paths:
            raise ValueError(
                f"duplicate bundle product path {relative.as_posix()!r}"
            )
        seen_paths.add(relative.as_posix())
        product_path = root / relative
        if (
            not product_path.is_file()
            or _sha256(product_path) != product["sha256"]
            or product_path.stat().st_size != int(product["size_bytes"])
        ):
            raise ValueError(
                f"radiation product hash mismatch: {product['path']}"
            )
        if product["kind"] == "volume_scalar_flux":
            _validate_scalar_flux_hdf5(
                product_path,
                product=product,
                inventory_magnet_ids=set(magnet_ids),
                source=manifest["source"],
                provenance=canonical_provenance,
                geometry=manifest["geometry"],
                nuclear_data=manifest["nuclear_data"],
                materials=manifest["materials"],
            )
        if product["kind"] == "boundary_phase_space":
            if product["magnet_id"] not in magnet_ids:
                raise ValueError(
                    "boundary product magnet is absent from inventory"
                )
            boundary_banks.append(
                _validate_boundary_hdf5(
                    product_path,
                    product=product,
                    source=manifest["source"],
                    provenance=canonical_provenance,
                    geometry=manifest["geometry"],
                )
            )
        if product["kind"] == "reaction_production":
            from .magnet_reaction_production import (
                validate_magnet_reaction_production,
            )

            validate_magnet_reaction_production(product_path)
        if product["kind"] == "damage_and_gas_production":
            _validate_damage_gas_hdf5(
                product_path,
                product=product,
                inventory_magnet_ids=set(magnet_ids),
                source=manifest["source"],
                provenance=canonical_provenance,
                geometry=manifest["geometry"],
                nuclear_data=manifest["nuclear_data"],
            )
        if product["kind"] == "activation_ready_metadata":
            from .activation_ready_metadata import (
                read_activation_ready_metadata,
            )

            read_activation_ready_metadata(product_path)
        if product["kind"] == "magnet_geometry_interchange":
            from .magnet_geometry_interchange import (
                read_magnet_geometry_interchange,
            )

            read_magnet_geometry_interchange(product_path)
    expected_completeness = _bank_completeness_summary(boundary_banks)
    if manifest["bank_completeness"] != expected_completeness:
        raise ValueError(
            "bundle bank-completeness summary disagrees with products"
        )
    return manifest
