"""Neutral volume scalar-flux export from OpenMC statepoint tallies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCHEMA = "parastell.magnet_volume_scalar_flux/v1.0.0"
FIELD_SCHEMA = "parastell.magnet_scalar_flux_fields/v1.0.0"


def _text(value) -> str:
    return (
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
    )


def _read_volume_flux_tally(statepoint_path, tally_name):
    with h5py.File(statepoint_path) as statepoint:
        tallies = statepoint["tallies"]
        tally = next(
            (
                value
                for key, value in tallies.items()
                if key.startswith("tally ")
                and _text(value["name"][()]) == tally_name
            ),
            None,
        )
        if tally is None:
            raise ValueError(f"statepoint has no tally named {tally_name!r}")
        filters = [
            tallies[f"filters/filter {int(value)}"]
            for value in tally["filters"][:]
        ]
        types = [_text(value["type"][()]) for value in filters]
        if set(types) != {"cell", "particle", "energy"}:
            raise ValueError(f"{tally_name} is not a cell scalar-flux tally")
        dimensions = tuple(int(value["n_bins"][()]) for value in filters)
        results = tally["results"][:]
        realizations = int(tally["n_realizations"][()])
        if realizations <= 0:
            raise ValueError(f"{tally_name} has no realizations")
        sums = results[..., 0].reshape(dimensions + (-1,))
        squares = results[..., 1].reshape(dimensions + (-1,))
        mean = sums / realizations
        if realizations == 1:
            std_dev = np.zeros_like(mean)
        else:
            variance = np.maximum(
                0.0,
                (squares / realizations - mean**2) / (realizations - 1),
            )
            std_dev = np.sqrt(variance)
        cell_axis = types.index("cell")
        particle_axis = types.index("particle")
        energy_axis = types.index("energy")
        cell_ids = np.asarray(
            filters[cell_axis]["bins"][:], dtype=int
        ).reshape(-1)
        particles = [
            _text(value) for value in filters[particle_axis]["bins"][:]
        ]
        if len(particles) != 1:
            raise ValueError(f"{tally_name} must score exactly one particle")
        energy_edges = np.asarray(filters[energy_axis]["bins"][:], dtype=float)
        output_mean = np.zeros((len(cell_ids), len(energy_edges) - 1))
        output_std = np.zeros_like(output_mean)
        for cell_index in range(len(cell_ids)):
            for energy_index in range(len(energy_edges) - 1):
                selection = [0] * len(dimensions)
                selection[cell_axis] = cell_index
                selection[particle_axis] = 0
                selection[energy_axis] = energy_index
                output_mean[cell_index, energy_index] = float(
                    np.sum(mean[tuple(selection)])
                )
                output_std[cell_index, energy_index] = float(
                    np.sqrt(np.sum(std_dev[tuple(selection)] ** 2))
                )
    return {
        "particle": particles[0],
        "cell_ids": cell_ids,
        "energy_edges_eV": energy_edges,
        "track_length_mean_cm_per_source": output_mean,
        "track_length_std_dev_cm_per_source": output_std,
        "realizations": realizations,
    }


def _read_mesh_flux_tally(statepoint_path, tally_name):
    """Read one component-filtered mesh flux tally in OpenMC filter-bin order."""
    with h5py.File(statepoint_path) as statepoint:
        tallies = statepoint["tallies"]
        tally = next(
            (
                value
                for key, value in tallies.items()
                if key.startswith("tally ")
                and _text(value["name"][()]) == tally_name
            ),
            None,
        )
        if tally is None:
            raise ValueError(f"statepoint has no tally named {tally_name!r}")
        filters = [
            tallies[f"filters/filter {int(value)}"]
            for value in tally["filters"][:]
        ]
        types = [_text(value["type"][()]) for value in filters]
        if set(types) != {"cell", "mesh", "particle", "energy"}:
            raise ValueError(
                f"{tally_name} is not a component-filtered mesh scalar-flux tally"
            )
        dimensions = tuple(int(value["n_bins"][()]) for value in filters)
        realizations = int(tally["n_realizations"][()])
        if realizations <= 0:
            raise ValueError(f"{tally_name} has no realizations")
        results = tally["results"][:]
        sums = results[..., 0].reshape(dimensions + (-1,))
        squares = results[..., 1].reshape(dimensions + (-1,))
        mean = sums / realizations
        if realizations == 1:
            std_dev = np.zeros_like(mean)
        else:
            variance = np.maximum(
                0.0,
                (squares / realizations - mean**2) / (realizations - 1),
            )
            std_dev = np.sqrt(variance)
        mesh_axis = types.index("mesh")
        cell_axis = types.index("cell")
        particle_axis = types.index("particle")
        energy_axis = types.index("energy")
        particles = [
            _text(value) for value in filters[particle_axis]["bins"][:]
        ]
        if len(particles) != 1:
            raise ValueError(f"{tally_name} must score exactly one particle")
        cell_ids = [int(value) for value in filters[cell_axis]["bins"][:]]
        if len(cell_ids) != 1:
            raise ValueError(
                f"{tally_name} must filter exactly one component cell"
            )
        energy_edges = np.asarray(filters[energy_axis]["bins"][:], dtype=float)

        def arrange(values):
            ordered = np.moveaxis(
                values,
                (mesh_axis, energy_axis),
                (0, 1),
            )
            if any(size != 1 for size in ordered.shape[2:]):
                raise ValueError(
                    f"{tally_name} contains unsupported non-singleton filters"
                )
            return ordered.reshape(
                dimensions[mesh_axis], dimensions[energy_axis]
            )

        output_mean = arrange(mean)
        output_std = arrange(std_dev)
        mesh_filter = filters[mesh_axis]
        mesh_id = int(mesh_filter["bins"][()])
    return {
        "particle": particles[0],
        "cell_id": cell_ids[0],
        "mesh_id": mesh_id,
        "energy_edges_eV": energy_edges,
        "track_length_mean_cm_per_source": output_mean,
        "track_length_std_dev_cm_per_source": output_std,
        "realizations": realizations,
    }


def _precision_status(
    mean: np.ndarray,
    std_dev: np.ndarray,
    *,
    particle: str,
    realizations: int,
    minimum_realizations: int,
) -> np.ndarray:
    """Encode EMPTY/UNDER_RESOLVED/QUALIFIED without fabricating event ESS."""
    status = np.ones(mean.shape, dtype=np.uint8)
    status[(mean == 0.0) & (std_dev == 0.0)] = 0
    maximum_relative = 0.1 if particle == "neutron" else 0.2
    relative = np.divide(
        std_dev,
        mean,
        out=np.full_like(std_dev, np.inf),
        where=mean > 0.0,
    )
    if realizations >= minimum_realizations:
        status[(mean > 0.0) & (relative <= maximum_relative)] = 2
    return status


def build_scalar_flux_fields_from_statepoint(
    statepoint_path: str | Path,
    *,
    associations_path: str | Path,
    local_mesh_manifest_path: str | Path,
    material_intersection_volumes_path: str | Path,
    whole_tally_structures: Mapping[str, str] | None = None,
    local_energy_structures: Mapping[str, str] | None = None,
    openmc_cell_ids_by_dagmc_volume: Mapping[int, int] | None = None,
    minimum_realizations: int = 10,
) -> list[dict[str, Any]]:
    """Build whole-component and intersection-normalized local field payloads."""
    from .coil_frame import parallel_transport_frame
    from .magnet_local_mesh import LocalMeshDefinition
    from .material_intersection_volumes import (
        read_material_intersection_volume_manifest,
    )

    if minimum_realizations <= 1:
        raise ValueError("minimum_realizations must be greater than one")
    association = json.loads(
        Path(associations_path).read_text(encoding="utf-8")
    )
    local_collection = json.loads(
        Path(local_mesh_manifest_path).read_text(encoding="utf-8")
    )
    intersections = read_material_intersection_volume_manifest(
        material_intersection_volumes_path,
        local_mesh_manifest=local_collection,
        associations=association,
        require_qualified=True,
    )
    pairs = association["inventory"]["magnet_pairs"]
    explicit_cell_map = {
        int(key): int(value)
        for key, value in dict(openmc_cell_ids_by_dagmc_volume or {}).items()
    }
    component_by_cell: dict[int, dict[str, Any]] = {}
    pair_by_winding_cell: dict[int, Mapping[str, Any]] = {}
    for pair in pairs:
        for key in ("winding_pack", "casing"):
            component = pair[key]
            if component is None:
                continue
            dagmc_volume_id = int(component["volume_id"])
            cell_id = explicit_cell_map.get(dagmc_volume_id, dagmc_volume_id)
            if cell_id in component_by_cell:
                raise ValueError(f"duplicate magnet component cell {cell_id}")
            component_by_cell[cell_id] = {
                "pair": pair,
                "component": component,
                "component_role": str(component["component_role"]),
                "dagmc_volume_id": dagmc_volume_id,
                "openmc_cell_id": cell_id,
            }
        winding_volume_id = int(pair["winding_pack_volume_id"])
        pair_by_winding_cell[
            explicit_cell_map.get(winding_volume_id, winding_volume_id)
        ] = pair
    frames = {
        pair["magnet_id"]: parallel_transport_frame(
            association["centreline_points_by_coil"][pair["coil_id"]]
        )
        for pair in pairs
    }
    whole_structures = dict(
        whole_tally_structures
        or {
            "pstl_magnet_neutron_configured_fine_volume_flux": "configured_neutron",
            "pstl_magnet_neutron_ccfe_709_volume_flux": "CCFE-709",
            "pstl_magnet_neutron_ukaea_1102_volume_flux": "UKAEA-1102",
            "pstl_magnet_photon_configured_volume_flux": "configured_photon",
        }
    )
    local_structures = {
        "neutron": "configured_neutron",
        "photon": "configured_photon",
        **dict(local_energy_structures or {}),
    }
    fields: list[dict[str, Any]] = []
    for tally_name, structure in whole_structures.items():
        tally = _read_volume_flux_tally(statepoint_path, tally_name)
        try:
            selected = [
                component_by_cell[int(cell)] for cell in tally["cell_ids"]
            ]
        except KeyError as exc:
            raise ValueError(
                f"whole-volume scalar tally references unknown magnet cell {exc.args[0]}"
            ) from exc
        selected_pairs = [item["pair"] for item in selected]
        selected_components = [item["component"] for item in selected]
        centroids = np.asarray(
            [
                component["centroid_global_cm"]
                for component in selected_components
            ],
            dtype=float,
        )
        centreline_samples = [
            frames[pair["magnet_id"]].sample(centroid)
            for pair, centroid in zip(selected_pairs, centroids)
        ]
        name = tally_name.removeprefix("pstl_magnet_").removesuffix(
            "_volume_flux"
        )
        fields.append(
            {
                "name": name,
                "field_kind": "whole_volume",
                "particle": tally["particle"],
                "energy_edges_eV": tally["energy_edges_eV"],
                "energy_structure": structure,
                "track_length_mean_cm_per_source": tally[
                    "track_length_mean_cm_per_source"
                ],
                "track_length_std_dev_cm_per_source": tally[
                    "track_length_std_dev_cm_per_source"
                ],
                "volume_cm3": [
                    component["volume_cm3"]
                    for component in selected_components
                ],
                "region_ids": [
                    component["component_id"]
                    for component in selected_components
                ],
                "magnet_ids": [pair["magnet_id"] for pair in selected_pairs],
                "component_roles": [
                    item["component_role"] for item in selected
                ],
                "dagmc_volume_ids": np.asarray(
                    [item["dagmc_volume_id"] for item in selected],
                    dtype=np.int64,
                ),
                "openmc_cell_ids": np.asarray(
                    tally["cell_ids"], dtype=np.int64
                ),
                "global_centroid_cm": centroids,
                "nearest_centreline_global_cm": np.asarray(
                    [
                        sample["nearest_centreline_global_cm"]
                        for sample in centreline_samples
                    ],
                    dtype=float,
                ),
                "centreline_arclength_cm": np.asarray(
                    [
                        sample["centreline_arclength_cm"]
                        for sample in centreline_samples
                    ],
                    dtype=float,
                ),
                "normalized_arclength": np.asarray(
                    [
                        sample["normalized_arclength"]
                        for sample in centreline_samples
                    ],
                    dtype=float,
                ),
                "centreline_tangent": np.asarray(
                    [
                        sample["centreline_tangent"]
                        for sample in centreline_samples
                    ],
                    dtype=float,
                ),
                "centreline_radial": np.asarray(
                    [
                        sample["centreline_radial"]
                        for sample in centreline_samples
                    ],
                    dtype=float,
                ),
                "centreline_transverse": np.asarray(
                    [
                        sample["centreline_transverse"]
                        for sample in centreline_samples
                    ],
                    dtype=float,
                ),
                "local_centreline_coordinates_cm": np.asarray(
                    [
                        sample["local_centreline_coordinates_cm"]
                        for sample in centreline_samples
                    ],
                    dtype=float,
                ),
                "distance_to_centreline_cm": np.asarray(
                    [
                        sample["distance_to_centreline_cm"]
                        for sample in centreline_samples
                    ],
                    dtype=float,
                ),
                "centreline_linkage_status": np.full(
                    len(centreline_samples),
                    "LINKED_NEAREST_CENTRELINE_SEGMENT",
                    dtype=object,
                ),
                "frame_type": np.asarray(
                    [sample["frame_type"] for sample in centreline_samples],
                    dtype=object,
                ),
                "frame_quality_status": np.asarray(
                    [
                        sample["frame_quality_status"]
                        for sample in centreline_samples
                    ],
                    dtype=object,
                ),
                "material_ids": [
                    component["material"] for component in selected_components
                ],
                "material_semantics": (
                    "exact homogeneous DAGMC component cell with resolved "
                    "casing or winding-pack material tag"
                ),
                "estimator_scope": "track_length_inside_selected_DAGMC_cell",
                "volume_basis": "exact_DAGMC_component_volume",
                "event_effective_sample_size_available": False,
                "realizations": tally["realizations"],
                "resolution_status": _precision_status(
                    tally["track_length_mean_cm_per_source"],
                    tally["track_length_std_dev_cm_per_source"],
                    particle=tally["particle"],
                    realizations=tally["realizations"],
                    minimum_realizations=minimum_realizations,
                ),
            }
        )
    intersection_magnet_ids = set(intersections["meshes"])
    for cell_id, pair in sorted(pair_by_winding_cell.items()):
        magnet_id = pair["magnet_id"]
        if magnet_id not in intersection_magnet_ids:
            continue
        value = local_collection["meshes"][magnet_id]
        mesh = LocalMeshDefinition(
            magnet_id=value["magnet_id"],
            lower_left_local_cm=tuple(value["lower_left_local_cm"]),
            upper_right_local_cm=tuple(value["upper_right_local_cm"]),
            dimension=tuple(value["dimension"]),
            rotation_local_to_global=tuple(
                tuple(row) for row in value["rotation_local_to_global"]
            ),
            translation_global_cm=tuple(value["translation_global_cm"]),
            requested_resolution_cm=float(value["requested_resolution_cm"]),
            frame_type=value.get(
                "frame_type", "coil_centerline_parallel_transport"
            ),
        )
        metadata = mesh.bin_metadata(frames[magnet_id])
        for particle in ("neutron", "photon"):
            tally_name = f"pstl_magnet_{cell_id}_{particle}_local_mesh_flux"
            tally = _read_mesh_flux_tally(statepoint_path, tally_name)
            if tally["particle"] != particle:
                raise ValueError(f"{tally_name} particle identity mismatch")
            if tally["cell_id"] != cell_id:
                raise ValueError(
                    f"{tally_name} component cell identity mismatch"
                )
            if len(metadata["mesh_bin_ids"]) != len(
                tally["track_length_mean_cm_per_source"]
            ):
                raise ValueError(
                    f"{tally_name} does not match the local mesh manifest"
                )
            volume_row = intersections["meshes"][magnet_id]
            if int(volume_row["openmc_cell_id"]) != cell_id:
                raise ValueError(
                    f"{magnet_id} intersection volumes use another component cell"
                )
            if volume_row["material_tag"] != pair["winding_pack"]["material"]:
                raise ValueError(
                    f"{magnet_id} intersection material identity mismatch"
                )
            intersection_volumes = np.asarray(
                volume_row["intersection_volume_cm3"], dtype=float
            )
            intersection_std = np.asarray(
                volume_row["standard_deviation_cm3"], dtype=float
            )
            if intersection_volumes.shape != (mesh.bin_count,):
                raise ValueError(
                    f"{magnet_id} intersection volumes do not align with mesh bins"
                )
            selected_bins = intersection_volumes > 0.0
            if not np.any(selected_bins):
                raise ValueError(
                    f"{magnet_id} has no positive winding-pack mesh intersections"
                )

            def selected(name):
                return np.asarray(metadata[name])[selected_bins]

            geometric_volume = np.asarray(metadata["volume_cm3"], dtype=float)
            fields.append(
                {
                    "name": f"{magnet_id}_{particle}_local_mesh",
                    "field_kind": "local_mesh",
                    "particle": particle,
                    "energy_edges_eV": tally["energy_edges_eV"],
                    "energy_structure": local_structures[particle],
                    "track_length_mean_cm_per_source": tally[
                        "track_length_mean_cm_per_source"
                    ][selected_bins],
                    "track_length_std_dev_cm_per_source": tally[
                        "track_length_std_dev_cm_per_source"
                    ][selected_bins],
                    "volume_cm3": intersection_volumes[selected_bins],
                    "geometric_bin_volume_cm3": geometric_volume[
                        selected_bins
                    ],
                    "intersection_volume_std_dev_cm3": intersection_std[
                        selected_bins
                    ],
                    "material_volume_fraction": (
                        intersection_volumes[selected_bins]
                        / geometric_volume[selected_bins]
                    ),
                    "local_mesh_source_bin_index": np.flatnonzero(
                        selected_bins
                    ),
                    "region_ids": selected("mesh_bin_ids"),
                    "magnet_ids": np.full(
                        int(np.count_nonzero(selected_bins)),
                        magnet_id,
                        dtype=object,
                    ),
                    "global_centroid_cm": selected("global_centroid_cm"),
                    "mesh_local_centroid_cm": selected(
                        "mesh_local_centroid_cm"
                    ),
                    "nearest_centreline_global_cm": selected(
                        "nearest_centreline_global_cm"
                    ),
                    "centreline_arclength_cm": selected(
                        "centreline_arclength_cm"
                    ),
                    "normalized_arclength": selected("normalized_arclength"),
                    "centreline_tangent": selected("centreline_tangent"),
                    "centreline_radial": selected("centreline_radial"),
                    "centreline_transverse": selected("centreline_transverse"),
                    "local_centreline_coordinates_cm": selected(
                        "local_centreline_coordinates_cm"
                    ),
                    "distance_to_centreline_cm": selected(
                        "distance_to_centreline_cm"
                    ),
                    "centreline_linkage_status": selected(
                        "centreline_linkage_status"
                    ),
                    "frame_type": selected("frame_type"),
                    "frame_quality_status": selected("frame_quality_status"),
                    "material_ids": np.full(
                        int(np.count_nonzero(selected_bins)),
                        volume_row["material_tag"],
                        dtype=object,
                    ),
                    "component_roles": np.full(
                        int(np.count_nonzero(selected_bins)),
                        "winding_pack",
                        dtype=object,
                    ),
                    "dagmc_volume_ids": np.full(
                        int(np.count_nonzero(selected_bins)),
                        int(volume_row["dagmc_volume_id"]),
                        dtype=np.int64,
                    ),
                    "openmc_cell_ids": np.full(
                        int(np.count_nonzero(selected_bins)),
                        cell_id,
                        dtype=np.int64,
                    ),
                    "material_semantics": (
                        "track length and normalization are restricted to the "
                        "selected homogeneous winding-pack component"
                    ),
                    "estimator_scope": (
                        "track_length_inside_selected_DAGMC_cell_and_rotated_mesh_bin"
                    ),
                    "volume_basis": "qualified_component_intersection_volume",
                    "material_intersection_volume_artifact_sha256": intersections[
                        "artifact_sha256"
                    ],
                    "event_effective_sample_size_available": False,
                    "realizations": tally["realizations"],
                    "resolution_status": _precision_status(
                        tally["track_length_mean_cm_per_source"][
                            selected_bins
                        ],
                        tally["track_length_std_dev_cm_per_source"][
                            selected_bins
                        ],
                        particle=particle,
                        realizations=tally["realizations"],
                        minimum_realizations=minimum_realizations,
                    ),
                }
            )
    return fields


def export_volume_scalar_flux(
    output_path: str | Path,
    *,
    statepoint_path: str | Path,
    tally_names: Sequence[str],
    cell_volumes_cm3: Mapping[int, float],
    physical_source_rate_per_s: float | None,
    magnet_ids_by_cell: Mapping[int, str] | None = None,
) -> dict:
    """Export volume-normalized track-length flux without solver-specific syntax."""
    if not tally_names:
        raise ValueError("at least one volume scalar-flux tally is required")
    volumes = {
        int(key): float(value) for key, value in cell_volumes_cm3.items()
    }
    if any(
        not np.isfinite(value) or value <= 0.0 for value in volumes.values()
    ):
        raise ValueError("cell volumes must be positive and finite")
    if (
        physical_source_rate_per_s is not None
        and physical_source_rate_per_s <= 0.0
    ):
        raise ValueError("physical source rate must be positive")
    axes = []
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype("utf-8")
    with h5py.File(output, "w") as target:
        target.attrs["schema"] = SCHEMA
        groups = target.create_group("volume_scalar_flux")
        for tally_name in tally_names:
            tally = _read_volume_flux_tally(statepoint_path, tally_name)
            cell_ids = tally["cell_ids"]
            missing = set(cell_ids.tolist()) - set(volumes)
            if missing:
                raise ValueError(
                    f"missing DAGMC volumes for cells {sorted(missing)}"
                )
            cell_volume = np.asarray(
                [volumes[int(value)] for value in cell_ids]
            )
            mean = (
                tally["track_length_mean_cm_per_source"] / cell_volume[:, None]
            )
            std_dev = (
                tally["track_length_std_dev_cm_per_source"]
                / cell_volume[:, None]
            )
            relative_error = np.divide(
                std_dev,
                mean,
                out=np.where(std_dev == 0.0, 0.0, np.inf),
                where=mean != 0.0,
            )
            name = tally_name.removeprefix("pstl_magnet_").removesuffix(
                "_volume_flux"
            )
            group = groups.create_group(name)
            group.attrs["particle"] = tally["particle"]
            group.attrs["quantity"] = "volume_scalar_flux"
            group.attrs["per_source_units"] = "1/cm2/source"
            group.create_dataset("cell_ids", data=cell_ids)
            group.create_dataset("volume_cm3", data=cell_volume)
            group.create_dataset(
                "energy_edges_eV", data=tally["energy_edges_eV"]
            )
            group.create_dataset(
                "magnet_ids",
                data=np.asarray(
                    [
                        (magnet_ids_by_cell or {}).get(
                            int(value), f"cell-{int(value)}"
                        )
                        for value in cell_ids
                    ],
                    dtype=object,
                ),
                dtype=strings,
            )
            group.create_dataset(
                "track_length_mean_cm_per_source",
                data=tally["track_length_mean_cm_per_source"],
            )
            group.create_dataset(
                "track_length_std_dev_cm_per_source",
                data=tally["track_length_std_dev_cm_per_source"],
            )
            group.create_dataset("mean_per_source", data=mean)
            group.create_dataset("std_dev_per_source", data=std_dev)
            group.create_dataset("relative_error", data=relative_error)
            if physical_source_rate_per_s is not None:
                group.attrs["physical_units"] = "particles/cm2/s"
                group.create_dataset(
                    "mean_physical", data=mean * physical_source_rate_per_s
                )
                group.create_dataset(
                    "std_dev_physical",
                    data=std_dev * physical_source_rate_per_s,
                )
            axes.append(
                {
                    "name": name,
                    "particle": tally["particle"],
                    "groups": len(tally["energy_edges_eV"]) - 1,
                    "cell_ids": cell_ids.tolist(),
                }
            )
        manifest = {
            "schema": SCHEMA,
            "created_utc": datetime.now(UTC).isoformat(),
            "quantity": "volume_scalar_flux",
            "estimator": "OpenMC track-length flux divided by explicit cell volume",
            "normalization": {
                "per_source_units": "1/cm2/source",
                "physical_units": (
                    "particles/cm2/s"
                    if physical_source_rate_per_s is not None
                    else None
                ),
                "physical_source_rate_per_s": physical_source_rate_per_s,
            },
            "energy_axes": axes,
        }
        target.create_dataset(
            "manifest_json",
            data=json.dumps(manifest, sort_keys=True),
            dtype=strings,
        )
    return manifest


def qualify_flux_statistics(
    *,
    raw_count: int,
    sum_weights: float,
    sum_squared_weights: float,
    mean: float,
    standard_error: float,
    qualified_ess: float = 25.0,
    marginal_ess: float = 4.0,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Assign an explicit resolution status; an empty bin is not a physical zero."""
    if (
        raw_count < 0
        or min(sum_weights, sum_squared_weights, mean, standard_error) < 0.0
    ):
        raise ValueError("flux statistics must be nonnegative")
    ess = (
        sum_weights * sum_weights / sum_squared_weights
        if sum_squared_weights > 0.0
        else 0.0
    )
    if raw_count == 0:
        status = "EMPTY"
    elif ess >= qualified_ess:
        status = "QUALIFIED"
    elif ess >= marginal_ess:
        status = "MARGINAL"
    else:
        status = "INSUFFICIENT_STATISTICS"
    # Poisson zero-count upper bound, expressed in expected weighted events.
    upper = -np.log(1.0 - confidence_level) if raw_count == 0 else None
    return {
        "raw_count": int(raw_count),
        "sum_weights": float(sum_weights),
        "sum_squared_weights": float(sum_squared_weights),
        "effective_sample_size": float(ess),
        "mean": float(mean),
        "standard_error": float(standard_error),
        "relative_uncertainty": (
            float(standard_error / mean) if mean > 0.0 else None
        ),
        "status": status,
        "empty_bin_is_physical_zero": False,
        "zero_count_upper_bound_expected_events": upper,
        "confidence_level": confidence_level if upper is not None else None,
    }


def _hash_edges(edges: np.ndarray) -> str:
    payload = json.dumps(
        [float(value) for value in edges], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def export_scalar_flux_fields(
    output_path: str | Path,
    *,
    fields: Sequence[Mapping[str, Any]],
    physical_source_rate_per_s: float,
    provenance: Mapping[str, Any],
    material_manifest_sha256: str,
) -> dict[str, Any]:
    """Write whole-volume and local-mesh scalar flux with exact normalization."""
    if not fields:
        raise ValueError("at least one scalar-flux field is required")
    if (
        not np.isfinite(physical_source_rate_per_s)
        or physical_source_rate_per_s <= 0.0
    ):
        raise ValueError("physical source rate must be positive and finite")
    required_provenance = {
        "raw_h5m_sha256",
        "canonical_geometry_fingerprint",
        "source_definition_sha256",
        "source_mesh_sha256",
        "nuclear_data_manifest_sha256",
    }
    missing_provenance = required_provenance - set(provenance)
    if missing_provenance:
        raise ValueError(
            f"scalar-flux provenance is missing {sorted(missing_provenance)}"
        )
    entries = []
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype("utf-8")
    with h5py.File(output, "w") as handle:
        handle.attrs["schema"] = FIELD_SCHEMA
        handle.attrs["quantity"] = "volume_scalar_flux"
        root = handle.create_group("scalar_flux_fields")
        for field in fields:
            required = {
                "name",
                "field_kind",
                "particle",
                "energy_edges_eV",
                "track_length_mean_cm_per_source",
                "track_length_std_dev_cm_per_source",
                "volume_cm3",
                "region_ids",
                "magnet_ids",
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
                "material_ids",
            }
            missing = required - set(field)
            if missing:
                raise ValueError(
                    f"scalar-flux field is missing {sorted(missing)}"
                )
            if field["field_kind"] not in {"whole_volume", "local_mesh"}:
                raise ValueError(
                    "field_kind must be whole_volume or local_mesh"
                )
            particle = str(field["particle"])
            if particle not in {"neutron", "photon"}:
                raise ValueError(
                    "scalar-flux particle must be neutron or photon"
                )
            edges = np.asarray(field["energy_edges_eV"], dtype=float)
            if (
                edges.ndim != 1
                or len(edges) < 2
                or not np.all(np.isfinite(edges))
                or np.any(np.diff(edges) <= 0.0)
            ):
                raise ValueError(
                    "energy edges must be finite and strictly increasing"
                )
            volumes = np.asarray(field["volume_cm3"], dtype=float)
            mean_track = np.asarray(
                field["track_length_mean_cm_per_source"], dtype=float
            )
            std_track = np.asarray(
                field["track_length_std_dev_cm_per_source"], dtype=float
            )
            expected = (len(volumes), len(edges) - 1)
            if mean_track.shape != expected or std_track.shape != expected:
                raise ValueError(
                    "track-length arrays do not align with regions and energy"
                )
            if (
                np.any(~np.isfinite(volumes))
                or np.any(volumes <= 0.0)
                or np.any(~np.isfinite(mean_track))
                or np.any(~np.isfinite(std_track))
                or np.any(mean_track < 0.0)
                or np.any(std_track < 0.0)
            ):
                raise ValueError(
                    "scalar-flux inputs must be finite and nonnegative"
                )
            flux = mean_track / volumes[:, None]
            flux_std = std_track / volumes[:, None]
            physical = flux * physical_source_rate_per_s
            physical_std = flux_std * physical_source_rate_per_s
            relative = np.divide(
                flux_std,
                flux,
                out=np.full_like(flux, np.inf),
                where=flux > 0.0,
            )
            safe_name = "".join(
                value if value.isalnum() or value in "-_" else "_"
                for value in str(field["name"])
            )
            if safe_name in root:
                raise ValueError(f"duplicate scalar-flux field {safe_name!r}")
            group = root.create_group(safe_name)
            group.attrs["particle"] = particle
            group.attrs["field_kind"] = field["field_kind"]
            group.attrs["quantity"] = "volume_scalar_flux"
            group.attrs["per_source_units"] = "1/cm2/source"
            group.attrs["physical_units"] = "particles/cm2/s"
            group.attrs["energy_values"] = "group_integrated"
            group.attrs["material_semantics"] = str(
                field.get("material_semantics", "region_material_identity")
            )
            group.attrs["estimator_scope"] = str(
                field.get("estimator_scope", "region_track_length")
            )
            group.attrs["volume_basis"] = str(
                field.get("volume_basis", "explicit_region_volume")
            )
            if "material_intersection_volume_artifact_sha256" in field:
                group.attrs["material_intersection_volume_artifact_sha256"] = (
                    str(field["material_intersection_volume_artifact_sha256"])
                )
            group.attrs["event_effective_sample_size_available"] = bool(
                field.get("event_effective_sample_size_available", False)
            )
            group.attrs["centreline_linkage_available"] = True
            group.attrs["centreline_linkage_contract"] = (
                "nearest point on continuous coil centreline with parallel-transport frame"
            )
            group.attrs["statistical_basis"] = (
                "OpenMC batch-mean standard deviation; event-level ESS is unavailable"
            )
            if "realizations" in field:
                group.attrs["realizations"] = int(field["realizations"])
            for name, value in (
                ("energy_edges_eV", edges),
                ("volume_cm3", volumes),
                ("track_length_mean_cm_per_source", mean_track),
                ("track_length_std_dev_cm_per_source", std_track),
                ("mean_per_source", flux),
                ("std_dev_per_source", flux_std),
                ("mean_physical", physical),
                ("std_dev_physical", physical_std),
                ("relative_uncertainty", relative),
                (
                    "global_centroid_cm",
                    np.asarray(field["global_centroid_cm"], dtype=float),
                ),
                (
                    "local_centreline_coordinates_cm",
                    np.asarray(
                        field["local_centreline_coordinates_cm"], dtype=float
                    ),
                ),
                (
                    "nearest_centreline_global_cm",
                    np.asarray(
                        field["nearest_centreline_global_cm"], dtype=float
                    ),
                ),
                (
                    "centreline_arclength_cm",
                    np.asarray(field["centreline_arclength_cm"], dtype=float),
                ),
                (
                    "normalized_arclength",
                    np.asarray(field["normalized_arclength"], dtype=float),
                ),
                (
                    "centreline_tangent",
                    np.asarray(field["centreline_tangent"], dtype=float),
                ),
                (
                    "centreline_radial",
                    np.asarray(field["centreline_radial"], dtype=float),
                ),
                (
                    "centreline_transverse",
                    np.asarray(field["centreline_transverse"], dtype=float),
                ),
                (
                    "distance_to_centreline_cm",
                    np.asarray(
                        field["distance_to_centreline_cm"], dtype=float
                    ),
                ),
            ):
                group.create_dataset(name, data=value)
            if "mesh_local_centroid_cm" in field:
                group.create_dataset(
                    "mesh_local_centroid_cm",
                    data=np.asarray(
                        field["mesh_local_centroid_cm"], dtype=float
                    ),
                )
            optional_region_fields = {
                "geometric_bin_volume_cm3": float,
                "intersection_volume_std_dev_cm3": float,
                "material_volume_fraction": float,
                "local_mesh_source_bin_index": np.int64,
            }
            for name, dtype in optional_region_fields.items():
                if name not in field:
                    continue
                values = np.asarray(field[name], dtype=dtype)
                if values.shape != (len(volumes),) or np.any(
                    ~np.isfinite(values)
                ):
                    raise ValueError(
                        f"{name} must contain finite values aligned with regions"
                    )
                if np.any(values < 0.0):
                    raise ValueError(f"{name} must be nonnegative")
                group.create_dataset(name, data=values)
            if "material_volume_fraction" in group and np.any(
                group["material_volume_fraction"][...] > 1.0 + 1.0e-10
            ):
                raise ValueError("material volume fractions cannot exceed one")
            for name in (
                "region_ids",
                "magnet_ids",
                "material_ids",
                "centreline_linkage_status",
                "frame_type",
                "frame_quality_status",
            ):
                values = np.asarray(field[name], dtype=object)
                if values.shape != (len(volumes),):
                    raise ValueError(
                        f"{name} must align with scalar-flux regions"
                    )
                group.create_dataset(name, data=values, dtype=strings)
            if "component_roles" in field:
                roles = np.asarray(field["component_roles"], dtype=object)
                if roles.shape != (len(volumes),) or any(
                    not str(value) for value in roles
                ):
                    raise ValueError(
                        "component_roles must align with scalar-flux regions"
                    )
                group.create_dataset(
                    "component_roles", data=roles, dtype=strings
                )
            identity_fields = {
                name: np.asarray(field[name], dtype=np.int64)
                for name in ("dagmc_volume_ids", "openmc_cell_ids")
                if name in field
            }
            if identity_fields and set(identity_fields) != {
                "dagmc_volume_ids",
                "openmc_cell_ids",
            }:
                raise ValueError(
                    "DAGMC volume and OpenMC cell IDs must be supplied together"
                )
            for name, values in identity_fields.items():
                if values.shape != (len(volumes),) or np.any(values <= 0):
                    raise ValueError(
                        f"{name} must contain positive IDs aligned with regions"
                    )
                group.create_dataset(name, data=values)
            if "resolution_status" in field:
                status = np.asarray(field["resolution_status"], dtype=np.uint8)
                if status.shape != expected:
                    raise ValueError(
                        "resolution_status must align with regions and energy"
                    )
                if np.any(status > 2):
                    raise ValueError(
                        "resolution_status contains an unknown code"
                    )
                group.create_dataset("resolution_status", data=status)
            if group["global_centroid_cm"].shape != (len(volumes), 3) or group[
                "local_centreline_coordinates_cm"
            ].shape != (len(volumes), 3):
                raise ValueError(
                    "global and local centroids must have shape (regions, 3)"
                )
            vector_metadata = (
                "nearest_centreline_global_cm",
                "centreline_tangent",
                "centreline_radial",
                "centreline_transverse",
            )
            scalar_metadata = (
                "centreline_arclength_cm",
                "normalized_arclength",
                "distance_to_centreline_cm",
            )
            if any(
                group[name].shape != (len(volumes), 3)
                or np.any(~np.isfinite(group[name][...]))
                for name in vector_metadata
            ) or any(
                group[name].shape != (len(volumes),)
                or np.any(~np.isfinite(group[name][...]))
                for name in scalar_metadata
            ):
                raise ValueError(
                    "centreline metadata does not align with regions"
                )
            if np.any(group["normalized_arclength"][...] < 0.0) or np.any(
                group["normalized_arclength"][...] > 1.0
            ):
                raise ValueError(
                    "normalized centreline arclength is outside [0, 1]"
                )
            linkage = group["centreline_linkage_status"].asstr()[...]
            if np.any(linkage != "LINKED_NEAREST_CENTRELINE_SEGMENT"):
                raise ValueError(
                    "scalar-flux region has unavailable centreline linkage"
                )
            if "mesh_local_centroid_cm" in group and group[
                "mesh_local_centroid_cm"
            ].shape != (len(volumes), 3):
                raise ValueError(
                    "mesh-local centroids must have shape (regions, 3)"
                )
            entries.append(
                {
                    "name": safe_name,
                    "field_kind": field["field_kind"],
                    "particle": particle,
                    "regions": len(volumes),
                    "energy_groups": len(edges) - 1,
                    "energy_edges_sha256": _hash_edges(edges),
                    "energy_structure": field.get(
                        "energy_structure", "user_supplied"
                    ),
                    "volume_basis": group.attrs["volume_basis"],
                    "material_semantics": group.attrs["material_semantics"],
                    "event_effective_sample_size_available": bool(
                        group.attrs["event_effective_sample_size_available"]
                    ),
                    "centreline_linkage_available": True,
                    "centreline_metadata_fields": [
                        "nearest_centreline_global_cm",
                        "centreline_arclength_cm",
                        "normalized_arclength",
                        "centreline_tangent",
                        "centreline_radial",
                        "centreline_transverse",
                        "local_centreline_coordinates_cm",
                        "distance_to_centreline_cm",
                        "frame_type",
                        "frame_quality_status",
                    ],
                    "component_identity_fields": [
                        name
                        for name in (
                            "component_roles",
                            "dagmc_volume_ids",
                            "openmc_cell_ids",
                        )
                        if name in group
                    ],
                    "material_intersection_volume_artifact_sha256": (
                        group.attrs.get(
                            "material_intersection_volume_artifact_sha256"
                        )
                    ),
                }
            )
        manifest = {
            "schema": FIELD_SCHEMA,
            "created_utc": datetime.now(UTC).isoformat(),
            "quantity": "volume_scalar_flux",
            "estimator": "OpenMC track-length flux divided by explicit region volume",
            "formula": "phi_physical = track_length_per_source / volume * source_rate",
            "resolution_status_codes": {
                "0": "EMPTY_NOT_A_PHYSICAL_ZERO",
                "1": "UNDER_RESOLVED",
                "2": "QUALIFIED_BATCH_PRECISION",
            },
            "effective_sample_size": {
                "event_level_available": False,
                "reason": "OpenMC statepoints expose batch moments, not event weights",
            },
            "normalization": {
                "per_source_units": "1/cm2/source",
                "physical_units": "particles/cm2/s",
                "physical_source_rate_per_s": physical_source_rate_per_s,
            },
            "material_manifest_sha256": material_manifest_sha256,
            "provenance": dict(provenance),
            "fields": entries,
        }
        handle.create_dataset(
            "manifest_json",
            data=json.dumps(manifest, sort_keys=True),
            dtype=strings,
        )
    return manifest


def validate_spectra_pka_ready_flux(
    path: str | Path,
    *,
    field_name: str,
) -> dict[str, Any]:
    """Validate exact CCFE-709 scalar flux and reject surface-current substitutes."""
    from .energy_groups import get_structure

    expected_edges = np.asarray(
        get_structure("CCFE-709", particle="neutron").edges_eV, dtype=float
    )
    with h5py.File(path) as source:
        if str(source.attrs.get("quantity", "")) != "volume_scalar_flux":
            raise ValueError(
                "SPECTRA-PKA input must be scalar flux, not surface current"
            )
        manifest = json.loads(source["manifest_json"].asstr()[()])
        group = source[f"scalar_flux_fields/{field_name}"]
        if str(group.attrs["particle"]) != "neutron":
            raise ValueError("CCFE-709 SPECTRA-PKA input must be neutron flux")
        edges = group["energy_edges_eV"][...]
        if edges.shape != (710,) or not np.array_equal(edges, expected_edges):
            raise ValueError(
                "SPECTRA-PKA input does not contain exact CCFE-709 boundaries"
            )
        required = {
            "mean_physical",
            "std_dev_physical",
            "volume_cm3",
            "material_ids",
            "magnet_ids",
        }
        missing = required - set(group)
        if missing:
            raise ValueError(
                f"SPECTRA-PKA scalar flux is missing {sorted(missing)}"
            )
        if group["mean_physical"].shape[1] != 709:
            raise ValueError("SPECTRA-PKA flux must have 709 ascending groups")
    return {
        "status": "PASS",
        "field_name": field_name,
        "groups": 709,
        "edge_ordering": "ascending",
        "units": "particles/cm2/s",
        "material_manifest_sha256": manifest["material_manifest_sha256"],
        "provenance": manifest["provenance"],
    }
