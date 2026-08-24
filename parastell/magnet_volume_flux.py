"""Neutral volume scalar-flux export from OpenMC statepoint tallies."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np


SCHEMA = "parastell.magnet_volume_scalar_flux/v1.0.0"


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
            "created_utc": datetime.now(timezone.utc).isoformat(),
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
