"""Export particle-resolved OpenMC magnet heating without changing its basis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np


SCHEMA = "parastell.magnet_heating/v1.0.0"
EV_TO_JOULE = 1.602176634e-19


def _filter_named(tally, name: str):
    matches = [item for item in tally.filters if type(item).__name__ == name]
    if len(matches) != 1:
        raise ValueError(f"heating tally requires exactly one {name}")
    return matches[0]


def _decode_particle(value) -> str:
    if hasattr(value, "name"):
        return str(value.name).lower()
    return str(value).lower()


def export_magnet_heating(
    output_path: str | Path,
    *,
    statepoint_path: str | Path,
    tally_names: Sequence[str],
    cell_volumes_cm3: Mapping[int, float],
    cell_magnet_ids: Mapping[int, str],
    cell_component_roles: Mapping[int, str] | None = None,
    physical_source_rate_per_s: float,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Export OpenMC ``heating`` scores as eV/source, W, and W/cm3.

    Photon heating follows OpenMC's convention: energy deposited by locally
    created electrons and positrons is attributed to the parent photon.
    """
    if physical_source_rate_per_s <= 0.0:
        raise ValueError("physical_source_rate_per_s must be positive")
    if not tally_names:
        raise ValueError("at least one heating tally name is required")

    import openmc

    rows = []
    seen_particles = set()
    with openmc.StatePoint(str(statepoint_path)) as statepoint:
        for tally_name in tally_names:
            tally = statepoint.get_tally(name=tally_name)
            if list(tally.scores) != ["heating"]:
                raise ValueError(f"{tally_name!r} must score only 'heating'")
            cell_filter = _filter_named(tally, "CellFilter")
            particle_filter = _filter_named(tally, "ParticleFilter")
            energy_filter = _filter_named(tally, "EnergyFilter")
            particles = [
                _decode_particle(item) for item in particle_filter.bins
            ]
            if len(particles) != 1 or particles[0] not in {
                "neutron",
                "photon",
            }:
                raise ValueError(
                    f"{tally_name!r} must select one neutron or photon particle"
                )
            particle = particles[0]
            if particle in seen_particles:
                raise ValueError(f"duplicate {particle} heating tally")
            seen_particles.add(particle)

            cell_ids = np.asarray(cell_filter.bins, dtype=np.int64).reshape(-1)
            energy_edges = np.asarray(energy_filter.values, dtype=float)
            n_energy = energy_edges.size - 1
            mean = np.asarray(tally.mean, dtype=float).reshape(
                cell_ids.size, 1, n_energy, 1, 1
            )[:, 0, :, 0, 0]
            std_dev = np.asarray(tally.std_dev, dtype=float).reshape(
                cell_ids.size, 1, n_energy, 1, 1
            )[:, 0, :, 0, 0]
            if not np.all(np.isfinite(mean)) or not np.all(
                np.isfinite(std_dev)
            ):
                raise ValueError(f"{tally_name!r} contains non-finite values")
            if np.any(mean < 0.0) or np.any(std_dev < 0.0):
                raise ValueError(f"{tally_name!r} contains negative values")

            missing = [
                int(cell)
                for cell in cell_ids
                if int(cell) not in cell_volumes_cm3
            ]
            if missing:
                raise ValueError(f"missing cell volumes for {missing}")
            missing = [
                int(cell)
                for cell in cell_ids
                if int(cell) not in cell_magnet_ids
            ]
            if missing:
                raise ValueError(f"missing magnet IDs for cells {missing}")
            if cell_component_roles is not None:
                missing = [
                    int(cell)
                    for cell in cell_ids
                    if int(cell) not in cell_component_roles
                ]
                if missing:
                    raise ValueError(
                        f"missing component roles for cells {missing}"
                    )
            volumes = np.asarray(
                [cell_volumes_cm3[int(cell)] for cell in cell_ids], dtype=float
            )
            if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
                raise ValueError("cell volumes must be positive and finite")

            watts_factor = physical_source_rate_per_s * EV_TO_JOULE
            mean_w = mean * watts_factor
            std_w = std_dev * watts_factor
            relative_error = np.divide(
                std_dev,
                mean,
                out=np.zeros_like(std_dev),
                where=mean > 0.0,
            )
            rows.append(
                {
                    "particle": particle,
                    "tally_name": tally_name,
                    "cell_ids": cell_ids,
                    "magnet_ids": [
                        cell_magnet_ids[int(cell)] for cell in cell_ids
                    ],
                    "component_roles": (
                        [cell_component_roles[int(cell)] for cell in cell_ids]
                        if cell_component_roles is not None
                        else None
                    ),
                    "volumes_cm3": volumes,
                    "energy_edges_eV": energy_edges,
                    "mean_eV_per_source": mean,
                    "std_dev_eV_per_source": std_dev,
                    "mean_W": mean_w,
                    "std_dev_W": std_w,
                    "mean_W_cm3": mean_w / volumes[:, None],
                    "std_dev_W_cm3": std_w / volumes[:, None],
                    "relative_error": relative_error,
                }
            )

    manifest = {
        "schema": SCHEMA,
        "statepoint": str(Path(statepoint_path).resolve()),
        "physical_source_rate_per_s": float(physical_source_rate_per_s),
        "raw_normalization": "eV/source",
        "physical_normalization": "W and W/cm3",
        "uncertainty": "OpenMC one-standard-deviation tally uncertainty",
        "estimator": "OpenMC heating score",
        "photon_heating_semantics": (
            "local electron and positron deposition is attributed to the parent photon"
        ),
        "particles": sorted(seen_particles),
        "provenance": dict(provenance or {}),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.attrs["schema"] = SCHEMA
        handle.create_dataset(
            "manifest_json", data=json.dumps(manifest, sort_keys=True)
        )
        root = handle.create_group("heating")
        for row in rows:
            group = root.create_group(row["particle"])
            group.attrs["tally_name"] = row["tally_name"]
            for key in (
                "cell_ids",
                "volumes_cm3",
                "energy_edges_eV",
                "mean_eV_per_source",
                "std_dev_eV_per_source",
                "mean_W",
                "std_dev_W",
                "mean_W_cm3",
                "std_dev_W_cm3",
                "relative_error",
            ):
                group.create_dataset(key, data=row[key])
            group.create_dataset(
                "magnet_ids",
                data=np.asarray(row["magnet_ids"], dtype=string_dtype),
            )
            if row["component_roles"] is not None:
                group.create_dataset(
                    "component_roles",
                    data=np.asarray(
                        row["component_roles"], dtype=string_dtype
                    ),
                )
    manifest["path"] = str(output_path.resolve())
    return manifest
