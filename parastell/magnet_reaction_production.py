"""Versioned export of magnet reaction and particle-production tallies."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SCHEMA = "parastell.magnet_reaction_production/v1.0.0"
REACTION_TALLY = "pstl_magnet_neutron_reactions"
PRODUCTION_TALLIES = {
    particle: f"pstl_magnet_production_{particle}"
    for particle in ("photon", "neutron", "electron", "positron")
}


def _filter(tally: Any, class_name: str) -> tuple[int, Any]:
    matches = [
        (index, item)
        for index, item in enumerate(tally.filters)
        if type(item).__name__ == class_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{tally.name!r} requires exactly one {class_name}; found {len(matches)}"
        )
    return matches[0]


def _energy_edges(filter_: Any, *, production: bool = False) -> np.ndarray:
    if production:
        bins = list(filter_.bins)
        particles = {str(item[0]) for item in bins}
        if len(particles) != 1:
            raise ValueError(
                "production filter must contain one outgoing particle"
            )
        intervals = [(float(item[1]), float(item[2])) for item in bins]
    else:
        intervals = [tuple(map(float, item)) for item in filter_.bins]
    if not intervals:
        raise ValueError("energy filter cannot be empty")
    edges = np.asarray([intervals[0][0], *(item[1] for item in intervals)])
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy boundaries must be finite and increasing")
    if any(
        left[1] != right[0] for left, right in zip(intervals, intervals[1:])
    ):
        raise ValueError("energy bins must be exactly contiguous")
    return edges


def _ordered_values(
    tally: Any, response_filter_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    cell_axis, cell_filter = _filter(tally, "CellFilter")
    energy_axis, energy_filter = _filter(tally, "EnergyFilter")
    response_axis, response_filter = _filter(tally, response_filter_name)
    shape = [int(item.num_bins) for item in tally.filters]
    shape.extend((int(tally.num_nuclides), int(tally.num_scores)))
    if tally.num_nuclides != 1 or tally.num_scores != 1:
        raise ValueError("export requires one total-nuclide and one score bin")
    retained = {cell_axis, energy_axis, response_axis}
    for index, filter_ in enumerate(tally.filters):
        if index not in retained and int(filter_.num_bins) != 1:
            raise ValueError(
                f"unsupported non-singleton filter {type(filter_).__name__}"
            )

    def arrange(values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=float).reshape(shape)
        array = np.moveaxis(
            array, (cell_axis, energy_axis, response_axis), (0, 1, 2)
        )
        trailing = tuple(range(3, array.ndim))
        if any(array.shape[index] != 1 for index in trailing):
            raise ValueError("unexpected non-singleton tally dimensions")
        return array.reshape(array.shape[:3])

    mean = arrange(tally.mean)
    std_dev = arrange(tally.std_dev)
    if np.any(~np.isfinite(mean)) or np.any(mean < 0.0):
        raise ValueError(f"{tally.name!r} contains invalid means")
    if np.any(~np.isfinite(std_dev)) or np.any(std_dev < 0.0):
        raise ValueError(
            f"{tally.name!r} contains invalid standard deviations"
        )
    cells = np.asarray(cell_filter.bins, dtype=np.int64)
    incident_edges = _energy_edges(energy_filter)
    return cells, incident_edges, mean, std_dev, response_filter


def _relative_error(mean: np.ndarray, std_dev: np.ndarray) -> np.ndarray:
    return np.divide(
        std_dev,
        mean,
        out=np.zeros_like(std_dev),
        where=mean > 0.0,
    )


def _string_dataset(
    group: h5py.Group, name: str, values: Sequence[str]
) -> None:
    group.create_dataset(
        name, data=np.asarray(values, dtype=h5py.string_dtype())
    )


def export_magnet_reaction_production(
    statepoint: str | Path | Any,
    output_path: str | Path,
    *,
    magnet_ids: Sequence[str] | None = None,
    physical_source_rate_per_s: float | None = None,
    transported_particles: Mapping[str, bool] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Export independently scored reaction and production distributions.

    The score is the OpenMC ``events`` estimator. Production arrays are indexed
    ``[magnet, incident-energy, outgoing-energy]`` and reaction arrays are
    indexed ``[magnet, incident-energy, reaction]``. Produced-particle identity
    is deliberately separate from whether OpenMC transported that particle.
    """

    if physical_source_rate_per_s is not None and (
        not np.isfinite(physical_source_rate_per_s)
        or physical_source_rate_per_s <= 0.0
    ):
        raise ValueError("physical source rate must be finite and positive")
    if isinstance(statepoint, (str, Path)):
        import openmc

        context = openmc.StatePoint(str(statepoint))
    else:
        context = nullcontext(statepoint)
    transported = {
        "neutron": True,
        "photon": True,
        "electron": False,
        "positron": False,
        **dict(transported_particles or {}),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with context as handle:
        reaction = handle.get_tally(name=REACTION_TALLY)
        cells, incident_edges, mean, std_dev, reaction_filter = (
            _ordered_values(reaction, "ReactionFilter")
        )
        ids = tuple(magnet_ids or (f"magnet-{cell}" for cell in cells))
        if len(ids) != len(cells):
            raise ValueError("magnet IDs must align with cell IDs")
        reaction_labels = tuple(str(value) for value in reaction_filter.bins)
        products = {}
        for particle, tally_name in PRODUCTION_TALLIES.items():
            tally = handle.get_tally(name=tally_name)
            pcells, p_incident, pmean, pstd, production_filter = (
                _ordered_values(tally, "ParticleProductionFilter")
            )
            if not np.array_equal(pcells, cells):
                raise ValueError(
                    f"{tally_name!r} cell axis does not match reactions"
                )
            if not np.array_equal(p_incident, incident_edges):
                raise ValueError(
                    f"{tally_name!r} incident-energy axis does not match"
                )
            outgoing = {str(item[0]) for item in production_filter.bins}
            if outgoing != {particle}:
                raise ValueError(
                    f"{tally_name!r} outgoing-particle identity mismatch"
                )
            products[particle] = (
                _energy_edges(production_filter, production=True),
                pmean,
                pstd,
            )

        manifest = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "normalization": {
                "per_source": "events per OpenMC source history",
                "physical_source_rate_per_s": physical_source_rate_per_s,
            },
            "uncertainty": "OpenMC reported standard deviation of the mean",
            "incident_particle": "neutron",
            "score": "events",
            "provenance": dict(provenance or {}),
        }
        with h5py.File(output, "w") as stream:
            stream.attrs["schema"] = SCHEMA
            stream.create_dataset("manifest_json", data=json.dumps(manifest))
            reactions = stream.create_group("reactions/neutron")
            reactions.create_dataset("cell_ids", data=cells)
            _string_dataset(reactions, "magnet_ids", ids)
            reactions.create_dataset(
                "incident_energy_edges_eV", data=incident_edges
            )
            _string_dataset(reactions, "reaction_labels", reaction_labels)
            reactions.create_dataset("mean_events_per_source", data=mean)
            reactions.create_dataset("std_dev_events_per_source", data=std_dev)
            reactions.create_dataset(
                "relative_error", data=_relative_error(mean, std_dev)
            )
            if physical_source_rate_per_s is not None:
                reactions.create_dataset(
                    "mean_events_per_s", data=mean * physical_source_rate_per_s
                )
                reactions.create_dataset(
                    "std_dev_events_per_s",
                    data=std_dev * physical_source_rate_per_s,
                )
            for particle, (edges, pmean, pstd) in products.items():
                group = stream.create_group(f"production/{particle}")
                group.attrs["produced_particle"] = particle
                group.attrs["transported_by_openmc"] = bool(
                    transported.get(particle, False)
                )
                group.create_dataset("cell_ids", data=cells)
                _string_dataset(group, "magnet_ids", ids)
                group.create_dataset(
                    "incident_energy_edges_eV", data=incident_edges
                )
                group.create_dataset("outgoing_energy_edges_eV", data=edges)
                group.create_dataset("mean_events_per_source", data=pmean)
                group.create_dataset("std_dev_events_per_source", data=pstd)
                group.create_dataset(
                    "relative_error", data=_relative_error(pmean, pstd)
                )
                if physical_source_rate_per_s is not None:
                    group.create_dataset(
                        "mean_events_per_s",
                        data=pmean * physical_source_rate_per_s,
                    )
                    group.create_dataset(
                        "std_dev_events_per_s",
                        data=pstd * physical_source_rate_per_s,
                    )
    return output


def validate_magnet_reaction_production(path: str | Path) -> dict[str, Any]:
    """Reject malformed reaction/production products and return a summary."""

    with h5py.File(path, "r") as stream:
        if stream.attrs.get("schema") != SCHEMA:
            raise ValueError("unsupported magnet reaction/production schema")
        manifest = json.loads(stream["manifest_json"][()].decode())
        reactions = stream["reactions/neutron"]
        required = {
            "cell_ids",
            "magnet_ids",
            "incident_energy_edges_eV",
            "reaction_labels",
            "mean_events_per_source",
            "std_dev_events_per_source",
            "relative_error",
        }
        if missing := required.difference(reactions):
            raise ValueError(f"reaction payload missing {sorted(missing)}")
        cells = reactions["cell_ids"][...]
        reaction_shape = reactions["mean_events_per_source"].shape
        expected = (
            len(cells),
            len(reactions["incident_energy_edges_eV"]) - 1,
            len(reactions["reaction_labels"]),
        )
        if reaction_shape != expected:
            raise ValueError("reaction payload shape does not match its axes")
        totals = {}
        for particle in PRODUCTION_TALLIES:
            name = f"production/{particle}"
            if name not in stream:
                raise ValueError(f"missing {name}")
            group = stream[name]
            shape = group["mean_events_per_source"].shape
            expected = (
                len(cells),
                len(group["incident_energy_edges_eV"]) - 1,
                len(group["outgoing_energy_edges_eV"]) - 1,
            )
            if shape != expected:
                raise ValueError(
                    f"{name} payload shape does not match its axes"
                )
            totals[particle] = float(
                group["mean_events_per_source"][...].sum()
            )
        return {
            "schema": manifest["schema"],
            "magnets": len(cells),
            "reaction_events_per_source": float(
                reactions["mean_events_per_source"][...].sum()
            ),
            "production_events_per_source": totals,
        }
