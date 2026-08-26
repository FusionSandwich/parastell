"""Neutral export of OpenMC magnet damage-energy and gas-production tallies."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .openmc16 import GAS_PRODUCTION_SCORES

SCHEMA = "parastell.magnet_damage_gas/v1.0.0"
DAMAGE_TALLY = "pstl_magnet_neutron_damage_energy"
GAS_TALLY = "pstl_magnet_gas_production"
RESPONSE_NAMES = ("damage-energy", *GAS_PRODUCTION_SCORES)
GAS_OPENMC_SCORES = {
    "H1-production": "(n,Xp)",
    "H2-production": "(n,Xd)",
    "H3-production": "(n,Xt)",
    "He3-production": "(n,X3He)",
    "He4-production": "(n,Xa)",
}
EV_TO_JOULE = 1.602176634e-19


def _inventory_mapping(inventory: Any) -> Mapping[str, Any]:
    if hasattr(inventory, "to_dict"):
        inventory = inventory.to_dict()
    if not isinstance(inventory, Mapping):
        raise TypeError(
            "tally_inventory must be a mapping or expose to_dict()"
        )
    required = {"damage_energy", "gas_production", "response_availability"}
    if missing := required.difference(inventory):
        raise ValueError(f"tally inventory missing {sorted(missing)}")
    return inventory


def _response_statuses(
    inventory: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = inventory["response_availability"]
    if not isinstance(raw, Mapping):
        raise TypeError("response_availability must be a mapping")
    statuses: dict[str, dict[str, Any]] = {}
    for response in RESPONSE_NAMES:
        if response not in raw or not isinstance(raw[response], Mapping):
            raise ValueError(f"missing availability status for {response!r}")
        item = dict(raw[response])
        if type(item.get("available")) is not bool:
            raise ValueError(f"{response!r} availability must be boolean")
        if not isinstance(item.get("status"), str) or not item["status"]:
            raise ValueError(f"{response!r} status must be a nonempty string")
        statuses[response] = item

    damage_expected = statuses["damage-energy"]["available"]
    damage_name = inventory["damage_energy"]
    if damage_expected and damage_name != DAMAGE_TALLY:
        raise ValueError(
            f"available damage-energy response must use {DAMAGE_TALLY!r}"
        )
    if not damage_expected and damage_name is not None:
        raise ValueError("unavailable damage-energy response has a tally name")

    gas_expected = any(
        statuses[score]["available"] for score in GAS_PRODUCTION_SCORES
    )
    gas_name = inventory["gas_production"]
    if gas_expected and gas_name != GAS_TALLY:
        raise ValueError(f"available gas response must use {GAS_TALLY!r}")
    if not gas_expected and gas_name is not None:
        raise ValueError("unavailable gas responses have a tally name")
    return statuses


def _filter(tally: Any, class_name: str) -> tuple[int, Any]:
    matches = [
        (index, item)
        for index, item in enumerate(tally.filters)
        if type(item).__name__ == class_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{tally.name!r} requires exactly one {class_name}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _decode_particle(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name).lower()
    return str(value).lower()


def _energy_edges(filter_: Any) -> np.ndarray:
    intervals = [tuple(map(float, item)) for item in filter_.bins]
    if not intervals:
        raise ValueError("incident-energy filter cannot be empty")
    if any(len(item) != 2 for item in intervals):
        raise ValueError("incident-energy bins must be intervals")
    edges = np.asarray([intervals[0][0], *(item[1] for item in intervals)])
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError(
            "incident-energy boundaries must be finite and increasing"
        )
    if any(left[1] != right[0] for left, right in pairwise(intervals)):
        raise ValueError("incident-energy bins must be exactly contiguous")
    return edges


def _tally_values(
    tally: Any, *, expected_scores: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tuple(str(score) for score in tally.scores) != tuple(expected_scores):
        raise ValueError(
            f"{tally.name!r} scores do not match the tally inventory: "
            f"expected {list(expected_scores)!r}, found {list(tally.scores)!r}"
        )
    if len(tally.filters) != 3:
        raise ValueError(
            f"{tally.name!r} must contain only cell, neutron, and energy filters"
        )
    cell_axis, cell_filter = _filter(tally, "CellFilter")
    particle_axis, particle_filter = _filter(tally, "ParticleFilter")
    energy_axis, energy_filter = _filter(tally, "EnergyFilter")
    particles = [_decode_particle(item) for item in particle_filter.bins]
    if particles != ["neutron"]:
        raise ValueError(f"{tally.name!r} must select only incident neutrons")
    if int(tally.num_nuclides) != 1:
        raise ValueError(
            f"{tally.name!r} must use exactly one total-nuclide bin"
        )
    if int(tally.num_scores) != len(expected_scores):
        raise ValueError(f"{tally.name!r} score-axis length is inconsistent")

    cells = np.asarray(cell_filter.bins, dtype=np.int64).reshape(-1)
    if cells.size == 0 or np.unique(cells).size != cells.size:
        raise ValueError(
            f"{tally.name!r} cell IDs must be nonempty and unique"
        )
    edges = _energy_edges(energy_filter)
    filter_shape = [int(filter_.num_bins) for filter_ in tally.filters]
    full_shape = (*filter_shape, 1, len(expected_scores))

    def arrange(values: Any, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.size != int(np.prod(full_shape)):
            raise ValueError(f"{tally.name!r} {label} shape is inconsistent")
        array = array.reshape(full_shape)
        array = np.moveaxis(
            array,
            (cell_axis, energy_axis, particle_axis),
            (0, 1, 2),
        )
        expected_shape = (
            cells.size,
            edges.size - 1,
            1,
            1,
            len(expected_scores),
        )
        if array.shape != expected_shape:
            raise ValueError(f"{tally.name!r} {label} axes are inconsistent")
        return array[:, :, 0, 0, :]

    mean = arrange(tally.mean, "mean")
    std_dev = arrange(tally.std_dev, "standard-deviation")
    if np.any(~np.isfinite(mean)) or np.any(mean < 0.0):
        raise ValueError(f"{tally.name!r} contains invalid means")
    if np.any(~np.isfinite(std_dev)) or np.any(std_dev < 0.0):
        raise ValueError(
            f"{tally.name!r} contains invalid standard deviations"
        )
    return cells, edges, mean, std_dev


def _cell_metadata(
    cells: np.ndarray,
    cell_magnet_ids: Mapping[int, str],
    cell_volumes_cm3: Mapping[int, float],
) -> tuple[tuple[str, ...], np.ndarray]:
    missing_ids = [
        int(cell) for cell in cells if int(cell) not in cell_magnet_ids
    ]
    if missing_ids:
        raise ValueError(f"missing magnet IDs for cells {missing_ids}")
    missing_volumes = [
        int(cell) for cell in cells if int(cell) not in cell_volumes_cm3
    ]
    if missing_volumes:
        raise ValueError(f"missing cell volumes for {missing_volumes}")
    magnet_ids = tuple(str(cell_magnet_ids[int(cell)]) for cell in cells)
    if any(not value for value in magnet_ids):
        raise ValueError("magnet IDs must be nonempty")
    volumes = np.asarray(
        [cell_volumes_cm3[int(cell)] for cell in cells], dtype=float
    )
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("cell volumes must be positive and finite")
    return magnet_ids, volumes


def _relative_error(mean: np.ndarray, std_dev: np.ndarray) -> np.ndarray:
    return np.divide(
        std_dev,
        mean,
        out=np.full_like(std_dev, np.nan),
        where=mean > 0.0,
    )


def _dataset(
    group: h5py.Group,
    name: str,
    values: Any,
    *,
    units: str,
    axes: str | None = None,
) -> h5py.Dataset:
    result = group.create_dataset(name, data=values)
    result.attrs["units"] = units
    if axes is not None:
        result.attrs["axes"] = axes
    return result


def _write_common_axes(
    group: h5py.Group,
    *,
    cells: np.ndarray,
    magnet_ids: Sequence[str],
    component_roles: Sequence[str] | None,
    volumes: np.ndarray,
    edges: np.ndarray,
) -> None:
    _dataset(group, "cell_ids", cells, units="dimensionless", axes="magnet")
    magnet_dataset = group.create_dataset(
        "magnet_ids",
        data=np.asarray(magnet_ids, dtype=h5py.string_dtype(encoding="utf-8")),
    )
    magnet_dataset.attrs["axes"] = "magnet"
    if component_roles is not None:
        roles = tuple(str(value) for value in component_roles)
        if len(roles) != len(cells) or any(not value for value in roles):
            raise ValueError("component roles must align with cells")
        role_dataset = group.create_dataset(
            "component_roles",
            data=np.asarray(roles, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        role_dataset.attrs["axes"] = "magnet"
    _dataset(group, "cell_volumes_cm3", volumes, units="cm3", axes="magnet")
    _dataset(
        group,
        "incident_energy_edges_eV",
        edges,
        units="eV",
        axes="incident_energy_edge",
    )


def _write_response_status(
    stream: h5py.File, statuses: Mapping[str, Mapping[str, Any]]
) -> None:
    group = stream.create_group("response_status")
    strings = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(
        "response", data=np.asarray(RESPONSE_NAMES, dtype=strings)
    )
    group.create_dataset(
        "configured_status",
        data=np.asarray(
            [statuses[name]["status"] for name in RESPONSE_NAMES],
            dtype=strings,
        ),
    )
    group.create_dataset(
        "available",
        data=np.asarray(
            [statuses[name]["available"] for name in RESPONSE_NAMES],
            dtype=bool,
        ),
    )
    group.create_dataset(
        "product_status",
        data=np.asarray(
            [
                (
                    "SCORED"
                    if statuses[name]["available"]
                    else statuses[name]["status"]
                )
                for name in RESPONSE_NAMES
            ],
            dtype=strings,
        ),
    )
    group.attrs["unavailable_semantics"] = (
        "An unavailable response is absent, not a scored zero."
    )


def export_magnet_damage_gas(
    statepoint: str | Path | Any,
    output_path: str | Path,
    *,
    tally_inventory: Mapping[str, Any] | Any,
    cell_magnet_ids: Mapping[int, str],
    cell_component_roles: Mapping[int, str] | None = None,
    cell_volumes_cm3: Mapping[int, float],
    physical_source_rate_per_s: float,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Export damage energy and gas-nuclide production without deriving DPA/appm."""

    if (
        not np.isfinite(physical_source_rate_per_s)
        or physical_source_rate_per_s <= 0
    ):
        raise ValueError("physical source rate must be finite and positive")
    inventory = _inventory_mapping(tally_inventory)
    statuses = _response_statuses(inventory)
    available_gas_scores = tuple(
        score
        for score in GAS_PRODUCTION_SCORES
        if statuses[score]["available"]
    )
    if isinstance(statepoint, (str, Path)):
        import openmc

        context = openmc.StatePoint(str(statepoint))
        statepoint_reference = str(Path(statepoint).resolve())
    else:
        context = nullcontext(statepoint)
        statepoint_reference = None

    damage_values = None
    gas_values = None
    reference_cells = None
    reference_edges = None
    with context as handle:
        if statuses["damage-energy"]["available"]:
            try:
                damage_tally = handle.get_tally(name=DAMAGE_TALLY)
            except Exception as exc:
                raise ValueError(
                    f"required tally {DAMAGE_TALLY!r} is missing"
                ) from exc
            damage_values = _tally_values(
                damage_tally, expected_scores=("damage-energy",)
            )
            reference_cells, reference_edges = damage_values[:2]
        if available_gas_scores:
            try:
                gas_tally = handle.get_tally(name=GAS_TALLY)
            except Exception as exc:
                raise ValueError(
                    f"required tally {GAS_TALLY!r} is missing"
                ) from exc
            gas_values = _tally_values(
                gas_tally,
                expected_scores=tuple(
                    GAS_OPENMC_SCORES[score] for score in available_gas_scores
                ),
            )
            if reference_cells is not None and not np.array_equal(
                gas_values[0], reference_cells
            ):
                raise ValueError("damage and gas tally cell axes do not match")
            if reference_edges is not None and not np.array_equal(
                gas_values[1], reference_edges
            ):
                raise ValueError(
                    "damage and gas incident-energy axes do not match"
                )
            reference_cells, reference_edges = gas_values[:2]

    if reference_cells is None or reference_edges is None:
        raise ValueError(
            "no damage-energy or gas-production response is available"
        )
    magnet_ids, volumes = _cell_metadata(
        reference_cells, cell_magnet_ids, cell_volumes_cm3
    )
    component_roles = None
    if cell_component_roles is not None:
        missing_roles = [
            int(cell)
            for cell in reference_cells
            if int(cell) not in cell_component_roles
        ]
        if missing_roles:
            raise ValueError(
                f"missing component roles for cells {missing_roles}"
            )
        component_roles = tuple(
            str(cell_component_roles[int(cell)]) for cell in reference_cells
        )
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "statepoint": statepoint_reference,
        "incident_particle": "neutron",
        "physical_source_rate_per_s": float(physical_source_rate_per_s),
        "uncertainty": "OpenMC reported one-standard-deviation tally uncertainty",
        "zero_score_interpretation": (
            "A zero tally estimate is not a claim of a physical zero."
        ),
        "damage_energy_semantics": (
            "OpenMC damage-energy in eV/source; no displacement threshold, "
            "NRT/arc-DPA model, or DPA conversion has been applied."
        ),
        "gas_production_semantics": (
            "OpenMC gas-nuclide production in atoms/source; no irradiation-time, "
            "host-atom inventory, depletion, or appm conversion has been applied."
        ),
        "response_status": statuses,
        "provenance": dict(provenance or {}),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as stream:
        stream.attrs["schema"] = SCHEMA
        stream.create_dataset(
            "manifest_json", data=json.dumps(manifest, sort_keys=True)
        )
        _write_response_status(stream, statuses)
        if damage_values is not None:
            cells, edges, mean, std_dev = damage_values
            group = stream.create_group("damage_energy/neutron")
            group.attrs["tally_name"] = DAMAGE_TALLY
            group.attrs["score"] = "damage-energy"
            group.attrs["is_dpa"] = False
            group.attrs["axis_order"] = "magnet,incident_energy"
            _write_common_axes(
                group,
                cells=cells,
                magnet_ids=magnet_ids,
                component_roles=component_roles,
                volumes=volumes,
                edges=edges,
            )
            mean = mean[..., 0]
            std_dev = std_dev[..., 0]
            axes = "magnet,incident_energy"
            _dataset(
                group, "mean_eV_per_source", mean, units="eV/source", axes=axes
            )
            _dataset(
                group,
                "std_dev_eV_per_source",
                std_dev,
                units="eV/source",
                axes=axes,
            )
            _dataset(
                group,
                "mean_eV_per_source_cm3",
                mean / volumes[:, None],
                units="eV/source/cm3",
                axes=axes,
            )
            _dataset(
                group,
                "std_dev_eV_per_source_cm3",
                std_dev / volumes[:, None],
                units="eV/source/cm3",
                axes=axes,
            )
            _dataset(
                group,
                "mean_J_per_s",
                mean * physical_source_rate_per_s * EV_TO_JOULE,
                units="J/s",
                axes=axes,
            )
            _dataset(
                group,
                "std_dev_J_per_s",
                std_dev * physical_source_rate_per_s * EV_TO_JOULE,
                units="J/s",
                axes=axes,
            )
            _dataset(
                group,
                "mean_J_per_s_cm3",
                mean
                * physical_source_rate_per_s
                * EV_TO_JOULE
                / volumes[:, None],
                units="J/s/cm3",
                axes=axes,
            )
            _dataset(
                group,
                "std_dev_J_per_s_cm3",
                std_dev
                * physical_source_rate_per_s
                * EV_TO_JOULE
                / volumes[:, None],
                units="J/s/cm3",
                axes=axes,
            )
            _dataset(
                group,
                "relative_error",
                _relative_error(mean, std_dev),
                units="dimensionless",
                axes=axes,
            )
            _dataset(
                group,
                "sample_status_code",
                np.where(mean > 0.0, 1, 0).astype(np.uint8),
                units="dimensionless",
                axes=axes,
            )
        if gas_values is not None:
            cells, edges, mean, std_dev = gas_values
            group = stream.create_group("gas_production/neutron")
            group.attrs["tally_name"] = GAS_TALLY
            group.attrs["axis_order"] = "magnet,incident_energy,gas_nuclide"
            group.attrs["is_appm"] = False
            _write_common_axes(
                group,
                cells=cells,
                magnet_ids=magnet_ids,
                component_roles=component_roles,
                volumes=volumes,
                edges=edges,
            )
            group.create_dataset(
                "score_labels",
                data=np.asarray(
                    available_gas_scores, dtype=h5py.string_dtype()
                ),
            )
            group.create_dataset(
                "openmc_score_labels",
                data=np.asarray(
                    [
                        GAS_OPENMC_SCORES[score]
                        for score in available_gas_scores
                    ],
                    dtype=h5py.string_dtype(),
                ),
            )
            axes = "magnet,incident_energy,gas_nuclide"
            _dataset(
                group,
                "mean_atoms_per_source",
                mean,
                units="atoms/source",
                axes=axes,
            )
            _dataset(
                group,
                "std_dev_atoms_per_source",
                std_dev,
                units="atoms/source",
                axes=axes,
            )
            _dataset(
                group,
                "mean_atoms_per_source_cm3",
                mean / volumes[:, None, None],
                units="atoms/source/cm3",
                axes=axes,
            )
            _dataset(
                group,
                "std_dev_atoms_per_source_cm3",
                std_dev / volumes[:, None, None],
                units="atoms/source/cm3",
                axes=axes,
            )
            _dataset(
                group,
                "mean_atoms_per_s",
                mean * physical_source_rate_per_s,
                units="atoms/s",
                axes=axes,
            )
            _dataset(
                group,
                "std_dev_atoms_per_s",
                std_dev * physical_source_rate_per_s,
                units="atoms/s",
                axes=axes,
            )
            _dataset(
                group,
                "mean_atoms_per_s_cm3",
                mean * physical_source_rate_per_s / volumes[:, None, None],
                units="atoms/s/cm3",
                axes=axes,
            )
            _dataset(
                group,
                "std_dev_atoms_per_s_cm3",
                std_dev * physical_source_rate_per_s / volumes[:, None, None],
                units="atoms/s/cm3",
                axes=axes,
            )
            _dataset(
                group,
                "relative_error",
                _relative_error(mean, std_dev),
                units="dimensionless",
                axes=axes,
            )
            _dataset(
                group,
                "sample_status_code",
                np.where(mean > 0.0, 1, 0).astype(np.uint8),
                units="dimensionless",
                axes=axes,
            )
        legend = stream.create_group("sample_status_legend")
        legend.attrs["0"] = "EMPTY_NOT_A_PHYSICAL_ZERO"
        legend.attrs["1"] = "ESTIMATED"
    return output


def _validate_axes(
    group: h5py.Group,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    required = {
        "cell_ids",
        "magnet_ids",
        "cell_volumes_cm3",
        "incident_energy_edges_eV",
    }
    if missing := required.difference(group):
        raise ValueError(f"response axes missing {sorted(missing)}")
    cells = group["cell_ids"][...]
    magnets = group["magnet_ids"].asstr()[...]
    volumes = group["cell_volumes_cm3"][...]
    edges = group["incident_energy_edges_eV"][...]
    if (
        cells.ndim != 1
        or len(cells) == 0
        or len(np.unique(cells)) != len(cells)
    ):
        raise ValueError("invalid response cell axis")
    if magnets.shape != cells.shape or any(not value for value in magnets):
        raise ValueError("invalid response magnet axis")
    if (
        volumes.shape != cells.shape
        or np.any(~np.isfinite(volumes))
        or np.any(volumes <= 0)
    ):
        raise ValueError("invalid response volumes")
    if (
        edges.ndim != 1
        or len(edges) < 2
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("invalid response energy axis")
    expected_units = {
        "cell_ids": "dimensionless",
        "cell_volumes_cm3": "cm3",
        "incident_energy_edges_eV": "eV",
    }
    for name, units in expected_units.items():
        if group[name].attrs.get("units") != units:
            raise ValueError(f"{group.name}/{name} has invalid units")
    return cells, magnets, volumes, len(edges) - 1


def _validated_dataset(
    group: h5py.Group,
    name: str,
    *,
    expected_shape: tuple[int, ...],
    units: str,
) -> np.ndarray:
    if name not in group or group[name].shape != expected_shape:
        raise ValueError(f"{group.name}/{name} has an invalid shape")
    if group[name].attrs.get("units") != units:
        raise ValueError(f"{group.name}/{name} has invalid units")
    values = group[name][...]
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{group.name}/{name} contains invalid values")
    return values


def _validate_scaled_dataset(
    group: h5py.Group,
    name: str,
    expected: np.ndarray,
    *,
    units: str,
) -> None:
    values = _validated_dataset(
        group, name, expected_shape=expected.shape, units=units
    )
    if not np.allclose(values, expected, rtol=1.0e-13, atol=0.0):
        raise ValueError(f"{group.name}/{name} has inconsistent normalization")


def _validate_statistics(
    group: h5py.Group,
    *,
    expected_shape: tuple[int, ...],
    mean_name: str,
    std_name: str,
    units: str,
) -> tuple[np.ndarray, np.ndarray]:
    mean = _validated_dataset(
        group, mean_name, expected_shape=expected_shape, units=units
    )
    std_dev = _validated_dataset(
        group, std_name, expected_shape=expected_shape, units=units
    )
    for name in ("relative_error", "sample_status_code"):
        if name not in group or group[name].shape != expected_shape:
            raise ValueError(f"{group.name}/{name} has an invalid shape")
        if group[name].attrs.get("units") != "dimensionless":
            raise ValueError(f"{group.name}/{name} has invalid units")
    relative = group["relative_error"][...]
    status = group["sample_status_code"][...]
    if np.any(np.isfinite(relative[mean == 0])) or np.any(
        ~np.isfinite(relative[mean > 0])
    ):
        raise ValueError(
            f"{group.name} relative errors misrepresent zero estimates"
        )
    if not np.array_equal(status, np.where(mean > 0.0, 1, 0)):
        raise ValueError(f"{group.name} sample status is inconsistent")
    return mean, std_dev


def validate_magnet_damage_gas(path: str | Path) -> dict[str, Any]:
    """Validate a neutral damage/gas product without importing OpenMC."""

    with h5py.File(path, "r") as stream:
        if stream.attrs.get("schema") != SCHEMA:
            raise ValueError("unsupported magnet damage/gas schema")
        manifest = json.loads(stream["manifest_json"][()].decode())
        if manifest.get("schema") != SCHEMA:
            raise ValueError("manifest schema does not match product schema")
        source_rate = manifest.get("physical_source_rate_per_s")
        if (
            not isinstance(source_rate, (int, float))
            or not np.isfinite(source_rate)
            or source_rate <= 0
        ):
            raise ValueError("manifest physical source rate is invalid")
        status_group = stream.get("response_status")
        if status_group is None:
            raise ValueError("response status is missing")
        required_status = {
            "response",
            "configured_status",
            "available",
            "product_status",
        }
        if missing := required_status.difference(status_group):
            raise ValueError(f"response status missing {sorted(missing)}")
        responses = tuple(status_group["response"].asstr()[...])
        if responses != RESPONSE_NAMES:
            raise ValueError(
                "response status inventory is incomplete or reordered"
            )
        availability = status_group["available"][...]
        if availability.shape != (len(RESPONSE_NAMES),):
            raise ValueError("response availability shape is invalid")
        configured = tuple(status_group["configured_status"].asstr()[...])
        product_status = tuple(status_group["product_status"].asstr()[...])
        manifest_status = manifest.get("response_status")
        if not isinstance(manifest_status, Mapping):
            raise TypeError("manifest response status is invalid")
        expected_configured = tuple(
            manifest_status[name]["status"] for name in RESPONSE_NAMES
        )
        expected_availability = np.asarray(
            [manifest_status[name]["available"] for name in RESPONSE_NAMES],
            dtype=bool,
        )
        expected_product_status = tuple(
            "SCORED" if available else status
            for available, status in zip(
                expected_availability, expected_configured
            )
        )
        if (
            configured != expected_configured
            or not np.array_equal(availability, expected_availability)
            or product_status != expected_product_status
        ):
            raise ValueError("response status disagrees with manifest")
        if "sample_status_legend" not in stream:
            raise ValueError("sample status legend is missing")

        damage_total = None
        damage_available = bool(availability[0])
        if damage_available != ("damage_energy/neutron" in stream):
            raise ValueError(
                "damage-energy payload disagrees with response status"
            )
        reference_cells = None
        reference_magnets = None
        if damage_available:
            group = stream["damage_energy/neutron"]
            if (
                group.attrs.get("tally_name") != DAMAGE_TALLY
                or group.attrs.get("score") != "damage-energy"
                or bool(group.attrs.get("is_dpa"))
            ):
                raise ValueError("damage-energy metadata is invalid")
            cells, magnets, volumes, n_energy = _validate_axes(group)
            mean, std_dev = _validate_statistics(
                group,
                expected_shape=(len(cells), n_energy),
                mean_name="mean_eV_per_source",
                std_name="std_dev_eV_per_source",
                units="eV/source",
            )
            _validate_scaled_dataset(
                group,
                "mean_eV_per_source_cm3",
                mean / volumes[:, None],
                units="eV/source/cm3",
            )
            _validate_scaled_dataset(
                group,
                "std_dev_eV_per_source_cm3",
                std_dev / volumes[:, None],
                units="eV/source/cm3",
            )
            _validate_scaled_dataset(
                group,
                "mean_J_per_s",
                mean * source_rate * EV_TO_JOULE,
                units="J/s",
            )
            _validate_scaled_dataset(
                group,
                "std_dev_J_per_s",
                std_dev * source_rate * EV_TO_JOULE,
                units="J/s",
            )
            _validate_scaled_dataset(
                group,
                "mean_J_per_s_cm3",
                mean * source_rate * EV_TO_JOULE / volumes[:, None],
                units="J/s/cm3",
            )
            _validate_scaled_dataset(
                group,
                "std_dev_J_per_s_cm3",
                std_dev * source_rate * EV_TO_JOULE / volumes[:, None],
                units="J/s/cm3",
            )
            damage_total = float(mean.sum())
            reference_cells = cells
            reference_magnets = magnets

        available_gas = tuple(
            response
            for response, available in zip(
                RESPONSE_NAMES[1:], availability[1:]
            )
            if available
        )
        if bool(available_gas) != ("gas_production/neutron" in stream):
            raise ValueError(
                "gas-production payload disagrees with response status"
            )
        gas_totals: dict[str, float] = {}
        if available_gas:
            group = stream["gas_production/neutron"]
            if group.attrs.get("tally_name") != GAS_TALLY or bool(
                group.attrs.get("is_appm")
            ):
                raise ValueError("gas-production metadata is invalid")
            cells, magnets, volumes, n_energy = _validate_axes(group)
            if reference_cells is not None and not np.array_equal(
                cells, reference_cells
            ):
                raise ValueError(
                    "damage and gas product cell axes do not match"
                )
            if reference_magnets is not None and not np.array_equal(
                magnets, reference_magnets
            ):
                raise ValueError(
                    "damage and gas product magnet axes do not match"
                )
            if (
                "score_labels" not in group
                or "openmc_score_labels" not in group
            ):
                raise ValueError("gas score identity axes are missing")
            scores = tuple(group["score_labels"].asstr()[...])
            if scores != available_gas:
                raise ValueError(
                    "gas score axis disagrees with response status"
                )
            openmc_scores = tuple(group["openmc_score_labels"].asstr()[...])
            if openmc_scores != tuple(
                GAS_OPENMC_SCORES[score] for score in scores
            ):
                raise ValueError("OpenMC gas score aliases are invalid")
            mean, std_dev = _validate_statistics(
                group,
                expected_shape=(len(cells), n_energy, len(scores)),
                mean_name="mean_atoms_per_source",
                std_name="std_dev_atoms_per_source",
                units="atoms/source",
            )
            _validate_scaled_dataset(
                group,
                "mean_atoms_per_source_cm3",
                mean / volumes[:, None, None],
                units="atoms/source/cm3",
            )
            _validate_scaled_dataset(
                group,
                "std_dev_atoms_per_source_cm3",
                std_dev / volumes[:, None, None],
                units="atoms/source/cm3",
            )
            _validate_scaled_dataset(
                group,
                "mean_atoms_per_s",
                mean * source_rate,
                units="atoms/s",
            )
            _validate_scaled_dataset(
                group,
                "std_dev_atoms_per_s",
                std_dev * source_rate,
                units="atoms/s",
            )
            _validate_scaled_dataset(
                group,
                "mean_atoms_per_s_cm3",
                mean * source_rate / volumes[:, None, None],
                units="atoms/s/cm3",
            )
            _validate_scaled_dataset(
                group,
                "std_dev_atoms_per_s_cm3",
                std_dev * source_rate / volumes[:, None, None],
                units="atoms/s/cm3",
            )
            gas_totals = {
                score: float(mean[..., index].sum())
                for index, score in enumerate(scores)
            }
        return {
            "schema": SCHEMA,
            "magnets": len(
                set(
                    reference_magnets
                    if reference_magnets is not None
                    else magnets
                )
            ),
            "damage_energy_eV_per_source": damage_total,
            "gas_atoms_per_source": gas_totals,
            "unavailable_responses": [
                name
                for name, available in zip(RESPONSE_NAMES, availability)
                if not available
            ],
        }
