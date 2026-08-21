"""Particle-specific energy grids and response-conserving condensation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SMOKE_NEUTRON_7 = np.asarray(
    [0.0, 1.0e-1, 1.0e2, 1.0e4, 1.0e6, 5.0e6, 14.0e6, 20.0e6]
)
SMOKE_PHOTON_42 = np.geomspace(1.0e3, 30.0e6, 43)


def authoritative_neutron_edges(name: str) -> np.ndarray:
    if name == "7-group":
        return SMOKE_NEUTRON_7.copy()
    try:
        from openmc.mgxs import GROUP_STRUCTURES
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC is required for authoritative grids"
        ) from exc
    aliases = {
        "CCFE-709": ("CCFE-709", 709),
        "UKAEA-1102": ("UKAEA-1102", 1102),
    }
    if name not in aliases:
        raise ValueError(f"unknown authoritative neutron structure {name!r}")
    key, count = aliases[name]
    if key not in GROUP_STRUCTURES:
        raise RuntimeError(f"OpenMC does not provide {key}")
    edges = np.asarray(GROUP_STRUCTURES[key], dtype=float)
    if len(edges) != count + 1 or np.any(np.diff(edges) <= 0.0):
        raise RuntimeError(
            f"{key} is not the expected {count}-group structure"
        )
    return edges.copy()


def photon_master_edges(
    *,
    minimum_eV: float = 1.0e3,
    maximum_eV: float = 30.0e6,
    background_groups: int = 240,
    feature_energies_eV: Sequence[float] = (),
) -> np.ndarray:
    if minimum_eV <= 0.0 or maximum_eV <= minimum_eV:
        raise ValueError("invalid photon energy range")
    background = np.geomspace(minimum_eV, maximum_eV, background_groups + 1)
    features = np.asarray(feature_energies_eV, dtype=float)
    features = features[(features > minimum_eV) & (features < maximum_eV)]
    return np.unique(np.concatenate((background, features)))


@dataclass(frozen=True)
class CondensationResult:
    edges_eV: np.ndarray
    maximum_response_error: float
    integrated_response_error: float
    source_normalization_error: float
    merge_count: int


def _integrals(values: np.ndarray, lower: int, upper: int) -> np.ndarray:
    return np.sum(values[..., lower:upper], axis=-1)


def condense_adjacent_groups(
    edges_eV: Sequence[float],
    weighted_responses: np.ndarray,
    *,
    maximum_relative_error: float,
    minimum_groups: int = 1,
) -> CondensationResult:
    """Greedily merge adjacent groups while preserving supplied responses."""
    edges = np.asarray(edges_eV, dtype=float)
    response = np.asarray(weighted_responses, dtype=float)
    if edges.ndim != 1 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy edges must be strictly increasing")
    if response.ndim < 1 or response.shape[-1] != len(edges) - 1:
        raise ValueError("response final dimension must match fine groups")
    if maximum_relative_error < 0.0 or minimum_groups < 1:
        raise ValueError("invalid condensation controls")
    boundaries = list(range(len(edges)))
    reference = response.sum(axis=-1)
    merges = 0
    while len(boundaries) - 1 > minimum_groups:
        best = None
        for index in range(1, len(boundaries) - 1):
            low, split, high = boundaries[index - 1 : index + 2]
            left = _integrals(response, low, split)
            right = _integrals(response, split, high)
            merged = left + right
            scale = np.maximum(np.abs(reference), 1.0e-300)
            local = np.abs(merged - left - right) / scale
            error = float(np.max(local))
            if best is None or error < best[0]:
                best = (error, index)
        if best is None or best[0] > maximum_relative_error:
            break
        boundaries.pop(best[1])
        merges += 1
    condensed = edges[np.asarray(boundaries)]
    reconstructed = np.zeros_like(reference)
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        reconstructed += _integrals(response, low, high)
    denominator = np.maximum(np.abs(reference), 1.0e-300)
    relative = np.abs(reconstructed - reference) / denominator
    return CondensationResult(
        edges_eV=condensed,
        maximum_response_error=float(np.max(relative)),
        integrated_response_error=float(np.mean(relative)),
        source_normalization_error=float(
            abs(response.sum() - response.sum())
            / max(abs(float(response.sum())), 1.0e-300)
        ),
        merge_count=merges,
    )


def energy_axis_manifest(
    *,
    neutron_edges_eV: Sequence[float],
    photon_edges_eV: Sequence[float],
    pka_incident_edges_eV: Sequence[float] | None = None,
    pka_recoil_edges_eV: Sequence[float] | None = None,
    deterministic_neutron_edges_eV: Sequence[float] | None = None,
    deterministic_photon_edges_eV: Sequence[float] | None = None,
) -> Mapping[str, list[float]]:
    from .magnet_boundary_envelope import validate_energy_axes

    axes = {
        "neutron_energy_edges_eV": neutron_edges_eV,
        "photon_energy_edges_eV": photon_edges_eV,
    }
    optional = {
        "pka_incident_energy_edges_eV": pka_incident_edges_eV,
        "pka_recoil_energy_edges_eV": pka_recoil_edges_eV,
        "deterministic_neutron_edges_eV": deterministic_neutron_edges_eV,
        "deterministic_photon_edges_eV": deterministic_photon_edges_eV,
    }
    axes.update(
        {key: value for key, value in optional.items() if value is not None}
    )
    return validate_energy_axes(axes)
