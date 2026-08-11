"""Deterministic replay of a magnet-boundary source through HTS layers.

The reactor calculation and the local conductor calculation intentionally use
separate spatial scales.  This module consumes the weighted boundary phase
space and its companion OpenMC current tally, reconstructs an absolute
per-source boundary measure, and applies an exact planar characteristics
operator to an explicitly layered conductor.

The operator implemented here is the uncollided/removal component of a larger
multigroup deterministic model.  It is useful as a closure gate and as a
response-conditioned thin-layer operator.  It does not invent scattering,
PKA, or energy-deposition data: non-zero material coefficients must be supplied
in a response library with provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPLAY_SCHEMA = "parastell.hts_multilayer_replay"
REPLAY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class HTSLayer:
    """One explicit layer in a planar coated-conductor stack."""

    name: str
    material: str
    thickness_cm: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("layer name cannot be empty")
        if not self.material.strip():
            raise ValueError("layer material cannot be empty")
        thickness = float(self.thickness_cm)
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("layer thickness_cm must be finite and positive")
        object.__setattr__(self, "thickness_cm", thickness)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "material": self.material,
            "thickness_cm": self.thickness_cm,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class MultilayerStack:
    """Ordered set of layers, listed from reactor side to magnet interior."""

    name: str
    layers: tuple[HTSLayer, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stack name cannot be empty")
        if not self.layers:
            raise ValueError("at least one layer is required")
        names = [layer.name for layer in self.layers]
        if len(names) != len(set(names)):
            raise ValueError("layer names must be unique")
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def total_thickness_cm(self) -> float:
        return float(sum(layer.thickness_cm for layer in self.layers))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layers": [layer.to_dict() for layer in self.layers],
            "total_thickness_cm": self.total_thickness_cm,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class MaterialResponseLibrary:
    """Groupwise material coefficients used by the thin-layer operator.

    ``removal_xs_cm_1`` and ``deposition_fraction`` are nested mappings of
    material -> particle -> one value per energy group.  Removal coefficients
    are macroscopic inverse-centimetre coefficients.  Deposition fractions are
    dimensionless fractions of particle energy assigned when a removal event
    occurs.  They are optional and default to zero.
    """

    energy_bounds_eV: tuple[float, ...]
    removal_xs_cm_1: Mapping[str, Mapping[str, Sequence[float]]]
    deposition_fraction: Mapping[str, Mapping[str, Sequence[float]]] = field(
        default_factory=dict
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bounds = _strictly_increasing(self.energy_bounds_eV, "energy bounds")
        object.__setattr__(self, "energy_bounds_eV", bounds)
        group_count = len(bounds) - 1
        removal = _normalise_response_mapping(
            self.removal_xs_cm_1,
            group_count,
            "removal_xs_cm_1",
            maximum=None,
        )
        deposition = _normalise_response_mapping(
            self.deposition_fraction,
            group_count,
            "deposition_fraction",
            maximum=1.0,
        )
        object.__setattr__(self, "removal_xs_cm_1", removal)
        object.__setattr__(self, "deposition_fraction", deposition)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def group_count(self) -> int:
        return len(self.energy_bounds_eV) - 1

    def coefficient(
        self,
        material: str,
        particle: str,
        group_index: int,
    ) -> tuple[float, float]:
        try:
            removal = self.removal_xs_cm_1[material][particle][group_index]
        except KeyError as exc:
            raise KeyError(
                f"response library has no removal coefficients for "
                f"material={material!r}, particle={particle!r}"
            ) from exc
        deposition = self.deposition_fraction.get(material, {}).get(
            particle, (0.0,) * self.group_count
        )[group_index]
        return float(removal), float(deposition)

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy_bounds_eV": list(self.energy_bounds_eV),
            "removal_xs_cm_1": _jsonable(self.removal_xs_cm_1),
            "deposition_fraction": _jsonable(self.deposition_fraction),
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class ReplaySummary:
    """Scalar closure metrics returned by :func:`replay_phase_space`."""

    input_current_per_source: float
    transmitted_current_per_source: float
    removed_current_per_source: float
    balance_residual_per_source: float
    relative_balance_error: float
    selected_record_count: int
    rejected_record_count: int
    boundary_tally_name: str
    normalization_mode: str
    direction_basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_current_per_source": self.input_current_per_source,
            "transmitted_current_per_source": (
                self.transmitted_current_per_source
            ),
            "removed_current_per_source": self.removed_current_per_source,
            "balance_residual_per_source": self.balance_residual_per_source,
            "relative_balance_error": self.relative_balance_error,
            "selected_record_count": self.selected_record_count,
            "rejected_record_count": self.rejected_record_count,
            "boundary_tally_name": self.boundary_tally_name,
            "normalization_mode": self.normalization_mode,
            "direction_basis": self.direction_basis,
        }


def verification_rebco_stack() -> MultilayerStack:
    """Return an explicit, non-manufacturer-specific REBCO verification stack.

    Thicknesses are representative test values chosen to exercise micron and
    sub-millimetre layers.  They are not a specification for a commercial tape
    and must be replaced by measured values for production calculations.
    """

    micrometre_to_cm = 1.0e-4
    layers = (
        HTSLayer("reactor_side_copper", "copper", 20 * micrometre_to_cm),
        HTSLayer("silver_cap", "silver", 2 * micrometre_to_cm),
        HTSLayer("rebco", "rebco", 1 * micrometre_to_cm),
        HTSLayer("buffer_stack", "buffer", 0.2 * micrometre_to_cm),
        HTSLayer("hastelloy_substrate", "hastelloy", 50 * micrometre_to_cm),
        HTSLayer("backside_copper", "copper", 20 * micrometre_to_cm),
        HTSLayer("solder", "solder", 20 * micrometre_to_cm),
        HTSLayer("insulation", "insulation", 50 * micrometre_to_cm),
    )
    return MultilayerStack(
        name="verification_rebco_coated_conductor",
        layers=layers,
        metadata={
            "classification": "verification_geometry",
            "manufacturer_specific": False,
            "thickness_warning": (
                "Replace with measured conductor architecture before using "
                "the result for design or lifetime prediction."
            ),
        },
    )


def zero_response_library(
    stack: MultilayerStack,
    energy_bounds_eV: Sequence[float],
    particles: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> MaterialResponseLibrary:
    """Create a transparent response library for exact streaming closure."""

    bounds = _strictly_increasing(energy_bounds_eV, "energy bounds")
    zeros = [0.0] * (len(bounds) - 1)
    removal = {
        material: {particle: list(zeros) for particle in particles}
        for material in {layer.material for layer in stack.layers}
    }
    return MaterialResponseLibrary(
        energy_bounds_eV=bounds,
        removal_xs_cm_1=removal,
        metadata={
            "classification": "zero_response_closure",
            "physical_material_model": False,
            **dict(metadata or {}),
        },
    )


def constant_response_library(
    stack: MultilayerStack,
    energy_bounds_eV: Sequence[float],
    particles: Sequence[str],
    removal_xs_cm_1: float,
    *,
    deposition_fraction: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> MaterialResponseLibrary:
    """Create a verification-only constant-coefficient response library."""

    coefficient = float(removal_xs_cm_1)
    fraction = float(deposition_fraction)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("removal_xs_cm_1 must be finite and non-negative")
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("deposition_fraction must lie in [0, 1]")
    bounds = _strictly_increasing(energy_bounds_eV, "energy bounds")
    groups = len(bounds) - 1
    materials = {layer.material for layer in stack.layers}
    removal = {
        material: {particle: [coefficient] * groups for particle in particles}
        for material in materials
    }
    deposition = {
        material: {particle: [fraction] * groups for particle in particles}
        for material in materials
    }
    return MaterialResponseLibrary(
        energy_bounds_eV=bounds,
        removal_xs_cm_1=removal,
        deposition_fraction=deposition,
        metadata={
            "classification": "analytic_verification_only",
            "physical_material_model": False,
            **dict(metadata or {}),
        },
    )


def replay_phase_space(
    phase_space_path: str | Path,
    spectra_path: str | Path,
    output_path: str | Path,
    *,
    stack: MultilayerStack,
    response_library: MaterialResponseLibrary,
    boundary_tally_name: str | None = None,
    tally_direction: str | None = "incoming",
    record_direction: str | None = "incoming",
    per_record_inward_normals_global: np.ndarray | None = None,
    fixed_inward_normal_global: Sequence[float] | None = None,
    allow_local_axis_proxy: bool = False,
    local_inward_axis: int = 2,
    x_edges_cm: Sequence[float] | None = None,
    y_edges_cm: Sequence[float] | None = None,
    minimum_incident_cosine: float = 1.0e-8,
    provenance: Mapping[str, Any] | None = None,
) -> ReplaySummary:
    """Replay a weighted boundary source through explicit HTS layers.

    The companion current tally fixes the absolute current per source particle;
    the source bank supplies the joint distribution within each particle,
    energy, and surface bin.  Bank record count is never used as an absolute
    rate.
    """

    phase_space_path = Path(phase_space_path)
    spectra_path = Path(spectra_path)
    output_path = Path(output_path)
    if not phase_space_path.is_file():
        raise FileNotFoundError(phase_space_path)
    if not spectra_path.is_file():
        raise FileNotFoundError(spectra_path)

    phase, phase_metadata, region_name = _read_phase_space(phase_space_path)
    tally_name, tally = _read_boundary_tally(
        spectra_path,
        region_name=region_name,
        tally_name=boundary_tally_name,
    )
    record_weights, normalization = _normalise_bank_to_current(
        phase,
        tally,
        tally_direction=tally_direction,
    )

    direction = np.asarray(phase["direction_global"], dtype=float)
    mu_inward, direction_basis = _resolve_incident_cosine(
        phase,
        direction,
        per_record_inward_normals_global=per_record_inward_normals_global,
        fixed_inward_normal_global=fixed_inward_normal_global,
        allow_local_axis_proxy=allow_local_axis_proxy,
        local_inward_axis=local_inward_axis,
    )

    record_mask = np.isfinite(record_weights) & (record_weights > 0.0)
    record_mask &= np.isfinite(mu_inward)
    record_mask &= mu_inward > float(minimum_incident_cosine)
    if record_direction is not None and "magnet_direction" in phase:
        labels = _as_strings(phase["magnet_direction"])
        record_mask &= labels == record_direction

    particle_names = _as_strings(phase["particle_name"])
    energy_eV = np.asarray(phase["energy_eV"], dtype=float)
    group_index = _energy_group_indices(
        energy_eV, response_library.energy_bounds_eV
    )
    record_mask &= group_index >= 0

    particles = tuple(sorted(set(particle_names[record_mask].tolist())))
    if not particles:
        raise RuntimeError("no phase-space records satisfy replay selection")
    particle_index = {name: index for index, name in enumerate(particles)}

    positions = np.asarray(phase["position_local_cm"], dtype=float)
    x_edges = _spatial_edges(positions[:, 0], x_edges_cm)
    y_edges = _spatial_edges(positions[:, 1], y_edges_cm)
    x_index = np.searchsorted(x_edges, positions[:, 0], side="right") - 1
    y_index = np.searchsorted(y_edges, positions[:, 1], side="right") - 1
    x_index[positions[:, 0] == x_edges[-1]] = len(x_edges) - 2
    y_index[positions[:, 1] == y_edges[-1]] = len(y_edges) - 2
    record_mask &= (x_index >= 0) & (x_index < len(x_edges) - 1)
    record_mask &= (y_index >= 0) & (y_index < len(y_edges) - 1)

    shape = (
        len(x_edges) - 1,
        len(y_edges) - 1,
        len(stack.layers),
        len(particles),
        response_library.group_count,
    )
    incident = np.zeros(shape, dtype=float)
    transmitted = np.zeros(shape, dtype=float)
    removed = np.zeros(shape, dtype=float)
    track_length = np.zeros(shape, dtype=float)
    deposited_energy = np.zeros(shape, dtype=float)

    selected_indices = np.flatnonzero(record_mask)
    final_record_weight = np.zeros(len(record_weights), dtype=float)
    for record_index in selected_indices:
        particle = particle_names[record_index]
        particle_bin = particle_index[particle]
        energy_bin = int(group_index[record_index])
        ix = int(x_index[record_index])
        iy = int(y_index[record_index])
        cosine = float(mu_inward[record_index])
        energy = float(energy_eV[record_index])
        current = float(record_weights[record_index])

        for layer_index, layer in enumerate(stack.layers):
            removal_xs, deposition = response_library.coefficient(
                layer.material, particle, energy_bin
            )
            path_length = layer.thickness_cm / cosine
            optical_depth = removal_xs * path_length
            attenuation = float(np.exp(-optical_depth))
            outgoing = current * attenuation
            removed_here = current - outgoing
            if removal_xs > 0.0:
                track_here = current * (-np.expm1(-optical_depth)) / removal_xs
            else:
                track_here = current * path_length

            index = (ix, iy, layer_index, particle_bin, energy_bin)
            incident[index] += current
            transmitted[index] += outgoing
            removed[index] += removed_here
            track_length[index] += track_here
            deposited_energy[index] += removed_here * energy * deposition
            current = outgoing

        final_record_weight[record_index] = current

    input_current = float(np.sum(record_weights[selected_indices]))
    final_current = float(np.sum(final_record_weight[selected_indices]))
    removed_current = float(np.sum(removed))
    residual = input_current - final_current - removed_current
    denominator = max(abs(input_current), np.finfo(float).tiny)
    relative_error = abs(residual) / denominator

    summary = ReplaySummary(
        input_current_per_source=input_current,
        transmitted_current_per_source=final_current,
        removed_current_per_source=removed_current,
        balance_residual_per_source=residual,
        relative_balance_error=relative_error,
        selected_record_count=len(selected_indices),
        rejected_record_count=len(record_weights) - len(selected_indices),
        boundary_tally_name=tally_name,
        normalization_mode=normalization["mode"],
        direction_basis=direction_basis,
    )

    manifest = {
        "schema": REPLAY_SCHEMA,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_phase_space": str(phase_space_path),
        "source_spectra": str(spectra_path),
        "source_region": region_name,
        "phase_space_metadata": phase_metadata,
        "boundary_tally_name": tally_name,
        "normalization": normalization,
        "direction_basis": direction_basis,
        "minimum_incident_cosine": minimum_incident_cosine,
        "record_direction": record_direction,
        "tally_direction": tally_direction,
        "stack": stack.to_dict(),
        "response_library": response_library.to_dict(),
        "summary": summary.to_dict(),
        "operator_scope": {
            "implemented": (
                "exact planar characteristics for uncollided/removal transport"
            ),
            "not_implemented": [
                "within-layer scattering redistribution",
                "secondary-particle production",
                "PKA recoil matrices",
                "temperature-dependent defect retention",
            ],
        },
        "provenance": _jsonable(dict(provenance or {})),
    }
    _write_replay_output(
        output_path,
        manifest=manifest,
        stack=stack,
        particles=particles,
        energy_bounds_eV=response_library.energy_bounds_eV,
        x_edges_cm=x_edges,
        y_edges_cm=y_edges,
        incident=incident,
        transmitted=transmitted,
        removed=removed,
        track_length=track_length,
        deposited_energy=deposited_energy,
        phase=phase,
        record_weights=record_weights,
        final_record_weight=final_record_weight,
        mu_inward=mu_inward,
        record_mask=record_mask,
        x_index=x_index,
        y_index=y_index,
        group_index=group_index,
    )
    return summary


def analytic_transmission(
    incident_current: float,
    removal_xs_cm_1: float,
    thickness_cm: float,
    incident_cosine: float,
) -> float:
    """Return the exact planar uncollided current after one homogeneous layer."""

    if incident_cosine <= 0.0:
        raise ValueError("incident_cosine must be positive")
    if removal_xs_cm_1 < 0.0:
        raise ValueError("removal_xs_cm_1 must be non-negative")
    if thickness_cm < 0.0:
        raise ValueError("thickness_cm must be non-negative")
    return float(
        incident_current
        * np.exp(-removal_xs_cm_1 * thickness_cm / incident_cosine)
    )


def _read_phase_space(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    with h5py.File(path, "r") as source:
        if "phase_space" not in source:
            raise ValueError(f"{path} has no phase_space group")
        phase = {
            name: dataset[()]
            for name, dataset in source["phase_space"].items()
        }
        region = str(source.attrs.get("region", ""))
        metadata: dict[str, Any] = {}
        if "metadata_json" in source:
            metadata = json.loads(source["metadata_json"].asstr()[()])
    required = {
        "position_local_cm",
        "direction_global",
        "direction_local",
        "energy_eV",
        "weight",
        "surface_id_abs",
        "particle_name",
    }
    missing = required.difference(phase)
    if missing:
        raise ValueError(
            "phase-space file is missing required fields: "
            + ", ".join(sorted(missing))
        )
    return phase, metadata, region


def _read_boundary_tally(
    path: Path,
    *,
    region_name: str,
    tally_name: str | None,
) -> tuple[str, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as source:
        if "tallies" not in source:
            raise ValueError(f"{path} has no tallies group")
        candidates: list[str] = []
        for name, group in source["tallies"].items():
            role = str(group.attrs.get("role", ""))
            region = str(group.attrs.get("region", ""))
            if role == "boundary_current" and (
                not region_name or region == region_name
            ):
                candidates.append(name)
        if tally_name is None:
            if len(candidates) != 1:
                raise ValueError(
                    "expected one matching boundary-current tally, found "
                    f"{candidates}"
                )
            tally_name = candidates[0]
        if tally_name not in source["tallies"]:
            raise KeyError(f"boundary tally {tally_name!r} not found")
        group = source["tallies"][tally_name]
        if str(group.attrs.get("role", "")) != "boundary_current":
            raise ValueError(f"tally {tally_name!r} is not boundary_current")
        table = {name: dataset[()] for name, dataset in group.items()}
    return tally_name, table


def _normalise_bank_to_current(
    phase: Mapping[str, np.ndarray],
    tally: Mapping[str, np.ndarray],
    *,
    tally_direction: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale bank weights to current in particle/energy/surface bins."""

    mean_name = _find_column(tally, "mean")
    particle_name = _find_column(tally, "particle")
    surface_name = _find_column(tally, "surface")
    energy_low_name = _find_column(tally, "energy_low_ev")
    energy_high_name = _find_column(tally, "energy_high_ev")
    if None in {
        mean_name,
        particle_name,
        surface_name,
        energy_low_name,
        energy_high_name,
    }:
        raise ValueError(
            "boundary-current tally lacks particle/surface/energy/mean columns"
        )

    row_mask = np.ones(len(tally[mean_name]), dtype=bool)
    direction_name = _find_column(tally, "magnet_direction")
    if tally_direction is not None:
        if direction_name is None:
            raise ValueError(
                "tally_direction was requested but tally has no "
                "magnet_direction labels"
            )
        row_mask &= _as_strings(tally[direction_name]) == tally_direction

    target_by_bin: dict[tuple[str, int, float, float], float] = {}
    for index in np.flatnonzero(row_mask):
        key = (
            str(_as_strings(tally[particle_name])[index]),
            abs(int(tally[surface_name][index])),
            float(tally[energy_low_name][index]),
            float(tally[energy_high_name][index]),
        )
        target_by_bin[key] = target_by_bin.get(key, 0.0) + abs(
            float(tally[mean_name][index])
        )

    particle = _as_strings(phase["particle_name"])
    surface = np.asarray(phase["surface_id_abs"], dtype=int)
    energy = np.asarray(phase["energy_eV"], dtype=float)
    raw_weight = np.asarray(phase["weight"], dtype=float)
    result = np.zeros(len(raw_weight), dtype=float)
    records_by_bin: dict[tuple[str, int, float, float], list[int]] = {}
    for key in target_by_bin:
        name, surface_id, lower, upper = key
        mask = particle == name
        mask &= surface == surface_id
        mask &= energy >= lower
        if upper == max(item[3] for item in target_by_bin):
            mask &= energy <= upper
        else:
            mask &= energy < upper
        records_by_bin[key] = np.flatnonzero(mask).tolist()

    diagnostics: list[dict[str, Any]] = []
    for key, target in target_by_bin.items():
        indices = np.asarray(records_by_bin[key], dtype=int)
        raw_total = float(np.sum(raw_weight[indices])) if len(indices) else 0.0
        if target > 0.0 and raw_total <= 0.0:
            raise RuntimeError(
                "non-zero boundary-current bin has no sampled source records: "
                f"{key}"
            )
        scale = target / raw_total if raw_total > 0.0 else 0.0
        result[indices] += raw_weight[indices] * scale
        diagnostics.append(
            {
                "particle": key[0],
                "surface_id": key[1],
                "energy_low_eV": key[2],
                "energy_high_eV": key[3],
                "target_current_per_source": target,
                "bank_weight_sum": raw_total,
                "scale": scale,
                "record_count": int(len(indices)),
            }
        )

    unmatched_positive = (raw_weight > 0.0) & (result == 0.0)
    return result, {
        "mode": "particle_energy_surface_current_conditioning",
        "tally_direction": tally_direction,
        "bins": diagnostics,
        "target_current_per_source": float(sum(target_by_bin.values())),
        "normalised_record_current_per_source": float(np.sum(result)),
        "unmatched_positive_weight_records": int(
            np.count_nonzero(unmatched_positive)
        ),
        "bank_record_count_is_not_an_absolute_rate": True,
    }


def _resolve_incident_cosine(
    phase: Mapping[str, np.ndarray],
    direction_global: np.ndarray,
    *,
    per_record_inward_normals_global: np.ndarray | None,
    fixed_inward_normal_global: Sequence[float] | None,
    allow_local_axis_proxy: bool,
    local_inward_axis: int,
) -> tuple[np.ndarray, str]:
    record_count = len(direction_global)
    if per_record_inward_normals_global is not None:
        normals = _normalise_vectors(
            per_record_inward_normals_global,
            expected_rows=record_count,
            name="per-record inward normals",
        )
        return (
            np.einsum("ij,ij->i", direction_global, normals),
            "per_record_inward_normal_global",
        )
    if fixed_inward_normal_global is not None:
        normal = _normalise_vector(
            fixed_inward_normal_global, "fixed inward normal"
        )
        return direction_global @ normal, "fixed_inward_normal_global"
    if "mu_outward" in phase:
        mu_outward = np.asarray(phase["mu_outward"], dtype=float)
        if np.all(np.isfinite(mu_outward)):
            return -mu_outward, "phase_space_mu_outward"
    if allow_local_axis_proxy:
        if local_inward_axis not in {0, 1, 2}:
            raise ValueError("local_inward_axis must be 0, 1, or 2")
        direction_local = np.asarray(phase["direction_local"], dtype=float)
        return (
            np.abs(direction_local[:, local_inward_axis]),
            "absolute_local_axis_proxy",
        )
    raise ValueError(
        "incident cosine is unavailable; supply per-record normals, a fixed "
        "normal, or explicitly allow the local-axis proxy"
    )


def _write_replay_output(
    output_path: Path,
    *,
    manifest: Mapping[str, Any],
    stack: MultilayerStack,
    particles: Sequence[str],
    energy_bounds_eV: Sequence[float],
    x_edges_cm: np.ndarray,
    y_edges_cm: np.ndarray,
    incident: np.ndarray,
    transmitted: np.ndarray,
    removed: np.ndarray,
    track_length: np.ndarray,
    deposited_energy: np.ndarray,
    phase: Mapping[str, np.ndarray],
    record_weights: np.ndarray,
    final_record_weight: np.ndarray,
    mu_inward: np.ndarray,
    record_mask: np.ndarray,
    x_index: np.ndarray,
    y_index: np.ndarray,
    group_index: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["schema"] = REPLAY_SCHEMA
            output.attrs["schema_version"] = REPLAY_SCHEMA_VERSION
            output.create_dataset(
                "manifest_json",
                data=json.dumps(_jsonable(manifest), sort_keys=True),
                dtype=string_dtype,
            )
            axes = output.create_group("axes")
            axes.create_dataset("x_edges_cm", data=x_edges_cm)
            axes.create_dataset("y_edges_cm", data=y_edges_cm)
            axes.create_dataset("energy_bounds_eV", data=energy_bounds_eV)
            axes.create_dataset(
                "particle",
                data=np.asarray(particles, dtype=object),
                dtype=string_dtype,
            )
            axes.create_dataset(
                "layer_name",
                data=np.asarray(
                    [layer.name for layer in stack.layers], dtype=object
                ),
                dtype=string_dtype,
            )
            axes.create_dataset(
                "layer_material",
                data=np.asarray(
                    [layer.material for layer in stack.layers], dtype=object
                ),
                dtype=string_dtype,
            )
            axes.create_dataset(
                "layer_thickness_cm",
                data=np.asarray(
                    [layer.thickness_cm for layer in stack.layers],
                    dtype=float,
                ),
            )

            response = output.create_group("response")
            response.create_dataset(
                "incident_current_per_source", data=incident
            )
            response.create_dataset(
                "transmitted_current_per_source", data=transmitted
            )
            response.create_dataset("removed_current_per_source", data=removed)
            response.create_dataset(
                "track_length_cm_per_source", data=track_length
            )
            response.create_dataset(
                "deposited_energy_eV_per_source", data=deposited_energy
            )
            response.attrs["dimension_order"] = (
                "x_bin,y_bin,layer,particle,energy_group"
            )

            records = output.create_group("records")
            records.create_dataset(
                "record_id",
                data=np.asarray(
                    phase.get("record_id", np.arange(len(record_weights))),
                    dtype=int,
                ),
            )
            records.create_dataset(
                "normalised_current_per_source", data=record_weights
            )
            records.create_dataset(
                "transmitted_current_per_source", data=final_record_weight
            )
            records.create_dataset("mu_inward", data=mu_inward)
            records.create_dataset("selected", data=record_mask)
            records.create_dataset("x_bin", data=x_index)
            records.create_dataset("y_bin", data=y_index)
            records.create_dataset("energy_group", data=group_index)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _normalise_response_mapping(
    mapping: Mapping[str, Mapping[str, Sequence[float]]],
    group_count: int,
    name: str,
    maximum: float | None,
) -> dict[str, dict[str, tuple[float, ...]]]:
    result: dict[str, dict[str, tuple[float, ...]]] = {}
    for material, particles in mapping.items():
        if not str(material).strip():
            raise ValueError(f"{name} contains an empty material name")
        result[str(material)] = {}
        for particle, values in particles.items():
            array = np.asarray(values, dtype=float)
            if array.shape != (group_count,):
                raise ValueError(
                    f"{name}[{material!r}][{particle!r}] must have "
                    f"{group_count} values"
                )
            if not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"{name} coefficients must be non-negative")
            if maximum is not None and np.any(array > maximum):
                raise ValueError(f"{name} coefficients exceed {maximum}")
            result[str(material)][str(particle)] = tuple(
                float(value) for value in array
            )
    return result


def _energy_group_indices(
    energy_eV: np.ndarray, bounds: Sequence[float]
) -> np.ndarray:
    bounds_array = np.asarray(bounds, dtype=float)
    indices = np.searchsorted(bounds_array, energy_eV, side="right") - 1
    indices[energy_eV == bounds_array[-1]] = len(bounds_array) - 2
    invalid = (indices < 0) | (indices >= len(bounds_array) - 1)
    indices[invalid] = -1
    return indices.astype(int)


def _spatial_edges(
    values: np.ndarray, provided: Sequence[float] | None
) -> np.ndarray:
    if provided is not None:
        return np.asarray(_strictly_increasing(provided, "spatial edges"))
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.asarray([-0.5, 0.5], dtype=float)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower == upper:
        margin = max(1.0e-12, abs(lower) * 1.0e-12)
        return np.asarray([lower - margin, upper + margin], dtype=float)
    margin = (upper - lower) * 1.0e-12
    return np.asarray([lower - margin, upper + margin], dtype=float)


def _strictly_increasing(
    values: Sequence[float], name: str
) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if not np.all(np.isfinite(array)) or np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return tuple(float(value) for value in array)


def _normalise_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} cannot be zero")
    return vector / norm


def _normalise_vectors(
    values: np.ndarray, *, expected_rows: int, name: str
) -> np.ndarray:
    vectors = np.asarray(values, dtype=float)
    if vectors.shape != (expected_rows, 3):
        raise ValueError(
            f"{name} must have shape ({expected_rows}, 3), got "
            f"{vectors.shape}"
        )
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(vectors)) or np.any(norms <= 0.0):
        raise ValueError(f"{name} contains invalid or zero vectors")
    return vectors / norms[:, None]


def _find_column(
    table: Mapping[str, np.ndarray], *candidates: str
) -> str | None:
    normalised = {
        name: "".join(character for character in name if character.isalnum())
        for name in table
    }
    for candidate in candidates:
        candidate_key = "".join(
            character for character in candidate if character.isalnum()
        )
        for name, key in normalised.items():
            if key == candidate_key or candidate_key in key:
                return name
    return None


def _as_strings(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    if array.dtype.kind == "O":
        return np.asarray(
            [
                (
                    value.decode("utf-8")
                    if isinstance(value, (bytes, np.bytes_))
                    else str(value)
                )
                for value in array
            ],
            dtype=str,
        )
    return array.astype(str)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
