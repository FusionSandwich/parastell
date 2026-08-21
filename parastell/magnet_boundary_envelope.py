"""Closed magnet-envelope and correlated boundary-source contract.

This module is the production extension of the plane-oriented v1 handoff.  A
closed envelope is made from every DAGMC surface bounding one magnet volume.
The canonical payload is a correlated list of crossing records; histograms are
derived products and are never independently normalised marginals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


SCHEMA_NAME = "parastell.magnet_boundary_source"
SCHEMA_VERSION = "2.0.0"
SCHEMA_URI = f"{SCHEMA_NAME}/v{SCHEMA_VERSION}"
PARTICLE_PDG = {"neutron": 2112, "photon": 22}
ENERGY_AXIS_NAMES = (
    "neutron_energy_edges_eV",
    "photon_energy_edges_eV",
    "pka_incident_energy_edges_eV",
    "pka_recoil_energy_edges_eV",
    "deterministic_neutron_edges_eV",
    "deterministic_photon_edges_eV",
)
REQUIRED_RECORD_FIELDS = (
    "record_id",
    "history_id",
    "position_global_cm",
    "position_local_cm",
    "direction_global",
    "direction_local",
    "outward_normal_global",
    "energy_eV",
    "weight",
    "weight_std_dev",
    "particle",
    "particle_pdg",
    "surface_id",
    "envelope_id",
    "crossing_sense",
    "surface_role",
    "time_s",
    "mu",
    "azimuth_rad",
    "grazing",
    "patch_id",
    "energy_group",
    "angle_bin_id",
)


def _unit(vector: Sequence[float], name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError(f"{name} cannot be zero")
    return value / norm


def _edges(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) < 2:
        raise ValueError(f"{name} must contain at least two edges")
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return result


def _hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authoritative_energy_edges(name: str = "CCFE-709") -> np.ndarray:
    """Return an exact OpenMC group structure; never synthesize CCFE-709."""
    try:
        from openmc.mgxs import GROUP_STRUCTURES
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC is required to resolve authoritative group structures"
        ) from exc
    if name not in GROUP_STRUCTURES:
        raise RuntimeError(
            f"authoritative OpenMC energy structure {name!r} is unavailable"
        )
    values = _edges(GROUP_STRUCTURES[name], f"GROUP_STRUCTURES[{name!r}]")
    if name == "CCFE-709" and len(values) != 710:
        raise RuntimeError(
            "OpenMC CCFE-709 does not contain exactly 709 groups"
        )
    return values.copy()


def production_mu_edges() -> np.ndarray:
    """Twenty-six bins, including six dedicated grazing bins in |mu| <= .1."""
    return np.asarray(
        [
            -1.0,
            -0.9,
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            -2 / 30,
            -1 / 30,
            0.0,
            1 / 30,
            2 / 30,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.975,
            1.0,
        ],
        dtype=float,
    )


def production_phi_edges() -> np.ndarray:
    return np.linspace(-np.pi, np.pi, 17)


@dataclass(frozen=True)
class EnvelopeSurface:
    surface_id: int
    role: str
    area_cm2: float
    centroid_global_cm: tuple[float, float, float]
    outward_normal_global: tuple[float, float, float]
    toroidal_direction_global: tuple[float, float, float]
    poloidal_direction_global: tuple[float, float, float]
    u_edges_cm: tuple[float, ...]
    v_edges_cm: tuple[float, ...]
    openmc_normal_sign: int = 1
    topology_edge_ids: tuple[str, ...] = ()
    vector_area_global_cm2: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if int(self.surface_id) <= 0:
            raise ValueError("surface_id must be positive")
        if not self.role:
            raise ValueError("surface role cannot be empty")
        if not np.isfinite(self.area_cm2) or self.area_cm2 <= 0.0:
            raise ValueError("surface area must be positive and finite")
        n = _unit(self.outward_normal_global, "outward normal")
        t = _unit(self.toroidal_direction_global, "toroidal direction")
        p = _unit(self.poloidal_direction_global, "poloidal direction")
        if max(abs(np.dot(n, t)), abs(np.dot(n, p)), abs(np.dot(t, p))) > 1e-8:
            raise ValueError("surface frame must be orthogonal")
        if np.dot(np.cross(t, p), n) < 1.0 - 1e-8:
            raise ValueError("surface frame must be right handed: t x p = n")
        _edges(self.u_edges_cm, "u_edges_cm")
        _edges(self.v_edges_cm, "v_edges_cm")
        if self.openmc_normal_sign not in (-1, 1):
            raise ValueError("openmc_normal_sign must be -1 or +1")
        if self.vector_area_global_cm2 is not None:
            vector_area = np.asarray(self.vector_area_global_cm2, dtype=float)
            if vector_area.shape != (3,) or not np.all(
                np.isfinite(vector_area)
            ):
                raise ValueError(
                    "vector_area_global_cm2 must be a finite vector"
                )

    @property
    def frame(self) -> np.ndarray:
        return np.vstack(
            (
                self.toroidal_direction_global,
                self.poloidal_direction_global,
                self.outward_normal_global,
            )
        )

    def local_position(self, positions: np.ndarray) -> np.ndarray:
        return (positions - np.asarray(self.centroid_global_cm)) @ self.frame.T

    def local_direction(self, directions: np.ndarray) -> np.ndarray:
        return directions @ self.frame.T

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "role": self.role,
            "area_cm2": self.area_cm2,
            "centroid_global_cm": list(self.centroid_global_cm),
            "outward_normal_global": list(self.outward_normal_global),
            "toroidal_direction_global": list(self.toroidal_direction_global),
            "poloidal_direction_global": list(self.poloidal_direction_global),
            "u_edges_cm": list(self.u_edges_cm),
            "v_edges_cm": list(self.v_edges_cm),
            "openmc_normal_sign": self.openmc_normal_sign,
            "topology_edge_ids": list(self.topology_edge_ids),
            "vector_area_global_cm2": (
                list(self.vector_area_global_cm2)
                if self.vector_area_global_cm2 is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvelopeSurface":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class MagnetBoundaryEnvelope:
    envelope_id: str
    magnet_component: str
    dagmc_volume_id: int
    surfaces: tuple[EnvelopeSurface, ...]
    dagmc_geometry_sha256: str
    units: str = "cm"
    watertight: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.envelope_id or not self.magnet_component:
            raise ValueError("envelope and magnet identifiers cannot be empty")
        if self.dagmc_volume_id <= 0 or not self.surfaces:
            raise ValueError(
                "a closed envelope requires a positive DAGMC volume and surfaces"
            )
        ids = [item.surface_id for item in self.surfaces]
        if len(ids) != len(set(ids)):
            raise ValueError("closed envelope contains duplicate surface IDs")
        if self.units != "cm":
            raise ValueError("production envelope geometry units must be cm")
        if not self.watertight:
            raise ValueError("magnet boundary envelope is not watertight")
        edge_counts: dict[str, int] = {}
        for surface in self.surfaces:
            for edge in surface.topology_edge_ids:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        bad = {key: count for key, count in edge_counts.items() if count != 2}
        if edge_counts and bad:
            raise ValueError(f"non-manifold/open envelope edges: {bad}")
        vector_area = sum(
            (
                (
                    np.asarray(item.vector_area_global_cm2)
                    if item.vector_area_global_cm2 is not None
                    else item.area_cm2 * np.asarray(item.outward_normal_global)
                )
                for item in self.surfaces
            ),
            start=np.zeros(3),
        )
        total_area = sum(item.area_cm2 for item in self.surfaces)
        if np.linalg.norm(vector_area) > max(1e-7 * total_area, 1e-8):
            raise ValueError(
                "surface normals do not close the envelope vector area"
            )

    @property
    def surface_ids(self) -> tuple[int, ...]:
        return tuple(item.surface_id for item in self.surfaces)

    def surface(self, surface_id: int) -> EnvelopeSurface:
        for item in self.surfaces:
            if item.surface_id == int(surface_id):
                return item
        raise KeyError(
            f"surface {surface_id} is not part of envelope {self.envelope_id}"
        )

    def classify(
        self,
        surface_id: int,
        direction: Sequence[float],
        tolerance: float = 1e-12,
    ) -> tuple[float, str, bool]:
        mu = float(
            np.dot(
                _unit(direction, "particle direction"),
                self.surface(surface_id).outward_normal_global,
            )
        )
        if abs(mu) <= tolerance:
            return mu, "grazing", True
        return mu, "outgoing" if mu > 0.0 else "incoming", False

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "magnet_component": self.magnet_component,
            "dagmc_volume_id": self.dagmc_volume_id,
            "dagmc_geometry_sha256": self.dagmc_geometry_sha256,
            "units": self.units,
            "watertight": self.watertight,
            "surfaces": [item.to_dict() for item in self.surfaces],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MagnetBoundaryEnvelope":
        return cls(
            envelope_id=value["envelope_id"],
            magnet_component=value["magnet_component"],
            dagmc_volume_id=int(value["dagmc_volume_id"]),
            dagmc_geometry_sha256=value["dagmc_geometry_sha256"],
            units=value.get("units", "cm"),
            watertight=bool(value["watertight"]),
            surfaces=tuple(
                EnvelopeSurface.from_dict(item) for item in value["surfaces"]
            ),
            metadata=value.get("metadata", {}),
        )


@dataclass
class CorrelatedBoundaryBank:
    columns: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(REQUIRED_RECORD_FIELDS) - set(self.columns)
        if missing:
            raise ValueError(
                f"canonical bank is missing fields: {sorted(missing)}"
            )
        lengths = {len(np.asarray(value)) for value in self.columns.values()}
        if len(lengths) != 1:
            raise ValueError(
                "canonical bank columns have inconsistent lengths"
            )
        n = lengths.pop()
        for name in (
            "position_global_cm",
            "position_local_cm",
            "direction_global",
            "direction_local",
        ):
            if np.asarray(self.columns[name]).shape != (n, 3):
                raise ValueError(f"{name} must have shape (records, 3)")
        if np.any(~np.isfinite(self.columns["weight"])) or np.any(
            self.columns["weight"] < 0.0
        ):
            raise ValueError("bank weights must be finite and nonnegative")

    def __len__(self) -> int:
        return len(self.columns["record_id"])

    @property
    def integrated_current(self) -> float:
        return float(np.sum(self.columns["weight"]))

    def angular_metrics(self) -> dict[str, float]:
        weight = np.asarray(self.columns["weight"], dtype=float)
        mu = np.asarray(self.columns["mu"], dtype=float)
        total = float(weight.sum())
        if total <= 0.0:
            return {
                key: 0.0
                for key in (
                    "mean_mu",
                    "p2",
                    "forward_fraction",
                    "backward_fraction",
                    "grazing_fraction",
                )
            }
        return {
            "mean_mu": float(np.dot(weight, mu) / total),
            "p2": float(np.dot(weight, 0.5 * (3.0 * mu * mu - 1.0)) / total),
            "forward_fraction": float(weight[mu > 0.0].sum() / total),
            "backward_fraction": float(weight[mu < 0.0].sum() / total),
            "grazing_fraction": float(weight[np.abs(mu) <= 0.1].sum() / total),
        }


@dataclass(frozen=True)
class IndependentClosure:
    """Comparison of independently accumulated estimators without scaling."""

    tally_mean: float
    tally_std_dev: float
    bank_mean: float
    bank_std_dev: float
    projection_mean: float
    projection_std_dev: float
    replay_mean: float
    replay_std_dev: float
    numerical_tolerance: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.tally_mean,
                self.tally_std_dev,
                self.bank_mean,
                self.bank_std_dev,
                self.projection_mean,
                self.projection_std_dev,
                self.replay_mean,
                self.replay_std_dev,
                self.numerical_tolerance,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values[1::2] < 0.0):
            raise ValueError(
                "closure values must be finite with nonnegative uncertainty"
            )

    def comparison(self, other: str) -> dict[str, float | bool]:
        means = {
            "bank": (self.bank_mean, self.bank_std_dev),
            "projection": (self.projection_mean, self.projection_std_dev),
            "replay": (self.replay_mean, self.replay_std_dev),
        }
        if other not in means:
            raise ValueError(f"unknown closure estimator {other!r}")
        mean, sigma = means[other]
        difference = mean - self.tally_mean
        combined = float(np.hypot(self.tally_std_dev, sigma))
        allowance = 3.0 * combined + self.numerical_tolerance
        z_score = difference / combined if combined > 0.0 else np.inf
        if difference == 0.0 and combined == 0.0:
            z_score = 0.0
        return {
            "difference": float(difference),
            "combined_std_dev": combined,
            "z_score": float(z_score),
            "passes": bool(abs(difference) <= allowance),
        }

    @property
    def passes(self) -> bool:
        return all(
            bool(self.comparison(name)["passes"])
            for name in ("bank", "projection", "replay")
        )


def validate_energy_axes(
    axes: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    """Validate particle-specific energy dimensions for the v2 contract."""
    unknown = set(axes) - set(ENERGY_AXIS_NAMES)
    if unknown:
        raise ValueError(f"unknown energy axes: {sorted(unknown)}")
    required = {"neutron_energy_edges_eV", "photon_energy_edges_eV"}
    missing = required - set(axes)
    if missing:
        raise ValueError(f"missing particle energy axes: {sorted(missing)}")
    return {
        name: _edges(values, name).tolist() for name, values in axes.items()
    }


def build_correlated_bank(
    envelope: MagnetBoundaryEnvelope,
    *,
    position_global_cm: np.ndarray,
    direction_global: np.ndarray,
    energy_eV: Sequence[float],
    raw_weight: Sequence[float],
    particle: Sequence[str],
    surface_id: Sequence[int],
    time_s: Sequence[float] | None = None,
    history_id: Sequence[int] | None = None,
    energy_edges_eV: Sequence[float] | None = None,
    energy_edges_by_particle: Mapping[str, Sequence[float]] | None = None,
    mu_edges: Sequence[float] | None = None,
    phi_edges_rad: Sequence[float] | None = None,
    outward_normal_global: np.ndarray | None = None,
) -> CorrelatedBoundaryBank:
    positions = np.asarray(position_global_cm, dtype=float)
    directions = np.asarray(direction_global, dtype=float)
    energies = np.asarray(energy_eV, dtype=float)
    weights = np.asarray(raw_weight, dtype=float)
    particles = np.asarray(particle, dtype=object).astype(str)
    surfaces = np.asarray(surface_id, dtype=int)
    n = len(energies)
    if positions.shape != (n, 3) or directions.shape != (n, 3):
        raise ValueError(
            "position and direction arrays must have shape (records, 3)"
        )
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("particle directions cannot be zero")
    directions = directions / norms[:, None]
    if energy_edges_by_particle is not None and energy_edges_eV is not None:
        raise ValueError(
            "provide either a shared or particle-specific energy grid"
        )
    if energy_edges_by_particle is None:
        if energy_edges_eV is None:
            raise ValueError("an energy grid is required")
        particle_edges = {
            name: _edges(energy_edges_eV, "energy_edges_eV")
            for name in set(particles)
        }
    else:
        missing_particles = set(particles) - set(energy_edges_by_particle)
        if missing_particles:
            raise ValueError(
                f"missing energy grids for particles: {sorted(missing_particles)}"
            )
        particle_edges = {
            name: _edges(values, f"{name}_energy_edges_eV")
            for name, values in energy_edges_by_particle.items()
        }
    m_edges = _edges(
        mu_edges if mu_edges is not None else production_mu_edges(), "mu_edges"
    )
    p_edges = _edges(
        phi_edges_rad if phi_edges_rad is not None else production_phi_edges(),
        "phi_edges_rad",
    )
    local_position = np.empty_like(positions)
    local_direction = np.empty_like(directions)
    record_normals = np.empty_like(directions)
    roles = np.empty(n, dtype=object)
    sense = np.empty(n, dtype=object)
    grazing = np.zeros(n, dtype=bool)
    patch = np.full(n, -1, dtype=int)
    for sid in np.unique(surfaces):
        face = envelope.surface(int(sid))
        mask = surfaces == sid
        local_position[mask] = face.local_position(positions[mask])
        if outward_normal_global is None:
            normals_here = np.repeat(
                np.asarray(face.outward_normal_global)[None, :],
                int(np.count_nonzero(mask)),
                axis=0,
            )
        else:
            normals_here = np.asarray(outward_normal_global, dtype=float)[mask]
            normal_lengths = np.linalg.norm(normals_here, axis=1)
            if np.any(normal_lengths <= 0.0):
                raise ValueError("per-record outward normals cannot be zero")
            normals_here = normals_here / normal_lengths[:, None]
        record_normals[mask] = normals_here
        toroidal = np.asarray(face.toroidal_direction_global, dtype=float)
        tangents = (
            toroidal
            - np.sum(toroidal[None, :] * normals_here, axis=1)[:, None]
            * normals_here
        )
        tangent_lengths = np.linalg.norm(tangents, axis=1)
        if np.any(tangent_lengths <= 1.0e-12):
            raise ValueError(
                "surface toroidal direction is parallel to a normal"
            )
        tangents /= tangent_lengths[:, None]
        poloidal = np.cross(normals_here, tangents)
        local_direction[mask] = np.column_stack(
            (
                np.sum(directions[mask] * tangents, axis=1),
                np.sum(directions[mask] * poloidal, axis=1),
                np.sum(directions[mask] * normals_here, axis=1),
            )
        )
        roles[mask] = face.role
        mu_here = local_direction[mask, 2]
        sense[mask] = np.where(
            mu_here > 1e-12,
            "outgoing",
            np.where(mu_here < -1e-12, "incoming", "grazing"),
        )
        grazing[mask] = np.abs(mu_here) <= 0.1
        iu = (
            np.searchsorted(
                face.u_edges_cm, local_position[mask, 0], side="right"
            )
            - 1
        )
        iv = (
            np.searchsorted(
                face.v_edges_cm, local_position[mask, 1], side="right"
            )
            - 1
        )
        u_values = local_position[mask, 0]
        v_values = local_position[mask, 1]
        u_span = face.u_edges_cm[-1] - face.u_edges_cm[0]
        v_span = face.v_edges_cm[-1] - face.v_edges_cm[0]
        edge_tolerance = 1.0e-9
        in_bounds = (
            (u_values >= face.u_edges_cm[0] - edge_tolerance * u_span)
            & (u_values <= face.u_edges_cm[-1] + edge_tolerance * u_span)
            & (v_values >= face.v_edges_cm[0] - edge_tolerance * v_span)
            & (v_values <= face.v_edges_cm[-1] + edge_tolerance * v_span)
        )
        iu = np.clip(iu, 0, len(face.u_edges_cm) - 2)
        iv = np.clip(iv, 0, len(face.v_edges_cm) - 2)
        valid = (
            in_bounds
            & (iu >= 0)
            & (iu < len(face.u_edges_cm) - 1)
            & (iv >= 0)
            & (iv < len(face.v_edges_cm) - 1)
        )
        if np.any(~valid):
            bad_local = local_position[mask][~valid]
            raise ValueError(
                f"surface-source positions lie outside surface {sid} patch map; "
                f"local extrema={bad_local.min(axis=0).tolist()} to "
                f"{bad_local.max(axis=0).tolist()}, "
                f"u={face.u_edges_cm[0]}..{face.u_edges_cm[-1]}, "
                f"v={face.v_edges_cm[0]}..{face.v_edges_cm[-1]}"
            )
        patch_values = np.full(mask.sum(), -1, dtype=int)
        patch_values[valid] = (
            iu[valid] + (len(face.u_edges_cm) - 1) * iv[valid]
        )
        patch[mask] = patch_values
    mu = local_direction[:, 2]
    phi = np.arctan2(local_direction[:, 1], local_direction[:, 0])
    eg = np.full(n, -1, dtype=int)
    for name in set(particles):
        mask = particles == name
        edges = particle_edges[name]
        values = energies[mask]
        groups = np.searchsorted(edges, values, side="right") - 1
        groups[values == edges[-1]] = len(edges) - 2
        if np.any(groups < 0) or np.any(groups >= len(edges) - 1):
            raise ValueError(
                f"{name} surface-source energy lies outside configured groups"
            )
        eg[mask] = groups
    mb = np.searchsorted(m_edges, mu, side="right") - 1
    mb[mu == m_edges[-1]] = len(m_edges) - 2
    pb = np.searchsorted(p_edges, phi, side="right") - 1
    pb[phi == p_edges[-1]] = len(p_edges) - 2
    if np.any(patch < 0):
        raise ValueError(
            "surface-source position lies outside a surface patch map"
        )
    if any(item not in PARTICLE_PDG for item in particles):
        raise ValueError(
            "canonical bank currently supports neutron and photon"
        )
    columns = {
        "record_id": np.arange(n, dtype=np.int64),
        "history_id": np.asarray(
            history_id if history_id is not None else np.full(n, -1),
            dtype=np.int64,
        ),
        "position_global_cm": positions,
        "position_local_cm": local_position,
        "direction_global": directions,
        "direction_local": local_direction,
        "outward_normal_global": record_normals,
        "energy_eV": energies,
        "weight": weights,
        "weight_std_dev": np.zeros(n),
        "particle": particles,
        "particle_pdg": np.asarray(
            [PARTICLE_PDG[item] for item in particles], dtype=np.int64
        ),
        "surface_id": surfaces,
        "envelope_id": np.full(n, envelope.envelope_id, dtype=object),
        "crossing_sense": sense.astype(str),
        "surface_role": roles.astype(str),
        "time_s": np.asarray(
            time_s if time_s is not None else np.zeros(n), dtype=float
        ),
        "mu": mu,
        "azimuth_rad": phi,
        "grazing": grazing,
        "patch_id": patch,
        "energy_group": eg,
        "angle_bin_id": mb * (len(p_edges) - 1) + pb,
    }
    return CorrelatedBoundaryBank(
        columns,
        {
            "history_id_available": history_id is not None,
            "mu_edges": m_edges.tolist(),
            "phi_edges_rad": p_edges.tolist(),
            "energy_axes": {
                f"{name}_energy_edges_eV": edges.tolist()
                for name, edges in particle_edges.items()
            },
            **(
                {
                    "energy_edges_eV": next(
                        iter(particle_edges.values())
                    ).tolist()
                }
                if energy_edges_by_particle is None
                else {}
            ),
        },
    )


def condition_on_independent_current(
    bank: CorrelatedBoundaryBank, current_rows: Iterable[Mapping[str, Any]]
) -> CorrelatedBoundaryBank:
    """Scale joint records within tally strata without destroying correlation."""
    c = bank.columns
    keys = list(
        zip(
            c["surface_id"],
            c["particle"],
            c["energy_group"],
            c["crossing_sense"],
        )
    )
    key_to_indices: dict[tuple[Any, ...], list[int]] = {}
    for index, key in enumerate(keys):
        key_to_indices.setdefault(tuple(key), []).append(index)
    weight = np.zeros(len(bank), dtype=float)
    std = np.zeros(len(bank), dtype=float)
    tally_total = 0.0
    used: set[tuple[Any, ...]] = set()
    for row in current_rows:
        key = (
            int(row["surface_id"]),
            str(row["particle"]),
            int(row["energy_group"]),
            str(row["crossing_sense"]),
        )
        mean = float(row["mean"])
        sigma = float(row.get("std_dev", 0.0))
        if mean < 0.0 or sigma < 0.0 or not np.isfinite(mean + sigma):
            raise ValueError(f"invalid independent current row {key}")
        indices = key_to_indices.get(key, [])
        raw = float(np.sum(c["weight"][indices])) if indices else 0.0
        if mean > 0.0 and raw <= 0.0:
            raise ValueError(
                f"positive tally stratum has no bank records: {key}"
            )
        if raw > 0.0:
            fraction = np.asarray(c["weight"][indices], dtype=float) / raw
            weight[indices] = mean * fraction
            std[indices] = sigma * fraction
        tally_total += mean
        used.add(key)
    unscored = {
        key
        for key in key_to_indices
        if key not in used and np.sum(c["weight"][key_to_indices[key]]) > 0.0
    }
    if unscored:
        raise ValueError(
            f"bank contains records outside independent tally strata: {sorted(unscored)}"
        )
    result = {name: np.array(value, copy=True) for name, value in c.items()}
    result["weight"] = weight
    result["weight_std_dev"] = std
    output = CorrelatedBoundaryBank(result, dict(bank.metadata))
    error = abs(output.integrated_current - tally_total)
    if error > max(1e-12, 1e-10 * max(tally_total, 1.0)):
        raise RuntimeError("bank-to-tally current closure failed")
    output.metadata.update(
        {
            "independent_tally_current": tally_total,
            "bank_current": output.integrated_current,
            "closure_absolute_error": error,
        }
    )
    return output


def conservative_projection(
    bank: CorrelatedBoundaryBank,
) -> dict[str, np.ndarray]:
    """Project complete correlated records to surface/patch/E/angle/species/sense."""
    c = bank.columns
    particles = sorted(set(c["particle"].astype(str)))
    senses = ("incoming", "outgoing", "grazing")
    surface_ids = sorted(set(int(item) for item in c["surface_id"]))
    shape = (
        len(surface_ids),
        int(np.max(c["patch_id"])) + 1,
        int(np.max(c["energy_group"])) + 1,
        int(np.max(c["angle_bin_id"])) + 1,
        len(particles),
        len(senses),
    )
    mean = np.zeros(shape)
    variance = np.zeros(shape)
    si = {value: index for index, value in enumerate(surface_ids)}
    pi = {value: index for index, value in enumerate(particles)}
    xi = {value: index for index, value in enumerate(senses)}
    for index in range(len(bank)):
        target = (
            si[int(c["surface_id"][index])],
            int(c["patch_id"][index]),
            int(c["energy_group"][index]),
            int(c["angle_bin_id"][index]),
            pi[str(c["particle"][index])],
            xi[str(c["crossing_sense"][index])],
        )
        mean[target] += c["weight"][index]
        variance[target] += c["weight_std_dev"][index] ** 2
    if not np.isclose(
        mean.sum(), bank.integrated_current, rtol=1e-12, atol=1e-12
    ):
        raise RuntimeError(
            "correlated projection does not conserve bank current"
        )
    return {
        "mean": mean,
        "std_dev": np.sqrt(variance),
        "surface_ids": np.asarray(surface_ids),
        "particles": np.asarray(particles, dtype=object),
        "senses": np.asarray(senses, dtype=object),
    }


def write_handoff(
    path: str | Path,
    envelope: MagnetBoundaryEnvelope,
    bank: CorrelatedBoundaryBank,
    *,
    provenance: Mapping[str, Any],
    normalization: Mapping[str, Any],
) -> dict[str, Any]:
    projection = conservative_projection(bank)
    required_norm = {
        "basis",
        "particles_per_source_history",
        "area_basis",
        "energy_bin_width",
        "solid_angle_measure",
        "quantity",
        "time_basis",
    }
    missing = required_norm - set(normalization)
    if missing:
        raise ValueError(
            f"normalization contract is missing {sorted(missing)}"
        )
    manifest = {
        "schema": SCHEMA_URI,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "envelope": envelope.to_dict(),
        "particles": sorted(set(bank.columns["particle"].astype(str))),
        "energy_axes": (
            bank.metadata["energy_axes"]
            if "energy_axes" in bank.metadata
            else {"energy_edges_eV": bank.metadata["energy_edges_eV"]}
        ),
        "mu_edges": bank.metadata["mu_edges"],
        "phi_edges_rad": bank.metadata["phi_edges_rad"],
        "quantity": "partial current",
        "normal_convention": "mu = Omega dot n_outward; incoming mu < 0",
        "normalization": dict(normalization),
        "provenance": dict(provenance),
        "angular_metrics": bank.angular_metrics(),
        "record_count": len(bank),
        "integrated_current": bank.integrated_current,
        "bank_metadata": bank.metadata,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype("utf-8")
    with h5py.File(output, "w") as target:
        target.attrs["schema"] = SCHEMA_URI
        target.attrs["schema_version"] = SCHEMA_VERSION
        target.create_dataset(
            "manifest_json",
            data=json.dumps(manifest, sort_keys=True),
            dtype=strings,
        )
        records = target.create_group("records")
        for name, value in bank.columns.items():
            array = np.asarray(value)
            (
                records.create_dataset(
                    name, data=array.astype(object), dtype=strings
                )
                if array.dtype.kind in "OU"
                else records.create_dataset(name, data=array)
            )
        projected = target.create_group("projection")
        for name, value in projection.items():
            array = np.asarray(value)
            (
                projected.create_dataset(
                    name, data=array.astype(object), dtype=strings
                )
                if array.dtype.kind in "OU"
                else projected.create_dataset(name, data=array)
            )
    return manifest


def read_handoff(
    path: str | Path,
) -> tuple[dict[str, Any], MagnetBoundaryEnvelope, CorrelatedBoundaryBank]:
    with h5py.File(path, "r") as source:
        if (
            source.attrs.get("schema") != SCHEMA_URI
            or source.attrs.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("incompatible magnet boundary-source schema")
        if not {"manifest_json", "records", "projection"}.issubset(source):
            raise ValueError("malformed handoff: required groups are absent")
        manifest = json.loads(source["manifest_json"].asstr()[()])
        columns = {
            name: (
                dataset.asstr()[()]
                if dataset.dtype.kind in "OS"
                else dataset[()]
            )
            for name, dataset in source["records"].items()
        }
    envelope = MagnetBoundaryEnvelope.from_dict(manifest["envelope"])
    bank = CorrelatedBoundaryBank(
        columns, dict(manifest.get("bank_metadata", {}))
    )
    expected = conservative_projection(bank)["mean"]
    with h5py.File(path, "r") as source:
        actual = source["projection/mean"][()]
    if expected.shape != actual.shape or not np.allclose(
        expected, actual, rtol=1e-12, atol=1e-12
    ):
        raise ValueError(
            "stored projection is incompatible with canonical records"
        )
    return manifest, envelope, bank


def source_mesh_provenance(
    source_mesh_path: str | Path,
    strengths_path: str | Path,
    *,
    vmec_path: str | Path,
    parameters: Mapping[str, Any],
    strengths: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(strengths, dtype=float)
    if (
        values.ndim != 1
        or len(values) == 0
        or np.any(values < 0.0)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "ParaStell source strengths must be a finite nonnegative vector"
        )
    return {
        "source_kind": "parastell.SourceMesh",
        "fallback_source_used": False,
        "plasma_conditions": "parastell.source_mesh.default_plasma_conditions",
        "reaction_rate": "parastell.source_mesh.default_reaction_rate",
        "parameters": dict(parameters),
        "vmec": {
            "path": str(Path(vmec_path).resolve()),
            "sha256": _hash(vmec_path),
        },
        "source_mesh": {
            "path": str(Path(source_mesh_path).resolve()),
            "sha256": _hash(source_mesh_path),
        },
        "strengths": {
            "path": str(Path(strengths_path).resolve()),
            "sha256": _hash(strengths_path),
            "tetrahedron_count": len(values),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "integral_reactions_per_s": float(values.sum()),
        },
    }
