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
SCHEMA_VERSION = "2.2.0"
SCHEMA_URI = f"{SCHEMA_NAME}/v{SCHEMA_VERSION}"
SUPPORTED_SCHEMA_VERSIONS = ("2.0.0", "2.1.0", SCHEMA_VERSION)
SUPPORTED_SCHEMA_URIS = tuple(
    f"{SCHEMA_NAME}/v{version}" for version in SUPPORTED_SCHEMA_VERSIONS
)
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
)
OPTIONAL_RECORD_FIELDS = (
    "source_file_id",
    "source_record_index",
    "history_id",
    "time_s",
    "weight_std_dev",
    "facet_id",
    "canonical_facet_id",
    "facet_index",
    "barycentric_coordinates",
    "reconstructed_position_global_cm",
    "signed_plane_residual_cm",
    "nearest_point_residual_cm",
    "distance_to_facet_residual_cm",
    "inside_facet",
    "facet_mapping_status",
    "centreline_arclength_cm",
    "normalized_arclength",
    "centreline_tangent",
    "centreline_radial",
    "centreline_transverse",
    "parallel_transport_tangent",
    "parallel_transport_width_axis",
    "parallel_transport_thickness_normal",
    "local_centreline_coordinates_cm",
    "distance_to_centreline_cm",
    "frame_type",
    "frame_quality_status",
    "cell_id",
    "material_id",
    "parent_id",
)
BANK_CLASSIFICATIONS = (
    "COMPLETE_CROSSING_BANK",
    "SAMPLED_CROSSING_BANK",
    "TRUNCATED_INVALID_BANK",
)
FACET_MATCH_CLASSIFICATIONS = (
    "EXACT_FACET_MATCH",
    "EDGE_TOLERANCE_MATCH",
    "VERTEX_TOLERANCE_MATCH",
    "NO_VALID_FACET_MATCH",
)


def classify_crossing_bank(
    *,
    stored_record_count: int,
    selected_record_count: int,
    max_particles_per_file: int | None,
    max_source_files: int | None,
    source_file_count: int,
    mpi_ranks: int | None = None,
    sampling_applied: bool = False,
) -> dict[str, Any]:
    """Classify surface-bank completeness from explicit capture accounting."""
    counts = (stored_record_count, selected_record_count, source_file_count)
    if any(int(value) < 0 for value in counts) or source_file_count == 0:
        raise ValueError("surface-bank counts must be nonnegative with a file")
    if selected_record_count > stored_record_count:
        raise ValueError("selected crossings cannot exceed stored records")
    if max_particles_per_file is not None and max_particles_per_file <= 0:
        raise ValueError("max_particles_per_file must be positive")
    if max_source_files is not None and max_source_files <= 0:
        raise ValueError("max_source_files must be positive")
    capacity = (
        int(max_particles_per_file) * int(max_source_files)
        if max_particles_per_file is not None and max_source_files is not None
        else None
    )
    cap_reached = capacity is not None and stored_record_count >= capacity
    missing_config = max_particles_per_file is None or max_source_files is None
    if cap_reached:
        classification = "TRUNCATED_INVALID_BANK"
    elif sampling_applied or missing_config:
        classification = "SAMPLED_CROSSING_BANK"
    else:
        classification = "COMPLETE_CROSSING_BANK"
    return {
        "classification": classification,
        "stored_record_count": int(stored_record_count),
        "selected_record_count": int(selected_record_count),
        "source_file_count": int(source_file_count),
        "max_particles_per_file": max_particles_per_file,
        "max_source_files": max_source_files,
        "configured_capacity": capacity,
        "cap_reached": bool(cap_reached),
        "sampling_applied": bool(sampling_applied),
        "mpi_ranks": mpi_ranks,
    }


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
    """Return a vendored, checksum-validated neutron structure."""

    from .energy_groups import get_structure

    try:
        structure = get_structure(name, particle="neutron")
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f"authoritative energy structure {name!r} is unavailable"
        ) from exc
    return _edges(structure.edges_eV, name)


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
        if "facet_mapping_status" in self.columns:
            statuses = np.asarray(self.columns["facet_mapping_status"]).astype(
                str
            )
            unknown = set(statuses) - set(FACET_MATCH_CLASSIFICATIONS)
            if unknown:
                raise ValueError(
                    f"unknown facet-match classifications: {sorted(unknown)}"
                )
        if "canonical_facet_id" in self.columns and "facet_id" in self.columns:
            if not np.array_equal(
                np.asarray(self.columns["canonical_facet_id"]).astype(str),
                np.asarray(self.columns["facet_id"]).astype(str),
            ):
                raise ValueError("facet_id must alias canonical_facet_id")

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

    def population_statistics(self) -> dict[str, Any]:
        """Report count, weighted count, and ESS by species and crossing sense."""
        weights = np.asarray(self.columns["weight"], dtype=float)
        particles = np.asarray(self.columns["particle"]).astype(str)
        senses = np.asarray(self.columns["crossing_sense"]).astype(str)
        surface_ids = np.asarray(self.columns["surface_id"], dtype=int)

        def summarize(mask: np.ndarray) -> dict[str, float | int]:
            selected = weights[mask]
            total = float(selected.sum())
            square_sum = float(np.dot(selected, selected))
            effective_sample_size = (
                float(total * total / square_sum) if square_sum > 0.0 else 0.0
            )
            if not len(selected):
                status = "EMPTY"
            elif effective_sample_size >= 25.0:
                status = "QUALIFIED"
            elif effective_sample_size >= 4.0:
                status = "MARGINAL"
            else:
                status = "INSUFFICIENT_STATISTICS"
            return {
                "record_count": int(mask.sum()),
                "weighted_count": total,
                "sum_weight_squared": square_sum,
                "effective_sample_size": effective_sample_size,
                "relative_counting_uncertainty": (
                    float(np.sqrt(square_sum) / total) if total > 0.0 else 0.0
                ),
                "status": status,
            }

        rows = []
        for surface_id in sorted(set(surface_ids.tolist())):
            for particle in sorted(set(particles.tolist())):
                for sense in ("incoming", "outgoing", "grazing"):
                    mask = (
                        (surface_ids == surface_id)
                        & (particles == particle)
                        & (senses == sense)
                    )
                    rows.append(
                        {
                            "surface_id": int(surface_id),
                            "particle": particle,
                            "crossing_sense": sense,
                            **summarize(mask),
                        }
                    )
        return {
            "overall": summarize(np.ones(len(self), dtype=bool)),
            "by_surface_particle_sense": rows,
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
    record_id: Sequence[int] | None = None,
    source_file_id: Sequence[int] | None = None,
    source_record_index: Sequence[int] | None = None,
    facet_mapping: Mapping[str, Sequence[Any]] | None = None,
    centreline_frame: Any | None = None,
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
        "record_id": (
            np.arange(n, dtype=np.int64)
            if record_id is None
            else np.asarray(record_id, dtype=np.int64)
        ),
        "position_global_cm": positions,
        "position_local_cm": local_position,
        "direction_global": directions,
        "direction_local": local_direction,
        "outward_normal_global": record_normals,
        "energy_eV": energies,
        "weight": weights,
        "particle": particles,
        "particle_pdg": np.asarray(
            [PARTICLE_PDG[item] for item in particles], dtype=np.int64
        ),
        "surface_id": surfaces,
        "envelope_id": np.full(n, envelope.envelope_id, dtype=object),
        "crossing_sense": sense.astype(str),
        "surface_role": roles.astype(str),
        "mu": mu,
        "azimuth_rad": phi,
        "grazing": grazing,
        "patch_id": patch,
        "energy_group": eg,
        "angle_bin_id": mb * (len(p_edges) - 1) + pb,
    }
    if (
        len(columns["record_id"]) != n
        or len(set(columns["record_id"].tolist())) != n
    ):
        raise ValueError("record IDs must be unique and align with records")
    optional_columns = {
        "source_file_id": source_file_id,
        "source_record_index": source_record_index,
    }
    for name, value in optional_columns.items():
        if value is not None:
            array = np.asarray(value, dtype=np.int64)
            if array.shape != (n,):
                raise ValueError(f"{name} must align with records")
            columns[name] = array
    if facet_mapping is not None:
        allowed = {
            "facet_id",
            "canonical_facet_id",
            "facet_index",
            "barycentric_coordinates",
            "reconstructed_position_global_cm",
            "signed_plane_residual_cm",
            "nearest_point_residual_cm",
            "distance_to_facet_residual_cm",
            "inside_facet",
            "facet_mapping_status",
        }
        unknown = set(facet_mapping) - allowed
        if unknown:
            raise ValueError(
                f"unknown facet-mapping fields: {sorted(unknown)}"
            )
        # v2.1 callers may still supply only the legacy four fields.  The
        # producer's v2.2 export path supplies and validates the full set.
        required = {
            "facet_id",
            "facet_index",
            "barycentric_coordinates",
            "distance_to_facet_residual_cm",
        }
        missing = required - set(facet_mapping)
        if missing:
            raise ValueError(f"facet mapping is missing {sorted(missing)}")
        for name, value in facet_mapping.items():
            array = np.asarray(value)
            expected = (
                (n, 3)
                if name
                in {
                    "barycentric_coordinates",
                    "reconstructed_position_global_cm",
                }
                else (n,)
            )
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            columns[name] = array
        if "canonical_facet_id" not in columns:
            columns["canonical_facet_id"] = np.asarray(
                columns["facet_id"], dtype=object
            )
        if "facet_mapping_status" in columns:
            invalid = (
                np.asarray(columns["facet_mapping_status"]).astype(str)
                == "NO_VALID_FACET_MATCH"
            )
            if np.any(invalid):
                raise ValueError(
                    "canonical boundary bank cannot contain "
                    "NO_VALID_FACET_MATCH records"
                )
        barycentric = np.asarray(
            columns["barycentric_coordinates"], dtype=float
        )
        if not np.allclose(
            barycentric.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-7
        ) or np.any(barycentric < -1.0e-7):
            raise ValueError("facet barycentric coordinates are invalid")
    if centreline_frame is not None:
        frame_values = centreline_frame.sample(positions)
        for name in (
            "centreline_arclength_cm",
            "normalized_arclength",
            "centreline_tangent",
            "centreline_radial",
            "centreline_transverse",
            "local_centreline_coordinates_cm",
            "distance_to_centreline_cm",
        ):
            columns[name] = np.asarray(frame_values[name])
        columns["parallel_transport_tangent"] = np.asarray(
            frame_values["centreline_tangent"]
        )
        columns["parallel_transport_width_axis"] = np.asarray(
            frame_values["centreline_radial"]
        )
        columns["parallel_transport_thickness_normal"] = np.asarray(
            frame_values["centreline_transverse"]
        )
        columns["frame_type"] = np.full(
            n, frame_values["frame_type"], dtype=object
        )
        columns["frame_quality_status"] = np.full(
            n, frame_values["frame_quality_status"], dtype=object
        )
    if history_id is not None:
        columns["history_id"] = np.asarray(history_id, dtype=np.int64)
    if time_s is not None:
        columns["time_s"] = np.asarray(time_s, dtype=float)
    source_fields = {
        "position_global_cm",
        "direction_global",
        "energy_eV",
        "weight",
        "particle",
        "surface_id",
        "time_s",
        "history_id",
        "source_file_id",
        "source_record_index",
    }
    field_availability = {}
    for name in (*REQUIRED_RECORD_FIELDS, *OPTIONAL_RECORD_FIELDS):
        available = name in columns
        field_availability[name] = {
            "available": available,
            "origin": (
                "source_provided"
                if available and name in source_fields
                else "derived" if available else "unavailable"
            ),
        }
    field_availability["time_s"]["semantics"] = (
        "prompt_particle_flight_time" if time_s is not None else "not_provided"
    )
    field_availability["weight_std_dev"][
        "semantics"
    ] = "per_record_uncertainty_not_exposed"
    field_availability["facet_id"]["semantics"] = (
        "canonical_dagmc_facet_identity"
        if "facet_id" in columns
        else "facet_geometry_not_supplied"
    )
    field_availability["canonical_facet_id"]["semantics"] = (
        "canonical_dagmc_facet_identity"
        if "canonical_facet_id" in columns
        else "facet_geometry_not_supplied"
    )
    field_availability["frame_type"]["semantics"] = (
        "coil_centerline_parallel_transport_engineering_frame"
        if "frame_type" in columns
        else "coil_centreline_not_supplied"
    )
    return CorrelatedBoundaryBank(
        columns,
        {
            "field_availability": field_availability,
            "unavailable_field_semantics": {
                name: "not_exposed_by_openmc_surface_source"
                for name in (
                    "history_id",
                    "parent_id",
                    "cell_id",
                    "material_id",
                )
                if name not in columns
            },
            "canonical_record_policy": (
                "raw source-bank transport contributions; no tally conditioning"
            ),
            "surface_ids": list(envelope.surface_ids),
            "surface_patch_counts": {
                str(surface.surface_id): (
                    (len(surface.u_edges_cm) - 1)
                    * (len(surface.v_edges_cm) - 1)
                )
                for surface in envelope.surfaces
            },
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


def derive_tally_conditioned_bank(
    bank: CorrelatedBoundaryBank, current_rows: Iterable[Mapping[str, Any]]
) -> CorrelatedBoundaryBank:
    """Create a noncanonical tally-conditioned consumer distribution."""
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
    metadata = dict(bank.metadata)
    availability = dict(metadata.get("field_availability", {}))
    availability["weight_std_dev"] = {
        "available": True,
        "origin": "derived",
        "semantics": "allocated tally stratum uncertainty; not measured per record",
    }
    metadata.update(
        {
            "field_availability": availability,
            "dataset_role": "derived_tally_conditioned_distribution",
            "canonical_bank": False,
        }
    )
    output = CorrelatedBoundaryBank(result, metadata)
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


def condition_on_independent_current(
    bank: CorrelatedBoundaryBank, current_rows: Iterable[Mapping[str, Any]]
) -> CorrelatedBoundaryBank:
    """Backward-compatible alias for a derived, noncanonical distribution."""
    return derive_tally_conditioned_bank(bank, current_rows)


def conservative_projection(
    bank: CorrelatedBoundaryBank,
) -> dict[str, np.ndarray]:
    """Project complete correlated records to surface/patch/E/angle/species/sense."""
    c = bank.columns
    observed_particles = set(c["particle"].astype(str))
    energy_group_counts = {}
    for name, edges in bank.metadata.get("energy_axes", {}).items():
        if name.endswith("_energy_edges_eV"):
            particle = name.removesuffix("_energy_edges_eV")
            energy_group_counts[particle] = len(edges) - 1
    particles = sorted(observed_particles | set(energy_group_counts))
    senses = ("incoming", "outgoing", "grazing")
    observed_surface_ids = set(int(item) for item in c["surface_id"])
    surface_ids = sorted(
        set(int(item) for item in bank.metadata.get("surface_ids", ()))
        | observed_surface_ids
    )
    patch_counts = {
        int(surface_id): int(count)
        for surface_id, count in bank.metadata.get(
            "surface_patch_counts", {}
        ).items()
    }
    for patch in bank.metadata.get("adaptive_surface_patches", {}).get(
        "patches", ()
    ):
        surface_id = int(patch["surface_id"])
        patch_counts[surface_id] = max(
            patch_counts.get(surface_id, 0), int(patch["patch_id"]) + 1
        )
    observed_patch_count = int(np.max(c["patch_id"])) + 1 if len(bank) else 0
    patch_count = max([observed_patch_count, *patch_counts.values()])
    observed_energy_count = (
        int(np.max(c["energy_group"])) + 1 if len(bank) else 0
    )
    energy_count = max([observed_energy_count, *energy_group_counts.values()])
    mu_count = max(len(bank.metadata.get("mu_edges", ())) - 1, 0)
    phi_count = max(len(bank.metadata.get("phi_edges_rad", ())) - 1, 0)
    configured_angle_count = mu_count * phi_count
    observed_angle_count = (
        int(np.max(c["angle_bin_id"])) + 1 if len(bank) else 0
    )
    angle_count = max(configured_angle_count, observed_angle_count)
    shape = (
        len(surface_ids),
        patch_count,
        energy_count,
        angle_count,
        len(particles),
        len(senses),
    )
    mean = np.zeros(shape)
    variance = np.zeros(shape)
    si = {value: index for index, value in enumerate(surface_ids)}
    pi = {value: index for index, value in enumerate(particles)}
    xi = {value: index for index, value in enumerate(senses)}
    record_sigma = c.get("weight_std_dev")
    use_counting_model = (
        bank.metadata.get("projection_uncertainty_model")
        == "weighted_event_counting_approximation"
    )
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
        if record_sigma is not None:
            variance[target] += record_sigma[index] ** 2
        elif use_counting_model:
            variance[target] += c["weight"][index] ** 2
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
        "surface_patch_counts": np.asarray(
            [
                patch_counts.get(surface_id, observed_patch_count)
                for surface_id in surface_ids
            ]
        ),
        "particles": np.asarray(particles, dtype=object),
        "particle_energy_group_counts": np.asarray(
            [
                energy_group_counts.get(particle, observed_energy_count)
                for particle in particles
            ]
        ),
        "senses": np.asarray(senses, dtype=object),
        "mu_bin_count": np.asarray(mu_count),
        "phi_bin_count": np.asarray(phi_count),
    }


def write_handoff(
    path: str | Path,
    envelope: MagnetBoundaryEnvelope,
    bank: CorrelatedBoundaryBank,
    *,
    provenance: Mapping[str, Any],
    normalization: Mapping[str, Any],
    facet_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    projection = conservative_projection(bank)
    provenance_geometry = provenance.get("dagmc_geometry_sha256")
    if (
        provenance_geometry is not None
        and str(provenance_geometry) != envelope.dagmc_geometry_sha256
    ):
        raise ValueError(
            "boundary-source geometry SHA disagrees with the envelope"
        )
    envelope_fingerprint = envelope.metadata.get(
        "canonical_geometry_fingerprint"
    )
    provenance_fingerprint = provenance.get("canonical_geometry_fingerprint")
    if (
        envelope_fingerprint is not None
        and provenance_fingerprint is not None
        and str(envelope_fingerprint) != str(provenance_fingerprint)
    ):
        raise ValueError(
            "boundary-source canonical geometry fingerprint mismatch"
        )
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
    normalized_facet_catalog = None
    facet_catalog_manifest = {
        "available": False,
        "reason": "faceted DAGMC envelope catalog was not supplied",
    }
    if facet_catalog is not None:
        required_catalog = {
            "facet_id",
            "facet_index",
            "surface_id",
            "surface_role",
            "facet_centroid_global_cm",
            "outward_normal_global",
            "facet_area_cm2",
            "centreline_linkage_available",
            "centreline_linkage_status",
        }
        missing_catalog = required_catalog - set(facet_catalog)
        if missing_catalog:
            raise ValueError(
                f"facet catalog is missing {sorted(missing_catalog)}"
            )
        normalized_facet_catalog = {
            name: np.asarray(value)
            for name, value in facet_catalog.items()
            if name != "centreline_linkage_available"
        }
        if "canonical_facet_id" not in normalized_facet_catalog:
            normalized_facet_catalog["canonical_facet_id"] = np.asarray(
                normalized_facet_catalog["facet_id"], dtype=object
            )
        facet_count = len(normalized_facet_catalog["facet_id"])
        if facet_count == 0:
            raise ValueError("facet catalog must contain at least one facet")
        for name, values in normalized_facet_catalog.items():
            if values.shape[0] != facet_count:
                raise ValueError(f"facet catalog field {name!r} is misaligned")
        if (
            len(set(normalized_facet_catalog["facet_id"].astype(str)))
            != facet_count
        ):
            raise ValueError("facet catalog IDs must be unique")
        if not np.array_equal(
            normalized_facet_catalog["facet_id"].astype(str),
            normalized_facet_catalog["canonical_facet_id"].astype(str),
        ):
            raise ValueError("facet_id must alias canonical_facet_id")
        if not set(
            normalized_facet_catalog["surface_id"].astype(int)
        ).issubset(set(envelope.surface_ids)):
            raise ValueError(
                "facet catalog contains a foreign envelope surface"
            )
        for name in (
            "facet_centroid_global_cm",
            "outward_normal_global",
        ):
            values = np.asarray(normalized_facet_catalog[name], dtype=float)
            if values.shape != (facet_count, 3) or np.any(
                ~np.isfinite(values)
            ):
                raise ValueError(f"facet catalog field {name!r} is invalid")
        facet_complete_fields = {
            "canonical_facet_id",
            "dagmc_volume_id",
            "surface_id",
            "triangle_vertices_global_cm",
            "triangle_outward_normal_global",
            "triangle_area_cm2",
            "facet_centroid_global_cm",
        }
        facet_complete = facet_complete_fields.issubset(
            normalized_facet_catalog
        )
        if facet_complete:
            vertices = np.asarray(
                normalized_facet_catalog["triangle_vertices_global_cm"],
                dtype=float,
            )
            triangle_normals = np.asarray(
                normalized_facet_catalog["triangle_outward_normal_global"],
                dtype=float,
            )
            if vertices.shape != (facet_count, 3, 3) or np.any(
                ~np.isfinite(vertices)
            ):
                raise ValueError("facet triangle vertices are invalid")
            if triangle_normals.shape != (facet_count, 3) or np.any(
                ~np.isfinite(triangle_normals)
            ):
                raise ValueError("facet triangle normals are invalid")
            if not np.allclose(
                np.linalg.norm(triangle_normals, axis=1),
                1.0,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError("facet triangle normals are not unit vectors")
            volume_ids = np.asarray(
                normalized_facet_catalog["dagmc_volume_id"], dtype=int
            )
            if np.any(volume_ids != int(envelope.dagmc_volume_id)):
                raise ValueError(
                    "facet catalog DAGMC volume identity disagrees with envelope"
                )
        areas = np.asarray(
            normalized_facet_catalog["facet_area_cm2"], dtype=float
        )
        if (
            areas.shape != (facet_count,)
            or np.any(~np.isfinite(areas))
            or np.any(areas <= 0.0)
        ):
            raise ValueError("facet catalog areas must be finite and positive")
        linkage_available = bool(facet_catalog["centreline_linkage_available"])
        linkage = normalized_facet_catalog["centreline_linkage_status"].astype(
            str
        )
        if linkage_available:
            required_linkage = {
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
            }
            missing_linkage = required_linkage - set(normalized_facet_catalog)
            if missing_linkage:
                raise ValueError(
                    "linked facet catalog is missing "
                    f"{sorted(missing_linkage)}"
                )
            if np.any(linkage != "LINKED_NEAREST_CENTRELINE_SEGMENT"):
                raise ValueError(
                    "linked facet catalog contains unavailable rows"
                )
        elif np.any(linkage != "UNAVAILABLE_CENTRELINE_FRAME_NOT_SUPPLIED"):
            raise ValueError("unlinked facet catalog has inconsistent status")
        digest_payload = {
            "centreline_linkage_available": linkage_available,
            **{
                name: values.tolist()
                for name, values in sorted(normalized_facet_catalog.items())
            },
        }
        facet_catalog_sha256 = hashlib.sha256(
            json.dumps(
                digest_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        facet_catalog_manifest = {
            "available": True,
            "facet_count": facet_count,
            "sha256": facet_catalog_sha256,
            "centreline_linkage_available": linkage_available,
            "fields": sorted(normalized_facet_catalog),
            "facet_complete_v22": facet_complete,
            "facet_complete_fields": sorted(facet_complete_fields),
        }
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
        "population_statistics": bank.population_statistics(),
        "record_count": len(bank),
        "integrated_current": bank.integrated_current,
        "bank_metadata": bank.metadata,
        "field_availability": bank.metadata.get("field_availability", {}),
        "canonical_bank": bank.metadata.get("canonical_bank", True),
        "canonical_record_policy": bank.metadata.get(
            "canonical_record_policy",
            "raw record weights; no hidden tally conditioning",
        ),
        "time_semantics": "prompt particle flight time, not irradiation time",
        "facet_catalog": facet_catalog_manifest,
        "facet_complete_boundary": {
            "schema_revision": SCHEMA_URI,
            "record_fields_complete": all(
                name in bank.columns
                for name in (
                    "canonical_facet_id",
                    "barycentric_coordinates",
                    "reconstructed_position_global_cm",
                    "signed_plane_residual_cm",
                    "nearest_point_residual_cm",
                    "inside_facet",
                    "facet_mapping_status",
                )
            ),
            "all_records_valid": bool(
                "facet_mapping_status" in bank.columns
                and np.all(
                    np.asarray(bank.columns["facet_mapping_status"]).astype(
                        str
                    )
                    != "NO_VALID_FACET_MATCH"
                )
            ),
            "catalog_complete": bool(
                facet_catalog_manifest.get("facet_complete_v22", False)
            ),
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype("utf-8")

    def create_numeric_dataset(group, name, array):
        options = (
            {"compression": "gzip", "compression_opts": 4, "shuffle": True}
            if array.ndim > 0 and array.size >= 1024
            else {}
        )
        return group.create_dataset(name, data=array, **options)

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
                else create_numeric_dataset(records, name, array)
            )
        projected = target.create_group("projection")
        for name, value in projection.items():
            array = np.asarray(value)
            (
                projected.create_dataset(
                    name, data=array.astype(object), dtype=strings
                )
                if array.dtype.kind in "OU"
                else create_numeric_dataset(projected, name, array)
            )
        if normalized_facet_catalog is not None:
            catalog = target.create_group("facet_catalog")
            catalog.attrs["centreline_linkage_available"] = bool(
                facet_catalog_manifest["centreline_linkage_available"]
            )
            catalog.attrs["sha256"] = facet_catalog_manifest["sha256"]
            for name, value in normalized_facet_catalog.items():
                array = np.asarray(value)
                (
                    catalog.create_dataset(
                        name, data=array.astype(object), dtype=strings
                    )
                    if array.dtype.kind in "OU"
                    else create_numeric_dataset(catalog, name, array)
                )
    return manifest


def read_handoff(
    path: str | Path,
) -> tuple[dict[str, Any], MagnetBoundaryEnvelope, CorrelatedBoundaryBank]:
    with h5py.File(path, "r") as source:
        schema = str(source.attrs.get("schema", ""))
        version = str(source.attrs.get("schema_version", ""))
        if (
            schema not in SUPPORTED_SCHEMA_URIS
            or version not in SUPPORTED_SCHEMA_VERSIONS
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
        catalog_contract = manifest.get("facet_catalog", {"available": False})
        if bool(catalog_contract.get("available")) != (
            "facet_catalog" in source
        ):
            raise ValueError(
                "facet catalog availability contradicts the manifest"
            )
        if catalog_contract.get("available"):
            catalog_group = source["facet_catalog"]
            catalog_values = {
                name: (
                    dataset.asstr()[()]
                    if dataset.dtype.kind in "OS"
                    else dataset[()]
                )
                for name, dataset in catalog_group.items()
            }
            facet_count = int(catalog_contract["facet_count"])
            if facet_count <= 0 or any(
                np.asarray(value).shape[0] != facet_count
                for value in catalog_values.values()
            ):
                raise ValueError("stored facet catalog is misaligned")
            if (
                len(set(np.asarray(catalog_values["facet_id"]).astype(str)))
                != facet_count
            ):
                raise ValueError("stored facet catalog IDs are not unique")
            digest_payload = {
                "centreline_linkage_available": bool(
                    catalog_group.attrs["centreline_linkage_available"]
                ),
                **{
                    name: np.asarray(value).tolist()
                    for name, value in sorted(catalog_values.items())
                },
            }
            actual_catalog_sha256 = hashlib.sha256(
                json.dumps(
                    digest_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if actual_catalog_sha256 != catalog_contract.get("sha256") or str(
                catalog_group.attrs.get("sha256", "")
            ) != catalog_contract.get("sha256"):
                raise ValueError("stored facet catalog hash is invalid")
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


def assign_adaptive_surface_patches(
    envelope: MagnetBoundaryEnvelope,
    bank: CorrelatedBoundaryBank,
    *,
    target_effective_sample_size: float = 25.0,
    minimum_records: int = 4,
    maximum_depth: int = 5,
) -> CorrelatedBoundaryBank:
    """Derive conservative surface patches without changing canonical records."""
    if target_effective_sample_size <= 0.0:
        raise ValueError("target_effective_sample_size must be positive")
    if minimum_records <= 0 or maximum_depth < 0:
        raise ValueError("adaptive patch limits are invalid")
    columns = {
        name: np.array(value, copy=True)
        for name, value in bank.columns.items()
    }
    positions = np.asarray(columns["position_local_cm"], dtype=float)
    weights = np.asarray(columns["weight"], dtype=float)
    surface_ids = np.asarray(columns["surface_id"], dtype=int)
    patch_ids = np.full(len(bank), -1, dtype=int)
    patch_metadata = []

    def effective_sample_size(indices: np.ndarray) -> float:
        selected = weights[indices]
        total = float(selected.sum())
        squares = float(np.dot(selected, selected))
        return float(total * total / squares) if squares > 0.0 else 0.0

    for surface in envelope.surfaces:
        indices = np.flatnonzero(surface_ids == surface.surface_id)
        root = (
            float(surface.u_edges_cm[0]),
            float(surface.u_edges_cm[-1]),
            float(surface.v_edges_cm[0]),
            float(surface.v_edges_cm[-1]),
        )
        leaves: list[tuple[np.ndarray, tuple[float, ...], int]] = []

        def split(
            selected: np.ndarray,
            bounds: tuple[float, ...],
            depth: int,
        ) -> None:
            if (
                depth >= maximum_depth
                or len(selected) < 2 * minimum_records
                or effective_sample_size(selected)
                < 2.0 * target_effective_sample_size
            ):
                leaves.append((selected, bounds, depth))
                return
            u0, u1, v0, v1 = bounds
            local = positions[selected, :2]
            spans = np.asarray([u1 - u0, v1 - v0])
            variances = np.var(local, axis=0) / np.maximum(
                spans * spans, 1e-30
            )
            axis = int(np.argmax(variances))
            midpoint = (bounds[2 * axis] + bounds[2 * axis + 1]) / 2.0
            lower_mask = local[:, axis] < midpoint
            lower = selected[lower_mask]
            upper = selected[~lower_mask]
            if (
                len(lower) < minimum_records
                or len(upper) < minimum_records
                or effective_sample_size(lower) < target_effective_sample_size
                or effective_sample_size(upper) < target_effective_sample_size
            ):
                leaves.append((selected, bounds, depth))
                return
            lower_bounds = list(bounds)
            upper_bounds = list(bounds)
            lower_bounds[2 * axis + 1] = midpoint
            upper_bounds[2 * axis] = midpoint
            split(lower, tuple(lower_bounds), depth + 1)
            split(upper, tuple(upper_bounds), depth + 1)

        split(indices, root, 0)
        root_area = (root[1] - root[0]) * (root[3] - root[2])
        area_sum = 0.0
        for patch_id, (selected, bounds, depth) in enumerate(leaves):
            patch_ids[selected] = patch_id
            rectangle_area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
            area = float(surface.area_cm2 * rectangle_area / root_area)
            area_sum += area
            patch_metadata.append(
                {
                    "surface_id": surface.surface_id,
                    "surface_role": surface.role,
                    "patch_id": patch_id,
                    "u_bounds_cm": [bounds[0], bounds[1]],
                    "v_bounds_cm": [bounds[2], bounds[3]],
                    "area_cm2": area,
                    "depth": depth,
                    "record_count": int(len(selected)),
                    "weighted_count": float(weights[selected].sum()),
                    "effective_sample_size": effective_sample_size(selected),
                }
            )
        if not np.isclose(area_sum, surface.area_cm2, rtol=1e-12, atol=1e-12):
            raise RuntimeError(
                f"adaptive patch areas do not close on surface {surface.surface_id}"
            )
    if np.any(patch_ids < 0):
        raise RuntimeError(
            "adaptive patch assignment omitted boundary records"
        )
    columns["patch_id"] = patch_ids
    metadata = dict(bank.metadata)
    metadata["adaptive_surface_patches"] = {
        "target_effective_sample_size": float(target_effective_sample_size),
        "minimum_records": int(minimum_records),
        "maximum_depth": int(maximum_depth),
        "patches": patch_metadata,
        "area_conservative": True,
        "canonical_records_modified": False,
    }
    metadata["surface_patch_counts"] = {
        str(surface.surface_id): sum(
            patch["surface_id"] == surface.surface_id
            for patch in patch_metadata
        )
        for surface in envelope.surfaces
    }
    output = CorrelatedBoundaryBank(columns, metadata)
    if not np.isclose(
        output.integrated_current,
        bank.integrated_current,
        rtol=1e-14,
        atol=1e-14,
    ):
        raise RuntimeError("adaptive patching changed integrated current")
    return output
