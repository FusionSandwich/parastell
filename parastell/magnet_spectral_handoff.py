"""OpenMC spectral handoff utilities for ParaStell magnet models.

This module builds energy-, particle-, direction-, and space-resolved OpenMC
Tallies around coarse reactor-scale magnet regions and exports their results to
an HDF5 contract suitable for local deterministic transport models. It also
configures OpenMC surface-source writing and converts the resulting phase-space
bank into a magnet-local coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import openmc
import yaml


SCHEMA_NAME = "parastell.magnet_spectral_handoff"
SCHEMA_VERSION = "0.1.0"
EV_TO_J = 1.602176634e-19
DEFAULT_GAS_PRODUCTION_SCORES = (
    "H1-production",
    "H2-production",
    "H3-production",
    "He3-production",
    "He4-production",
)
_VALID_SOURCE_DIRECTIONS = {"incoming", "outgoing", "both", "all"}
_LEGACY_PARTICLE_CODES = {
    0: ("neutron", 2112),
    1: ("photon", 22),
    2: ("electron", 11),
    3: ("positron", -11),
}
_PDG_PARTICLE_CODES = {
    2112: ("neutron", 2112),
    22: ("photon", 22),
    11: ("electron", 11),
    -11: ("positron", -11),
}
_ROLE_OFFSETS = {
    "cell_flux": 1,
    "mesh_flux": 2,
    "boundary_current": 3,
    "heating": 4,
    "damage_energy": 5,
    "gas_production": 6,
}


def _resolve_energy_bounds(
    data: Mapping[str, Any],
) -> tuple[tuple[float, ...], str | None]:
    explicit = data.get("energy_bounds_eV")
    structure = data.get("energy_group_structure")
    if explicit is not None and structure is not None:
        raise ValueError(
            "set either energy_bounds_eV or energy_group_structure, not both"
        )
    if explicit is not None:
        return tuple(explicit), None
    if structure is None:
        raise ValueError(
            "energy_bounds_eV or energy_group_structure is required"
        )

    from openmc.mgxs import GROUP_STRUCTURES

    structure = str(structure)
    try:
        bounds = GROUP_STRUCTURES[structure]
    except KeyError as exc:
        raise ValueError(
            f"unknown OpenMC energy group structure {structure!r}"
        ) from exc
    return tuple(float(value) for value in bounds), structure


def _as_float_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of real numbers") from exc
    return result


def _validate_edges(
    values: Iterable[float],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, ...]:
    edges = _as_float_tuple(values, name)
    if len(edges) < 2:
        raise ValueError(f"{name} must contain at least two bin edges")
    if not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} must contain only finite values")
    if any(upper <= lower for lower, upper in zip(edges, edges[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    if minimum is not None and edges[0] < minimum:
        raise ValueError(f"{name} cannot start below {minimum}")
    if maximum is not None and edges[-1] > maximum:
        raise ValueError(f"{name} cannot end above {maximum}")
    return edges


def _positive_ids(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive OpenMC IDs")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} cannot contain duplicate IDs")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    return value


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    if not slug:
        raise ValueError("region names must contain an alphanumeric character")
    return slug.lower()


def _normalise_axis(values: Iterable[float], name: str) -> np.ndarray:
    axis = np.asarray(tuple(values), dtype=float)
    if axis.shape != (3,):
        raise ValueError(f"{name} must have exactly three components")
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must be a finite, non-zero vector")
    return axis / norm


def _normalise_particle_codes(
    codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Normalize OpenMC 0.15 enum and OpenMC 0.16 PDG particle codes."""
    raw = np.asarray(codes, dtype=int)
    unique = set(int(value) for value in np.unique(raw))
    if unique.issubset(_LEGACY_PARTICLE_CODES):
        mapping = _LEGACY_PARTICLE_CODES
        encoding = "openmc_0.15_enum"
    elif unique.issubset(_PDG_PARTICLE_CODES):
        mapping = _PDG_PARTICLE_CODES
        encoding = "pdg"
    else:
        mapping = {**_LEGACY_PARTICLE_CODES, **_PDG_PARTICLE_CODES}
        encoding = "mixed_or_unknown"

    names = np.empty(raw.shape, dtype=object)
    pdg = np.zeros(raw.shape, dtype=int)
    for index, code in np.ndenumerate(raw):
        particle = mapping.get(int(code))
        if particle is None:
            names[index] = f"unknown_{int(code)}"
        else:
            names[index], pdg[index] = particle
    return names, pdg, encoding


def _resolve_relative_mesh_filenames(
    data: Mapping[str, Any], base_directory: Path
) -> None:
    """Resolve unstructured tally meshes relative to their YAML file."""
    for region in data.get("regions", ()):
        if not isinstance(region, Mapping):
            continue
        mesh = region.get("mesh")
        if not isinstance(mesh, Mapping):
            continue
        kind = str(mesh.get("kind", "")).lower().replace("-", "_")
        filename = mesh.get("filename")
        if kind != "unstructured" or not filename:
            continue
        path = Path(str(filename)).expanduser()
        if not path.is_absolute():
            mesh["filename"] = str((base_directory / path).resolve())


@dataclass(frozen=True)
class CoordinateFrame:
    """Right-handed Cartesian frame attached to a magnet or tape stack.

    The three axes are expressed in the reactor-scale global coordinate system.
    Position and direction vectors are transformed into local coordinates by
    projecting them onto these axes.
    """

    origin_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    x_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    y_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    z_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    labels: tuple[str, str, str] = (
        "local_x",
        "local_y",
        "local_z",
    )

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_cm, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin_cm must contain three finite coordinates")

        x_axis = _normalise_axis(self.x_axis, "x_axis")
        y_axis = _normalise_axis(self.y_axis, "y_axis")
        z_axis = _normalise_axis(self.z_axis, "z_axis")
        matrix = np.vstack((x_axis, y_axis, z_axis))

        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1.0e-8):
            raise ValueError("coordinate-frame axes must be orthonormal")
        if np.linalg.det(matrix) <= 0.0:
            raise ValueError("coordinate-frame axes must be right-handed")
        if len(self.labels) != 3 or any(not label for label in self.labels):
            raise ValueError("labels must contain three non-empty strings")

        object.__setattr__(self, "origin_cm", tuple(float(x) for x in origin))
        object.__setattr__(self, "x_axis", tuple(float(x) for x in x_axis))
        object.__setattr__(self, "y_axis", tuple(float(x) for x in y_axis))
        object.__setattr__(self, "z_axis", tuple(float(x) for x in z_axis))
        object.__setattr__(self, "labels", tuple(str(x) for x in self.labels))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> CoordinateFrame:
        if not data:
            return cls()
        return cls(
            origin_cm=tuple(data.get("origin_cm", (0.0, 0.0, 0.0))),
            x_axis=tuple(data.get("x_axis", (1.0, 0.0, 0.0))),
            y_axis=tuple(data.get("y_axis", (0.0, 1.0, 0.0))),
            z_axis=tuple(data.get("z_axis", (0.0, 0.0, 1.0))),
            labels=tuple(
                data.get("labels", ("local_x", "local_y", "local_z"))
            ),
        )

    @property
    def matrix(self) -> np.ndarray:
        """Return the global-to-local rotation matrix."""
        return np.vstack((self.x_axis, self.y_axis, self.z_axis))

    def transform_positions(self, positions_cm: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_cm, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_cm must have shape (N, 3)")
        return (positions - np.asarray(self.origin_cm)) @ self.matrix.T

    def transform_directions(self, directions: np.ndarray) -> np.ndarray:
        vectors = np.asarray(directions, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError("directions must have shape (N, 3)")
        return vectors @ self.matrix.T

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_cm": list(self.origin_cm),
            "x_axis": list(self.x_axis),
            "y_axis": list(self.y_axis),
            "z_axis": list(self.z_axis),
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class MeshSpec:
    """Serializable OpenMC tally-mesh definition."""

    kind: str
    filename: str | None = None
    library: str = "moab"
    lower_left_cm: tuple[float, float, float] | None = None
    upper_right_cm: tuple[float, float, float] | None = None
    dimension: tuple[int, int, int] | None = None
    mesh_id: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        kind = self.kind.lower().replace("-", "_")
        if kind not in {"regular", "unstructured"}:
            raise ValueError("mesh kind must be 'regular' or 'unstructured'")
        object.__setattr__(self, "kind", kind)

        if self.mesh_id is not None and int(self.mesh_id) <= 0:
            raise ValueError("mesh_id must be positive")

        if kind == "unstructured":
            if not self.filename:
                raise ValueError("unstructured meshes require filename")
            if self.library not in {"moab", "libmesh"}:
                raise ValueError(
                    "unstructured mesh library must be moab/libmesh"
                )
        else:
            if self.lower_left_cm is None or self.upper_right_cm is None:
                raise ValueError(
                    "regular meshes require lower_left_cm and upper_right_cm"
                )
            if self.dimension is None:
                raise ValueError("regular meshes require dimension")
            lower = np.asarray(self.lower_left_cm, dtype=float)
            upper = np.asarray(self.upper_right_cm, dtype=float)
            dimension = np.asarray(self.dimension, dtype=int)
            if lower.shape != (3,) or upper.shape != (3,):
                raise ValueError("regular-mesh bounds must have three values")
            if not np.all(np.isfinite(lower)) or not np.all(
                np.isfinite(upper)
            ):
                raise ValueError("regular-mesh bounds must be finite")
            if np.any(upper <= lower):
                raise ValueError("upper_right_cm must exceed lower_left_cm")
            if dimension.shape != (3,) or np.any(dimension <= 0):
                raise ValueError(
                    "dimension must contain three positive integers"
                )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> MeshSpec | None:
        if not data:
            return None
        return cls(
            kind=str(data["kind"]),
            filename=data.get("filename"),
            library=str(data.get("library", "moab")),
            lower_left_cm=(
                tuple(data["lower_left_cm"])
                if data.get("lower_left_cm") is not None
                else None
            ),
            upper_right_cm=(
                tuple(data["upper_right_cm"])
                if data.get("upper_right_cm") is not None
                else None
            ),
            dimension=(
                tuple(int(value) for value in data["dimension"])
                if data.get("dimension") is not None
                else None
            ),
            mesh_id=(
                int(data["mesh_id"])
                if data.get("mesh_id") is not None
                else None
            ),
            name=data.get("name"),
        )

    def build(self) -> openmc.MeshBase:
        if self.kind == "unstructured":
            return openmc.UnstructuredMesh(
                self.filename,
                self.library,
                mesh_id=self.mesh_id,
                name=self.name or "",
            )

        mesh = openmc.RegularMesh(mesh_id=self.mesh_id, name=self.name or "")
        mesh.lower_left = self.lower_left_cm
        mesh.upper_right = self.upper_right_cm
        mesh.dimension = self.dimension
        return mesh

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "filename": self.filename,
            "library": self.library,
            "lower_left_cm": (
                list(self.lower_left_cm)
                if self.lower_left_cm is not None
                else None
            ),
            "upper_right_cm": (
                list(self.upper_right_cm)
                if self.upper_right_cm is not None
                else None
            ),
            "dimension": list(self.dimension) if self.dimension else None,
            "mesh_id": self.mesh_id,
            "name": self.name,
        }


@dataclass(frozen=True)
class MagnetRegion:
    """Coarse reactor-scale region representing one magnet interface."""

    name: str
    cell_ids: tuple[int, ...]
    source_region_id: str | int | None = None
    surface_ids: tuple[int, ...] = ()
    phase_space_cell_id: int | None = None
    magnet_id: str | int | None = None
    coil_id: str | int | None = None
    winding_pack_id: str | int | None = None
    volume_cm3: float | None = None
    cell_volumes_cm3: Mapping[int, float] = field(default_factory=dict)
    surface_areas_cm2: Mapping[int, float] = field(default_factory=dict)
    surface_normal_signs: Mapping[int, int] = field(default_factory=dict)
    surface_outward_normals_global: Mapping[
        int, tuple[float, float, float]
    ] = field(default_factory=dict)
    coordinate_frame: CoordinateFrame = field(default_factory=CoordinateFrame)
    mesh: MeshSpec | None = None
    damage_nuclides: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("magnet region name cannot be empty")
        cell_ids = _positive_ids(self.cell_ids, "cell_ids")
        if not cell_ids:
            raise ValueError("cell_ids must contain at least one OpenMC cell")
        surface_ids = _positive_ids(self.surface_ids, "surface_ids")
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "surface_ids", surface_ids)

        if self.phase_space_cell_id is not None:
            source_cell = int(self.phase_space_cell_id)
            if source_cell <= 0:
                raise ValueError("phase_space_cell_id must be positive")
            object.__setattr__(self, "phase_space_cell_id", source_cell)

        if self.volume_cm3 is not None:
            volume = float(self.volume_cm3)
            if not np.isfinite(volume) or volume <= 0.0:
                raise ValueError("volume_cm3 must be finite and positive")
            object.__setattr__(self, "volume_cm3", volume)

        cell_volumes: dict[int, float] = {}
        for cell_id, volume in self.cell_volumes_cm3.items():
            cell_id = int(cell_id)
            volume = float(volume)
            if cell_id not in cell_ids:
                raise ValueError(
                    "cell_volumes_cm3 keys must appear in cell_ids"
                )
            if not np.isfinite(volume) or volume <= 0.0:
                raise ValueError("cell volumes must be finite and positive")
            cell_volumes[cell_id] = volume
        object.__setattr__(self, "cell_volumes_cm3", cell_volumes)

        areas: dict[int, float] = {}
        for surface_id, area in self.surface_areas_cm2.items():
            surface_id = int(surface_id)
            area = float(area)
            if surface_id not in surface_ids:
                raise ValueError(
                    "surface_areas_cm2 keys must appear in surface_ids"
                )
            if not np.isfinite(area) or area <= 0.0:
                raise ValueError("surface areas must be finite and positive")
            areas[surface_id] = area
        object.__setattr__(self, "surface_areas_cm2", areas)

        signs: dict[int, int] = {}
        for surface_id, sign in self.surface_normal_signs.items():
            surface_id = int(surface_id)
            sign = int(sign)
            if surface_id not in surface_ids:
                raise ValueError(
                    "surface_normal_signs keys must appear in surface_ids"
                )
            if sign not in {-1, 1}:
                raise ValueError("surface normal signs must be -1 or +1")
            signs[surface_id] = sign
        object.__setattr__(self, "surface_normal_signs", signs)

        outward_normals: dict[int, tuple[float, float, float]] = {}
        for surface_id, normal in self.surface_outward_normals_global.items():
            surface_id = int(surface_id)
            if surface_id not in surface_ids:
                raise ValueError(
                    "surface_outward_normals_global keys must appear in "
                    "surface_ids"
                )
            unit_normal = _normalise_axis(
                normal, f"surface {surface_id} outward normal"
            )
            outward_normals[surface_id] = tuple(
                float(value) for value in unit_normal
            )
        object.__setattr__(
            self, "surface_outward_normals_global", outward_normals
        )
        object.__setattr__(
            self,
            "damage_nuclides",
            tuple(str(x) for x in self.damage_nuclides),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MagnetRegion:
        return cls(
            name=str(data["name"]),
            cell_ids=tuple(data["cell_ids"]),
            source_region_id=data.get("source_region_id"),
            surface_ids=tuple(data.get("surface_ids", ())),
            phase_space_cell_id=data.get("phase_space_cell_id"),
            magnet_id=data.get("magnet_id"),
            coil_id=data.get("coil_id"),
            winding_pack_id=data.get("winding_pack_id"),
            volume_cm3=data.get("volume_cm3"),
            cell_volumes_cm3={
                int(key): value
                for key, value in data.get("cell_volumes_cm3", {}).items()
            },
            surface_areas_cm2={
                int(key): value
                for key, value in data.get("surface_areas_cm2", {}).items()
            },
            surface_normal_signs={
                int(key): value
                for key, value in data.get("surface_normal_signs", {}).items()
            },
            surface_outward_normals_global={
                int(key): tuple(value)
                for key, value in data.get(
                    "surface_outward_normals_global", {}
                ).items()
            },
            coordinate_frame=CoordinateFrame.from_mapping(
                data.get("coordinate_frame")
            ),
            mesh=MeshSpec.from_mapping(data.get("mesh")),
            damage_nuclides=tuple(data.get("damage_nuclides", ())),
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def source_cell_id(self) -> int:
        if self.phase_space_cell_id is not None:
            return self.phase_space_cell_id
        if len(self.cell_ids) == 1:
            return self.cell_ids[0]
        raise ValueError(
            f"region {self.name!r} has multiple cells; set "
            "phase_space_cell_id for directional surface-source output"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_region_id": self.source_region_id,
            "magnet_id": self.magnet_id,
            "coil_id": self.coil_id,
            "winding_pack_id": self.winding_pack_id,
            "cell_ids": list(self.cell_ids),
            "surface_ids": list(self.surface_ids),
            "phase_space_cell_id": self.phase_space_cell_id,
            "volume_cm3": self.volume_cm3,
            "cell_volumes_cm3": {
                str(key): value for key, value in self.cell_volumes_cm3.items()
            },
            "surface_areas_cm2": {
                str(key): value
                for key, value in self.surface_areas_cm2.items()
            },
            "surface_normal_signs": {
                str(key): value
                for key, value in self.surface_normal_signs.items()
            },
            "surface_outward_normals_global": {
                str(key): list(value)
                for key, value in (self.surface_outward_normals_global.items())
            },
            "coordinate_frame": self.coordinate_frame.to_dict(),
            "mesh": self.mesh.to_dict() if self.mesh else None,
            "damage_nuclides": list(self.damage_nuclides),
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class MagnetSpectralHandoff:
    """Definition and exporter for a reactor-to-magnet spectral handoff."""

    regions: tuple[MagnetRegion, ...]
    energy_bounds_eV: tuple[float, ...]
    energy_group_structure: str | None = None
    particles: tuple[str, ...] = ("neutron", "photon")
    mu_bounds: tuple[float, ...] = (-1.0, 0.0, 1.0)
    time_bounds_s: tuple[float, ...] | None = None
    polar_bounds_rad: tuple[float, ...] | None = None
    azimuthal_bounds_rad: tuple[float, ...] | None = None
    source_rate_per_s: float | None = None
    tally_id_base: int = 9_000_000
    include_cell_flux: bool = True
    include_mesh_flux: bool = True
    include_boundary_current: bool = True
    include_heating: bool = True
    include_damage_energy: bool = True
    gas_production_scores: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("at least one magnet region is required")
        names = [region.name for region in self.regions]
        if len(names) != len(set(names)):
            raise ValueError("magnet region names must be unique")
        slugs = [_slugify(name) for name in names]
        if len(slugs) != len(set(slugs)):
            raise ValueError(
                "magnet region names must remain unique after slugification"
            )

        energy = _validate_edges(
            self.energy_bounds_eV,
            "energy_bounds_eV",
            minimum=0.0,
        )
        if energy[0] < 0.0:
            raise ValueError("energy_bounds_eV cannot contain negative values")
        object.__setattr__(self, "energy_bounds_eV", energy)
        if self.energy_group_structure is not None:
            structure = str(self.energy_group_structure).strip()
            if not structure:
                raise ValueError("energy_group_structure cannot be empty")
            object.__setattr__(self, "energy_group_structure", structure)

        particles = tuple(str(value).lower() for value in self.particles)
        if not particles:
            raise ValueError("particles cannot be empty")
        if len(particles) != len(set(particles)):
            raise ValueError("particles cannot contain duplicates")
        object.__setattr__(self, "particles", particles)

        mu = _validate_edges(
            self.mu_bounds,
            "mu_bounds",
            minimum=-1.0,
            maximum=1.0,
        )
        object.__setattr__(self, "mu_bounds", mu)

        if self.time_bounds_s is not None:
            time = _validate_edges(
                self.time_bounds_s,
                "time_bounds_s",
                minimum=0.0,
            )
            object.__setattr__(self, "time_bounds_s", time)
        if self.polar_bounds_rad is not None:
            polar = _validate_edges(
                self.polar_bounds_rad,
                "polar_bounds_rad",
                minimum=0.0,
                maximum=float(np.pi),
            )
            object.__setattr__(self, "polar_bounds_rad", polar)
        if self.azimuthal_bounds_rad is not None:
            azimuthal = _validate_edges(
                self.azimuthal_bounds_rad,
                "azimuthal_bounds_rad",
                minimum=-float(np.pi),
                maximum=float(np.pi),
            )
            object.__setattr__(self, "azimuthal_bounds_rad", azimuthal)

        if self.source_rate_per_s is not None:
            source_rate = float(self.source_rate_per_s)
            if not np.isfinite(source_rate) or source_rate <= 0.0:
                raise ValueError(
                    "source_rate_per_s must be finite and positive"
                )
            object.__setattr__(self, "source_rate_per_s", source_rate)

        tally_id_base = int(self.tally_id_base)
        if tally_id_base <= 0:
            raise ValueError("tally_id_base must be positive")
        object.__setattr__(self, "tally_id_base", tally_id_base)

        gas_scores = tuple(str(score) for score in self.gas_production_scores)
        if len(gas_scores) != len(set(gas_scores)):
            raise ValueError("gas-production scores cannot contain duplicates")
        object.__setattr__(self, "gas_production_scores", gas_scores)

        if not any(
            (
                self.include_cell_flux,
                self.include_mesh_flux,
                self.include_boundary_current,
                self.include_heating,
                self.include_damage_energy,
                bool(gas_scores),
            )
        ):
            raise ValueError("at least one handoff tally must be enabled")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MagnetSpectralHandoff:
        tally_data = data.get("tallies", {})
        normalization = data.get("normalization", {})
        energy_bounds, energy_structure = _resolve_energy_bounds(data)
        gas_setting = tally_data.get("gas_production", False)
        if gas_setting is True:
            gas_scores = DEFAULT_GAS_PRODUCTION_SCORES
        elif gas_setting is False or gas_setting is None:
            gas_scores = ()
        else:
            gas_scores = tuple(gas_setting)

        return cls(
            regions=tuple(
                MagnetRegion.from_mapping(region) for region in data["regions"]
            ),
            energy_bounds_eV=energy_bounds,
            energy_group_structure=energy_structure,
            particles=tuple(data.get("particles", ("neutron", "photon"))),
            mu_bounds=tuple(data.get("mu_bounds", (-1.0, 0.0, 1.0))),
            time_bounds_s=(
                tuple(data["time_bounds_s"])
                if data.get("time_bounds_s") is not None
                else None
            ),
            polar_bounds_rad=(
                tuple(data["polar_bounds_rad"])
                if data.get("polar_bounds_rad") is not None
                else None
            ),
            azimuthal_bounds_rad=(
                tuple(data["azimuthal_bounds_rad"])
                if data.get("azimuthal_bounds_rad") is not None
                else None
            ),
            source_rate_per_s=normalization.get("source_rate_per_s"),
            tally_id_base=int(data.get("tally_id_base", 9_000_000)),
            include_cell_flux=bool(tally_data.get("cell_flux", True)),
            include_mesh_flux=bool(tally_data.get("mesh_flux", True)),
            include_boundary_current=bool(
                tally_data.get("boundary_current", True)
            ),
            include_heating=bool(tally_data.get("heating", True)),
            include_damage_energy=bool(tally_data.get("damage_energy", True)),
            gas_production_scores=gas_scores,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> MagnetSpectralHandoff:
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, Mapping):
            raise ValueError("handoff YAML root must be a mapping")
        _resolve_relative_mesh_filenames(data, config_path.parent)
        return cls.from_mapping(data)

    def region(self, name: str) -> MagnetRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"unknown magnet region {name!r}")

    @staticmethod
    def _tally_name(region: MagnetRegion, role: str) -> str:
        return f"pstl_magnet_{_slugify(region.name)}_{role}"

    def _tally_id(self, region: MagnetRegion, role: str) -> int:
        region_index = self.regions.index(region)
        return self.tally_id_base + 100 * region_index + _ROLE_OFFSETS[role]

    def tally_catalog(self) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for region in self.regions:
            roles: list[tuple[str, str]] = []
            if self.include_cell_flux:
                roles.append(("cell_flux", "particle-cm/source"))
            if self.include_mesh_flux and region.mesh is not None:
                roles.append(("mesh_flux", "particle-cm/source"))
            if self.include_boundary_current and region.surface_ids:
                roles.append(("boundary_current", "particle/source"))
            if self.include_heating:
                roles.append(("heating", "eV/source"))
            if self.include_damage_energy:
                roles.append(("damage_energy", "eV/source"))
            if self.gas_production_scores:
                roles.append(("gas_production", "products/source"))
            for role, units in roles:
                catalog.append(
                    {
                        "region": region.name,
                        "role": role,
                        "name": self._tally_name(region, role),
                        "id": self._tally_id(region, role),
                        "raw_units": units,
                    }
                )
        return catalog

    def _phase_filters(
        self,
        spatial_filter: openmc.Filter,
        *,
        particles: Sequence[str] | None = None,
    ) -> list[openmc.Filter]:
        filters: list[openmc.Filter] = [
            spatial_filter,
            openmc.ParticleFilter(tuple(particles or self.particles)),
            openmc.EnergyFilter(self.energy_bounds_eV),
        ]
        if self.time_bounds_s is not None:
            filters.append(openmc.TimeFilter(self.time_bounds_s))
        return filters

    def build_tallies(self) -> openmc.Tallies:
        """Build all OpenMC tallies required by the handoff contract."""
        if self.include_boundary_current and not hasattr(
            openmc, "MuSurfaceFilter"
        ):
            raise RuntimeError(
                "direction-resolved boundary current requires OpenMC >=0.15.1"
            )

        tallies = openmc.Tallies()
        for region in self.regions:
            cell_filter = openmc.CellFilter(region.cell_ids)

            if self.include_cell_flux:
                tally = openmc.Tally(
                    tally_id=self._tally_id(region, "cell_flux"),
                    name=self._tally_name(region, "cell_flux"),
                )
                tally.filters = self._phase_filters(cell_filter)
                tally.scores = ["flux"]
                tallies.append(tally)

            if self.include_mesh_flux and region.mesh is not None:
                tally = openmc.Tally(
                    tally_id=self._tally_id(region, "mesh_flux"),
                    name=self._tally_name(region, "mesh_flux"),
                )
                tally.filters = self._phase_filters(
                    openmc.MeshFilter(region.mesh.build())
                )
                tally.scores = ["flux"]
                tallies.append(tally)

            if self.include_boundary_current and region.surface_ids:
                filters = self._phase_filters(
                    openmc.SurfaceFilter(region.surface_ids)
                )
                filters.append(openmc.MuSurfaceFilter(self.mu_bounds))
                if self.polar_bounds_rad is not None:
                    filters.append(openmc.PolarFilter(self.polar_bounds_rad))
                if self.azimuthal_bounds_rad is not None:
                    filters.append(
                        openmc.AzimuthalFilter(self.azimuthal_bounds_rad)
                    )
                tally = openmc.Tally(
                    tally_id=self._tally_id(region, "boundary_current"),
                    name=self._tally_name(region, "boundary_current"),
                )
                tally.filters = filters
                tally.scores = ["current"]
                tally.estimator = "analog"
                tallies.append(tally)

            if self.include_heating:
                tally = openmc.Tally(
                    tally_id=self._tally_id(region, "heating"),
                    name=self._tally_name(region, "heating"),
                )
                tally.filters = self._phase_filters(cell_filter)
                tally.scores = ["heating"]
                tallies.append(tally)

            if self.include_damage_energy:
                tally = openmc.Tally(
                    tally_id=self._tally_id(region, "damage_energy"),
                    name=self._tally_name(region, "damage_energy"),
                )
                tally.filters = self._phase_filters(
                    cell_filter, particles=("neutron",)
                )
                tally.scores = ["damage-energy"]
                if region.damage_nuclides:
                    tally.nuclides = list(region.damage_nuclides)
                tallies.append(tally)

            if self.gas_production_scores:
                tally = openmc.Tally(
                    tally_id=self._tally_id(region, "gas_production"),
                    name=self._tally_name(region, "gas_production"),
                )
                tally.filters = self._phase_filters(
                    cell_filter, particles=("neutron",)
                )
                tally.scores = list(self.gas_production_scores)
                tallies.append(tally)

        return tallies

    def attach_to_model(
        self,
        model: openmc.Model,
        *,
        enable_photon_transport: bool = True,
    ) -> openmc.Tallies:
        """Append handoff tallies to an existing OpenMC model."""
        new_tallies = self.build_tallies()
        if model.tallies is None:
            model.tallies = openmc.Tallies()

        existing_names = {tally.name for tally in model.tallies}
        duplicate_names = existing_names.intersection(
            tally.name for tally in new_tallies
        )
        existing_ids = {tally.id for tally in model.tallies}
        duplicate_ids = existing_ids.intersection(
            tally.id for tally in new_tallies
        )
        if duplicate_names or duplicate_ids:
            details = []
            if duplicate_names:
                details.append("names=" + ",".join(sorted(duplicate_names)))
            if duplicate_ids:
                details.append(
                    "ids="
                    + ",".join(str(value) for value in sorted(duplicate_ids))
                )
            raise ValueError(
                "model already contains handoff tally identifiers: "
                + "; ".join(details)
            )

        for tally in new_tallies:
            model.tallies.append(tally)

        if enable_photon_transport and "photon" in self.particles:
            model.settings.photon_transport = True
        return new_tallies

    def configure_surface_source(
        self,
        settings: openmc.Settings,
        region_name: str,
        *,
        direction: str = "incoming",
        max_particles: int = 100_000,
        max_source_files: int = 1,
        mcpl: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Configure native OpenMC phase-space banking at a magnet boundary.

        ``incoming`` uses ``cellto``, ``outgoing`` uses ``cellfrom``, ``both``
        uses ``cell``, and ``all`` applies only the supplied surface IDs.
        """
        direction = direction.lower()
        if direction not in _VALID_SOURCE_DIRECTIONS:
            allowed = ", ".join(sorted(_VALID_SOURCE_DIRECTIONS))
            raise ValueError(f"direction must be one of {allowed}")
        if int(max_particles) <= 0:
            raise ValueError("max_particles must be positive")
        if int(max_source_files) <= 0:
            raise ValueError("max_source_files must be positive")
        if settings.surf_source_write and not overwrite:
            raise ValueError(
                "settings already define surf_source_write; use overwrite=True"
            )

        region = self.region(region_name)
        if not self.include_boundary_current:
            raise ValueError(
                "phase-space banking requires boundary_current=true so the "
                "bank has a companion normalization tally"
            )
        if not region.surface_ids:
            raise ValueError(
                "phase-space banking requires explicit surface_ids so the "
                "bank and boundary-current tally describe the same interface"
            )

        config: dict[str, Any] = {"max_particles": int(max_particles)}
        config["surface_ids"] = list(region.surface_ids)
        if max_source_files != 1:
            config["max_source_files"] = int(max_source_files)
        if mcpl:
            raise ValueError(
                "the deterministic handoff requires OpenMC HDF5 surface "
                "sources; MCPL does not preserve the full surface-ID contract"
            )

        if direction != "all":
            key = {
                "incoming": "cellto",
                "outgoing": "cellfrom",
                "both": "cell",
            }[direction]
            config[key] = region.source_cell_id

        settings.surf_source_write = config
        return config

    def to_manifest(
        self,
        *,
        statepoint_path: str | Path | None = None,
        surface_source: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "openmc_version": getattr(openmc, "__version__", "unknown"),
            "energy_bounds_eV": list(self.energy_bounds_eV),
            "energy_group_structure": self.energy_group_structure,
            "particles": list(self.particles),
            "mu_bounds": list(self.mu_bounds),
            "time_bounds_s": (
                list(self.time_bounds_s)
                if self.time_bounds_s is not None
                else None
            ),
            "polar_bounds_rad": (
                list(self.polar_bounds_rad)
                if self.polar_bounds_rad is not None
                else None
            ),
            "azimuthal_bounds_rad": (
                list(self.azimuthal_bounds_rad)
                if self.azimuthal_bounds_rad is not None
                else None
            ),
            "tally_id_base": self.tally_id_base,
            "minimum_openmc_version": "0.15.1",
            "tally_switches": {
                "cell_flux": self.include_cell_flux,
                "mesh_flux": self.include_mesh_flux,
                "boundary_current": self.include_boundary_current,
                "heating": self.include_heating,
                "damage_energy": self.include_damage_energy,
                "gas_production": bool(self.gas_production_scores),
            },
            "normalization": {
                "source_rate_per_s": self.source_rate_per_s,
                "rule": (
                    "multiply OpenMC per-source results by source_rate_per_s; "
                    "divide cell flux by volume and boundary current by area"
                ),
            },
            "regions": [region.to_dict() for region in self.regions],
            "tallies": self.tally_catalog(),
            "statepoint_path": (
                str(statepoint_path) if statepoint_path is not None else None
            ),
            "surface_source": _jsonable(surface_source or {}),
            "native_surface_source_fields": [
                "position",
                "direction",
                "energy",
                "time",
                "weight",
                "delayed_group",
                "surface_id",
                "particle_type",
            ],
            "native_surface_source_fields_unavailable": [
                "history_id",
                "parent_id",
                "cell_id",
                "material_id",
            ],
            "phase_space_output_fields_added_by_parastell": [
                "record_id",
                "source_file_index",
                "source_record_index",
                "source_region_id",
                "source_region_name",
                "position_local_cm",
                "direction_local",
                "surface_outward_normal_global",
                "surface_outward_normal_local",
                "mu_outward",
                "magnet_direction",
                "direction_label_basis",
                "particle_name",
                "particle_pdg",
            ],
            "normalization_warning": (
                "Do not infer absolute interface rate from the number of "
                "banked phase-space records. Normalize with the companion "
                "boundary-current tally."
            ),
            "metadata": _jsonable(self.metadata),
        }

    def write_manifest(
        self,
        path: str | Path,
        *,
        statepoint_path: str | Path | None = None,
        surface_source: Mapping[str, Any] | None = None,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.to_manifest(
            statepoint_path=statepoint_path,
            surface_source=surface_source,
        )
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    def export_statepoint(
        self,
        statepoint_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Export handoff tallies from an OpenMC statepoint to tidy HDF5."""
        statepoint_path = Path(statepoint_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        temporary_path.unlink(missing_ok=True)
        statepoint = openmc.StatePoint(statepoint_path)
        try:
            with h5py.File(temporary_path, "w") as output:
                output.attrs["schema"] = SCHEMA_NAME
                output.attrs["schema_version"] = SCHEMA_VERSION
                output.attrs["source_statepoint"] = str(statepoint_path)
                manifest = self.to_manifest(statepoint_path=statepoint_path)
                output.create_dataset(
                    "manifest_json",
                    data=json.dumps(manifest, sort_keys=True),
                    dtype=h5py.string_dtype("utf-8"),
                )

                tallies_group = output.create_group("tallies")
                for item in self.tally_catalog():
                    try:
                        tally = statepoint.get_tally(id=item["id"])
                    except LookupError as exc:
                        raise RuntimeError(
                            f"statepoint is missing tally {item['name']!r}"
                        ) from exc

                    dataframe = _tally_dataframe(tally)
                    table, column_map = _flatten_dataframe(dataframe)
                    region = self.region(item["region"])
                    _add_derived_columns(
                        table,
                        role=item["role"],
                        region=region,
                        source_rate_per_s=self.source_rate_per_s,
                        tally=tally,
                        row_count=len(dataframe),
                    )

                    group = tallies_group.create_group(item["name"])
                    group.attrs["region"] = item["region"]
                    group.attrs["role"] = item["role"]
                    group.attrs["raw_units"] = item["raw_units"]
                    group.attrs["tally_id"] = int(tally.id)
                    group.attrs["column_map_json"] = json.dumps(
                        column_map, sort_keys=True
                    )
                    group.attrs["row_count"] = len(dataframe)
                    _write_column_group(group, table)
            temporary_path.replace(output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        finally:
            statepoint.close()

        return output_path

    def export_surface_source(
        self,
        surface_source_paths: str | Path | Sequence[str | Path],
        output_path: str | Path,
        *,
        region_name: str,
        selection: str,
        record_outward_normals_global: np.ndarray | None = None,
        record_normal_basis: str = "external_per_record_outward_normal",
    ) -> Path:
        """Convert native OpenMC source banks to a local-frame HDF5 file."""
        selection = selection.lower()
        if selection not in _VALID_SOURCE_DIRECTIONS:
            allowed = ", ".join(sorted(_VALID_SOURCE_DIRECTIONS))
            raise ValueError(f"selection must be one of {allowed}")
        region = self.region(region_name)
        if not self.include_boundary_current or not region.surface_ids:
            raise ValueError(
                "surface-source export requires an enabled companion "
                "boundary-current tally and explicit surface_ids"
            )

        if isinstance(surface_source_paths, (str, Path)):
            paths = [Path(surface_source_paths)]
        else:
            paths = [Path(path) for path in surface_source_paths]
        if not paths:
            raise ValueError("at least one surface-source path is required")

        banks: list[dict[str, np.ndarray]] = []
        source_file_metadata: list[dict[str, Any]] = []
        for file_index, path in enumerate(paths):
            bank_data, file_metadata = _read_source_bank(path, file_index)
            banks.append(bank_data)
            source_file_metadata.append(file_metadata)
        bank = {
            name: np.concatenate([entry[name] for entry in banks], axis=0)
            for name in banks[0]
        }
        bank["record_id"] = np.arange(len(bank["energy_eV"]), dtype=int)
        bank["surface_id_abs"] = np.abs(
            np.asarray(bank["surface_id"], dtype=int)
        )
        expected_surfaces = set(region.surface_ids)
        observed_surfaces = set(int(value) for value in bank["surface_id_abs"])
        unexpected_surfaces = sorted(observed_surfaces - expected_surfaces)
        if unexpected_surfaces:
            raise ValueError(
                "surface source contains IDs outside the configured magnet "
                f"interface: {unexpected_surfaces}"
            )

        record_count = len(bank["energy_eV"])
        region_identifier = (
            region.source_region_id
            if region.source_region_id is not None
            else (
                region.magnet_id
                if region.magnet_id is not None
                else _slugify(region.name)
            )
        )
        bank["source_region_id"] = np.full(
            record_count, str(region_identifier), dtype=object
        )
        bank["source_region_name"] = np.full(
            record_count, region.name, dtype=object
        )
        bank["magnet_id"] = np.full(
            record_count,
            "" if region.magnet_id is None else str(region.magnet_id),
            dtype=object,
        )
        bank["coil_id"] = np.full(
            record_count,
            "" if region.coil_id is None else str(region.coil_id),
            dtype=object,
        )
        bank["winding_pack_id"] = np.full(
            record_count,
            (
                ""
                if region.winding_pack_id is None
                else str(region.winding_pack_id)
            ),
            dtype=object,
        )

        local_position = region.coordinate_frame.transform_positions(
            bank["position_global_cm"]
        )
        local_direction = region.coordinate_frame.transform_directions(
            bank["direction_global"]
        )
        outward_global, outward_local, mu_outward = _surface_normal_columns(
            bank,
            region,
            record_outward_normals_global=record_outward_normals_global,
        )
        phase_space_table = dict(bank)
        phase_space_table["position_local_cm"] = local_position
        phase_space_table["direction_local"] = local_direction
        phase_space_table["surface_outward_normal_global"] = outward_global
        phase_space_table["surface_outward_normal_local"] = outward_local
        phase_space_table["mu_outward"] = mu_outward
        geometric_basis = (
            record_normal_basis
            if record_outward_normals_global is not None
            else "configured_outward_normal"
        )
        magnet_direction, direction_basis = _phase_space_direction_columns(
            mu_outward,
            selection,
            geometric_basis=geometric_basis,
        )
        phase_space_table["magnet_direction"] = magnet_direction
        phase_space_table["direction_label_basis"] = direction_basis
        direction_validation = _phase_space_direction_validation(
            magnet_direction, direction_basis, selection
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        temporary_path.unlink(missing_ok=True)
        try:
            with h5py.File(temporary_path, "w") as output:
                output.attrs["schema"] = SCHEMA_NAME
                output.attrs["schema_version"] = SCHEMA_VERSION
                output.attrs["region"] = region_name
                output.attrs["selection"] = selection
                output.attrs["record_count"] = len(local_position)
                output.attrs["normalization_tally"] = self._tally_name(
                    region, "boundary_current"
                )
                output.attrs["normalization_tally_id"] = self._tally_id(
                    region, "boundary_current"
                )
                output.attrs["direction_validation_json"] = json.dumps(
                    direction_validation, sort_keys=True
                )
                output.attrs["absolute_normalization"] = (
                    "Use the companion boundary-current tally; record count "
                    "alone is not an absolute rate."
                )

                phase_space = output.create_group("phase_space")
                _write_column_group(phase_space, phase_space_table)

                metadata = {
                    "region": region.to_dict(),
                    "selection": selection,
                    "source_files": source_file_metadata,
                    "source_rate_per_s": self.source_rate_per_s,
                    "direction_validation": direction_validation,
                    "coordinate_frame": region.coordinate_frame.to_dict(),
                    "record_normal_basis": geometric_basis,
                    "particle_code_contract": (
                        "particle_code_raw preserves the source file; "
                        "particle_name and particle_pdg are normalized across "
                        "OpenMC 0.15 enum and OpenMC 0.16 PDG encodings"
                    ),
                    "fields_not_available_in_native_openmc_source_bank": [
                        "history_id",
                        "parent_id",
                        "cell_id",
                        "material_id",
                    ],
                }
                output.create_dataset(
                    "metadata_json",
                    data=json.dumps(metadata, sort_keys=True),
                    dtype=h5py.string_dtype("utf-8"),
                )

            temporary_path.replace(output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return output_path


def _repeat_and_tile(
    values: np.ndarray, stride: int, data_size: int
) -> np.ndarray:
    repeated = np.repeat(values, int(stride))
    if len(repeated) == 0 or data_size % len(repeated) != 0:
        raise ValueError("mesh-filter bins do not divide the tally data size")
    return np.tile(repeated, data_size // len(repeated))


def _compatible_mesh_filter_dataframe(
    mesh_filter: openmc.MeshFilter,
    data_size: int,
    stride: int,
    **_: Any,
) -> Any:
    """OpenMC 0.16 mesh DataFrame logic for OpenMC 0.15.x users."""
    import pandas as pd

    mesh = mesh_filter.mesh
    mesh_key = f"mesh {mesh.id}"
    if isinstance(mesh, openmc.UnstructuredMesh):
        labels = ("element",)
        index_start = 0
    else:
        labels = tuple(getattr(mesh, "axis_labels", ("x", "y", "z")))
        index_start = 1

    dimensions = tuple(int(value) for value in mesh.dimension)
    columns: dict[tuple[str, str], np.ndarray] = {}
    running_stride = int(stride)
    for label, dimension in zip(labels, dimensions):
        columns[(mesh_key, str(label))] = _repeat_and_tile(
            np.arange(index_start, index_start + dimension),
            running_stride,
            int(data_size),
        )
        running_stride *= dimension
    return pd.DataFrame(columns)


def _tally_dataframe(tally: Any) -> Any:
    """Return a tally DataFrame, including OpenMC 0.15 unstructured meshes."""
    has_unstructured = any(
        isinstance(item, openmc.MeshFilter)
        and isinstance(item.mesh, openmc.UnstructuredMesh)
        for item in getattr(tally, "filters", ())
    )
    if not has_unstructured:
        return tally.get_pandas_dataframe(paths=False)

    original = openmc.MeshFilter.get_pandas_dataframe
    openmc.MeshFilter.get_pandas_dataframe = _compatible_mesh_filter_dataframe
    try:
        return tally.get_pandas_dataframe(paths=False)
    finally:
        openmc.MeshFilter.get_pandas_dataframe = original


def _flatten_column_name(column: Any) -> str:
    if isinstance(column, tuple):
        parts = [
            str(part).strip()
            for part in column
            if str(part).strip() not in {"", "None", "nan"}
        ]
        value = "__".join(parts)
    else:
        value = str(column)
    value = value.lower().replace("std. dev.", "std_dev")
    value = value.replace("[", "_").replace("]", "")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "column"


def _flatten_dataframe(dataframe: Any) -> tuple[dict[str, np.ndarray], dict]:
    result: dict[str, np.ndarray] = {}
    column_map: dict[str, str] = {}
    used: set[str] = set()
    for column in dataframe.columns:
        base_name = _flatten_column_name(column)
        name = base_name
        suffix = 2
        while name in used:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used.add(name)
        result[name] = dataframe[column].to_numpy()
        column_map[name] = str(column)
    return result, column_map


def _find_column(
    table: Mapping[str, np.ndarray], *candidates: str
) -> str | None:
    for candidate in candidates:
        if candidate in table:
            return candidate
    for name in table:
        if any(candidate in name for candidate in candidates):
            return name
    return None


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=np.isfinite(denominator) & (denominator > 0.0),
    )


def _cell_row_volumes(
    table: Mapping[str, np.ndarray], region: MagnetRegion
) -> np.ndarray | None:
    cell_name = _find_column(table, "cell")
    if cell_name is not None and region.cell_volumes_cm3:
        return np.asarray(
            [
                region.cell_volumes_cm3.get(int(cell_id), np.nan)
                for cell_id in table[cell_name]
            ],
            dtype=float,
        )
    if len(region.cell_ids) == 1 and region.volume_cm3 is not None:
        return np.full(len(next(iter(table.values()))), region.volume_cm3)
    return None


def _mesh_row_volumes(
    table: Mapping[str, np.ndarray], tally: Any, row_count: int
) -> np.ndarray | None:
    for mesh_filter in getattr(tally, "filters", ()):
        if not isinstance(mesh_filter, openmc.MeshFilter):
            continue
        mesh = mesh_filter.mesh
        try:
            volumes = np.asarray(mesh.volumes, dtype=float)
        except (AttributeError, RuntimeError, ValueError):
            return None

        prefix = f"mesh_{mesh.id}_"
        if isinstance(mesh, openmc.UnstructuredMesh):
            element_name = _find_column(table, prefix + "element")
            if element_name is None:
                return None
            indices = np.asarray(table[element_name], dtype=int)
            flat = volumes.reshape(-1)
            if np.any(indices < 0) or np.any(indices >= len(flat)):
                return None
            return flat[indices]

        axis_labels = tuple(getattr(mesh, "axis_labels", ("x", "y", "z")))
        index_columns: list[np.ndarray] = []
        for label in axis_labels:
            name = _find_column(table, prefix + str(label).lower())
            if name is None:
                return None
            index_columns.append(np.asarray(table[name], dtype=int) - 1)
        try:
            row_volumes = volumes[tuple(index_columns)]
        except (IndexError, ValueError):
            return None
        if len(row_volumes) != row_count:
            return None
        return np.asarray(row_volumes, dtype=float)
    return None


def _add_differential_column(
    table: dict[str, np.ndarray],
    name: str,
    values: np.ndarray,
    width: np.ndarray | None,
) -> None:
    if width is not None:
        table[name + "_eV_1"] = _safe_divide(values, width)


def _add_derived_columns(
    table: dict[str, np.ndarray],
    *,
    role: str,
    region: MagnetRegion,
    source_rate_per_s: float | None,
    tally: Any,
    row_count: int,
) -> None:
    if role == "boundary_current":
        _add_direction_labels(table, region)

    mean_name = _find_column(table, "mean")
    std_name = _find_column(table, "std_dev")
    if mean_name is None or std_name is None:
        return

    mean = np.asarray(table[mean_name], dtype=float)
    std_dev = np.asarray(table[std_name], dtype=float)
    relative_error = np.full(mean.shape, np.nan, dtype=float)
    nonzero = mean != 0.0
    relative_error[nonzero] = std_dev[nonzero] / np.abs(mean[nonzero])
    table["relative_error"] = relative_error

    width: np.ndarray | None = None
    energy_low_name = _find_column(table, "energy_low_ev")
    energy_high_name = _find_column(table, "energy_high_ev")
    if energy_low_name and energy_high_name:
        width = np.asarray(table[energy_high_name], dtype=float) - np.asarray(
            table[energy_low_name], dtype=float
        )
        table["energy_width_eV"] = width
        table["mean_per_eV"] = _safe_divide(mean, width)
        table["std_dev_per_eV"] = _safe_divide(std_dev, width)

    cell_volumes = None
    if role in {
        "cell_flux",
        "heating",
        "damage_energy",
        "gas_production",
    }:
        cell_volumes = _cell_row_volumes(table, region)
        if cell_volumes is not None:
            table["cell_volume_cm3"] = cell_volumes

    if role == "cell_flux" and cell_volumes is not None:
        flux = _safe_divide(mean, cell_volumes)
        flux_std = _safe_divide(std_dev, cell_volumes)
        table["flux_per_source_cm_2"] = flux
        table["flux_std_dev_per_source_cm_2"] = flux_std
        _add_differential_column(table, "flux_per_source_cm_2", flux, width)
        _add_differential_column(
            table, "flux_std_dev_per_source_cm_2", flux_std, width
        )

    elif role == "mesh_flux":
        mesh_volumes = _mesh_row_volumes(table, tally, row_count)
        if mesh_volumes is not None:
            table["mesh_volume_cm3"] = mesh_volumes
            flux = _safe_divide(mean, mesh_volumes)
            flux_std = _safe_divide(std_dev, mesh_volumes)
            table["flux_per_source_cm_2"] = flux
            table["flux_std_dev_per_source_cm_2"] = flux_std
            _add_differential_column(
                table, "flux_per_source_cm_2", flux, width
            )
            _add_differential_column(
                table,
                "flux_std_dev_per_source_cm_2",
                flux_std,
                width,
            )

    elif role == "boundary_current":
        surface_name = _find_column(table, "surface")
        if surface_name and region.surface_areas_cm2:
            areas = np.asarray(
                [
                    region.surface_areas_cm2.get(int(surface), np.nan)
                    for surface in table[surface_name]
                ],
                dtype=float,
            )
            table["surface_area_cm2"] = areas
            current = _safe_divide(mean, areas)
            current_std = _safe_divide(std_dev, areas)
            table["current_per_source_cm_2"] = current
            table["current_std_dev_per_source_cm_2"] = current_std
            _add_differential_column(
                table, "current_per_source_cm_2", current, width
            )
            _add_differential_column(
                table,
                "current_std_dev_per_source_cm_2",
                current_std,
                width,
            )

    elif role == "heating":
        heating_j = mean * EV_TO_J
        heating_j_std = std_dev * EV_TO_J
        table["heating_J_per_source"] = heating_j
        table["heating_std_dev_J_per_source"] = heating_j_std
        if cell_volumes is not None:
            table["heating_J_cm_3_per_source"] = _safe_divide(
                heating_j, cell_volumes
            )
            table["heating_std_dev_J_cm_3_per_source"] = _safe_divide(
                heating_j_std, cell_volumes
            )

    elif role == "damage_energy" and cell_volumes is not None:
        table["damage_energy_eV_cm_3_per_source"] = _safe_divide(
            mean, cell_volumes
        )
        table["damage_energy_std_dev_eV_cm_3_per_source"] = _safe_divide(
            std_dev, cell_volumes
        )

    elif role == "gas_production" and cell_volumes is not None:
        table["production_cm_3_per_source"] = _safe_divide(mean, cell_volumes)
        table["production_std_dev_cm_3_per_source"] = _safe_divide(
            std_dev, cell_volumes
        )

    if source_rate_per_s is None:
        return

    rate = mean * source_rate_per_s
    rate_std_dev = std_dev * source_rate_per_s
    table["mean_rate_per_s"] = rate
    table["std_dev_rate_per_s"] = rate_std_dev

    if role in {"cell_flux", "mesh_flux"}:
        volume_name = (
            "cell_volume_cm3" if role == "cell_flux" else "mesh_volume_cm3"
        )
        if volume_name in table:
            flux = _safe_divide(rate, table[volume_name])
            flux_std = _safe_divide(rate_std_dev, table[volume_name])
            table["flux_cm_2_s_1"] = flux
            table["flux_std_dev_cm_2_s_1"] = flux_std
            _add_differential_column(table, "flux_cm_2_s_1", flux, width)
            _add_differential_column(
                table, "flux_std_dev_cm_2_s_1", flux_std, width
            )

    elif role == "boundary_current" and "surface_area_cm2" in table:
        current = _safe_divide(rate, table["surface_area_cm2"])
        current_std = _safe_divide(rate_std_dev, table["surface_area_cm2"])
        table["current_density_cm_2_s_1"] = current
        table["current_density_std_dev_cm_2_s_1"] = current_std
        _add_differential_column(
            table, "current_density_cm_2_s_1", current, width
        )
        _add_differential_column(
            table,
            "current_density_std_dev_cm_2_s_1",
            current_std,
            width,
        )

    elif role == "heating":
        power = rate * EV_TO_J
        table["heating_W"] = power
        table["heating_std_dev_W"] = rate_std_dev * EV_TO_J
        if cell_volumes is not None:
            table["heating_W_cm_3"] = _safe_divide(power, cell_volumes)
            table["heating_std_dev_W_cm_3"] = _safe_divide(
                rate_std_dev * EV_TO_J, cell_volumes
            )

    elif role == "damage_energy" and cell_volumes is not None:
        table["damage_energy_eV_cm_3_s_1"] = _safe_divide(rate, cell_volumes)
        table["damage_energy_std_dev_eV_cm_3_s_1"] = _safe_divide(
            rate_std_dev, cell_volumes
        )

    elif role == "gas_production" and cell_volumes is not None:
        table["production_cm_3_s_1"] = _safe_divide(rate, cell_volumes)
        table["production_std_dev_cm_3_s_1"] = _safe_divide(
            rate_std_dev, cell_volumes
        )


def _add_direction_labels(
    table: dict[str, np.ndarray], region: MagnetRegion
) -> None:
    low_name = _find_column(table, "musurface_low")
    high_name = _find_column(table, "musurface_high")
    surface_name = _find_column(table, "surface")
    if low_name is None or high_name is None:
        return

    low = np.asarray(table[low_name], dtype=float)
    high = np.asarray(table[high_name], dtype=float)
    labels: list[str] = []
    for index, (lower, upper) in enumerate(zip(low, high)):
        if upper <= 0.0:
            mu_sign = -1
        elif lower >= 0.0:
            mu_sign = 1
        else:
            labels.append("mixed_mu")
            continue

        if surface_name is None:
            labels.append("negative_mu" if mu_sign < 0 else "positive_mu")
            continue
        surface_id = int(table[surface_name][index])
        normal_sign = region.surface_normal_signs.get(surface_id)
        if normal_sign is None:
            labels.append("negative_mu" if mu_sign < 0 else "positive_mu")
            continue
        outward_mu_sign = mu_sign * normal_sign
        labels.append("incoming" if outward_mu_sign < 0 else "outgoing")

    table["magnet_direction"] = np.asarray(labels, dtype=object)


def _phase_space_direction_columns(
    mu_outward: np.ndarray,
    selection: str,
    *,
    geometric_basis: str = "configured_outward_normal",
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu_outward, dtype=float)
    labels = np.full(mu.shape, "unknown", dtype=object)
    basis = np.full(mu.shape, "unavailable", dtype=object)
    known = np.isfinite(mu)
    labels[known & (mu < -1.0e-12)] = "incoming"
    labels[known & (mu > 1.0e-12)] = "outgoing"
    labels[known & (np.abs(mu) <= 1.0e-12)] = "grazing"
    basis[known] = geometric_basis

    if selection in {"incoming", "outgoing"}:
        inferred = ~known
        labels[inferred] = selection
        basis[inferred] = "openmc_cell_selection"
    return labels, basis


def _phase_space_direction_validation(
    labels: np.ndarray, basis: np.ndarray, selection: str
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=object)
    basis = np.asarray(basis, dtype=object)
    counts = {
        label: int(np.count_nonzero(labels == label))
        for label in ("incoming", "outgoing", "grazing", "unknown")
    }
    basis_counts = {
        str(label): int(np.count_nonzero(basis == label))
        for label in np.unique(basis)
    }
    expected = None
    mismatches = 0
    if selection in {"incoming", "outgoing"}:
        expected = selection
        geometric = ~np.isin(basis, ("openmc_cell_selection", "unavailable"))
        directional = np.isin(labels, ("incoming", "outgoing"))
        mismatches = int(
            np.count_nonzero(geometric & directional & (labels != selection))
        )
    return {
        "selection": selection,
        "expected_direction": expected,
        "counts": counts,
        "basis_counts": basis_counts,
        "geometric_direction_mismatches": mismatches,
        "note": (
            "Configured constant normals are valid only for planar or locally "
            "planar patches. For cellto/cellfrom banks, records without a "
            "configured normal inherit the guaranteed OpenMC cell-selection "
            "direction."
        ),
    }


def _surface_normal_columns(
    bank: Mapping[str, np.ndarray],
    region: MagnetRegion,
    *,
    record_outward_normals_global: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(bank["surface_id"])
    outward_global = np.full((count, 3), np.nan, dtype=float)
    if record_outward_normals_global is not None:
        supplied = np.asarray(record_outward_normals_global, dtype=float)
        if supplied.shape != (count, 3):
            raise ValueError(
                "record_outward_normals_global must have shape "
                f"({count}, 3), got {supplied.shape}"
            )
        norms = np.linalg.norm(supplied, axis=1)
        if not np.all(np.isfinite(supplied)) or np.any(norms <= 0.0):
            raise ValueError(
                "record_outward_normals_global contains invalid vectors"
            )
        outward_global = supplied / norms[:, None]
    if record_outward_normals_global is None:
        for (
            surface_id,
            normal,
        ) in region.surface_outward_normals_global.items():
            mask = (
                np.abs(np.asarray(bank["surface_id"], dtype=int)) == surface_id
            )
            outward_global[mask] = normal

    outward_local = region.coordinate_frame.transform_directions(
        outward_global
    )
    direction = np.asarray(bank["direction_global"], dtype=float)
    mu_outward = np.einsum("ij,ij->i", direction, outward_global)
    return outward_global, outward_local, mu_outward


def _write_column_group(
    group: h5py.Group, table: Mapping[str, np.ndarray]
) -> None:
    string_dtype = h5py.string_dtype("utf-8")
    for name, values in table.items():
        array = np.asarray(values)
        if array.dtype.kind in {"O", "U", "S"}:
            encoded = np.asarray(array, dtype=str).astype(object)
            group.create_dataset(name, data=encoded, dtype=string_dtype)
        else:
            group.create_dataset(name, data=array)


def _extract_xyz(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.names:
        names = values.dtype.names
        if all(axis in names for axis in ("x", "y", "z")):
            return np.column_stack(
                (values["x"], values["y"], values["z"])
            ).astype(float)
    if values.ndim == 2 and values.shape[1] == 3:
        return values.astype(float)
    raise ValueError(f"source-bank field {name!r} is not a 3-vector")


def _read_source_bank(
    path: Path, file_index: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(path, "r") as source:
        if "source_bank" not in source:
            raise ValueError(f"{path} does not contain a source_bank dataset")
        bank = source["source_bank"][()]
        attributes = {
            str(key): _jsonable(value) for key, value in source.attrs.items()
        }

    required = {
        "r",
        "u",
        "E",
        "time",
        "wgt",
        "delayed_group",
        "surf_id",
        "particle",
    }
    missing = required.difference(bank.dtype.names or ())
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} source_bank is missing: {missing_text}")

    raw_particle_codes = np.asarray(bank["particle"], dtype=int)
    particle_names, particle_pdg, encoding = _normalise_particle_codes(
        raw_particle_codes
    )
    count = len(bank)
    data = {
        "position_global_cm": _extract_xyz(bank["r"], "r"),
        "direction_global": _extract_xyz(bank["u"], "u"),
        "energy_eV": np.asarray(bank["E"], dtype=float),
        "time_s": np.asarray(bank["time"], dtype=float),
        "weight": np.asarray(bank["wgt"], dtype=float),
        "delayed_group": np.asarray(bank["delayed_group"], dtype=int),
        "surface_id": np.asarray(bank["surf_id"], dtype=int),
        "particle_code_raw": raw_particle_codes,
        "particle_name": particle_names,
        "particle_pdg": particle_pdg,
        "source_file_index": np.full(count, file_index, dtype=int),
        "source_record_index": np.arange(count, dtype=int),
    }
    metadata = {
        "path": str(path),
        "record_count": count,
        "particle_code_encoding": encoding,
        "attributes": attributes,
    }
    return data, metadata
