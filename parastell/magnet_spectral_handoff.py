"""OpenMC spectral handoff utilities for ParaStell magnet models.

This module builds energy-, particle-, direction-, and space-resolved OpenMC
Tallies around coarse reactor-scale magnet regions and exports their results to
an HDF5 contract suitable for local deterministic transport models. It also
configures OpenMC surface-source writing and converts the resulting phase-space
bank into a magnet-local coordinate frame.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import openmc
import yaml


SCHEMA_NAME = "parastell.magnet_boundary_source"
SCHEMA_VERSION = "1.0.0"
SCHEMA_URI = f"{SCHEMA_NAME}/v{SCHEMA_VERSION}"
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
SMOKE_NEUTRON_7_GROUP_BOUNDS_EV = (
    0.0,
    1.0e5,
    1.0e6,
    5.0e6,
    10.0e6,
    14.0e6,
    14.2e6,
    20.0e6,
)


def software_validation_energy_bounds() -> tuple[float, ...]:
    """Return a 175+ group software-scaling structure.

    This is deliberately not advertised as a nuclear-data group structure. It
    combines logarithmic coverage with explicit fusion-region boundaries so
    array sizing, serialization, and replay can be tested at useful scale.
    """

    logarithmic = np.geomspace(1.0e-5, 20.0e6, 177)
    fusion = np.asarray((13.5e6, 14.0e6, 14.1e6, 14.2e6, 14.5e6))
    return tuple(
        float(value) for value in np.unique(np.r_[0.0, logarithmic, fusion])
    )


PARASTELL_GROUP_STRUCTURES = {
    "PSTL-SMOKE-7": SMOKE_NEUTRON_7_GROUP_BOUNDS_EV,
    "PSTL-SOFTWARE-175+": software_validation_energy_bounds(),
}


def available_energy_group_structures() -> dict[str, dict[str, Any]]:
    """Return discoverable ParaStell and OpenMC energy structures."""

    structures = {
        name: {
            "group_count": len(edges) - 1,
            "classification": (
                "software_validation"
                if name == "PSTL-SOFTWARE-175+"
                else "fast_smoke_test"
            ),
            "source": "parastell",
        }
        for name, edges in PARASTELL_GROUP_STRUCTURES.items()
    }
    try:
        from openmc.mgxs import GROUP_STRUCTURES

        structures.update(
            {
                str(name): {
                    "group_count": len(edges) - 1,
                    "classification": "openmc_named_structure",
                    "source": "openmc.mgxs.GROUP_STRUCTURES",
                }
                for name, edges in GROUP_STRUCTURES.items()
            }
        )
    except ImportError:
        pass
    return dict(sorted(structures.items()))


def load_energy_group_edges(
    path: str | Path,
    *,
    units: str = "eV",
) -> tuple[float, ...]:
    """Load energy edges from JSON, CSV/text, or NumPy storage."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get(
                "energy_bounds", payload.get("energy_bounds_eV")
            )
        if payload is None:
            raise ValueError("JSON energy file has no energy_bounds array")
        values = np.asarray(payload, dtype=float)
    elif suffix in {".csv", ".txt", ".dat"}:
        values = np.loadtxt(
            source, delimiter="," if suffix == ".csv" else None
        )
    else:
        raise ValueError(
            "energy-group files must be JSON, CSV, text, or NumPy"
        )
    scale = {"ev": 1.0, "kev": 1.0e3, "mev": 1.0e6}.get(units.lower())
    if scale is None:
        raise ValueError("energy units must be eV, keV, or MeV")
    return _validate_edges(
        np.asarray(values, dtype=float).reshape(-1) * scale,
        "energy_bounds_eV",
        minimum=0.0,
    )


def _resolve_energy_bounds(
    data: Mapping[str, Any],
    *,
    base_directory: Path | None = None,
) -> tuple[tuple[float, ...], str | None]:
    explicit = data.get("energy_bounds_eV")
    structure = data.get("energy_group_structure")
    file_spec = data.get("energy_group_file")
    if (
        sum(value is not None for value in (explicit, structure, file_spec))
        != 1
    ):
        raise ValueError(
            "set exactly one of energy_bounds_eV, energy_group_structure, or "
            "energy_group_file"
        )
    if explicit is not None:
        return tuple(explicit), None
    if file_spec is not None:
        if isinstance(file_spec, Mapping):
            filename = file_spec.get("path")
            units = str(file_spec.get("units", "eV"))
        else:
            filename = file_spec
            units = "eV"
        if not filename:
            raise ValueError("energy_group_file requires a path")
        path = Path(str(filename)).expanduser()
        if not path.is_absolute() and base_directory is not None:
            path = base_directory / path
        return load_energy_group_edges(path, units=units), None

    structure = str(structure)
    if structure in PARASTELL_GROUP_STRUCTURES:
        return PARASTELL_GROUP_STRUCTURES[structure], structure
    from openmc.mgxs import GROUP_STRUCTURES

    try:
        bounds = GROUP_STRUCTURES[structure]
    except KeyError as exc:
        raise ValueError(
            f"unknown OpenMC energy group structure {structure!r}"
        ) from exc
    return tuple(float(value) for value in bounds), structure


def _coerce_id(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _coerce_region_id(value: Any, *, name: str) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    return text


def _optional_units(value: Any) -> str:
    if value is None:
        return "cm"
    text = str(value).strip()
    if not text:
        raise ValueError("units cannot be empty")
    return text


def _solid_angle_from_mu_bounds(mu_bounds: Sequence[float]) -> np.ndarray:
    values = np.asarray(mu_bounds, dtype=float)
    return 2.0 * np.pi * np.diff(values)


def _safe_float_list(
    values: Sequence[float] | np.ndarray | None,
) -> list[float] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    array = array.ravel()
    return [float(value) for value in array]


def _time_now_iso8601() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_hash(value: Any) -> str:
    data = json.dumps(_jsonable(value), sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _vector_to_bin_index(
    values: np.ndarray, bounds: tuple[float, ...]
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=int)
    values = np.asarray(values, dtype=float)
    edges = np.asarray(bounds, dtype=float)
    index = np.digitize(values, edges, right=False) - 1
    index = index.astype(int)
    upper_boundary = np.isclose(values, edges[-1], rtol=0.0, atol=0.0)
    index[upper_boundary] = (values[upper_boundary] > 0.0).astype(int)
    invalid = ~np.isfinite(values) | (index < 0) | (index >= len(edges) - 1)
    invalid[upper_boundary] = False
    index[invalid] = -1
    return index


def _default_polar_bounds_rad() -> tuple[float, float]:
    return (0.0, float(np.pi))


def _default_azimuthal_bounds_rad() -> tuple[float, float]:
    return (-float(np.pi), float(np.pi))


def _solid_angle_from_bins(
    mu_edges: Sequence[float],
    azimuth_edges: Sequence[float],
) -> np.ndarray:
    mu = np.asarray(mu_edges, dtype=float)
    az = np.asarray(azimuth_edges, dtype=float)
    mu_width = np.abs(np.diff(mu))
    az_width = np.abs(np.diff(az))
    return np.outer(mu_width, az_width)


def _safe_shape_tuple(
    values: Sequence[int], default: tuple[int, int, int] | None = None
) -> tuple[int, int, int]:
    if values is None:
        return default or (0, 0, 0)
    array = tuple(int(value) for value in values)
    if len(array) != 3:
        raise ValueError("dimension must contain 3 values")
    return array


def _normalise_angle_angle(values: np.ndarray) -> np.ndarray:
    angle = np.asarray(values, dtype=float)
    angle = np.mod(angle + np.pi, 2.0 * np.pi) - np.pi
    return angle


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
class MagnetCouplingPlane:
    """Finite rectangular phase-space interface attached to one magnet."""

    plane_id: str
    magnet_region_id: str | int
    name: str
    role: str
    origin_cm: tuple[float, float, float]
    normal_global: tuple[float, float, float]
    u_global: tuple[float, float, float]
    v_global: tuple[float, float, float]
    width_cm: float
    height_cm: float
    u_edges_cm: tuple[float, ...]
    v_edges_cm: tuple[float, ...]
    magnet_component: str
    surface_id: int
    normal_orientation: str = "outward_from_magnet"
    entry_sense: str = "mu < -mu_tolerance"
    exit_sense: str = "mu > mu_tolerance"
    coordinate_units: str = "cm"
    area_units: str = "cm2"
    geometry_source: Mapping[str, Any] = field(default_factory=dict)
    construction_method: str = "explicit"
    mu_tolerance: float = 1.0e-12
    adjacency_tolerance_cm: float = 1.0e-6
    association: Mapping[str, Any] = field(default_factory=dict)
    proxy: bool = False

    def __post_init__(self) -> None:
        for field_name in ("plane_id", "name", "magnet_component"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be empty")
        object.__setattr__(
            self,
            "magnet_region_id",
            _coerce_region_id(self.magnet_region_id, name="magnet_region_id"),
        )
        role = str(self.role).lower()
        if role not in {"entry", "exit", "diagnostic"}:
            raise ValueError("plane role must be entry, exit, or diagnostic")
        object.__setattr__(self, "role", role)
        origin = np.asarray(self.origin_cm, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("plane origin must contain three finite values")
        normal = _normalise_axis(self.normal_global, "plane normal")
        u_axis = _normalise_axis(self.u_global, "plane u vector")
        v_axis = _normalise_axis(self.v_global, "plane v vector")
        frame = np.vstack((u_axis, v_axis, normal))
        if not np.allclose(frame @ frame.T, np.eye(3), atol=1.0e-10):
            raise ValueError(
                "plane u, v, and normal vectors must be orthonormal"
            )
        if not np.allclose(np.cross(u_axis, v_axis), normal, atol=1.0e-10):
            raise ValueError("plane basis must satisfy u x v = normal")
        width = float(self.width_cm)
        height = float(self.height_cm)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("plane width_cm must be finite and positive")
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError("plane height_cm must be finite and positive")
        u_edges = _validate_edges(self.u_edges_cm, "u_edges_cm")
        v_edges = _validate_edges(self.v_edges_cm, "v_edges_cm")
        if not np.allclose((u_edges[0], u_edges[-1]), (-width / 2, width / 2)):
            raise ValueError("u_edges_cm must span the complete plane width")
        if not np.allclose(
            (v_edges[0], v_edges[-1]), (-height / 2, height / 2)
        ):
            raise ValueError("v_edges_cm must span the complete plane height")
        tolerance = float(self.mu_tolerance)
        adjacency = float(self.adjacency_tolerance_cm)
        if not np.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
            raise ValueError("mu_tolerance must be finite and lie in [0, 1)")
        if not np.isfinite(adjacency) or adjacency < 0.0:
            raise ValueError("adjacency_tolerance_cm must be non-negative")
        object.__setattr__(self, "origin_cm", tuple(float(x) for x in origin))
        object.__setattr__(
            self, "normal_global", tuple(float(x) for x in normal)
        )
        object.__setattr__(self, "u_global", tuple(float(x) for x in u_axis))
        object.__setattr__(self, "v_global", tuple(float(x) for x in v_axis))
        object.__setattr__(self, "width_cm", width)
        object.__setattr__(self, "height_cm", height)
        object.__setattr__(self, "u_edges_cm", u_edges)
        object.__setattr__(self, "v_edges_cm", v_edges)
        object.__setattr__(
            self, "surface_id", _coerce_id(self.surface_id, name="surface_id")
        )
        object.__setattr__(self, "mu_tolerance", tolerance)
        object.__setattr__(self, "adjacency_tolerance_cm", adjacency)
        object.__setattr__(self, "geometry_source", dict(self.geometry_source))
        object.__setattr__(self, "association", dict(self.association))
        if self.proxy and self.construction_method != "first_wall_proxy":
            raise ValueError(
                "proxy planes must use first_wall_proxy construction"
            )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        magnet_region_id: str | int | None = None,
        magnet_component: str | None = None,
    ) -> MagnetCouplingPlane:
        placement = dict(data.get("placement", {}))
        extent = dict(data.get("extent", {}))
        spatial = dict(data.get("spatial_bins", {}))
        mode = str(
            placement.get("mode", data.get("construction_method", "explicit"))
        )
        origin = placement.get("origin_cm", data.get("origin_cm"))
        normal = placement.get("normal", data.get("normal_global"))
        reference = placement.get("reference_direction", data.get("u_global"))
        if origin is None or normal is None or reference is None:
            raise ValueError(
                f"{mode} plane definitions require resolved origin, normal, and "
                "reference_direction values"
            )
        normal_axis = _normalise_axis(normal, "plane normal")
        reference_axis = np.asarray(reference, dtype=float)
        reference_axis -= np.dot(reference_axis, normal_axis) * normal_axis
        u_axis = _normalise_axis(reference_axis, "plane reference direction")
        v_axis = np.cross(normal_axis, u_axis)
        width = float(extent.get("width_cm", data.get("width_cm")))
        height = float(extent.get("height_cm", data.get("height_cm")))
        u_spec = data.get("u_edges_cm", spatial.get("u", 1))
        v_spec = data.get("v_edges_cm", spatial.get("v", 1))
        u_edges = (
            np.linspace(-width / 2, width / 2, int(u_spec) + 1)
            if np.isscalar(u_spec)
            else u_spec
        )
        v_edges = (
            np.linspace(-height / 2, height / 2, int(v_spec) + 1)
            if np.isscalar(v_spec)
            else v_spec
        )
        region_id = data.get("magnet_region_id", magnet_region_id)
        component = data.get("magnet_component", magnet_component)
        return cls(
            plane_id=str(data.get("plane_id", data.get("id", ""))),
            magnet_region_id=region_id,
            name=str(data.get("name", data.get("id", ""))),
            role=str(data.get("role", "diagnostic")),
            origin_cm=tuple(origin),
            normal_global=tuple(normal_axis),
            u_global=tuple(u_axis),
            v_global=tuple(v_axis),
            width_cm=width,
            height_cm=height,
            u_edges_cm=tuple(u_edges),
            v_edges_cm=tuple(v_edges),
            magnet_component=str(component or ""),
            surface_id=int(data["surface_id"]),
            normal_orientation=str(
                data.get("normal_orientation", "outward_from_magnet")
            ),
            geometry_source=dict(data.get("geometry_source", {})),
            construction_method=mode,
            mu_tolerance=float(data.get("mu_tolerance", 1.0e-12)),
            adjacency_tolerance_cm=float(
                data.get("adjacency_tolerance_cm", 1.0e-6)
            ),
            association=dict(data.get("association", {})),
            proxy=mode == "first_wall_proxy" or bool(data.get("proxy", False)),
        )

    @classmethod
    def from_magnet_solid(
        cls,
        solid: Any,
        *,
        plane_id: str,
        magnet_region_id: str | int,
        magnet_component: str,
        role: str,
        approximate_location_cm: Sequence[float],
        reference_direction: Sequence[float],
        width_cm: float,
        height_cm: float,
        u_bins: int,
        v_bins: int,
        surface_id: int,
        geometry_source: Mapping[str, Any],
    ) -> MagnetCouplingPlane:
        """Construct a tangent patch from the nearest face of a CAD solid."""

        target = np.asarray(approximate_location_cm, dtype=float)
        faces = list(solid.Faces())
        if not faces:
            raise ValueError("selected magnet component has no faces")
        centers = np.asarray(
            [
                [face.Center().x, face.Center().y, face.Center().z]
                for face in faces
            ]
        )
        face = faces[int(np.argmin(np.linalg.norm(centers - target, axis=1)))]
        center = face.Center()
        origin = np.asarray((center.x, center.y, center.z), dtype=float)
        normal_value = face.normalAt()
        normal = np.asarray(
            (normal_value.x, normal_value.y, normal_value.z), dtype=float
        )
        solid_center = solid.Center()
        outward = origin - np.asarray(
            (solid_center.x, solid_center.y, solid_center.z)
        )
        if np.dot(normal, outward) < 0.0:
            normal *= -1.0
        payload = {
            "id": plane_id,
            "name": plane_id,
            "role": role,
            "surface_id": surface_id,
            "magnet_region_id": magnet_region_id,
            "magnet_component": magnet_component,
            "placement": {
                "mode": "magnet_attached",
                "origin_cm": origin,
                "normal": normal,
                "reference_direction": reference_direction,
            },
            "extent": {"width_cm": width_cm, "height_cm": height_cm},
            "spatial_bins": {"u": u_bins, "v": v_bins},
            "geometry_source": dict(geometry_source),
            "association": {
                "minimum_plane_to_magnet_distance_cm": 0.0,
                "criterion": "plane origin is the center of the nearest selected CAD face",
                "approximate_location_cm": target.tolist(),
            },
        }
        return cls.from_mapping(payload)

    @property
    def area_cm2(self) -> float:
        return self.width_cm * self.height_cm

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return len(self.u_edges_cm) - 1, len(self.v_edges_cm) - 1

    def local_coordinates(self, positions_global_cm: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_global_cm, dtype=float)
        delta = positions - np.asarray(self.origin_cm)
        return np.column_stack(
            (
                delta @ np.asarray(self.u_global),
                delta @ np.asarray(self.v_global),
                delta @ np.asarray(self.normal_global),
            )
        )

    def build_openmc_partition(
        self,
        fill: Any,
        world_region: Any,
        *,
        slab_half_thickness_cm: float = 1.0e-4,
        cell_id_base: int = 8_800_000,
    ) -> tuple[list[openmc.Cell], openmc.Plane]:
        """Build a finite, transport-neutral partition around this plane."""

        half = float(slab_half_thickness_cm)
        if not np.isfinite(half) or half <= 0.0:
            raise ValueError("slab_half_thickness_cm must be positive")
        origin = np.asarray(self.origin_cm)
        u_axis = np.asarray(self.u_global)
        v_axis = np.asarray(self.v_global)
        normal = np.asarray(self.normal_global)

        def plane(
            axis: np.ndarray, offset: float, surface_id: int
        ) -> openmc.Plane:
            return openmc.Plane(
                a=float(axis[0]),
                b=float(axis[1]),
                c=float(axis[2]),
                d=float(np.dot(axis, origin) + offset),
                surface_id=surface_id,
            )

        center = plane(normal, 0.0, self.surface_id)
        u_low = plane(u_axis, -self.width_cm / 2, self.surface_id + 1)
        u_high = plane(u_axis, self.width_cm / 2, self.surface_id + 2)
        v_low = plane(v_axis, -self.height_cm / 2, self.surface_id + 3)
        v_high = plane(v_axis, self.height_cm / 2, self.surface_id + 4)
        n_low = plane(normal, -half, self.surface_id + 5)
        n_high = plane(normal, half, self.surface_id + 6)
        patch = +u_low & -u_high & +v_low & -v_high & +n_low & -n_high
        regions = (
            world_region & patch & -center,
            world_region & patch & +center,
            world_region & ~patch,
        )
        cells = [
            openmc.Cell(cell_id=cell_id_base + index, region=region, fill=fill)
            for index, region in enumerate(regions)
        ]
        return cells, center

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane_id": self.plane_id,
            "magnet_region_id": self.magnet_region_id,
            "name": self.name,
            "role": self.role,
            "origin_cm": list(self.origin_cm),
            "normal_global": list(self.normal_global),
            "u_global": list(self.u_global),
            "v_global": list(self.v_global),
            "width_cm": self.width_cm,
            "height_cm": self.height_cm,
            "u_edges_cm": list(self.u_edges_cm),
            "v_edges_cm": list(self.v_edges_cm),
            "spatial_shape": list(self.spatial_shape),
            "area_cm2": self.area_cm2,
            "magnet_component": self.magnet_component,
            "surface_id": self.surface_id,
            "normal_orientation": self.normal_orientation,
            "entry_sense": self.entry_sense,
            "exit_sense": self.exit_sense,
            "coordinate_units": self.coordinate_units,
            "area_units": self.area_units,
            "geometry_source": _jsonable(self.geometry_source),
            "construction_method": self.construction_method,
            "mu_tolerance": self.mu_tolerance,
            "adjacency_tolerance_cm": self.adjacency_tolerance_cm,
            "association": _jsonable(self.association),
            "proxy": self.proxy,
        }


@dataclass(frozen=True)
class MagnetRegion:
    """Coarse reactor-scale region representing one magnet interface."""

    name: str
    cell_ids: tuple[int, ...]
    region_id: str | int | None = None
    source_region_id: str | int | None = None
    surface_ids: tuple[int, ...] = ()
    phase_space_cell_id: int | None = None
    magnet_id: str | int | None = None
    coil_id: str | int | None = None
    winding_pack_id: str | int | None = None
    magnet_component: str | None = None
    coupling_planes: tuple[MagnetCouplingPlane, ...] = ()
    geometry_source_mode: str = "legacy_surface"
    entry_surface_id: int | None = None
    exit_surface_id: int | None = None
    units: str = "cm"
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
        object.__setattr__(
            self,
            "region_id",
            _coerce_region_id(self.region_id, name="region_id"),
        )
        object.__setattr__(self, "units", _optional_units(self.units))
        cell_ids = _positive_ids(self.cell_ids, "cell_ids")
        if not cell_ids:
            raise ValueError("cell_ids must contain at least one OpenMC cell")
        surface_ids = _positive_ids(self.surface_ids, "surface_ids")
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "surface_ids", surface_ids)
        planes = tuple(self.coupling_planes)
        plane_ids = [plane.plane_id for plane in planes]
        if len(plane_ids) != len(set(plane_ids)):
            raise ValueError(
                "coupling plane IDs must be unique within a region"
            )
        for plane in planes:
            if (
                self.region_id is not None
                and plane.magnet_region_id != self.region_id
            ):
                raise ValueError(
                    "coupling plane magnet_region_id does not match region_id"
                )
            if plane.surface_id not in surface_ids:
                raise ValueError(
                    "coupling-plane surface_id must appear in surface_ids"
                )
        mode = str(self.geometry_source_mode)
        if mode not in {"production", "first_wall_proxy", "legacy_surface"}:
            raise ValueError("geometry_source_mode is not recognized")
        if mode == "first_wall_proxy" and any(
            not plane.proxy for plane in planes
        ):
            raise ValueError(
                "first_wall_proxy regions may contain only proxy planes"
            )
        object.__setattr__(self, "coupling_planes", planes)
        object.__setattr__(self, "geometry_source_mode", mode)
        object.__setattr__(
            self,
            "entry_surface_id",
            _coerce_id(self.entry_surface_id, name="entry_surface_id"),
        )
        object.__setattr__(
            self,
            "exit_surface_id",
            _coerce_id(self.exit_surface_id, name="exit_surface_id"),
        )
        if (
            self.entry_surface_id is not None
            and self.entry_surface_id not in surface_ids
        ):
            raise ValueError("entry_surface_id must appear in surface_ids")
        if (
            self.exit_surface_id is not None
            and self.exit_surface_id not in surface_ids
        ):
            raise ValueError("exit_surface_id must appear in surface_ids")

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
        region_id = data.get("region_id", data.get("id"))
        component = data.get("magnet_component")
        planes = tuple(
            MagnetCouplingPlane.from_mapping(
                item,
                magnet_region_id=region_id,
                magnet_component=component,
            )
            for item in data.get("coupling_planes", ())
        )
        surface_ids = list(data.get("surface_ids", ()))
        surface_ids.extend(
            plane.surface_id
            for plane in planes
            if plane.surface_id not in surface_ids
        )
        return cls(
            name=str(data["name"]),
            region_id=region_id,
            cell_ids=tuple(data["cell_ids"]),
            source_region_id=data.get("source_region_id"),
            surface_ids=tuple(surface_ids),
            phase_space_cell_id=data.get("phase_space_cell_id"),
            magnet_id=data.get("magnet_id"),
            coil_id=data.get("coil_id"),
            winding_pack_id=data.get("winding_pack_id"),
            magnet_component=component,
            coupling_planes=planes,
            geometry_source_mode=data.get(
                "geometry_source_mode", "legacy_surface"
            ),
            entry_surface_id=data.get("entry_surface_id"),
            exit_surface_id=data.get("exit_surface_id"),
            units=data.get("units", "cm"),
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
            "region_id": self.region_id,
            "source_region_id": self.source_region_id,
            "magnet_id": self.magnet_id,
            "coil_id": self.coil_id,
            "winding_pack_id": self.winding_pack_id,
            "magnet_component": self.magnet_component,
            "geometry_source_mode": self.geometry_source_mode,
            "coupling_planes": [
                plane.to_dict() for plane in self.coupling_planes
            ],
            "entry_surface_id": self.entry_surface_id,
            "exit_surface_id": self.exit_surface_id,
            "units": self.units,
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
    mu_bins: int | None = None
    time_bounds_s: tuple[float, ...] | None = None
    polar_bounds_rad: tuple[float, ...] | None = None
    azimuthal_bounds_rad: tuple[float, ...] | None = None
    source_rate_per_s: float | None = None
    source_definition: Mapping[str, Any] = field(default_factory=dict)
    source_definition_hash: str | None = None
    dagmc_geometry_hash: str | None = None
    dagmc_geometry_path: str | None = None
    parastell_git_sha: str | None = None
    openmc_git_sha: str | None = None
    run_statistics: Mapping[str, Any] = field(default_factory=dict)
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
        plane_ids = [
            plane.plane_id
            for region in self.regions
            for plane in region.coupling_planes
        ]
        if len(plane_ids) != len(set(plane_ids)):
            raise ValueError("coupling plane IDs must be globally unique")

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
        mu_bin_count = len(mu) - 1
        if self.mu_bins is not None and int(self.mu_bins) != mu_bin_count:
            raise ValueError(
                "mu_bins must match the number of intervals in mu_bounds"
            )
        object.__setattr__(self, "mu_bins", mu_bin_count)

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
        else:
            object.__setattr__(
                self, "polar_bounds_rad", _default_polar_bounds_rad()
            )
        if self.azimuthal_bounds_rad is not None:
            azimuthal = _validate_edges(
                self.azimuthal_bounds_rad,
                "azimuthal_bounds_rad",
                minimum=-float(np.pi),
                maximum=float(np.pi),
            )
            object.__setattr__(self, "azimuthal_bounds_rad", azimuthal)
        else:
            object.__setattr__(
                self,
                "azimuthal_bounds_rad",
                _default_azimuthal_bounds_rad(),
            )

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
        if self.source_definition_hash is not None:
            text = str(self.source_definition_hash).strip()
            if not text:
                raise ValueError("source_definition_hash cannot be empty")
            object.__setattr__(self, "source_definition_hash", text)
        if self.dagmc_geometry_hash is not None:
            geometry_hash = str(self.dagmc_geometry_hash).strip()
            if not geometry_hash:
                raise ValueError("dagmc_geometry_hash cannot be empty")
            object.__setattr__(self, "dagmc_geometry_hash", geometry_hash)
        if self.dagmc_geometry_path is not None:
            geometry_path = str(self.dagmc_geometry_path).strip()
            if not geometry_path:
                raise ValueError("dagmc_geometry_path cannot be empty")
            object.__setattr__(self, "dagmc_geometry_path", geometry_path)
        object.__setattr__(self, "run_statistics", dict(self.run_statistics))
        object.__setattr__(
            self, "source_definition", dict(self.source_definition)
        )
        if not self.source_definition:
            object.__setattr__(self, "source_definition", {})

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        base_directory: Path | None = None,
    ) -> MagnetSpectralHandoff:
        tally_data = data.get("tallies", {})
        normalization = data.get("normalization", {})
        energy_bounds, energy_structure = _resolve_energy_bounds(
            data, base_directory=base_directory
        )
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
            mu_bins=data.get("mu_bins"),
            mu_bounds=tuple(data.get("mu_bounds", (-1.0, 0.0, 1.0))),
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
            source_definition=_jsonable(data.get("source_definition", {})),
            source_definition_hash=data.get("source_definition_hash"),
            dagmc_geometry_hash=data.get("dagmc_geometry_hash"),
            dagmc_geometry_path=data.get("dagmc_geometry_path"),
            parastell_git_sha=data.get("parastell_git_sha"),
            openmc_git_sha=data.get("openmc_git_sha"),
            run_statistics=dict(data.get("run_statistics", {})),
            time_bounds_s=(
                tuple(data["time_bounds_s"])
                if data.get("time_bounds_s") is not None
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
        return cls.from_mapping(data, base_directory=config_path.parent)

    @property
    def coupling_planes(self) -> tuple[MagnetCouplingPlane, ...]:
        return tuple(
            plane
            for region in self.regions
            for plane in region.coupling_planes
        )

    @property
    def interface_status(self) -> str:
        if not self.coupling_planes:
            return "legacy_surface"
        if any(plane.proxy for plane in self.coupling_planes):
            return "proxy"
        return "production"

    def plane(self, plane_id: str) -> MagnetCouplingPlane:
        for plane in self.coupling_planes:
            if plane.plane_id == plane_id:
                return plane
        raise KeyError(f"unknown coupling plane {plane_id!r}")

    def validate_plane_contract(
        self, *, require_production: bool = False
    ) -> dict[str, Any]:
        errors: list[str] = []
        if not self.coupling_planes:
            errors.append("no finite coupling planes are configured")
        if require_production:
            if self.interface_status != "production":
                errors.append(
                    "production validation rejects proxy/legacy interfaces"
                )
            for name, value in (
                ("dagmc_geometry_path", self.dagmc_geometry_path),
                ("dagmc_geometry_hash", self.dagmc_geometry_hash),
                (
                    "source_definition_hash",
                    self.normalization_contract()["source_definition_hash"],
                ),
            ):
                if value is None or not str(value).strip():
                    errors.append(f"production metadata {name} is required")
            for region in self.regions:
                if region.region_id is None:
                    errors.append(f"region {region.name!r} has no region_id")
                if not region.magnet_component:
                    errors.append(
                        f"region {region.name!r} has no magnet_component"
                    )
            for plane in self.coupling_planes:
                if plane.proxy:
                    errors.append(
                        f"plane {plane.plane_id!r} is a diagnostic proxy"
                    )
                distance = plane.association.get(
                    "minimum_plane_to_magnet_distance_cm"
                )
                if distance is None or not np.isfinite(float(distance)):
                    errors.append(
                        f"plane {plane.plane_id!r} has no geometry distance evidence"
                    )
                elif float(distance) > plane.adjacency_tolerance_cm:
                    errors.append(
                        f"plane {plane.plane_id!r} is not adjacent to its magnet"
                    )
        return {
            "status": "pass" if not errors else "fail",
            "interface_status": self.interface_status,
            "plane_count": len(self.coupling_planes),
            "planes": [plane.to_dict() for plane in self.coupling_planes],
            "errors": errors,
        }

    def normalization_contract(self) -> dict[str, Any]:
        mode = (
            "per_second"
            if self.source_rate_per_s is not None
            else "per_source"
        )
        quantity_scale = (
            "current or flux / source_particle"
            if mode == "per_source"
            else "current or flux / s"
        )
        return {
            "mode": mode,
            "source_rate_per_s": self.source_rate_per_s,
            "particles_per_source_history": None,
            "time_basis": (
                "per_source"
                if self.source_rate_per_s is None
                else "per_second"
            ),
            "quantity_scale": quantity_scale,
            "permitted_mix": (
                "Never combine per-source and per-second fields in one "
                "quantity."
            ),
            "surface_tally_reference": (
                self._tally_name(self.regions[0], "boundary_current")
                if self.include_boundary_current
                else None
            ),
            "normalization_rule": (
                "Companion boundary-current tally and binwise bank scaling "
                "is required for absolute replay"
            ),
            "source_definition_hash": (
                self.source_definition_hash
                if self.source_definition_hash is not None
                else _stable_hash(self.source_definition)
            ),
            "dagmc_geometry_hash": self.dagmc_geometry_hash,
        }

    def angular_contract(self) -> dict[str, Any]:
        mu_delta = np.diff(self.mu_bounds)
        polar_delta = np.diff(self.polar_bounds_rad)
        azimuthal_delta = np.diff(self.azimuthal_bounds_rad)
        angular = {
            "mu_bins": int(len(self.mu_bounds) - 1),
            "mu_bin_count": int(len(self.mu_bounds) - 1),
            "mu_bounds": list(self.mu_bounds),
            "mu_bin_width": _safe_float_list(mu_delta),
            "mu_solid_angle_sr": _safe_float_list(
                _solid_angle_from_mu_bounds(self.mu_bounds)
            ),
            "normal_convention": "global_surface_normal_with_sign",
            "label": "mu = Ω · n_outward",
            "polar_bin_count": int(len(self.polar_bounds_rad) - 1),
            "polar_bounds_rad": list(self.polar_bounds_rad),
            "polar_width": _safe_float_list(polar_delta),
            "azimuthal_bin_count": int(len(self.azimuthal_bounds_rad) - 1),
            "azimuthal_bounds_rad": list(self.azimuthal_bounds_rad),
            "azimuthal_width": _safe_float_list(azimuthal_delta),
            "angular_bin_count": (
                int(len(self.mu_bounds) - 1)
                * int(len(self.azimuthal_bounds_rad) - 1)
            ),
        }
        angular["angular_bin_solid_angle_sr"] = _safe_float_list(
            _solid_angle_from_bins(self.mu_bounds, self.azimuthal_bounds_rad)
        )
        return angular

    def spatial_contract(self) -> dict[str, Any]:
        if len(self.regions) != 1:
            return {
                "regions": [
                    {
                        "region": region.name,
                        "region_id": region.region_id,
                        "mesh": region.mesh.to_dict() if region.mesh else None,
                        "coupling_planes": [
                            plane.to_dict() for plane in region.coupling_planes
                        ],
                    }
                    for region in self.regions
                ]
            }
        region = self.regions[0]
        return {
            "region": region.name,
            "region_id": region.region_id,
            "mesh": region.mesh.to_dict() if region.mesh else None,
            "coupling_planes": [
                plane.to_dict() for plane in region.coupling_planes
            ],
            "units": region.units,
            "entry_surface_id": region.entry_surface_id,
            "exit_surface_id": region.exit_surface_id,
            "tape_width_axis": (
                region.coordinate_frame.labels[0]
                if region.coordinate_frame.labels
                else "x_axis"
            ),
            "tape_length_axis": (
                region.coordinate_frame.labels[1]
                if region.coordinate_frame.labels
                else "y_axis"
            ),
            "thickness_axis": (
                region.coordinate_frame.labels[2]
                if region.coordinate_frame.labels
                else "z_axis"
            ),
            "units": region.units,
        }

    def region(self, name: str) -> MagnetRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"unknown magnet region {name!r}")

    def add_openmc_tallies(
        self,
        model: openmc.Model,
        *,
        enable_photon_transport: bool = True,
    ) -> openmc.Tallies:
        return self.attach_to_model(
            model, enable_photon_transport=enable_photon_transport
        )

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
            "schema": SCHEMA_URI,
            "schema_version": SCHEMA_VERSION,
            "created_utc": _time_now_iso8601(),
            "openmc_version": getattr(openmc, "__version__", "unknown"),
            "openmc_git_sha": self.openmc_git_sha,
            "parastell_version": "0.1.0",
            "parastell_git_sha": self.parastell_git_sha or _git_sha(),
            "interface_status": self.interface_status,
            "energy_bounds_eV": list(self.energy_bounds_eV),
            "energy_group_structure": self.energy_group_structure,
            "particles": list(self.particles),
            "tally_id_base": self.tally_id_base,
            "minimum_openmc_version": "0.15.1",
            "angular": self.angular_contract(),
            "spatial": self.spatial_contract(),
            "contract": self.normalization_contract(),
            "time_bounds_s": (
                list(self.time_bounds_s)
                if self.time_bounds_s is not None
                else None
            ),
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
                "source_rate_reference": "particles/s",
                "particles_per_source_history": None,
                "rule": (
                    "Use boundary-current tally bins to scale source-bank "
                    "weights before deterministic replay."
                ),
                "surface_tally_basis": self._tally_name(
                    self.regions[0], "boundary_current"
                ),
            },
            "regions": [region.to_dict() for region in self.regions],
            "tallies": self.tally_catalog(),
            "statepoint_path": (
                str(statepoint_path) if statepoint_path is not None else None
            ),
            "surface_source": _jsonable(surface_source or {}),
            "source_definition": _jsonable(self.source_definition),
            "source_definition_hash": self.normalization_contract()[
                "source_definition_hash"
            ],
            "dagmc_geometry_hash": self.dagmc_geometry_hash,
            "dagmc_geometry_path": self.dagmc_geometry_path,
            "coupling_planes": [
                plane.to_dict() for plane in self.coupling_planes
            ],
            "run_statistics": _jsonable(self.run_statistics),
            "native_surface_source_fields": [
                "position_global_cm",
                "direction_global",
                "position_local_cm",
                "direction_local",
                "energy_eV",
                "time_s",
                "weight",
                "delayed_group",
                "surface_id_abs",
                "particle_name",
                "particle_code_raw",
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
                "magnet_id",
                "coil_id",
                "winding_pack_id",
                "magnet_direction",
                "crossing_sense",
                "surface_outward_normal_global",
                "surface_outward_normal_local",
                "mu_outward",
                "position_bin_x",
                "position_bin_y",
                "position_bin_z",
                "mu_bin",
                "polar_bin",
                "azimuthal_bin",
                "direction_label_basis",
                "particle_pdg",
                "surface_area_cm2",
                "coupling_plane_id",
                "plane_u_cm",
                "plane_v_cm",
                "plane_distance_cm",
                "position_bin_u",
                "position_bin_v",
            ],
            "normalization_warning": (
                "Do not infer absolute interface rate from number of banked "
                "phase-space records. Normalize with companion boundary-current "
                "tally using direction-conditioned scaling."
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

    def validate_exported_contract(
        self,
        spectra_path: str | Path,
        phase_space_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Validate schema-consistent payloads written by this workflow."""
        output: dict[str, Any] = {
            "spectra_path": str(Path(spectra_path)),
            "phase_space_path": (
                None
                if phase_space_path is None
                else str(Path(phase_space_path))
            ),
            "status": "pass",
            "errors": [],
        }
        spectra_path = Path(spectra_path)
        with h5py.File(spectra_path, "r") as spectra:
            schema = str(spectra.attrs.get("schema", ""))
            schema_version = str(spectra.attrs.get("schema_version", ""))
            if schema != SCHEMA_URI and schema != SCHEMA_NAME:
                output["status"] = "fail"
                output["errors"].append(
                    f"spectra schema mismatch: {schema}/{schema_version}"
                )
            if "tallies" not in spectra:
                output["status"] = "fail"
                output["errors"].append("spectra has no tallies group")
            if "tallies" in spectra:
                boundary_groups = [
                    str(name)
                    for name, group in spectra["tallies"].items()
                    if str(group.attrs.get("role", "")) == "boundary_current"
                ]
                if self.include_boundary_current and not boundary_groups:
                    output["status"] = "fail"
                    output["errors"].append(
                        "boundary_current tally missing from spectra"
                    )
        if phase_space_path is None:
            return output

        phase_space_path = Path(phase_space_path)
        with h5py.File(phase_space_path, "r") as source:
            schema = str(source.attrs.get("schema", ""))
            schema_version = str(source.attrs.get("schema_version", ""))
            if schema != SCHEMA_URI and schema != SCHEMA_NAME:
                output["status"] = "fail"
                output["errors"].append(
                    f"phase-space schema mismatch: {schema}/{schema_version}"
                )
            if "phase_space" not in source:
                output["status"] = "fail"
                output["errors"].append(
                    "phase-space file missing phase_space group"
                )
            else:
                phase = source["phase_space"]
                for field in ("position_local_cm", "direction_local"):
                    if field not in phase:
                        output["status"] = "fail"
                        output["errors"].append(
                            f"phase-space missing required field {field}"
                        )
                for field in (
                    "position_global_cm",
                    "direction_global",
                    "energy_eV",
                    "time_s",
                    "weight",
                    "particle_name",
                    "particle_pdg",
                    "surface_id_abs",
                    "mu_outward",
                    "magnet_direction",
                    "crossing_sense",
                    "is_entering",
                    "is_exiting",
                    "polar_rad",
                    "azimuthal_rad",
                    "mu_bin",
                    "polar_bin",
                    "azimuthal_bin",
                    "position_bin_x",
                    "position_bin_y",
                    "position_bin_z",
                    "position_bin_index",
                    "surface_area_cm2",
                ):
                    if field not in phase:
                        output["status"] = "fail"
                        output["errors"].append(
                            f"phase-space missing required field {field}"
                        )
            try:
                metadata = json.loads(source["metadata_json"].asstr()[()])
            except Exception:
                metadata = None
            output["metadata"] = metadata
            output["record_count"] = (
                int(len(source["phase_space"]["position_local_cm"]))
                if "phase_space" in source
                and "position_local_cm" in source["phase_space"]
                else 0
            )
            output["region_name"] = str(source.attrs.get("region", ""))
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
                output.attrs["schema"] = SCHEMA_URI
                output.attrs["schema_version"] = SCHEMA_VERSION
                output.attrs["source_statepoint"] = str(statepoint_path)
                manifest = self.to_manifest(statepoint_path=statepoint_path)
                statepoint_statistics = {
                    name: _jsonable(value)
                    for name in (
                        "n_realizations",
                        "n_particles",
                        "n_batches",
                        "current_batch",
                    )
                    if (value := getattr(statepoint, name, None)) is not None
                }
                manifest["run_statistics"] = {
                    **manifest.get("run_statistics", {}),
                    **statepoint_statistics,
                }
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

        plane_columns: dict[str, np.ndarray] | None = None
        if region.coupling_planes:
            plane_columns = _coupling_plane_columns(region, bank)
            finite_patch = plane_columns.pop("finite_patch_mask")
            if not np.any(finite_patch):
                raise RuntimeError(
                    "surface source has no crossings inside finite patches"
                )
            bank = {
                name: np.asarray(values)[finite_patch]
                for name, values in bank.items()
            }
            plane_columns = {
                name: np.asarray(values)[finite_patch]
                for name, values in plane_columns.items()
            }
            if record_outward_normals_global is not None:
                record_outward_normals_global = np.asarray(
                    record_outward_normals_global
                )[finite_patch]

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
        plane_normals = (
            plane_columns["plane_normal_global"]
            if plane_columns is not None
            else record_outward_normals_global
        )
        outward_global, outward_local, mu_outward = _surface_normal_columns(
            bank,
            region,
            record_outward_normals_global=plane_normals,
        )
        polar_rad, azimuth_rad = _local_spherical_bins(local_direction)
        mu_bin = _vector_to_bin_index(mu_outward, self.mu_bounds)
        polar_bin = _vector_to_bin_index(polar_rad, self.polar_bounds_rad)
        azimuthal_bin = _vector_to_bin_index(
            azimuth_rad, self.azimuthal_bounds_rad
        )
        phase_space_table = dict(bank)
        phase_space_table["position_local_cm"] = local_position
        phase_space_table["direction_local"] = local_direction
        phase_space_table["polar_rad"] = polar_rad
        phase_space_table["azimuthal_rad"] = azimuth_rad
        phase_space_table["surface_outward_normal_global"] = outward_global
        phase_space_table["surface_outward_normal_local"] = outward_local
        phase_space_table["mu_outward"] = mu_outward
        phase_space_table["mu_bin"] = mu_bin
        phase_space_table["polar_bin"] = polar_bin
        phase_space_table["azimuthal_bin"] = azimuthal_bin
        phase_space_table["position_global_cm"] = bank["position_global_cm"]
        phase_space_table["direction_global"] = bank["direction_global"]
        phase_space_table["energy_eV"] = bank["energy_eV"]
        phase_space_table["weight"] = bank["weight"]
        phase_space_table["surface_id_abs"] = bank["surface_id_abs"]
        phase_space_table["particle_name"] = bank["particle_name"]
        phase_space_table["particle_pdg"] = bank["particle_pdg"]
        phase_space_table["particle_code_raw"] = bank["particle_code_raw"]
        phase_space_table["surface_area_cm2"] = np.asarray(
            [
                region.surface_areas_cm2.get(int(surface), np.nan)
                for surface in bank["surface_id_abs"]
            ],
            dtype=float,
        )
        if plane_columns is not None:
            phase_space_table.update(plane_columns)
            phase_space_table["surface_area_cm2"] = plane_columns[
                "plane_area_cm2"
            ]
        mesh_bins = _mesh_bin_indices(region, local_position)
        if mesh_bins is None:
            mesh = {
                "position_bin_x": np.full(record_count, -1, dtype=int),
                "position_bin_y": np.full(record_count, -1, dtype=int),
                "position_bin_z": np.full(record_count, -1, dtype=int),
                "position_bin_index": np.full(record_count, -1, dtype=int),
            }
        else:
            mesh = mesh_bins
        phase_space_table.update(mesh)
        if plane_columns is not None:
            phase_space_table["position_bin_x"] = plane_columns[
                "position_bin_u"
            ]
            phase_space_table["position_bin_y"] = plane_columns[
                "position_bin_v"
            ]
            phase_space_table["position_bin_z"] = np.zeros(
                record_count, dtype=int
            )
            phase_space_table["position_bin_index"] = plane_columns[
                "position_bin_index"
            ]
        geometric_basis = (
            record_normal_basis
            if record_outward_normals_global is not None
            else "configured_outward_normal"
        )
        magnet_direction, direction_basis = _phase_space_direction_columns(
            mu_outward,
            selection,
            geometric_basis=geometric_basis,
            mu_tolerance=(
                max(plane.mu_tolerance for plane in region.coupling_planes)
                if region.coupling_planes
                else 1.0e-12
            ),
        )
        phase_space_table["magnet_direction"] = magnet_direction
        phase_space_table["direction_label_basis"] = direction_basis
        crossing_sense = np.full(record_count, "unknown", dtype=object)
        crossing_sense[magnet_direction == "incoming"] = "entering"
        crossing_sense[magnet_direction == "outgoing"] = "exiting"
        crossing_sense[magnet_direction == "grazing"] = "grazing"

        phase_space_table["crossing_sense"] = crossing_sense
        phase_space_table["is_entering"] = crossing_sense == "entering"
        phase_space_table["is_exiting"] = crossing_sense == "exiting"
        direction_validation = _phase_space_direction_validation(
            magnet_direction, direction_basis, selection
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        temporary_path.unlink(missing_ok=True)
        try:
            with h5py.File(temporary_path, "w") as output:
                output.attrs["schema"] = SCHEMA_URI
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
                output.attrs["normalization_contract_json"] = json.dumps(
                    self.normalization_contract(), sort_keys=True
                )
                output.attrs["angular_contract_json"] = json.dumps(
                    self.angular_contract(), sort_keys=True
                )
                output.attrs["spatial_contract_json"] = json.dumps(
                    self.spatial_contract(), sort_keys=True
                )
                output.attrs[
                    "source_definition_hash"
                ] = self.normalization_contract().get(
                    "source_definition_hash", ""
                )
                output.attrs["source_rate_per_s"] = (
                    self.source_rate_per_s
                    if self.source_rate_per_s is not None
                    else np.nan
                )
                output.attrs["selected_crossing_sense"] = ",".join(
                    sorted(set(crossing_sense.tolist()))
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
                    "surface_area_lookup_cm2": (
                        region.surface_areas_cm2
                        if region.surface_areas_cm2
                        else None
                    ),
                    "spatial_bin_mapping": self.spatial_contract(),
                    "angular_bin_mapping": self.angular_contract(),
                    "coupling_planes": [
                        plane.to_dict() for plane in region.coupling_planes
                    ],
                    "finite_patch_filtering": bool(region.coupling_planes),
                    "crossing_counts": {
                        label: int(np.count_nonzero(crossing_sense == label))
                        for label in (
                            "entering",
                            "exiting",
                            "grazing",
                            "unknown",
                        )
                    },
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
    empty = (mean == 0.0) & ~np.isfinite(std_dev)
    if np.any(empty):
        std_dev = std_dev.copy()
        std_dev[empty] = 0.0
        table[std_name] = std_dev
    relative_error = np.full(mean.shape, np.nan, dtype=float)
    nonzero = mean != 0.0
    relative_error[nonzero] = std_dev[nonzero] / np.abs(mean[nonzero])
    relative_error[~nonzero] = 0.0
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
    mean_name = _find_column(table, "mean")
    if low_name is None or high_name is None:
        return

    low = np.asarray(table[low_name], dtype=float)
    high = np.asarray(table[high_name], dtype=float)
    labels: list[str] = []
    for index, (lower, upper) in enumerate(zip(low, high)):
        if surface_name is not None and mean_name is not None:
            surface_id = int(table[surface_name][index])
            normal_sign = region.surface_normal_signs.get(surface_id)
            signed_current = float(table[mean_name][index])
            if (
                normal_sign is not None
                and np.isfinite(signed_current)
                and signed_current != 0.0
            ):
                outward_current = signed_current * normal_sign
                labels.append(
                    "incoming" if outward_current < 0.0 else "outgoing"
                )
                continue
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
    mu_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu_outward, dtype=float)
    labels = np.full(mu.shape, "unknown", dtype=object)
    basis = np.full(mu.shape, "unavailable", dtype=object)
    known = np.isfinite(mu)
    labels[known & (mu < -mu_tolerance)] = "incoming"
    labels[known & (mu > mu_tolerance)] = "outgoing"
    labels[known & (np.abs(mu) <= mu_tolerance)] = "grazing"
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


def _local_spherical_bins(
    directions_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return polar and azimuth arrays for local direction vectors."""
    vectors = np.asarray(directions_local, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("direction vectors must be shape (N, 3)")
    radius = np.linalg.norm(vectors, axis=1)
    safe = np.divide(
        vectors,
        radius[:, None],
        out=np.zeros_like(vectors),
        where=radius[:, None] != 0,
    )
    polar = np.arccos(np.clip(safe[:, 2], -1.0, 1.0))
    azimuth = np.arctan2(safe[:, 1], safe[:, 0])
    return polar, _normalise_angle_angle(azimuth)


def _coupling_plane_columns(
    region: MagnetRegion,
    bank: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    count = len(bank["surface_id_abs"])
    surface_ids = np.asarray(bank["surface_id_abs"], dtype=int)
    positions = np.asarray(bank["position_global_cm"], dtype=float)
    plane_id = np.full(count, "", dtype=object)
    role = np.full(count, "", dtype=object)
    local_u = np.full(count, np.nan)
    local_v = np.full(count, np.nan)
    distance = np.full(count, np.nan)
    u_bin = np.full(count, -1, dtype=int)
    v_bin = np.full(count, -1, dtype=int)
    flat_bin = np.full(count, -1, dtype=int)
    normals = np.full((count, 3), np.nan)
    areas = np.full(count, np.nan)
    finite = np.zeros(count, dtype=bool)
    for plane in region.coupling_planes:
        mask = surface_ids == plane.surface_id
        if not np.any(mask):
            continue
        coordinates = plane.local_coordinates(positions[mask])
        selected = np.flatnonzero(mask)
        plane_id[selected] = plane.plane_id
        role[selected] = plane.role
        local_u[selected] = coordinates[:, 0]
        local_v[selected] = coordinates[:, 1]
        distance[selected] = coordinates[:, 2]
        normals[selected] = plane.normal_global
        areas[selected] = plane.area_cm2
        u_values = (
            np.searchsorted(plane.u_edges_cm, coordinates[:, 0], side="right")
            - 1
        )
        v_values = (
            np.searchsorted(plane.v_edges_cm, coordinates[:, 1], side="right")
            - 1
        )
        u_values[np.isclose(coordinates[:, 0], plane.u_edges_cm[-1])] = (
            plane.spatial_shape[0] - 1
        )
        v_values[np.isclose(coordinates[:, 1], plane.v_edges_cm[-1])] = (
            plane.spatial_shape[1] - 1
        )
        inside = (
            (u_values >= 0)
            & (u_values < plane.spatial_shape[0])
            & (v_values >= 0)
            & (v_values < plane.spatial_shape[1])
            & (np.abs(coordinates[:, 2]) <= plane.adjacency_tolerance_cm)
        )
        u_bin[selected] = u_values
        v_bin[selected] = v_values
        flat_bin[selected] = u_values + plane.spatial_shape[0] * v_values
        finite[selected] = inside
    return {
        "coupling_plane_id": plane_id,
        "coupling_plane_role": role,
        "plane_u_cm": local_u,
        "plane_v_cm": local_v,
        "plane_distance_cm": distance,
        "position_bin_u": u_bin,
        "position_bin_v": v_bin,
        "position_bin_index": flat_bin,
        "plane_normal_global": normals,
        "plane_area_cm2": areas,
        "finite_patch_mask": finite,
    }


def _mesh_bin_indices(
    region: MagnetRegion,
    local_position_cm: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Return rectilinear mesh bin indices for the region mesh."""
    if region.mesh is None or region.mesh.kind != "regular":
        return None
    if not local_position_cm.size:
        return {
            "position_bin_x": np.zeros(0, dtype=int),
            "position_bin_y": np.zeros(0, dtype=int),
            "position_bin_z": np.zeros(0, dtype=int),
            "position_bin_index": np.zeros(0, dtype=int),
        }
    mesh = region.mesh
    lower = np.asarray(mesh.lower_left_cm, dtype=float)
    upper = np.asarray(mesh.upper_right_cm, dtype=float)
    dimension = np.asarray(_safe_shape_tuple(mesh.dimension), dtype=int)
    spacing = upper - lower
    if np.any(spacing <= 0.0):
        raise ValueError(
            "mesh spacing must be positive in all directions for phase-space "
            "spatial binning"
        )
    valid = np.all(np.isfinite(local_position_cm), axis=1)
    valid &= np.all(local_position_cm >= lower, axis=1)
    valid &= np.all(local_position_cm <= upper, axis=1)
    normalized = (local_position_cm - lower) / spacing
    indices = np.floor(normalized * dimension).astype(int)
    local_position_cm = np.asarray(local_position_cm)
    indices[~valid] = -1
    if local_position_cm.size:
        for axis in range(3):
            at_upper = valid & (
                local_position_cm[:, axis] >= upper[axis] - 1.0e-12
            )
            indices[:, axis] = np.where(
                at_upper & (indices[:, axis] >= dimension[axis]),
                dimension[axis] - 1,
                indices[:, axis],
            )
    flat = indices[:, 0] + dimension[0] * (
        indices[:, 1] + dimension[1] * indices[:, 2]
    )
    flat[~valid] = -1
    return {
        "position_bin_x": indices[:, 0].astype(int),
        "position_bin_y": indices[:, 1].astype(int),
        "position_bin_z": indices[:, 2].astype(int),
        "position_bin_index": flat.astype(int),
    }


def _write_column_group(
    group: h5py.Group, table: Mapping[str, np.ndarray]
) -> None:
    string_dtype = h5py.string_dtype("utf-8")
    for name, values in table.items():
        array = np.asarray(values)
        if array.dtype.kind in {"O", "U", "S"}:
            encoded = np.asarray(array, dtype=str).astype(object)
            options = (
                {"compression": "gzip", "shuffle": True}
                if encoded.size > 128
                else {}
            )
            group.create_dataset(
                name, data=encoded, dtype=string_dtype, **options
            )
        else:
            options = (
                {"compression": "gzip", "shuffle": True}
                if array.size > 128
                else {}
            )
            group.create_dataset(name, data=array, **options)


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
