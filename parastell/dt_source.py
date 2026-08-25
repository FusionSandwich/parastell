"""ParaStell spatial source coupled to OpenMC 0.16 D-T spectra."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def _hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DTSourceAudit:
    mesh_sha256: str
    tetrahedra: int
    source_rate_per_s: float
    source_volume_cm3: float
    minimum_density_cm3_s: float
    maximum_density_cm3_s: float
    minimum_temperature_eV: float
    maximum_temperature_eV: float
    minimum_mean_energy_eV: float
    maximum_mean_energy_eV: float
    minimum_width_eV: float
    maximum_width_eV: float
    sampling_strength_sum: float
    transport_normalization: str


@dataclass(frozen=True)
class TaggedSourceMeshState:
    """Restartable source state reconstructed from ParaStell MOAB tags."""

    strengths: np.ndarray
    volumes: np.ndarray
    ion_temperatures_eV: np.ndarray
    source_element_cfs: np.ndarray


def source_convergence_observables(source_mesh) -> dict[str, dict[str, float]]:
    """Return strength-weighted spatial and D-T energy source moments.

    The energy distributions are the same OpenMC 0.16 D-T distributions used
    by :func:`build_temperature_dependent_mesh_source`.  Repeated ion
    temperatures are evaluated once so the finest source mesh does not create
    hundreds of thousands of identical distribution objects.
    """
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC 0.16 is required for D-T source convergence moments"
        ) from exc
    if not hasattr(openmc.stats, "fusion_neutron_spectrum"):
        raise RuntimeError("OpenMC 0.16 fusion_neutron_spectrum is required")

    strengths = np.asarray(source_mesh.strengths, dtype=float)
    centroids = np.asarray(
        source_mesh.source_element_centroids_cm, dtype=float
    )
    temperatures = np.asarray(source_mesh.ion_temperatures_eV, dtype=float)
    count = len(strengths)
    if (
        count == 0
        or centroids.shape != (count, 3)
        or len(temperatures) != count
    ):
        raise ValueError("source convergence arrays do not align")
    if np.any(strengths < 0.0) or not np.all(np.isfinite(strengths)):
        raise ValueError("source strengths must be finite and nonnegative")
    total = float(np.sum(strengths))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("source strength sum must be positive")
    probabilities = strengths / total
    means = np.empty(count, dtype=float)
    second_moments = np.empty(count, dtype=float)
    cache: dict[float, tuple[float, float]] = {}
    for index, temperature in enumerate(temperatures):
        key = float(temperature)
        if key not in cache:
            spectrum = openmc.stats.fusion_neutron_spectrum(
                ion_temp=key, reactants="DT"
            )
            mean = float(spectrum.mean_value)
            standard_deviation = float(spectrum.std_dev)
            cache[key] = (
                mean,
                standard_deviation**2 + mean**2,
            )
        means[index], second_moments[index] = cache[key]

    first = np.sum(probabilities[:, None] * centroids, axis=0)
    second = np.sum(probabilities[:, None] * centroids**2, axis=0)
    return {
        "spatial_moments": {
            "mean_x_cm": float(first[0]),
            "mean_y_cm": float(first[1]),
            "mean_z_cm": float(first[2]),
            "mean_x2_cm2": float(second[0]),
            "mean_y2_cm2": float(second[1]),
            "mean_z2_cm2": float(second[2]),
        },
        "dt_energy_moments": {
            "mean_eV": float(np.dot(probabilities, means)),
            "mean_squared_eV2": float(np.dot(probabilities, second_moments)),
        },
    }


def read_tagged_source_mesh(mesh_path: str | Path) -> TaggedSourceMeshState:
    """Read source strength, volume, temperature, and CFS tags from H5M."""
    try:
        from pymoab import core, types
    except ImportError as exc:
        raise RuntimeError(
            "PyMOAB is required to read a ParaStell source mesh"
        ) from exc
    mb = core.Core()
    mb.load_file(str(Path(mesh_path).resolve()))
    tetrahedra = list(mb.get_entities_by_type(0, types.MBTET))
    if not tetrahedra:
        raise ValueError("ParaStell source mesh contains no tetrahedra")

    def values(name):
        try:
            tag = mb.tag_get_handle(name)
        except RuntimeError as exc:
            raise ValueError(
                f"source mesh is missing required MOAB tag {name!r}"
            ) from exc
        return np.asarray(
            mb.tag_get_data(tag, tetrahedra, flat=True), dtype=float
        )

    state = TaggedSourceMeshState(
        values("Source Strength"),
        values("Volume"),
        values("Ion Temperature eV"),
        values("Source Element CFS"),
    )
    count = len(tetrahedra)
    if any(
        len(getattr(state, name)) != count
        for name in state.__dataclass_fields__
    ):
        raise ValueError("source mesh tags do not align with tetrahedra")
    return state


def build_temperature_dependent_mesh_source(source_mesh, mesh_path):
    """Build an OpenMC MeshSource without replacing ParaStell spatial weights."""
    import openmc

    if not hasattr(openmc.stats, "fusion_neutron_spectrum"):
        raise RuntimeError("OpenMC 0.16 fusion_neutron_spectrum is required")
    strengths = np.asarray(source_mesh.strengths, dtype=float)
    volumes = np.abs(np.asarray(source_mesh.volumes, dtype=float))
    temperatures = np.asarray(source_mesh.ion_temperatures_eV, dtype=float)
    if not (len(strengths) == len(volumes) == len(temperatures)):
        raise ValueError(
            "source mesh strength/volume/temperature arrays differ"
        )
    if len(strengths) == 0 or np.any(strengths < 0.0):
        raise ValueError(
            "source mesh must contain nonnegative source elements"
        )
    total_strength = float(strengths.sum())
    if not np.isfinite(total_strength) or total_strength <= 0.0:
        raise ValueError(
            "source mesh total physical strength must be positive"
        )
    sampling_probabilities = strengths / total_strength
    sources = []
    means = []
    widths = []
    for probability, temperature in zip(sampling_probabilities, temperatures):
        spectrum = openmc.stats.fusion_neutron_spectrum(
            ion_temp=float(temperature), reactants="DT"
        )
        means.append(float(spectrum.mean_value))
        widths.append(float(spectrum.std_dev))
        sources.append(
            openmc.IndependentSource(
                energy=spectrum,
                angle=openmc.stats.Isotropic(),
                particle="neutron",
                # OpenMC source strengths participate in tally normalization.
                # Keep transport per source history by using probabilities
                # summing to one.  The physical rate remains in DTSourceAudit
                # and is applied only when physical units are requested.
                strength=float(probability),
            )
        )
    mesh = openmc.UnstructuredMesh(mesh_path, "moab")
    mesh_source = openmc.MeshSource(mesh, sources)
    density = np.divide(
        strengths,
        volumes,
        out=np.zeros_like(strengths),
        where=volumes > 0.0,
    )
    audit = DTSourceAudit(
        mesh_sha256=_hash(mesh_path),
        tetrahedra=len(strengths),
        source_rate_per_s=float(strengths.sum()),
        source_volume_cm3=float(volumes.sum()),
        minimum_density_cm3_s=float(density.min()),
        maximum_density_cm3_s=float(density.max()),
        minimum_temperature_eV=float(temperatures.min()),
        maximum_temperature_eV=float(temperatures.max()),
        minimum_mean_energy_eV=float(np.min(means)),
        maximum_mean_energy_eV=float(np.max(means)),
        minimum_width_eV=float(np.min(widths)),
        maximum_width_eV=float(np.max(widths)),
        sampling_strength_sum=float(sampling_probabilities.sum()),
        transport_normalization="per source history",
    )
    return mesh_source, audit
