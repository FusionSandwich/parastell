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
