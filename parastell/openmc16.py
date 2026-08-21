"""OpenMC 0.16 integration primitives for magnet-boundary production runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
from typing import Any, Mapping, Sequence

import numpy as np


OPENMC_MINIMUM = (0, 16, 0)
PDG_PARTICLES = {
    "neutron": 2112,
    "photon": 22,
    "electron": 11,
    "positron": -11,
}
REQUIRED_APIS = (
    "SurfaceFilter",
    "MuSurfaceFilter",
    "MeshSurfaceFilter",
    "ReactionFilter",
    "ParticleProductionFilter",
    "SolidRayTracePlot",
    "SlicePlot",
    "VoxelPlot",
)
REQUIRED_SETTINGS = (
    "surf_source_write",
    "surface_grazing_cutoff",
    "surface_grazing_ratio",
    "collision_track",
    "photon_transport",
)


def _openmc():
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC 0.16 is required for this operation"
        ) from exc
    return openmc


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split("+")[0].split(".")
    return tuple(int(item) for item in parts[:3])


def capability_report() -> dict[str, Any]:
    """Exercise required Python APIs and return machine-readable provenance."""
    openmc = _openmc()
    settings = openmc.Settings()
    classes = {name: hasattr(openmc, name) for name in REQUIRED_APIS}
    setting_apis = {
        name: hasattr(settings, name) for name in REQUIRED_SETTINGS
    }
    report = {
        "version": openmc.__version__,
        "version_supported": _version_tuple(openmc.__version__)
        >= OPENMC_MINIMUM,
        "classes": classes,
        "settings": setting_apis,
        "fusion_neutron_spectrum": hasattr(
            openmc.stats, "fusion_neutron_spectrum"
        ),
        "fusion_neutron_spectrum_signature": (
            str(inspect.signature(openmc.stats.fusion_neutron_spectrum))
            if hasattr(openmc.stats, "fusion_neutron_spectrum")
            else None
        ),
        "particle_pdg": {
            name: int(getattr(openmc.ParticleType, name.upper()))
            for name in PDG_PARTICLES
        },
    }
    report["passes"] = bool(
        report["version_supported"]
        and all(classes.values())
        and all(setting_apis.values())
        and report["fusion_neutron_spectrum"]
        and report["particle_pdg"] == PDG_PARTICLES
    )
    return report


def require_capabilities() -> dict[str, Any]:
    report = capability_report()
    if not report["passes"]:
        raise RuntimeError(f"OpenMC 0.16 capability gate failed: {report}")
    return report


def configure_transport(
    settings,
    *,
    grazing_cutoff: float = 1.0e-8,
    grazing_ratio: float = 16.0,
    collision_track: Mapping[str, Any] | None = None,
) -> None:
    """Enable coupled transport and explicit 0.16 grazing semantics."""
    require_capabilities()
    if grazing_cutoff <= 0.0 or grazing_ratio <= 0.0:
        raise ValueError("grazing controls must be positive")
    settings.photon_transport = True
    settings.surface_grazing_cutoff = float(grazing_cutoff)
    settings.surface_grazing_ratio = float(grazing_ratio)
    if collision_track is not None:
        settings.collision_track = dict(collision_track)


@dataclass(frozen=True)
class TallyInventory:
    current: tuple[str, ...]
    directional_current: tuple[str, ...]
    surface_flux: tuple[str, ...]
    reactions: str | None
    production: tuple[str, ...]
    heating: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_envelope_tallies(
    model,
    *,
    surface_ids: Sequence[int],
    cell_ids: Sequence[int],
    neutron_edges_eV: Sequence[float],
    photon_edges_eV: Sequence[float],
    reaction_bins: Sequence[str | int] = (2, 4, 16, 102, 103, 107),
    produced_particles: Sequence[str | int] = (
        "photon",
        "neutron",
        "electron",
        "positron",
    ),
) -> TallyInventory:
    """Attach independent current, flux, reaction, production, and heating tallies."""
    require_capabilities()
    openmc = _openmc()
    surfaces = [int(value) for value in surface_ids]
    cells = [int(value) for value in cell_ids]
    if not surfaces or not cells:
        raise ValueError(
            "envelope surface IDs and magnet cell IDs are required"
        )
    tallies = model.tallies if model.tallies is not None else openmc.Tallies()
    inventory: dict[str, list[str] | str | None] = {
        "current": [],
        "directional_current": [],
        "surface_flux": [],
        "reactions": None,
        "production": [],
        "heating": [],
    }
    for particle, edges in (
        ("neutron", neutron_edges_eV),
        ("photon", photon_edges_eV),
    ):
        filters = [
            openmc.SurfaceFilter(surfaces),
            openmc.ParticleFilter([particle]),
            openmc.EnergyFilter(edges),
        ]
        current = openmc.Tally(name=f"pstl_envelope_{particle}_current")
        current.filters = filters
        current.scores = ["current"]
        tallies.append(current)
        inventory["current"].append(current.name)
        directional = openmc.Tally(
            name=f"pstl_envelope_{particle}_directional_current"
        )
        directional.filters = [
            openmc.SurfaceFilter(surfaces),
            openmc.MuSurfaceFilter([-1.0, 0.0, 1.0]),
            openmc.ParticleFilter([particle]),
            openmc.EnergyFilter(edges),
        ]
        directional.scores = ["current"]
        tallies.append(directional)
        inventory["directional_current"].append(directional.name)
        flux = openmc.Tally(name=f"pstl_envelope_{particle}_surface_flux")
        flux.filters = filters
        flux.scores = ["flux"]
        tallies.append(flux)
        inventory["surface_flux"].append(flux.name)
        heating = openmc.Tally(name=f"pstl_magnet_{particle}_heating")
        heating.filters = [
            openmc.CellFilter(cells),
            openmc.ParticleFilter([particle]),
            openmc.EnergyFilter(edges),
        ]
        heating.scores = ["heating"]
        tallies.append(heating)
        inventory["heating"].append(heating.name)

    reactions = openmc.Tally(name="pstl_magnet_neutron_reactions")
    reactions.filters = [
        openmc.CellFilter(cells),
        openmc.ParticleFilter(["neutron"]),
        openmc.EnergyFilter(neutron_edges_eV),
        openmc.ReactionFilter(reaction_bins),
    ]
    reactions.scores = ["events"]
    tallies.append(reactions)
    inventory["reactions"] = reactions.name

    for produced in produced_particles:
        name = str(produced).lower()
        outgoing_edges = (
            neutron_edges_eV if name == "neutron" else photon_edges_eV
        )
        production = openmc.Tally(name=f"pstl_magnet_production_{name}")
        production.filters = [
            openmc.CellFilter(cells),
            openmc.ParticleFilter(["neutron"]),
            openmc.EnergyFilter(neutron_edges_eV),
            openmc.ParticleProductionFilter(
                [produced], energies=outgoing_edges
            ),
        ]
        production.scores = ["events"]
        tallies.append(production)
        inventory["production"].append(production.name)
    model.tallies = tallies
    return TallyInventory(
        current=tuple(inventory["current"]),
        directional_current=tuple(inventory["directional_current"]),
        surface_flux=tuple(inventory["surface_flux"]),
        reactions=str(inventory["reactions"]),
        production=tuple(inventory["production"]),
        heating=tuple(inventory["heating"]),
    )


def rotated_mesh_filter(
    mesh, rotation: Sequence[Sequence[float]], translation
):
    openmc = _openmc()
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.allclose(
        matrix @ matrix.T, np.eye(3), atol=1.0e-10
    ):
        raise ValueError("mesh rotation must be an orthogonal 3x3 matrix")
    result = openmc.MeshFilter(mesh)
    result.rotation = matrix
    result.translation = np.asarray(translation, dtype=float)
    return result
