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
TALLY_PROFILES = {
    "core": frozenset({"core"}),
    "magnet_damage": frozenset({"core", "damage", "gas", "reactions"}),
    "activation_ready": frozenset(
        {"core", "damage", "gas", "reactions", "production"}
    ),
    "full_diagnostics": frozenset(
        {"core", "damage", "gas", "reactions", "production", "surface"}
    ),
    "magnet_damage_and_handoff": frozenset(
        {"core", "damage", "gas", "reactions", "production", "surface"}
    ),
}
GAS_PRODUCTION_SCORES = (
    "H1-production",
    "H2-production",
    "H3-production",
    "He3-production",
    "He4-production",
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
    volume_flux: tuple[str, ...]
    reactions: str | None
    nuclide_reactions: tuple[str, ...]
    production: tuple[str, ...]
    heating: tuple[str, ...]
    total_heating: str | None = None
    local_mesh_flux: tuple[str, ...] = ()
    local_mesh_heating: tuple[str, ...] = ()
    local_mesh_damage: tuple[str, ...] = ()
    local_mesh_gas: tuple[str, ...] = ()
    damage_energy: str | None = None
    gas_production: str | None = None
    profile: str = "full_diagnostics"
    response_availability: Mapping[str, Mapping[str, Any]] | None = None

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
    volume_flux_energy_axes: (
        Mapping[str, tuple[str, Sequence[float]]] | None
    ) = None,
    tally_profile: str = "full_diagnostics",
    local_mesh_filters_by_cell: Mapping[int, Any] | None = None,
    supported_responses: Sequence[str] | None = None,
    nuclide_mt_requests: Mapping[str, Sequence[str | int]] | None = None,
) -> TallyInventory:
    """Attach independent current, flux, reaction, production, and heating tallies."""
    require_capabilities()
    openmc = _openmc()
    if tally_profile not in TALLY_PROFILES:
        raise ValueError(
            f"unknown tally profile {tally_profile!r}; choose from {sorted(TALLY_PROFILES)}"
        )
    enabled = TALLY_PROFILES[tally_profile]
    supported = (
        set(supported_responses) if supported_responses is not None else None
    )

    def availability(name: str, requested: bool = True) -> dict[str, Any]:
        if not requested:
            return {"status": "NOT_REQUESTED", "available": False}
        if supported is not None and name not in supported:
            return {
                "status": "UNAVAILABLE_IN_CONFIGURED_NUCLEAR_DATA",
                "available": False,
            }
        return {
            "status": (
                "AVAILABLE"
                if supported is not None
                else "CONFIGURED_API_AVAILABLE_DATA_UNVERIFIED"
            ),
            "available": True,
        }

    response_report: dict[str, dict[str, Any]] = {}
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
        "volume_flux": [],
        "reactions": None,
        "nuclide_reactions": [],
        "production": [],
        "heating": [],
        "local_mesh_flux": [],
        "local_mesh_heating": [],
        "local_mesh_damage": [],
        "local_mesh_gas": [],
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

        for cell, mesh_filter in sorted(
            (local_mesh_filters_by_cell or {}).items()
        ):
            if int(cell) not in cells:
                raise ValueError(
                    f"local mesh was provided for unselected cell {int(cell)}"
                )
            mesh_flux = openmc.Tally(
                name=f"pstl_magnet_{int(cell)}_{particle}_local_mesh_flux"
            )
            mesh_flux.filters = [
                mesh_filter,
                openmc.ParticleFilter([particle]),
                openmc.EnergyFilter(edges),
            ]
            mesh_flux.scores = ["flux"]
            tallies.append(mesh_flux)
            inventory["local_mesh_flux"].append(mesh_flux.name)
            mesh_heat = openmc.Tally(
                name=f"pstl_magnet_{int(cell)}_{particle}_local_mesh_heating"
            )
            mesh_heat.filters = [
                mesh_filter,
                openmc.ParticleFilter([particle]),
                openmc.EnergyFilter(edges),
            ]
            mesh_heat.scores = ["heating"]
            tallies.append(mesh_heat)
            inventory["local_mesh_heating"].append(mesh_heat.name)

    flux_axes = volume_flux_energy_axes or {
        "neutron_configured": ("neutron", neutron_edges_eV),
        "photon_configured": ("photon", photon_edges_eV),
    }
    for label, (particle, edges) in flux_axes.items():
        safe_label = "".join(
            character if character.isalnum() else "_" for character in label
        ).strip("_")
        volume_flux = openmc.Tally(
            name=f"pstl_magnet_{safe_label}_volume_flux"
        )
        volume_flux.filters = [
            openmc.CellFilter(cells),
            openmc.ParticleFilter([particle]),
            openmc.EnergyFilter(edges),
        ]
        volume_flux.scores = ["flux"]
        tallies.append(volume_flux)
        inventory["volume_flux"].append(volume_flux.name)

    total_heating = openmc.Tally(name="pstl_magnet_total_heating")
    total_heating.filters = [
        openmc.CellFilter(cells),
        openmc.ParticleFilter(["neutron", "photon"]),
        openmc.EnergyFilter(
            np.unique(np.concatenate((neutron_edges_eV, photon_edges_eV)))
        ),
    ]
    total_heating.scores = ["heating"]
    tallies.append(total_heating)

    reactions_name = None
    if "reactions" in enabled:
        reactions = openmc.Tally(name="pstl_magnet_neutron_reactions")
        reactions.filters = [
            openmc.CellFilter(cells),
            openmc.ParticleFilter(["neutron"]),
            openmc.EnergyFilter(neutron_edges_eV),
            openmc.ReactionFilter(reaction_bins),
        ]
        reactions.scores = ["events"]
        tallies.append(reactions)
        reactions_name = reactions.name
    response_report["reaction_families"] = availability(
        "reaction_families", "reactions" in enabled
    )

    if nuclide_mt_requests and "reactions" not in enabled:
        raise ValueError(
            "nuclide/MT requests require a profile with reaction tallies"
        )
    for nuclide, requested_reactions in sorted(
        (nuclide_mt_requests or {}).items()
    ):
        nuclide_name = str(nuclide).strip()
        reactions_for_nuclide = tuple(requested_reactions)
        if not nuclide_name or not reactions_for_nuclide:
            raise ValueError("nuclide/MT requests cannot be empty")
        safe_nuclide = "".join(
            character if character.isalnum() else "_"
            for character in nuclide_name
        ).strip("_")
        response_name = f"nuclide_mt:{nuclide_name}"
        status = availability(response_name, True)
        response_report[response_name] = status
        if not status["available"]:
            continue
        reaction = openmc.Tally(
            name=f"pstl_magnet_{safe_nuclide}_mt_reactions"
        )
        reaction.filters = [
            openmc.CellFilter(cells),
            openmc.ParticleFilter(["neutron"]),
            openmc.EnergyFilter(neutron_edges_eV),
            openmc.ReactionFilter(reactions_for_nuclide),
        ]
        reaction.nuclides = [nuclide_name]
        reaction.scores = ["events"]
        tallies.append(reaction)
        inventory["nuclide_reactions"].append(reaction.name)

    damage_name = None
    damage_status = availability("damage-energy", "damage" in enabled)
    response_report["damage-energy"] = damage_status
    if damage_status["available"]:
        damage = openmc.Tally(name="pstl_magnet_neutron_damage_energy")
        damage.filters = [
            openmc.CellFilter(cells),
            openmc.ParticleFilter(["neutron"]),
            openmc.EnergyFilter(neutron_edges_eV),
        ]
        damage.scores = ["damage-energy"]
        tallies.append(damage)
        damage_name = damage.name
        for cell, mesh_filter in sorted(
            (local_mesh_filters_by_cell or {}).items()
        ):
            local_damage = openmc.Tally(
                name=f"pstl_magnet_{int(cell)}_neutron_local_mesh_damage_energy"
            )
            local_damage.filters = [
                mesh_filter,
                openmc.ParticleFilter(["neutron"]),
                openmc.EnergyFilter(neutron_edges_eV),
            ]
            local_damage.scores = ["damage-energy"]
            tallies.append(local_damage)
            inventory["local_mesh_damage"].append(local_damage.name)

    gas_name = None
    gas_scores = [
        score
        for score in GAS_PRODUCTION_SCORES
        if availability(score, "gas" in enabled)["available"]
    ]
    for score in GAS_PRODUCTION_SCORES:
        response_report[score] = availability(score, "gas" in enabled)
    if gas_scores:
        gas = openmc.Tally(name="pstl_magnet_gas_production")
        gas.filters = [
            openmc.CellFilter(cells),
            openmc.ParticleFilter(["neutron"]),
            openmc.EnergyFilter(neutron_edges_eV),
        ]
        gas.scores = gas_scores
        tallies.append(gas)
        gas_name = gas.name
        for cell, mesh_filter in sorted(
            (local_mesh_filters_by_cell or {}).items()
        ):
            local_gas = openmc.Tally(
                name=f"pstl_magnet_{int(cell)}_neutron_local_mesh_gas"
            )
            local_gas.filters = [
                mesh_filter,
                openmc.ParticleFilter(["neutron"]),
                openmc.EnergyFilter(neutron_edges_eV),
            ]
            local_gas.scores = gas_scores
            tallies.append(local_gas)
            inventory["local_mesh_gas"].append(local_gas.name)

    production_requested = "production" in enabled
    for produced in produced_particles:
        name = str(produced).lower()
        status = availability(f"produce:{name}", production_requested)
        response_report[f"produce:{name}"] = status
        if not status["available"]:
            continue
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
        volume_flux=tuple(inventory["volume_flux"]),
        reactions=reactions_name,
        nuclide_reactions=tuple(inventory["nuclide_reactions"]),
        production=tuple(inventory["production"]),
        heating=tuple(inventory["heating"]),
        total_heating=total_heating.name,
        local_mesh_flux=tuple(inventory["local_mesh_flux"]),
        local_mesh_heating=tuple(inventory["local_mesh_heating"]),
        local_mesh_damage=tuple(inventory["local_mesh_damage"]),
        local_mesh_gas=tuple(inventory["local_mesh_gas"]),
        damage_energy=damage_name,
        gas_production=gas_name,
        profile=tally_profile,
        response_availability=response_report,
    )


def add_reactor_component_tallies(
    model,
    *,
    component_cell_ids: Mapping[str, int],
    neutron_edges_eV: Sequence[float],
    photon_edges_eV: Sequence[float],
) -> dict[str, Any]:
    """Attach the minimum global reactor and breeder-accounting tallies.

    These tallies are cell-resolved over the physical DAGMC volumes.  The
    continuous magnet layer is one physical cell; engineering coil identities
    are intentionally not fabricated as global transport cells.
    """
    require_capabilities()
    openmc = _openmc()
    required = {
        "first_wall",
        "breeder",
        "back_wall",
        "high_temperature_shield",
        "vacuum_vessel",
        "low_temperature_shield",
        "magnets",
    }
    if not required.issubset(component_cell_ids):
        missing = sorted(required - set(component_cell_ids))
        raise ValueError(f"reactor component cell IDs omit {missing}")
    physical = {
        name: int(component_cell_ids[name]) for name in sorted(required)
    }
    if len(set(physical.values())) != len(physical) or any(
        value <= 0 for value in physical.values()
    ):
        raise ValueError(
            "reactor component cell IDs must be unique and positive"
        )
    cells = [physical[name] for name in sorted(physical)]
    tallies = model.tallies if model.tallies is not None else openmc.Tallies()
    names: dict[str, Any] = {"component_cell_ids": physical}

    for particle, edges in (
        ("neutron", neutron_edges_eV),
        ("photon", photon_edges_eV),
    ):
        flux = openmc.Tally(name=f"pstl_reactor_{particle}_component_flux")
        flux.filters = [
            openmc.CellFilter(cells),
            openmc.ParticleFilter([particle]),
            openmc.EnergyFilter(edges),
        ]
        flux.scores = ["flux"]
        tallies.append(flux)
        heating = openmc.Tally(
            name=f"pstl_reactor_{particle}_component_heating"
        )
        heating.filters = list(flux.filters)
        heating.scores = ["heating"]
        tallies.append(heating)
        names[f"{particle}_component_flux"] = flux.name
        names[f"{particle}_component_heating"] = heating.name

    reactions = openmc.Tally(name="pstl_reactor_neutron_component_reactions")
    reactions.filters = [
        openmc.CellFilter(cells),
        openmc.ParticleFilter(["neutron"]),
        openmc.EnergyFilter(neutron_edges_eV),
        openmc.ReactionFilter([16, 17, 102]),
    ]
    reactions.scores = ["events"]
    tallies.append(reactions)
    names["component_reactions"] = reactions.name

    breeder = physical["breeder"]
    tbr = openmc.Tally(name="pstl_breeder_tritium_production")
    tbr.filters = [
        openmc.CellFilter([breeder]),
        openmc.ParticleFilter(["neutron"]),
        openmc.EnergyFilter(neutron_edges_eV),
    ]
    tbr.scores = ["H3-production"]
    tallies.append(tbr)
    names["breeder_tritium_production"] = tbr.name
    names["tbr_semantics"] = (
        "H3 production per source history in modeled 90-degree period; "
        "apply physical source normalization exactly once"
    )
    names["statistics_qualified"] = False
    model.tallies = tallies
    return names


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
