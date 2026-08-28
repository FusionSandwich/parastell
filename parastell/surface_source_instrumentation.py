"""Input-general OpenMC 0.16 instrumentation for closed magnet boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def build_surface_instrumentation_spec(
    *,
    surface_ids: Sequence[int],
    energy_edges_by_particle: Mapping[str, Sequence[float]],
    openmc_normal_sign_by_surface: Mapping[int, int],
    max_particles_per_process: int,
    max_source_files: int,
    mpi_ranks: int,
    coupling_interface: str = "homogenized_magnet_outer_boundary",
) -> dict:
    """Build a geometry-ID-neutral, fail-closed instrumentation contract."""

    surfaces = tuple(int(value) for value in surface_ids)
    if (
        not surfaces
        or len(surfaces) != len(set(surfaces))
        or min(surfaces) <= 0
    ):
        raise ValueError("surface_ids must be unique positive integers")
    supported_interfaces = {
        "homogenized_magnet_outer_boundary",
        "outer_casing_external",
        "winding_pack",
    }
    if coupling_interface not in supported_interfaces:
        raise ValueError(
            f"unsupported coupling_interface {coupling_interface!r}"
        )
    signs = {
        int(key): int(value)
        for key, value in openmc_normal_sign_by_surface.items()
    }
    if set(signs) != set(surfaces) or any(
        value not in {-1, 1} for value in signs.values()
    ):
        raise ValueError(
            "openmc_normal_sign_by_surface must map every selected surface "
            "to -1 or +1 from DAGMC forward/reverse topology"
        )
    if (
        isinstance(max_particles_per_process, bool)
        or int(max_particles_per_process) != max_particles_per_process
        or int(max_particles_per_process) <= 0
    ):
        raise ValueError(
            "max_particles_per_process must be a positive integer"
        )
    if (
        isinstance(max_source_files, bool)
        or int(max_source_files) != max_source_files
        or int(max_source_files) <= 0
    ):
        raise ValueError("max_source_files must be a positive integer")
    if (
        isinstance(mpi_ranks, bool)
        or int(mpi_ranks) != mpi_ranks
        or int(mpi_ranks) <= 0
    ):
        raise ValueError("mpi_ranks must be a positive integer")
    axes = {}
    for particle, values in energy_edges_by_particle.items():
        if particle not in {"neutron", "photon"}:
            raise ValueError(f"unsupported surface particle {particle!r}")
        edges = np.asarray(values, dtype=float)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
            or edges[0] < 0.0
        ):
            raise ValueError(f"{particle} energy edges are invalid")
        axes[particle] = edges.tolist()
    if not axes:
        raise ValueError("at least one particle energy axis is required")
    per_process = int(max_particles_per_process)
    file_count = int(max_source_files)
    ranks = int(mpi_ranks)
    return {
        "schema": "parastell.surface_source_instrumentation/v1.0.0",
        "coupling_interface": coupling_interface,
        "surface_ids": sorted(surfaces),
        "particles": sorted(axes),
        "energy_edges_eV": axes,
        "native_mu_edges": [-1.0, 0.0, 1.0],
        "openmc_normal_sign_by_surface": {
            str(key): signs[key] for key in sorted(signs)
        },
        "canonical_direction_mapping": (
            "canonical_mu = openmc_normal_sign_by_surface[surface_id] * "
            "native_mu; incoming canonical_mu < 0; outgoing canonical_mu > 0"
        ),
        "settings": {
            "surface_ids": sorted(surfaces),
            "max_particles": per_process,
            "max_source_files": file_count,
        },
        "mpi_ranks": ranks,
        "per_file_capacity": per_process * ranks,
        "configured_capacity": per_process * ranks * file_count,
        "records_both_directions": True,
        "outward_orientation_source": "dagmc_forward_reverse_topology",
        "canonical_weight_contract": "OpenMC wgt divided only by exact source histories",
    }


def apply_openmc16_surface_instrumentation(
    model, spec: Mapping
) -> tuple[str, ...]:
    """Apply a validated specification without changing physical geometry."""

    import openmc

    configure_openmc16_surface_bank(model, spec)
    tallies = model.tallies if model.tallies is not None else openmc.Tallies()
    existing_names = {tally.name for tally in tallies}
    created = []
    for particle in spec["particles"]:
        name = f"pstl_envelope_{particle}_directional_current"
        if name in existing_names:
            raise ValueError(f"duplicate tally name {name!r}")
        tally = openmc.Tally(name=name)
        tally.filters = [
            openmc.SurfaceFilter(spec["surface_ids"]),
            openmc.MuSurfaceFilter(spec["native_mu_edges"]),
            openmc.ParticleFilter([particle]),
            openmc.EnergyFilter(spec["energy_edges_eV"][particle]),
        ]
        tally.scores = ["current"]
        tallies.append(tally)
        existing_names.add(name)
        created.append(name)
    model.tallies = tallies
    return tuple(created)


def configure_openmc16_surface_bank(model, spec: Mapping) -> None:
    """Configure the crossing bank without creating duplicate current tallies."""
    import openmc

    if openmc.__version__ != "0.16.0":
        raise RuntimeError(
            f"expected OpenMC 0.16.0, found {openmc.__version__}"
        )
    if spec.get("schema") != "parastell.surface_source_instrumentation/v1.0.0":
        raise ValueError("unknown surface instrumentation schema")
    required = {"surface_ids", "max_particles", "max_source_files"}
    settings = spec.get("settings")
    if not isinstance(settings, Mapping) or set(settings) != required:
        raise ValueError("surface source settings are incomplete")
    model.settings.surf_source_write = dict(settings)
    if "photon" in spec.get("particles", ()):
        model.settings.photon_transport = True
