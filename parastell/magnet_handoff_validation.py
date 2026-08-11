"""Executable scientific validation gates for the magnet spectral handoff.

The gates are intentionally ordered:

1. a planar OpenMC current/source-bank closure problem;
2. a small ParaStell DAGMC sector using the repository's real source mesh;
3. deterministic replay through explicit REBCO conductor layers.

Each gate emits machine-readable evidence and fails closed.  The sector asset
is a ParaStell first-wall test model and is labelled as a magnet-interface
proxy; it validates the software coupling, not a reactor lifetime prediction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import openmc

from .hts_multilayer import analytic_transmission
from .hts_multilayer import constant_response_library
from .hts_multilayer import replay_phase_space
from .hts_multilayer import verification_rebco_stack
from .hts_multilayer import zero_response_library
from .invessel_build import VMECSurface
from .magnet_spectral_handoff import CoordinateFrame
from .magnet_spectral_handoff import MagnetRegion
from .magnet_spectral_handoff import MagnetSpectralHandoff
from .magnet_spectral_handoff import MeshSpec
from .nwl_utils import compute_flux_coordinates
from .pystell.read_vmec import VMECData


VALIDATION_SCHEMA = "parastell.magnet_handoff_validation"
VALIDATION_SCHEMA_VERSION = "1.0"


def run_planar_interface_gate(
    output_dir: str | Path,
    *,
    particles: int = 5_000,
    threads: int = 2,
) -> dict[str, Any]:
    """Run an exactly solvable planar interface and multilayer closure gate."""

    output_dir = _fresh_directory(output_dir)
    run_dir = output_dir / "openmc"
    run_dir.mkdir()

    model, handoff = _build_planar_model(particles)
    statepoint = Path(model.run(cwd=run_dir, threads=threads))
    if not statepoint.is_absolute():
        statepoint = run_dir / statepoint
    source_path = run_dir / "surface_source.h5"
    if not statepoint.is_file() or not source_path.is_file():
        raise RuntimeError("planar OpenMC run did not produce required files")

    spectra_path = output_dir / "magnet_spectra.h5"
    phase_path = output_dir / "hts_phase_space_source.h5"
    handoff.export_statepoint(statepoint, spectra_path)
    handoff.export_surface_source(
        source_path,
        phase_path,
        region_name="planar_magnet_interface",
        selection="incoming",
    )

    stack = verification_rebco_stack()
    energy_bounds = handoff.energy_bounds_eV
    transparent_library = zero_response_library(
        stack,
        energy_bounds,
        ("neutron",),
        metadata={"gate": "planar_streaming_closure"},
    )
    transparent_path = output_dir / "transparent_multilayer_replay.h5"
    transparent = replay_phase_space(
        phase_path,
        spectra_path,
        transparent_path,
        stack=stack,
        response_library=transparent_library,
        provenance={"gate": "planar_interface_closure"},
    )

    verification_removal = 2.5
    analytic_library = constant_response_library(
        stack,
        energy_bounds,
        ("neutron",),
        verification_removal,
        metadata={"gate": "analytic_attenuation"},
    )
    analytic_path = output_dir / "analytic_multilayer_replay.h5"
    analytic = replay_phase_space(
        phase_path,
        spectra_path,
        analytic_path,
        stack=stack,
        response_library=analytic_library,
        provenance={"gate": "planar_analytic_attenuation"},
    )

    current = _boundary_current_total(
        spectra_path,
        direction="incoming",
    )
    source_records = _phase_space_record_count(phase_path)
    expected_transmission = analytic_transmission(
        analytic.input_current_per_source,
        verification_removal,
        stack.total_thickness_cm,
        1.0,
    )
    analytic_error = abs(
        analytic.transmitted_current_per_source - expected_transmission
    ) / max(abs(expected_transmission), np.finfo(float).tiny)

    tolerances = {
        "openmc_current_absolute": 1.0e-12,
        "transparent_balance_relative": 1.0e-12,
        "transparent_transmission_relative": 1.0e-12,
        "analytic_transmission_relative": 1.0e-12,
    }
    transparent_transmission_error = abs(
        transparent.transmitted_current_per_source
        - transparent.input_current_per_source
    ) / max(abs(transparent.input_current_per_source), np.finfo(float).tiny)
    checks = {
        "openmc_current_is_unity": abs(current - 1.0)
        <= tolerances["openmc_current_absolute"],
        "surface_source_nonempty": source_records > 0,
        "transparent_balance": transparent.relative_balance_error
        <= tolerances["transparent_balance_relative"],
        "transparent_transmission": transparent_transmission_error
        <= tolerances["transparent_transmission_relative"],
        "analytic_attenuation": analytic_error
        <= tolerances["analytic_transmission_relative"],
    }
    result = {
        "gate": "planar_interface_closure",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "tolerances": tolerances,
        "metrics": {
            "openmc_boundary_current_per_source": current,
            "surface_source_record_count": source_records,
            "transparent_replay": transparent.to_dict(),
            "transparent_transmission_relative_error": (
                transparent_transmission_error
            ),
            "analytic_replay": analytic.to_dict(),
            "analytic_expected_transmission_per_source": (
                expected_transmission
            ),
            "analytic_transmission_relative_error": analytic_error,
        },
        "artifacts": {
            "statepoint": str(statepoint),
            "surface_source": str(source_path),
            "spectra": str(spectra_path),
            "phase_space": str(phase_path),
            "transparent_replay": str(transparent_path),
            "analytic_replay": str(analytic_path),
        },
        "versions": _software_versions(),
    }
    _write_json(output_dir / "gate_result.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("planar interface closure gate failed")
    return result


def run_parastell_sector_gate(
    output_dir: str | Path,
    *,
    assets_dir: str | Path,
    particles: int = 2_000,
    threads: int = 2,
    wall_s: float = 1.08,
) -> dict[str, Any]:
    """Run the repository's real DAGMC/source-mesh sector and replay it."""

    output_dir = _fresh_directory(output_dir)
    run_dir = output_dir / "openmc"
    run_dir.mkdir()
    assets_dir = Path(assets_dir).resolve()
    assets = _sector_assets(assets_dir)

    model, handoff = _build_sector_model(
        assets,
        particles=particles,
    )
    statepoint = Path(model.run(cwd=run_dir, threads=threads))
    if not statepoint.is_absolute():
        statepoint = run_dir / statepoint
    source_path = run_dir / "surface_source.h5"
    if not statepoint.is_file() or not source_path.is_file():
        raise RuntimeError("sector OpenMC run did not produce required files")

    source_positions = _source_positions(source_path)
    if len(source_positions) == 0:
        raise RuntimeError("sector surface source is empty")
    magnet_outward_normals = _vmec_magnet_outward_normals(
        source_positions,
        assets["vmec"],
        wall_s=wall_s,
        num_threads=max(1, min(threads, os.cpu_count() or 1)),
    )

    spectra_path = output_dir / "magnet_spectra.h5"
    phase_path = output_dir / "hts_phase_space_source.h5"
    handoff.export_statepoint(statepoint, spectra_path)
    handoff.export_surface_source(
        source_path,
        phase_path,
        region_name="parastell_first_wall_proxy",
        selection="all",
        record_outward_normals_global=magnet_outward_normals,
        record_normal_basis="vmec_reference_surface_per_record",
    )

    phase_metrics = _phase_space_direction_metrics(phase_path)
    if phase_metrics["record_count"] == 0:
        raise RuntimeError("exported sector phase space is empty")

    stack = verification_rebco_stack()
    transparent_library = zero_response_library(
        stack,
        handoff.energy_bounds_eV,
        ("neutron",),
        metadata={
            "gate": "sector_to_multilayer_streaming_closure",
            "physical_material_model": False,
        },
    )
    x_edges, y_edges = _four_bin_edges(_phase_space_positions(phase_path))
    replay_path = output_dir / "sector_multilayer_replay.h5"
    replay = replay_phase_space(
        phase_path,
        spectra_path,
        replay_path,
        stack=stack,
        response_library=transparent_library,
        tally_direction=None,
        record_direction="incoming",
        x_edges_cm=x_edges,
        y_edges_cm=y_edges,
        provenance={
            "gate": "real_parastell_sector_to_explicit_hts_layers",
            "interface_classification": "first_wall_magnet_proxy",
            "normal_reconstruction": (
                "VMEC reference-surface finite-difference normal at each "
                "surface-source record"
            ),
        },
    )

    current = _boundary_current_total(spectra_path, direction=None)
    current_match_error = abs(replay.input_current_per_source - current) / max(
        abs(current), np.finfo(float).tiny
    )
    response_shape = _response_shape(replay_path)
    tolerances = {
        "incoming_fraction_minimum": 0.95,
        "current_match_relative": 1.0e-12,
        "replay_balance_relative": 1.0e-12,
    }
    checks = {
        "real_dagmc_asset_used": assets["dagmc"].is_file(),
        "real_source_mesh_used": assets["source_mesh"].is_file(),
        "surface_source_nonempty": phase_metrics["record_count"] > 0,
        "vmec_normals_are_directionally_consistent": (
            phase_metrics["incoming_fraction"]
            >= tolerances["incoming_fraction_minimum"]
        ),
        "tally_bank_current_match": current_match_error
        <= tolerances["current_match_relative"],
        "multilayer_balance": replay.relative_balance_error
        <= tolerances["replay_balance_relative"],
        "explicit_layer_axis_present": response_shape[2] == len(stack.layers),
        "heterogeneous_xy_mesh_present": response_shape[0] > 1
        and response_shape[1] > 1,
    }
    result = {
        "gate": "real_parastell_sector_and_multilayer_replay",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "tolerances": tolerances,
        "metrics": {
            "boundary_current_per_source": current,
            "phase_space": phase_metrics,
            "current_match_relative_error": current_match_error,
            "replay": replay.to_dict(),
            "response_shape": response_shape,
            "response_dimension_order": (
                "x_bin,y_bin,layer,particle,energy_group"
            ),
        },
        "assets": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in assets.items()
            if isinstance(path, Path)
        },
        "scope": {
            "geometry": "real ParaStell repository DAGMC sector test asset",
            "source": "real ParaStell tetrahedral source mesh and strengths",
            "interface": (
                "first-wall surface used as a magnet-boundary software "
                "integration proxy"
            ),
            "material_response": (
                "transparent closure library; no physical REBCO attenuation "
                "or PKA claim"
            ),
        },
        "artifacts": {
            "statepoint": str(statepoint),
            "surface_source": str(source_path),
            "spectra": str(spectra_path),
            "phase_space": str(phase_path),
            "multilayer_replay": str(replay_path),
        },
        "versions": _software_versions(),
    }
    _write_json(output_dir / "gate_result.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("ParaStell sector/multilayer gate failed")
    return result


def run_all_gates(
    output_dir: str | Path,
    *,
    assets_dir: str | Path,
    planar_particles: int = 5_000,
    sector_particles: int = 2_000,
    threads: int = 2,
) -> dict[str, Any]:
    """Execute both transport gates and write a consolidated attestation."""

    output_dir = _fresh_directory(output_dir)
    planar = run_planar_interface_gate(
        output_dir / "planar",
        particles=planar_particles,
        threads=threads,
    )
    sector = run_parastell_sector_gate(
        output_dir / "sector",
        assets_dir=assets_dir,
        particles=sector_particles,
        threads=threads,
    )
    result = {
        "schema": VALIDATION_SCHEMA,
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": (
            "PASS"
            if planar["status"] == "PASS" and sector["status"] == "PASS"
            else "FAIL"
        ),
        "gates": {
            "planar_interface_closure": planar,
            "real_sector_multilayer_replay": sector,
        },
        "versions": _software_versions(),
    }
    _write_json(output_dir / "validation_attestation.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("magnet handoff scientific validation failed")
    return result


def _build_planar_model(
    particles: int,
) -> tuple[openmc.Model, MagnetSpectralHandoff]:
    openmc.reset_auto_ids()
    x_min = openmc.XPlane(
        x0=-1.0,
        boundary_type="vacuum",
        surface_id=100,
    )
    interface = openmc.XPlane(x0=0.0, surface_id=101)
    x_max = openmc.XPlane(
        x0=1.0,
        boundary_type="vacuum",
        surface_id=102,
    )
    y_min = openmc.YPlane(
        y0=-0.5,
        boundary_type="reflective",
        surface_id=103,
    )
    y_max = openmc.YPlane(
        y0=0.5,
        boundary_type="reflective",
        surface_id=104,
    )
    z_min = openmc.ZPlane(
        z0=-0.5,
        boundary_type="reflective",
        surface_id=105,
    )
    z_max = openmc.ZPlane(
        z0=0.5,
        boundary_type="reflective",
        surface_id=106,
    )
    transverse = +y_min & -y_max & +z_min & -z_max
    left = openmc.Cell(
        cell_id=1,
        region=+x_min & -interface & transverse,
    )
    right = openmc.Cell(
        cell_id=2,
        region=+interface & -x_max & transverse,
    )
    geometry = openmc.Geometry([left, right])

    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((-0.5, 0.0, 0.0))
    source.angle = openmc.stats.Monodirectional((1.0, 0.0, 0.0))
    source.energy = openmc.stats.Discrete([14.1e6], [1.0])
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = int(particles)
    settings.batches = 1
    settings.seed = 31_415
    settings.source = [source]

    model = openmc.Model(geometry=geometry, settings=settings)
    region = MagnetRegion(
        name="planar_magnet_interface",
        cell_ids=(2,),
        phase_space_cell_id=2,
        surface_ids=(101,),
        source_region_id="planar-interface",
        magnet_id="planar-magnet",
        surface_areas_cm2={101: 1.0},
        surface_normal_signs={101: -1},
        surface_outward_normals_global={101: (-1.0, 0.0, 0.0)},
        coordinate_frame=CoordinateFrame(
            x_axis=(0.0, 1.0, 0.0),
            y_axis=(0.0, 0.0, 1.0),
            z_axis=(1.0, 0.0, 0.0),
            labels=("tape_width", "tape_length", "tape_depth"),
        ),
        metadata={"geometry": "exact_planar_void_interface"},
    )
    handoff = MagnetSpectralHandoff(
        regions=(region,),
        energy_bounds_eV=(0.0, 20.0e6),
        particles=("neutron",),
        mu_bounds=(-1.0, 0.0, 1.0),
        include_cell_flux=False,
        include_mesh_flux=False,
        include_boundary_current=True,
        include_heating=False,
        include_damage_energy=False,
        metadata={"validation_gate": "planar_interface_closure"},
    )
    handoff.attach_to_model(model, enable_photon_transport=False)
    handoff.configure_surface_source(
        settings,
        region.name,
        direction="incoming",
        max_particles=max(2 * particles, 1),
    )
    return model, handoff


def _build_sector_model(
    assets: Mapping[str, Path],
    *,
    particles: int,
) -> tuple[openmc.Model, MagnetSpectralHandoff]:
    openmc.reset_auto_ids()
    toroidal_extent = math.radians(15.0)
    dagmc_universe = openmc.DAGMCUniverse(
        str(assets["dagmc"]),
        auto_geom_ids=False,
    )
    periodic_initial = openmc.YPlane(
        boundary_type="periodic",
        surface_id=9990,
    )
    periodic_final = openmc.Plane(
        a=math.sin(toroidal_extent),
        b=-math.cos(toroidal_extent),
        c=0.0,
        d=0.0,
        boundary_type="periodic",
        surface_id=9991,
    )
    vacuum = openmc.Sphere(
        r=10_000.0,
        boundary_type="vacuum",
        surface_id=9992,
    )
    period_region = -vacuum & +periodic_initial & +periodic_final
    period_cell = openmc.Cell(
        cell_id=9999,
        region=period_region,
        fill=dagmc_universe,
    )
    geometry = openmc.Geometry([period_cell])

    source_mesh = openmc.UnstructuredMesh(
        str(assets["source_mesh"]),
        "moab",
    )
    strengths = np.load(assets["strengths"])
    source = openmc.IndependentSource()
    source.space = openmc.stats.MeshSpatial(
        source_mesh,
        strengths=strengths,
        volume_normalized=False,
    )
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([14.1e6], [1.0])
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = int(particles)
    settings.batches = 1
    settings.seed = 27_182
    settings.source = [source]
    openmc.Materials.cross_sections = str(assets["cross_sections"])

    model = openmc.Model(geometry=geometry, settings=settings)
    region = MagnetRegion(
        name="parastell_first_wall_proxy",
        cell_ids=(9999,),
        phase_space_cell_id=9999,
        surface_ids=(1,),
        source_region_id="parastell-sector-first-wall",
        magnet_id="integration-proxy",
        coordinate_frame=CoordinateFrame(
            labels=("global_x", "global_y", "global_z"),
        ),
        mesh=MeshSpec(
            kind="unstructured",
            filename=str(assets["source_mesh"]),
            library="moab",
            mesh_id=8_100_001,
            name="parastell_source_mesh_flux",
        ),
        metadata={
            "classification": "first_wall_magnet_interface_proxy",
            "toroidal_extent_deg": 15.0,
            "dagmc_geometry": str(assets["dagmc"]),
            "source_mesh": str(assets["source_mesh"]),
        },
    )
    handoff = MagnetSpectralHandoff(
        regions=(region,),
        energy_bounds_eV=(
            0.0,
            1.0e5,
            1.0e6,
            5.0e6,
            10.0e6,
            14.0e6,
            14.2e6,
            20.0e6,
        ),
        particles=("neutron",),
        mu_bounds=(-1.0, -0.5, 0.0, 0.5, 1.0),
        include_cell_flux=False,
        include_mesh_flux=True,
        include_boundary_current=True,
        include_heating=False,
        include_damage_energy=False,
        metadata={
            "validation_gate": "real_parastell_sector",
            "proxy_warning": (
                "The repository first-wall test surface validates coupling; "
                "it is not a resolved magnet geometry."
            ),
        },
    )
    handoff.attach_to_model(model, enable_photon_transport=False)
    handoff.configure_surface_source(
        settings,
        region.name,
        direction="all",
        max_particles=max(2 * particles, 1),
    )
    return model, handoff


def _vmec_magnet_outward_normals(
    positions_cm: np.ndarray,
    vmec_path: Path,
    *,
    wall_s: float,
    num_threads: int,
) -> np.ndarray:
    toroidal, poloidal = compute_flux_coordinates(
        vmec_path,
        wall_s,
        positions_cm,
        num_threads,
        1.0e-5,
    )
    vmec_surface = VMECSurface(VMECData(vmec_path))
    dtheta = 1.0e-4
    dphi = 1.0e-4
    ds = 1.0e-4
    normals = np.empty_like(positions_cm, dtype=float)
    for index, (phi, theta) in enumerate(zip(toroidal, poloidal)):
        theta_values = np.asarray([theta - dtheta, theta + dtheta])
        theta_points = vmec_surface.angles_to_xyz(
            phi,
            theta_values,
            wall_s,
            100.0,
        )
        phi_minus = vmec_surface.angles_to_xyz(
            phi - dphi,
            np.asarray([theta]),
            wall_s,
            100.0,
        )[0]
        phi_plus = vmec_surface.angles_to_xyz(
            phi + dphi,
            np.asarray([theta]),
            wall_s,
            100.0,
        )[0]
        s_minus = vmec_surface.angles_to_xyz(
            phi,
            np.asarray([theta]),
            wall_s - ds,
            100.0,
        )[0]
        s_plus = vmec_surface.angles_to_xyz(
            phi,
            np.asarray([theta]),
            wall_s + ds,
            100.0,
        )[0]
        tangent_theta = theta_points[1] - theta_points[0]
        tangent_phi = phi_plus - phi_minus
        plasma_outward = np.cross(tangent_theta, tangent_phi)
        if np.dot(plasma_outward, s_plus - s_minus) < 0.0:
            plasma_outward *= -1.0
        norm = np.linalg.norm(plasma_outward)
        if not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(
                f"failed to reconstruct VMEC normal for record {index}"
            )
        plasma_outward /= norm
        normals[index] = -plasma_outward
    return normals


def _sector_assets(assets_dir: Path) -> dict[str, Path]:
    assets = {
        "dagmc": assets_dir / "nwl_geom.h5m",
        "source_mesh": assets_dir / "source_mesh.h5m",
        "strengths": assets_dir / "strengths.npy",
        "vmec": assets_dir / "wout_vmec.nc",
        "cross_sections": assets_dir / "cross_sections" / "cross_sections.xml",
    }
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing sector assets: " + ", ".join(missing))
    return assets


def _boundary_current_total(
    spectra_path: Path,
    *,
    direction: str | None,
) -> float:
    with h5py.File(spectra_path, "r") as source:
        groups = [
            group
            for group in source["tallies"].values()
            if str(group.attrs.get("role", "")) == "boundary_current"
        ]
        if len(groups) != 1:
            raise ValueError("expected exactly one boundary-current tally")
        group = groups[0]
        mean_name = _find_dataset(group, "mean")
        mask = np.ones(len(group[mean_name]), dtype=bool)
        if direction is not None:
            direction_name = _find_dataset(group, "magnet_direction")
            labels = group[direction_name].asstr()[:]
            mask &= labels == direction
        return float(np.sum(np.abs(group[mean_name][:][mask])))


def _phase_space_direction_metrics(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as source:
        phase = source["phase_space"]
        labels = phase["magnet_direction"].asstr()[:]
        basis = phase["direction_label_basis"].asstr()[:]
        mu = phase["mu_outward"][:]
    count = len(labels)
    incoming = int(np.count_nonzero(labels == "incoming"))
    return {
        "record_count": count,
        "incoming_count": incoming,
        "outgoing_count": int(np.count_nonzero(labels == "outgoing")),
        "grazing_count": int(np.count_nonzero(labels == "grazing")),
        "unknown_count": int(np.count_nonzero(labels == "unknown")),
        "incoming_fraction": incoming / count if count else 0.0,
        "finite_mu_fraction": (
            float(np.count_nonzero(np.isfinite(mu))) / count if count else 0.0
        ),
        "direction_basis": sorted(set(basis.tolist())),
    }


def _source_positions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as source:
        bank = source["source_bank"]
        position = bank["r"]
        return np.column_stack(
            (position["x"], position["y"], position["z"])
        ).astype(float)


def _phase_space_positions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as source:
        return source["phase_space"]["position_local_cm"][:]


def _phase_space_record_count(path: Path) -> int:
    with h5py.File(path, "r") as source:
        return int(source.attrs["record_count"])


def _four_bin_edges(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = []
    for axis in (0, 1):
        lower = float(np.min(positions[:, axis]))
        upper = float(np.max(positions[:, axis]))
        if lower == upper:
            upper = lower + 1.0e-9
        margin = (upper - lower) * 1.0e-12
        edges.append(np.linspace(lower - margin, upper + margin, 5))
    return edges[0], edges[1]


def _response_shape(path: Path) -> list[int]:
    with h5py.File(path, "r") as source:
        return list(source["response"]["incident_current_per_source"].shape)


def _find_dataset(group: h5py.Group, candidate: str) -> str:
    key = "".join(character for character in candidate if character.isalnum())
    for name in group:
        normalised = "".join(
            character for character in name if character.isalnum()
        )
        if normalised == key or key in normalised:
            return name
    raise KeyError(candidate)


def _fresh_directory(path: str | Path) -> Path:
    path = Path(path).resolve()
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _software_versions() -> dict[str, str]:
    try:
        import parastell

        parastell_version = getattr(parastell, "__version__", "0.1.0")
    except Exception:
        parastell_version = "unknown"
    return {
        "python": sys.version.split()[0],
        "openmc": getattr(openmc, "__version__", "unknown"),
        "parastell": str(parastell_version),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ParaStell magnet-handoff scientific validation gates"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("magnet_handoff_validation"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("tests/files_for_tests"),
    )
    parser.add_argument("--planar-particles", type=int, default=5_000)
    parser.add_argument("--sector-particles", type=int, default=2_000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--gate",
        choices=("all", "planar", "sector"),
        default="all",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.gate == "planar":
        result = run_planar_interface_gate(
            args.output_dir,
            particles=args.planar_particles,
            threads=args.threads,
        )
    elif args.gate == "sector":
        result = run_parastell_sector_gate(
            args.output_dir,
            assets_dir=args.assets_dir,
            particles=args.sector_particles,
            threads=args.threads,
        )
    else:
        result = run_all_gates(
            args.output_dir,
            assets_dir=args.assets_dir,
            planar_particles=args.planar_particles,
            sector_particles=args.sector_particles,
            threads=args.threads,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
