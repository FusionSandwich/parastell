"""Executable scientific validation gates for the magnet spectral handoff.

The gates are intentionally ordered:

1. a planar OpenMC current/source-bank closure problem;
2. a small ParaStell DAGMC sector using the repository's real source mesh;
3. a CAD-derived, magnet-attached finite coupling plane in real DAGMC;
4. deterministic replay through explicit REBCO conductor layers.

Each gate emits machine-readable evidence and fails closed.  The sector asset
is a ParaStell first-wall test model and is labelled as a magnet-interface
proxy; it validates the software coupling, not a reactor lifetime prediction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
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
from .magnet_spectral_handoff import MagnetCouplingPlane
from .magnet_spectral_handoff import MagnetRegion
from .magnet_spectral_handoff import MagnetSpectralHandoff
from .magnet_spectral_handoff import MeshSpec
from .magnet_spectral_handoff import software_validation_energy_bounds
from .nwl_utils import compute_flux_coordinates
from .pystell.read_vmec import VMECData


VALIDATION_SCHEMA = "parastell.magnet_handoff_validation"
VALIDATION_SCHEMA_VERSION = "1.0"
ACTIVE_BATCHES = 20


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
        tally_direction="incoming",
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

    total_current = _boundary_current_total(spectra_path, direction=None)
    incoming_current = _boundary_current_total(
        spectra_path,
        direction="incoming",
    )
    outgoing_current = _boundary_current_total(
        spectra_path,
        direction="outgoing",
    )
    grazing_current = _boundary_current_total(
        spectra_path,
        direction="grazing",
    )
    partitioned_current = incoming_current + outgoing_current + grazing_current
    current_partition_error = abs(total_current - partitioned_current) / max(
        abs(total_current),
        np.finfo(float).tiny,
    )
    current_match_error = abs(
        replay.input_current_per_source - incoming_current
    ) / max(
        abs(incoming_current),
        np.finfo(float).tiny,
    )
    response_shape = _response_shape(replay_path)
    partitioned_records = sum(
        phase_metrics[name]
        for name in (
            "incoming_count",
            "outgoing_count",
            "grazing_count",
            "unknown_count",
        )
    )
    tolerances = {
        "current_partition_relative": 1.0e-12,
        "current_match_relative": 1.0e-12,
        "replay_balance_relative": 1.0e-12,
    }
    checks = {
        "real_dagmc_asset_used": assets["dagmc"].is_file(),
        "real_source_mesh_used": assets["source_mesh"].is_file(),
        "surface_source_nonempty": phase_metrics["record_count"] > 0,
        "vmec_normals_are_finite": (
            phase_metrics["finite_mu_fraction"] == 1.0
        ),
        "direction_labels_partition_records": (
            partitioned_records == phase_metrics["record_count"]
            and phase_metrics["unknown_count"] == 0
        ),
        "direction_labels_match_mu_sign": (
            phase_metrics["label_sign_mismatch_count"] == 0
        ),
        "bidirectional_crossings_observed": (
            phase_metrics["incoming_count"] > 0
            and phase_metrics["outgoing_count"] > 0
        ),
        "boundary_current_partition": current_partition_error
        <= tolerances["current_partition_relative"],
        "incoming_tally_bank_current_match": current_match_error
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
            "boundary_current_per_source": {
                "total": total_current,
                "incoming": incoming_current,
                "outgoing": outgoing_current,
                "grazing": grazing_current,
            },
            "phase_space": phase_metrics,
            "current_partition_relative_error": current_partition_error,
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


def run_real_magnet_gate(
    output_dir: str | Path,
    *,
    assets_dir: str | Path,
    particles: int = 100,
    threads: int = 2,
) -> dict[str, Any]:
    """Run a real ParaStell magnet DAGMC model through HTS replay."""

    output_dir = _fresh_directory(output_dir)
    run_dir = output_dir / "scored_openmc"
    baseline_dir = output_dir / "unscored_openmc"
    run_dir.mkdir()
    baseline_dir.mkdir()
    assets_dir = Path(assets_dir).resolve()
    step_path = assets_dir / "magnet_geom_with_casing.step"
    cross_sections = assets_dir / "cross_sections" / "cross_sections.xml"
    missing = [
        path for path in (step_path, cross_sections) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing real-magnet validation assets: "
            + ", ".join(str(path) for path in missing)
        )

    dagmc_path, plane = _prepare_real_magnet_geometry(
        step_path,
        output_dir / "parastell_magnet.h5m",
    )
    openmc.Materials.cross_sections = str(cross_sections)
    scored_model, handoff = _build_real_magnet_model(
        dagmc_path,
        step_path,
        plane,
        particles=particles,
        include_interface=True,
    )
    handoff.validate_plane_contract(require_production=True)
    scored_statepoint = Path(scored_model.run(cwd=run_dir, threads=threads))
    if not scored_statepoint.is_absolute():
        scored_statepoint = run_dir / scored_statepoint
    source_path = run_dir / "surface_source.h5"
    if not scored_statepoint.is_file() or not source_path.is_file():
        raise RuntimeError(
            "real-magnet OpenMC run did not produce required files"
        )

    baseline_model, _ = _build_real_magnet_model(
        dagmc_path,
        step_path,
        plane,
        particles=particles,
        include_interface=False,
    )
    baseline_statepoint = Path(
        baseline_model.run(cwd=baseline_dir, threads=threads)
    )
    if not baseline_statepoint.is_absolute():
        baseline_statepoint = baseline_dir / baseline_statepoint

    spectra_path = output_dir / "magnet_spectra.h5"
    phase_path = output_dir / "hts_phase_space_source.h5"
    replay_path = output_dir / "magnet_multilayer_replay.h5"
    handoff.export_statepoint(scored_statepoint, spectra_path)
    handoff.export_surface_source(
        source_path,
        phase_path,
        region_name="parastell_magnet_entry",
        selection="incoming",
    )
    phase_metrics = _phase_space_direction_metrics(phase_path)
    stack = verification_rebco_stack()
    replay = replay_phase_space(
        phase_path,
        spectra_path,
        replay_path,
        stack=stack,
        response_library=zero_response_library(
            stack,
            handoff.energy_bounds_eV,
            ("neutron",),
            metadata={"gate": "real_magnet_transparent_replay"},
        ),
        x_edges_cm=np.asarray(plane.u_edges_cm),
        y_edges_cm=np.asarray(plane.v_edges_cm),
        provenance={
            "gate": "real_magnet_to_explicit_hts_layers",
            "plane_id": plane.plane_id,
        },
    )

    entering_current = _boundary_current_total(
        spectra_path, direction="incoming"
    )
    closure_error = abs(
        replay.input_current_per_source - entering_current
    ) / max(abs(entering_current), np.finfo(float).tiny)
    scored_leakage = _statepoint_tally_result(scored_statepoint, 8_899_999)
    baseline_leakage = _statepoint_tally_result(baseline_statepoint, 8_899_999)
    leakage_delta = abs(scored_leakage[0] - baseline_leakage[0])
    leakage_uncertainty = math.sqrt(
        scored_leakage[1] ** 2 + baseline_leakage[1] ** 2
    )
    nonperturbation_tolerance = max(5.0 * leakage_uncertainty, 1.0e-12)
    response_shape = _response_shape(replay_path)
    checks = {
        "real_parastell_magnet_step_used": step_path.is_file(),
        "real_dagmc_generated": dagmc_path.is_file(),
        "magnet_attached_not_proxy": (
            plane.construction_method == "magnet_attached" and not plane.proxy
        ),
        "finite_plane": plane.area_cm2 > 0.0,
        "spatially_binned": all(value > 1 for value in plane.spatial_shape),
        "fusion_resolving_energy_groups": len(handoff.energy_bounds_eV) - 1
        >= 175,
        "multiple_active_batches": ACTIVE_BATCHES >= 10,
        "surface_source_nonempty": phase_metrics["record_count"] > 0,
        "all_records_entering": (
            phase_metrics["incoming_count"] == phase_metrics["record_count"]
        ),
        "current_closure": closure_error <= 1.0e-12,
        "plane_nonperturbing": leakage_delta <= nonperturbation_tolerance,
        "multilayer_balance": replay.relative_balance_error <= 1.0e-12,
        "explicit_layer_axis_present": response_shape[2] == len(stack.layers),
    }
    result = {
        "gate": "real_parastell_magnet_dagmc_and_multilayer_replay",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "requested_histories": particles * ACTIVE_BATCHES,
            "active_batches": ACTIVE_BATCHES,
            "energy_groups": len(handoff.energy_bounds_eV) - 1,
            "spatial_bins": list(plane.spatial_shape),
            "angular_bins": [
                len(handoff.mu_bounds) - 1,
                len(handoff.azimuthal_bounds_rad) - 1,
            ],
            "integrated_entering_current_per_source": entering_current,
            "integrated_exported_current_per_source": replay.input_current_per_source,
            "closure_relative_error": closure_error,
            "scored_leakage_mean_std": list(scored_leakage),
            "unscored_leakage_mean_std": list(baseline_leakage),
            "nonperturbation_absolute_delta": leakage_delta,
            "nonperturbation_tolerance": nonperturbation_tolerance,
            "phase_space": phase_metrics,
            "replay": replay.to_dict(),
            "response_shape": response_shape,
        },
        "plane": plane.to_dict(),
        "assets": {
            "magnet_step": {
                "path": str(step_path),
                "sha256": _sha256(step_path),
            },
            "dagmc": {"path": str(dagmc_path), "sha256": _sha256(dagmc_path)},
        },
        "scope": {
            "geometry": "real ParaStell magnet and casing STEP converted to DAGMC",
            "source": "finite four-point 14.1 MeV neutron field directed into the magnet",
            "material": "low-density H-1 transport surrogate for interface validation",
            "damage_claim": "transport replay only; no DPA or PKA prediction",
        },
        "artifacts": {
            "statepoint": str(scored_statepoint),
            "baseline_statepoint": str(baseline_statepoint),
            "surface_source": str(source_path),
            "spectra": str(spectra_path),
            "phase_space": str(phase_path),
            "multilayer_replay": str(replay_path),
        },
        "versions": _software_versions(),
    }
    _write_json(output_dir / "gate_result.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("real ParaStell magnet/multilayer gate failed")
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
    assets = _sector_assets(Path(assets_dir).resolve())
    openmc.Materials.cross_sections = str(assets["cross_sections"])
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
    magnet = run_real_magnet_gate(
        output_dir / "magnet",
        assets_dir=assets_dir,
        particles=max(100, sector_particles // ACTIVE_BATCHES),
        threads=threads,
    )
    result = {
        "schema": VALIDATION_SCHEMA,
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": (
            "PASS"
            if all(
                gate["status"] == "PASS" for gate in (planar, sector, magnet)
            )
            else "FAIL"
        ),
        "gates": {
            "planar_interface_closure": planar,
            "real_sector_multilayer_replay": sector,
            "real_magnet_multilayer_replay": magnet,
        },
        "versions": _software_versions(),
    }
    _write_json(output_dir / "validation_attestation.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("magnet handoff scientific validation failed")
    return result


def _prepare_real_magnet_geometry(
    step_path: Path,
    dagmc_path: Path,
) -> tuple[Path, MagnetCouplingPlane]:
    import cadquery as cq
    from cad_to_dagmc import CadToDagmc

    imported = cq.importers.importStep(str(step_path))
    solids = list(imported.solids().vals())
    if not solids:
        raise RuntimeError(f"magnet STEP contains no solids: {step_path}")
    solid = solids[0]
    bounds = solid.BoundingBox()
    center = solid.Center()
    face_plane = MagnetCouplingPlane.from_magnet_solid(
        solid,
        plane_id="magnet_entry_plane",
        magnet_region_id="parastell-magnet-0",
        magnet_component="coil-casing-0",
        role="entry",
        approximate_location_cm=(bounds.xmax, center.y, center.z),
        reference_direction=(0.0, 0.0, 1.0),
        width_cm=80.0,
        height_cm=80.0,
        u_bins=4,
        v_bins=4,
        surface_id=8_800_000,
        geometry_source={
            "path": str(step_path),
            "sha256": _sha256(step_path),
            "solid_index": 0,
        },
    )
    offset_cm = 0.05
    origin = np.asarray(face_plane.origin_cm) + offset_cm * np.asarray(
        face_plane.normal_global
    )
    plane = replace(
        face_plane,
        origin_cm=tuple(origin),
        adjacency_tolerance_cm=0.1,
        association={
            **face_plane.association,
            "minimum_plane_to_magnet_distance_cm": offset_cm,
            "criterion": (
                "plane is parallel to and 0.05 cm outward from the nearest "
                "selected CAD magnet face"
            ),
        },
    )

    converter = CadToDagmc()
    volume_count = converter.add_stp_file(
        str(step_path),
        material_tags=["magnet"] * len(solids),
    )
    if volume_count != len(solids):
        raise RuntimeError(
            f"CAD-to-DAGMC found {volume_count} volumes; expected {len(solids)}"
        )
    converter.export_dagmc_h5m_file(
        filename=str(dagmc_path),
        meshing_backend="gmsh",
        min_mesh_size=5.0,
        max_mesh_size=20.0,
        h5m_backend="pymoab",
    )
    if not dagmc_path.is_file():
        raise RuntimeError("CAD-to-DAGMC did not produce the magnet H5M file")
    return dagmc_path, plane


def _build_real_magnet_model(
    dagmc_path: Path,
    step_path: Path,
    plane: MagnetCouplingPlane,
    *,
    particles: int,
    include_interface: bool,
) -> tuple[openmc.Model, MagnetSpectralHandoff]:
    openmc.reset_auto_ids()
    magnet = openmc.Material(material_id=8_800_100, name="magnet")
    magnet.add_nuclide("H1", 1.0)
    magnet.set_density("g/cm3", 1.0e-10)
    dagmc_universe = openmc.DAGMCUniverse(
        str(dagmc_path),
        auto_geom_ids=False,
    )
    origin = np.asarray(plane.origin_cm)
    world = openmc.Sphere(
        x0=float(origin[0]),
        y0=float(origin[1]),
        z0=float(origin[2]),
        r=2_000.0,
        boundary_type="vacuum",
        surface_id=8_899_998,
    )
    world_region = -world
    if include_interface:
        cells, _ = plane.build_openmc_partition(
            dagmc_universe,
            world_region,
            cell_id_base=8_810_000,
        )
        geometry = openmc.Geometry(cells)
    else:
        geometry = openmc.Geometry(
            [
                openmc.Cell(
                    cell_id=8_820_000,
                    region=world_region,
                    fill=dagmc_universe,
                )
            ]
        )

    normal = np.asarray(plane.normal_global)
    u_axis = np.asarray(plane.u_global)
    v_axis = np.asarray(plane.v_global)
    sources: list[openmc.IndependentSource] = []
    source_points: list[list[float]] = []
    for u_offset, v_offset in (
        (-10.0, -10.0),
        (-10.0, 10.0),
        (10.0, -10.0),
        (10.0, 10.0),
    ):
        point = origin + 5.0 * normal + u_offset * u_axis + v_offset * v_axis
        source = openmc.IndependentSource()
        source.space = openmc.stats.Point(tuple(point))
        source.angle = openmc.stats.Monodirectional(tuple(-normal))
        source.energy = openmc.stats.Discrete([14.1e6], [1.0])
        source.strength = 0.25
        sources.append(source)
        source_points.append(point.tolist())
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = int(particles)
    settings.batches = ACTIVE_BATCHES
    settings.seed = 16_180
    settings.source = sources
    model = openmc.Model(
        geometry=geometry,
        materials=openmc.Materials([magnet]),
        settings=settings,
    )
    leakage = openmc.Tally(tally_id=8_899_999, name="world_leakage_current")
    leakage.filters = [
        openmc.SurfaceFilter(world),
        openmc.ParticleFilter(["neutron"]),
    ]
    leakage.scores = ["current"]
    model.tallies = openmc.Tallies([leakage])

    frame = CoordinateFrame(
        origin_cm=plane.origin_cm,
        x_axis=plane.u_global,
        y_axis=plane.v_global,
        z_axis=plane.normal_global,
        labels=("tape_width", "toroidal", "through_thickness_outward"),
    )
    region = MagnetRegion(
        name="parastell_magnet_entry",
        cell_ids=(8_810_000,),
        region_id="parastell-magnet-0",
        source_region_id="finite-fusion-source",
        phase_space_cell_id=8_810_000,
        magnet_id="parastell-magnet-0",
        magnet_component="coil-casing-0",
        entry_surface_id=plane.surface_id,
        surface_ids=(plane.surface_id,),
        surface_areas_cm2={plane.surface_id: plane.area_cm2},
        surface_normal_signs={plane.surface_id: 1},
        surface_outward_normals_global={plane.surface_id: plane.normal_global},
        coordinate_frame=frame,
        coupling_planes=(plane,),
        geometry_source_mode="production",
        metadata={
            "geometry": "real ParaStell magnet with casing",
            "tape_orientation": {
                "width": "plane u",
                "toroidal": "plane v",
                "through_thickness": "negative plane normal into magnet",
            },
        },
    )
    source_definition = {
        "particle": "neutron",
        "energy_eV": 14.1e6,
        "positions_cm": source_points,
        "direction_global": (-normal).tolist(),
        "strength_per_point": 0.25,
    }
    handoff = MagnetSpectralHandoff(
        regions=(region,),
        energy_bounds_eV=software_validation_energy_bounds(),
        particles=("neutron",),
        mu_bounds=(-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0),
        azimuthal_bounds_rad=tuple(np.linspace(-math.pi, math.pi, 9)),
        source_definition=source_definition,
        dagmc_geometry_path=str(dagmc_path),
        dagmc_geometry_hash=_sha256(dagmc_path),
        parastell_git_sha=_repository_sha(),
        openmc_git_sha=os.environ.get(
            "OPENMC_GIT_SHA",
            f"release-{openmc.__version__}-git-metadata-unavailable",
        ),
        run_statistics={
            "active_batches": ACTIVE_BATCHES,
            "particles_per_batch": int(particles),
            "requested_histories": int(particles) * ACTIVE_BATCHES,
        },
        include_cell_flux=False,
        include_mesh_flux=False,
        include_boundary_current=True,
        include_heating=False,
        include_damage_energy=False,
        metadata={
            "validation_gate": "real_parastell_magnet",
            "interface_is_proxy": False,
        },
    )
    if include_interface:
        handoff.attach_to_model(model, enable_photon_transport=False)
        handoff.configure_surface_source(
            settings,
            region.name,
            direction="incoming",
            max_particles=max((ACTIVE_BATCHES + 1) * particles, 1),
        )
    return model, handoff


def _statepoint_tally_result(
    statepoint_path: Path,
    tally_id: int,
) -> tuple[float, float]:
    statepoint = openmc.StatePoint(statepoint_path)
    try:
        tally = statepoint.get_tally(id=tally_id)
        return float(np.sum(tally.mean)), float(
            np.sqrt(np.sum(tally.std_dev**2))
        )
    finally:
        statepoint.close()


def _repository_sha() -> str:
    injected = os.environ.get("PARASTELL_GIT_SHA", "").strip()
    if injected:
        return injected
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "cannot resolve ParaStell provenance; set PARASTELL_GIT_SHA to "
            "the exact live commit in containerized runs"
        ) from exc
    return result.stdout.strip()


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
    settings.batches = ACTIVE_BATCHES
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
        max_particles=max((ACTIVE_BATCHES + 1) * particles, 1),
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
    settings.batches = ACTIVE_BATCHES
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
        surface_normal_signs={1: -1},
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
        max_particles=max(min(500, 2 * particles), 1),
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
    label_sign_mismatch = (
        ((labels == "incoming") & ~(mu < -1.0e-12))
        | ((labels == "outgoing") & ~(mu > 1.0e-12))
        | ((labels == "grazing") & ~(np.abs(mu) <= 1.0e-12))
    )
    return {
        "record_count": count,
        "incoming_count": incoming,
        "outgoing_count": int(np.count_nonzero(labels == "outgoing")),
        "grazing_count": int(np.count_nonzero(labels == "grazing")),
        "unknown_count": int(np.count_nonzero(labels == "unknown")),
        "incoming_fraction": incoming / count if count else 0.0,
        "label_sign_mismatch_count": int(
            np.count_nonzero(label_sign_mismatch)
        ),
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
    parser.add_argument("--magnet-particles", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--gate",
        choices=("all", "planar", "sector", "magnet"),
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
    elif args.gate == "magnet":
        result = run_real_magnet_gate(
            args.output_dir,
            assets_dir=args.assets_dir,
            particles=args.magnet_particles,
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
