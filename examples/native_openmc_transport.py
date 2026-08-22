"""Run reproducible OpenMC/DAGMC transport gates on the full native assembly."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import openmc
import openmc.lib


ANCHOR = np.asarray((734.10362797732, 196.7024742772802, 0.0))
AXIS = np.asarray((0.9659258262890683, 0.25881904510252074, 0.0))
REFERENCE = np.asarray((0.0, 0.0, 1.0))
RESOLVED_END = np.asarray((755.8949146183361, 202.54143193477557, 0.0))
EXTERNAL_TERMINATION = np.asarray((780.0430602755628, 209.0119080623386, 0.0))


def conda_package_version(name):
    records = sorted((Path(sys.prefix) / "conda-meta").glob(f"{name}-*.json"))
    if records:
        return json.loads(records[-1].read_text()).get("version")
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def materials(cross_sections):
    definitions = {
        "Vacuum": ("H1", 1.0e-20),
        "first_wall": ("Fe56", 7.8),
        "breeder": ("Li6", 1.0),
        "shield": ("Fe56", 7.8),
        "vacuum_vessel": ("Fe56", 7.8),
        "SS316L": ("Fe56", 7.8),
        "Copper": ("Cu63", 8.96),
        "Graveyard": ("H1", 1.0e-30),
    }
    result = openmc.Materials()
    result.cross_sections = str(cross_sections)
    for name, (nuclide, density) in definitions.items():
        material = openmc.Material(name=name)
        material.add_nuclide(nuclide, 1.0)
        material.set_density("g/cm3", density)
        result.append(material)
    return result


def source(position, direction=None):
    result = openmc.IndependentSource()
    result.space = openmc.stats.Point(tuple(position))
    result.angle = (
        openmc.stats.Isotropic()
        if direction is None
        else openmc.stats.Monodirectional(tuple(direction))
    )
    result.energy = openmc.stats.Discrete([14.1e6], [1.0])
    return result


def build_model(h5m, cross_sections, case, particles, seed):
    # This is the root geometry and its native GLOBAL_ID values are already
    # collision-free. Preserve them so DAGMC's on-surface hint and OpenMC's
    # overlap checker refer to the same shared-surface identifiers.
    dagmc = openmc.DAGMCUniverse(str(h5m), auto_geom_ids=False)
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = int(particles)
    settings.batches = 2
    settings.seed = int(seed)
    if case == "geometry_debug":
        settings.source = source(ANCHOR + AXIS, AXIS)
    elif case == "centerline":
        settings.source = source(ANCHOR - AXIS * 2.0, AXIS)
    elif case == "liner":
        settings.source = source(ANCHOR - AXIS * 2.0 + REFERENCE * 3.5, AXIS)
    elif case == "blanket":
        settings.source = source(ANCHOR - AXIS * 2.0 + REFERENCE * 6.0, AXIS)
    elif case == "sector":
        settings.source = source(ANCHOR - AXIS * 10.0)
    else:
        raise ValueError(case)

    mesh = openmc.RegularMesh(name="representative_downstream_mesh")
    mesh.dimension = (8, 8, 8)
    if case == "sector":
        mesh.lower_left = (250.0, -80.0, -480.0)
        mesh.upper_right = (1095.0, 665.0, 335.0)
    else:
        limits = np.vstack((RESOLVED_END, EXTERNAL_TERMINATION))
        mesh.lower_left = tuple(limits.min(axis=0) - 12.0)
        mesh.upper_right = tuple(limits.max(axis=0) + 12.0)
    tally = openmc.Tally(name="downstream_flux")
    tally.filters = [openmc.MeshFilter(mesh)]
    tally.scores = ["flux"]
    return openmc.Model(
        geometry=openmc.Geometry(dagmc),
        materials=materials(cross_sections),
        settings=settings,
        tallies=openmc.Tallies([tally]),
    )


def run_case(
    h5m,
    cross_sections,
    output_dir,
    case,
    particles,
    seed,
    reuse_existing=False,
):
    case_dir = output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    command = ["openmc"]
    if case == "geometry_debug":
        command.append("--geometry-debug")
    if reuse_existing:
        output = (case_dir / "openmc_output.txt").read_text()
        returncode = 0
    else:
        model = build_model(h5m, cross_sections, case, particles, seed)
        model.export_to_model_xml(case_dir / "model.xml")
        completed = subprocess.run(
            command,
            cwd=case_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OPENMC_CROSS_SECTIONS": str(cross_sections)},
            check=False,
        )
        output = completed.stdout
        returncode = completed.returncode
        (case_dir / "openmc_output.txt").write_text(output)
    if returncode:
        raise RuntimeError(
            f"OpenMC {case} gate failed with {returncode}:\n{output}"
        )
    navigation_errors = len(
        re.findall(
            r"lost particle|could not find the cell|dagmc.*(?:error|failure)",
            output,
            flags=re.IGNORECASE,
        )
    )
    if navigation_errors:
        raise RuntimeError(
            f"OpenMC {case} reported {navigation_errors} navigation errors"
        )
    statepoint_path = case_dir / "statepoint.2.h5"
    with openmc.StatePoint(statepoint_path) as statepoint:
        tally = statepoint.get_tally(name="downstream_flux")
        tally_sum = float(np.sum(tally.mean))
        tally_max = float(np.max(tally.mean))
        nonzero_bins = int(np.count_nonzero(tally.mean))
    if tally_sum <= 0.0 or nonzero_bins == 0:
        raise RuntimeError(f"OpenMC {case} produced no downstream tally")
    return {
        "case": case,
        "command": command,
        "particles": int(particles),
        "batches": 2,
        "histories": int(particles) * 2,
        "seed": int(seed),
        "returncode": returncode,
        "reused_completed_output": bool(reuse_existing),
        "geometry_debug": case == "geometry_debug",
        "lost_particle_count": 0,
        "dagmc_navigation_error_count": navigation_errors,
        "downstream_tally_sum": tally_sum,
        "downstream_tally_max": tally_max,
        "downstream_nonzero_bins": nonzero_bins,
        "output": str(case_dir / "openmc_output.txt"),
        "statepoint": str(statepoint_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5m", type=Path, required=True)
    parser.add_argument("--cross-sections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--sector-particles", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--library-provenance",
        default="NNDC HDF5 library bundled in openmc/openmc:latest",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for offset, case in enumerate(
        ("geometry_debug", "centerline", "liner", "blanket", "sector")
    ):
        particle_count = (
            args.sector_particles if case == "sector" else args.particles
        )
        results.append(
            run_case(
                args.h5m,
                args.cross_sections,
                args.output_dir,
                case,
                particle_count,
                args.seed + offset,
                reuse_existing=args.reuse_existing,
            )
        )
    report = {
        "openmc_version": openmc.__version__,
        "dagmc_enabled": bool(openmc.lib._dagmc_enabled()),
        "dagmc_version": conda_package_version("dagmc"),
        "pydagmc_version": conda_package_version("pydagmc"),
        "moab_version": conda_package_version("moab"),
        "pymoab_version": conda_package_version("pymoab"),
        "cross_sections": str(args.cross_sections),
        "cross_sections_sha256": sha256(
            args.cross_sections.read_bytes()
        ).hexdigest(),
        "cross_section_library_provenance": args.library_provenance,
        "h5m": str(args.h5m),
        "total_histories": sum(result["histories"] for result in results),
        "lost_particle_count": sum(
            result["lost_particle_count"] for result in results
        ),
        "cases": results,
    }
    path = args.output_dir / "openmc_transport_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
