"""Run a bounded, geometry-immutable OpenMC 0.16 surface qualification.

The input model supplies the physical source, materials, and CSG wrapper.  This
script changes only the DAGMC filename, bounded run controls, and diagnostic
surface instrumentation.  The H5M itself is opened read-only by OpenMC and is
never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import openmc

from parastell.surface_source_instrumentation import (
    apply_openmc16_surface_instrumentation,
    build_surface_instrumentation_spec,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _surface_signs(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    magnets = payload.get("magnets")
    if not isinstance(magnets, list) or not magnets:
        raise ValueError("surface manifest has no magnets")
    signs: dict[int, int] = {}
    for magnet in magnets:
        for surface in magnet.get("surfaces", ()):
            surface_id = int(surface["dagmc_surface_id"])
            sign = int(surface["magnet_outward_normal_multiplier"])
            if sign not in {-1, 1}:
                raise ValueError("surface normal multiplier must be -1 or +1")
            if surface_id in signs and signs[surface_id] != sign:
                raise ValueError(
                    "surface manifest has conflicting normal signs"
                )
            signs[surface_id] = sign
    declared = sorted(int(value) for value in payload["all_surface_ids"])
    if declared != sorted(signs):
        raise ValueError("surface manifest IDs disagree with magnet surfaces")
    return signs


def _replace_dagmc_filename(model: openmc.Model, dagmc_path: Path) -> int:
    universes = model.geometry.get_all_universes()
    dagmc_universes = [
        universe
        for universe in universes.values()
        if isinstance(universe, openmc.DAGMCUniverse)
    ]
    if len(dagmc_universes) != 1:
        raise ValueError(
            "qualification model must contain exactly one DAGMC universe"
        )
    dagmc_universes[0].filename = str(dagmc_path)
    return int(dagmc_universes[0].id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_xml", type=Path)
    parser.add_argument("dagmc_h5m", type=Path)
    parser.add_argument("surface_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8_280_731)
    parser.add_argument("--max-particles", type=int, default=100_000)
    args = parser.parse_args()

    if openmc.__version__ != "0.16.0":
        raise RuntimeError(
            f"expected OpenMC 0.16.0, found {openmc.__version__}"
        )
    model_xml = args.model_xml.resolve(strict=True)
    dagmc_path = args.dagmc_h5m.resolve(strict=True)
    surface_manifest = args.surface_manifest.resolve(strict=True)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if min(args.particles, args.batches, args.seed, args.max_particles) <= 0:
        raise ValueError("run controls must be positive integers")

    signs = _surface_signs(surface_manifest)
    model = openmc.Model.from_model_xml(model_xml)
    dagmc_universe_id = _replace_dagmc_filename(model, dagmc_path)
    model.settings.run_mode = "fixed source"
    model.settings.particles = args.particles
    model.settings.batches = args.batches
    model.settings.seed = args.seed
    model.settings.output = {"tallies": False}
    model.tallies = openmc.Tallies()
    instrumentation = build_surface_instrumentation_spec(
        surface_ids=sorted(signs),
        energy_edges_by_particle={
            "neutron": [0.0, 0.5, 1.0e5, 1.0e6, 20.0e6],
            "photon": [0.0, 1.0e5, 1.0e6, 20.0e6],
        },
        openmc_normal_sign_by_surface=signs,
        max_particles_per_process=args.max_particles,
        max_source_files=1,
        mpi_ranks=1,
        coupling_interface="homogenized_magnet_outer_boundary",
    )
    tally_names = apply_openmc16_surface_instrumentation(
        model, instrumentation
    )
    model_path = output / "model.xml"
    model.export_to_model_xml(model_path)

    log_path = output / "openmc.log"
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            ["openmc", "-s", "1"],
            cwd=output,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    receipt = {
        "schema": "parastell.bounded_surface_qualification_run/v1.0.0",
        "status": (
            "TRANSPORT_PASS" if completed.returncode == 0 else "TRANSPORT_FAIL"
        ),
        "claim": "BOUNDED_NONPRODUCTION",
        "openmc_version": openmc.__version__,
        "return_code": int(completed.returncode),
        "histories": int(args.particles * args.batches),
        "particles_per_batch": int(args.particles),
        "batches": int(args.batches),
        "seed": int(args.seed),
        "dagmc_universe_id": dagmc_universe_id,
        "dagmc": {"path": str(dagmc_path), "sha256": _sha256(dagmc_path)},
        "input_model_xml": {
            "path": str(model_xml),
            "sha256": _sha256(model_xml),
        },
        "surface_manifest": {
            "path": str(surface_manifest),
            "sha256": _sha256(surface_manifest),
        },
        "output_model_xml_sha256": _sha256(model_path),
        "openmc_log_sha256": _sha256(log_path),
        "instrumentation": instrumentation,
        "tally_names": list(tally_names),
        "physical_h5m_mutation": False,
    }
    receipt_path = output / "TRANSPORT_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"OpenMC failed; see {log_path}")
    print(receipt_path)


if __name__ == "__main__":
    main()
