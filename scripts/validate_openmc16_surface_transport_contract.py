"""Bounded OpenMC 0.16 transport proof for surface-bank/tally integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import openmc

from parastell.surface_source_phase_space import read_openmc16_surface_sources


HISTORIES = 2_000
SEED = 7_104_291
SURFACE_ID = 17


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _statepoint_directional_current(path: Path) -> tuple[float, float]:
    with h5py.File(path, "r") as statepoint:
        tallies = statepoint["tallies"]
        tally = next(
            group
            for name, group in tallies.items()
            if name.startswith("tally ")
            and group["name"][()].decode() == "contract_directional_current"
        )
        results = np.asarray(tally["results"])
        realizations = int(tally["n_realizations"][()])
        means = results[..., 0].reshape(-1) / realizations
    if means.shape != (2,):
        raise RuntimeError(f"unexpected directional tally shape {means.shape}")
    return float(abs(means[0])), float(abs(means[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if openmc.__version__ != "0.16.0":
        raise RuntimeError(
            f"expected OpenMC 0.16.0, found {openmc.__version__}"
        )
    boundary = openmc.Sphere(
        r=10.0,
        surface_id=SURFACE_ID,
        boundary_type="vacuum",
    )
    cell = openmc.Cell(cell_id=31, region=-boundary)
    model = openmc.Model(geometry=openmc.Geometry([cell]))
    model.settings.run_mode = "fixed source"
    model.settings.particles = HISTORIES
    model.settings.batches = 1
    model.settings.seed = SEED
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0.0, 0.0, 0.0))
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([14.1e6], [1.0])
    source.particle = "neutron"
    model.settings.source = [source]
    model.settings.surf_source_write = {
        "surface_ids": [SURFACE_ID],
        "max_particles": 2 * HISTORIES,
        "max_source_files": 1,
    }
    tally = openmc.Tally(name="contract_directional_current")
    tally.filters = [
        openmc.SurfaceFilter([SURFACE_ID]),
        openmc.MuSurfaceFilter([-1.0, 0.0, 1.0]),
        openmc.ParticleFilter(["neutron"]),
        openmc.EnergyFilter([0.0, 20.0e6]),
    ]
    tally.scores = ["current"]
    model.tallies = openmc.Tallies([tally])
    model.run(cwd=output, threads=1)

    bank_path = output / "surface_source.h5"
    statepoint_path = output / "statepoint.1.h5"
    model_path = output / "model.xml"
    columns, phase_manifest = read_openmc16_surface_sources(
        [bank_path],
        source_histories=HISTORIES,
        history_binding={
            "kind": "fixed_source_run",
            "run_id": "openmc16-vacuum-sphere-contract-v1",
            "settings_payload_path": str(model_path),
            "settings_payload_sha256": _hash(model_path),
            "statepoint_path": str(statepoint_path),
            "statepoint_sha256": _hash(statepoint_path),
        },
        requested_surface_ids=[SURFACE_ID],
    )
    incoming_tally, outgoing_tally = _statepoint_directional_current(
        statepoint_path
    )
    outgoing_bank = float(np.sum(columns["weight_per_source_history"]))
    direction_norms = np.linalg.norm(columns["direction_global"], axis=1)
    radial_normals = (
        columns["position_global_cm"]
        / np.linalg.norm(columns["position_global_cm"], axis=1)[:, None]
    )
    mu = np.sum(columns["direction_global"] * radial_normals, axis=1)
    checks = {
        "one_crossing_per_history": len(columns["surface_id"]) == HISTORIES,
        "all_requested_surface": bool(
            np.all(columns["surface_id"] == SURFACE_ID)
        ),
        "all_neutrons": bool(np.all(columns["particle_pdg"] == 2112)),
        "unit_directions": bool(
            np.allclose(direction_norms, 1.0, rtol=0.0, atol=1.0e-12)
        ),
        "outward_mu": bool(np.all(mu > 1.0 - 1.0e-10)),
        "incoming_tally_zero": abs(incoming_tally) <= 1.0e-14,
        "outgoing_tally_one": abs(outgoing_tally - 1.0) <= 1.0e-14,
        "bank_tally_equality": abs(outgoing_bank - outgoing_tally) <= 1.0e-14,
        "capacity_not_reached": len(columns["surface_id"]) < 2 * HISTORIES,
    }
    receipt = {
        "schema": "parastell.openmc16_surface_transport_contract/v1.0.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "openmc_version": openmc.__version__,
        "histories": HISTORIES,
        "seed": SEED,
        "surface_id": SURFACE_ID,
        "surface_source_configuration": model.settings.surf_source_write,
        "record_count": int(len(columns["surface_id"])),
        "bank_integrated_current_per_source": outgoing_bank,
        "tally_incoming_current_per_source": incoming_tally,
        "tally_outgoing_current_per_source": outgoing_tally,
        "checks": checks,
        "phase_space_manifest": phase_manifest,
        "bank_sha256": _hash(bank_path),
        "statepoint_sha256": _hash(statepoint_path),
        "scope": "bounded vacuum-sphere transport mechanism proof",
        "scientific_geometry_claim": "NOT_PERFORMED",
    }
    receipt_path = output / "OPENMC16_SURFACE_TRANSPORT_CONTRACT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["status"] != "PASS":
        raise RuntimeError(f"surface transport contract failed: {checks}")
    print(receipt_path)


if __name__ == "__main__":
    main()
