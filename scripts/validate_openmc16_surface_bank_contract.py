"""Create and verify an exact OpenMC 0.16 surface-bank phase-space fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import openmc

from parastell.surface_source_phase_space import read_openmc16_surface_sources


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    expected = [
        openmc.SourceParticle(
            r=(1.25, -2.5, 3.75),
            u=(1.0, 0.0, 0.0),
            E=14.1e6,
            time=1.0e-9,
            wgt=1.0,
            delayed_group=0,
            surf_id=17,
            particle="neutron",
        ),
        openmc.SourceParticle(
            r=(-4.5, 5.25, 6.0),
            u=(0.0, 1.0, 0.0),
            E=2.2e6,
            time=3.0e-9,
            wgt=0.125,
            delayed_group=0,
            surf_id=18,
            particle="photon",
        ),
    ]
    bank_path = output / "openmc16_surface_bank.h5"
    openmc.write_source_file(expected, bank_path)
    columns, manifest = read_openmc16_surface_sources(
        [bank_path],
        source_histories=200,
        history_binding={
            "kind": "serializer_fixture",
            "fixture_id": "openmc16-native-sourceparticle-roundtrip-v1",
            "source_histories": 200,
        },
        requested_surface_ids=[17, 18],
    )
    checks = {
        "position": np.allclose(
            columns["position_global_cm"],
            [[1.25, -2.5, 3.75], [-4.5, 5.25, 6.0]],
            rtol=0.0,
            atol=0.0,
        ),
        "direction": np.allclose(
            columns["direction_global"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            rtol=0.0,
            atol=0.0,
        ),
        "energy": np.array_equal(columns["energy_eV"], [14.1e6, 2.2e6]),
        "time": np.array_equal(columns["time_s"], [1.0e-9, 3.0e-9]),
        "openmc_weight": np.array_equal(
            columns["openmc_weight"], [1.0, 0.125]
        ),
        "normalized_weight": np.array_equal(
            columns["weight_per_source_history"], [0.005, 0.000625]
        ),
        "surface": np.array_equal(columns["surface_id"], [17, 18]),
        "particle": np.array_equal(columns["particle_pdg"], [2112, 22]),
        "delayed_group": np.array_equal(columns["delayed_group"], [0, 0]),
    }
    receipt = {
        "schema": "parastell.openmc16_surface_bank_runtime_validation/v1.0.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "openmc_version": openmc.__version__,
        "bank_path": str(bank_path),
        "bank_sha256": _hash(bank_path),
        "checks": checks,
        "phase_space_manifest": manifest,
        "transport_histories": 0,
        "note": "serializer/reader contract only; not a transport or geometry result",
    }
    receipt_path = output / "OPENMC16_SURFACE_BANK_RUNTIME_VALIDATION.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["status"] != "PASS":
        raise RuntimeError(
            "OpenMC 0.16 surface-bank contract validation failed"
        )
    print(receipt_path)


if __name__ == "__main__":
    main()
