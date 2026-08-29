"""Create a synthetic, non-production ALARA workflow smoke package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from parastell.activation_campaign import build_activation_campaign
from parastell.alara_activation import build_alara_handoff, write_alara_package
from parastell.energy_groups import (
    VITAMIN_J_175_EDGES_EV,
    VITAMIN_J_175_EDGES_SHA256,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    provenance = {
        "raw_h5m_sha256": _digest("synthetic-no-h5m"),
        "canonical_geometry_fingerprint": _digest("synthetic-no-geometry"),
        "statepoint_sha256": _digest("synthetic-no-statepoint"),
        "nuclear_data_sha256": _digest("synthetic-no-openmc-data"),
        "physical_source_rate_per_s": 1.0e20,
        "source_rate_scope": "synthetic_one_zone_alara_smoke",
        "normalization": "per_source_history",
        "statistics_classification": "WORKFLOW_SMOKE_ONLY",
        "provenance_origin": "SYNTHETIC_WORKFLOW_FIXTURE",
    }
    material = {
        "material_id": "synthetic-pure-cu63",
        "composition_sha256": _digest("synthetic-pure-cu63"),
        "alara_constituents": [
            {
                "element_symbol": "cu:63",
                "relative_density": 1.0,
                "volume_fraction": 1.0,
            }
        ],
    }
    handoff = build_alara_handoff(
        activation_rows=[
            {
                "domain_id": "synthetic-cu63-zone",
                "volume_cm3": 30.0,
                "energy_edges_eV": list(VITAMIN_J_175_EDGES_EV),
                "bin_integrated_flux_per_source": [1.0e-8] * 175,
                "physical_source_rate_per_s": 1.0e20,
                "material": material,
            }
        ],
        campaign=build_activation_campaign(),
        group_structure_sha256=VITAMIN_J_175_EDGES_SHA256,
        openmc_provenance=provenance,
    )
    paths = write_alara_package(args.output, handoff)
    print(
        json.dumps(
            {
                "status": "READY_FOR_BOUNDED_EXECUTION",
                "claim": "WORKFLOW_SMOKE_ONLY",
                "output": str(args.output.resolve()),
                "file_count": len(paths),
                "group_structure_sha256": VITAMIN_J_175_EDGES_SHA256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
