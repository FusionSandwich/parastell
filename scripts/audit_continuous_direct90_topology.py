#!/usr/bin/env python3
"""Create native/PyDAGMC topology evidence for the continuous magnet layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from parastell.continuous_radial_contract import COMPONENT_ORDER
from parastell.continuous_radial_contract import H5M_WRITEBACK_SCHEMA
from parastell.continuous_radial_contract import sha256_file
from parastell.dagmc_qualification import audit_dagmc_topology


EXPECTED_MATERIAL_COUNTS = {
    "Vacuum": 2,
    "back_wall": 1,
    "breeder": 1,
    "first_wall": 1,
    "high_temperature_shield": 1,
    "homogenized_magnet": 1,
    "low_temperature_shield": 1,
    "vacuum_vessel": 1,
}


def audit(
    dagmc: Path,
    output: Path,
    writeback_path: Path,
    *,
    expected_dagmc_sha256: str,
    expected_writeback_sha256: str,
    vector_area_relative_tolerance: float = 1.0e-8,
) -> dict:
    dagmc = dagmc.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only output exists: {output}")
    if sha256_file(dagmc) != expected_dagmc_sha256:
        raise ValueError("DAGMC H5M hash mismatch")
    if sha256_file(writeback_path) != expected_writeback_sha256:
        raise ValueError("H5M writeback hash mismatch")
    writeback = json.loads(writeback_path.read_text(encoding="utf-8"))
    if (
        writeback.get("schema") != H5M_WRITEBACK_SCHEMA
        or writeback.get("pass") is not True
        or writeback.get("h5m_sha256") != expected_dagmc_sha256
        or writeback.get("semantic_roles_by_global_volume_id")
        != list(COMPONENT_ORDER)
        or writeback.get("continuous_magnet_volume_global_id") != 9
    ):
        raise ValueError("continuous H5M writeback is invalid")
    topology = audit_dagmc_topology(
        dagmc,
        magnet_material="homogenized_magnet",
        expected_magnet_count=1,
        expected_material_counts=EXPECTED_MATERIAL_COUNTS,
        vector_area_relative_tolerance=vector_area_relative_tolerance,
    )
    magnet = next(
        (
            row
            for row in topology.get("volume_envelopes", ())
            if int(row.get("volume_id", -1)) == 9
        ),
        None,
    )
    passed = bool(
        topology.get("topology_gate_pass") is True
        and topology.get("h5m_unchanged") is True
        and topology.get("magnet_count") == 1
        and magnet
        and magnet.get("closed") is True
    )
    inventory = {
        "schema": "parastell.continuous_magnet_volume_inventory/v1.0.0",
        "status": "PASS" if passed else "BLOCKED_NATIVE_TOPOLOGY",
        "raw_h5m_sha256": expected_dagmc_sha256,
        "h5m_unchanged": topology.get("h5m_unchanged"),
        "volume_count": len(topology.get("volume_envelopes", ())),
        "component_order": list(COMPONENT_ORDER),
        "continuous_magnet": (
            {
                "volume_id": 9,
                "material_tag": magnet.get("material_tag"),
                "complete_boundary_surface_ids": magnet.get("surface_ids"),
                "complete_boundary_closed": magnet.get("closed"),
                "bounding_box_cm": magnet.get("bounding_box_cm"),
                "volume_cm3": magnet.get("volume_cm3"),
                "vector_area_closure_relative": magnet.get(
                    "vector_area_closure_relative"
                ),
            }
            if magnet
            else None
        ),
        "physical_h5m_mutation": False,
    }
    output.mkdir(parents=False)
    (output / "DAGMC_TOPOLOGY_AUDIT.json").write_text(
        json.dumps(topology, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "CONTINUOUS_MAGNET_VOLUME_INVENTORY.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("writeback", type=Path)
    parser.add_argument("--expected-dagmc-sha256", required=True)
    parser.add_argument("--expected-writeback-sha256", required=True)
    args = parser.parse_args()
    result = audit(
        args.dagmc,
        args.output,
        args.writeback,
        expected_dagmc_sha256=args.expected_dagmc_sha256,
        expected_writeback_sha256=args.expected_writeback_sha256,
    )
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
