"""Write create-only DAGMC topology and original-magnet inventories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

from parastell.dagmc_qualification import audit_dagmc_topology
from parastell.reference_geometry import sha256_file


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _observed_hash(path: Path) -> tuple[str | None, str | None]:
    try:
        return sha256_file(path), None
    except Exception as error:  # Preserve evidence even for unreadable input.
        return None, f"{type(error).__name__}: {error}"


def _failed_inventory(
    *,
    dagmc_sha256: str | None,
    criteria_sha256: str | None,
    stage: str,
    error: str,
) -> dict[str, Any]:
    return {
        "schema": "parastell.original_magnet_envelopes/v1.1.0",
        "raw_h5m_sha256": dagmc_sha256,
        "acceptance_criteria_sha256": criteria_sha256,
        "magnet_count": 0,
        "all_envelopes_close": False,
        "magnets": [],
        "audit_stage": stage,
        "audit_exception": error,
        "inventory_gate_pass": False,
    }


def _write_failure_receipts(
    output: Path,
    *,
    dagmc: Path,
    expected_dagmc_sha256: str,
    observed_dagmc_sha256_before: str | None,
    criteria_path: Path,
    expected_criteria_sha256: str,
    criteria_sha256: str | None,
    stage: str,
    error: Exception,
    topology: dict[str, Any] | None,
) -> None:
    after, after_error = _observed_hash(dagmc)
    message = f"{type(error).__name__}: {error}"
    failure = {
        "schema": "parastell.dagmc_topology_audit/v1.1.0",
        "dagmc_path": str(dagmc),
        "expected_raw_h5m_sha256": expected_dagmc_sha256,
        "raw_h5m_sha256_before": observed_dagmc_sha256_before,
        "raw_h5m_sha256_after": after,
        "raw_h5m_after_hash_error": after_error,
        "h5m_unchanged": (
            after == observed_dagmc_sha256_before
            if after is not None and observed_dagmc_sha256_before is not None
            else None
        ),
        "acceptance_criteria_path": str(criteria_path),
        "expected_acceptance_criteria_sha256": expected_criteria_sha256,
        "acceptance_criteria_sha256": criteria_sha256,
        "audit_stage": stage,
        "audit_exception": message,
        # None means the exception occurred before this fact was established.
        "pydagmc_load_attempted": (
            topology.get("pydagmc_load_attempted")
            if isinstance(topology, dict)
            else None
        ),
        "volume_envelopes": [],
        "magnet_count": 0,
        "magnet_volume_ids": [],
        "all_magnet_envelopes_close": False,
        "topology_gate_pass": False,
    }
    inventory = _failed_inventory(
        dagmc_sha256=after,
        criteria_sha256=criteria_sha256,
        stage=stage,
        error=message,
    )
    (output / "dagmc_topology_audit.json").write_text(
        _json_text(failure), encoding="utf-8"
    )
    (output / "original_magnet_envelope_inventory.json").write_text(
        _json_text(inventory), encoding="utf-8"
    )


def _magnet_inventory(
    topology: dict[str, Any],
    *,
    dagmc_sha256: str,
    criteria_sha256: str,
) -> dict[str, Any]:
    magnets = [
        row
        for row in topology.get("volume_envelopes", [])
        if row["material_tag"].strip().lower() == "magnets"
    ]

    def order(row: dict[str, Any]) -> tuple[float, float, float, float]:
        lower = row["bounding_box_cm"]["lower"]
        upper = row["bounding_box_cm"]["upper"]
        center = [(left + right) / 2.0 for left, right in zip(lower, upper)]
        return (
            math.atan2(center[1], center[0]) % (2.0 * math.pi),
            center[2],
            center[0],
            center[1],
        )

    magnets.sort(key=order)
    all_close = len(magnets) == 18 and all(row["closed"] for row in magnets)
    inventory_gate_pass = (
        topology.get("topology_gate_pass") is True and all_close
    )
    return {
        "schema": "parastell.original_magnet_envelopes/v1.1.0",
        "raw_h5m_sha256": dagmc_sha256,
        "acceptance_criteria_sha256": criteria_sha256,
        "magnet_count": len(magnets),
        "all_envelopes_close": all_close,
        "inventory_gate_pass": inventory_gate_pass,
        "magnets": [
            {
                "magnet_id": f"magnet-{index:04d}",
                "volume_id": row["volume_id"],
                "material_tag": row["material_tag"],
                "component_name": row["component_name"],
                "outer_envelope_surface_ids": row["surface_ids"],
                "surface_count": len(row["surface_ids"]),
                "bounding_box_cm": row["bounding_box_cm"],
                "volume_cm3": row["volume_cm3"],
                "signed_volume_cm3": row["signed_volume_cm3"],
                "edge_multiplicity_error_count": row[
                    "edge_multiplicity_error_count"
                ],
                "directed_edge_orientation_error_count": row[
                    "directed_edge_orientation_error_count"
                ],
                "vector_area_closure_relative": row[
                    "vector_area_closure_relative"
                ],
                "degenerate_triangle_count": row["degenerate_triangle_count"],
                "closed": row["closed"],
                "physical_role": "original_homogenized_magnet_solid",
                "coupling_interface": "homogenized_magnet_outer",
                "casing_winding_split": False,
            }
            for index, row in enumerate(magnets)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-dagmc-sha256", required=True)
    parser.add_argument("--acceptance-criteria", required=True, type=Path)
    parser.add_argument("--expected-acceptance-criteria-sha256", required=True)
    arguments = parser.parse_args()

    dagmc = arguments.dagmc.resolve()
    output = arguments.output_dir.resolve()
    criteria_path = arguments.acceptance_criteria.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    stage = "dagmc_input_hash"
    dagmc_sha256: str | None = None
    criteria_sha256: str | None = None
    topology: dict[str, Any] | None = None
    try:
        dagmc_sha256 = sha256_file(dagmc)
        if dagmc_sha256 != arguments.expected_dagmc_sha256:
            raise ValueError("DAGMC H5M hash mismatch")

        stage = "acceptance_criteria_hash"
        criteria_sha256 = sha256_file(criteria_path)
        if criteria_sha256 != arguments.expected_acceptance_criteria_sha256:
            raise ValueError("acceptance-criteria hash mismatch")

        stage = "acceptance_criteria_parse"
        criteria = json.loads(criteria_path.read_text(encoding="utf-8"))

        stage = "native_and_pydagmc_topology_audit"
        topology = audit_dagmc_topology(
            dagmc,
            expected_magnet_count=18,
            vector_area_relative_tolerance=float(
                criteria["magnet_envelopes"]["vector_area_relative_tolerance"]
            ),
        )
        topology["acceptance_criteria_path"] = str(criteria_path)
        topology["acceptance_criteria_sha256"] = criteria_sha256

        stage = "magnet_inventory"
        inventory = _magnet_inventory(
            topology,
            dagmc_sha256=arguments.expected_dagmc_sha256,
            criteria_sha256=criteria_sha256,
        )

        stage = "finite_json_serialization"
        topology_text = _json_text(topology)
        inventory_text = _json_text(inventory)

        stage = "receipt_write"
        (output / "dagmc_topology_audit.json").write_text(
            topology_text, encoding="utf-8"
        )
        (output / "original_magnet_envelope_inventory.json").write_text(
            inventory_text, encoding="utf-8"
        )
    except Exception as error:
        _write_failure_receipts(
            output,
            dagmc=dagmc,
            expected_dagmc_sha256=arguments.expected_dagmc_sha256,
            observed_dagmc_sha256_before=dagmc_sha256,
            criteria_path=criteria_path,
            expected_criteria_sha256=(
                arguments.expected_acceptance_criteria_sha256
            ),
            criteria_sha256=criteria_sha256,
            stage=stage,
            error=error,
            topology=topology,
        )
        return 1

    return (
        0
        if topology.get("topology_gate_pass") is True
        and inventory.get("inventory_gate_pass") is True
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
