"""Fail-closed contract for the authoritative continuous WISTELL-D model.

This module deliberately does not describe the alternate 18 swept-coil CAD.
The accepted global model is nine nested radial volumes; its final volume is a
continuous homogenized magnet layer.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping


SOURCE_MANIFEST_SCHEMA = "parastell.parametric_source_cad/v1.0.0"
SOURCE_AUDIT_SCHEMA = "parastell.continuous_radial_source_cad_audit/v1.0.0"
SOURCE_AUDIT_SEAL_SCHEMA = (
    "parastell.continuous_radial_source_cad_audit_seal/v1.0.0"
)
H5M_WRITEBACK_SCHEMA = (
    "parastell.continuous_radial_direct90_h5m_writeback/v1.0.0"
)
COMPONENT_ORDER = (
    "chamber",
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "vacuum_gap",
    "magnets",
)
MATERIAL_TAGS = (
    "Vacuum",
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "Vacuum",
    "homogenized_magnet",
)
ADJACENT_ROLE_PAIRS = tuple(zip(COMPONENT_ORDER, COMPONENT_ORDER[1:]))
NONADJACENT_ROLE_PAIRS = tuple(
    (left, right)
    for index, left in enumerate(COMPONENT_ORDER)
    for right in COMPONENT_ORDER[index + 2 :]
)
COUPLING_INTERFACE = "continuous_magnet_layer_inner_boundary"
MAGNET_ROLE = COMPONENT_ORDER[-1]
GAP_ROLE = COMPONENT_ORDER[-2]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject any source packet that is not the nine-volume global model."""
    plan = manifest.get("plan", {})
    safety = plan.get("safety", {})
    expected_layers = list(COMPONENT_ORDER[1:])
    if (
        manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("status") != "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING"
        or manifest.get("transport_eligible") is not False
        or plan.get("device") != "WISTELL-D"
        or float(plan.get("extent_degrees", math.nan)) != 90.0
        or float(plan.get("field_period_degrees", math.nan)) != 90.0
        or plan.get("grid_shape") != [80, 90]
        or plan.get("magnet_representation") != "radial_envelope"
        or plan.get("radial_build_mode") != "isolated_cumulative_shells"
        or plan.get("component_order") != list(COMPONENT_ORDER)
        or tuple(manifest.get("actual_invessel_component_order", ()))
        != COMPONENT_ORDER
        or manifest.get("radial_build_mode") != "isolated_cumulative_shells"
        or safety.get("ports") is not False
        or safety.get("post_build_transforms") is not False
        or safety.get("imported_physical_solids") is not False
    ):
        raise ValueError(
            "source manifest is not the authoritative direct-90 radial envelope"
        )
    vmec = manifest.get("live_vmec_metadata", {})
    if (
        vmec.get("pass") is not True
        or vmec.get("n_field_periods") != 4
        or vmec.get("stellarator_symmetric") is not True
        or vmec.get("lasym_vmec_logical") is not False
    ):
        raise ValueError("live WISTELL-D VMEC identity is incomplete")
    inventory = manifest.get("magnet_inventory", {})
    if (
        inventory.get("representation") != "radial_envelope"
        or inventory.get("material") != "homogenized_magnet"
        or "magnets" in inventory
        or "coil_count" in inventory
        or "solid_count" in inventory
    ):
        raise ValueError("global magnet is not one continuous radial layer")
    evidence = manifest.get("component_evidence", {})
    if tuple(evidence) != COMPONENT_ORDER or any(
        row.get("brep_valid") is not True
        or row.get("solid_count") != 1
        or not math.isfinite(float(row.get("volume_cm3", math.nan)))
        or float(row["volume_cm3"]) <= 0.0
        for row in evidence.values()
    ):
        raise ValueError("nine-volume source component evidence is invalid")
    stages = manifest.get("radial_build_construction_stages", ())
    if len(stages) != len(expected_layers):
        raise ValueError(
            "isolated cumulative construction-stage count is invalid"
        )
    for index, (stage, component) in enumerate(zip(stages, expected_layers)):
        if (
            stage.get("stage_index") != index
            or stage.get("component") != component
            or not all(
                isinstance(stage.get(key), str) and len(stage[key]) == 64
                for key in (
                    "cumulative_inner_matrix_sha256",
                    "layer_matrix_sha256",
                    "cumulative_outer_matrix_sha256",
                )
            )
            or (
                index
                and stage.get("cumulative_inner_matrix_sha256")
                != stages[index - 1].get("cumulative_outer_matrix_sha256")
            )
        ):
            raise ValueError("cumulative radial construction proof is broken")


def expected_step_names() -> tuple[str, ...]:
    return tuple(f"{name}.step" for name in COMPONENT_ORDER)


def validate_source_artifacts(
    source_root: str | Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    root = Path(source_root)
    artifacts = manifest.get("artifacts", {})
    if set(path.name for path in root.glob("*.step")) != set(
        expected_step_names()
    ):
        raise ValueError(
            "source STEP inventory is not exactly nine radial volumes"
        )
    if "magnet_set.step" in artifacts:
        raise ValueError("swept-coil STEP is forbidden in the continuous lane")
    rows = []
    for name, expected in sorted(artifacts.items()):
        path = root / name
        size = path.stat().st_size if path.is_file() else None
        digest = sha256_file(path) if path.is_file() else None
        passed = size == expected.get("bytes") and digest == expected.get(
            "sha256"
        )
        rows.append(
            {
                "path": name,
                "expected_bytes": expected.get("bytes"),
                "actual_bytes": size,
                "expected_sha256": expected.get("sha256"),
                "actual_sha256": digest,
                "pass": passed,
            }
        )
    if not rows or not all(row["pass"] for row in rows):
        raise ValueError("source artifact integrity failed")
    return rows


def radial_separation_proof(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return construction proofs for every nonadjacent radial pair.

    The exact Boolean checks are reserved for the eight touching adjacent
    pairs. Strictly positive resolved layers and the chained cumulative
    boundaries prove that a nonadjacent shell has at least one intervening
    positive-thickness shell.
    """
    resolved = manifest.get("resolved_layers", {})
    for component in COMPONENT_ORDER[1:]:
        row = resolved.get(component, {})
        if (
            row.get("shape") != [80, 90]
            or row.get("finite") is not True
            or row.get("strictly_positive") is not True
            or row.get("poloidally_closed") is not True
            or float(row.get("minimum_cm", math.nan)) <= 0.0
        ):
            raise ValueError("resolved positive-thickness proof is invalid")
    return [
        {
            "components": [left, right],
            "intervening_components": list(
                COMPONENT_ORDER[
                    COMPONENT_ORDER.index(left)
                    + 1 : COMPONENT_ORDER.index(right)
                ]
            ),
            "proof": "strictly_positive_ordered_radial_stack_and_nested_shell_construction",
            "exact_boolean_required": False,
            "pass": True,
        }
        for left, right in NONADJACENT_ROLE_PAIRS
    ]
