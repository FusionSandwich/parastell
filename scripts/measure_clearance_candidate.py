"""Measure public-reference clearance and derive local candidate matrices."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cadquery as cq
import numpy as np

from parastell import invessel_build as ivb
from parastell.cad_radial_clearance import measure_first_wall_to_magnets
from parastell.cad_radial_clearance import (
    project_dense_deficit_to_control_grid,
)
from parastell.public_geometry_candidate import POLOIDAL_ANGLES_DEG
from parastell.public_geometry_candidate import TOROIDAL_ANGLES_DEG
from parastell.public_geometry_candidate import WALL_S
from parastell.public_geometry_candidate import public_reference_radial_build
from parastell.public_geometry_candidate import thickness_matrices
from parastell.pystell import read_vmec
from parastell.radial_build_constraints import constrain_radial_build
from parastell.radial_build_constraints import (
    eliminate_interpolated_layer_increase,
)
from parastell.reference_geometry import sha256_file


def _json_array(values: np.ndarray):
    return [
        [_json_value(value) for value in row] for row in np.asarray(values)
    ]


def _json_value(value: float):
    value = float(value)
    return value if np.isfinite(value) else None


def _policy_result(
    original: dict[str, np.ndarray],
    projected_deficit: np.ndarray,
    *,
    clearance_cm: float,
    reduction_order: tuple[str, ...],
    minima: dict[str, float],
) -> dict:
    total = np.sum(np.stack(list(original.values())), axis=0)
    effective_available = total + clearance_cm - projected_deficit
    try:
        result = constrain_radial_build(
            original,
            effective_available,
            clearance_cm=clearance_cm,
            reduction_order=reduction_order,
            minimum_thickness_cm=minima,
        )
        corrected, interpolation_refinement = (
            eliminate_interpolated_layer_increase(
                original,
                result.corrected,
                layer="breeder",
                toroidal_control_angles_deg=TOROIDAL_ANGLES_DEG,
                poloidal_control_angles_deg=POLOIDAL_ANGLES_DEG,
                minimum_thickness_cm=minima["breeder"],
            )
        )
    except ValueError as error:
        return {
            "status": "INFEASIBLE",
            "reason": str(error),
            "reduction_order": list(reduction_order),
            "minimum_thickness_cm": minima,
        }
    return {
        "status": "MATRICES_DERIVED_PENDING_SOURCE_CAD_BUILD",
        "reduction_order": list(reduction_order),
        "minimum_thickness_cm": minima,
        "summary": result.summary(),
        "interpolation_refinement": interpolation_refinement,
        "corrected_thickness_matrices_cm": {
            name: _json_array(values) for name, values in corrected.items()
        },
        "reduction_matrices_cm": {
            name: _json_array(original[name] - values)
            for name, values in corrected.items()
        },
    }


def measure(
    vmec_path: Path,
    magnet_step_path: Path,
    output_dir: Path,
    *,
    clearance_cm: float,
    dense_samples: int,
    tessellation_tolerance_cm: float,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"create-only output already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    vmec = read_vmec.VMECData(str(vmec_path))
    ref_surf = ivb.VMECSurface(vmec)
    magnet_workplane = cq.importers.importStep(str(magnet_step_path))
    magnet_solids = magnet_workplane.solids().vals()
    if len(magnet_solids) != 18:
        raise RuntimeError(
            f"expected 18 source-CAD magnet solids, got {len(magnet_solids)}"
        )

    original_build = public_reference_radial_build()
    dense_build = ivb.InVesselBuild(
        ref_surf,
        ivb.RadialBuild(
            TOROIDAL_ANGLES_DEG,
            POLOIDAL_ANGLES_DEG,
            WALL_S,
            original_build,
        ),
        num_ribs=dense_samples,
        num_rib_pts=dense_samples,
        scale=100.0,
    )
    dense_build.populate_surfaces()
    dense_phi = np.rad2deg(dense_build._toroidal_angles_exp)
    dense_theta = np.rad2deg(dense_build._poloidal_angles_exp)
    measurement = measure_first_wall_to_magnets(
        ref_surf,
        dense_phi,
        dense_theta,
        WALL_S,
        magnet_solids,
        tessellation_tolerance_cm=tessellation_tolerance_cm,
    )
    dense_total = dense_build.Surfaces["vacuum_vessel"].offset_mat
    distances = measurement["distances_cm"]
    dense_deficit = np.maximum(dense_total + clearance_cm - distances, 0.0)
    dense_deficit[~np.isfinite(dense_deficit)] = 0.0
    projected_deficit = project_dense_deficit_to_control_grid(
        dense_deficit,
        dense_phi,
        dense_theta,
        TOROIDAL_ANGLES_DEG,
        POLOIDAL_ANGLES_DEG,
    )
    original = thickness_matrices(original_build)

    report = {
        "schema": "parastell.clearance_candidate_measurement/v1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": "public_r1_negative_reference_inputs",
        "inputs": {
            "vmec_path": str(vmec_path.resolve()),
            "vmec_sha256": sha256_file(vmec_path),
            "magnet_step_path": str(magnet_step_path.resolve()),
            "magnet_step_sha256": sha256_file(magnet_step_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "measurement": {
            "method": "first positive ray hit on actual outer-magnet B-rep tessellation",
            "requested_dense_samples": dense_samples,
            "actual_dense_shape": list(distances.shape),
            "clearance_target_cm": clearance_cm,
            "tessellation_tolerance_cm": tessellation_tolerance_cm,
            "triangle_count": measurement["triangle_count"],
            "vertex_count": measurement["vertex_count"],
            "hit_count": int(np.count_nonzero(measurement["hit_mask"])),
            "maximum_dense_deficit_cm": float(np.max(dense_deficit)),
            "changed_control_count": int(np.count_nonzero(projected_deficit)),
            "distances_cm": _json_array(distances),
            "dense_deficit_cm": _json_array(dense_deficit),
            "projected_control_deficit_cm": _json_array(projected_deficit),
        },
        "policies": {
            "A1_breeder_only": _policy_result(
                original,
                projected_deficit,
                clearance_cm=clearance_cm,
                reduction_order=("breeder",),
                minima={"breeder": 5.0},
            ),
            "A2_breeder_then_local_shield": _policy_result(
                original,
                projected_deficit,
                clearance_cm=clearance_cm,
                reduction_order=("breeder", "shield"),
                minima={"breeder": 5.0, "shield": 5.0},
            ),
        },
        "qualification": "DERIVATION_ONLY_NOT_GLOBAL_MINIMUM_CLEARANCE_PROOF",
    }
    (output_dir / "clearance_measurement.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vmec", type=Path)
    parser.add_argument("magnet_step", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clearance-cm", type=float, default=5.0)
    parser.add_argument("--dense-samples", type=int, default=61)
    parser.add_argument("--tessellation-tolerance-cm", type=float, default=0.1)
    arguments = parser.parse_args()
    if arguments.dense_samples < 9:
        parser.error("--dense-samples must be at least 9")
    measure(
        arguments.vmec,
        arguments.magnet_step,
        arguments.output_dir,
        clearance_cm=arguments.clearance_cm,
        dense_samples=arguments.dense_samples,
        tessellation_tolerance_cm=arguments.tessellation_tolerance_cm,
    )


if __name__ == "__main__":
    main()
