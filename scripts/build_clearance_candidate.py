"""Build a create-only clearance-constrained public-reference candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import parastell.parastell as ps
from parastell.public_geometry_candidate import MAGNET_SAMPLE_MOD
from parastell.public_geometry_candidate import MAGNET_THICKNESS_CM
from parastell.public_geometry_candidate import MAGNET_WIDTH_CM
from parastell.public_geometry_candidate import POLOIDAL_ANGLES_DEG
from parastell.public_geometry_candidate import SOURCE_CFS_SAMPLES
from parastell.public_geometry_candidate import SOURCE_POLOIDAL_SAMPLES
from parastell.public_geometry_candidate import SOURCE_TOROIDAL_SAMPLES
from parastell.public_geometry_candidate import TOROIDAL_ANGLES_DEG
from parastell.public_geometry_candidate import TOROIDAL_EXTENT_DEG
from parastell.public_geometry_candidate import WALL_S
from parastell.public_geometry_candidate import public_reference_radial_build
from parastell.public_geometry_candidate import replace_thickness_matrices
from parastell.reference_geometry import sha256_file


def build(
    vmec_path: Path,
    coils_path: Path,
    measurement_path: Path,
    output_dir: Path,
    *,
    policy: str,
    min_mesh_size_cm: float,
    max_mesh_size_cm: float,
    construct_only: bool = False,
    acceptance_criteria_path: Path | None = None,
    expected_acceptance_criteria_sha256: str | None = None,
    expected_vmec_sha256: str | None = None,
    expected_coils_sha256: str | None = None,
    expected_measurement_sha256: str | None = None,
) -> None:
    if not construct_only and (
        acceptance_criteria_path is None
        or expected_acceptance_criteria_sha256 is None
        or expected_vmec_sha256 is None
        or expected_coils_sha256 is None
        or expected_measurement_sha256 is None
    ):
        raise ValueError(
            "a full candidate build requires all expected input hashes"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"create-only output already exists: {output_dir}"
        )
    vmec_sha256 = sha256_file(vmec_path)
    coils_sha256 = sha256_file(coils_path)
    measurement_sha256 = sha256_file(measurement_path)
    if not construct_only:
        if vmec_sha256 != expected_vmec_sha256:
            raise ValueError("VMEC hash mismatch")
        if coils_sha256 != expected_coils_sha256:
            raise ValueError("coil hash mismatch")
        if measurement_sha256 != expected_measurement_sha256:
            raise ValueError("clearance-measurement hash mismatch")
        criteria_sha256 = sha256_file(acceptance_criteria_path)
        if criteria_sha256 != expected_acceptance_criteria_sha256:
            raise ValueError("acceptance-criteria hash mismatch")
        criteria = json.loads(
            acceptance_criteria_path.read_text(encoding="utf-8")
        )
        registered_levels = {
            tuple(float(value) for value in level)
            for level in (
                criteria["faceting"]["coarse_mesh_size_cm"],
                criteria["faceting"]["refined_mesh_size_cm"],
            )
        }
        if (
            float(min_mesh_size_cm),
            float(max_mesh_size_cm),
        ) not in registered_levels:
            raise ValueError("faceting settings are not preregistered")
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    policy_data = measurement["policies"][policy]
    if policy_data["status"] != "MATRICES_DERIVED_PENDING_SOURCE_CAD_BUILD":
        raise RuntimeError(
            f"selected policy is not buildable: {policy_data['status']}"
        )
    matrices = {
        name: np.asarray(values, dtype=float)
        for name, values in policy_data[
            "corrected_thickness_matrices_cm"
        ].items()
    }
    radial_build = replace_thickness_matrices(
        public_reference_radial_build(), matrices
    )

    output_dir.mkdir(parents=True)
    matrix_receipt = {
        "schema": "parastell.clearance_candidate_build/v1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "vmec_path": str(vmec_path.resolve()),
        "vmec_sha256": vmec_sha256,
        "coils_path": str(coils_path.resolve()),
        "coils_sha256": coils_sha256,
        "measurement_path": str(measurement_path.resolve()),
        "measurement_sha256": measurement_sha256,
        "acceptance_criteria": (
            {
                "path": str(acceptance_criteria_path.resolve()),
                "sha256": sha256_file(acceptance_criteria_path),
            }
            if acceptance_criteria_path is not None
            else None
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "geometry_invariants": {
            "magnet_width_cm": MAGNET_WIDTH_CM,
            "magnet_thickness_cm": MAGNET_THICKNESS_CM,
            "case_thickness_cm": 0.0,
            "magnet_transform": "native_unmodified",
            "wall_s": WALL_S,
            "toroidal_extent_deg": TOROIDAL_EXTENT_DEG,
            "source_samples": [
                SOURCE_CFS_SAMPLES,
                SOURCE_POLOIDAL_SAMPLES,
                SOURCE_TOROIDAL_SAMPLES,
            ],
        },
        "faceting": {
            "min_mesh_size_cm": min_mesh_size_cm,
            "max_mesh_size_cm": max_mesh_size_cm,
            "algorithm": 1,
        },
        "corrected_thickness_matrices_cm": {
            name: values.tolist() for name, values in matrices.items()
        },
        "status": "BUILD_STARTED",
        "construct_only": bool(construct_only),
    }
    receipt_path = output_dir / "candidate_build_manifest.json"
    receipt_path.write_text(json.dumps(matrix_receipt, indent=2) + "\n")

    stellarator = ps.Stellarator(str(vmec_path))
    stellarator.construct_invessel_build(
        TOROIDAL_ANGLES_DEG,
        POLOIDAL_ANGLES_DEG,
        WALL_S,
        radial_build,
    )
    stellarator.export_invessel_build_step(export_dir=output_dir)
    stellarator.construct_magnets_from_filaments(
        str(coils_path),
        MAGNET_WIDTH_CM,
        MAGNET_THICKNESS_CM,
        TOROIDAL_EXTENT_DEG,
        sample_mod=MAGNET_SAMPLE_MOD,
    )
    stellarator.export_magnets_step(
        filename="magnet_set",
        export_dir=output_dir,
    )
    stellarator.construct_source_mesh(
        np.linspace(0.0, 1.0, SOURCE_CFS_SAMPLES),
        np.linspace(0.0, 360.0, SOURCE_POLOIDAL_SAMPLES),
        np.linspace(0.0, TOROIDAL_EXTENT_DEG, SOURCE_TOROIDAL_SAMPLES),
    )
    stellarator.export_source_mesh(
        filename="source_mesh", export_dir=output_dir
    )
    if construct_only:
        matrix_receipt["status"] = "CONSTRUCTION_COMPLETE"
        matrix_receipt["artifacts"] = {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path != receipt_path
        }
        receipt_path.write_text(json.dumps(matrix_receipt, indent=2) + "\n")
        return
    stellarator.build_cad_to_dagmc_model()
    stellarator.export_cad_to_dagmc(
        filename="dagmc",
        export_dir=output_dir,
        min_mesh_size=min_mesh_size_cm,
        max_mesh_size=max_mesh_size_cm,
        algorithm=1,
    )

    matrix_receipt["status"] = "BUILD_COMPLETE_PENDING_QUALIFICATION"
    matrix_receipt["artifacts"] = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path != receipt_path
    }
    receipt_path.write_text(json.dumps(matrix_receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vmec", type=Path)
    parser.add_argument("coils", type=Path)
    parser.add_argument("measurement", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--policy",
        choices=("A1_breeder_only", "A2_breeder_then_local_shield"),
        default="A1_breeder_only",
    )
    parser.add_argument("--min-mesh-size-cm", type=float, default=5.0)
    parser.add_argument("--max-mesh-size-cm", type=float, default=20.0)
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--acceptance-criteria", type=Path)
    parser.add_argument("--expected-acceptance-criteria-sha256")
    parser.add_argument("--expected-vmec-sha256")
    parser.add_argument("--expected-coils-sha256")
    parser.add_argument("--expected-measurement-sha256")
    arguments = parser.parse_args()
    build(
        arguments.vmec,
        arguments.coils,
        arguments.measurement,
        arguments.output_dir,
        policy=arguments.policy,
        min_mesh_size_cm=arguments.min_mesh_size_cm,
        max_mesh_size_cm=arguments.max_mesh_size_cm,
        construct_only=arguments.construct_only,
        acceptance_criteria_path=arguments.acceptance_criteria,
        expected_acceptance_criteria_sha256=(
            arguments.expected_acceptance_criteria_sha256
        ),
        expected_vmec_sha256=arguments.expected_vmec_sha256,
        expected_coils_sha256=arguments.expected_coils_sha256,
        expected_measurement_sha256=arguments.expected_measurement_sha256,
    )


if __name__ == "__main__":
    main()
