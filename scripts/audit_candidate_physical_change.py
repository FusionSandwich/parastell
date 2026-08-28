"""Quantify every physical change from vanilla R1 to one recovery candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cadquery as cq
import numpy as np

from parastell import invessel_build as ivb
from parastell.public_geometry_candidate import POLOIDAL_ANGLES_DEG
from parastell.public_geometry_candidate import TOROIDAL_ANGLES_DEG
from parastell.public_geometry_candidate import WALL_S
from parastell.public_geometry_candidate import public_reference_radial_build
from parastell.public_geometry_candidate import thickness_matrices
from parastell.pystell import read_vmec
from parastell.radial_build_constraints import (
    sampled_interpolated_layer_thickness,
)
from parastell.reference_geometry import sha256_file
from parastell.source_geometry_identity import canonical_step_payload_sha256
from parastell.source_geometry_identity import source_mesh_fingerprint


COMPONENT_FILES = (
    "chamber.step",
    "first_wall.step",
    "breeder.step",
    "back_wall.step",
    "shield.step",
    "vacuum_vessel.step",
)

REFERENCE_REQUIRED_ARTIFACTS = frozenset(
    {
        "back_wall.step",
        "breeder.step",
        "build.log",
        "chamber.step",
        "coils.example",
        "dagmc.h5m",
        "first_wall.step",
        "magnet_mesh.h5m",
        "magnet_set.step",
        "shield.step",
        "source_mesh.h5m",
        "vacuum_vessel.step",
        "weight_window_mesh.h5m",
        "wout_vmec.nc",
    }
)
CANDIDATE_REQUIRED_ARTIFACTS = frozenset(
    {
        *COMPONENT_FILES,
        "dagmc.h5m",
        "magnet_set.step",
        "source_mesh.h5m",
    }
)


def _one_solid(path: Path):
    solids = cq.importers.importStep(str(path)).solids().vals()
    if len(solids) != 1:
        raise RuntimeError(f"expected one solid in {path}, got {len(solids)}")
    return solids[0]


def _metrics(shape) -> dict:
    center = shape.Center()
    box = shape.BoundingBox()
    return {
        "volume_cm3": float(shape.Volume()),
        "area_cm2": float(shape.Area()),
        "centroid_cm": [float(center.x), float(center.y), float(center.z)],
        "bounding_box_cm": {
            "minimum": [float(box.xmin), float(box.ymin), float(box.zmin)],
            "maximum": [float(box.xmax), float(box.ymax), float(box.zmax)],
        },
    }


def _component_comparison(
    reference_dir: Path, candidate_dir: Path
) -> list[dict]:
    rows = []
    for filename in COMPONENT_FILES:
        reference_path = reference_dir / filename
        candidate_path = candidate_dir / filename
        reference = _metrics(_one_solid(reference_path))
        candidate = _metrics(_one_solid(candidate_path))
        rows.append(
            {
                "component": filename.removesuffix(".step"),
                "reference": {
                    **reference,
                    "raw_sha256": sha256_file(reference_path),
                    "canonical_step_sha256": canonical_step_payload_sha256(
                        reference_path
                    ),
                },
                "candidate": {
                    **candidate,
                    "raw_sha256": sha256_file(candidate_path),
                    "canonical_step_sha256": canonical_step_payload_sha256(
                        candidate_path
                    ),
                },
                "canonical_step_identical": (
                    canonical_step_payload_sha256(reference_path)
                    == canonical_step_payload_sha256(candidate_path)
                ),
                "volume_delta_cm3": (
                    candidate["volume_cm3"] - reference["volume_cm3"]
                ),
                "volume_relative_delta": (
                    candidate["volume_cm3"] / reference["volume_cm3"] - 1.0
                ),
                "area_delta_cm2": (
                    candidate["area_cm2"] - reference["area_cm2"]
                ),
                "area_relative_delta": (
                    candidate["area_cm2"] / reference["area_cm2"] - 1.0
                ),
            }
        )
    return rows


def _first_wall_affected_area(
    vmec_path: Path,
    phi_deg: np.ndarray,
    theta_deg: np.ndarray,
    reduction_cm: np.ndarray,
) -> dict:
    vmec = read_vmec.VMECData(str(vmec_path))
    reference_surface = ivb.VMECSurface(vmec)
    surface = ivb.Surface(
        reference_surface,
        WALL_S,
        np.deg2rad(theta_deg),
        np.deg2rad(phi_deg),
        np.full(reduction_cm.shape, 5.0),
        100.0,
    )
    surface.populate_ribs()
    surface.calculate_loci()
    points = np.asarray(surface.get_loci())
    first = points[:-1, :-1]
    second = points[1:, :-1]
    third = points[1:, 1:]
    fourth = points[:-1, 1:]
    areas = 0.5 * np.linalg.norm(
        np.cross(second - first, third - first), axis=2
    )
    areas += 0.5 * np.linalg.norm(
        np.cross(third - first, fourth - first), axis=2
    )
    changed = reduction_cm > 1.0e-10
    changed_cells = (
        changed[:-1, :-1]
        | changed[1:, :-1]
        | changed[1:, 1:]
        | changed[:-1, 1:]
    )
    return {
        "method": "sampled first-wall quadrilateral area on the same phi/theta grid",
        "total_sector_first_wall_area_cm2": float(np.sum(areas)),
        "affected_first_wall_area_cm2": float(np.sum(areas[changed_cells])),
        "affected_first_wall_area_fraction": float(
            np.sum(areas[changed_cells]) / np.sum(areas)
        ),
        "changed_cell_count": int(np.count_nonzero(changed_cells)),
        "cell_count": int(changed_cells.size),
    }


def _verify_manifest_artifacts(
    directory: Path,
    manifest_name: str,
    *,
    required_artifacts: frozenset[str] = frozenset(),
) -> dict:
    manifest_path = directory / manifest_name
    if not manifest_path.is_file():
        return {
            "pass": False,
            "manifest_path": str(manifest_path.resolve()),
            "reason": "manifest missing",
            "artifacts": [],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_artifacts = manifest.get("artifacts", {})
    missing_required = sorted(required_artifacts - set(declared_artifacts))
    rows = []
    for filename, expected in sorted(declared_artifacts.items()):
        path = directory / filename
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = sha256_file(path) if exists else None
        rows.append(
            {
                "filename": filename,
                "exists": exists,
                "expected_size_bytes": expected.get("size_bytes"),
                "actual_size_bytes": actual_size,
                "expected_sha256": expected.get("sha256"),
                "actual_sha256": actual_hash,
                "pass": exists
                and actual_size == expected.get("size_bytes")
                and actual_hash == expected.get("sha256"),
            }
        )
    return {
        "pass": bool(rows)
        and not missing_required
        and all(row["pass"] for row in rows),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "required_artifacts": sorted(required_artifacts),
        "missing_required_artifacts": missing_required,
        "artifacts": rows,
    }


def audit(
    reference_dir: Path,
    candidate_dir: Path,
    measurement_path: Path,
    output_dir: Path,
    *,
    policy: str,
    acceptance_criteria_path: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"create-only output already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    reference_integrity = _verify_manifest_artifacts(
        reference_dir,
        "VANILLA_BUILD_MANIFEST.json",
        required_artifacts=REFERENCE_REQUIRED_ARTIFACTS,
    )
    candidate_integrity = _verify_manifest_artifacts(
        candidate_dir,
        "candidate_build_manifest.json",
        required_artifacts=CANDIDATE_REQUIRED_ARTIFACTS,
    )
    candidate_manifest = candidate_integrity.get("manifest", {})
    criteria_sha256 = sha256_file(acceptance_criteria_path)
    criteria = json.loads(acceptance_criteria_path.read_text(encoding="utf-8"))
    reconstruction_criteria = criteria["continuous_reconstruction"]
    physical_criteria = criteria["physical_change"]
    toroidal_step_deg = float(
        reconstruction_criteria["maximum_toroidal_sampling_step_deg"]
    )
    poloidal_step_deg = float(
        reconstruction_criteria["maximum_poloidal_sampling_step_deg"]
    )
    minimum_breeder_thickness_cm = float(
        reconstruction_criteria["continuous_minimum_breeder_thickness_cm"]
    )
    candidate_matrices = {
        name: np.asarray(values, dtype=float)
        for name, values in measurement["policies"][policy][
            "corrected_thickness_matrices_cm"
        ].items()
    }
    reference_matrices = thickness_matrices(public_reference_radial_build())
    thickness = sampled_interpolated_layer_thickness(
        reference_matrices,
        candidate_matrices,
        layer="breeder",
        toroidal_control_angles_deg=TOROIDAL_ANGLES_DEG,
        poloidal_control_angles_deg=POLOIDAL_ANGLES_DEG,
        toroidal_step_deg=toroidal_step_deg,
        poloidal_step_deg=poloidal_step_deg,
    )
    continuous_map_path = output_dir / "continuous_breeder_thickness.npz"
    np.savez_compressed(
        continuous_map_path,
        toroidal_angles_deg=thickness.toroidal_angles_deg,
        poloidal_angles_deg=thickness.poloidal_angles_deg,
        reference_thickness_cm=thickness.reference_thickness_cm,
        candidate_thickness_cm=thickness.candidate_thickness_cm,
        reduction_cm=thickness.reduction_cm,
    )
    changed_layers = [
        name
        for name in reference_matrices
        if not np.array_equal(
            reference_matrices[name], candidate_matrices[name]
        )
    ]
    component_rows = _component_comparison(reference_dir, candidate_dir)
    magnet_reference = canonical_step_payload_sha256(
        reference_dir / "magnet_set.step"
    )
    magnet_candidate = canonical_step_payload_sha256(
        candidate_dir / "magnet_set.step"
    )
    source_reference = source_mesh_fingerprint(
        reference_dir / "source_mesh.h5m"
    )
    source_candidate = source_mesh_fingerprint(
        candidate_dir / "source_mesh.h5m"
    )
    thickness_summary = thickness.summary()
    affected_area = _first_wall_affected_area(
        reference_dir / "wout_vmec.nc",
        thickness.toroidal_angles_deg,
        thickness.poloidal_angles_deg,
        thickness.reduction_cm,
    )
    rows_by_name = {row["component"]: row for row in component_rows}
    criteria_contract = {
        "toroidal_sampling_step_deg": toroidal_step_deg,
        "poloidal_sampling_step_deg": poloidal_step_deg,
        "minimum_breeder_thickness_cm": minimum_breeder_thickness_cm,
        "actual_cad_deltas_required": physical_criteria[
            "actual_component_cad_volume_and_area_deltas_required"
        ]
        is True,
        "first_wall_change_forbidden": physical_criteria[
            "first_wall_geometry_change_allowed"
        ]
        is False,
        "magnet_change_forbidden": physical_criteria[
            "magnet_geometry_change_allowed"
        ]
        is False,
    }
    criteria_contract_pass = all(
        value
        for value in criteria_contract.values()
        if isinstance(value, bool)
    )
    report = {
        "schema": "parastell.candidate_physical_change/v1.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_dir": str(reference_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "measurement": {
            "path": str(measurement_path.resolve()),
            "sha256": sha256_file(measurement_path),
            "policy": policy,
        },
        "acceptance_criteria": {
            "path": str(acceptance_criteria_path.resolve()),
            "sha256": criteria_sha256,
        },
        "criteria_contract": criteria_contract,
        "reference_integrity": reference_integrity,
        "candidate_integrity": candidate_integrity,
        "audit_implementation": {
            "script_sha256": sha256_file(Path(__file__)),
            "radial_build_constraints_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "parastell"
                / "radial_build_constraints.py"
            ),
            "source_geometry_identity_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "parastell"
                / "source_geometry_identity.py"
            ),
            "public_geometry_candidate_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "parastell"
                / "public_geometry_candidate.py"
            ),
            "build_clearance_candidate_sha256": sha256_file(
                Path(__file__).resolve().parent
                / "build_clearance_candidate.py"
            ),
            "measure_clearance_candidate_sha256": sha256_file(
                Path(__file__).resolve().parent
                / "measure_clearance_candidate.py"
            ),
        },
        "generated_artifacts": {
            "continuous_breeder_thickness.npz": {
                "size_bytes": continuous_map_path.stat().st_size,
                "sha256": sha256_file(continuous_map_path),
            }
        },
        "declared_thickness_matrix_changes": {
            "changed_layers": changed_layers,
            "only_breeder_changed": changed_layers == ["breeder"],
        },
        "continuous_breeder_thickness": thickness_summary,
        "affected_area": affected_area,
        "components": component_rows,
        "magnet_identity": {
            "reference_canonical_step_sha256": magnet_reference,
            "candidate_canonical_step_sha256": magnet_candidate,
            "canonical_identical": magnet_reference == magnet_candidate,
        },
        "source_mesh_identity": {
            "reference": source_reference,
            "candidate": source_candidate,
            "canonical_identical": (
                source_reference["canonical_fingerprint"]
                == source_candidate["canonical_fingerprint"]
            ),
        },
        "gate": {
            "reference_manifest_and_artifacts_pass": reference_integrity[
                "pass"
            ],
            "candidate_manifest_and_artifacts_pass": candidate_integrity[
                "pass"
            ],
            "candidate_measurement_bound": (
                candidate_manifest.get("measurement_sha256")
                == sha256_file(measurement_path)
            ),
            "candidate_policy_bound": candidate_manifest.get("policy")
            == policy,
            "candidate_matrices_bound": all(
                np.array_equal(
                    np.asarray(
                        candidate_manifest.get(
                            "corrected_thickness_matrices_cm", {}
                        ).get(name, []),
                        dtype=float,
                    ),
                    values,
                )
                for name, values in candidate_matrices.items()
            ),
            "candidate_acceptance_criteria_bound": (
                candidate_manifest.get("acceptance_criteria", {}).get("sha256")
                == criteria_sha256
            ),
            "candidate_is_full_build": (
                candidate_manifest.get("construct_only") is False
                and candidate_manifest.get("status")
                == "BUILD_COMPLETE_PENDING_QUALIFICATION"
            ),
            "candidate_build_script_matches": (
                candidate_manifest.get("script_sha256")
                == sha256_file(
                    Path(__file__).resolve().parent
                    / "build_clearance_candidate.py"
                )
            ),
            "measurement_script_matches": (
                measurement.get("inputs", {}).get("script_sha256")
                == sha256_file(
                    Path(__file__).resolve().parent
                    / "measure_clearance_candidate.py"
                )
            ),
            "criteria_contract_pass": criteria_contract_pass,
            "only_breeder_thickness_matrix_changed": changed_layers
            == ["breeder"],
            "minimum_continuous_breeder_thickness_pass": (
                thickness_summary["minimum_candidate_thickness_cm"]
                >= minimum_breeder_thickness_cm
            ),
            "no_breeder_thickness_increase": (
                thickness_summary["maximum_increase_cm"] <= 1.0e-10
            ),
            "poloidal_closure_pass": (
                thickness_summary["poloidal_closure_max_abs_cm"] <= 1.0e-8
            ),
            "chamber_canonical_identical": rows_by_name["chamber"][
                "canonical_step_identical"
            ],
            "first_wall_canonical_identical": rows_by_name["first_wall"][
                "canonical_step_identical"
            ],
            "magnet_canonical_identical": magnet_reference == magnet_candidate,
            "source_mesh_canonical_identical": (
                source_reference["canonical_fingerprint"]
                == source_candidate["canonical_fingerprint"]
            ),
            "breeder_cad_change_nonzero": (
                rows_by_name["breeder"]["volume_delta_cm3"] < -1.0e-6
                and abs(rows_by_name["breeder"]["area_delta_cm2"]) > 1.0e-6
            ),
            "induced_outer_component_volume_changes_quantified": all(
                rows_by_name[name]["volume_delta_cm3"] < -1.0e-6
                for name in ("back_wall", "shield", "vacuum_vessel")
            ),
            "changed_area_quantified": (
                affected_area["affected_first_wall_area_cm2"] > 0.0
            ),
        },
    }
    report["physical_change_gate_pass"] = all(report["gate"].values())
    (output_dir / "candidate_physical_change.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("measurement", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--policy", default="A1_breeder_only")
    parser.add_argument("--acceptance-criteria", type=Path, required=True)
    arguments = parser.parse_args()
    audit(
        arguments.reference_dir,
        arguments.candidate_dir,
        arguments.measurement,
        arguments.output_dir,
        policy=arguments.policy,
        acceptance_criteria_path=arguments.acceptance_criteria,
    )


if __name__ == "__main__":
    main()
