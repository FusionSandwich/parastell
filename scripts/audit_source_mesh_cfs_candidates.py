"""Audit and select from an immutable set of source-mesh CFS candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from parastell.parametric_openmc16_model import _canonical_sha
from parastell.reference_geometry import sha256_file
from parastell.source_domain import audit_source_domain
from parastell.source_geometry_identity import source_mesh_fingerprint
from scripts.select_source_mesh_outer_cfs_cap import (
    SCHEMA,
    _outer_boundary_clearance,
    _validated_caps,
)

SOURCE_BOUNDARY_TOLERANCE_CM = 1.0e-6
SECTOR_ANGLE_TOLERANCE_DEGREES = 1.0e-12


def _verified_receipt(
    path: Path, expected_sha256: str, *, label: str
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} file hash mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    content_digest = receipt.get("receipt_content_sha256")
    content = dict(receipt)
    content.pop("receipt_content_sha256", None)
    if content_digest != _canonical_sha(content):
        raise ValueError(f"{label} content hash mismatch")
    return receipt


def _verified_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    manifest = _verified_receipt(
        path, expected_sha256, label="candidate-manifest"
    )
    if (
        manifest.get("schema") != "parastell.source_mesh_cfs_candidates/v1.0.0"
        or manifest.get("status") != "CANDIDATES_BUILT_CONTAINMENT_PENDING"
        or manifest.get("geometry_mutated") is not False
        or manifest.get("inner_cfs_planes_mutated") is not False
    ):
        raise ValueError("candidate manifest is not eligible for audit")
    return manifest


def _verified_control(path: Path, expected_sha256: str) -> dict[str, Any]:
    control = _verified_receipt(path, expected_sha256, label="audit-control")
    if control.get("schema") != (
        "parastell.source_mesh_outer_cfs_audit_control/v1.0.0"
    ):
        raise ValueError("unsupported source-mesh audit-control schema")
    source_volume_id = control.get("source_volume_id")
    if (
        isinstance(source_volume_id, bool)
        or not isinstance(source_volume_id, int)
        or source_volume_id <= 0
    ):
        raise ValueError("source volume ID must be positive")
    source_material = control.get("source_material")
    if not isinstance(source_material, str) or not source_material.strip():
        raise ValueError("source material must be non-empty")
    clearance = float(control.get("minimum_clearance_cm", 0.0))
    source_loss = float(control.get("maximum_omitted_source_fraction", -1.0))
    cut_angles = control.get("periodic_cut_plane_angles_degrees")
    cut_tolerance = control.get("periodic_cut_plane_tolerance_cm")
    if not math.isfinite(clearance) or clearance <= 0.0:
        raise ValueError("minimum clearance must be finite and positive")
    if not math.isfinite(source_loss) or not 0.0 <= source_loss < 1.0:
        raise ValueError("maximum omitted source fraction is invalid")
    if (
        not isinstance(cut_angles, list)
        or len(cut_angles) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in cut_angles
        )
    ):
        raise ValueError("periodic cut-plane angles are invalid")
    if not float(cut_angles[0]) < float(cut_angles[1]):
        raise ValueError(
            "periodic cut-plane angles must be unique and ordered"
        )
    if (
        isinstance(cut_tolerance, bool)
        or not isinstance(cut_tolerance, (int, float))
        or not math.isfinite(float(cut_tolerance))
        or float(cut_tolerance) <= 0.0
        or float(cut_tolerance) > SOURCE_BOUNDARY_TOLERANCE_CM
    ):
        raise ValueError("periodic cut-plane tolerance is invalid")
    for key in ("dagmc_sha256", "candidate_manifest_sha256"):
        value = str(control.get(key, ""))
        if len(value) != 64:
            raise ValueError(f"audit-control {key} is invalid")
    return control


def _validate_cut_plane_binding(
    control: dict[str, Any], manifest: dict[str, Any]
) -> None:
    expected_cut_angles = [0.0, float(manifest["extent_degrees"])]
    if not np.allclose(
        np.asarray(control["periodic_cut_plane_angles_degrees"], dtype=float),
        np.asarray(expected_cut_angles, dtype=float),
        rtol=0.0,
        atol=SECTOR_ANGLE_TOLERANCE_DEGREES,
    ):
        raise ValueError(
            "periodic cut-plane angles do not match the modeled sector extent"
        )


def _candidate_passes(
    *,
    domain_pass: bool,
    all_samples_enclosed: bool,
    minimum_clearance_cm: float,
    required_clearance_cm: float,
    reference_rate: float,
    candidate_rate: float,
    omitted_fraction: float,
    maximum_omitted_fraction: float,
    manifest_identity_matches: bool,
    input_immutability_pass: bool,
) -> bool:
    return bool(
        domain_pass
        and all_samples_enclosed
        and math.isfinite(minimum_clearance_cm)
        and minimum_clearance_cm >= required_clearance_cm
        and math.isfinite(reference_rate)
        and reference_rate > 0.0
        and math.isfinite(candidate_rate)
        and candidate_rate > 0.0
        and math.isfinite(omitted_fraction)
        and 0.0 <= omitted_fraction <= maximum_omitted_fraction
        and manifest_identity_matches
        and input_immutability_pass
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc_h5m", type=Path)
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("audit_control", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-audit-control-sha256", required=True)
    args = parser.parse_args()

    dagmc = args.dagmc_h5m.resolve(strict=True)
    manifest_path = args.candidate_manifest.resolve(strict=True)
    control_path = args.audit_control.resolve(strict=True)
    control = _verified_control(
        control_path, args.expected_audit_control_sha256
    )
    if sha256_file(dagmc) != control["dagmc_sha256"]:
        raise ValueError("DAGMC H5M hash mismatch")
    manifest = _verified_manifest(
        manifest_path, control["candidate_manifest_sha256"]
    )
    _validate_cut_plane_binding(control, manifest)
    if args.output_directory.exists():
        raise FileExistsError(
            f"create-only output exists: {args.output_directory}"
        )

    caps = _validated_caps(
        manifest["caps_descending"], manifest["radial_points"]
    )
    rows = manifest["candidates"]
    if len(rows) != len(caps):
        raise ValueError("candidate rows and preregistered caps disagree")
    vmec = Path(manifest["vmec"]["path"]).resolve(strict=True)
    reference = Path(manifest["reference_source_mesh"]["path"]).resolve(
        strict=True
    )
    immutable_expected = {
        "dagmc": control["dagmc_sha256"],
        "candidate_manifest": control["candidate_manifest_sha256"],
        "audit_control": args.expected_audit_control_sha256,
        "vmec": manifest["vmec"]["sha256"],
        "reference_source_mesh": manifest["reference_source_mesh"]["sha256"],
    }
    immutable_paths = {
        "dagmc": dagmc,
        "candidate_manifest": manifest_path,
        "audit_control": control_path,
        "vmec": vmec,
        "reference_source_mesh": reference,
    }
    for label, path in immutable_paths.items():
        if sha256_file(path) != immutable_expected[label]:
            raise ValueError(f"{label} pre-audit hash mismatch")

    reference_identity = source_mesh_fingerprint(reference)
    if (
        reference_identity["canonical_fingerprint"]
        != manifest["reference_source_mesh"]["canonical_fingerprint"]
    ):
        raise ValueError("reference source-mesh fingerprint mismatch")
    reference_rate = float(reference_identity["total_source_strength_n_per_s"])
    if not math.isfinite(reference_rate) or reference_rate <= 0.0:
        raise ValueError("reference source rate must be finite and positive")

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audited_rows = []
    provisional_selected_index = None
    candidate_mutation_detected = False
    for rank, (cap, row) in enumerate(zip(caps, rows, strict=True)):
        if (
            int(row["rank_descending"]) != rank
            or float(row["outer_cfs_cap"]) != cap
        ):
            raise ValueError("candidate rank or cap differs from manifest")
        candidate = Path(row["source_mesh"]["path"]).resolve(strict=True)
        expected_candidate_sha = row["source_mesh"]["sha256"]
        candidate_label = f"candidate_{rank:02d}_source_mesh"
        immutable_paths[candidate_label] = candidate
        immutable_expected[candidate_label] = expected_candidate_sha
        pre_candidate_sha = sha256_file(candidate)
        if pre_candidate_sha != expected_candidate_sha:
            raise ValueError(f"candidate {rank} source-mesh hash mismatch")
        domain = audit_source_domain(
            dagmc,
            candidate,
            reference,
            source_volume_id=int(control["source_volume_id"]),
            source_material=str(control["source_material"]),
            require_reference_identity=False,
            periodic_cut_plane_angles_degrees=control[
                "periodic_cut_plane_angles_degrees"
            ],
            periodic_cut_plane_tolerance_cm=float(
                control["periodic_cut_plane_tolerance_cm"]
            ),
            boundary_tolerance_cm=SOURCE_BOUNDARY_TOLERANCE_CM,
        )
        clearance = _outer_boundary_clearance(
            dagmc,
            vmec,
            outer_cfs_cap=cap,
            poloidal_points=manifest["poloidal_points"],
            toroidal_points=manifest["toroidal_points"],
            extent_degrees=manifest["extent_degrees"],
            source_volume_id=int(control["source_volume_id"]),
            source_material=str(control["source_material"]),
        )
        candidate_identity = domain["source_mesh_identity"]
        candidate_rate = float(
            candidate_identity["total_source_strength_n_per_s"]
        )
        omitted_fraction = (reference_rate - candidate_rate) / reference_rate
        manifest_identity_matches = bool(
            candidate_identity["canonical_fingerprint"]
            == row["source_mesh"]["canonical_fingerprint"]
            and np.isclose(
                candidate_rate,
                float(row["physical_source_rate_n_per_s"]),
                rtol=1.0e-13,
                atol=0.0,
            )
            and np.isclose(
                omitted_fraction,
                float(row["omitted_edge_source_fraction"]),
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        )
        post_candidate_sha = sha256_file(candidate)
        candidate_immutability_pass = bool(
            pre_candidate_sha == post_candidate_sha == expected_candidate_sha
        )
        candidate_pass = _candidate_passes(
            domain_pass=domain["source_domain_gate_pass"],
            all_samples_enclosed=clearance["all_samples_enclosed"],
            minimum_clearance_cm=clearance["minimum_clearance_cm"],
            required_clearance_cm=float(control["minimum_clearance_cm"]),
            reference_rate=reference_rate,
            candidate_rate=candidate_rate,
            omitted_fraction=omitted_fraction,
            maximum_omitted_fraction=float(
                control["maximum_omitted_source_fraction"]
            ),
            manifest_identity_matches=manifest_identity_matches,
            input_immutability_pass=candidate_immutability_pass,
        )
        domain_path = output / f"candidate-{rank:02d}-SOURCE_DOMAIN_AUDIT.json"
        domain_path.write_text(
            json.dumps(domain, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        audited_rows.append(
            {
                "rank_descending": rank,
                "outer_cfs_cap": cap,
                "candidate_source_mesh": {
                    "path": str(candidate),
                    "sha256_before": pre_candidate_sha,
                    "sha256_after": post_candidate_sha,
                    "expected_sha256": expected_candidate_sha,
                    "input_immutability_pass": candidate_immutability_pass,
                },
                "source_domain_audit": {
                    "path": str(domain_path),
                    "sha256": sha256_file(domain_path),
                    "invalid_sample_point_count": domain[
                        "invalid_sample_point_count"
                    ],
                    "pass": domain["source_domain_gate_pass"],
                    "reference_identity_required": False,
                },
                "outer_boundary_clearance": clearance,
                "physical_source_rate_n_per_s": candidate_rate,
                "omitted_edge_source_fraction": omitted_fraction,
                "maximum_omitted_source_fraction": control[
                    "maximum_omitted_source_fraction"
                ],
                "manifest_identity_matches": manifest_identity_matches,
                "candidate_pass": candidate_pass,
            }
        )
        if not candidate_immutability_pass:
            candidate_mutation_detected = True
            break
        if candidate_pass:
            provisional_selected_index = rank
            break

    post_audit_hashes = {
        label: sha256_file(path) for label, path in immutable_paths.items()
    }
    for row in audited_rows:
        label = f"candidate_{row['rank_descending']:02d}_source_mesh"
        final_sha = post_audit_hashes[label]
        candidate_receipt = row["candidate_source_mesh"]
        candidate_receipt["sha256_after"] = final_sha
        candidate_receipt["input_immutability_pass"] = bool(
            final_sha == candidate_receipt["expected_sha256"]
        )
        if not candidate_receipt["input_immutability_pass"]:
            row["candidate_pass"] = False
    input_immutability_pass = bool(
        not candidate_mutation_detected
        and post_audit_hashes == immutable_expected
    )
    selected_index = (
        provisional_selected_index if input_immutability_pass else None
    )
    if not input_immutability_pass:
        for row in audited_rows:
            row["candidate_pass"] = False
    status = (
        "SOURCE_MESH_OUTER_CFS_CAP_SELECTED"
        if selected_index is not None
        else (
            "BLOCKED_INPUT_MUTATION"
            if not input_immutability_pass
            else "BLOCKED_NO_SOURCE_MESH_CFS_CAP_PASSED"
        )
    )
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "selection_rule": (
            "first passing cap in the preregistered strictly descending list"
        ),
        "audit_control": {
            "path": str(control_path),
            "sha256": post_audit_hashes["audit_control"],
            "receipt_content_sha256": control["receipt_content_sha256"],
        },
        "candidate_manifest": {
            "path": str(manifest_path),
            "sha256": post_audit_hashes["candidate_manifest"],
            "receipt_content_sha256": manifest["receipt_content_sha256"],
        },
        "geometry_mutated": False,
        "inner_source_cfs_planes_mutated": False,
        "dagmc_h5m": {
            "path": str(dagmc),
            "sha256": post_audit_hashes["dagmc"],
        },
        "vmec": {"path": str(vmec), "sha256": post_audit_hashes["vmec"]},
        "reference_source_mesh": {
            "path": str(reference),
            "sha256": post_audit_hashes["reference_source_mesh"],
            "canonical_fingerprint": reference_identity[
                "canonical_fingerprint"
            ],
            "physical_source_rate_n_per_s": reference_rate,
        },
        "source_volume_id": int(control["source_volume_id"]),
        "source_material": str(control["source_material"]),
        "minimum_clearance_cm": float(control["minimum_clearance_cm"]),
        "maximum_omitted_source_fraction": float(
            control["maximum_omitted_source_fraction"]
        ),
        "periodic_cut_plane_angles_degrees": [
            float(value)
            for value in control["periodic_cut_plane_angles_degrees"]
        ],
        "periodic_cut_plane_tolerance_cm": float(
            control["periodic_cut_plane_tolerance_cm"]
        ),
        "source_boundary_tolerance_cm": SOURCE_BOUNDARY_TOLERANCE_CM,
        "preregistered_caps": caps,
        "candidates": audited_rows,
        "provisional_selected_candidate_index": provisional_selected_index,
        "selected_candidate_index": selected_index,
        "selected_candidate": (
            audited_rows[selected_index]
            if selected_index is not None
            else None
        ),
        "expected_input_hashes": immutable_expected,
        "post_audit_input_hashes": post_audit_hashes,
        "input_immutability_pass": input_immutability_pass,
        "production_run_authorized": False,
    }
    result["receipt_content_sha256"] = _canonical_sha(result)
    result_path = output / "SOURCE_MESH_OUTER_CFS_SELECTION.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(result_path)
    if selected_index is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
