"""Fail-closed validation of localized DAGMC surface-crossing records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .magnet_surface_audit import validate_coupling_interface


def _text(values: Any) -> np.ndarray:
    return np.asarray(values).astype(str)


def _errors(
    mask: np.ndarray, label: str, errors: list[dict[str, Any]]
) -> None:
    for index in np.flatnonzero(mask)[:25]:
        errors.append({"record_index": int(index), "failure": label})


def validate_boundary_localization(
    columns: Mapping[str, Any],
    facet_catalog: Mapping[str, Any],
    *,
    envelope_surface_ids: Any,
    coupling_interface: Any,
    barycentric_tolerance: float = 1.0e-7,
    reconstruction_tolerance_cm: float = 1.0e-6,
    normal_tolerance: float = 1.0e-8,
    crossing_sense_tolerance: float = 1.0e-12,
    near_grazing_mu_cutoff: float = 0.1,
) -> dict[str, Any]:
    """Reconstruct every bank point and validate topology-derived semantics."""
    interface = validate_coupling_interface(coupling_interface)
    required = {
        "position_global_cm",
        "direction_global",
        "outward_normal_global",
        "surface_id",
        "facet_id",
        "facet_index",
        "barycentric_coordinates",
        "mu",
        "crossing_sense",
        "grazing",
    }
    missing = required - set(columns)
    catalog_required = {
        "facet_id",
        "facet_index",
        "surface_id",
        "facet_vertices_global_cm",
        "outward_normal_global",
    }
    missing_catalog = catalog_required - set(facet_catalog)
    if missing or missing_catalog:
        return {
            "status": "BLOCKED_REQUIRED_LOCALIZATION_FIELDS_ABSENT",
            "pass": False,
            "coupling_interface": interface,
            "missing_record_fields": sorted(missing),
            "missing_catalog_fields": sorted(missing_catalog),
            "record_count": int(
                len(next(iter(columns.values()))) if columns else 0
            ),
        }

    position = np.asarray(columns["position_global_cm"], dtype=float)
    direction = np.asarray(columns["direction_global"], dtype=float)
    normals = np.asarray(columns["outward_normal_global"], dtype=float)
    surface = np.asarray(columns["surface_id"], dtype=int)
    facet_id = _text(columns["facet_id"])
    facet_index = np.asarray(columns["facet_index"], dtype=int)
    barycentric = np.asarray(columns["barycentric_coordinates"], dtype=float)
    count = len(surface)
    shapes = {
        "position_global_cm": (count, 3),
        "direction_global": (count, 3),
        "outward_normal_global": (count, 3),
        "barycentric_coordinates": (count, 3),
    }
    for name, shape in shapes.items():
        if np.asarray(columns[name]).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    catalog_ids = _text(facet_catalog["facet_id"])
    if len(set(catalog_ids.tolist())) != len(catalog_ids):
        raise ValueError("facet catalog IDs must be globally unique")
    lookup = {value: index for index, value in enumerate(catalog_ids)}
    catalog_row = np.asarray([lookup.get(value, -1) for value in facet_id])
    errors: list[dict[str, Any]] = []
    missing_facet = catalog_row < 0
    _errors(missing_facet, "facet_id_not_in_catalog", errors)
    safe_row = np.clip(catalog_row, 0, max(len(catalog_ids) - 1, 0))
    catalog_surface = np.asarray(facet_catalog["surface_id"], dtype=int)[
        safe_row
    ]
    catalog_index = np.asarray(facet_catalog["facet_index"], dtype=int)[
        safe_row
    ]
    _errors(
        ~missing_facet & (surface != catalog_surface),
        "record_surface_disagrees_with_facet_catalog",
        errors,
    )
    _errors(
        ~missing_facet & (facet_index != catalog_index),
        "record_facet_index_disagrees_with_catalog",
        errors,
    )
    accepted_surfaces = {int(value) for value in envelope_surface_ids}
    foreign_surface = np.asarray(
        [int(value) not in accepted_surfaces for value in surface]
    )
    _errors(foreign_surface, "surface_not_in_selected_envelope", errors)

    bary_ok = (
        np.all(np.isfinite(barycentric), axis=1)
        & np.all(barycentric >= -float(barycentric_tolerance), axis=1)
        & np.all(barycentric <= 1.0 + float(barycentric_tolerance), axis=1)
        & np.isclose(
            barycentric.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=float(barycentric_tolerance),
        )
    )
    _errors(~bary_ok, "barycentric_coordinates_outside_tolerance", errors)
    vertices = np.asarray(
        facet_catalog["facet_vertices_global_cm"], dtype=float
    )
    if vertices.shape != (len(catalog_ids), 3, 3):
        raise ValueError(
            "facet catalog vertices must have shape (facets, 3, 3)"
        )
    reconstructed = np.einsum("ni,nij->nj", barycentric, vertices[safe_row])
    reconstruction_error = np.linalg.norm(reconstructed - position, axis=1)
    reconstruction_ok = (
        ~missing_facet
        & np.isfinite(reconstruction_error)
        & (reconstruction_error <= float(reconstruction_tolerance_cm))
    )
    _errors(
        ~reconstruction_ok, "global_position_reconstruction_failed", errors
    )

    catalog_normal = np.asarray(
        facet_catalog["outward_normal_global"], dtype=float
    )[safe_row]
    normal_error = np.linalg.norm(normals - catalog_normal, axis=1)
    normal_ok = (
        ~missing_facet
        & np.isfinite(normal_error)
        & (normal_error <= float(normal_tolerance))
    )
    _errors(~normal_ok, "outward_normal_disagrees_with_dagmc_sense", errors)

    direction_norm = np.linalg.norm(direction, axis=1)
    normal_norm = np.linalg.norm(normals, axis=1)
    unit_ok = np.isclose(
        direction_norm, 1.0, atol=normal_tolerance
    ) & np.isclose(normal_norm, 1.0, atol=normal_tolerance)
    _errors(~unit_ok, "direction_or_normal_not_unit", errors)
    expected_mu = np.einsum("ij,ij->i", direction, normals)
    mu = np.asarray(columns["mu"], dtype=float)
    mu_ok = np.isclose(mu, expected_mu, rtol=0.0, atol=normal_tolerance)
    _errors(~mu_ok, "mu_not_direction_dot_outward_normal", errors)
    expected_sense = np.where(
        expected_mu > float(crossing_sense_tolerance),
        "outgoing",
        np.where(
            expected_mu < -float(crossing_sense_tolerance),
            "incoming",
            "grazing",
        ),
    )
    sense_ok = _text(columns["crossing_sense"]) == expected_sense
    _errors(~sense_ok, "crossing_sense_disagrees_with_mu", errors)
    expected_near_grazing = np.abs(expected_mu) <= float(
        near_grazing_mu_cutoff
    )
    grazing_ok = (
        np.asarray(columns["grazing"], dtype=bool) == expected_near_grazing
    )
    _errors(~grazing_ok, "near_grazing_flag_disagrees_with_mu", errors)

    frame_status = "NOT_AVAILABLE"
    frame_ok = np.ones(count, dtype=bool)
    frame_fields = {
        "centreline_tangent",
        "centreline_radial",
        "centreline_transverse",
        "centreline_arclength_cm",
        "normalized_arclength",
    }
    if frame_fields.issubset(columns):
        tangent = np.asarray(columns["centreline_tangent"], dtype=float)
        radial = np.asarray(columns["centreline_radial"], dtype=float)
        transverse = np.asarray(columns["centreline_transverse"], dtype=float)
        arclength = np.asarray(columns["centreline_arclength_cm"], dtype=float)
        normalized = np.asarray(columns["normalized_arclength"], dtype=float)
        frame_ok = (
            np.isclose(
                np.linalg.norm(tangent, axis=1), 1.0, atol=normal_tolerance
            )
            & np.isclose(
                np.linalg.norm(radial, axis=1), 1.0, atol=normal_tolerance
            )
            & np.isclose(
                np.linalg.norm(transverse, axis=1), 1.0, atol=normal_tolerance
            )
            & np.all(
                np.isclose(
                    np.cross(tangent, radial),
                    transverse,
                    atol=normal_tolerance,
                ),
                axis=1,
            )
            & np.isfinite(arclength)
            & (arclength >= 0.0)
            & np.isfinite(normalized)
            & (normalized >= -normal_tolerance)
            & (normalized <= 1.0 + normal_tolerance)
        )
        frame_status = (
            "PASS_RIGHT_HANDED_CENTRELINE_FRAME"
            if np.all(frame_ok)
            else "FAIL_CENTRELINE_FRAME"
        )
        _errors(~frame_ok, "centreline_frame_or_arclength_invalid", errors)

    failures = {
        "foreign_surface": int(np.count_nonzero(foreign_surface)),
        "missing_facet": int(np.count_nonzero(missing_facet)),
        "barycentric": int(np.count_nonzero(~bary_ok)),
        "reconstruction": int(np.count_nonzero(~reconstruction_ok)),
        "normal": int(np.count_nonzero(~normal_ok)),
        "unit_vector": int(np.count_nonzero(~unit_ok)),
        "mu": int(np.count_nonzero(~mu_ok)),
        "sense": int(np.count_nonzero(~sense_ok)),
        "near_grazing": int(np.count_nonzero(~grazing_ok)),
        "frame": int(np.count_nonzero(~frame_ok)),
    }
    passed = all(value == 0 for value in failures.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "coupling_interface": interface,
        "record_count": count,
        "tolerances": {
            "barycentric": float(barycentric_tolerance),
            "reconstruction_cm": float(reconstruction_tolerance_cm),
            "normal": float(normal_tolerance),
            "crossing_sense_abs_mu": float(crossing_sense_tolerance),
            "near_grazing_abs_mu": float(near_grazing_mu_cutoff),
        },
        "frame_status": frame_status,
        "failure_counts": failures,
        "maximum_reconstruction_error_cm": (
            float(np.max(reconstruction_error)) if count else 0.0
        ),
        "maximum_normal_error": float(np.max(normal_error)) if count else 0.0,
        "sample_failures": errors[:100],
    }


def audit_handoff_localization_file(
    handoff_path: str | Path,
    *,
    facet_catalog_with_vertices: Mapping[str, Any],
    expected_envelope_surface_ids: Any,
    coupling_interface: Any,
    source_histories: int,
    barycentric_tolerance: float = 1.0e-7,
    reconstruction_tolerance_cm: float = 1.0e-6,
    normal_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Audit a v2 handoff plus the stricter Prompt-A interface semantics."""
    import h5py

    from .magnet_boundary_envelope import read_handoff

    path = Path(handoff_path).resolve()
    manifest, envelope, bank = read_handoff(path)
    with h5py.File(path, "r") as source:
        stored_catalog = {
            name: (
                dataset.asstr()[()]
                if dataset.dtype.kind in "OS"
                else dataset[()]
            )
            for name, dataset in source["facet_catalog"].items()
        }
    supplied = {
        name: np.asarray(value)
        for name, value in facet_catalog_with_vertices.items()
        if name != "centreline_linkage_available"
    }
    stored_ids = _text(stored_catalog["facet_id"])
    supplied_ids = _text(supplied["facet_id"])
    if set(stored_ids.tolist()) != set(supplied_ids.tolist()):
        raise ValueError(
            "reconstructed DAGMC facet catalog identity disagrees"
        )
    supplied_lookup = {
        value: index for index, value in enumerate(supplied_ids)
    }
    order = np.asarray(
        [supplied_lookup[value] for value in stored_ids], dtype=int
    )
    catalog = dict(stored_catalog)
    catalog["facet_vertices_global_cm"] = np.asarray(
        supplied["facet_vertices_global_cm"], dtype=float
    )[order]
    expected = tuple(
        sorted(int(value) for value in expected_envelope_surface_ids)
    )
    declared = tuple(sorted(int(value) for value in envelope.surface_ids))
    localization = validate_boundary_localization(
        bank.columns,
        catalog,
        envelope_surface_ids=expected,
        coupling_interface=coupling_interface,
        barycentric_tolerance=barycentric_tolerance,
        reconstruction_tolerance_cm=reconstruction_tolerance_cm,
        normal_tolerance=normal_tolerance,
        crossing_sense_tolerance=float(
            bank.metadata.get("crossing_sense_tolerance_abs_mu", 1.0e-12)
        ),
        near_grazing_mu_cutoff=float(
            bank.metadata.get("near_grazing_mu_cutoff", 0.1)
        ),
    )
    completeness = dict(bank.metadata.get("surface_bank_completeness", {}))
    same_run = bank.metadata.get("same_run_integrity_closure", {})
    same_run_pass = all(
        bool(row.get("passes_integrity_tolerance"))
        for row in same_run.values()
        if isinstance(row, Mapping)
    )
    policy = str(bank.metadata.get("canonical_record_policy", "")).lower()
    canonical_weight_pass = (
        "raw source-bank transport contributions" in policy
        and "no tally conditioning" in policy
        and int(source_histories) > 0
    )
    declared_match = declared == expected
    passed = (
        localization["pass"]
        and declared_match
        and completeness.get("classification") == "COMPLETE_CROSSING_BANK"
        and not bool(completeness.get("cap_reached"))
        and not bool(completeness.get("sampling_applied"))
        and same_run_pass
        and canonical_weight_pass
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "handoff_path": str(path),
        "handoff_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "coupling_interface": validate_coupling_interface(coupling_interface),
        "source_histories": int(source_histories),
        "declared_envelope_surface_ids": list(declared),
        "expected_envelope_surface_ids": list(expected),
        "declared_envelope_matches_required_interface": declared_match,
        "extra_declared_surface_ids": sorted(set(declared) - set(expected)),
        "missing_declared_surface_ids": sorted(set(expected) - set(declared)),
        "bank_completeness": completeness,
        "same_run_bank_tally_integrity_pass": same_run_pass,
        "canonical_weight_policy_pass": canonical_weight_pass,
        "canonical_weight_normalization": (
            "raw OpenMC record weight divided only by exact source histories"
        ),
        "manifest_normalization": manifest.get("normalization", {}),
        "localization": localization,
    }


def write_boundary_localization_audit(
    output_path: str | Path, reports: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        "schema": "parastell.boundary_localization_audit/v1",
        "interfaces": dict(reports),
        "pass": all(bool(row.get("pass")) for row in reports.values()),
    }
    result["status"] = "PASS" if result["pass"] else "FAIL"
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
