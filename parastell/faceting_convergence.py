"""Resolution-independent qualification of two DAGMC faceting levels."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SNAPSHOT_SCHEMA = "parastell.faceting_snapshot/v1.0.0"
REPORT_SCHEMA = "parastell.faceting_convergence/v1.0.0"
PROTOCOL_SHA256 = (
    "d1f4216a279ffe6608e3b334d85828bb126036833e1d17aa241880c73ec8cacf"
)
PUBLIC_REFERENCE_OVERLAP_REPORT_SHA256 = (
    "f30fd91d4ed6f797e498acc600138b5006f335ac698bfdaad8da8af66ccb7f62"
)
FORMER_OVERLAP_WITNESSES_CM = {
    "magnet-0005": [1160.39, 380.476, 128.459],
    "magnet-0006": [1095.15, 498.581, 92.5898],
    "magnet-0011": [476.445, 1095.28, -86.2664],
    "magnet-0012": [364.23, 1158.37, -121.329],
}


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def _native_qualification_pass(report: Mapping[str, Any]) -> bool:
    watertight = report.get("check_watertight", {})
    overlaps = report.get("overlap_checks", [])
    precisions = sorted(row.get("points_per_edge") for row in overlaps)
    return bool(
        report.get("h5m_unchanged") is True
        and watertight.get("pass") is True
        and watertight.get("unmatched_edge_count") == 0
        and watertight.get("unsealed_surface_count") == 0
        and watertight.get("unsealed_volume_count") == 0
        and precisions == [1, 2, 4]
        and all(
            row.get("pass") is True
            and row.get("terminal_overlap_location_count") == 0
            and row.get("parsed_overlap_location_count") == 0
            and row.get("row_count_matches_terminal") is True
            and row.get("precision_header_pass") is True
            for row in overlaps
        )
    )


def _volume_map(
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, dict], list[str]]:
    rows = snapshot.get("semantic_volumes", [])
    identifiers = [str(row.get("semantic_id", "")) for row in rows]
    errors = []
    if not rows:
        errors.append("semantic volume inventory is empty")
    if any(not value for value in identifiers):
        errors.append("semantic volume identity is missing")
    duplicates = sorted(
        value for value, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        errors.append(f"duplicate semantic volume identities: {duplicates}")
    return {str(row.get("semantic_id")): dict(row) for row in rows}, errors


def _snapshot_gate(
    snapshot: Mapping[str, Any],
    *,
    expected_level: str,
    expected_settings: Sequence[float],
    criteria_sha256: str,
    protocol_sha256: str,
) -> tuple[bool, list[str]]:
    reasons = []
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        reasons.append("snapshot schema mismatch")
    if snapshot.get("level") != expected_level:
        reasons.append("faceting level label mismatch")
    if snapshot.get("criteria_sha256") != criteria_sha256:
        reasons.append("acceptance-criteria hash mismatch")
    if snapshot.get("faceting_protocol_sha256") != protocol_sha256:
        reasons.append("faceting-protocol hash mismatch")
    if snapshot.get("construct_only") is not False:
        reasons.append("construct-only build cannot qualify")
    if snapshot.get("build_status") != "BUILD_COMPLETE_PENDING_QUALIFICATION":
        reasons.append("build status is not qualification-ready")
    if snapshot.get("raw_h5m_sha256_before") != snapshot.get(
        "raw_h5m_sha256_after"
    ):
        reasons.append("H5M changed while faceting evidence was collected")
    settings = snapshot.get("faceting_settings", {})
    actual = [settings.get("minimum_size_cm"), settings.get("maximum_size_cm")]
    if actual != [float(value) for value in expected_settings]:
        reasons.append("faceting settings differ from preregistration")
    if settings.get("algorithm") != 1:
        reasons.append("faceting algorithm differs from preregistration")
    if snapshot.get("native_topology_gate_pass") is not True:
        reasons.append("native topology gate failed")
    if snapshot.get("pydagmc_topology_gate_pass") is not True:
        reasons.append("PyDAGMC topology gate failed")
    if not _native_qualification_pass(
        snapshot.get("native_qualification", {})
    ):
        reasons.append("native watertight/overlap evidence failed")
    _, volume_errors = _volume_map(snapshot)
    reasons.extend(volume_errors)
    return not reasons, reasons


def _certified_deviation_pass(
    report: Mapping[str, Any],
    *,
    level: str,
    h5m_sha256: str,
    physical_input_identity_sha256: str,
    protocol_sha256: str,
    maximum_cover_radius_cm: float,
    solver_margin_cm: float,
) -> bool:
    maximum_projected = report.get("maximum_projected_distance_cm")
    actual_cover = report.get("actual_maximum_cover_radius_cm")
    solver_margin = report.get("solver_margin_cm")
    brep_tolerance = report.get("maximum_brep_entity_tolerance_cm")
    upper_bound = report.get("upper_bound_cm")
    values_valid = bool(
        _finite_nonnegative(maximum_projected)
        and _finite_positive(actual_cover)
        and _finite_nonnegative(solver_margin)
        and _finite_nonnegative(brep_tolerance)
        and _finite_positive(upper_bound)
    )
    if values_valid:
        expected_upper_bound = (
            float(maximum_projected)
            + float(actual_cover)
            + float(solver_margin)
            + float(brep_tolerance)
        )
        arithmetic_pass = math.isclose(
            float(upper_bound),
            expected_upper_bound,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    else:
        arithmetic_pass = False
    source_hashes = report.get("source_solid_sha256", {})
    physical_triangle_count = report.get("physical_triangle_count")
    covered_triangle_count = report.get("covered_triangle_count")
    incidence_count = report.get("physical_triangle_incidence_count")
    covered_incidence_count = report.get("covered_triangle_incidence_count")
    projected_count = report.get("projected_subtriangle_centroid_count")
    expected_incidence_digest = report.get(
        "expected_triangle_incidence_digest_sha256"
    )
    covered_incidence_digest = report.get(
        "covered_triangle_incidence_digest_sha256"
    )
    expected_physical_digest = report.get(
        "expected_physical_triangle_digest_sha256"
    )
    covered_physical_digest = report.get(
        "covered_physical_triangle_digest_sha256"
    )
    return bool(
        report.get("schema") == "parastell.facet_deviation_certificate/v1.0.0"
        and report.get("level") == level
        and report.get("faceting_protocol_sha256") == protocol_sha256
        and report.get("raw_h5m_sha256_before") == h5m_sha256
        and report.get("raw_h5m_sha256_after") == h5m_sha256
        and report.get("physical_input_identity_sha256")
        == physical_input_identity_sha256
        and report.get("direction") == "dagmc_facets_to_source_cad_boundary"
        and report.get("certified_upper_bound") is True
        and report.get("all_triangles_covered") is True
        and report.get("all_physical_volume_triangles") is True
        and report.get("all_magnet_triangles") is True
        and report.get("coverage_reconciliation_pass") is True
        and report.get("projection_target_semantic_match_pass") is True
        and report.get("projection_error_count") == 0
        and isinstance(physical_triangle_count, int)
        and isinstance(covered_triangle_count, int)
        and isinstance(incidence_count, int)
        and isinstance(covered_incidence_count, int)
        and isinstance(projected_count, int)
        and physical_triangle_count > 0
        and covered_triangle_count == physical_triangle_count
        and incidence_count >= physical_triangle_count
        and covered_incidence_count == incidence_count
        and projected_count >= covered_incidence_count
        and _valid_sha256(expected_incidence_digest)
        and covered_incidence_digest == expected_incidence_digest
        and _valid_sha256(expected_physical_digest)
        and covered_physical_digest == expected_physical_digest
        and report.get("triangle_id_digest_sha256")
        == expected_incidence_digest
        and isinstance(source_hashes, Mapping)
        and bool(source_hashes)
        and all(_valid_sha256(value) for value in source_hashes.values())
        and values_valid
        and float(actual_cover) <= float(maximum_cover_radius_cm)
        and math.isclose(
            float(solver_margin),
            float(solver_margin_cm),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and arithmetic_pass
    )


def compare_faceting_levels(
    coarse: Mapping[str, Any],
    refined: Mapping[str, Any],
    *,
    criteria: Mapping[str, Any],
    criteria_sha256: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    source_cad_clearance: Mapping[str, Any],
    coarse_deviation: Mapping[str, Any],
    refined_deviation: Mapping[str, Any],
    former_overlap_zones: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two already-collected evidence snapshots without raw-ID coupling."""
    faceting = criteria["faceting"]
    separation = criteria["separated_interface"]
    protocol_gate_pass = bool(
        protocol_sha256 == PROTOCOL_SHA256
        and protocol.get("schema")
        == "parastell.faceting_comparison_protocol/v1.0.0"
        and protocol.get("levels", {})
        .get("coarse", {})
        .get("minimum_mesh_size_cm")
        == float(faceting["coarse_mesh_size_cm"][0])
        and protocol.get("levels", {})
        .get("coarse", {})
        .get("maximum_mesh_size_cm")
        == float(faceting["coarse_mesh_size_cm"][1])
        and protocol.get("levels", {})
        .get("refined", {})
        .get("minimum_mesh_size_cm")
        == float(faceting["refined_mesh_size_cm"][0])
        and protocol.get("levels", {})
        .get("refined", {})
        .get("maximum_mesh_size_cm")
        == float(faceting["refined_mesh_size_cm"][1])
    )
    coarse_snapshot_pass, coarse_reasons = _snapshot_gate(
        coarse,
        expected_level="coarse",
        expected_settings=faceting["coarse_mesh_size_cm"],
        criteria_sha256=criteria_sha256,
        protocol_sha256=protocol_sha256,
    )
    refined_snapshot_pass, refined_reasons = _snapshot_gate(
        refined,
        expected_level="refined",
        expected_settings=faceting["refined_mesh_size_cm"],
        criteria_sha256=criteria_sha256,
        protocol_sha256=protocol_sha256,
    )
    physical_identity_pass = bool(
        coarse.get("physical_input_identity")
        and coarse.get("physical_input_identity")
        == refined.get("physical_input_identity")
    )
    physical_input_identity_sha256 = _json_sha256(
        coarse.get("physical_input_identity")
    )
    physical_identity_hash_pass = bool(
        physical_identity_pass
        and coarse.get("physical_input_identity_sha256")
        == physical_input_identity_sha256
        and refined.get("physical_input_identity_sha256")
        == physical_input_identity_sha256
    )
    source_mesh_identity_pass = bool(
        coarse.get("source_mesh_canonical_fingerprint")
        and coarse.get("source_mesh_canonical_fingerprint")
        == refined.get("source_mesh_canonical_fingerprint")
    )
    topology_identity_pass = bool(
        coarse.get("semantic_topology_signature")
        and coarse.get("semantic_topology_signature")
        == refined.get("semantic_topology_signature")
    )

    coarse_volumes, coarse_volume_errors = _volume_map(coarse)
    refined_volumes, refined_volume_errors = _volume_map(refined)
    semantic_ids_match = bool(coarse_volumes) and set(coarse_volumes) == set(
        refined_volumes
    )
    volume_rows = []
    area_rows = []
    if semantic_ids_match:
        for semantic_id in sorted(coarse_volumes):
            coarse_row = coarse_volumes[semantic_id]
            refined_row = refined_volumes[semantic_id]
            refined_volume = float(refined_row.get("volume_cm3", float("nan")))
            coarse_volume = float(coarse_row.get("volume_cm3", float("nan")))
            relative_volume = (
                abs(coarse_volume - refined_volume) / abs(refined_volume)
                if _finite_positive(abs(refined_volume))
                else float("inf")
            )
            volume_rows.append(
                {
                    "semantic_id": semantic_id,
                    "coarse_volume_cm3": coarse_volume,
                    "refined_volume_cm3": refined_volume,
                    "relative_difference": relative_volume,
                    "pass": math.isfinite(relative_volume)
                    and relative_volume
                    <= float(
                        faceting[
                            "maximum_component_volume_relative_difference"
                        ]
                    ),
                }
            )
            coarse_area = float(
                coarse_row.get("total_boundary_area_cm2", float("nan"))
            )
            refined_area = float(
                refined_row.get("total_boundary_area_cm2", float("nan"))
            )
            relative_area = (
                abs(coarse_area - refined_area) / abs(refined_area)
                if _finite_positive(abs(refined_area))
                else float("inf")
            )
            is_magnet = refined_row.get("material") == "magnets"
            area_rows.append(
                {
                    "semantic_id": semantic_id,
                    "material": refined_row.get("material"),
                    "coarse_area_cm2": coarse_area,
                    "refined_area_cm2": refined_area,
                    "relative_difference": relative_area,
                    "gated": is_magnet,
                    "pass": (not is_magnet)
                    or (
                        math.isfinite(relative_area)
                        and relative_area
                        <= float(
                            faceting["maximum_magnet_area_relative_difference"]
                        )
                    ),
                }
            )
    volume_gate_pass = semantic_ids_match and all(
        row["pass"] for row in volume_rows
    )
    magnet_area_gate_pass = semantic_ids_match and all(
        row["pass"] for row in area_rows
    )
    magnet_rows_coarse = [
        row
        for row in coarse_volumes.values()
        if row.get("material") == "magnets"
    ]
    magnet_rows_refined = [
        row
        for row in refined_volumes.values()
        if row.get("material") == "magnets"
    ]
    magnet_envelope_gate_pass = bool(
        len(magnet_rows_coarse) == len(magnet_rows_refined) == 18
        and all(row.get("envelope_pass") is True for row in magnet_rows_coarse)
        and all(
            row.get("envelope_pass") is True for row in magnet_rows_refined
        )
    )

    exact_clearance = source_cad_clearance.get("minimum_clearance_cm")
    exact_clearance_pass = bool(
        source_cad_clearance.get("source_cad_gate_pass") is True
        and _finite_positive(exact_clearance)
        and float(exact_clearance)
        >= float(separation["minimum_accepted_clearance_cm"])
    )
    deviation_protocol = protocol.get("facet_deviation_certificate", {})
    deviation_evidence_pass = _certified_deviation_pass(
        coarse_deviation,
        level="coarse",
        h5m_sha256=str(coarse.get("raw_h5m_sha256_before")),
        physical_input_identity_sha256=physical_input_identity_sha256,
        protocol_sha256=protocol_sha256,
        maximum_cover_radius_cm=float(
            deviation_protocol.get("maximum_cover_radius_cm", float("nan"))
        ),
        solver_margin_cm=float(
            deviation_protocol.get("solver_margin_cm", float("nan"))
        ),
    ) and _certified_deviation_pass(
        refined_deviation,
        level="refined",
        h5m_sha256=str(refined.get("raw_h5m_sha256_before")),
        physical_input_identity_sha256=physical_input_identity_sha256,
        protocol_sha256=protocol_sha256,
        maximum_cover_radius_cm=float(
            deviation_protocol.get("maximum_cover_radius_cm", float("nan"))
        ),
        solver_margin_cm=float(
            deviation_protocol.get("solver_margin_cm", float("nan"))
        ),
    )
    clearance_margin_pass = bool(
        exact_clearance_pass
        and deviation_evidence_pass
        and float(exact_clearance)
        > 2.0 * float(refined_deviation["upper_bound_cm"])
    )
    faceted_clearance_pass = bool(
        _finite_positive(coarse.get("faceted_clearance_cm"))
        and _finite_positive(refined.get("faceted_clearance_cm"))
    )

    zone_rows = []
    for row in former_overlap_zones:
        zone_id = str(row.get("zone_id"))
        expected_witness = FORMER_OVERLAP_WITNESSES_CM.get(zone_id)
        witness = row.get("witness_cm")
        witness_pass = bool(
            isinstance(witness, list)
            and expected_witness is not None
            and len(witness) == 3
            and all(
                math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for actual, expected in zip(witness, expected_witness)
            )
        )
        role_rows = []
        for role in ("magnet", "vac_vessel"):
            coarse_role = row.get("coarse", {}).get(role, {})
            refined_role = row.get("refined", {}).get(role, {})
            coarse_edge = coarse_role.get("maximum_edge_cm")
            refined_edge = refined_role.get("maximum_edge_cm")
            coarse_triangles = coarse_role.get("triangle_count")
            refined_triangles = refined_role.get("triangle_count")
            role_rows.append(
                bool(
                    coarse_role.get("covered") is True
                    and refined_role.get("covered") is True
                    and _finite_positive(coarse_edge)
                    and _finite_positive(refined_edge)
                    and float(refined_edge) < float(coarse_edge)
                    and isinstance(coarse_triangles, int)
                    and isinstance(refined_triangles, int)
                    and coarse_triangles > 0
                    and refined_triangles > coarse_triangles
                )
            )
        passed = bool(
            row.get("public_reference_overlap_report_sha256")
            == PUBLIC_REFERENCE_OVERLAP_REPORT_SHA256
            and witness_pass
            and math.isclose(
                float(row.get("neighborhood_radius_cm", float("nan"))),
                40.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and row.get("refinement_classification")
            == "GLOBAL_REFINEMENT_COVERS_FORMER_ZONES"
            and all(role_rows)
        )
        zone_rows.append({**dict(row), "pass": passed})
    required_zone_ids = {
        "magnet-0005",
        "magnet-0006",
        "magnet-0011",
        "magnet-0012",
    }
    observed_zone_ids = {str(row.get("zone_id")) for row in zone_rows}
    former_zone_gate_pass = observed_zone_ids == required_zone_ids and all(
        row["pass"] for row in zone_rows
    )

    gates = {
        "faceting_protocol_binding_pass": protocol_gate_pass,
        "coarse_snapshot_pass": coarse_snapshot_pass,
        "refined_snapshot_pass": refined_snapshot_pass,
        "physical_input_identity_pass": physical_identity_hash_pass,
        "source_mesh_identity_pass": source_mesh_identity_pass,
        "semantic_volume_identity_pass": semantic_ids_match
        and not coarse_volume_errors
        and not refined_volume_errors,
        "semantic_topology_identity_pass": topology_identity_pass,
        "per_volume_stability_pass": volume_gate_pass,
        "magnet_area_stability_pass": magnet_area_gate_pass,
        "all_magnet_envelopes_pass": magnet_envelope_gate_pass,
        "faceted_clearances_positive_pass": faceted_clearance_pass,
        "source_cad_clearance_pass": exact_clearance_pass,
        "certified_deviation_evidence_pass": deviation_evidence_pass,
        "strict_clearance_margin_pass": clearance_margin_pass,
        "former_overlap_zone_refinement_pass": former_zone_gate_pass,
    }
    gate = all(gates.values())
    if not coarse_snapshot_pass or not refined_snapshot_pass:
        status = "UNQUALIFIED_INPUT_EVIDENCE"
    elif not physical_identity_hash_pass or not source_mesh_identity_pass:
        status = "UNQUALIFIED_SOURCE_CAD_IDENTITY"
    elif not semantic_ids_match or not topology_identity_pass:
        status = "UNQUALIFIED_SEMANTIC_MAPPING"
    elif not deviation_evidence_pass:
        status = "UNQUALIFIED_REFINED_FACET_DEVIATION"
    elif gate:
        status = "FACETING_CONVERGENCE_PASS"
    else:
        status = "FACETING_CONVERGENCE_FAIL"
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "criteria_sha256": criteria_sha256,
        "faceting_protocol_sha256": protocol_sha256,
        "gates": gates,
        "coarse_snapshot_reasons": coarse_reasons,
        "refined_snapshot_reasons": refined_reasons,
        "semantic_volume_errors": {
            "coarse": coarse_volume_errors,
            "refined": refined_volume_errors,
        },
        "per_volume_comparison": volume_rows,
        "per_volume_area_comparison": area_rows,
        "source_cad_clearance_cm": exact_clearance,
        "coarse_deviation": dict(coarse_deviation),
        "refined_deviation": dict(refined_deviation),
        "former_overlap_zones": zone_rows,
        "faceting_convergence_gate_pass": gate,
        "scientific_scope": (
            "two-level thresholded resolution stability; not a convergence-order "
            "or extrapolated discretization-error estimate"
        ),
    }
