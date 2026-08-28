import copy

import pytest

from parastell.faceting_convergence import compare_faceting_levels
from parastell.faceting_convergence import _json_sha256
from parastell.faceting_convergence import PROTOCOL_SHA256
from parastell.faceting_convergence import (
    PUBLIC_REFERENCE_OVERLAP_REPORT_SHA256,
)


CRITERIA = {
    "faceting": {
        "coarse_mesh_size_cm": [5.0, 20.0],
        "refined_mesh_size_cm": [2.5, 10.0],
        "maximum_component_volume_relative_difference": 0.0025,
        "maximum_magnet_area_relative_difference": 0.01,
    },
    "separated_interface": {"minimum_accepted_clearance_cm": 5.0},
}
PROTOCOL = {
    "schema": "parastell.faceting_comparison_protocol/v1.0.0",
    "levels": {
        "coarse": {
            "minimum_mesh_size_cm": 5.0,
            "maximum_mesh_size_cm": 20.0,
        },
        "refined": {
            "minimum_mesh_size_cm": 2.5,
            "maximum_mesh_size_cm": 10.0,
        },
    },
    "facet_deviation_certificate": {
        "maximum_cover_radius_cm": 0.25,
        "solver_margin_cm": 1.0e-7,
    },
}


def _native_qualification():
    return {
        "h5m_unchanged": True,
        "check_watertight": {
            "pass": True,
            "unmatched_edge_count": 0,
            "unsealed_surface_count": 0,
            "unsealed_volume_count": 0,
        },
        "overlap_checks": [
            {
                "points_per_edge": precision,
                "pass": True,
                "terminal_overlap_location_count": 0,
                "parsed_overlap_location_count": 0,
                "row_count_matches_terminal": True,
                "precision_header_pass": True,
            }
            for precision in (1, 2, 4)
        ],
    }


def _snapshot(level):
    settings = [5.0, 20.0] if level == "coarse" else [2.5, 10.0]
    physical_identity = {"vmec": "d" * 64, "step": "e" * 64}
    volumes = [
        {
            "semantic_id": "reactor-vacuum",
            "material": "Vacuum",
            "volume_cm3": 1000.0,
            "total_boundary_area_cm2": 600.0,
            "envelope_pass": True,
        }
    ]
    volumes.extend(
        {
            "semantic_id": f"magnet-{index:04d}",
            "material": "magnets",
            "volume_cm3": 100.0,
            "total_boundary_area_cm2": 50.0,
            "envelope_pass": True,
        }
        for index in range(18)
    )
    return {
        "schema": "parastell.faceting_snapshot/v1.0.0",
        "level": level,
        "criteria_sha256": "a" * 64,
        "faceting_protocol_sha256": PROTOCOL_SHA256,
        "construct_only": False,
        "build_status": "BUILD_COMPLETE_PENDING_QUALIFICATION",
        "raw_h5m_sha256_before": ("b" if level == "coarse" else "c") * 64,
        "raw_h5m_sha256_after": ("b" if level == "coarse" else "c") * 64,
        "faceting_settings": {
            "minimum_size_cm": settings[0],
            "maximum_size_cm": settings[1],
            "algorithm": 1,
        },
        "native_topology_gate_pass": True,
        "pydagmc_topology_gate_pass": True,
        "native_qualification": _native_qualification(),
        "physical_input_identity": physical_identity,
        "physical_input_identity_sha256": _json_sha256(physical_identity),
        "source_mesh_canonical_fingerprint": "f" * 64,
        "semantic_topology_signature": {"adjacencies": [["a", "EXTERIOR"]]},
        "semantic_volumes": volumes,
        "faceted_clearance_cm": 5.1,
    }


def _deviation(snapshot, value=0.2):
    cover_radius = 0.1
    solver_margin = 1.0e-7
    brep_tolerance = 0.0
    return {
        "schema": "parastell.facet_deviation_certificate/v1.0.0",
        "level": snapshot["level"],
        "faceting_protocol_sha256": PROTOCOL_SHA256,
        "raw_h5m_sha256_before": snapshot["raw_h5m_sha256_before"],
        "raw_h5m_sha256_after": snapshot["raw_h5m_sha256_before"],
        "physical_input_identity_sha256": snapshot[
            "physical_input_identity_sha256"
        ],
        "direction": "dagmc_facets_to_source_cad_boundary",
        "certified_upper_bound": True,
        "all_triangles_covered": True,
        "all_physical_volume_triangles": True,
        "all_magnet_triangles": True,
        "coverage_reconciliation_pass": True,
        "projection_target_semantic_match_pass": True,
        "projection_error_count": 0,
        "physical_triangle_count": 100,
        "covered_triangle_count": 100,
        "physical_triangle_incidence_count": 120,
        "covered_triangle_incidence_count": 120,
        "projected_subtriangle_centroid_count": 400,
        "triangle_id_digest_sha256": "1" * 64,
        "expected_triangle_incidence_digest_sha256": "1" * 64,
        "covered_triangle_incidence_digest_sha256": "1" * 64,
        "expected_physical_triangle_digest_sha256": "3" * 64,
        "covered_physical_triangle_digest_sha256": "3" * 64,
        "source_solid_sha256": {"magnet_set.step": "2" * 64},
        "maximum_projected_distance_cm": (
            value - cover_radius - solver_margin - brep_tolerance
        ),
        "actual_maximum_cover_radius_cm": cover_radius,
        "solver_margin_cm": solver_margin,
        "maximum_brep_entity_tolerance_cm": brep_tolerance,
        "upper_bound_cm": value,
    }


def _zones():
    witnesses = {
        5: [1160.39, 380.476, 128.459],
        6: [1095.15, 498.581, 92.5898],
        11: [476.445, 1095.28, -86.2664],
        12: [364.23, 1158.37, -121.329],
    }
    return [
        {
            "zone_id": f"magnet-{index:04d}",
            "public_reference_overlap_report_sha256": (
                PUBLIC_REFERENCE_OVERLAP_REPORT_SHA256
            ),
            "witness_cm": witnesses[index],
            "neighborhood_radius_cm": 40.0,
            "refinement_classification": (
                "GLOBAL_REFINEMENT_COVERS_FORMER_ZONES"
            ),
            "coarse": {
                role: {
                    "covered": True,
                    "maximum_edge_cm": 5.0,
                    "triangle_count": 10,
                }
                for role in ("magnet", "vac_vessel")
            },
            "refined": {
                role: {
                    "covered": True,
                    "maximum_edge_cm": 2.5,
                    "triangle_count": 40,
                }
                for role in ("magnet", "vac_vessel")
            },
        }
        for index in (5, 6, 11, 12)
    ]


def _compare(coarse=None, refined=None, **kwargs):
    coarse = coarse or _snapshot("coarse")
    refined = refined or _snapshot("refined")
    return compare_faceting_levels(
        coarse,
        refined,
        criteria=CRITERIA,
        criteria_sha256="a" * 64,
        protocol=PROTOCOL,
        protocol_sha256=PROTOCOL_SHA256,
        source_cad_clearance=kwargs.pop(
            "source_cad_clearance",
            {"source_cad_gate_pass": True, "minimum_clearance_cm": 5.2},
        ),
        coarse_deviation=kwargs.pop("coarse_deviation", _deviation(coarse)),
        refined_deviation=kwargs.pop("refined_deviation", _deviation(refined)),
        former_overlap_zones=kwargs.pop("former_overlap_zones", _zones()),
        **kwargs,
    )


def test_two_level_semantic_comparison_passes_with_distinct_h5m_hashes():
    report = _compare()
    assert report["status"] == "FACETING_CONVERGENCE_PASS"
    assert report["faceting_convergence_gate_pass"] is True


@pytest.mark.parametrize(
    "relative_difference, expected",
    [(0.0025, True), (0.0025001, False)],
)
def test_per_volume_threshold_boundary(relative_difference, expected):
    refined = _snapshot("refined")
    refined["semantic_volumes"][0]["volume_cm3"] = 1000.0
    coarse = _snapshot("coarse")
    coarse["semantic_volumes"][0]["volume_cm3"] = 1000.0 * (
        1.0 + relative_difference
    )
    assert (
        _compare(coarse, refined)["gates"]["per_volume_stability_pass"]
        is expected
    )


@pytest.mark.parametrize(
    "relative_difference, expected", [(0.01, True), (0.010001, False)]
)
def test_magnet_area_threshold_boundary(relative_difference, expected):
    coarse = _snapshot("coarse")
    refined = _snapshot("refined")
    coarse["semantic_volumes"][1]["total_boundary_area_cm2"] = 50.0 * (
        1.0 + relative_difference
    )
    assert (
        _compare(coarse, refined)["gates"]["magnet_area_stability_pass"]
        is expected
    )


def test_nonmagnet_area_drift_is_advisory_only():
    coarse = _snapshot("coarse")
    coarse["semantic_volumes"][0]["total_boundary_area_cm2"] = 1000.0
    report = _compare(coarse)
    assert report["gates"]["magnet_area_stability_pass"] is True
    reactor_row = next(
        row
        for row in report["per_volume_area_comparison"]
        if row["semantic_id"] == "reactor-vacuum"
    )
    assert reactor_row["gated"] is False


def test_raw_ids_and_triangle_counts_are_not_part_of_semantic_comparison():
    coarse = _snapshot("coarse")
    refined = _snapshot("refined")
    for index, row in enumerate(coarse["semantic_volumes"]):
        row["native_volume_id"] = index + 1
        row["triangle_count"] = 10
    for index, row in enumerate(refined["semantic_volumes"]):
        row["native_volume_id"] = 100 - index
        row["triangle_count"] = 40
    assert _compare(coarse, refined)["faceting_convergence_gate_pass"] is True


def test_topology_or_identity_drift_is_unqualified():
    refined = _snapshot("refined")
    refined["semantic_topology_signature"] = {
        "adjacencies": [["b", "EXTERIOR"]]
    }
    assert (
        _compare(refined=refined)["status"] == "UNQUALIFIED_SEMANTIC_MAPPING"
    )
    refined = _snapshot("refined")
    refined["physical_input_identity"]["step"] = "0" * 64
    assert (
        _compare(refined=refined)["status"]
        == "UNQUALIFIED_SOURCE_CAD_IDENTITY"
    )


def test_strict_clearance_margin_rejects_equality():
    report = _compare(
        source_cad_clearance={
            "source_cad_gate_pass": True,
            "minimum_clearance_cm": 5.0,
        },
        refined_deviation=_deviation(_snapshot("refined"), 2.5),
    )
    assert report["gates"]["strict_clearance_margin_pass"] is False
    assert report["status"] == "FACETING_CONVERGENCE_FAIL"


def test_sampled_or_incomplete_deviation_is_unqualified():
    deviation = _deviation(_snapshot("refined"))
    deviation["certified_upper_bound"] = False
    report = _compare(refined_deviation=deviation)
    assert report["status"] == "UNQUALIFIED_REFINED_FACET_DEVIATION"


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual_maximum_cover_radius_cm", 0.250001),
        ("covered_triangle_count", 99),
        ("projection_target_semantic_match_pass", False),
        ("triangle_id_digest_sha256", "bad"),
    ],
)
def test_deviation_certificate_rejects_incomplete_or_unbound_proof(
    field, value
):
    refined = _snapshot("refined")
    deviation = _deviation(refined)
    deviation[field] = value
    report = _compare(refined=refined, refined_deviation=deviation)
    assert report["gates"]["certified_deviation_evidence_pass"] is False


def test_deviation_certificate_rejects_upper_bound_arithmetic_drift():
    refined = _snapshot("refined")
    deviation = _deviation(refined)
    deviation["maximum_projected_distance_cm"] += 0.01
    assert (
        _compare(refined=refined, refined_deviation=deviation)["status"]
        == "UNQUALIFIED_REFINED_FACET_DEVIATION"
    )


def test_former_zone_requires_exact_public_witness_binding_and_both_roles():
    zones = _zones()
    zones[0]["witness_cm"][0] += 1.0e-6
    assert (
        _compare(former_overlap_zones=zones)["gates"][
            "former_overlap_zone_refinement_pass"
        ]
        is False
    )
    zones = _zones()
    zones[0]["refined"]["vac_vessel"]["covered"] = False
    assert (
        _compare(former_overlap_zones=zones)["gates"][
            "former_overlap_zone_refinement_pass"
        ]
        is False
    )


def test_native_overlap_or_missing_zone_prevents_pass():
    refined = _snapshot("refined")
    refined["native_qualification"]["overlap_checks"][2][
        "terminal_overlap_location_count"
    ] = 1
    assert _compare(refined=refined)["status"] == "UNQUALIFIED_INPUT_EVIDENCE"
    assert (
        _compare(former_overlap_zones=_zones()[:-1])[
            "faceting_convergence_gate_pass"
        ]
        is False
    )


def test_missing_magnet_or_open_envelope_prevents_pass():
    refined = _snapshot("refined")
    refined["semantic_volumes"][-1]["envelope_pass"] = False
    assert (
        _compare(refined=refined)["gates"]["all_magnet_envelopes_pass"]
        is False
    )
    missing = copy.deepcopy(refined)
    missing["semantic_volumes"].pop()
    assert (
        _compare(refined=missing)["gates"]["semantic_volume_identity_pass"]
        is False
    )
