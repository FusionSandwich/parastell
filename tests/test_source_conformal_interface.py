import json

import numpy as np
import pytest

from parastell.source_conformal_interface import (
    P4_REGION_ORDER,
    audit_shared_interface_topology,
    audit_source_conformal_containment_arrays,
    build_targeted_p4_plan,
    extract_source_master_interface,
    summarize_p4_results,
    write_compact_json_create_only,
)


def _quarter_sector_source():
    # Two tetrahedra fill a pyramid whose y=0 and x=0 faces are the two
    # periodic sector seams.  All remaining faces are the curved-interface
    # analogue used by this analytic connectivity fixture.
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    tetrahedra = np.asarray([[0, 1, 2, 3], [0, 1, 3, 4]])
    coordinates = vertices[tetrahedra]
    volumes = (
        np.abs(
            np.linalg.det(
                np.stack(
                    (
                        coordinates[:, 1] - coordinates[:, 0],
                        coordinates[:, 2] - coordinates[:, 0],
                        coordinates[:, 3] - coordinates[:, 0],
                    ),
                    axis=1,
                )
            )
        )
        / 6.0
    )
    strengths = np.asarray([3.0, 5.0])
    return vertices, tetrahedra, volumes, strengths


def _master_and_shared():
    vertices, tetrahedra, _, _ = _quarter_sector_source()
    master = extract_source_master_interface(vertices, tetrahedra)
    shared = audit_shared_interface_topology(
        master,
        shared_triangles=master["interface_triangles"],
        surface_owners=[
            ["chamber", "first_wall"] for _ in master["interface_triangles"]
        ],
    )
    return master, shared


def test_master_interface_comes_only_from_source_connectivity_and_seams_close():
    vertices, tetrahedra, _, _ = _quarter_sector_source()
    original = vertices.copy()
    master = extract_source_master_interface(vertices, tetrahedra)

    assert np.array_equal(vertices, original)
    assert master["coordinates_modified"] is False
    assert master["source_boundary_triangle_count"] == 6
    assert master["periodic_seam_triangle_count"] == 2
    assert master["interface_triangle_count"] == 4
    assert master["periodic_seam_open_edge_counts"] == [3, 3]
    assert master["sector_seam_continuity_pass"] is True
    assert master["source_master_interface_pass"] is True
    assert len(master["master_interface_sha256"]) == 64
    assert np.allclose(
        np.linalg.norm(master["interface_normal_unit_vectors"], axis=1), 1.0
    )


def test_shared_interface_requires_exact_connectivity_owners_and_opposite_senses():
    master, shared = _master_and_shared()
    assert shared["shared_interface_topology_pass"] is True

    changed = np.asarray(master["interface_triangles"], dtype=int)
    changed[0, 0] = 99
    rejected = audit_shared_interface_topology(
        master,
        shared_triangles=changed,
        surface_owners=[["chamber", "first_wall"] for _ in changed],
    )
    assert rejected["gates"]["shared_connectivity_exact_pass"] is False
    assert rejected["shared_interface_topology_pass"] is False

    reversed_normal = np.asarray(master["interface_triangles"], dtype=int)
    reversed_normal[0, [1, 2]] = reversed_normal[0, [2, 1]]
    rejected_normal = audit_shared_interface_topology(
        master,
        shared_triangles=reversed_normal,
        surface_owners=[["chamber", "first_wall"] for _ in reversed_normal],
    )
    assert rejected_normal["gates"]["shared_connectivity_exact_pass"] is True
    assert rejected_normal["gates"]["outward_normal_orientation_pass"] is False
    assert rejected_normal["shared_interface_topology_pass"] is False

    duplicate_owner = audit_shared_interface_topology(
        master,
        shared_triangles=master["interface_triangles"],
        surface_owners=[
            ["chamber", "chamber"] for _ in master["interface_triangles"]
        ],
    )
    assert duplicate_owner["gates"]["surface_ownership_pass"] is False
    assert duplicate_owner["shared_interface_topology_pass"] is False

    same_sense = audit_shared_interface_topology(
        master,
        shared_triangles=master["interface_triangles"],
        surface_owners=[
            ["chamber", "first_wall"] for _ in master["interface_triangles"]
        ],
        chamber_sense=1,
        first_wall_sense=1,
    )
    assert same_sense["gates"]["opposite_surface_senses_pass"] is False


def test_containment_is_connectivity_based_and_preserves_source_strength():
    vertices, tetrahedra, volumes, strengths = _quarter_sector_source()
    master, shared = _master_and_shared()
    report = audit_source_conformal_containment_arrays(
        vertices,
        tetrahedra,
        volumes,
        strengths,
        reference_vertices_cm=vertices[::-1],
        reference_tetrahedra=np.asarray(
            [
                [4, 3, 1, 0],
                [4, 3, 2, 1],
            ]
        ),
        reference_tagged_volumes_cm3=volumes[::-1],
        reference_source_strengths_n_per_s=strengths[::-1],
        master_interface=master,
        shared_interface=shared,
    )

    assert report["constructional_containment_pass"] is True
    assert report["source_semantic_identity_pass"] is True
    assert report["source_strength_relative_difference"] == 0.0
    assert report["vertex_occurrence_classification"]["OUTSIDE"] == 0
    assert report["quadrature_classification"]["OUTSIDE"] == 0
    assert report["distance_only_interface_classification_allowed"] is False
    assert report["source_projection_used"] is False
    assert report["source_rescaling_used"] is False
    # Analytic construction proof alone is never a physical transport gate.
    assert report["physical_candidate"] is False
    assert report["transport_eligible"] is False


def test_containment_fails_source_strength_invariance_without_rescaling():
    vertices, tetrahedra, volumes, strengths = _quarter_sector_source()
    master, shared = _master_and_shared()
    report = audit_source_conformal_containment_arrays(
        vertices,
        tetrahedra,
        volumes,
        strengths * 1.0001,
        reference_vertices_cm=vertices,
        reference_tetrahedra=tetrahedra,
        reference_tagged_volumes_cm3=volumes,
        reference_source_strengths_n_per_s=strengths,
        master_interface=master,
        shared_interface=shared,
    )
    assert report["gates"]["source_strength_invariance_pass"] is False
    assert report["constructional_containment_pass"] is False
    assert report["source_strengths_modified"] is False
    assert report["transport_eligible"] is False


def test_physical_eligibility_requires_all_explicit_successor_inputs():
    vertices, tetrahedra, volumes, strengths = _quarter_sector_source()
    master, shared = _master_and_shared()
    report = audit_source_conformal_containment_arrays(
        vertices,
        tetrahedra,
        volumes,
        strengths,
        reference_vertices_cm=vertices,
        reference_tetrahedra=tetrahedra,
        reference_tagged_volumes_cm3=volumes,
        reference_source_strengths_n_per_s=strengths,
        master_interface=master,
        shared_interface=shared,
        physical_candidate_evidence={
            "schema": "parastell.source_conformal_physical_candidate/v1.0.0",
            "input_class": "PHYSICAL",
            "master_interface_sha256": master["master_interface_sha256"],
            "dagmc_h5m_sha256": "a" * 64,
            "outer_layer_mapping_sha256": "b" * 64,
            "material_identity_sha256": "c" * 64,
            "full_geometry_topology_pass": True,
            "fixture_only": False,
        },
    )
    assert report["transport_eligible"] is True

    incomplete = audit_source_conformal_containment_arrays(
        vertices,
        tetrahedra,
        volumes,
        strengths,
        reference_vertices_cm=vertices,
        reference_tetrahedra=tetrahedra,
        reference_tagged_volumes_cm3=volumes,
        reference_source_strengths_n_per_s=strengths,
        master_interface=master,
        shared_interface=shared,
        physical_candidate_evidence={"input_class": "PHYSICAL"},
    )
    assert incomplete["physical_candidate"] is False
    assert incomplete["transport_eligible"] is False


def test_p4_plan_is_precision_four_partitioned_and_resumable():
    plan = build_targeted_p4_plan(
        geometry_sha256="a" * 64,
        interface_sha256="b" * 64,
        timeout_seconds_per_region=120,
        region_bounds_cm={"SOURCE_CHAMBER_INTERFACE": [0, 0, 0, 1, 1, 1]},
    )
    assert [row["region_id"] for row in plan["regions"]] == list(
        P4_REGION_ORDER
    )
    assert all(row["precision"] == 4 for row in plan["regions"])

    partial = summarize_p4_results(
        plan,
        [
            {
                "region_id": "SOURCE_CHAMBER_INTERFACE",
                "precision": 4,
                "status": "PASS",
                "overlap_count": 0,
                "geometry_sha256": plan["geometry_sha256"],
                "master_interface_sha256": plan["master_interface_sha256"],
                "plan_sha256": plan["canonical_plan_sha256"],
            },
            {
                "region_id": "SECTOR_SEAMS",
                "precision": 4,
                "status": "TIMEOUT",
                "geometry_sha256": plan["geometry_sha256"],
                "master_interface_sha256": plan["master_interface_sha256"],
                "plan_sha256": plan["canonical_plan_sha256"],
            },
        ],
    )
    assert partial["status"] == "BLOCKED_RUNTIME"
    assert partial["timeout_is_physics_failure"] is False
    assert partial["targeted_p4_pass"] is False
    assert partial["global_p4_allowed"] is False
    assert partial["completed_region_count"] == 2

    with pytest.raises(ValueError, match="not hash-bound"):
        summarize_p4_results(
            plan,
            [
                {
                    "region_id": "SOURCE_CHAMBER_INTERFACE",
                    "precision": 4,
                    "status": "PASS",
                }
            ],
        )

    complete = summarize_p4_results(
        plan,
        [
            {
                "region_id": region,
                "precision": 4,
                "status": "PASS",
                "geometry_sha256": plan["geometry_sha256"],
                "master_interface_sha256": plan["master_interface_sha256"],
                "plan_sha256": plan["canonical_plan_sha256"],
            }
            for region in P4_REGION_ORDER
        ],
    )
    assert complete["status"] == "PASS"
    assert complete["targeted_p4_pass"] is True
    assert complete["global_p4_allowed"] is True


def test_compact_receipts_are_create_only_and_strict_json(tmp_path):
    path = tmp_path / "receipt.json"
    identity = write_compact_json_create_only(
        path, {"status": "BLOCKED_INPUT"}
    )
    assert len(identity["sha256"]) == 64
    assert json.loads(path.read_text())["status"] == "BLOCKED_INPUT"
    with pytest.raises(FileExistsError):
        write_compact_json_create_only(path, {"status": "PASS"})
