import numpy as np

from scripts.benchmark_tier_a_finite_tape_atlas import (
    PACKET_HASHES,
    benchmark,
    finite_tape_fixture,
)


def test_tier_a_finite_tape_fixture_retains_geometry_ratio_and_faces():
    catalog, phase, faces, angles = finite_tape_fixture()
    extents = np.ptp(catalog["vertices_global_cm"].reshape(-1, 3), axis=0)
    assert extents[0] / extents[2] == 1.0e7
    assert len(set(faces)) == 6
    assert set(angles) == {90.0, 45.0, 10.0, 5.0, 1.0}
    assert len(phase["surface_id"]) == 102
    assert all(len(value) == 64 for value in PACKET_HASHES.values())


def test_tier_a_six_face_cache_is_equivalent_and_nonproduction():
    receipt = benchmark(repeats=1)
    assert receipt["all_six_faces_pass"] is True
    assert receipt["all_octants_pass"] is True
    assert receipt["cached_uncached_equivalence_pass"] is True
    assert receipt["incoming_record_count"] == receipt["query_count"]
    assert receipt["grazing_record_count"] == 0
    assert receipt["maximum_reconstruction_residual_cm"] <= 1.0e-15
    assert receipt["minimum_candidate_thickness_um"] == 0.1
    assert receipt["length_to_thickness_ratio"] == 1.0e7
    assert receipt["finite_tape_transport_solve_claimed"] is False
    assert receipt["physical_qualification_claimed"] is False
