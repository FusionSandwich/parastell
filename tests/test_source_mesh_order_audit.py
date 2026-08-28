import numpy as np
import pytest

from parastell.source_mesh_order_audit import (
    audit_source_order_arrays,
    audit_tetrahedral_seam_topology,
    deduplicate_half_period_seam,
)


def test_source_order_requires_exact_vector_order_and_rate():
    report = audit_source_order_arrays(
        [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], expected_rate_per_s=6.0
    )
    assert report["element_order_exact_match"] is True
    assert report["source_rate_closure_pass"] is True
    assert report["order_and_rate_gate_pass"] is True


def test_source_order_rejects_permutation_despite_equal_sum():
    report = audit_source_order_arrays(
        [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], expected_rate_per_s=6.0
    )
    assert report["element_order_exact_match"] is False
    assert report["source_rate_closure_pass"] is True
    assert report["order_and_rate_gate_pass"] is False


@pytest.mark.parametrize("bad", [[1.0, np.nan], [1.0, -1.0]])
def test_source_order_rejects_nonfinite_and_negative_values(bad):
    with pytest.raises(ValueError):
        audit_source_order_arrays(bad, bad, expected_rate_per_s=1.0)


def test_half_period_expansion_merges_only_shared_seam():
    root_two = np.sqrt(2.0)
    original = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
    mate = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 2.0]])
    unique, indices, audit = deduplicate_half_period_seam(original, mate)
    assert unique.shape == (3, 3)
    assert indices.tolist() == [0, 1, 2, 1]
    assert audit["merged_vertex_count"] == 1
    assert audit["all_merged_vertices_on_seam"] is True
    assert np.isclose(np.hypot(1.0, 1.0), root_two)


def test_half_period_expansion_rejects_nonseam_collision():
    coordinates = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="non-seam collision"):
        deduplicate_half_period_seam(coordinates, coordinates)


def test_source_order_rejects_invalid_expected_rate():
    with pytest.raises(ValueError, match="finite and positive"):
        audit_source_order_arrays([1.0], [1.0], expected_rate_per_s=np.nan)


def test_tetrahedral_seam_topology_rejects_unmatched_internal_face():
    points = np.array(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    report = audit_tetrahedral_seam_topology(points, [[0, 1, 2, 3]])
    assert report["unmatched_internal_seam_face_count"] == 1
    assert report["conformal_internal_seam_pass"] is False
