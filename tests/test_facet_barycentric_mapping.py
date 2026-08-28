import numpy as np
import pytest

from parastell.dagmc_envelope import DagmcEnvelope
from parastell.dagmc_envelope import FacetedSurface


def _surface():
    return FacetedSurface(
        surface_id=29,
        role="plasma_facing",
        triangles_cm=np.asarray(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        ),
        triangle_normals_outward=np.asarray([[0.0, 0.0, 1.0]]),
        triangle_areas_cm2=np.asarray([0.5]),
        centroid_global_cm=np.asarray([1.0 / 3.0, 1.0 / 3.0, 0.0]),
        canonical_facet_ids=("facet-stable",),
        dagmc_volume_id=8,
    )


@pytest.mark.parametrize(
    ("point", "classification", "barycentric"),
    [
        ([0.2, 0.3, 1.0e-8], "EXACT_FACET_MATCH", [0.5, 0.2, 0.3]),
        ([0.5, 0.5, -1.0e-8], "EDGE_TOLERANCE_MATCH", [0.0, 0.5, 0.5]),
        ([1.0, 0.0, 0.0], "VERTEX_TOLERANCE_MATCH", [0.0, 1.0, 0.0]),
    ],
)
def test_facet_vertex_edge_interior_mapping(
    point, classification, barycentric
):
    result = _surface().locate(point)

    assert result["mapping_status"] == classification
    assert result["canonical_facet_id"] == "facet-stable"
    assert result["inside_facet"]
    assert result["barycentric_coordinates"] == pytest.approx(barycentric)
    reconstructed = result["reconstructed_position_global_cm"]
    assert reconstructed == pytest.approx([point[0], point[1], 0.0])
    assert result["signed_plane_residual_cm"] == pytest.approx(point[2])
    assert result["nearest_point_residual_cm"] == pytest.approx(abs(point[2]))


def test_invalid_point_does_not_receive_a_nearest_centroid_facet_id():
    result = _surface().locate([4.0, 4.0, 0.0])

    assert result["mapping_status"] == "NO_VALID_FACET_MATCH"
    assert result["canonical_facet_id"] == ""
    assert result["facet_index"] == -1
    assert not result["inside_facet"]
    assert sum(result["barycentric_coordinates"]) == pytest.approx(1.0)
    assert result["nearest_point_residual_cm"] > 1.0


def test_small_physical_edge_offset_is_an_edge_tolerance_match():
    result = _surface().locate(
        [0.500002, 0.500002, 0.0], source_tolerance_cm=1.0e-5
    )

    assert result["mapping_status"] == "EDGE_TOLERANCE_MATCH"
    assert result["canonical_facet_id"] == "facet-stable"
    assert result["nearest_point_residual_cm"] < 1.0e-5


def test_production_mapping_rejects_an_unmatched_record():
    envelope = DagmcEnvelope(None, (_surface(),), 3, 0, 0.0)
    with pytest.raises(ValueError, match="NO_VALID_FACET_MATCH"):
        envelope.facet_mappings([29], [[4.0, 4.0, 0.0]], require_valid=True)


def test_outward_normal_and_one_sided_plane_residual_are_consistent():
    above = _surface().locate([0.2, 0.2, 2.0e-6])
    below = _surface().locate([0.2, 0.2, -2.0e-6])

    assert above["signed_plane_residual_cm"] > 0.0
    assert below["signed_plane_residual_cm"] < 0.0
    assert above["outward_normal_global"] == pytest.approx([0.0, 0.0, 1.0])


def test_noncoplanar_shared_edge_is_rejected_without_native_facet_id():
    surface = FacetedSurface(
        surface_id=29,
        role="plasma_facing",
        triangles_cm=np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            ]
        ),
        triangle_normals_outward=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        ),
        triangle_areas_cm2=np.asarray([0.5, 0.5]),
        centroid_global_cm=np.asarray([1.0 / 3.0, 1.0 / 6.0, -1.0 / 6.0]),
        canonical_facet_ids=("xy", "xz"),
        dagmc_volume_id=8,
    )
    with pytest.raises(ValueError, match="ambiguous noncoplanar facet edge"):
        surface.locate([0.25, 0.0, 0.0])
