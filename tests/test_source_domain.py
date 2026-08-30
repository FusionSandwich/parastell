import numpy as np

from parastell.source_domain import SOURCE_QUADRATURE_BARYCENTRICS
from parastell.source_domain import _ClosedSurface
from parastell.source_domain import _remove_periodic_cut_face_vertex_exemptions
from parastell.source_domain import audit_source_tetrahedra_arrays
from parastell.source_domain import periodic_cut_face_triangles
from parastell.source_domain import periodic_cut_face_vertex_mask
from parastell.source_domain import select_source_volume


def test_closed_surface_vtk_fallback_classifies_inside_and_outside_points():
    triangles = np.asarray(
        [
            [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ],
        dtype=float,
    )
    surface = _ClosedSurface(triangles)
    assert surface.n_open_edges == 0
    selected, distance = surface.classify(
        np.asarray([[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]])
    )
    assert selected.tolist() == [1, 0]
    assert distance[0] == 0.0
    assert distance[1] != 0.0

    selected, distance = surface.classify(
        np.asarray([[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]]),
        compute_all_distances=True,
    )
    assert selected.tolist() == [1, 0]
    assert distance[0] != 0.0
    assert distance[1] != 0.0


def _two_tetrahedra():
    first = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    second = first + np.asarray([2.0, 0.0, 0.0])
    return np.stack((first, second))


def test_source_tetrahedron_audit_is_order_independent_and_positive():
    tetrahedra = _two_tetrahedra()
    volumes = np.asarray([1.0 / 6.0, 1.0 / 6.0])
    strengths = np.asarray([3.0, 5.0])
    forward = audit_source_tetrahedra_arrays(tetrahedra, volumes, strengths)
    reverse = audit_source_tetrahedra_arrays(
        tetrahedra[::-1], volumes[::-1], strengths[::-1]
    )

    assert forward["tetrahedron_data_gate_pass"] is True
    assert forward["invalid_tetrahedron_count"] == 0
    assert (
        forward["selected_tetrahedron"]["canonical_payload_sha256"]
        == reverse["selected_tetrahedron"]["canonical_payload_sha256"]
    )
    assert (
        forward["selected_source_point_cm"]
        == reverse["selected_source_point_cm"]
    )


def test_source_tetrahedron_audit_rejects_bad_tags_and_negative_strength():
    tetrahedra = _two_tetrahedra()
    report = audit_source_tetrahedra_arrays(
        tetrahedra,
        np.asarray([1.0 / 6.0, 1.0]),
        np.asarray([1.0, -1.0]),
    )

    assert report["tetrahedron_data_gate_pass"] is False
    assert report["tagged_volume_mismatch_count"] == 1
    assert report["negative_source_strength_count"] == 1
    assert report["invalid_tetrahedron_count"] == 1


def test_source_quadrature_matches_parastell_five_point_rule():
    assert SOURCE_QUADRATURE_BARYCENTRICS.shape == (5, 4)
    assert np.allclose(SOURCE_QUADRATURE_BARYCENTRICS.sum(axis=1), 1.0)
    assert np.all(SOURCE_QUADRATURE_BARYCENTRICS > 0.0)
    assert np.allclose(
        SOURCE_QUADRATURE_BARYCENTRICS[0], [0.25, 0.25, 0.25, 0.25]
    )


def test_source_volume_selection_requires_exact_id_not_unique_material():
    class Volume:
        def __init__(self, material):
            self.material = material

    volumes = {1: Volume("Vacuum"), 7: Volume("Vacuum")}
    assert select_source_volume(volumes, 1, "Vacuum") is volumes[1]
    with np.testing.assert_raises_regex(ValueError, "does not exist"):
        select_source_volume(volumes, 2, "Vacuum")
    with np.testing.assert_raises_regex(ValueError, "does not match"):
        select_source_volume(volumes, 7, "breeder")


def _cut_face_patches():
    return (
        np.asarray([[[1, 0, 0], [2, 0, 0], [1, 0, 1]]], dtype=float),
        np.asarray([[[0, 1, 0], [0, 2, 0], [0, 1, 1]]], dtype=float),
    )


def test_periodic_cut_face_mask_requires_actual_triangle_membership():
    points = np.asarray(
        [
            [1.25, 0.0, 0.25],
            [0.0, 1.25, 0.25],
            [1.0e6, 0.0, 0.25],
            [1.0, 1.0, 0.0],
        ]
    )
    assert periodic_cut_face_vertex_mask(
        points, _cut_face_patches(), tolerance_cm=1.0e-8
    ).tolist() == [True, True, False, False]


def test_periodic_cut_faces_are_extracted_from_source_volume_triangles():
    class Surface:
        triangle_coords = np.concatenate(
            (
                _cut_face_patches()[0],
                _cut_face_patches()[1],
                np.asarray([[[1, 1, 0], [2, 1, 0], [1, 2, 1]]]),
            )
        )

    class Volume:
        surfaces = [Surface()]

    patches = periodic_cut_face_triangles(
        Volume(), [0.0, 90.0], tolerance_cm=1.0e-8
    )
    assert [len(patch) for patch in patches] == [1, 1]


def test_periodic_cut_face_exemption_never_removes_quadrature_points():
    points = np.asarray([[1.25, 0.0, 0.25], [1.25, 0.0, 0.25]], dtype=float)
    retained, exempted = _remove_periodic_cut_face_vertex_exemptions(
        np.asarray([0, 1]),
        1,
        points,
        _cut_face_patches(),
        tolerance_cm=1.0e-8,
    )
    assert exempted == 1
    assert retained.tolist() == [1]
