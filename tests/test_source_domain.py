import numpy as np

from parastell.source_domain import SOURCE_QUADRATURE_BARYCENTRICS
from parastell.source_domain import audit_source_tetrahedra_arrays


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
