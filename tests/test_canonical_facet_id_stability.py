import numpy as np

from parastell.dagmc_envelope import _canonical_facet_id


def _identity(triangle, *, fingerprint="a" * 64, volume_id=8, surface_id=29):
    return _canonical_facet_id(
        fingerprint,
        volume_id,
        surface_id,
        np.asarray(triangle, dtype=float),
        1,
        1.0e-6,
    )


def test_canonical_facet_id_is_stable_under_cyclic_triangle_ordering():
    triangle = [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 3.0, 3.0]]
    assert _identity(triangle) == _identity(triangle[1:] + triangle[:1])
    assert _identity(triangle) == _identity(triangle[2:] + triangle[:2])


def test_canonical_facet_id_changes_with_geometry_or_lineage():
    triangle = [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 3.0, 3.0]]
    changed = [row[:] for row in triangle]
    changed[2][1] += 2.0e-6
    baseline = _identity(triangle)

    assert _identity(changed) != baseline
    assert _identity(triangle, fingerprint="b" * 64) != baseline
    assert _identity(triangle, volume_id=9) != baseline
    assert _identity(triangle, surface_id=30) != baseline
