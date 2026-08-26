from types import SimpleNamespace

import numpy as np

from parastell.magnet_surface_audit import audit_surface_subset


def test_tetrahedral_volume_closes_and_face_subset_does_not():
    volume = SimpleNamespace(id=1)
    outside = SimpleNamespace(id=2)
    points = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    volume.surfaces = tuple(
        SimpleNamespace(
            id=index + 1,
            forward_volume=volume,
            reverse_volume=outside,
            triangle_coords=points[list(face)],
        )
        for index, face in enumerate(faces)
    )
    closed = audit_surface_subset(volume, (1, 2, 3, 4))
    open_subset = audit_surface_subset(volume, (1, 2, 3))
    assert closed.closed
    assert closed.bad_edge_count == 0
    assert closed.vector_area_closure_relative == 0.0
    assert not open_subset.closed
    assert open_subset.bad_edge_count == 3
