import numpy as np
import pytest

from parastell.cad_radial_clearance import first_triangle_hit_distances
from parastell.cad_radial_clearance import (
    project_dense_deficit_to_control_grid,
)


def test_first_triangle_hit_returns_nearest_positive_distance():
    vertices = np.array(
        [
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [0.0, 1.0, 2.0],
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [0.0, 1.0, 5.0],
        ]
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    distances = first_triangle_hit_distances(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 4.0]]),
        vertices,
        triangles,
        maximum_distance_cm=10.0,
    )
    np.testing.assert_allclose(distances, [2.0])


def test_first_triangle_hit_reports_no_hit_as_positive_infinity():
    distances = first_triangle_hit_distances(
        np.array([[2.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]]),
        np.array([[0, 1, 2]]),
        maximum_distance_cm=10.0,
    )
    assert np.isposinf(distances[0])


def test_first_triangle_hit_rejects_invalid_triangle_index():
    with pytest.raises(ValueError, match="out of range"):
        first_triangle_hit_distances(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0]]),
            np.zeros((3, 3)),
            np.array([[0, 1, 3]]),
            maximum_distance_cm=10.0,
        )


def test_dense_deficit_projection_is_conservative_and_closes_poloidally():
    dense = np.zeros((5, 5))
    dense[2, 2] = 3.0
    dense[1, 0] = 2.0
    projected = project_dense_deficit_to_control_grid(
        dense,
        np.linspace(0.0, 90.0, 5),
        np.linspace(0.0, 360.0, 5),
        np.linspace(0.0, 90.0, 3),
        np.linspace(0.0, 360.0, 3),
    )
    assert np.max(projected) == 3.0
    assert np.count_nonzero(projected == 3.0) == 1
    assert np.count_nonzero(projected == 2.0) == 4
    np.testing.assert_array_equal(projected[:, 0], projected[:, -1])
