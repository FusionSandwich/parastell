import numpy as np

from scripts.compare_regenerated_cad_to_dagmc import triangle_metrics


def test_triangle_metrics_for_unit_tetrahedron():
    triangles = np.asarray(
        [
            [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ],
        dtype=float,
    )
    result = triangle_metrics(triangles)

    assert result["volume_cm3"] == np.float64(1.0 / 6.0)
    assert np.allclose(result["volume_centroid_cm"], (0.25, 0.25, 0.25))
    assert np.allclose(result["bounding_box_cm"], ((0, 0, 0), (1, 1, 1)))
