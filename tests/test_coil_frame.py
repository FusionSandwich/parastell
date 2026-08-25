import numpy as np
import pytest

from parastell.coil_frame import FRAME_TYPE, parallel_transport_frame


def test_periodic_parallel_transport_frame_is_continuous_and_right_handed():
    angle = np.linspace(0.0, 2.0 * np.pi, 65)
    points = np.column_stack(
        (100.0 * np.cos(angle), 100.0 * np.sin(angle), 5.0 * np.sin(2 * angle))
    )
    frame = parallel_transport_frame(points)
    assert len(frame.points_cm) == 64
    assert frame.total_length_cm > 600.0
    assert np.allclose(
        np.cross(frame.tangents, frame.radial_directions),
        frame.transverse_directions,
    )
    assert np.allclose(
        np.sum(frame.tangents * frame.radial_directions, axis=1), 0.0
    )
    sample = frame.sample([[100.5, 0.0, 0.0], [0.0, 99.5, 0.0]])
    assert sample["frame_type"] == FRAME_TYPE
    assert np.all(sample["normalized_arclength"] >= 0.0)
    assert np.all(sample["normalized_arclength"] <= 1.0)
    assert np.all(sample["distance_to_centreline_cm"] < 2.0)


def test_parallel_transport_rejects_duplicate_points():
    with pytest.raises(ValueError, match="duplicate adjacent"):
        parallel_transport_frame([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_empty_frame_sample_preserves_vector_shapes():
    angle = np.linspace(0.0, 2.0 * np.pi, 33)
    frame = parallel_transport_frame(
        np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    )
    sampled = frame.sample(np.empty((0, 3)))
    assert sampled["centreline_tangent"].shape == (0, 3)
    assert sampled["local_centreline_coordinates_cm"].shape == (0, 3)
    assert sampled["centreline_arclength_cm"].shape == (0,)
