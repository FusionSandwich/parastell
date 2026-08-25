import numpy as np

from parastell.coil_frame import parallel_transport_frame
from parastell.magnet_local_mesh import LocalMeshDefinition
from parastell.magnet_local_mesh import build_local_mesh_definition
from parastell.magnet_local_mesh import qualify_local_mesh_nonoverlap


def test_local_mesh_centroids_volumes_and_frame_identity():
    angle = np.linspace(0.0, 2.0 * np.pi, 33)
    frame = parallel_transport_frame(
        np.column_stack(
            (10 * np.cos(angle), 10 * np.sin(angle), np.zeros_like(angle))
        )
    )
    sample = frame.sample([10.0, 0.0, 0.0])
    mesh = build_local_mesh_definition(
        "magnet-A",
        bounding_box_global_cm=[[8.0, -1.0, -1.0], [12.0, 1.0, 1.0]],
        centreline_sample=sample,
        resolution_cm=1.0,
    )
    metadata = mesh.bin_metadata()
    first_x_sweep = metadata["indices_ijk"][: mesh.dimension[0]]
    assert first_x_sweep.tolist() == [
        [index, 0, 0] for index in range(mesh.dimension[0])
    ]
    if mesh.dimension[1] > 1:
        assert metadata["indices_ijk"][mesh.dimension[0]].tolist() == [
            0,
            1,
            0,
        ]
    assert len(metadata["mesh_bin_ids"]) == mesh.bin_count
    assert np.all(metadata["volume_cm3"] > 0.0)
    assert metadata["global_centroid_cm"].shape == (mesh.bin_count, 3)
    assert mesh.to_dict()["frame_type"] == "coil_centerline_parallel_transport"


def test_local_mesh_bins_link_to_nearest_continuous_centreline_frame():
    angle = np.linspace(0.0, 2.0 * np.pi, 65)
    frame = parallel_transport_frame(
        np.column_stack(
            (20 * np.cos(angle), 20 * np.sin(angle), np.zeros_like(angle))
        )
    )
    sample = frame.sample([20.0, 0.0, 0.0])
    mesh = build_local_mesh_definition(
        "magnet-linked",
        bounding_box_global_cm=[[18.0, -1.0, -1.0], [22.0, 1.0, 1.0]],
        centreline_sample=sample,
        resolution_cm=1.0,
    )

    metadata = mesh.bin_metadata(frame)
    independently_sampled = frame.sample(metadata["global_centroid_cm"])

    assert metadata["centreline_linkage_available"]
    assert np.all(
        metadata["centreline_linkage_status"]
        == "LINKED_NEAREST_CENTRELINE_SEGMENT"
    )
    assert np.allclose(
        metadata["centreline_arclength_cm"],
        independently_sampled["centreline_arclength_cm"],
    )
    assert np.allclose(
        metadata["local_centreline_coordinates_cm"],
        independently_sampled["local_centreline_coordinates_cm"],
    )
    assert np.all(metadata["normalized_arclength"] >= 0.0)
    assert np.all(metadata["normalized_arclength"] <= 1.0)
    assert np.allclose(
        np.cross(
            metadata["centreline_tangent"], metadata["centreline_radial"]
        ),
        metadata["centreline_transverse"],
    )
    assert metadata["mesh_local_centroid_cm"].shape == (mesh.bin_count, 3)
    assert np.all(
        metadata["frame_type"] == "coil_centerline_parallel_transport"
    )


def test_local_mesh_does_not_fabricate_missing_centreline_linkage():
    angle = np.linspace(0.0, 2.0 * np.pi, 33)
    frame = parallel_transport_frame(
        np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    )
    mesh = build_local_mesh_definition(
        "magnet-unlinked",
        bounding_box_global_cm=[[-2.0, -2.0, -1.0], [2.0, 2.0, 1.0]],
        centreline_sample=frame.sample([1.0, 0.0, 0.0]),
        resolution_cm=2.0,
    )

    metadata = mesh.bin_metadata()

    assert not metadata["centreline_linkage_available"]
    assert np.all(
        metadata["centreline_linkage_status"]
        == "UNAVAILABLE_CENTRELINE_FRAME_NOT_SUPPLIED"
    )
    assert "centreline_arclength_cm" not in metadata
    assert "distance_to_centreline_cm" not in metadata


def _identity_mesh(magnet_id, translation):
    return LocalMeshDefinition(
        magnet_id=magnet_id,
        lower_left_local_cm=(-1.0, -1.0, -1.0),
        upper_right_local_cm=(1.0, 1.0, 1.0),
        dimension=(1, 1, 1),
        rotation_local_to_global=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_global_cm=translation,
        requested_resolution_cm=2.0,
    )


def test_nonoverlap_requires_both_geometric_proof_and_component_filter():
    separated = (
        _identity_mesh("magnet-A", (0.0, 0.0, 0.0)),
        _identity_mesh("magnet-B", (10.0, 0.0, 0.0)),
    )
    unfiltered = qualify_local_mesh_nonoverlap(
        separated, cell_filter_applied=False
    )
    assert unfiltered["geometric_pairwise_disjoint_proven"]
    assert not unfiltered["nonoverlap_qualified"]
    assert unfiltered["blocking_reasons"] == [
        "LOCAL_MESH_TALLIES_ARE_NOT_COMPONENT_FILTERED"
    ]

    filtered = qualify_local_mesh_nonoverlap(
        separated, cell_filter_applied=True
    )
    assert filtered["nonoverlap_qualified"]
    assert filtered["status"] == "QUALIFIED"

    overlapping = (
        separated[0],
        _identity_mesh("magnet-C", (1.0, 0.0, 0.0)),
    )
    unresolved = qualify_local_mesh_nonoverlap(
        overlapping, cell_filter_applied=True
    )
    assert not unresolved["geometric_pairwise_disjoint_proven"]
    assert not unresolved["nonoverlap_qualified"]
    assert unresolved["blocking_reasons"] == [
        "PAIRWISE_GEOMETRIC_DISJOINTNESS_NOT_PROVEN"
    ]
