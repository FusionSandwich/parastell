import numpy as np
import pytest

from parastell.dagmc_envelope import FacetedSurface
from parastell.dagmc_envelope import DagmcEnvelope
from parastell.dagmc_envelope import MagnetVolumeRecord
from parastell.dagmc_envelope import MagnetVolumeInventory
from parastell.dagmc_envelope import _pair_magnet_records
from parastell.dagmc_envelope import select_magnet_pairs
from parastell.coil_frame import parallel_transport_frame


def record(volume_id, role, centroid, bounds, coil_id=""):
    return MagnetVolumeRecord(
        volume_id,
        role,
        role,
        (volume_id + 100,),
        component_name=coil_id,
        component_id=f"component-{volume_id}",
        coil_id=coil_id,
        centroid_global_cm=centroid,
        bounding_box_cm=bounds,
        volume_cm3=10.0,
    )


def test_facet_mapping_retains_barycentric_coordinates_and_residual():
    surface = FacetedSurface(
        7,
        "plasma_facing",
        np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        np.asarray([[0.0, 0.0, 1.0]]),
        np.asarray([0.5]),
        np.asarray([1 / 3, 1 / 3, 0.0]),
        ("facet-stable",),
    )
    result = surface.locate([0.25, 0.25, 2.0e-8])
    assert result["facet_id"] == "facet-stable"
    assert result["mapping_status"] == "CONTAINING_FACET"
    assert result["barycentric_coordinates"] == pytest.approx(
        [0.5, 0.25, 0.25]
    )
    assert result["distance_to_facet_residual_cm"] == pytest.approx(2.0e-8)


def test_facet_catalog_links_canonical_rows_to_centreline_frame():
    surface = FacetedSurface(
        7,
        "plasma_facing",
        np.asarray(
            [
                [[10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]],
                [[10.0, 1.0, 0.0], [10.0, 1.0, 1.0], [10.0, 0.0, 1.0]],
            ]
        ),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([0.5, 0.5]),
        np.asarray([10.0, 0.5, 0.5]),
        ("facet-z", "facet-a"),
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 65)
    frame = parallel_transport_frame(
        np.column_stack(
            (10 * np.cos(angle), 10 * np.sin(angle), np.zeros_like(angle))
        )
    )
    envelope = DagmcEnvelope(None, (surface,), 5, 0, 0.0, frame)

    metadata = envelope.facet_metadata()

    assert metadata["facet_id"].tolist() == ["facet-a", "facet-z"]
    assert metadata["facet_index"].tolist() == [1, 0]
    assert metadata["centreline_linkage_available"]
    assert metadata["centreline_arclength_cm"].shape == (2,)
    assert metadata["normalized_arclength"].shape == (2,)
    assert metadata["distance_to_centreline_cm"].shape == (2,)
    assert np.allclose(
        np.cross(
            metadata["centreline_tangent"], metadata["centreline_radial"]
        ),
        metadata["centreline_transverse"],
    )
    assert np.all(metadata["frame_quality_status"] == frame.quality_status)


def test_facet_catalog_marks_unavailable_centreline_without_fake_values():
    surface = FacetedSurface(
        2,
        "other",
        np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        np.asarray([[0.0, 0.0, 1.0]]),
        np.asarray([0.5]),
        np.asarray([1 / 3, 1 / 3, 0.0]),
        ("facet-only",),
    )
    metadata = DagmcEnvelope(None, (surface,), 3, 0, 0.0).facet_metadata()

    assert not metadata["centreline_linkage_available"]
    assert metadata["centreline_linkage_status"].tolist() == [
        "UNAVAILABLE_CENTRELINE_FRAME_NOT_SUPPLIED"
    ]
    assert "centreline_arclength_cm" not in metadata
    assert "frame_quality_status" not in metadata


def test_winding_pack_casing_pairing_and_selection_are_deterministic():
    pack_a = record(
        16, "winding_pack", (0, 0, 0), ((-1, -1, -1), (1, 1, 1)), "A"
    )
    pack_b = record(
        22, "winding_pack", (10, 0, 0), ((9, -1, -1), (11, 1, 1)), "B"
    )
    casing_a = record(
        15, "magnet_casing", (0, 0, 0), ((-2, -2, -2), (2, 2, 2))
    )
    casing_b = record(
        21, "magnet_casing", (10, 0, 0), ((8, -2, -2), (12, 2, 2))
    )
    pairs = _pair_magnet_records((pack_b, pack_a), (casing_b, casing_a))
    assert [(pair.coil_id, pair.casing.volume_id) for pair in pairs] == [
        ("A", 15),
        ("B", 21),
    ]
    inventory = MagnetVolumeInventory(
        "model.h5m",
        "0" * 64,
        (pack_a, casing_a, pack_b, casing_b),
        "1" * 64,
        pairs,
    )
    assert (
        select_magnet_pairs(inventory, "magnet-B")[0].winding_pack.volume_id
        == 22
    )
    assert select_magnet_pairs(inventory, 16)[0].magnet_id == "magnet-A"
    with pytest.raises(ValueError, match="resolved to 0"):
        select_magnet_pairs(inventory, "missing")
