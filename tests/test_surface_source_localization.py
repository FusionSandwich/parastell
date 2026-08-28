import numpy as np
import pytest

from parastell.surface_source_localization import localize_surface_crossings


def _catalog():
    return {
        "facet_id": ["surface-7-facet-a", "surface-7-facet-b"],
        "surface_id": [7, 7],
        "vertices_global_cm": np.asarray(
            [
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[1, 1, 0], [0, 1, 0], [1, 0, 0]],
            ],
            dtype=float,
        ),
        "outward_normal_global": np.asarray([[0, 0, 1], [0, 0, 1]]),
        "normal_source": "dagmc_forward_reverse_topology",
    }


def test_facet_barycentric_normal_mu_and_local_frame_roundtrip():
    phase = {
        "position_global_cm": np.asarray([[0.25, 0.25, 0], [0.75, 0.75, 0]]),
        "direction_global": np.asarray([[0, 0, -1], [0, 0, 1]]),
        "surface_id": np.asarray([7, 7]),
        "energy_eV": np.asarray([14.1e6, 1.0e6]),
    }
    localized, audit = localize_surface_crossings(phase, _catalog())
    assert audit["status"] == "PASS"
    assert audit["incoming_records"] == 1
    assert audit["outgoing_records"] == 1
    assert localized["facet_id"].tolist() == [
        "surface-7-facet-a",
        "surface-7-facet-b",
    ]
    assert np.all(localized["barycentric_coordinates"] >= -1.0e-12)
    assert np.allclose(
        localized["reconstructed_position_global_cm"],
        phase["position_global_cm"],
    )
    assert np.array_equal(localized["mu"], [-1.0, 1.0])
    assert localized["crossing_sense"].tolist() == ["incoming", "outgoing"]
    for frame in localized["local_frame_global"]:
        assert np.allclose(np.cross(frame[0], frame[1]), frame[2])
    assert np.allclose(localized["direction_local"][:, 2], [-1.0, 1.0])


def test_topology_normal_source_and_containment_fail_closed():
    phase = {
        "position_global_cm": np.asarray([[0.25, 0.25, 0]]),
        "direction_global": np.asarray([[0, 0, -1]]),
        "surface_id": np.asarray([7]),
    }
    catalog = _catalog()
    catalog["normal_source"] = "global_coordinate_heuristic"
    with pytest.raises(ValueError, match="forward/reverse topology"):
        localize_surface_crossings(phase, catalog)

    outside = {**phase, "position_global_cm": np.asarray([[2, 2, 0]])}
    with pytest.raises(ValueError, match="not contained"):
        localize_surface_crossings(outside, _catalog())
