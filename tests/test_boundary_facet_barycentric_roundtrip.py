import numpy as np
import pytest

from parastell.boundary_localization import validate_boundary_localization
from parastell.dagmc_envelope import FacetedSurface


def test_strict_facet_localization_and_barycentric_roundtrip():
    surface = FacetedSurface(
        7,
        "plasma_facing",
        np.asarray([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float),
        np.asarray([[0, 0, 1]], dtype=float),
        np.asarray([0.5]),
        np.asarray([1 / 3, 1 / 3, 0]),
        ("facet-a",),
    )
    point = np.asarray([0.25, 0.25, 0.0])
    mapping = surface.locate(
        point, require_containing=True, residual_tolerance_cm=1.0e-12
    )
    catalog = surface.facet_metadata()
    columns = {
        "position_global_cm": point[None, :],
        "direction_global": np.asarray([[0, 0, -1]], dtype=float),
        "outward_normal_global": np.asarray([[0, 0, 1]], dtype=float),
        "surface_id": np.asarray([7]),
        "facet_id": np.asarray([mapping["facet_id"]], dtype=object),
        "facet_index": np.asarray([mapping["facet_index"]]),
        "barycentric_coordinates": mapping["barycentric_coordinates"][None, :],
        "mu": np.asarray([-1.0]),
        "crossing_sense": np.asarray(["incoming"]),
        "grazing": np.asarray([False]),
    }
    result = validate_boundary_localization(
        columns,
        catalog,
        envelope_surface_ids=(7,),
        coupling_interface="winding_pack",
    )
    assert result["pass"]
    with pytest.raises(ValueError, match="not contained"):
        surface.locate([2, 2, 0], require_containing=True)
    with pytest.raises(ValueError, match="residual"):
        surface.locate(
            [0.25, 0.25, 0.1],
            require_containing=True,
            residual_tolerance_cm=1.0e-3,
        )
