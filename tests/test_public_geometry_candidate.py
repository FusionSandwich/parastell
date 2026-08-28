import numpy as np

from parastell.public_geometry_candidate import public_reference_radial_build
from parastell.public_geometry_candidate import replace_thickness_matrices
from parastell.public_geometry_candidate import thickness_matrices


def test_public_reference_build_preserves_frozen_layer_semantics():
    build = public_reference_radial_build()
    assert tuple(build) == (
        "first_wall",
        "breeder",
        "back_wall",
        "shield",
        "vacuum_vessel",
    )
    assert build["vacuum_vessel"]["mat_tag"] == "vac_vessel"
    np.testing.assert_array_equal(build["first_wall"]["thickness_matrix"], 5.0)
    np.testing.assert_array_equal(build["shield"]["thickness_matrix"], 50.0)
    np.testing.assert_array_equal(
        build["vacuum_vessel"]["thickness_matrix"], 10.0
    )
    assert set(np.unique(build["breeder"]["thickness_matrix"])) == {
        25.0,
        45.0,
        65.0,
        75.0,
    }


def test_replacement_changes_only_thickness_arrays():
    original = public_reference_radial_build()
    matrices = thickness_matrices(original)
    matrices["breeder"][4, 4] -= 1.0
    replaced = replace_thickness_matrices(original, matrices)
    assert replaced["breeder"]["thickness_matrix"][4, 4] == 74.0
    assert original["breeder"]["thickness_matrix"][4, 4] == 75.0
    assert replaced["vacuum_vessel"]["mat_tag"] == "vac_vessel"
