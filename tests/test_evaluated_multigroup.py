import numpy as np
import pytest

from parastell.evaluated_multigroup import MaterialSpec
from parastell.evaluated_multigroup import collapse_lethargy_weighted


def test_constant_cross_section_collapses_exactly():
    values = collapse_lethargy_weighted(
        [1.0, 10.0, 100.0],
        lambda energy: np.full_like(energy, 3.25),
        minimum_eV=1.0,
        maximum_eV=100.0,
    )
    assert values == pytest.approx([3.25, 3.25], rel=2.0e-3)


def test_out_of_library_groups_are_explicit_zero():
    values = collapse_lethargy_weighted(
        [1.0, 10.0, 100.0],
        lambda energy: np.ones_like(energy),
        minimum_eV=10.0,
        maximum_eV=20.0,
    )
    assert values[0] == 0.0
    assert values[1] > 0.0


def test_material_fractions_must_be_normalized():
    with pytest.raises(ValueError, match="sum to one"):
        MaterialSpec("bad", 1.0, {"Cu": 0.8}, "test")
