from collections import OrderedDict

import numpy as np
import pytest

from parastell.periodic_geometry import expand_half_period_scalar_matrix
from parastell.periodic_geometry import expand_half_period_thickness_fields
from parastell.periodic_geometry import FieldPeriodContract
from parastell.periodic_geometry import radial_build_from_thickness_fields
from parastell.periodic_geometry import validate_full_period_scalar_matrix


def _half_period_matrix(rows=16, columns=31):
    matrix = np.empty((rows, columns), dtype=float)
    for row in range(rows):
        matrix[row, :-1] = 1.0 + row + np.arange(columns - 1) / 100.0
    matrix[0, 16:] = matrix[0, 14::-1]
    matrix[-1, 16:] = matrix[-1, 14::-1]
    matrix[:, -1] = matrix[:, 0]
    return matrix


def test_wistell_d_one_full_period_is_ninety_degrees():
    contract = FieldPeriodContract(n_field_periods=4)

    assert contract.one_period_extent_degrees == 90.0
    assert contract.modeled_extent_degrees == 90.0
    assert np.allclose(contract.toroidal_angles(31), np.linspace(0, 90, 31))


def test_half_period_expansion_preserves_independent_half_and_closes_period():
    half = _half_period_matrix()
    frozen = half.copy()

    full = expand_half_period_scalar_matrix(half)

    assert full.shape == (31, 31)
    assert np.array_equal(half, frozen)
    assert np.allclose(full[:16], half)
    assert np.allclose(full[:, 0], full[:, -1])
    assert np.allclose(full[0], full[-1])
    validate_full_period_scalar_matrix(full)


def test_expand_positive_thickness_fields_and_create_radial_build():
    fields = OrderedDict(
        [
            ("first_wall", _half_period_matrix() + 4.0),
            ("gap", _half_period_matrix() + 6.0),
            ("magnets", _half_period_matrix() + 30.0),
        ]
    )

    expanded = expand_half_period_thickness_fields(fields)
    radial_build = radial_build_from_thickness_fields(
        fields, vacuum_components=("gap",)
    )

    assert list(expanded) == list(fields)
    assert all(values.shape == (31, 31) for values in expanded.values())
    assert radial_build["gap"]["mat_tag"] == "Vacuum"
    assert radial_build["magnets"]["mat_tag"] == "magnets"
    assert np.array_equal(
        radial_build["first_wall"]["thickness_matrix"],
        expanded["first_wall"],
    )


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.ones((1, 31)), "at least two"),
        (np.ones((16, 30)), "odd number"),
        (np.full((16, 31), np.nan), "non-finite"),
    ],
)
def test_invalid_half_period_fields_fail_closed(matrix, message):
    with pytest.raises(ValueError, match=message):
        expand_half_period_scalar_matrix(matrix)


def test_open_half_period_poloidal_profile_is_rejected():
    matrix = _half_period_matrix()
    matrix[3, -1] += 1.0

    with pytest.raises(ValueError, match="not closed"):
        expand_half_period_scalar_matrix(matrix)


def test_contract_rejects_nonphysical_period_counts():
    with pytest.raises(ValueError, match="positive"):
        FieldPeriodContract(n_field_periods=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        FieldPeriodContract(n_field_periods=4, periods_modeled=5)
