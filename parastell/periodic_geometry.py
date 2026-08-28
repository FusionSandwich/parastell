"""Helpers for constructing exactly one full stellarator field period.

The functions in this module are geometry-neutral.  They convert scalar
half-period design fields into the full-period matrices expected by ParaStell
without depending on a particular VMEC equilibrium, material set, or OpenMC
model.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np


def _enforce_scalar_helical_symmetry(matrix: np.ndarray) -> np.ndarray:
    """Scalar-only form of ParaStell's symmetry operation.

    Keeping this small operation local avoids importing optional MOAB bindings
    merely to manipulate NumPy thickness fields.
    """
    matrix = np.asarray(matrix, dtype=float).copy()
    num_rows, num_columns = matrix.shape
    matrix[:, -1] = matrix[:, 0]
    matrix[0] = np.concatenate(
        [
            matrix[0, : math.ceil(num_columns / 2)],
            np.flip(matrix[0, : math.floor(num_columns / 2)]),
        ]
    )
    flattened = matrix.reshape(num_rows * num_columns)
    first_half = flattened[: math.ceil(flattened.size / 2)]
    last_half = np.flip(
        flattened.copy()[: math.floor(flattened.size / 2)], axis=0
    )
    return np.concatenate([first_half, last_half]).reshape(matrix.shape)


@dataclass(frozen=True)
class FieldPeriodContract:
    """Describe a model containing an integer number of field periods."""

    n_field_periods: int
    periods_modeled: int = 1
    start_angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.n_field_periods, bool) or not isinstance(
            self.n_field_periods, (int, np.integer)
        ):
            raise TypeError("n_field_periods must be an integer")
        if isinstance(self.periods_modeled, bool) or not isinstance(
            self.periods_modeled, (int, np.integer)
        ):
            raise TypeError("periods_modeled must be an integer")
        if self.n_field_periods <= 0:
            raise ValueError("n_field_periods must be positive")
        if self.periods_modeled <= 0:
            raise ValueError("periods_modeled must be positive")
        if self.periods_modeled > self.n_field_periods:
            raise ValueError(
                "periods_modeled cannot exceed the device field-period count"
            )
        if not np.isfinite(self.start_angle_degrees):
            raise ValueError("start_angle_degrees must be finite")

    @property
    def one_period_extent_degrees(self) -> float:
        """Toroidal extent of one complete field period."""
        return 360.0 / self.n_field_periods

    @property
    def modeled_extent_degrees(self) -> float:
        """Toroidal extent represented by this contract."""
        return self.periods_modeled * self.one_period_extent_degrees

    @property
    def end_angle_degrees(self) -> float:
        return self.start_angle_degrees + self.modeled_extent_degrees

    def toroidal_angles(self, num_profiles: int) -> np.ndarray:
        """Return regularly spaced toroidal profiles for the modeled sector."""
        if isinstance(num_profiles, bool) or not isinstance(
            num_profiles, (int, np.integer)
        ):
            raise TypeError("num_profiles must be an integer")
        if num_profiles < 2:
            raise ValueError("num_profiles must be at least two")
        return np.linspace(
            self.start_angle_degrees,
            self.end_angle_degrees,
            num=int(num_profiles),
        )


def expand_half_period_scalar_matrix(
    half_period_matrix: np.ndarray,
) -> np.ndarray:
    """Expand one helically symmetric half-period scalar field.

    The input contains both half-period boundary profiles.  The returned full
    period therefore has ``2 * n_toroidal - 1`` profiles.  ParaStell's native
    symmetry routine supplies the second half and enforces toroidal/poloidal
    closure.  The input array is never mutated.
    """
    half_period_matrix = np.asarray(half_period_matrix, dtype=float)
    if half_period_matrix.ndim != 2:
        raise ValueError("half_period_matrix must be two-dimensional")
    num_toroidal, num_poloidal = half_period_matrix.shape
    if num_toroidal < 2:
        raise ValueError("half-period matrix requires at least two profiles")
    if num_poloidal < 3 or num_poloidal % 2 == 0:
        raise ValueError(
            "poloidal profiles must use an odd number of points including "
            "the repeated closure point"
        )
    if not np.all(np.isfinite(half_period_matrix)):
        raise ValueError("half-period matrix contains non-finite values")
    if not np.allclose(half_period_matrix[:, 0], half_period_matrix[:, -1]):
        raise ValueError("half-period poloidal profiles are not closed")

    full_period = np.zeros((2 * num_toroidal - 1, num_poloidal), dtype=float)
    full_period[:num_toroidal] = half_period_matrix
    full_period = _enforce_scalar_helical_symmetry(full_period)
    validate_full_period_scalar_matrix(full_period)
    return full_period


def validate_full_period_scalar_matrix(matrix: np.ndarray) -> None:
    """Fail if a scalar field does not satisfy ParaStell period symmetry."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("full-period matrix must be two-dimensional")
    num_toroidal, num_poloidal = matrix.shape
    if num_toroidal < 3 or num_toroidal % 2 == 0:
        raise ValueError("full-period matrix requires an odd profile count")
    if num_poloidal < 3 or num_poloidal % 2 == 0:
        raise ValueError("full-period poloidal point count must be odd")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("full-period matrix contains non-finite values")
    if not np.allclose(matrix[:, 0], matrix[:, -1]):
        raise ValueError("full-period poloidal profiles are not closed")
    expected = _enforce_scalar_helical_symmetry(matrix)
    if not np.allclose(matrix, expected):
        raise ValueError(
            "matrix does not satisfy full-period helical symmetry"
        )


def expand_half_period_thickness_fields(
    fields: Mapping[str, np.ndarray],
) -> OrderedDict[str, np.ndarray]:
    """Expand an ordered collection of half-period scalar thickness fields."""
    if not fields:
        raise ValueError("at least one thickness field is required")
    expanded: OrderedDict[str, np.ndarray] = OrderedDict()
    expected_shape = None
    for name, values in fields.items():
        if not isinstance(name, str) or not name:
            raise ValueError("thickness field names must be nonempty strings")
        full_period = expand_half_period_scalar_matrix(values)
        if expected_shape is None:
            expected_shape = full_period.shape
        elif full_period.shape != expected_shape:
            raise ValueError("all thickness fields must have the same shape")
        if not np.all(full_period > 0.0):
            raise ValueError(
                f"thickness field {name!r} is not strictly positive"
            )
        expanded[name] = full_period
    return expanded


def radial_build_from_thickness_fields(
    fields: Mapping[str, np.ndarray],
    *,
    vacuum_components: Sequence[str] = (),
) -> OrderedDict[str, dict[str, object]]:
    """Create a ParaStell radial-build dictionary without device assumptions."""
    vacuum = set(vacuum_components)
    unknown = vacuum - set(fields)
    if unknown:
        raise ValueError(
            "vacuum components are absent from thickness fields: "
            + ", ".join(sorted(unknown))
        )
    expanded = expand_half_period_thickness_fields(fields)
    return OrderedDict(
        (
            name,
            {
                "thickness_matrix": matrix,
                "mat_tag": "Vacuum" if name in vacuum else name,
            },
        )
        for name, matrix in expanded.items()
    )
