import numpy as np
import pytest

from parastell.radial_build_constraints import constrain_radial_build
from parastell.radial_build_constraints import (
    eliminate_interpolated_layer_increase,
)
from parastell.radial_build_constraints import (
    sampled_interpolated_layer_thickness,
)


def _build():
    return {
        "first_wall": np.full((3, 3), 5.0),
        "breeder": np.full((3, 3), 25.0),
        "back_wall": np.full((3, 3), 5.0),
        "shield": np.full((3, 3), 50.0),
        "vacuum_vessel": np.full((3, 3), 10.0),
    }


def test_constraint_preserves_unaffected_samples_and_protected_layers():
    original = _build()
    available = np.full((3, 3), 97.0)
    available[1, 1] = 82.0

    result = constrain_radial_build(
        original,
        available,
        clearance_cm=2.0,
        reduction_order=("breeder", "shield"),
        minimum_thickness_cm={"breeder": 5.0, "shield": 5.0},
    )

    assert result.feasible
    assert result.changed_mask.sum() == 1
    assert result.corrected["breeder"][1, 1] == 10.0
    assert result.corrected["shield"][1, 1] == 50.0
    for name in ("first_wall", "back_wall", "vacuum_vessel"):
        np.testing.assert_array_equal(result.corrected[name], original[name])
    for name in original:
        np.testing.assert_array_equal(original[name], _build()[name])


def test_constraint_uses_declared_reduction_priority_and_minima():
    result = constrain_radial_build(
        _build(),
        np.full((3, 3), 67.0),
        clearance_cm=2.0,
        reduction_order=("breeder", "shield"),
        minimum_thickness_cm={"breeder": 5.0, "shield": 40.0},
    )

    np.testing.assert_array_equal(result.corrected["breeder"], 5.0)
    np.testing.assert_array_equal(result.corrected["shield"], 40.0)


def test_constraint_fails_closed_when_protected_build_cannot_fit():
    with pytest.raises(ValueError, match="infeasible"):
        constrain_radial_build(
            _build(),
            np.full((3, 3), 20.0),
            clearance_cm=2.0,
            reduction_order=("breeder", "shield"),
            minimum_thickness_cm={"breeder": 5.0, "shield": 40.0},
        )


@pytest.mark.parametrize(
    "available, message",
    [
        (np.ones((2, 2)), "must match"),
        (np.zeros((3, 3)), "strictly positive"),
    ],
)
def test_constraint_rejects_invalid_available_space(available, message):
    with pytest.raises(ValueError, match=message):
        constrain_radial_build(
            _build(),
            available,
            clearance_cm=2.0,
            reduction_order=("breeder", "shield"),
        )


def test_constraint_treats_positive_infinity_as_unconstrained():
    result = constrain_radial_build(
        _build(),
        np.full((3, 3), np.inf),
        clearance_cm=5.0,
        reduction_order=("breeder",),
    )
    assert result.feasible
    assert not np.any(result.changed_mask)


def test_sampled_interpolated_thickness_uses_cumulative_offsets():
    unit = np.ones((4, 4))
    reference = {
        "first_wall": unit * 5.0,
        "breeder": unit * 25.0,
        "shield": unit * 50.0,
    }
    candidate = {name: values.copy() for name, values in reference.items()}
    candidate["breeder"][1, 1] = 15.0
    result = sampled_interpolated_layer_thickness(
        reference,
        candidate,
        layer="breeder",
        toroidal_control_angles_deg=[0.0, 30.0, 60.0, 90.0],
        poloidal_control_angles_deg=[0.0, 120.0, 240.0, 360.0],
        toroidal_step_deg=10.0,
        poloidal_step_deg=20.0,
    )
    summary = result.summary()
    assert summary["minimum_candidate_thickness_cm"] == pytest.approx(15.0)
    assert summary["maximum_reduction_cm"] == pytest.approx(10.0)
    assert summary["changed_sample_count"] > 0
    np.testing.assert_allclose(result.reference_thickness_cm, 25.0)


def test_sampled_interpolated_thickness_reports_unauthorized_increase():
    unit = np.ones((4, 4))
    reference = {"breeder": unit * 25.0}
    candidate = {"breeder": unit * 25.0}
    candidate["breeder"][1, 1] = 26.0
    result = sampled_interpolated_layer_thickness(
        reference,
        candidate,
        layer="breeder",
        toroidal_control_angles_deg=[0.0, 30.0, 60.0, 90.0],
        poloidal_control_angles_deg=[0.0, 120.0, 240.0, 360.0],
        toroidal_step_deg=10.0,
        poloidal_step_deg=20.0,
    )
    summary = result.summary()
    assert summary["maximum_increase_cm"] == pytest.approx(1.0)
    assert summary["increased_sample_count"] > 0
    assert summary["changed_sample_count"] >= summary["increased_sample_count"]


def test_refinement_removes_dense_increase_without_touching_other_layers():
    unit = np.ones((4, 4))
    reference = {
        "first_wall": unit * 5.0,
        "breeder": np.array(
            [
                [75.0, 75.0, 25.0, 75.0],
                [75.0, 25.0, 25.0, 75.0],
                [75.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 25.0, 75.0],
            ]
        ),
    }
    candidate = {name: values.copy() for name, values in reference.items()}
    candidate["breeder"][1, 1] -= 5.0
    corrected, receipt = eliminate_interpolated_layer_increase(
        reference,
        candidate,
        layer="breeder",
        toroidal_control_angles_deg=[0.0, 30.0, 60.0, 90.0],
        poloidal_control_angles_deg=[0.0, 120.0, 240.0, 360.0],
        minimum_thickness_cm=5.0,
        toroidal_step_deg=5.0,
        poloidal_step_deg=10.0,
    )
    assert receipt["status"] == "CONVERGED"
    assert receipt["final_summary"]["maximum_increase_cm"] <= 1.0e-8
    np.testing.assert_array_equal(corrected["first_wall"], unit * 5.0)
    assert np.all(corrected["breeder"] <= reference["breeder"] + 1.0e-8)
