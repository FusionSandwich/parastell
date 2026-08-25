import pytest

from parastell.magnet_boundary_envelope import classify_crossing_bank
from parastell.surface_bank_capacity import (
    plan_surface_bank_capacity,
    validate_crossing_bank_accounting,
    validate_qualification_bank,
)


def test_capacity_uses_confidence_upper_crossings_plus_margin():
    plan = plan_surface_bank_capacity(
        observed_crossings=100,
        calibration_histories=100_000,
        planned_histories=2_000_000,
        confidence_level=0.99,
        capacity_margin=1.25,
        max_source_files=4,
        mpi_ranks=16,
    )
    assert plan["expected_crossings"] == pytest.approx(2000.0)
    assert plan["confidence_upper_crossings"] > plan["expected_crossings"]
    assert plan["required_capacity"] >= (
        plan["confidence_upper_crossings"] * 1.25
    )
    assert plan["configured_capacity"] >= plan["required_capacity"]
    assert plan["max_particles_per_file"] * 4 == (plan["configured_capacity"])


def test_complete_sampled_and_truncated_semantics_match_canonical_classifier():
    complete = classify_crossing_bank(
        stored_record_count=80,
        selected_record_count=40,
        max_particles_per_file=100,
        max_source_files=1,
        source_file_count=1,
        mpi_ranks=4,
    )
    assert validate_qualification_bank(complete)["classification"] == (
        "COMPLETE_CROSSING_BANK"
    )

    sampled = classify_crossing_bank(
        stored_record_count=80,
        selected_record_count=40,
        max_particles_per_file=None,
        max_source_files=None,
        source_file_count=1,
        sampling_applied=True,
    )
    assert validate_crossing_bank_accounting(sampled)["classification"] == (
        "SAMPLED_CROSSING_BANK"
    )
    with pytest.raises(ValueError, match="requires a complete"):
        validate_qualification_bank(sampled)

    truncated = classify_crossing_bank(
        stored_record_count=100,
        selected_record_count=40,
        max_particles_per_file=100,
        max_source_files=1,
        source_file_count=1,
    )
    assert validate_crossing_bank_accounting(truncated)["classification"] == (
        "TRUNCATED_INVALID_BANK"
    )
    with pytest.raises(ValueError, match="requires a complete"):
        validate_qualification_bank(truncated)


def test_declared_classification_and_capacity_plan_fail_closed():
    plan = plan_surface_bank_capacity(
        observed_crossings=10,
        calibration_histories=100_000,
        planned_histories=500_000,
        max_source_files=2,
    )
    accounting = classify_crossing_bank(
        stored_record_count=10,
        selected_record_count=10,
        max_particles_per_file=plan["max_particles_per_file"],
        max_source_files=plan["max_source_files"],
        source_file_count=1,
    )
    validated = validate_qualification_bank(accounting, capacity_plan=plan)
    assert validated["configured_capacity"] >= plan["required_capacity"]

    forged = {**accounting, "classification": "SAMPLED_CROSSING_BANK"}
    with pytest.raises(ValueError, match="classification is inconsistent"):
        validate_crossing_bank_accounting(forged)

    impossible_file_count = {**accounting, "source_file_count": 3}
    with pytest.raises(ValueError, match="exceeds max_source_files"):
        validate_crossing_bank_accounting(impossible_file_count)

    too_small = classify_crossing_bank(
        stored_record_count=10,
        selected_record_count=10,
        max_particles_per_file=1,
        max_source_files=1,
        source_file_count=1,
    )
    with pytest.raises(ValueError, match="below the confidence-upper plan"):
        validate_crossing_bank_accounting(too_small, capacity_plan=plan)


def test_zero_calibration_count_still_allocates_from_exact_upper_bound():
    plan = plan_surface_bank_capacity(
        observed_crossings=0,
        calibration_histories=100_000,
        planned_histories=500_000,
        confidence_level=0.95,
        capacity_margin=1.25,
    )
    assert plan["expected_crossings"] == 0.0
    assert plan["confidence_upper_crossings"] > 0.0
    assert plan["required_capacity"] > 1
