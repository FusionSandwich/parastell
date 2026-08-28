import pytest

from parastell.surface_source_accounting import (
    validate_surface_source_accounting,
)


def _evidence():
    return {
        "run_id": "bounded-a",
        "seed": 17,
        "source_histories": 1000,
        "requested_surface_ids": [7, 8],
        "max_particles_per_process": 100,
        "max_source_files": 2,
        "mpi_ranks": 2,
        "source_files": [
            {
                "path": "surface_source.1.h5",
                "sha256": "a" * 64,
                "record_count": 80,
            },
            {
                "path": "surface_source.h5",
                "sha256": "b" * 64,
                "record_count": 20,
            },
        ],
        "stored_record_count": 100,
        "selected_record_count": 100,
        "sampling_applied": False,
        "terminal_writer_evidence": {
            "available": True,
            "records_by_file": [80, 20],
        },
    }


def test_complete_bank_uses_mpi_scaled_capacity_and_terminal_counts():
    result = validate_surface_source_accounting(_evidence())
    assert result["classification"] == "COMPLETE_CROSSING_BANK"
    assert result["per_file_capacity"] == 200
    assert result["configured_capacity"] == 400


def test_capacity_reached_and_missing_terminal_evidence_fail_closed():
    full = _evidence()
    full["source_files"] = [
        {"path": "surface_source.h5", "sha256": "c" * 64, "record_count": 200}
    ]
    full["stored_record_count"] = 200
    full["selected_record_count"] = 200
    full["terminal_writer_evidence"]["records_by_file"] = [200]
    result = validate_surface_source_accounting(full)
    assert result["classification"] == "TRUNCATED_INVALID_BANK"
    assert result["capacity_reached"]

    missing = _evidence()
    missing["terminal_writer_evidence"] = {"available": False}
    assert validate_surface_source_accounting(missing)["classification"] == (
        "TRUNCATED_INVALID_BANK"
    )


def test_sampling_and_forged_or_misaligned_evidence_are_rejected():
    sampled = _evidence()
    sampled["sampling_applied"] = True
    assert validate_surface_source_accounting(sampled)["classification"] == (
        "SAMPLED_CROSSING_BANK"
    )

    forged = _evidence()
    forged["classification"] = "SAMPLED_CROSSING_BANK"
    with pytest.raises(ValueError, match="disagrees"):
        validate_surface_source_accounting(forged)

    mismatched = _evidence()
    mismatched["terminal_writer_evidence"]["records_by_file"] = [99, 1]
    assert validate_surface_source_accounting(mismatched)[
        "classification"
    ] == ("TRUNCATED_INVALID_BANK")
