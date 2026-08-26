import pytest

from parastell.surface_bank_capacity import validate_surface_bank_run_evidence


def _evidence():
    return {
        "run_id": "run-a",
        "seed": 17,
        "source_histories": 1000,
        "requested_surface_ids": [1, 2],
        "max_particles_per_file": 100,
        "max_source_files": 2,
        "mpi_ranks": 2,
        "stored_record_count": 12,
        "selected_record_count": 12,
        "source_file_count": 2,
        "source_files": [
            {"path": "rank0.h5", "sha256": "a" * 64},
            {"path": "rank1.h5", "sha256": "b" * 64},
        ],
        "sampling_applied": False,
        "terminal_writer_evidence": {
            "available": True,
            "aggregate_records_written": 12,
            "records_written_by_rank": [5, 7],
        },
    }


def test_complete_bank_requires_terminal_and_rank_accounting():
    result = validate_surface_bank_run_evidence(_evidence())
    assert result["classification"] == "COMPLETE_CROSSING_BANK"
    assert result["terminal_writer_evidence_valid"]


def test_duplicate_bank_content_is_rejected():
    value = _evidence()
    value["source_files"][1]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="duplicate"):
        validate_surface_bank_run_evidence(value)


def test_capacity_reached_is_invalid():
    value = _evidence()
    value["stored_record_count"] = 200
    value["selected_record_count"] = 200
    value["terminal_writer_evidence"]["aggregate_records_written"] = 200
    value["terminal_writer_evidence"]["records_written_by_rank"] = [100, 100]
    with pytest.raises(ValueError, match="complete crossing bank"):
        validate_surface_bank_run_evidence(value)
