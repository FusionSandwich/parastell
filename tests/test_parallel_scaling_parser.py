import pytest

from parastell.parallel_scaling import (
    INCOMPLETE,
    PASS,
    allowed_cores,
    evaluate_scaling_ladder,
    physical_cores,
)


def _row(cores, wall_seconds):
    return {
        "cores": cores,
        "histories": 500_000,
        "wall_seconds": wall_seconds,
    }


def test_physical_core_policy_does_not_count_smt_threads():
    assert (
        physical_cores(sockets=2, cores_per_socket=64, threads_per_core=2)
        == 128
    )
    assert allowed_cores(128) == 32
    assert allowed_cores(192) == 48


def test_incomplete_ladder_never_invents_a_scaling_fit():
    result = evaluate_scaling_ladder([_row(8, 515.0)])
    assert result["status"] == INCOMPLETE
    assert result["missing_core_counts"] == [16, 32, 64]
    assert result["scaling_fit_status"] == "NO_MEASURED_FIT"
    assert result["measurements"][0]["histories_per_second"] == pytest.approx(
        500_000 / 515.0
    )


def test_complete_ladder_reports_measured_efficiency():
    result = evaluate_scaling_ladder(
        [_row(8, 80), _row(16, 42), _row(32, 23), _row(64, 14)]
    )
    assert result["status"] == PASS
    assert result["missing_core_counts"] == []
    assert result["parallel_efficiency_relative_to_8_cores"]["64"] < 1.0


def test_duplicate_core_counts_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        evaluate_scaling_ladder([_row(8, 80), _row(8, 81)])
