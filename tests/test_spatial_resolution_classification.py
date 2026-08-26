import pytest

from parastell.spatial_resolution import classify_spatial_estimator


@pytest.mark.parametrize(
    "relative,expected",
    [
        (0.20, "QUALIFIED"),
        (0.40, "MARGINAL"),
        (0.80, "INSUFFICIENT_STATISTICS"),
    ],
)
def test_weighted_bin_classification(relative, expected):
    assert (
        classify_spatial_estimator(
            raw_records=10,
            sum_weights=1.0,
            sum_squared_weights=0.1,
            relative_uncertainty=relative,
        )
        == expected
    )


def test_empty_and_unobservable_bins_fail_closed():
    assert (
        classify_spatial_estimator(
            raw_records=0,
            sum_weights=0.0,
            sum_squared_weights=0.0,
            relative_uncertainty=None,
        )
        == "EMPTY"
    )
    assert (
        classify_spatial_estimator(
            raw_records=None,
            sum_weights=None,
            sum_squared_weights=None,
            relative_uncertainty=0.1,
        )
        == "INSUFFICIENT_STATISTICS"
    )
