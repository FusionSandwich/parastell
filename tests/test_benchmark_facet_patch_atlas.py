from scripts.benchmark_facet_patch_atlas import benchmark


def test_preprocessing_and_query_benchmark_is_diagnostic_and_equivalent():
    receipt = benchmark(side=3, repeats=1)

    assert receipt["facet_count"] == 18
    assert receipt["query_count_per_repeat"] == 18
    assert receipt["preprocessing_nanoseconds"] > 0
    assert receipt["cached_query_median_nanoseconds"] > 0
    assert receipt["uncached_query_median_nanoseconds"] > 0
    assert receipt["cached_uncached_equivalence_pass"] is True
    assert receipt["physical_qualification_claimed"] is False
    assert receipt["arbitrary_polyhedral_radiant_sweep_claimed"] is False
