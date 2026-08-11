"""Regression tests for exported OpenMC tally-column resolution."""

import numpy as np

from parastell.hts_multilayer import _find_column


def test_exact_surface_column_wins_over_mu_surface_columns():
    table = {
        "musurface_high": np.asarray([1.0]),
        "musurface_low": np.asarray([0.0]),
        "surface": np.asarray([101]),
    }

    assert _find_column(table, "surface") == "surface"
