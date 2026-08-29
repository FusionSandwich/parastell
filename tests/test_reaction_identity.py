from __future__ import annotations

import pytest

from parastell.reaction_identity import (
    canonicalize_nuclide_mt_requests,
    reaction_matrix_identity,
)


def test_isotope_mt_identity_is_explicit_and_hash_bound():
    row = reaction_matrix_identity(
        nuclide="W184", mt=102, nuclear_data_sha256="a" * 64
    )
    assert row == {
        "nuclide": "W184",
        "mt": 102,
        "reaction_label": "n,gamma",
        "nuclear_data_sha256": "a" * 64,
        "missing_semantics": "MISSING_IS_NOT_ZERO",
    }


def test_requests_reject_duplicate_or_ambiguous_mt_values():
    with pytest.raises(ValueError, match="repeats"):
        canonicalize_nuclide_mt_requests({"W184": [102, 102]})
    with pytest.raises(ValueError, match="MT"):
        canonicalize_nuclide_mt_requests({"W184": ["(n,gamma)"]})
    with pytest.raises(ValueError, match="nuclide"):
        canonicalize_nuclide_mt_requests({"tungsten": [102]})
