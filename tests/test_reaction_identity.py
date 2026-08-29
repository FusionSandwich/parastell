from __future__ import annotations

import pytest

from parastell.reaction_identity import (
    canonicalize_nuclide_mt_requests,
    derive_material_nuclide_mt_requests,
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


def test_material_isotopes_drive_explicit_mt_policy():
    result = derive_material_nuclide_mt_requests(
        [
            {
                "material_id": "rebco",
                "composition_sha256": "a" * 64,
                "isotopes": {
                    "O16": 0.6,
                    "Cu63": 0.3,
                    "Cu65": 0.1,
                    "O17": 0.0,
                },
            },
            {
                "material_id": "steel",
                "composition_sha256": "b" * 64,
                "isotopes": {"Fe56": 1.0},
            },
        ],
        default_mts=[2, 102],
        nuclide_overrides={"Cu63": [16, 103, 107]},
    )
    assert result["nuclide_mt_requests"] == {
        "Cu63": [16, 103, 107],
        "Cu65": [2, 102],
        "Fe56": [2, 102],
        "O16": [2, 102],
    }
    assert result["policy"]["zero_fraction_isotopes_requested"] is False
    assert len(result["derivation_sha256"]) == 64


def test_material_mt_derivation_rejects_natural_or_absent_override():
    material = {
        "material_id": "bad",
        "composition_sha256": "a" * 64,
        "isotopes": {"Fe": 1.0},
    }
    with pytest.raises(ValueError, match="invalid OpenMC nuclide"):
        derive_material_nuclide_mt_requests([material], default_mts=[2])
    material["isotopes"] = {"Fe56": 1.0}
    with pytest.raises(ValueError, match="absent from materials"):
        derive_material_nuclide_mt_requests(
            [material], default_mts=[2], nuclide_overrides={"W184": [102]}
        )
