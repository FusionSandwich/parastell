from __future__ import annotations

import pytest

from scripts.select_source_mesh_outer_cfs_cap import (
    _selection_index,
    _validated_caps,
)


def test_caps_are_preregistered_in_strict_descending_order():
    assert _validated_caps([0.999, 0.995, 0.99], 11) == [
        0.999,
        0.995,
        0.99,
    ]


@pytest.mark.parametrize(
    "caps", [[0.99, 0.995], [0.99, 0.99], [1.001], [0.9], []]
)
def test_caps_reject_ambiguous_or_radially_invalid_search(caps):
    with pytest.raises(ValueError, match="outer CFS caps"):
        _validated_caps(caps, 11)


def test_selector_returns_first_passing_descending_candidate():
    assert (
        _selection_index(
            [
                {"outer_cfs_cap": 0.999, "candidate_pass": False},
                {"outer_cfs_cap": 0.995, "candidate_pass": True},
                {"outer_cfs_cap": 0.99, "candidate_pass": True},
            ]
        )
        == 1
    )
    assert _selection_index([{"candidate_pass": False}]) is None


def test_candidate_builder_import_is_geometry_runtime_neutral():
    from scripts import build_source_mesh_cfs_candidates as builder

    assert builder.SCHEMA == "parastell.source_mesh_cfs_candidates/v1.0.0"
