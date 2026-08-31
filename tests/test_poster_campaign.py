from __future__ import annotations

import copy

import pytest

from parastell.poster_campaign import (
    SCHEMA,
    build_poster_campaign_manifest,
    validate_poster_campaign_manifest,
    write_poster_campaign_manifest,
)


def _baseline_case():
    return {
        "geometry": {
            "layout": "wistell-sector",
            "breeder": {"enabled": True, "scale": 1.0},
            "shield": {"enabled": True, "thickness": 2.0},
            "thickness_cm": 1.0,
            "selected": {
                "locations": ["inboard", "outboard"],
                "label": "default",
            },
            "materials": {"first_wall": "W", "vacuum_vessel": "SS"},
        }
    }


def _cases():
    return [
        {"geometry": {"breeder": {"scale": 1.1}}},
        {"geometry": {"breeder": {"enabled": False}}},
    ]


def test_staged_poster_manifest_tracks_changed_fields_and_staging():
    manifest = build_poster_campaign_manifest(
        campaign_id="wistell-d-poster-demo",
        baseline_case=_baseline_case(),
        controlled_breeder_cases=_cases(),
        shield_cases=[{"geometry": {"shield": {"thickness": 2.4}}}],
        spatial_thickness_cases=[
            {"geometry": {"thickness_cm": 1.5}},
            {"geometry": {"thickness_cm": 2.0, "layout": "alternative"}},
        ],
        selected_location_cases=[
            {"geometry": {"selected": {"label": "upper"}}}
        ],
    )

    assert manifest["schema"] == SCHEMA
    assert manifest["case_strategy"] == "staged_without_cartesian_product"
    assert manifest["non_cartesian"] is True
    assert manifest["case_count"] == 1 + 2 + 1 + 2 + 1
    assert manifest["baseline_case_id"] == "wistell-d-poster-demo-baseline"
    assert manifest["cases"][0]["stage"] == "baseline"
    assert manifest["cases"][0]["case_index"] == 0
    assert (
        manifest["cases"][1]["source_case_id"] == manifest["baseline_case_id"]
    )
    assert manifest["stages"][0]["stage"] == "controlled_breeder"
    assert manifest["stages"][0]["case_count"] == 2
    assert manifest["manifest_sha256"]

    stage_one = manifest["cases"][1]
    stage_two = manifest["cases"][-1]
    assert "geometry.breeder.scale" in stage_one["changed_fields"]
    assert "geometry.selected.label" in stage_two["changed_fields"]
    assert "geometry.thickness_cm" in manifest["cases"][4]["changed_fields"]


def test_manifest_stages_are_single_parent_non_cartesian():
    manifest = build_poster_campaign_manifest(
        campaign_id="wistell-d-poster-no-mix",
        baseline_case=_baseline_case(),
        controlled_breeder_cases=_cases(),
    )
    baseline_id = manifest["baseline_case_id"]
    for case in manifest["cases"]:
        if case["stage"] == "baseline":
            continue
        assert case["source_case_id"] == baseline_id
        assert case["stage"] in {
            "controlled_breeder",
            "shield",
            "spatial_thickness",
            "selected_location",
        }


def test_manifest_rejects_diff_metadata_inconsistency():
    manifest = build_poster_campaign_manifest(
        campaign_id="wistell-d-poster-bad",
        baseline_case=_baseline_case(),
        shield_cases=[{"geometry": {"shield": {"thickness": 2.4}}}],
    )
    manifest["cases"][1]["changed_fields"] = [
        "geometry.layout"
    ]  # intentionally wrong
    manifest["cases"][1]["held_fixed_fields"] = [
        field
        for field in manifest["cases"][1]["held_fixed_fields"]
        if field != "geometry.layout"
    ]
    with pytest.raises(ValueError, match="diff metadata is inconsistent"):
        validate_poster_campaign_manifest(manifest)


def test_manifest_requires_staged_strategy_and_complete_cases():
    manifest = build_poster_campaign_manifest(
        campaign_id="wistell-d-poster-empty",
        baseline_case=_baseline_case(),
    )
    assert manifest["cases"][0]["case_index"] == 0
    assert manifest["case_count"] == 1

    with pytest.raises(ValueError, match="baseline_case_id|baseline case id"):
        bad = copy.deepcopy(manifest)
        bad["baseline_case_id"] = "wrong"
        bad["cases"][0]["case_id"] = "wrong"
        validate_poster_campaign_manifest(bad)


def test_manifest_writer_is_create_only(tmp_path):
    manifest = build_poster_campaign_manifest(
        campaign_id="wistell-d-poster-write",
        baseline_case=_baseline_case(),
    )
    destination = tmp_path / "poster_campaign.json"
    first = write_poster_campaign_manifest(destination, manifest)
    with pytest.raises(FileExistsError):
        write_poster_campaign_manifest(destination, manifest)
    assert first == destination
