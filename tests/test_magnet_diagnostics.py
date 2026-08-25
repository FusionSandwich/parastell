import json

import pytest

from parastell.magnet_diagnostics import validate_figure_manifest
from parastell.magnet_stage_handlers import render_diagnostics


def test_render_diagnostics_records_unbiased_fallback(tmp_path):
    pytest.importorskip("matplotlib")
    unbiased = tmp_path / "unbiased.json"
    weighted = tmp_path / "weighted.json"
    skipped = {
        "schema": "parastell.magnet_weight_window_responses/v1.0.0",
        "status": "SKIPPED_UNBIASED_FALLBACK",
        "classification": "REJECTED_INSTABILITY",
        "responses": [],
    }
    unbiased.write_text(json.dumps(skipped), encoding="utf-8")
    weighted.write_text(json.dumps(skipped), encoding="utf-8")
    figure = tmp_path / "comparison.png"
    manifest_path = tmp_path / "figure_manifest.json"

    result = render_diagnostics(
        {
            "unbiased_responses_path": str(unbiased),
            "weight_window_responses_path": str(weighted),
            "comparison_figure_path": str(figure),
            "figure_manifest_path": str(manifest_path),
            "component_names": ["winding_pack"],
            "component_colors": {"winding_pack": "#4C78A8"},
        },
        tmp_path,
    )

    assert figure.is_file()
    assert result["rendering_parameters"]["status"] == (
        "NO_COMPARABLE_WEIGHT_WINDOW_RESPONSES"
    )
    assert result["rendering_parameters"]["unbiased_payload_status"] == (
        "SKIPPED_UNBIASED_FALLBACK"
    )
    assert validate_figure_manifest(manifest_path) == result


def test_render_diagnostics_rejects_mismatched_response_inventory(tmp_path):
    unbiased = tmp_path / "unbiased.json"
    weighted = tmp_path / "weighted.json"
    unbiased.write_text(
        json.dumps({"responses": [{"response_id": "left", "mean": 1.0}]}),
        encoding="utf-8",
    )
    weighted.write_text(
        json.dumps({"responses": [{"response_id": "right", "mean": 1.0}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inventories do not match"):
        render_diagnostics(
            {
                "unbiased_responses_path": str(unbiased),
                "weight_window_responses_path": str(weighted),
                "comparison_figure_path": str(tmp_path / "comparison.png"),
                "figure_manifest_path": str(tmp_path / "manifest.json"),
            },
            tmp_path,
        )
