from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from parastell.activation_campaign import (
    COOLING_OFFSETS_S,
    IRRADIATION_CHECKPOINTS_S,
    build_activation_campaign,
    validate_activation_campaign,
    write_activation_campaign,
)


def test_requested_full_power_schedule_and_cooling_branches_are_exact():
    campaign = build_activation_campaign()
    assert tuple(campaign["irradiation_checkpoint_s"]) == (
        IRRADIATION_CHECKPOINTS_S
    )
    assert tuple(campaign["cooling_offset_s"]) == COOLING_OFFSETS_S
    assert campaign["prompt_source_during_cooling"] is False
    assert campaign["delayed_photon_source_separate_from_prompt"] is True
    assert len(campaign["schedule_sha256"]) == 64


def test_checked_in_schedule_matches_the_builder():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / ("wistell_d_activation_schedule.json")
    )
    checked_in = json.loads(path.read_text(encoding="utf-8"))
    built = build_activation_campaign()
    built.pop("schedule_sha256")
    assert checked_in == built
    validate_activation_campaign(checked_in)


def test_schedule_rejects_serialized_cooling_or_copied_source_rate():
    campaign = build_activation_campaign()
    campaign.pop("schedule_sha256")
    serialized = copy.deepcopy(campaign)
    serialized["cooling_semantics"] = "serial"
    with pytest.raises(ValueError, match="independent"):
        validate_activation_campaign(serialized)
    copied = copy.deepcopy(campaign)
    copied["full_power_source_rate_binding"]["copied_absolute_rate"] = True
    with pytest.raises(ValueError, match="source-rate"):
        validate_activation_campaign(copied)


def test_schedule_writer_is_create_only(tmp_path):
    path = tmp_path / "schedule.json"
    write_activation_campaign(path, build_activation_campaign())
    with pytest.raises(FileExistsError):
        write_activation_campaign(path, build_activation_campaign())
