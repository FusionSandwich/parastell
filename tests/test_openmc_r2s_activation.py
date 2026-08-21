from pathlib import Path

import pytest

from parastell.activation.model import ActivationSchedule, ActivationStep
from parastell.activation.openmc_r2s import OpenMCR2SConfiguration


def test_activation_schedule_preserves_source_rate_and_cooling():
    schedule = ActivationSchedule(
        "pilot",
        (
            ActivationStep(10.0, 2.0e20, "full power"),
            ActivationStep(20.0, 0.0, "shutdown"),
        ),
    )
    assert schedule.timesteps_s == (10.0, 20.0)
    assert schedule.source_rates_n_s == (2.0e20, 0.0)
    assert schedule.cooling_result_indices == (2,)
    assert schedule.as_dict()["integrated_source_neutrons"] == 2.0e21


def test_openmc_configuration_rejects_nonexistent_chain(tmp_path):
    schedule = ActivationSchedule("test", (ActivationStep(1.0, 1.0, "on"),))
    with pytest.raises(FileNotFoundError):
        OpenMCR2SConfiguration(
            tmp_path / "missing.xml", schedule, tmp_path / "run"
        )


def test_openmc_configuration_accepts_direct_reaction_rates(tmp_path):
    chain = tmp_path / "chain.xml"
    chain.write_text("<depletion_chain />", encoding="ascii")
    schedule = ActivationSchedule("test", (ActivationStep(1.0, 1.0, "on"),))
    config = OpenMCR2SConfiguration(chain, schedule, tmp_path / "run")
    assert config.reaction_rate_mode == "direct"
    assert isinstance(config.chain_file, Path)
