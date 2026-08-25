import json

import pytest

from parastell.magnet_field_workflow import MagnetRadiationWorkflow
from parastell.magnet_field_workflow import StageStatus, StaleStageError


def test_stage_reuse_and_stale_input_refusal(tmp_path):
    source = tmp_path / "input.dat"
    source.write_text("first", encoding="utf-8")
    output = tmp_path / "artifacts" / "validated.json"
    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "geometry_features": {"ports": False},
                "output_directory": str(tmp_path / "artifacts"),
                "stages": {
                    "validate_inputs": {
                        "inputs": [str(source)],
                        "outputs": [str(output)],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    workflow = MagnetRadiationWorkflow.from_json(config_path)
    validation = workflow.validate_configuration()
    assert validation["configured_stages"] == ["validate_inputs"]
    assert validation["declared_output_count"] == 1

    def action(stage_config, root):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return {"validated": True}

    first = workflow.run_stage("validate_inputs", action)
    assert first.status == StageStatus.PASS
    assert not first.reused
    second = workflow.run_stage("validate_inputs", action)
    assert second.reused
    forced = workflow.run_stage("validate_inputs", action, force=True)
    assert not forced.reused
    assert forced.status == StageStatus.PASS
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(StaleStageError):
        workflow.run_stage("validate_inputs", action)
    assert (
        workflow.inspect()["stages"]["validate_inputs"]["status"]
        == "STALE_INPUT"
    )


def test_port_configuration_is_rejected(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text(
        json.dumps(
            {
                "geometry_features": {"ports": True},
                "output_directory": str(tmp_path / "output"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ports: false"):
        MagnetRadiationWorkflow.from_json(config)


def test_configuration_validation_rejects_output_escape(tmp_path):
    config = tmp_path / "escape.json"
    config.write_text(
        json.dumps(
            {
                "geometry_features": {"ports": False},
                "output_directory": str(tmp_path / "artifacts"),
                "stages": {
                    "validate_inputs": {
                        "outputs": [str(tmp_path / "outside.json")]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    workflow = MagnetRadiationWorkflow.from_json(config)
    with pytest.raises(ValueError, match="outside output_directory"):
        workflow.validate_configuration()
