import hashlib
import json
from pathlib import Path

import pytest

from parastell import magnet_stage_handlers
from parastell.magnet_field_cli import STAGE_COMMANDS, _parser, main
from parastell.magnet_field_workflow import (
    STAGE_DEPENDENCIES,
    STAGES,
    MagnetRadiationWorkflow,
)
from parastell.magnet_stage_handlers import build_source_convergence_ladder
from parastell.magnet_stage_handlers import qualify_source_convergence
from parastell.source_convergence import (
    CONVERGED_STATUS,
    INCOMPLETE_STATUS,
    MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS,
    REQUIRED_CANDIDATE_SHAPES,
    SOURCE_METRICS,
)


def _hash(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest(path, shape, delta=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = 1.0 + delta
    value = {
        "schema": "parastell.magnet_source_stage/v1.0.0",
        "source_mesh": {
            "path": str(path.with_suffix(".h5m")),
            "sha256": _hash(f"mesh:{shape}"),
            "size_bytes": 123,
            "shape": list(shape),
            "tetrahedra": 10,
            "physical_source_rate_per_s": 1.0e20 * scale,
        },
        "convergence_observables": {
            "spatial_moments": {
                "mean_x_cm": 100.0 * scale,
                "mean_y_cm": 200.0 * scale,
                "mean_z_cm": 50.0 * scale,
                "mean_x2_cm2": 10_000.0 * scale**2 + 1_000.0,
                "mean_y2_cm2": 40_000.0 * scale**2 + 1_000.0,
                "mean_z2_cm2": 2_500.0 * scale**2 + 500.0,
            },
            "dt_energy_moments": {
                "mean_eV": 14.1e6 * scale,
                "mean_squared_eV2": (14.1e6 * scale) ** 2 + 1.2e12,
            },
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _tolerances(limit=0.01):
    return {
        metric: {"maximum_relative_change": limit} for metric in SOURCE_METRICS
    }


def _response_report(path, delta=0.0, cost=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = 1.0 + delta
    responses = {}
    for index, response_id in enumerate(
        MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS
    ):
        if response_id == "whole_magnet.hotspot_location":
            responses[response_id] = {
                "value_xyz_cm": [100.0 * scale, 20.0, -5.0],
                "units": "cm",
                "coordinate_frame": "global_cartesian_cm",
                "magnet_id": "magnet-0001",
                "bin_id": f"bin-{index}",
            }
        else:
            responses[response_id] = {
                "value": (
                    cost
                    if response_id == "computational_cost"
                    else (index + 1.0) * scale
                ),
                "units": (
                    "cpu-s"
                    if response_id == "computational_cost"
                    else "per source"
                ),
            }
            if response_id.startswith("whole_magnet.incoming_"):
                responses[response_id].update(
                    {
                        "direction": "incoming",
                        "crossing_selection": "inward_only",
                        "estimator": "direction_resolved_surface_current",
                    }
                )
    path.write_text(
        json.dumps({"whole_magnet_responses": responses}), encoding="utf-8"
    )
    return path


def _stage(tmp_path, *, existing=4):
    paths = []
    deltas = (0.20, 0.08, 0.005, 0.0)
    costs = (1.0, 2.0, 4.0, 20.0)
    for index, (shape, delta, cost) in enumerate(
        zip(REQUIRED_CANDIDATE_SHAPES, deltas, costs)
    ):
        path = tmp_path / f"source-{index}.json"
        response_path = tmp_path / f"responses-{index}.json"
        if index < existing:
            _manifest(path, shape, delta)
            _response_report(response_path, delta, cost)
        paths.append(path)
    output = tmp_path / "source-convergence.json"
    return {
        "candidates": [
            {
                "shape": list(shape),
                "source_manifest_path": str(path),
                "response_report_path": str(
                    tmp_path / f"responses-{index}.json"
                ),
            }
            for index, (shape, path) in enumerate(
                zip(REQUIRED_CANDIDATE_SHAPES, paths)
            )
        ],
        "source_metric_tolerances": _tolerances(),
        "response_metric_tolerances": {
            response_id: (
                {"qualification_role": "selection_only"}
                if response_id == "computational_cost"
                else (
                    {"maximum_absolute_change": 1.0}
                    if response_id == "whole_magnet.hotspot_location"
                    else {"maximum_relative_change": 0.01}
                )
            )
            for response_id in MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS
        },
        "required_response_metric_ids": list(
            MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS
        ),
        "qualification_path": str(output),
    }, output


def test_handler_writes_explicit_incomplete_evidence_for_missing_manifests(
    tmp_path,
):
    stage, output = _stage(tmp_path, existing=1)
    result = qualify_source_convergence(stage, tmp_path)
    value = json.loads(output.read_text(encoding="utf-8"))

    assert result["overall_status"] == INCOMPLETE_STATUS
    assert result["available_candidate_count"] == 1
    assert len(result["missing_configured_files"]) == 6
    assert value["overall_status"] == INCOMPLETE_STATUS
    assert value["default_selection"]["selected_shape"] is None
    assert len(value["missing_evidence"]) >= 3


def test_handler_qualifies_complete_ladder_and_rejects_wrong_inventory(
    tmp_path,
):
    stage, output = _stage(tmp_path)
    result = qualify_source_convergence(stage, tmp_path)
    assert result["overall_status"] == CONVERGED_STATUS
    assert result["available_candidate_count"] == 4
    assert result["default_selection"]["status"] == (
        "SELECTED_LOWEST_COST_QUALIFIED_PAIR"
    )
    assert result["default_selection"]["selected_shape"] == [7, 41, 31]
    assert (
        json.loads(output.read_text(encoding="utf-8"))["overall_status"]
        == CONVERGED_STATUS
    )

    stage["candidates"][-1]["shape"] = [7, 41, 31]
    with pytest.raises(ValueError, match="exactly match the required ladder"):
        qualify_source_convergence(stage, tmp_path)


def test_hotspot_magnet_identity_change_prevents_convergence(tmp_path):
    stage, output = _stage(tmp_path)
    final_report = Path(stage["candidates"][-1]["response_report_path"])
    value = json.loads(final_report.read_text(encoding="utf-8"))
    value["whole_magnet_responses"]["whole_magnet.hotspot_location"][
        "magnet_id"
    ] = "magnet-0002"
    final_report.write_text(json.dumps(value), encoding="utf-8")

    result = qualify_source_convergence(stage, tmp_path)
    assert result["overall_status"] == "NOT_CONVERGED"
    artifact = json.loads(output.read_text(encoding="utf-8"))
    hotspot = artifact["comparisons"][-1]["metrics"][
        "whole_magnet_response.whole_magnet.hotspot_location"
    ]
    assert hotspot["same_magnet"] is False
    assert hotspot["passes"] is False


@pytest.mark.parametrize(
    ("field", "substitution"),
    (
        ("estimator", "signed_net_current"),
        ("crossing_selection", "all_crossings"),
    ),
)
def test_incoming_current_rejects_signed_or_all_crossing_substitutions(
    tmp_path, field, substitution
):
    stage, _ = _stage(tmp_path)
    report = Path(stage["candidates"][-1]["response_report_path"])
    value = json.loads(report.read_text(encoding="utf-8"))
    value["whole_magnet_responses"]["whole_magnet.incoming_neutron_current"][
        field
    ] = substitution
    report.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="inward-only, direction-resolved"):
        qualify_source_convergence(stage, tmp_path)


def test_stage_and_additive_cli_command_are_registered():
    assert "build_source_convergence_ladder" in STAGES
    assert "qualify_source_convergence" in STAGES
    assert STAGE_DEPENDENCIES["build_source_convergence_ladder"] == (
        "validate_inputs",
    )
    assert STAGE_DEPENDENCIES["qualify_source_convergence"] == (
        "build_source",
        "build_source_convergence_ladder",
    )
    assert (
        STAGE_COMMANDS["build-source-convergence-ladder"]
        == "build_source_convergence_ladder"
    )
    assert (
        STAGE_COMMANDS["qualify-source-convergence"]
        == "qualify_source_convergence"
    )
    arguments = _parser().parse_args(
        ["qualify-source-convergence", "workflow.json", "--force"]
    )
    assert arguments.command == "qualify-source-convergence"
    assert arguments.force is True
    ladder_arguments = _parser().parse_args(
        ["build-source-convergence-ladder", "workflow.json", "--force"]
    )
    assert ladder_arguments.command == "build-source-convergence-ladder"
    assert ladder_arguments.force is True


def test_cli_executes_incomplete_source_qualification_without_openmc(
    tmp_path, capsys
):
    stage, output = _stage(tmp_path / "artifacts", existing=1)
    stage["inputs"] = [
        path
        for candidate in stage["candidates"]
        for path in (
            candidate["source_manifest_path"],
            candidate["response_report_path"],
        )
    ]
    stage["outputs"] = [str(output)]
    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "geometry_features": {"ports": False},
                "output_directory": str(tmp_path / "artifacts"),
                "stages": {
                    "validate_inputs": {},
                    "build_source": {},
                    "build_source_convergence_ladder": {},
                    "qualify_source_convergence": stage,
                },
            }
        ),
        encoding="utf-8",
    )
    workflow = MagnetRadiationWorkflow.from_json(config_path)
    workflow.run_stage("validate_inputs", lambda stage, root: {"ok": True})
    workflow.run_stage("build_source", lambda stage, root: {"ok": True})
    workflow.run_stage(
        "build_source_convergence_ladder",
        lambda stage, root: {"ok": True},
    )

    assert main(["qualify-source-convergence", str(config_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "PASS"
    assert printed["result"]["overall_status"] == INCOMPLETE_STATUS
    assert output.is_file()


def test_smoke_and_production_configs_declare_four_distinct_candidates(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PARASTELL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("OPENMC_CROSS_SECTIONS", str(tmp_path / "xs.xml"))
    monkeypatch.setenv("PARASTELL_COMMIT", "a" * 40)
    root = Path(__file__).resolve().parents[1]

    for name in (
        "magnet_radiation_smoke.json",
        "magnet_radiation_production.json",
    ):
        workflow = MagnetRadiationWorkflow.from_json(root / "examples" / name)
        validation = workflow.validate_configuration()
        assert validation["status"] == "PASS"
        stage = workflow.config["stages"]["qualify_source_convergence"]
        shapes = [
            tuple(candidate["shape"]) for candidate in stage["candidates"]
        ]
        paths = [
            candidate["source_manifest_path"]
            for candidate in stage["candidates"]
        ]
        assert shapes == list(REQUIRED_CANDIDATE_SHAPES)
        assert len(paths) == len(set(paths)) == 4
        assert stage["required_response_metric_ids"] == list(
            MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS
        )
        assert set(stage["response_metric_tolerances"]) == set(
            MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS
        )
        response_paths = [
            candidate["response_report_path"]
            for candidate in stage["candidates"]
        ]
        assert len(response_paths) == len(set(response_paths)) == 4

        ladder = workflow.config["stages"]["build_source_convergence_ladder"]
        primary = tuple(ladder["primary_shape"])
        ladder_shapes = {
            tuple(candidate["shape"]) for candidate in ladder["candidates"]
        }
        assert ladder_shapes == set(REQUIRED_CANDIDATE_SHAPES) - {primary}
        assert len(ladder["outputs"]) == len(set(ladder["outputs"])) == 6
        primary_outputs = set(
            workflow.config["stages"]["build_source"]["outputs"]
        )
        assert set(ladder["outputs"]).isdisjoint(primary_outputs)

    production = MagnetRadiationWorkflow.from_json(
        root / "examples" / "magnet_radiation_production.json"
    )
    final_candidate = production.config["stages"][
        "qualify_source_convergence"
    ]["candidates"][-1]
    assert final_candidate["source_manifest_path"].endswith(
        "source_manifest.json"
    )


def test_ladder_handler_builds_exact_non_primary_shapes_without_openmc(
    tmp_path, monkeypatch
):
    primary_shape = REQUIRED_CANDIDATE_SHAPES[0]
    candidates = []
    outputs = []
    for shape in REQUIRED_CANDIDATE_SHAPES[1:]:
        label = "x".join(str(value) for value in shape)
        source_path = tmp_path / f"source-{label}.h5m"
        manifest_path = tmp_path / f"source-{label}.json"
        candidates.append(
            {
                "shape": list(shape),
                "source_mesh_path": str(source_path),
                "source_manifest_path": str(manifest_path),
            }
        )
        outputs.extend((str(source_path), str(manifest_path)))

    def fake_build(candidate):
        source_path = Path(candidate["source_mesh_path"])
        source_path.write_bytes(b"mesh")
        return json.loads(
            _manifest(
                Path(candidate["source_manifest_path"]),
                tuple(candidate["mesh_shape"]),
            ).read_text(encoding="utf-8")
        )

    monkeypatch.setattr(
        magnet_stage_handlers, "_build_source_candidate", fake_build
    )
    stage = {
        "inputs": [],
        "outputs": outputs,
        "geometry_config": str(tmp_path / "config.yaml"),
        "vmec_path": str(tmp_path / "wout.nc"),
        "primary_shape": list(primary_shape),
        "primary_source_mesh_path": str(tmp_path / "source-primary.h5m"),
        "primary_source_manifest_path": str(tmp_path / "source-primary.json"),
        "candidates": candidates,
    }
    result = build_source_convergence_ladder(stage, tmp_path)
    assert result["built_candidate_count"] == 3
    assert {tuple(row["shape"]) for row in result["candidates"]} == set(
        REQUIRED_CANDIDATE_SHAPES[1:]
    )
    assert all(Path(path).is_file() for path in outputs)
    assert all(row["convergence_observables"] for row in result["candidates"])

    stage["outputs"] = stage["outputs"][:-1]
    with pytest.raises(ValueError, match="exactly its six candidate outputs"):
        build_source_convergence_ladder(stage, tmp_path)
