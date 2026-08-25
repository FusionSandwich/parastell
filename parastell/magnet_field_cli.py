"""Additive ``parastell magnet-field`` command namespace."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


STAGE_COMMANDS = {
    "validate-inputs": "validate_inputs",
    "build-source": "build_source",
    "build-source-convergence-ladder": "build_source_convergence_ladder",
    "qualify-source-convergence": "qualify_source_convergence",
    "build": "build_geometry",
    "validate-geometry": "validate_geometry",
    "inventory-magnets": "inventory_magnets",
    "prepare-openmc": "prepare_unbiased_model",
    "build-tally-meshes": "build_tally_meshes",
    "build-ww-mesh": "build_weight_window_mesh",
    "prepare-unbiased": "prepare_unbiased_model",
    "run-unbiased": "run_unbiased_model",
    "run-unbiased-campaign": "run_unbiased_qualification_campaign",
    "qualify-statistics": "qualify_production_statistics",
    "prepare-ww-generation": "prepare_weight_window_generation",
    "run-ww-generation": "run_weight_window_generation",
    "run-ww-qualification": "run_weight_window_qualification_campaign",
    "qualify-ww-stage": "qualify_weight_windows",
    "prepare-production": "prepare_production_model",
    "postprocess": "postprocess",
    "export-bundle": "export_bundle",
    "render-diagnostics": "render_diagnostics",
}


def _json(value):
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _parser():
    parser = argparse.ArgumentParser(prog="parastell magnet-field")
    commands = parser.add_subparsers(dest="command", required=True)
    validate_config = commands.add_parser("validate-config")
    validate_config.add_argument("config")
    list_magnets = commands.add_parser("list-magnets")
    list_magnets.add_argument("model")
    inspect_magnet = commands.add_parser("inspect-magnet")
    inspect_magnet.add_argument("model")
    inspect_magnet.add_argument("--magnet-id", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("config")
    inspect_ww = commands.add_parser("inspect-ww")
    inspect_ww.add_argument("contract")
    qualify = commands.add_parser("qualify-ww")
    qualify.add_argument("unbiased")
    qualify.add_argument("weight_window")
    qualify.add_argument("--output")
    validate = commands.add_parser("validate")
    validate.add_argument("--bundle")
    validate.add_argument("--config")
    for command in STAGE_COMMANDS:
        stage = commands.add_parser(command)
        stage.add_argument("config")
        stage.add_argument("--force", action="store_true")
    return parser


def _load_workflow(path):
    from .magnet_field_workflow import MagnetRadiationWorkflow

    return MagnetRadiationWorkflow.from_file(path)


def _load_handler(workflow, stage):
    reference = workflow.config.get("stage_handlers", {}).get(stage)
    if not reference:
        module = importlib.import_module(".magnet_stage_handlers", __package__)
        handler = getattr(module, stage, None)
        if handler is None:
            raise ValueError(
                f"configuration has no explicit stage_handlers entry for {stage!r}"
            )
        return handler
    module_name, separator, attribute = str(reference).partition(":")
    if not separator:
        raise ValueError("stage handler must use 'module:function' syntax")
    handler = getattr(importlib.import_module(module_name), attribute)
    if not callable(handler):
        raise TypeError(f"stage handler {reference!r} is not callable")
    return handler


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "validate-config":
        workflow = _load_workflow(args.config)
        _json(
            {
                **workflow.validate_configuration(),
                "config": str(workflow.config_path),
                "ports": "PROHIBITED_AND_ABSENT",
            }
        )
        return 0
    if args.command in {"list-magnets", "inspect-magnet"}:
        from .dagmc_envelope import discover_magnet_volumes
        from .dagmc_envelope import select_magnet_pairs

        inventory = discover_magnet_volumes(args.model)
        if args.command == "list-magnets":
            _json(inventory.to_dict())
        else:
            _json(select_magnet_pairs(inventory, args.magnet_id)[0].to_dict())
        return 0
    if args.command == "inspect":
        _json(_load_workflow(args.config).inspect())
        return 0
    if args.command == "inspect-ww":
        value = json.loads(Path(args.contract).read_text(encoding="utf-8"))
        artifact = Path(value["weight_window_artifact"]["path"])
        from .weight_windows import _sha256

        value["artifact_hash_valid"] = (
            artifact.is_file()
            and _sha256(artifact) == value["weight_window_artifact"]["sha256"]
        )
        _json(value)
        return 0
    if args.command == "qualify-ww":
        from .weight_windows import qualify_weight_windows

        unbiased = json.loads(Path(args.unbiased).read_text(encoding="utf-8"))
        weighted = json.loads(
            Path(args.weight_window).read_text(encoding="utf-8")
        )
        value = qualify_weight_windows(unbiased, weighted)
        if args.output:
            Path(args.output).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(value)
        return 0
    if args.command == "validate":
        result = {}
        if args.bundle:
            from .magnet_radiation_field_bundle import (
                read_radiation_field_bundle,
            )

            result["bundle"] = read_radiation_field_bundle(args.bundle)
        if args.config:
            result["workflow"] = _load_workflow(args.config).inspect()
        if not result:
            raise ValueError("validate requires --bundle and/or --config")
        _json(result)
        return 0
    stage = STAGE_COMMANDS[args.command]
    workflow = _load_workflow(args.config)
    result = workflow.run_stage(
        stage, _load_handler(workflow, stage), force=args.force
    )
    _json(result.manifest)
    return 0
