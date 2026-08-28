#!/usr/bin/env python3
"""Validate, plan, or build generic ParaStell source CAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from parastell.parametric_geometry import (
    build_source_cad,
    docker_build_argv,
    execute_docker_build,
    load_plan,
    read_vmec_metadata,
    resolve_geometry,
)


def _load(arguments: argparse.Namespace):
    return load_plan(arguments.config, arguments.input_root)


def validate(arguments: argparse.Namespace) -> None:
    plan = _load(arguments)
    resolved = resolve_geometry(plan)
    live_vmec = read_vmec_metadata(plan)
    print(
        json.dumps(
            {
                "plan": plan.receipt(),
                "live_vmec_metadata": live_vmec,
                "resolved_layers": resolved.statistics(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def build(arguments: argparse.Namespace) -> None:
    plan = _load(arguments)
    resolved = resolve_geometry(plan)
    manifest = build_source_cad(
        resolved,
        arguments.output_root,
        source_revision=arguments.source_revision,
        repository_root=REPOSITORY_ROOT,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def docker_command(arguments: argparse.Namespace) -> None:
    plan = _load(arguments)
    resolve_geometry(plan)
    argv = docker_build_argv(
        plan,
        arguments.output_root,
        repository_root=arguments.repository_root,
        source_revision=arguments.source_revision,
    )
    payload = {"argv": argv, "executed": bool(arguments.execute)}
    if arguments.execute:
        payload["return_code"] = execute_docker_build(argv)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("return_code", 0) != 0:
        raise SystemExit(payload["return_code"])


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create-only parametric ParaStell geometry infrastructure"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser("validate")
    _common(validate_parser)
    validate_parser.set_defaults(function=validate)

    build_parser = commands.add_parser("build-source-cad")
    _common(build_parser)
    build_parser.add_argument("--output-root", required=True)
    build_parser.add_argument("--source-revision", required=True)
    build_parser.set_defaults(function=build)

    docker_parser = commands.add_parser("docker-command")
    _common(docker_parser)
    docker_parser.add_argument("--output-root", required=True)
    docker_parser.add_argument("--repository-root", required=True)
    docker_parser.add_argument("--source-revision", required=True)
    docker_parser.add_argument(
        "--execute",
        action="store_true",
        help="explicitly execute the validated command; omission prints only",
    )
    docker_parser.set_defaults(function=docker_command)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
