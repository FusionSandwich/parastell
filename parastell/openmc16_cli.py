"""Public command-line surface for the OpenMC 0.16 magnet workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .magnet_boundary_envelope import read_handoff
from .magnet_energy_architecture import (
    authoritative_neutron_edges,
    photon_master_edges,
)
from .openmc16 import capability_report


COMMANDS = (
    "environment-audit",
    "source-audit",
    "geometry-inventory",
    "envelope-validation",
    "prepare-tallies",
    "run-transport",
    "export-phase-space",
    "validate-closure",
    "compare-energy-structures",
    "generate-response-groups",
    "project-deterministic-source",
    "replay-multilayer-model",
    "inspect-particle-production",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parastell-openmc16-handoff")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("environment-audit")
    audit.add_argument("--output", type=Path)
    envelope = sub.add_parser("envelope-validation")
    envelope.add_argument("handoff", type=Path)
    groups = sub.add_parser("compare-energy-structures")
    groups.add_argument("--output", type=Path)
    for name in COMMANDS:
        if name in {
            "environment-audit",
            "envelope-validation",
            "compare-energy-structures",
        }:
            continue
        command = sub.add_parser(name)
        command.add_argument("config", type=Path)
    return parser


def _write(value, output: Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True)
    if output is None:
        print(text)
    else:
        output.write_text(text + "\n", encoding="utf-8")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "environment-audit":
        _write(capability_report(), args.output)
        return 0
    if args.command == "envelope-validation":
        manifest, envelope, bank = read_handoff(args.handoff)
        _write(
            {
                "schema": manifest["schema"],
                "envelope_id": envelope.envelope_id,
                "surface_ids": envelope.surface_ids,
                "records": len(bank),
                "watertight": envelope.watertight,
            },
            None,
        )
        return 0
    if args.command == "compare-energy-structures":
        result = {
            "neutron": {
                "7-group": 7,
                "CCFE-709": len(authoritative_neutron_edges("CCFE-709")) - 1,
                "UKAEA-1102": len(authoritative_neutron_edges("UKAEA-1102"))
                - 1,
            },
            "photon": {"reference": len(photon_master_edges()) - 1},
        }
        _write(result, args.output)
        return 0
    raise RuntimeError(
        f"{args.command} requires the production workflow configuration adapter"
    )


if __name__ == "__main__":
    raise SystemExit(main())
