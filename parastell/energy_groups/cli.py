"""Command-line interface for the energy-structure registry."""

from __future__ import annotations

import argparse
import json

from .registry import compare_structures
from .registry import get_structure
from .registry import list_structures
from .registry import load_custom_structure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parastell energy-groups")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list built-in structures")
    listing.add_argument("--json", action="store_true")
    inspect = commands.add_parser("inspect", help="inspect one structure")
    inspect.add_argument("name")
    inspect.add_argument("--descending", action="store_true")
    inspect.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate", help="validate a custom file")
    validate.add_argument("path")
    validate.add_argument(
        "--particle", choices=("neutron", "photon"), required=True
    )
    validate.add_argument("--units", default="eV")
    compare = commands.add_parser("compare", help="compare exact boundaries")
    compare.add_argument("first")
    compare.add_argument("second")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        rows = [structure.as_dict() for structure in list_structures()]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                groups = (
                    "continuous"
                    if row["group_count"] is None
                    else row["group_count"]
                )
                print(
                    f"{row['name']:<20} {row['particle']:<8} "
                    f"{str(groups):>10}  {row['status']}"
                )
    elif args.command == "inspect":
        data = get_structure(args.name).as_dict(descending=args.descending)
        print(json.dumps(data, indent=2))
    elif args.command == "validate":
        data = load_custom_structure(
            args.path, particle=args.particle, units=args.units
        ).as_dict()
        print(json.dumps(data, indent=2))
    else:
        print(
            json.dumps(compare_structures(args.first, args.second), indent=2)
        )
    return 0
