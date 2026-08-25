"""Dependency-safe top-level CLI dispatcher."""

from __future__ import annotations

import argparse
import sys


def _help_parser():
    parser = argparse.ArgumentParser(
        prog="parastell",
        description=(
            "Build ParaStell geometry with the legacy CONFIG arguments, or use "
            "the additive 'magnet-field' producer namespace."
        ),
    )
    parser.add_argument(
        "filename", nargs="?", help="legacy ParaStell YAML configuration"
    )
    parser.add_argument("-e", "--export_dir", metavar="")
    parser.add_argument("-l", "--logger", action="store_true")
    parser.add_argument("-i", "--ivb", action="store_true")
    parser.add_argument("-m", "--magnets", action="store_true")
    parser.add_argument("-s", "--source", action="store_true")
    parser.add_argument("-n", "--nwl", action="store_true")
    parser.epilog = "Additional namespace: parastell magnet-field --help"
    return parser


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "magnet-field":
        from .magnet_field_cli import main as magnet_field_main

        return magnet_field_main(arguments[1:])
    if not arguments or arguments in (["-h"], ["--help"]):
        _help_parser().parse_args(arguments)
        return 0
    # Preserve the original positional CLI and import its CAD stack only here.
    from .parastell import parastell

    original = sys.argv
    try:
        sys.argv = [original[0], *arguments]
        return parastell()
    finally:
        sys.argv = original
