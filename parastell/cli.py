"""Lightweight top-level ParaStell command dispatcher."""

from __future__ import annotations

import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "energy-groups":
        from .energy_groups.cli import main as energy_groups_main

        return energy_groups_main(arguments[1:])

    from .parastell import parastell

    if argv is None:
        return parastell()
    original = sys.argv
    try:
        sys.argv = [original[0], *arguments]
        return parastell()
    finally:
        sys.argv = original
