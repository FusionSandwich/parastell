#!/usr/bin/env python3
"""Resolve and create one continuous-global/local-coil CAD case."""

from __future__ import annotations

import argparse
from pathlib import Path

from parastell.continuous_local_magnet_case import (
    resolve_continuous_local_magnet_case,
)
from parastell.continuous_local_magnet_geometry import (
    build_continuous_local_magnet_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    plan = resolve_continuous_local_magnet_case(args.case)
    build_continuous_local_magnet_geometry(plan, args.output_root)


if __name__ == "__main__":
    main()
