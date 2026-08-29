#!/usr/bin/env python3
"""Create a hash-bound coil-filament inventory without constructing CAD."""

from __future__ import annotations

import argparse
from pathlib import Path

from parastell.coil_filament_inventory import (
    build_coil_filament_inventory,
    write_inventory_create_only,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coil_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-line", type=int, default=3)
    parser.add_argument("--scale-to-cm", type=float, default=100.0)
    parser.add_argument("--input-coordinate-units", default="m")
    parser.add_argument("--sector-start-degrees", type=float, default=0.0)
    parser.add_argument("--sector-extent-degrees", type=float, default=360.0)
    args = parser.parse_args()
    inventory = build_coil_filament_inventory(
        args.coil_file,
        start_line=args.start_line,
        scale_to_cm=args.scale_to_cm,
        input_coordinate_units=args.input_coordinate_units,
        sector_start_degrees=args.sector_start_degrees,
        sector_extent_degrees=args.sector_extent_degrees,
    )
    write_inventory_create_only(args.output, inventory)


if __name__ == "__main__":
    main()
