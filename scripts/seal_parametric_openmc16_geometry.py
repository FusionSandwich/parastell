"""Create a PyMOAB-derived handoff for an OpenMC 0.16-only runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from parastell.parametric_openmc16_geometry_seal import (
    create_geometry_transport_seal,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_control", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-control-sha256", required=True)
    args = parser.parse_args()
    create_geometry_transport_seal(
        args.build_control,
        args.expected_control_sha256,
        args.output_directory,
    )


if __name__ == "__main__":
    main()
