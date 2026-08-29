"""Create a hash-bound direct-period OpenMC 0.16 model from qualified inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from parastell.parametric_openmc16_model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_control", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-control-sha256", required=True)
    args = parser.parse_args()
    build_model(
        args.build_control,
        args.expected_control_sha256,
        args.output_directory.resolve(),
    )


if __name__ == "__main__":
    main()
