#!/usr/bin/env python3
"""Export OpenMC 0.16 XML from a split-runtime geometry transport seal."""

from __future__ import annotations

import argparse
from pathlib import Path

from parastell.parametric_openmc16_sealed_runtime import (
    export_sealed_openmc16_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seal", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-seal-sha256", required=True)
    args = parser.parse_args()
    export_sealed_openmc16_model(
        args.seal, args.expected_seal_sha256, args.output_directory
    )


if __name__ == "__main__":
    main()
