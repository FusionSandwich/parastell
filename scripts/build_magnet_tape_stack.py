#!/usr/bin/env python3
"""Create a validated solver-neutral heterogeneous magnet tape-stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parastell.magnet_tape_stack import (
    build_tape_stack_manifest,
    write_manifest_create_only,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("definition", type=Path)
    parser.add_argument(
        "materials",
        type=Path,
        help="JSON list of parastell.material_identity/v1.0.0 records",
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    material_records = json.loads(
        args.materials.resolve(strict=True).read_text(encoding="utf-8")
    )
    manifest = build_tape_stack_manifest(args.definition, material_records)
    write_manifest_create_only(args.output, manifest)


if __name__ == "__main__":
    main()
