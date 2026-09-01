"""Create a hash-bound ParaStell blanket material bundle and OpenMC XML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parastell.blanket_materials import (
    PRESETS,
    build_blanket_material_bundle,
    write_blanket_material_bundle,
    write_openmc_materials_xml,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pure_materials_json", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="dcll")
    parser.add_argument(
        "--role-recipe-overrides",
        type=Path,
        help="JSON object mapping component roles to approved recipe IDs",
    )
    parser.add_argument("--temperature-k", type=float, default=293.6)
    args = parser.parse_args()
    overrides = None
    if args.role_recipe_overrides is not None:
        overrides = json.loads(
            args.role_recipe_overrides.resolve(strict=True).read_text(
                encoding="utf-8"
            )
        )
    bundle = build_blanket_material_bundle(
        args.pure_materials_json,
        preset=args.preset,
        role_recipe_overrides=overrides,
        temperature_K=args.temperature_k,
    )
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_blanket_material_bundle(
        output / "blanket_material_bundle.json", bundle
    )
    write_openmc_materials_xml(output / "materials.xml", bundle)
    print(output)


if __name__ == "__main__":
    main()
