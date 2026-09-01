"""Create-only compiler for the staged WISTELL-D P00--P07 poster cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parastell.blanket_materials import (
    write_blanket_material_bundle,
    write_openmc_materials_xml,
)
from parastell.poster_parametric_cases import (
    build_poster_parametric_case_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pure_materials_json", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--temperature-k", type=float, default=293.6)
    args = parser.parse_args()

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = build_poster_parametric_case_plan(
        args.pure_materials_json,
        geometry_path=args.geometry,
        source_path=args.source,
        temperature_K=args.temperature_k,
    )
    for case in plan["cases"]:
        bundle = case.get("material_bundle")
        if bundle is None:
            continue
        case_directory = output / case["case_id"]
        case_directory.mkdir()
        write_blanket_material_bundle(
            case_directory / "blanket_material_bundle.json", bundle
        )
        write_openmc_materials_xml(case_directory / "materials.xml", bundle)

    plan_path = output / "POSTER_PARAMETRIC_CASE_PLAN.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(plan_path)


if __name__ == "__main__":
    main()
