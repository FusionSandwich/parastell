"""Re-export a retained hash-matched OpenMC 0.16 bank as facet-complete v2.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parastell.retained_openmc16_handoff import reexport_retained_handoff_v22


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-recipe", type=Path, required=True)
    parser.add_argument("--dagmc", type=Path, required=True)
    parser.add_argument("--retained-v21", type=Path, required=True)
    parser.add_argument("--statepoint", type=Path, required=True)
    parser.add_argument("--surface-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--exporter-commit", required=True)
    args = parser.parse_args()
    manifest = reexport_retained_handoff_v22(
        args.output,
        geometry_recipe_path=args.geometry_recipe,
        dagmc_path=args.dagmc,
        retained_v21_path=args.retained_v21,
        statepoint_path=args.statepoint,
        surface_source_path=args.surface_source,
        exporter_commit=args.exporter_commit,
    )
    receipt = manifest["retained_reexport_receipt"] | {
        "schema": "parastell.retained_openmc16_v22_reexport/v1",
        "record_count": manifest["record_count"],
        "integrated_current_per_source": manifest["integrated_current"],
        "facet_complete_boundary": manifest["facet_complete_boundary"],
    }
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.receipt}")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
