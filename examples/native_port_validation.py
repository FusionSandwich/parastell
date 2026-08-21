"""Generate native DAGMC and discrete-PLC port validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from port_validation import create_validation_invessel_build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-mesh-size", type=float, default=8.0)
    parser.add_argument("--max-mesh-size", type=float, default=30.0)
    parser.add_argument("--skip-volume-mesh", action="store_true")
    args = parser.parse_args()
    model = create_validation_invessel_build(native=True)
    result = model.export_native_port_artifacts(
        args.output_dir,
        tetrahedralize=not args.skip_volume_mesh,
        min_mesh_size=args.min_mesh_size,
        max_mesh_size=args.max_mesh_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
