"""Generate native DAGMC and discrete-PLC port validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import parastell.parastell as ps

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
        tetrahedralize=False,
        min_mesh_size=args.min_mesh_size,
        max_mesh_size=args.max_mesh_size,
    )
    if not args.skip_volume_mesh:
        stellarator = ps.Stellarator.__new__(ps.Stellarator)
        stellarator.invessel_build = model
        mesh_path = stellarator.export_invessel_build_mesh_moab(
            tuple(model.radial_build.radial_build),
            "ported_sector_native_volume_mesh",
            export_dir=args.output_dir,
            geometry_source="native_surface_complex",
            min_mesh_size=args.min_mesh_size,
            max_mesh_size=args.max_mesh_size,
        )
        validation = model.native_volume_mesh_validation.to_dict()
        validation["geometry_source"] = "native_surface_complex"
        validation["h5m"] = str(mesh_path)
        validation["vtk"] = str(mesh_path.with_suffix(".vtk"))
        validation_path = (
            args.output_dir / "native_volume_mesh_validation.json"
        )
        validation_path.write_text(json.dumps(validation, indent=2) + "\n")
        result["volume_mesh_validation"] = validation
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
