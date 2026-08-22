"""Generate the magnet-inclusive native port production validation model."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pydagmc
from pymoab import core

import parastell.magnet_coils as magnet_coils
import parastell.parastell as ps
from parastell.dagmc_assembly import (
    audit_dagmc_model,
    dagmc_component_names,
    ray_region_sequence,
)

from port_validation import create_validation_invessel_build


def create_validation_stellarator():
    """Construct the native IVB and representative cased filament magnets."""
    native_ivb = create_validation_invessel_build(native=True)
    coils = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "files_for_tests"
        / "coils.example"
    )
    magnets = magnet_coils.MagnetSetFromFilaments(
        coils,
        width=40.0,
        thickness=50.0,
        toroidal_extent=30.0,
        case_thickness=5.0,
        sample_mod=10,
        mat_tag=("SS316L", "Copper"),
    )
    magnets.populate_magnet_coils()
    # Keep the standard filament shapes and nested casing/conductor geometry,
    # but place this validation set above the sector so the fixture represents
    # a physically clear full assembly rather than the intentionally crossing
    # test filaments in their source coordinates.
    magnet_translation = (0.0, 0.0, 1000.0)
    translation = np.asarray(magnet_translation)
    for coil in magnets.magnet_coils:
        coil.coords = coil.coords + translation
        coil.center_of_mass = coil.center_of_mass + translation
        coil.filament.coords = coil.coords
    magnets.build_magnet_coils()
    magnet_boxes = [solid.BoundingBox() for solid in magnets.all_coil_solids]
    magnet_bounds = {
        "lower": [
            min(getattr(box, key) for box in magnet_boxes)
            for key in ("xmin", "ymin", "zmin")
        ],
        "upper": [
            max(getattr(box, key) for box in magnet_boxes)
            for key in ("xmax", "ymax", "zmax")
        ],
    }
    native_max_z = max(
        float(np.max(surface.triangles[:, :, 2]))
        for surface in native_ivb.native_port_complex.surfaces
    )
    if magnet_bounds["lower"][2] <= native_max_z + 10.0:
        raise RuntimeError("Validation magnet set is not clear of the IVB")

    stellarator = ps.Stellarator.__new__(ps.Stellarator)
    comparison_ivb = create_validation_invessel_build(native=False)
    stellarator.invessel_build = comparison_ivb
    stellarator.magnet_set = magnets
    stellarator.port_magnet_collision_report = ()
    stellarator.component_ledger = ()
    stellarator.use_pydagmc = True
    stellarator._logger = native_ivb._logger
    stellarator.check_port_magnet_clearance()
    stellarator.invessel_build = native_ivb
    stellarator.validation_magnet_translation = magnet_translation
    stellarator.validation_magnet_bounds = magnet_bounds
    return stellarator


def file_record(path):
    path = Path(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _surface_between(model, names, left_suffix, right_suffix):
    for surface in model.surfaces:
        sense_names = {
            names[int(volume.id)]
            for volume in surface.senses
            if volume is not None
        }
        if any(name.endswith(left_suffix) for name in sense_names) and any(
            name.endswith(right_suffix) for name in sense_names
        ):
            return surface
    raise RuntimeError(
        f"No assembled surface joins {left_suffix!r} and {right_suffix!r}"
    )


def _outward_ray(model, surface, initial_id, required_prefix):
    triangle = np.asarray(surface.triangle_coords).reshape((-1, 3, 3))[0]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    center = triangle.mean(axis=0)
    candidates = []
    for sign in (-1.0, 1.0):
        direction = sign * normal
        sequence = ray_region_sequence(
            model, center - 1.0e-5 * direction, direction, initial_id
        )
        if tuple(sequence[: len(required_prefix)]) == tuple(required_prefix):
            candidates.append(sequence)
    if not candidates:
        raise RuntimeError(
            f"No facet ray has required prefix {required_prefix!r}"
        )
    return max(
        candidates,
        key=lambda sequence: (
            sequence[-1] == "external",
            len(sequence) == len(set(sequence)),
            -len(sequence),
        ),
    )


def assembled_ray_sequences(stellarator):
    model = stellarator.pydagmc_model
    names = dagmc_component_names(model)
    ids = {name: volume_id for volume_id, name in names.items()}
    complex_ = stellarator.invessel_build.native_port_complex
    port = complex_.port
    anchor = np.asarray(port.placement.anchor)
    axis = np.asarray(port.placement.local_axis)
    reference = np.asarray(port.placement.local_reference)
    origin = anchor - 2.0 * axis
    liner_offset = port.cross_section.radius + port.liner.thickness / 2.0
    blanket_offset = (
        complex_.source_model._port_aperture_half_width(port) + 2.0
    )
    conductor = next(
        name for name in ids if name.endswith("__inner_conductor")
    )
    casing = conductor.replace("__inner_conductor", "__outer_casing")
    conductor_casing = _surface_between(
        model, names, "__inner_conductor", "__outer_casing"
    )
    casing_graveyard = _surface_between(
        model, names, "__outer_casing", "graveyard"
    )
    outer = next(
        surface for surface in model.surfaces if None in surface.senses
    )
    return {
        "port_centerline": list(
            ray_region_sequence(model, origin, axis, ids["plasma"])
        ),
        "liner": list(
            ray_region_sequence(
                model,
                origin + reference * liner_offset,
                axis,
                ids["plasma"],
            )
        ),
        "adjacent_blanket": list(
            ray_region_sequence(
                model,
                origin + reference * blanket_offset,
                axis,
                ids["plasma"],
            )
        ),
        "magnet_conductor": list(
            _outward_ray(
                model,
                conductor_casing,
                ids[conductor],
                (conductor, casing),
            )
        ),
        "magnet_casing": list(
            _outward_ray(
                model,
                casing_graveyard,
                ids[casing],
                (casing, "graveyard"),
            )
        ),
        "graveyard": list(
            _outward_ray(
                model, outer, ids["graveyard"], ("graveyard", "external")
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-mesh-size", type=float, default=20.0)
    parser.add_argument("--max-mesh-size", type=float, default=50.0)
    parser.add_argument("--graveyard-margin", type=float, default=75.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stellarator = create_validation_stellarator()
    stellarator.build_pydagmc_model(
        magnet_exporter="cad_to_dagmc",
        filename="filament_magnets_physical",
        export_dir=args.output_dir,
        min_mesh_size=args.min_mesh_size,
        max_mesh_size=args.max_mesh_size,
        graveyard_margin=args.graveyard_margin,
    )
    h5m = args.output_dir / "ported_sector_magnets_native_dagmc.h5m"
    stellarator.export_pydagmc_model(h5m.name, export_dir=args.output_dir)
    vtk = h5m.with_suffix(".vtk")
    stellarator.pydagmc_model.mb.write_file(str(vtk))
    audit = audit_dagmc_model(stellarator.pydagmc_model)
    loaded_mb = core.Core()
    loaded_mb.load_file(str(h5m))
    loaded_model = pydagmc.Model(str(h5m))
    reloaded_audit = audit_dagmc_model(loaded_model)
    result = {
        "geometry_source": "native_point_cloud_aperture_plus_physical_magnets",
        "filament_magnet_translation": list(
            stellarator.validation_magnet_translation
        ),
        "filament_magnet_bounding_box": stellarator.validation_magnet_bounds,
        "dagmc": audit,
        "dagmc_file_audit": {
            "pymoab_load": True,
            "pydagmc_load": True,
            **reloaded_audit,
        },
        "magnet_volume_ids": stellarator.magnet_volume_ids,
        "port_volume_ids": {
            key: value
            for key, value in stellarator.invessel_build.native_port_complex.volume_ids.items()
            if key.endswith(("__void", "__liner"))
        },
        "graveyard": stellarator.global_graveyard,
        "collision": [
            record.to_dict()
            for record in stellarator.port_magnet_collision_report
        ],
        "component_ledger": [
            record.to_dict() for record in stellarator.component_ledger
        ],
        "native_ivb_topology": (
            stellarator.invessel_build.native_port_complex.topology_summary()
        ),
        "ray_sequences": assembled_ray_sequences(stellarator),
        "files": [file_record(h5m), file_record(vtk)],
    }
    validation = args.output_dir / "full_assembly_validation.json"
    validation.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
