"""Run the representative native-port faceting and PLC convergence study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

import parastell.invessel_build as ivb
from parastell.native_port_geometry import build_native_port_surface_complex

from port_validation import (
    ValidationReferenceSurface,
    create_validation_invessel_build,
)


def create_model(num_ribs, num_rib_pts):
    toroidal_angles = [0.0, 10.0, 20.0, 30.0]
    poloidal_angles = [0.0, 90.0, 180.0, 270.0, 360.0]
    thickness = np.ones((4, 5)) * 5.0
    build = ivb.RadialBuild(
        toroidal_angles,
        poloidal_angles,
        1.08,
        {
            name: {"thickness_matrix": thickness, "mat_tag": name}
            for name in (
                "first_wall",
                "breeder",
                "shield",
                "vacuum_vessel",
            )
        },
    )
    port = {
        "name": "real_heating_port",
        "placement": {
            "mode": "surface",
            "anchor": {
                "reference": "plasma_surface",
                "toroidal_angle": 15.0,
                "poloidal_angle": 0.0,
            },
            "axis": {"mode": "outward_normal"},
            "roll": 0.0,
            "max_search_length": 500.0,
        },
        "cross_section": {"shape": "circle", "radius": 3.0},
        "extent": {
            "start": {"reference": "plasma_surface"},
            "end": {
                "reference": "layer",
                "layer": "vacuum_vessel",
                "fraction": 1.0,
            },
            "outer_extension": 25.0,
        },
        "liner": {"enabled": True, "thickness": 1.0, "mat_tag": "SS316L"},
        "fill": {"mat_tag": "Vacuum"},
        "repetition": {"mode": "single"},
        "expected_layers": [
            "first_wall",
            "breeder",
            "shield",
            "vacuum_vessel",
        ],
    }
    model = ivb.InVesselBuild(
        ValidationReferenceSurface(),
        build,
        num_ribs=num_ribs,
        num_rib_pts=num_rib_pts,
        ports=[port],
    )
    model.populate_surfaces()
    model.calculate_loci()
    return model


def tool_result(command):
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "output": completed.stdout,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-mesh-size", type=float, default=15.0)
    parser.add_argument("--max-mesh-size", type=float, default=45.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cad_model = create_validation_invessel_build(native=False)
    comparison_names = (
        "first_wall",
        "breeder",
        "shield",
        "vacuum_vessel",
        "real_heating_port__void",
        "real_heating_port__liner",
    )
    cad_reference = {
        name: float(cad_model.Components[name].Volume())
        for name in comparison_names
    }
    cases = (
        (12, 45, 0.05),
        (13, 49, 0.075),
        (13, 49, 0.060),
        (13, 49, 0.05),
        (17, 65, 0.05),
    )
    results = []
    production_reference = None
    for num_ribs, num_rib_pts, chord_tolerance in cases:
        name = f"r{num_ribs}_p{num_rib_pts}_c{chord_tolerance:g}".replace(
            ".", "p"
        )
        case_dir = args.output_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        model = create_model(num_ribs, num_rib_pts)
        complex_ = build_native_port_surface_complex(
            model,
            aperture_chord_tolerance=chord_tolerance,
            vertex_merge_tolerance=1.0e-9,
        )
        h5m = complex_.write_dagmc(case_dir / "native_dagmc.h5m")
        structural = complex_.validate()
        references = complex_.reference_volumes()
        mesh = complex_.tetrahedralize(args.min_mesh_size, args.max_mesh_size)
        mesh_result = mesh.validate()
        mesh_path = mesh.write(case_dir / "native_volume_mesh.h5m")
        watertight = tool_result(["check_watertight", str(h5m)])
        overlap = tool_result(["overlap_check", str(h5m), "-p", "10"])
        if not watertight["passed"] or not overlap["passed"]:
            raise RuntimeError(f"Convergence tools failed for {name}")
        if num_ribs == 13 and num_rib_pts == 49 and chord_tolerance == 0.05:
            production_reference = {
                key: references[key] for key in comparison_names
            }
        results.append(
            {
                "case": name,
                "num_ribs": num_ribs,
                "num_rib_pts": num_rib_pts,
                "aperture_chord_tolerance": chord_tolerance,
                "vertex_merge_tolerance": complex_.vertex_merge_tolerance,
                "loop_point_counts": [
                    len(loop.inner_points) - 1 for loop in complex_.loops
                ],
                "surface_count": len(complex_.surfaces),
                "facet_count": complex_.triangle_count,
                "reference_volumes": {
                    key: references[key] for key in comparison_names
                },
                "cadquery_reference_relative_difference": {
                    key: abs(references[key] - cad_reference[key])
                    / cad_reference[key]
                    for key in comparison_names
                },
                "centerline_region_sequence": list(
                    structural.centerline_region_sequence
                ),
                "watertightness": watertight,
                "overlap": overlap,
                "tetrahedron_count": mesh_result.tetrahedron_count,
                "mesh_quality": {
                    "minimum_scaled_jacobian": min(
                        mesh_result.region_minimum_scaled_jacobian.values()
                    ),
                    "minimum_mean_ratio": min(
                        mesh_result.region_minimum_mean_ratio.values()
                    ),
                    "minimum_radius_ratio": min(
                        mesh_result.region_minimum_radius_ratio.values()
                    ),
                    "minimum_dihedral_angle": min(
                        mesh_result.region_minimum_dihedral_angle.values()
                    ),
                    "maximum_dihedral_angle": max(
                        mesh_result.region_maximum_dihedral_angle.values()
                    ),
                    "quality_threshold_counts": (
                        mesh_result.region_quality_threshold_counts
                    ),
                },
                "files": {"dagmc": str(h5m), "volume_mesh": str(mesh_path)},
                "topology_sha256": complex_.topology_summary()["sha256"],
            }
        )
    for result in results:
        result["production_reference_relative_difference"] = {
            key: abs(value - production_reference[key])
            / production_reference[key]
            for key, value in result["reference_volumes"].items()
        }
    report = {
        "schema_version": "1.0",
        "excluded_coarse_resolutions": [
            {
                "num_ribs": 9,
                "num_rib_pts": 33,
                "reason": "end:layer aperture loop is not ordered outward",
            },
            {
                "num_ribs": 11,
                "num_rib_pts": 41,
                "reason": "end:layer aperture loop is not ordered outward",
            },
        ],
        "excluded_chord_tolerances": [
            {
                "aperture_chord_tolerance": 0.1,
                "reason": "16-point end:layer aperture loop is not ordered outward",
            },
            {
                "aperture_chord_tolerance": 0.025,
                "reason": "32-point end:layer aperture loop is not ordered outward",
            },
        ],
        "production_default": {
            "num_ribs": 13,
            "num_rib_pts": 49,
            "aperture_chord_tolerance": 0.05,
            "vertex_merge_tolerance": 1.0e-9,
            "min_mesh_size": args.min_mesh_size,
            "max_mesh_size": args.max_mesh_size,
        },
        "cadquery_reference_volumes": cad_reference,
        "cases": results,
    }
    path = args.output_dir / "faceting_convergence.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
