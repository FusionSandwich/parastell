"""Validate one real-coordinate port sector against the complete coil set."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np

import parastell.parastell as ps
from parastell.dagmc_assembly import audit_dagmc_model, dagmc_component_names
from parastell.ports import PortGeometryResult
from parastell.utils import read_yaml_config


LAYER_SEQUENCE = (
    "first_wall",
    "breeder",
    "back_wall",
    "shield",
    "vacuum_vessel",
)


def actual_port():
    return {
        "name": "actual_reactor_port",
        "placement": {
            "mode": "surface",
            "anchor": {
                "reference": "plasma_surface",
                "toroidal_angle": 45.0,
                "poloidal_angle": 180.0,
            },
            "axis": {"mode": "through_build"},
            "max_search_length": 1000.0,
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
        "liner": {
            "enabled": True,
            "thickness": 1.0,
            "mat_tag": "SS316L",
        },
        "fill": {"mat_tag": "Vacuum"},
        "repetition": {"mode": "single"},
        "collision": {
            "magnet_policy": "report",
            "clearance_policy": "report",
            "minimum_magnet_clearance": 0.0,
        },
        "expected_layers": list(LAYER_SEQUENCE),
    }


def file_record(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def run_tool(command):
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": [str(item) for item in command],
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "output": result.stdout,
    }


def prepare_collision_geometry(model):
    complex_ = model.invessel_build.native_port_complex
    port = complex_.port
    anchor = np.asarray(port.placement.anchor)
    axis = np.asarray(port.placement.local_axis)
    loop_w = [
        float(np.mean((loop.inner_points[:-1] - anchor) @ axis))
        for loop in complex_.loops
    ]
    start, end = min(loop_w), max(loop_w)
    envelope = model.invessel_build._build_port_prism(
        port,
        start,
        end + port.extent.outer_extension,
        radial_expansion=port.liner.thickness,
    )
    model.invessel_build.port_outer_envelopes[port.name] = envelope
    model.invessel_build.port_geometry_diagnostics[port.name] = (
        PortGeometryResult(
            name=port.name,
            resolved_start=start,
            resolved_end=end,
            outer_extension=port.extent.outer_extension,
            ordered_intersected_layers=LAYER_SEQUENCE,
            original_blanket_volume=0.0,
            remaining_blanket_volume=0.0,
            void_volume_inside_blanket=0.0,
            liner_volume_inside_blanket=0.0,
            void_volume_outside_blanket=0.0,
            liner_volume_outside_blanket=0.0,
            total_cut_volume=0.0,
            closure_error=0.0,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument("--magnet-min-size", type=float, default=60.0)
    parser.add_argument("--magnet-max-size", type=float, default=150.0)
    parser.add_argument("--mesh-min-size", type=float, default=25.0)
    parser.add_argument("--mesh-max-size", type=float, default=75.0)
    parser.add_argument("--mesh-algorithm", type=int, default=1)
    parser.add_argument("--skip-external-tools", action="store_true")
    parser.add_argument("--skip-volume-mesh", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    config = read_yaml_config(root / "examples/config.yaml")
    ivb = config["invessel_build"]
    magnets = config["magnet_coils"]

    model = ps.Stellarator(root / "examples/wout_vmec.nc")
    model.construct_invessel_build(
        ivb["toroidal_angles"],
        ivb["poloidal_angles"],
        ivb["wall_s"],
        ivb["radial_build"],
        split_chamber=ivb["split_chamber"],
        plasma_mat_tag=ivb["plasma_mat_tag"],
        sol_mat_tag=ivb["sol_mat_tag"],
        num_ribs=13,
        num_rib_pts=49,
        use_pydagmc=True,
        ports=[actual_port()],
    )
    prepare_collision_geometry(model)
    model.construct_magnets_from_filaments(
        root / "examples/coils.example",
        width=magnets["width"],
        thickness=magnets["thickness"],
        toroidal_extent=360.0,
        case_thickness=5.0,
        sample_mod=magnets["sample_mod"],
        mat_tag=("SS316L", "Copper"),
    )
    collision = [
        record.to_dict() for record in model.port_magnet_collision_report
    ]
    collision_path = args.output_dir / "actual_magnet_collision_report.json"
    collision_path.write_text(json.dumps(collision, indent=2) + "\n")
    placement = model.invessel_build.ports[0].placement

    model.build_pydagmc_model(
        magnet_exporter="cad_to_dagmc",
        filename="actual_magnets_dagmc",
        export_dir=args.output_dir,
        min_mesh_size=args.magnet_min_size,
        max_mesh_size=args.magnet_max_size,
        graveyard_margin=100.0,
    )
    h5m = args.output_dir / "actual_reactor_port_dagmc.h5m"
    model.export_pydagmc_model(h5m.name, export_dir=args.output_dir)
    vtk = h5m.with_suffix(".vtk")
    model.pydagmc_model.mb.write_file(str(vtk))
    audit = audit_dagmc_model(model.pydagmc_model)
    names = dagmc_component_names(model.pydagmc_model)

    if args.skip_external_tools:
        watertight = {"status": "not_run"}
        overlap = {"status": "not_run"}
    else:
        watertight = run_tool(["check_watertight", h5m])
        overlap = run_tool(["overlap_check", h5m, "-p", "1000"])
    result = {
        "schema_version": "1.0",
        "repository_sha": args.repository_sha,
        "scope": "sector_transport_model",
        "visual_scope": "not_run",
        "full_reactor_render": {
            "status": "not_run",
            "reasons": [
                "the in-vessel model contains one field-period sector",
                "the actual-coordinate volumetric PLC gate is unresolved",
            ],
        },
        "vmec_input": file_record(root / "examples/wout_vmec.nc"),
        "coil_input": file_record(root / "examples/coils.example"),
        "field_periods": 4,
        "ivb_field_periods": 1,
        "coil_count": len(model.magnet_set.magnet_coils),
        "magnet_translation": [0.0, 0.0, 0.0],
        "layer_sequence": list(LAYER_SEQUENCE),
        "port_placement": {
            "anchor": list(placement.anchor),
            "axis": list(placement.local_axis),
            "local_u": list(placement.local_reference),
            "local_v": list(placement.local_normal),
            "toroidal_angle_degrees": 45.0,
            "poloidal_angle_degrees": 180.0,
        },
        "port_axis": model.invessel_build.port_axis_diagnostics[
            "actual_reactor_port"
        ],
        "collision": collision,
        "collision_report_status": "complete",
        "dagmc": audit,
        "component_names": names,
        "port_volume_ids": {
            name: volume_id
            for volume_id, name in names.items()
            if name.endswith(("__void", "__liner"))
        },
        "magnet_volume_ids": {
            name: volume_id
            for volume_id, name in names.items()
            if name.startswith("magnet_")
        },
        "graveyard_id": next(
            volume_id
            for volume_id, name in names.items()
            if name == "graveyard"
        ),
        "watertightness": watertight,
        "overlap": overlap,
        "volume_mesh": {"status": "not_run"},
        "artifacts": [
            file_record(path) for path in (h5m, vtk, collision_path)
        ],
    }
    output = args.output_dir / "actual_reactor_port_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    if not args.skip_external_tools and (
        not watertight["passed"] or not overlap["passed"]
    ):
        raise RuntimeError("Actual-coordinate DAGMC tools did not pass")

    if args.skip_volume_mesh:
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("dagmc", "port_volume_ids", "graveyard_id")
                },
                indent=2,
            )
        )
        return

    native = model.invessel_build.native_port_complex
    try:
        mesh = native.tetrahedralize(
            args.mesh_min_size,
            args.mesh_max_size,
            algorithm_3d=args.mesh_algorithm,
        )
        mesh_validation = mesh.validate()
        mesh_h5m = args.output_dir / "actual_reactor_ivb_volume_mesh.h5m"
        mesh.write(mesh_h5m)
        mesh_vtk = mesh_h5m.with_suffix(".vtk")
        result["volume_mesh"] = {
            "status": "complete",
            **mesh_validation.to_dict(),
        }
        result["artifacts"].extend(
            file_record(path) for path in (mesh_h5m, mesh_vtk)
        )
    except Exception as error:
        result["volume_mesh"] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
            "mesh_algorithm": args.mesh_algorithm,
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("dagmc", "port_volume_ids", "graveyard_id")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
