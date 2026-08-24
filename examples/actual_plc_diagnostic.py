"""Run the frozen six-case actual in-vessel radial PLC diagnostic.

This entry point constructs no magnets and performs no transport work.  Its
outputs are PLC/topology diagnostics, never qualified transport geometry.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import subprocess
import sys
import time

# Bind imports to this exact checkout rather than an installed ParaStell copy.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import parastell.parastell as ps
from parastell.native_port_geometry import (
    DEFAULT_APERTURE_CHORD_TOLERANCE,
    DEFAULT_VERTEX_MERGE_TOLERANCE,
    build_native_port_surface_complex,
)
from parastell.plc_diagnostic import (
    CASE_MATRIX,
    INTERSECTION_TOLERANCE_CM,
    SPATIAL_CELL_SIZE_CM,
    canonical_sha256,
    case_by_name,
    classify_case,
    exception_record,
    file_record,
    fresh_directory,
    inspect_edge_facet_intersections,
    process_failure_record,
    serializable_intersection_report,
    terminal_classification,
    write_json,
    write_minimal_vtk,
)
from parastell.utils import read_yaml_config
from reactor_port_validation import actual_port


SCOPE = "PLC/topology diagnostic; not qualified transport geometry"
MIN_MESH_SIZE_CM = 25.0
MAX_MESH_SIZE_CM = 75.0
MESH_ALGORITHM_3D = 1
RELEVANT_INPUTS = (
    "examples/wout_vmec.nc",
    "examples/config.yaml",
    "examples/coils.example",
    "examples/reactor_port_validation.py",
    "examples/actual_plc_diagnostic.py",
    "parastell/parastell.py",
    "parastell/invessel_build.py",
    "parastell/native_port_geometry.py",
    "parastell/plc_diagnostic.py",
    "parastell/port_aperture.py",
    "parastell/ports.py",
    "parastell/pystell/read_vmec.py",
    "parastell/utils.py",
)


def _git(root, *arguments):
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _module_version(name):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        if version is None:
            try:
                from importlib.metadata import version as distribution_version

                version = distribution_version(name)
            except Exception:
                version = None
        return {
            "status": "present",
            "version": version,
            "path": getattr(module, "__file__", None),
        }
    except Exception as error:
        return {
            "status": "absent",
            "error_type": type(error).__name__,
            "message": str(error),
        }


def environment_manifest(
    image_reference=None, image_id=None, image_digest=None
):
    packages = {
        name: _module_version(name)
        for name in (
            "parastell",
            "numpy",
            "scipy",
            "netCDF4",
            "yaml",
            "cadquery",
            "gmsh",
            "pydagmc",
            "pymoab",
            "cad_to_dagmc",
            "openmc",
            "pytest",
        )
    }
    return {
        "schema_version": "1.0",
        "scope": SCOPE,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "container": {
            "image_reference": image_reference,
            "image_id": image_id,
            "image_digest": image_digest,
            "network": "none",
        },
        "packages": packages,
        "declared_tolerances": {
            "geometry_units": "cm",
            "aperture_chord_tolerance_cm": DEFAULT_APERTURE_CHORD_TOLERANCE,
            "vertex_merge_tolerance_cm": DEFAULT_VERTEX_MERGE_TOLERANCE,
            "intersection_tolerance_cm": INTERSECTION_TOLERANCE_CM,
            "spatial_index_cell_size_cm": SPATIAL_CELL_SIZE_CM,
            "gmsh_mesh_size_min_cm": MIN_MESH_SIZE_CM,
            "gmsh_mesh_size_max_cm": MAX_MESH_SIZE_CM,
            "gmsh_algorithm_3d": MESH_ALGORITHM_3D,
        },
    }


def input_manifest(root, repository):
    records = [
        file_record(root / relative, logical_name=relative)
        for relative in RELEVANT_INPUTS
    ]
    port_configuration = actual_port()
    return {
        "schema_version": "1.0",
        "scope": SCOPE,
        "repository": repository,
        "files": records,
        "port_configuration": port_configuration,
        "port_configuration_canonical_sha256": canonical_sha256(
            port_configuration
        ),
        "coil_input_constructed": False,
        "coil_input_included_because_referenced": True,
    }


def _build_complex(root, case):
    config = read_yaml_config(root / "examples/config.yaml")
    ivb = config["invessel_build"]
    model = ps.Stellarator(root / "examples/wout_vmec.nc")
    model.construct_invessel_build(
        ivb["toroidal_angles"],
        ivb["poloidal_angles"],
        ivb["wall_s"],
        ivb["radial_build"],
        split_chamber=ivb["split_chamber"],
        plasma_mat_tag=ivb["plasma_mat_tag"],
        sol_mat_tag=ivb["sol_mat_tag"],
        num_ribs=case.num_ribs,
        num_rib_pts=case.num_rib_pts,
        use_pydagmc=True,
        ports=[actual_port()],
    )
    if case.port:
        return model.invessel_build.native_port_complex
    return build_native_port_surface_complex(
        model.invessel_build,
        stitch_port=False,
        include_graveyard=True,
    )


def run_single_case(root, output_dir, case_name, repository_sha):
    case = case_by_name(case_name)
    output_dir = fresh_directory(output_dir)
    started = time.monotonic()
    complex_ = _build_complex(root, case)
    report = inspect_edge_facet_intersections(complex_)
    public_report = serializable_intersection_report(report)
    report_path = None
    vtk_path = None
    if report["intersection_count"]:
        report_path = write_json(
            output_dir / "plc_intersection_report.json", public_report
        )
        vtk_path = write_minimal_vtk(
            output_dir / "minimal_implicated_facets.vtk", report
        )

    surface_counts = Counter(surface.kind for surface in complex_.surfaces)
    pre_gmsh_result = {
        "schema_version": "1.0",
        "scope": SCOPE,
        "case": case.to_dict(),
        "repository_sha": repository_sha,
        "sampling_basis": (
            "shared port-centred refined angular grid; uncut radial surfaces"
            if not case.port
            else "shared port-centred refined angular grid; port patch stitched"
        ),
        "port_geometry_present": case.port,
        "magnets_constructed": False,
        "full_assembly_constructed": False,
        "declared_tolerances": {
            "units": "cm",
            "aperture_chord": complex_.aperture_chord_tolerance,
            "vertex_merge": complex_.vertex_merge_tolerance,
            "intersection": INTERSECTION_TOLERANCE_CM,
            "gmsh_mesh_size_min": MIN_MESH_SIZE_CM,
            "gmsh_mesh_size_max": MAX_MESH_SIZE_CM,
        },
        "entity_counts": report["entity_counts"],
        "surface_kind_counts": dict(sorted(surface_counts.items())),
        "topology_summary": complex_.topology_summary(),
        "intersection_count": report["intersection_count"],
        "intersection_report": (
            file_record(report_path) if report_path else None
        ),
        "minimal_vtk": file_record(vtk_path) if vtk_path else None,
        "downstream_operation": "Gmsh discrete PLC tetrahedralization",
    }
    pre_gmsh_path = write_json(
        output_dir / "pre_gmsh_result.json", pre_gmsh_result
    )

    downstream_error = None
    mesh_validation = None
    try:
        mesh = complex_.tetrahedralize(
            MIN_MESH_SIZE_CM,
            MAX_MESH_SIZE_CM,
            terminal_output=True,
            algorithm_3d=MESH_ALGORITHM_3D,
        )
        mesh_validation = mesh.validate().to_dict()
    except Exception as error:
        downstream_error = exception_record(error)

    classification = classify_case(
        case.port, report["intersection_count"], downstream_error
    )
    artifacts = [
        path
        for path in (report_path, vtk_path, pre_gmsh_path)
        if path is not None
    ]
    result = {
        **pre_gmsh_result,
        "downstream_error": downstream_error,
        "gmsh_log": [],
        "mesh_validation": mesh_validation,
        "classification": classification,
        "passed": classification == "PASS_NO_REPRODUCIBLE_PLC_FAILURE",
        "elapsed_seconds": time.monotonic() - started,
        "artifacts": [file_record(path) for path in artifacts],
    }
    write_json(output_dir / "case_result.json", result)
    print(
        json.dumps(
            {
                "case": case.name,
                "classification": classification,
                "intersections": report["intersection_count"],
                "downstream_error": (
                    downstream_error["message"] if downstream_error else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


def _external_path(external_root, relative):
    if external_root is None:
        return None
    return str(PureWindowsPath(external_root).joinpath(*relative.parts))


def _artifact_records(directory, external_root=None):
    records = []
    for path in sorted(Path(directory).rglob("*")):
        if (
            not path.is_file()
            or path.name == "geometry_diagnostic_manifest.json"
        ):
            continue
        relative = path.relative_to(directory)
        records.append(
            file_record(
                path,
                logical_name=relative.as_posix(),
                external_path=_external_path(external_root, relative),
            )
        )
    return records


def run_matrix(args):
    root = Path(__file__).resolve().parents[1]
    expected_branch = "ports/actual-plc-diagnostic-20260823"
    try:
        branch = _git(root, "branch", "--show-current")
        head = _git(root, "rev-parse", "HEAD")
        status = _git(
            root, "status", "--porcelain=v1", "--untracked-files=all"
        )
        remotes = _git(root, "remote", "-v").splitlines()
        identity_verification = "container_git"
    except RuntimeError:
        if not (
            args.repository_clean_verified
            and args.repository_branch
            and args.repository_remote
        ):
            raise
        branch = args.repository_branch
        head = args.repository_sha
        status = ""
        remotes = args.repository_remote
        identity_verification = "explicit_host_preflight"
    if branch != expected_branch or head != args.repository_sha or status:
        raise RuntimeError(
            "Repository identity gate failed: "
            f"branch={branch!r}, head={head!r}, dirty={bool(status)}"
        )
    output_dir = fresh_directory(args.output_dir)
    repository = {
        "root": str(root),
        "host_root": args.repository_host_root,
        "branch": branch,
        "commit": head,
        "remotes": remotes,
        "clean_at_start": True,
        "identity_verification": identity_verification,
    }
    inputs = input_manifest(root, repository)
    environment = environment_manifest(
        args.image_reference, args.image_id, args.image_digest
    )
    input_path = write_json(output_dir / "input_manifest.json", inputs)
    environment_path = write_json(
        output_dir / "environment_manifest.json", environment
    )
    input_record = file_record(input_path)
    environment_record = file_record(environment_path)

    case_results = []
    receipts = []
    for case in CASE_MATRIX:
        case_dir = output_dir / case.name
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-case",
            case.name,
            "--output-dir",
            str(case_dir),
            "--repository-sha",
            head,
        ]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        payload_path = case_dir / "case_result.json"
        pre_gmsh_path = case_dir / "pre_gmsh_result.json"
        payload = None
        if payload_path.exists():
            payload = json.loads(payload_path.read_text())
        elif pre_gmsh_path.exists():
            payload = json.loads(pre_gmsh_path.read_text())
            failure = process_failure_record(
                completed.returncode, completed.stderr
            )
            classification = classify_case(
                case.port, payload["intersection_count"], failure
            )
            payload.update(
                {
                    "downstream_error": failure,
                    "gmsh_log": [],
                    "mesh_validation": None,
                    "classification": classification,
                    "passed": False,
                    "elapsed_seconds": None,
                    "process_return_code": completed.returncode,
                    "artifacts": [
                        file_record(path)
                        for path in sorted(case_dir.iterdir())
                        if path.is_file() and path != payload_path
                    ],
                }
            )
            write_json(payload_path, payload)
        if payload is not None:
            case_results.append(payload)
        receipt = {
            "schema_version": "1.0",
            "scope": SCOPE,
            "case": case.to_dict(),
            "exact_command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "repository": repository,
            "input_manifest": input_record,
            "environment_manifest": environment_record,
            "input_hashes": inputs["files"],
            "port_configuration_canonical_sha256": inputs[
                "port_configuration_canonical_sha256"
            ],
            "environment": environment,
            "exception": (
                payload.get("downstream_error")
                if payload
                else {
                    "type": "DiagnosticProcessFailure",
                    "message": "Case process did not produce case_result.json",
                    "traceback": completed.stderr,
                }
            ),
            "generated_entity_counts": (
                payload.get("entity_counts") if payload else None
            ),
            "classification": (
                payload.get("classification")
                if payload
                else "BLOCKED_ENVIRONMENT_OR_INPUT_IDENTITY"
            ),
            "outputs": (
                _artifact_records(
                    case_dir,
                    (
                        str(PureWindowsPath(args.external_root) / case.name)
                        if args.external_root
                        else None
                    ),
                )
                if case_dir.exists()
                else []
            ),
        }
        receipt_path = write_json(case_dir / "case_receipt.json", receipt)
        receipts.append(file_record(receipt_path))
        if payload is None and case.name == "A0":
            break

    primary = terminal_classification(case_results)
    if len(case_results) != len(CASE_MATRIX):
        primary = "BLOCKED_ENVIRONMENT_OR_INPUT_IDENTITY"
    matrix = [
        {
            "case": item["case"]["name"],
            "port": item["case"]["port"],
            "resolution": item["case"]["resolution"],
            "intersection_count": item["intersection_count"],
            "downstream_exception_type": (
                item["downstream_error"]["type"]
                if item["downstream_error"]
                else None
            ),
            "downstream_message": (
                item["downstream_error"]["message"]
                if item["downstream_error"]
                else None
            ),
            "classification": item["classification"],
        }
        for item in case_results
    ]
    summary = {
        "schema_version": "1.0",
        "scope": SCOPE,
        "repository": repository,
        "case_matrix": matrix,
        "terminal_classification": primary,
        "qualified_transport_geometry": False,
        "next_stage_authorized": False,
    }
    summary_path = write_json(
        output_dir / "plc_diagnostic_summary.json", summary
    )
    intersection_cases = [
        item for item in case_results if item["intersection_count"]
    ]
    if intersection_cases:
        aggregate = {
            "schema_version": "1.0",
            "scope": SCOPE,
            "terminal_classification": primary,
            "cases": [
                {
                    "case": item["case"],
                    "report": item["intersection_report"],
                    "minimal_vtk": item["minimal_vtk"],
                    "intersection_count": item["intersection_count"],
                }
                for item in intersection_cases
            ],
        }
        write_json(output_dir / "plc_intersection_report.json", aggregate)
    manifest = {
        "schema_version": "1.0",
        "scope": SCOPE,
        "repository": repository,
        "external_root": args.external_root,
        "terminal_classification": primary,
        "artifacts": _artifact_records(output_dir, args.external_root),
        "summary": file_record(summary_path),
        "receipt_count": len(receipts),
    }
    manifest_path = write_json(
        output_dir / "geometry_diagnostic_manifest.json", manifest
    )
    print(
        json.dumps(
            {
                "terminal_classification": primary,
                "manifest": file_record(manifest_path),
                "case_matrix": matrix,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if primary == "PASS_NO_REPRODUCIBLE_PLC_FAILURE" else 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument(
        "--single-case", choices=[case.name for case in CASE_MATRIX]
    )
    parser.add_argument("--external-root")
    parser.add_argument("--repository-host-root")
    parser.add_argument("--repository-branch")
    parser.add_argument("--repository-remote", action="append")
    parser.add_argument("--repository-clean-verified", action="store_true")
    parser.add_argument("--image-reference")
    parser.add_argument("--image-id")
    parser.add_argument("--image-digest")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.single_case:
        return run_single_case(
            root, args.output_dir, args.single_case, args.repository_sha
        )
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
