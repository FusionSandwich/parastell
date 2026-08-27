#!/usr/bin/env python3
"""Create and audit Prompt-01-v3 WISTELL-D geometry artifacts.

All output commands are create-only.  The script uses the already-local
CadQuery/ParaStell runtime and never installs a dependency or discovers a
fallback geometry.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Any, Mapping

import cadquery as cq
import h5py
import numpy as np
from scipy.io import netcdf_file
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parastell.geometry_provider import (
    EXPECTED_LAYER_ORDER,
    WISTELL_D_ACCEPTANCE_SCHEMA,
    WISTELL_D_GEOMETRY_INPUT_MODE,
    WISTELL_D_SOURCE_HASHES,
    canonical_patch_instances,
    complete_pairwise_audit,
    derive_wistell_d_transforms,
    require_complete_pairwise_acceptance,
    sha256_file,
)


LINEAGE_COMMIT = "398032b8c0b4e7c0459c602f2af1e73b3fca0b9a"
LANE_BASE = "cff61d06cf1d6129337841bdb80061f6a8f2b998"
N_TOROIDAL = 16
N_POLOIDAL = 31
N_CONTROL_POINTS = 452
N_UNIQUE_POLOIDAL = N_POLOIDAL - 1
N_SYMMETRIC_PROFILE = math.ceil(N_POLOIDAL / 2)
SOURCE_CAD_GRID = (61, 121)
OVERLAP_TOLERANCE_CM3 = 1.0e-5

RADIAL_NAMES = (
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "vacuum_gap",
    "magnet_envelope",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"create-only output already exists: {path}")
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_paths(root: Path) -> dict[str, Path]:
    return {name: root / name for name in WISTELL_D_SOURCE_HASHES}


def verify_source_set(root: Path) -> dict[str, Any]:
    rows = {}
    for name, expected in WISTELL_D_SOURCE_HASHES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"authoritative source is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"authoritative source hash mismatch: {path}")
        rows[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return rows


def array_to_symmetric_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (N_CONTROL_POINTS,):
        raise ValueError(f"expected 452 canonical values, got {values.shape}")
    matrix = np.zeros((N_TOROIDAL, N_POLOIDAL), dtype=float)
    for canonical_id in range(N_CONTROL_POINTS):
        point_id = canonical_id
        if point_id >= N_SYMMETRIC_PROFILE:
            point_id += N_UNIQUE_POLOIDAL - N_SYMMETRIC_PROFILE
        toroidal_id = point_id // N_UNIQUE_POLOIDAL
        poloidal_id = point_id % N_UNIQUE_POLOIDAL
        matrix[toroidal_id, poloidal_id] = values[canonical_id]
    matrix[0, N_SYMMETRIC_PROFILE:] = np.flip(
        matrix[0, : N_POLOIDAL // 2]
    )
    matrix[-1, N_SYMMETRIC_PROFILE:] = np.flip(
        matrix[-1, : N_POLOIDAL // 2]
    )
    matrix[:, -1] = matrix[:, 0]
    return matrix


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "finite": bool(np.isfinite(values).all()),
    }


def make_radial_build(input_root: Path) -> tuple[OrderedDict, OrderedDict, dict]:
    blanket_boundary = np.load(input_root / "blanket_boundary.npy")
    magnet_boundary = np.load(input_root / "magnet_boundary.npy")
    nwl = np.load(input_root / "nwl.npy")
    nwl_normalized = (nwl - nwl.min()) / (nwl.max() - nwl.min())
    breeder_requested = 25.0 + 30.0 * nwl_normalized
    hts = np.full(N_CONTROL_POINTS, 20.0)
    breeder_limit = blanket_boundary - 4.0 - 5.0 - 10.0 - hts - 5.0
    breeder = np.minimum(breeder_requested, breeder_limit)
    lts = blanket_boundary - 4.0 - breeder - 5.0 - hts - 10.0
    gap = magnet_boundary - blanket_boundary
    arrays = OrderedDict(
        (
            ("first_wall", np.full(N_CONTROL_POINTS, 4.0)),
            ("breeder", breeder),
            ("back_wall", np.full(N_CONTROL_POINTS, 5.0)),
            ("high_temperature_shield", hts),
            ("vacuum_vessel", np.full(N_CONTROL_POINTS, 10.0)),
            ("low_temperature_shield", lts),
            ("vacuum_gap", gap),
            ("magnet_envelope", np.full(N_CONTROL_POINTS, 30.0)),
        )
    )
    radial_build = OrderedDict(
        (
            name,
            {
                "thickness_matrix": array_to_symmetric_matrix(values),
                "mat_tag": "Vacuum" if name == "vacuum_gap" else name,
            },
        )
        for name, values in arrays.items()
    )
    cumulative = np.cumsum(np.vstack(tuple(arrays.values())), axis=0)
    diagnostics = {
        "canonical_control_count": N_CONTROL_POINTS,
        "all_thicknesses_finite": bool(
            all(np.isfinite(values).all() for values in arrays.values())
        ),
        "all_thicknesses_strictly_positive": bool(
            all(np.all(values > 0.0) for values in arrays.values())
        ),
        "strict_radial_order_at_all_canonical_controls": bool(
            np.all(np.diff(cumulative, axis=0) > 0.0)
        ),
        "thickness_cm": {name: summarize(values) for name, values in arrays.items()},
        "nwl_mw_per_m2": summarize(nwl),
        "nwl_normalized": summarize(nwl_normalized),
        "blanket_boundary_cm": summarize(blanket_boundary),
        "magnet_boundary_cm": summarize(magnet_boundary),
        "breeder_clipped_control_count": int(
            np.count_nonzero(breeder < breeder_requested)
        ),
        "outer_blanket_residual_cm": summarize(
            blanket_boundary
            - sum(arrays[name] for name in RADIAL_NAMES[:6])
        ),
        "magnet_boundary_residual_cm": summarize(
            magnet_boundary
            - sum(arrays[name] for name in RADIAL_NAMES[:7])
        ),
    }
    return radial_build, arrays, diagnostics


class _NetcdfVariableAdapter:
    def __init__(self, variable):
        self.variable = variable

    def __getitem__(self, key):
        return self.variable[key]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.variable.data, dtype=dtype)


class _NetcdfDatasetAdapter:
    def __init__(self, filename):
        self.dataset = netcdf_file(filename, mode="r", mmap=False)
        self.variables = {
            name: _NetcdfVariableAdapter(variable)
            for name, variable in self.dataset.variables.items()
        }

    def close(self):
        self.dataset.close()


def load_geometry_only_parastell():
    """Load local ParaStell while stubbing only unused optional exporters."""
    for name in (
        "cad_to_dagmc",
        "gmsh",
        "pydagmc",
        "pymoab.core",
        "pymoab.types",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    pymoab = sys.modules.setdefault("pymoab", types.ModuleType("pymoab"))
    pymoab.core = sys.modules["pymoab.core"]
    pymoab.types = sys.modules["pymoab.types"]
    netcdf4 = sys.modules.setdefault("netCDF4", types.ModuleType("netCDF4"))
    netcdf4.Dataset = _NetcdfDatasetAdapter
    import parastell
    import parastell.parastell as ps

    return parastell, ps


def shape_metrics(shape: Any) -> dict[str, Any]:
    center = shape.Center()
    box = shape.BoundingBox()
    return {
        "valid_brep": bool(shape.isValid()),
        "solid_count": len(shape.Solids()),
        "volume_cm3": float(shape.Volume()),
        "surface_area_cm2": float(shape.Area()),
        "centroid_cm": [float(center.x), float(center.y), float(center.z)],
        "bounding_box_cm": {
            "min": [float(box.xmin), float(box.ymin), float(box.zmin)],
            "max": [float(box.xmax), float(box.ymax), float(box.zmax)],
        },
    }


def intersection_volume(left: Any, right: Any) -> float:
    common = left.intersect(right)
    if common is None or common.isNull():
        return 0.0
    return float(common.Volume())


def load_step_components(root: Path, names: tuple[str, ...]) -> OrderedDict:
    return OrderedDict(
        (name, cq.importers.importStep(str(root / f"{name}.step")).val())
        for name in names
    )


def artifact_rows(root: Path, files: Mapping[str, Path]) -> dict[str, Any]:
    rows = {}
    for role, path in files.items():
        path = path.resolve()
        rows[role] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return rows


def audit_existing_31x61(args: argparse.Namespace) -> None:
    source_root = Path(args.step_root).resolve()
    output = Path(args.output).resolve()
    names = (
        "chamber",
        "first_wall",
        "breeder",
        "back_wall",
        "hts",
        "vacuum_vessel",
        "lts",
        "gap",
        "magnets",
    )
    started = time.time()
    components = load_step_components(source_root, names)
    metrics = {name: shape_metrics(shape) for name, shape in components.items()}
    pairwise = complete_pairwise_audit(
        components, intersection_volume, tolerance_cm3=args.tolerance_cm3
    )
    nonzero = [row for row in pairwise["pairs"] if row["overlap"]]
    payload = {
        "schema": "wistell_d.source_cad_31x61_rejection/v1.0.0",
        "generated_utc": utc_now(),
        "classification": "REJECTED_LINEAGE_EVIDENCE_ONLY",
        "source_root": str(source_root),
        "cad_grid": [31, 61],
        "component_metrics": metrics,
        "complete_pairwise_audit": pairwise,
        "rejection": {
            "required_pair_count": 36,
            "all_nine_solids_individually_valid": all(
                row["valid_brep"] and row["solid_count"] == 1
                for row in metrics.values()
            ),
            "nonzero_pairs": nonzero,
            "adjacent_only_logic_would_accept": all(
                not row["overlap"]
                for row in pairwise["pairs"]
                if row["adjacent_in_radial_stack"]
            ),
        },
        "input_steps": artifact_rows(
            source_root,
            {name: source_root / f"{name}.step" for name in names},
        ),
        "elapsed_seconds": time.time() - started,
    }
    if not nonzero:
        raise RuntimeError("31x61 rejection was not reproduced")
    write_json_create_only(output, payload)
    print(json.dumps({"output": str(output), "nonzero_pairs": nonzero}, indent=2))


def expanded_surface_diagnostics(stellarator: Any) -> dict[str, Any]:
    surfaces = OrderedDict(
        (name, surface.get_loci())
        for name, surface in stellarator.invessel_build.Surfaces.items()
    )
    names = tuple(surfaces)
    clearance_rows = []
    control_mask = np.zeros(SOURCE_CAD_GRID, dtype=bool)
    control_mask[::4, ::4] = True
    for inner_name, outer_name in zip(names[:-1], names[1:]):
        distances = np.linalg.norm(
            surfaces[outer_name] - surfaces[inner_name], axis=2
        )
        clearance_rows.append(
            {
                "inner": inner_name,
                "outer": outer_name,
                "all_sample_min_cm": float(distances.min()),
                "off_control_sample_min_cm": float(distances[~control_mask].min()),
                "off_control_sample_count": int(np.count_nonzero(~control_mask)),
            }
        )

    transforms = derive_wistell_d_transforms(nfp=4, lasym=True)
    start_axis = np.diag([1.0, -1.0, -1.0])
    end_axis = transforms["half_period_mate"].linear
    seam_rows = []
    for name, loci in surfaces.items():
        start = loci[0]
        end = loci[-1]
        start_residual = np.max(
            np.linalg.norm(start @ start_axis.T - start[::-1], axis=1)
        )
        end_residual = np.max(
            np.linalg.norm(end @ end_axis.T - end[::-1], axis=1)
        )
        seam_rows.append(
            {
                "surface": name,
                "start_profile_stellarator_symmetry_max_cm": float(start_residual),
                "end_profile_stellarator_symmetry_max_cm": float(end_residual),
                "start_toroidal_plane_max_abs_y_cm": float(np.abs(start[:, 1]).max()),
                "end_toroidal_plane_max_abs_x_minus_y_cm": float(
                    np.abs(end[:, 0] - end[:, 1]).max()
                ),
            }
        )

    samples = []
    for surface_name in (names[0], names[1], names[-2], names[-1]):
        loci = surfaces[surface_name]
        for toroidal_index, poloidal_index in ((0, 0), (30, 30), (60, 60), (60, 120)):
            samples.append(
                {
                    "surface": surface_name,
                    "toroidal_index": toroidal_index,
                    "poloidal_index": poloidal_index,
                    "coordinate_cm": loci[toroidal_index, poloidal_index].tolist(),
                }
            )
    return {
        "clearance_samples": clearance_rows,
        "sector_end_cap_checks": seam_rows,
        "sampled_interface_coordinates": samples,
    }


def build_45(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output root exists: {output_root}")
    input_before = verify_source_set(input_root)
    output_root.mkdir(parents=True)
    started = time.time()
    parastell, ps = load_geometry_only_parastell()
    radial_build, arrays, thickness = make_radial_build(input_root)
    np.savez(output_root / "canonical_thickness_arrays_cm.npz", **arrays)

    stellarator = ps.Stellarator(str(input_root / "wout_wistell-d.nc"))
    stellarator.construct_invessel_build(
        np.linspace(0.0, 45.0, N_TOROIDAL),
        np.linspace(0.0, 360.0, N_POLOIDAL),
        1.0,
        radial_build,
        split_chamber=False,
        num_ribs=SOURCE_CAD_GRID[0],
        num_rib_pts=SOURCE_CAD_GRID[1],
    )
    stellarator.export_invessel_build_step(export_dir=str(output_root))
    components = OrderedDict(stellarator.invessel_build.Components)
    combined = cq.Compound.makeCompound(list(components.values()))
    combined_path = output_root / "wistell_d_45d_61x121_source_cad.step"
    cq.exporters.export(combined, str(combined_path))

    component_metrics = {
        name: shape_metrics(shape) for name, shape in components.items()
    }
    surface_diagnostics = expanded_surface_diagnostics(stellarator)
    pairwise = complete_pairwise_audit(
        components, intersection_volume, tolerance_cm3=args.tolerance_cm3
    )
    require_complete_pairwise_acceptance(pairwise)
    input_after = verify_source_set(input_root)
    if input_before != input_after:
        raise RuntimeError("authoritative source set changed during CAD build")

    source_files = source_paths(input_root)
    artifacts = artifact_rows(
        output_root,
        {
            "source_step": combined_path,
            "canonical_thickness_arrays": output_root
            / "canonical_thickness_arrays_cm.npz",
            **{
                f"component_step:{name}": output_root / f"{name}.step"
                for name in components
            },
        },
    )
    payload = {
        "schema": WISTELL_D_ACCEPTANCE_SCHEMA,
        "geometry_input_mode": WISTELL_D_GEOMETRY_INPUT_MODE,
        "generated_utc": utc_now(),
        "classification": "WISTELL_D_45D_61X121_SOURCE_CAD_PASS",
        "lane": {
            "base_sha": LANE_BASE,
            "head_sha_at_build": git_head(Path.cwd()),
        },
        "source": {
            "lineage_root": str(input_root),
            "lineage_commit": LINEAGE_COMMIT,
            "hashes": dict(WISTELL_D_SOURCE_HASHES),
            "files": input_after,
            "immutability_check": "PASS",
            "parastell_module": str(Path(parastell.__file__).resolve()),
        },
        "model": {
            "device": "WISTELL-D",
            "role": "canonical_45_degree_half_field_period_source_geometry",
            "toroidal_extent_degrees": 45.0,
            "wall_s": 1.0,
            "canonical_control_count": N_CONTROL_POINTS,
            "control_grid": [16, 31],
            "source_cad_grid": list(SOURCE_CAD_GRID),
            "magnet_representation": "continuous_30_cm_magnet_envelope",
            "global_explicit_coils": False,
            "component_order": list(EXPECTED_LAYER_ORDER),
        },
        "thickness_validation": thickness,
        "expanded_surface_validation": surface_diagnostics,
        "components": component_metrics,
        "materials": {
            name: ("Vacuum" if name in ("chamber", "vacuum_gap") else name)
            for name in components
        },
        "complete_pairwise_audit": pairwise,
        "source_domain": {
            "mesh_path": str(source_files["source_mesh.h5m"].resolve()),
            "mesh_sha256": WISTELL_D_SOURCE_HASHES["source_mesh.h5m"],
            "toroidal_extent_degrees": [0.0, 45.0],
            "rate_scope": "UNRESOLVED_REQUIRES_LANE_01B_NORMALIZATION",
        },
        "physical_boundaries": {
            "toroidal_start": "stellarator_transform_boundary_at_0_degrees",
            "toroidal_end": "stellarator_transform_boundary_at_45_degrees",
            "outer_magnet_envelope": "transport_outer_boundary_role_unassigned",
        },
        "transforms": {
            name: transform.receipt()
            for name, transform in derive_wistell_d_transforms(
                nfp=4, lasym=True
            ).items()
        },
        "canonical_patch_map": canonical_patch_instances(),
        "local_frames": {},
        "artifacts": artifacts,
        "elapsed_seconds": time.time() - started,
    }
    acceptance_path = output_root / "SOURCE_CAD_61X121_ACCEPTANCE.json"
    write_json_create_only(acceptance_path, payload)
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "output": str(acceptance_path),
                "source_step_sha256": artifacts["source_step"]["sha256"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )


def vmec_data(path: Path) -> dict[str, Any]:
    with netcdf_file(path, mode="r", mmap=False) as data:
        variables = data.variables
        return {
            "nfp": int(np.asarray(variables["nfp"].data)),
            "lasym": bool(np.asarray(variables["lasym__logical__"].data)),
            "rmnc": np.asarray(variables["rmnc"].data).copy(),
            "zmns": np.asarray(variables["zmns"].data).copy(),
            "xm": np.asarray(variables["xm"].data).copy(),
            "xn": np.asarray(variables["xn"].data).copy(),
        }


def evaluate_vmec_surface(
    vmec: Mapping[str, Any], theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    theta, phi = np.broadcast_arrays(theta, phi)
    angle = (
        vmec["xm"][:, None] * theta.ravel()[None, :]
        - vmec["xn"][:, None] * phi.ravel()[None, :]
    )
    radius = np.sum(vmec["rmnc"][-1, :, None] * np.cos(angle), axis=0)
    z_coord = np.sum(vmec["zmns"][-1, :, None] * np.sin(angle), axis=0)
    points = np.column_stack(
        (radius * np.cos(phi.ravel()), radius * np.sin(phi.ravel()), z_coord)
    )
    return points.reshape(theta.shape + (3,))


def parse_coils(path: Path) -> list[np.ndarray]:
    curves = []
    current = []
    for line in path.read_text(encoding="utf-8").splitlines():
        columns = line.split()
        if len(columns) < 4:
            continue
        try:
            row = [float(value) for value in columns[:4]]
        except ValueError:
            continue
        current.append(row[:3])
        if row[3] == 0.0:
            curves.append(np.asarray(current, dtype=float))
            current = []
    if current or not curves:
        raise ValueError("coil file contains an incomplete or empty filament set")
    return curves


def audit_symmetry(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    verify_source_set(input_root)
    vmec = vmec_data(input_root / "wout_wistell-d.nc")
    transforms = derive_wistell_d_transforms(
        nfp=vmec["nfp"], lasym=not vmec["lasym"]
    )
    mate = transforms["half_period_mate"]
    period = transforms["field_period"]
    theta = np.linspace(0.0, 2.0 * np.pi, 49)[:, None]
    phi = np.linspace(0.0, np.pi / vmec["nfp"], 25)[None, :]
    canonical = evaluate_vmec_surface(vmec, theta, phi)
    expected_mate = evaluate_vmec_surface(
        vmec, -theta, 2.0 * np.pi / vmec["nfp"] - phi
    )
    vmec_residual = np.linalg.norm(
        mate.transform_points(canonical) - expected_mate, axis=2
    )

    curves = parse_coils(input_root / "coils.wistell-d")
    coil_points = np.vstack(curves)
    coil_tree = cKDTree(coil_points)
    coil_distance = coil_tree.query(mate.transform_points(coil_points), k=1)[0]
    centroids = np.vstack([curve[:-1].mean(axis=0) for curve in curves])
    centroid_tree = cKDTree(centroids)
    centroid_distance = centroid_tree.query(
        mate.transform_points(centroids), k=1
    )[0]

    with h5py.File(input_root / "source_mesh.h5m", "r") as mesh:
        source_nodes = mesh["tstt/nodes/coordinates"][:]
        tet_shape = list(mesh["tstt/elements/Tet4/connectivity"].shape)
    source_phi = np.degrees(np.arctan2(source_nodes[:, 1], source_nodes[:, 0]))
    seam_nodes = source_nodes[np.isclose(source_phi, 45.0, atol=1e-10)]
    seam_tree = cKDTree(seam_nodes)
    seam_distance = seam_tree.query(mate.transform_points(seam_nodes), k=1)[0]
    mate_source_phi = np.degrees(
        np.arctan2(
            mate.transform_points(source_nodes)[:, 1],
            mate.transform_points(source_nodes)[:, 0],
        )
    )

    start_axis = np.diag([1.0, -1.0, -1.0])
    start = np.eye(4)
    start[:3, :3] = start_axis
    closure = {
        "half_period_involution_inf_norm": float(
            np.linalg.norm(mate.array @ mate.array - np.eye(4), ord=np.inf)
        ),
        "period_power_nfp_inf_norm": float(
            np.linalg.norm(
                np.linalg.matrix_power(period.array, vmec["nfp"]) - np.eye(4),
                ord=np.inf,
            )
        ),
        "generator_product_vs_period_inf_norm": float(
            np.linalg.norm(mate.array @ start - period.array, ord=np.inf)
        ),
    }
    payload = {
        "schema": "wistell_d.symmetry_transforms/v1.0.0",
        "generated_utc": utc_now(),
        "classification": "WISTELL_D_EXACT_HALF_PERIOD_TRANSFORM_PASS",
        "evidence": {
            "source_hashes": dict(WISTELL_D_SOURCE_HASHES),
            "nfp": vmec["nfp"],
            "lasym_vmec_logical": vmec["lasym"],
            "stellarator_symmetric": not vmec["lasym"],
            "phase_origin_degrees": 0.0,
            "canonical_domain_degrees": [0.0, 45.0],
            "derived_domain_degrees": [45.0, 90.0],
            "frame": "right_handed_cartesian_toroidal_axis_z_origin_0_0_0",
            "cad_units": "cm",
            "coil_units": "m",
            "source_scope": "45_degree_half_field_period",
            "rate_scope": "UNRESOLVED_REQUIRES_LANE_01B_NORMALIZATION",
        },
        "transforms": {
            name: transform.receipt() for name, transform in transforms.items()
        },
        "group_closure": closure,
        "verification": {
            "vmec_sample_count": int(vmec_residual.size),
            "vmec_point_max_residual_m": float(vmec_residual.max()),
            "vmec_point_rms_residual_m": float(
                np.sqrt(np.mean(vmec_residual**2))
            ),
            "coil_curve_count": len(curves),
            "coil_point_count": int(len(coil_points)),
            "coil_set_max_nearest_residual_m": float(coil_distance.max()),
            "coil_centroid_max_nearest_residual_m": float(
                centroid_distance.max()
            ),
            "source_node_count": int(len(source_nodes)),
            "source_tet_connectivity_shape": tet_shape,
            "source_canonical_phi_range_degrees": [
                float(source_phi.min()),
                float(source_phi.max()),
            ],
            "source_mate_phi_range_degrees": [
                float(mate_source_phi.min()),
                float(mate_source_phi.max()),
            ],
            "source_seam_node_count": int(len(seam_nodes)),
            "source_seam_set_max_residual_cm": float(seam_distance.max()),
            "boundary_scalar_fields": {
                name: {
                    "canonical_count": int(len(np.load(input_root / name))),
                    "start_profile_symmetry_max": float(
                        np.abs(
                            array_to_symmetric_matrix(np.load(input_root / name))[0]
                            - array_to_symmetric_matrix(np.load(input_root / name))[0][::-1]
                        ).max()
                    ),
                    "end_profile_symmetry_max": float(
                        np.abs(
                            array_to_symmetric_matrix(np.load(input_root / name))[-1]
                            - array_to_symmetric_matrix(np.load(input_root / name))[-1][::-1]
                        ).max()
                    ),
                }
                for name in (
                    "nwl.npy",
                    "blanket_boundary.npy",
                    "magnet_boundary.npy",
                )
            },
        },
        "transform_kind": (
            "proper_180_degree_rotation_about_shared_seam_radial_axis; "
            "equivalent_to_combined_toroidal_reflection_and_z_reversal"
        ),
        "not_a_simple_45_degree_rotation": True,
    }
    thresholds = (
        payload["verification"]["vmec_point_max_residual_m"] < 1.0e-10
        and payload["verification"]["coil_set_max_nearest_residual_m"] < 1.0e-5
        and payload["verification"]["source_seam_set_max_residual_cm"] < 1.0e-8
        and max(closure.values()) < 1.0e-12
    )
    if not thresholds:
        payload["classification"] = "WISTELL_D_EXACT_HALF_PERIOD_TRANSFORM_BLOCKED"
        raise RuntimeError(json.dumps(payload, indent=2))
    write_json_create_only(output, payload)
    print(json.dumps({"output": str(output), "verification": payload["verification"]}, indent=2))


def derive_90(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    transform_path = Path(args.transform_receipt).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output root exists: {output_root}")
    source_manifest = json.loads(
        (source_root / "SOURCE_CAD_61X121_ACCEPTANCE.json").read_text(
            encoding="utf-8"
        )
    )
    if source_manifest.get("classification") != (
        "WISTELL_D_45D_61X121_SOURCE_CAD_PASS"
    ):
        raise ValueError("45-degree source CAD is not accepted")
    transform_receipt = json.loads(transform_path.read_text(encoding="utf-8"))
    if transform_receipt.get("classification") != (
        "WISTELL_D_EXACT_HALF_PERIOD_TRANSFORM_PASS"
    ):
        raise ValueError("symmetry transform receipt is not accepted")
    output_root.mkdir(parents=True)
    started = time.time()
    names = tuple(source_manifest["model"]["component_order"])
    source_components = load_step_components(source_root, names)
    expanded = OrderedDict()
    expansion_rows = []
    for name, source in source_components.items():
        mate = source.rotate((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), 180.0)
        common_volume = intersection_volume(source, mate)
        joined = source.fuse(mate).clean()
        source_metrics = shape_metrics(source)
        joined_metrics = shape_metrics(joined)
        volume_scale = joined_metrics["volume_cm3"] / source_metrics["volume_cm3"]
        row = {
            "component": name,
            "source_mate_intersection_volume_cm3": common_volume,
            "joined_solid_count": joined_metrics["solid_count"],
            "volume_scale_90d_over_45d": volume_scale,
            "source_surface_area_cm2": source_metrics["surface_area_cm2"],
            "joined_surface_area_cm2": joined_metrics["surface_area_cm2"],
            "joined_valid_brep": joined_metrics["valid_brep"],
        }
        if (
            common_volume > args.tolerance_cm3
            or not joined_metrics["valid_brep"]
            or joined_metrics["solid_count"] != 1
            or not math.isclose(volume_scale, 2.0, rel_tol=2.0e-8, abs_tol=1.0e-8)
        ):
            raise RuntimeError(f"90-degree seam expansion failed for {name}: {row}")
        expanded[name] = joined
        expansion_rows.append(row)
        cq.exporters.export(joined, str(output_root / f"{name}.step"))

    combined = cq.Compound.makeCompound(list(expanded.values()))
    combined_path = output_root / "wistell_d_90d_symmetry_derived.step"
    cq.exporters.export(combined, str(combined_path))
    pairwise = complete_pairwise_audit(
        expanded, intersection_volume, tolerance_cm3=args.tolerance_cm3
    )
    require_complete_pairwise_acceptance(pairwise)
    components = {name: shape_metrics(shape) for name, shape in expanded.items()}
    artifacts = artifact_rows(
        output_root,
        {
            "source_step": combined_path,
            "transform_receipt": transform_path,
            **{
                f"component_step:{name}": output_root / f"{name}.step"
                for name in expanded
            },
        },
    )
    payload = {
        "schema": WISTELL_D_ACCEPTANCE_SCHEMA,
        "geometry_input_mode": WISTELL_D_GEOMETRY_INPUT_MODE,
        "generated_utc": utc_now(),
        "classification": "WISTELL_D_90D_TRANSPORT_SOURCE_CAD_PASS",
        "source": source_manifest["source"],
        "model": {
            **source_manifest["model"],
            "role": "derived_90_degree_one_field_period_transport_geometry",
            "toroidal_extent_degrees": 90.0,
            "canonical_source_geometry": str(source_root),
            "derived_not_independently_reconstructed": True,
        },
        "components": components,
        "materials": source_manifest["materials"],
        "complete_pairwise_audit": pairwise,
        "seam_validation": {
            "shared_seam_degrees": 45.0,
            "operation": "rigid_transform_then_same_material_BRep_fuse",
            "component_results": expansion_rows,
            "positive_volume_overlap_count": sum(
                row["source_mate_intersection_volume_cm3"] > args.tolerance_cm3
                for row in expansion_rows
            ),
            "gap_or_disconnected_component_count": sum(
                row["joined_solid_count"] != 1 for row in expansion_rows
            ),
        },
        "source_domain": {
            **source_manifest["source_domain"],
            "derived_support": "canonical_plus_half_period_mate",
            "derived_toroidal_extent_degrees": [0.0, 90.0],
            "rate_scope": "UNRESOLVED_REQUIRES_LANE_01B_NORMALIZATION",
        },
        "physical_boundaries": {
            "toroidal_start": "field_period_boundary_at_0_degrees",
            "internal_45_degree_seam": "removed_by_same_material_BRep_fuse",
            "toroidal_end": "field_period_boundary_at_90_degrees",
            "outer_magnet_envelope": "transport_outer_boundary_role_unassigned",
        },
        "transforms": transform_receipt["transforms"],
        "canonical_patch_map": canonical_patch_instances(
            instance_ids=("canonical", "mate")
        ),
        "local_frames": {},
        "artifacts": artifacts,
        "elapsed_seconds": time.time() - started,
    }
    acceptance_path = output_root / "SOURCE_CAD_90D_ACCEPTANCE.json"
    write_json_create_only(acceptance_path, payload)
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "output": str(acceptance_path),
                "step_sha256": artifacts["source_step"]["sha256"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    reject = subparsers.add_parser("audit-existing-31x61")
    reject.add_argument("--step-root", required=True)
    reject.add_argument("--output", required=True)
    reject.add_argument("--tolerance-cm3", type=float, default=OVERLAP_TOLERANCE_CM3)
    reject.set_defaults(function=audit_existing_31x61)

    build = subparsers.add_parser("build-45")
    build.add_argument("--input-root", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--tolerance-cm3", type=float, default=OVERLAP_TOLERANCE_CM3)
    build.set_defaults(function=build_45)

    symmetry = subparsers.add_parser("audit-symmetry")
    symmetry.add_argument("--input-root", required=True)
    symmetry.add_argument("--output", required=True)
    symmetry.set_defaults(function=audit_symmetry)

    expand = subparsers.add_parser("derive-90")
    expand.add_argument("--source-root", required=True)
    expand.add_argument("--transform-receipt", required=True)
    expand.add_argument("--output-root", required=True)
    expand.add_argument("--tolerance-cm3", type=float, default=OVERLAP_TOLERANCE_CM3)
    expand.set_defaults(function=derive_90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
