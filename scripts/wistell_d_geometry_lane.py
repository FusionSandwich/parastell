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
    canonical_json_sha256,
    canonical_patch_instances,
    complete_pairwise_audit,
    derive_wistell_d_transforms,
    require_complete_pairwise_acceptance,
    sha256_file,
    validate_wistell_d_manifest,
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
    matrix[0, N_SYMMETRIC_PROFILE:] = np.flip(matrix[0, : N_POLOIDAL // 2])
    matrix[-1, N_SYMMETRIC_PROFILE:] = np.flip(matrix[-1, : N_POLOIDAL // 2])
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


def make_radial_build(
    input_root: Path,
) -> tuple[OrderedDict, OrderedDict, dict]:
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
        "thickness_cm": {
            name: summarize(values) for name, values in arrays.items()
        },
        "nwl_mw_per_m2": summarize(nwl),
        "nwl_normalized": summarize(nwl_normalized),
        "blanket_boundary_cm": summarize(blanket_boundary),
        "magnet_boundary_cm": summarize(magnet_boundary),
        "breeder_clipped_control_count": int(
            np.count_nonzero(breeder < breeder_requested)
        ),
        "outer_blanket_residual_cm": summarize(
            blanket_boundary - sum(arrays[name] for name in RADIAL_NAMES[:6])
        ),
        "magnet_boundary_residual_cm": summarize(
            magnet_boundary - sum(arrays[name] for name in RADIAL_NAMES[:7])
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
    metrics = {
        name: shape_metrics(shape) for name, shape in components.items()
    }
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
    print(
        json.dumps({"output": str(output), "nonzero_pairs": nonzero}, indent=2)
    )


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
                "off_control_sample_min_cm": float(
                    distances[~control_mask].min()
                ),
                "off_control_sample_count": int(
                    np.count_nonzero(~control_mask)
                ),
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
                "start_profile_stellarator_symmetry_max_cm": float(
                    start_residual
                ),
                "end_profile_stellarator_symmetry_max_cm": float(end_residual),
                "start_toroidal_plane_max_abs_y_cm": float(
                    np.abs(start[:, 1]).max()
                ),
                "end_toroidal_plane_max_abs_x_minus_y_cm": float(
                    np.abs(end[:, 0] - end[:, 1]).max()
                ),
            }
        )

    samples = []
    for surface_name in (names[0], names[1], names[-2], names[-1]):
        loci = surfaces[surface_name]
        for toroidal_index, poloidal_index in (
            (0, 0),
            (30, 30),
            (60, 60),
            (60, 120),
        ):
            samples.append(
                {
                    "surface": surface_name,
                    "toroidal_index": toroidal_index,
                    "poloidal_index": poloidal_index,
                    "coordinate_cm": loci[
                        toroidal_index, poloidal_index
                    ].tolist(),
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
    if args.construct_only:
        input_after = verify_source_set(input_root)
        if input_before != input_after:
            raise RuntimeError(
                "authoritative source set changed during construction"
            )
        construction_payload = {
            "schema": "wistell_d.geometry_construction/v1.0.0",
            "geometry_input_mode": WISTELL_D_GEOMETRY_INPUT_MODE,
            "generated_utc": utc_now(),
            "classification": "WISTELL_D_45D_61X121_CONSTRUCTION_PASS",
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
                "toroidal_extent_degrees": 45.0,
                "wall_s": 1.0,
                "canonical_control_count": N_CONTROL_POINTS,
                "control_grid": [16, 31],
                "source_cad_grid": list(SOURCE_CAD_GRID),
                "magnet_representation": "continuous_30_cm_magnet_envelope",
                "global_explicit_coils": False,
            },
            "thickness_validation": thickness,
            "expanded_surface_validation": expanded_surface_diagnostics(
                stellarator
            ),
            "components": {
                name: shape_metrics(shape)
                for name, shape in stellarator.invessel_build.Components.items()
            },
            "elapsed_seconds": time.time() - started,
        }
        construction_path = output_root / "CONSTRUCTION_RECEIPT.json"
        write_json_create_only(construction_path, construction_payload)
        print(
            json.dumps(
                {
                    "classification": construction_payload["classification"],
                    "output": str(construction_path),
                    "elapsed_seconds": construction_payload["elapsed_seconds"],
                },
                indent=2,
            )
        )
        return
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
        "local_frames": {
            "status": "PENDING_M2_QUALIFICATION",
            "frame_count": 0,
            "frames": [],
            "accepted_for_prompt02": False,
        },
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
        raise ValueError(
            "coil file contains an incomplete or empty filament set"
        )
    return curves


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths <= 0.0) or not np.isfinite(lengths).all():
        raise ValueError("cannot normalize a zero or nonfinite vector")
    return vectors / lengths


def magnet_envelope_loci(input_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Recreate only the accepted 61x121 interface point clouds, not CAD."""
    _, ps = load_geometry_only_parastell()
    import parastell.invessel_build as ivb

    radial_build, _, _ = make_radial_build(input_root)
    stellarator = ps.Stellarator(str(input_root / "wout_wistell-d.nc"))
    radial = ivb.RadialBuild(
        np.linspace(0.0, 45.0, N_TOROIDAL),
        np.linspace(0.0, 360.0, N_POLOIDAL),
        1.0,
        radial_build,
        split_chamber=False,
        logger=stellarator._logger,
    )
    build = ivb.InVesselBuild(
        stellarator._ref_surf,
        radial,
        logger=stellarator._logger,
        num_ribs=SOURCE_CAD_GRID[0],
        num_rib_pts=SOURCE_CAD_GRID[1],
    )
    build.populate_surfaces()
    build.calculate_loci()
    inner = build.Surfaces["vacuum_gap"].get_loci()
    outer = build.Surfaces["magnet_envelope"].get_loci()
    if inner.shape != (*SOURCE_CAD_GRID, 3) or outer.shape != inner.shape:
        raise RuntimeError("unexpected magnet-envelope surface grid")
    return inner, outer


def triangulate_envelope_boundary(
    loci: np.ndarray,
    expected_normal: np.ndarray,
    *,
    boundary_role_code: int,
) -> dict[str, np.ndarray]:
    vertices = []
    centroids = []
    normals = []
    areas = []
    toroidal_cells = []
    poloidal_cells = []
    triangle_ids = []
    for toroidal_cell in range(SOURCE_CAD_GRID[0] - 1):
        for poloidal_cell in range(SOURCE_CAD_GRID[1] - 1):
            corners = (
                loci[toroidal_cell, poloidal_cell],
                loci[toroidal_cell + 1, poloidal_cell],
                loci[toroidal_cell + 1, poloidal_cell + 1],
                loci[toroidal_cell, poloidal_cell + 1],
            )
            expected = _normalize(
                np.mean(
                    (
                        expected_normal[toroidal_cell, poloidal_cell],
                        expected_normal[toroidal_cell + 1, poloidal_cell],
                        expected_normal[toroidal_cell + 1, poloidal_cell + 1],
                        expected_normal[toroidal_cell, poloidal_cell + 1],
                    ),
                    axis=0,
                )
            )
            for triangle_id, indices in enumerate(((0, 1, 2), (0, 2, 3))):
                triangle = np.asarray([corners[index] for index in indices])
                cross = np.cross(
                    triangle[1] - triangle[0], triangle[2] - triangle[0]
                )
                if float(np.dot(cross, expected)) < 0.0:
                    triangle[[1, 2]] = triangle[[2, 1]]
                    cross = -cross
                area = 0.5 * float(np.linalg.norm(cross))
                if area <= 0.0 or not math.isfinite(area):
                    raise RuntimeError("degenerate magnet-envelope facet")
                vertices.append(triangle)
                centroids.append(triangle.mean(axis=0))
                normals.append(cross / np.linalg.norm(cross))
                areas.append(area)
                toroidal_cells.append(toroidal_cell)
                poloidal_cells.append(poloidal_cell)
                triangle_ids.append(triangle_id)
    count = len(areas)
    return {
        "vertices_cm": np.asarray(vertices),
        "centroids_cm": np.asarray(centroids),
        "outward_normals": np.asarray(normals),
        "areas_cm2": np.asarray(areas),
        "boundary_role_code": np.full(
            count, boundary_role_code, dtype=np.uint8
        ),
        "toroidal_cell": np.asarray(toroidal_cells, dtype=np.uint16),
        "poloidal_cell": np.asarray(poloidal_cells, dtype=np.uint16),
        "triangle_in_cell": np.asarray(triangle_ids, dtype=np.uint8),
    }


def concatenate_facet_tables(
    tables: tuple[dict[str, np.ndarray], ...],
) -> dict[str, np.ndarray]:
    keys = tuple(tables[0])
    if any(tuple(table) != keys for table in tables):
        raise ValueError("facet tables have incompatible columns")
    return {
        key: np.concatenate([table[key] for table in tables]) for key in keys
    }


def transform_facet_table(
    table: Mapping[str, np.ndarray], linear: np.ndarray
) -> dict[str, np.ndarray]:
    transformed = {
        key: np.asarray(value).copy() for key, value in table.items()
    }
    for key in ("vertices_cm", "centroids_cm", "outward_normals"):
        transformed[key] = np.asarray(table[key]) @ linear.T
    return transformed


def symmetry_cut_facets(
    inner: np.ndarray,
    outer: np.ndarray,
    *,
    toroidal_index: int,
    outward: np.ndarray,
) -> dict[str, np.ndarray]:
    vertices = []
    normals = []
    areas = []
    poloidal_cells = []
    triangle_ids = []
    outward = _normalize(outward)
    for poloidal_cell in range(SOURCE_CAD_GRID[1] - 1):
        corners = (
            inner[toroidal_index, poloidal_cell],
            inner[toroidal_index, poloidal_cell + 1],
            outer[toroidal_index, poloidal_cell + 1],
            outer[toroidal_index, poloidal_cell],
        )
        for triangle_id, indices in enumerate(((0, 1, 2), (0, 2, 3))):
            triangle = np.asarray([corners[index] for index in indices])
            cross = np.cross(
                triangle[1] - triangle[0], triangle[2] - triangle[0]
            )
            if float(np.dot(cross, outward)) < 0.0:
                triangle[[1, 2]] = triangle[[2, 1]]
                cross = -cross
            vertices.append(triangle)
            normals.append(cross / np.linalg.norm(cross))
            areas.append(0.5 * np.linalg.norm(cross))
            poloidal_cells.append(poloidal_cell)
            triangle_ids.append(triangle_id)
    vertices_array = np.asarray(vertices)
    return {
        "vertices_cm": vertices_array,
        "centroids_cm": vertices_array.mean(axis=1),
        "outward_normals": np.asarray(normals),
        "areas_cm2": np.asarray(areas),
        "poloidal_cell": np.asarray(poloidal_cells, dtype=np.uint16),
        "triangle_in_cell": np.asarray(triangle_ids, dtype=np.uint8),
    }


def control_grid_aliases(canonical_id: int) -> list[tuple[int, int]]:
    point_id = canonical_id
    if point_id >= N_SYMMETRIC_PROFILE:
        point_id += N_UNIQUE_POLOIDAL - N_SYMMETRIC_PROFILE
    toroidal_control = point_id // N_UNIQUE_POLOIDAL
    poloidal_control = point_id % N_UNIQUE_POLOIDAL
    aliases = [(toroidal_control, poloidal_control)]
    if toroidal_control in (0, N_TOROIDAL - 1) and 0 < poloidal_control < 15:
        aliases.append(
            (toroidal_control, N_UNIQUE_POLOIDAL - poloidal_control)
        )
    return aliases


def patch_cells(
    toroidal_control: int, poloidal_control: int
) -> list[tuple[int, int]]:
    expanded_toroidal = 4 * toroidal_control
    expanded_poloidal = 4 * poloidal_control
    cells = []
    for toroidal_cell in range(SOURCE_CAD_GRID[0] - 1):
        nearest_toroidal = min(
            range(N_TOROIDAL),
            key=lambda index: abs((toroidal_cell + 0.5) - 4 * index),
        )
        if nearest_toroidal != toroidal_control:
            continue
        for poloidal_cell in range(SOURCE_CAD_GRID[1] - 1):
            delta = abs((poloidal_cell + 0.5) - expanded_poloidal)
            periodic_delta = min(delta, SOURCE_CAD_GRID[1] - 1 - delta)
            nearest_poloidal = min(
                range(N_UNIQUE_POLOIDAL),
                key=lambda index: min(
                    abs((poloidal_cell + 0.5) - 4 * index),
                    SOURCE_CAD_GRID[1]
                    - 1
                    - abs((poloidal_cell + 0.5) - 4 * index),
                ),
            )
            if nearest_poloidal == poloidal_control and periodic_delta <= 2.0:
                cells.append((toroidal_cell, poloidal_cell))
    if not cells:
        raise RuntimeError("canonical control patch has no source-CAD cells")
    return cells


def local_engineering_frame(
    inner: np.ndarray, outer: np.ndarray, row: int, column: int
) -> dict[str, Any]:
    outward = _normalize(outer[row, column] - inner[row, column])
    if row == 0:
        toroidal = outer[row + 1, column] - outer[row, column]
    elif row == SOURCE_CAD_GRID[0] - 1:
        toroidal = outer[row, column] - outer[row - 1, column]
    else:
        toroidal = outer[row + 1, column] - outer[row - 1, column]
    toroidal = toroidal - float(np.dot(toroidal, outward)) * outward
    toroidal = _normalize(toroidal)
    poloidal = _normalize(np.cross(outward, toroidal))
    matrix = np.column_stack((toroidal, poloidal, outward))
    return {
        "origin_cm": (
            0.5 * (inner[row, column] + outer[row, column])
        ).tolist(),
        "toroidal_unit": toroidal.tolist(),
        "poloidal_unit": poloidal.tolist(),
        "radially_outward_unit": outward.tolist(),
        "basis_columns_determinant": float(np.linalg.det(matrix)),
    }


def qualify_engineering_frame(
    frame: Mapping[str, Any],
    instance_linear: np.ndarray,
    *,
    frame_id: str,
) -> dict[str, Any]:
    qualified = dict(frame)
    for key in (
        "origin_cm",
        "toroidal_unit",
        "poloidal_unit",
        "radially_outward_unit",
    ):
        qualified[key] = (
            np.asarray(qualified[key], dtype=float) @ instance_linear.T
        ).tolist()
    basis = np.column_stack(
        (
            qualified["toroidal_unit"],
            qualified["poloidal_unit"],
            qualified["radially_outward_unit"],
        )
    )
    origin = np.asarray(qualified["origin_cm"], dtype=float)
    forward = np.eye(4)
    forward[:3, :3] = basis
    forward[:3, 3] = origin
    inverse = np.eye(4)
    inverse[:3, :3] = basis.T
    inverse[:3, 3] = -basis.T @ origin
    qualified.update(
        {
            "local_engineering_frame_id": frame_id,
            "frame_kind": "ORTHONORMAL_MAGNET_ALIGNED_TOROIDAL_POLOIDAL_RADIAL",
            "tape_twist_resolved": False,
            "forward_transform": forward.tolist(),
            "inverse_transform": inverse.tolist(),
            "forward_inverse_closure_inf_norm": float(
                np.linalg.norm(forward @ inverse - np.eye(4), ord=np.inf)
            ),
            "basis_columns_determinant": float(np.linalg.det(basis)),
        }
    )
    qualified["frame_fingerprint_sha256"] = canonical_json_sha256(qualified)
    return qualified


def cad_face_inventory(
    step_path: Path,
    inner: np.ndarray,
    outer: np.ndarray,
    *,
    extent_degrees: float,
) -> list[dict[str, Any]]:
    shape = cq.importers.importStep(str(step_path)).val()
    inner_tree = cKDTree(inner.reshape(-1, 3))
    outer_tree = cKDTree(outer.reshape(-1, 3))
    rows = []
    for index, face in enumerate(shape.Faces()):
        center_vector = face.Center()
        center = np.array([center_vector.x, center_vector.y, center_vector.z])
        vertices = np.array([[v.X, v.Y, v.Z] for v in face.Vertices()])
        phi_zero_residual = float(np.abs(vertices[:, 1]).max())
        end_radians = math.radians(extent_degrees)
        end_normal = np.array(
            [-math.sin(end_radians), math.cos(end_radians), 0.0]
        )
        phi_end_residual = float(np.abs(vertices @ end_normal).max())
        cut_role = None
        if phi_zero_residual <= 1.0e-5:
            cut_role = "PURE_SYMMETRY_CUT_PHI_0"
        elif phi_end_residual <= 1.0e-5:
            cut_role = f"PURE_SYMMETRY_CUT_PHI_{extent_degrees:g}"
        if cut_role is None:
            inner_distance, inner_index = inner_tree.query(center)
            outer_distance, outer_index = outer_tree.query(center)
            if inner_distance < outer_distance:
                role = "MAGNET_ENVELOPE_INNER_PHYSICAL_BOUNDARY"
                radial = (
                    outer.reshape(-1, 3)[inner_index]
                    - inner.reshape(-1, 3)[inner_index]
                )
                expected = -_normalize(radial)
            else:
                role = "MAGNET_ENVELOPE_OUTER_PHYSICAL_BOUNDARY"
                radial = (
                    outer.reshape(-1, 3)[outer_index]
                    - inner.reshape(-1, 3)[outer_index]
                )
                expected = _normalize(radial)
            try:
                raw_vector = face.normalAt()
                raw = _normalize(
                    np.array([raw_vector.x, raw_vector.y, raw_vector.z])
                )
                outward_normal = (
                    raw if float(np.dot(raw, expected)) >= 0.0 else -raw
                )
                normal_status = "CAD_NORMAL_ORIENTED_BY_ENVELOPE_RADIAL_SENSE"
            except Exception as exc:
                outward_normal = expected
                normal_status = f"RADIAL_SENSE_FALLBACK:{type(exc).__name__}"
        else:
            role = cut_role
            outward_normal = None
            normal_status = "EXCLUDED_PURE_SYMMETRY_CUT"
        rows.append(
            {
                "cad_face_id": index,
                "geometry_type": face.geomType(),
                "area_cm2": float(face.Area()),
                "center_cm": center.tolist(),
                "vertex_count": len(vertices),
                "role": role,
                "included_in_m2": cut_role is None,
                "outward_normal_at_center": (
                    None if outward_normal is None else outward_normal.tolist()
                ),
                "normal_status": normal_status,
                "symmetry_plane_residual_cm": {
                    "phi_0": phi_zero_residual,
                    f"phi_{extent_degrees:g}": phi_end_residual,
                },
            }
        )
    return rows


def build_m2_contract(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    source_root = Path(args.source_root).resolve()
    derived_root = Path(args.derived_root).resolve()
    transform_path = Path(args.transform_receipt).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output root exists: {output_root}")
    verify_source_set(input_root)
    source_acceptance = source_root / "SOURCE_CAD_61X121_ACCEPTANCE.json"
    derived_acceptance = derived_root / "SOURCE_CAD_90D_ACCEPTANCE.json"
    for path in (source_acceptance, derived_acceptance, transform_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = json.loads(source_acceptance.read_text(encoding="utf-8"))
    derived_manifest = json.loads(
        derived_acceptance.read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("classification")
        != "WISTELL_D_45D_61X121_SOURCE_CAD_PASS"
    ):
        raise ValueError("source CAD is not accepted")
    if (
        derived_manifest.get("classification")
        != "WISTELL_D_90D_TRANSPORT_SOURCE_CAD_PASS"
    ):
        raise ValueError("derived CAD is not accepted")
    transform_receipt = json.loads(transform_path.read_text(encoding="utf-8"))
    if (
        transform_receipt.get("classification")
        != "WISTELL_D_EXACT_HALF_PERIOD_TRANSFORM_PASS"
    ):
        raise ValueError("transform receipt is not accepted")
    linear = np.asarray(
        transform_receipt["transforms"]["half_period_mate"]["matrix"],
        dtype=float,
    )[:3, :3]

    output_root.mkdir(parents=True)
    inner, outer = magnet_envelope_loci(input_root)
    radial = _normalize(outer - inner)
    inner_table = triangulate_envelope_boundary(
        inner, -radial, boundary_role_code=0
    )
    outer_table = triangulate_envelope_boundary(
        outer, radial, boundary_role_code=1
    )
    facets_45 = concatenate_facet_tables((inner_table, outer_table))
    facets_45["facet_id"] = np.arange(
        len(facets_45["areas_cm2"]), dtype=np.uint32
    )
    facets_45["symmetry_instance_code"] = np.zeros(
        len(facets_45["areas_cm2"]), dtype=np.uint8
    )
    facets_45["parent_45d_facet_id"] = facets_45["facet_id"].copy()
    facets_mate = transform_facet_table(facets_45, linear)
    facets_mate["symmetry_instance_code"] = np.ones(
        len(facets_mate["areas_cm2"]), dtype=np.uint8
    )
    facets_90 = concatenate_facet_tables((facets_45, facets_mate))
    facets_90["facet_id"] = np.arange(
        len(facets_90["areas_cm2"]), dtype=np.uint32
    )

    start_cut = symmetry_cut_facets(
        inner, outer, toroidal_index=0, outward=np.array([0.0, -1.0, 0.0])
    )
    seam_angle = math.radians(45.0)
    end_cut = symmetry_cut_facets(
        inner,
        outer,
        toroidal_index=SOURCE_CAD_GRID[0] - 1,
        outward=np.array([-math.sin(seam_angle), math.cos(seam_angle), 0.0]),
    )
    mate_start_cut = transform_facet_table(start_cut, linear)

    facets_45_path = output_root / "M2_PHYSICAL_FACETS_45D.npz"
    facets_90_path = output_root / "M2_PHYSICAL_FACETS_90D.npz"
    cuts_path = output_root / "M2_EXCLUDED_SYMMETRY_CUT_FACETS.npz"
    np.savez_compressed(facets_45_path, **facets_45)
    np.savez_compressed(facets_90_path, **facets_90)
    np.savez_compressed(
        cuts_path,
        canonical_phi0_vertices_cm=start_cut["vertices_cm"],
        canonical_phi0_areas_cm2=start_cut["areas_cm2"],
        canonical_phi45_internal_seam_vertices_cm=end_cut["vertices_cm"],
        canonical_phi45_internal_seam_areas_cm2=end_cut["areas_cm2"],
        derived_phi90_vertices_cm=mate_start_cut["vertices_cm"],
        derived_phi90_areas_cm2=mate_start_cut["areas_cm2"],
    )

    inner_cell_area = inner_table["areas_cm2"].reshape(60, 120, 2).sum(axis=2)
    outer_cell_area = outer_table["areas_cm2"].reshape(60, 120, 2).sum(axis=2)
    nwl = np.load(input_root / "nwl.npy")
    normalized_nwl = (nwl - nwl.min()) / (nwl.max() - nwl.min())
    mapping = []
    for canonical_id in range(N_CONTROL_POINTS):
        aliases = control_grid_aliases(canonical_id)
        for instance_id, instance_linear in (
            ("canonical", np.eye(3)),
            ("mate", linear),
        ):
            alias_rows = []
            for alias_index, (toroidal_control, poloidal_control) in enumerate(
                aliases
            ):
                row = 4 * toroidal_control
                column = 4 * poloidal_control
                cells = patch_cells(toroidal_control, poloidal_control)
                inner_area = float(
                    sum(inner_cell_area[cell] for cell in cells)
                )
                outer_area = float(
                    sum(outer_cell_area[cell] for cell in cells)
                )
                frame_id = (
                    f"LEF:{instance_id}:t{toroidal_control:02d}:"
                    f"p{poloidal_control:02d}"
                )
                frame = qualify_engineering_frame(
                    local_engineering_frame(inner, outer, row, column),
                    instance_linear,
                    frame_id=frame_id,
                )
                instance_facet_offset = (
                    0
                    if instance_id == "canonical"
                    else len(facets_45["areas_cm2"])
                )
                inner_facet_ids = [
                    instance_facet_offset
                    + 2 * (cell[0] * (SOURCE_CAD_GRID[1] - 1) + cell[1])
                    + triangle_id
                    for cell in cells
                    for triangle_id in (0, 1)
                ]
                outer_facet_ids = [
                    instance_facet_offset
                    + len(inner_table["areas_cm2"])
                    + 2 * (cell[0] * (SOURCE_CAD_GRID[1] - 1) + cell[1])
                    + triangle_id
                    for cell in cells
                    for triangle_id in (0, 1)
                ]
                canonical_45_patch_id = (
                    f"M2:canonical:location:t{toroidal_control:02d}:"
                    f"p{poloidal_control:02d}"
                )
                alias_rows.append(
                    {
                        "alias_index": alias_index,
                        "source_control_grid": [
                            toroidal_control,
                            poloidal_control,
                        ],
                        "source_cad_grid": [row, column],
                        "toroidal_degrees_45d": 3.0 * toroidal_control,
                        "poloidal_degrees": 12.0 * poloidal_control,
                        "canonical_45_patch_id": canonical_45_patch_id,
                        "local_engineering_frame_id": frame_id,
                        "frame_kind": frame["frame_kind"],
                        "tape_twist_resolved": frame["tape_twist_resolved"],
                        "forward_transform": frame["forward_transform"],
                        "inverse_transform": frame["inverse_transform"],
                        "magnet_inner_coordinate_cm": (
                            inner[row, column] @ instance_linear.T
                        ).tolist(),
                        "magnet_outer_coordinate_cm": (
                            outer[row, column] @ instance_linear.T
                        ).tolist(),
                        "engineering_frame": frame,
                        "inner_patch": {
                            "patch_id": (
                                f"M2:{instance_id}:inner:t{toroidal_control:02d}:"
                                f"p{poloidal_control:02d}"
                            ),
                            "source_cad_cells": [list(cell) for cell in cells],
                            "area_cm2": inner_area,
                            "outward_sense": "toward_vacuum_gap",
                            "global_90_facet_ids": inner_facet_ids,
                        },
                        "outer_patch": {
                            "patch_id": (
                                f"M2:{instance_id}:outer:t{toroidal_control:02d}:"
                                f"p{poloidal_control:02d}"
                            ),
                            "source_cad_cells": [list(cell) for cell in cells],
                            "area_cm2": outer_area,
                            "outward_sense": "away_from_magnet_envelope",
                            "global_90_facet_ids": outer_facet_ids,
                        },
                    }
                )
            mapping.append(
                {
                    "canonical_control_id": canonical_id,
                    "nwl_mw_per_m2": float(nwl[canonical_id]),
                    "nwl_normalized": float(normalized_nwl[canonical_id]),
                    "symmetry_instance_id": instance_id,
                    "transform_id": (
                        "identity"
                        if instance_id == "canonical"
                        else "half_period_mate"
                    ),
                    "aliases": alias_rows,
                }
            )

    mapping_path = output_root / "CANONICAL_452_M2_MAPPING.json"
    source_step = source_root / "magnet_envelope.step"
    derived_step = derived_root / "magnet_envelope.step"
    cad_faces_45 = cad_face_inventory(
        source_step, inner, outer, extent_degrees=45.0
    )
    mate_inner = inner @ linear.T
    mate_outer = outer @ linear.T
    full_inner = np.concatenate((inner, mate_inner), axis=0)
    full_outer = np.concatenate((outer, mate_outer), axis=0)
    cad_faces_90 = cad_face_inventory(
        derived_step, full_inner, full_outer, extent_degrees=90.0
    )
    surface_ids_by_role = {
        "winding_pack": sorted(
            row["cad_face_id"]
            for row in cad_faces_90
            if row["role"] == "MAGNET_ENVELOPE_INNER_PHYSICAL_BOUNDARY"
        ),
        "outer_casing": sorted(
            row["cad_face_id"]
            for row in cad_faces_90
            if row["role"] == "MAGNET_ENVELOPE_OUTER_PHYSICAL_BOUNDARY"
        ),
    }
    if any(not ids for ids in surface_ids_by_role.values()):
        raise RuntimeError(
            "90-degree CAD face inventory lacks an M2 physical role"
        )
    facets_90_sha256 = sha256_file(facets_90_path)
    local_frame_rows = []
    product_rows = []
    for record in mapping:
        for alias in record["aliases"]:
            inner_patch = alias["inner_patch"]
            outer_patch = alias["outer_patch"]
            inner_patch["global_90_surface_ids"] = surface_ids_by_role[
                "winding_pack"
            ]
            outer_patch["global_90_surface_ids"] = surface_ids_by_role[
                "outer_casing"
            ]
            global_surface_ids = sorted(
                set(
                    inner_patch["global_90_surface_ids"]
                    + outer_patch["global_90_surface_ids"]
                )
            )
            global_facet_ids = sorted(
                inner_patch["global_90_facet_ids"]
                + outer_patch["global_90_facet_ids"]
            )
            frame = alias["engineering_frame"]
            local_frame_rows.append(
                {
                    "canonical_control_point_id": record[
                        "canonical_control_id"
                    ],
                    "canonical_45_patch_id": alias["canonical_45_patch_id"],
                    "symmetry_instance_id": record["symmetry_instance_id"],
                    "transform_id": record["transform_id"],
                    "global_90_surface_ids": global_surface_ids,
                    "global_90_facet_ids": global_facet_ids,
                    "local_engineering_frame_id": alias[
                        "local_engineering_frame_id"
                    ],
                    "forward_transform": alias["forward_transform"],
                    "inverse_transform": alias["inverse_transform"],
                    "frame_kind": alias["frame_kind"],
                    "tape_twist_resolved": alias["tape_twist_resolved"],
                    "frame_fingerprint_sha256": frame[
                        "frame_fingerprint_sha256"
                    ],
                }
            )
            for component_role, patch, normal_sign in (
                ("winding_pack", inner_patch, -1.0),
                ("outer_casing", outer_patch, 1.0),
            ):
                role_payload = {
                    "component_role": component_role,
                    "physical_model": "continuous_homogenized_magnet_envelope",
                    "exact_coil_or_tape_surface": False,
                    "canonical_control_point_id": record[
                        "canonical_control_id"
                    ],
                    "canonical_45_patch_id": alias["canonical_45_patch_id"],
                    "patch_id": patch["patch_id"],
                    "symmetry_instance_id": record["symmetry_instance_id"],
                    "transform_id": record["transform_id"],
                    "area_cm2": patch["area_cm2"],
                    "outward_normal": (
                        normal_sign
                        * np.asarray(
                            frame["radially_outward_unit"], dtype=float
                        )
                    ).tolist(),
                    "local_engineering_frame_id": alias[
                        "local_engineering_frame_id"
                    ],
                    "engineering_frame": frame,
                    "global_90_surface_ids": patch["global_90_surface_ids"],
                    "global_90_facet_ids": patch["global_90_facet_ids"],
                    "tape_twist_resolved": False,
                }
                role_payload["facet_fingerprint_sha256"] = (
                    canonical_json_sha256(
                        {
                            "m2_physical_facets_90d_sha256": facets_90_sha256,
                            "global_90_facet_ids": patch[
                                "global_90_facet_ids"
                            ],
                        }
                    )
                )
                role_payload["patch_fingerprint_sha256"] = (
                    canonical_json_sha256(role_payload)
                )
                product_rows.append(role_payload)

    write_json_create_only(
        mapping_path,
        {
            "schema": "wistell_d.canonical_452_m2_mapping/v1.1.0",
            "geometry_input_mode": WISTELL_D_GEOMETRY_INPUT_MODE,
            "canonical_control_count": N_CONTROL_POINTS,
            "symmetry_instance_count": 2,
            "records": mapping,
            "local_frames": {
                "status": "PASS",
                "frame_count": len(local_frame_rows),
                "frames": local_frame_rows,
            },
            "source_hashes": dict(WISTELL_D_SOURCE_HASHES),
        },
    )
    product_inventory_path = (
        output_root / "PROMPT02_GEOMETRY_PRODUCT_INVENTORY.json"
    )
    write_json_create_only(
        product_inventory_path,
        {
            "schema": "wistell_d.prompt02_geometry_product_inventory/v1.0.0",
            "generated_utc": utc_now(),
            "classification": "P2G5_GEOMETRY_PATCH_CATALOG_PASS",
            "representation": "continuous_homogenized_magnet_envelope",
            "exact_component_roles": ["outer_casing", "winding_pack"],
            "tape_twist_resolved": False,
            "role_semantics": {
                "outer_casing": (
                    "logical engineering role on the physical outer boundary of "
                    "the homogenized envelope"
                ),
                "winding_pack": (
                    "logical engineering role on the physical inner boundary of "
                    "the homogenized envelope; not an exact tape or coil surface"
                ),
            },
            "patch_count": len(product_rows),
            "patches": product_rows,
            "source_hashes": dict(WISTELL_D_SOURCE_HASHES),
        },
    )
    prompt02_acceptance_paths = {}
    for extent_label, raw_manifest, raw_path, allowed_instances in (
        ("45D", source_manifest, source_acceptance, {"canonical"}),
        (
            "90D",
            derived_manifest,
            derived_acceptance,
            {"canonical", "mate"},
        ),
    ):
        accepted_frames = [
            row
            for row in local_frame_rows
            if row["symmetry_instance_id"] in allowed_instances
        ]
        frame_contract = {
            "status": "PASS",
            "accepted_for_prompt02": True,
            "frame_count": len(accepted_frames),
            "selected_patch_scope": "ALL_CANONICAL_452_NWL_CONTROL_PATCH_ALIASES",
            "frames": accepted_frames,
        }
        qualified_manifest = {
            **raw_manifest,
            "generated_utc": utc_now(),
            "local_frames": frame_contract,
            "prompt02_producer_gate": {
                "P2G2_local_engineering_frames": "PASS",
                "canonical_control_count": N_CONTROL_POINTS,
                "canonical_452_identity_preserved": True,
                "exact_45_to_90_transform_preserved": True,
                "tape_twist_resolved": False,
                "representation": "continuous_homogenized_magnet_envelope",
            },
            "artifacts": {
                **raw_manifest["artifacts"],
                **artifact_rows(
                    output_root,
                    {
                        "raw_cad_acceptance": raw_path,
                        "m2_mapping": mapping_path,
                        "prompt02_geometry_product_inventory": (
                            product_inventory_path
                        ),
                    },
                ),
            },
        }
        qualified_path = (
            output_root / f"PROMPT02_{extent_label}_CAD_ACCEPTANCE.json"
        )
        write_json_create_only(qualified_path, qualified_manifest)
        prompt02_acceptance_paths[extent_label] = qualified_path
    artifact_files = {
        "physical_facets_45d": facets_45_path,
        "physical_facets_90d": facets_90_path,
        "excluded_symmetry_cut_facets": cuts_path,
        "canonical_452_mapping": mapping_path,
        "prompt02_geometry_product_inventory": product_inventory_path,
        "prompt02_45d_cad_acceptance": prompt02_acceptance_paths["45D"],
        "prompt02_90d_cad_acceptance": prompt02_acceptance_paths["90D"],
        "source_acceptance": source_acceptance,
        "derived_acceptance": derived_acceptance,
        "transform_receipt": transform_path,
        "source_magnet_step": source_step,
        "derived_magnet_step": derived_step,
    }
    payload = {
        "schema": "wistell_d.m2_geometry_handshake/v1.0.0",
        "generated_utc": utc_now(),
        "classification": "M2_GEOMETRY_INVENTORY_PASS",
        "geometry_input_mode": WISTELL_D_GEOMETRY_INPUT_MODE,
        "representation": {
            "kind": "continuous_homogenized_30_cm_magnet_envelope",
            "exact_coil_or_tape_surfaces": False,
            "logical_patches_are_not_physical_coils": True,
        },
        "boundary_roles": {
            "M1": "MAGNET_ENVELOPE_INNER_ENTRY",
            "M2": "MAGNET_ENVELOPE_COMPLETE_PHYSICAL_BOUNDARY",
            "M2_included": [
                "MAGNET_ENVELOPE_INNER_PHYSICAL_BOUNDARY",
                "MAGNET_ENVELOPE_OUTER_PHYSICAL_BOUNDARY",
            ],
            "M2_excluded": ["PURE_SECTOR_OR_FIELD_PERIOD_SYMMETRY_CUT"],
        },
        "parametric_facet_inventory": {
            "45d": {
                "facet_count": int(len(facets_45["areas_cm2"])),
                "area_cm2": float(facets_45["areas_cm2"].sum()),
                "inner_area_cm2": float(inner_table["areas_cm2"].sum()),
                "outer_area_cm2": float(outer_table["areas_cm2"].sum()),
            },
            "90d": {
                "facet_count": int(len(facets_90["areas_cm2"])),
                "area_cm2": float(facets_90["areas_cm2"].sum()),
                "area_scale_from_45d": float(
                    facets_90["areas_cm2"].sum() / facets_45["areas_cm2"].sum()
                ),
            },
            "ancestry": (
                "90D canonical facets retain parent_45d_facet_id; mate facets use "
                "the exact half_period_mate transform"
            ),
            "normal_sense": (
                "outward from homogenized magnet volume; inner normals point toward "
                "vacuum_gap and outer normals point away from the envelope"
            ),
        },
        "symmetry_cut_exclusions": {
            "45d_phi0_facet_count": int(len(start_cut["areas_cm2"])),
            "45d_phi45_facet_count": int(len(end_cut["areas_cm2"])),
            "90d_phi0_facet_count": int(len(start_cut["areas_cm2"])),
            "90d_phi90_facet_count": int(len(mate_start_cut["areas_cm2"])),
            "internal_phi45_seam": "REMOVED_BY_SAME_MATERIAL_BREP_FUSE",
            "included_in_m2": False,
        },
        "cad_surface_inventory": {"45d": cad_faces_45, "90d": cad_faces_90},
        "canonical_mapping": {
            "canonical_control_count": N_CONTROL_POINTS,
            "record_count": len(mapping),
            "instance_ids": ["canonical", "mate"],
            "mapping_path": str(mapping_path),
            "local_frame_count": len(local_frame_rows),
        },
        "prompt02_product_inventory": {
            "roles": ["outer_casing", "winding_pack"],
            "patch_count": len(product_rows),
            "tape_twist_resolved": False,
        },
        "source_hashes": dict(WISTELL_D_SOURCE_HASHES),
        "artifacts": artifact_rows(output_root, artifact_files),
    }
    if not math.isclose(
        payload["parametric_facet_inventory"]["90d"]["area_scale_from_45d"],
        2.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "90-degree M2 area does not equal two 45-degree instances"
        )
    write_json_create_only(output_root / "M2_GEOMETRY_HANDSHAKE.json", payload)
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "output": str(output_root),
            },
            indent=2,
        )
    )


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
                            array_to_symmetric_matrix(
                                np.load(input_root / name)
                            )[0]
                            - array_to_symmetric_matrix(
                                np.load(input_root / name)
                            )[0][::-1]
                        ).max()
                    ),
                    "end_profile_symmetry_max": float(
                        np.abs(
                            array_to_symmetric_matrix(
                                np.load(input_root / name)
                            )[-1]
                            - array_to_symmetric_matrix(
                                np.load(input_root / name)
                            )[-1][::-1]
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
        payload["classification"] = (
            "WISTELL_D_EXACT_HALF_PERIOD_TRANSFORM_BLOCKED"
        )
        raise RuntimeError(json.dumps(payload, indent=2))
    write_json_create_only(output, payload)
    print(
        json.dumps(
            {"output": str(output), "verification": payload["verification"]},
            indent=2,
        )
    )


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
        volume_scale = (
            joined_metrics["volume_cm3"] / source_metrics["volume_cm3"]
        )
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
            or not math.isclose(
                volume_scale, 2.0, rel_tol=2.0e-8, abs_tol=1.0e-8
            )
        ):
            raise RuntimeError(
                f"90-degree seam expansion failed for {name}: {row}"
            )
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
    components = {
        name: shape_metrics(shape) for name, shape in expanded.items()
    }
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
        "local_frames": {
            "status": "PENDING_M2_QUALIFICATION",
            "frame_count": 0,
            "frames": [],
            "accepted_for_prompt02": False,
        },
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


def mesh_cad_to_dagmc(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output root exists: {output_root}")
    manifest_names = (
        "SOURCE_CAD_61X121_ACCEPTANCE.json",
        "SOURCE_CAD_90D_ACCEPTANCE.json",
    )
    manifest_paths = [source_root / name for name in manifest_names]
    existing = [path for path in manifest_paths if path.is_file()]
    if len(existing) != 1:
        raise ValueError(
            "source root must contain exactly one CAD acceptance manifest"
        )
    manifest_path = existing[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_classification = {
        manifest_names[0]: "WISTELL_D_45D_61X121_SOURCE_CAD_PASS",
        manifest_names[1]: "WISTELL_D_90D_TRANSPORT_SOURCE_CAD_PASS",
    }[manifest_path.name]
    if manifest.get("classification") != expected_classification:
        raise ValueError(
            "CAD acceptance classification is not eligible for meshing"
        )
    if manifest.get("geometry_input_mode") != WISTELL_D_GEOMETRY_INPUT_MODE:
        raise ValueError("CAD acceptance has the wrong geometry input mode")
    if manifest.get("model", {}).get("source_cad_grid") != list(
        SOURCE_CAD_GRID
    ):
        raise ValueError(
            "CAD acceptance is not based on the 61x121 source CAD"
        )

    import cad_to_dagmc
    import pydagmc

    output_root.mkdir(parents=True)
    names = tuple(manifest["model"]["component_order"])
    components = load_step_components(source_root, names)
    geometry = cq.Compound.makeCompound(list(components.values()))
    material_tags = [manifest["materials"][name] for name in names]
    h5m_path = (
        output_root
        / f"wistell_d_{int(manifest['model']['toroidal_extent_degrees'])}d_{args.label}.h5m"
    )
    started = time.time()
    gmsh_object = cad_to_dagmc.init_gmsh()
    try:
        _, volumes = cad_to_dagmc.get_volumes(
            gmsh_object, geometry, method="in memory"
        )
        if len(volumes) != len(names):
            raise RuntimeError(
                f"CAD-to-DAGMC imported {len(volumes)} volumes; expected {len(names)}"
            )
        cad_to_dagmc.set_sizes_for_mesh(
            gmsh_object,
            min_mesh_size=args.min_mesh_size_cm,
            max_mesh_size=args.max_mesh_size_cm,
            mesh_algorithm=args.algorithm,
        )
        gmsh_object.model.mesh.generate(dim=2)
        vertices, triangles_by_solid_and_by_face = (
            cad_to_dagmc.mesh_to_vertices_and_triangles(volumes)
        )
    finally:
        gmsh_object.finalize()
    cad_to_dagmc.vertices_to_h5m(
        vertices,
        triangles_by_solid_and_by_face,
        material_tags,
        h5m_filename=h5m_path,
    )

    model = pydagmc.Model(str(h5m_path))
    surface_rows = []
    for surface in model.surfaces:
        senses = [
            None if volume is None else int(volume.id)
            for volume in surface.senses
        ]
        surface_rows.append(
            {
                "surface_id": int(surface.id),
                "triangle_count": int(surface.num_triangles),
                "area_cm2": float(surface.area),
                "sense_volume_ids": senses,
                "parent_volume_ids": sorted(
                    int(volume.id) for volume in surface.volumes
                ),
            }
        )
    volume_rows = []
    for volume in model.volumes:
        volume_rows.append(
            {
                "volume_id": int(volume.id),
                "material": volume.material,
                "surface_ids": sorted(
                    int(surface.id) for surface in volume.surfaces
                ),
                "triangle_reference_count": int(volume.num_triangles),
            }
        )
    topology = {
        "volume_count": len(model.volumes),
        "surface_count": len(model.surfaces),
        "group_names": sorted(model.group_names),
        "volumes_without_material": [
            int(volume.id) for volume in model.volumes_without_material
        ],
        "zero_triangle_surface_ids": [
            row["surface_id"]
            for row in surface_rows
            if row["triangle_count"] <= 0
        ],
        "missing_sense_surface_ids": [
            row["surface_id"]
            for row in surface_rows
            if row["sense_volume_ids"] == [None, None]
        ],
        "surface_triangle_count": int(
            sum(row["triangle_count"] for row in surface_rows)
        ),
        "surface_area_cm2": float(
            sum(row["area_cm2"] for row in surface_rows)
        ),
    }
    topology_pass = (
        topology["volume_count"] == len(names)
        and not topology["volumes_without_material"]
        and not topology["zero_triangle_surface_ids"]
        and not topology["missing_sense_surface_ids"]
        and sum(len(volumes) for volumes in model.volumes_by_material.values())
        == len(names)
    )
    if not topology_pass:
        raise RuntimeError(f"DAGMC topology failed: {topology}")
    payload = {
        "schema": "wistell_d.dagmc_qualification/v1.0.0",
        "generated_utc": utc_now(),
        "classification": "DAGMC_H5M_TOPOLOGY_PASS_OPENMC_DEBUG_PENDING",
        "geometry_input_mode": WISTELL_D_GEOMETRY_INPUT_MODE,
        "extent_degrees": manifest["model"]["toroidal_extent_degrees"],
        "faceting_label": args.label,
        "faceting": {
            "backend": "cad_to_dagmc",
            "min_mesh_size_cm": args.min_mesh_size_cm,
            "max_mesh_size_cm": args.max_mesh_size_cm,
            "gmsh_algorithm": args.algorithm,
            "source_cad_grid": list(SOURCE_CAD_GRID),
        },
        "source_cad_acceptance": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "source_hashes": dict(WISTELL_D_SOURCE_HASHES),
        "topology": topology,
        "surfaces": surface_rows,
        "volumes": volume_rows,
        "openmc_geometry_debug": {
            "status": "BLOCKED",
            "reason": (
                "Bateman geometry-only policy forbids OpenMC transport and the "
                "authorized local runtime has no OpenMC installation"
            ),
            "two_seed_debug_run_performed": False,
        },
        "artifacts": artifact_rows(output_root, {"dagmc_h5m": h5m_path}),
        "elapsed_seconds": time.time() - started,
    }
    write_json_create_only(output_root / "DAGMC_QUALIFICATION.json", payload)
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "output": str(output_root / "DAGMC_QUALIFICATION.json"),
                "h5m_sha256": payload["artifacts"]["dagmc_h5m"]["sha256"],
            },
            indent=2,
        )
    )


def load_mesh_qualification(
    root: Path, *, extent_degrees: float, label: str
) -> tuple[dict[str, Any], Path]:
    receipt_path = root / "DAGMC_QUALIFICATION.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("classification") != (
        "DAGMC_H5M_TOPOLOGY_PASS_OPENMC_DEBUG_PENDING"
    ):
        raise ValueError(f"DAGMC topology is not accepted: {receipt_path}")
    if not math.isclose(
        float(receipt.get("extent_degrees", math.nan)),
        extent_degrees,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"DAGMC extent mismatch: {receipt_path}")
    if receipt.get("faceting_label") != label:
        raise ValueError(f"DAGMC faceting-label mismatch: {receipt_path}")
    h5m_files = list(root.glob("*.h5m"))
    if len(h5m_files) != 1:
        raise ValueError(f"expected exactly one H5M in {root}")
    h5m_path = h5m_files[0].resolve()
    expected_hash = receipt["artifacts"]["dagmc_h5m"]["sha256"]
    if sha256_file(h5m_path) != expected_hash:
        raise ValueError(f"DAGMC H5M hash mismatch: {h5m_path}")
    return receipt, h5m_path


def component_clearance_inventory(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    minimum_nonadjacent_clearance_cm: float,
    interface_gap_tolerance_cm: float,
) -> dict[str, Any]:
    names = tuple(manifest["model"]["component_order"])
    if names != EXPECTED_LAYER_ORDER:
        raise ValueError("clearance audit component order is not canonical")
    pairwise = manifest["complete_pairwise_audit"]
    require_complete_pairwise_acceptance(pairwise)
    components = load_step_components(root, names)
    rows = []
    failures = []
    for left_index, left_name in enumerate(names[:-1]):
        for right_index, right_name in enumerate(
            names[left_index + 1 :], start=left_index + 1
        ):
            adjacent = right_index == left_index + 1
            try:
                distance = float(
                    components[left_name].distance(components[right_name])
                )
                if not math.isfinite(distance) or distance < 0.0:
                    raise ValueError(
                        "CAD distance must be finite and nonnegative"
                    )
                error = None
            except Exception as exc:
                distance = None
                error = f"{type(exc).__name__}: {exc}"
            if error is not None:
                status = "BOOLEAN_FAILURE"
            elif adjacent and distance > interface_gap_tolerance_cm:
                status = "ADJACENT_INTERFACE_GAP_FAILURE"
            elif not adjacent and distance < minimum_nonadjacent_clearance_cm:
                status = "NONADJACENT_CLEARANCE_FAILURE"
            else:
                status = "PASS"
            row = {
                "left": left_name,
                "right": right_name,
                "left_index": left_index,
                "right_index": right_index,
                "adjacent_in_radial_stack": adjacent,
                "minimum_distance_cm": distance,
                "distance_boolean_error": error,
                "required_max_interface_gap_cm": (
                    interface_gap_tolerance_cm if adjacent else None
                ),
                "required_minimum_clearance_cm": (
                    None if adjacent else minimum_nonadjacent_clearance_cm
                ),
                "status": status,
            }
            rows.append(row)
            if status != "PASS":
                failures.append(row)
    nonadjacent_distances = [
        row["minimum_distance_cm"]
        for row in rows
        if not row["adjacent_in_radial_stack"]
        and row["minimum_distance_cm"] is not None
    ]
    return {
        "component_order": list(names),
        "expected_pair_count": len(names) * (len(names) - 1) // 2,
        "evaluated_pair_count": len(rows),
        "distance_boolean_failure_count": sum(
            row["distance_boolean_error"] is not None for row in rows
        ),
        "overlap_boolean_failure_count": pairwise["boolean_failure_count"],
        "overlap_count": pairwise["overlap_count"],
        "minimum_nonadjacent_clearance_cm": min(nonadjacent_distances),
        "required_minimum_nonadjacent_clearance_cm": (
            minimum_nonadjacent_clearance_cm
        ),
        "required_maximum_adjacent_interface_gap_cm": (
            interface_gap_tolerance_cm
        ),
        "failure_count": len(failures),
        "pairs": rows,
    }


def audit_prompt02_clearance(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    derived_root = Path(args.derived_root).resolve()
    output = Path(args.output).resolve()
    source_manifest_path = source_root / "SOURCE_CAD_61X121_ACCEPTANCE.json"
    derived_manifest_path = derived_root / "SOURCE_CAD_90D_ACCEPTANCE.json"
    transform_path = Path(args.transform_receipt).resolve()
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    derived_manifest = json.loads(
        derived_manifest_path.read_text(encoding="utf-8")
    )
    if source_manifest.get("classification") != (
        "WISTELL_D_45D_61X121_SOURCE_CAD_PASS"
    ):
        raise ValueError("45-degree CAD is not accepted")
    if derived_manifest.get("classification") != (
        "WISTELL_D_90D_TRANSPORT_SOURCE_CAD_PASS"
    ):
        raise ValueError("90-degree CAD is not accepted")
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    if transform.get("classification") != (
        "WISTELL_D_EXACT_HALF_PERIOD_TRANSFORM_PASS"
    ):
        raise ValueError("exact symmetry transform is not accepted")

    mesh_roots = {
        "45d_selected": Path(args.source_selected_mesh_root).resolve(),
        "45d_refined": Path(args.source_refined_mesh_root).resolve(),
        "90d_selected": Path(args.derived_selected_mesh_root).resolve(),
        "90d_refined": Path(args.derived_refined_mesh_root).resolve(),
    }
    mesh_artifacts = {}
    for role, root in mesh_roots.items():
        extent = 45.0 if role.startswith("45d") else 90.0
        label = "selected" if role.endswith("selected") else "refined"
        receipt, h5m_path = load_mesh_qualification(
            root, extent_degrees=extent, label=label
        )
        mesh_artifacts[f"{role}_h5m"] = h5m_path
        mesh_artifacts[f"{role}_qualification"] = (
            root / "DAGMC_QUALIFICATION.json"
        )
        if receipt["topology"]["volume_count"] != len(EXPECTED_LAYER_ORDER):
            raise ValueError(f"wrong DAGMC volume count for {role}")

    clearance = {
        "45d": component_clearance_inventory(
            source_root,
            source_manifest,
            minimum_nonadjacent_clearance_cm=(
                args.minimum_nonadjacent_clearance_cm
            ),
            interface_gap_tolerance_cm=args.interface_gap_tolerance_cm,
        ),
        "90d": component_clearance_inventory(
            derived_root,
            derived_manifest,
            minimum_nonadjacent_clearance_cm=(
                args.minimum_nonadjacent_clearance_cm
            ),
            interface_gap_tolerance_cm=args.interface_gap_tolerance_cm,
        ),
    }
    seam = derived_manifest["seam_validation"]
    seam_pass = (
        seam["positive_volume_overlap_count"] == 0
        and seam["gap_or_disconnected_component_count"] == 0
        and all(row["joined_valid_brep"] for row in seam["component_results"])
        and all(
            math.isclose(
                row["volume_scale_90d_over_45d"],
                2.0,
                rel_tol=2.0e-8,
                abs_tol=1.0e-8,
            )
            for row in seam["component_results"]
        )
    )
    gate_pass = (
        seam_pass
        and all(row["failure_count"] == 0 for row in clearance.values())
        and all(
            row["distance_boolean_failure_count"] == 0
            and row["overlap_boolean_failure_count"] == 0
            and row["overlap_count"] == 0
            for row in clearance.values()
        )
    )
    payload = {
        "schema": "wistell_d.prompt02_geometry_clearance/v1.0.0",
        "generated_utc": utc_now(),
        "classification": (
            "P2G3_ALL_PAIR_CLEARANCE_AND_SEAM_PASS"
            if gate_pass
            else "P2G3_ALL_PAIR_CLEARANCE_AND_SEAM_BLOCKED"
        ),
        "geometry_input_mode": WISTELL_D_GEOMETRY_INPUT_MODE,
        "all_pair_clearance": clearance,
        "boolean_failure_count": sum(
            row["distance_boolean_failure_count"]
            + row["overlap_boolean_failure_count"]
            for row in clearance.values()
        ),
        "overlap_count": sum(
            row["overlap_count"] for row in clearance.values()
        ),
        "seam_validation": {
            "status": "PASS" if seam_pass else "BLOCKED",
            "shared_seam_degrees": seam["shared_seam_degrees"],
            "operation": seam["operation"],
            "positive_volume_overlap_count": seam[
                "positive_volume_overlap_count"
            ],
            "gap_or_disconnected_component_count": seam[
                "gap_or_disconnected_component_count"
            ],
            "component_results": seam["component_results"],
            "transform_id": "half_period_mate",
            "transform_sha256": sha256_file(transform_path),
        },
        "source_hashes": dict(WISTELL_D_SOURCE_HASHES),
        "accepted_artifacts": artifact_rows(
            output.parent,
            {
                "45d_combined_step": (
                    source_root / "wistell_d_45d_61x121_source_cad.step"
                ),
                "90d_combined_step": (
                    derived_root / "wistell_d_90d_symmetry_derived.step"
                ),
                "45d_cad_acceptance": source_manifest_path,
                "90d_cad_acceptance": derived_manifest_path,
                "transform_receipt": transform_path,
                **mesh_artifacts,
            },
        ),
    }
    if not gate_pass:
        raise RuntimeError(json.dumps(payload, indent=2))
    write_json_create_only(output, payload)
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "output": str(output),
            },
            indent=2,
        )
    )


def qualify_prompt02_geometry(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    derived_root = Path(args.derived_root).resolve()
    m2_root = Path(args.m2_root).resolve()
    clearance_path = Path(args.clearance_receipt).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output root exists: {output_root}")
    source_path = source_root / "SOURCE_CAD_61X121_ACCEPTANCE.json"
    derived_path = derived_root / "SOURCE_CAD_90D_ACCEPTANCE.json"
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    derived_manifest = json.loads(derived_path.read_text(encoding="utf-8"))
    mapping_path = m2_root / "CANONICAL_452_M2_MAPPING.json"
    product_path = m2_root / "PROMPT02_GEOMETRY_PRODUCT_INVENTORY.json"
    m2_path = m2_root / "M2_GEOMETRY_HANDSHAKE.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    product = json.loads(product_path.read_text(encoding="utf-8"))
    m2 = json.loads(m2_path.read_text(encoding="utf-8"))
    clearance = json.loads(clearance_path.read_text(encoding="utf-8"))
    if mapping.get("local_frames", {}).get("status") != "PASS":
        raise ValueError("M2 local-frame mapping is not accepted")
    if product.get("classification") != "P2G5_GEOMETRY_PATCH_CATALOG_PASS":
        raise ValueError("Prompt-2 geometry product inventory is not accepted")
    if m2.get("classification") != "M2_GEOMETRY_INVENTORY_PASS":
        raise ValueError("M2 geometry inventory is not accepted")
    if clearance.get("classification") != (
        "P2G3_ALL_PAIR_CLEARANCE_AND_SEAM_PASS"
    ):
        raise ValueError("Prompt-2 clearance receipt is not accepted")

    mesh_roots = {
        "45d_selected": Path(args.source_selected_mesh_root).resolve(),
        "45d_refined": Path(args.source_refined_mesh_root).resolve(),
        "90d_selected": Path(args.derived_selected_mesh_root).resolve(),
        "90d_refined": Path(args.derived_refined_mesh_root).resolve(),
    }
    mesh_files = {}
    for role, root in mesh_roots.items():
        extent = 45.0 if role.startswith("45d") else 90.0
        label = "selected" if role.endswith("selected") else "refined"
        _, mesh_files[role] = load_mesh_qualification(
            root, extent_degrees=extent, label=label
        )

    all_frames = mapping["local_frames"]["frames"]
    output_root.mkdir(parents=True)
    accepted_paths = {}
    for extent_label, root, raw_path, raw_manifest, instances in (
        ("45D", source_root, source_path, source_manifest, {"canonical"}),
        (
            "90D",
            derived_root,
            derived_path,
            derived_manifest,
            {"canonical", "mate"},
        ),
    ):
        frames = [
            frame
            for frame in all_frames
            if frame["symmetry_instance_id"] in instances
        ]
        mesh_prefix = extent_label.lower()
        artifact_files = {
            "source_step": root
            / (
                "wistell_d_45d_61x121_source_cad.step"
                if extent_label == "45D"
                else "wistell_d_90d_symmetry_derived.step"
            ),
            **{
                f"component_step:{name}": root / f"{name}.step"
                for name in raw_manifest["model"]["component_order"]
            },
            "selected_h5m": mesh_files[f"{mesh_prefix}_selected"],
            "refined_h5m": mesh_files[f"{mesh_prefix}_refined"],
            "raw_cad_acceptance": raw_path,
            "m2_mapping": mapping_path,
            "m2_geometry_handshake": m2_path,
            "prompt02_geometry_product_inventory": product_path,
            "prompt02_clearance_receipt": clearance_path,
        }
        qualified = {
            **raw_manifest,
            "generated_utc": utc_now(),
            "local_frames": {
                "status": "PASS",
                "accepted_for_prompt02": True,
                "frame_count": len(frames),
                "selected_patch_scope": (
                    "ALL_CANONICAL_452_NWL_CONTROL_PATCH_ALIASES"
                ),
                "frames": frames,
            },
            "prompt02_producer_gate": {
                "P2G2_local_engineering_frames": "PASS",
                "P2G3_no_overlap_min_clearance_and_seam": "PASS",
                "P2G5_geometry_product_inventory": "PASS",
                "selected_and_refined_h5m_hashes_bound": True,
                "openmc_geometry_debug": "BLOCKED_LOCAL_RUNTIME_UNAVAILABLE",
                "canonical_452_identity_preserved": True,
                "exact_45_to_90_transform_preserved": True,
                "tape_twist_resolved": False,
            },
            "artifacts": artifact_rows(output_root, artifact_files),
        }
        validate_wistell_d_manifest(
            qualified,
            required_extent_degrees=float(
                qualified["model"]["toroidal_extent_degrees"]
            ),
        )
        acceptance_path = (
            output_root / f"PROMPT02_{extent_label}_ACCEPTANCE.json"
        )
        write_json_create_only(acceptance_path, qualified)
        accepted_paths[extent_label] = acceptance_path

    bundle = {
        "schema": "wistell_d.geometry_acceptance_bundle/v1.0.0",
        "generated_utc": utc_now(),
        "classification": "GEOMETRY_PROVIDER_PASS_OPENMC_DEBUG_BLOCKED",
        "geometry_input_mode": WISTELL_D_GEOMETRY_INPUT_MODE,
        "source_hashes": dict(WISTELL_D_SOURCE_HASHES),
        "provider_route": {
            "mode": "READ_ONLY_HASH_VERIFIED",
            "default_extent": "90D",
            "example_fallback_available": False,
        },
        "prompt02_gates": {
            "P2G2": "PASS",
            "P2G3": "PASS",
            "P2G5_geometry_half": "PASS",
            "openmc_geometry_debug": "BLOCKED_LOCAL_RUNTIME_UNAVAILABLE",
        },
        "artifacts": artifact_rows(
            output_root,
            {
                "accepted_45d_manifest": accepted_paths["45D"],
                "accepted_90d_manifest": accepted_paths["90D"],
                "clearance_receipt": clearance_path,
                "m2_geometry_handshake": m2_path,
                "canonical_452_mapping": mapping_path,
                "prompt02_geometry_product_inventory": product_path,
            },
        ),
    }
    write_json_create_only(
        output_root / "GEOMETRY_ACCEPTANCE_BUNDLE.json", bundle
    )
    print(
        json.dumps(
            {
                "classification": bundle["classification"],
                "output": str(output_root),
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
    reject.add_argument(
        "--tolerance-cm3", type=float, default=OVERLAP_TOLERANCE_CM3
    )
    reject.set_defaults(function=audit_existing_31x61)

    build = subparsers.add_parser("build-45")
    build.add_argument("--input-root", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--construct-only", action="store_true")
    build.add_argument(
        "--tolerance-cm3", type=float, default=OVERLAP_TOLERANCE_CM3
    )
    build.set_defaults(function=build_45)

    symmetry = subparsers.add_parser("audit-symmetry")
    symmetry.add_argument("--input-root", required=True)
    symmetry.add_argument("--output", required=True)
    symmetry.set_defaults(function=audit_symmetry)

    expand = subparsers.add_parser("derive-90")
    expand.add_argument("--source-root", required=True)
    expand.add_argument("--transform-receipt", required=True)
    expand.add_argument("--output-root", required=True)
    expand.add_argument(
        "--tolerance-cm3", type=float, default=OVERLAP_TOLERANCE_CM3
    )
    expand.set_defaults(function=derive_90)

    m2 = subparsers.add_parser("build-m2-contract")
    m2.add_argument("--input-root", required=True)
    m2.add_argument("--source-root", required=True)
    m2.add_argument("--derived-root", required=True)
    m2.add_argument("--transform-receipt", required=True)
    m2.add_argument("--output-root", required=True)
    m2.set_defaults(function=build_m2_contract)

    mesh = subparsers.add_parser("mesh-cad")
    mesh.add_argument("--source-root", required=True)
    mesh.add_argument("--output-root", required=True)
    mesh.add_argument(
        "--label", choices=("selected", "refined"), required=True
    )
    mesh.add_argument("--min-mesh-size-cm", type=float, required=True)
    mesh.add_argument("--max-mesh-size-cm", type=float, required=True)
    mesh.add_argument("--algorithm", type=int, default=1)
    mesh.set_defaults(function=mesh_cad_to_dagmc)

    clearance = subparsers.add_parser("audit-prompt02-clearance")
    clearance.add_argument("--source-root", required=True)
    clearance.add_argument("--derived-root", required=True)
    clearance.add_argument("--transform-receipt", required=True)
    clearance.add_argument("--source-selected-mesh-root", required=True)
    clearance.add_argument("--source-refined-mesh-root", required=True)
    clearance.add_argument("--derived-selected-mesh-root", required=True)
    clearance.add_argument("--derived-refined-mesh-root", required=True)
    clearance.add_argument("--output", required=True)
    clearance.add_argument(
        "--minimum-nonadjacent-clearance-cm", type=float, default=1.0e-4
    )
    clearance.add_argument(
        "--interface-gap-tolerance-cm", type=float, default=1.0e-4
    )
    clearance.set_defaults(function=audit_prompt02_clearance)

    qualify = subparsers.add_parser("qualify-prompt02")
    qualify.add_argument("--source-root", required=True)
    qualify.add_argument("--derived-root", required=True)
    qualify.add_argument("--m2-root", required=True)
    qualify.add_argument("--clearance-receipt", required=True)
    qualify.add_argument("--source-selected-mesh-root", required=True)
    qualify.add_argument("--source-refined-mesh-root", required=True)
    qualify.add_argument("--derived-selected-mesh-root", required=True)
    qualify.add_argument("--derived-refined-mesh-root", required=True)
    qualify.add_argument("--output-root", required=True)
    qualify.set_defaults(function=qualify_prompt02_geometry)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
