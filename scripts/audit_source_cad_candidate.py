"""Audit source-CAD solids before a geometry candidate can reach DAGMC."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path, PurePosixPath

import cadquery as cq
import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Pnt

from parastell.reference_geometry import sha256_file
from parastell.source_geometry_identity import canonical_step_payload_sha256
from parastell.source_geometry_identity import source_mesh_fingerprint


COMPONENT_FILES = (
    "chamber.step",
    "first_wall.step",
    "breeder.step",
    "back_wall.step",
    "shield.step",
    "vacuum_vessel.step",
)

REFERENCE_REQUIRED_ARTIFACTS = frozenset(
    {
        "back_wall.step",
        "breeder.step",
        "build.log",
        "chamber.step",
        "coils.example",
        "dagmc.h5m",
        "first_wall.step",
        "magnet_mesh.h5m",
        "magnet_set.step",
        "shield.step",
        "source_mesh.h5m",
        "vacuum_vessel.step",
        "weight_window_mesh.h5m",
        "wout_vmec.nc",
    }
)
CANDIDATE_REQUIRED_ARTIFACTS = frozenset(
    {
        *COMPONENT_FILES,
        "dagmc.h5m",
        "magnet_set.step",
        "source_mesh.h5m",
    }
)

MAXIMUM_WITNESS_DISTANCE_RESIDUAL_CM = 1.0e-8
MAXIMUM_WITNESS_POINT_SHAPE_RESIDUAL_CM = 1.0e-8

_WORKER_COMPONENTS = None
_WORKER_MAGNETS = None
_WORKER_INTERSECTION_TOLERANCE_CM3 = None
_WORKER_DUPLICATE_DISTANCE_CM = None


def _one_solid(path: Path):
    solids = cq.importers.importStep(str(path)).solids().vals()
    if len(solids) != 1:
        raise RuntimeError(f"expected one solid in {path}, got {len(solids)}")
    return solids[0]


def _finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _shape_row(shape, *, identity: str) -> dict:
    box = shape.BoundingBox()
    center = shape.Center()
    shells = shape.Shells()
    shell_closed = [bool(shell.wrapped.Closed()) for shell in shells]
    row = {
        "identity": identity,
        "volume_cm3": float(shape.Volume()),
        "area_cm2": float(shape.Area()),
        "centroid_cm": [float(center.x), float(center.y), float(center.z)],
        "bounding_box_cm": {
            "minimum": [float(box.xmin), float(box.ymin), float(box.zmin)],
            "maximum": [float(box.xmax), float(box.ymax), float(box.zmax)],
        },
        "valid": bool(shape.isValid()),
        "topological_type": shape.ShapeType(),
        # OpenCascade's Closed flag is meaningful on each bounding shell;
        # TopoDS_Solid itself commonly leaves that flag unset even for a
        # valid closed box.  Requiring the Solid flag caused every valid STEP
        # solid in the first A1 audit to false-fail.
        "topods_solid_closed_flag_advisory": bool(shape.wrapped.Closed()),
        "shell_count": len(shells),
        "shell_closed_flags": shell_closed,
        "closed": bool(shells) and all(shell_closed),
        "face_count": len(shape.Faces()),
        "edge_count": len(shape.Edges()),
    }
    numeric_values = (
        [row["volume_cm3"], row["area_cm2"]]
        + row["centroid_cm"]
        + row["bounding_box_cm"]["minimum"]
        + row["bounding_box_cm"]["maximum"]
    )
    row["numeric_values_finite"] = all(
        _finite_number(value) for value in numeric_values
    )
    return row


def _intersection(first, second, *, multithread: bool = True) -> dict:
    first_box = first.BoundingBox()
    second_box = second.BoundingBox()
    separated = any(
        high_first < low_second or high_second < low_first
        for low_first, high_first, low_second, high_second in (
            (first_box.xmin, first_box.xmax, second_box.xmin, second_box.xmax),
            (first_box.ymin, first_box.ymax, second_box.ymin, second_box.ymax),
            (first_box.zmin, first_box.zmax, second_box.zmin, second_box.zmax),
        )
    )
    if separated:
        return {
            "status": "EXACT_AABB_DISJOINT",
            "volume_cm3": 0.0,
            "result_boundary_area_cm2": 0.0,
            "connected_solid_count": 0,
        }
    try:
        operation = BRepAlgoAPI_Common(first.wrapped, second.wrapped)
        operation.SetRunParallel(multithread)
        operation.Build()
    except Exception as error:
        return {
            "status": "ERROR",
            "reason": str(error),
            "volume_cm3": None,
            "result_boundary_area_cm2": None,
            "connected_solid_count": None,
        }
    if not operation.IsDone():
        return {
            "status": "ERROR",
            "reason": "Open CASCADE common operation did not complete",
            "volume_cm3": None,
            "result_boundary_area_cm2": None,
            "connected_solid_count": None,
        }
    result = cq.Shape.cast(operation.Shape())
    if result.isNull():
        return {
            "status": "MEASURED_EMPTY_INTERSECTION",
            "volume_cm3": 0.0,
            "result_boundary_area_cm2": 0.0,
            "connected_solid_count": 0,
        }
    solids = result.Solids()
    measured = {
        "status": "MEASURED",
        "volume_cm3": float(result.Volume()),
        "result_boundary_area_cm2": float(result.Area()),
        "connected_solid_count": len(solids),
    }
    if not all(
        _finite_number(measured[field])
        for field in ("volume_cm3", "result_boundary_area_cm2")
    ):
        return {
            "status": "ERROR",
            "reason": "non-finite Open CASCADE intersection measure",
            "volume_cm3": None,
            "result_boundary_area_cm2": None,
            "connected_solid_count": None,
        }
    return measured


def _distance(
    first,
    second,
    *,
    multithread: bool = True,
    witnesses_required: bool = True,
) -> dict:
    try:
        calculation = BRepExtrema_DistShapeShape(first.wrapped, second.wrapped)
        calculation.SetMultiThread(multithread)
        calculation.Perform()
    except Exception as error:
        return {"status": "ERROR", "reason": str(error)}
    if not calculation.IsDone() or calculation.NbSolution() < 1:
        return {
            "status": "ERROR",
            "reason": "Open CASCADE distance calculation returned no solution",
        }
    reported_distance = float(calculation.Value())
    if not _finite_number(reported_distance):
        return {
            "status": "ERROR",
            "reason": "non-finite Open CASCADE distance",
        }
    if not witnesses_required:
        return {
            "status": "MEASURED",
            "method": "OpenCascade BRepExtrema_DistShapeShape multithreaded",
            "minimum_distance_cm": reported_distance,
            "solution_count": int(calculation.NbSolution()),
            "witnesses_required": False,
        }
    witnesses = []
    for index in range(1, calculation.NbSolution() + 1):
        first_point = calculation.PointOnShape1(index)
        second_point = calculation.PointOnShape2(index)
        first_coordinates = [
            float(first_point.X()),
            float(first_point.Y()),
            float(first_point.Z()),
        ]
        second_coordinates = [
            float(second_point.X()),
            float(second_point.Y()),
            float(second_point.Z()),
        ]
        recomputed_distance = float(
            np.linalg.norm(
                np.asarray(second_coordinates) - np.asarray(first_coordinates)
            )
        )
        reported_distance = float(calculation.Value())
        first_vertex = BRepBuilderAPI_MakeVertex(
            gp_Pnt(*first_coordinates)
        ).Vertex()
        second_vertex = BRepBuilderAPI_MakeVertex(
            gp_Pnt(*second_coordinates)
        ).Vertex()
        first_membership = BRepExtrema_DistShapeShape(
            first.wrapped, first_vertex
        )
        second_membership = BRepExtrema_DistShapeShape(
            second.wrapped, second_vertex
        )
        first_membership.Perform()
        second_membership.Perform()
        first_residual = (
            float(first_membership.Value())
            if first_membership.IsDone() and first_membership.NbSolution() >= 1
            else None
        )
        second_residual = (
            float(second_membership.Value())
            if second_membership.IsDone()
            and second_membership.NbSolution() >= 1
            else None
        )
        witnesses.append(
            {
                "first_point_cm": first_coordinates,
                "second_point_cm": second_coordinates,
                "recomputed_distance_cm": recomputed_distance,
                "reported_distance_residual_cm": abs(
                    recomputed_distance - reported_distance
                ),
                "distance_reconstruction_pass": abs(
                    recomputed_distance - reported_distance
                )
                <= MAXIMUM_WITNESS_DISTANCE_RESIDUAL_CM,
                "first_point_shape_residual_cm": first_residual,
                "second_point_shape_residual_cm": second_residual,
                "point_membership_tolerance_cm": (
                    MAXIMUM_WITNESS_POINT_SHAPE_RESIDUAL_CM
                ),
                "point_membership_pass": all(
                    _finite_number(value)
                    for value in (first_residual, second_residual)
                )
                and max(first_residual, second_residual)
                <= MAXIMUM_WITNESS_POINT_SHAPE_RESIDUAL_CM,
            }
        )
    if not all(
        all(_finite_number(value) for value in witness["first_point_cm"])
        and all(_finite_number(value) for value in witness["second_point_cm"])
        and _finite_number(witness["recomputed_distance_cm"])
        and _finite_number(witness["reported_distance_residual_cm"])
        and _finite_number(witness["first_point_shape_residual_cm"])
        and _finite_number(witness["second_point_shape_residual_cm"])
        for witness in witnesses
    ):
        return {
            "status": "ERROR",
            "reason": "non-finite Open CASCADE distance or witness value",
        }
    return {
        "status": "MEASURED",
        "method": "OpenCascade BRepExtrema_DistShapeShape multithreaded",
        "minimum_distance_cm": reported_distance,
        "solution_count": int(calculation.NbSolution()),
        "witnesses_required": True,
        "maximum_witness_distance_residual_cm": max(
            witness["reported_distance_residual_cm"] for witness in witnesses
        ),
        "witness_distance_residual_tolerance_cm": (
            MAXIMUM_WITNESS_DISTANCE_RESIDUAL_CM
        ),
        "witnesses": witnesses,
    }


def _write_progress(path: Path, stage: str, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": stage, "row": row}) + "\n")


def _verify_manifest_artifacts(
    directory: Path,
    manifest_name: str,
    *,
    required_artifacts: frozenset[str] = frozenset(),
) -> dict:
    manifest_path = directory / manifest_name
    if not manifest_path.is_file():
        return {
            "pass": False,
            "manifest_path": str(manifest_path.resolve()),
            "reason": "manifest missing",
            "artifacts": [],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_artifacts = manifest.get("artifacts", {})
    missing_required = sorted(required_artifacts - set(declared_artifacts))
    rows = []
    for filename, expected in sorted(declared_artifacts.items()):
        path = directory / filename
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = sha256_file(path) if exists else None
        rows.append(
            {
                "filename": filename,
                "exists": exists,
                "expected_size_bytes": expected.get("size_bytes"),
                "actual_size_bytes": actual_size,
                "expected_sha256": expected.get("sha256"),
                "actual_sha256": actual_hash,
                "pass": exists
                and actual_size == expected.get("size_bytes")
                and actual_hash == expected.get("sha256"),
            }
        )
    return {
        "pass": bool(rows)
        and not missing_required
        and all(row["pass"] for row in rows),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "required_artifacts": sorted(required_artifacts),
        "missing_required_artifacts": missing_required,
        "artifacts": rows,
    }


def _shape_physical_comparison(first, second, *, identity: str) -> dict:
    first_row = _shape_row(first, identity=identity)
    second_row = _shape_row(second, identity=identity)
    absolute = {
        "volume_cm3": abs(first_row["volume_cm3"] - second_row["volume_cm3"]),
        "area_cm2": abs(first_row["area_cm2"] - second_row["area_cm2"]),
        "centroid_cm": float(
            np.max(
                np.abs(
                    np.asarray(first_row["centroid_cm"])
                    - np.asarray(second_row["centroid_cm"])
                )
            )
        ),
        "bounding_box_cm": float(
            np.max(
                np.abs(
                    np.asarray(
                        first_row["bounding_box_cm"]["minimum"]
                        + first_row["bounding_box_cm"]["maximum"]
                    )
                    - np.asarray(
                        second_row["bounding_box_cm"]["minimum"]
                        + second_row["bounding_box_cm"]["maximum"]
                    )
                )
            )
        ),
    }
    scale = max(first_row["volume_cm3"], second_row["volume_cm3"], 1.0)
    physical_equal = all(
        (
            absolute["volume_cm3"] <= 1.0e-9 * scale,
            absolute["area_cm2"]
            <= 1.0e-9
            * max(first_row["area_cm2"], second_row["area_cm2"], 1.0),
            absolute["centroid_cm"] <= 1.0e-8,
            absolute["bounding_box_cm"] <= 1.0e-8,
            first_row["face_count"] == second_row["face_count"],
            first_row["edge_count"] == second_row["edge_count"],
        )
    )
    return {
        "identity": identity,
        "absolute_differences": absolute,
        "topology_equal": first_row["face_count"] == second_row["face_count"]
        and first_row["edge_count"] == second_row["edge_count"],
        "physical_equal_within_tolerance": physical_equal,
    }


def _clearance_summary(vessel_rows: list[dict], requirement_cm: float) -> dict:
    expected_magnet_ids = {f"magnet-{index:04d}" for index in range(18)}
    actual_magnet_ids = [row.get("magnet_id") for row in vessel_rows]
    identity_pass = (
        len(actual_magnet_ids) == 18
        and len(set(actual_magnet_ids)) == 18
        and set(actual_magnet_ids) == expected_magnet_ids
    )
    measured = [
        row["distance"]["minimum_distance_cm"]
        for row in vessel_rows
        if row["distance"]["status"] == "MEASURED"
        and _finite_number(row["distance"].get("minimum_distance_cm"))
    ]
    error_count = sum(
        row["distance"]["status"] != "MEASURED" for row in vessel_rows
    )
    witness_failure_count = 0
    for row in vessel_rows:
        distance = row["distance"]
        witnesses = distance.get("witnesses", [])
        valid_witnesses = bool(witnesses) and all(
            len(witness.get("first_point_cm", [])) == 3
            and len(witness.get("second_point_cm", [])) == 3
            and np.all(np.isfinite(witness["first_point_cm"]))
            and np.all(np.isfinite(witness["second_point_cm"]))
            and np.isfinite(witness.get("recomputed_distance_cm", np.nan))
            and witness.get("distance_reconstruction_pass") is True
            and witness.get("point_membership_pass") is True
            and _finite_number(witness.get("reported_distance_residual_cm"))
            and witness.get("reported_distance_residual_cm", float("inf"))
            <= MAXIMUM_WITNESS_DISTANCE_RESIDUAL_CM
            and _finite_number(witness.get("first_point_shape_residual_cm"))
            and witness.get("first_point_shape_residual_cm", float("inf"))
            <= MAXIMUM_WITNESS_POINT_SHAPE_RESIDUAL_CM
            and _finite_number(witness.get("second_point_shape_residual_cm"))
            and witness.get("second_point_shape_residual_cm", float("inf"))
            <= MAXIMUM_WITNESS_POINT_SHAPE_RESIDUAL_CM
            for witness in witnesses
        )
        if (
            distance.get("status") != "MEASURED"
            or distance.get("solution_count") != len(witnesses)
            or not valid_witnesses
        ):
            witness_failure_count += 1
    minimum = min(measured) if measured else None
    return {
        "minimum_vessel_to_magnet_clearance_cm": minimum,
        "distance_measurement_count": len(measured),
        "distance_error_count": error_count,
        "distance_witness_failure_count": witness_failure_count,
        "expected_magnet_ids": sorted(expected_magnet_ids),
        "actual_magnet_ids": actual_magnet_ids,
        "magnet_identity_set_pass": identity_pass,
        "clearance_pass": len(vessel_rows) == 18
        and identity_pass
        and len(measured) == 18
        and error_count == 0
        and witness_failure_count == 0
        and minimum is not None
        and minimum >= requirement_cm,
    }


def _pair_identity_contract(
    magnet_component: list[dict],
    component_pairs: list[dict],
    magnet_pairs: list[dict],
) -> dict:
    component_names = [name.removesuffix(".step") for name in COMPONENT_FILES]
    magnet_ids = [f"magnet-{index:04d}" for index in range(18)]
    expected_magnet_component = {
        (component, magnet)
        for component in component_names
        for magnet in magnet_ids
    }
    actual_magnet_component = [
        (row.get("component"), row.get("magnet_id"))
        for row in magnet_component
    ]
    expected_component_pairs = {
        tuple(sorted(pair))
        for pair in itertools.combinations(component_names, 2)
    }
    actual_component_pairs = [
        tuple(sorted(row.get("components", []))) for row in component_pairs
    ]
    expected_magnet_pairs = {
        tuple(pair) for pair in itertools.combinations(magnet_ids, 2)
    }
    actual_magnet_pairs = [
        tuple(row.get("magnets", [])) for row in magnet_pairs
    ]

    def contract(actual: list[tuple], expected: set[tuple]) -> dict:
        return {
            "expected_count": len(expected),
            "actual_count": len(actual),
            "unique_actual_count": len(set(actual)),
            "missing": [
                list(value) for value in sorted(expected - set(actual))
            ],
            "unexpected": [
                list(value) for value in sorted(set(actual) - expected)
            ],
            "pass": len(actual) == len(expected)
            and len(set(actual)) == len(expected)
            and set(actual) == expected,
        }

    result = {
        "magnet_component": contract(
            actual_magnet_component, expected_magnet_component
        ),
        "component_pair": contract(
            actual_component_pairs, expected_component_pairs
        ),
        "magnet_pair": contract(actual_magnet_pairs, expected_magnet_pairs),
    }
    result["pass"] = all(row["pass"] for row in result.values())
    return result


def _remap_receipt_artifact_path(
    receipt_path: str,
    *,
    receipt_artifact_root: str | None,
    actual_artifact_root: Path | None,
) -> Path:
    if (receipt_artifact_root is None) != (actual_artifact_root is None):
        raise ValueError(
            "receipt and actual artifact roots must be provided together"
        )
    if receipt_artifact_root is None:
        return Path(receipt_path)
    logical = PurePosixPath(receipt_path)
    logical_root = PurePosixPath(receipt_artifact_root)
    if not logical.is_absolute() or not logical_root.is_absolute():
        raise ValueError("receipt artifact paths and roots must be absolute")
    try:
        relative = logical.relative_to(logical_root)
    except ValueError as error:
        raise ValueError(
            f"receipt path is outside the declared artifact root: {receipt_path}"
        ) from error
    if ".." in relative.parts:
        raise ValueError(
            f"receipt path contains parent traversal: {receipt_path}"
        )
    resolved_root = actual_artifact_root.resolve()
    resolved_target = resolved_root.joinpath(*relative.parts).resolve()
    if not resolved_target.is_relative_to(resolved_root):
        raise ValueError(
            f"remapped receipt path escapes the actual artifact root: {receipt_path}"
        )
    return resolved_target


def _physical_support_evidence(
    physical_change: dict,
    *,
    receipt_artifact_root: str | None = None,
    actual_artifact_root: Path | None = None,
) -> dict:
    rows = []
    measurement = physical_change.get("measurement", {})
    measurement_receipt_path = str(measurement.get("path", ""))
    measurement_path = _remap_receipt_artifact_path(
        measurement_receipt_path,
        receipt_artifact_root=receipt_artifact_root,
        actual_artifact_root=actual_artifact_root,
    )
    measurement_exists = measurement_path.is_file()
    measurement_hash = (
        sha256_file(measurement_path) if measurement_exists else None
    )
    rows.append(
        {
            "role": "clearance_measurement",
            "receipt_path": measurement_receipt_path,
            "actual_path": str(measurement_path.resolve()),
            "exists": measurement_exists,
            "expected_sha256": measurement.get("sha256"),
            "actual_sha256": measurement_hash,
            "pass": measurement_exists
            and measurement_hash == measurement.get("sha256"),
        }
    )
    report_path = Path(str(physical_change.get("_receipt_path", "")))
    for filename, expected in sorted(
        physical_change.get("generated_artifacts", {}).items()
    ):
        path = report_path.parent / filename
        exists = path.is_file()
        actual_hash = sha256_file(path) if exists else None
        actual_size = path.stat().st_size if exists else None
        rows.append(
            {
                "role": "physical_change_support",
                "path": str(path),
                "exists": exists,
                "expected_sha256": expected.get("sha256"),
                "actual_sha256": actual_hash,
                "expected_size_bytes": expected.get("size_bytes"),
                "actual_size_bytes": actual_size,
                "pass": exists
                and actual_hash == expected.get("sha256")
                and actual_size == expected.get("size_bytes"),
            }
        )
    return {
        "artifacts": rows,
        "pass": bool(rows) and all(row["pass"] for row in rows),
    }


def _configure_pair_worker(
    candidate_dir: str,
    intersection_tolerance_cm3: float,
    duplicate_shape_distance_cm: float,
) -> None:
    """Load an independent read-only CAD copy inside one worker process."""

    global _WORKER_COMPONENTS
    global _WORKER_MAGNETS
    global _WORKER_INTERSECTION_TOLERANCE_CM3
    global _WORKER_DUPLICATE_DISTANCE_CM
    directory = Path(candidate_dir)
    _WORKER_COMPONENTS = {
        filename.removesuffix(".step"): _one_solid(directory / filename)
        for filename in COMPONENT_FILES
    }
    _WORKER_MAGNETS = (
        cq.importers.importStep(str(directory / "magnet_set.step"))
        .solids()
        .vals()
    )
    if len(_WORKER_MAGNETS) != 18:
        raise RuntimeError(
            f"expected 18 worker magnet solids, got {len(_WORKER_MAGNETS)}"
        )
    _WORKER_INTERSECTION_TOLERANCE_CM3 = intersection_tolerance_cm3
    _WORKER_DUPLICATE_DISTANCE_CM = duplicate_shape_distance_cm


def _worker_magnet_component(task: tuple[str, int]) -> dict:
    component_name, index = task
    component = _WORKER_COMPONENTS[component_name]
    magnet = _WORKER_MAGNETS[index]
    separation = _distance(
        component,
        magnet,
        multithread=True,
        witnesses_required=component_name == "vacuum_vessel",
    )
    if (
        separation.get("status") == "MEASURED"
        and separation["minimum_distance_cm"] > _WORKER_DUPLICATE_DISTANCE_CM
    ):
        intersection = {
            "status": "EXACT_BREP_DISTANCE_DISJOINT",
            "volume_cm3": 0.0,
            "result_boundary_area_cm2": 0.0,
            "connected_solid_count": 0,
            "exact_minimum_distance_cm": separation["minimum_distance_cm"],
        }
    else:
        intersection = _intersection(component, magnet, multithread=True)
    row = {
        "component": component_name,
        "magnet_id": f"magnet-{index:04d}",
        **intersection,
        "separation_distance": separation,
        "distance": separation,
    }
    row["intersection_pass"] = (
        intersection["status"] != "ERROR"
        and _finite_number(intersection.get("volume_cm3"))
        and intersection["volume_cm3"] <= _WORKER_INTERSECTION_TOLERANCE_CM3
    )
    return row


def _worker_component_pair(task: tuple[str, str]) -> dict:
    first_name, second_name = task
    intersection = _intersection(
        _WORKER_COMPONENTS[first_name],
        _WORKER_COMPONENTS[second_name],
        multithread=True,
    )
    row = {"components": [first_name, second_name], **intersection}
    row["intersection_pass"] = (
        intersection["status"] != "ERROR"
        and _finite_number(intersection.get("volume_cm3"))
        and intersection["volume_cm3"] <= _WORKER_INTERSECTION_TOLERANCE_CM3
    )
    return row


def _worker_magnet_pair(task: tuple[int, int]) -> dict:
    first_index, second_index = task
    first = _WORKER_MAGNETS[first_index]
    second = _WORKER_MAGNETS[second_index]
    intersection = _intersection(first, second, multithread=True)
    first_center = first.Center()
    second_center = second.Center()
    first_box = first.BoundingBox()
    second_box = second.BoundingBox()
    location_delta = max(
        abs(first_center.x - second_center.x),
        abs(first_center.y - second_center.y),
        abs(first_center.z - second_center.z),
        abs(first_box.xmin - second_box.xmin),
        abs(first_box.xmax - second_box.xmax),
        abs(first_box.ymin - second_box.ymin),
        abs(first_box.ymax - second_box.ymax),
        abs(first_box.zmin - second_box.zmin),
        abs(first_box.zmax - second_box.zmax),
    )
    same_mass_properties = abs(
        first.Volume() - second.Volume()
    ) <= 1.0e-9 * max(first.Volume(), second.Volume(), 1.0) and abs(
        first.Area() - second.Area()
    ) <= 1.0e-9 * max(
        first.Area(), second.Area(), 1.0
    )
    duplicate_distance = (
        _distance(first, second, multithread=True)
        if same_mass_properties
        and location_delta <= _WORKER_DUPLICATE_DISTANCE_CM
        else {
            "status": "NOT_REQUESTED",
            "reason": "mass properties or global location differ",
        }
    )
    duplicate_shape = (
        same_mass_properties
        and location_delta <= _WORKER_DUPLICATE_DISTANCE_CM
        and duplicate_distance.get("status") == "MEASURED"
        and duplicate_distance.get("minimum_distance_cm", float("inf"))
        <= _WORKER_DUPLICATE_DISTANCE_CM
    )
    row = {
        "magnets": [
            f"magnet-{first_index:04d}",
            f"magnet-{second_index:04d}",
        ],
        **intersection,
        "duplicate_shape_check": {
            "threshold_cm": _WORKER_DUPLICATE_DISTANCE_CM,
            "same_mass_properties": same_mass_properties,
            "maximum_location_delta_cm": float(location_delta),
            "distance": duplicate_distance,
            "duplicate_shape": duplicate_shape,
            "exact_common_volume_cm3": intersection.get("volume_cm3"),
            "proof_basis": (
                "a coincident or sub-micrometre-displaced duplicate positive-"
                "volume solid necessarily has common volume above the frozen "
                "intersection tolerance; the exact common-volume gate is "
                "therefore controlling, with distance and mass properties "
                "retained as supplementary evidence"
            ),
            "pass": not duplicate_shape,
        },
    }
    row["intersection_pass"] = (
        intersection["status"] != "ERROR"
        and _finite_number(intersection.get("volume_cm3"))
        and intersection["volume_cm3"] <= _WORKER_INTERSECTION_TOLERANCE_CM3
    )
    row["duplicate_shape_check"]["pass"] = bool(
        row["duplicate_shape_check"]["pass"] and row["intersection_pass"]
    )
    return row


def audit(
    candidate_dir: Path,
    output_dir: Path,
    *,
    reference_dir: Path,
    acceptance_criteria_path: Path,
    expected_acceptance_criteria_sha256: str,
    physical_change_report_path: Path,
    expected_physical_change_report_sha256: str,
    expected_reference_manifest_sha256: str,
    workers: int,
    physical_receipt_candidate_dir: Path | None = None,
    physical_receipt_reference_dir: Path | None = None,
    receipt_artifact_root: str | None = None,
    actual_artifact_root: Path | None = None,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"create-only output already exists: {output_dir}"
        )
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    script_sha256_before = sha256_file(Path(__file__))
    identity_module_path = (
        Path(__file__).resolve().parents[1]
        / "parastell"
        / "source_geometry_identity.py"
    )
    identity_module_sha256_before = sha256_file(identity_module_path)
    criteria_sha256 = sha256_file(acceptance_criteria_path)
    if criteria_sha256 != expected_acceptance_criteria_sha256:
        raise ValueError("acceptance-criteria hash mismatch")
    physical_change_sha256 = sha256_file(physical_change_report_path)
    if physical_change_sha256 != expected_physical_change_report_sha256:
        raise ValueError("physical-change receipt hash mismatch")
    criteria = json.loads(acceptance_criteria_path.read_text(encoding="utf-8"))
    source_criteria = criteria["source_cad"]
    intersection_tolerance_cm3 = float(
        source_criteria["maximum_unintended_intersection_volume_cm3"]
    )
    clearance_requirement_cm = float(
        criteria["separated_interface"]["minimum_accepted_clearance_cm"]
    )
    minimum_positive_volume_cm3 = float(
        source_criteria["minimum_positive_solid_volume_cm3"]
    )
    duplicate_shape_distance_cm = float(
        source_criteria["maximum_duplicate_shape_distance_cm"]
    )
    physical_change = json.loads(
        physical_change_report_path.read_text(encoding="utf-8")
    )
    physical_change["_receipt_path"] = str(
        physical_change_report_path.resolve()
    )
    output_dir.mkdir(parents=True)
    progress_path = output_dir / "source_cad_progress.jsonl"
    reference_integrity = _verify_manifest_artifacts(
        reference_dir,
        "VANILLA_BUILD_MANIFEST.json",
        required_artifacts=REFERENCE_REQUIRED_ARTIFACTS,
    )
    candidate_integrity = _verify_manifest_artifacts(
        candidate_dir,
        "candidate_build_manifest.json",
        required_artifacts=CANDIDATE_REQUIRED_ARTIFACTS,
    )
    candidate_manifest = candidate_integrity.get("manifest", {})
    candidate_manifest_sha256 = candidate_integrity.get("manifest_sha256")
    logical_candidate_dir = (
        physical_receipt_candidate_dir or candidate_dir
    ).resolve()
    logical_reference_dir = (
        physical_receipt_reference_dir or reference_dir
    ).resolve()
    physical_support_before = _physical_support_evidence(
        physical_change,
        receipt_artifact_root=receipt_artifact_root,
        actual_artifact_root=actual_artifact_root,
    )
    physical_change_binding = {
        "path": str(physical_change_report_path.resolve()),
        "sha256": physical_change_sha256,
        "expected_sha256": expected_physical_change_report_sha256,
        "hash_matches_expected": physical_change_sha256
        == expected_physical_change_report_sha256,
        "physical_change_gate_pass": physical_change.get(
            "physical_change_gate_pass"
        )
        is True,
        "candidate_dir_matches": physical_change.get("candidate_dir")
        == str(logical_candidate_dir),
        "candidate_actual_dir": str(candidate_dir.resolve()),
        "candidate_receipt_logical_dir": str(logical_candidate_dir),
        "candidate_path_remapped": candidate_dir.resolve()
        != logical_candidate_dir,
        "candidate_manifest_matches": physical_change.get(
            "candidate_integrity", {}
        ).get("manifest_sha256")
        == candidate_manifest_sha256,
        "reference_dir_matches": physical_change.get("reference_dir")
        == str(logical_reference_dir),
        "reference_actual_dir": str(reference_dir.resolve()),
        "reference_receipt_logical_dir": str(logical_reference_dir),
        "reference_path_remapped": reference_dir.resolve()
        != logical_reference_dir,
        "reference_manifest_matches_expected": reference_integrity.get(
            "manifest_sha256"
        )
        == expected_reference_manifest_sha256,
        "physical_receipt_reference_manifest_matches": physical_change.get(
            "reference_integrity", {}
        ).get("manifest_sha256")
        == expected_reference_manifest_sha256,
        "acceptance_criteria_matches": physical_change.get(
            "acceptance_criteria", {}
        ).get("sha256")
        == criteria_sha256,
    }
    physical_change_binding["pass"] = (
        all(
            value
            for key, value in physical_change_binding.items()
            if key
            in {
                "physical_change_gate_pass",
                "candidate_dir_matches",
                "candidate_manifest_matches",
                "reference_dir_matches",
                "reference_manifest_matches_expected",
                "physical_receipt_reference_manifest_matches",
                "acceptance_criteria_matches",
                "hash_matches_expected",
            }
        )
        and physical_support_before["pass"]
    )

    components = {
        filename.removesuffix(".step"): _one_solid(candidate_dir / filename)
        for filename in COMPONENT_FILES
    }
    magnet_path = candidate_dir / "magnet_set.step"
    magnets = cq.importers.importStep(str(magnet_path)).solids().vals()
    if len(magnets) != 18:
        raise RuntimeError(f"expected 18 magnet solids, got {len(magnets)}")

    component_inventory = []
    for name, solid in components.items():
        row = _shape_row(solid, identity=name)
        row.update(
            {
                "path": str((candidate_dir / f"{name}.step").resolve()),
                "sha256": sha256_file(candidate_dir / f"{name}.step"),
            }
        )
        component_inventory.append(row)
    magnet_inventory = [
        _shape_row(solid, identity=f"magnet-{index:04d}")
        for index, solid in enumerate(magnets)
    ]
    reference_magnets = (
        cq.importers.importStep(str(reference_dir / "magnet_set.step"))
        .solids()
        .vals()
    )
    if len(reference_magnets) != 18:
        raise RuntimeError(
            f"expected 18 reference magnet solids, got {len(reference_magnets)}"
        )
    magnet_physical_comparison = [
        _shape_physical_comparison(
            reference_magnets[index],
            magnet,
            identity=f"magnet-{index:04d}",
        )
        for index, magnet in enumerate(magnets)
    ]

    magnet_component_tasks = [
        (component_name, index)
        for component_name in components
        for index in range(len(magnets))
    ]
    component_pair_tasks = list(itertools.combinations(components, 2))
    magnet_pair_tasks = list(itertools.combinations(range(18), 2))

    def collect_rows(mapper, function, tasks, stage):
        rows = []
        for row in mapper(function, tasks):
            rows.append(row)
            _write_progress(progress_path, stage, row)
        return rows

    if workers == 1:
        global _WORKER_COMPONENTS
        global _WORKER_MAGNETS
        global _WORKER_INTERSECTION_TOLERANCE_CM3
        global _WORKER_DUPLICATE_DISTANCE_CM
        _WORKER_COMPONENTS = components
        _WORKER_MAGNETS = magnets
        _WORKER_INTERSECTION_TOLERANCE_CM3 = intersection_tolerance_cm3
        _WORKER_DUPLICATE_DISTANCE_CM = duplicate_shape_distance_cm
        mapper = map
        magnet_component = collect_rows(
            mapper,
            _worker_magnet_component,
            magnet_component_tasks,
            "magnet_component",
        )
        component_pairs = collect_rows(
            mapper,
            _worker_component_pair,
            component_pair_tasks,
            "component_pair",
        )
        magnet_pairs = collect_rows(
            mapper, _worker_magnet_pair, magnet_pair_tasks, "magnet_pair"
        )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_configure_pair_worker,
            initargs=(
                str(candidate_dir.resolve()),
                intersection_tolerance_cm3,
                duplicate_shape_distance_cm,
            ),
        ) as executor:
            magnet_component = collect_rows(
                executor.map,
                _worker_magnet_component,
                magnet_component_tasks,
                "magnet_component",
            )
            component_pairs = collect_rows(
                executor.map,
                _worker_component_pair,
                component_pair_tasks,
                "component_pair",
            )
            magnet_pairs = collect_rows(
                executor.map,
                _worker_magnet_pair,
                magnet_pair_tasks,
                "magnet_pair",
            )

    vessel_rows = [
        row for row in magnet_component if row["component"] == "vacuum_vessel"
    ]
    clearance_summary = _clearance_summary(
        vessel_rows, clearance_requirement_cm
    )
    pair_identity_contract = _pair_identity_contract(
        magnet_component, component_pairs, magnet_pairs
    )

    candidate_source = source_mesh_fingerprint(
        candidate_dir / "source_mesh.h5m"
    )
    reference_source = source_mesh_fingerprint(
        reference_dir / "source_mesh.h5m"
    )
    reference_comparison = {
        "magnet_step_sha256": sha256_file(reference_dir / "magnet_set.step"),
        "magnet_step_byte_identical": sha256_file(magnet_path)
        == sha256_file(reference_dir / "magnet_set.step"),
        "magnet_step_canonical_sha256": canonical_step_payload_sha256(
            reference_dir / "magnet_set.step"
        ),
        "candidate_magnet_step_canonical_sha256": (
            canonical_step_payload_sha256(magnet_path)
        ),
        "magnet_step_canonical_identical": (
            canonical_step_payload_sha256(magnet_path)
            == canonical_step_payload_sha256(reference_dir / "magnet_set.step")
        ),
        "source_mesh_sha256": sha256_file(reference_dir / "source_mesh.h5m"),
        "source_mesh_byte_identical": sha256_file(
            candidate_dir / "source_mesh.h5m"
        )
        == sha256_file(reference_dir / "source_mesh.h5m"),
        "candidate_source_mesh_identity": candidate_source,
        "reference_source_mesh_identity": reference_source,
        "source_mesh_canonical_identical": (
            candidate_source["canonical_fingerprint"]
            == reference_source["canonical_fingerprint"]
        ),
        "magnet_physical_comparison": magnet_physical_comparison,
        "all_magnets_physically_equal": all(
            row["physical_equal_within_tolerance"]
            for row in magnet_physical_comparison
        ),
    }

    reference_integrity_after = _verify_manifest_artifacts(
        reference_dir,
        "VANILLA_BUILD_MANIFEST.json",
        required_artifacts=REFERENCE_REQUIRED_ARTIFACTS,
    )
    candidate_integrity_after = _verify_manifest_artifacts(
        candidate_dir,
        "candidate_build_manifest.json",
        required_artifacts=CANDIDATE_REQUIRED_ARTIFACTS,
    )
    physical_support_after = _physical_support_evidence(
        physical_change,
        receipt_artifact_root=receipt_artifact_root,
        actual_artifact_root=actual_artifact_root,
    )
    postflight = {
        "acceptance_criteria_sha256_after": sha256_file(
            acceptance_criteria_path
        ),
        "physical_change_report_sha256_after": sha256_file(
            physical_change_report_path
        ),
        "script_sha256_after": sha256_file(Path(__file__)),
        "source_geometry_identity_sha256_after": sha256_file(
            identity_module_path
        ),
        "reference_integrity_after": reference_integrity_after,
        "candidate_integrity_after": candidate_integrity_after,
        "physical_support_after": physical_support_after,
    }
    postflight["input_immutability_pass"] = bool(
        postflight["acceptance_criteria_sha256_after"] == criteria_sha256
        and postflight["physical_change_report_sha256_after"]
        == physical_change_sha256
        and postflight["script_sha256_after"] == script_sha256_before
        and postflight["source_geometry_identity_sha256_after"]
        == identity_module_sha256_before
        and reference_integrity_after == reference_integrity
        and candidate_integrity_after == candidate_integrity
        and physical_support_after == physical_support_before
    )

    report = {
        "schema": "parastell.source_cad_candidate_audit/v1.4.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": str(candidate_dir.resolve()),
        "candidate_h5m_sha256": sha256_file(candidate_dir / "dagmc.h5m"),
        "magnet_step_sha256": sha256_file(magnet_path),
        "source_mesh_sha256": sha256_file(candidate_dir / "source_mesh.h5m"),
        "criteria": {
            "maximum_unintended_intersection_volume_cm3": intersection_tolerance_cm3,
            "minimum_vessel_to_magnet_clearance_cm": clearance_requirement_cm,
            "minimum_positive_solid_volume_cm3": minimum_positive_volume_cm3,
            "maximum_duplicate_shape_distance_cm": duplicate_shape_distance_cm,
            "acceptance_criteria_path": str(
                acceptance_criteria_path.resolve()
            ),
            "acceptance_criteria_sha256": criteria_sha256,
            "workers": workers,
            "parallelization": (
                "independent OpenCascade operations in bounded worker processes; "
                "each worker loads an immutable CAD copy; the container CPU cap "
                "bounds aggregate internal OpenCascade execution"
            ),
        },
        "reference_integrity": reference_integrity,
        "candidate_integrity": candidate_integrity,
        "physical_change_binding": physical_change_binding,
        "artifact_path_relocation": {
            "receipt_artifact_root": receipt_artifact_root,
            "actual_artifact_root": (
                str(actual_artifact_root.resolve())
                if actual_artifact_root is not None
                else None
            ),
            "active": receipt_artifact_root is not None,
        },
        "physical_change_support_evidence": physical_support_before,
        "pair_identity_contract": pair_identity_contract,
        "postflight": postflight,
        "criteria_contract": {
            "global_distance_witnesses_required": source_criteria[
                "global_distance_witnesses_required_for_each_magnet"
            ]
            is True,
            "distance_errors_allowed_is_zero": source_criteria[
                "distance_errors_allowed"
            ]
            == 0,
            "source_mesh_semantic_identity_required": source_criteria[
                "source_mesh_semantic_identity_required"
            ]
            is True,
            "magnet_step_canonical_identity_required": source_criteria[
                "magnet_step_canonical_identity_required"
            ]
            is True,
        },
        "audit_implementation": {
            "script_sha256_before": script_sha256_before,
            "script_sha256_after": postflight["script_sha256_after"],
            "source_geometry_identity_sha256_before": (
                identity_module_sha256_before
            ),
            "source_geometry_identity_sha256_after": postflight[
                "source_geometry_identity_sha256_after"
            ],
        },
        "component_inventory": component_inventory,
        "magnet_inventory": magnet_inventory,
        "magnet_component_pairs": magnet_component,
        "component_pairs": component_pairs,
        "magnet_pairs": magnet_pairs,
        "summary": {
            "component_count": len(components),
            "magnet_count": len(magnets),
            "invalid_component_count": sum(
                not row["numeric_values_finite"]
                or row["volume_cm3"] <= minimum_positive_volume_cm3
                or not row["valid"]
                or not row["closed"]
                for row in component_inventory
            ),
            "invalid_magnet_count": sum(
                not row["numeric_values_finite"]
                or row["volume_cm3"] <= minimum_positive_volume_cm3
                or not row["valid"]
                or not row["closed"]
                for row in magnet_inventory
            ),
            "magnet_component_intersection_failures": sum(
                not row["intersection_pass"] for row in magnet_component
            ),
            "component_pair_intersection_failures": sum(
                not row["intersection_pass"] for row in component_pairs
            ),
            "magnet_pair_intersection_failures": sum(
                not row["intersection_pass"] for row in magnet_pairs
            ),
            "duplicate_magnet_shape_failures": sum(
                not row["duplicate_shape_check"]["pass"]
                for row in magnet_pairs
            ),
            **clearance_summary,
        },
        "reference_comparison": reference_comparison,
    }
    summary = report["summary"]
    postflight["final_acceptance_criteria_sha256"] = sha256_file(
        acceptance_criteria_path
    )
    postflight["final_physical_change_report_sha256"] = sha256_file(
        physical_change_report_path
    )
    postflight["final_script_sha256"] = sha256_file(Path(__file__))
    postflight["final_source_geometry_identity_sha256"] = sha256_file(
        identity_module_path
    )
    postflight["input_immutability_pass"] = bool(
        postflight["input_immutability_pass"]
        and postflight["final_acceptance_criteria_sha256"] == criteria_sha256
        and postflight["final_physical_change_report_sha256"]
        == physical_change_sha256
        and postflight["final_script_sha256"] == script_sha256_before
        and postflight["final_source_geometry_identity_sha256"]
        == identity_module_sha256_before
    )
    report["source_cad_physical_gate_pass"] = all(
        (
            summary["invalid_component_count"] == 0,
            summary["invalid_magnet_count"] == 0,
            summary["magnet_component_intersection_failures"] == 0,
            summary["component_pair_intersection_failures"] == 0,
            summary["magnet_pair_intersection_failures"] == 0,
            summary["duplicate_magnet_shape_failures"] == 0,
            summary["clearance_pass"],
            reference_comparison["magnet_step_canonical_identical"],
            reference_comparison["all_magnets_physically_equal"],
            reference_comparison["source_mesh_canonical_identical"],
        )
    )
    report["source_cad_gate_pass"] = all(
        (
            report["source_cad_physical_gate_pass"],
            reference_integrity["pass"],
            candidate_integrity["pass"],
            physical_change_binding["pass"],
            physical_support_before["pass"],
            pair_identity_contract["pass"],
            postflight["input_immutability_pass"],
            candidate_manifest.get("acceptance_criteria", {}).get("sha256")
            == criteria_sha256,
            candidate_manifest.get("construct_only") is False,
            candidate_manifest.get("status")
            == "BUILD_COMPLETE_PENDING_QUALIFICATION",
            all(report["criteria_contract"].values()),
        )
    )
    serialized = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary = output_dir / ".source_cad_audit.json.tmp"
    final = output_dir / "source_cad_audit.json"
    temporary.write_text(serialized, encoding="utf-8")
    if json.loads(temporary.read_text(encoding="utf-8")) != report:
        raise RuntimeError("source-CAD receipt readback mismatch")
    temporary.replace(final)
    report_sha256 = sha256_file(final)
    seal = {
        "schema": "parastell.source_cad_candidate_audit_seal/v1.0.0",
        "source_cad_audit_sha256": report_sha256,
        "audit_script_sha256": script_sha256_before,
        "input_immutability_pass": postflight["input_immutability_pass"],
        "source_cad_gate_pass": report["source_cad_gate_pass"],
    }
    seal_text = (
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    seal_temporary = output_dir / ".source_cad_audit_seal.json.tmp"
    seal_temporary.write_text(seal_text, encoding="utf-8")
    if json.loads(seal_temporary.read_text(encoding="utf-8")) != seal:
        raise RuntimeError("source-CAD seal readback mismatch")
    seal_temporary.replace(output_dir / "source_cad_audit_seal.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--acceptance-criteria", type=Path, required=True)
    parser.add_argument("--expected-acceptance-criteria-sha256", required=True)
    parser.add_argument("--physical-change-report", type=Path, required=True)
    parser.add_argument(
        "--expected-physical-change-report-sha256", required=True
    )
    parser.add_argument("--expected-reference-manifest-sha256", required=True)
    parser.add_argument(
        "--physical-receipt-candidate-dir",
        type=Path,
        help=(
            "logical candidate path recorded by the hash-bound physical-change "
            "receipt; use only when immutable inputs are relocated to another host"
        ),
    )
    parser.add_argument(
        "--physical-receipt-reference-dir",
        type=Path,
        help=(
            "logical reference path recorded by the hash-bound physical-change "
            "receipt; use only when immutable inputs are relocated to another host"
        ),
    )
    parser.add_argument(
        "--receipt-artifact-root",
        help="logical artifact-root prefix stored by the physical receipt",
    )
    parser.add_argument(
        "--actual-artifact-root",
        type=Path,
        help="read-only artifact root containing the relocated receipt inputs",
    )
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    audit(
        arguments.candidate_dir,
        arguments.output_dir,
        reference_dir=arguments.reference_dir,
        acceptance_criteria_path=arguments.acceptance_criteria,
        expected_acceptance_criteria_sha256=(
            arguments.expected_acceptance_criteria_sha256
        ),
        physical_change_report_path=arguments.physical_change_report,
        expected_physical_change_report_sha256=(
            arguments.expected_physical_change_report_sha256
        ),
        expected_reference_manifest_sha256=(
            arguments.expected_reference_manifest_sha256
        ),
        physical_receipt_candidate_dir=(
            arguments.physical_receipt_candidate_dir
        ),
        physical_receipt_reference_dir=(
            arguments.physical_receipt_reference_dir
        ),
        receipt_artifact_root=arguments.receipt_artifact_root,
        actual_artifact_root=arguments.actual_artifact_root,
        workers=arguments.workers,
    )


if __name__ == "__main__":
    main()
