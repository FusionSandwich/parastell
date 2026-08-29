"""Create-only casing/winding CAD for the continuous-global/local-coil lane."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .coil_filament_inventory import coordinate_identity, sha256_file
from .continuous_local_magnet_case import PLAN_SCHEMA

MANIFEST_SCHEMA = "parastell.continuous_local_magnet_geometry/v1.0.0"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dependencies():
    import cadquery as cq

    from .magnet_coils import MagnetSetFromFilaments

    return cq, MagnetSetFromFilaments


def _vector(value: Any) -> list[float]:
    result = [float(item) for item in value.toTuple()]
    if len(result) != 3 or any(not math.isfinite(item) for item in result):
        raise RuntimeError("CAD vector is invalid")
    return result


def _metrics(shape: Any, label: str) -> dict[str, Any]:
    bounds = shape.BoundingBox()
    result = {
        "valid": bool(shape.isValid()),
        "volume_cm3": float(shape.Volume()),
        "centroid_cm": _vector(shape.Center()),
        "bounds_cm": [
            float(bounds.xmin),
            float(bounds.ymin),
            float(bounds.zmin),
            float(bounds.xmax),
            float(bounds.ymax),
            float(bounds.zmax),
        ],
    }
    if (
        result["valid"] is not True
        or not math.isfinite(result["volume_cm3"])
        or result["volume_cm3"] <= 0.0
        or any(not math.isfinite(value) for value in result["bounds_cm"])
    ):
        raise RuntimeError(f"{label} is not a positive valid CAD solid")
    return result


def _max_difference(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise RuntimeError("CAD metric dimensions differ")
    return max(abs(left - right) for left, right in zip(first, second))


def _construct_set(
    magnet_class: Any,
    request: Mapping[str, Any],
    *,
    case_thickness: float,
    selected_index: int | None,
):
    parser = request["parser"]
    sector = request["sector"]
    if not math.isclose(float(sector["start_degrees"]), 0.0, abs_tol=1.0e-12):
        raise ValueError(
            "this builder supports native sectors starting at zero only"
        )
    kwargs = {
        "case_thickness": case_thickness,
        "sample_mod": int(request["sample_mod"]),
        "start_line": int(parser["start_line"]),
        "scale": float(parser["scale_to_cm"]),
    }
    if selected_index is not None:
        kwargs["filament_indices"] = (selected_index,)
    selected = magnet_class(
        request["coil_input"]["path"],
        float(request["outer_width_cm"]),
        float(request["outer_thickness_cm"]),
        float(sector["extent_degrees"]),
        **kwargs,
    )
    selected.populate_magnet_coils()
    return selected


def _selected_index(magnet_class: Any, request: Mapping[str, Any]) -> int:
    discovery = _construct_set(
        magnet_class, request, case_thickness=0.0, selected_index=None
    )
    expected = request["filament_coordinates_sha256"]
    matches = [
        index
        for index, filament in enumerate(discovery.filaments)
        if coordinate_identity(filament.coords)["coordinates_sha256"]
        == expected
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "selected filament is absent or duplicated after finite-sector filtering"
        )
    return matches[0]


def _build_selected(
    magnet_class: Any,
    request: Mapping[str, Any],
    *,
    case_thickness: float,
    index: int,
):
    result = _construct_set(
        magnet_class,
        request,
        case_thickness=case_thickness,
        selected_index=index,
    )
    if len(result.filaments) != 1 or len(result.magnet_coils) != 1:
        raise RuntimeError("local selection did not resolve to one filament")
    result.build_magnet_coils()
    if len(result.coil_solids) != 1:
        raise RuntimeError("selected filament did not create one coil group")
    return result


def build_continuous_local_magnet_geometry(
    plan: Mapping[str, Any],
    output_root: str | Path,
    *,
    intersection_tolerance_cm3: float = 1.0e-6,
    parity_absolute_tolerance_cm3: float = 1.0e-6,
    parity_relative_tolerance: float = 1.0e-9,
    coordinate_tolerance_cm: float = 1.0e-8,
) -> dict[str, Any]:
    """Build only local CAD; never compare it to the global radial layer."""
    if (
        not isinstance(plan, Mapping)
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("status")
        != "CONTINUOUS_GLOBAL_AND_LOCAL_COIL_COUPLING_BOUND_NOT_EXECUTED"
        or plan.get("claims", {}).get("global_swept_coils") is not False
        or plan.get("local_geometry_request", {}).get(
            "global_volume_parity_claim"
        )
        is not False
    ):
        raise ValueError("local geometry plan is not accepted")
    request = plan["local_geometry_request"]
    transform = plan.get("surface_source", {}).get("transform", {})
    if not np.allclose(
        transform.get("rotation_global_to_local"),
        np.eye(3),
        rtol=0.0,
        atol=1.0e-12,
    ) or not np.allclose(
        transform.get("translation_cm"), np.zeros(3), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "native-coordinate CAD builder requires an identity coupling transform"
        )
    tolerances = {
        "intersection_volume_cm3": float(intersection_tolerance_cm3),
        "parity_absolute_volume_cm3": float(parity_absolute_tolerance_cm3),
        "parity_relative": float(parity_relative_tolerance),
        "coordinate_cm": float(coordinate_tolerance_cm),
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in tolerances.values()
    ):
        raise ValueError("CAD tolerances must be finite and nonnegative")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)

    before = {}
    for name, row in plan.get("bindings", {}).items():
        path = Path(str(row.get("path", ""))).resolve(strict=True)
        digest = sha256_file(path)
        if digest != row.get("sha256"):
            raise ValueError(
                f"binding {name} changed before local CAD construction"
            )
        before[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }

    cq, magnet_class = _dependencies()
    index = _selected_index(magnet_class, request)
    split = _build_selected(
        magnet_class,
        request,
        case_thickness=float(request["case_thickness_cm"]),
        index=index,
    )
    unsplit_set = _build_selected(
        magnet_class, request, case_thickness=0.0, index=index
    )
    split_identity = coordinate_identity(split.filaments[0].coords)
    unsplit_identity = coordinate_identity(unsplit_set.filaments[0].coords)
    expected = request["filament_coordinates_sha256"]
    if (
        split_identity["coordinates_sha256"] != expected
        or unsplit_identity["coordinates_sha256"] != expected
        or len(split.coil_solids[0]) != 2
        or len(unsplit_set.coil_solids[0]) != 1
    ):
        raise RuntimeError(
            "local filament identity or solid multiplicity changed"
        )

    casing, winding = split.coil_solids[0]
    unsplit = unsplit_set.coil_solids[0][0]
    split_union = casing.fuse(winding)
    metrics = {
        "outer_casing": _metrics(casing, "outer casing"),
        "winding_pack": _metrics(winding, "winding pack"),
        "split_union": _metrics(split_union, "split union"),
        "same_local_input_unsplit_reference": _metrics(
            unsplit, "local unsplit reference"
        ),
    }
    intersection = float(casing.intersect(winding).Volume())
    extra = float(split_union.cut(unsplit).Volume())
    missing = float(unsplit.cut(split_union).Volume())
    reference_volume = metrics["same_local_input_unsplit_reference"][
        "volume_cm3"
    ]
    parity_limit = (
        tolerances["parity_absolute_volume_cm3"]
        + tolerances["parity_relative"] * reference_volume
    )
    volume_delta = abs(metrics["split_union"]["volume_cm3"] - reference_volume)
    centroid_delta = _max_difference(
        metrics["split_union"]["centroid_cm"],
        metrics["same_local_input_unsplit_reference"]["centroid_cm"],
    )
    bounds_delta = _max_difference(
        metrics["split_union"]["bounds_cm"],
        metrics["same_local_input_unsplit_reference"]["bounds_cm"],
    )
    if intersection > tolerances["intersection_volume_cm3"]:
        raise RuntimeError("local casing and winding interiors overlap")
    if max(extra, missing, volume_delta) > parity_limit:
        raise RuntimeError(
            "local split union differs from same-input unsplit coil"
        )
    if max(centroid_delta, bounds_delta) > tolerances["coordinate_cm"]:
        raise RuntimeError(
            "local split union position differs from unsplit coil"
        )

    filament_id = request["filament_id"]
    paths = {
        "outer_casing": output / f"{filament_id}_outer_casing.step",
        "winding_pack": output / f"{filament_id}_winding_pack.step",
        "assembly": output / f"{filament_id}_assembly.step",
    }
    cq.exporters.export(casing, str(paths["outer_casing"]))
    cq.exporters.export(winding, str(paths["winding_pack"]))
    cq.exporters.export(
        cq.Compound.makeCompound([casing, winding]), str(paths["assembly"])
    )
    if any(
        not path.is_file() or path.stat().st_size <= 0
        for path in paths.values()
    ):
        raise RuntimeError("local STEP export is missing or empty")
    reload_metrics = {}
    for role in ("outer_casing", "winding_pack"):
        solids = cq.importers.importStep(str(paths[role])).solids().vals()
        if len(solids) != 1:
            raise RuntimeError(f"{role} STEP reload is not one solid")
        reload_metrics[role] = _metrics(solids[0], f"reloaded {role}")
        allowed = (
            tolerances["parity_absolute_volume_cm3"]
            + tolerances["parity_relative"] * metrics[role]["volume_cm3"]
        )
        if (
            abs(
                reload_metrics[role]["volume_cm3"]
                - metrics[role]["volume_cm3"]
            )
            > allowed
        ):
            raise RuntimeError(f"{role} STEP volume changed on reload")

    after = {
        name: {
            "path": row["path"],
            "bytes": Path(row["path"]).stat().st_size,
            "sha256": sha256_file(row["path"]),
        }
        for name, row in before.items()
    }
    if after != before:
        raise RuntimeError(
            "a bound input changed during local CAD construction"
        )
    artifacts = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "LOCAL_CASING_WINDING_CAD_BUILT_LOCAL_PARITY_PASS",
        "created_utc": datetime.now(UTC).isoformat(),
        "case_id": plan["case_id"],
        "filament_id": filament_id,
        "source_plan_sha256": _canonical_sha256(dict(plan)),
        "local_geometry_request": dict(request),
        "source_filament_identity": split_identity,
        "solid_metrics": metrics,
        "casing_winding_intersection_volume_cm3": intersection,
        "same_local_input_unsplit_parity": {
            "extra_volume_cm3": extra,
            "missing_volume_cm3": missing,
            "absolute_volume_delta_cm3": volume_delta,
            "maximum_centroid_delta_cm": centroid_delta,
            "maximum_bounds_delta_cm": bounds_delta,
            "pass": True,
        },
        "global_geometry_comparison": {
            "volume_parity_claim": False,
            "centroid_parity_claim": False,
            "reason": "global model is a continuous radial magnet layer",
        },
        "step_reload": {"component_metrics": reload_metrics, "pass": True},
        "input_bindings_before": before,
        "input_bindings_after": after,
        "tolerances": tolerances,
        "artifacts": artifacts,
        "claims": {
            "global_geometry_modified": False,
            "global_swept_coils": False,
            "local_cad_coordinate_frame": "global_native",
            "dagmc_generated": False,
            "transport_executed": False,
            "production_authorized": False,
        },
    }
    manifest["manifest_content_sha256"] = _canonical_sha256(manifest)
    manifest_path = output / "LOCAL_MAGNET_GEOMETRY_MANIFEST.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest
