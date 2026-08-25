"""Dependency-light evidence for ParaStell source-mesh convergence.

This module deliberately does not read MOAB meshes or import OpenMC.  The
producer that creates a source manifest is responsible for recording the
integrated source rate and the required source moments in that manifest.  This
module binds those observations to the manifest bytes, compares the prescribed
refinement ladder, and writes a self-validating neutral JSON artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

SCHEMA = "parastell.source_mesh_convergence/v1.0.0"
SOURCE_MANIFEST_SCHEMA = "parastell.magnet_source_stage/v1.0.0"
REQUIRED_CANDIDATE_SHAPES = (
    (3, 9, 9),
    (5, 21, 17),
    (7, 41, 31),
    (11, 81, 61),
)
SPATIAL_MOMENTS = (
    "mean_x_cm",
    "mean_y_cm",
    "mean_z_cm",
    "mean_x2_cm2",
    "mean_y2_cm2",
    "mean_z2_cm2",
)
DT_ENERGY_MOMENTS = ("mean_eV", "mean_squared_eV2")
SOURCE_METRICS = (
    "integrated_source_rate_per_s",
    *(f"spatial.{name}" for name in SPATIAL_MOMENTS),
    *(f"dt_energy.{name}" for name in DT_ENERGY_MOMENTS),
)
MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS = (
    "whole_magnet.incoming_neutron_current",
    "whole_magnet.incoming_photon_current",
    "whole_magnet.neutron_flux",
    "whole_magnet.photon_flux",
    "whole_magnet.heating",
    "whole_magnet.hotspot_location",
    "computational_cost",
)

CONVERGED_STATUS = "CONVERGED_EXPLICIT_THRESHOLDS"
NOT_CONVERGED_STATUS = "NOT_CONVERGED"
INCOMPLETE_STATUS = "INCOMPLETE_EVIDENCE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _file_reference(
    path_value: Any,
    *,
    expected_sha256: Any = None,
    label: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256(path)
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise ValueError(f"declared {label} hash is not a SHA-256 digest")
        if digest != expected_sha256:
            raise ValueError(f"declared {label} hash does not match the file")
    value = _json_mapping(path, label)
    return (
        {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "schema": value.get("schema"),
        },
        value,
    )


def _shape(value: Any) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in value
        )
    ):
        raise TypeError("source mesh shape must contain three integers")
    result = tuple(int(item) for item in value)
    if result not in REQUIRED_CANDIDATE_SHAPES:
        raise ValueError(
            f"source mesh shape {list(result)} is not in the required ladder"
        )
    return result


def _optional_moments(
    source: Any,
    names: Sequence[str],
    prefix: str,
    missing: list[str],
) -> dict[str, float]:
    if source is None:
        source = {}
    if not isinstance(source, Mapping):
        raise TypeError(f"{prefix} moments must be a mapping")
    unknown = set(source) - set(names)
    if unknown:
        raise ValueError(f"unknown {prefix} moments: {sorted(unknown)}")
    result = {}
    for name in names:
        if name not in source:
            missing.append(f"{prefix}.{name}")
        else:
            result[name] = _finite_float(
                source[name], f"{prefix} moment {name}"
            )
    return result


def _normalize_response_values(
    report: Mapping[str, Any],
    configured_ids: Sequence[str],
    missing: list[str],
) -> dict[str, dict[str, Any]]:
    raw = report.get("whole_magnet_responses", {})
    if not isinstance(raw, Mapping):
        raise TypeError("whole-magnet responses must be a mapping")
    result = {}
    for response_id in configured_ids:
        if response_id not in raw:
            missing.append(f"whole_magnet_response.{response_id}")
            continue
        row = raw[response_id]
        if not isinstance(row, Mapping):
            raise TypeError(
                f"whole-magnet response {response_id!r} must be a mapping"
            )
        units = row.get("units")
        if not isinstance(units, str) or not units:
            raise ValueError(
                f"whole-magnet response {response_id!r} needs explicit units"
            )
        if response_id == "whole_magnet.hotspot_location":
            coordinates = row.get("value_xyz_cm")
            if (
                not isinstance(coordinates, Sequence)
                or isinstance(coordinates, (str, bytes))
                or len(coordinates) != 3
            ):
                raise ValueError(
                    "whole-magnet hotspot location requires value_xyz_cm"
                )
            response_value = [
                _finite_float(value, "hotspot location coordinate")
                for value in coordinates
            ]
            magnet_id = row.get("magnet_id")
            bin_id = row.get("bin_id")
            coordinate_frame = row.get("coordinate_frame")
            if not isinstance(magnet_id, str) or not magnet_id:
                raise ValueError("hotspot location requires a magnet ID")
            if not isinstance(bin_id, (str, int)) or isinstance(bin_id, bool):
                raise ValueError("hotspot location requires a bin ID")
            if coordinate_frame != "global_cartesian_cm" or units != "cm":
                raise ValueError(
                    "hotspot location requires global Cartesian cm coordinates"
                )
            result[response_id] = {
                "value": response_value,
                "units": units,
                "coordinate_frame": coordinate_frame,
                "magnet_id": magnet_id,
                "bin_id": str(bin_id),
            }
            continue
        if response_id in {
            "whole_magnet.incoming_neutron_current",
            "whole_magnet.incoming_photon_current",
        }:
            if (
                row.get("direction") != "incoming"
                or row.get("crossing_selection") != "inward_only"
                or row.get("estimator") != "direction_resolved_surface_current"
            ):
                raise ValueError(
                    "incoming current requires inward-only, direction-resolved "
                    "surface-current evidence"
                )
        response_value = _finite_float(
            row.get("value"), f"whole-magnet response {response_id!r}"
        )
        if response_id == "computational_cost" and response_value <= 0.0:
            raise ValueError("computational cost must be positive")
        result[response_id] = {
            "value": response_value,
            "units": units,
        }
        if response_id.startswith("whole_magnet.incoming_"):
            result[response_id].update(
                {
                    "direction": "incoming",
                    "crossing_selection": "inward_only",
                    "estimator": "direction_resolved_surface_current",
                }
            )
    return result


def _normalize_candidate(
    source: Mapping[str, Any], configured_response_ids: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise TypeError("source convergence candidates must be mappings")
    manifest_ref, manifest = _file_reference(
        source.get("source_manifest_path"),
        expected_sha256=source.get("source_manifest_sha256"),
        label="source manifest",
    )
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("unsupported source manifest schema")
    mesh = manifest.get("source_mesh")
    if not isinstance(mesh, Mapping):
        raise TypeError("source manifest source_mesh must be a mapping")
    shape = _shape(mesh.get("shape"))
    mesh_sha256 = mesh.get("sha256")
    if not _is_sha256(mesh_sha256):
        raise ValueError("source manifest mesh hash must be a SHA-256 digest")
    rate = _positive_float(
        mesh.get("physical_source_rate_per_s"), "integrated source rate"
    )
    expected_shape = source.get("expected_shape")
    if expected_shape is not None and _shape(expected_shape) != shape:
        raise ValueError(
            "source manifest shape disagrees with the configured candidate shape"
        )

    missing = []
    observations = manifest.get("convergence_observables", {})
    if not isinstance(observations, Mapping):
        raise TypeError("source convergence observations must be a mapping")
    spatial = _optional_moments(
        observations.get("spatial_moments"),
        SPATIAL_MOMENTS,
        "spatial",
        missing,
    )
    energy = _optional_moments(
        observations.get("dt_energy_moments"),
        DT_ENERGY_MOMENTS,
        "dt_energy",
        missing,
    )
    for axis in ("x", "y", "z"):
        mean_name = f"mean_{axis}_cm"
        second_name = f"mean_{axis}2_cm2"
        if (
            mean_name in spatial
            and second_name in spatial
            and spatial[second_name] < spatial[mean_name] ** 2
        ):
            raise ValueError(
                f"spatial second moment for {axis} is below mean squared"
            )
    if "mean_eV" in energy and energy["mean_eV"] <= 0.0:
        raise ValueError("D-T mean energy must be positive")
    if (
        "mean_eV" in energy
        and "mean_squared_eV2" in energy
        and energy["mean_squared_eV2"] < energy["mean_eV"] ** 2
    ):
        raise ValueError("D-T second energy moment is below mean squared")

    report_ref = None
    responses = {}
    report_path = source.get("response_report_path")
    if report_path is not None:
        report_ref, report = _file_reference(
            report_path,
            expected_sha256=source.get("response_report_sha256"),
            label="response report",
        )
        responses = _normalize_response_values(
            report, configured_response_ids, missing
        )
    elif configured_response_ids:
        missing.append("response_report")
        missing.extend(
            f"whole_magnet_response.{item}" for item in configured_response_ids
        )

    return {
        "shape": list(shape),
        "source_manifest": manifest_ref,
        "source_mesh_sha256": mesh_sha256,
        "integrated_source_rate_per_s": rate,
        "spatial_moments": spatial,
        "dt_energy_moments": energy,
        "response_report": report_ref,
        "whole_magnet_responses": responses,
        "missing_observations": sorted(set(missing)),
    }


def _normalize_tolerance(
    value: Any, label: str, *, allow_selection_only: bool = False
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} tolerance must be a mapping")
    allowed = {"maximum_relative_change", "maximum_absolute_change"}
    if allow_selection_only:
        allowed.add("qualification_role")
    if unknown := set(value) - allowed:
        raise ValueError(
            f"unknown {label} tolerance fields: {sorted(unknown)}"
        )
    if not value:
        raise ValueError(f"{label} tolerance must be nonempty")
    role = value.get("qualification_role", "convergence")
    if role not in {"convergence", "selection_only"}:
        raise ValueError(f"{label} qualification role is invalid")
    if role == "selection_only" and not allow_selection_only:
        raise ValueError(f"{label} cannot be selection-only")
    result = {}
    for name in sorted(set(value) - {"qualification_role"}):
        threshold = _finite_float(value[name], f"{label} {name}")
        if threshold < 0.0:
            raise ValueError(f"{label} tolerances cannot be negative")
        result[name] = threshold
    if role == "selection_only":
        result["qualification_role"] = role
    elif not result:
        raise ValueError(f"{label} tolerance needs a numeric threshold")
    return result


def _normalize_tolerances(
    source: Mapping[str, Mapping[str, Any]] | None,
    responses: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    source = dict(source or {})
    if unknown := set(source) - set(SOURCE_METRICS):
        raise ValueError(
            f"unknown source convergence metrics: {sorted(unknown)}"
        )
    source_result = {
        name: _normalize_tolerance(value, f"source metric {name!r}")
        for name, value in sorted(source.items())
    }
    response_result = {}
    for name, value in sorted(dict(responses or {}).items()):
        if not isinstance(name, str) or not name:
            raise ValueError("response metric IDs must be nonempty strings")
        response_result[name] = _normalize_tolerance(
            value,
            f"response metric {name!r}",
            allow_selection_only=True,
        )
    return source_result, response_result


def _normalize_required_response_ids(
    values: Sequence[str] | None,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("required response metric IDs must be a sequence")
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError("required response metric IDs must be nonempty")
    if len(result) != len(set(result)):
        raise ValueError("required response metric IDs must be unique")
    return result


def _metric_values(candidate: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "integrated_source_rate_per_s": candidate[
            "integrated_source_rate_per_s"
        ]
    }
    values.update(
        {
            f"spatial.{name}": value
            for name, value in candidate["spatial_moments"].items()
        }
    )
    values.update(
        {
            f"dt_energy.{name}": value
            for name, value in candidate["dt_energy_moments"].items()
        }
    )
    values.update(
        {
            f"whole_magnet_response.{name}": row["value"]
            for name, row in candidate["whole_magnet_responses"].items()
        }
    )
    return values


def _compare(
    coarse: Any, fine: Any, tolerance: Mapping[str, Any]
) -> dict[str, Any]:
    if isinstance(coarse, list) or isinstance(fine, list):
        if not (
            isinstance(coarse, list)
            and isinstance(fine, list)
            and len(coarse) == len(fine) == 3
        ):
            raise ValueError("response vector shapes changed")
        absolute = math.sqrt(
            sum(
                (fine_value - coarse_value) ** 2
                for coarse_value, fine_value in zip(coarse, fine)
            )
        )
        coarse_scale = math.sqrt(sum(value**2 for value in coarse))
        fine_scale = math.sqrt(sum(value**2 for value in fine))
        scale = max(coarse_scale, fine_scale)
    else:
        absolute = abs(fine - coarse)
        scale = max(abs(fine), abs(coarse))
    relative = absolute / scale if scale else 0.0
    checks = []
    if "maximum_relative_change" in tolerance:
        checks.append(
            {
                "kind": "relative",
                "observed": relative,
                "threshold": tolerance["maximum_relative_change"],
                "passes": relative <= tolerance["maximum_relative_change"],
            }
        )
    if "maximum_absolute_change" in tolerance:
        checks.append(
            {
                "kind": "absolute",
                "observed": absolute,
                "threshold": tolerance["maximum_absolute_change"],
                "passes": absolute <= tolerance["maximum_absolute_change"],
            }
        )
    role = tolerance.get("qualification_role", "convergence")
    return {
        "coarse_value": coarse,
        "fine_value": fine,
        "absolute_change": absolute,
        "relative_change": relative,
        "threshold_logic": (
            "selection_only_not_a_convergence_bound"
            if role == "selection_only"
            else "any_configured_bound"
        ),
        "qualification_role": role,
        "threshold_checks": checks,
        "passes": (
            True
            if role == "selection_only"
            else any(check["passes"] for check in checks)
        ),
    }


def _evaluate_normalized(
    candidates: Sequence[Mapping[str, Any]],
    source_tolerances: Mapping[str, Mapping[str, float]],
    response_tolerances: Mapping[str, Mapping[str, float]],
    required_response_ids: Sequence[str] = (),
) -> dict[str, Any]:
    shape_order = {
        shape: index for index, shape in enumerate(REQUIRED_CANDIDATE_SHAPES)
    }
    shapes = [tuple(candidate["shape"]) for candidate in candidates]
    if len(set(shapes)) != len(shapes):
        raise ValueError("source convergence candidate shapes must be unique")
    candidates = sorted(
        candidates, key=lambda item: shape_order[tuple(item["shape"])]
    )
    by_shape = {tuple(item["shape"]): item for item in candidates}

    missing = []
    for shape in REQUIRED_CANDIDATE_SHAPES:
        if shape not in by_shape:
            missing.append(f"candidate_shape.{list(shape)}")
    for metric in SOURCE_METRICS:
        if metric not in source_tolerances:
            missing.append(f"source_tolerance.{metric}")
    for response_id in required_response_ids:
        if response_id not in response_tolerances:
            missing.append(f"response_tolerance.{response_id}")
    for candidate in candidates:
        shape_label = "x".join(str(value) for value in candidate["shape"])
        missing.extend(
            f"candidate.{shape_label}.{item}"
            for item in candidate["missing_observations"]
        )

    expected_response_units: dict[str, str] = {}
    for response_id in response_tolerances:
        for candidate in candidates:
            row = candidate["whole_magnet_responses"].get(response_id)
            if row is None:
                continue
            previous = expected_response_units.setdefault(
                response_id, row["units"]
            )
            if row["units"] != previous:
                raise ValueError(
                    f"whole-magnet response {response_id!r} changed units"
                )

    tolerances = dict(source_tolerances)
    tolerances.update(
        {
            f"whole_magnet_response.{name}": value
            for name, value in response_tolerances.items()
        }
    )
    comparisons = []
    for coarse_shape, fine_shape in zip(
        REQUIRED_CANDIDATE_SHAPES[:-1], REQUIRED_CANDIDATE_SHAPES[1:]
    ):
        if coarse_shape not in by_shape or fine_shape not in by_shape:
            continue
        coarse_values = _metric_values(by_shape[coarse_shape])
        fine_values = _metric_values(by_shape[fine_shape])
        metrics = {}
        for metric, tolerance in sorted(tolerances.items()):
            if metric in coarse_values and metric in fine_values:
                comparison = _compare(
                    coarse_values[metric], fine_values[metric], tolerance
                )
                if metric == (
                    "whole_magnet_response." "whole_magnet.hotspot_location"
                ):
                    coarse_hotspot = by_shape[coarse_shape][
                        "whole_magnet_responses"
                    ]["whole_magnet.hotspot_location"]
                    fine_hotspot = by_shape[fine_shape][
                        "whole_magnet_responses"
                    ]["whole_magnet.hotspot_location"]
                    same_magnet = (
                        coarse_hotspot["magnet_id"]
                        == fine_hotspot["magnet_id"]
                    )
                    comparison.update(
                        {
                            "coarse_magnet_id": coarse_hotspot["magnet_id"],
                            "coarse_bin_id": coarse_hotspot["bin_id"],
                            "fine_magnet_id": fine_hotspot["magnet_id"],
                            "fine_bin_id": fine_hotspot["bin_id"],
                            "same_magnet": same_magnet,
                            "passes": comparison["passes"] and same_magnet,
                        }
                    )
                metrics[metric] = comparison
        comparisons.append(
            {
                "coarse_shape": list(coarse_shape),
                "fine_shape": list(fine_shape),
                "metrics": metrics,
            }
        )

    qualification_shapes = [
        list(REQUIRED_CANDIDATE_SHAPES[-2]),
        list(REQUIRED_CANDIDATE_SHAPES[-1]),
    ]
    qualification = next(
        (
            item
            for item in comparisons
            if item["coarse_shape"] == qualification_shapes[0]
            and item["fine_shape"] == qualification_shapes[1]
        ),
        None,
    )
    expected_metric_count = len(SOURCE_METRICS) + len(response_tolerances)
    if (
        qualification is None
        or len(qualification["metrics"]) != expected_metric_count
    ):
        missing.append("complete_finest_pair_comparison")
    missing = sorted(set(missing))
    if missing:
        status = INCOMPLETE_STATUS
    elif all(row["passes"] for row in qualification["metrics"].values()):
        status = CONVERGED_STATUS
    else:
        status = NOT_CONVERGED_STATUS
    selected_shape = None
    selection_status = "NO_DEFAULT_INCOMPLETE_EVIDENCE"
    if status == NOT_CONVERGED_STATUS:
        selection_status = "NO_DEFAULT_NOT_CONVERGED"
    elif status == CONVERGED_STATUS:
        coarse = by_shape[REQUIRED_CANDIDATE_SHAPES[-2]]
        fine = by_shape[REQUIRED_CANDIDATE_SHAPES[-1]]
        coarse_cost = coarse["whole_magnet_responses"].get(
            "computational_cost"
        )
        fine_cost = fine["whole_magnet_responses"].get("computational_cost")
        if coarse_cost is not None and fine_cost is not None:
            selected_shape = (
                REQUIRED_CANDIDATE_SHAPES[-2]
                if coarse_cost["value"] <= fine_cost["value"]
                else REQUIRED_CANDIDATE_SHAPES[-1]
            )
            selection_status = "SELECTED_LOWEST_COST_QUALIFIED_PAIR"
        else:
            selected_shape = REQUIRED_CANDIDATE_SHAPES[-1]
            selection_status = "SELECTED_FINEST_WITHOUT_COST_METRIC"
    default_selection = {
        "status": selection_status,
        "selected_shape": (
            list(selected_shape) if selected_shape is not None else None
        ),
        "conservative_fallback_shape": list(REQUIRED_CANDIDATE_SHAPES[-1]),
        "selection_basis": (
            "A default is selected only after the complete finest pair passes "
            "all explicit thresholds; when cost is available, the lower-cost "
            "member of that qualified pair is selected."
        ),
    }
    return {
        "candidates": list(candidates),
        "comparisons": comparisons,
        "missing_evidence": missing,
        "qualification_pair": qualification_shapes,
        "qualification_metric_count": expected_metric_count,
        "overall_status": status,
        "default_selection": default_selection,
    }


def _stable_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": value["schema"],
        "required_candidate_shapes": value["required_candidate_shapes"],
        "source_metric_tolerances": value["source_metric_tolerances"],
        "response_metric_tolerances": value["response_metric_tolerances"],
        "required_response_metric_ids": value["required_response_metric_ids"],
        "candidates": value["candidates"],
        "comparisons": value["comparisons"],
        "missing_evidence": value["missing_evidence"],
        "qualification_pair": value["qualification_pair"],
        "qualification_metric_count": value["qualification_metric_count"],
        "overall_status": value["overall_status"],
        "default_selection": value["default_selection"],
        "claim_scope": value["claim_scope"],
    }


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _stable_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_source_mesh_convergence(
    candidates: Sequence[Mapping[str, Any]],
    *,
    source_metric_tolerances: Mapping[str, Mapping[str, Any]] | None = None,
    response_metric_tolerances: Mapping[str, Mapping[str, Any]] | None = None,
    required_response_metric_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a prescribed four-shape source-mesh refinement campaign.

    Candidate observations are read from hash-bound JSON source manifests and,
    when response metrics are configured, hash-bound response reports.  Missing
    candidates, observations, reports, or explicit source thresholds yield an
    ``INCOMPLETE_EVIDENCE`` artifact rather than a convergence claim.
    """

    source_tolerances, response_tolerances = _normalize_tolerances(
        source_metric_tolerances, response_metric_tolerances
    )
    required_response_ids = _normalize_required_response_ids(
        required_response_metric_ids
    )
    configured_response_ids = tuple(
        sorted(set(response_tolerances).union(required_response_ids))
    )
    normalized = [
        _normalize_candidate(candidate, configured_response_ids)
        for candidate in candidates
    ]
    evaluation = _evaluate_normalized(
        normalized,
        source_tolerances,
        response_tolerances,
        required_response_ids,
    )
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "required_candidate_shapes": [
            list(shape) for shape in REQUIRED_CANDIDATE_SHAPES
        ],
        "source_metric_tolerances": source_tolerances,
        "response_metric_tolerances": response_tolerances,
        "required_response_metric_ids": list(required_response_ids),
        **evaluation,
        "claim_scope": (
            "Convergence applies only to the explicit finest-pair thresholds "
            "and the recorded source and configured whole-magnet metrics. "
            "It does not establish transport statistical qualification."
        ),
    }
    result["evidence_sha256"] = _payload_sha256(result)
    return result


def write_source_mesh_convergence(
    path: str | Path,
    candidates: Sequence[Mapping[str, Any]],
    *,
    source_metric_tolerances: Mapping[str, Mapping[str, Any]] | None = None,
    response_metric_tolerances: Mapping[str, Mapping[str, Any]] | None = None,
    required_response_metric_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write a neutral JSON source-mesh convergence artifact."""

    result = evaluate_source_mesh_convergence(
        candidates,
        source_metric_tolerances=source_metric_tolerances,
        response_metric_tolerances=response_metric_tolerances,
        required_response_metric_ids=required_response_metric_ids,
    )
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result["path"] = str(output)
    return result


def validate_source_mesh_convergence(
    path: str | Path, *, verify_bound_files: bool = False
) -> dict[str, Any]:
    """Validate evidence hashes and recompute comparisons without OpenMC."""

    source = Path(path).resolve()
    value = _json_mapping(source, "source convergence artifact")
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported source convergence schema")
    if value.get("required_candidate_shapes") != [
        list(shape) for shape in REQUIRED_CANDIDATE_SHAPES
    ]:
        raise ValueError("source convergence candidate ladder is invalid")
    if value.get("evidence_sha256") != _payload_sha256(value):
        raise ValueError("source convergence evidence hash is invalid")
    source_tolerances, response_tolerances = _normalize_tolerances(
        value.get("source_metric_tolerances"),
        value.get("response_metric_tolerances"),
    )
    required_response_ids = _normalize_required_response_ids(
        value.get("required_response_metric_ids")
    )
    recalculated = _evaluate_normalized(
        value.get("candidates", ()),
        source_tolerances,
        response_tolerances,
        required_response_ids,
    )
    for name in (
        "candidates",
        "comparisons",
        "missing_evidence",
        "qualification_pair",
        "qualification_metric_count",
        "overall_status",
        "default_selection",
    ):
        if recalculated[name] != value.get(name):
            raise ValueError("source convergence comparisons are inconsistent")
    if verify_bound_files:
        for candidate in value["candidates"]:
            for reference_name in ("source_manifest", "response_report"):
                reference = candidate.get(reference_name)
                if reference is None:
                    continue
                bound = Path(reference["path"]).resolve()
                if not bound.is_file():
                    raise FileNotFoundError(bound)
                if bound.stat().st_size != reference["size_bytes"]:
                    raise ValueError(f"bound {reference_name} size changed")
                if _sha256(bound) != reference["sha256"]:
                    raise ValueError(f"bound {reference_name} hash changed")
            source_reference = candidate["source_manifest"]
            response_reference = candidate.get("response_report")
            candidate_input = {
                "source_manifest_path": source_reference["path"],
                "source_manifest_sha256": source_reference["sha256"],
                "expected_shape": candidate["shape"],
            }
            if response_reference is not None:
                candidate_input.update(
                    {
                        "response_report_path": response_reference["path"],
                        "response_report_sha256": response_reference["sha256"],
                    }
                )
            configured_response_ids = tuple(
                sorted(set(response_tolerances).union(required_response_ids))
            )
            reconstructed = _normalize_candidate(
                candidate_input, configured_response_ids
            )
            if reconstructed != candidate:
                raise ValueError(
                    "serialized source candidate disagrees with bound files"
                )
    return {
        "schema": SCHEMA,
        "evidence_sha256": value["evidence_sha256"],
        "candidate_count": len(value["candidates"]),
        "comparison_count": len(value["comparisons"]),
        "overall_status": value["overall_status"],
        "default_selection": value["default_selection"],
        "missing_evidence_count": len(value["missing_evidence"]),
    }
