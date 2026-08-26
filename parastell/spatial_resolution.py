"""Finite-resolution qualification for surface and local-volume estimators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SPATIAL_STATUSES = (
    "QUALIFIED",
    "MARGINAL",
    "INSUFFICIENT_STATISTICS",
    "EMPTY",
)


def classify_spatial_estimator(
    *,
    raw_records: int | None,
    sum_weights: float | None,
    sum_squared_weights: float | None,
    relative_uncertainty: float | None,
    qualified_max_relative: float = 0.25,
    marginal_max_relative: float = 0.50,
) -> str:
    """Classify one finite bin without interpreting no score as physics."""
    if raw_records is not None and int(raw_records) < 0:
        raise ValueError("raw_records cannot be negative")
    for name, value in (
        ("sum_weights", sum_weights),
        ("sum_squared_weights", sum_squared_weights),
        ("relative_uncertainty", relative_uncertainty),
    ):
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if raw_records == 0 or sum_weights == 0.0:
        return "EMPTY"
    if (
        raw_records is None
        or sum_weights is None
        or sum_squared_weights is None
    ):
        return "INSUFFICIENT_STATISTICS"
    if relative_uncertainty is None:
        return "INSUFFICIENT_STATISTICS"
    if relative_uncertainty <= qualified_max_relative:
        return "QUALIFIED"
    if relative_uncertainty <= marginal_max_relative:
        return "MARGINAL"
    return "INSUFFICIENT_STATISTICS"


def build_spatial_resolution_report(
    short_run_analysis: Mapping[str, Any],
    local_mesh_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    requested = [5.0, 2.0, 1.0, 0.5]
    actual = float(local_mesh_manifest["resolution_cm"])
    representative = short_run_analysis["representative_magnet"]
    surface_patch_rows = []
    for interface_name in ("outer_bank", "winding_bank"):
        for row in representative[interface_name]["surface_patch_statistics"]:
            value = dict(row)
            value["interface"] = interface_name
            if value["raw_records"] == 0:
                value["status"] = "EMPTY"
                value["zero_interpretation"] = (
                    "no record in a complete bank; physical zero is not claimed"
                )
            surface_patch_rows.append(value)
    local_rows = {}
    for particle in ("neutron", "photon"):
        rows = []
        for row in representative["local_mesh_flux"][particle]:
            value = dict(row)
            value.update(
                {
                    "raw_records": None,
                    "sum_weights": None,
                    "sum_squared_weights": None,
                    "status": "INSUFFICIENT_STATISTICS",
                    "classification_reason": (
                        "OpenMC track-length statepoint exposes batch moments, "
                        "not event-level records; all 500k local-bin relative "
                        "uncertainties also exceed the marginal threshold"
                    ),
                }
            )
            rows.append(value)
        local_rows[particle] = rows
    candidates = [
        {
            "requested_scale_cm": value,
            "neutron": "NOT_RUN",
            "photon": "NOT_RUN",
            "heating": "NOT_RUN",
            "reason": (
                "Prompt-1B bounded resource caps produced only the 100 cm "
                "coarse mesh; no tally exists at this requested scale"
            ),
        }
        for value in requested
    ]
    return {
        "schema": "parastell.spatial_resolution_qualification/v1",
        "requested_candidate_scales_cm": requested,
        "actual_prompt1b_requested_resolution_cm": actual,
        "actual_representative_bin_widths_cm": local_mesh_manifest["meshes"][
            "example-stellarator-sector-00-90deg-coil-0005"
        ]["actual_bin_widths_cm"],
        "qualification_thresholds": {
            "qualified_max_relative_uncertainty": 0.25,
            "marginal_max_relative_uncertainty": 0.50,
        },
        "candidate_results": candidates,
        "surface_patch_estimators": surface_patch_rows,
        "local_volume_estimators": local_rows,
        "local_mesh_heating": representative["local_mesh_heating"],
        "finest_defensible_scale": {
            "neutron": "NO_QUALIFIED_SCALE",
            "photon": "NO_QUALIFIED_SCALE",
            "heating": "NO_QUALIFIED_SCALE_NOT_TALLIED",
        },
        "overall_status": "INSUFFICIENT_STATISTICS",
        "interpretation": (
            "This is a sampled phase-space measure and a set of finite-bin "
            "estimators; no field at every mathematical point is claimed."
        ),
    }


def write_spatial_resolution_report(
    output_path: str | Path,
    short_run_analysis_path: str | Path,
    local_mesh_manifest_path: str | Path,
) -> dict[str, Any]:
    analysis = json.loads(
        Path(short_run_analysis_path).read_text(encoding="utf-8")
    )
    meshes = json.loads(
        Path(local_mesh_manifest_path).read_text(encoding="utf-8")
    )
    report = build_spatial_resolution_report(analysis, meshes)
    Path(output_path).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
