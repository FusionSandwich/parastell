"""Conservative capacity planning for ParaStell crossing banks."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .magnet_boundary_envelope import (
    BANK_CLASSIFICATIONS,
    classify_crossing_bank,
)
from .photon_calibration import garwood_poisson_interval


def plan_surface_bank_capacity(
    *,
    observed_crossings: int,
    calibration_histories: int,
    planned_histories: int,
    confidence_level: float = 0.99,
    capacity_margin: float = 1.25,
    max_source_files: int = 1,
    mpi_ranks: int = 1,
) -> dict[str, Any]:
    """Scale an exact crossing-rate bound into an uncapped bank capacity."""

    integer_values = {
        "observed_crossings": observed_crossings,
        "calibration_histories": calibration_histories,
        "planned_histories": planned_histories,
        "max_source_files": max_source_files,
        "mpi_ranks": mpi_ranks,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name} must be an integer")
    observed_crossings = int(observed_crossings)
    calibration_histories = int(calibration_histories)
    planned_histories = int(planned_histories)
    max_source_files = int(max_source_files)
    mpi_ranks = int(mpi_ranks)
    if observed_crossings < 0:
        raise ValueError("observed_crossings cannot be negative")
    if any(
        value <= 0
        for value in (
            calibration_histories,
            planned_histories,
            max_source_files,
            mpi_ranks,
        )
    ):
        raise ValueError("history, file, and MPI counts must be positive")
    margin = float(capacity_margin)
    if not np.isfinite(margin) or margin <= 1.0:
        raise ValueError("capacity_margin must be greater than one")

    lower, upper = garwood_poisson_interval(
        observed_crossings, confidence_level=confidence_level
    )
    exposure_scale = planned_histories / calibration_histories
    expected = observed_crossings * exposure_scale
    confidence_upper = upper * exposure_scale
    required_capacity = max(1, int(math.ceil(confidence_upper * margin)))
    max_particles_per_file = int(
        math.ceil(required_capacity / max_source_files)
    )
    configured_capacity = max_particles_per_file * max_source_files
    return {
        "observed_crossings": observed_crossings,
        "calibration_histories": calibration_histories,
        "planned_histories": planned_histories,
        "observed_rate_per_history": (
            observed_crossings / calibration_histories
        ),
        "garwood_rate_per_history_interval": {
            "confidence_level": float(confidence_level),
            "lower": lower / calibration_histories,
            "upper": upper / calibration_histories,
        },
        "expected_crossings": expected,
        "confidence_upper_crossings": confidence_upper,
        "capacity_margin": margin,
        "required_capacity": required_capacity,
        "max_particles_per_file": max_particles_per_file,
        "max_source_files": max_source_files,
        "configured_capacity": configured_capacity,
        "capacity_above_confidence_upper": (
            configured_capacity - confidence_upper
        ),
        "configured_to_confidence_upper_ratio": (
            configured_capacity / confidence_upper
            if confidence_upper > 0.0
            else None
        ),
        "mpi_ranks": mpi_ranks,
        "mpi_contract": (
            "capacity is validated from aggregate stored/source-file "
            "accounting; rank-local file behavior must be recorded by the run"
        ),
    }


def validate_crossing_bank_accounting(
    accounting: Mapping[str, Any],
    *,
    require_complete: bool = False,
    capacity_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the canonical classifier and reject forged semantics."""

    required = {
        "stored_record_count",
        "selected_record_count",
        "max_particles_per_file",
        "max_source_files",
        "source_file_count",
    }
    missing = required - set(accounting)
    if missing:
        raise ValueError(f"bank accounting is missing {sorted(missing)}")
    if accounting["max_source_files"] is not None and int(
        accounting["source_file_count"]
    ) > int(accounting["max_source_files"]):
        raise ValueError("source_file_count exceeds max_source_files")
    recomputed = classify_crossing_bank(
        stored_record_count=int(accounting["stored_record_count"]),
        selected_record_count=int(accounting["selected_record_count"]),
        max_particles_per_file=(
            None
            if accounting["max_particles_per_file"] is None
            else int(accounting["max_particles_per_file"])
        ),
        max_source_files=(
            None
            if accounting["max_source_files"] is None
            else int(accounting["max_source_files"])
        ),
        source_file_count=int(accounting["source_file_count"]),
        mpi_ranks=(
            None
            if accounting.get("mpi_ranks") is None
            else int(accounting["mpi_ranks"])
        ),
        sampling_applied=bool(accounting.get("sampling_applied", False)),
    )
    declared = accounting.get("classification")
    if declared is not None and declared not in BANK_CLASSIFICATIONS:
        raise ValueError("unknown declared bank classification")
    if declared is not None and declared != recomputed["classification"]:
        raise ValueError("declared bank classification is inconsistent")
    for name in ("cap_reached", "configured_capacity", "sampling_applied"):
        if name in accounting and accounting[name] != recomputed[name]:
            raise ValueError(f"declared bank {name} is inconsistent")
    if capacity_plan is not None:
        configured = recomputed["configured_capacity"]
        required_capacity = int(capacity_plan["required_capacity"])
        if configured is None or configured < required_capacity:
            raise ValueError(
                "bank capacity is below the confidence-upper plan"
            )
        if recomputed["max_particles_per_file"] != int(
            capacity_plan["max_particles_per_file"]
        ) or recomputed["max_source_files"] != int(
            capacity_plan["max_source_files"]
        ):
            raise ValueError("bank configuration disagrees with capacity plan")
    if require_complete and recomputed["classification"] != (
        "COMPLETE_CROSSING_BANK"
    ):
        raise ValueError("qualification requires a complete crossing bank")
    return recomputed


def validate_qualification_bank(
    accounting: Mapping[str, Any],
    *,
    capacity_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require COMPLETE rather than SAMPLED or TRUNCATED bank semantics."""

    return validate_crossing_bank_accounting(
        accounting,
        require_complete=True,
        capacity_plan=capacity_plan,
    )
