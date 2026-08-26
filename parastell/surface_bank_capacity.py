"""Conservative capacity planning for ParaStell crossing banks."""

from __future__ import annotations

import math
from pathlib import Path
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


def validate_surface_bank_run_evidence(
    evidence: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate immutable run identity and aggregate writer accounting.

    The OpenMC writer's terminal record counts remain authoritative.  Merely
    observing a file smaller than its configured capacity is not sufficient
    evidence of completeness.
    """
    required = {
        "run_id",
        "seed",
        "source_histories",
        "requested_surface_ids",
        "max_particles_per_file",
        "max_source_files",
        "mpi_ranks",
        "stored_record_count",
        "selected_record_count",
        "source_file_count",
        "source_files",
        "sampling_applied",
        "terminal_writer_evidence",
    }
    missing = required - set(evidence)
    if missing:
        raise ValueError(
            f"surface-bank run evidence is missing {sorted(missing)}"
        )
    run_id = str(evidence["run_id"]).strip()
    if not run_id:
        raise ValueError("run_id cannot be empty")
    for name in ("seed", "source_histories", "mpi_ranks"):
        value = evidence[name]
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    requested = [int(value) for value in evidence["requested_surface_ids"]]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("requested_surface_ids must be nonempty and unique")
    files = list(evidence["source_files"])
    if len(files) != int(evidence["source_file_count"]):
        raise ValueError("source file list disagrees with source_file_count")
    paths = [str(Path(row["path"])) for row in files]
    hashes = [str(row["sha256"]).lower() for row in files]
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        raise ValueError("duplicate source-bank path or content hash detected")
    if any(
        len(value) != 64 or set(value) - set("0123456789abcdef")
        for value in hashes
    ):
        raise ValueError("source-bank SHA-256 values are malformed")
    terminal = evidence["terminal_writer_evidence"]
    if not isinstance(terminal, Mapping) or not bool(
        terminal.get("available")
    ):
        raise ValueError("terminal writer evidence is required")
    terminal_count = terminal.get("aggregate_records_written")
    if terminal_count is None or int(terminal_count) != int(
        evidence["stored_record_count"]
    ):
        raise ValueError("terminal writer count disagrees with stored records")
    accounting = validate_crossing_bank_accounting(
        evidence, require_complete=require_complete
    )
    stored_by_rank = terminal.get("records_written_by_rank")
    if stored_by_rank is not None:
        if len(stored_by_rank) != int(evidence["mpi_ranks"]):
            raise ValueError("rank-local writer accounting is incomplete")
        if sum(int(value) for value in stored_by_rank) != int(
            evidence["stored_record_count"]
        ):
            raise ValueError(
                "rank-local writer counts do not sum to aggregate"
            )
    return {
        **accounting,
        "run_id": run_id,
        "seed": int(evidence["seed"]),
        "source_histories": int(evidence["source_histories"]),
        "requested_surface_ids": sorted(requested),
        "source_file_sha256": hashes,
        "terminal_writer_evidence_valid": True,
    }
