"""Evidence-bound parsing and projections for OpenMC scaling studies.

The helpers in this module deliberately distinguish measured observations from
ideal-scaling illustrations.  A production estimate cannot be reported as
measured when the requested scaling ladder is incomplete.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

REQUIRED_CORE_COUNTS = (8, 16, 32, 64)
PASS = "PASS_MEASURED_SCALING_LADDER"
INCOMPLETE = "NOT_RUN_INCOMPLETE_SCALING_LADDER"


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def allowed_cores(physical_cores: int, fraction: float = 0.25) -> int:
    """Return the integer per-node policy limit for physical cores."""

    if isinstance(physical_cores, bool) or not isinstance(physical_cores, int):
        raise TypeError("physical_cores must be an integer")
    if physical_cores <= 0:
        raise ValueError("physical_cores must be positive")
    policy_fraction = _positive(fraction, "fraction")
    if policy_fraction > 1.0:
        raise ValueError("fraction cannot exceed one")
    return math.floor(physical_cores * policy_fraction)


def physical_cores(
    *, sockets: int, cores_per_socket: int, threads_per_core: int
) -> int:
    """Resolve physical cores without confusing Slurm CPUs with cores."""

    values = (sockets, cores_per_socket, threads_per_core)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise TypeError("topology values must be integers")
    if any(value <= 0 for value in values):
        raise ValueError("topology values must be positive")
    # The threads value is validated because it is essential evidence, even
    # though sockets * cores/socket is already the physical-core count.
    return sockets * cores_per_socket


def normalize_benchmark(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one benchmark row and derive throughput and CPU-hours."""

    if not isinstance(row, Mapping):
        raise TypeError("benchmark row must be a mapping")
    cores = row.get("cores")
    histories = row.get("histories")
    if isinstance(cores, bool) or not isinstance(cores, int) or cores <= 0:
        raise ValueError("benchmark cores must be a positive integer")
    if (
        isinstance(histories, bool)
        or not isinstance(histories, int)
        or histories <= 0
    ):
        raise ValueError("benchmark histories must be a positive integer")
    wall_seconds = _positive(row.get("wall_seconds"), "wall_seconds")
    result = dict(row)
    result["histories_per_second"] = histories / wall_seconds
    result["cpu_hours"] = cores * wall_seconds / 3600.0
    return result


def evaluate_scaling_ladder(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_histories: int = 250_000_000,
) -> dict[str, Any]:
    """Evaluate measured ladder completeness and compute bounded projections."""

    if (
        isinstance(target_histories, bool)
        or not isinstance(target_histories, int)
        or target_histories <= 0
    ):
        raise ValueError("target_histories must be a positive integer")
    normalized = [normalize_benchmark(row) for row in rows]
    by_core = {row["cores"]: row for row in normalized}
    if len(by_core) != len(normalized):
        raise ValueError("benchmark core counts must be unique")
    missing = [core for core in REQUIRED_CORE_COUNTS if core not in by_core]
    status = INCOMPLETE if missing else PASS
    measured_projections = {
        str(core): target_histories / row["histories_per_second"]
        for core, row in sorted(by_core.items())
    }
    efficiencies = {}
    if 8 in by_core:
        baseline = by_core[8]["histories_per_second"]
        for core, row in sorted(by_core.items()):
            efficiencies[str(core)] = (
                row["histories_per_second"] / baseline / (core / 8.0)
            )
    return {
        "status": status,
        "required_core_counts": list(REQUIRED_CORE_COUNTS),
        "missing_core_counts": missing,
        "measurements": normalized,
        "parallel_efficiency_relative_to_8_cores": efficiencies,
        "target_histories": target_histories,
        "measured_wall_time_projection_seconds": measured_projections,
        "scaling_fit_status": (
            "MEASURED_FIT_AVAILABLE" if not missing else "NO_MEASURED_FIT"
        ),
    }
