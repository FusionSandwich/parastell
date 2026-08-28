"""Fail-closed completeness accounting for OpenMC surface-source banks."""

from __future__ import annotations

from collections.abc import Mapping


BANK_CLASSIFICATIONS = {
    "COMPLETE_CROSSING_BANK",
    "SAMPLED_CROSSING_BANK",
    "TRUNCATED_INVALID_BANK",
}


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_surface_source_accounting(evidence: Mapping) -> dict:
    """Validate terminal file accounting and classify bank completeness.

    In OpenMC 0.16, ``max_particles`` is a per-MPI-process limit.  Thus one
    file can contain at most ``max_particles * mpi_ranks`` records, and the
    total configured capacity also includes ``max_source_files``.  A bank that
    exactly reaches a file capacity is conservatively invalid because omitted
    later crossings cannot be disproved from the HDF5 file alone.
    """

    required = {
        "run_id",
        "seed",
        "source_histories",
        "requested_surface_ids",
        "max_particles_per_process",
        "max_source_files",
        "mpi_ranks",
        "source_files",
        "stored_record_count",
        "selected_record_count",
        "sampling_applied",
        "terminal_writer_evidence",
    }
    missing = required - set(evidence)
    if missing:
        raise ValueError(f"surface-source evidence omits {sorted(missing)}")
    run_id = str(evidence["run_id"]).strip()
    if not run_id:
        raise ValueError("run_id cannot be empty")
    seed = _positive_integer(evidence["seed"], "seed")
    histories = _positive_integer(
        evidence["source_histories"], "source_histories"
    )
    max_particles = _positive_integer(
        evidence["max_particles_per_process"], "max_particles_per_process"
    )
    max_files = _positive_integer(
        evidence["max_source_files"], "max_source_files"
    )
    mpi_ranks = _positive_integer(evidence["mpi_ranks"], "mpi_ranks")
    requested = [int(value) for value in evidence["requested_surface_ids"]]
    if (
        not requested
        or len(requested) != len(set(requested))
        or min(requested) <= 0
    ):
        raise ValueError(
            "requested_surface_ids must be unique positive integers"
        )

    files = list(evidence["source_files"])
    if not files or len(files) > max_files:
        raise ValueError(
            "source file count is zero or exceeds max_source_files"
        )
    paths = [str(row["path"]) for row in files]
    hashes = [str(row["sha256"]).lower() for row in files]
    counts = [int(row["record_count"]) for row in files]
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        raise ValueError("duplicate source-bank path or content hash")
    if any(
        len(value) != 64 or set(value) - set("0123456789abcdef")
        for value in hashes
    ):
        raise ValueError("source-bank SHA-256 is malformed")
    if any(value < 0 for value in counts):
        raise ValueError("source-bank record counts cannot be negative")
    stored = int(evidence["stored_record_count"])
    selected = int(evidence["selected_record_count"])
    if stored < 0 or selected < 0 or selected > stored:
        raise ValueError("stored/selected record accounting is invalid")
    if sum(counts) != stored:
        raise ValueError(
            "source file counts do not sum to stored_record_count"
        )

    terminal = evidence["terminal_writer_evidence"]
    terminal_valid = isinstance(terminal, Mapping) and bool(
        terminal.get("available")
    )
    if terminal_valid:
        terminal_counts = [
            int(value) for value in terminal.get("records_by_file", ())
        ]
        terminal_valid = terminal_counts == counts
    file_capacity = max_particles * mpi_ranks
    configured_capacity = file_capacity * max_files
    capacity_reached = any(value >= file_capacity for value in counts)
    sampling = bool(evidence["sampling_applied"])
    if capacity_reached or not terminal_valid:
        classification = "TRUNCATED_INVALID_BANK"
    elif sampling:
        classification = "SAMPLED_CROSSING_BANK"
    else:
        classification = "COMPLETE_CROSSING_BANK"
    declared = evidence.get("classification")
    if declared is not None and declared != classification:
        raise ValueError(
            "declared classification disagrees with recomputation"
        )
    return {
        "classification": classification,
        "run_id": run_id,
        "seed": seed,
        "source_histories": histories,
        "requested_surface_ids": sorted(requested),
        "max_particles_per_process": max_particles,
        "max_source_files": max_files,
        "mpi_ranks": mpi_ranks,
        "per_file_capacity": file_capacity,
        "configured_capacity": configured_capacity,
        "source_file_count": len(files),
        "stored_record_count": stored,
        "selected_record_count": selected,
        "foreign_record_count": stored - selected,
        "capacity_reached": capacity_reached,
        "sampling_applied": sampling,
        "terminal_writer_evidence_valid": terminal_valid,
        "source_file_sha256": hashes,
    }
